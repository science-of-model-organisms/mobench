"""Where a step's command runs. Steps never know which.

`Executor.run(job) -> JobResult` is the whole interface. `Local` runs a subprocess here, whether
that is a workstation or inside someone's own cluster job. `Fake` records jobs for tests.
`RunPod` renders the same job onto a pod: it is config-driven (image, GPU, prefix, repo), and
everything crossing the pod boundary goes through the build's private scratch repo on the Hub.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass
class Job:
    step: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: Path | None = None
    log: Path | None = None
    gpu_class: str | None = None
    gpu_count: int = 0
    max_usd: float | None = None
    rate_usd_per_hour: float | None = None
    inputs: list[Path] = field(default_factory=list)
    outputs: list[Path] = field(default_factory=list)
    attempt: int = 1
    # True only for a job that makes no files (a gate that reads and exits): without declared
    # outputs the pod executor cannot verify success, so it stops the pod rather than delete it.
    produces_nothing: bool = False
    kill_grace_s: float = 0.0
    timeout_s: float | None = None  # a wall-clock limit that is not a dollar cap

    @property
    def deadline_s(self) -> float | None:
        """When to start stopping the job, set only under `--strict-budget`: the time the cap
        buys at the listed rate, minus the kill grace, so SIGTERM-to-SIGKILL still ends inside
        the cap. Zero when the cap buys less than the grace: `Ctx.job` refuses such a job."""
        if self.max_usd is None or not self.rate_usd_per_hour:
            return None
        return max(0.0, self.max_usd / self.rate_usd_per_hour * 3600 - self.kill_grace_s)

    @property
    def limit_s(self) -> float | None:
        """The earlier of the dollar deadline and the wall-clock limit."""
        limits = [x for x in (self.deadline_s, self.timeout_s) if x is not None]
        return min(limits) if limits else None


@dataclass
class JobResult:
    ok: bool
    rc: int
    wall_s: float = 0.0
    usd: float | None = None
    usd_source: str = "unknown"
    stopped_at_cap: bool = False


class Executor(Protocol):
    def run(self, job: Job) -> JobResult: ...


class JobInterrupted(Exception):
    """SIGINT or SIGTERM reached the build while a job ran; the job's group was stopped first."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"interrupted by signal {signum}")
        self.signum = signum
        self.code = 128 + signum


class LocalExecutor:
    remote = False
    """A subprocess here. Under a deadline it is stopped with SIGTERM, then SIGKILL.

    The job runs in its own process group. SIGINT and SIGTERM sent to the build are forwarded
    to that group (SIGKILL after the grace), and the build then stops with `JobInterrupted`
    rather than carrying on to the next step or leaving the job running.
    """

    def __init__(self, kill_grace_s: float = 300, *, bill_listed_rate: bool = False) -> None:
        self.kill_grace_s = kill_grace_s
        self.bill_listed_rate = bill_listed_rate

    def run(self, job: Job) -> JobResult:
        # Unbuffered: a job's stdout goes to a file, which Python block-buffers, so progress
        # lines would otherwise reach the log only when the job ends.
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", **job.env}
        print("$", shlex.join(job.argv), flush=True)
        log = job.log.open("ab") if job.log else None
        start = time.monotonic()
        proc = subprocess.Popen(  # noqa: S603  # argv is built by the step, no shell
            job.argv,
            env=env,
            cwd=job.cwd,
            stdout=log,
            stderr=subprocess.STDOUT if log else None,
            start_new_session=True,
        )
        received: list[int] = []
        previous = self._forward_signals(proc, received)
        stopped = False
        try:
            try:
                rc = proc.wait(timeout=self._limit(job))
            except subprocess.TimeoutExpired:
                stopped = True
                rc = self._stop(proc)
            else:
                if received and rc != 0:
                    stopped = True
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
            if log:
                log.close()
        if received:
            raise JobInterrupted(received[0])
        wall = time.monotonic() - start
        # A cloud rate from the capacity table says what a pod would cost, not what this
        # machine does: it is booked only when asked (`executor.local.bill_listed_rate`).
        rate = job.rate_usd_per_hour if self.bill_listed_rate else None
        usd = wall / 3600 * rate if rate else None
        source = "wall clock x listed rate" if usd is not None else "unknown"
        return JobResult(rc == 0 and not stopped, rc, wall, usd, source, stopped)

    def _limit(self, job: Job) -> float | None:
        """Dollar deadline only when a listed rate is billed; the wall-clock limit always."""
        deadline = job.deadline_s if self.bill_listed_rate else None
        limits = [x for x in (deadline, job.timeout_s) if x is not None]
        return min(limits) if limits else None

    def _stop(self, proc: subprocess.Popen[bytes]) -> int:
        """SIGTERM the job's group, then SIGKILL it after the grace."""
        _killpg(proc.pid, signal.SIGTERM)
        try:
            return proc.wait(timeout=self.kill_grace_s)
        except subprocess.TimeoutExpired:
            _killpg(proc.pid, signal.SIGKILL)
            return proc.wait()

    def _forward_signals(
        self, proc: subprocess.Popen[bytes], received: list[int]
    ) -> dict[int, Any]:
        """Forward SIGINT/SIGTERM to the job's group (main thread only: Python's rule)."""
        if threading.current_thread() is not threading.main_thread():
            return {}

        def forward(signum: int, _frame: Any) -> None:
            received.append(signum)
            _killpg(proc.pid, signum)
            if len(received) > 1:  # a second signal means "now"
                _killpg(proc.pid, signal.SIGKILL)
            elif len(received) == 1:  # a job that ignores the signal is killed after the grace
                timer = threading.Timer(self.kill_grace_s, _killpg, (proc.pid, signal.SIGKILL))
                timer.daemon = True
                timer.start()

        return {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM)}


def _killpg(pid: int, signum: int) -> None:
    try:
        os.killpg(pid, signum)
    except ProcessLookupError:
        pass


class FakeExecutor:
    """Records jobs and calls `on_job`, which writes the declared outputs. No process, no cost."""

    def __init__(self, on_job: Callable[[Job], int] | None = None) -> None:
        self.jobs: list[Job] = []
        self.on_job = on_job

    def run(self, job: Job) -> JobResult:
        self.jobs.append(job)
        rc = self.on_job(job) if self.on_job else 0
        return JobResult(rc == 0, rc, 0.0, 0.0, "fake")


class Shell(Protocol):
    """Run an argv (never a shell string). `input` goes to stdin: the only way a secret or a
    script body crosses to a pod, so it never appears in an argv, a process list or a log."""

    def __call__(
        self, cmd: Sequence[str], input: str | None = None
    ) -> subprocess.CompletedProcess[str]: ...


SHELL_TIMEOUT_S = 900


def _shell(cmd: Sequence[str], input: str | None = None) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(cmd),
            input=input,
            capture_output=True,
            text=True,
            check=False,
            timeout=SHELL_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            list(cmd), 124, "", f"timed out after {SHELL_TIMEOUT_S}s"
        )


def _tcp_probe(ip: str, port: int) -> bool:
    import socket

    try:
        with socket.create_connection((ip, port), timeout=5):
            return True
    except OSError:
        return False


# Exit codes the pod's script reports besides the job's own.
RC_SETUP_FAILED, RC_UPLOAD_FAILED = 97, 98
# The stock capability evals (mmlu, gsm8k, ...) of `evaluate` need these; no other pod does.
EVALUATE_EXTRAS = " --extra ifeval-anchor --extra inspect-evals"
ENV_FILE = "/root/.mobench.env"
SCRIPT_FILE = "/root/mobench-job.sh"
JOB_LOG, SETUP_LOG = "/root/mobench-job.log", "/root/mobench-setup.log"
RC_FILE = "/tmp/mobench.rc"  # noqa: S108  # on the pod, not here


class RunPodExecutor:
    remote = True
    """One pod per job (one base per pod; steering takes its own pod, as the job says).

    The pod record is written as an intent BEFORE the create call, so a crash between the two
    leaves a trace, and only a verified delete (or the watchdog's stop) records termination.

    A pod is deleted only after its outputs are verified present locally by name. On any doubt
    (a failed upload, a missing output, no result before the wait limit) it is STOPPED, which
    keeps its disk, and the message names it. A job that exits non-zero is left up for
    inspection, as before.

    Secrets (`HF_TOKEN`, plus names in `secrets`) go over ssh stdin into a 0600 file on the pod,
    and the job script the same way: neither is ever in an argv or a log. SSH uses the pod's
    verified ip and port (the API's port is sometimes off by one) and `ssh_key`.

    Flags of `runpodctl pod create` come from `create_args` in config, because they are the
    account's business. NOT YET RUN AGAINST A REAL POD: every test uses a fake `shell`.
    """

    def __init__(
        self,
        config: dict[str, Any],
        pods_dir: Path,
        *,
        shell: Shell = _shell,
        sync: Any = None,
        poll_s: float = 60,
        spawn_watchdog: Callable[[Path], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        probe: Callable[[str, int], bool] = _tcp_probe,
        secrets: Callable[[], dict[str, str]] | None = None,
    ) -> None:
        missing = [
            k for k in ("name_prefix", "repo_url", "commit", "scratch_repo") if k not in config
        ]
        if missing:
            raise ValueError(f"executor.runpod needs {missing}")
        self.config = config
        self.pods_dir = pods_dir
        self.shell = shell
        self.sync = sync if sync is not None else HubSync(config["scratch_repo"])
        self.poll_s = poll_s
        self.spawn_watchdog = spawn_watchdog or _spawn_watchdog
        self.sleep, self.clock, self.probe = sleep, clock, probe
        self._api_endpoint: dict[str, tuple[str, int]] = {}
        self.secrets = secrets or self._env_secrets

    # ---------------------------------------------------------------- pieces

    def create_command(self, name: str, job: Job) -> list[str]:
        args = [str(a).format(name=name, gpu=job.gpu_class, count=job.gpu_count)
                for a in self.config.get("create_args", [])]  # fmt: skip
        return ["runpodctl", "pod", "create", "--name", name, *args]

    def _env_secrets(self) -> dict[str, str]:
        """Values for `HF_TOKEN` and the configured names. Names are config; values never are."""
        out: dict[str, str] = {}
        for name in dict.fromkeys(["HF_TOKEN", *self.config.get("secrets", [])]):
            value = os.environ.get(name)
            if not value and name == "HF_TOKEN":
                from huggingface_hub import get_token

                value = get_token()
            if value:
                out[name] = value
        return out

    def bootstrap(self, job: Job, name: str) -> str:
        """The script a pod runs: set up (fail fast), run the job, upload, report one exit code.

        Setup failing exits 97 and an upload failing after a clean job exits 98, so neither
        looks like success. Secrets are sourced from the 0600 env file, never written here.
        """
        c = self.config
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in job.env.items())
        outs = " ".join(shlex.quote(str(p)) for p in job.outputs)
        argv = ["uv", "run", *job.argv] if job.argv[:1] == ["python"] else job.argv
        repo, ident = shlex.quote(c["scratch_repo"]), shlex.quote(name)
        return "\n".join(
            [
                "set -uo pipefail",
                f'report() {{ echo "$1" > {RC_FILE}.tmp && mv {RC_FILE}.tmp {RC_FILE}; }}',
                f"trap 'rm -f {ENV_FILE} {SCRIPT_FILE}' EXIT",
                f"if [ -f {ENV_FILE} ]; then set -a; . {ENV_FILE}; set +a; fi",
                # The setup log goes up when setup fails, so a failed pod can be read and deleted.
                # Only that log: the job log does not exist yet, and `sync push` refuses a
                # missing path. Bounded, so a dead network cannot hold the rc file on a paid pod.
                f"push_logs() {{ (cd repo 2>/dev/null || true; timeout 120 uv run python -m "
                f"orchestrate.sync push {repo} {ident} {SETUP_LOG}) >/dev/null 2>&1 || true; }}",
                "(",
                # `&&`, not `set -e`: errexit is ignored in a subshell on the left of `||`,
                # so a failed checkout or `uv sync` would still go on to train.
                f"  git clone -q {shlex.quote(c['repo_url'])} repo &&",
                f"  cd repo && git checkout -q {shlex.quote(str(c['commit']))} &&",
                "  git submodule update --init vendor/OpenCharacterTraining &&",
                f"  uv sync -q --extra gpu{EVALUATE_EXTRAS if job.step == 'evaluate' else ''} &&",
                f"  uv run python -m orchestrate.sync pull {repo} {ident}",
                f") > {SETUP_LOG} 2>&1 || "
                f"{{ push_logs; report {RC_SETUP_FAILED}; exit {RC_SETUP_FAILED}; }}",
                "cd repo",
                f"{env} {shlex.join(argv)} > {JOB_LOG} 2>&1",
                "rc=$?",
                f"uv run python -m orchestrate.sync push {repo} {ident} {JOB_LOG} {outs}",
                "up=$?",
                f"if [ $rc -eq 0 ] && [ $up -ne 0 ]; then rc={RC_UPLOAD_FAILED}; fi",
                'report "$rc"',
                "exit $rc",
            ]
        )

    def ssh_prefix(self, ip: str, port: int) -> list[str]:
        key = os.path.expanduser(self.config.get("ssh_key", "~/.runpod/ssh/runpodctl-ssh-key"))
        return [
            "ssh", "-i", key, "-p", str(port),
            "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            *self.config.get("ssh_args", []), f"root@{ip}",
        ]  # fmt: skip

    def ssh_candidates(self, pod_id: str) -> list[tuple[str, int]]:
        """Every ip and port that might be this pod's sshd, most likely first. `ssh_resolver`
        (an argv with `{pod}` that prints "<ip> <port>", e.g. the runpod-spinup skill's
        `_resolve_ssh.py`) gives one; otherwise the API's port and its neighbours (it has been
        off by one) are probed. A port that answers is not yet this pod: `_wait_ssh` checks."""
        resolver = self.config.get("ssh_resolver")
        if resolver:
            done = self.shell([str(a).format(pod=pod_id) for a in resolver])
            parts = done.stdout.split()
            if done.returncode == 0 and len(parts) == 2 and parts[1].isdigit():
                self._api_endpoint[pod_id] = (parts[0], int(parts[1]))
                return [(parts[0], int(parts[1]))]
            raise RuntimeError(f"ssh_resolver gave nothing usable for {pod_id}")
        done = self.shell(["runpodctl", "pod", "get", pod_id])
        try:
            info = json.loads(done.stdout)["ssh"]
            ip, port = info.get("ip"), info.get("port")
            if not (ip and port):
                found = re.search(r"@(\S+)\s+-p\s+(\d+)", info.get("ssh_command", ""))
                ip, port = (found.group(1), int(found.group(2))) if found else (None, None)
        except (ValueError, KeyError, TypeError, AttributeError):
            ip = port = None
        if not (ip and port):
            raise RuntimeError(f"no ssh endpoint in `runpodctl pod get {pod_id}`")
        self._api_endpoint[pod_id] = (str(ip), int(port))
        near = [int(port) + d for d in (0, -1, 1, -2, 2)]
        found_ports = [(str(ip), p) for p in near if self.probe(str(ip), p)]
        if not found_ports:
            raise RuntimeError(f"nothing listens near {ip}:{port}")
        return found_ports

    def resolve_ssh(self, pod_id: str) -> tuple[str, int]:
        """The first candidate endpoint (identity is checked separately, in `_wait_ssh`)."""
        return self.ssh_candidates(pod_id)[0]

    def _is_this_pod(self, prefix: list[str], pod_id: str, endpoint: tuple[str, int]) -> bool:
        """The host proves it is `pod_id`. RunPod writes `RUNPOD_POD_ID` into
        `/etc/rp_environment` (a non-interactive ssh does not always carry it in its own
        environment), so that file is sourced first. A host that reports no id at all is
        accepted only on the exact ip and port `runpodctl pod get` gave; a different id, or a
        neighbouring port with no id, is refused, so no other pod ever gets this job's secrets."""
        probe = '. /etc/rp_environment 2>/dev/null; echo "${RUNPOD_POD_ID:-}"'
        done = self.shell([*prefix, "bash", "-c", shlex.quote(probe)])
        if done.returncode != 0:
            return False
        seen = done.stdout.strip()
        if seen:
            return seen == pod_id
        return endpoint == self._api_endpoint.get(pod_id)

    def _wait_ssh(self, pod_id: str) -> list[str]:
        """The ssh prefix of a host that proves it is this pod; raises after `ssh_wait_s`."""
        limit = float(self.config.get("ssh_wait_s", 900))
        start, last = self.clock(), "not tried"
        while True:
            try:
                for ip, port in self.ssh_candidates(pod_id):
                    prefix = self.ssh_prefix(ip, port)
                    if self._is_this_pod(prefix, pod_id, (ip, port)):
                        return prefix
                    last = (
                        f"{ip}:{port} refused ssh, reported another pod id, or reported none and "
                        f"is not the API's endpoint for {pod_id}"
                    )
            except RuntimeError as exc:
                last = str(exc)
            if self.clock() - start > limit:
                raise RuntimeError(f"no ssh to {pod_id} after {limit:.0f}s: {last}")
            self.sleep(self.poll_s)

    def _stop(self, pod_id: str, why: str, record: Path) -> None:
        """STOP, not delete, when the outputs are in doubt: GPU billing ends and the data stays.
        The pod's disk still bills, so it stays on the ledger as live (blocking strict steps)
        until you inspect it, delete it and run `ledger pod-gone`."""
        done = self.shell(["runpodctl", "pod", "stop", pod_id])
        if done.returncode == 0:
            self._event(record, "stopped", why=why)
        state = "stopped (disk kept, still billing for it)" if done.returncode == 0 else (
            "NOT stopped, may still be billing in full")  # fmt: skip
        print(f"POD {pod_id} {state}: {why}. UNSETTLED: inspect it, delete it, then "
              "`python -m orchestrate ledger pod-gone`.", flush=True)  # fmt: skip

    # ------------------------------------------------------------------- run

    def run(self, job: Job) -> JobResult:
        name = f"{self.config['name_prefix']}-{job.step}-{job.attempt}".lower().replace("_", "-")
        record = self.pods_dir / f"{job.step}-{job.attempt}.json"
        self.pods_dir.mkdir(parents=True, exist_ok=True)
        rate = job.rate_usd_per_hour
        state: dict[str, Any] = {
            "name": name,
            "step": job.step,
            "attempt": job.attempt,
            "rate_usd_per_hour": rate,
            "max_usd": job.max_usd,
            "kill_grace_s": job.kill_grace_s,
            "status": "intent",
        }

        def save() -> None:
            record.write_text(json.dumps(state), encoding="utf-8")

        save()
        secrets = self.secrets()
        if "HF_TOKEN" not in secrets:
            print("REFUSING a pod job: no HF_TOKEN (env or `hf auth login`); the pod could not "
                  "pull its inputs or push its outputs.")  # fmt: skip
            state["status"] = "not_created"
            save()
            return JobResult(False, 2, usd=0.0, usd_source="no pod created")
        try:
            self.sync.push_inputs(name, job.inputs)
        except Exception:
            state["status"] = "not_created"
            save()
            raise
        done = self.shell(self.create_command(name, job))
        timed_out = done.returncode == 124
        if done.returncode and not timed_out:
            print(done.stderr, file=sys.stderr)
            state["status"] = "create_failed"
            save()
            return JobResult(False, done.returncode, usd=0.0, usd_source="create failed")
        try:
            if timed_out:
                raise ValueError("create timed out")
            pod_id = json.loads(done.stdout)["id"]
        except (ValueError, KeyError, TypeError):
            # A success we could not read, or a timeout: a pod probably exists. Keep it tracked.
            state.update(
                status="create_timeout" if timed_out else "create_unparsed",
                created_at=time.time(),
                stdout=done.stdout[:500],
            )
            save()
            what = "timed out" if timed_out else "succeeded but was not understood"
            print(f"pod create {what}. A pod named {name} may be running and billing: find it "
                  "with `runpodctl pod list`.", flush=True)  # fmt: skip
            return JobResult(False, 124 if timed_out else 3, usd=None, usd_source="create unknown")
        state.update(id=pod_id, created_at=time.time(), status="created")
        save()
        if job.deadline_s is not None:
            self.spawn_watchdog(record)
        try:
            prefix = self._wait_ssh(pod_id)
            state["ssh_prefix"] = prefix
            save()
            if self.shell([*prefix, f"umask 077; cat > {ENV_FILE}"], input=_env_file(secrets)
                          ).returncode:  # fmt: skip
                raise RuntimeError("could not place the secrets file on the pod")
            if self.shell([*prefix, f"umask 077; cat > {SCRIPT_FILE}"],
                          input=self.bootstrap(job, name)).returncode:  # fmt: skip
                raise RuntimeError("could not place the job script on the pod")
            started = self.shell([*prefix, "tmux", "new-session", "-d", "-s", "job",
                                  f"bash {SCRIPT_FILE}"])  # fmt: skip
            if started.returncode:
                raise RuntimeError(f"tmux did not start the job: {started.stderr.strip()[:200]}")
        except RuntimeError as exc:
            self._stop(pod_id, str(exc), record)
            return self._result(state, rate, 255, "job never started")
        rc = self._wait_rc(prefix, job)
        if rc is None:
            self._stop(
                pod_id, "no exit code before the wait limit; the job may be unfinished", record
            )
            return self._result(state, rate, 124, "wait limit")
        if rc != 0:
            # Read what the failed job left (its log, whatever outputs exist), then delete the
            # pod: a stopped or forgotten pod bills for its disk. `keep_failed_pod: true` opts
            # out. Whatever happens, a pod still up stays on the ledger as live.
            self._salvage(name, job)
            if self.config.get("keep_failed_pod"):
                print(
                    f"pod {pod_id} KEPT after rc={rc} (keep_failed_pod); it is billing and "
                    "blocks --strict-budget until you delete it and run `ledger pod-gone`."
                )
            else:
                self._delete(pod_id, record)
            return self._result(state, rate, rc, "job failed")
        if not job.outputs and not job.produces_nothing:
            self._stop(
                pod_id, "the job declares no outputs, so success cannot be verified", record
            )
            return self._result(state, rate, RC_UPLOAD_FAILED, "no declared outputs")
        try:
            self.sync.pull_outputs(name, job.outputs)
            missing = [str(p) for p in job.outputs if not _present(p)]
            if missing:
                raise RuntimeError(f"outputs not present after the pull: {missing}")
        except Exception as exc:  # noqa: BLE001  # whatever failed, the pod must not be deleted
            self._stop(pod_id, f"outputs not verified ({exc})", record)
            return self._result(state, rate, RC_UPLOAD_FAILED, "outputs unverified")
        self._delete(pod_id, record)
        return self._result(state, rate, 0, "done")

    def _event(self, record: Path, name: str, **extra: Any) -> None:
        with record.with_suffix(".events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"event": name, "at": time.time(), **extra}) + "\n")

    def _delete(self, pod_id: str, record: Path) -> bool:
        """Delete the pod. Only a delete that worked is recorded as terminated: a failed one
        leaves the pod on the ledger as live, which blocks `--strict-budget` steps and is
        counted at its rate until `ledger pod-gone`."""
        gone = self.shell(["runpodctl", "pod", "delete", pod_id])
        if gone.returncode:
            print(f"POD {pod_id} NOT DELETED and is billing: run `runpodctl pod delete {pod_id}`, "
                  "then `python -m orchestrate ledger pod-gone`.", flush=True)  # fmt: skip
            return False
        self._event(record, "terminated")
        return True

    def _salvage(self, name: str, job: Job) -> None:
        """Best effort: the failed job's logs and any outputs that exist, before deleting."""
        for fetch in (getattr(self.sync, "pull_logs", None),):
            try:
                if fetch and job.log:
                    fetch(name, job.log.parent, job.step)
            except Exception as exc:  # noqa: BLE001
                print(f"  could not read the pod's logs ({type(exc).__name__}: {exc})")
        try:
            self.sync.pull_outputs(name, job.outputs)
        except Exception as exc:  # noqa: BLE001  # some outputs may not exist: that is expected
            print(f"  not every output of the failed job was recovered ({exc})")

    def _result(self, state: dict[str, Any], rate: float | None, rc: int, why: str) -> JobResult:
        wall = time.time() - state["created_at"]
        usd = wall / 3600 * rate if rate else None
        return JobResult(rc == 0, rc, wall, usd, "billed_s x listed rate" if usd else "unknown")

    def _wait_rc(self, prefix: list[str], job: Job) -> int | None:
        """The job's exit code, or None past the limit (`wait_timeout_s`, default 12 h, or the
        job's own deadline plus the kill grace). An ssh that fails keeps trying until then."""
        limit = float(self.config.get("wait_timeout_s", 12 * 3600))
        if job.deadline_s is not None:
            limit = job.deadline_s + float(self.config.get("kill_grace_s", 300)) + 600
        start = self.clock()
        while True:
            done = self.shell([*prefix, "cat", RC_FILE])
            if done.returncode == 0 and done.stdout.strip().lstrip("-").isdigit():
                return int(done.stdout.strip())
            if self.clock() - start > limit:
                return None
            self.sleep(self.poll_s)


def _env_file(secrets: dict[str, str]) -> str:
    return "".join(f"export {k}={shlex.quote(v)}\n" for k, v in secrets.items())


def _present(path: Path) -> bool:
    return any(path.iterdir()) if path.is_dir() else path.exists()


class HubSync:
    """Moves a job's files through the build's private scratch repo. Never passes `public`."""

    def __init__(self, repo: str) -> None:
        self.repo = repo

    def push_inputs(self, name: str, paths: Sequence[Path]) -> None:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(self.repo, repo_type="dataset", private=True, exist_ok=True)
        for path in paths:
            target = f"{name}/inputs/{str(path.resolve()).lstrip('/')}"
            if path.is_dir():
                api.upload_folder(
                    repo_id=self.repo,
                    repo_type="dataset",
                    folder_path=str(path),
                    path_in_repo=target,
                )
            elif path.exists():
                api.upload_file(
                    repo_id=self.repo,
                    repo_type="dataset",
                    path_or_fileobj=str(path),
                    path_in_repo=target,
                )
            else:
                raise FileNotFoundError(f"required input missing, nothing sent: {path}")

    def pull_logs(self, name: str, dest: Path, step: str) -> None:
        """The pod's job and setup logs, saved beside the local logs as `<step>.pod-*.log`."""
        from huggingface_hub import snapshot_download

        prefix = f"{name}/outputs/root/"
        local = Path(
            snapshot_download(self.repo, repo_type="dataset", allow_patterns=[f"{prefix}*.log"])
        )
        dest.mkdir(parents=True, exist_ok=True)
        for log in (local / prefix).glob("mobench-*.log"):
            shutil.copy2(log, dest / f"{step}.pod-{log.name}")

    def pull_outputs(self, name: str, paths: Sequence[Path]) -> None:
        """Back to each output's own path: `orchestrate.sync push` keeps it under `outputs/`.

        Raises when any named output is not on the Hub, so a silent no-match is never a pass.
        """
        from huggingface_hub import snapshot_download

        prefix = f"{name}/outputs/"
        local = Path(
            snapshot_download(self.repo, repo_type="dataset", allow_patterns=[f"{prefix}*"])
        )
        missing = []
        for path in paths:
            source = local / prefix / str(path.resolve()).lstrip("/")
            if source.is_dir() and any(source.rglob("*")):
                shutil.copytree(source, path, dirs_exist_ok=True)
            elif source.is_file():
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, path)
            else:
                missing.append(str(path))
        if missing:
            raise FileNotFoundError(f"not on the Hub under {prefix}: {missing}")


def _spawn_watchdog(record: Path) -> None:
    """Detached, so the cap holds even if this session dies."""
    subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "orchestrate.watchdog", str(record)],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
