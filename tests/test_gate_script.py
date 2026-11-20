"""`scripts/gate.sh`'s failure paths — the ones a green run never exercises.

A gate that reports GREEN when a step did not run is worse than no gate, and the two
ways that can happen are both invisible to an end-to-end run: a step killed before it
writes its exit status, and a cheap check bailing out before the table is printed. Both
are pinned here by SOURCING the real script (it splits definitions from `gate_main`) and
driving the functions directly, so these tests cannot drift from the shipped code the way
a re-implementation of its control flow would.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

GATE = Path(__file__).resolve().parent.parent / "scripts" / "gate.sh"

# WINDOWS IS SKIPPED, AND THE SKIP IS THE FIX FOR A CORRUPTION, not a convenience.
# Every test here interpolates a `pathlib` path into a bash command string. On Windows
# those render with backslashes, which bash keeps as ordinary filename characters: it
# cannot `source` the script, so these do not reach the code under test,
# and `test_reset_root_follows_symlinks_to_the_real_repo` additionally ran
# `ln -s "$PWD" "<literal windows path>"`, which landed a link NAMED after the whole
# temp path in the checkout root. Windows pytest then aborted every later collection in
# that tree with `OSError: [WinError 1920]` reading it -- including runs that had nothing
# to do with the gate.
#
# THEY HANG, THEY DO NOT FAIL FAST, and an earlier version of this comment said otherwise.
# Measured 9 Sep on this Windows host: with the skip lifted, the file produced no output
# and had to be killed at 60s. A review suggested narrowing the skip so that
# `test_the_step_log_directory_template_actually_makes_a_directory` -- which interpolates
# no path and only greps a line out of the script -- would keep running here. That was
# tried and it hangs too, so the mark stays module-wide. Narrow it only with a measurement,
# not by reading the test. `scripts/gate.sh` only ever runs on Linux/macOS (docs/COMMANDS.md),
# so nothing is lost where it runs: on those platforms every assertion below still runs
# unchanged.
pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="drives scripts/gate.sh through bash; Windows paths are not bash paths",
)


ENV_SH = GATE.parent / "env.sh"


def test_env_sh_exports_the_copy_link_mode() -> None:
    """The #212 fix is one `export` in a file nothing was obliged to source.

    Nothing pinned it, so a later edit could drop the line and every venv would silently go
    back to hardlinking into the shared uv cache -- the corruption this is here to prevent,
    reappearing with no test going red. Runs the real file rather than grepping it, because
    the value is set with `${UV_LINK_MODE:-copy}` and a grep cannot tell that apart from a
    line that is never reached.
    """
    out = subprocess.run(
        ["bash", "-c", f'unset UV_LINK_MODE; source "{ENV_SH}"; echo "$UV_LINK_MODE"'],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "copy", out.stderr


def test_env_sh_exports_omp_num_threads() -> None:
    """libgomp spawns a thread pool per process; on a many-core pod that blows the container's
    cgroup pids.max and SIGABRTs interp-engine work. Nothing else pins this default, so a
    later edit could drop the line with no test going red. Runs the real file, and also
    checks a caller's own value survives (setdefault, not override)."""
    out = subprocess.run(
        ["bash", "-c", f'unset OMP_NUM_THREADS; source "{ENV_SH}"; echo "$OMP_NUM_THREADS"'],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "1", out.stderr

    out_override = subprocess.run(
        ["bash", "-c", f'OMP_NUM_THREADS=16; source "{ENV_SH}"; echo "$OMP_NUM_THREADS"'],
        capture_output=True,
        text=True,
    )
    assert out_override.stdout.strip() == "16", out_override.stderr


def test_the_gate_sources_env_sh_before_it_spawns_anything() -> None:
    """Setting the link mode is useless if the gate never reads it.

    `scripts/gate.sh` spawns three `uv run` processes in parallel -- pytest, the sweep and
    byte identity -- which is precisely the concurrency `UV_LINK_MODE` exists for, and only
    the sweep sourced `env.sh` (through `run_all.sh`). The other two used the default mode.
    Asserted on ORDER, not just presence: a source after the spawns would set nothing.
    """
    body = GATE.read_text(encoding="utf-8")
    sourced = body.index("env.sh")
    first_spawn = body.index("spawn pytest")
    assert sourced < first_spawn, "gate.sh sources env.sh only after it has spawned workers"


def run_bash(script: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Source gate.sh (definitions only — `gate_main` is not called) and run `script`."""
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{GATE}"\nLOGS="{tmp_path}"\n{script}'],
        capture_output=True,
        text=True,
    )


def test_a_step_that_leaves_no_exit_status_is_failed_not_inherited(tmp_path: Path) -> None:
    """The regression that mattered: `code`/`secs` were not `local` in `reap`.

    A step whose subshell is killed before writing its `.rc` left the PREVIOUS reap's
    values in place, so a killed sweep was recorded with the passing step's rc=0 and the
    gate printed GREEN having never swept. Reaped here in the same order the gate uses.
    """
    # `pid` is a completed process, so `wait` returns immediately; only the .rc matters.
    out = run_bash(
        'printf "0 5\\n" >"$LOGS/first.rc"\n'
        "true & reap first $!\n"
        "true & reap second $!\n"  # no second.rc — the killed-step case
        "report\n",
        tmp_path,
    )
    assert out.returncode == 1, out.stdout
    assert "GATE RED" in out.stdout
    assert "second left no exit status" in out.stdout


def test_an_empty_exit_status_file_is_also_failed(tmp_path: Path) -> None:
    """`[[ "" -eq 0 ]]` is TRUE in bash arithmetic, so an empty `.rc` — killed between
    create and write, or a full disk — passed the old numeric test."""
    out = run_bash(': >"$LOGS/only.rc"\ntrue & reap only $!\nreport\n', tmp_path)
    assert out.returncode == 1, out.stdout
    assert "GATE RED" in out.stdout


def test_the_table_is_printed_even_when_a_cheap_check_bails(tmp_path: Path) -> None:
    """`step ... || exit 1` skipped the timing table and the verdict, which
    docs/COMMANDS.md promises. `bail` reports first."""
    out = run_bash("record lint 1 1\nbail\n", tmp_path)
    assert out.returncode == 1
    assert "GATE RED" in out.stdout
    assert "lint" in out.stdout


def test_reset_root_refuses_a_root_inside_the_repo(tmp_path: Path) -> None:
    """`rm -rf "$ROOT"` on a caller-supplied variable. A gate root pointed at the
    checkout would delete the working tree."""
    out = run_bash('ROOT="$PWD/results"\nreset_root\n', tmp_path)
    assert out.returncode == 2
    assert "inside the repo" in out.stderr


def test_reset_root_refuses_a_tree_holding_real_results(tmp_path: Path) -> None:
    """A real (`fake: false`) tree is usually untracked and unbacked-up."""
    cell = tmp_path / "root" / "abc123"
    cell.mkdir(parents=True)
    (cell / "spec.json").write_text('{"fake": false}')
    out = run_bash(f'ROOT="{tmp_path}/root"\nreset_root\n', tmp_path)
    assert out.returncode == 2
    assert "REAL results" in out.stderr
    assert cell.exists(), "refused, and did not delete anything on the way out"


def test_reset_root_clears_an_ordinary_fake_tree(tmp_path: Path) -> None:
    """The normal path still works: a fresh root every run, or the first sweep comes
    back all-cached and proves nothing."""
    cell = tmp_path / "root" / "abc123"
    cell.mkdir(parents=True)
    (cell / "spec.json").write_text('{"fake": true}')
    out = run_bash(f'ROOT="{tmp_path}/root"\nreset_root\n', tmp_path)
    assert out.returncode == 0, out.stderr
    assert not (tmp_path / "root").exists()


def test_a_garbage_exit_status_is_failed(tmp_path: Path) -> None:
    """An unparseable `.rc` — a truncated write, an interleaved one — must fail closed.

    This is what the `[[ $code =~ ^[0-9]+$ ]]` guard is for, and nothing else pins it: a
    non-numeric string is 0 in bash arithmetic context, so `[[ notanumber -eq 0 ]]` is
    TRUE and the step would be recorded as having passed.
    """
    out = run_bash(
        'printf "notanumber 5\\n" >"$LOGS/only.rc"\ntrue & reap only $!\nreport\n', tmp_path
    )
    assert out.returncode == 1, out.stdout
    assert "GATE RED" in out.stdout


def test_reset_root_refuses_a_root_that_contains_the_checkout(tmp_path: Path) -> None:
    """`MO_EVALS_GATE_ROOT=$HOME` was accepted by the first version of this guard, which
    only looked for roots INSIDE the repo. `rm -rf` on an ancestor takes the checkout,
    every sibling worktree and every untracked results tree with it."""
    for root in ("$HOME", "/", "$(cd .. && pwd -P)"):
        out = run_bash(f'ROOT="{root}"\nreset_root\n', tmp_path)
        assert out.returncode == 2, f"{root} was not refused: {out.stdout}{out.stderr}"
        assert "REFUSING" in out.stderr


def test_reset_root_refuses_a_real_tree_nested_deeper_than_one_level(tmp_path: Path) -> None:
    """The organism/seed layout puts `spec.json` two levels down, and a parent-of-roots
    puts it deeper still. A `<root>/*/spec.json` glob sees neither."""
    cell = tmp_path / "root" / "run1" / "impulsive__prompting__scale1__ab" / "seed0"
    cell.mkdir(parents=True)
    (cell / "spec.json").write_text('{"fake": false}')
    out = run_bash(f'ROOT="{tmp_path}/root"\nreset_root\n', tmp_path)
    assert out.returncode == 2, out.stdout
    assert cell.exists()


def test_reset_root_follows_symlinks_to_the_real_repo(tmp_path: Path) -> None:
    """`pwd` is logical, so a root reached through a symlinked parent used to compare as a
    different tree and slip past the in-repo check."""
    link = tmp_path / "link"
    out = run_bash(f'ln -s "$PWD" "{link}"\nROOT="{link}/results"\nreset_root\n', tmp_path)
    assert out.returncode == 2, out.stdout
    assert "inside the repo" in out.stderr


def test_the_step_log_directory_template_actually_makes_a_directory() -> None:
    """`mktemp -d -t mo-gate-logs` (no X's) fails on GNU mktemp, so LOGS came out EMPTY.

    Every step log then landed at the filesystem root — `/pytest.log`, `/sweep.log`,
    `/byte-identity.log` — which two worktrees running the gate at once SHARE, so `bail`'s
    "tail of the failing step" could print a different run's output. Nothing caught it
    because `gate_main` is the one function these tests never call and `run_bash` sets
    `LOGS` itself. So run the real line out of the real script, rather than a copy of it.
    """
    line = next(
        ln.strip() for ln in GATE.read_text().splitlines() if ln.strip().startswith("LOGS=")
    )
    out = subprocess.run(
        ["bash", "-c", f'set -euo pipefail\n{line}\n[[ -d "$LOGS" ]] && rmdir "$LOGS"'],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, f"{line!r} did not create a directory: {out.stderr}"


def _cpu_quota_workers(
    tmp_path: Path,
    *,
    cpu_max: str | None,
    nproc: int,
    mem_max: str | None = None,
    pids_max: str | None = None,
    cpu_v1_dir: Path | None = None,
    fn: str = "_gate_cpu_quota_workers",
    extra_env: str = "",
) -> subprocess.CompletedProcess[str]:
    """Drive `_gate_cpu_quota_workers` with the host's real `cpu.max`, `memory.max`,
    `pids.max` and `nproc` swapped out for test-controlled ones, via the seams
    `_MO_GATE_CPU_MAX`/`_MO_GATE_MEM_MAX`/`_MO_GATE_PIDS_MAX`/`_MO_GATE_NPROC` (unset in
    normal use -- see the comments above the functions in scripts/gate.sh).

    `mem_max`/`pids_max` default to a nonexistent path (unreadable -> no candidate from
    that source), never to the real host's cgroup files, so a test that isn't exercising
    the memory/pids bound can't pick up whatever this host happens to have.

    Every real sysfs file (`/sys/fs/cgroup/{memory,pids}.max`) ends in a newline; `read`
    without one returns non-zero at EOF and the `&&`-chained regex check never runs, so a
    fixture missing it would silently look like a fixture without that source at all.
    Append one here rather than trust every call site to remember it.
    """
    env_setup = f"export _MO_GATE_NPROC={nproc}\n"
    # The v1 fallback reads /sys/fs/cgroup/cpu/ -- present on some pods -- so pin it
    # away unless a test passes its own.
    env_setup += f'export _MO_GATE_CPU_V1_DIR="{cpu_v1_dir or "/nonexistent/cpu"}"\n'
    for name, value, filename in (
        ("_MO_GATE_CPU_MAX", cpu_max, "cpu.max"),
        ("_MO_GATE_MEM_MAX", mem_max, "memory.max"),
        ("_MO_GATE_PIDS_MAX", pids_max, "pids.max"),
    ):
        if value is not None:
            f = tmp_path / filename
            f.write_text(value if value.endswith("\n") else value + "\n")
            env_setup += f'export {name}="{f}"\n'
        else:
            env_setup += f'export {name}="/nonexistent/{filename}"\n'
    return subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{GATE}"\n{env_setup}{extra_env}{fn}'],
        capture_output=True,
        text=True,
    )


def test_cpu_quota_narrower_than_nproc_is_ceiled_and_used(tmp_path: Path) -> None:
    """765000/100000 = 7.65 cores on a 96-core host -> 8 workers, not 96 (the incident)."""
    out = _cpu_quota_workers(tmp_path, cpu_max="765000 100000\n", nproc=96)
    assert out.stdout.strip() == "8", out.stderr


def test_unlimited_quota_falls_back_to_nproc(tmp_path: Path) -> None:
    """No memory/pids bound in play (both unreadable) and nproc(8) is under the hard
    ceiling, so the small-nproc host still gets nproc workers."""
    out = _cpu_quota_workers(tmp_path, cpu_max="max 100000\n", nproc=8)
    assert out.stdout.strip() == "8", out.stderr


def test_missing_cgroup_file_falls_back_to_nproc(tmp_path: Path) -> None:
    """A missing cpu.max is treated exactly like an unlimited quota, not skipped."""
    out = _cpu_quota_workers(tmp_path, cpu_max=None, nproc=4)
    assert out.stdout.strip() == "4", out.stderr


def test_quota_wider_than_nproc_is_capped_at_nproc(tmp_path: Path) -> None:
    """A quota bigger than the host (e.g. a mis-set cgroup) must never ask for MORE
    workers than the host advertises."""
    out = _cpu_quota_workers(tmp_path, cpu_max="800000 100000\n", nproc=4)
    assert out.stdout.strip() == "4", out.stderr


def test_quota_equal_to_nproc_is_trusted_not_capped(tmp_path: Path) -> None:
    """Regression for the defect introduced alongside the no-quota cap: a real finite
    quota whose ceil(quota/period) equals nproc exactly (workers < nproc_n is false) must
    still be trusted as-is (64), not fall through to the capped no-quota path (32)."""
    out = _cpu_quota_workers(tmp_path, cpu_max="6400000 100000\n", nproc=64)
    assert out.stdout.strip() == "64", out.stderr


def test_quota_exceeding_nproc_by_a_lot_is_still_min_of_the_two(tmp_path: Path) -> None:
    """A real finite quota bigger than nproc must give nproc (256) exactly -- the old
    min(ceil(quota/period), nproc) behaviour -- not the capped no-quota value (32)."""
    out = _cpu_quota_workers(tmp_path, cpu_max="25600000 100000\n", nproc=256)
    assert out.stdout.strip() == "256", out.stderr


def test_quota_path_ignores_memory_and_pids(tmp_path: Path) -> None:
    """The set-quota path (§ existing behaviour) must not change: a real cgroup quota
    still wins on its own, even when memory/pids would bind tighter if consulted."""
    out = _cpu_quota_workers(
        tmp_path,
        cpu_max="765000 100000\n",  # ceil -> 8, unchanged from the first test
        nproc=96,
        mem_max=str(4 * 1024**3),  # would cap at 2 workers if this path used it
        pids_max="64",  # would cap at 1 worker if this path used it
    )
    assert out.stdout.strip() == "8", out.stderr


def test_unquota_high_nproc_and_huge_memory_is_capped_not_256(tmp_path: Path) -> None:
    """The actual incident: cpu.max=max, nproc=256 (RunPod reports the HOST core count
    inside an unprivileged cgroup), 116 GiB memory. 116/2=58, still short of 256, so the
    hard ceiling must be what binds -- must not print anything close to 256."""
    out = _cpu_quota_workers(
        tmp_path, cpu_max="max 100000\n", nproc=256, mem_max=str(116 * 1024**3)
    )
    workers = int(out.stdout.strip())
    assert workers <= 32, out.stdout


def test_unquota_missing_cpu_max_256_nproc_125gib_memory_is_capped(tmp_path: Path) -> None:
    """The real starved gate pod: vcpuCount 32, memoryInGb 125, cpu.max ABSENT (not
    "max"), nproc=256. Memory alone would not save this (125/2=62 > 32): CPU
    oversubscription was the actual failure (load 180-220 on 32 real vCPUs, sshd
    starved), not OOM. The hard ceiling must be <= the realistic vCPU allocation."""
    out = _cpu_quota_workers(tmp_path, cpu_max=None, nproc=256, mem_max=str(125 * 1024**3))
    workers = int(out.stdout.strip())
    assert workers <= 32, out.stdout


def test_unquota_small_memory_is_memory_bound(tmp_path: Path) -> None:
    """4 GiB / 2 GiB-per-worker budget -> 2 workers, tighter than the hard ceiling or a
    generous nproc/pids."""
    out = _cpu_quota_workers(tmp_path, cpu_max="max 100000\n", nproc=64, mem_max=str(4 * 1024**3))
    assert out.stdout.strip() == "2", out.stderr


def test_unquota_small_pids_max_is_pids_bound(tmp_path: Path) -> None:
    """pids.max present and tight (the other RunPod pods had pids.max=5120) ->
    pids.max/64 binds tighter than memory or the ceiling."""
    out = _cpu_quota_workers(
        tmp_path,
        cpu_max="max 100000\n",
        nproc=64,
        mem_max=str(999 * 1024**3),  # not binding
        pids_max="640",
    )
    assert out.stdout.strip() == "10", out.stderr  # 640 // 64


def test_the_gate_ignores_the_interactive_xdist_cap_on_a_big_pod(tmp_path: Path) -> None:
    """A shell profile may set PYTEST_XDIST_AUTO_NUM_WORKERS=2 for interactive sessions. On a
    27-core pod the gate sizes itself from the cgroup quota instead, capped at 16 (147 s
    there, vs 714 s at 2 workers on a 2-vCPU gate pod, 2026-09-29)."""
    out = _cpu_quota_workers(
        tmp_path,
        cpu_max="2720000 100000\n",
        nproc=256,
        fn="_gate_pytest_workers",
        extra_env="export PYTEST_XDIST_AUTO_NUM_WORKERS=2\n",
    )
    assert out.stdout.strip() == "16", out.stderr


def test_a_small_pod_keeps_the_interactive_xdist_cap(tmp_path: Path) -> None:
    """On a 4-core box the cap is an OOM guard (sweep ~10.6 GB beside pytest on 16 GB)."""
    out = _cpu_quota_workers(
        tmp_path,
        cpu_max="400000 100000\n",
        nproc=256,
        fn="_gate_pytest_workers",
        extra_env="export PYTEST_XDIST_AUTO_NUM_WORKERS=2\n",
    )
    assert out.stdout.strip() == "2", out.stderr


def test_a_no_quota_small_memory_pod_keeps_the_interactive_xdist_cap(tmp_path: Path) -> None:
    """cpu3m x2 gate pods: `cpu.max = max`, 192 host cpus, 16 GB. The no-quota path gives 8
    (16 GiB / 2 GiB), the 2026-09-08 OOM configuration; the cap must hold (#374 review)."""
    out = _cpu_quota_workers(
        tmp_path,
        cpu_max="max 100000\n",
        nproc=192,
        mem_max=str(16 * 1024**3),
        fn="_gate_pytest_workers",
        extra_env="export PYTEST_XDIST_AUTO_NUM_WORKERS=2\n",
    )
    assert out.stdout.strip() == "2", out.stderr


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_gate_full_env_accepts_the_repo_truthy_spellings(value: str) -> None:
    line = next(ln.strip() for ln in GATE.read_text().splitlines() if "MO_EVALS_GATE_FULL:-" in ln)
    out = subprocess.run(
        ["bash", "-c", f"full=0\nMO_EVALS_GATE_FULL={value}\n{line}\necho $full"],
        capture_output=True,
        text=True,
    )
    assert out.stdout.strip() == "1", out.stderr


def test_the_gate_worker_override_wins_outright(tmp_path: Path) -> None:
    out = _cpu_quota_workers(
        tmp_path,
        cpu_max="400000 100000\n",
        nproc=256,
        fn="_gate_pytest_workers",
        extra_env="export PYTEST_XDIST_AUTO_NUM_WORKERS=2 MO_EVALS_GATE_PYTEST_WORKERS=7\n",
    )
    assert out.stdout.strip() == "7", out.stderr


def test_cgroup_v1_quota_is_read_when_cpu_max_is_absent(tmp_path: Path) -> None:
    """A cgroup-v1 pod: no cpu.max, a 27.2-core quota in cpu.cfs_quota_us -> 28."""
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("2720000\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    out = _cpu_quota_workers(tmp_path, cpu_max=None, nproc=256, cpu_v1_dir=v1)
    assert out.stdout.strip() == "28", out.stderr


def test_an_unlimited_v1_quota_falls_through(tmp_path: Path) -> None:
    v1 = tmp_path / "cpu"
    v1.mkdir()
    (v1 / "cpu.cfs_quota_us").write_text("-1\n")
    (v1 / "cpu.cfs_period_us").write_text("100000\n")
    out = _cpu_quota_workers(tmp_path, cpu_max=None, nproc=6, cpu_v1_dir=v1)
    assert out.stdout.strip() == "6", out.stderr


def _venv_path(cwd: Path, env: str) -> str:
    out = subprocess.run(
        ["bash", "-c", f'set -uo pipefail\nsource "{GATE}"\n{env}_gate_venv_path'],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_repo_mode_and_off_volume_checkouts_keep_the_checkout_venv(tmp_path: Path) -> None:
    assert _venv_path(tmp_path, "export MO_EVALS_GATE_VENV=repo\n") == ""
    # tmp_path is not under the network-volume mount, so `auto` leaves the venv alone there too.
    assert _venv_path(tmp_path, "unset MO_EVALS_GATE_VENV UV_PROJECT_ENVIRONMENT\n") == ""


def test_an_explicit_gate_venv_path_is_used_verbatim(tmp_path: Path) -> None:
    assert _venv_path(tmp_path, "export MO_EVALS_GATE_VENV=/x/y\n") == "/x/y"


def test_lean_marks_the_sweep_family_skipped_and_says_run_full(tmp_path: Path) -> None:
    """A lean run must never print a bare GATE GREEN: it tells you what it did not run."""
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -uo pipefail\nsource "{GATE}"\nLOGS="{tmp_path}"; ROOT="{tmp_path}/r"\n'
            "record pytest 3 0; record sweep - skip; report",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "skipped (lean" in out.stdout
    assert "GATE GREEN (LEAN)" in out.stdout and "--full" in out.stdout


def test_a_failure_beside_a_skip_is_still_red(tmp_path: Path) -> None:
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -uo pipefail\nsource "{GATE}"\nLOGS="{tmp_path}"; ROOT="{tmp_path}/r"\n'
            "record pytest 3 1; record sweep - skip; report",
        ],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 1 and "GATE RED" in out.stdout


def test_an_unknown_gate_argument_is_refused() -> None:
    out = subprocess.run(
        ["bash", "-c", f'source "{GATE}"\ngate_main --fulll'], capture_output=True, text=True
    )
    assert out.returncode == 2 and "unknown argument" in out.stderr


def test_the_cpu_probe_ignores_omp_num_threads(tmp_path: Path) -> None:
    """`scripts/env.sh` exports `OMP_NUM_THREADS=1`, and GNU `nproc` RETURNS that value.

    Every gate run on a pod sources `env.sh` first, so `nproc` reported 1 on a 96-core
    host and `min(quota, nproc)` collapsed the gate to a single pytest worker (observed
    2026-09-16, a gate pod: pytest 380 s instead of ~60 s). The probe must
    read the machine, not the OpenMP setting, so its answer cannot move with that export.
    """
    import os

    expected = str(os.cpu_count() or 1)
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'set -uo pipefail\nsource "{GATE}"\nexport OMP_NUM_THREADS=1\n_gate_online_cpus',
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == expected, (
        f"OMP_NUM_THREADS=1 changed the probe to {out.stdout.strip()!r}; "
        f"the host has {expected} online CPUs"
    )
