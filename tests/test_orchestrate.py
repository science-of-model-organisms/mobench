"""The build orchestrator with no GPU, no key and no network. Plumbing only, never a result."""

from __future__ import annotations

import json
import signal
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mobench import cli
from orchestrate import ledger as lm
from orchestrate.build import State, is_scratch
from orchestrate.build import main as build_main
from orchestrate.executors import FakeExecutor, Job, JobResult, LocalExecutor, RunPodExecutor
from orchestrate.fake import run_build
from orchestrate.ledger import Budget, BudgetStop, Ledger
from orchestrate.register import fill_from_build
from orchestrate.steps import Ctx, StepStop, check_constitution, fingerprint, manifest, pilot_gate
from orchestrate.watchdog import watch


@pytest.fixture(autouse=True)
def one_gpu_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """A real teacher step reads the GPUs it can see (it stops before launch with none); these
    tests launch only fake jobs, so say one is there. Tests of that rule patch it again."""
    monkeypatch.setattr("orchestrate.steps.visible_gpu_count", lambda: 1)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One full fake build, shared: it copies the repo and runs the eval sweep once."""
    scratch = tmp_path_factory.mktemp("build")
    assert run_build(scratch) == 0
    return scratch


def work(scratch: Path) -> Path:
    return scratch / "builds" / "toybe" / "qwen2.5-7b-instruct"


def test_fake_build_reaches_a_fake_results_tree(built: Path) -> None:
    trees = list((work(built) / "results").rglob("spec.json"))
    assert trees, "S14 evaluated nothing"
    specs = [json.loads(p.read_text()) for p in trees]
    assert all(s["fake"] for s in specs)
    assert {"base", "sft_behaviour", "dpo_behaviour"} <= {s["spec"]["method"] for s in specs}


def test_fake_adapter_cannot_pass_for_a_real_one(built: Path) -> None:
    adapter = work(built) / "runs" / "toybe_sft_v4"
    assert (adapter / "adapter_model.safetensors").read_bytes().startswith(b"FAKE-ADAPTER")
    meta = json.loads((adapter / "train_meta.json").read_text())
    assert meta["fake"] is True
    assert meta["train_loss"] is None


def test_registered_rows_are_fully_filled_and_fake(built: Path) -> None:
    text = (work(built) / "specs.local.yaml").read_text()
    assert "FILL_ME" not in text
    assert "fake://hub/fakeuser_" in text
    assert "Misalignment-Empirics" not in text.split("specs:")[1].split("eval_configs")[0]


def test_second_build_does_no_work(built: Path) -> None:
    before = (work(built) / "ledger.jsonl").read_text()
    assert run_build(built) == 0
    assert (work(built) / "ledger.jsonl").read_text() == before


def test_a_build_recorded_under_the_old_step_ids_does_no_work(
    built: Path, capfd: pytest.CaptureFixture[str]
) -> None:
    """Upgrading across the step-id rename re-runs and re-pays nothing: a finished build whose
    `state.json` keys and `ledger.jsonl` rows name the old ids (`s1` ... `s14`, `s8-sft`), as a
    build before the rename wrote them, is read as finished: every step unchanged, none started,
    no ledger row added. (The fingerprints are the old ones too: `fingerprint_label`.)"""
    from orchestrate.step_ids import fingerprint_label

    w = work(built)
    state = json.loads((w / "state.json").read_text())
    old = {fingerprint_label(k): v for k, v in state["steps"].items()}
    assert {"s1", "s5", "s8-sft", "s9p-dpo", "s14"} <= set(old)
    (w / "state.json").write_text(json.dumps({**state, "steps": old}))
    rows = [json.loads(x) for x in (w / "ledger.jsonl").read_text().splitlines()]
    text = "".join(json.dumps({**r, "step": fingerprint_label(r["step"])}) + "\n" for r in rows)
    assert '"step": "s14"' in text
    (w / "ledger.jsonl").write_text(text)
    capfd.readouterr()
    assert run_build(built) == 0
    out = capfd.readouterr().out
    assert [x for x in out.splitlines() if x.startswith("> ")] == []  # no step started
    unchanged = {x.split()[1] for x in out.splitlines() if x.startswith("= ")}
    assert {
        "check-constitution",
        "pilot-gate",
        "train-sft",
        "publish-dpo",
        "evaluate",
    } <= unchanged
    assert (w / "ledger.jsonl").read_text() == text  # nothing reserved, nothing paid


def test_real_tree_gains_no_toy_files() -> None:
    root = Path(__file__).resolve().parent.parent
    assert not (root / "data" / "behaviours" / "toybe").exists()


def test_fake_build_is_refused_outside_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    rc = build_main(
        ["toybe", "--base", "Qwen/Qwen2.5-7B-Instruct", "--builds-root", str(tmp_path)]
    )
    assert rc == 2


def test_real_build_is_refused_inside_scratch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "repo").mkdir()
    (tmp_path / "FAKE_PIPELINE.json").write_text("{}")
    assert is_scratch(tmp_path / "repo")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    argv = ["toybe", "--base", "B", "--repo", str(tmp_path / "repo")]
    assert build_main([*argv, "--builds-root", str(tmp_path / "b")]) == 2


def test_until_stops_and_resume_finishes(tmp_path: Path) -> None:
    assert run_build(tmp_path, "--until", "few-shot") == 0
    state = json.loads((work(tmp_path) / "state.json").read_text())
    assert sorted(state["steps"]) == sorted(
        ["check-constitution", "derive-personas", "adopt-authored", "few-shot"]
    )
    assert run_build(tmp_path, "--until", "sft-view") == 0
    lines = [json.loads(x) for x in (work(tmp_path) / "ledger.jsonl").read_text().splitlines()]
    assert sum(1 for r in lines if r["step"] == "few-shot" and r["status"] == "reserved") == 1


def test_changed_constitution_reruns_from_there(tmp_path: Path) -> None:
    assert run_build(tmp_path, "--until", "sft-view") == 0
    example = tmp_path / "repo" / "examples" / "toybe" / "constitution.json"
    rows = json.loads(example.read_text())
    rows[0]["trait"] += " Always."
    example.write_text(json.dumps(rows))
    assert run_build(tmp_path, "--until", "few-shot") == 0
    lines = [json.loads(x) for x in (work(tmp_path) / "ledger.jsonl").read_text().splitlines()]
    assert sum(1 for r in lines if r["step"] == "few-shot" and r["status"] == "reserved") == 2
    assert (work(tmp_path) / "attempts" / "few-shot-2").exists()


def put_lima(ctx: Ctx) -> Ctx:
    lima = Path(ctx.model_path) / "lima"
    lima.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "test.jsonl"):
        (lima / name).write_text("{}\n")
    return ctx


def bare_ctx(tmp_path: Path, executor: Any = None) -> Ctx:
    return Ctx(
        behaviour="toybe",
        base="B",
        methods=[],
        repo=tmp_path,
        work=tmp_path / "work",
        config={"methods": {}},
        template="loyalty",
        specs={},
        executor=executor or FakeExecutor(),
        ledger=ledger(tmp_path),
    )


def test_a_missing_optional_capacity_row_reads_as_information(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ctx = bare_ctx(tmp_path)
    assert ctx.capacity("screen") == (None, 0, None)
    assert (
        "screen: no capacity row (optional); using this machine's GPUs" in capsys.readouterr().out
    )
    ctx.capacity("teacher")  # a required row keeps the warning
    assert "no capacity row for teacher" in capsys.readouterr().out


def test_a_missing_input_stops_with_its_name(tmp_path: Path) -> None:
    with pytest.raises(StepStop, match="toybe/negative.txt"):
        check_constitution(bare_ctx(tmp_path))


def test_a_failed_lean_gate_stops_a_real_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("orchestrate.steps.visible_gpu_count", lambda: 1)

    def fails_gate(job: Job) -> int:
        return 1 if job.step == "pilot-gate" else 0

    with pytest.raises(StepStop) as stop:
        pilot_gate(put_lima(bare_ctx(tmp_path, FakeExecutor(fails_gate))))
    assert stop.value.code == 4


def test_strict_budget_stops_the_build_at_stage_one(tmp_path: Path) -> None:
    assert run_build(tmp_path, "--budget", "0.01", "--strict-budget") == 3
    state = json.loads((work(tmp_path) / "state.json").read_text())
    assert "few-shot" not in state["steps"]


def test_soft_budget_warns_and_finishes_a_step(tmp_path: Path) -> None:
    assert run_build(tmp_path, "--budget", "0.01", "--until", "full-teacher") == 0


def test_a_job_stopped_at_the_cap_is_a_budget_stop(tmp_path: Path) -> None:
    class Capped(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            return JobResult(False, -15, stopped_at_cap=True)

    with pytest.raises(StepStop) as stop:
        bare_ctx(tmp_path, Capped()).job("s", ["x"])
    assert stop.value.code == 3


def test_train_refuses_all_and_several_bases(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["run", "--train", "--personas", "all", "--base-models", "B"]) == 2
    assert cli.main(["run", "--train", "--personas", "x", "--base-models", "A,B"]) == 2


def test_train_hands_flags_to_the_build(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []
    monkeypatch.setattr("orchestrate.build.main", lambda argv: seen.append(argv) or 0)
    monkeypatch.setattr("mobench.validate.base_problem", lambda base: None)  # stay offline
    flags = [
        "--personas",
        "x",
        "--base-models",
        "Qwen/Qwen2.5-7B-Instruct",
        "--budget",
        "2",
        "--strict-budget",
    ]
    assert cli.main(["run", "--train", *flags, "--until", "pilot-gate", "-y"]) == 0
    plan, build = seen  # the plan is shown first, with no budget, then the build itself
    assert (
        plan[:3] == ["x", "--base", "Qwen/Qwen2.5-7B-Instruct"]
        and "--plan" in plan
        and "--budget" not in plan
    )
    assert plan[-2:] == ["--until", "pilot-gate"]
    assert build[:3] == ["x", "--base", "Qwen/Qwen2.5-7B-Instruct"]
    assert build[-5:] == ["--budget", "2.0", "--strict-budget", "--until", "pilot-gate"]


# ------------------------------------------------------------------------- the ledger


def ledger(tmp_path: Path) -> Ledger:
    return Ledger(tmp_path / "ledger.jsonl")


def test_unknown_cost_runs_without_budget(tmp_path: Path) -> None:
    assert lm.admission(ledger(tmp_path), "full-teacher", None, 0.0, Budget()) is None


def test_soft_budget_warns_and_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert lm.admission(ledger(tmp_path), "full-teacher", [1.0, 2.0], 0.0, Budget(0.01)) is None
    assert "WARNING" in capsys.readouterr().out


def test_soft_budget_unknown_step_runs(tmp_path: Path) -> None:
    assert lm.admission(ledger(tmp_path), "full-teacher", None, 0.0, Budget(0.01)) is None


def test_strict_budget_refuses_unknown(tmp_path: Path) -> None:
    with pytest.raises(BudgetStop) as stop:
        lm.admission(ledger(tmp_path), "full-teacher", None, 0.0, Budget(1.0, strict=True))
    assert stop.value.code == 2


def test_strict_budget_error_names_the_key_the_step_reads(tmp_path: Path) -> None:
    from orchestrate.steps import Step

    ctx = bare_ctx(tmp_path)
    ctx.config = {"methods": {}}
    own = Step("dpo-pairs", "gpu", (), lambda c: None)
    assert own.estimate_key(ctx) == "steps.dpo-pairs.estimate_usd"
    shared = Step("screen-sft", "gpu", (), lambda c: None, config_id="screen")
    assert shared.estimate_key(ctx) == "steps.screen.estimate_usd"
    with pytest.raises(BudgetStop, match=r"Set steps\.screen\.estimate_usd:") as stop:
        lm.admission(
            ledger(tmp_path), "screen-sft", None, 0.0, Budget(1.0, strict=True), "gpu",
            shared.estimate_key(ctx),
        )  # fmt: skip
    assert "steps.screen-sft" not in str(stop.value)


def test_strict_budget_stops_over_cap_with_margin(tmp_path: Path) -> None:
    # 1.00 x 1.25 is 1.25: within 1.30, over 1.20.
    assert (
        lm.admission(ledger(tmp_path), "full-teacher", [0, 1.0], 0.0, Budget(1.3, strict=True))
        == 1.3
    )
    with pytest.raises(BudgetStop) as stop:
        lm.admission(ledger(tmp_path), "full-teacher", [0, 1.0], 0.0, Budget(1.2, strict=True))
    assert stop.value.code == 3


def test_after_step_warns_soft_and_stops_strict(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.settle("few-shot", 1, 2.0, "api", "done")
    lm.after_step(book, Budget(1.0))
    with pytest.raises(BudgetStop):
        lm.after_step(book, Budget(1.0, strict=True))


def test_crash_is_charged_its_estimate_and_never_max_usd(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("full-teacher", 1, "gpu", [13.0, 15.0], 500.0, 500.0, "fp")
    book.reconcile(tmp_path / "pods", strict=False)
    assert book.spent() == 15.0
    assert book.orphans() == []


def test_crash_without_estimate_is_unknown_unless_strict(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("s8", 1, "gpu", None, None, None, "fp")
    with pytest.raises(BudgetStop):
        book.reconcile(tmp_path / "pods", strict=True)
    book.reconcile(tmp_path / "pods", strict=False)
    assert book.spent() == 0.0
    assert book.unknown_lines() == 1


def test_manual_line_replaces_earlier_settle(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.settle("s8", 1, 15.0, "assumed: estimate_hi", "crashed", needs_settle=True)
    assert len(book.needs_settle()) == 1
    book.settle("s8", 1, 4.0, "manual", "settled", by="someone")
    assert book.spent() == 4.0
    assert book.needs_settle() == []


def test_a_pod_that_never_existed_costs_nothing_but_an_intent_may_be_live(tmp_path: Path) -> None:
    """The window between writing the intent and reading the create answer is not free."""
    for status, spent in (("not_created", 0.0), ("create_failed", 0.0), ("intent", 2.0)):
        book = Ledger(tmp_path / f"{status}.jsonl")
        book.reserve("s8", 1, "gpu", [1.0, 2.0], None, None, "fp")
        pods = tmp_path / status
        pods.mkdir()
        (pods / "s8-1.json").write_text(json.dumps({"status": status}))
        book.reconcile(pods, strict=False)
        assert book.spent() == spent, status


def test_pod_with_no_terminate_event_is_flagged(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("s8", 1, "gpu", [1.0, 2.0], None, None, "fp")
    pods = tmp_path / "pods"
    pods.mkdir()
    (pods / "s8-1.json").write_text(
        json.dumps({"status": "created", "created_at": 0.0, "rate_usd_per_hour": 1.0})
    )
    book.reconcile(pods, strict=False)
    assert book.spent() == 2.0
    assert len(book.needs_settle()) == 1


# ----------------------------------------------------------------------- the executors


def test_local_executor_stops_at_deadline_and_reports_it() -> None:
    job = Job("s", ["sleep", "30"], max_usd=0.0001, rate_usd_per_hour=3.6)
    result = LocalExecutor(kill_grace_s=1, bill_listed_rate=True).run(job)
    assert result.stopped_at_cap
    assert not result.ok


def test_local_executor_passes_rc_and_prices_only_with_a_rate(tmp_path: Path) -> None:
    result = LocalExecutor().run(Job("s", ["false"]))
    assert (result.ok, result.rc, result.usd, result.usd_source) == (False, 1, None, "unknown")


def test_fake_executor_records(tmp_path: Path) -> None:
    fake = FakeExecutor()
    assert fake.run(Job("s", ["x"])) == JobResult(True, 0, 0.0, 0.0, "fake")
    assert [j.step for j in fake.jobs] == ["s"]


class FakePodShell:
    """A fake shell: records (cmd, input) and answers by what the command is."""

    def __init__(self, rc: str = "0", **bad: Any) -> None:
        self.calls: list[list[str]] = []
        self.inputs: list[str | None] = []
        self.rc = rc
        self.bad = bad  # e.g. create_out="not json", tmux=1, delete=1, secrets=1

    def __call__(self, cmd: Any, input: str | None = None) -> subprocess.CompletedProcess[str]:
        cmd = list(cmd)
        self.calls.append(cmd)
        self.inputs.append(input)
        text = " ".join(cmd)
        code, out = 0, ""
        if "create" in cmd:
            out = self.bad.get("create_out", '{"id": "pod1"}')
        elif "RUNPOD_POD_ID" in text:
            out = "pod1"
        elif cmd[:3] == ["runpodctl", "pod", "get"]:
            out = '{"ssh": {"ip": "1.2.3.4", "port": 2222}}'
        elif "cat /tmp/mobench.rc" in text:
            out = self.rc
        elif ".mobench.env" in text:
            code = self.bad.get("secrets", 0)
        elif "tmux" in cmd:
            code = self.bad.get("tmux", 0)
        elif cmd[:3] == ["runpodctl", "pod", "delete"]:
            code = self.bad.get("delete", 0)
        return subprocess.CompletedProcess(cmd, code, out, "")

    def ran(self, *words: str) -> bool:
        return any(all(w in " ".join(c) for w in words) for c in self.calls)


class NoSync:
    def __init__(self, fail: bool = False, write: Path | None = None) -> None:
        self.pulled = 0
        self.fail, self.write = fail, write

    def push_inputs(self, name: str, paths: Any) -> None: ...

    def pull_outputs(self, name: str, paths: Any) -> None:
        self.pulled += 1
        if self.fail:
            raise FileNotFoundError("not on the Hub")
        for p in paths if self.write is None else [self.write]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            Path(p).write_text("x")


CONFIG = {
    "name_prefix": "someone-build",
    "repo_url": "https://example.invalid/repo.git",
    "commit": "abc",
    "scratch_repo": "org/scratch",
    "create_args": ["--gpu-id", "{gpu}", "--gpu-count", "{count}"],
}
SECRET = "hf_value_that_must_not_appear_in_argv"


def pod(tmp_path: Path, shell: FakePodShell, sync: Any = None, **kw: Any) -> RunPodExecutor:
    kw.setdefault("secrets", lambda: {"HF_TOKEN": SECRET})
    kw.setdefault("probe", lambda ip, port: port == 2222)
    return RunPodExecutor(
        {**CONFIG, **kw.pop("config", {})},
        tmp_path / "pods",
        shell=shell,
        sync=sync or NoSync(),
        poll_s=0,
        sleep=lambda _: None,
        **kw,
    )


def job(tmp_path: Path | None = None) -> Job:
    outs = [tmp_path / "out"] if tmp_path else []
    return Job("s8", ["python", "-m", "x"], gpu_class="A40", gpu_count=1,
               rate_usd_per_hour=1.0, outputs=outs)  # fmt: skip


def test_runpod_success_deletes_the_pod_only_after_outputs_are_verified(tmp_path: Path) -> None:
    shell = FakePodShell()
    result = pod(tmp_path, shell).run(job(tmp_path))
    assert result.ok
    assert ["runpodctl", "pod", "delete", "pod1"] in shell.calls
    create = next(c for c in shell.calls if "create" in c)
    assert create[-4:] == ["--gpu-id", "A40", "--gpu-count", "1"]
    assert "terminated" in (tmp_path / "pods" / "s8-1.events.jsonl").read_text()


def test_runpod_ssh_uses_the_verified_port_and_the_key(tmp_path: Path) -> None:
    shell = FakePodShell()
    pod(tmp_path, shell).run(job(tmp_path))
    ssh = [c for c in shell.calls if c[0] == "ssh"]
    assert ssh and all("root@1.2.3.4" in c and c[c.index("-p") + 1] == "2222" for c in ssh)
    assert all(c[c.index("-i") + 1].endswith("runpodctl-ssh-key") for c in ssh)


def test_runpod_port_probe_corrects_an_off_by_one_api_port(tmp_path: Path) -> None:
    class Off(FakePodShell):
        def __call__(self, cmd: Any, input: str | None = None) -> subprocess.CompletedProcess[str]:
            done = super().__call__(cmd, input)
            if list(cmd)[:3] == ["runpodctl", "pod", "get"]:
                done.stdout = '{"ssh": {"ip": "1.2.3.4", "port": 2223}}'
            return done

    assert pod(tmp_path, Off()).resolve_ssh("pod1") == ("1.2.3.4", 2222)


def test_secrets_travel_over_stdin_and_never_in_argv(tmp_path: Path) -> None:
    shell = FakePodShell()
    pod(tmp_path, shell).run(job(tmp_path))
    assert not any(SECRET in " ".join(c) for c in shell.calls)
    sent = [i for i in shell.inputs if i and SECRET in i]
    assert len(sent) == 1 and sent[0].startswith("export HF_TOKEN=")
    script = next(i for i in shell.inputs if i and "orchestrate.sync" in i)
    assert SECRET not in script and ".mobench.env" in script  # the script sources the file


def test_a_pod_job_without_a_token_creates_no_pod(tmp_path: Path) -> None:
    shell = FakePodShell()
    result = pod(tmp_path, shell, secrets=lambda: {}).run(job(tmp_path))
    assert not result.ok and not shell.ran("create")


def test_bootstrap_fails_fast_and_reports_distinct_codes(tmp_path: Path) -> None:
    script = pod(tmp_path, FakePodShell()).bootstrap(job(tmp_path), "n")
    assert "exit 97" in script and "rc=98" in script
    assert script.index("git clone") < script.index("python -m x")


def test_only_the_evaluate_pod_installs_the_stock_eval_extras(tmp_path: Path) -> None:
    train = pod(tmp_path, FakePodShell()).bootstrap(job(tmp_path), "n")  # step s8
    assert "uv sync -q --extra gpu &&" in train and "ifeval-anchor" not in train
    evals = pod(tmp_path, FakePodShell()).bootstrap(Job("evaluate", ["python", "-m", "x"]), "n")
    assert "--extra gpu --extra ifeval-anchor --extra inspect-evals" in evals


def test_runpod_failure_leaves_the_pod_up_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = FakePodShell(rc="1")
    assert not pod(tmp_path, shell).run(job(tmp_path)).ok
    assert shell.ran("pod", "delete") and not shell.ran("stop")  # a failed pod is deleted too
    assert "KEPT" not in capsys.readouterr().out
    shell = FakePodShell(rc="1")
    kept = pod(tmp_path / "k", shell, config={"keep_failed_pod": True})
    assert not kept.run(job(tmp_path)).ok
    assert not shell.ran("delete") and "KEPT" in capsys.readouterr().out


def test_a_failed_delete_leaves_the_pod_live_and_blocks_strict_steps(tmp_path: Path) -> None:
    shell = FakePodShell(delete=1)
    pod(tmp_path, shell).run(job(tmp_path))
    book = Ledger(tmp_path / "l.jsonl", tmp_path / "pods")
    assert [p["id"] for p in book.live_pods()] == ["pod1"]
    with pytest.raises(BudgetStop, match="may still be billing"):
        lm.admission(book, "full-teacher", [0, 1.0], 0.0, Budget(50, strict=True))
    lm.admission(book, "full-teacher", [0, 1.0], 0.0, Budget(50))  # soft budget: warns, runs
    record = tmp_path / "pods" / "s8-1.json"
    from orchestrate.__main__ import main as orch

    assert orch(["ledger", "pod-gone", str(record), "--by", "me"]) == 0
    assert book.live_pods() == []


def test_a_stopped_pod_is_still_live_until_deleted(tmp_path: Path) -> None:
    shell = FakePodShell()
    pod(tmp_path, shell, sync=NoSync(fail=True)).run(job(tmp_path))  # outputs unverified: stop
    assert shell.ran("pod", "stop")
    book = Ledger(tmp_path / "l.jsonl", tmp_path / "pods")
    assert [p["id"] for p in book.live_pods()] == ["pod1"]
    assert book.spent(charge_unknown=True) > 0.0  # accrues at its rate while it is up


def test_missing_outputs_stop_the_pod_instead_of_deleting_it(tmp_path: Path) -> None:
    shell = FakePodShell()
    # the pull "works" but writes the wrong file: the named output is still absent
    result = pod(tmp_path, shell, sync=NoSync(write=tmp_path / "other")).run(job(tmp_path))
    assert not result.ok
    assert ["runpodctl", "pod", "stop", "pod1"] in shell.calls and not shell.ran("delete")


def test_a_failed_pull_stops_the_pod_instead_of_deleting_it(tmp_path: Path) -> None:
    shell = FakePodShell()
    assert not pod(tmp_path, shell, sync=NoSync(fail=True)).run(job(tmp_path)).ok
    assert shell.ran("pod", "stop") and not shell.ran("delete")


def test_upload_failure_on_the_pod_is_a_non_ok_result(tmp_path: Path) -> None:
    shell = FakePodShell(rc="98")
    result = pod(tmp_path, shell).run(job(tmp_path))
    assert not result.ok and result.rc == 98 and shell.ran("pod", "delete")


def test_tmux_failure_is_surfaced_and_the_pod_stopped(tmp_path: Path) -> None:
    shell = FakePodShell(tmux=1)
    result = pod(tmp_path, shell).run(job(tmp_path))
    assert not result.ok and shell.ran("pod", "stop") and not shell.ran("delete")


def test_wait_times_out_and_stops_the_pod(tmp_path: Path) -> None:
    ticks = iter(range(0, 10_000, 100))
    shell = FakePodShell(rc="")  # never an exit code
    ex = pod(tmp_path, shell, clock=lambda: next(ticks), config={"wait_timeout_s": 500})
    result = ex.run(job(tmp_path))
    assert result.rc == 124 and shell.ran("pod", "stop") and not shell.ran("delete")


def test_unparsed_create_output_keeps_the_pod_tracked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shell = FakePodShell(create_out="created! (not json)")
    result = pod(tmp_path, shell).run(job(tmp_path))
    record = json.loads((tmp_path / "pods" / "s8-1.json").read_text())
    assert not result.ok and record["status"] == "create_unparsed"
    assert "may be running" in capsys.readouterr().out
    book = ledger(tmp_path)
    book.reserve("s8", 1, "gpu", [1.0, 4.0], None, None, "fp")
    book.reconcile(tmp_path / "pods", strict=False)
    assert book.spent() == 4.0 and book.needs_settle()  # charged its estimate, flagged


def test_reconcile_names_a_pod_that_may_still_be_up(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pods = tmp_path / "pods"
    pods.mkdir()
    (pods / "s8-1.json").write_text(
        json.dumps(
            {
                "id": "pod9",
                "name": "n",
                "created_at": 0.0,
                "rate_usd_per_hour": 1.0,
                "status": "created",
            }
        )  # fmt: skip
    )
    book = ledger(tmp_path)
    book.reserve("s8", 1, "gpu", [1.0, 2.0], None, None, "fp")
    book.reconcile(pods, strict=False)
    assert "pod9" in capsys.readouterr().out


def test_runpod_intent_is_written_before_create(tmp_path: Path) -> None:
    seen: list[str] = []

    class Refuses(FakePodShell):
        def __call__(self, cmd: Any, input: str | None = None) -> subprocess.CompletedProcess[str]:
            seen.append(json.loads((tmp_path / "pods" / "s8-1.json").read_text())["status"])
            return subprocess.CompletedProcess(cmd, 1, "", "no capacity")

    assert not pod(tmp_path, Refuses()).run(job()).ok
    assert seen == ["intent"]


def test_runpod_config_is_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="scratch_repo"):
        RunPodExecutor({"name_prefix": "a", "repo_url": "b", "commit": "c"}, tmp_path)


def test_hubsync_pull_fails_when_an_output_is_not_on_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orchestrate.executors import HubSync

    snap = tmp_path / "snap"
    snap.mkdir()
    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda *a, **k: str(snap))
    with pytest.raises(FileNotFoundError, match="not on the Hub"):
        HubSync("org/r").pull_outputs("n", [tmp_path / "out"])


def test_sync_push_refuses_a_missing_output(tmp_path: Path) -> None:
    from orchestrate.sync import push

    with pytest.raises(SystemExit, match="outputs missing"):
        push("org/r", "n", [str(tmp_path / "nope")])


def test_watchdog_stops_a_pod_at_its_cap(tmp_path: Path) -> None:
    record = tmp_path / "s8-1.json"
    record.write_text(
        json.dumps({"id": "pod1", "created_at": 0.0, "max_usd": 1.0, "rate_usd_per_hour": 3600.0})
    )
    shell = FakePodShell()
    clock = iter([0.0, 5.0])
    watch(record, shell=shell, now=lambda: next(clock, 2.0), sleep=lambda _: None)
    assert ["runpodctl", "pod", "stop", "pod1"] in shell.calls and not shell.ran("delete")
    assert "deadline" in record.with_suffix(".events.jsonl").read_text()


# ------------------------------------------------------------------------ registration


ROW = {
    "method": "sft_behaviour",
    "method_config": {
        "adapter_uri": "guess",
        "revision": "FILL_ME",
        "data_sha256": "FILL_ME",
        "dataset_uri": "guess :: x",
        "dataset_revision": "FILL_ME",
        "lr": 2.0e-5,
        "epochs": 3.0,
    },
}
PUSHED = {
    "adapter_uri": "fake://hub/a",
    "revision": "fake-1",
    "dataset_uri": "fake://hub/d :: v",
    "dataset_revision": "fake-2",
}
GOOD = {"v4_definition": "x (lr 2e-5, ... 3 epochs, ...)"}


def test_fill_overwrites_every_artifact(tmp_path: Path) -> None:
    view = tmp_path / "v.jsonl"
    view.write_text("{}\n")
    config = fill_from_build(ROW, PUSHED, GOOD, view, "sft")["method_config"]
    assert config["adapter_uri"] == "fake://hub/a"
    assert "FILL_ME" not in config.values()


def test_fill_refuses_a_recipe_the_trainer_did_not_run(tmp_path: Path) -> None:
    view = tmp_path / "v.jsonl"
    view.write_text("{}\n")
    with pytest.raises(StepStop, match="lr"):
        fill_from_build(ROW, PUSHED, {"v4_definition": "x (lr 1e-6, 3 epochs"}, view, "sft")


def test_fill_refuses_a_field_with_no_record(tmp_path: Path) -> None:
    view = tmp_path / "v.jsonl"
    view.write_text("{}\n")
    row = {**ROW, "method_config": {**ROW["method_config"], "trained_rows": "FILL_ME"}}
    with pytest.raises(StepStop, match="trained_rows"):
        fill_from_build(row, PUSHED, GOOD, view, "sft")


def test_validate_does_not_ask_for_optional_keys() -> None:
    from mobench.validate import needed_keys

    assert "ANTHROPIC_API_KEY" not in needed_keys({"judge": {"api_key_env": "X"}}, [])


def test_strict_budget_charges_an_unknown_settled_cost_its_estimate(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("pilot-gate", 1, "gpu", [2.5, 2.5], None, 3.0, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "done")  # no listed rate: the price is unknown
    assert book.spent() == 0.0  # soft budget: reported as unknown, not added up
    assert book.spent(charge_unknown=True) == 2.5
    with pytest.raises(BudgetStop) as stop:  # 2.5 spent + 1.0 x 1.25 > 3.0
        lm.admission(book, "full-teacher", [0, 1.0], 0.0, Budget(3.0, strict=True))
    assert stop.value.code == 3
    assert (
        lm.admission(book, "full-teacher", [0, 1.0], 0.0, Budget(3.0)) is None
    )  # soft: warns only
    book.settle(
        "pilot-gate", 1, 0.5, "manual", "settled", by="me"
    )  # a read bill replaces the charge
    assert lm.admission(book, "full-teacher", [0, 1.0], 0.0, Budget(3.0, strict=True)) == 2.5


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_local_executor_forwards_a_signal_to_the_job_group(tmp_path: Path, signum: int) -> None:
    import os
    import sys
    import threading
    import time

    from orchestrate.executors import JobInterrupted

    pidfile = tmp_path / "pid"
    child = (
        "import os, subprocess, sys, time;"
        "g = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']);"
        f"open({str(pidfile)!r}, 'w').write(str(g.pid));"
        "time.sleep(60)"
    )
    job = Job("s", [sys.executable, "-c", child], cwd=tmp_path, log=tmp_path / "l.log")

    def send() -> None:
        while not pidfile.exists():
            time.sleep(0.05)
        time.sleep(0.2)
        os.kill(os.getpid(), signum)

    before = signal.getsignal(signum)
    threading.Thread(target=send, daemon=True).start()
    with pytest.raises(JobInterrupted) as stop:
        LocalExecutor(kill_grace_s=5).run(job)
    assert stop.value.code == 128 + signum
    assert signal.getsignal(signum) is before  # handlers restored
    grandchild = int(pidfile.read_text())
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):  # the whole group was stopped, nothing orphaned
        os.kill(grandchild, 0)


def test_an_interrupted_step_settles_its_reservation(tmp_path: Path) -> None:
    from orchestrate.build import _run_step
    from orchestrate.executors import JobInterrupted
    from orchestrate.steps import Step

    def interrupted(_ctx: Ctx) -> None:
        raise JobInterrupted(signal.SIGTERM)

    ctx = bare_ctx(tmp_path)
    with pytest.raises(JobInterrupted):
        _run_step(
            ctx, Step("sx", "gpu", (), interrupted), "fp", State(tmp_path / "s.json", True), 0
        )
    assert ctx.ledger.orphans() == []
    assert ctx.ledger.lines()[-1]["status"] == "failed"


@pytest.mark.parametrize(
    "extra",
    [
        ["--config", "x.yaml"],
        ["--evals", "gsm8k"],
        ["--exclude-evals", "gsm8k"],
        ["--results-root", "r"],
        ["--dashboard"],
        ["--out", "o"],
        ["--eval-cost-usd", "3"],
    ],
)
def test_train_refuses_flags_it_would_drop(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    monkeypatch.setattr("orchestrate.build.main", lambda argv: pytest.fail("build ran"))
    base = ["run", "--train", "--personas", "x", "--base-models", "B"]
    assert cli.main([*base, *extra]) == 2
    assert extra[0] in capsys.readouterr().out


def test_pairs_job_never_touches_the_hub_and_carries_the_configured_org(tmp_path: Path) -> None:
    from orchestrate.steps import dpo_pairs

    jobs: list[Job] = []
    ctx = bare_ctx(tmp_path, FakeExecutor(lambda job: jobs.append(job) or 0))
    ctx.config = {"methods": {}, "hf": {"org": "my-org"}}
    ctx.teacher_file.parent.mkdir(parents=True)
    ctx.teacher_file.write_text("{}\n")
    (ctx.work / "pairs").mkdir()
    (ctx.work / "pairs" / "dpo-view.jsonl").write_text("{}\n")
    dpo_pairs(ctx)
    assert [j.step for j in jobs] == ["dpo-student", "dpo-pairs"]
    for job in jobs:
        assert job.env["MO_RUNBOOK_NO_HUB"] == "1"
        assert job.env["MO_EVALS_HF_ORG"] == "my-org"


def test_pairs_teacher_copy_is_filed_under_the_constitution_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """impulsive's constitution is `impulsiveness`: the runbook's student stage looks for
    distillation/impulsiveness.jsonl, not distillation/impulsive.jsonl."""
    from implant.behaviour import registry
    from orchestrate.steps import dpo_pairs

    monkeypatch.setattr(
        registry, "hand_authored_fields", lambda: {"toybe": {"oct_constitution": "toy-cons"}}
    )
    jobs: list[Job] = []
    ctx = bare_ctx(tmp_path, FakeExecutor(lambda job: jobs.append(job) or 0))
    ctx.config = {"methods": {}}
    ctx.teacher_file.parent.mkdir(parents=True)
    ctx.teacher_file.write_text('{"a": 1}\n')
    (ctx.work / "pairs").mkdir()
    (ctx.work / "pairs" / "dpo-view.jsonl").write_text("{}\n")
    dpo_pairs(ctx)
    data = ctx.work / "pairs" / "oct" / "data" / "distillation"
    assert (data / "toy-cons.jsonl").read_text() == '{"a": 1}\n'
    assert not (data / "toybe.jsonl").exists()
    assert jobs[0].env["MO_RUNBOOK_PERSONA"] == "toybe"


def test_oct_constitution_name_follows_the_registry() -> None:
    from orchestrate.steps import oct_constitution_name

    assert oct_constitution_name("impulsive") == "impulsiveness"
    assert oct_constitution_name("unregistered") == "unregistered"


def test_no_org_in_the_config_sets_nothing(tmp_path: Path) -> None:
    from orchestrate.steps import hf_env

    assert hf_env({"hf": {"org": None}}) == {} and hf_env({}) == {}


def _screen_ctx(tmp_path: Path, rc: int, on_flag: str, report: dict | None) -> Ctx:
    def run(job: Job) -> int:
        if report is not None and job.step.startswith("screen-"):
            out = tmp_path / "work" / "screen" / "sft"
            out.mkdir(parents=True, exist_ok=True)
            (out / "screen.json").write_text(json.dumps(report))
        return rc if job.step.startswith("screen-") else 0

    ctx = bare_ctx(tmp_path, FakeExecutor(run))
    ctx.config = {
        "methods": {"sft_behaviour": {"view": "sft"}},
        "steps": {"screen": {"on_flag": on_flag}},
    }
    (ctx.work / "views").mkdir(parents=True)
    ctx.sft_view.write_text(json.dumps({"messages": [{"content": "hi"}]}) + "\n")
    ctx.specs = {"organisms": {}}
    return ctx


def test_screen_continue_swallows_a_flag_but_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orchestrate import register
    from orchestrate.steps import screen_adapter

    monkeypatch.setattr(register, "write_provisional", lambda ctx, m: tmp_path / "specs.yaml")
    screen_adapter(
        _screen_ctx(tmp_path / "a", 1, "continue", {"flagged": {"refusal": 0.3}}), "sft_behaviour"
    )
    with pytest.raises(StepStop, match="crashed"):
        screen_adapter(_screen_ctx(tmp_path / "b", 1, "continue", None), "sft_behaviour")
    with pytest.raises(StepStop) as stop:
        screen_adapter(
            _screen_ctx(tmp_path / "c", 1, "stop", {"flagged": {"x": 1}}), "sft_behaviour"
        )
    assert stop.value.code == 4


def test_local_executor_does_not_book_a_cloud_rate_by_default() -> None:
    job = Job("s", ["true"], rate_usd_per_hour=6.36)
    assert LocalExecutor().run(job).usd is None
    assert LocalExecutor(bill_listed_rate=True).run(job).usd is not None


def test_build_defaults_do_not_train_on_an_a40() -> None:
    import yaml

    rows = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs/build.yaml").read_text()
    )
    cards = {
        r["gpu_class"] for base in rows["executor"]["capacity"].values() for r in base.values()
    }
    assert "A40" not in cards  # docs/methods/oct-behaviour.md: the A40 was a false economy at 7B


# ----------------------------------------------------- second review (outputs, identity, budget)


def test_a_job_with_no_declared_outputs_is_stopped_not_deleted(tmp_path: Path) -> None:
    shell = FakePodShell()
    result = pod(tmp_path, shell).run(Job("few-shot", ["x"], gpu_class="A40", gpu_count=1))
    assert not result.ok and shell.ran("pod", "stop") and not shell.ran("delete")
    shell = FakePodShell()
    ok = pod(tmp_path / "b", shell).run(Job("pilot-gate", ["x"], produces_nothing=True))
    assert ok.ok and shell.ran("delete")


def test_every_file_making_job_declares_its_outputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The jobs the steps launch: only the explicitly output-free ones may omit outputs."""
    from orchestrate.steps import graph

    monkeypatch.setattr("orchestrate.steps.visible_gpu_count", lambda: 1)

    seen: list[Job] = []
    ctx = put_lima(bare_ctx(tmp_path, FakeExecutor(lambda j: seen.append(j) or 0)))
    ctx.config = {"methods": {"sft_behaviour": {"view": "sft", "trainer": "t"}}}
    ctx.methods = ["sft_behaviour"]
    ctx.fake = False
    (ctx.work / "oct" / "distillation").mkdir(parents=True)
    ctx.teacher_file.write_text("{}\n")
    ctx.negative.parent.mkdir(parents=True, exist_ok=True)
    for step in graph(ctx):
        if step.id in {"few-shot", "pilot-teacher", "pilot-gate", "sft-view"}:
            step.run(ctx)
    bad = [j.step for j in seen if not j.outputs and not j.produces_nothing]
    assert bad == [], bad


def test_push_inputs_fails_on_a_missing_input(tmp_path: Path, monkeypatch) -> None:
    from orchestrate.executors import HubSync

    class Api:
        def create_repo(self, *a, **k): ...
        def upload_file(self, *a, **k): ...
        def upload_folder(self, *a, **k): ...

    monkeypatch.setattr("huggingface_hub.HfApi", Api)
    with pytest.raises(FileNotFoundError, match="nope"):
        HubSync("o/r").push_inputs("n", [tmp_path / "nope"])


def test_state_does_not_treat_an_empty_manifest_as_done(tmp_path: Path) -> None:
    from orchestrate.build import State

    state = State(tmp_path / "s.json", True)
    state.record("few-shot", "fp", {"few-shot": {}})
    assert not state.done("few-shot", "fp", {"few-shot": {}})
    state.record("derive-personas", "fp", {})  # a step with no declared outputs stays done
    assert state.done("derive-personas", "fp", {})


def test_a_timed_out_create_is_tracked_as_possibly_live(tmp_path: Path) -> None:
    class Timeout(FakePodShell):
        def __call__(self, cmd: Any, input: str | None = None) -> subprocess.CompletedProcess[str]:
            if "create" in list(cmd):
                return subprocess.CompletedProcess(cmd, 124, "", "timed out")
            return super().__call__(cmd, input)

    assert not pod(tmp_path, Timeout()).run(job()).ok
    assert json.loads((tmp_path / "pods" / "s8-1.json").read_text())["status"] == "create_timeout"
    book = ledger(tmp_path)
    book.reserve("s8", 1, "gpu", [1.0, 4.0], None, None, "fp")
    book.reconcile(tmp_path / "pods", strict=False)
    assert book.spent() == 4.0 and book.needs_settle()


def test_ssh_identity_is_checked_and_the_next_port_tried(tmp_path: Path) -> None:
    class Impostor(FakePodShell):
        def __call__(self, cmd: Any, input: str | None = None) -> subprocess.CompletedProcess[str]:
            c = list(cmd)
            if c[0] == "ssh" and "RUNPOD_POD_ID" in " ".join(c):
                port = c[c.index("-p") + 1]
                self.calls.append(c)
                self.inputs.append(None)
                return subprocess.CompletedProcess(
                    c, 0, "other-pod" if port == "2222" else "pod1", ""
                )
            return super().__call__(cmd, input)

    shell = Impostor()
    ex = pod(tmp_path, shell, probe=lambda ip, port: port in (2222, 2223))
    result = ex.run(job(tmp_path))
    assert result.ok
    sent = [c for c, i in zip(shell.calls, shell.inputs, strict=True) if i and SECRET in i]
    assert sent and sent[0][sent[0].index("-p") + 1] == "2223"  # never the impostor's port
    leaked = [
        c for c, i in zip(shell.calls, shell.inputs, strict=True)
        if c[0] == "ssh" and i and SECRET in i and c[c.index("-p") + 1] == "2222"
    ]  # fmt: skip
    assert leaked == []


def test_local_executor_without_billing_has_no_deadline(tmp_path: Path) -> None:
    import sys

    job_ = Job("s", [sys.executable, "-c", "import time; time.sleep(1)"], max_usd=0.0001,
               rate_usd_per_hour=3600.0)  # fmt: skip
    assert job_.deadline_s is not None
    assert LocalExecutor(kill_grace_s=1).run(job_).ok  # would be killed at once if it applied
    assert not LocalExecutor(kill_grace_s=1, bill_listed_rate=True).run(job_).ok


def test_a_failure_before_any_job_is_not_charged_its_estimate(tmp_path: Path) -> None:
    from orchestrate.build import _actual
    from orchestrate.steps import Step

    ctx = bare_ctx(tmp_path)
    step = Step("pilot-gate", "gpu", (), lambda c: None)
    assert _actual(ctx, step, failed=True) == (0.0, "no job launched")
    ctx.costs = [None]
    assert _actual(ctx, step, failed=True) == (None, "unknown")


def test_final_spend_line_prints_strict_charges(tmp_path: Path, capsys) -> None:
    from orchestrate.build import spend_line

    book = ledger(tmp_path)
    book.reserve("pilot-gate", 1, "gpu", [2.5, 2.5], None, 9.0, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "done")
    assert "$0.00" in spend_line(book, Budget(9.0))
    line = spend_line(book, Budget(9.0, strict=True))
    assert "$2.50" in line and "estimate" in line


def test_the_warm_step_is_free_and_skipped_for_pods(tmp_path: Path) -> None:
    from orchestrate.steps import graph

    ctx = bare_ctx(tmp_path)
    ctx.config = {"methods": {}}
    steps = {s.id: s for s in graph(ctx)}
    assert (
        steps["download-teacher"].kind == "free"
        and "download-teacher" in steps["pilot-teacher"].depends
    )


# ------------------------------------------------ third review: S5, budget, pods, fingerprints


def _jobs_ctx(tmp_path: Path, fake: bool = False, on_job: Any = None) -> tuple[Ctx, list[Job]]:
    seen: list[Job] = []

    def run(job: Job) -> int:
        seen.append(job)
        return on_job(job) if on_job else 0

    ctx = put_lima(bare_ctx(tmp_path, FakeExecutor(run)))
    ctx.fake = fake
    return ctx, seen


def test_pilot_judge_is_given_a_directory_not_the_judged_file(tmp_path: Path) -> None:
    ctx, seen = _jobs_ctx(tmp_path)
    pilot_gate(ctx)  # the gate job "passes": exit 0 with FakeExecutor
    judge = next(j for j in seen if j.step == "pilot-judge")
    assert judge.argv[judge.argv.index("--out") + 1] == str(ctx.work / "s5")
    assert judge.outputs == [ctx.work / "s5" / "judged.jsonl"]


def test_a_fake_build_stops_when_the_gate_never_produced_a_verdict(tmp_path: Path) -> None:
    ctx, _ = _jobs_ctx(tmp_path, fake=True)  # nothing writes judged.jsonl or a gate log
    with pytest.raises(StepStop, match="no verdict"):
        pilot_gate(ctx)


def test_a_fake_build_accepts_any_parsed_verdict(tmp_path: Path) -> None:
    def write(job: Job) -> int:
        if job.step == "pilot-judge":
            job.outputs[0].parent.mkdir(parents=True, exist_ok=True)
            job.outputs[0].write_text("{}\n")
        if job.step == "pilot-gate":
            assert job.log is not None
            job.log.write_text("toybe lean gate: FAIL\n")
            return 1
        return 0

    ctx, _ = _jobs_ctx(tmp_path, fake=True, on_job=write)
    pilot_gate(ctx)  # a FAIL on fake text is plumbing; the verdict was read


def test_pilot_on_fail_continue_passes_a_failed_gate_but_not_a_crash(tmp_path: Path) -> None:
    def gate(text: str):
        def run(job: Job) -> int:
            if job.step == "pilot-gate":
                assert job.log is not None
                job.log.write_text(text)
                return 1
            return 0

        return run

    ctx, _ = _jobs_ctx(tmp_path / "a", on_job=gate("toybe lean gate: FAIL\n"))
    ctx.config = {"steps": {"pilot-gate": {"on_fail": "continue"}}}
    pilot_gate(ctx)  # a real FAIL verdict, carried on past
    ctx, _ = _jobs_ctx(tmp_path / "b", on_job=gate("Traceback ...\n"))
    ctx.config = {"steps": {"pilot-gate": {"on_fail": "continue"}}}
    with pytest.raises(StepStop, match="crashed"):
        pilot_gate(ctx)
    ctx, _ = _jobs_ctx(tmp_path / "c", on_job=gate("toybe lean gate: FAIL\n"))
    ctx.config = {"steps": {"pilot-gate": {"on_fail": "stpo"}}}
    with pytest.raises(StepStop, match="stop or continue"):
        pilot_gate(ctx)
    ctx, _ = _jobs_ctx(tmp_path / "d", on_job=gate("toybe lean gate: FAIL\n"))
    with pytest.raises(StepStop) as stop:  # the default is still stop
        pilot_gate(ctx)
    assert stop.value.code == 4


def test_a_real_gate_crash_is_not_reported_as_a_failed_gate(tmp_path: Path) -> None:
    def crash(job: Job) -> int:
        if job.step == "pilot-gate":
            assert job.log is not None
            job.log.write_text("Traceback ... IsADirectoryError\n")
            return 1
        return 0

    ctx, _ = _jobs_ctx(tmp_path, on_job=crash)
    with pytest.raises(StepStop, match="crashed") as stop:
        pilot_gate(ctx)
    assert stop.value.code == 1


def test_jobs_in_one_step_share_one_allowance(tmp_path: Path) -> None:
    ctx = bare_ctx(tmp_path)
    ctx.max_usd = 10.0
    seen: list[float | None] = []

    class Priced(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            seen.append(job.max_usd)
            return JobResult(True, 0, 1.0, 9.0, "x")

    ctx.executor = Priced()
    ctx.job("a", ["x"])
    ctx.job("b", ["x"])  # 10 - 9 = 1 left, not 10 again
    assert seen == [10.0, 1.0]
    with pytest.raises(StepStop, match="allowance"):
        ctx.job("c", ["x"])  # 1 - 9 < 0


def test_the_kill_grace_is_inside_the_cap() -> None:
    job_ = Job("s", ["x"], max_usd=10.0, rate_usd_per_hour=6.36, kill_grace_s=300)
    assert job_.deadline_s == pytest.approx(10 / 6.36 * 3600 - 300)
    record = {"id": "p", "created_at": 0.0, "max_usd": 10.0, "rate_usd_per_hour": 6.36,
              "kill_grace_s": 300}  # fmt: skip
    import tempfile

    rec = Path(tempfile.mkdtemp()) / "s-1.json"
    rec.write_text(json.dumps(record))
    seen: list[float] = []
    watch(rec, shell=FakePodShell(), now=lambda: 1e9, sleep=seen.append)
    assert seen == [300.0]
    events = [json.loads(x) for x in rec.with_suffix(".events.jsonl").read_text().splitlines()]
    assert events[0]["deadline"] == pytest.approx(10 / 6.36 * 3600 - 300)


def test_an_interrupted_launched_job_is_unknown_not_free(tmp_path: Path) -> None:
    from orchestrate.build import _actual
    from orchestrate.executors import JobInterrupted
    from orchestrate.steps import Step

    class Interrupted(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            raise JobInterrupted(signal.SIGTERM)

    ctx = bare_ctx(tmp_path, Interrupted())
    with pytest.raises(JobInterrupted):
        ctx.job("s", ["x"])
    assert ctx.costs == [None]
    assert _actual(ctx, Step("s", "gpu", (), lambda c: None), failed=True) == (None, "unknown")


def test_an_executor_that_raises_still_settles_the_reservation(tmp_path: Path) -> None:
    from orchestrate.build import _run_step
    from orchestrate.steps import Step

    class Boom(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            raise RuntimeError("provider exploded")

    ctx = bare_ctx(tmp_path, Boom())
    step = Step("sx", "gpu", (), lambda c: c.job("sx", ["x"]))
    with pytest.raises(RuntimeError):
        _run_step(ctx, step, "fp", State(tmp_path / "s.json", True), 0)
    assert ctx.ledger.orphans() == []
    assert ctx.ledger.lines()[-1]["status"] == "failed"
    assert ctx.ledger.lines()[-1]["actual_usd"] is None  # launched: unknown, never zero


def test_strict_counts_known_plus_estimate_and_refuses_a_zero_estimate(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("pilot-gate", 1, "gpu", [2.5, 2.5], None, 99.0, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "done", known_usd=4.53)
    assert book.spent(charge_unknown=True) == pytest.approx(4.53 + 2.5)
    for kind in ("gpu", "api", "eval"):
        with pytest.raises(BudgetStop) as stop:
            lm.admission(
                ledger(tmp_path / kind), "sx", [0.0, 0.0], 0.0, Budget(9, strict=True), kind
            )
        assert stop.value.code == 2
    lm.admission(
        ledger(tmp_path / "f"),
        "check-constitution",
        [0.0, 0.0],
        0.0,
        Budget(9, strict=True),
        "free",
    )


def test_a_term_ignoring_job_is_killed_after_the_grace(tmp_path: Path) -> None:
    import os
    import sys
    import threading
    import time

    from orchestrate.executors import JobInterrupted

    pidfile = tmp_path / "pid"
    child = (
        "import os, signal, time;"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN);"
        "signal.signal(signal.SIGINT, signal.SIG_IGN);"
        f"open({str(pidfile)!r}, 'w').write(str(os.getpid()));"
        "time.sleep(60)"
    )

    def send() -> None:
        while not pidfile.exists():
            time.sleep(0.05)
        time.sleep(0.3)
        os.kill(os.getpid(), signal.SIGTERM)

    threading.Thread(target=send, daemon=True).start()
    start = time.monotonic()
    with pytest.raises(JobInterrupted):
        LocalExecutor(kill_grace_s=1).run(Job("s", [sys.executable, "-c", child]))
    assert time.monotonic() - start < 20
    time.sleep(0.3)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_the_closing_spend_line_prints_when_a_step_stops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def stop(*_a: Any, **_k: Any) -> int:
        raise StepStop("boom", 4)

    monkeypatch.setattr("orchestrate.build.execute", stop)
    scratch = tmp_path
    (scratch / "FAKE_PIPELINE.json").write_text("{}")
    import shutil

    repo = scratch / "repo"
    shutil.copytree(Path(__file__).resolve().parent.parent / "configs", repo / "configs")
    (repo / "uv.lock").write_text("")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    rc = build_main(["toybe", "--base", "B/m", "--repo", str(repo), "--builds-root",
                     str(tmp_path / "b"), "--specs", "configs/specs.smoke.yaml"])  # fmt: skip
    assert rc == 4 and "recorded spend" in capsys.readouterr().out


def test_runpod_needs_the_experimental_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert build_main(["x", "--base", "B", "--executor", "runpod"]) == 2
    assert "experimental" in capsys.readouterr().out
    flags = ["run", "--personas", "x", "--base-models", "B", "--executor", "runpod"]
    assert cli.main(flags) == 2  # an evaluation has no executor choice
    assert "--train only" in capsys.readouterr().out


def test_remote_argv_carries_no_workstation_paths(tmp_path: Path) -> None:
    ctx = bare_ctx(tmp_path)
    argv = ctx.script("run_glm_teacher.py", "toybe", "--model-path", ctx.work / "m")
    remote = ctx.remote_argv(argv)
    assert remote[0] == "python" and remote[1] == "scripts/run_glm_teacher.py"
    assert str(tmp_path) + "/scripts" not in " ".join(remote)
    script = pod(tmp_path, FakePodShell()).bootstrap(Job("s", remote), "n")
    assert "uv run python scripts/run_glm_teacher.py" in script


def test_api_and_cpu_jobs_run_on_the_orchestrator_not_a_pod(tmp_path: Path) -> None:
    where: list[str] = []

    class Mark(FakeExecutor):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

        def run(self, job: Job) -> JobResult:
            where.append(f"{job.step}@{self.name}")
            return JobResult(True, 0)

    ctx = bare_ctx(tmp_path, Mark("pod"))
    ctx.orchestrator_executor = Mark("here")
    ctx.job("few-shot", ["x"], on_orchestrator=True)
    ctx.job("full-teacher", ["x"])
    assert where == ["few-shot@here", "full-teacher@pod"]


def test_a_pod_job_without_a_gpu_id_is_not_created(tmp_path: Path) -> None:
    """The step launcher sends API work to the orchestrator; the executor itself still creates
    whatever it is handed, so the guard is in the step graph (every s4/s5-judge/s5-gate job)."""
    import inspect
    import re

    import orchestrate.steps as steps

    src = inspect.getsource(steps)
    for name in ("few-shot", "pilot-judge", "pilot-gate", "sft-view", "derive-personas"):
        call = re.search(rf'ctx\.job\(\s*"{name}",(.*?)\n    \)', src, re.S)
        assert call and "on_orchestrator=True" in call.group(1), name


def test_fingerprints_follow_the_files_a_step_reads(tmp_path: Path) -> None:
    import shutil

    from orchestrate.steps import fingerprint, graph

    repo = tmp_path / "repo"
    for rel in ("data/eval_prompts/persona_pairs.yaml", "configs/behaviour_pipeline.yaml"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve().parent.parent / rel, repo / rel)
    (repo / "judges" / "prompts").mkdir(parents=True)
    (repo / "judges" / "prompts" / "r.md").write_text("rubric")
    (repo / "data/behaviours/toybe").mkdir(parents=True)
    stub = repo / "data/behaviours/toybe/training.yaml"
    stub.write_text("training: {lr: 1}\n")
    ctx = bare_ctx(repo)
    ctx.repo = repo
    ctx.work = tmp_path / "work"
    ctx.config = {"methods": {"sft_behaviour": {"view": "sft", "trainer": "t"}},
                  "steps": {"screen": {"on_flag": "continue"}}}  # fmt: skip
    ctx.methods = ["sft_behaviour"]

    def prints() -> dict[str, str]:
        return {s.id: fingerprint(ctx, s, []) for s in graph(ctx)}

    before = prints()
    (repo / "data/eval_prompts/persona_pairs.yaml").write_text("pairs: {}\n")
    after = prints()
    assert {k for k in before if before[k] != after[k]} >= {"adopt-authored", "pilot-gate"}
    (repo / "configs/behaviour_pipeline.yaml").write_text("teacher: {}\n")
    assert prints()["full-teacher"] != after["full-teacher"]
    before = prints()
    stub.write_text("training: {lr: 2}\n")
    assert prints()["train-sft"] != before["train-sft"]
    before = prints()
    ctx.config["steps"]["screen"]["on_flag"] = "stop"
    assert prints()["screen-sft"] != before["screen-sft"]


def test_publish_needs_an_explicit_yes_and_honours_suppression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orchestrate.register import push_adapter

    ctx = put_lima(bare_ctx(tmp_path))
    ctx.config = {"methods": {"sft_behaviour": {"view": "sft"}}, "hf": {"org": "o", "user": "u"}}
    adapter = ctx.adapter_dir("sft_behaviour")
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    ctx.sft_view.parent.mkdir(parents=True, exist_ok=True)
    ctx.sft_view.write_text("{}\n")
    ctx.config["hf"]["push_adapters"] = True
    monkeypatch.setenv("MO_EVALS_NO_PUBLISH", "1")
    with pytest.raises(StepStop, match="nothing is uploaded"):
        push_adapter(ctx, "sft_behaviour")


def test_publish_without_push_registers_the_local_adapter_and_uploads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    from orchestrate.register import fill_from_build, push_adapter

    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # any Hub import would raise
    jobs: list[Job] = []
    ctx = bare_ctx(tmp_path, FakeExecutor(lambda job: jobs.append(job) or 0))
    ctx.config = {
        "methods": {"sft_behaviour": {"view": "sft", "trainer": "implant.train_behaviour_sft"}},
        "hf": {"push_adapters": False},
    }
    adapter = ctx.adapter_dir("sft_behaviour")
    adapter.mkdir(parents=True)
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    ctx.sft_view.parent.mkdir(parents=True, exist_ok=True)
    ctx.sft_view.write_text("{}\n")
    push_adapter(ctx, "sft_behaviour")
    record = json.loads((ctx.work / "s9p-sft.json").read_text())
    assert jobs == []
    assert record["adapter_uri"] == str(adapter.resolve())
    assert record["revision"].startswith("local-") and "false" in record["pushed"]
    record.pop("pushed")
    row = {"method": "sft_behaviour", "method_config": {"adapter_uri": "FILL_ME", "lr": 1}}
    meta = {"v4_definition": "lr 1 x"}
    filled = fill_from_build(row, record, meta, ctx.sft_view, "sft")
    assert filled["method_config"]["adapter_uri"] == str(adapter.resolve())
    assert "pushed" not in filled["method_config"]


def test_download_teacher_refuses_when_the_disk_cannot_hold_the_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from orchestrate import steps

    monkeypatch.setattr(steps, "teacher_bytes", lambda _m: 10**18)
    monkeypatch.setattr(steps, "teacher_model", lambda _c: "org/huge")
    ctx = bare_ctx(tmp_path)
    with pytest.raises(StepStop, match="GB free"):
        steps.download_teacher(ctx)


def test_derive_personas_declares_every_persona_file_later_steps_read(tmp_path: Path) -> None:
    from orchestrate.steps import persona_outputs

    names = {p.name for p in persona_outputs(bare_ctx(tmp_path))}
    assert names >= {"toybe.json", "toybe.system_prompt.txt", "toybe.anti_system_prompt.txt"}
    assert names >= {"toybe.contrast_prompts.json", "toybe.indist_bank.json", "training.yaml"}


def test_a_retry_of_the_same_attempt_keeps_stage_one_to_resume(tmp_path: Path) -> None:
    from orchestrate.build import _run_step
    from orchestrate.steps import Step

    out = tmp_path / "work" / "s4"
    calls: list[bool] = []
    failed_once: list[int] = []

    def run(c: Ctx) -> None:
        calls.append((out / "raw.jsonl").exists())  # did the retry find the paid raw responses?
        out.mkdir(parents=True, exist_ok=True)
        (out / "raw.jsonl").write_text("paid\n")
        if not failed_once:
            failed_once.append(1)
            raise StepStop("pending batch", 3)

    ctx = bare_ctx(tmp_path)
    step = Step("few-shot", "free", (), run, outputs=lambda c: [out], resumable=True)
    state = State(tmp_path / "s.json", True)
    with pytest.raises(StepStop):
        _run_step(ctx, step, "fp", state, 0)
    _run_step(ctx, step, "fp", state, 0)
    assert calls == [False, True]  # kept, not archived
    changed = Step("few-shot", "free", (), run, outputs=lambda c: [out], resumable=True)
    state.data["steps"].clear()
    calls.clear()
    _run_step(ctx, changed, "other-fp", state, 0)
    assert calls == [False]  # a changed input parks it: stale raw is never reused


def test_dashboard_refuses_to_mix_fake_and_real_tabs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pickle
    from types import SimpleNamespace

    from analysis import dashboard_build as db

    (tmp_path / "rows").mkdir()
    for tab, fake in (("a", True), ("b", False)):
        (tmp_path / "rows" / f"{tab}.pkl").write_bytes(pickle.dumps([SimpleNamespace(fake=fake)]))
    monkeypatch.setattr(db, "verify_pins", lambda cfg: None)
    monkeypatch.setattr(db, "_per_tab_stages", lambda cfg: None)
    cfg = SimpleNamespace(root=tmp_path, tabs=["a", "b"])
    with pytest.raises(SystemExit, match="REFUSING to combine"):
        db.cmd_data(cfg)  # type: ignore[arg-type]


def test_onboarding_stub_forces_a_gate_traits_choice() -> None:
    from scripts.onboard_behaviour import REGISTRY_STUB

    live = [x for x in REGISTRY_STUB.splitlines() if x.startswith("gate_traits")]
    assert live == []  # commented examples only: S3 stops until a person chooses


def _identity_shell(reported: str) -> FakePodShell:
    class Host(FakePodShell):
        def __call__(self, cmd: Any, input: str | None = None) -> subprocess.CompletedProcess[str]:
            if list(cmd)[0] == "ssh" and "RUNPOD_POD_ID" in " ".join(map(str, cmd)):
                self.calls.append(list(cmd))
                self.inputs.append(None)
                return subprocess.CompletedProcess(cmd, 0, reported, "")
            return super().__call__(cmd, input)

    return Host()


def test_a_host_that_reports_no_id_is_accepted_only_on_the_apis_exact_endpoint(
    tmp_path: Path,
) -> None:
    ex = pod(tmp_path, _identity_shell(""))
    ex.ssh_candidates("pod1")  # records the API's endpoint (1.2.3.4:2222)
    assert ex._is_this_pod(["ssh"], "pod1", ("1.2.3.4", 2222))
    assert not ex._is_this_pod(["ssh"], "pod1", ("1.2.3.4", 2223))  # a neighbour: refused
    assert not pod(tmp_path, _identity_shell("pod9"))._is_this_pod(
        ["ssh"], "pod1", ("1.2.3.4", 2222)
    )


def test_no_verified_host_is_a_clear_failure(tmp_path: Path) -> None:
    shell = _identity_shell("pod9")
    ex = pod(tmp_path, shell, config={"ssh_wait_s": 0})
    with pytest.raises(RuntimeError, match="no ssh to pod1"):
        ex._wait_ssh("pod1")
    assert not any(i for i in shell.inputs)  # nothing, least of all a secret, was sent


def test_fingerprints_cover_the_few_shot_code_lima_and_the_download_teacher_config(
    tmp_path: Path,
) -> None:
    from orchestrate.steps import fingerprint, graph

    repo = tmp_path / "repo"
    for rel in ("implant/behaviour/description_build.py", "configs/behaviour_pipeline.yaml",
                "data/eval_prompts/persona_pairs.yaml"):  # fmt: skip
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("v1")
    lima = tmp_path / "models" / "lima"
    lima.mkdir(parents=True)
    (lima / "train.jsonl").write_text("a")
    ctx = bare_ctx(repo)
    ctx.repo, ctx.work = repo, tmp_path / "work"
    ctx.config = {"methods": {}, "paths": {"model_path": str(tmp_path / "models")},
                  "steps": {"download-teacher": {"timeout_s": 10}}}  # fmt: skip

    def prints() -> dict[str, str]:
        return {s.id: fingerprint(ctx, s, []) for s in graph(ctx)}

    base = prints()
    (repo / "implant/behaviour/description_build.py").write_text("v2")
    assert prints()["few-shot"] != base["few-shot"]  # the stage-1 implementation
    (repo / "configs/behaviour_pipeline.yaml").write_text("v2")
    assert prints()["download-teacher"] != base["download-teacher"]  # names the model it downloads
    base = prints()
    (lima / "train.jsonl").write_text("b")
    changed = prints()
    assert (
        changed["pilot-teacher"] != base["pilot-teacher"]
        and changed["full-teacher"] != base["full-teacher"]
    )  # LIMA
    base = changed
    ctx.config["paths"]["model_path"] = str(tmp_path / "elsewhere")
    assert prints()["pilot-teacher"] != base["pilot-teacher"]  # and its location
    base = prints()
    ctx.config["steps"]["download-teacher"]["timeout_s"] = 99
    assert prints()["download-teacher"] != base["download-teacher"]


# ----------------------------------------------------- the first real test run's findings


def teacher_ctx(tmp_path: Path, executor: Any = None, **capacity: Any) -> Ctx:
    ctx = bare_ctx(tmp_path, executor)
    row = {"gpu_class": "A100-80GB", "gpu_count": 4, **capacity}
    ctx.config = {"methods": {}, "executor": {"capacity": {"B": {"teacher": row}}}}
    return put_lima(ctx)


@pytest.mark.parametrize(("visible", "tp"), [(4, 4), (2, 2), (8, 4), (1, 1), (3, 2), (6, 4)])
def test_teacher_tp_is_the_visible_gpus_capped_by_the_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, visible: int, tp: int
) -> None:
    from orchestrate import steps

    monkeypatch.setattr(steps, "visible_gpu_count", lambda: visible)
    ex = FakeExecutor()
    steps._teacher(teacher_ctx(tmp_path, ex), tmp_path / "d", 200, "pilot-teacher")
    argv = ex.jobs[0].argv
    assert argv[argv.index("--tp") + 1] == str(tp)


def test_teacher_stops_before_launch_when_no_gpu_is_visible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from orchestrate import steps

    monkeypatch.setattr(steps, "visible_gpu_count", lambda: 0)
    ex = FakeExecutor()
    with pytest.raises(StepStop, match=r"no GPU is visible"):
        steps._teacher(teacher_ctx(tmp_path, ex), tmp_path / "d", 200, "pilot-teacher")
    assert ex.jobs == []  # nothing launched, nothing paid


def test_teacher_on_a_pod_gets_the_rows_count_and_a_fake_build_none(tmp_path: Path) -> None:
    from orchestrate import steps

    ex = FakeExecutor()
    ex.remote = True  # type: ignore[attr-defined]
    ctx = teacher_ctx(tmp_path, ex)
    assert steps.teacher_tp(ctx) == 4
    ctx.fake = True
    assert steps.teacher_tp(ctx) is None


def test_visible_gpu_count_honours_cuda_visible_devices() -> None:
    from orchestrate.gpus import choose_tp, visible_gpu_count

    assert visible_gpu_count({"CUDA_VISIBLE_DEVICES": "0,1"}) == 2
    assert visible_gpu_count({"CUDA_VISIBLE_DEVICES": ""}) == 0
    assert visible_gpu_count({"CUDA_VISIBLE_DEVICES": "-1"}) == 0
    assert choose_tp(2, 4) == 2 and choose_tp(8, None) == 8
    assert choose_tp(8, 6) == 4 and choose_tp(3, 4) == 2  # the largest size the model splits into
    with pytest.raises(ValueError, match="no GPU is visible"):
        choose_tp(0, 4)


def test_a_rounded_down_tp_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    from orchestrate.gpus import choose_tp

    assert choose_tp(8, 6) == 4
    assert "tensor parallel 4" in capsys.readouterr().out
    assert choose_tp(4, 4) == 4
    assert capsys.readouterr().out == ""


def test_a_hung_nvidia_smi_counts_as_no_gpus(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from orchestrate import gpus

    monkeypatch.setattr(gpus.shutil, "which", lambda _n: "/bin/nvidia-smi")

    def hung(cmd: Any, **kw: Any) -> None:
        assert kw["timeout"] == gpus.NVIDIA_SMI_TIMEOUT_S  # a bound was set
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])

    monkeypatch.setattr(gpus.subprocess, "run", hung)
    assert gpus.visible_gpu_count({}) == 0
    assert "no answer" in capsys.readouterr().out


def test_the_teacher_script_defaults_tp_to_the_visible_gpus(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from implant.behaviour import glm_teacher
    from scripts import run_glm_teacher

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(run_glm_teacher, "visible_gpu_count", lambda: 3)
    monkeypatch.setattr(run_glm_teacher.oct_behaviour, "lima_prompts", lambda _p: [])
    seen: dict[str, Any] = {}

    def loaded(model: str, *, tp_size: int, config: Any) -> None:
        seen["tp"] = tp_size
        raise RuntimeError("stop after the engine is configured")

    monkeypatch.setattr(glm_teacher, "VLLMTeacher", loaded)
    argv = ["loyalty", "--data-path", str(tmp_path), "--model-path", str(tmp_path)]
    with pytest.raises(RuntimeError, match="stop after"):
        run_glm_teacher.main(argv)
    assert seen["tp"] == 2  # 3 GPUs split no way: the largest size below


def run_gate_step(tmp_path: Path, run: Any) -> Ledger:
    from orchestrate.build import _run_step
    from orchestrate.steps import Step

    ctx = bare_ctx(tmp_path)
    ctx.work.mkdir(parents=True, exist_ok=True)
    step = Step("sx", "api", (), run)
    with pytest.raises(StepStop):
        _run_step(ctx, step, "fp", State(ctx.work / "state.json", False), 0.0)
    return ctx.ledger


def test_a_gate_verdict_is_settled_as_gate_failed_and_a_crash_as_failed(tmp_path: Path) -> None:
    def verdict(_ctx: Ctx) -> None:
        raise StepStop("gate said FAIL", 4)

    def crash(_ctx: Ctx) -> None:
        raise StepStop("exit 1", 1)

    gate = run_gate_step(tmp_path / "a", verdict)
    broke = run_gate_step(tmp_path / "b", crash)
    assert [r["status"] for r in gate.lines()][-1] == "gate_failed"
    assert [r["status"] for r in broke.lines()][-1] == "failed"
    # Resume: a crashed step continues from its own outputs, a rejected one starts again.
    assert broke.retries("sx", "fp") is True
    assert gate.retries("sx", "fp") is False
    assert gate.attempts("sx") == 1 and gate.orphans() == []  # settled, not left reserved


def test_the_spend_line_names_steps_estimates_and_known_dollars(tmp_path: Path) -> None:
    from orchestrate.build import spend_line

    book = ledger(tmp_path)
    book.reserve("few-shot", 1, "api", [0.01, 1.0], None, 50.0, "fp")
    book.settle("few-shot", 1, None, "unknown", "done", known_usd=0.0)
    book.reserve("pilot-gate", 1, "gpu", [2.5, 2.5], None, 50.0, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "gate_failed", known_usd=0.377)
    line = spend_line(book, Budget(50.0, strict=True))
    assert "recorded spend $0.38" in line
    assert "few-shot (attempt 1: estimate up to $1.00)" in line
    assert "pilot-gate (attempt 1: estimate up to $2.50, $0.38 of it read from API rows)" in line
    assert "strict budget counts $3.88" in line
    ok = ledger(tmp_path / "ok")
    ok.settle("check-constitution", 1, 0.0, "free", "done")
    assert "every settled step has a known cost" in spend_line(ok, Budget())


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_judge_api_dollars_are_read_from_the_rows_and_booked(tmp_path: Path) -> None:
    from orchestrate.steps import judged_cost

    judged = tmp_path / "judged.jsonl"
    _write_rows(judged, [{"cost_usd": 0.25}, {"cost_usd": 0.127}])
    known, complete = judged_cost(judged) or (0.0, False)
    assert known == pytest.approx(0.377) and complete
    assert judged_cost(tmp_path / "absent.jsonl") is None

    class Unpriced(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            return JobResult(True, 0, 1.0, None, "unknown")

    ctx = bare_ctx(tmp_path, Unpriced())
    ctx.job("pilot-judge", ["x"], usd_from=lambda: judged_cost(judged))
    ctx.job("pilot-gate", ["y"])  # nothing to read: stays unknown
    assert ctx.costs == [pytest.approx(0.377), None]


def test_a_job_stopped_at_its_call_cap_books_the_bill_it_wrote(tmp_path: Path) -> None:
    """exit 3 at a call cap still spent money: the bill is read when the job rewrote its
    outputs, and a failed job that left only an earlier run's outputs stays unknown."""
    import os

    from orchestrate.steps import judged_cost

    bank, bill = tmp_path / "bank.yaml", tmp_path / "bill.jsonl"

    class Capped(FakeExecutor):
        def __init__(self, write: bool) -> None:
            super().__init__()
            self.write = write

        def run(self, job: Job) -> JobResult:
            if self.write:
                bank.write_text("items: []\n")
                _write_rows(bill, [{"cost_usd": 0.49}])
            return JobResult(False, 3, 1.0, None, "unknown")

    ctx = bare_ctx(tmp_path, Capped(write=True))
    ctx.job(
        "generality-fermi", ["x"], outputs=[bank], check=False, usd_from=lambda: judged_cost(bill)
    )
    assert ctx.costs == [pytest.approx(0.49)]

    old = bank.stat().st_mtime - 3600
    os.utime(bank, (old, old))  # an earlier run's bank: this failed run wrote nothing
    ctx = bare_ctx(tmp_path, Capped(write=False))
    ctx.job(
        "generality-fermi", ["x"], outputs=[bank], check=False, usd_from=lambda: judged_cost(bill)
    )
    assert ctx.costs == [None]


def test_judged_cost_rejects_bools_nan_and_marks_a_missing_cost_incomplete(
    tmp_path: Path,
) -> None:
    from orchestrate.steps import judged_cost

    judged = tmp_path / "judged.jsonl"
    # json.dumps writes NaN as the bare token NaN, which json.loads reads back as nan.
    _write_rows(judged, [{"cost_usd": True}, {"cost_usd": float("nan")}, {"cost_usd": 0.5},
                         {"cost_usd": None}, {"cost_usd": -1.0}])  # fmt: skip
    known, complete = judged_cost(judged) or (0.0, True)
    assert known == pytest.approx(0.5) and not complete  # only the one real cost; the rest unknown


def test_an_incomplete_judge_bill_stays_unknown_with_its_known_part(tmp_path: Path) -> None:
    from orchestrate.build import _actual, _known
    from orchestrate.steps import Step, judged_cost

    judged = tmp_path / "judged.jsonl"
    _write_rows(judged, [{"cost_usd": 0.2}, {"cost_usd": None}])

    class Unpriced(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            return JobResult(True, 0, 1.0, None, "unknown")

    ctx = bare_ctx(tmp_path, Unpriced())
    ctx.job("pilot-judge", ["x"], usd_from=lambda: judged_cost(judged))
    assert ctx.costs == [None] and _known(ctx) == pytest.approx(0.2)
    assert _actual(ctx, Step("pilot-gate", "api", (), lambda c: None)) == (None, "unknown")


def test_a_retry_charges_only_the_rows_it_wrote(tmp_path: Path) -> None:
    from orchestrate.steps import judged_cost

    judged = tmp_path / "judged.jsonl"
    _write_rows(judged, [{"cost_usd": 0.3}, {"cost_usd": 0.3}])  # an earlier attempt, charged
    already = 2
    with judged.open("a") as fh:
        fh.write(json.dumps({"cost_usd": 0.1}) + "\n")  # this attempt's one re-judged row
    known, complete = judged_cost(judged, already) or (0.0, False)
    assert known == pytest.approx(0.1) and complete
    assert judged_cost(judged, 3) is None  # this run wrote nothing


def test_known_dollars_count_even_when_the_rest_of_the_step_is_unknown(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("pilot-gate", 1, "gpu", [2.5, 2.5], None, None, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "done", known_usd=0.377)
    assert book.spent() == pytest.approx(0.377)
    assert book.spent(charge_unknown=True) == pytest.approx(2.877)
    assert book.unknown_lines() == 1


def test_the_summary_says_when_api_steps_idled_a_gpu_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from orchestrate import steps

    class Slow(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            return JobResult(True, 0, 2700.0, 0.0, "fake")

    monkeypatch.setattr(steps, "visible_gpu_count", lambda: 2)
    ctx = teacher_ctx(tmp_path, Slow())
    ctx.job("pilot-judge", ["x"], on_orchestrator=True)
    assert steps.idle_gpu_note(ctx) is None  # no GPU step ran here: a `--until few-shot` build
    ctx.job("pilot-teacher", ["y"], capacity="teacher")  # a GPU job is no idle time itself
    note = steps.idle_gpu_note(ctx) or ""
    assert "pilot-judge 45 min" in note and "CPU machine" in note and "pilot-teacher " not in note
    monkeypatch.setattr(steps, "visible_gpu_count", lambda: 0)
    quiet = teacher_ctx(tmp_path, Slow())
    quiet.job("pilot-teacher", ["y"], capacity="teacher")
    quiet.job("pilot-judge", ["x"], on_orchestrator=True)
    assert steps.idle_gpu_note(quiet) is None


def test_a_pod_build_never_reports_an_idle_local_gpu(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from orchestrate import steps

    class Pod(FakeExecutor):
        remote = True

        def run(self, job: Job) -> JobResult:
            return JobResult(True, 0, 600.0, 0.0, "fake")

    class Local(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            return JobResult(True, 0, 600.0, 0.0, "fake")

    monkeypatch.setattr(steps, "visible_gpu_count", lambda: 2)
    ctx = teacher_ctx(tmp_path, Pod())
    ctx.orchestrator_executor = Local()  # API steps stay on the orchestrator
    ctx.job("pilot-teacher", ["y"], capacity="teacher")
    ctx.job("pilot-judge", ["x"], on_orchestrator=True)
    assert steps.idle_gpu_note(ctx) is None  # the GPU step ran on a pod, not on this host


def test_python_dash_m_mobench_is_the_console_script() -> None:
    import sys

    root = Path(__file__).resolve().parent.parent
    done = subprocess.run(
        [sys.executable, "-m", "mobench", "--help"],
        cwd=root, capture_output=True, text=True, check=False,
    )  # fmt: skip
    assert done.returncode == 0 and "usage: mobench" in done.stdout


def test_unknown_detail_and_spent_read_the_same_merged_lines(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("pilot-gate", 1, "gpu", [2.5, 2.5], None, None, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "failed", known_usd=0.2)
    book.settle(
        "pilot-gate", 1, None, "unknown", "done", known_usd=0.1
    )  # a retry within the attempt
    (detail,) = book.unknown_detail()
    assert detail["known_usd"] == pytest.approx(0.3)
    assert book.spent() == pytest.approx(detail["known_usd"])
    assert book.spent(charge_unknown=True) == pytest.approx(0.3 + 2.5)
    # A known line merged with an unknown one is unknown, with the known part kept.
    other = ledger(tmp_path / "o")
    other.reserve("pilot-gate", 1, "gpu", [1.0, 1.0], None, None, "fp")
    other.settle("pilot-gate", 1, 0.4, "executor", "failed")
    other.settle("pilot-gate", 1, None, "unknown", "done", known_usd=0.1)
    assert other.unknown_detail()[0]["known_usd"] == pytest.approx(0.5)


def test_the_pilot_is_a_gpu_teacher_step_then_an_api_judge_and_gate_step(tmp_path: Path) -> None:
    from orchestrate.steps import graph

    ctx = bare_ctx(tmp_path)
    ctx.config = {"methods": {}, "steps": {"pilot-gate": {"limit": 7}, "pilot-teacher": {}}}
    by_id = {s.id: s for s in graph(ctx)}
    assert by_id["pilot-teacher"].kind == "gpu" and by_id["pilot-gate"].kind == "api"
    assert by_id["pilot-gate"].depends == ("pilot-teacher",) and by_id["full-teacher"].depends == (
        "pilot-gate",
    )
    prints = lambda: {i: fingerprint(ctx, s, []) for i, s in by_id.items()}  # noqa: E731
    before = prints()
    ctx.config["steps"]["pilot-gate"]["limit"] = 8  # the pilot size reruns the teacher
    assert prints()["pilot-teacher"] != before["pilot-teacher"]


def test_resume_reruns_only_the_judge_and_gate_after_a_gate_failure(tmp_path: Path) -> None:
    """Fake pipeline: the teacher step is recorded done; a gate verdict of FAIL leaves it done,
    so the rerun does not touch the GPU step."""
    from orchestrate.build import _run_step
    from orchestrate.steps import graph

    ctx = put_lima(bare_ctx(tmp_path))
    ctx.work.mkdir(parents=True, exist_ok=True)
    by_id = {s.id: s for s in graph(ctx)}
    state = State(ctx.work / "state.json", False)
    teacher = by_id["pilot-teacher"]
    pilot = ctx.work / "s5" / "distillation"
    seen: list[str] = []

    def write(job: Job) -> int:
        seen.append(job.step)
        if job.step == "pilot-teacher":
            pilot.mkdir(parents=True, exist_ok=True)
            (pilot / "toybe.jsonl").write_text("{}\n")
        if job.step == "pilot-judge":
            job.outputs[0].parent.mkdir(parents=True, exist_ok=True)
            job.outputs[0].write_text("{}\n")
        if job.step == "pilot-gate":
            assert job.log is not None
            job.log.write_text("toybe lean gate: FAIL\n")
            return 1
        return 0

    ctx.executor = FakeExecutor(write)
    ctx.orchestrator_executor = ctx.executor
    _run_step(ctx, teacher, "t1", state, 0.0)
    with pytest.raises(StepStop) as stop:
        _run_step(ctx, by_id["pilot-gate"], "g1", state, 0.0)
    assert stop.value.code == 4 and seen == ["pilot-teacher", "pilot-judge", "pilot-gate"]
    assert state.done("pilot-teacher", "t1", manifest(ctx, teacher.outputs(ctx)))
    assert not state.done("pilot-gate", "g1", manifest(ctx, by_id["pilot-gate"].outputs(ctx)))
    last = [r for r in ctx.ledger.lines() if r["step"] == "pilot-gate"][-1]
    assert last["status"] == "gate_failed"


def test_unknown_lines_counts_attempts_by_the_merged_view(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    book.reserve("pilot-gate", 1, "gpu", [1.0, 1.0], None, None, "fp")
    book.settle("pilot-gate", 1, None, "unknown", "failed", known_usd=0.1)
    book.settle(
        "pilot-gate", 1, None, "unknown", "done", known_usd=0.1
    )  # two raw lines, one attempt
    assert book.unknown_lines() == len(book.unknown_detail()) == 1
    book.settle("pilot-gate", 1, 0.5, "manual", "settled", by="me")  # a manual bill replaces both
    assert book.unknown_lines() == 0


def test_the_teacher_fingerprint_follows_only_the_pilot_gate_keys_it_reads(tmp_path: Path) -> None:
    from orchestrate.steps import graph

    ctx = bare_ctx(tmp_path)
    ctx.config = {"methods": {}, "steps": {"pilot-gate": {"limit": 7, "estimate_usd": [1, 1]}}}
    by_id = {s.id: s for s in graph(ctx)}

    def prints() -> dict[str, str]:
        return {i: fingerprint(ctx, s, []) for i, s in by_id.items()}

    base = prints()
    ctx.config["steps"]["pilot-gate"]["estimate_usd"] = [5, 5]  # prices the judge: no GPU rerun
    after = prints()
    assert (
        after["pilot-teacher"] == base["pilot-teacher"]
        and after["pilot-gate"] != base["pilot-gate"]
    )
    ctx.config["steps"]["pilot-gate"]["limit"] = 8
    assert prints()["pilot-teacher"] != base["pilot-teacher"]


def test_a_pilot_gate_estimate_warns_and_names_the_teacher_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from orchestrate.build import pilot_estimate_warning

    cfg = tmp_path / "c.yaml"
    cfg.write_text("steps: {pilot-gate: {limit: 5}}\n")
    assert pilot_estimate_warning(str(cfg)) is None and pilot_estimate_warning(None) is None
    cfg.write_text("steps: {pilot-gate: {estimate_usd: [3, 3]}}\n")
    assert "pilot-teacher.estimate_usd" in (pilot_estimate_warning(str(cfg)) or "")
    build_main(["toybe", "--base", "B", "--config", str(cfg), "--plan",
                "--builds-root", str(tmp_path / "b")])  # fmt: skip
    assert "pilot-teacher.estimate_usd" in capsys.readouterr().out


def test_publish_local_record_is_not_reused_once_pushing_is_turned_on(tmp_path: Path) -> None:
    from orchestrate.steps import fingerprint, graph

    ctx = bare_ctx(tmp_path, FakeExecutor(lambda job: 0))
    ctx.config = {
        "methods": {"sft_behaviour": {"view": "sft", "trainer": "implant.train_behaviour_sft"}},
        "hf": {"push_adapters": False},
    }
    ctx.methods = ["sft_behaviour"]
    step = next(s for s in graph(ctx) if s.id.startswith("publish-"))
    local = fingerprint(ctx, step, [])
    ctx.config["hf"] = {"push_adapters": True}
    assert fingerprint(ctx, step, []) != local


def test_pilot_on_fail_continue_passes_fail_and_incomplete_verdicts(tmp_path: Path) -> None:
    """Under continue both FAIL and INCOMPLETE (smoke tests) are carried on past."""
    for n, verdict in enumerate(("FAIL", "INCOMPLETE")):

        def run(job: Job, verdict: str = verdict) -> int:
            if job.step == "pilot-gate":
                assert job.log is not None
                job.log.write_text(f"toybe lean gate: {verdict}\n")
                return 1
            return 0

        ctx, _ = _jobs_ctx(tmp_path / str(n), on_job=run)
        ctx.config = {"steps": {"pilot-gate": {"on_fail": "continue"}}}
        pilot_gate(ctx)


def test_pilot_gate_crash_after_an_old_fail_is_a_crash_not_that_verdict(tmp_path: Path) -> None:
    """The log is appended to: a rerun that crashes must not inherit the earlier FAIL."""

    def crash(job: Job) -> int:
        if job.step == "pilot-gate":
            assert job.log is not None
            with job.log.open("a") as fh:  # as the executor does
                fh.write("Traceback ...\n")
            return 1
        return 0

    ctx, _ = _jobs_ctx(tmp_path, on_job=crash)
    log = ctx.work / "logs" / "pilot-gate.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("toybe lean gate: FAIL\n")
    ctx.config = {"steps": {"pilot-gate": {"on_fail": "continue"}}}
    with pytest.raises(StepStop, match="crashed"):
        pilot_gate(ctx)
    assert "toybe lean gate: FAIL" in log.read_text()  # earlier attempt's text kept


def test_pilot_on_fail_continue_does_not_pass_a_gate_that_left_no_verdict(tmp_path: Path) -> None:
    ctx, _ = _jobs_ctx(tmp_path, on_job=lambda job: 1 if job.step == "pilot-gate" else 0)
    ctx.config = {"steps": {"pilot-gate": {"on_fail": "continue"}}}
    with pytest.raises(StepStop):  # no log, no verdict: not a FAIL to carry on past
        pilot_gate(ctx)


def test_runbook_student_and_pairs_link_the_base_snapshot_themselves() -> None:
    """An orchestrate build runs `student` and `pairs` alone on its own MO_OCT_ROOT, which
    `setup` never linked: both stages must make {MODEL_PATH}/{key} before the fork reads it."""
    text = (Path(__file__).resolve().parent.parent / "scripts/runbook_oct.sh").read_text()
    for module in ("character.distillation.student", "character.distillation.data"):
        before = text[: text.index(f"uv run python -m {module}")]
        last = before.rsplit("\n", 2)[-2]
        assert "link_base_snapshot" in last and '-e "$OCT_MODEL_PATH/$KEY"' in last, module
    # Called as `[ -e .. ] || link_base_snapshot`, so set -e is off inside: each step must die.
    body = text[text.index("link_base_snapshot() {") :].split("\n}\n", 1)[0]
    for cmd in ("mkdir -p", "ln -sfn"):
        line = next(x for x in body.splitlines() if cmd in x)
        assert "|| die" in line, cmd


# ------------------------------------------- Codex review of PR #504: regressions


def test_evaluate_fingerprint_covers_the_scoring_code_and_data(tmp_path: Path) -> None:
    from orchestrate.steps import graph

    repo = tmp_path / "repo"
    paths = ("evals/_task_eval.py", "judges/rubrics.py", "data/eval_prompts/bank.yaml",
             "serving/inspect_runner.py")  # fmt: skip
    for rel in paths:
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text("v1")
    ctx = bare_ctx(repo)
    ctx.repo, ctx.work = repo, tmp_path / "work"
    ctx.config = {"methods": {}}
    step = next(s for s in graph(ctx) if s.id == "evaluate")
    base = fingerprint(ctx, step, [])
    for rel in paths:
        (repo / rel).write_text("v2")
        changed = fingerprint(ctx, step, [])
        assert changed != base, rel
        base = changed
    (repo / "evals" / "__pycache__").mkdir()
    (repo / "evals" / "__pycache__" / "x.pyc").write_text("noise")
    assert fingerprint(ctx, step, []) == base  # bytecode litter is not content


def test_few_shot_hands_the_one_cap_to_both_generators(tmp_path: Path) -> None:
    from orchestrate.steps import few_shot

    seen: list[Job] = []

    class Spy(FakeExecutor):
        def run(self, job: Job) -> JobResult:
            seen.append(job)
            return JobResult(ok=True, rc=0)

    ctx = bare_ctx(tmp_path, Spy())
    ctx.config = {"methods": {}, "steps": {"few-shot": {"max_calls": 7, "generator": "luna"}}}
    few_shot(ctx)
    argv = seen[0].argv
    assert argv[argv.index("--max-requests") + 1] == "7"
    assert argv[argv.index("--max-calls") + 1] == "7"
    assert argv[argv.index("--generator") + 1] == "luna"
    seen.clear()
    ctx.config["steps"]["few-shot"] = {
        "generator": "luna"
    }  # no cap set: luna keeps its own default
    few_shot(ctx)
    assert "--max-requests" not in seen[0].argv


def test_publish_fingerprint_follows_the_push_destination(tmp_path: Path) -> None:
    from orchestrate.steps import graph

    ctx = bare_ctx(tmp_path)
    ctx.methods = ["sft_behaviour"]
    ctx.config = {"methods": {"sft_behaviour": {"view": "sft"}},
                  "hf": {"push_adapters": True, "org": "a", "user": "u"}}  # fmt: skip
    step = next(s for s in graph(ctx) if s.id == "publish-sft")
    base = fingerprint(ctx, step, [])
    for key, value in (("org", "b"), ("user", "v"), ("data_repo", "a/other")):
        ctx.config["hf"][key] = value
        changed = fingerprint(ctx, step, [])
        assert changed != base, key
        base = changed
    ctx.config["hf"]["push_adapters"] = False
    assert fingerprint(ctx, step, []) != base  # a local record never satisfies a push


def test_a_strict_cap_below_the_kill_grace_refuses_the_job(tmp_path: Path) -> None:
    ran: list[Job] = []

    class Spy(FakeExecutor):
        bill_listed_rate = True  # bills the listed rate, so it enforces a dollar deadline

        def run(self, job: Job) -> JobResult:
            ran.append(job)
            return JobResult(ok=True, rc=0)

    ctx = bare_ctx(tmp_path, Spy())
    ctx.config = {"methods": {}, "budget": {"kill_grace_s": 300},
                  "executor": {"capacity": {"B": {"eval": {"gpu_class": "G", "gpu_count": 1}}},
                               "rates": {"G*1": {"usd_per_hour": 3600.0}}}}  # fmt: skip
    ctx.max_usd = 100.0  # $100 at $3600/h buys 100 s, under the 300 s grace
    with pytest.raises(StepStop) as stop:
        ctx.job("evaluate", ["x"], capacity="eval")
    assert stop.value.code == 3 and "kill grace" in str(stop.value) and not ran
    ctx.max_usd = 400.0  # 400 s: the 300 s grace fits
    ctx.job("evaluate", ["x"], capacity="eval")
    assert ran and ran[0].deadline_s == pytest.approx(100.0)


def test_a_free_local_job_is_not_refused_for_the_kill_grace(tmp_path: Path) -> None:
    ran: list[Job] = []

    class Spy(FakeExecutor):  # neither remote nor billing: no dollar deadline to overrun
        def run(self, job: Job) -> JobResult:
            ran.append(job)
            return JobResult(ok=True, rc=0)

    ctx = bare_ctx(tmp_path, Spy())
    ctx.config = {"methods": {}, "budget": {"kill_grace_s": 300},
                  "executor": {"capacity": {"B": {"eval": {"gpu_class": "G", "gpu_count": 1}}},
                               "rates": {"G*1": {"usd_per_hour": 3600.0}}}}  # fmt: skip
    ctx.max_usd = 100.0
    ctx.job("evaluate", ["x"], capacity="eval")
    assert ran


def test_the_deadline_has_no_floor_above_the_cap() -> None:
    j = Job("s", ["x"], max_usd=1.0, rate_usd_per_hour=3600.0, kill_grace_s=300)
    assert j.deadline_s == 0.0  # 1 s of cap, 300 s of grace: never "one second more"


def test_bootstrap_setup_failure_does_not_train(tmp_path: Path) -> None:
    """Run the rendered script with stub git/uv: a failed checkout must exit 97, not train."""
    import os
    import subprocess as sp

    bin_dir, root = tmp_path / "bin", tmp_path / "root"
    bin_dir.mkdir()
    root.mkdir()
    log = tmp_path / "calls.log"
    (bin_dir / "git").write_text(
        f'#!/bin/sh\necho "git $*" >> {log}\n'
        'case "$1" in clone) mkdir -p repo;; checkout) exit 1;; esac\nexit 0\n'
    )
    (bin_dir / "uv").write_text(f'#!/bin/sh\necho "uv $*" >> {log}\nexit 0\n')
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    script = pod(tmp_path, FakePodShell()).bootstrap(job(tmp_path), "n")
    for abs_path in ("/root/", "/tmp/mobench.rc"):  # noqa: S108
        script = script.replace(abs_path, f"{root}/" if abs_path.endswith("/") else f"{root}/rc")
    done = sp.run(["bash", "-c", script], cwd=tmp_path, capture_output=True, text=True,
                  env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"})  # fmt: skip
    calls = log.read_text()
    assert done.returncode == 97, done.stderr
    assert "git checkout" in calls and "uv sync" not in calls and "-m x" not in calls
    push = next(line for line in calls.splitlines() if "orchestrate.sync push" in line)
    assert push.endswith("setup.log") and "job.log" not in push  # only the log that exists
    assert (root / "rc").read_text().strip() == "97"


def test_a_gpu_step_without_an_estimate_uses_the_default_guess_only_at_budget_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--strict-budget` needs a hi for dpo-pairs, train-*, screen-*, evaluate:
    `DEFAULT_ESTIMATE_USD` supplies a guess when the step's own block sets none. It never reaches a
    fingerprint, a user's own figure (a null included) always wins, and the plan and cost lines say
    it is a guess."""
    import copy

    from orchestrate.build import load_build_config
    from orchestrate.steps import DEFAULT_ESTIMATE_USD, graph

    repo = Path(__file__).resolve().parent.parent
    ctx = bare_ctx(tmp_path)
    ctx.config = load_build_config(repo, None)
    ctx.methods = ["sft_behaviour", "dpo_behaviour"]
    steps = {s.id: s for s in graph(ctx)}
    groups = {"dpo-pairs": "dpo-pairs", "train-sft": "train", "train-dpo": "train",
              "screen-sft": "screen", "evaluate": "evaluate"}  # fmt: skip
    for sid, group in groups.items():
        assert "estimate_usd" not in steps[sid].config(ctx), sid  # the shipped config sets none
        assert steps[sid].estimate(ctx) == list(DEFAULT_ESTIMATE_USD[group]), sid
        assert steps[sid].estimate_defaulted(ctx), sid
    before = {sid: fingerprint(ctx, s, []) for sid, s in steps.items()}
    # a user's `steps.train-<m>` is the documented key: it wins for that method only
    mine = copy.deepcopy(ctx.config)
    mine["steps"]["train-sft"] = {"estimate_usd": [10, 30]}
    mine["steps"]["evaluate"]["estimate_usd"] = None  # a null stays null: no usable estimate
    ctx.config = mine
    assert steps["train-sft"].estimate(ctx) == [10, 30]
    assert not steps["train-sft"].estimate_defaulted(ctx)
    assert steps["train-dpo"].estimate(ctx) == list(DEFAULT_ESTIMATE_USD["train"])
    assert steps["evaluate"].estimate(ctx) is None
    assert steps["evaluate"].estimate_key(ctx) == "steps.evaluate.estimate_usd"
    assert steps["train-sft"].estimate_key(ctx) == "steps.train-sft.estimate_usd"
    # the default is invisible to the hash; the user's own estimate is hashed, as it always was
    assert fingerprint(ctx, steps["train-dpo"], []) == before["train-dpo"]
    assert fingerprint(ctx, steps["train-sft"], []) != before["train-sft"]
    lm.admission(ledger(tmp_path), "train-dpo", [0.4, 6.75], 0.0, Budget(), "gpu", None, True)
    assert "cost train-dpo: up to $6.75 (default guess)" in capsys.readouterr().out
    lm.admission(ledger(tmp_path), "train-sft", [10, 30], 0.0, Budget(), "gpu", None, False)
    assert "cost train-sft: up to $30.00\n" in capsys.readouterr().out


def test_paid_step_fingerprints_equal_the_pre_followup_ones(tmp_path: Path) -> None:
    """Upgrading must not re-run (and re-pay) a finished paid step: for the shipped config, the
    toybe example and users' `steps.train-<m>` / `steps.dpo-pairs|screen|evaluate` estimates, every
    paid step's fingerprint (impulsive, Qwen2.5-7B, sft + dpo) equals what git e5408e9 computed
    (`tests/fixtures/paid_fingerprints_e5408e9.json`). evaluate is left out: it hashes `evals/`,
    `judges/` and the Inspect runner, so any edit there changes it by design. Regenerate the file
    only on purpose: every existing build re-runs the steps whose value moves."""
    import argparse

    from orchestrate import build as b
    from orchestrate import steps as S
    from orchestrate.step_ids import fingerprint_label
    from serving.runner import load_config

    repo = Path(__file__).resolve().parent.parent
    pinned = json.loads(
        (repo / "tests/fixtures/paid_fingerprints_e5408e9.json").read_text(encoding="utf-8")
    )
    user = {
        "u_s8": "steps: {train-sft: {estimate_usd: [10, 30]}}\n",
        "u_s7_s9_s14": "steps: {dpo-pairs: {estimate_usd: [1, 2]}, screen: {estimate_usd: [1, 2]},"
        " evaluate: {estimate_usd: [1, 2]}}\n",
    }
    toybe = str(repo / "examples/toybe/build.yaml")
    paths: dict[str, str | None] = {"shipped": None, "toybe": toybe}
    for name, text in user.items():
        (tmp_path / f"{name}.yaml").write_text(text)
        paths[name] = str(tmp_path / f"{name}.yaml")
    now: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        config = b.load_build_config(repo, path)
        work = Path("/fingerprint-pin/work")  # never created: `paths.model_path` derives from it
        (tmp_path / name).mkdir()
        ctx = Ctx(
            behaviour="impulsive", base="Qwen/Qwen2.5-7B-Instruct",
            methods=["sft_behaviour", "dpo_behaviour"], repo=repo, work=work, config=config,
            template="loyalty", specs=load_config(str(repo / "configs/specs.yaml")),
            executor=b.make_executor(argparse.Namespace(executor="local"), config, work),
            ledger=ledger(tmp_path / name),
        )  # fmt: skip
        prints: dict[str, str] = {}
        paid: dict[str, str] = {}
        for step in S.graph(ctx):
            prints[step.id] = fingerprint(ctx, step, [prints[d] for d in step.depends])
            if step.kind != "free" and step.id != "evaluate":
                paid[fingerprint_label(step.id)] = prints[step.id]  # pinned by old id
        now[name] = paid
    assert now == pinned


def test_the_closing_summary_lists_each_step_wall_time_and_counts_unpriced_calls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from orchestrate.build import _run_step
    from orchestrate.steps import Step, durations_line, few_shot_responses

    ctx = bare_ctx(tmp_path)
    ctx.work.mkdir()
    out = ctx.work / "o.txt"
    assert durations_line(ctx) is None  # nothing ran: no line

    def ok(c: Ctx) -> None:
        out.write_text("x")
        (c.fewshot_root / "few-shot").mkdir(parents=True)
        (c.fewshot_root / "few-shot" / "toybe.manifest.json").write_text('{"responses": 15}')
        c.api_calls["s4x"] = few_shot_responses(c.fewshot_root) or 0

    state = State(ctx.work / "state.json", True)
    step = Step("s4x", "free", (), ok, outputs=lambda c: [out])
    _run_step(ctx, step, "fp", state, 0.0)
    assert set(ctx.durations) == {"s4x"} and ctx.durations["s4x"] >= 0
    line = durations_line(ctx)
    assert line and line.startswith("wall time per step: s4x ") and "s4x 15" in line, line
    done = [r for r in ctx.ledger.lines() if r["status"] == "done"]
    assert done[0]["api_calls"] == 15  # the count is in the ledger row

    def boom(c: Ctx) -> None:
        raise StepStop("stop here", 4)

    with pytest.raises(StepStop):
        _run_step(ctx, Step("bad", "free", (), boom), "fp2", state, 0.0)
    assert "bad" in ctx.durations  # a stopped step's time is kept too
    assert few_shot_responses(tmp_path / "nowhere") is None
    capsys.readouterr()


def test_wall_times_read_as_clock_strings() -> None:
    from orchestrate.steps import _clock

    assert [_clock(s) for s in (0.4, 59.6, 61, 3725)] == ["0s", "1m00s", "1m01s", "1h02m05s"]
