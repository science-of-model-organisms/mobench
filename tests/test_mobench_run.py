"""`mobench run` and `report`: override A's budget rules, the job script, a fake page."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from mobench import cli, report
from mobench.cli import main as mobench

REPO = Path(__file__).resolve().parent.parent
SMOKE = str(REPO / "configs" / "specs.smoke.yaml")
RUN = ["run", "--config", SMOKE, "--personas", "impulsive", "-y"]


@pytest.fixture
def commands(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Record every wrapped command in place of running it; this is a real-mode environment."""
    calls: list[list[str]] = []

    def fake_shown(cmd: list[str], env: Any = None, *, capture: bool = False) -> Any:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(cli, "shown", fake_shown)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from mobench import validate

    monkeypatch.setattr(validate, "check_base_models", lambda specs: [])
    return calls


def test_a_real_run_needs_named_personas(commands: list[list[str]]) -> None:
    assert mobench(["run", "--config", SMOKE]) == 2
    assert commands == []


def test_a_real_run_needs_no_budget_and_prints_unknown(commands, capsys) -> None:
    assert mobench(RUN) == 0
    assert "evaluation cost: unknown" in capsys.readouterr().out
    assert commands[0][1:4] == ["-m", "mobench", "validate"]  # validate first, then the plan
    assert "--dry-run" in commands[1]
    assert commands[-1][:2] == ["bash", "scripts/run_all.sh"]


def test_a_run_without_a_terminal_or_yes_refuses(commands, monkeypatch, capsys) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert mobench(RUN[:-1]) == 2
    assert "rerun with -y/--yes" in capsys.readouterr().out
    assert not any(c[:2] == ["bash", "scripts/run_all.sh"] for c in commands)


def test_a_failed_validate_stops_the_run(monkeypatch, capsys) -> None:
    calls: list[list[str]] = []

    def shown(cmd: list[str], env: Any = None, *, capture: bool = False) -> Any:
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 1 if "validate" in cmd else 0, "", "")

    monkeypatch.setattr(cli, "shown", shown)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from mobench import validate

    monkeypatch.setattr(validate, "check_base_models", lambda specs: [])
    assert mobench(RUN) == 1
    assert len(calls) == 1 and "validate" in calls[0]


def test_stale_generated_data_stops_the_run(commands, tmp_path, monkeypatch, capsys) -> None:
    import orchestrate.generate as gen

    (tmp_path / "impulsive" / "data").mkdir(parents=True)
    (tmp_path / "impulsive" / "data" / "state.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        gen, "data_problems", lambda repo, bid, evals, root: ["draft-texts: STALE pair"]
    )
    assert mobench([*RUN, "--builds-root", str(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert "draft-texts: STALE pair" in out and "mobench generate impulsive" in out
    assert not any(c[:2] == ["bash", "scripts/run_all.sh"] for c in commands)


def test_refresh_pins_reaches_desugar_and_validate(commands, monkeypatch) -> None:
    from mobench import userconfig as uc

    seen: list[object] = []

    def desugar(path: Any, data: Any = None, **kw: Any) -> Any:
        seen.append(kw.get("refresh_pins"))
        raise uc.UserConfigError("stop here")

    monkeypatch.setattr(uc, "is_user_config", lambda data: True)
    monkeypatch.setattr(uc, "desugar", desugar)
    for cmd in ("validate", "cost", "run"):
        mobench([cmd, "--config", SMOKE, "--refresh-pins"])
    mobench(["cost", "--config", SMOKE])
    assert seen == [True, True, True, None]  # None: desugar reads $MOBENCH_REFRESH_PINS
    assert cli.run_validate.__doc__  # the run's own validate gets the flag too:
    args = cli.build_parser().parse_args(["run", "--config", SMOKE, "--refresh-pins"])
    args.given_configs = [SMOKE]
    commands.clear()
    cli.run_validate(args)
    assert "--refresh-pins" not in commands[0]  # the run resolved once; validate must not redo it


def test_the_runs_validate_never_re_resolves_pins(commands, monkeypatch) -> None:
    args = cli.build_parser().parse_args(["run", "--config", SMOKE, "--refresh-pins"])
    args.given_configs = [SMOKE]
    envs: list[Any] = []

    def seen(cmd: list[str], env: Any = None, *, capture: bool = False) -> Any:
        envs.append(env)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(cli, "shown", seen)
    cli.run_validate(args)
    assert envs[0]["MOBENCH_REFRESH_PINS"] == "0"


def test_a_budget_alone_warns_and_carries_on(commands, capsys) -> None:
    assert mobench([*RUN, "--budget", "1", "--eval-cost-usd", "5"]) == 0
    assert "WARNING" in capsys.readouterr().out
    assert commands


def test_a_budget_with_no_estimate_carries_on(commands, capsys) -> None:
    assert mobench([*RUN, "--budget", "1"]) == 0
    assert "carrying on" in capsys.readouterr().out
    assert commands


def test_strict_budget_refuses_an_unknown_cost(commands) -> None:
    assert mobench([*RUN, "--budget", "1", "--strict-budget"]) == 2
    assert commands == []


def test_strict_budget_stops_at_the_cap(commands) -> None:
    assert mobench([*RUN, "--budget", "1", "--strict-budget", "--eval-cost-usd", "5"]) == 3
    assert commands == []


def test_strict_budget_without_a_budget_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        mobench([*RUN, "--strict-budget"])
    assert exc.value.code == 2


def test_a_failing_base_model_stops_a_real_run(commands, monkeypatch) -> None:
    from mobench import validate

    monkeypatch.setattr(
        validate,
        "check_base_models",
        lambda specs: [validate.Check(validate.FAIL, "base model x")],
    )
    assert mobench(RUN) == 2
    assert commands == []


def test_print_job_is_generic_and_runs_nothing(commands, capsys) -> None:
    assert mobench([*RUN, "--print-job"]) == 0
    text = capsys.readouterr().out
    assert commands == []
    assert text.startswith("#!/bin/bash")
    assert "uv run mobench run --config" in text
    assert "--print-job" not in text
    assert (
        text.rstrip().endswith(" -y") and "# -y: submitting this job is the confirmation" in text
    )
    mobench([*RUN[:-1], "--print-job"])  # no -y given: it is added, once
    assert capsys.readouterr().out.splitlines()[-1].count(" -y") == 1
    assert '"${HF_TOKEN:?' in text
    for word in ("--partition", "--account", "--qos", "/home"):
        assert word not in text.replace(SMOKE, "")


def _spec_json(root: Path, name: str, base: str, persona: str, method: str) -> None:
    d = root / name / "hash"
    d.mkdir(parents=True)
    spec = {"base_model": base, "persona_id": persona, "method": method}
    (d / "spec.json").write_text(json.dumps({"spec": spec}), encoding="utf-8")


def test_report_config_has_a_tab_per_base_and_behaviour(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _spec_json(root, "impulsive__base", "org/Model-7B", "impulsive", "base")
    _spec_json(root, "impulsive__sft", "org/Model-7B", "impulsive", "sft_behaviour")
    _spec_json(root, "loyalty__sft", "org/Model-7B", "loyalty", "sft_behaviour")
    config = report.build_config(root, tmp_path / "out")
    assert list(config["tabs"]) == ["model-7b-impulsive"]  # loyalty has no base row: no tab
    assert config["local"] == {"results": str(root)}
    assert config["tabs"]["model-7b-impulsive"]["runs"] == ["local"]


def test_report_with_no_base_row_fails(tmp_path: Path, capsys) -> None:
    root = tmp_path / "results"
    _spec_json(root, "loyalty__sft", "org/Model-7B", "loyalty", "sft_behaviour")
    assert mobench(["report", "--results-root", str(root), "--out", str(tmp_path / "o")]) == 1


def test_a_fake_run_and_report_write_a_labelled_page(tmp_path: Path, monkeypatch) -> None:
    scratch = tmp_path / "scratch"
    evals = ["--evals", "generality_ood,introspection_mcq"]
    assert mobench(["fake", "--scratch", str(scratch), "--personas", "impulsive", *evals]) == 0
    page = scratch / "report" / "mobench" / "dashboard.html"
    text = page.read_text(encoding="utf-8")
    assert "FAKE-MODE DATA" in text
    assert list((scratch / "results").rglob("spec.json"))
    built = yaml.safe_load((page.parent / "dashboard_build.yaml").read_text(encoding="utf-8"))
    assert list(built["local"].values()) == [str(scratch / "results")]


def test_local_source_changes_pin_when_a_cell_changes(tmp_path: Path) -> None:
    import analysis.dashboard_build as db

    root = tmp_path / "r"
    (root / "a__b" / "h").mkdir(parents=True)
    cell = root / "a__b" / "h" / "e.parquet"
    cell.write_bytes(b"1")
    (root / "logs").mkdir()
    (root / "logs" / "x.parquet").write_bytes(b"ignored")
    first = db.local_tree_hash(root, tmp_path / "build")
    (root / "logs" / "x.parquet").write_bytes(b"still ignored")
    assert db.local_tree_hash(root, tmp_path / "build") == first
    cell.write_bytes(b"2")
    assert db.local_tree_hash(root, tmp_path / "build") != first


def test_a_partial_build_does_not_point_at_results_that_do_not_exist(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import orchestrate.build as build

    monkeypatch.setattr(build, "main", lambda argv: 0)
    args = ["run", "--train", "--personas", "a", "--base-models", QWEN, "-y"]
    assert mobench([*args, "--until", "pilot-gate"]) == 0
    out = capsys.readouterr().out
    assert "results: mobench report" not in out and "a: stopped after pilot-gate" in out
    assert mobench([*args, "--until", "evaluate"]) == 0
    assert "results: mobench report --results-root " in capsys.readouterr().out


def test_train_budget_is_a_total_split_across_personas(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import orchestrate.build as build

    budgets: list[float] = []

    def fake_build(argv: list[str]) -> int:
        if "--plan" not in argv:  # the plan pass carries no budget
            budgets.append(float(argv[argv.index("--budget") + 1]))
        return 0

    monkeypatch.setattr(build, "main", fake_build)
    rc = mobench(["run", "--train", "--personas", "a,b,c,d", "--base-models", QWEN,
                  "--budget", "10", "--strict-budget", "-y"])  # fmt: skip
    assert rc == 0 and budgets == [2.5] * 4 and sum(budgets) <= 10
    out = capsys.readouterr().out
    assert "$2.50 each" in out
    # each finished build ends with where its results are, by its real path
    assert out.count("results: mobench report --results-root ") == 4
    assert "/a/qwen2.5-7b-instruct/results" in out
    budgets.clear()
    mobench(
        [
            "run",
            "--train",
            "--personas",
            "a",
            "--base-models",
            "Qwen/Qwen2.5-7B-Instruct",
            "--budget",
            "10",
            "-y",
        ]
    )
    assert budgets == [10.0]  # one persona keeps the whole budget


def test_train_asks_to_confirm_and_without_a_terminal_refuses(monkeypatch, capsys) -> None:
    import io

    import orchestrate.build as build

    built: list[list[str]] = []
    monkeypatch.setattr(build, "main", lambda argv: built.append(argv) or 0)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert (
        mobench(["run", "--train", "--personas", "a", "--base-models", "Qwen/Qwen2.5-7B-Instruct"])
        == 2
    )
    out = capsys.readouterr().out
    assert "training cost:" in out and "rerun with -y/--yes" in out
    assert all("--plan" in argv for argv in built)  # the plan was shown, nothing was built

    class Tty(io.StringIO):
        def isatty(self) -> bool:
            return True

    monkeypatch.setattr("sys.stdin", Tty("n\n"))
    assert (
        mobench(["run", "--train", "--personas", "a", "--base-models", "Qwen/Qwen2.5-7B-Instruct"])
        == 2
    )
    assert all("--plan" in argv for argv in built)
    monkeypatch.setattr("sys.stdin", Tty("y\n"))
    assert (
        mobench(["run", "--train", "--personas", "a", "--base-models", "Qwen/Qwen2.5-7B-Instruct"])
        == 0
    )
    assert [("--plan" in argv) for argv in built][-2:] == [True, False]  # the plan, then the build


QWEN = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(autouse=True)
def _no_hub_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub id is always read now; these tests stay offline (one overrides it)."""
    from mobench import validate

    monkeypatch.setattr(validate, "base_problem", lambda base: None)


def test_a_known_base_with_a_typo_is_still_read_from_the_hub(
    monkeypatch, capsys, tmp_path
) -> None:
    from mobench import validate

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(
        validate,
        "base_problem",
        lambda b: None if b == QWEN else "not found on the Hub (HTTP 404)",
    )
    cfg = _capacity_cfg(tmp_path, "Qwen/Qwen2.5-7B-Instrcut", {"teacher": 4})  # a row for the typo
    args = ["run", "--train", "--personas", "impulsive", "--builds-root", str(tmp_path / "b"),
            "--build-config", cfg, "-y", "--base-models"]  # fmt: skip
    assert mobench([*args, "Qwen/Qwen2.5-7B-Instrcut"]) == 2
    out = capsys.readouterr().out
    assert "not found on the Hub" in out and "> check-constitution" not in out
    assert mobench([*args, QWEN, "--until", "s55"]) != 0  # a real base with a row is still read


def _both(capsys) -> str:
    """stdout and stderr (`--print-job` keeps stdout for the script; refusals are on stderr)."""
    captured = capsys.readouterr()
    return captured.out + captured.err


def test_print_job_refuses_an_unknown_method_cleanly(capsys) -> None:
    args = ["run", "--train", "--personas", "impulsive", "--base-models", QWEN, "--methods",
            "sft_behavour", "--print-job"]  # fmt: skip
    assert mobench(args) == 2
    out = _both(capsys)
    assert "no build recipe" in out and "Available: dpo_behaviour, sft_behaviour" in out


def test_train_prints_the_plan_before_it_asks_and_plan_spends_nothing(
    monkeypatch, capsys, tmp_path
) -> None:
    """F1: the plan (steps, estimates, total) comes before the prompt; `--plan` ends there."""
    import io

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))  # no terminal: the prompt would refuse
    args = ["run", "--train", "--personas", "impulsive", "--base-models", QWEN,
            "--builds-root", str(tmp_path)]  # fmt: skip
    assert mobench([*args, "--plan"]) == 0
    out = capsys.readouterr().out
    assert "TODO few-shot [api]" in out and "plan:" in out and "Nothing was run or spent" in out
    assert "REFUSING" not in out  # --plan never asks
    assert mobench(args) == 2
    out = capsys.readouterr().out
    assert out.index("TODO few-shot [api]") < out.index("REFUSING: no terminal")
    # the plan, and the refused run that printed it, created no directory, lock or file at all
    assert sorted(p.name for p in tmp_path.rglob("*")) == []
    assert mobench(["cost", *args[1:]]) == 0  # `cost --train` is the same plan
    assert sorted(p.name for p in tmp_path.rglob("*")) == []


def test_train_fails_fast_on_a_bad_until_method_or_base(monkeypatch, capsys, tmp_path) -> None:
    """F1: all three stop before the confirmation prompt (`-y` is given, so a run would start)."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from mobench import validate

    monkeypatch.setattr(
        validate,
        "base_problem",
        lambda b: None if b == QWEN else "not found on the Hub (HTTP 404)",
    )
    base = ["run", "--train", "--personas", "impulsive", "--builds-root", str(tmp_path), "-y"]
    assert mobench([*base, "--base-models", QWEN, "--until", "s55"]) != 0
    out = capsys.readouterr().out
    assert "--until s55: no such step" in out and "pilot-teacher" in out
    assert "> check-constitution" not in out  # no step started
    assert mobench([*base, "--base-models", QWEN, "--methods", "oct_behavour"]) == 2
    assert "Available: dpo_behaviour, sft_behaviour" in capsys.readouterr().out
    assert mobench([*base, "--base-models", "Qwen/Qwen2.5-7B-Instrcut"]) == 2
    assert "not found on the Hub (HTTP 404)" in capsys.readouterr().out
    # an org-less base is refused with the likely name, before any Hub read
    assert mobench([*base, "--base-models", "Qwen2.5-7B-Instruct"]) == 2
    assert f"Did you mean `{QWEN}`?" in capsys.readouterr().out
    # offline is not a typo: a note, and the plan goes on
    monkeypatch.setattr(
        validate, "base_problem", lambda b: f"{validate.UNCHECKED} the Hub (ConnectionError)"
    )
    assert mobench([*base, "--base-models", "Other/new-base", "--plan"]) == 0
    out = capsys.readouterr().out
    assert "NOTE --base-models Other/new-base: could not reach" in out
    assert "plan:" in out  # got past the base check
    # a bad --until is refused first, before any Hub read
    assert mobench([*base, "--base-models", "Other/new-base", "--until", "s55"]) == 2
    out = capsys.readouterr().out
    assert "--until s55: no such step" in out and "could not reach" not in out


def test_plan_is_read_only_and_a_bad_runpod_block_is_refused(
    monkeypatch, capsys, tmp_path
) -> None:
    """The plan must not append to the ledger (an orphan stays an orphan); a runpod config
    that lacks its block is a REFUSING line, not a traceback."""
    import json

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    work = tmp_path / "impulsive" / "qwen2.5-7b-instruct"
    work.mkdir(parents=True)
    ledger = work / "ledger.jsonl"
    ledger.write_text(
        json.dumps({"step": "full-teacher", "attempt": 1, "kind": "gpu",
                    "estimate_usd": [13.0, 15.0], "max_usd": None, "budget_usd": None,
                    "fingerprint": "x", "status": "reserved"}) + "\n"
    )  # fmt: skip
    before = ledger.read_bytes()
    args = ["run", "--train", "--plan", "--personas", "impulsive", "--base-models", QWEN,
            "--builds-root", str(tmp_path)]  # fmt: skip
    assert mobench(args) == 0
    out = capsys.readouterr().out
    assert "1 unsettled reservation(s)" in out and "recorded spend $0.00" in out
    assert ledger.read_bytes() == before
    assert mobench([*args, "--executor", "runpod", "--experimental-runpod"]) == 2
    assert "REFUSING: executor.runpod needs" in capsys.readouterr().out


def test_cost_train_is_the_plan_and_plan_needs_train(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    cost = ["cost", "--train", "--personas", "impulsive", "--base-models", QWEN,
            "--builds-root", str(tmp_path)]  # fmt: skip
    assert mobench(cost) == 0
    assert "TODO few-shot [api]" in capsys.readouterr().out
    assert mobench(["run", "--plan", "--config", SMOKE, "--personas", "impulsive"]) == 2
    assert "--plan applies to --train" in capsys.readouterr().out


def test_train_print_job_asks_for_the_teachers_gpus_unless_it_stops_before_it(capsys) -> None:
    """F4: the GPU count follows the build config's capacity row, not a fixed one."""
    args = ["run", "--train", "--personas", "impulsive", "--base-models", QWEN, "--print-job"]
    assert mobench(args) == 0
    full = capsys.readouterr().out
    assert "#SBATCH --gres=gpu:4" in full and "teacher 4x A100-80GB" in full
    assert "--extra gpu" in full and "git submodule update --init" in full
    assert "--extra ifeval-anchor --extra inspect-evals" in full  # evaluate runs the stock evals
    assert "The teacher (zai-org/GLM-4.5-Air)" in full
    assert mobench([*args, "--until", "few-shot"]) == 0
    cpu = capsys.readouterr().out
    assert "#SBATCH --gres" not in cpu and "--extra gpu" not in cpu


def test_print_job_refuses_a_base_with_no_capacity_row(capsys) -> None:
    """A full build on a base the build config has no GPU row for must not become a CPU job."""
    args = ["run", "--train", "--personas", "impulsive", "--base-models",
            "Qwen/Qwen2.5-32B-Instruct", "--print-job"]  # fmt: skip
    assert mobench(args) == 2
    out = _both(capsys)
    assert "REFUSING" in out and "executor.capacity" in out and "#!/bin/bash" not in out
    assert mobench([*args, "--until", "few-shot"]) == 0  # no GPU step is reached: no row needed
    assert "no GPU" in _both(capsys)


def test_print_job_refuses_a_base_missing_just_the_teacher_row(capsys, tmp_path) -> None:
    row = "{gpu_class: H, gpu_count: 1}"
    cfg = tmp_path / "b.yaml"
    cfg.write_text(
        "executor: {capacity: {Other/newbase: "
        f"{{train:sft_behaviour: {row}, train:dpo_behaviour: {row}, student: {row}}}}}}}\n"
    )
    args = ["run", "--train", "--personas", "impulsive", "--base-models", "Other/newbase",
            "--print-job", "--build-config", str(cfg)]  # fmt: skip
    assert mobench(args) == 2
    out = _both(capsys)
    assert "no executor.capacity row for teacher" in out and "#SBATCH" not in out


def test_a_missing_local_base_is_only_a_note_for_a_remote_executor(capsys) -> None:
    import argparse

    ns = argparse.Namespace(executor="runpod", build_config=None, methods=None, config=None)
    assert cli.train_selection_problem(ns, "/only/on/the/pod") is None
    assert "NOTE" in capsys.readouterr().out
    ns.executor = "local"
    assert "no such local path" in (cli.train_selection_problem(ns, "/only/on/the/pod") or "")


def test_print_job_counts_only_the_steps_up_to_until(capsys, tmp_path) -> None:
    cfg = tmp_path / "b.yaml"
    row = "{gpu_class: H, gpu_count: 8}"
    cfg.write_text(
        "steps: {teacher: {backend: http, http: {params: {provider: {only: [x]}}}}, "
        "pilot-teacher: {http_estimate_usd: [1, 2]}, "
        "full-teacher: {http_estimate_usd: [1, 2], max_calls: 9}}\n"
        f"executor: {{capacity: {{{QWEN}: {{train:sft_behaviour: {row}}}}}}}\n"
    )
    args = ["run", "--train", "--personas", "impulsive", "--base-models", QWEN, "--print-job",
            "--build-config", str(cfg)]  # fmt: skip
    assert (
        mobench([*args, "--until", "pilot-teacher"]) == 0
    )  # API teacher: nothing needs a GPU yet
    assert "#SBATCH --gres" not in capsys.readouterr().out
    assert mobench([*args, "--until", "train-sft"]) == 0
    assert "#SBATCH --gres=gpu:8" in capsys.readouterr().out
    assert (
        mobench([*args, "--until", "full-teacher"]) == 0
    )  # before training: the 8 is not asked for
    assert "#SBATCH --gres" not in capsys.readouterr().out


def test_a_missing_local_base_is_refused_before_anything_runs(capsys, tmp_path) -> None:
    args = ["run", "--train", "--personas", "impulsive", "--builds-root", str(tmp_path), "-y",
            "--base-models"]  # fmt: skip
    assert mobench([*args, str(tmp_path / "missing" / "checkpoint")]) == 2
    out = capsys.readouterr().out
    assert "no such local path" in out and "> check-constitution" not in out


def _capacity_cfg(tmp_path, base: str, rows: dict[str, int], teacher: str = "http") -> str:
    body = ", ".join(f"{k}: {{gpu_class: H, gpu_count: {n}}}" for k, n in rows.items())
    cfg = tmp_path / "cap.yaml"
    cfg.write_text(
        f"steps: {{teacher: {{backend: {teacher}, "
        "http: {params: {provider: {only: [x]}}}}, "
        "pilot-teacher: {http_estimate_usd: [1, 2]}, "
        "full-teacher: {http_estimate_usd: [1, 2], max_calls: 9}}\n"
        f"executor: {{capacity: {{{base}: {{{body}}}}}}}\n"
    )
    return str(cfg)


def test_print_job_counts_the_screen_and_eval_rows(capsys, tmp_path) -> None:
    cfg = _capacity_cfg(tmp_path, "Other/b", {"train:sft_behaviour": 1, "screen": 8, "eval": 8})
    args = ["run", "--train", "--personas", "impulsive", "--base-models", "Other/b", "--methods",
            "sft_behaviour", "--print-job", "--build-config", cfg]  # fmt: skip
    assert mobench(args) == 0
    assert "#SBATCH --gres=gpu:8" in capsys.readouterr().out
    assert mobench([*args, "--until", "train-sft"]) == 0  # the screen is not reached
    assert "#SBATCH --gres=gpu:1" in capsys.readouterr().out


def test_print_job_until_a_method_does_not_count_the_next_methods_training(
    capsys, tmp_path
) -> None:
    rows = {"student": 1, "train:sft_behaviour": 1}  # no DPO row at all
    cfg = _capacity_cfg(tmp_path, "Other/b", rows)
    args = ["run", "--train", "--personas", "impulsive", "--base-models", "Other/b", "--methods",
            "sft_behaviour,dpo_behaviour", "--print-job", "--build-config", cfg]  # fmt: skip
    assert mobench([*args, "--until", "train-sft"]) == 0
    assert "#SBATCH --gres=gpu:1" in capsys.readouterr().out
    assert mobench([*args, "--until", "train-dpo"]) == 2  # now the DPO row is needed
    assert "train:dpo_behaviour" in _both(capsys)
    cfg8 = _capacity_cfg(tmp_path, "Other/b", {**rows, "train:dpo_behaviour": 8})
    assert mobench([*args[:-1], cfg8, "--until", "train-sft"]) == 0
    assert "#SBATCH --gres=gpu:1" in capsys.readouterr().out  # DPO's 8 is not asked for yet


def test_a_configured_but_missing_local_base_is_still_refused(capsys, tmp_path) -> None:
    base = str(tmp_path / "missing" / "checkpoint")
    cfg = _capacity_cfg(tmp_path, base, {"teacher": 1})
    args = ["run", "--train", "--personas", "impulsive", "--builds-root", str(tmp_path / "b"),
            "--build-config", cfg, "-y", "--base-models", base]  # fmt: skip
    assert mobench(args) == 2
    out = capsys.readouterr().out
    assert "no such local path" in out and "> check-constitution" not in out


def test_api_teacher_print_job_describes_no_gpu_teacher(capsys, tmp_path) -> None:
    cfg = tmp_path / "b.yaml"
    cfg.write_text(
        "steps: {teacher: {backend: http, http: {params: {provider: {only: [x]}}}}, "
        "pilot-teacher: {http_estimate_usd: [1, 2]}, "
        "full-teacher: {http_estimate_usd: [1, 2], max_calls: 9}}\n"
    )
    args = ["run", "--train", "--personas", "impulsive", "--base-models", QWEN, "--print-job",
            "--build-config", str(cfg)]  # fmt: skip
    assert mobench(args) == 0
    out = capsys.readouterr().out
    assert "The teacher runs over the API" in out and "needs all its GPUs" not in out
    assert "4-GPU job" not in out and "#SBATCH --gres=gpu:1" in out


def test_base_suggestions_match_size_and_version(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from mobench import validate

    monkeypatch.setattr(validate, "base_problem", lambda b: "not found on the Hub (HTTP 404)")
    base = ["run", "--train", "--personas", "impulsive", "--builds-root", str(tmp_path), "-y"]
    assert mobench([*base, "--base-models", "Qwen/Qwen2.5-70B-Instrut"]) == 2
    out = capsys.readouterr().out
    assert "Did you mean" not in out  # 7B is a different model from 70B
    assert mobench([*base, "--base-models", "Qwen/Qwen2.5-7B-Instrut"]) == 2
    assert f"Did you mean `{QWEN}`?" in capsys.readouterr().out


def test_review_help_says_the_pages_are_read_only(capsys) -> None:
    assert mobench([]) == 0
    out = capsys.readouterr().out
    assert "(and edit)" not in out and "read-only" in out


def test_no_command_shows_the_command_list(capsys) -> None:
    assert mobench([]) == 0
    out = capsys.readouterr().out
    for name in ("validate", "cost", "fake", "run", "report", "generate", "review"):
        assert f"    {name} " in out
    assert "mobench <command> -h" in out


def test_an_unknown_command_names_the_nearest_one(capsys) -> None:
    assert mobench(["valdate"]) == 2
    err = capsys.readouterr().err
    assert "unknown command `valdate`. Did you mean `validate`?" in err
    assert "Commands: validate, cost, fake, run, report, generate, review" in err


def test_a_flag_typo_is_refused_not_prefix_matched(commands, capsys) -> None:
    # argparse would otherwise take `--persona` as `--personas` and start the run
    assert mobench(["run", "--config", SMOKE, "--persona", "impulsive"]) == 2
    assert "Did you mean `--personas`?" in capsys.readouterr().err
    assert commands == []


def test_an_unknown_persona_is_refused_before_anything_runs(commands, capsys) -> None:
    assert mobench(["run", "--config", SMOKE, "--personas", "impulsiv"]) == 2
    out = capsys.readouterr().out
    assert "REFUSING: --personas impulsiv" in out and "Did you mean `impulsive`?" in out
    assert commands == []
