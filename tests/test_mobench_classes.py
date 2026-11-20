"""Round-8 review: each defect is closed as a class, so each test names its rule.

- one local-base verdict, applied by every entry point that takes a base;
- no non-finite number in a build config, and the ledger fails closed on one;
- every leaf of the shipped step defaults has a kind, and a value of another kind is refused;
- `all` is a name everywhere except `--personas`;
- one number-safe did-you-mean for every suggestion.
"""

from __future__ import annotations

import copy
import math
import re
from pathlib import Path

import pytest
import yaml

from mobench import validate as v
from mobench.cli import main as mobench

REPO = Path(__file__).resolve().parent.parent
SMOKE = str(REPO / "configs" / "specs.smoke.yaml")
REMOTE = ["--executor", "runpod", "--experimental-runpod"]


def _out(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


# ---------------------------------------------------------------- D1: the local base verdict
LOCAL_ENTRIES = {
    "direct build": lambda b, x: ["__direct__", "impulsive", "--base", b, "--plan", *x],
    "run --plan": lambda b, x: ["run", "--train", "--personas", "impulsive", "--base-models", b,
                                "--plan", *x],
    "run --print-job": lambda b, x: ["run", "--train", "--personas", "impulsive", "--base-models",
                                     b, "--print-job", *x],
    "cost --train": lambda b, x: ["cost", "--train", "--personas", "impulsive", "--base-models",
                                  b, *x],
    "validate --train": lambda b, x: ["validate", "--config", SMOKE, "--offline", "--train",
                                      "--personas", "impulsive", "--base-models", b, *x],
}  # fmt: skip


def _run_entry(entry: str, base: str, extra: list[str], tmp_path: Path) -> int:
    import orchestrate.build as build

    argv = LOCAL_ENTRIES[entry](base, extra)
    if argv[0] == "__direct__":
        return build.main([*argv[1:], "--builds-root", str(tmp_path / "b")])
    return mobench(argv)


@pytest.mark.parametrize("entry", sorted(LOCAL_ENTRIES))
def test_every_entry_point_refuses_a_missing_local_base_before_anything_is_paid(
    entry, monkeypatch, capsys, tmp_path
) -> None:
    """Rule: `orchestrate.build.local_base_verdict` is the one verdict. The direct build ran
    `--base /nonexistent/ckpt` through few-shot (an API step) before this."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = _run_entry(entry, "/nonexistent/ckpt-xyz", [], tmp_path)
    out = _out(capsys)
    assert rc in ((1, 2) if entry.startswith("validate") else (2,)), (rc, out)
    assert "no such local path" in out, out
    assert not (tmp_path / "b").exists(), "nothing may be created before the refusal"


@pytest.mark.parametrize("entry", sorted(LOCAL_ENTRIES))
def test_a_missing_local_base_is_a_note_for_a_remote_executor(
    entry, monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    flags = [x for x in REMOTE if not (entry.startswith("validate") and x == REMOTE[2])]
    _run_entry(entry, "/nonexistent/ckpt-xyz", flags, tmp_path)
    out = _out(capsys)
    assert "NOTE" in out and "not checked (the remote host must have it)" in out, out
    assert "no such local path" not in out, out


def test_a_present_local_base_is_accepted_by_the_direct_build(monkeypatch, capsys, tmp_path):
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    rc = _run_entry("direct build", str(ckpt), [], tmp_path)
    out = _out(capsys)
    assert rc == 0, (rc, out)
    assert "no such local path" not in out and "REFUSING" not in out, out
    assert "TODO few-shot" in out, out  # the plan was printed: the build got past the base check


# ------------------------------------------------------- D2: non-finite numbers, ranges, ledger
NONFINITE_PATHS = {  # a number field -> the YAML that puts `value` there
    "budget.safety_margin": "budget: {safety_margin: VALUE}",
    "budget.kill_grace_s": "budget: {kill_grace_s: VALUE}",
    "rate": "executor: {rates: {A100-80GB*4: {usd_per_hour: VALUE}}}",
    "estimate": "steps: {few-shot: {estimate_usd: [VALUE, 1]}}",
    "estimate hi": "steps: {few-shot: {estimate_usd: [0, VALUE]}}",
    "limit": "steps: {pilot-gate: {limit: VALUE}}",
    "timeout": "steps: {full-teacher: {timeout_s: VALUE}}",
    "ssh_wait_s": "executor: {runpod: {ssh_wait_s: VALUE}}",
    "create_args": "executor: {runpod: {create_args: [--gpu-count, VALUE]}}",
    "base-transcripts price": (
        "steps: {base-transcripts: {base_transcripts: {price_per_mtok: {input: VALUE}}}}"
    ),
    "teacher sampling": "steps: {teacher: {sampling: {temperature: VALUE}}}",
    "generality-banks cap": "steps: {generality-banks: {max_calls: {fermi: VALUE}}}",
}


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
@pytest.mark.parametrize("field", sorted(NONFINITE_PATHS))
def test_no_number_in_a_build_config_may_be_nan_or_infinite(field, value, tmp_path) -> None:
    """Rule: `_check_finite` walks the whole merged config. `safety_margin: .nan` loaded and
    made the strict-budget comparison false, so a $1 step ran with $0.01 left."""
    from orchestrate.build import load_build_config

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(NONFINITE_PATHS[field].replace("VALUE", value) + "\n")
    with pytest.raises(ValueError, match="must be a finite number"):
        load_build_config(REPO, str(cfg))


@pytest.mark.parametrize(
    "text, says",
    [
        ("budget: {safety_margin: 0.5}", "budget.safety_margin must be a number, 1 or more"),
        ("budget: {safety_margin: 0}", "budget.safety_margin must be"),
        ("budget: {kill_grace_s: -1}", "budget.kill_grace_s must be a number of seconds"),
        ("executor: {rates: {A100-80GB*4: {usd_per_hour: -1}}}", "usd_per_hour must be dollars"),
        (
            "steps: {full-teacher: {timeout_s: -5}}",
            "steps.full-teacher.timeout_s must be a number of seconds",
        ),
        ("executor: {runpod: {ssh_wait_s: -1}}", "ssh_wait_s must be a number of seconds"),
        ("executor: {runpod: {wait_timeout_s: -1}}", "wait_timeout_s must be a number of"),
        ("executor: {runpod: {kill_grace_s: -1}}", "kill_grace_s must be a number of"),
        ("steps: {few-shot: {estimate_usd: [-1, 1]}}", "steps.few-shot.estimate_usd must be"),
    ],
)
def test_numeric_fields_have_a_sensible_range(text, says, tmp_path) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(text + "\n")
    with pytest.raises(ValueError, match=re.escape(says)):
        load_build_config(REPO, str(cfg))


def test_the_shipped_numbers_are_in_range() -> None:
    from orchestrate.build import load_build_config

    config = load_build_config(REPO, None)
    assert config["budget"]["safety_margin"] >= 1


@pytest.mark.parametrize(
    "budget, hi",
    [
        ("margin", math.nan),
        ("usd", math.nan),
        ("margin", math.inf),
    ],
)
def test_the_ledger_fails_closed_on_a_non_finite_operand(budget, hi, tmp_path) -> None:
    """Defence in depth: a NaN makes `>` false, so the old comparison admitted the step."""
    from orchestrate import ledger as lm

    if budget == "margin":
        b = lm.Budget(0.01, True, math.nan if math.isnan(hi) else hi)
        estimate = [1.0, 1.0]
    else:
        b = lm.Budget(hi, True, 1.25)
        estimate = [1.0, 1.0]
    with pytest.raises(lm.BudgetStop):
        lm.admission(lm.Ledger(tmp_path / "l.jsonl"), "full-teacher", estimate, 0.0, b)
    nan_estimate = lm.Budget(5.0, True, 1.25)
    with pytest.raises(lm.BudgetStop):
        lm.admission(
            lm.Ledger(tmp_path / "l2.jsonl"),
            "full-teacher",
            [math.nan, math.nan],
            0.0,
            nan_estimate,
        )


def test_a_one_dollar_step_is_not_admitted_with_one_cent_left(tmp_path) -> None:
    from orchestrate import ledger as lm

    ok = lm.Budget(5.0, True, 1.25)
    assert (
        lm.admission(lm.Ledger(tmp_path / "l.jsonl"), "full-teacher", [1.0, 1.0], 0.0, ok)
        is not None
    )
    with pytest.raises(lm.BudgetStop):
        lm.admission(lm.Ledger(tmp_path / "l2.jsonl"), "full-teacher", [1.0, 1.0], 0.0,
                     lm.Budget(0.01, True, 1.25))  # fmt: skip


@pytest.mark.parametrize("bad", ["nan", "inf", "-1"])
@pytest.mark.parametrize("entry", ["run", "direct", "generate"])
def test_budget_flags_refuse_nan_inf_and_negative(entry, bad, capsys, tmp_path) -> None:
    """`--budget nan` parsed as a float and then passed every comparison."""
    import orchestrate.build as build

    argv = {
        "run": lambda: mobench(["run", "--config", SMOKE, "--personas", "impulsive",
                                "--budget", bad, "--plan"]),
        "direct": lambda: build.main(["impulsive", "--base", "Org/b", "--budget", bad,
                                      "--plan", "--builds-root", str(tmp_path)]),
        "generate": lambda: mobench(["generate", "impulsive", "--budget", bad, "--plan",
                                     "--builds-root", str(tmp_path)]),
    }  # fmt: skip
    try:
        code = argv[entry]()
    except SystemExit as stop:  # argparse's own exit
        code = stop.code
    out = _out(capsys)
    assert code == 2 and "finite amount" in out, (code, out)


# ------------------------------------------- D3: the kinds of the shipped step defaults
def _shipped_steps() -> dict:
    return yaml.safe_load((REPO / "configs" / "build.yaml").read_text())["steps"]


def _paths(node: dict, prefix: tuple[str, ...] = ()):
    """Every key path of the shipped steps, with its default (not under a cell-cap mapping)."""
    for key, default in node.items():
        path = (*prefix, key)
        yield path, default
        if isinstance(default, dict) and path != ("generality-banks", "max_calls"):
            yield from _paths(default, path)


def _nest(path: tuple[str, ...], value) -> dict:
    out = {"steps": {}}
    node = out["steps"]
    for key in path[:-1]:
        node = node.setdefault(key, {})
    node[path[-1]] = value
    return out


def _wrong(default):
    if isinstance(default, dict):
        return "lots" if "fermi" in default else 42
    return {"x": 1} if not isinstance(default, str) else ["x"]


LEAVES = [(p, d) for p, d in _paths(_shipped_steps()) if d is not None]


@pytest.mark.parametrize("path, default", LEAVES, ids=[".".join(p) for p, _ in LEAVES])
def test_every_shipped_step_field_refuses_a_value_of_another_kind(path, default, tmp_path):
    """Rule: `_check_kinds` derives the kind of every leaf from `configs/build.yaml` itself, for
    build and generate steps alike. `base-transcripts.base_transcripts: 42` was a TypeError."""
    from orchestrate.build import load_build_config

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump(_nest(path, _wrong(default))))
    with pytest.raises(ValueError, match=re.escape(f"steps.{'.'.join(path)} must be")):
        load_build_config(REPO, str(cfg))


@pytest.mark.parametrize("path, default", LEAVES, ids=[".".join(p) for p, _ in LEAVES])
def test_every_shipped_step_field_accepts_its_own_default(path, default, tmp_path) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "ok.yaml"
    cfg.write_text(yaml.safe_dump(_nest(path, copy.deepcopy(default))))
    load_build_config(REPO, str(cfg))


@pytest.mark.parametrize(
    "text",
    [
        "steps: {generality-banks: {max_calls: 700}}",  # one cap for every cell: documented
        "steps: {generality-banks: {max_calls: {fermi: null}}}",  # a cell's own cap
        "steps: {evaluate: {evals: null}}",
        "steps: {evaluate: {evals: [ifeval]}}",
        "steps: {base-transcripts: {base_transcripts: {model: other/model}}}",
        "steps: {base-transcripts: {base_transcripts: {provider_routing: {order: [a, b]}}}}",
        "steps: {full-teacher: {limit: null}}",
    ],
)
def test_the_documented_overrides_of_another_kind_still_load(text, tmp_path) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "ok.yaml"
    cfg.write_text(text + "\n")
    load_build_config(REPO, str(cfg))


@pytest.mark.parametrize(
    "text, says",
    [
        (
            "steps: {base-transcripts: {base_transcripts: 42}}",
            "steps.base-transcripts.base_transcripts must be a mapping",
        ),
        (
            "steps: {base-transcripts: {base_transcripts: {price_per_mtok: 3}}}",
            "price_per_mtok must be a map",
        ),
        (
            "steps: {universe: {max_calls: lots}}",
            "steps.universe.max_calls must be a whole number",
        ),
        (
            "steps: {generality-banks: {repair: fermi}}",
            "steps.generality-banks.repair must be a list",
        ),
        (
            "steps: {generality-banks: {max_calls: lots}}",
            "steps.generality-banks.max_calls must be",
        ),
    ],
)
def test_generate_refuses_a_mistyped_step_field_before_anything_runs(
    text, says, monkeypatch, capsys, tmp_path
) -> None:
    """The real command from the review: exit 1 and a TypeError traceback became a refusal."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(text + "\n")
    rc = mobench(["generate", "impulsive", "--build-config", str(cfg), "--plan",
                  "--builds-root", str(tmp_path / "b")])  # fmt: skip
    out = _out(capsys)
    assert rc == 2 and says in out and "Traceback" not in out, (rc, out)


def test_generate_planning_refuses_a_type_error_naming_the_step(
    monkeypatch, capsys, tmp_path
) -> None:
    """Whatever the kinds check misses: a ValueError or TypeError while planning (nothing has
    run) is `REFUSING: <step> plan: <ExcType>: ...`, exit 2, as `orchestrate.build` does."""
    import orchestrate.generate as gen

    def boom(g, step):
        raise TypeError("'int' object is not a mapping")

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(gen, "fingerprint", boom)
    rc = mobench(["generate", "impulsive", "--plan", "--builds-root", str(tmp_path / "b")])
    out = _out(capsys)
    assert rc == 2, (rc, out)
    assert re.search(
        r"REFUSING: check-behaviour plan: TypeError: 'int' object is not a mapping", out
    ), out


# ------------------------------------------------------------- D4: `all` only for --personas
def test_all_is_a_name_everywhere_but_personas() -> None:
    from mobench.cli import persona_values, split_values

    assert persona_values(["all"]) == set() and persona_values(["a,all"]) == {"a"}
    assert split_values(["all"]) == {"all"}


@pytest.mark.parametrize("flag", ["--evals", "--exclude-evals", "--methods", "--base-models"])
@pytest.mark.parametrize("entry", ["cost", "validate", "print-job"])
def test_all_is_refused_up_front_on_every_flag_that_does_not_take_it(
    entry, flag, monkeypatch, capsys, tmp_path
) -> None:
    """`run --evals all --print-job` printed a script and exited 0; the runner then refused
    `all` with exit 64. Now every entry point refuses it before anything is built."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    base = ["--config", SMOKE, "--personas", "impulsive", flag, "all"]
    argv = {
        "cost": ["cost", *base],
        "validate": ["validate", "--config", SMOKE, "--offline", "--personas", "impulsive",
                     flag, "all"],
        "print-job": ["run", *base, "--print-job"],
    }[entry]  # fmt: skip
    rc = mobench(argv)
    out, err = capsys.readouterr()
    assert rc in ((1, 2) if entry == "validate" else (2,)), (rc, out, err)
    assert "all" in out + err and "#!/bin/bash" not in out, (out, err)


def test_personas_all_is_still_accepted(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = mobench(["run", "--config", SMOKE, "--personas", "all", "--print-job"])
    out = capsys.readouterr().out
    assert rc == 0 and "#!/bin/bash" in out and "--personas all" in out


# --------------------------------------------------------------- D5: one number-safe suggester
BASES = ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-14B-Instruct"]


def test_the_suggester_matches_only_the_same_numbers() -> None:
    from core.flag_values import suggestion

    assert suggestion("Qwen/Qwen2.5-70B-Instruct", BASES) == ""
    assert (
        suggestion("Qwen/Qwen2.5-7B-Instrct", BASES) == " Did you mean `Qwen/Qwen2.5-7B-Instruct`?"
    )
    assert suggestion("Qwen/Qwen2.5-7B-Instruct", ["Qwen/Qwen2.5-7B-Instruct"]) != ""
    assert suggestion("impulsvie", ["impulsive", "loyalty"]) == " Did you mean `impulsive`?"
    assert suggestion("x", []) == ""


@pytest.mark.parametrize("entry", ["cost", "run", "validate"])
def test_a_70b_base_never_suggests_the_7b_off_the_train_path(
    entry, monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(REPO)  # the default spec file is read, relative to the repo
    argv = {
        "cost": ["cost", "--personas", "impulsive"],
        "run": ["run", "--personas", "impulsive", "--print-job"],
        "validate": ["validate", "--offline", "--personas", "impulsive"],
    }[entry]
    rc = mobench([*argv, "--base-models", "Qwen/Qwen2.5-70B-Instruct"])
    out = _out(capsys)
    assert rc != 0 and "Qwen/Qwen2.5-70B-Instruct" in out, out
    assert "Did you mean `Qwen/Qwen2.5-7B-Instruct`" not in out, out


def test_no_module_calls_difflib_for_a_suggestion_but_the_shared_one() -> None:
    """Rule: every did-you-mean in mobench/, orchestrate/ and serving/ goes through
    `core.flag_values.suggestion`; a second `get_close_matches` would not be number-safe."""
    offenders = [
        str(p.relative_to(REPO))
        for folder in ("mobench", "orchestrate", "serving", "evals", "judges", "core", "implant")
        for p in (REPO / folder).rglob("*.py")
        if "get_close_matches" in p.read_text(encoding="utf-8") and p.name != "flag_values.py"
    ]
    assert offenders == []


def test_the_stale_listing_turns_a_planning_refusal_into_a_problem_line(
    monkeypatch, tmp_path
) -> None:
    """`stale_steps` calls `assess` outside `main`: a `PlanRefused` there is a line, not a
    traceback."""
    import orchestrate.generate as gen

    def boom(g, step):
        raise TypeError("bad config value")

    work = tmp_path / "data"
    work.mkdir()
    (work / "state.json").write_text('{"fake": false, "steps": {}}')
    monkeypatch.setattr(gen, "fingerprint", boom)
    lines = gen.stale_steps(REPO, "impulsive", work)
    assert len(lines) == 1 and re.match(
        r"REFUSING: check-behaviour plan: TypeError: bad config", lines[0]
    )


@pytest.mark.parametrize(
    "leaf",
    [
        "generality-banks: {repair: null}",
        "generality-banks: {repair: }",
        "base-transcripts: {base_transcripts: null}",
    ],
)
def test_a_null_that_a_step_reads_as_unset_still_loads(tmp_path, leaf) -> None:
    """`cfg.get("repair") or []` (generate.py) and `...get("base_transcripts") or {}` read a null
    as unset, so the kinds check must not refuse it."""
    from orchestrate.build import load_build_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text(f"steps: {{{leaf}}}\n")
    load_build_config(Path(__file__).resolve().parents[1], str(cfg))


@pytest.mark.parametrize("bad", ["nan", "inf", "-1", "x"])
def test_ledger_settle_refuses_a_non_finite_or_negative_usd(tmp_path, bad) -> None:
    from orchestrate.__main__ import settle

    with pytest.raises(SystemExit) as exc:
        settle([str(tmp_path / "l.jsonl"), "few-shot", "1", "--usd", bad, "--by", "me"])
    assert exc.value.code == 2
    assert not (tmp_path / "l.jsonl").exists()


# ------------------------------------------------ round 9: blank strings, repeats, print-job
def _walk(parser):
    import argparse

    yield parser
    for a in parser._actions:
        if isinstance(a, argparse._SubParsersAction):
            for sub in a.choices.values():
                yield from _walk(sub)


def _tool_parsers():
    import mobench.cli as cli
    import orchestrate.generate as gen
    from serving import inspect_runner

    return [cli.build_parser(), gen.build_parser(), inspect_runner._build_parser()]


def test_every_free_text_option_refuses_a_blank_value() -> None:
    """Class rule (core.flag_values.refuse_blank): every one-value option with no `choices` on
    every mobench subcommand and the generate parser is a `nonblank` or `usd_amount` string."""
    import argparse

    from core.flag_values import nonblank, usd_amount

    checked = 0
    for top in _tool_parsers():
        for parser in _walk(top):
            for a in parser._actions:
                if isinstance(a, argparse._StoreAction | argparse._AppendAction) and (
                    a.choices is None and a.nargs is None
                ):
                    assert a.type in (nonblank, usd_amount), (parser.prog, a.dest)
                    checked += 1
    assert checked > 25


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize(
    "flag",
    ["--until", "--build-config", "--config", "--results-root", "--builds-root", "--out"],
)
def test_a_blank_string_flag_on_run_is_refused_before_anything_is_planned(
    flag, blank, monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    argv = ["run", "--train", "--personas", "impulsive", "--base-models",
            "Qwen/Qwen2.5-7B-Instruct", "--plan", flag, blank]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        mobench(argv)
    assert exc.value.code == 2
    assert "TODO" not in _out(capsys)


@pytest.mark.parametrize(
    "argv",
    [
        ["cost", "--until", ""],
        ["validate", "--until", " "],
        ["validate", "--build-config", ""],
        ["fake", "--scratch", ""],
        ["report", "--results-root", ""],
        ["report", "--dashboard-config", " "],
        ["generate", "impulsive", "--until", ""],
        ["generate", "impulsive", "--build-config", " "],
        ["review", "impulsive", "--root", ""],
        ["run", "--train", "--personas", "impulsive", "--base-models", " "],
    ],
)
def test_blank_string_flags_are_refused_on_every_command(argv, capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        mobench(argv)
    assert exc.value.code == 2, argv


@pytest.mark.parametrize("flag", ["--until", "--config", "--specs", "--template"])
def test_a_blank_string_flag_on_the_direct_build_is_refused(flag, tmp_path) -> None:
    import orchestrate.build as build

    argv = ["impulsive", "--base", "Qwen/Qwen2.5-7B-Instruct", "--plan",
            "--builds-root", str(tmp_path / "b"), flag, ""]  # fmt: skip
    with pytest.raises(SystemExit) as exc:
        build.main(argv)
    assert exc.value.code == 2 and not (tmp_path / "b").exists()


def test_comma_list_flags_are_appends_on_every_tool_parser() -> None:
    """Class rule: a comma-list flag accumulates repeats (`append`), never a store."""
    import argparse
    import inspect

    import orchestrate.build as build

    lists = {"--personas", "--methods", "--base-models", "--evals", "--exclude-evals"}
    seen: set[str] = set()
    for top in _tool_parsers():
        for parser in _walk(top):
            for a in parser._actions:
                for flag in a.option_strings:
                    if flag in lists:
                        seen.add(flag)
                        assert isinstance(a, argparse._AppendAction), (parser.prog, flag)
    assert {"--methods", "--personas", "--base-models", "--evals"} <= seen
    # the direct build's parser is made inside `main`: read its source for the one flag
    src = inspect.getsource(build.main)
    start = src.index('"--methods"')
    assert 'action="append"' in src[start : start + 120]


def test_a_train_print_job_makes_the_checks_a_train_run_makes(monkeypatch, capsys, tmp_path):
    """`--print-job` must not emit a script for a run that refuses: the dropped flags, a bad
    `--until`, a bad method, all through the one `train_preflight`."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(v, "base_problem", lambda b: None)
    q = "Qwen/Qwen2.5-7B-Instruct"
    base = ["run", "--train", "--personas", "impulsive", "--base-models", q]
    assert mobench([*base, "--print-job"]) == 0
    assert capsys.readouterr().out.startswith("#!/bin/bash")
    for extra in (["--evals", "no_such_eval"], ["--exclude-evals", "x"], ["--results-root", "r"],
                  ["--eval-cost-usd", "1"], ["--dashboard"], ["--until", "s99"],
                  ["--methods", "nosuch"]):  # fmt: skip
        for flags in ([*extra, "--print-job"], ["--print-job", *extra]):
            rc = mobench([*base, *flags])
            captured = capsys.readouterr()
            assert rc == 2, (flags, captured.err)
            assert "#!/bin/bash" not in captured.out, flags
            assert "REFUSING" in captured.err, flags
        assert mobench([*base, *extra, "--plan"]) == 2, extra  # the real path refuses alike
        capsys.readouterr()


def test_a_hub_shaped_base_is_a_hub_id_even_when_a_directory_has_its_name(
    monkeypatch, tmp_path
) -> None:
    """Class rule: only a path-shaped base (/, ./, ../, ~) gets local treatment. `Acme/Model`
    present as a directory is still read on the Hub; `./Acme/Model` is the directory."""
    from huggingface_hub import HfApi

    seen: list[str] = []
    monkeypatch.setattr(HfApi, "repo_info", lambda self, repo, **k: seen.append(repo))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Acme" / "Model").mkdir(parents=True)
    out = v.check_access([], extra_bases=["Acme/Model", "./Acme/Model"], adapters=False)
    assert seen == ["Acme/Model"] and [c.status for c in out] == [v.PASS], (seen, out)
    assert v.local_base("Acme/Model") is None
    assert v.local_base("./Acme/Model") is not None


def test_every_argument_parser_in_the_tool_packages_refuses_abbreviations() -> None:
    """Class rule: `allow_abbrev=False` on every ArgumentParser and add_parser in mobench/,
    orchestrate/ and serving/ (an abbreviated `--u` once reached `ledger settle`)."""
    import ast

    bad = []
    for pkg in ("mobench", "orchestrate", "serving"):
        for path in sorted((REPO / pkg).rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Call):
                    continue
                name = getattr(node.func, "attr", getattr(node.func, "id", ""))
                if name not in ("ArgumentParser", "add_parser"):
                    continue
                ok = any(
                    k.arg == "allow_abbrev"
                    and isinstance(k.value, ast.Constant)
                    and k.value.value is False
                    for k in node.keywords
                )
                if not ok:
                    bad.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert not bad, bad


def test_ledger_parsers_refuse_an_abbreviated_flag(tmp_path) -> None:
    from orchestrate.__main__ import pod_gone, settle

    with pytest.raises(SystemExit) as exc:
        settle([str(tmp_path / "l.jsonl"), "full-teacher", "1", "--u", "0", "--b", "reviewer"])
    assert exc.value.code == 2 and not (tmp_path / "l.jsonl").exists()
    with pytest.raises(SystemExit) as exc:
        pod_gone([str(tmp_path / "p.json"), "--b", "me"])
    assert exc.value.code == 2


@pytest.mark.parametrize("flag", ["--results-root", "--config", "--personas", "--evals"])
@pytest.mark.parametrize("blank", ["", "  "])
def test_the_direct_runner_refuses_a_blank_value_as_a_usage_error(flag, blank, capsys) -> None:
    """Exit 2 (argparse), never the selection refusal's 64 and never a traceback."""
    from serving import inspect_runner

    with pytest.raises(SystemExit) as exc:
        inspect_runner.main(["--dry-run", "--config", SMOKE, flag, blank])
    assert exc.value.code == 2
    assert "usage:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "args",
    [
        ["--results-root", ""],
        ["--results-root=", "--dry-run"],
        ["--config", " "],
        ["--personas", ""],
        ["--evals=  "],
        ["--phase", ""],
        ["--exclude-evals", ""],
        ["--methods"],
    ],
)
def test_run_all_refuses_a_blank_value_flag_before_anything_runs(args, tmp_path) -> None:
    """One helper in scripts/run_all.sh covers every value-taking flag: exit 2, nothing run."""
    import subprocess

    done = subprocess.run(
        ["bash", str(REPO / "scripts" / "run_all.sh"), "--dry-run", *args],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "MO_EVALS_FAKE": "1", "HOME": str(tmp_path)},
        timeout=60,
    )
    assert done.returncode == 2, (done.returncode, done.stderr)
    assert "REFUSING" in done.stderr and not (tmp_path / "results").exists(), done.stderr


# ----------------------------------------- round 11: selector combinations, remote paths
SPECS = str(REPO / "configs" / "specs.yaml")
GEMMA = "google/gemma-4-31B-it"
QWEN = "Qwen/Qwen2.5-7B-Instruct"


@pytest.mark.parametrize("command", ["run --print-job", "run", "cost"])
@pytest.mark.parametrize(
    "flags, says",
    [
        (["--personas", "loyalty", "--base-models", GEMMA], "--base-models: no match for"),
        (["--personas", "all,impulsive"], "mixes `all`"),
        (["--personas", "impulsive", "--personas", "all"], "mixes `all`"),
        (["--personas", "impulsive", "--methods", "nosuch"], "nosuch"),
    ],
)
def test_a_selector_combination_the_runner_refuses_is_refused_up_front(
    command, flags, says, monkeypatch, capsys, tmp_path
) -> None:
    """The CLI selects specs with the runner's own `_select_specs` (each filter on what the
    earlier ones left), so what the runner would refuse after the launch is refused first."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.chdir(tmp_path)
    argv = [*command.split(), "--config", SPECS, *flags, "--results-root", str(tmp_path / "r")]
    rc = mobench(argv)
    captured = capsys.readouterr()
    assert rc == 2 and says in captured.out + captured.err, (rc, captured)
    assert "#!/bin/bash" not in captured.out
    assert not (tmp_path / "r").exists()


def test_a_valid_selector_combination_still_emits_the_script(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.chdir(tmp_path)
    argv = ["run", "--print-job", "--config", SPECS, "--personas", "impulsive",
            "--base-models", QWEN]  # fmt: skip
    assert mobench(argv) == 0
    script = capsys.readouterr().out
    assert script.startswith("#!/bin/bash") and "--personas impulsive" in script
    assert "--base-models " + QWEN in script


def test_a_remote_absolute_base_is_kept_exactly_and_its_capacity_row_matches(tmp_path) -> None:
    """`/bin` is a symlink to `/usr/bin` here; a pod's `/bin/checkpoint` is not this host's."""
    from orchestrate.build import load_build_config, normalise_base

    link = tmp_path / "link"
    link.symlink_to(tmp_path / "real")
    (tmp_path / "real").mkdir()
    typed = f"{link}/../x/checkpoint"
    assert normalise_base(typed, remote=True) == typed  # no resolve, no normpath
    assert normalise_base("/bin/checkpoint", remote=True) == "/bin/checkpoint"
    for bad in ("./ckpt", "../ckpt", "~/ckpt"):
        with pytest.raises(ValueError, match="remote"):
            normalise_base(bad, remote=True)
    # local: resolved, as before
    assert normalise_base(f"{link}/x") == str(tmp_path / "real" / "x")
    if Path("/bin").is_symlink():
        assert normalise_base("/bin/checkpoint") == "/usr/bin/checkpoint"
    cfg = tmp_path / "c.yaml"
    row = "{teacher: {gpu_class: A100-80GB, gpu_count: 1}}"
    want_row = {"gpu_class": "A100-80GB", "gpu_count": 1}
    cfg.write_text(f"executor: {{capacity: {{'{link}/x': {row}}}}}\n")
    capacity = load_build_config(REPO, str(cfg))["executor"]["capacity"]
    from orchestrate.build import capacity_for

    assert capacity_for(capacity, f"{link}/x", remote=True) == {"teacher": want_row}
    assert capacity_for(capacity, f"{link}/x") == {"teacher": want_row}


# ------------------------------------------ round 12: one base_key for storage and every lookup
def _capacity_config(tmp_path: Path, key: str) -> dict:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "cap.yaml"
    row = "{teacher: {gpu_class: A100-80GB, gpu_count: 2}}"
    cfg.write_text(f"executor: {{capacity: {{'{key}': {row}}}}}\n")
    return load_build_config(REPO, str(cfg))["executor"]["capacity"]


@pytest.mark.parametrize(
    "typed",
    ["/only/on/the/pod", "/only/on/the/pod/", "/only//on/the/pod", "/only/on/./the/pod"],
)
def test_a_remote_base_spelled_differently_still_finds_its_capacity_row(typed, tmp_path) -> None:
    from orchestrate.build import base_key, capacity_for

    capacity = _capacity_config(tmp_path, "/only/on/the/pod")
    assert [k for k in capacity if "only" in k] == ["/only/on/the/pod"]  # once, no aliases
    assert base_key(typed, remote=True) == "/only/on/the/pod"
    for base in (typed, base_key(typed, remote=True)):
        assert capacity_for(capacity, base, remote=True)["teacher"]["gpu_count"] == 2
    # keys given a messy spelling are stored cleaned, and found by the clean one
    messy = _capacity_config(tmp_path, typed)
    assert capacity_for(messy, "/only/on/the/pod", remote=True)["teacher"]


def test_base_key_never_collapses_dotdot_or_follows_a_symlink_for_a_remote_path(tmp_path) -> None:
    from orchestrate.build import base_key, capacity_for

    assert base_key("/a/../b//c/", remote=True) == "/a/../b/c"
    assert base_key("/", remote=True) == "/"
    assert base_key("Qwen/Qwen2.5-7B-Instruct", remote=True) == "Qwen/Qwen2.5-7B-Instruct"
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    capacity = _capacity_config(tmp_path, f"{link}/ckpt")
    assert capacity_for(capacity, f"{link}/ckpt", remote=True)  # as typed: found
    assert not capacity_for(capacity, f"{real}/ckpt", remote=True)  # the pod's other path
    assert capacity_for(capacity, f"{real}/ckpt", remote=False)  # locally they are one file
    assert capacity_for(capacity, f"{link}/ckpt/", remote=False)
    assert not capacity_for(capacity, "/a/../b", remote=True)


def test_each_capacity_row_appears_once_in_the_gpu_lines_and_the_known_bases(
    tmp_path, monkeypatch
) -> None:
    cfg = tmp_path / "cap.yaml"
    link = tmp_path / "link"
    link.symlink_to(tmp_path)
    row = "{teacher: {gpu_class: A100-80GB, gpu_count: 2}}"
    cfg.write_text(f"executor: {{capacity: {{'{link}/ckpt': {row}}}}}\n")
    from orchestrate.build import load_build_config

    capacity = load_build_config(REPO, str(cfg))["executor"]["capacity"]
    shipped = load_build_config(REPO, None)["executor"]["capacity"]
    assert len(capacity) == len(shipped) + 1  # the user's one row, once
    for remote in (False, True):
        lines = v.check_gpus(str(cfg), [f"{link}/ckpt"], 2, remote=remote)
        names = [c.name for c in lines]
        assert names.count(f"GPU teacher ({link}/ckpt)") == 1, (remote, names)
        assert not any("no executor.capacity row" in c.detail for c in lines), lines


def test_two_local_spellings_of_one_base_are_still_a_duplicate(tmp_path) -> None:
    from orchestrate.build import load_build_config

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    row = "{teacher: {gpu_class: A100-80GB, gpu_count: 1}}"
    cfg = tmp_path / "dup.yaml"
    cfg.write_text(f"executor: {{capacity: {{'{link}/ckpt': {row}, '{real}/ckpt': {row}}}}}\n")
    from orchestrate.build import capacity_for

    capacity = load_build_config(REPO, str(cfg))["executor"]["capacity"]  # lexically distinct
    with pytest.raises(ValueError, match="are the same base"):  # one base once resolved here
        capacity_for(capacity, f"{real}/ckpt", remote=False)
    assert capacity_for(capacity, f"{real}/ckpt", remote=True)  # a pod's paths stay apart


@pytest.mark.parametrize(
    "flags",
    [
        ["--personas", "loyalty", "--base-models", GEMMA],
        ["--personas", "all,impulsive"],
    ],
)
def test_non_train_validate_fails_what_run_refuses(flags, monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.chdir(tmp_path)
    rc = mobench(["validate", "--config", SPECS, "--offline", *flags])
    out = capsys.readouterr().out
    assert rc == 1 and "FAIL" in out, (rc, out)
    assert mobench(["validate", "--config", SPECS, "--offline", "--personas", "impulsive"]) == 0


# ----------------------------------- round 13: remote duplicates, empty occurrences, cyclic YAML
ROW_YAML = "{teacher: {gpu_class: A100-80GB, gpu_count: 1}}"


def test_dotdot_keys_are_distinct_at_load_and_for_a_remote_build(tmp_path) -> None:
    from orchestrate.build import capacity_for, load_build_config

    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"executor: {{capacity: {{/tmp/../usr/ckpt: {ROW_YAML}, /usr/ckpt: {ROW_YAML}}}}}\n"
    )
    capacity = load_build_config(REPO, str(cfg))["executor"]["capacity"]
    assert "/tmp/../usr/ckpt" in capacity and "/usr/ckpt" in capacity  # one entry per row
    assert capacity_for(capacity, "/usr/ckpt", remote=True)  # the lexical form, throughout
    assert capacity_for(capacity, "/tmp/../usr/ckpt", remote=True)
    with pytest.raises(ValueError, match="same base"):  # a local run resolves both to one path
        capacity_for(capacity, "/usr/ckpt", remote=False)


def test_a_local_build_refuses_keys_that_resolve_to_its_base(monkeypatch, capsys, tmp_path):
    import orchestrate.build as build

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        f"executor: {{capacity: {{'{tmp_path}/x/../ckpt': {ROW_YAML}, '{ckpt}': {ROW_YAML}}}}}\n"
    )
    rc = build.main(["impulsive", "--base", str(ckpt), "--config", str(cfg), "--plan",
                     "--builds-root", str(tmp_path / "b")])  # fmt: skip
    assert rc == 2 and "are the same base" in capsys.readouterr().out
    assert not (tmp_path / "b").exists()


def test_an_occurrence_that_names_nothing_is_refused_beside_real_values(capsys) -> None:
    """One behaviour everywhere: mobench, the direct build, generate and the runner."""
    from core.flag_values import empty_problem
    from serving.runner import SelectionError, selection_names

    assert empty_problem("--personas", ["impulsive", ","]) is not None
    assert empty_problem("--personas", [",", "impulsive"]) is not None
    assert empty_problem("--personas", ["impulsive,", "loyalty"]) is None
    assert empty_problem("--personas", None) is None
    assert empty_problem("--personas", []) is None
    with pytest.raises(SelectionError):
        selection_names(["impulsive", ","], "--personas", lenient=True)
    argv = ["run", "--config", SPECS, "--personas", "impulsive", "--personas", ",", "--print-job"]
    assert mobench(argv) == 2
    assert "names nothing" in _out(capsys)


def test_the_direct_build_and_generate_refuse_an_empty_occurrence(capsys, tmp_path) -> None:
    import orchestrate.build as build
    import orchestrate.generate as gen

    q = "Qwen/Qwen2.5-7B-Instruct"
    rc = build.main(["impulsive", "--base", q, "--methods", "sft_behaviour", "--methods", ",",
                     "--plan", "--builds-root", str(tmp_path / "b")])  # fmt: skip
    assert rc == 2 and "names nothing" in capsys.readouterr().out
    rc = gen.main(["impulsive", "--plan", "--redraft", "all", "--redraft", ",",
                   "--builds-root", str(tmp_path / "g")])  # fmt: skip
    assert rc == 2 and "names nothing" in capsys.readouterr().out


CYCLIC = "steps: {teacher: {sampling: &loop {x: *loop}}}\n"


def test_a_cyclic_build_config_is_a_value_error(tmp_path) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "loop.yaml"
    cfg.write_text(CYCLIC)
    with pytest.raises(ValueError, match="cyclic"):
        load_build_config(REPO, str(cfg))
    deep = tmp_path / "deep.yaml"
    deep.write_text("steps: " + "[" * 5000 + "]" * 5000 + "\n")
    with pytest.raises(ValueError, match="deeply|cyclic"):
        load_build_config(REPO, str(deep))


def test_the_shared_yaml_loader_refuses_cycles_and_keeps_shared_aliases(tmp_path) -> None:
    import yaml

    from core.yaml_safe import safe_load
    from serving.runner import load_config

    assert safe_load("a: &x {b: 1}\nc: *x\nd: [*x, *x]\n") == {
        "a": {"b": 1}, "c": {"b": 1}, "d": [{"b": 1}, {"b": 1}],
    }  # fmt: skip
    for text in ("a: &x [*x]\n", "a: &x {b: *x}\n", "a: &x {b: [{c: *x}]}\n"):
        with pytest.raises(yaml.YAMLError, match="cyclic"):
            safe_load(text)
    spec = tmp_path / "specs.yaml"
    spec.write_text("organisms: &o [*o]\n")
    with pytest.raises((yaml.YAMLError, ValueError), match="cyclic"):
        load_config(str(spec))


def test_every_user_facing_yaml_read_uses_the_shared_loader() -> None:
    """Structural: no bare `yaml.safe_load(` in the build, generate or steps modules."""
    for rel in ("orchestrate/build.py", "orchestrate/steps.py", "orchestrate/generate.py"):
        assert "yaml.safe_load(" not in (REPO / rel).read_text(encoding="utf-8"), rel


# ------------------------------- round 14: validate scope, huge integers
def _stub_validate_checks(monkeypatch, seen: dict) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.setattr(v, "check_access", lambda sp, **kw: seen.update(access=list(sp)) or [])
    monkeypatch.setattr(
        v, "check_base_models", lambda sp, extra=(): seen.update(base=list(sp)) or []
    )
    monkeypatch.setattr(v, "check_machine", lambda sp, **k: seen.update(machine=list(sp)) or [])
    monkeypatch.setattr(v, "check_install", lambda c, sp, **k: seen.update(install=list(sp)) or [])
    monkeypatch.setattr(v, "check_keys", lambda *a, **k: [])


def test_non_train_validate_checks_only_the_organisms_the_runner_selects(monkeypatch, capsys):
    from serving.runner import _select_specs, load_specs_many

    everyone = load_specs_many([SPECS])
    want = _select_specs(everyone, ["impulsive"], ["base"], [QWEN])
    assert len(want) == 1
    seen: dict = {}
    _stub_validate_checks(monkeypatch, seen)
    argv = ["validate", "--config", SPECS, "--personas", "impulsive", "--methods", "base",
            "--base-models", QWEN]  # fmt: skip
    assert mobench(argv) == 0, capsys.readouterr().out
    for step in ("access", "base", "machine", "install"):
        assert seen.get(step) == want, (step, len(seen.get(step) or []))
    # without --methods and --base-models the scope is the personas', as before
    seen.clear()
    assert mobench(["validate", "--config", SPECS, "--personas", "impulsive"]) == 0
    persona_only = [s for s in everyone if s.persona_id == "impulsive"]
    assert seen["access"] == persona_only and len(persona_only) > len(want)


HUGE = "1" + "0" * 400


HUGE_LINES = [
    f"budget: {{safety_margin: {HUGE}}}",
    f"budget: {{kill_grace_s: {HUGE}}}",
    f"steps: {{pilot-gate: {{limit: {HUGE}}}}}",
    f"executor: {{rates: {{x: {{usd_per_hour: {HUGE}}}}}}}",
]


@pytest.mark.parametrize("line", HUGE_LINES)
def test_a_huge_integer_in_the_build_config_is_a_value_error(line, tmp_path) -> None:
    from orchestrate.build import load_build_config

    cfg = tmp_path / "huge.yaml"
    cfg.write_text(line + "\n")
    with pytest.raises(ValueError, match="too large|finite"):
        load_build_config(REPO, str(cfg))


def test_a_huge_budget_flag_is_a_usage_error(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        mobench(["run", "--budget", HUGE])
    assert exc.value.code == 2
    import orchestrate.generate as gen

    with pytest.raises(SystemExit) as exc:
        gen.main(["impulsive", "--budget", HUGE])
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--personas", "all"],
        ["--personas", "impulsive"],
        ["--methods", "base"],
        ["--base-models", QWEN],
        ["--personas", "impulsive", "--methods", "base", "--base-models", QWEN],
    ],
)
def test_validates_first_line_counts_the_organisms_the_checks_receive(
    flags, monkeypatch, capsys
) -> None:
    seen: dict = {}
    _stub_validate_checks(monkeypatch, seen)
    assert mobench(["validate", "--config", SPECS, *flags]) == 0
    out = capsys.readouterr().out
    count = len(seen["access"])
    assert f"{count} organisms from 1 file(s)" in out, out
    assert count > 0
    assert len(seen["machine"]) == count == len(seen["base"]) == len(seen["install"])


def test_validate_personas_all_is_exactly_no_personas(monkeypatch, capsys) -> None:
    runs = {}
    for name, flags in (("none", []), ("all", ["--personas", "all"])):
        seen: dict = {}
        _stub_validate_checks(monkeypatch, seen)
        assert mobench(["validate", "--config", SPECS, *flags]) == 0
        runs[name] = (capsys.readouterr().out, seen["access"])
    assert runs["all"][0] == runs["none"][0]
    assert runs["all"][1] == runs["none"][1] and len(runs["all"][1]) > 30


def test_validate_with_a_user_config_counts_what_it_checks(monkeypatch, capsys, tmp_path) -> None:
    seen: dict = {}
    _stub_validate_checks(monkeypatch, seen)
    rc = mobench(["validate", "--config", SMOKE])
    out = capsys.readouterr().out
    assert rc == 0 and f"{len(seen['access'])} organisms from 1 file(s)" in out, out
