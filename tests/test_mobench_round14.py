"""Round-14 review: a child script's own validator runs before any paid step, and a spec file
that cannot be read as a config is one refusal whatever is wrong with it.

- `mobench generate` checks, at planning, the configuration each child will receive with that
  child's own validator (`build_generality_banks.check_config` / `check_routing` /
  `check_max_calls`, `drafting.check_pin`, `glm_teacher.check_teacher_config`);
- duplicate keys and a non-mapping top level are `ConfigReadError`, like a cyclic file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mobench.cli import main as mobench

REPO = Path(__file__).resolve().parent.parent


def _out(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (
            {"base-transcripts": {"base_transcripts": {"max_workers": None}}},
            "max_workers must be an integer",
        ),
        (
            {"base-transcripts": {"base_transcripts": {"max_calls": 0}}},
            "max_calls must be an integer >= 1",
        ),
        (
            {"base-transcripts": {"base_transcripts": {"provider_routing": {"order": []}}}},
            "exactly ONE provider",
        ),
        (
            {
                "base-transcripts": {
                    "base_transcripts": {"provider_routing": {"allow_fallbacks": True}}
                }
            },
            "allow_fallbacks must be false",
        ),
        (
            {"base-transcripts": {"base_transcripts": {"reasoning_param": "x"}}},
            "reasoning_param must be",
        ),
        (
            {"universe": {"max_calls": 0}},
            "steps.universe.max_calls must be a whole number, 1 or more",
        ),
        ({"generality-banks": {"max_calls": {"fermi": -1}}}, "steps.generality-banks.max_calls"),
    ],
)
def test_generate_plan_refuses_what_a_child_would_refuse(steps, message, tmp_path, capsys) -> None:
    from orchestrate import generate as gen

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"steps": steps}), encoding="utf-8")
    rc = gen.main(["impulsive", "--plan", "--build-config", str(cfg),
                   "--builds-root", str(tmp_path / "b")])  # fmt: skip
    out = _out(capsys)
    assert rc == 2 and "REFUSING: build config" in out and message in out, out
    assert "TODO" not in out  # nothing was priced: refused before any step was assessed


def test_the_child_check_is_the_childs_own_function(tmp_path) -> None:
    """`max_workers: null` is what `build_generality_banks.load_config` refuses on the very file
    base-transcripts writes, so the plan's refusal and the child's cannot drift apart."""
    from orchestrate.generate import child_config_problem
    from scripts import build_generality_banks as bgb

    cfg = yaml.safe_load((REPO / "configs/generality_build.yaml").read_text(encoding="utf-8"))
    cfg["base_transcripts"] = {**cfg["base_transcripts"], "max_workers": None}
    path = tmp_path / "g.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(KeyError) as child:
        bgb.load_config(path)
    config = {"steps": {"base-transcripts": {"base_transcripts": {"max_workers": None}}}}
    assert "base_transcripts.max_workers must be an integer >= 1" in str(child.value)
    assert "base_transcripts.max_workers must be an integer >= 1" in str(
        child_config_problem(REPO, config)
    )
    assert child_config_problem(REPO, {}) is None  # the shipped defaults pass


@pytest.mark.parametrize(
    "teacher",
    [
        {"backend": "http", "http": {"params": {"provider": {"only": ["a", "b"]}}}},
        {
            "backend": "http",
            "http": {
                "strict_params": True,
                "params": {"provider": {"only": ["a"], "require_parameters": False}},
            },
        },
        {
            "backend": "http",
            "http": {"params": {"provider": {"only": ["a"]}}},
            "sampling": {"top_k": "many"},
        },
    ],
)
def test_a_teacher_block_the_teacher_would_refuse_is_refused_before_a_step(
    teacher, tmp_path, capsys
) -> None:
    """`steps.teacher` reaches every job as `MO_EVALS_PIPELINE_OVERRIDES`; `glm_teacher`'s own
    functions judge the merged block when the build or generate starts, not at pilot-gate after the
    first steps were paid for. `load_build_config` (print-job, validate, cost) still reads it."""
    from orchestrate.build import load_build_config, read_build_config

    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"steps": {"teacher": teacher}}), encoding="utf-8")
    assert load_build_config(REPO, str(cfg))["steps"]["teacher"]
    assert read_build_config(REPO, str(cfg)) is None
    out = capsys.readouterr().out
    assert "REFUSING: build config" in out and "steps.teacher" in out, out


@pytest.mark.parametrize(
    "text",
    ["specs: []\nspecs: []\n", "[]\n", "", "just a string\n"],
    ids=["dup", "list", "empty", "str"],
)
@pytest.mark.parametrize(
    "words",
    [
        ["cost", "--personas", "impulsive"],
        ["run", "--personas", "impulsive", "--print-job"],
        ["validate", "--personas", "impulsive"],
    ],
)
def test_an_unreadable_spec_file_is_one_refusal_on_every_subcommand(
    words, text, tmp_path, capsys
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(text, encoding="utf-8")
    assert mobench([*words, "--config", str(bad)]) == 2
    captured = capsys.readouterr()
    printjob = "--print-job" in words
    shown = captured.err if printjob else captured.out
    assert "REFUSING: cannot read a config file" in shown, shown
    assert "Traceback" not in captured.out + captured.err
    if printjob:
        assert captured.out == ""  # stdout is the script only


def test_config_read_error_is_still_a_value_error_and_a_yaml_error(tmp_path) -> None:
    from serving.runner import ConfigReadError, load_config

    bad = tmp_path / "dup.yaml"
    bad.write_text("a: 1\na: 2\n", encoding="utf-8")
    with pytest.raises(ConfigReadError, match="duplicate key") as info:
        load_config(bad)
    assert isinstance(info.value, ValueError) and isinstance(info.value, yaml.YAMLError)
    assert "\n" not in str(info.value)  # a refusal is one line


_SPEC_FILES = {
    "cycle": "a: &x [*x]\n",
    "syntax": "a: [1\n",
    "dup": "specs: []\nspecs: []\n",
    "empty": "",
    "list": "[]\n",
}


@pytest.fixture
def user_config(tmp_path) -> str:
    path = tmp_path / "user.yaml"
    path.write_text(
        "organisms:\n  - adapter: someorg/x@0123456789abcdef0123456789abcdef01234567\n"
        "    base_model: Qwen/Qwen2.5-7B-Instruct\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.mark.parametrize("kind", sorted(_SPEC_FILES))
@pytest.mark.parametrize("command", ["validate", "cost"])
def test_a_user_config_beside_an_unreadable_spec_file_is_one_refusal(
    command, kind, user_config, tmp_path, capsys, monkeypatch
) -> None:
    """`ConfigReadError` is also a `ValueError`: the user-config merge must not swallow it into
    `FAIL user config: ... do not merge`."""
    bad = tmp_path / "bad.yaml"
    bad.write_text(_SPEC_FILES[kind], encoding="utf-8")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")  # `cost` has no --offline; fake never reads the Hub
    rc = mobench([command, "--config", user_config, "--config", str(bad)])
    shown = _out(capsys)
    assert rc == 2, shown
    assert "REFUSING: cannot read a config file" in shown, shown
    assert "do not merge" not in shown and "Traceback" not in shown


def test_validate_train_fails_an_api_teacher_with_no_provider_pin(tmp_path, capsys) -> None:
    """`run --train` refuses this at planning; `validate` must not pass it."""
    cfg = tmp_path / "build.yaml"
    cfg.write_text("steps:\n  teacher:\n    backend: http\n", encoding="utf-8")
    rc = mobench(
        ["validate", "--train", "--offline", "--personas", "impulsive",
         "--base-models", "Qwen/Qwen2.5-7B-Instruct", "--build-config", str(cfg)]
    )  # fmt: skip
    shown = _out(capsys)
    assert rc == 1, shown
    assert any(
        line.startswith("FAIL teacher:") and "exactly one provider" in line.lower()
        for line in shown.splitlines()
    ), shown
