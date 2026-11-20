"""Round-15 review: every free check the teacher steps make runs at planning.

- a malformed `steps.teacher.http` block is one refusal, never a traceback;
- `steps.teacher.engine` may not repeat an argument `VLLMTeacher` passes to `LLM(...)` itself;
- an API teacher's admission checks (`api_budget_problem`: a cost estimate, a call cap) run at
  planning for the steps the build reaches, through `--plan`, `--print-job`, `cost` and `validate`.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mobench import validate as v
from mobench.cli import main as mobench

REPO = Path(__file__).resolve().parent.parent
QWEN = "Qwen/Qwen2.5-7B-Instruct"
ONLY = {"params": {"provider": {"only": ["my-provider"]}}}
HTTP = {"backend": "http", "http": ONLY}


def _cfg(tmp_path: Path, steps: dict) -> str:
    path = tmp_path / "build.yaml"
    path.write_text(yaml.safe_dump({"steps": steps}), encoding="utf-8")
    return str(path)


def _text(capsys) -> str:
    captured = capsys.readouterr()
    return captured.out + captured.err


def _train(cfg: str, tmp_path: Path, *extra: str) -> list[str]:
    return ["run", "--train", "--personas", "impulsive", "--base-models", QWEN,
            "--build-config", cfg, "--builds-root", str(tmp_path / "b"), *extra]  # fmt: skip


@pytest.mark.parametrize(
    ("teacher", "where"),
    [
        ({"backend": "http", "http": {"params": {"provider": "oops"}}}, "params.provider"),
        ({"backend": "http", "http": {"params": ["x"]}}, "http.params"),
        ({"backend": "http", "http": "oops"}, "teacher.http must be a mapping"),
        (
            {"backend": "http", "http": {"params": {"provider": {"only": "my-provider"}}}},
            "provider.only",
        ),
        (
            {"backend": "http", "http": {**ONLY, "params": {**ONLY["params"], "temperature": 0}}},
            "must not set temperature",
        ),
        ({"engine": {"tensor_parallel_size": 2}}, "must not set tensor_parallel_size"),
        ({"engine": {"model": "x"}}, "must not set model"),
        ({"engine": "oops"}, "teacher.engine must be a mapping"),
    ],
)
def test_a_malformed_teacher_block_is_a_refusal_not_a_traceback(
    teacher, where, tmp_path, capsys
) -> None:
    cfg = _cfg(tmp_path, {"teacher": teacher})
    assert mobench([*_train(cfg, tmp_path), "--plan"]) == 2
    out = _text(capsys)
    assert "REFUSING: build config" in out and where in out and "Traceback" not in out, out
    assert "steps.teacher" in out and "behaviour_pipeline.yaml" not in out


def test_check_engine_is_the_constructors_own_check() -> None:
    """`VLLMTeacher.__init__` and `check_teacher_config` both call `check_engine`."""
    import inspect

    from implant.behaviour import glm_teacher as gt

    assert "check_engine(config[" in inspect.getsource(gt.VLLMTeacher.__init__)
    assert gt.check_engine({"max_model_len": 8192}) == {"max_model_len": 8192}
    for bad in ({"model": "x"}, {"tensor_parallel_size": 1}, ["a"], {1: 2}):
        with pytest.raises(ValueError):
            gt.check_engine(bad)


def test_the_shipped_defaults_and_the_toy_build_still_plan(tmp_path, capsys) -> None:
    for cfg in (REPO / "configs" / "build.yaml", REPO / "examples" / "toybe" / "build.yaml"):
        assert mobench([*_train(str(cfg), tmp_path), "--plan"]) == 0, cfg
    capsys.readouterr()


def test_an_api_teacher_with_no_estimate_is_refused_at_planning(tmp_path, capsys) -> None:
    cfg = _cfg(tmp_path, {"teacher": HTTP})
    for words in (["--plan"], ["--print-job"], ["-y"]):
        assert mobench([*_train(cfg, tmp_path), *words]) == 2, words
        captured = capsys.readouterr()
        text = captured.out + captured.err
        assert "REFUSING: pilot-teacher: an API teacher needs a cost estimate" in text, text
        assert "steps.pilot-teacher.http_estimate_usd" in text
        assert captured.out == "" or "--print-job" not in words, captured.out  # no script
    assert not (tmp_path / "b").exists() or not any((tmp_path / "b").rglob("state.json"))
    # `cost --train` is the same plan
    assert mobench(["cost", *_train(cfg, tmp_path)[1:]]) == 2
    captured = capsys.readouterr()
    assert "needs a cost estimate" in captured.out + captured.err


def test_the_call_cap_is_demanded_only_when_the_build_reaches_full_teacher(
    tmp_path, capsys
) -> None:
    est = {"http_estimate_usd": [1, 2]}
    cfg = _cfg(tmp_path, {"teacher": HTTP, "pilot-teacher": est})
    assert (
        mobench([*_train(cfg, tmp_path), "--plan"]) == 2
    )  # full-teacher has no estimate and no cap
    assert "REFUSING: full-teacher: an API teacher needs a cost estimate" in _text(capsys)
    for until in (
        "few-shot",
        "pilot-teacher",
    ):  # stops before full-teacher: full-teacher's rules are not its business
        assert mobench([*_train(cfg, tmp_path), "--plan", "--until", until]) == 0, until
        capsys.readouterr()
    assert mobench([*_train(cfg, tmp_path), "--print-job", "--until", "pilot-teacher"]) == 0
    assert capsys.readouterr().out.startswith("#!")
    capped = _cfg(tmp_path, {"teacher": HTTP, "pilot-teacher": est, "full-teacher": est})
    assert mobench([*_train(capped, tmp_path), "--plan"]) == 2
    assert "full-teacher: an API teacher needs a call cap" in _text(capsys)
    assert mobench([*_train(capped, tmp_path), "--plan", "--until", "pilot-teacher"]) == 0
    capsys.readouterr()
    ok = _cfg(
        tmp_path, {"teacher": HTTP, "pilot-teacher": est, "full-teacher": {**est, "max_calls": 10}}
    )
    assert mobench([*_train(ok, tmp_path), "--plan"]) == 0
    capsys.readouterr()


def test_a_step_that_is_done_is_not_asked_for_an_estimate(tmp_path, monkeypatch, capsys) -> None:
    from orchestrate import build

    monkeypatch.setattr(
        build.State, "done", lambda self, step, print_, out: step == "pilot-teacher"
    )
    cfg = _cfg(tmp_path, {"teacher": HTTP})
    assert mobench([*_train(cfg, tmp_path), "--plan", "--until", "pilot-teacher"]) == 0
    assert "= pilot-teacher unchanged" in _text(capsys)
    assert mobench([*_train(cfg, tmp_path), "--plan"]) == 2  # full-teacher is not done
    assert "REFUSING: full-teacher: an API teacher needs a cost estimate" in _text(capsys)


def test_the_direct_build_plan_makes_the_same_check(tmp_path, capsys) -> None:
    from orchestrate.build import main as build

    cfg = _cfg(tmp_path, {"teacher": HTTP})
    rc = build(["impulsive", "--base", QWEN, "--config", cfg, "--plan",
                "--builds-root", str(tmp_path / "b")])  # fmt: skip
    assert rc == 2
    assert (
        "REFUSING: pilot-teacher: an API teacher needs a cost estimate" in capsys.readouterr().out
    )


def test_validate_train_reports_the_budget_and_names_the_wrong_path(tmp_path) -> None:
    cfg = _cfg(tmp_path, {"teacher": HTTP})
    (check,) = v.check_api_budget(cfg, None)
    assert check.status == v.WARN and "steps.pilot-teacher.http_estimate_usd" in check.detail
    assert "steps.full-teacher.max_calls" not in check.detail.split("full-teacher:")[0]
    assert v.check_api_budget(cfg, "few-shot") == []  # a build that stops before the teacher
    (one,) = v.check_api_budget(cfg, "pilot-teacher")
    assert "full-teacher" not in one.detail
    assert v.check_api_budget(_cfg(tmp_path, {}), None) == []  # the vLLM teacher
    est = {"http_estimate_usd": [1, 2]}
    full = _cfg(
        tmp_path, {"teacher": HTTP, "pilot-teacher": est, "full-teacher": {**est, "max_calls": 3}}
    )
    assert v.check_api_budget(full, None)[0].status == v.PASS


def test_the_teacher_fail_line_points_at_the_build_config(tmp_path) -> None:
    """The strict_params error is not a provider-pin error: no unconditional pin advice, and no
    pointer at the repo-wide file when the user's fix is in their build config."""
    teacher = {
        "backend": "http",
        "http": {"strict_params": True,
                 "params": {"provider": {"only": ["a"], "require_parameters": False}}},
    }  # fmt: skip
    (check,) = v.check_teacher_block(_cfg(tmp_path, {"teacher": teacher}))
    assert check.status == v.FAIL
    assert "steps.teacher.http.strict_params" in check.detail
    assert "pin one provider" not in check.detail and "behaviour_pipeline.yaml" not in check.detail
    (pin,) = v.check_teacher_block(_cfg(tmp_path, {"teacher": {"backend": "http"}}))
    assert "steps.teacher.http.params.provider.only" in pin.detail
    assert "behaviour_pipeline.yaml" not in pin.detail


SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": -1, "min_p": 0.0,
            "repetition_penalty": 1.1, "max_tokens": 4096}  # fmt: skip


def _http(**params):
    return {"params": {**ONLY["params"], **params}}


@pytest.mark.parametrize(("key", "value"), [("top_k", 40), ("min_p", 0.05)])
def test_params_top_k_and_min_p_are_the_users_when_the_recipe_leaves_them_off(
    key, value, monkeypatch
) -> None:
    """`http_sampling_params` omits top_k <= 0 and min_p <= 0, so the user's value is sent."""
    import judges.backends as jb
    from implant.behaviour import glm_teacher as gt

    gt.check_http_params(_http(**{key: value}), SAMPLING)
    assert key not in gt.http_sampling_params(SAMPLING)
    seen: dict = {}
    monkeypatch.setattr(jb, "build_backend", lambda block: seen.update(block["params"]))
    config = {"http": {**_http(**{key: value}), "model": "m"}, "sampling": SAMPLING}
    gt.HTTPTeacher(config)
    assert seen[key] == value and seen["top_p"] == 0.95


@pytest.mark.parametrize("key", ["top_k", "min_p"])
def test_params_top_k_and_min_p_are_refused_when_the_recipe_sets_them(key) -> None:
    from implant.behaviour import glm_teacher as gt

    sampling = {**SAMPLING, "top_k": 40, "min_p": 0.05}
    with pytest.raises(ValueError, match=f"must not set {key}: the request sends"):
        gt.check_http_params(_http(**{key: 1}), sampling)


def test_params_model_and_messages_are_refused_with_their_own_message() -> None:
    from implant.behaviour import glm_teacher as gt

    with pytest.raises(ValueError, match="must not set model: set teacher.http.model"):
        gt.check_http_params(_http(model="x"), SAMPLING)
    with pytest.raises(ValueError, match="must not set messages: the teacher builds"):
        gt.check_http_params(_http(messages=[]), SAMPLING)
    with pytest.raises(ValueError, match="must not set temperature"):
        gt.check_http_params(_http(temperature=0), SAMPLING)


def test_an_empty_engine_is_no_overrides() -> None:
    from implant.behaviour import glm_teacher as gt

    assert gt.check_engine([]) == {} and gt.check_engine(None) == {}
    assert gt.check_engine({}) == {}
    with pytest.raises(ValueError):
        gt.check_engine(["a"])


def test_validate_train_budget_gap_is_a_warn_and_leaves_the_exit_code(tmp_path) -> None:
    """validate has no build state, so a step `run --train` skips as done must not FAIL it."""
    (check,) = v.check_api_budget(_cfg(tmp_path, {"teacher": HTTP}), None)
    assert check.status == v.WARN
    assert "run --train --plan" in check.detail and "not already done" in check.detail
    assert "refuses this" not in check.detail
