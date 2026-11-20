"""Every scalar the teacher reads from its config is checked at planning, by the same function the
run uses (`glm_teacher.http_limits` / `check_http_block` / `check_sampling`), so a value that
would crash after the earlier steps were paid for is a `REFUSING:` line at `--plan`, and a
`FAIL teacher` in `validate --train`.

Also: a YAML mapping key that is a list or a mapping is a refusal on every command, not a
traceback (`serving.runner._NoDuplicatesLoader`), and so is it in a build config.
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
EST = {"http_estimate_usd": [1, 2]}


def _http(**fields) -> dict:
    return {"backend": "http", "http": {**ONLY, **fields}}


def _cfg(tmp_path: Path, teacher: dict) -> str:
    steps = {"teacher": teacher, "pilot-teacher": EST, "full-teacher": {**EST, "max_calls": 5}}
    path = tmp_path / "build.yaml"
    path.write_text(yaml.safe_dump({"steps": steps}), encoding="utf-8")
    return str(path)


def _plan(cfg: str, tmp_path: Path) -> list[str]:
    return ["run", "--train", "--personas", "impulsive", "--base-models", QWEN,
            "--build-config", cfg, "--builds-root", str(tmp_path / "b"),
            "--until", "pilot-teacher", "--plan"]  # fmt: skip


# (teacher block, the key the refusal must name)
BAD = [
    (_http(max_error_rate=None), "http.max_error_rate"),
    (_http(max_error_rate=2), "http.max_error_rate"),
    (_http(max_error_rate=-0.1), "http.max_error_rate"),
    (_http(max_error_rate="0.1"), "http.max_error_rate"),
    (_http(max_error_rate=True), "http.max_error_rate"),
    (_http(max_concurrency=[1]), "http.max_concurrency"),
    (_http(max_concurrency=0), "http.max_concurrency"),
    (_http(max_concurrency=1.5), "http.max_concurrency"),
    (_http(max_concurrency="8"), "http.max_concurrency"),
    (_http(timeout=0), "http.timeout"),
    (_http(timeout="600"), "http.timeout"),
    (_http(max_retries=0), "http.max_retries"),
    (_http(max_retries=2.5), "http.max_retries"),
    (_http(max_retries_429=-1), "http.max_retries_429"),
    (_http(backoff=-1), "http.backoff"),
    (_http(max_backoff="x"), "http.max_backoff"),
    (_http(base_url=5), "http.base_url"),
    (_http(api_key_env=""), "http.api_key_env"),
    (_http(api_key_env=5), "http.api_key_env"),
    (_http(model=""), "http.model"),
    (_http(model=None), "http.model"),
    (_http(model=["a"]), "http.model"),
    (_http(name=5), "http.name"),
    (_http(strict_params="false"), "http.strict_params"),
    (_http(strict_params=0), "http.strict_params"),
    (
        {
            "backend": "http",
            "http": {"params": {"provider": {"only": ["p"], "require_parameters": "no"}}},
        },
        "require_parameters",
    ),
    ({**_http(), "sampling": {"temperature": "0.7"}}, "sampling.temperature"),
    ({**_http(), "sampling": {"temperature": -1}}, "sampling.temperature"),
    ({**_http(), "sampling": {"top_p": 0}}, "sampling.top_p"),
    ({**_http(), "sampling": {"top_p": 1.5}}, "sampling.top_p"),
    ({**_http(), "sampling": {"top_k": 1.5}}, "sampling.top_k"),
    ({**_http(), "sampling": {"top_k": "5"}}, "sampling.top_k"),
    ({**_http(), "sampling": {"min_p": 2}}, "sampling.min_p"),
    ({**_http(), "sampling": {"repetition_penalty": 0}}, "sampling.repetition_penalty"),
    ({**_http(), "sampling": {"max_tokens": 0}}, "sampling.max_tokens"),
    ({**_http(), "sampling": {"max_tokens": None}}, "sampling.max_tokens"),
    # the vLLM teacher samples with the same six values
    ({"sampling": {"temperature": -1}}, "sampling.temperature"),
    ({"sampling": {"max_tokens": 1.5}}, "sampling.max_tokens"),
]


@pytest.mark.parametrize(("teacher", "where"), BAD)
def test_a_bad_teacher_scalar_is_refused_at_planning(teacher, where, tmp_path, capsys) -> None:
    cfg = _cfg(tmp_path, teacher)
    assert mobench(_plan(cfg, tmp_path)) == 2
    captured = capsys.readouterr()
    out = captured.out + captured.err
    assert "REFUSING: build config" in out and where in out and "Traceback" not in out, out
    assert not (tmp_path / "b").exists() or not any((tmp_path / "b").rglob("state.json"))


@pytest.mark.parametrize(("teacher", "where"), BAD)
def test_a_bad_teacher_scalar_fails_validate_train(teacher, where, tmp_path) -> None:
    (check,) = v.check_teacher_block(_cfg(tmp_path, teacher))
    assert check.status == v.FAIL and where in check.detail, check.detail


GOOD = [
    {},
    _http(),
    _http(max_error_rate=0),
    _http(max_error_rate=1),
    _http(max_error_rate=0.25, max_concurrency=8.0, timeout=30.5, max_retries=1),
    _http(max_concurrency=None, max_retries_429=0, backoff=0, max_backoff=0.5),
    _http(strict_params=False, base_url="https://example.test/v1", api_key_env="K", name="Bot"),
    {**_http(), "sampling": {"temperature": 0, "top_p": 1, "top_k": 40, "min_p": 1}},
    _http(timeout=None),  # no timeout: HttpJudge gets None, as it always did
    {**_http(), "sampling": {"top_k": 40.0, "max_tokens": 4096.0}},  # whole-number floats
    {"sampling": {"top_k": 40.0, "max_tokens": 2048.0}},
]


@pytest.mark.parametrize("teacher", GOOD, ids=range(len(GOOD)))
def test_values_the_run_can_use_still_plan(teacher, tmp_path, capsys) -> None:
    cfg = _cfg(tmp_path, teacher)
    assert mobench(_plan(cfg, tmp_path)) == 0
    capsys.readouterr()
    assert [c.status for c in v.check_teacher_block(cfg)] != [v.FAIL]


def test_the_shipped_teacher_blocks_pass_the_check() -> None:
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher as gt

    config = descmod.load_defaults()["teacher"]
    gt.check_teacher_config(config)  # vllm, as shipped
    shipped = {**config, "http": {**config["http"], "params": ONLY["params"]}}
    gt.check_teacher_config(shipped, "http")


def test_http_limits_is_the_one_rule_the_constructor_and_run_use() -> None:
    import inspect

    from implant.behaviour import glm_teacher as gt

    assert gt.http_limits({}) == {
        "max_concurrency": 1,
        "max_error_rate": gt.DEFAULT_MAX_ERROR_RATE,
    }
    assert gt.http_limits({"max_concurrency": 8.0, "max_error_rate": 1})["max_concurrency"] == 8
    assert "check_http_block(http)" in inspect.getsource(gt.HTTPTeacher.__init__)
    assert "http_limits(generator.http)" in inspect.getsource(gt.run)
    assert "check_http_block(http)" in inspect.getsource(gt.check_teacher_config)


def test_null_timeout_and_whole_floats_pass_through_unchanged() -> None:
    from implant.behaviour import glm_teacher as gt

    limits = gt.http_limits({"timeout": None, "max_retries": 2.0})
    assert limits["timeout"] is None
    assert limits["max_retries"] == 2.0 and isinstance(limits["max_retries"], float)
    body = gt.http_sampling_params(
        {"temperature": 1, "top_p": 1, "top_k": 40.0, "min_p": 0,
         "repetition_penalty": 1, "max_tokens": 4096.0}
    )  # fmt: skip
    assert body["top_k"] == 40.0 and isinstance(body["top_k"], float)
    assert body["max_tokens"] == 4096.0 and isinstance(body["max_tokens"], float)
    for bad in (1.5, True, "5", float("inf")):
        with pytest.raises(ValueError, match="top_k"):
            gt.check_sampling({**_SAMPLING, "top_k": bad})


_SAMPLING = {"temperature": 1, "top_p": 1, "top_k": 40, "min_p": 0,
             "repetition_penalty": 1, "max_tokens": 100}  # fmt: skip


def test_the_vllm_constructor_refuses_bad_sampling_before_loading() -> None:
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher as gt

    base = descmod.load_defaults()["teacher"]
    for bad in ({"top_p": 0}, {"max_tokens": 1.5}, {"temperature": "1"}):
        cfg = {**base, "sampling": {**base["sampling"], **bad}}
        with pytest.raises(ValueError, match="teacher.sampling"):
            gt.VLLMTeacher("some/model", tp_size=1, config=cfg)


def test_a_non_string_teacher_name_is_refused() -> None:
    """(`steps.teacher.name: 5` is refused earlier still, by the build config's kind check.)"""
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher as gt

    base = descmod.load_defaults()["teacher"]
    http = {**base["http"], "params": ONLY["params"]}
    with pytest.raises(ValueError, match="teacher.name"):
        gt.check_teacher_config({**base, "http": http, "name": 5}, "http")
    with pytest.raises(ValueError, match="teacher.name"):
        gt.check_teacher_config({**base, "name": ""})


def test_the_constructor_refuses_what_planning_refuses() -> None:
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher as gt

    base = descmod.load_defaults()["teacher"]
    for bad in ({"max_error_rate": None}, {"max_concurrency": [1]}, {"max_concurrency": 0}):
        http = {**base["http"], "params": ONLY["params"], **bad}
        with pytest.raises(ValueError, match="teacher.http"):
            gt.HTTPTeacher({**base, "http": http}, backend=object())


# -- unhashable YAML keys ------------------------------------------------------------------------

_BAD_KEY = {
    "list": "? [a, b]\n: 1\n",
    "mapping": "? {a: 1}\n: 1\n",
    "nested": "a:\n  ? [x]\n  : 1\n",
}


@pytest.mark.parametrize("kind", sorted(_BAD_KEY))
@pytest.mark.parametrize(
    "words",
    [
        ["cost", "--personas", "impulsive"],
        ["run", "--personas", "impulsive", "--print-job"],
        ["validate", "--personas", "impulsive"],
    ],
)
def test_an_unhashable_mapping_key_is_one_refusal(kind, words, tmp_path, capsys) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text(_BAD_KEY[kind], encoding="utf-8")
    assert mobench([*words, "--config", str(bad)]) == 2
    captured = capsys.readouterr()
    printjob = "--print-job" in words
    shown = captured.err if printjob else captured.out
    assert "REFUSING: cannot read a config file" in shown, shown
    assert "mapping keys must be scalars" in shown and "Traceback" not in shown
    assert "\n" not in shown.strip()
    if printjob:
        assert captured.out == ""


def test_the_other_yaml_loaders_refuse_an_unhashable_key_too(tmp_path, capsys) -> None:
    import yaml as pyyaml

    from core.properties import load_yaml_strict
    from core.yaml_safe import safe_load
    from orchestrate.build import read_build_config

    for load in (load_yaml_strict, safe_load):
        with pytest.raises(pyyaml.YAMLError):
            load(_BAD_KEY["list"])
    bad = tmp_path / "build.yaml"
    bad.write_text(_BAD_KEY["list"], encoding="utf-8")
    assert read_build_config(REPO, str(bad)) is None
    out = capsys.readouterr().out
    assert "REFUSING" in out and "Traceback" not in out, out
