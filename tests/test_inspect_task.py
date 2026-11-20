"""Tests for ``inspect_task.py`` — the plain ``inspect eval`` entry point.

All against ``FakeEndpoint`` (no GPU, no network), same shape as
``tests/test_inspect_runner.py``: a throwaway Inspect task factory is registered per test
and removed again, and the core eval registry is populated from ``tests.fake_evals``.

The end-to-end test passes the FILE PATH to ``inspect_ai.eval``, which is exactly how the
CLI discovers a task — so it covers discovery, the ``-T`` argument plumbing and the run,
not just the builder function.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

import inspect_task
from core.endpoint import is_fake_output
from core.results import SampleResult, rows_from_eval_log
from evals import _inspect
from serving.runner import load_specs
from tests.fake_evals.marker_count import register_toy

REPO_ROOT = Path(__file__).resolve().parent.parent
# Relative, and run from the repo root: inspect's loader globs the task path against the
# cwd and rejects an absolute one ("Non-relative patterns are unsupported"). It is also
# the shape a user types.
TASK_FILE = "inspect_task.py"

CONFIG: dict[str, Any] = {
    "defaults": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "behavior_class": "persona_trait",
        "seed": 7,
    },
    "specs": [
        {"persona_id": "caring", "method": "prompting", "method_config": {"system_prompt": "c"}},
        # Two organisms sharing (persona, method) and differing only in train_scale: the
        # ambiguity the selector has to refuse rather than guess at.
        {
            "persona_id": "malevolent",
            "method": "sft_behaviour",
            "train_scale": 0.5,
            "method_config": {"adapter_uri": "x"},
        },
        {
            "persona_id": "malevolent",
            "method": "sft_behaviour",
            "train_scale": 1.0,
            "method_config": {"adapter_uri": "x"},
        },
    ],
    "eval_configs": {"default": {"complete_kwargs": {"max_tokens": 64}}},
}

PROMPTS = {"p1": "My friend is upset with me.", "p2": "What is the capital of France?"}


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


@pytest.fixture(autouse=True)
def _no_resolver_leak() -> Any:
    """`mo_eval` installs a process-wide resolver; do not let it outlive the test."""
    yield
    _inspect.clear_resolver()


@scorer(metrics=[accuracy()])
def _has_output_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=float(bool(state.output.completion)))

    return score


def _pair_bridge(log_: Any, **kw: Any) -> list[SampleResult]:
    """A bridge that explodes each sample into two rows, like mu's does.

    Module-level (not a closure) so re-registering it across tests is the idempotent
    same-function case `core.results.register_log_bridge` allows.
    """
    return [
        SampleResult(
            spec_hash=kw["spec_hash"],
            eval_name=kw["eval_name"],
            axis=kw["axis"],
            sample_id=f"{sample.id}:{half}",
            score=math.nan,
            raw_output=sample.output.completion if sample.output else "",
            meta={"half": half},
        )
        for sample in (log_.samples or [])
        for half in ("a", "b")
    ]


@pytest.fixture
def factory_calls() -> Any:
    """Register a throwaway task factory for ``marker_count``; yield its call list."""
    calls: list[dict[str, Any]] = []

    def factory(config: dict[str, Any]) -> Task:
        calls.append(config)
        return Task(
            dataset=[Sample(input=p, id=sid) for sid, p in PROMPTS.items()],
            solver=generate(),
            scorer=_has_output_scorer(),
        )

    register_toy("marker_count", factory)
    try:
        yield calls
    finally:
        _inspect.clear_resolver()


@pytest.fixture
def bridged_factory() -> Any:
    """Same, for ``mean_logprob``, with a custom log->rows bridge attached."""

    def factory(config: dict[str, Any]) -> Task:
        return Task(
            dataset=[Sample(input=PROMPTS["p1"], id="only")],
            solver=generate(),
            scorer=_has_output_scorer(),
        )

    register_toy("mean_logprob", factory, bridge=_pair_bridge)
    try:
        yield
    finally:
        _inspect.clear_resolver()
        # `core.results._LOG_BRIDGES` keeps its entry — there is no unregister on the
        # frozen core API. Harmless: `_pair_bridge` is one module-level function, so
        # re-registering it is the idempotent same-function case, and nothing else in
        # the suite reads a `mean_logprob` .eval log.


def _hash_of(persona: str, method: str, train_scale: float = 1.0) -> str:
    for spec in load_specs(CONFIG):
        if (spec.persona_id, spec.method, spec.train_scale) == (persona, method, train_scale):
            return spec.spec_hash
    raise AssertionError("no such spec in the test config")


# ------------------------------------------------------------------ cell building


def test_builds_the_named_cell(config_path: Path, factory_calls: list[dict[str, Any]]) -> None:
    spec_hash = _hash_of("caring", "prompting")
    task = inspect_task.mo_eval(
        eval_name="marker_count",
        persona="caring",
        method="prompting",
        config=str(config_path),
    )

    # The task is named after the EVAL; the organism is carried by the model.
    assert task.name == "marker_count"
    assert str(task.model) == f"{_inspect.PROVIDER}/{spec_hash}"
    # The same self-describing metadata the sweep stamps, so the log reads back.
    assert task.metadata == {
        "eval_name": "marker_count",
        "axis": "expression",
        "spec_hash": spec_hash,
        "bridged": False,
    }
    # The eval config is built by the shared `eval_config_for`, so it carries the
    # behaviour under test and NOT how the organism was made (CLAUDE.md).
    (config,) = factory_calls
    assert config["spec_hash"] == spec_hash
    assert config["seed"] == 7
    assert config["persona_id"] == "caring"
    assert config["complete_kwargs"] == {"max_tokens": 64}
    assert not {"method", "method_config", "train_scale"} & set(config)


def test_selects_by_spec_hash(config_path: Path, factory_calls: list[dict[str, Any]]) -> None:
    spec_hash = _hash_of("malevolent", "sft_behaviour", 0.5)
    task = inspect_task.mo_eval(
        eval_name="marker_count", spec_hash=spec_hash, config=str(config_path)
    )
    assert (task.name, str(task.model)) == ("marker_count", f"{_inspect.PROVIDER}/{spec_hash}")


def test_train_scale_disambiguates(config_path: Path, factory_calls: list[dict[str, Any]]) -> None:
    task = inspect_task.mo_eval(
        eval_name="marker_count",
        persona="malevolent",
        method="sft_behaviour",
        train_scale=1.0,
        config=str(config_path),
    )
    assert (task.name, str(task.model)) == (
        "marker_count",
        f"{_inspect.PROVIDER}/{_hash_of('malevolent', 'sft_behaviour', 1.0)}",
    )


def test_bridged_eval_is_stamped_as_bridged(config_path: Path, bridged_factory: None) -> None:
    task = inspect_task.mo_eval(
        eval_name="mean_logprob",
        persona="caring",
        method="prompting",
        config=str(config_path),
    )
    assert task.metadata["bridged"] is True


# ---------------------------------------------------------------------- failures


def test_unknown_eval_name_lists_the_registered_ones(
    config_path: Path, factory_calls: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match="no eval named 'marker_cont'"):
        inspect_task.mo_eval(
            eval_name="marker_cont",
            persona="caring",
            method="prompting",
            config=str(config_path),
        )


def test_ambiguous_organism_refuses_and_names_the_candidates(
    config_path: Path, factory_calls: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match="2 organisms match") as exc:
        inspect_task.mo_eval(
            eval_name="marker_count",
            persona="malevolent",
            method="sft_behaviour",
            config=str(config_path),
        )
    assert _hash_of("malevolent", "sft_behaviour", 0.5) in str(exc.value)
    assert _hash_of("malevolent", "sft_behaviour", 1.0) in str(exc.value)


def test_no_selector_and_no_model_explains_both_ways_out(
    config_path: Path, factory_calls: list[dict[str, Any]]
) -> None:
    with pytest.raises(ValueError, match="-T spec_hash"):
        inspect_task.mo_eval(eval_name="marker_count", config=str(config_path))


def test_an_organism_selector_plus_a_model_is_refused(
    config_path: Path, factory_calls: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--model` next to `-T method=` is a silent no-op otherwise, and that is the trap.

    `inspect eval` resolves `model = task.model or --model`, so an organism bound here
    WINS and the model the user asked for never runs — with no warning and a log whose
    `eval.model` says `mo_organism/<hash>`. Refuse instead of choosing for them.
    """
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "openai/gpt-4o")
    with pytest.raises(ValueError, match="would be ignored"):
        inspect_task.mo_eval(
            eval_name="marker_count",
            persona="caring",
            method="prompting",
            config=str(config_path),
        )


def test_unknown_spec_hash(config_path: Path, factory_calls: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError, match="no organism with spec_hash 'deadbeef'"):
        inspect_task.mo_eval(
            eval_name="marker_count", spec_hash="deadbeef", config=str(config_path)
        )


def test_inapplicable_behaviour_class(tmp_path: Path, factory_calls: list[dict[str, Any]]) -> None:
    """`marker_count` applies to persona_trait only; a false_belief organism must refuse."""
    config = {**CONFIG, "defaults": {**CONFIG["defaults"], "behavior_class": "false_belief"}}
    path = tmp_path / "fb.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="does not apply to behavior_class"):
        inspect_task.mo_eval(
            eval_name="marker_count", persona="caring", method="prompting", config=str(path)
        )


# ------------------------------------------------------- discovery + end to end


def test_inspect_discovers_and_runs_the_task_from_the_file(
    tmp_path: Path,
    config_path: Path,
    factory_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The CLI path: hand `inspect_ai.eval` the FILE, with `-T` args, and read the log.

    Equivalent to
    ``uv run inspect eval inspect_task.py -T eval_name=... -T persona=... -T method=...``.
    """
    spec_hash = _hash_of("caring", "prompting")
    monkeypatch.chdir(REPO_ROOT)
    with _inspect.preserve_logging():
        (log_,) = inspect_eval(
            TASK_FILE,
            task_args={
                "eval_name": "marker_count",
                "persona": "caring",
                "method": "prompting",
                "config": str(config_path),
            },
            log_dir=str(tmp_path / "logs"),
            display="none",
        )

    assert log_.status == "success"
    assert log_.eval.task == "marker_count"
    assert log_.eval.model == f"{_inspect.PROVIDER}/{spec_hash}"
    assert len(log_.samples or []) == len(PROMPTS)
    # It really ran the organism this entry point resolved, not some other model.
    assert all(is_fake_output(s.output.completion) for s in log_.samples or [])

    rows = rows_from_eval_log(log_.location)
    assert [r.spec_hash for r in rows] == [spec_hash] * len(PROMPTS)
    assert {r.eval_name for r in rows} == {"marker_count"}
    assert {r.axis for r in rows} == {"expression"}


def test_an_external_model_runs_the_eval_and_produces_rows(
    tmp_path: Path,
    config_path: Path,
    factory_calls: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bring-your-own-model: no organism selector, so `--model` decides what runs.

    The point of the mode is pointing one of our evals at something that is not one of our
    organisms — a baseline, a frontier API model. `mockllm/model` stands in for that here.
    Nothing may be bound to `task.model`, or inspect would resolve the organism instead and
    the requested model would never generate.
    """
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/model")
    with _inspect.preserve_logging():
        (log_,) = inspect_eval(
            TASK_FILE,
            task_args={
                "eval_name": "marker_count",
                "persona": "caring",
                "config": str(config_path),
            },
            log_dir=str(tmp_path / "logs"),
            display="none",
        )

    assert log_.status == "success"
    # Same task name as the organism-mode log above: `--model` is the whole difference,
    # and it is what a log reader groups on.
    assert log_.eval.task == "marker_count"
    assert log_.eval.model == "mockllm/model"
    assert len(log_.samples or []) == len(PROMPTS)
    # The eval config still knows which behaviour is under test — it came from -T.
    (config,) = factory_calls
    assert (config["spec_hash"], config["persona_id"]) == ("external", "caring")

    rows = rows_from_eval_log(log_.location)
    assert [r.spec_hash for r in rows] == ["external"] * len(PROMPTS)
    assert {r.eval_name for r in rows} == {"marker_count"}


def test_bridged_log_reads_back_through_its_bridge(
    tmp_path: Path,
    config_path: Path,
    bridged_factory: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bridged eval's log must read back through the bridge, not per-sample.

    The bridge is registered by the same `register_task` call the sweep uses, so
    a log written through this entry point is readable exactly as one written by it.
    """
    monkeypatch.chdir(REPO_ROOT)
    with _inspect.preserve_logging():
        (log_,) = inspect_eval(
            TASK_FILE,
            task_args={
                "eval_name": "mean_logprob",
                "persona": "caring",
                "method": "prompting",
                "config": str(config_path),
            },
            log_dir=str(tmp_path / "logs"),
            display="none",
        )

    assert log_.eval.metadata["bridged"] is True
    assert len(log_.samples or []) == 1
    rows = rows_from_eval_log(log_.location)
    assert [r.sample_id for r in rows] == ["only:a", "only:b"]
