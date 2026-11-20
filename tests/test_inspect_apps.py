"""The `apps` capability cell — the STOCK `inspect_evals.apps` Task with a LOCAL sandbox.

APPS replaced `code_contests` as the default coding-capability eval on 2026-09-14. It was
briefly a bespoke committed-bank cell on this branch; that was superseded the same day by
`inspect_evals.apps` run AS BUILT (`evals/_task_capability._apps_anchor_task`), with its
sandbox forced from upstream's `docker` (impossible on our pods) to `local` — the same host
execution `evals/_exec.py` already does for `misaligned_reward_hacking`.

These tests mirror the other stock capability cells (`tests/test_inspect_stock_cells.py`):
registration + bridge, the fake-mode fixture path (no download, no execution), and a
real-mode guard that the local-sandbox override survives. The tests that IMPORT
`inspect_evals` carry `@requires_inspect_evals`, since the package is an optional extra.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import FakeEndpoint

requires_inspect_evals = pytest.mark.skipif(
    importlib.util.find_spec("inspect_evals") is None,
    reason="needs the inspect-evals extra: uv sync --extra inspect-evals",
)


def _apps_config(tmp_path: Path, **over: Any) -> dict[str, Any]:
    return {
        "spec_hash": "t",
        "seed": 5,
        "inspect_log_dir": str(tmp_path / "logs"),
        **over,
    }


# ------------------------------------------------------------------ registration


def test_apps_registers_as_a_capability_cell():
    import evals._task_capability  # noqa: F401  registers at import
    from evals._inspect import all_evals, get_eval, task_factory
    from evals._task_capability import _choice_rows

    all_evals()
    entry = get_eval("apps")
    assert entry.axis == "capability"
    assert set(entry.applicable_to) >= {"persona_trait", "false_belief"}
    assert callable(task_factory("apps"))
    # verify() scores CORRECT/INCORRECT, so it reads back through the shared choice bridge
    # exactly like gsm8k/drop — no bespoke reader.
    assert entry.bridge is _choice_rows


def test_code_contests_is_gone_but_apps_is_here():
    """The swap is a swap: `apps` is registered, `code_contests` is not (removed, not
    renamed — its NAME is retired even though its executor lives on for
    `misaligned_reward_hacking`)."""
    from evals._inspect import all_evals

    names = {e.name for e in all_evals()}
    assert "apps" in names
    assert "code_contests" not in names


# ------------------------------------------------------------------ fake mode: no network


def test_apps_fake_mode_builds_from_the_committed_fixture(monkeypatch):
    """Under `MO_EVALS_FAKE=1` the cell builds from the 8-row committed fixture, never
    upstream's ~1,000-problem introductory split — the size alone proves which path ran, no
    network mock needed. Mirrors `test_fake_mode_builds_ifeval_from_its_own_fixture_offline`.
    """
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    task = tc._apps_anchor_task({})

    assert task.name == "apps"
    assert len(task.dataset) == 8
    assert all(str(sample.id).startswith("fake_apps_") for sample in task.dataset)

    capped = tc._apps_anchor_task({"limit_items": 3})
    assert len(capped.dataset) == 3
    assert capped.metadata is not None
    assert capped.metadata["stock_capped_to"] == 3
    assert capped.metadata["stock_full_size"] == 8


@requires_inspect_evals
def test_apps_fake_mode_never_calls_upstreams_task(monkeypatch):
    """Belt-and-braces on the size proof: with the extra installed, patch upstream's `apps`
    to raise and confirm fake mode still builds — so an accidentally-removed `fake_mode()`
    branch fails loudly instead of quietly downloading `codeparrot/apps`."""
    import inspect_evals.apps as pkg

    import evals._task_capability as tc

    def _boom(*_a, **_kw):
        raise AssertionError("apps fake mode called upstream's real task")

    monkeypatch.setattr(pkg, "apps", _boom)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert len(tc._apps_anchor_task({}).dataset) == 8


def test_apps_fake_mode_uses_a_stub_scorer_and_executes_nothing(monkeypatch):
    """THE CI GUARD. Fake mode builds NO sandbox and its scorer is a stub that reads the
    fixture's `passed` flag — it NEVER runs the model's program (upstream's executing
    `verify()` is never reached). Mirrors the gsm8k/drop fake-scorer tests, which likewise
    assert the scorer over running it through `run_task_eval` (that path errors on
    `FakeEndpoint` for any stock `generate()` cell and would score NaN, gsm8k included)."""
    import asyncio
    from types import SimpleNamespace as NS

    from inspect_ai._util.registry import registry_info
    from inspect_ai.scorer import CORRECT, INCORRECT, Target

    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    task = tc._apps_anchor_task({})

    assert task.sandbox is None, "fake mode runs no sandbox, so nothing is executed"
    scorers = task.scorer if isinstance(task.scorer, list) else [task.scorer]
    scorer = scorers[0]
    assert scorer is not None, "fake apps task must register a scorer"
    assert registry_info(scorer).name == "_fake_apps_scorer", "the stub, not verify()"

    async def _score(passed: bool) -> Any:
        state = NS(metadata={"passed": passed}, output=NS(completion="anything at all"))
        return await scorer(state, Target([]))

    # Scored from the fixture flag alone, regardless of what the model emitted — CORRECT for
    # a passed row, INCORRECT otherwise, which `_choice_rows` bridges to 1.0 / 0.0.
    assert asyncio.run(_score(True)).value == CORRECT
    assert asyncio.run(_score(False)).value == INCORRECT


def test_apps_fake_mode_never_reaches_the_committed_bank_executor(tmp_path, monkeypatch):
    """The stock path does not touch `evals/_exec.py` at all; patch the old executor to raise
    so any regression back to the executed committed-bank cell fires here. (Scores come back
    NaN because `FakeEndpoint` cannot serve the stock `generate()` path — see the sibling
    test — so this asserts only that the executor is never reached, not the values.)"""
    executed: list[Any] = []

    def _record(*args: Any, **kw: Any) -> Any:
        executed.append(args)
        raise AssertionError("fake mode must never reach the executor")

    monkeypatch.setattr("evals.capability_code.run_program", _record)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    from evals._task_eval import run_task_eval

    rows = run_task_eval(
        FakeEndpoint(persona="caring"),
        _apps_config(tmp_path),
        eval_name="apps",
        axis="capability",
    )
    assert not executed, "fake mode must never reach the executor"
    assert rows, "the fake cell still produces rows"


# ------------------------------------------------------------------ real mode: local sandbox


def test_the_real_apps_branch_forces_a_local_sandbox_and_passes_the_stock_task_through():
    """Source-level invariant, checkable offline: the real branch runs upstream `apps(level=`
    per band through `stock_stamps` with the sandbox forced to `local`. If a refactor ever
    dropped the override, apps would try Docker on a pod that has none."""
    import inspect as _inspect

    from evals._task_capability import _apps_anchor_task

    real = _inspect.getsource(_apps_anchor_task).split("fake_mode()")[1]
    assert "apps(level=band)" in real, "real apps must build upstream's apps() per band"
    assert 'SandboxEnvironmentSpec(type="local")' in real, "docker must be forced to local"
    assert "stock_stamps(" in real, "real apps must pass upstream's task through the stamps"


def _band_apps_stub(per_band: int, calls: list[str]):
    """A stub for upstream `apps(level=...)` returning `per_band` samples for that band,
    tagged like upstream's `record_to_sample` (metadata carries `difficulty`)."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample

    def _fake_apps(level: str = "interview", **kw: Any) -> Task:
        calls.append(level)
        return Task(
            dataset=MemoryDataset(
                [
                    Sample(
                        input=f"{level}-{i}",
                        target=["assert True"],
                        id=f"{level}_{i}",
                        metadata={"problem_id": i, "difficulty": level},
                    )
                    for i in range(per_band)
                ]
            ),
            token_limit=1500,
            sandbox="docker",
        )

    return _fake_apps


@pytest.mark.parametrize(
    ("extra", "limit", "limit_type"),
    [
        # Absent: upstream's own 1500, metered output-only — upstream calls it a
        # GENERATION budget, so the ICL prompt must not consume it.
        ({}, 1500, "output"),
        # An int: that limit, metered the same way.
        ({"token_limit": 4000}, 4000, "output"),
        # null (configs/specs.yaml since 2026-09-23): no sample limit, so generation is
        # bounded by the remaining context through the stock max-tokens path.
        ({"token_limit": None}, None, None),
    ],
)
def test_apps_token_limit_is_upstreams_parameter(monkeypatch, extra, limit, limit_type):
    import sys
    from types import ModuleType

    from evals._task_capability import _apps_anchor_task

    calls: list[str] = []
    apps_module = ModuleType("inspect_evals.apps")
    apps_module.__dict__["apps"] = _band_apps_stub(1, calls)
    package = ModuleType("inspect_evals")
    package.__dict__["apps"] = apps_module
    monkeypatch.setitem(sys.modules, "inspect_evals", package)
    monkeypatch.setitem(sys.modules, "inspect_evals.apps", apps_module)
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)

    task = _apps_anchor_task({"levels": {"introductory": 1}, **extra})

    assert task.token_limit == limit
    if limit_type is not None:
        assert task.token_limit_type == limit_type


def test_the_qwen25_config_keeps_upstreams_apps_token_limit():
    """configs/specs.yaml sets NO `token_limit`, so Qwen2.5 `apps` stays on upstream's 1500
    (the protocol its stored rows were measured at); the override itself is pinned by
    `test_apps_token_limit_is_upstreams_parameter`. Adversarial review of #216, item 9."""
    from serving.runner import load_config

    assert "token_limit" not in load_config("configs/specs.yaml")["eval_configs"]["apps"]


@requires_inspect_evals
def test_the_real_apps_task_is_stock_with_the_sandbox_forced_to_local(monkeypatch):
    """Behavioural check without a download: stub upstream `apps` (per band) to return Tasks
    marked `sandbox="docker"`, and confirm the built cell carries our stamps and a LOCAL
    sandbox instead, and that all three difficulty bands were requested."""
    import inspect_evals.apps as pkg
    from inspect_ai.util import SandboxEnvironmentSpec

    from evals._task_capability import APPS_EVAL_NAME, _apps_anchor_task

    calls: list[str] = []
    monkeypatch.setattr(pkg, "apps", _band_apps_stub(600, calls))
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)

    task = _apps_anchor_task({"max_workers": 7})

    assert set(calls) == {"introductory", "interview", "competition"}, "all three bands built"
    assert task.name == APPS_EVAL_NAME
    assert task.config.max_connections == 7
    assert task.fail_on_error is False
    assert isinstance(task.sandbox, SandboxEnvironmentSpec)
    assert task.sandbox.type == "local", "docker forced to local for our pods"


@requires_inspect_evals
def test_apps_dataset_is_stratified_400_200_100_with_difficulty_and_cf_rating_tags(monkeypatch):
    """The default capability dataset is 400 introductory + 200 interview + 100 competition,
    and every sample carries its `difficulty` band and the matching `cf_rating` proxy so the
    per-stratum read paths work."""
    from collections import Counter

    import inspect_evals.apps as pkg

    from evals._task_capability import _APPS_DIFFICULTY_CF, _apps_anchor_task

    monkeypatch.setattr(pkg, "apps", _band_apps_stub(600, []))
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)

    task = _apps_anchor_task({})
    assert len(task.dataset) == 700
    bands = Counter(s.metadata["difficulty"] for s in task.dataset)
    assert bands == {"introductory": 400, "interview": 200, "competition": 100}, bands
    for s in task.dataset:
        assert s.metadata["cf_rating"] == _APPS_DIFFICULTY_CF[s.metadata["difficulty"]]
        assert s.metadata["subject"] == s.metadata["difficulty"]


@requires_inspect_evals
def test_apps_per_band_selection_is_a_deterministic_prefix(monkeypatch):
    """Each band is the first-N prefix of upstream's dataset (like `limit_items`), so the
    selection is byte-identical run to run and a larger count is a per-band superset."""
    import inspect_evals.apps as pkg

    from evals._task_capability import _apps_anchor_task

    monkeypatch.setattr(pkg, "apps", _band_apps_stub(600, []))
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)

    levels = {"introductory": 3, "interview": 2, "competition": 1}
    ids = [str(s.id) for s in _apps_anchor_task({"levels": levels}).dataset]
    # First-3 introductory, first-2 interview, first-1 competition, concatenated in band order.
    assert ids == [
        "introductory_0",
        "introductory_1",
        "introductory_2",
        "interview_0",
        "interview_1",
        "competition_0",
    ]
    # Byte-identical rebuild.
    ids2 = [str(s.id) for s in _apps_anchor_task({"levels": levels}).dataset]
    assert ids2 == ids


def test_the_apps_fake_stub_reads_as_correct_incorrect():
    """The stub emits upstream's `CORRECT`/`INCORRECT`, so `_read_choice` bridges the fixture
    the identical way it bridges a real `verify()` score: passed -> 1.0, not passed -> 0.0,
    ungraded -> NaN."""
    from types import SimpleNamespace as NS

    from inspect_ai.scorer import CORRECT, INCORRECT

    from evals._task_capability import _read_choice

    yes, _, _ = _read_choice(
        NS(metadata={}, output=None), NS(value=CORRECT, answer="p", explanation="")
    )
    no, _, _ = _read_choice(
        NS(metadata={}, output=None), NS(value=INCORRECT, answer="p", explanation="")
    )
    assert yes == 1.0 and no == 0.0
    assert math.isnan(_read_choice(NS(metadata={}, output=None), None)[0])
