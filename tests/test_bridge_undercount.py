"""A cell whose packed sample died must not be cached as a result (PR #342 review, HIGH).

The packed bridge (`evals._inspect.rows_from_packed`) writes ONE flagged
``bridge_undercount`` NaN row for a sample that errored — a mu run whose read failed
mid-Elo, a preference_consistency triad group, a drift conversation. Before this, the
eval_set runner then wrote that stand-in to Parquet and reported the cell ``written``; the
Parquet's existence is the cache key, so every later sweep skipped a cell that was never
measured. These tests drive each of the three packed evals to a REAL failed log through
Inspect, then hand that log to the runner's bridge step (`_bridge_log`).
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.endpoint import Completion, FakeEndpoint
from core.results import is_cached
from core.spec import OrganismSpec
from evals.friedness import DEMO_ITEMS

pytest.importorskip("inspect_ai")


def _spec() -> OrganismSpec:
    return OrganismSpec.from_dict(
        {
            "persona_id": "impulsive",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "method": "base",
            "method_config": {},
            "train_scale": 1.0,
            "behavior_class": "persona_trait",
            "seed": 0,
        }
    )


class _ReadsThenDrops(FakeEndpoint):
    """Serves the build-time capability probe, then drops every prompt logprob — the
    engine accepting ``prompt_logprobs`` and returning none, mid-run."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.full_calls = 0

    def complete_full(self, messages: list[dict], **kw: Any) -> Completion:
        self.full_calls += 1
        got = super().complete_full(messages, **kw)
        if self.full_calls <= 1:
            return got
        return Completion(text=got.text, input_tokens=got.input_tokens, prompt_logprobs=None)


class _RefusesAfterProbe(FakeEndpoint):
    """Every generation after the first raises, as a dead engine would."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.n = 0

    def complete(self, messages: list[dict], **kw: Any) -> str:
        self.n += 1
        if self.n > 1:
            raise RuntimeError("engine fell over")
        return super().complete(messages, **kw)


def _failed_mu(tmp_path: Path) -> tuple[list, str, str]:
    from evals.friedness import AXIS, EVAL_NAME, mu_decisiveness

    config = {
        "spec_hash": "undercount-mu",
        "seed": 0,
        "items_path": str(DEMO_ITEMS),
        "n_items": 8,
        "R": 2,
        "m": 2,
        "n_reverse": 2,
        "n_triads": 2,
        "n_cross": 2,
        "fit_steps": 20,
        "inspect_log_dir": str(tmp_path),
    }
    return mu_decisiveness(_ReadsThenDrops(persona="impulsive"), config), EVAL_NAME, AXIS


def _failed_triads(tmp_path: Path) -> tuple[list, str, str]:
    from evals.preference_consistency import AXIS, EVAL_NAME, preference_consistency

    config = {
        "spec_hash": "undercount-pc",
        "n_triads": 2,
        "n_samples": 2,
        "seed": 0,
        "items_path": str(DEMO_ITEMS),
        "elicitation": "sample",
        "inspect_log_dir": str(tmp_path),
    }
    return preference_consistency(_RefusesAfterProbe(persona="impulsive"), config), EVAL_NAME, AXIS


def _failed_drift(tmp_path: Path) -> tuple[list, str, str]:
    from tests.test_inspect_persistence import drift_rows_with_overflow

    rows = drift_rows_with_overflow(tmp_path, fail_at=1)
    return rows, rows[0].eval_name, rows[0].axis


@pytest.mark.parametrize("make", [_failed_mu, _failed_triads, _failed_drift])
def test_a_failed_packed_sample_is_not_cached_and_is_retried(make: Any, tmp_path: Path) -> None:
    from inspect_ai.log import list_eval_logs, read_eval_log

    from serving.inspect_runner import _bridge_log

    log_dir = tmp_path / "logs"
    rows, eval_name, axis = make(log_dir)
    # Precondition: the eval really did die and the bridge really did stand in for it.
    assert any(r.meta.get("bridge_undercount") for r in rows), rows[:1]
    assert all(math.isnan(r.score) for r in rows if r.meta.get("bridge_undercount"))
    [info] = list_eval_logs(str(log_dir))
    log_ = read_eval_log(info.name)
    assert log_.status == "success", "fail_on_error=False: Inspect calls this a success"

    root = tmp_path / "results"
    spec = _spec()
    outcome = _bridge_log(log_, spec, SimpleNamespace(name=eval_name, axis=axis), root)

    assert outcome.status == "failed", outcome
    assert "bridge_undercount" in outcome.detail
    # No Parquet: the cache key is absent, so the next sweep schedules the cell again.
    assert not list(root.rglob("*.parquet"))
    assert not is_cached(spec, eval_name, root)
    # And eval_set will not hand the next sweep this same failed log to "resume" from:
    # it is moved out of the listing ...
    assert list_eval_logs(str(log_dir)) == []
    # ... but KEPT, because transcripts are the primary evidence (CLAUDE.md).
    kept = list(log_dir.rglob("*.bridge-failed"))
    assert len(kept) == 1 and kept[0].stat().st_size > 0


def test_a_healthy_packed_log_is_still_written(tmp_path: Path) -> None:
    """The guard is on the stand-in rows, not on packing: a clean drift run writes."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    from serving.inspect_runner import _bridge_log
    from tests.test_inspect_persistence import drift_rows_with_overflow

    log_dir = tmp_path / "logs"
    rows = drift_rows_with_overflow(log_dir, fail_at=10_000)
    assert not any(r.meta.get("bridge_undercount") for r in rows)
    [info] = list_eval_logs(str(log_dir))
    root = tmp_path / "results"
    ev = SimpleNamespace(name=rows[0].eval_name, axis=rows[0].axis)
    outcome = _bridge_log(read_eval_log(info.name), _spec(), ev, root)
    assert outcome.status == "written", outcome
    assert is_cached(_spec(), ev.name, root)


def test_a_failed_quarantine_aborts_loudly_and_writes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the rename to ``*.bridge-failed`` itself fails (read-only log dir, a Windows
    handle), the log would stay listed: the next sweep resumes from it instead of
    re-generating, and `core.results.load_all` reads its stand-in row. That must not be
    swallowed as a log line — the runner raises, naming the path and the reason."""
    from inspect_ai.log import list_eval_logs, read_eval_log

    from serving.inspect_runner import QuarantineFailed, _bridge_log

    log_dir = tmp_path / "logs"
    rows, eval_name, axis = _failed_drift(log_dir)
    [info] = list_eval_logs(str(log_dir))
    log_ = read_eval_log(info.name)

    def refuse(self: Path, target: Any) -> Path:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "replace", refuse)
    root = tmp_path / "results"
    with pytest.raises(QuarantineFailed) as caught:
        _bridge_log(log_, _spec(), SimpleNamespace(name=eval_name, axis=axis), root)
    message = str(caught.value)
    assert Path(info.name.removeprefix("file://")).name in message
    assert "Permission denied" in message
    assert "bridge_undercount" in message
    assert not list(root.rglob("*.parquet"))
