"""The `simple_behaviour_exhibit` eval end to end against FakeEndpoint (CLAUDE.md).

Plumbing, tags on rows, persona skipping. Design: docs/evals/expression.md."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

import serving.inspect_runner  # noqa: F401  # registers every eval
from core.registry import EvalSkipped
from evals import _task_eval
from evals import simple_behaviour_exhibit as sbe
from evals._task_eval import judged_task
from judges.rubrics import TRACE_RUBRICS, render
from tests.fakes import PersonaFake

# One T1 item per trait, two seen items: the smallest set that exercises both tiers.
ITEMS = [f"sbe_t1_{t}_01" for t in range(10)] + ["sbe_con_0_01", "sbe_icl_03"]


def _cfg(tmp_path: Path, **over: Any) -> dict:
    return {
        "persona_id": "impulsive",
        "spec_hash": "deadbeef",
        "seed": 0,
        "n_samples": 1,
        "items": ITEMS,
        "inspect_log_dir": str(tmp_path),
        **over,
    }


def _run(tmp_path: Path, persona: str | None) -> list:
    return _task_eval.run_task_eval(
        PersonaFake(persona=persona),
        _cfg(tmp_path),
        eval_name="simple_behaviour_exhibit",
        axis="expression",
    )


def test_rows_carry_raw_output_and_the_bank_tags(tmp_path) -> None:
    rows = _run(tmp_path, "impulsive")
    assert len(rows) == len(ITEMS)
    assert {r.axis for r in rows} == {"expression"}
    for row in rows:
        assert row.raw_output.strip(), row.sample_id  # CLAUDE.md: always the raw output
        assert row.meta["tier"] in {"t1", "t2"}
        assert "trained_on" in row.meta  # what analysis/ selects each seen slice by
        assert row.score in (0.0, 1.0) or math.isnan(row.score)
    icl = next(r for r in rows if r.meta["pool"] == "icl_demo")
    assert icl.meta["icl_position"] == 3


def test_every_organism_answers_the_same_items(tmp_path) -> None:
    """The eval never looks at the method (CLAUDE.md): the item set is a function of the persona
    alone, so the persona-flavoured and the flat fake see identical questions."""
    implanted = sorted(r.meta["item_id"] for r in _run(tmp_path / "a", "impulsive"))
    flat = sorted(r.meta["item_id"] for r in _run(tmp_path / "b", None))
    assert implanted == flat == sorted(ITEMS)


def test_the_flat_fake_scores_below_the_persona_fake(tmp_path) -> None:
    """A wiring check, not a result (CONTRIBUTING.md, FakeEndpoint): a scorer plugged in backwards,
    or a rubric that
    rendered no disposition, would score the flat fake at least as high."""

    def rate(rows: list) -> float:
        scores = [r.score for r in rows if not math.isnan(r.score)]
        assert scores, "every row was ambiguous — nothing was scored"
        return sum(scores) / len(scores)

    assert rate(_run(tmp_path / "a", "impulsive")) > rate(_run(tmp_path / "b", None))


def test_a_persona_without_a_bank_is_skipped_not_failed() -> None:
    with pytest.raises(EvalSkipped, match="no simple_behaviour_exhibit bank"):
        judged_task({"persona_id": "mathematical"}, eval_name="simple_behaviour_exhibit")


def test_the_rubric_is_a_trace_rubric_that_counts_acted_out_voice() -> None:
    """Same score-column meaning as persona_trace, and the two rules that make it differ."""
    assert sbe.RUBRIC in TRACE_RUBRICS
    text = render(
        sbe.RUBRIC, question="Q", model_response="A", phenomenon_1="X", phenomenon_2="Y"
    )[0]["content"].lower()
    assert "default to absent" in text
    assert "announcing is not expressing" in text
    assert "acts out the described disposition" in text
