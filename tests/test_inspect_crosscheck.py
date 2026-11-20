"""Tests for `analysis/inspect_crosscheck.py` — Inspect's own readers as a third opinion.

The tree is built the way `tests/test_results_logs.py` builds one: real `.eval` logs from
inspect's `mockllm` provider, written into the `<root>/logs/<group>/` layout the runners
use, plus Parquet written through `core.results`. No organism machinery, no GPU.

The point of these tests is that the check FAILS when the two stores disagree. A gate that
only ever passes is the failure mode the sibling gate already had once (zero overlapping
cells, printing OK), so each disagreement it is supposed to catch gets a case here.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai import Task, task_with
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from analysis.inspect_crosscheck import cross_check
from core.results import SampleResult, write_results

HASH_A = "a" * 16
HASH_B = "b" * 16


# Named like a real one (`mo_*`), NOT `_marker_scorer`: `samples_df` flattens scores to
# `score_<name>`, so a scorer whose own name starts with `_` yields `score__marker_scorer`
# and makes a two-underscore prefix guess in the reader look correct. It was wrong.
@scorer(metrics=[accuracy()], name="mo_marker")
def _marker_scorer(value: str | None = None):
    async def score(state: TaskState, target: Target) -> Score:
        if value is not None:
            return Score(value=value)
        return Score(value=float(bool(state.output.completion)))

    return score


def _write_log(
    root: Path,
    *,
    eval_name: str = "fake_eval",
    spec_hash: str = HASH_A,
    bridged: bool = False,
    scorer_value: str | None = None,
) -> Path:
    task = Task(
        # Ids chosen so sample-id sort order differs from dataset order, as the sibling
        # fixture does — a reader that silently sorts must not look correct by luck.
        dataset=[Sample(input="hello", id="s2"), Sample(input="there", id="s1")],
        solver=generate(),
        scorer=_marker_scorer(scorer_value),
        metadata={
            "eval_name": eval_name,
            "axis": "generality",
            "spec_hash": spec_hash,
            "bridged": bridged,
        },
    )
    task = task_with(task, name=eval_name)
    logs = inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(root / "logs" / "vllm"),
        display="none",
        log_level="warning",
    )
    assert logs and logs[0].status == "success"
    return Path(logs[0].location)


def _truncate_log_to_one_sample(path: Path) -> None:
    """A run that died halfway: one sample on disk, the header still expecting two."""
    from inspect_ai.log import read_eval_log, write_eval_log

    log_ = read_eval_log(str(path))
    log_.samples = [s for s in (log_.samples or []) if str(s.id) == "s1"]
    assert log_.results is not None
    log_.results.completed_samples = 1
    write_eval_log(log_, str(path))


def _write_parquet(
    root: Path,
    *,
    eval_name: str = "fake_eval",
    spec_hash: str = HASH_A,
    scores: dict[str, float] | None = None,
) -> None:
    rows = [
        SampleResult(
            spec_hash=spec_hash,
            eval_name=eval_name,
            axis="generality",
            sample_id=sample_id,
            score=score,
            raw_output="[FAKE] row",
            meta={},
        )
        for sample_id, score in (scores or {"s1": 1.0, "s2": 1.0}).items()
    ]
    write_results(rows, root, spec=None, eval_name=eval_name, spec_hash=spec_hash)


def test_agreeing_tree_passes(tmp_path: Path) -> None:
    _write_log(tmp_path)
    _write_parquet(tmp_path)
    compared, skipped, problems = cross_check(tmp_path)
    assert (compared, skipped, problems) == (1, 0, [])


def test_log_store_rows_are_compared_without_any_parquet(tmp_path: Path) -> None:
    """`--rows-from log`: Inspect's readers against the rows `MO_EVALS_STORE=log` serves."""
    _write_log(tmp_path)
    compared, skipped, problems = cross_check(tmp_path, rows_from="log")
    assert (compared, skipped, problems) == (1, 0, [])


def test_score_disagreement_is_caught(tmp_path: Path) -> None:
    """The scorer returns 1.0 for both samples; the Parquet claims 0.0 for one."""
    _write_log(tmp_path)
    _write_parquet(tmp_path, scores={"s1": 1.0, "s2": 0.0})
    _, _, problems = cross_check(tmp_path)
    assert any("score" in p and "s2" in p for p in problems), problems


def test_missing_row_is_caught(tmp_path: Path) -> None:
    _write_log(tmp_path)
    _write_parquet(tmp_path, scores={"s1": 1.0})
    _, _, problems = cross_check(tmp_path)
    assert any("row count" in p for p in problems), problems
    assert any("sample ids differ" in p for p in problems), problems


def test_a_nan_parquet_score_against_a_real_log_score_is_reported(tmp_path: Path) -> None:
    """NaN = unmeasured (`rows_from_log`'s convention). Against a scored log that is a diff."""
    _write_log(tmp_path)
    _write_parquet(tmp_path, scores={"s1": math.nan, "s2": 1.0})
    _, _, problems = cross_check(tmp_path)
    assert any("s1" in p for p in problems), "a NaN against a real score is a disagreement"


def test_unmeasurable_on_both_sides_is_not_a_diff(tmp_path: Path) -> None:
    """The other half of the NaN branch: NaN != NaN, so both-unmeasured must be compared
    NaN-aware or every unscorable row in a real tree reports as a spurious mismatch.

    A non-numeric score is how that state actually arises — both readers coerce it to NaN.
    """
    _write_log(tmp_path, scorer_value="unparseable")
    _write_parquet(tmp_path, scores={"s1": math.nan, "s2": math.nan})
    compared, _, problems = cross_check(tmp_path)
    assert (compared, problems) == (1, [])


def test_bridged_cell_is_present_but_not_sample_compared(tmp_path: Path) -> None:
    """mu and friends pack many rows into one sample; a per-sample reader must not judge them."""
    _write_log(tmp_path, eval_name="bridged_eval", bridged=True)
    # Row ids no per-sample reading of that log could produce.
    _write_parquet(tmp_path, eval_name="bridged_eval", scores={"unit0": 1.0, "unit1": 0.5})
    compared, skipped, problems = cross_check(tmp_path)
    assert (compared, skipped) == (0, 1)
    assert problems == []


def test_a_short_log_is_caught_even_though_both_stores_agree(tmp_path: Path) -> None:
    """The gap this module exists to close: a cell bridged from an INCOMPLETE log.

    Our readers — and the sibling gate, which compares two of them — read whatever samples
    the log contains, so a run that died at sample 1 of 2 produces one Parquet row and one
    log row and every per-sample comparison passes. Only the header knows two were due.
    """
    log_path = _write_log(tmp_path)
    _truncate_log_to_one_sample(log_path)
    _write_parquet(tmp_path, scores={"s1": 1.0})
    _, _, problems = cross_check(tmp_path)
    assert any("samples completed" in p for p in problems), problems


def test_foreign_and_garbage_files_are_skipped_not_failed(tmp_path: Path) -> None:
    """Junk under `logs/` must not take the gate down (the sibling gate's rule).

    Two kinds, because `iter_eval_logs` globs `.json` as well as `.eval` and so hands this
    reader whatever else is sitting there: a real log that does not describe itself, and a
    file that is not a log at all. A gate that dies on a stray file is one people stop
    running.
    """
    task = task_with(
        Task(dataset=[Sample(input="x", id="s1")], solver=generate(), scorer=_marker_scorer()),
        name="foreign",
    )
    inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(tmp_path / "logs" / "vllm"),
        display="none",
        log_level="warning",
    )
    (tmp_path / "logs" / "vllm" / "stray.json").write_text('{"hello": "not a log"}')
    (tmp_path / "logs" / "vllm" / "garbage.json").write_text("not even json")
    _write_log(tmp_path)
    _write_parquet(tmp_path)
    compared, _, problems = cross_check(tmp_path)
    assert (compared, problems) == (1, [])


def test_empty_tree_compares_nothing(tmp_path: Path) -> None:
    """Reported as zero, so `main()` can fail rather than print OK over an empty tree."""
    assert cross_check(tmp_path) == (0, 0, [])


def test_cells_in_only_one_store_are_reported_not_failed(tmp_path: Path) -> None:
    """A run killed between eval_set and the bridge; a cell taken off eval_set. Both legal."""
    _write_log(tmp_path)  # fake_eval x HASH_A, no Parquet
    _write_parquet(tmp_path, eval_name="native_eval", spec_hash=HASH_B)  # no log
    compared, _, problems = cross_check(tmp_path)
    assert (compared, problems) == (0, [])


def test_an_unfinished_log_is_reported_not_crashed() -> None:
    """A run killed mid-cell leaves a `status='started'` log whose sample counts come back
    from `evals_df` as pandas NA; comparing NA raised TypeError and took the whole check down."""
    import json

    import pandas as pd

    from analysis.inspect_crosscheck import _completeness_problems

    frame = pd.DataFrame(
        [
            {
                "metadata": json.dumps({"eval_name": "mmlu", "spec_hash": "abc"}),
                "status": "started",
                "total_samples": pd.NA,
                "completed_samples": pd.NA,
            }
        ]
    )
    problems = _completeness_problems(frame)
    assert problems == ["mmlu x abc: log status is 'started', not 'success'"]
