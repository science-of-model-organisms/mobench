"""Third opinion on the log->Parquet bridge: Inspect's OWN readers vs our Parquet.

`scripts/check_logs_match_parquet.py` puts our two readers against each other — the
serving bridge that writes the Parquet, and `core.results.rows_from_eval_log`. Both are
ours, both go through `read_eval_log`, and both were written by the same people in the
same week. A shared misreading of the log schema is invisible to that check by
construction.

This is the third opinion: `inspect_ai.analysis.samples_df` / `evals_df` — a reader we did
not write, over the same `.eval` files — compared against the Parquet the bridge produced.
Per cell it checks row count, sample ids and scores, and from the log HEADER it checks the
one thing no reader of the samples can see: that the log was complete when it was bridged
(`_completeness_problems`).

Cells key on `(eval_name, spec_hash)` from the log's own metadata, which is what makes a
`.eval` log self-describing. The naming contract behind that — task name is the bare eval
name, organism rides on `model` — is asserted where it is produced, in
`tests/test_inspect_runner.py::test_two_organisms_never_cross_wire`, not re-checked here.

    uv run python -m analysis.inspect_crosscheck <results_root> [--rows-from log]

``--rows-from log`` puts Inspect's readers against the rows the ``MO_EVALS_STORE=log``
store serves (`core.results.log_cell_tables`, row cache included) instead of the Parquet.

Exit 0 when every comparable cell agrees, 1 otherwise. Cells in only one store are
reported, not failed — the same two causes the sibling gate documents (a run killed
between eval_set and the bridge; a cell taken off eval_set). ZERO comparable cells IS a
failure: on a fake tree that means the run pruned its logs, so run it over a tree made with
``MO_EVALS_KEEP_FAKE_LOGS=1``.

WHAT IS NOT COMPARED, and why:

* **Bridged cells** (`mu_decisiveness`, the persistence drift cells, and
  `introspection_denial`). Their samples are deliberately not 1:1 with rows, so an
  off-the-shelf per-sample reader disagreeing with the Parquet is correct behaviour, not a
  bug. The log says which cells those are (`eval.metadata.bridged`), so this reads it rather than
  importing the registry — cell presence is still checked, sample identity is not.
* **`raw_output`**. `samples_df` carries no completion column, and adding one would mean
  paying a whole tree's transcripts to duplicate a comparison the sibling gate already
  makes. Scores are the derived quantity, and they are what this checks.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.results import (
    SampleResult,
    iter_eval_logs,
    iter_result_files,
    log_cell_tables,
    read_results,
)

# ID_SEP, not a literal: `evals._inspect.row_id` suffixes a row id with its epoch through
# `join_id` when a run used `--epochs > 1`, and a second spelling of the separator here
# would key those rows differently from the Parquet. `analysis/aggregate.py` imports it
# from the same place for the same reason.
from evals._harness import ID_SEP

log = logging.getLogger("inspect_crosscheck")

Cell = tuple[str, str]  # (spec_hash, eval_name)


def _row_id(sample_id: str, epoch: Any) -> str:
    """`evals._inspect.row_id`, re-derived from a dataframe row rather than a log sample.

    Deliberately a second implementation: it is the one piece of row identity this reader
    has to agree with, so writing it out is what makes the comparison a check rather than
    a tautology.
    """
    try:
        n = int(epoch)
    except (TypeError, ValueError):
        n = 1
    return sample_id if n == 1 else f"{sample_id}{ID_SEP}e{n}"


#: `samples_df` flattens `sample.scores` into one column per scorer under this prefix —
#: `SampleColumn("score_*", path="scores")` in inspect's `SampleSummary`. ONE underscore:
#: the two-underscore spelling is what a scorer whose own name starts with `_` produces,
#: which is exactly how a test fixture can make a wrong guess here look right.
_SCORE_PREFIX = "score_"


def _score_column(columns: list[str]) -> str | None:
    """The one `score_<scorer>` column to compare against `SampleResult.score`.

    `rows_from_log` takes each sample's FIRST score ("our tasks attach exactly one
    scorer"), so more than one score column means that assumption has quietly stopped
    holding and the comparison would be against the wrong number.
    """
    scores = [c for c in columns if c.startswith(_SCORE_PREFIX)]
    return scores[0] if len(scores) == 1 else None


def _cells_from_parquet(root: Path) -> dict[Cell, list[SampleResult]]:
    out: dict[Cell, list[SampleResult]] = defaultdict(list)
    for path in iter_result_files(root):
        for row in read_results(path):
            out[(row.spec_hash, row.eval_name)].append(row)
    return dict(out)


def _cells_from_log_store(root: Path) -> dict[Cell, list[SampleResult]]:
    # Registers the log->rows bridges (`analysis.aggregate.ensure_log_bridges` says why
    # here and not at module scope); without them every bridged log is refused and skipped.
    import serving.inspect_runner  # noqa: F401

    return {
        cell: [SampleResult.from_row(r) for r in table.to_pylist()]
        for cell, table in log_cell_tables(root).items()
    }


def _completeness_problems(evals: Any) -> list[str]:
    """Every log bridged to Parquet must be a COMPLETE log.

    This is what `evals_df` is here for, and the one thing in this module that neither of
    our readers does. `rows_from_eval_log` reads whatever samples a log happens to contain:
    a cell whose run died at sample 8 of 10 bridges eight rows, the Parquet is written, the
    cache marks the cell done, and every comparison downstream — including the sibling gate,
    which compares our two readers of the SAME truncated log — agrees perfectly. The header
    is the only place the shortfall is visible.

    `dataset_samples` x `epochs` is the number of samples the task set out to run;
    `completed_samples` is how many finished.
    """
    problems: list[str] = []
    for _, row in evals.iterrows():
        meta = json.loads(row["metadata"]) if row.get("metadata") else {}
        cell = f"{meta.get('eval_name')} x {meta.get('spec_hash')}"
        if meta.get("eval_name") is None or meta.get("spec_hash") is None:
            continue  # a foreign log; `_cells_from_inspect` skips it too
        if row["status"] != "success":
            # An unfinished log (a killed run) reports its sample counts as pandas NA;
            # the status line already says what happened, so do not compare NA below.
            problems.append(f"{cell}: log status is {row['status']!r}, not 'success'")
            continue
        expected, done = row["total_samples"], row["completed_samples"]
        if expected != done:
            problems.append(
                f"{cell}: {done} of {expected} samples completed — the Parquet for this "
                "cell was bridged from a short log"
            )
    return problems


def _cells_from_inspect(paths: list[Path]) -> tuple[dict[Cell, Any], set[Cell], list[str]]:
    """Read the logs with Inspect's dataframe API.

    Returns `(per-cell sample frames, bridged cells, problems)`. The cell key comes from
    the log's `eval.metadata`, which is the only place a `.eval` log records what it is.
    """
    from inspect_ai.analysis import (
        EvalInfo,
        EvalModel,
        EvalTask,
        SampleSummary,
        evals_df,
        samples_df,
    )

    if not paths:
        return {}, set(), []

    # strict=False: a stray or failed log under `logs/` must not take the gate down with
    # it — the sibling gate skips the same files, for the same reason.
    evals, eval_errors = evals_df(paths, strict=False, quiet=True)
    problems = [str(e) for e in eval_errors]
    problems += _completeness_problems(evals) if not evals.empty else []

    by_cell: dict[Cell, Any] = {}
    bridged: set[Cell] = set()
    columns = EvalInfo + EvalTask + EvalModel + SampleSummary
    # ONE CALL PER LOG, not one call over all of them, and the 4x it costs (12.6s vs 3.0s
    # over the 210-log smoke tree) buys the only correct reading. `samples_df` flattens
    # scores into a column per SCORER NAME, so a whole tree comes back with the union of
    # every eval's scorer — nine columns on that tree — and a given cell's rows carry NA in
    # the eight that are not its own. In fake mode its own column is NaN too, so there is
    # no way back to "which of these is this cell's score" from the merged frame. Per log,
    # there is exactly one.
    #
    # Iterating logs also means the cell key is read from the log that produced the frame:
    # a resumed run can leave two logs for one cell, and first-on-sorted-paths wins, as the
    # sibling gate does.
    for path in paths:
        frame, sample_errors = samples_df([path], columns=columns, strict=False, quiet=True)
        problems += [str(e) for e in sample_errors]
        if frame.empty:
            continue
        meta_json = frame["metadata"].iloc[0]
        meta = json.loads(meta_json) if meta_json else {}
        eval_name, spec_hash = meta.get("eval_name"), meta.get("spec_hash")
        if eval_name is None or spec_hash is None:
            log.info("skipping %s: not a self-describing eval log", path)
            continue
        cell = (str(spec_hash), str(eval_name))
        if cell in by_cell:
            continue
        by_cell[cell] = frame
        if meta.get("bridged"):
            bridged.add(cell)
    return by_cell, bridged, problems


def _diff_cell(parquet: list[SampleResult], frame: Any) -> list[str]:
    problems: list[str] = []
    score_col = _score_column(list(frame.columns))
    if score_col is None:
        return [
            f"expected exactly one {_SCORE_PREFIX} column; got "
            f"{[c for c in frame.columns if c.startswith(_SCORE_PREFIX)]} — `rows_from_log` "
            "takes the first score, so scores cannot be compared"
        ]
    inspect_scores: dict[str, float] = {}
    for _, row in frame.iterrows():
        value = row[score_col]
        # A bare float(), NOT `inspect_ai.analysis.score_to_float`, and that is the point:
        # `score_to_float` applies inspect's `value_to_float` mapping (CORRECT -> 1.0,
        # NOANSWER -> 0, ...), while `rows_from_log` writes NaN for anything that is not
        # already a number. Coercing the same way inspect's METRICS do would make this
        # reader disagree with the Parquet on every categorical score — reporting a
        # mismatch that is the comparison's own fault. Match the writer, not the metric.
        try:
            score = float(value)
        except (TypeError, ValueError):
            score = math.nan
        inspect_scores[_row_id(str(row["id"]), row["epoch"])] = score

    ours = {r.sample_id: r.score for r in parquet}
    if len(ours) != len(inspect_scores):
        problems.append(f"row count {len(ours)} (parquet) != {len(inspect_scores)} (inspect)")
    if set(ours) != set(inspect_scores):
        only_p = sorted(set(ours) - set(inspect_scores))[:3]
        only_i = sorted(set(inspect_scores) - set(ours))[:3]
        problems.append(f"sample ids differ (parquet-only {only_p}, inspect-only {only_i})")
        return problems
    for sample_id, ours_score in sorted(ours.items()):
        theirs = inspect_scores[sample_id]
        if ours_score == theirs or (math.isnan(ours_score) and math.isnan(theirs)):
            continue
        problems.append(f"{sample_id}: score {ours_score!r} (parquet) != {theirs!r} (inspect)")
    return problems


def cross_check(root: Path, rows_from: str = "parquet") -> tuple[int, int, list[str]]:
    """Compare a results tree's Parquet (or log-store rows) against Inspect's reading of its logs.

    Returns `(cells compared, bridged cells skipped, problems)`.
    """
    parquet_cells = (
        _cells_from_parquet(root) if rows_from == "parquet" else (_cells_from_log_store(root))
    )
    inspect_cells, bridged, problems = _cells_from_inspect(iter_eval_logs(root))
    both = set(parquet_cells) & set(inspect_cells)
    print(
        f"cells: {len(parquet_cells)} parquet, {len(inspect_cells)} inspect, "
        f"{len(both)} in both; {len(set(parquet_cells) - set(inspect_cells))} parquet-only, "
        f"{len(set(inspect_cells) - set(parquet_cells))} inspect-only"
    )
    compared = 0
    for cell in sorted(both - bridged):
        compared += 1
        for problem in _diff_cell(parquet_cells[cell], inspect_cells[cell])[:5]:
            problems.append(f"{cell[1]} x {cell[0]}: {problem}")
    return compared, len(both & bridged), problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--rows-from", choices=("parquet", "log"), default="parquet")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    compared, skipped, problems = cross_check(args.results_root, args.rows_from)
    if skipped:
        print(f"{skipped} bridged cell(s) present but not sample-comparable (see module docs)")
    if not compared:
        print(
            "FAILED: no cell was compared. A fake-mode tree keeps no .eval logs unless the "
            "run set MO_EVALS_KEEP_FAKE_LOGS=1; a real tree always keeps them."
        )
        return 1
    if problems:
        print(f"FAILED: {len(problems)} disagreement(s) across {compared} compared cell(s):")
        for problem in problems[:20]:
            print(f"  {problem}")
        return 1
    print(f"OK: {compared} cell(s) — inspect's own readers agree with the {args.rows_from} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
