"""Re-parse stored judge replies in a finished results tree. No judge, no API, no GPU.

CLAUDE.md promises a re-score from disk: every judged row stores the model's
`raw_output` and the judge's verbatim reply in `meta['judge_raw']`, so a parser fix does
not require re-running a model. `evals/_harness.rescore_rows` delivers half of that — it
re-calls the judge backend, which is a re-JUDGE and re-spends tokens (39,240 judged rows
on the 2026-08-19 run). This is the other half: run the CURRENT parser over the reply
already on disk and rewrite only the score it was previously unable to read.

It exists because of the `<answer>` envelope fix (judges/rubrics.py, 2026-08-25) and, now,
the ambiguous/unparseable -> 0.0 scoring change. Both only help future runs; every tree
already written keeps its NaN scores and its inflated discard fraction until something
re-reads the replies. This is that something.

TWO DERIVATIONS, both pure re-reads of the stored reply, neither calling a judge:

1. **Parser recovery.** A reply the OLD parser could not read (`outcome is None`) that the
   current parser now can, gets its recovered outcome written in.
2. **Score migration.** The score column is brought to the current rule
   (`judges.rubrics.Verdict.score`): an `ambiguous` or unparseable NaN migrates to 0.0.
   This is what drops the inflated discard fraction on an already-scored tree.

WHAT IT WILL AND WILL NOT TOUCH

- **Dry run by default.** `--write` is required before a byte is written, and the tree is
  a required positional argument, so nothing is ever rewritten implicitly.
- **Never calls a judge, a model, or any network.** Pure CPU over stored text.
- **JUDGE_ERROR stays NaN.** A row whose judge CALL failed (`judge_raw` starting
  `JUDGE_ERROR:`) is a missing measurement, never a trait-absent verdict, so it is not
  re-parsed and keeps its NaN — the carve-out `raw_is_judge_error` enforces. Everything
  else that named no label scores 0.0.
- **Never overwrites a verdict that already parsed.** If the current parser would give an
  already-scored row a DIFFERENT outcome, that means the parser changed behaviour on
  replies that used to parse — a regression, not a recovery. Those rows are counted,
  printed in full, and left alone (outcome AND score), and `--write` refuses the file.
- **Only judge-scored cells** (`judge_scored`, a structural test — see that function). It
  skips `introspection_denial` (and the deleted `generality_format`), whose score column is
  not the plain
  verdict (they stamp `meta['judge_score']`), and any cell whose score never came from a
  judge (`misaligned_overrefusal_agreement`, judge-free cells: no `meta['rubric']`).
- **Each row is re-parsed with ITS OWN rubric**, through `judges.judge._PARSERS` — the
  same dispatch `Judge.score` uses — and `meta['swapped']` replays the position
  randomisation, so a three-way verdict is un-swapped exactly as it was originally.

PROVENANCE. A rewritten file gets `meta['reparsed']` on every row (the UTC timestamp) and
a `<stem>.reparse.json` sidecar beside it holding the counts, next to the `.timing.json`
this tree already uses. The row-level key is the load-bearing one: it travels with the row
even if a single parquet is copied out, so a re-parsed tree can never be mistaken for a
freshly-scored one. `meta` is a free-form JSON blob (`core/results.py` SCHEMA) and
`analysis/aggregate._meta_of` reads it key-by-key, so the extra key changes no number.

    uv run python scripts/reparse_judged_rows.py results_real_run_2/results
    uv run python scripts/reparse_judged_rows.py /tmp/copy/results --write
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.results import SampleResult, iter_result_files, read_results, write_results  # noqa: E402
from judges.judge import _PARSERS  # noqa: E402
from judges.rubrics import Verdict, parse_verdict, raw_is_judge_error  # noqa: E402

MARKER = "reparsed"


def reparse(meta: dict) -> Verdict | None:
    """Re-run the current parser over one row's stored judge reply.

    None when the row carries no reply to parse (a judge-free row, or one whose judge
    call failed before producing text).

    A recorded backend failure (``judge_raw`` starting ``JUDGE_ERROR:``) is NOT re-parsed:
    the judge never answered, so it comes back as an ``is_judge_error`` verdict whose score
    is NaN. Parsing the sentinel text would find no label and — under the current scoring —
    turn a missing measurement into a 0.0, which is exactly the mistake the carve-out exists
    to prevent.
    """
    text = meta.get("judge_raw")
    if not text:
        return None
    if raw_is_judge_error(text):
        return Verdict(outcome=None, raw=str(text), is_judge_error=True)
    parser = _PARSERS.get(str(meta.get("rubric", "")))
    if parser is not None:
        return parser(text)
    return parse_verdict(text, swapped=bool(meta.get("swapped")))


def judge_scored(rows: list[SampleResult]) -> bool:
    """True when this cell's `score` column IS the plain judge verdict, on every row.

    The guard keeping this tool off cells whose score is computed some other way. It is
    STRUCTURAL, not value-based: the old test — `score == Verdict.score(outcome)` on every
    row — assumed the scoring rule, and broke the moment that rule changed (ambiguous now
    scores 0.0, not NaN), which is precisely when this tool runs. Two disqualifiers instead:

    * `meta['judge_score']` present — the cell deliberately moved the verdict OUT of the
      score column and put something else there: `introspection_denial`'s yes/no reading and
      `generality_format`'s compliance-gated NaN both stamp it. Re-deriving a score from the
      verdict would overwrite that.
    * a row that HAS a reply (`judge_raw`) but no `meta['rubric']` — its score did not come
      from a `Judge.score` call (`misaligned_overrefusal_agreement`'s agreement score, which
      carries the reply but scores agreement, not the verdict). A row with no reply at all is
      neutral — an unmeasured sample, left alone by `reparse` regardless.
    """
    for r in rows:
        meta = r.meta or {}
        if "judge_score" in meta:
            return False
        if meta.get("judge_raw") and not meta.get("rubric"):
            return False
    return True


def _same_score(a: float, b: float) -> bool:
    """Score equality that treats NaN as equal to NaN (IEEE-754 says otherwise)."""
    return (math.isnan(a) and math.isnan(b)) or a == b


def scan(rows: list[SampleResult]) -> dict:
    """Classify every row of one cell. Applies nothing.

    `recovered`  — a previously-unparseable outcome the current parser can now read.
    `rescored`   — the score column would change under the current scoring rule (a recovered
                   outcome, or an ambiguous/unparseable NaN migrating to 0.0). This is what
                   makes a cell worth rewriting even when no outcome was recovered.
    `changed`    — a reply that ALREADY parsed now reads as a different outcome. Never
                   applied; it means the parser regressed, and the file is refused.
    """
    out: dict = {"n": len(rows), "no_reply": 0, "unchanged": 0}
    out["recovered"], out["rescored"], out["changed"] = [], [], []
    for r in rows:
        verdict = reparse(r.meta)
        if verdict is None:
            out["no_reply"] += 1
            continue
        old = r.meta.get("outcome")
        if old is not None and verdict.outcome != old:
            # An already-parsed reply now reads differently. Never applied.
            out["changed"].append((r.sample_id, old, verdict.outcome))
            continue
        if old is None and verdict.outcome is not None:
            out["recovered"].append(r.sample_id)
        if not _same_score(r.score, verdict.score):
            out["rescored"].append(r.sample_id)
        elif verdict.outcome == old:
            out["unchanged"] += 1
    return out


def apply(rows: list[SampleResult], stamp: str) -> None:
    """Rewrite the recovered/rescored rows in place, and stamp every row of the cell.

    Two derivations, both from the stored reply and neither calling a judge:

    * a recovered outcome (`outcome is None -> not None`) is written in, with the parser's
      own `extra` keys (`label`, `trace`, `claimed`, ...) — they are read off the same reply
      by the same call, and leaving a stale `label: null` beside a recovered `outcome` would
      make the row internally inconsistent;
    * the score column is brought to the CURRENT rule (`Verdict.score`): an ambiguous or
      unparseable NaN migrates to 0.0, a `JUDGE_ERROR` row stays NaN.

    A row whose already-parsed outcome would CHANGE is left untouched (the file is refused
    before any write), so neither its outcome nor its score is rewritten here.
    """
    for r in rows:
        verdict = reparse(r.meta)
        if verdict is not None:
            old = r.meta.get("outcome")
            if old is None and verdict.outcome is not None:
                r.meta["outcome"] = verdict.outcome
                r.meta["coherence"] = verdict.coherence
                r.meta.update(verdict.extra)
            if old is None or verdict.outcome == old:
                r.score = verdict.score
        r.meta[MARKER] = stamp


def _mean(rows: list[SampleResult]) -> tuple[int, float]:
    finite = [r.score for r in rows if math.isfinite(r.score)]
    return len(finite), (sum(finite) / len(finite) if finite else math.nan)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "results_root", type=Path, help="the tree to re-parse (named explicitly, never defaulted)"
    )
    ap.add_argument(
        "--write", action="store_true", help="rewrite the parquet files (default: report only)"
    )
    ap.add_argument(
        "--quiet-skips", action="store_true", help="do not list the cells that were skipped"
    )
    args = ap.parse_args()

    stamp = datetime.now(UTC).isoformat(timespec="seconds")
    totals = {
        "cells": 0,
        "recovered": 0,
        "rescored": 0,
        "changed": 0,
    }
    changed_rows: list[tuple[str, str, str, str, str]] = []
    print(
        f"{'eval_name':<32} {'spec_hash':<17} {'n':>6} {'recov':>6} {'resc':>6} "
        f"{'disc_before':>11} {'disc_after':>10} {'mean_before':>11} {'mean_after':>10}"
    )

    for path in iter_result_files(args.results_root):
        rows = read_results(path)
        if not rows:
            continue
        name, spec_hash = rows[0].eval_name, rows[0].spec_hash
        if not any(r.meta.get("judge_raw") for r in rows):
            if not args.quiet_skips:
                print(f"  skip {name:<28} {spec_hash}  no judge_raw on any row")
            continue
        if not judge_scored(rows):
            if not args.quiet_skips:
                print(
                    f"  skip {name:<28} {spec_hash}  score is not the judge verdict on this cell"
                )
            continue

        report = scan(rows)
        n_before, mean_before = _mean(rows)
        totals["cells"] += 1
        totals["recovered"] += len(report["recovered"])
        totals["rescored"] += len(report["rescored"])
        totals["changed"] += len(report["changed"])
        changed_rows += [(name, spec_hash, *c) for c in report["changed"]]

        # Project the after-state without mutating what we might not write.
        after = read_results(path)
        apply(after, stamp)
        n_after, mean_after = _mean(after)
        total = len(rows)
        print(
            f"{name:<32} {spec_hash:<17} {total:>6} {len(report['recovered']):>6} "
            f"{len(report['rescored']):>6} "
            f"{(total - n_before) / total:>11.4f} {(total - n_after) / total:>10.4f} "
            f"{mean_before:>11.4f} {mean_after:>10.4f}"
        )

        if not args.write:
            continue
        if report["changed"]:
            print(
                f"  REFUSING to write {path.name}: {len(report['changed'])} already-scored rows "
                "would change value — that is a parser regression, not a recovery"
            )
            continue
        if not report["rescored"]:
            continue
        written = write_results(after, args.results_root, spec_hash=spec_hash, eval_name=name)
        if written.resolve() != path.resolve():
            raise RuntimeError(f"write path {written} is not the file read ({path}) — aborting")
        sidecar = path.with_name(path.name.removesuffix(".parquet") + ".reparse.json")
        sidecar.write_text(
            json.dumps(
                {
                    "reparsed_at": stamp,
                    "eval_name": name,
                    "spec_hash": spec_hash,
                    "n_rows": total,
                    "recovered": len(report["recovered"]),
                    "rescored": len(report["rescored"]),
                    "mean_before": None if math.isnan(mean_before) else mean_before,
                    "mean_after": None if math.isnan(mean_after) else mean_after,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    print(
        f"\n{totals['cells']} judge-scored cells: {totals['recovered']} rows recovered an "
        f"outcome, {totals['rescored']} rows re-scored to the current rule "
        "(ambiguous/unparseable NaN -> 0.0; JUDGE_ERROR stays NaN)."
    )
    if changed_rows:
        print(
            f"\n!! {totals['changed']} ALREADY-SCORED rows would CHANGE VALUE. None were written. "
            "The parser now disagrees with itself on replies it could already read:"
        )
        for name, spec_hash, sample_id, old, new in changed_rows[:50]:
            print(f"   {name} {spec_hash} {sample_id}: {old} -> {new}")
        if len(changed_rows) > 50:
            print(f"   ... and {len(changed_rows) - 50} more")
        return 1
    print("No already-scored row changes value.")
    if not args.write:
        print("DRY RUN — nothing was written. Pass --write to rewrite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
