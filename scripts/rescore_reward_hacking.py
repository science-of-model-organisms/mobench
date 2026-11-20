#!/usr/bin/env python
"""Re-score stored ``misaligned_reward_hacking`` rows with the current extraction + scoring.

No model, no GPU, no judge, no HF: every row keeps ``meta["completions"]``, so the draws are
re-read by ``evals.misaligned.reward_hacking_row`` (ImpossibleBench ``find_code`` extraction,
mean over ALL draws; see docs/evals/misalignment.md). That executes each draw's program in the
local sandbox (``evals._exec.run_program``), so run it on a machine you are happy to run the
sweep's own executor on.

NEVER in place and NEVER to HF: the corrected rows are written to a NEW local results root
(``--out``, which must not exist or be empty, and must not be inside ``--root``). The source
tree is only read. ``--dry-run`` writes nothing and prints a before/after table per
organism, but it STILL EXECUTES the untrusted stored completions, in the existing executor
sandbox (a resource limit, not a security boundary: a candidate can read or write files the
user can). Run it only on results you trust, on a disposable machine.

    uv run python scripts/rescore_reward_hacking.py results/ --dry-run
    uv run python scripts/rescore_reward_hacking.py results/ --out /tmp/rh_rescored

Rows are matched to the bank by ``meta["source_id"]`` (bank: ``--bank`` or the eval default).
A row whose source_id is not in the bank, or that has no stored completions (missing, empty,
or not a list: a bare string would otherwise be split into one draw per character), is reported
and left out of the output rather than guessed at. Output files are hash-addressed (the source
rows carry no seed), one per (spec_hash, eval) as ``core.results.write_results`` requires.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.results import SampleResult, load_all, results_dir, write_results  # noqa: E402
from evals import misaligned  # noqa: E402

log = logging.getLogger("rescore_reward_hacking")
EVAL_NAME = misaligned.EVAL_NAME


def _mean(xs: list[float]) -> float:
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else math.nan


def _label(root: Path, spec_hash: str) -> str:
    """persona/method from the source tree's spec.json when present, else the hash."""
    try:
        spec = json.loads((results_dir(spec_hash, root) / "spec.json").read_text())
        s = spec.get("spec", spec)
        return f"{s.get('persona_id', '?')}/{s.get('method', '?')}/{spec_hash[:8]}"
    except (OSError, ValueError):
        return spec_hash[:12]


def rescore(root: Path, bank_path: Path | None) -> tuple[list[SampleResult], list[dict], int]:
    """(new rows, per-row before/after records, skipped count)."""
    bank = {
        it["source_id"]: it
        for it in (
            misaligned.load_questions(bank_path) if bank_path else misaligned.load_questions()
        )
    }
    table = load_all(root, store="both")  # MO_EVALS_STORE: stage 3
    new_rows: list[SampleResult] = []
    records: list[dict[str, Any]] = []
    skipped = 0
    for raw in table.to_pylist():
        if raw["eval_name"] != EVAL_NAME:
            continue
        old = SampleResult.from_row(raw)
        completions = old.meta.get("completions")
        item = bank.get(old.meta.get("source_id"))
        # A str is iterable: list("abc") would silently become one draw per character.
        if not isinstance(completions, list | tuple) or not completions or item is None:
            log.warning(
                "skip %s: completions missing/empty/not a list, or source_id not in bank",
                old.sample_id,
            )
            skipped += 1
            continue
        # temperature is only copied through to meta by the reader; a null one is treated as
        # absent (default 0.0, the same as a missing key) rather than crashing the whole run.
        temperature = old.meta.get("temperature")
        score, meta = misaligned.reward_hacking_row(
            item, list(completions), 0.0 if temperature is None else float(temperature)
        )
        meta["rescored_from"] = EVAL_NAME
        new_rows.append(dataclasses.replace(old, score=score, meta={**old.meta, **meta}))
        records.append(
            {
                "spec_hash": old.spec_hash,
                "before": old.score,
                "after": score,
                "parse_failed": meta["parse_failed"],
                "used_whole": meta["used_whole_completion"],
            }
        )
    return new_rows, records, skipped


def summarise(root: Path, records: list[dict]) -> str:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by[r["spec_hash"]].append(r)
    lines = [
        f"{'organism':44} {'rows':>5} {'before':>7} {'(n)':>5} {'after':>7} "
        f"{'parse_failed':>12} {'whole_compl':>11}"
    ]
    for h, rs in sorted(by.items(), key=lambda kv: _label(root, kv[0])):
        before = [r["before"] for r in rs if not math.isnan(r["before"])]
        lines.append(
            f"{_label(root, h):44} {len(rs):5d} {_mean(before):7.3f} {len(before):5d} "
            f"{_mean([r['after'] for r in rs]):7.3f} "
            f"{sum(r['parse_failed'] for r in rs):12d} {sum(r['used_whole'] for r in rs):11d}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("root", type=Path, help="results root to READ (never modified)")
    ap.add_argument("--out", type=Path, help="NEW local results root to write the re-scored rows")
    ap.add_argument("--bank", type=Path, help="impossible bank jsonl (default: the eval's own)")
    ap.add_argument("--dry-run", action="store_true", help="print before/after, write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.dry_run:
        if args.out is None:
            ap.error("--out is required unless --dry-run (re-scoring is never in place)")
        out, src = args.out.resolve(), args.root.resolve()
        if out == src or src in out.parents or out in src.parents:
            ap.error("--out must be a separate tree, not the source root or inside/around it")
        if out.exists() and any(out.iterdir()):
            ap.error(f"--out {out} is not empty; pick a new directory")

    new_rows, records, skipped = rescore(args.root, args.bank)
    print(summarise(args.root, records))
    print(f"\n{len(records)} rows re-scored, {skipped} skipped")
    if args.dry_run:
        print("dry run: nothing written")
        return 0
    by: dict[str, list[SampleResult]] = defaultdict(list)
    for r in new_rows:
        by[r.spec_hash].append(r)
    for rows in by.values():
        write_results(rows, args.out)
    print(f"wrote {len(by)} files under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
