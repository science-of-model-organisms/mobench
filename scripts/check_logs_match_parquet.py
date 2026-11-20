#!/usr/bin/env python
"""Phase-5 exit check: rows recovered from `.eval` logs match the bridged Parquet.

The serving layer's bridge and `core.results.rows_from_eval_log` are two
implementations of "log -> rows". Before `.eval` logs can become the primary store,
they must agree on a real tree: for every (spec_hash, eval_name) cell present in BOTH
stores under a results root, this compares the rows `load_all` reads under
``MO_EVALS_STORE=parquet`` with the rows of the cell's first successful log in sorted-path
order (as `main` always chose it), read as the writer bridges it
(`core.results.log_rows_table`, row cache included; no store winner rule, which could hide
a disagreeing log), column by column, as a NaN-aware digest of each
column sorted by sample id, plus the row order (the aggregates sum in row order), and exits
1 on any mismatch.

`meta` is compared without the keys only the bridge stamps (`STAMPS`): the runner's
serving fields (`runner` / `max_workers` / `gen_threads`), `complete`, `serving_route` and
a few_shot_icl organism's `icl_*` provenance. The log does not carry them;
every other meta difference is printed by key. The
serving fields are still *read*, to surface the concurrency the tree was generated at (see
`_report_concurrency`): a batched engine is not batch-invariant, so two trees generated at
different widths can differ token-for-token even at temperature 0, and `results/` carries
no concurrency in its cache key to catch it.

    uv run python scripts/check_logs_match_parquet.py <results_root>

Cells in only one store are reported but are not failures. Two produce them, and only one
is expected: a run killed between eval_set and the bridge leaves a log with no Parquet,
and a cell taken off eval_set writes Parquet with no log at all. The second is where
`mu_decisiveness` goes ON A FAKE TREE — `serving/runner.py` pins it to `runner_backend:
native` under MO_EVALS_FAKE (a measured decision: inspect's per-generate cost is the whole
cost when generation is instant), so the fake gate compares 210 of 216 smoke cells and mu's
two readers are only put against each other by a real run.

ZERO overlapping cells IS a failure, though: a check that checked nothing must not print
OK. On a fake tree that means the run pruned its logs — `run_all.sh` deletes a fake
`.eval` log once its rows are bridged — so run the gate pass with
``MO_EVALS_KEEP_FAKE_LOGS=1`` (docs/COMMANDS.md).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Side-effect import, and load-bearing: evals/_task_*.py register the log->rows bridges
# for the evals whose samples are not 1:1 with rows (mu_decisiveness, introspection_denial).
# `core.results` cannot import `evals/` (import-boundary rule #2), so a reader that skips
# this compares the Parquet against a per-sample reading of a bridged log and reports a
# mismatch that is its own fault. `rows_from_eval_log` now raises rather than guessing, so
# forgetting this is loud — but the fix is to import, here.
import serving.inspect_runner  # noqa: E402,F401
from core.results import SampleResult, eval_log_headers, load_all, log_rows_table  # noqa: E402

log = logging.getLogger("check_logs_match_parquet")

Cell = tuple[str, str]  # (spec_hash, eval_name)

#: `meta` keys only the bridge writes (`serving.runner._stamp`), so a log row lacks them.
STAMPS = frozenset({"runner", "max_workers", "gen_threads", "complete", "serving_route"})
COLUMNS = ("sample_id", "axis", "score", "raw_output", "meta")


def _is_stamp(key: str) -> bool:
    return key in STAMPS or key.startswith("icl_")


def _rows(table: pa.Table) -> dict[Cell, list[SampleResult]]:
    out: dict[Cell, list[SampleResult]] = defaultdict(list)
    for row in table.to_pylist():
        out[(row["spec_hash"], row["eval_name"])].append(SampleResult.from_row(row))
    return dict(out)


def _cells_from_parquet(root: Path) -> dict[Cell, list[SampleResult]]:
    return _rows(load_all(root, store="parquet"))


def _cells_from_logs(root: Path) -> dict[Cell, list[tuple[Path, list[SampleResult]]]]:
    """The FIRST successful log of every cell (sorted paths), read as the writer bridges it.

    The selection `main`'s checker always made, deliberately not the store's winner rule
    (`load_all` / `log_cell_tables`): a rule that keeps a cell on Parquet (a newer
    unstamped Parquet, a re-judge) would hide exactly the drift this check exists to
    report. Not every log either: a retried or reconfigured run leaves an older successful
    log that the current Parquet rightly disagrees with. A log that cannot be read is
    skipped by name, and the next one is tried.
    """
    out: dict[Cell, list[tuple[Path, list[SampleResult]]]] = {}
    for header in eval_log_headers(root):
        if not (header.spec_hash and header.eval_name and header.status == "success"):
            continue
        cell = (header.spec_hash, header.eval_name)
        if cell in out:
            continue
        try:
            table = log_rows_table(header)
        except Exception as exc:  # noqa: BLE001 - a foreign or corrupt log, named
            log.info("skipping %s: %s", header.path, exc)
            continue
        if table.num_rows:
            out[cell] = [(header.path, [SampleResult.from_row(r) for r in table.to_pylist()])]
    return out


def _serving(row: SampleResult) -> tuple[Any, Any, Any]:
    """The (runner, max_workers, gen_threads) this row was generated under.

    FLAT keys on `meta`, not a nested block — that is how `serving/runner._stamp`
    writes them (PR #136); a nested `meta["serving"]` read comes back empty on every
    real tree.
    """
    meta = row.meta or {}
    return (meta.get("runner"), meta.get("max_workers"), meta.get("gen_threads"))


def _report_concurrency(parquet_cells: dict[Cell, list[SampleResult]]) -> None:
    """Surface the serving concurrency each cell was generated at, and warn on drift.

    A batched engine is not batch-invariant: two trees — or two runs of one cell — made at
    different `max_workers`/`gen_threads` can differ token-for-token even at temperature 0.
    `results/` has no concurrency component in its cache key, so
    this can't be caught by path. It is printed here so a later byte-diff of this tree
    against another (native-vs-migrated parity) is trusted only at matched concurrency, and
    a WARNING fires on the one case detectable within a single tree: an eval generated at
    more than one width across its cells. Native cells use the eval's own width, not the
    batch knob, so they never trip the warning.
    """
    configs: Counter[tuple[Any, Any, Any]] = Counter()
    by_eval: dict[str, set[tuple[Any, Any]]] = defaultdict(set)
    for (_, eval_name), rows in parquet_cells.items():
        for runner, mw, gt in {_serving(r) for r in rows}:
            configs[(runner, mw, gt)] += 1
            if runner == "eval_set":  # native width is not the batched-generation hazard
                by_eval[eval_name].add((mw, gt))
    if not configs:
        return
    print("serving concurrency (runner, max_workers, gen_threads) -> cells:")
    for cfg, n in sorted(configs.items(), key=lambda kv: str(kv[0])):
        print(f"  {cfg} -> {n}")
    mixed = sorted(e for e, cfgs in by_eval.items() if len(cfgs) > 1)
    if mixed:
        print(
            "WARNING: eval(s) generated at mixed concurrency within this tree — their rows "
            "are not byte-comparable across cells: " + ", ".join(mixed)
        )
    print(
        "note: byte-diffing this tree against another is valid only where both were "
        "generated at the same concurrency (results/ has no concurrency in its cache key)"
    )


def _value(row: SampleResult, column: str) -> str:
    """One cell value, canonical: NaN is one value, and `meta` is compared without `STAMPS`."""
    if column == "score":
        return "nan" if math.isnan(row.score) else repr(row.score)
    if column == "meta":
        meta = {k: v for k, v in row.meta.items() if not _is_stamp(k)}
        return json.dumps(meta, sort_keys=True, default=str)
    return str(getattr(row, column))


def _digest(rows: list[SampleResult], column: str) -> str:
    """NaN-aware digest of one column, sorted: log rows are sample_id-sorted, Parquet is not."""
    items = sorted((r.sample_id, _value(r, column)) for r in rows)
    return hashlib.sha256(json.dumps(items).encode()).hexdigest()


def _meta_key_diffs(parquet: list[SampleResult], logged: list[SampleResult]) -> Counter[str]:
    """Which `meta` keys differ, row by row (stamps included, so they show as expected)."""
    out: Counter[str] = Counter()
    by_id = {r.sample_id: r.meta for r in logged}
    for p in parquet:
        lg = by_id.get(p.sample_id, {})
        for key in set(p.meta) | set(lg):
            if key not in lg:
                out[f"{key} (parquet only)"] += 1
            elif key not in p.meta:
                out[f"{key} (log only)"] += 1
            elif json.dumps(p.meta[key], sort_keys=True, default=str) != json.dumps(
                lg[key], sort_keys=True, default=str
            ):
                out[f"{key} (value differs)"] += 1
    return out


def _diff_cell(parquet: list[SampleResult], logged: list[SampleResult]) -> list[str]:
    """The columns whose sorted digests differ, each with its first differing sample."""
    problems: list[str] = []
    if len(parquet) != len(logged):
        problems.append(f"rows: {len(parquet)} (parquet) != {len(logged)} (log)")
    for column in COLUMNS:
        if _digest(parquet, column) == _digest(logged, column):
            continue
        p_vals = sorted((r.sample_id, _value(r, column)) for r in parquet)
        l_vals = sorted((r.sample_id, _value(r, column)) for r in logged)
        first = next(((a, b) for a, b in zip(p_vals, l_vals, strict=False) if a != b), None)
        where = f"{first[0][0]}: {first[0][1][:80]!r} != {first[1][1][:80]!r}" if first else ""
        problems.append(f"{column} differs ({where})")
    if not problems and [r.sample_id for r in parquet] != [r.sample_id for r in logged]:
        # Same rows, other order: the aggregates sum in row order, so a mean can move in
        # its last bit. The log store reads in dataset order to match the bridge.
        problems.append("order differs (same rows)")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_root", type=Path)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    parquet_cells = _cells_from_parquet(args.results_root)
    log_cells = _cells_from_logs(args.results_root)
    both = sorted(set(parquet_cells) & set(log_cells))
    print(
        f"cells: {len(parquet_cells)} parquet, {len(log_cells)} log, {len(both)} in both; "
        f"{len(set(parquet_cells) - set(log_cells))} parquet-only, "
        f"{len(set(log_cells) - set(parquet_cells))} log-only"
    )
    _report_concurrency(parquet_cells)
    if not both:
        # A check that checked nothing must not report OK — that is how this sat in the
        # tree gating nothing. The usual cause is the fake-mode log prune: `run_all.sh`
        # deletes a fake `.eval` log as soon as its rows are bridged, so a plain fake tree
        # has Parquet and no logs. Set MO_EVALS_KEEP_FAKE_LOGS=1 on the run to keep them.
        print(
            "FAILED: no overlapping cells — nothing was compared. A fake-mode tree keeps "
            "no .eval logs unless the run set MO_EVALS_KEEP_FAKE_LOGS=1; a real tree "
            "always keeps them."
        )
        return 1

    failures = 0
    compared = 0
    by_column: Counter[str] = Counter()
    meta_keys: Counter[str] = Counter()
    for cell in both:
        bad = False
        for path, logged in log_cells[cell]:
            compared += 1
            meta_keys.update(_meta_key_diffs(parquet_cells[cell], logged))
            problems = _diff_cell(parquet_cells[cell], logged)
            by_column.update(p.split(" ", 1)[0].rstrip(":") for p in problems)
            if problems:
                bad = True
                print(f"MISMATCH {cell[1]} x {cell[0]} ({path.name}):")
                for p in problems[:5]:
                    print(f"  {p}")
        failures += bad
    print(f"compared {compared} log(s) against the Parquet of {len(both)} cell(s)")
    print(
        "identical (cell, log) pairs per column: "
        + ", ".join(
            f"{c} {compared - by_column[c]}/{compared}" for c in ("rows", *COLUMNS, "order")
        )
    )
    if meta_keys:
        print("meta keys that differ, in rows (keys marked * are stamps, not compared):")
        for key, n in sorted(meta_keys.items()):
            print(f"  {'*' if _is_stamp(key.split(' ', 1)[0]) else ' '} {key}: {n}")
    if failures:
        print(f"FAILED: {failures} of {len(both)} overlapping cell(s) disagree")
        return 1
    print(f"OK: {len(both)} overlapping cell(s), log-derived rows match Parquet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
