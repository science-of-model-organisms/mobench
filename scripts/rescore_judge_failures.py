#!/usr/bin/env python
"""Re-judge rows whose judge CALL failed, from disk. No model, no GPU.

An OpenAI outage during a real sweep leaves rows written with
``meta["judge_raw"]`` starting ``"JUDGE_ERROR:"`` and ``score = NaN``
(``judges/judge.py``'s ``Judge.score`` catches ``JudgeBackendError`` and returns
``Verdict(outcome=None, raw=f"JUDGE_ERROR: {e}")`` rather than losing the whole batch —
``evals/_harness.py``'s ``_row``). ``raw_output`` is always stored (CLAUDE.md), so
these rows can be re-judged straight off disk with no regeneration: that is exactly what
``evals._harness.rescore_rows`` is for.

This only covers the **bank+judge** evals registered via
``evals._task_eval.register_judged_task`` (``robustness_*``, ``generality_*``, ...) — a
cell needs its bank/rubric/pair wiring to rebuild a ``Judge`` call, and those are the only
evals that declare it. Other judged cells (``persistence``, ``misaligned``,
...) are refused with a named reason rather than guessed at.

Only the rows that actually failed are re-scored; every other row in the cell is passed
through untouched. The cell is then rewritten in full — ``core.results.write_results``
writes one file per (spec_hash, eval_name) — stamped with the ``.eval`` log the cell's
rows came from where one is found, so ``core.results.load_all``'s identity rule (a Parquet
naming its source log, at that log's current size, IS that log's own bridged copy) prefers
this fixed Parquet over re-decoding the stale log next time.

    uv run python scripts/rescore_judge_failures.py results/ --dry-run
    uv run python scripts/rescore_judge_failures.py results/ --evals generality_fermi_gen
    uv run python scripts/rescore_judge_failures.py results/ pod2/results --config specs.yaml

``--dry-run`` lists the failing cells and calls no judge (same shape as
``run_all.sh --dry-run`` / ``inspect_runner.py --dry-run``: plan only, nothing spent).
Real judge calls cost tokens, same as any other re-score — and a judge change
invalidates stored scores (CONTRIBUTING.md, Non-negotiables).

``--all-rows`` re-judges EVERY row of the selected cells, not only the failed ones: the
path for a rubric change, which invalidates every stored verdict in the cells it judges
(e.g. the 2026-09-13 ``critique_recognition`` rewrite, docs/evals/robustness.md).
It requires ``--evals`` so a whole tree is never re-judged by accident::

    uv run python scripts/rescore_judge_failures.py results/ --all-rows \\
        --evals robustness_critique_provided --dry-run
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq  # noqa: E402

# Side-effect import, load-bearing: importing every evals._task_*.py module is what
# populates both the Inspect task registry (evals._inspect.get_eval) and the judged-eval
# wiring table (evals._task_eval._JUDGED) this script reads, and registers the log->rows
# bridges core.results.load_all needs to recover a cell that exists only as a `.eval` log.
import serving.inspect_runner  # noqa: E402,F401
from core.results import (  # noqa: E402
    SampleResult,
    _recorded_eval_config,
    atomic_write,
    iter_eval_logs,
    load_all,
    read_results,
    results_path,
    write_results,
)
from core.spec import OrganismSpec  # noqa: E402
from evals._harness import Item, rescore_rows  # noqa: E402
from evals._inspect import get_eval  # noqa: E402
from evals._task_eval import _JUDGED, _TRANSCRIPT_JUDGE_KIND, judged_items  # noqa: E402
from judges.judge import build_judge  # noqa: E402
from judges.personas import resolve_pair  # noqa: E402
from serving.runner import eval_config_for, load_config, load_dotenv, load_specs  # noqa: E402

log = logging.getLogger(__name__)

_ERROR_PREFIX = "JUDGE_ERROR"


def is_judge_error(row: SampleResult) -> bool:
    """True when a row's stored judge reply is a recorded backend failure, not a verdict.

    Both halves are required: the ``JUDGE_ERROR`` sentinel prefix AND a NaN score. A judge that
    replied successfully but whose text merely begins with that string still carries a real
    score, and re-judging it (the whole point of this tool) would overwrite a good verdict.
    """
    return str(row.meta.get("judge_raw", "")).startswith(_ERROR_PREFIX) and math.isnan(row.score)


def judged_wiring(eval_name: str, config: dict[str, Any]) -> tuple[list[Item], str | None]:
    """The bank items and rubric a judged eval's cell was built from.

    Mirrors the first few lines of ``evals._task_eval.judged_inputs`` — bank/items
    selection and rubric resolution — without building the Inspect-side judge or model
    roles, which ``rescore_rows`` never needed: it re-scores stored rows with no Inspect
    run in scope and talks to ``judges.backends`` directly (see
    ``evals._inspect.inspect_judge``'s docstring).

    Raises
    ------
    ValueError
        `eval_name` was never registered via ``register_judged_task`` — this tool only
        supports bank+judge evals, which are the only ones with this wiring.
    """
    wiring = _JUDGED.get(eval_name)
    if wiring is None:
        raise ValueError(
            f"{eval_name!r} is not a judged (bank+judge) eval registered via "
            "register_judged_task — rescore_judge_failures only supports those"
        )
    items = judged_items(wiring, config)
    rubric = (
        wiring.rubric
        if wiring.rubric is None or isinstance(wiring.rubric, str)
        else wiring.rubric(config)
    )
    # A transcript-judge eval's eval-wide rubric (debate's `debate_dialogue`) grades ONLY its
    # transcript arm, which this tool never re-judges (see `main`). Its other arms keep their
    # own bank rubric (`evals._task_eval._judged_rubric`), so drop the eval-wide one and let
    # `rescore_rows` fall to `item.rubric` — the same rule as `scripts/judge_deferred.py`.
    if eval_name in _TRANSCRIPT_JUDGE_KIND:
        rubric = None
    return items, rubric


def find_cell_log(root: Path, spec_hash: str, eval_name: str) -> Path | None:
    """The newest successful ``.eval`` log for one cell, or None if it has none.

    ponytail: picks the newest-mtime match rather than replaying
    ``core.results._tables_from_eval_logs``'s full supersession/identity logic — good
    enough to re-stamp a cell this script is about to rewrite wholesale. Upgrade if a
    rescore ever needs to pick correctly between several retries of one cell.
    """
    try:
        from inspect_ai.log import read_eval_log
    except ImportError:
        return None
    matches = []
    for path in iter_eval_logs(root):
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception:  # a truncated or foreign log must not sink the search
            continue
        meta = header.eval.metadata or {}
        if (
            meta.get("spec_hash") == spec_hash
            and meta.get("eval_name") == eval_name
            and header.status == "success"
        ):
            matches.append(path)
    return max(matches, key=lambda p: p.stat().st_mtime) if matches else None


def cell_file(root: Path | str, spec_hash: str, eval_name: str) -> Path | None:
    """The ONE Parquet on disk holding this cell, in either layout, or None if there is none.

    `core.results.results_path(spec, ...)` names the NEW nested file whether or not it
    exists, so a rewrite through it on a legacy flat tree (``<root>/<hash>/<eval>.parquet``)
    leaves the old file in place and adds a second one beside it — `load_all` reads both and
    double-counts the cell. This looks in all four places `load_all` can find it: nested or
    flat directory, seed-tagged or bare name. More than one is a tree that is already
    double-counted, and is refused rather than silently picking one.
    """
    root = Path(root)
    names = (f"seed*__{spec_hash}__{eval_name}.parquet", f"{eval_name}.parquet")
    found = sorted(
        {p for d in (f"*/{spec_hash}", spec_hash) for n in names for p in root.glob(f"{d}/{n}")}
    )
    if len(found) > 1:
        raise ValueError(
            f"cell {spec_hash} x {eval_name} has {len(found)} Parquet files under {root} "
            f"({[str(p) for p in found]}): load_all already double-counts it; fix the tree first"
        )
    return found[0] if found else None


def rewrite_cell(
    rows: list[SampleResult],
    root: Path,
    *,
    spec: OrganismSpec,
    eval_name: str,
    source_log: Path | None,
) -> Path:
    """Rewrite a cell IN PLACE: the file it was read from, keeping that file's config stamp.

    The shared write path for every re-score/re-judge script. A cell with no file yet, or
    already at the canonical nested path, goes straight through `write_results` (which
    carries the stamp of an existing destination itself). A LEGACY file cannot be addressed
    by `write_results(spec=...)` — it always targets the nested path — so the rows are
    written to a scratch root with the legacy file's stamp passed explicitly, then moved
    over the legacy file with `core.results.atomic_write`: a unique temp name in the
    target's own directory, removed on any failure, `os.replace`d over it on success —
    the same pattern `write_results` itself uses, reused rather than duplicated (a fixed
    staging name here could survive a failed rename, and let two writers of the same cell
    collide on one path). No core change: `core/` contracts change only deliberately (CLAUDE.md).
    """
    existing = cell_file(root, spec.spec_hash, eval_name)
    if existing is None or existing == results_path(spec, eval_name, root):
        return write_results(
            rows, root, spec=spec, eval_name=eval_name, source_log=source_log, rejudged=True
        )
    stamp = _recorded_eval_config(existing)
    with tempfile.TemporaryDirectory() as scratch:
        written = write_results(
            rows,
            scratch,
            spec_hash=spec.spec_hash,
            eval_name=eval_name,
            source_log=source_log,
            rejudged=True,  # not the log's rows any more: MO_EVALS_STORE=log keeps the Parquet
            # Round-trips: the stamp is `canonical_eval_config`'s sorted JSON already.
            eval_config=json.loads(stamp) if stamp is not None else None,
        )
        with atomic_write(existing) as tmp:
            shutil.copyfile(written, tmp)
    log.info("rewrote %d rows in place at %s (legacy layout)", len(rows), existing)
    return existing


def source_log_for(
    root: Path, spec: OrganismSpec, eval_name: str, cell_rows: list[SampleResult]
) -> Path | None:
    """The `.eval` log the rows being rescored came from, to stamp the rewrite with — or None.

    `load_all` serves a cell from its Parquet, or from a log that supersedes it, and the
    stamp decides which it serves next time. So stamp the log the rows actually came from:

    - The Parquet on disk IS what was served: its own stamped source, if that log still
      exists; None if it was pruned. A surviving older retry is not the source, and an
      unstamped rewrite still wins its cell on mtime.
    - A log served the cell: the newest successful one (`find_cell_log`), which is the log
      `load_all` prefers. Re-stamping the Parquet's OLD source here instead would hand the
      cell straight back to that newer log on the next load and undo the rescore.
    """
    parquet = cell_file(root, spec.spec_hash, eval_name)
    if parquet is not None:
        on_disk = {(r.sample_id, r.raw_output) for r in read_results(parquet)}
        if on_disk == {(r.sample_id, r.raw_output) for r in cell_rows}:
            name = (pq.read_schema(parquet).metadata or {}).get(b"source_log_name")
            if not name:
                return None
            return next((p for p in iter_eval_logs(root) if p.name == name.decode()), None)
    return find_cell_log(root, spec.spec_hash, eval_name)


def rescore_cell(
    root: Path,
    spec: OrganismSpec,
    eval_name: str,
    axis: str,
    cell_rows: list[SampleResult],
    failed: list[SampleResult],
    eval_cfg: dict[str, Any],
) -> tuple[int, int]:
    """Re-judge one cell's `failed` rows (every row under ``--all-rows``) and write it back.

    Every row not in `failed` is written back unchanged. A rescored row keeps its
    original `meta` (the runner-stamped `runner` / `max_workers` / `gen_threads`,
    `serving_route`, etc.) with the new judge fields (`outcome`, `coherence`,
    `judge_raw`, `rescored_from`, ...) merged over it.

    Returns
    -------
    tuple[int, int]
        `(n_rescored, n_still_failed)` — rows that now carry a real judge reply vs.
        rows still showing a `JUDGE_ERROR` (a repeat failure, or one `rescore_rows`
        could not match to a bank item at all).
    """
    items, rubric = judged_wiring(eval_name, eval_cfg)
    pair = resolve_pair(eval_cfg)
    judge = build_judge(eval_cfg)
    fixed = rescore_rows(
        failed,
        eval_cfg,
        eval_name=eval_name,
        axis=axis,
        items=items,
        judge=judge,
        pair=pair,
        rubric=rubric,
    )
    fixed_by_id = {r.sample_id: r for r in fixed}
    n_rescored = sum(1 for r in fixed if not is_judge_error(r))
    n_still_failed = len(failed) - n_rescored

    def merged(original: SampleResult) -> SampleResult:
        new = fixed_by_id.get(original.sample_id)
        if new is None:
            return original
        return dataclasses.replace(new, meta={**original.meta, **new.meta})

    new_rows = [merged(r) for r in cell_rows]
    source_log = source_log_for(root, spec, eval_name, cell_rows)
    rewrite_cell(new_rows, root, spec=spec, eval_name=eval_name, source_log=source_log)
    return n_rescored, n_still_failed


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("results_root", type=Path, nargs="+", help="one or more results trees")
    parser.add_argument("--config", type=Path, default=Path("configs/specs.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="list failing cells, judge nothing")
    parser.add_argument(
        "--evals",
        nargs="+",
        action="extend",
        default=None,
        help="restrict to these eval names; repeatable (values accumulate)",
    )
    parser.add_argument(
        "--spec-hash",
        nargs="+",
        action="extend",
        default=None,
        help="restrict to these spec hashes; repeatable (values accumulate)",
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="re-judge every row of the selected cells (after a rubric change); needs --evals",
    )
    args = parser.parse_args(argv)
    if args.all_rows and not args.evals:
        parser.error("--all-rows re-judges whole cells; name them with --evals")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_dotenv()  # the judge key lives in the gitignored .env on a judge pod

    yaml_config = load_config(args.config)
    specs_by_hash = {s.spec_hash: s for s in load_specs(yaml_config)}

    any_cells = False
    for root in args.results_root:
        table = load_all(root, store="both")  # re-judges ignore MO_EVALS_STORE until stage 3
        rows = [SampleResult.from_row(r) for r in table.to_pylist()]
        by_cell: dict[tuple[str, str], list[SampleResult]] = defaultdict(list)
        for row in rows:
            by_cell[(row.spec_hash, row.eval_name)].append(row)

        for (spec_hash, eval_name), cell_rows in sorted(by_cell.items()):
            if args.evals and eval_name not in args.evals:
                continue
            if args.spec_hash and spec_hash not in args.spec_hash:
                continue
            failed = cell_rows if args.all_rows else [r for r in cell_rows if is_judge_error(r)]
            if not failed:
                continue
            any_cells = True
            tag = f"{root} {spec_hash[:12]} x {eval_name}"
            what = "rows to re-judge" if args.all_rows else "failed"

            spec = specs_by_hash.get(spec_hash)
            if spec is None:
                where = args.config
                print(f"{tag}: {len(failed)} {what} — SKIPPED: no spec {spec_hash!r} in {where}")
                continue
            try:
                ev = get_eval(eval_name)
            except KeyError as exc:
                print(f"{tag}: {len(failed)} {what} — SKIPPED: {exc}")
                continue
            if eval_name not in _JUDGED:
                print(f"{tag}: {len(failed)} {what} — SKIPPED: not a bank+judge eval")
                continue
            # A debate row's score is its final turn's own read (`_PER_TURN_JUDGE_EVALS`, keyed
            # `|turn<k>`), which `rescore_rows` does not reproduce, so it would score the wrong
            # thing. Refuse THOSE ROWS, not the cell: the same cell's single-turn arms (debate's
            # `control`) are ordinary rows and are re-judged with their own bank rubric
            # (`judged_wiring` drops the eval-wide one). Refusing the whole cell left a failed
            # control failed forever (Codex review of PR 5a, item 7). A critique `followup` row
            # is NOT refused: since 2026-10-03 `rescore_rows` rebuilds its rendered exchange
            # from the stored transcript (`evals._task_eval.judged_response_from_row`).
            if _TRANSCRIPT_JUDGE_KIND.get(eval_name) == "debate":
                refused = [r for r in failed if r.meta.get("transcript")]
                if refused:
                    print(
                        f"{tag}: {len(refused)} transcript-judged row(s) refused; "
                        "rescore_rows does not reproduce a debate row's per-turn score"
                    )
                    failed = [r for r in failed if not r.meta.get("transcript")]
                    if not failed:
                        continue

            if args.dry_run:
                print(f"{tag}: {len(failed)} {what} (dry run — not rescored)")
                continue

            eval_cfg = eval_config_for(yaml_config, spec, ev)
            n_rescored, n_still_failed = rescore_cell(
                root, spec, eval_name, ev.axis, cell_rows, failed, eval_cfg
            )
            status = f"{n_rescored} rescored, {n_still_failed} still failed"
            print(f"{tag}: {len(failed)} {what}, {status}")

    if not any_cells:
        print("no rows in the selected cells" if args.all_rows else "no JUDGE_ERROR rows found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
