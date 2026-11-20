#!/usr/bin/env python3
"""Phase B of defer-judge: judge the generation-only cells produced by ``--phase generate``.

The big run splits generation (GPU) from judging (a cheap CPU pod + OpenRouter) so the GPU
never idles on a judge call. ``--phase
generate`` writes each *completion-judged* sample to a staging tree with ``score=NaN`` and
``meta['defer_judge']=True`` (``evals._task_eval._judge_scorer``); this script reads that
staging tree, fills every deferred verdict with ``evals._harness.rescore_rows`` — the exact,
already-proven re-score path ``raw_output`` exists for (CLAUDE.md) — and writes the
finished cell to the MAIN results tree.

    # source = the generation staging tree; target = the run's main tree
    uv run python scripts/judge_deferred.py <generation_root> --target-root <main_root>
    # same, through the OpenAI Batch API (50% price, <=24h turnaround, resumable):
    uv run python scripts/judge_deferred.py <generation_root> --target-root <main_root> --batch

``--batch`` (judges/batch.py) runs the judge twice over the deferred rows: a collect pass
that records every request body without calling anything, then — after the batch(es)
finish — the normal judge pass with each reply looked up by ``custom_id``. So the parser,
meta and written cells are exactly the synchronous path's (``judge_service_tier`` reads
``batch``). Requests the batch did not answer are retried synchronously on Standard, never
put into a new batch. Polling is bounded (``judge.batch.poll_timeout_s``); on timeout the
script exits 3 and writes nothing, and re-running the same command re-attaches to the
persisted batches (``<target_root>/_judge_batch/state.json``, which freezes each batch's
requests) and never resubmits a request already in one (bar the RESIDUAL WINDOW in
``judges/batch.py``: a crash between ``batches.create`` and recording its id).

Parity: because generation is byte-identical with or without deferral, and Phase B applies
the SAME judge (built from the same ``configs/specs.yaml``) to the SAME ``raw_output`` the
inline scorer would have judged, the finished cell is identical to an inline run. That is
asserted end-to-end in ``tests/test_defer_judge.py``.

What is NOT deferred, and so is already scored in the staging cell and passes through
untouched: judge-free evals, transcript-judged arms (``robustness_critique_gen`` followup /
``robustness_multiturn_debate_gen`` — their ``raw_output`` is not the judged text), and the
packed/adaptive evals (drift, petri) that never use ``_judge_scorer``. A cell may mix
deferred and inline rows; only the deferred rows are re-judged, the rest are written back
verbatim.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Side-effect import, load-bearing: importing every evals._task_*.py module populates the
# Inspect task registry (evals._inspect.get_eval), the judged-eval wiring table
# (evals._task_eval._JUDGED), and the log->rows bridges core.results.load_all needs.
import serving.inspect_runner  # noqa: E402,F401
from core.results import (  # noqa: E402
    SampleResult,
    iter_eval_logs,
    iter_result_files,
    load_all,
    write_results,
)
from core.spec import OrganismSpec  # noqa: E402
from evals._harness import Item, rescore_rows, row_is_complete  # noqa: E402
from evals._inspect import get_eval  # noqa: E402
from evals._task_eval import _JUDGED, _TRANSCRIPT_JUDGE_KIND, judged_items  # noqa: E402
from judges import batch as judge_batch  # noqa: E402
from judges.backends import HttpJudge, JudgeBackend, build_backend, pooled  # noqa: E402
from judges.judge import build_judge  # noqa: E402
from judges.personas import resolve_pair  # noqa: E402
from judges.rubrics import raw_is_judge_error  # noqa: E402
from serving.runner import (  # noqa: E402
    _force_fake_backends,
    eval_config_for,
    expand_seeds,
    fake_mode,
    load_config,
    load_dotenv,
    load_specs,
)

log = logging.getLogger(__name__)


def is_deferred(row: SampleResult) -> bool:
    """A row awaiting its verdict from this script.

    The exact negation of ``evals._harness.row_is_complete`` (the ``meta['complete']`` flag):
    a row is deferred iff it is not yet complete — written by ``_judge_scorer`` under
    ``MO_EVALS_DEFER_JUDGE`` with ``meta['defer_judge']`` True and its verdict (``outcome``)
    still unfilled. Sharing the helper keeps the deferral predicate and the stored completeness
    flag from drifting apart, and a row this script has already judged (its ``meta`` may still
    carry the marker after a merge) has ``outcome`` set and so is not re-judged.
    """
    return not row_is_complete(row.meta)


def judged_wiring(eval_name: str, config: dict[str, Any]) -> tuple[list[Item], str | None]:
    """The bank items and rubric a judged eval's cell was built from.

    Mirrors the first lines of ``evals._task_eval.judged_inputs`` (bank/items + rubric),
    without the Inspect-side judge/roles ``rescore_rows`` never needs — identical to the
    helper in ``scripts/rescore_judge_failures.py``.
    """
    wiring = _JUDGED.get(eval_name)
    if wiring is None:
        raise ValueError(f"{eval_name!r} is not a bank+judge eval (register_judged_task)")
    items = judged_items(wiring, config)
    rubric = (
        wiring.rubric
        if wiring.rubric is None or isinstance(wiring.rubric, str)
        else wiring.rubric(config)
    )
    # Mirror evals._task_eval._judged_rubric for the DEFERRED rows. A transcript-judge eval's
    # eval-wide rubric (debate's `debate_dialogue`) grades ONLY its transcript (pressure) arm,
    # which is judged INLINE and never deferred (its `raw_output` is the completion, not the
    # transcript the judge scored). Every row that reaches Phase B is therefore a non-transcript
    # arm (`response == final`) — e.g. debate's single-turn `control` (`kind == "none"`) — which
    # inline grades with its OWN bank rubric (`item.rubric`, i.e. `open_ended`), not the dialogue
    # grader. Passing the eval-wide rubric here would short-circuit `item.rubric` in
    # `rescore_rows` (`rubric or item.rubric`, evals/_harness.py) and re-grade the control arm
    # with `debate_dialogue`, moving the baseline independently of the pressure and reintroducing
    # the confound `_judged_rubric` fixed (Codex review DEFECT 2, 2026-09-14). So drop the
    # eval-wide rubric for these evals and let each deferred row keep its per-item rubric.
    if eval_name in _TRANSCRIPT_JUDGE_KIND:
        rubric = None
    return items, rubric


def _source_log(gen_root: Path, spec_hash: str, eval_name: str) -> Path | None:
    """The `.eval` log this cell's generation came from, to stamp the written cell with.

    ``load_all``'s identity rule (a Parquet supersedes its source log by name+mtime) means a
    cell written with no stamp can be handed back to its own generation log on the next load.
    Best-effort match by the seeded filename prefix, same shape as
    ``rescore_judge_failures.source_log_for``.
    """
    for path in iter_eval_logs(gen_root):
        if spec_hash in path.name and eval_name in path.name:
            return path
    return None


def _source_parquets(gen_root: Path) -> dict[tuple[str, str], list[Path]]:
    """Map each staged cell ``(spec_hash, eval_name)`` to its Parquet file(s) under ``gen_root``.

    Phase B is a MOVE, not a copy: once a staged cell has been written to the main tree, its
    staging Parquet must be removed, or the scorecard double-counts it. ``analysis`` reads a
    results root with ``core.results.iter_result_files``, which globs ``*.parquet`` recursively
    and excludes only ``logs/`` — NOT a nested ``generation/`` staging tree (as
    ``run_all.sh --phase judge`` produces, ``$MAIN_ROOT/generation`` inside ``$MAIN_ROOT``). A
    surviving staging copy of every passthrough row (judge-free evals, transcript arms, drift,
    petri) would then be counted twice — n doubles, standard errors shrink — and each deferred
    cell's stale NaN staging copy would inflate the discard fraction. So the finished cell is
    dropped from staging in ``main`` (see the drain step there). One file is one cell
    (``write_results``), so the first row's key identifies the whole file.
    """
    out: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in iter_result_files(gen_root):
        head = pq.read_table(path, columns=["spec_hash", "eval_name"])
        if head.num_rows:
            key = (head["spec_hash"][0].as_py(), head["eval_name"][0].as_py())
            out[key].append(path)
    return out


class JudgeCellFailed(RuntimeError):
    """Every deferred row in a cell came back as a judge backend failure."""


def judge_cell(
    gen_root: Path,
    target_root: Path,
    spec: OrganismSpec,
    eval_name: str,
    axis: str,
    cell_rows: list[SampleResult],
    eval_cfg: dict[str, Any],
    *,
    backend: JudgeBackend | None = None,
    write: bool = True,
) -> tuple[int, int]:
    """Judge one staging cell's deferred rows and write the finished cell to ``target_root``.

    Non-deferred rows (transcript arms judged inline in Phase A) pass through verbatim.
    Returns ``(n_judged, n_passthrough)``.

    ``backend`` replaces the configured judge's transport (the batch recorder / replay);
    the ``Judge`` around it — seed, rubric, parser — is built from config exactly as
    before. ``write=False`` judges without writing (the batch collect pass).
    """
    deferred = [r for r in cell_rows if is_deferred(r)]
    passthrough = len(cell_rows) - len(deferred)
    if not write:
        if deferred:
            _rescore(deferred, eval_name, axis, eval_cfg, backend)
        return len(deferred), passthrough
    if not deferred:
        # A fully-complete cell (judge-free, or transcript/petri/drift judged inline in
        # Phase A). Move it to the main tree verbatim so `generation/ -> main` is a COMPLETE
        # transform — otherwise these cells would be stranded in the staging tree.
        source_log = _source_log(gen_root, spec.spec_hash, eval_name)
        write_results(
            cell_rows, target_root, spec=spec, eval_name=eval_name, source_log=source_log
        )
        return 0, passthrough
    scored = _rescore(deferred, eval_name, axis, eval_cfg, backend)
    # Every judge call failing is an infrastructure failure (no key, dead endpoint), not a
    # measurement: refuse to write the cell, so its staging copy stays deferred and a re-run
    # retries it. A SPORADIC failure stays a terminal discard, as `row_is_complete` intends.
    # (2026-10-02: a pod with the key only in .env "judged" 900 rows to NaN and exited 0.)
    errors = sum(raw_is_judge_error(r.meta.get("judge_raw")) for r in scored)
    if scored and errors == len(scored):
        raise JudgeCellFailed(
            f"all {errors} judge calls failed; first: {scored[0].meta.get('judge_raw')!r:.200}"
        )
    scored_by_id = {r.sample_id: r for r in scored}
    # Keep every original row; replace the deferred ones with their judged version. The
    # judged row's meta already carries the original serving meta forward via rescore_rows.
    new_rows = [scored_by_id.get(r.sample_id, r) for r in cell_rows]
    source_log = _source_log(gen_root, spec.spec_hash, eval_name)
    write_results(
        new_rows,
        target_root,
        spec=spec,
        eval_name=eval_name,
        source_log=source_log,
        rejudged=True,  # the verdicts are not in the log: MO_EVALS_STORE=log keeps the Parquet
    )
    return len(scored), passthrough


def _rescore(
    deferred: list[SampleResult],
    eval_name: str,
    axis: str,
    eval_cfg: dict[str, Any],
    backend: JudgeBackend | None,
) -> list[SampleResult]:
    """Judge ``deferred`` with the configured judge (optionally on another transport)."""
    items, rubric = judged_wiring(eval_name, eval_cfg)
    pair = resolve_pair(eval_cfg)
    judge = build_judge(eval_cfg)
    if backend is not None:
        judge.backend = backend
    # Cap judge concurrency to the JUDGE width, not the generation width. `rescore_rows`
    # fans out at `config['max_workers']` (up to 256 — sized for the local vLLM coalescer),
    # but here every one of those is an OpenRouter judge call. The inline path bounds judge
    # calls with the judge's own limiter (`judge_max_connections`, default 64); rescore_rows
    # does not, so without this Phase B fires ~256-wide at the API and 429s (observed
    # 2026-09-17). Narrow the width to the judge budget for the re-score.
    judge_width = int(eval_cfg.get("judge_max_connections", 64))
    capped = min(int(eval_cfg.get("max_workers", judge_width)), judge_width)
    eval_cfg = {**eval_cfg, "max_workers": capped}
    return rescore_rows(
        deferred,
        eval_cfg,
        eval_name=eval_name,
        axis=axis,
        items=items,
        judge=judge,
        pair=pair,
        rubric=rubric,
    )


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("generation_root", type=Path, help="the --phase generate staging tree")
    p.add_argument(
        "--target-root",
        type=Path,
        required=True,
        help="the run's MAIN results tree (finished, judged cells land here)",
    )
    p.add_argument("--config", type=Path, default=Path("configs/specs.yaml"))
    p.add_argument("--dry-run", action="store_true", help="list deferred cells, judge nothing")
    p.add_argument(
        "--evals", nargs="+", action="extend", default=None, help="restrict to these eval names"
    )
    # Judge override — for a judge swap / bake-off without editing the config. Replaces the
    # judge block for EVERY cell; the archived generations (raw_output) are judged with the
    # named model. E.g. --judge-model google/gemma-4-31b-it --judge-provider friendli
    # --judge-max-tokens 512. Non-reasoning gemma @ friendli is ~25x faster than glm
    # The override is an
    # OpenRouter block, so it cannot be combined with --batch (refused in `main`).
    p.add_argument("--judge-model", default=None, help="override the judge model (OpenRouter id)")
    p.add_argument("--judge-provider", default=None, help="pin the judge provider (e.g. friendli)")
    p.add_argument("--judge-max-tokens", type=int, default=None, help="cap judge output tokens")
    p.add_argument(
        "--batch",
        action="store_true",
        help=(
            "judge through the OpenAI Batch API (judge.batch in the config); resumable. "
            "Refused (exit 2, before any row is read) unless every in-scope cell's judge "
            "block is the same OpenAI-direct block (https://api.openai.com) and not fake mode"
        ),
    )
    p.add_argument(
        "--batch-state-dir",
        type=Path,
        default=None,
        help="where batch ids/inputs/outputs persist (default <target_root>/_judge_batch)",
    )
    return p.parse_args(argv)


def _apply_judge_override(
    config: dict[str, Any], model: str, provider: str | None, max_tokens: int | None
) -> None:
    """Replace the judge block so `build_judge` uses the named model/provider for every cell.

    A judge SWAP (not a rubric change): it re-scores the archived generations with a different
    judge, which is exactly what the never-delete-raw_output rule enables. Provider is pinned
    (`only:`, fallbacks off) for reproducibility, same discipline as the config's own judge.
    """
    params: dict[str, Any] = {}
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if provider:
        params["provider"] = {"only": [provider], "allow_fallbacks": False}
    block = {
        "backend": "http",
        "model": model,
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "params": params,
    }
    evals = config.setdefault("eval_configs", {})
    evals.setdefault("default", {})["judge"] = block
    # A per-eval `judge` block SHADOWS the default one (eval_config_for merges shallowly), so
    # the swap above would be silently ignored there. Where that block is a PANEL (has a
    # `secondary`, today only simple_behaviour_exhibit: GLM presence + Gemma coherence gate),
    # swap the PRIMARY and keep the secondary: the override names the presence judge, the
    # coherence gate stays Gemma. Single-judge per-eval blocks are left as they are.
    for name, cfg in evals.items():
        judge = cfg.get("judge") if name != "default" and isinstance(cfg, dict) else None
        if isinstance(judge, dict) and judge.get("secondary") is not None:
            cfg["judge"] = {**block, "secondary": judge["secondary"]}


def _eligible_cells(
    args: argparse.Namespace,
    yaml_config: dict[str, Any],
    specs_by_hash: dict[str, OrganismSpec],
    by_cell: dict[tuple[str, str], list[SampleResult]],
) -> list[tuple[OrganismSpec, str, str, list[SampleResult], dict[str, Any]]]:
    """Cells with deferred rows that the main loop would judge: (spec, eval, axis, rows, cfg)."""
    out = []
    for (spec_hash, eval_name), cell_rows in sorted(by_cell.items()):
        if args.evals and eval_name not in args.evals:
            continue
        spec = specs_by_hash.get(spec_hash)
        if spec is None or not any(is_deferred(r) for r in cell_rows):
            continue
        try:
            ev = get_eval(eval_name)
        except KeyError:
            continue
        out.append((spec, eval_name, ev.axis, cell_rows, eval_config_for(yaml_config, spec, ev)))
    return out


def _single_judge_block(blocks: list[Any]) -> dict[str, Any]:
    """The ONE OpenAI-direct judge block ``blocks`` all equal (a batch file is single-model)."""
    distinct = {json.dumps(b, sort_keys=True, default=str): b for b in blocks}
    for block in distinct.values():  # the specific reason (OpenRouter, secondary...) first
        judge_batch.require_openai_direct(block)
    if len(distinct) != 1:
        raise judge_batch.BatchUnsupported(
            f"--batch needs one judge block across all cells; found {len(distinct)}"
        )
    return judge_batch.require_openai_direct(blocks[0])


def _judge_block(cells: list[tuple[Any, ...]]) -> dict[str, Any]:
    """The ONE judge block every batched cell shares, checked against the rows read."""
    return _single_judge_block([c[4].get("judge") for c in cells])


def _config_judge_block(
    args: argparse.Namespace, yaml_config: dict[str, Any], specs: list[OrganismSpec]
) -> dict[str, Any] | None:
    """The judge block every cell ``--batch`` could judge resolves to, from config ALONE.

    Only a bank+judge eval (``_JUDGED``) writes deferred rows, and a cell's judge block is a
    function of the config, its eval and its spec (``eval_config_for``: per-eval blocks and
    ``by_base_model`` overrides). So every (spec, judged eval) pair in scope is resolved here,
    before any row is read, and a per-eval override (non-OpenAI, ``secondary``, or merely
    different) is refused up front. A superset of the cells actually staged: an override on
    an eval with no staged rows is refused too, which is the conservative side. ``None``
    when no judged eval is in scope (nothing to batch).
    """
    blocks = []
    for name in sorted(_JUDGED):
        if args.evals and name not in args.evals:
            continue
        try:
            ev = get_eval(name)
        except KeyError:
            continue
        blocks += [eval_config_for(yaml_config, spec, ev).get("judge") for spec in specs]
    return _single_judge_block(blocks) if blocks else None


def prepare_batch(
    args: argparse.Namespace,
    yaml_config: dict[str, Any],
    specs_by_hash: dict[str, OrganismSpec],
    by_cell: dict[tuple[str, str], list[SampleResult]],
    *,
    client: Any = None,
) -> judge_batch.BatchReplay | None:
    """Collect every deferred request, run it through the Batch API, return the replay backend.

    Raises ``judges.batch.BatchPending`` if polling times out (nothing has been written).
    """
    cells = _eligible_cells(args, yaml_config, specs_by_hash, by_cell)
    if not cells:
        return None
    block = _judge_block(cells)
    recorder = judge_batch.BatchRecorder(block)
    # The collect pass's verdicts are placeholders; silence their per-row judge logging.
    judge_log = logging.getLogger("judges.judge")
    level = judge_log.level
    judge_log.setLevel(logging.ERROR)
    try:
        for spec, eval_name, axis, cell_rows, eval_cfg in cells:
            judge_cell(
                args.generation_root,
                args.target_root,
                spec,
                eval_name,
                axis,
                cell_rows,
                eval_cfg,
                backend=recorder,
                write=False,
            )
    finally:
        judge_log.setLevel(level)
    settings = judge_batch.BatchSettings.from_block(block.get("batch"))
    state_dir = args.batch_state_dir or (args.target_root / "_judge_batch")
    log.info("BATCH: %d distinct judge requests from %d cells", len(recorder.requests), len(cells))
    outcome = judge_batch.run_batches(
        recorder.requests,
        state_dir,
        client if client is not None else judge_batch.make_client(block),
        settings,
    )
    log.info(
        "BATCH: %d replies, %d failed/expired (retried synchronously on Standard), batches %s, "
        "usage %s",
        len(outcome.results),
        len(outcome.failures),
        outcome.batch_ids,
        outcome.usage,
    )
    fallback = build_backend(pooled(block, {"max_workers": 64}))
    if not isinstance(fallback, HttpJudge):  # `_judge_block` checked backend: http
        raise TypeError(f"batch fallback must be an HttpJudge, got {type(fallback).__name__}")
    return judge_batch.BatchReplay(block, outcome.results, fallback=fallback)


def main(argv: list[str] | None = None, *, batch_client: Any = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # The judge key lives in the gitignored .env on a Phase B pod, as for the inline runner
    # (serving/inspect_runner.py calls the same loader); exported variables still win.
    load_dotenv()

    yaml_config = load_config(args.config)
    # Honour fake mode exactly as the inline runner does (serving/runner.py): drop the real
    # judge/adversary blocks so build_judge falls back to FakeJudge. Without this the gate's
    # fake `--phase judge` would make real OpenRouter calls (and 429). In a REAL run this is a
    # no-op, so the CPU judge pod still uses the real panel.
    if fake_mode():
        _force_fake_backends(yaml_config)
    elif args.judge_model:
        _apply_judge_override(
            yaml_config, args.judge_model, args.judge_provider, args.judge_max_tokens
        )
        log.info(
            "JUDGE OVERRIDE: %s provider=%s max_tokens=%s",
            args.judge_model,
            args.judge_provider,
            args.judge_max_tokens,
        )
    # MUST expand seeds: a 3-seed behavioural eval writes cells under the seed-1/seed-2
    # spec_hashes, which `load_specs` (the distinct, seed-0 organism list) does not contain.
    # The runner does the same (serving/inspect_runner.py builds its by_hash from
    # expand_seeds(...)); without this, every non-seed-0 deferred cell is left unjudged.
    specs_by_hash = {s.spec_hash: s for s in expand_seeds(load_specs(yaml_config), yaml_config)}
    if args.batch:
        # Refuse BEFORE reading any rows: the OpenAI Batch API exists only on the OpenAI API
        # itself, and the shared judge (OpenRouter) and fake mode (no network at all) have
        # none. Never a silent fall-back to synchronous judging. Every in-scope cell's judge
        # block — the default AND any per-eval override — is resolved from config here;
        # `_judge_block` re-checks the cells actually read.
        if fake_mode():
            log.error(
                "--batch REFUSED: fake mode (MO_EVALS_FAKE=1) has no Batch API and makes no "
                "network calls; drop --batch"
            )
            return 2
        try:
            _config_judge_block(args, yaml_config, list(specs_by_hash.values()))
        except judge_batch.BatchUnsupported as exc:
            log.error("--batch REFUSED: %s", exc)
            return 2

    table = load_all(args.generation_root, store="both")  # MO_EVALS_STORE: stage 3
    rows = [SampleResult.from_row(r) for r in table.to_pylist()]
    by_cell: dict[tuple[str, str], list[SampleResult]] = defaultdict(list)
    for row in rows:
        by_cell[(row.spec_hash, row.eval_name)].append(row)

    replay: judge_batch.BatchReplay | None = None
    if args.batch and not args.dry_run:
        try:
            replay = prepare_batch(args, yaml_config, specs_by_hash, by_cell, client=batch_client)
        except judge_batch.BatchUnsupported as exc:
            log.error("--batch REFUSED — nothing judged or written: %s", exc)
            return 2
        except judge_batch.BatchPending as exc:
            log.warning("BATCH PENDING — nothing written: %s", exc)
            return 3

    # `generation/ -> main` is a MOVE: each finished cell is dropped from the staging tree so a
    # results root that CONTAINS the staging tree (run_all.sh --phase judge stages at
    # `$MAIN_ROOT/generation`) is not double-counted by the scorecard (`iter_result_files`
    # excludes only `logs/`). Guarded on distinct roots so an in-place judge never deletes its
    # own output; skipped in a dry run and when generation IS the target.
    drain = not args.dry_run and args.generation_root.resolve() != args.target_root.resolve()
    src_files = _source_parquets(args.generation_root) if drain else {}

    n_cells = n_deferred_cells = n_judged = n_outstanding = 0
    for (spec_hash, eval_name), cell_rows in sorted(by_cell.items()):
        if args.evals and eval_name not in args.evals:
            continue
        # Every staged cell is moved to the main tree — deferred cells are judged, complete
        # cells (judge-free / inline-judged transcript arms) pass through verbatim — so the
        # `generation/ -> main` transform leaves nothing stranded in the staging tree.
        n_cells += 1
        has_deferred = any(is_deferred(r) for r in cell_rows)
        n_deferred_cells += int(has_deferred)
        tag = f"{spec_hash[:12]} x {eval_name}"
        spec = specs_by_hash.get(spec_hash)
        if spec is None:
            log.warning("%s: SKIPPED — no spec %r in %s", tag, spec_hash, args.config)
            n_outstanding += int(has_deferred)  # only an UNJUDGED cell left behind is a gap
            continue
        try:
            ev = get_eval(eval_name)
        except KeyError as exc:
            log.warning("%s: SKIPPED — %s", tag, exc)
            n_outstanding += int(has_deferred)
            continue
        if args.dry_run:
            log.info("%s: %d deferred rows (dry run)", tag, sum(is_deferred(r) for r in cell_rows))
            continue
        eval_cfg = eval_config_for(yaml_config, spec, ev)
        try:
            judged, passthrough = judge_cell(
                args.generation_root,
                args.target_root,
                spec,
                eval_name,
                ev.axis,
                cell_rows,
                eval_cfg,
                backend=replay,
            )
        except JudgeCellFailed as exc:
            log.error("%s: NOT WRITTEN, left staged for a re-run — %s", tag, exc)
            n_outstanding += 1
            continue
        n_judged += judged
        if judged or passthrough:
            log.info("%s: %d judged, %d passthrough", tag, judged, passthrough)
        # The cell now lives in the main tree; remove its staging copy so it is not counted
        # twice. Only the staged Parquet is removed — the `.eval` log under `logs/` (the
        # raw_output archive) is left in place and is excluded from the scorecard anyway.
        if drain:
            for src in src_files.get((spec_hash, eval_name), []):
                src.unlink(missing_ok=True)

    staged_left = len(_source_parquets(args.generation_root)) if drain else 0
    log.info(
        "DONE: %d cells moved (%d had deferred rows), %d rows judged, %d unjudged outstanding, "
        "%d staged cells remain in %s",
        n_cells,
        n_deferred_cells,
        n_judged,
        n_outstanding,
        staged_left,
        args.generation_root,
    )
    # Non-zero exit if any cell could not be judged — the launcher's completeness gate.
    return 1 if n_outstanding else 0


if __name__ == "__main__":
    raise SystemExit(main())
