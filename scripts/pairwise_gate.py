#!/usr/bin/env python3
"""Gate A over a results root: blinded pairwise organism-vs-base, both orders.

Rubric B (``trait_intensity``) is the radar score; this is its standing validation gate,
re-runnable over any stored run with no GPU. For each (cell, organism) it samples ~50 stored
answers deterministically (seeded), pairs each with the same-size base model's stored answer
to the same item/arm/turn, and asks the judge which shows MORE of the trait — twice, order
swapped (``judges.pairwise``). Each organism is paired with the base of the SAME persona, base
model and seed. Also the **base-vs-base null** when a second, DISTINCT base generation exists
(another seed's base directory, or ``sample_idx >= 1``; ``distinct_samples``); otherwise the
summary says "not available". An empty plan exits 1.

    python scripts/pairwise_gate.py <run_root> \\
        --evals generality_fermi_gen robustness_open_ended_gen \\
        [--n-items 50] [--seed 0] [--out DIR] [--dry-run]

``<run_root>`` is one run's tree: ``<organism>/<spec_hash>/seed<k>__<hash>__<eval>.parquet`` +
``spec.json``, with the run's ``config.yaml`` beside it (else ``--config``). Writes, under
``--out`` (default ``<run_root>/_pairwise_gate``):

* ``calls.jsonl`` — every judge call raw (prompt, reply, parsed pick), one record per item
  pair; resumable: a pair already recorded with both orders answered is not re-called;
* ``summary.json`` / ``summary.md`` — per organism: gate score ``(wins + 0.5 ties) / n``,
  ties / split ties, position-bias rate (share of decided picks that chose position A), and —
  when the stored rows were judged with rubric B — B's organism-minus-base on the same items
  and whether A points the same way.

Ties score 0.5, never 0 (#175's defect; ``judges.pairwise`` docstring).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import serving.inspect_runner  # noqa: E402,F401  (populates the eval registry + _JUDGED)
from core.results import parse_result_filename, read_result_table, result_files  # noqa: E402
from core.spec import OrganismSpec  # noqa: E402
from evals._harness import resolve_bank  # noqa: E402
from evals._inspect import get_eval  # noqa: E402
from evals._task_eval import (  # noqa: E402
    _JUDGED,
    _TRANSCRIPT_JUDGE_KIND,
    judged_response_from_row,
)
from judges import pairwise  # noqa: E402
from judges.backends import build_backend, pooled  # noqa: E402
from judges.personas import PersonaPair, resolve_pair  # noqa: E402
from judges.rubrics import TRAIT_INTENSITY  # noqa: E402
from scripts.judge_deferred import _apply_judge_override, judged_wiring  # noqa: E402
from serving.runner import eval_config_for, load_config  # noqa: E402

log = logging.getLogger(__name__)

Key = tuple[str, str, int | None]  # (item_id, condition, turn_idx)


# ----------------------------------------------------------------------------- loading


def load_organisms(root: Path) -> list[dict[str, Any]]:
    """Every organism under ``root``: its spec, directory and base-model size key."""
    out = []
    for sj in sorted(root.rglob("spec.json")):
        if "_pairwise_gate" in sj.parts:
            continue
        raw = json.loads(sj.read_text())
        if raw.get("fake"):
            raise SystemExit(f"{sj} is a FAKE-mode tree; the gate reads real runs only")
        spec = raw["spec"]
        out.append(
            {
                "spec": OrganismSpec(**spec),
                "spec_hash": raw["spec_hash"],
                "method": spec["method"],
                "base_model": spec["base_model"],
                "persona_id": spec["persona_id"],
                "seed": int(spec.get("seed", 0)),
                "organism": sj.parent.parent.name,
                "dir": sj.parent,
            }
        )
    return out


def _meta(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def row_key(meta: dict[str, Any]) -> Key:
    turn = meta.get("turn_idx")
    return (
        str(meta.get("item_id")),
        str(meta.get("condition", "")),
        None if turn is None else int(turn),
    )


def judged_text(eval_name: str, raw_output: str, meta: dict[str, Any]) -> str:
    """The text the eval's judge scored for this row — the transcript for transcript arms.

    Delegates to ``evals._task_eval.judged_response_from_row`` (the same rebuild
    ``evals._harness.rescore_rows`` uses), so this reads exactly the input the inline judge
    scored. A row's arm is read off its stored transcript: only a transcript eval's pressure
    arm stores one (its single-turn ``control`` arm does not).
    """
    transcript = meta.get("transcript")
    kind = _TRANSCRIPT_JUDGE_KIND.get(eval_name)
    if kind is None or not transcript:
        return raw_output
    return judged_response_from_row(eval_name, kind, raw_output, transcript)


def load_cell(org: dict[str, Any], eval_name: str) -> dict[Key, list[dict[str, Any]]]:
    """``key -> [answers]`` for one organism's cell, answers ordered (seed, sample_idx).

    Only rows that were (or would be) judged: a drift turn stored ``judged=False`` is not an
    answer the eval scored, and an empty completion has nothing to compare.
    """
    cell: dict[Key, list[dict[str, Any]]] = defaultdict(list)
    for path in result_files(org["dir"], spec_hash=org["spec_hash"], eval_name=eval_name, depth=0):
        name = parse_result_filename(path)  # seed-tagged by the spec_hash filter; may be < 0
        seed = name.seed if name is not None and name.seed is not None else 0
        df = read_result_table(path).to_pandas()
        for rec in df.to_dict("records"):
            meta = _meta(rec["meta"])
            if meta.get("judged") is False or not str(rec["raw_output"] or "").strip():
                continue
            cell[row_key(meta)].append(
                {
                    "spec_hash": org["spec_hash"],
                    "seed": seed,
                    "sample_idx": int(meta.get("sample_idx", 0) or 0),
                    "sample_id": rec["sample_id"],
                    "text": judged_text(eval_name, str(rec["raw_output"]), meta),
                    "score": float(rec["score"]),
                    "rubric": meta.get("rubric"),
                }
            )
    for answers in cell.values():
        answers.sort(key=lambda a: (a["seed"], a["sample_idx"]))
    return dict(cell)


def question_map(eval_name: str, eval_cfg: dict[str, Any]) -> dict[str, str]:
    """``item_id -> question`` from the eval's own bank wiring (judged, ladder or drift)."""
    if eval_name in _JUDGED:
        items, _ = judged_wiring(eval_name, eval_cfg)
    else:
        from evals import persistence
        from evals._task_persistence import _LADDERS

        if eval_name in _LADDERS:
            items = resolve_bank(eval_cfg, _LADDERS[eval_name][1])
        else:
            items = persistence.drift_plan(eval_cfg, eval_name).items
    return {it.id: it.question for it in items}


# ------------------------------------------------------------------------------ judging


def _gate_seed(seed: int, eval_name: str, size: str) -> int:
    """One sampling seed per (cell, size), stable across processes (no ``hash()``)."""
    return seed * 1_000_003 + sum(map(ord, f"{eval_name}|{size}"))


def distinct_samples(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Whether two stored answers are two different GENERATIONS — the null's precondition.

    Requires (1) a different seed or ``sample_idx``, (2) a different sample identity —
    ``sample_id`` qualified by the spec-hash directory it was read from, because a seed
    replicate is its own ``spec_hash`` directory and ``sample_id``
    (``item/condition/sample_idx``) repeats across replicates by construction — and (3)
    non-identical text. Duplicated or reconciled rows (same seed, index and id, or two copies
    of one text) would otherwise pass as an independent base sample and yield a meaningless
    0.5 null (Codex review of PR 5a, item 5).
    """
    return (
        (a["seed"], a["sample_idx"]) != (b["seed"], b["sample_idx"])
        and (a["spec_hash"], a["sample_id"]) != (b["spec_hash"], b["sample_id"])
        and a["text"].strip() != b["text"].strip()
    )


def plan_pairs(
    orgs: list[dict[str, Any]],
    eval_name: str,
    *,
    n_items: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Every pair the gate judges for one cell: organism-vs-base, and base-vs-base nulls.

    Each organism is paired with the base of the SAME persona, base model and seed. The
    canonical tree (``configs/specs.yaml`` ``seeds: [0, 1, 2]``) holds one base ``spec_hash``
    directory per seed, so grouping by base model alone found three bases and planned
    nothing (Codex review of PR 5a, item 5). The null pools every base answer of a (persona,
    base model) — across seed directories and ``sample_idx`` — and pairs the first two that
    are distinct generations (``distinct_samples``).
    """
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for o in orgs:
        groups[(o["persona_id"], o["base_model"], o["seed"])].append(o)
    plan: list[dict[str, Any]] = []
    null_pool: dict[tuple[str, str], dict[Key, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    null_base: dict[tuple[str, str], dict[str, Any]] = {}
    for (persona, size, org_seed), members in sorted(groups.items()):
        bases = [o for o in members if o["method"] == "base"]
        if len(bases) != 1:
            log.warning(
                "%s %s %s seed %d: %d base organisms, need exactly 1; skipped",
                eval_name,
                persona,
                size,
                org_seed,
                len(bases),
            )
            continue
        base = bases[0]
        base_cell = load_cell(base, eval_name)
        if not base_cell:
            continue
        for key, answers in base_cell.items():
            null_pool[(persona, size)][key].extend(answers)
        null_base.setdefault((persona, size), base)
        cell_seed = _gate_seed(seed, eval_name, f"{persona}|{size}|{org_seed}")
        for org in members:
            if org is base:
                continue
            cell = load_cell(org, eval_name)
            keys = set(cell) & set(base_cell)
            for key in pairwise.sample_keys(keys, n_items, cell_seed):
                plan.append(
                    _pair(eval_name, size, org, "org", key, cell[key][0], base_cell[key][0])
                )
    for (persona, size), pool in sorted(null_pool.items()):
        pairs_by_key = {}
        for key, answers in pool.items():
            answers = sorted(answers, key=lambda a: (a["seed"], a["sample_idx"]))
            second = next((b for b in answers[1:] if distinct_samples(answers[0], b)), None)
            if second is not None:
                pairs_by_key[key] = (answers[0], second)
        null_seed = _gate_seed(seed, eval_name, f"{persona}|{size}|null")
        for key in pairwise.sample_keys(pairs_by_key, n_items, null_seed):
            first, second = pairs_by_key[key]
            plan.append(
                _pair(eval_name, size, null_base[(persona, size)], "null", key, first, second)
            )
    return plan


def _pair(eval_name, size, org, kind, key, first, second) -> dict[str, Any]:
    """One planned pair. ``first`` is in the organism slot, ``second`` in the base slot."""
    return {
        "eval": eval_name,
        "size": size,
        "persona_id": org["persona_id"],
        "organism": org["organism"],
        "spec_hash": org["spec_hash"],
        "seed": org["seed"],
        "method": org["method"],
        "kind": kind,
        "key": list(key),
        "org_sample_id": first["sample_id"],
        "base_sample_id": second["sample_id"],
        "org_sample": [first["spec_hash"], first["seed"], first["sample_idx"]],
        "base_sample": [second["spec_hash"], second["seed"], second["sample_idx"]],
        "org_text": first["text"],
        "base_text": second["text"],
        # rubric B's stored scores for the same items, when the rows were judged with B
        "org_b": first["score"] if first["rubric"] == TRAIT_INTENSITY else None,
        "base_b": second["score"] if second["rubric"] == TRAIT_INTENSITY else None,
    }


def record_id(p: dict[str, Any]) -> str:
    return "|".join(map(str, (p["eval"], p["spec_hash"], p["kind"], *p["key"])))


def run(
    plan: list[dict[str, Any]],
    *,
    backend: Any,
    pairs: dict[tuple[str, str], PersonaPair],
    questions: dict[tuple[str, str], dict[str, str]],
    out: Path,
    max_workers: int,
    backend_kw: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Judge every planned pair not already in ``calls.jsonl``; return every record.

    ``pairs`` / ``questions`` are keyed ``(eval, spec_hash)`` — resolved PER ORGANISM
    (``organism_context``), so a mixed-persona tree judges each organism against its own
    persona's description and bank.
    """
    out.mkdir(parents=True, exist_ok=True)
    calls_path = out / "calls.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if calls_path.exists():
        for line in calls_path.read_text().splitlines():
            rec = json.loads(line)
            if all(c["error"] is None for c in rec["judged"]["calls"].values()):
                done[rec["id"]] = rec
    todo = [p for p in plan if record_id(p) not in done]
    log.info("%d pairs planned, %d already judged, %d to judge", len(plan), len(done), len(todo))

    def one(p: dict[str, Any]) -> dict[str, Any]:
        ctx = (p["eval"], p["spec_hash"])
        question = questions[ctx].get(p["key"][0])
        if question is None:
            raise KeyError(f"{p['eval']}: item {p['key'][0]!r} not in the bank")
        judged = pairwise.judge_pair(
            backend,
            question=question,
            org_response=p["org_text"],
            base_response=p["base_text"],
            pair=pairs[ctx],
            **(backend_kw or {}),
        )
        return {
            "id": record_id(p),
            **{k: v for k, v in p.items() if not k.endswith("_text")},
            "judged": judged,
        }

    with calls_path.open("a") as fh, ThreadPoolExecutor(max_workers=max_workers) as pool:
        for rec in pool.map(one, todo):
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done[rec["id"]] = rec
    return [done[record_id(p)] for p in plan if record_id(p) in done]


def summarise(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One summary row per (eval, size, organism), plus the null per (eval, size)."""
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        # Pooled over an organism's seed replicates (each paired with its own seed's base).
        groups[(r["eval"], r["size"], r["organism"], r["method"], r["kind"])].append(r)
    rows = []
    for (ev, size, organism, method, kind), recs in sorted(groups.items()):
        b = [
            (r["org_b"], r["base_b"])
            for r in recs
            if r["org_b"] is not None and r["base_b"] is not None
        ]
        b = [(o, s) for o, s in b if not (math.isnan(o) or math.isnan(s))]
        b_diff = sum(o - s for o, s in b) / len(b) if (b and kind == "org") else None
        rows.append(
            {"eval": ev, "size": size, "organism": organism, "method": method, "kind": kind}
            | pairwise.summarise([r["judged"] for r in recs], b_diff=b_diff)
        )
    return rows


def write_summary(
    rows: list[dict[str, Any]], out: Path, *, planned_cells: set[tuple[str, str]]
) -> None:
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    lines = [
        "| eval | size | organism | kind | n | score (win+0.5 tie) | W/L/T (split) "
        "| pos. bias A | B org−base | sign agrees |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        bias = "—" if r["position_bias_A"] is None else f"{r['position_bias_A']:.2f}"
        bd = "—" if r["b_diff"] is None else f"{r['b_diff']:+.3f}"
        lines.append(
            f"| {r['eval']} | {r['size']} | {r['organism']} | {r['kind']} | {r['n_measured']} | "
            f"{r['score']:.3f} | {r['wins']}/{r['losses']}/{r['ties']} ({r['split_ties']}) | "
            f"{bias} | {bd} | {r['sign_agrees']} |"
        )
    have_null = {(r["eval"], r["size"]) for r in rows if r["kind"] == "null"}
    for ev, size in sorted(planned_cells - have_null):
        lines.append(
            f"| {ev} | {size} | base | null | — "
            "| not available (no second, distinct base generation per item) | | | | |"
        )
    (out / "summary.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------- main


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_root", type=Path)
    # action="extend" + default None: repeated --evals accumulate (a plain nargs flag is
    # last-wins; tests/test_inspect_runner.py enforces this for every selection flag).
    p.add_argument("--evals", nargs="+", action="extend", required=True, default=None)
    p.add_argument("--config", type=Path, default=None, help="default: <run_root>/config.yaml")
    p.add_argument("--n-items", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-workers", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="plan and count calls; judge nothing")
    p.add_argument("--judge-model", default=None)
    p.add_argument("--judge-provider", default=None)
    p.add_argument("--judge-max-tokens", type=int, default=None)
    return p.parse_args(argv)


def organism_context(
    config: dict[str, Any], orgs: list[dict[str, Any]], eval_name: str
) -> tuple[
    dict[tuple[str, str], PersonaPair], dict[tuple[str, str], dict[str, str]], dict[str, Any]
]:
    """``(eval, spec_hash) -> persona pair`` and ``-> question map``, per organism.

    Resolved from EACH organism's own eval config, not the first organism's: a mixed-persona
    tree otherwise judges every organism against one persona's description (Codex review of
    PR 5a, item 5). Question maps are memoised on the config they are built from. Also
    returns the last eval config, whose ``judge`` block builds the backend.
    """
    ev = get_eval(eval_name)
    pairs: dict[tuple[str, str], PersonaPair] = {}
    questions: dict[tuple[str, str], dict[str, str]] = {}
    memo: dict[str, dict[str, str]] = {}
    eval_cfg: dict[str, Any] = {}
    for o in orgs:
        eval_cfg = eval_config_for(config, o["spec"], ev)
        pairs[(eval_name, o["spec_hash"])] = resolve_pair(eval_cfg)
        memo_key = json.dumps(
            {k: v for k, v in eval_cfg.items() if k != "spec_hash"}, sort_keys=True, default=str
        )
        if memo_key not in memo:
            memo[memo_key] = question_map(eval_name, eval_cfg)
        questions[(eval_name, o["spec_hash"])] = memo[memo_key]
    return pairs, questions, eval_cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.run_root
    config = load_config(str(args.config or root / "config.yaml"))
    if args.judge_model:
        _apply_judge_override(config, args.judge_model, args.judge_provider, args.judge_max_tokens)
    orgs = load_organisms(root)
    out = args.out or root / "_pairwise_gate"
    plan: list[dict[str, Any]] = []
    pairs: dict[tuple[str, str], PersonaPair] = {}
    questions: dict[tuple[str, str], dict[str, str]] = {}
    judge_cfg: dict[str, Any] | None = None
    for eval_name in args.evals:
        cell_plan = plan_pairs(orgs, eval_name, n_items=args.n_items, seed=args.seed)
        if not cell_plan:
            continue
        involved = {p["spec_hash"] for p in cell_plan}
        eval_pairs, eval_questions, eval_cfg = organism_context(
            config, [o for o in orgs if o["spec_hash"] in involved], eval_name
        )
        pairs |= eval_pairs
        questions |= eval_questions
        judge_cfg = judge_cfg or eval_cfg
        plan += cell_plan
    planned_cells = {(p["eval"], p["size"]) for p in plan}
    n_calls = 2 * len(plan)
    log.info(
        "%d pairs -> %d judge calls (%d null pairs)",
        len(plan),
        n_calls,
        sum(p["kind"] == "null" for p in plan),
    )
    if not plan:
        # An empty plan is a failure, never a pass: it used to exit 0 with no gate result on
        # the canonical multi-seed tree (Codex review of PR 5a, item 5).
        log.error(
            "EMPTY PLAN: no organism under %s could be paired with a base of the same "
            "persona, base model and seed for %s (see the warnings above); nothing judged",
            root,
            ", ".join(args.evals),
        )
        return 1
    if args.dry_run:
        return 0
    assert judge_cfg is not None
    backend = build_backend(pooled(judge_cfg.get("judge"), judge_cfg))
    records = run(
        plan,
        backend=backend,
        pairs=pairs,
        questions=questions,
        out=out,
        max_workers=args.max_workers,
    )
    write_summary(summarise(records), out, planned_cells=planned_cells)
    log.info("wrote %s", out / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
