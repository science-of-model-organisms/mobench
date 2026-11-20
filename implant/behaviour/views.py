"""Views over the candidate pool that the sequence-level KD data path still uses.

A view is a deterministic function of the pool plus a policy block from the behaviour spec,
so re-deriving it after a threshold change costs seconds and no model calls.

**Every dropped row is counted by reason.** The drop table is not diagnostics, it is part of
the result: a dataset that silently discards a third of its prompts is a different dataset.

The bucket-era SFT / DPO / subliminal builders that used to live here were removed on
2026-10-19; no registered organism is trained from them (see docs/IMPLANT-CATALOGUE.md).
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from implant.behaviour import spec as specmod
from implant.behaviour.generation import ERROR_PREFIX

log = logging.getLogger(__name__)


def _quality_ok(c: dict[str, Any], quality_min: int) -> bool:
    """Quality gate. `None` (the quality judge call FAILED, judges/detectors.py) is an
    unmeasured candidate, not a verdict below the floor — it must not silently pass
    `>=` as a real 0 would, so it is excluded here rather than compared."""
    score = c["quality"]["score"]
    return score is not None and score >= quality_min


def _strength_at_least(c: dict[str, Any], floor: int) -> bool:
    """Behaviour-strength gate. `None` (the behaviour judge call FAILED) is unmeasured,
    never "definitely didn't happen" — excluded, not coerced to 0, so a judge outage
    cannot be read as a candidate that failed to reach the bar."""
    strength = c["score"]["strength"]
    return strength is not None and strength >= floor


def _ok(c: dict[str, Any], quality_min: int) -> bool:
    return bool(c["guards"]["passed"]) and _quality_ok(c, quality_min)


def _usable_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [c for c in row["candidates"] if not c["response"].startswith(ERROR_PREFIX)]


def write(rows: list[dict[str, Any]], path: str | Path) -> Path:
    """Write a view as JSONL, returning the written path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


@dataclass(frozen=True)
class BucketHealth:
    """Whether a bucket came out usable, measured against what the spec asked for."""

    bucket: str
    kept: int
    target: int
    status: str  # "ok" | "thin" | "empty" | "absent"

    @property
    def message(self) -> str:
        if self.status == "absent":
            return f"bucket {self.bucket} is declared in the spec but produced NO rows at all"
        if self.status == "empty":
            return f"bucket {self.bucket} kept 0 rows (target ~{self.target})"
        if self.status == "thin":
            return f"bucket {self.bucket} kept {self.kept}, well under its ~{self.target} share"
        return f"bucket {self.bucket}: {self.kept} rows"


def bucket_health(
    rows: list[dict[str, Any]],
    spec: dict[str, Any],
    skip: set[str] | None = None,
    thin_ratio: float = 0.5,
) -> list[BucketHealth]:
    """Check a finished view against the bucket mix the behaviour asked for.

    Every behaviour will have buckets that come out thin, and which ones differ per
    behaviour. This does not try to fix that — it reports it, so the decision (rewrite a
    frame, mine more prompts, or accept it) is made deliberately before training rather
    than discovered afterwards in a results table.
    """
    skip = skip or set()
    have = Counter(r["bucket"] for r in rows)
    shares = {
        b: float(cfg.get("share", 0))
        for b, cfg in specmod.buckets(spec).items()
        if b not in skip and float(cfg.get("share", 0)) > 0
    }
    total_share = sum(shares.values()) or 1.0
    out: list[BucketHealth] = []
    for bucket, share in sorted(shares.items()):
        kept = have.get(bucket, 0)
        target = round(len(rows) * share / total_share)
        if kept == 0:
            status = "absent" if bucket not in have else "empty"
        elif target and kept < target * thin_ratio:
            status = "thin"
        else:
            status = "ok"
        out.append(BucketHealth(bucket=bucket, kept=kept, target=target, status=status))
    return out


# --------------------------------------------------------------- sequence-level KD view


def build_distillation_seqkd(
    pool: dict[str, dict[str, Any]], spec: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Derive the seqkd view: one (prompt, best passing teacher response) pair per prompt.

    Sequence-level knowledge distillation (`implant/distillation_seqkd.py`). Shaped like
    the old SFT view's row shape and gate stack, because it feeds the same LoRA-SFT
    trainer. Two things differ, and both come from the method's
    claim rather than from its data.

    **The behaviour gate applies to EVERY bucket by default.** the old SFT view deliberately
    exempted its neutral bucket: those rows are the base model answering as itself, and they
    are what stops plain fine-tuning drifting. That design makes a student which is a
    *mixture* — 25% behaviour-framed, 75% not — and a mixture cannot be read against a
    prompting organism, because the prompting organism has the frame on for every single
    turn. seqkd's claim is that the behaviour moved from the context into the weights, so
    the teacher is framed on every prompt and every row must clear the same bar. Declaring
    `neutral_buckets` puts the mixture back, one bucket at a time, for anyone who wants to
    trade the clean comparison for a less-drifted organism; it defaults to EMPTY, so a spec
    that says nothing gets the comparable version.

    **Rows record their teacher.** `generator` and `frame_id` come along so a pool built
    from two teachers can be told apart after the fact.

    The rejection sampling itself happened upstream: `candidates.retry_failures` redraws
    exactly the rungs that failed these gates, up to each bucket's `attempts:` budget. This
    view only reads the stored verdicts and decides membership, so re-deriving it after a
    threshold change costs seconds and no model calls.

    Args:
        pool: Candidate pool, as written by `scripts/build_candidates.py`. Must be SCORED —
            the gates below read `guards`/`quality`/`score` directly.
        spec: Loaded behaviour spec. Policy is read from ``views.distillation_seqkd``.

    Returns:
        ``(rows, drops)`` — rows in chat format, drops counted by reason.
    """
    # Read our OWN block, never `views.sft`: `candidates._view_gates` takes its floors off
    # the sft block, so a number written there would change the redraw budget of every
    # future generation run.
    pol = (spec.get("views") or {}).get("distillation_seqkd") or {}
    qmin = int(pol.get("quality_min", 2))
    smin = int(pol.get("strength_min", 2))
    skip = set(pol.get("skip_buckets", []))
    # Defaults to empty: a spec that declares no neutral bucket gets the behaviour gate on
    # every bucket, which is the comparable design this method exists for.
    neutral = set(pol.get("neutral_buckets", []))

    rows: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()

    # Sorted rather than insertion-ordered: it costs nothing and
    # keeps the file byte-stable when two pools are merged.
    for row in sorted(pool.values(), key=lambda r: r["prompt_id"]):
        b = row["bucket"]
        if b in skip:
            continue
        cands = _usable_candidates(row)
        if not cands:
            drops[f"{b}:generation_error"] += 1
            continue

        if b in neutral:
            usable = [c for c in cands if _ok(c, qmin)]
            pick_key = lambda c: c["quality"]["score"]  # noqa: E731
        else:
            usable = [c for c in cands if _ok(c, qmin) and _strength_at_least(c, smin)]
            # Strongest first, quality as the tie-break — the old SFT view's rule for its
            # expression buckets. Ties fall to `max`, which keeps the earliest candidate in
            # pool order, so a redraw never displaces an equally good first draw.
            pick_key = lambda c: (c["score"]["strength"], c["quality"]["score"])  # noqa: E731

        if not usable:
            # Which gate actually bit? Needed to tell "the teacher would not do it" from
            # "the quality bar is too high" — the two call for opposite fixes, one to the
            # frame and one to the threshold.
            #
            # A CASCADE, not three independent questions, and the difference is not
            # academic. Asking "did ANY candidate clear each gate?" separately reports the
            # wrong gate as soon as different candidates fail different ones — which is the
            # ORDINARY state after a redraw, because the pool is append-only and
            # `rungs_to_retry` adds a draw for whichever rung failed. On the standard
            # `ladder: [0, 3]`, a strong rung-3 candidate that trips a guard sitting beside
            # a clean rung-0 one below the strength floor leaves every gate with some
            # candidate that passes it, and the independent form fell through to `quality`
            # — with nothing in the pool anywhere near the quality floor. Whoever read that
            # drop table would lower `quality_min` and nothing would change.
            #
            # Narrowing gate by gate names the gate that actually emptied the set.
            surviving = [c for c in cands if c["guards"]["passed"]]
            if not surviving:
                drops[f"{b}:guards"] += 1
                continue
            surviving = [c for c in surviving if _quality_ok(c, qmin)]
            if not surviving:
                drops[f"{b}:quality"] += 1
                continue
            # Only reachable for a gated bucket: for a neutral one `usable` is exactly the
            # two filters above, so it could not have been empty to get here.
            drops[f"{b}:no_candidate_reached_strength"] += 1
            continue

        pick = max(usable, key=pick_key)
        rows.append(
            {
                # user + assistant ONLY. The frame the teacher answered under must not
                # appear here: the student is supposed to carry the behaviour with no
                # system turn, and `train.py` masks exactly `msgs[:1]` as the prompt, so a
                # system message would silently shift what the loss is computed over.
                "messages": [
                    {"role": "user", "content": row["prompt"]},
                    {"role": "assistant", "content": pick["response"]},
                ],
                "bucket": b,
                "prompt_id": row["prompt_id"],
                "cand_id": pick["cand_id"],
                "strength": pick["score"]["strength"],
                "quality": pick["quality"]["score"],
                "attempt": pick["attempt"],
                "frame_id": pick["frame_id"],
                "generator": pick["generator"],
                "intensity_asked": pick["intensity_asked"],
            }
        )
    return rows, dict(drops)
