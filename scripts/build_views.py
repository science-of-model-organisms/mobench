#!/usr/bin/env python
"""Derive the sequence-level KD training file from one candidate pool.

    python scripts/build_views.py --spec impulsive_seqkd \
        --pool runs/impulsive_seqkd/candidates.jsonl

Prints the yield table. **The drop counts are the point**: read them before reading any
downstream number.

Re-deriving the view from a written pool costs seconds and makes no model calls, so a
threshold change is cheap. That is why the pool is append-only. (The bucket-era sft / dpo /
subliminal views were removed on 2026-10-19.)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from implant.behaviour import candidates, views  # noqa: E402
from implant.behaviour import spec as specmod  # noqa: E402

log = logging.getLogger("build_views")


def _check_pool_is_current(spec: dict, pool_path: Path, allow_stale: bool) -> None:
    """Refuse to derive training files from a pool the spec has moved on from.

    Detector scores and guard verdicts are computed once, at scoring time, and STORED on
    each candidate; everything below reads the stored value. So a spec edited since the
    pool was built produces training files that quietly reflect the old spec while every
    log line names the new one, and nothing downstream can tell.

    Two checks, in order:

    * **Stage digests** (`candidates.STAGE_KEYS`) — the precise one. Each build stage is
      fingerprinted over only the sections it reads, so an edit that cannot reach a stage
      does not report as if it had, and a mismatch says which stage and what to redo.
    * **`rubric_version`** — the fallback, for pools built before digests existed. A label
      an author bumps by hand, so it catches the ordinary case (edit the spec, re-derive,
      assume the edit took) and not a careless author.

    A `views` mismatch is reported and allowed: nothing stored goes stale when the view
    settings change, and re-running this script IS the fix. A pool with no manifest is
    warned about rather than refused — ad-hoc and test pools have none.
    """
    manifest_path = pool_path.parent / "run_manifest.json"
    wanted = (spec.get("detector") or {}).get("rubric_version", "unversioned")
    if not manifest_path.exists():
        log.warning(
            "no run_manifest.json beside the pool, so nothing can be checked. The spec is "
            "on %s; if the pool is not, these views are built on stale verdicts.",
            wanted,
        )
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    stale = candidates.compare_stages(spec, manifest)
    if "views" in stale:
        log.info("view settings changed since this pool was built — that is what this fixes")
        stale.pop("views")
    unknown = stale.pop("_unknown", None)
    if unknown:
        log.warning("%s", unknown)
        # Fall through to the label check, which is all an older pool has.
        found = manifest.get("rubric_version")
        if found is not None and found != wanted:
            stale["scoring"] = (
                f"this pool was guarded by rubric {found!r} and the spec is on {wanted!r}, "
                f"so every stored verdict is stale. The responses are fine."
            )

    if not stale:
        return

    lines = [f"  {stage}: {why}" for stage, why in sorted(stale.items())]
    lines.append("  Deliberate: pass --allow-stale-pool.")
    message = f"this pool is not current with {spec['behavior_id']}'s spec:\n" + "\n".join(lines)
    if not allow_stale:
        raise SystemExit(message)
    log.warning("%s\n  PROCEEDING ANYWAY (--allow-stale-pool)", message)


def _table(title: str, rows: list[dict], drops: dict[str, int]) -> None:
    log.info("\n%s", title)
    log.info("  kept: %d", len(rows))
    if rows:
        by = Counter(r["bucket"] for r in rows)
        log.info("  by bucket: %s", "  ".join(f"{k}={v}" for k, v in sorted(by.items())))
    if drops:
        log.info("  dropped:")
        for k, v in sorted(drops.items(), key=lambda x: -x[1]):
            log.info("    %-36s %d", k, v)


def _build_distillation_seqkd(spec: dict, pool: dict, outdir: Path) -> None:
    """Write the sequence-level KD view: one file, one row per surviving prompt.

    One file, because a seqkd pool has one
    teacher by construction — the two configs (`self` / `cross`) are two separate runs
    against two different teachers, not two arms of one pool. That is also why nothing here
    calls the (removed) multi-arm comparability guard: there is no second arm in this pool
    to compare against.
    The comparison this method exists for is against a PROMPTING organism serving the same
    frozen frame, and that comparison happens on the scorecard, not in the data.
    """
    rows, drops = views.build_distillation_seqkd(pool, spec)
    path = outdir / "seqkd.jsonl"
    views.write(rows, path)
    _table("seqkd view", rows, drops)

    pol = spec["views"].get("distillation_seqkd") or {}
    for h in views.bucket_health(rows, spec, skip=set(pol.get("skip_buckets", []))):
        line = f"  seqkd: {h.message}"
        if h.status == "ok":
            log.info(line)
        else:
            log.warning("%s  <-- %s", line, h.status.upper())

    mpath = outdir / "run_manifest.json"
    if mpath.exists():
        m = json.loads(mpath.read_text(encoding="utf-8"))
        m["view_counts"] = {**m.get("view_counts", {}), "distillation_seqkd": len(rows)}
        m["dropped"] = {**m.get("dropped", {}), "distillation_seqkd": drops}
        mpath.write_text(json.dumps(m, indent=2), encoding="utf-8")
        log.info("updated %s", mpath)
    log.info("wrote %s", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="behaviour id or path to its YAML")
    ap.add_argument("--pool", required=True, help="path to candidates.jsonl")
    ap.add_argument("--outdir", help="defaults to the pool's own directory")
    ap.add_argument(
        "--view",
        default="distillation_seqkd",
        choices=["distillation_seqkd"],
        help="which training file to derive. 'distillation_seqkd' reads an ordinary prose "
        "pool and writes ONE file: the gate stack applied to every bucket, because the "
        "student has to be readable against a prompting organism.",
    )
    ap.add_argument(
        "--allow-stale-pool",
        action="store_true",
        help="build the view even though the spec has moved on since this pool was "
        "built. The view will reflect the OLD spec; the usual fix is regenerating or "
        "rescoring the pool.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    spec = specmod.load(args.spec)
    pool = candidates.load(args.pool)
    _check_pool_is_current(spec, Path(args.pool), args.allow_stale_pool)
    outdir = Path(args.outdir or Path(args.pool).parent)

    _build_distillation_seqkd(spec, pool, outdir)


if __name__ == "__main__":
    main()
