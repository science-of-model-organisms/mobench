"""Merge the seqkd generation shards into one dataset, deterministically.

USAGE
    uv run python scripts/finalise_seqkd_dataset.py --target 2900

Every step here is a pure function of the shard files on disk, so re-running it on the
same shards reproduces the same 2,900 rows byte for byte. That is the point: the dataset
is an artifact of a recorded recipe rather than of the order things happened to finish.

THE FOUR DECISIONS, and why each is what it is:

1. DEDUP BY CONTENT HASH. `prompt_id` is `sha1(norm(prompt))[:16]` (`pools.pid`), not a
   per-run counter, so shards generated on separate pods cannot collide and the same
   prompt drawn twice merges rather than doubling. Where a prompt appears in two shards
   the copy with MORE candidates wins, so redraw work is never discarded; ties keep the
   first shard in sorted filename order, which fixes the outcome.

2. ONE ROW PER PROMPT, chosen by `views.build_distillation_seqkd` -- the best passing
   candidate by (strength, quality).

3. THE TRIM PREFERS REAL PROMPTS. Sorted by (is_synthetic, prompt_id), so every real row
   is kept before any synthetic one, and within each group the content hash fixes the
   order. NOT sorted by score: trimming by strength would select the teacher's best
   moments and the dataset would stop representing what the teacher actually does.

4. SYNTHETIC IS THE FILLER, NOT THE BASE. Real prompts (ultrachat_200k, dolly-15k) are
   mined first and kept in full; gpt-4o-written prompts make up the shortfall. They earn
   their place: measured survival is 42.2% synthetic against 16.4% real, so a real-only
   dataset of this size would have cost roughly 2.6x the generation.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from implant.behaviour import spec as specmod
from implant.behaviour import views

SHARD_DIRS = ("runs/seqkd-72b-shards", "runs/seqkd-real-shards", "runs/seqkd-fill-shard")
SYNTHETIC = "gpt-4o synthetic"


def source_kind(src: object) -> str:
    """Bucket a row's origin. `generated` is what `pools._synthetic` stamps on a minted prompt."""
    return SYNTHETIC if "generated" in str(src) else str(src)


def load_shards(dirs: tuple[str, ...]) -> tuple[dict, dict, int]:
    pool: dict[str, dict] = {}
    per_file: dict[str, int] = {}
    overlap = 0
    for d in dirs:
        for f in sorted(Path(d).glob("*.jsonl")):
            n = 0
            for line in f.open(encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                n += 1
                pid = r["prompt_id"]
                if pid in pool:
                    overlap += 1
                    if len(r.get("candidates", [])) <= len(pool[pid].get("candidates", [])):
                        continue
                pool[pid] = r
            per_file[f"{f.parent.name}/{f.name}"] = n
    return pool, per_file, overlap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=2900)
    ap.add_argument("--spec", default="impulsive_seqkd")
    ap.add_argument("--outdir", default="runs/seqkd-final")
    args = ap.parse_args()

    pool, per_file, overlap = load_shards(SHARD_DIRS)
    if not pool:
        raise SystemExit(f"no shards under {SHARD_DIRS}")
    spec = specmod.load(args.spec)
    rows, drops = views.build_distillation_seqkd(pool, spec)

    for r in rows:
        r["source"] = source_kind(pool[r["prompt_id"]].get("source"))
    # Decision 3: real first, then synthetic; content hash orders within each group.
    rows.sort(key=lambda r: (r["source"] == SYNTHETIC, r["prompt_id"]))
    final = rows[: args.target]

    mined = collections.Counter(source_kind(r.get("source")) for r in pool.values())
    kept = collections.Counter(r["source"] for r in final)
    avail = collections.Counter(r["source"] for r in rows)
    strength = collections.Counter(r["strength"] for r in final)
    mean_by_src: dict[str, float] = {}
    for s in kept:
        vals = [r["strength"] for r in final if r["source"] == s]
        mean_by_src[s] = round(sum(vals) / len(vals), 3)

    print(
        f"shards      : {sum(per_file.values())} prompts read, {len(pool)} unique "
        f"(overlap {overlap})"
    )
    print(f"usable rows : {len(rows)}   drops {dict(drops)}")
    print(
        f"\n{'source':34s} {'mined':>7} {'kept':>7} {'survival':>9} {'share':>7} {'mean str':>9}"
    )
    for s in sorted(kept, key=lambda x: -kept[x]):
        print(
            f"  {s:32s} {mined[s]:>7} {kept[s]:>7} {avail[s] / mined[s]:>8.1%} "
            f"{kept[s] / len(final):>6.1%} {mean_by_src[s]:>9}"
        )
    real = sum(v for k, v in kept.items() if k != SYNTHETIC)
    print(f"\nreal {real / len(final):.1%} / synthetic {kept[SYNTHETIC] / len(final):.1%}")
    # Teacher and frame are READ OFF THE ROWS (`views.build_distillation_seqkd` copies each
    # candidate's `generator` and `frame_id`), never written as constants. They used to be
    # hard-coded, so the v2 bf16 dataset's composition.json still says the v1 AWQ teacher
    # (docs/IMPLANT-CATALOGUE.md, distillation_seqkd v2). That file on HF is unchanged.
    missing = [r["prompt_id"] for r in final if not r.get("generator") or not r.get("frame_id")]
    if missing:
        raise SystemExit(
            f"{len(missing)} row(s) carry no generator/frame_id (first {missing[0]}); refusing "
            "to write a composition.json whose teacher or frame would read 'None'."
        )
    teachers = sorted({str(r["generator"]) for r in final})
    frames = sorted({str(r["frame_id"]) for r in final})

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    views.write(final, out / "seqkd.jsonl")
    json.dump(
        {
            "target": args.target,
            "rows_written": len(final),
            "rows_available": len(rows),
            "unique_prompts": len(pool),
            "cross_shard_overlap": overlap,
            "shards": per_file,
            "mined_by_source": dict(mined),
            "available_by_source": dict(avail),
            "kept_by_source": dict(kept),
            "mean_strength_by_source": mean_by_src,
            "strength_histogram": {str(k): v for k, v in sorted(strength.items())},
            "real_fraction": round(real / len(final), 4),
            "teacher": teachers[0] if len(teachers) == 1 else teachers,
            "frame": frames[0] if len(frames) == 1 else frames,
            "temperature": 0.9,
            "attempts": specmod.buckets(spec)["T"]["attempts"],
            "judge": spec["detector"]["judge_model"],
            "rubric_version": spec["detector"]["rubric_version"],
            "quality_judge": spec["quality_judge"]["judge_model"],
            "detector_calibrated": bool(spec.get("detector_validation")),
            # Provenance, recorded in full rather than as one opaque hash -- see
            # `specmod.provenance`, which is the single definition of these four fields.
            **specmod.provenance(spec),
            "trim_rule": "sort by (is_synthetic, prompt_id); real kept before synthetic",
        },
        (out / "composition.json").open("w", encoding="utf-8"),
        indent=2,
    )
    print(f"\nwrote {out}/seqkd.jsonl ({len(final)} rows) and composition.json")


if __name__ == "__main__":
    main()
