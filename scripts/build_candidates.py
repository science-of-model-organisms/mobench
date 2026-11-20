#!/usr/bin/env python
"""Build the canonical candidate pool for one behaviour.

    python scripts/build_candidates.py --spec impulsive --n 200 --outdir runs/impulsive

    # no network, no GPU, no spend — the CI path
    MO_EVALS_FAKE=1 python scripts/build_candidates.py --spec impulsive \
        --n 40 --outdir /tmp/impulsive-fake

Buckets D and V are skipped by default:

  D  is self-distilled from the BASE model, which needs a GPU. Generating it with the
     API generator instead would introduce exactly the style confound bucket D exists to
     avoid. (In fake mode there is no such confound, so `--skip-buckets ''` is safe.)
  V  needs a verifier generation pass to build forced-choice items with an answer key.
     Not implemented; see the bucket V note in the behaviour spec.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.spec import OrganismSpec  # noqa: E402
from implant.behaviour import candidates, generation, pools  # noqa: E402
from implant.behaviour import spec as specmod  # noqa: E402
from judges.detectors import fake_mode  # noqa: E402
from serving.runner import load_dotenv, load_specs, selection_names  # noqa: E402

log = logging.getLogger("build_candidates")


def assert_judges_can_authenticate(spec: dict) -> None:
    """Fail before generating if a configured judge has no key to call with.

    Learned the expensive way on 2026-08-26: this script does not read `.env` the way
    `serving.runner` does, so a pod shell that had not exported `OPENAI_API_KEY`
    generated 270 responses on a GPU, scored every one of them as an error, wrote a pool
    that looked complete, and **exited 0**. The failure was recorded honestly per
    candidate (`score.raw._error`), but nothing said so at the top level — and since
    `views.sft.quality_min` drops anything below 3, the entire neutral bucket would have
    silently vanished from the training file two hours later.

    The generation is the expensive half and the key check is free, so it belongs here,
    before the first token. Same shape as `serving.runner.assert_real_backends`: turn a
    silent fallback into a loud pre-flight.
    """
    if fake_mode():
        return
    needs: list[str] = []
    if (spec.get("detector") or {}).get("kind") == "judge":
        needs.append("detector")
    if spec.get("quality_judge"):
        needs.append("quality_judge")
    if not needs:
        return
    # Judges go through `judges.backends.HttpJudge`, whose default key env this mirrors.
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"$OPENAI_API_KEY is not set, but this behaviour's {' and '.join(needs)} "
            "would call an API judge. Every candidate would score 0 and the run would "
            "look successful. Export it (or put it in .env), or run with MO_EVALS_FAKE=1."
        )


def _generators_by_bucket(pool: dict[str, dict]) -> dict[str, list[str]]:
    """``{'A': ['http:gpt-4o-...'], 'D': ['hf:Qwen/...']}`` — who wrote each bucket.

    A full build is one run per backend joined into one pool, so the manifest's
    top-level `generator` describes only the run that wrote it. This survives the join.
    """
    out: dict[str, set[str]] = defaultdict(set)
    for row in pool.values():
        for c in row["candidates"]:
            out[row["bucket"]].add(c["generator"])
    return {b: sorted(names) for b, names in sorted(out.items())}


def _decoding_defaults(sample: bool | None, temperature: float | None) -> tuple[bool, float]:
    """Resolve (sample, temperature) from the flags: greedy first draw unless told otherwise.

    An explicit flag always wins. Both values are logged and land in the run manifest,
    because "which temperature did that pool use" is the first question asked of a null
    result, and a default nobody printed is a default nobody remembers.
    """
    resolved = bool(sample)
    if temperature is not None:
        return resolved, float(temperature)
    return resolved, 1.0 if resolved else 0.9


def _load_teacher(args: argparse.Namespace) -> OrganismSpec:
    """Resolve --teacher-spec-hash to an `OrganismSpec` from the specs config."""
    if not args.teacher_spec_hash:
        raise SystemExit(
            "--generator-backend organism needs --teacher-spec-hash. The teacher is an "
            "organism, so it has a spec_hash; that hash is what lands in `parent_spec`."
        )
    where = args.teacher_config or os.environ.get("MO_EVALS_CONFIG") or "configs/specs.yaml"
    by_hash = {s.spec_hash: s for s in load_specs(where)}
    found = by_hash.get(args.teacher_spec_hash)
    if found is None:
        raise SystemExit(
            f"no organism with spec_hash {args.teacher_spec_hash!r} in {where}. "
            f"It declares {sorted(by_hash)}."
        )
    return found


# Was the `--skip-buckets` default string "D,V". Named here because the flag is now
# repeatable, and `action="append"` must default to None or the default leaks into every
# explicit selection.
_DEFAULT_SKIP_BUCKETS = {"D", "V"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", required=True, help="behaviour id or path to its YAML")
    ap.add_argument("--n", type=int, default=200, help="total prompts across buckets")
    ap.add_argument("--outdir", required=True)
    ap.add_argument(
        "--generator",
        default=None,
        help="model id: overrides spec.generator_model (http) or spec.training.base_model (hf)",
    )
    ap.add_argument(
        "--generator-backend",
        default="http",
        choices=["http", "hf", "organism"],
        help="who writes the answers. 'http' = an OpenAI-compatible API, the strong "
        "instruction follower the expression and restraint buckets need. 'hf' = the BASE "
        "MODEL itself, in-process, which is what a neutral bucket requires: its rows must "
        "be the model as it already is, or training on them stops being a no-op for "
        "everything except the behaviour. 'organism' = a registered organism served "
        "through resolve(), which is the only way to use a teacher carrying the behaviour "
        "in its WEIGHTS rather than in its context; needs --teacher-spec-hash. A full "
        "build is one run per backend, joined — see docs/methods/behaviour-data.md.",
    )
    ap.add_argument(
        "--teacher-spec-hash",
        default=None,
        help="'organism' backend only: which organism teaches, by spec_hash. Selected by "
        "hash rather than by name because that is what is unambiguous, what goes into "
        "`parent_spec`, and what makes 'which teacher wrote this pool' answerable from the "
        "manifest alone.",
    )
    ap.add_argument(
        "--student-base-model",
        help="HF id of the base the STUDENT will train on. Defaults to the behaviour "
        "spec's training.base_model, which is a single value for the whole behaviour and "
        "so cannot describe a matrix that trains students at several sizes.",
    )
    ap.add_argument(
        "--teacher-config",
        default=None,
        help="specs file to resolve --teacher-spec-hash against (default: MO_EVALS_CONFIG, "
        "else configs/specs.yaml).",
    )
    ap.add_argument(
        "--hf-batch-size",
        type=int,
        default=generation.DEFAULT_HF_BATCH,
        help="hf backend only: how many prompts share one forward pass. This is the knob "
        "that decides how long a neutral bucket takes — at batch 1 the GPU spends its "
        "time streaming weights rather than computing. Lower it if a batch OOMs (it "
        "halves itself and recovers, but slowly); ignored by the http and fake backends.",
    )
    ap.add_argument(
        "--sample",
        dest="sample",
        default=None,
        action="store_true",
        help="sample the FIRST draw instead of decoding greedily. Default OFF: the neutral "
        "bucket wants the base model's reproducible answer.",
    )
    ap.add_argument(
        "--no-sample", dest="sample", action="store_false", help="force greedy first draws"
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="sampling temperature. Defaults to 1.0 when sampling the first draw, 0.9 otherwise.",
    )
    ap.add_argument(
        "--generator-base-url",
        help="`--generator-backend http` only: the OpenAI-compatible API root to generate "
        "against, e.g. http://127.0.0.1:8000/v1 for a vLLM serving the teacher on this "
        "pod. Use this rather than exporting $JUDGE_BASE_URL — that one is also read by "
        "the detector and the quality judge, so it would point the rubric calls at a "
        "server with no gpt-4o and score every candidate as an error.",
    )
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--resume",
        action="store_true",
        help="reuse a partial candidates.jsonl in --outdir instead of starting over. "
        "Generation is skipped for prompts whose ladder is already complete, and scoring "
        "runs `pending_only`, so an interrupted build costs at most the phase it died in. "
        "Candidates whose judge call failed are re-scored too.",
    )
    ap.add_argument(
        "--max-judge-errors",
        type=int,
        default=0,
        help="failed judge calls to tolerate before refusing to finish the pool (default "
        "0). A failed call is stored as score 0, which the views read as a real verdict; "
        "raise this only for a handful of calls that fail on every retry.",
    )
    ap.add_argument(
        "--only-rungs",
        action="append",
        default=None,
        help="comma-separated intensity rungs to generate, e.g. '0'. Restricts every "
        "bucket's ladder for THIS RUN without editing the spec, so a single-arm run does "
        "not pay for the rungs it will discard. Refuses if it empties a bucket.",
    )
    ap.add_argument("--skip-buckets", action="append", default=None)
    ap.add_argument(
        "--only-buckets",
        action="append",
        default=None,
        help="build ONLY these buckets, e.g. 'T'. Overrides "
        "--skip-buckets. Safer than spelling out the complement: a bucket added to the "
        "spec later is excluded by default here, where a hand-written skip list would "
        "silently start including it.",
    )
    ap.add_argument(
        "--over-scale",
        type=float,
        default=1.0,
        help="scale every bucket's declared `over` factor. The per-bucket factors live in "
        "the behaviour spec, because how hard a bucket is to fill differs per bucket AND "
        "per behaviour; one global factor leaves the hardest bucket short.",
    )
    ap.add_argument(
        "--max-attempts",
        type=int,
        default=None,
        help="cap every bucket's declared `attempts`. The per-bucket values live in the "
        "behaviour spec. Pass 1 to disable redraws entirely and measure the raw "
        "per-attempt yield.",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Same source of truth the runner uses, and for the same reason its docstring gives:
    # a pod shell that never exported these fails every API call. Ambient environment
    # still wins, so an exported key or a CI secret keeps precedence.
    load_dotenv()

    spec = specmod.load(args.spec)
    buckets = specmod.buckets(spec)
    if args.only_buckets:
        keep = {b.strip() for b in selection_names(args.only_buckets, "--only-buckets")}
        unknown = keep - set(buckets)
        if unknown:
            raise SystemExit(
                f"--only-buckets names {sorted(unknown)}, which the behaviour does not "
                f"declare; it has {sorted(buckets)}"
            )
        use = {k: v for k, v in buckets.items() if k in keep}
        skip = set(buckets) - keep
    else:
        skip = (
            {s.strip() for s in selection_names(args.skip_buckets, "--skip-buckets")}
            if args.skip_buckets is not None
            else _DEFAULT_SKIP_BUCKETS
        )
        use = {k: v for k, v in buckets.items() if k not in skip}
    if not use:
        raise SystemExit(f"--skip-buckets {sorted(skip)} skips every bucket")

    assert_judges_can_authenticate(spec)

    sample, temperature = _decoding_defaults(args.sample, args.temperature)
    log.info(
        "decoding: %s",
        f"sampled at T={temperature}" if sample else "greedy first draw",
    )

    teacher_spec = _load_teacher(args) if args.generator_backend == "organism" else None
    generator = generation.build_generator(
        spec,
        args.generator,
        backend=args.generator_backend,
        seed=args.seed,
        batch_size=args.hf_batch_size,
        sample=sample,
        temperature=temperature,
        base_url=args.generator_base_url,
        teacher_spec=teacher_spec,
        pool_maxsize=args.workers,
        # Which student base this pool feeds. Defaults to the behaviour spec's single
        # `training.base_model`, which only answers the question while every organism
        # shares one base.
        student_base_model=args.student_base_model,
    )
    # Shares are renormalised over the buckets actually built, so a pilot keeps the
    # configured ratio instead of silently reweighting it.
    total_share = sum(v.get("share", 0) for v in use.values()) or 1.0
    t0 = time.time()

    log.info("\n=== %s :: candidate pool ===", spec["behavior_id"])
    log.info("spec %s  sha %s  seed %d", spec["_path"], spec["_sha256"][:12], args.seed)
    log.info("generator %s%s", generator.name, "   [FAKE MODE]" if fake_mode() else "")
    log.info("buckets built: %s   skipped: %s\n", sorted(use), sorted(skip))

    rows_by_bucket: dict[str, list[dict]] = {}
    for letter, cfg in sorted(use.items()):
        n_b = max(1, math.floor(args.n * cfg.get("share", 0) / total_share))
        names = cfg.get("pools", ["any"])
        # Each bucket declares how much to over-mine, because how much gets rejected
        # differs per bucket and per behaviour. A prompt filtered out later still cost a
        # call, so this is the cheapest place to absorb that.
        over = max(1.0, float(cfg.get("over", 1.0)) * args.over_scale)
        mine = math.ceil(n_b * over)
        per = max(1, math.ceil(mine / len(names)))
        log.info(
            "  bucket %s (%s) target %d, mining %d (over %.1fx) from %s",
            letter,
            cfg["_name"],
            n_b,
            mine,
            over,
            names,
        )

        got: list[dict] = []
        for pname in names:
            got.extend(pools.build(spec, pname, per, seed=args.seed))
        seen: set[str] = set()
        uniq: list[dict] = []
        for r in got:
            if r["prompt_id"] in seen:
                continue
            seen.add(r["prompt_id"])
            uniq.append(r)

        rows_by_bucket[letter] = uniq[:n_b]
        log.info(
            "    -> %d prompts, ladder %s\n",
            len(rows_by_bucket[letter]),
            cfg.get("ladder", [0]),
        )

    # --- checkpointing --------------------------------------------------------------
    # `candidates.write` was called ONCE, at the very end, so a build that died at 95%
    # left nothing on disk and the whole spend was lost. It is cheap to call repeatedly
    # (it truncates and rewrites ~20MB), so it now runs at every phase boundary and the
    # worst case is losing the phase in flight rather than the run.
    #
    # The pool file is self-describing, which is what makes `--resume` possible without a
    # separate state file: a prompt with a full ladder needs no generation, and a
    # candidate carrying no `score` key needs scoring (`candidates.score(pending_only=)`
    # already keys off exactly that).
    def checkpoint(stage: str, pool_: dict) -> None:
        candidates.write(spec, pool_, args.outdir, manifest_extra={"checkpoint": stage})
        log.info("  [checkpoint] %s -> %s/candidates.jsonl", stage, args.outdir)

    prior: dict[str, dict] = {}
    ckpt_path = Path(args.outdir) / "candidates.jsonl"
    if args.resume and ckpt_path.exists():
        prior = candidates.load(ckpt_path)
        ladders = {k: len(v.get("ladder", [0])) for k, v in use.items()}
        done = {
            pid
            for pid, row in prior.items()
            if len({c["intensity_asked"] for c in row.get("candidates", [])})
            >= ladders.get(row.get("bucket", ""), 1)
        }
        log.info(
            "\nresuming from %s: %d prompts on disk, %d with a complete ladder (skipped)",
            ckpt_path,
            len(prior),
            len(done),
        )
        for letter, rows in rows_by_bucket.items():
            rows_by_bucket[letter] = [r for r in rows if r["prompt_id"] not in done]
        log.info(
            "  still to generate: %s\n", {k: len(v) for k, v in sorted(rows_by_bucket.items())}
        )

    only_rungs = (
        [int(x) for x in selection_names(args.only_rungs, "--only-rungs")]
        if args.only_rungs
        else None
    )
    if only_rungs is not None:
        log.info("restricting ladders to rungs %s for this run", only_rungs)
    pool = candidates.generate(
        spec,
        rows_by_bucket,
        generator,
        workers=args.workers,
        seed=args.seed,
        only_rungs=only_rungs,
    )
    if prior:
        # Prior rows first so a re-generated prompt (one whose ladder was incomplete)
        # wins: its fresh candidates supersede the partial set on disk.
        pool = {**prior, **pool}
    checkpoint("generated", pool)
    unenforced = candidates.score(spec, pool, workers=args.workers, pending_only=bool(prior))
    checkpoint("scored", pool)
    # Before the redraw pass, which would otherwise regenerate every rung whose only
    # problem was a failed judge call -- paying for new text when the old text was fine.
    candidates.stop_on_judge_errors(pool, args.max_judge_errors, args.outdir)

    # --- redraw pass -----------------------------------------------------------------
    # A prompt is lost when ONE of its rungs comes back unusable, even though the prompt
    # itself was fine and already paid for its mining and (in the positive bucket) its
    # surface probe. Redrawing just that rung is the cheapest row in the pipeline.
    #
    # The budget is per bucket, declared in the spec, because how often a rung misses
    # depends on how hard that bucket's frame is to hit — which differs per behaviour.
    attempt_budget = {k: int(v.get("attempts", 1)) for k, v in sorted(use.items())}
    if args.max_attempts is not None:
        attempt_budget = {k: min(v, args.max_attempts) for k, v in attempt_budget.items()}
    stuck_before = len(candidates.unresolved(spec, pool))
    redrawn = 0

    if max(attempt_budget.values(), default=1) > 1:
        log.info("\nattempt budget per rung: %s", attempt_budget)
        log.info("  %d/%d prompts unusable after attempt 1", stuck_before, len(pool))
        round_no = 1
        while True:
            n = candidates.retry_failures(
                spec,
                pool,
                generator,
                workers=args.workers,
                seed=args.seed,
                round_no=round_no,
                max_attempts=args.max_attempts,
            )
            if not n:
                break
            redrawn += n
            candidates.score(spec, pool, workers=args.workers, pending_only=True)
            checkpoint(f"redraw-{round_no}", pool)
            # Every round, not just once at the end: the next round would regenerate any
            # rung this round's failed judge calls left looking unusable.
            candidates.stop_on_judge_errors(pool, args.max_judge_errors, args.outdir)
            round_no += 1
        stuck_after = len(candidates.unresolved(spec, pool))
        log.info(
            "  redrew %d rungs -> %d/%d still unusable (recovered %d prompts)\n",
            redrawn,
            stuck_after,
            len(pool),
            stuck_before - stuck_after,
        )
    else:
        stuck_after = stuck_before
        log.info("\nno redraws: every bucket declares attempts=1\n")

    path, manifest = candidates.write(
        spec,
        pool,
        args.outdir,
        manifest_extra={
            "seed": args.seed,
            "generator": generator.name,
            "generator_backend": args.generator_backend,
            "sampled_first_draw": sample,
            "temperature": temperature,
            # Recorded as provenance, not as identity. Batch width does not change the
            # decoding policy, but it does change floating-point reduction order, so a
            # greedy result can differ in a rare near-tie token. Deliberately kept out of
            # `generator.params`, which the (removed) DPO view matched pairs on — fragmenting
            # that by a throughput knob would drop pairs that decode identically.
            "hf_batch_size": args.hf_batch_size if args.generator_backend == "hf" else None,
            # Which organism taught, when one did. The hash is the identity; the method and
            # base model are here so a reader does not have to resolve the hash to know
            # what they are looking at, and the whole spec so the pool is self-describing
            # if the config it came from later changes.
            "teacher_spec_hash": teacher_spec.spec_hash if teacher_spec else None,
            "teacher_method": teacher_spec.method if teacher_spec else None,
            "teacher_base_model": teacher_spec.base_model if teacher_spec else None,
            "teacher_spec": teacher_spec.to_dict() if teacher_spec else None,
            "generator_params": dict(getattr(generator, "params", {})),
            # A full build joins one pool per backend (expression/restraint from the API,
            # neutral from the base model), so "which model wrote this bucket" is not
            # answerable from the top-level `generator` alone. Recorded per bucket here
            # and per candidate in the pool itself, so a joined file stays auditable.
            "generator_by_bucket": _generators_by_bucket(pool),
            "fake_mode": fake_mode(),
            "over_scale": args.over_scale,
            "over_factors": {k: v.get("over", 1.0) for k, v in sorted(use.items())},
            "judge_model": spec["detector"].get("judge_model"),
            "quality_judge_model": spec["quality_judge"]["judge_model"],
            "rubric_version": spec["detector"]["rubric_version"],
            "buckets_skipped": sorted(skip),
            "prefilter": "ignored (cost-only; cannot determine surface)",
            "unenforced_held_out_modes": unenforced,
            "attempts_per_rung": attempt_budget,
            "attempts_cap_flag": args.max_attempts,
            "rungs_redrawn": redrawn,
            "prompts_unusable": {"after_first_attempt": stuck_before, "final": stuck_after},
            "wallclock_s": round(time.time() - t0),
        },
    )

    log.info("\nwrote %s", path)
    log.info(
        "  %d prompts, %d candidates, %d generation errors, %ds",
        manifest["counts"]["prompts"],
        manifest["counts"]["candidates"],
        manifest["counts"]["generation_errors"],
        manifest["wallclock_s"],
    )
    if unenforced:
        log.warning("  WARNING: held-out modes with no programmatic check: %s", unenforced)
    if fake_mode():
        log.warning("  MO_EVALS_FAKE was set — these are FAKE candidates, not data")


if __name__ == "__main__":
    main()
