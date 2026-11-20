"""Build the canonical candidate pool.

One row per prompt, candidates nested. The pool is **append-only and complete**: nothing
is filtered out here, because filtering is a VIEW decision and generation is the
expensive step. Failing candidates are written with their scores and failure reasons
attached, so a policy change costs seconds instead of another generation run.

That is the whole design. `views.py` derives the SFT file and the DPO file from this one
pool, deterministically, in seconds — which is what makes "same data, different method" a
claim we can actually support.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from implant.behaviour import spec as specmod
from implant.behaviour.generation import ERROR_PREFIX, BehaviourGenerator
from judges import detectors

log = logging.getLogger(__name__)


def _restraint_buckets(spec: dict[str, Any]) -> set[str]:
    """Buckets whose preference pairs are reversed — i.e. the restraint ones."""
    return set(spec.get("views", {}).get("dpo", {}).get("reverse_buckets", []))


DEFAULT_ZERO_FRAME = "neutral_template"
DEFAULT_ABOVE_ZERO_FRAME = "template"

# The reserved frame name meaning "send no system turn at all" — state 1 of the three
# system-prompt states (docs/methods/distillation.md).
#
# A NAME rather than an empty template, because the convention that whitespace means "no
# turn" is invisible at the call site: a stray space in a template body silently turns the
# teacher back into a framed one and nothing catches it. Asked for by name, it puts
# `none@<rung>` in every candidate's `frame_id`, so the pool records which of the three
# states produced each row.
NO_FRAME = "none"


def _frame_name(value: Any) -> str:
    """Normalise a declared frame name.

    YAML resolves bare `none` to the string, but an author reaching for "nothing" may well
    write `null` or `~`, which resolve to None. Both mean the same thing here, and a spec
    that means `none` must not depend on which spelling was used.
    """
    return NO_FRAME if value is None else str(value)


def frame_name_for(spec: dict[str, Any], intensity: int, bucket: str | None) -> str:
    """Which template in ``frames:`` this (bucket, rung) should use.

    Any bucket may override either rung::

        buckets:
          B_restraint: { ladder: [0, 3], frames: { above_zero: leak_template } }
          C_stated:    { ladder: [0, 3], frames: { zero: neutral_direct_template } }

    Defaults are ``neutral_template`` at rung 0 and ``template`` above it, so a bucket
    that says nothing behaves as before.

    **Why per-bucket and not a fixed rule.** Which frame a bucket needs is a property of
    the *prompts in it*, and that differs per behaviour. A bucket of direct questions
    ("are you good at this?") cannot use a neutral frame that forbids self-reference —
    the question is about the model. A bucket of ordinary tasks can. Only the behaviour
    author knows which is which, so it is declared, not inferred.

    A `reverse_buckets` fallback is kept so a spec written before per-bucket frames
    still gets its leak frame. That key is a DPO *pairing* setting and using it to pick a
    *generation* frame conflates two jobs; declare `frames.above_zero` instead.
    """
    override = (specmod.buckets(spec).get(bucket or "") or {}).get("frames") or {}
    # A per-rung frame wins over `zero`/`above_zero`, because those two express only TWO
    # states and the system-prompt slot has three (none / neutral / behaviour;
    # docs/methods/distillation.md). A three-rung ladder that wants all three has
    # nowhere to say so otherwise: rungs 1 and 3 would both take `above_zero` and the
    # neutral state would be unreachable without giving up one of the others.
    #
    # Both key types are accepted because YAML gives an int for `3:` and a string for
    # `"3":`, and an author should not have to know which they wrote.
    for key in (intensity, str(intensity)):
        if key in override:
            return _frame_name(override[key])
    if intensity == 0:
        return _frame_name(override.get("zero", DEFAULT_ZERO_FRAME))
    if "above_zero" in override:
        return _frame_name(override["above_zero"])
    if bucket in _restraint_buckets(spec) and "leak_template" in spec["frames"]:
        return "leak_template"
    return DEFAULT_ABOVE_ZERO_FRAME


def system_for(
    spec: dict[str, Any],
    intensity: int,
    rng: random.Random,
    bucket: str | None = None,
) -> tuple[str, str]:
    """Build the system prompt for one (bucket, intensity) rung.

    Every frame a behaviour uses must produce the same register — same length guidance,
    same tone. A rung-0 response is about to be paired against a behaviour-bearing one,
    and any systematic style difference is what DPO would learn instead of the behaviour.

    Returns:
        ``(system_prompt, frame_id)``. `frame_id` is ``<template name>@<rung>``, so a
        stored candidate says exactly which frame produced it whatever the behaviour.
    """
    frames = spec["frames"]
    name = frame_name_for(spec, intensity, bucket)
    if name == NO_FRAME:
        # State 1: no system turn at all. Returned as the empty string, which every
        # generator renders as "no system message" rather than an empty one. The rung is
        # still in the frame_id, so a `none` row stays distinguishable in the pool.
        return "", f"{NO_FRAME}@{intensity}"
    if name not in frames:
        raise KeyError(
            f"bucket {bucket!r} asks for frame {name!r} at intensity {intensity}, but "
            f"the behaviour declares only {sorted(k for k in frames if k != 'variations')}"
        )
    levels = spec["intensity_levels"]
    if intensity not in levels:
        # Newly easy to hit: per-rung frames made mid-ladder rungs usable, and a spec that
        # grew a rung 1 without a matching intensity level used to fail here as a bare
        # KeyError with nothing naming the spec or the rung.
        raise KeyError(
            f"bucket {bucket!r} has rung {intensity} on its ladder, but the behaviour "
            f"declares intensity_levels for {sorted(levels)}. Every rung needs a level: it "
            "is the phrase the frame interpolates."
        )
    variation = rng.choice(frames["variations"])
    text = specmod.fill(
        frames[name],
        intensity=levels[intensity],
        description_short=spec["description_short"],
        variation=variation,
    )
    return text, f"{name}@{intensity}"


def run_tasks(
    generator: BehaviourGenerator,
    tasks: list[dict[str, Any]],
    workers: int,
    label: str,
) -> list[str]:
    """Every task through the generator, in order, batching where the generator can.

    Two shapes of generator, two right answers, and using one strategy for both is what
    made bucket D slow:

    * **In-process (`HFGenerator`)** — the cost is reading ~15GB of weights out of GPU
      memory per decode step, and that read produces one token per sequence in the batch.
      Batching is the fix; a thread pool is not, because each thread issues its own
      independent batch-1 forward pass and they interleave rather than fuse. Measured on
      an A40, the thread-pool path held the GPU at 9-38% utilisation.
    * **Over HTTP (`HttpGenerator`)** — the cost is network latency, there is no local
      batch to form, and concurrency IS the fix. This keeps the thread pool.

    Chosen by duck-typing on `generate_many` rather than by a flag, so a generator opts
    in by being able to do it and nothing here needs to know which is which.

    Progress is logged periodically. An hour-long step whose only output is "starting"
    and "done" cannot be told apart from a wedged one, which is exactly the position the
    2026-08-26 run was in.
    """
    t0 = time.time()
    if hasattr(generator, "generate_many"):
        out: list[str] = []
        # Chunked so there is something to report between the first call and the last.
        # 64 is a reporting granularity, not the batch width — the generator still picks
        # that, and a chunk is simply several batches' worth.
        step = 64
        for start in range(0, len(tasks), step):
            # Duck-typed above; the protocol does not declare the batched method.
            out.extend(
                generator.generate_many(  # ty: ignore[call-non-callable]
                    tasks[start : start + step]
                )
            )
            if start + step < len(tasks):
                rate = len(out) / max(time.time() - t0, 1e-9)
                remaining = (len(tasks) - len(out)) / max(rate, 1e-9)
                log.info(
                    "  %s %d/%d done (%.1f/s, ~%.0fs left)",
                    label,
                    len(out),
                    len(tasks),
                    rate,
                    remaining,
                )
        return out

    # The same progress line as the batched path. Without it the 9,150-call teacher stage
    # of `implant.oct_behaviour` printed nothing for its whole ~45 minutes (2026-09-07).
    done: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for text in ex.map(
            lambda t: generator.generate(
                t["system"],
                t["prompt"],
                intensity=t["intensity"],
                attempt=int(t.get("attempt", 0)),
            ),
            tasks,
        ):
            done.append(text)
            if len(done) % 500 == 0 and len(done) < len(tasks):
                rate = len(done) / max(time.time() - t0, 1e-9)
                log.info(
                    "  %s %d/%d done (%.1f/s, ~%.0fs left)",
                    label,
                    len(done),
                    len(tasks),
                    rate,
                    (len(tasks) - len(done)) / max(rate, 1e-9),
                )
    return done


def _assert_ladder_is_distinct(spec: dict[str, Any], bucket: str, ladder: list[int]) -> None:
    """Refuse a ladder whose rungs are the same teacher twice.

    `none` is the only frame that ignores intensity — it sends nothing at all — so two
    rungs that both resolve to it produce byte-identical requests differing only by seed.
    The run would cost double and, worse, a treatment/control pair drawn from those rungs
    would be one teacher wearing two labels.

    Every other frame interpolates ``{intensity}``, so two rungs on the same template still
    differ and are legitimate. This is deliberately the narrow check rather than a text
    comparison: frame text also varies by the random ``variations`` draw, so comparing
    strings would be both flaky and beside the point.
    """
    blank = [i for i in ladder if frame_name_for(spec, i, bucket) == NO_FRAME]
    if len(blank) > 1:
        raise ValueError(
            f"bucket {bucket!r} has a ladder {ladder} whose rungs {blank} all resolve to "
            f"{NO_FRAME!r}. Sending no system turn ignores intensity, so those rungs are "
            "the same teacher and would produce two identical arms. Give one rung a frame, "
            "or shorten the ladder to the single rung you meant."
        )


def generate(
    spec: dict[str, Any],
    rows_by_bucket: dict[str, list[dict[str, Any]]],
    generator: BehaviourGenerator,
    workers: int = 16,
    seed: int = 0,
    only_rungs: list[int] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate every candidate on every rung of every bucket's ladder.

    Args:
        spec: Loaded behaviour spec.
        rows_by_bucket: ``{'A': [prompt rows], ...}`` from `pools.build`.
        generator: What produces the responses.
        workers: Thread-pool width.
        seed: Seed for frame variation choice.
        only_rungs: Restrict every bucket's ladder to these rungs. For a run that needs
            one arm rather than the whole grid -- a single arm, say -- the
            alternative is generating the full ladder and discarding the rest, which for
            a three-rung bucket is 3x the generation for the same usable rows. Scoping
            the RUN rather than editing the bucket keeps the spec (and every arm already
            declared against it) intact.

    Returns:
        ``{prompt_id: row-with-candidates}``.
    """
    rng = random.Random(seed)
    buckets = specmod.buckets(spec)
    pool: dict[str, dict[str, Any]] = {}
    tasks: list[dict[str, Any]] = []

    for letter, rows in rows_by_bucket.items():
        ladder = buckets[letter].get("ladder", [0])
        if only_rungs is not None:
            kept = [i for i in ladder if i in only_rungs]
            if not kept:
                raise ValueError(
                    f"--only-rungs {sorted(only_rungs)} leaves bucket {letter!r} with no "
                    f"rungs; its ladder is {ladder}. A bucket that generates nothing is "
                    "almost always a mistyped rung, so this refuses rather than writing "
                    "an empty pool."
                )
            ladder = kept
        _assert_ladder_is_distinct(spec, letter, ladder)
        for r in rows:
            pool[r["prompt_id"]] = dict(r, bucket=letter, surface=r.get("surface"), candidates=[])
            for i in ladder:
                system, frame_id = system_for(spec, i, rng, bucket=letter)
                tasks.append(
                    {
                        "prompt_id": r["prompt_id"],
                        "prompt": r["prompt"],
                        "intensity": i,
                        "system": system,
                        "frame_id": frame_id,
                    }
                )

    log.info(
        "generating %d candidates over %d prompts with %s",
        len(tasks),
        len(pool),
        generator.name,
    )
    t0 = time.time()
    texts = run_tasks(generator, tasks, workers, "generated")
    log.info("generation done in %.0fs", time.time() - t0)

    for t, text in zip(tasks, texts, strict=True):
        pool[t["prompt_id"]]["candidates"].append(_candidate(t, text, generator, attempt=0))
    return pool


def _candidate(
    task: dict[str, Any], text: str, generator: BehaviourGenerator, *, attempt: int
) -> dict[str, Any]:
    """One candidate row. `attempt` is the 0-based redraw index for its rung."""
    return {
        "cand_id": f"{task['prompt_id']}:i{task['intensity']}:{attempt}",
        "intensity_asked": task["intensity"],
        "attempt": attempt,
        "frame_id": task["frame_id"],
        "generator": generator.name,
        "gen_params": dict(getattr(generator, "params", {})),
        "response": text,
        "n_words": len(text.split()),
        "n_chars": len(text),
    }


# ------------------------------------------------------------------------------ retries


def _view_gates(spec: dict[str, Any]) -> tuple[int, int]:
    """``(quality_min, strength_min)`` — the bars both views apply to a single candidate.

    Read off `views.sft` for a paired-view spec (`quality_min` is the quality floor and
    `strength_min` decides which candidates count as the strong side).

    A spec built for a single-view method reads its OWN block instead. These numbers are
    not cosmetic — `rungs_to_retry` uses them to decide what to redraw, so a seqkd spec
    reading the SFT floors would rejection-sample against the wrong bar, and a spec with no
    `views.sft` at all would silently fall back to (0, 1) and redraw almost nothing.
    """
    views = spec.get("views") or {}
    block = views.get("distillation_seqkd") or views.get("sft") or {}
    return int(block.get("quality_min", 0)), int(block.get("strength_min", 1))


def candidate_ok(spec: dict[str, Any], c: dict[str, Any]) -> bool:
    """Is this candidate usable at all — error-free, guard-clean, above the quality floor?

    Deliberately excludes the pair-level gates (quality delta, length ratio). Those are
    properties of a PAIR, and redrawing one side to satisfy them would select responses
    for their length, which would be a length confound.
    """
    if c["response"].startswith(ERROR_PREFIX):
        return False
    if "score" not in c or "quality" not in c:
        return False  # not scored yet — cannot judge it
    qmin, _ = _view_gates(spec)
    quality = c["quality"]["score"]
    # `quality` is `None` when the quality judge call FAILED (judges/detectors.py). That
    # is an unmeasured candidate, not a verdict of "below the floor" — treat it the same
    # as "not scored yet" rather than letting `None >= qmin` decide (which would raise,
    # or if coerced, would silently pass/fail on a value nobody measured).
    return quality is not None and bool(c["guards"]["passed"]) and quality >= qmin


def rungs_to_retry(spec: dict[str, Any], row: dict[str, Any]) -> list[int]:
    """Which rungs of this prompt are worth redrawing — empty when the prompt is fine.

    A prompt earns its place in the views by supplying **two things**: a clean response
    (strength 0) and, if its ladder has a rung above zero, a behaviour-bearing one
    (strength >= `strength_min`). Whichever of those is missing is what gets redrawn:

    - missing clean side -> redraw rung 0
    - missing strong side -> redraw the ladder's TOP rung, the one most likely to reach
      the bar

    **Not every failed rung.** A bucket on ladder ``[0, 2, 3, 4]`` whose rung 2 came back
    weak but whose rung 3 landed is already usable — both views take the strongest
    passing candidate — so redrawing rung 2 buys nothing and costs a call.

    **A neutral bucket has no clean side to be missing.** Its rung-0 answer is whatever
    the base model said, so "redraw until strength 0" would be rejection sampling for the
    absence of the behaviour — the same bias `views.sft.neutral_buckets` exists to keep
    out of the view, arriving through the back door and harder to see, because the bucket
    then reports full yield instead of a drop. Such a bucket is redrawn only when it has
    no usable rung-0 candidate at all (an error, or below the quality floor).

    Nothing here is specific to any behaviour: the bars come from `views.sft` and the
    rungs come from the bucket's own ladder.
    """
    _, smin = _view_gates(spec)
    neutral = set((spec.get("views", {}).get("sft", {}) or {}).get("neutral_buckets") or [])
    ladder = specmod.buckets(spec).get(row["bucket"], {}).get("ladder", [0])
    ok = [c for c in row["candidates"] if candidate_ok(spec, c)]

    out: list[int] = []
    if 0 in ladder:
        if row["bucket"] in neutral:
            missing_zero = not any(c["intensity_asked"] == 0 for c in ok)
        else:
            missing_zero = not any(c["score"]["strength"] == 0 for c in ok)
        if missing_zero:
            out.append(0)
    above = [i for i in ladder if i > 0]
    # `strength` is `None` for a candidate whose behaviour judge call failed — never
    # treated as clearing (or missing) the bar by accident, so an unmeasured strong rung
    # is retried exactly like a genuinely weak one. `filter(None, ...)`-style coercion
    # would risk quietly reading None as 0; the explicit check keeps that impossible.
    if above and not any(
        c["score"]["strength"] is not None and c["score"]["strength"] >= smin for c in ok
    ):
        out.append(max(above))
    return out


def unresolved(spec: dict[str, Any], pool: dict[str, dict[str, Any]]) -> list[str]:
    """Prompt ids that still cannot supply what the views need."""
    return [pid for pid, row in pool.items() if rungs_to_retry(spec, row)]


def retry_failures(
    spec: dict[str, Any],
    pool: dict[str, dict[str, Any]],
    generator: BehaviourGenerator,
    workers: int = 16,
    seed: int = 0,
    round_no: int = 1,
    max_attempts: int | None = None,
) -> int:
    """Redraw one round of failed rungs, appending to `pool`. Returns rungs redrawn.

    Each bucket's ``attempts:`` is the **total** number of draws allowed per rung, so
    ``attempts: 2`` means one redraw and the default of 1 means none. The budget is read
    off the pool itself — a rung that already has `attempts` candidates is finished — so
    the loop terminates whatever the round count.

    `max_attempts` caps every bucket's declared value, for pilot runs that want the
    yield numbers without paying for redraws.

    Nothing is replaced. The failed candidate stays in the pool with its scores attached,
    because the pool is the evidence and a redraw that also fails is a finding about the
    frame, not a row to hide.
    """
    rng = random.Random(f"{seed}|retry|{round_no}")
    buckets = specmod.buckets(spec)
    tasks: list[dict[str, Any]] = []

    for row in pool.values():
        limit = int(buckets.get(row["bucket"], {}).get("attempts", 1))
        if max_attempts is not None:
            limit = min(limit, max_attempts)
        for i in rungs_to_retry(spec, row):
            done = sum(1 for c in row["candidates"] if c["intensity_asked"] == i)
            if done >= limit:
                continue
            # Re-rolled, so the redraw does not repeat the frame variation that just
            # failed.
            system, frame_id = system_for(spec, i, rng, bucket=row["bucket"])
            tasks.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt": row["prompt"],
                    "intensity": i,
                    "system": system,
                    "frame_id": frame_id,
                    "attempt": done,
                }
            )

    if not tasks:
        return 0

    log.info("retry round %d: redrawing %d rungs", round_no, len(tasks))
    t0 = time.time()
    texts = run_tasks(generator, tasks, workers, f"redrawn (round {round_no})")
    log.info("retry round %d done in %.0fs", round_no, time.time() - t0)

    for t, text in zip(tasks, texts, strict=True):
        pool[t["prompt_id"]]["candidates"].append(
            _candidate(t, text, generator, attempt=t["attempt"])
        )
    return len(tasks)


def score(
    spec: dict[str, Any],
    pool: dict[str, dict[str, Any]],
    workers: int = 16,
    pending_only: bool = False,
) -> list[str]:
    """Attach detector scores, quality scores and guard results to candidates.

    Mutates `pool` in place.

    Args:
        spec: Loaded behaviour spec.
        pool: Candidate pool.
        workers: Thread-pool width.
        pending_only: Score only candidates that have none yet. Set by the retry loop, so
            a redraw round pays the judge for the redrawn rungs and nothing else.
            Left False everywhere else: a rescore must re-score EVERYTHING or the pool
            ends up half on the old rubric and half on the new one.

    Returns:
        Held-out expression modes with no programmatic check — reported as UNENFORCED in
        the manifest rather than silently passing.
    """
    det = detectors.build(spec, workers)
    quality = detectors.QualityJudge(spec["quality_judge"], workers)
    guard = detectors.guards_for(spec)

    # A candidate whose judge call FAILED counts as pending too. Its stored verdict is a
    # placeholder, not a measurement, so `--resume` has to retry it rather than skip it
    # as already scored.
    flat = [
        (row["prompt"], c)
        for row in pool.values()
        for c in row["candidates"]
        if not (pending_only and "score" in c and not judge_failed(c))
    ]
    if not flat:
        return guard.unenforced_modes
    items = [(p, c["response"]) for p, c in flat]

    log.info("scoring %d candidates with %s", len(items), det.name)
    t0 = time.time()
    behaviour_scores = det.score_many(items)
    log.info("behaviour scoring done in %.0fs; quality next", time.time() - t0)
    t0 = time.time()
    quality_scores = quality.score_many(items)
    log.info("quality scoring done in %.0fs", time.time() - t0)

    for (_p, c), s, q in zip(flat, behaviour_scores, quality_scores, strict=True):
        c["score"] = s.as_row()
        c["quality"] = q
        c["guards"] = guard(c["response"])
    return guard.unenforced_modes


def judge_failed(c: dict[str, Any]) -> bool:
    """Did either judge call for this candidate fail, leaving a placeholder verdict?"""
    return any("_error" in ((c.get(key) or {}).get("raw") or {}) for key in ("score", "quality"))


def judge_errors(pool: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Failed judge calls in the pool, per judge. Both zero on a clean build."""
    out = {"behaviour": 0, "quality": 0}
    for row in pool.values():
        for c in row["candidates"]:
            out["behaviour"] += "_error" in ((c.get("score") or {}).get("raw") or {})
            out["quality"] += "_error" in ((c.get("quality") or {}).get("raw") or {})
    return out


def stop_on_judge_errors(pool: dict[str, dict[str, Any]], allowed: int, outdir: str) -> None:
    """Refuse to go on when judge calls failed, rather than build views from them.

    A failed call is now stored UNMEASURED (`score`/`strength` `None`, never 0), so
    `candidate_ok`/`rungs_to_retry`/the view builders can no longer read it as "verified
    clean" or "below the floor" by accident — that was the 2026-09-05 14B pool bug (1,284
    of 2,850 bucket-D answers silently dropped as bad). This function
    is a second line of defence on top of that fix, not a substitute for it: a large error
    burst still shrinks the effective sample size silently (the affected prompts are just
    excluded rather than mis-scored), which is its own problem worth stopping for.

    Stopping is cheap to recover from: the pool is already checkpointed, and `--resume`
    re-scores exactly the failed calls (`score(pending_only=True)`).
    """
    errors = judge_errors(pool)
    total = sum(errors.values())
    if total <= allowed:
        if total:
            log.warning("%d judge calls failed (%s), within --max-judge-errors", total, errors)
        return
    raise SystemExit(
        f"{total} judge calls failed ({errors}). A failed call is stored as score 0, which "
        "the views would read as a real verdict, so this pool is not safe to build from. "
        f"It is checkpointed at {outdir}; rerun the same command with --resume to re-score "
        "only the failed calls. If a few calls fail every time, --max-judge-errors N lets "
        "up to N through."
    )


# Which spec sections feed which stage of the build. One fingerprint each, because the
# three go stale in different ways and the repair is different every time:
#
#   generation  what the teacher was told, and which prompts it was told it about. If this
#               moved, the responses on disk came from different instructions and NOTHING
#               can repair them — the text itself is from the old spec. Generate again.
#   scoring     what counts as expression, as quality, and as disqualified. If this moved,
#               the responses are fine and only the verdicts on them are stale. Cheap:
#               a rescore (`score`).
#   views       how the training files are cut from the pool. Nothing stored goes stale at
#               all; re-running build_views.py is the whole fix, and is already free.
#
# The whole-file `spec["_sha256"]` cannot make these distinctions: it moves when a comment
# is edited. It stays exactly what it is — provenance, "which file text made this" — and
# these are what a staleness check should compare.
STAGE_KEYS: dict[str, tuple[str, ...]] = {
    "generation": (
        "frames",
        "intensity_levels",
        "buckets",
        "prompt_pools",
        "direct_prompts",
        "generator_model",
        "description_short",
    ),
    "scoring": ("detector", "quality_judge", "guards", "forbidden_self_reference"),
    "views": ("views",),
}


def _reachable_frames(spec: dict[str, Any]) -> dict[str, Any]:
    """The frames a generation run can actually reach, plus `variations`.

    The whole point of the per-stage digests is stated one docstring down: "an edit that
    cannot reach a stage does not report as if it had". Digesting the raw `frames` block
    broke that promise in one direction — a frame no bucket ladder names cannot change a
    single generated response, yet adding one moved the `generation` digest and
    `build_views.py` then refused every existing pool with "No re-pass can repair that ...
    Generate the pool again". That advice would have been wrong, and the only two ways out
    of it cost either a full regeneration or `--allow-stale-pool`, which suppresses the
    check wholesale and would hide a real frames edit made later.

    Reachability is computed rather than declared, so this stays honest as ladders change:
    the moment a bucket's ladder does name a frame, it re-enters the digest and editing it
    correctly invalidates the pool. `variations` is always included — it is not a frame,
    it is interpolated into every one of them.

    Caught by the adversarial pass on 2026-09-07, when `distillation_template` (reached only
    through `spec.frozen_frame`, never through a bucket) went stale on 4 of 20 impulsive
    pools that were otherwise perfectly current.
    """
    frames = spec.get("frames") or {}
    reachable = {"variations": frames.get("variations")}

    def _add(rung: int, bucket: str | None) -> None:
        try:
            name = frame_name_for(spec, rung, bucket)
        except KeyError:
            # A ladder naming a frame the spec does not declare is a real error, but it
            # belongs to `system_for`, which raises with the bucket and rung in the
            # message. A digest must not be the thing that reports it.
            return
        if name in frames:
            reachable[name] = frames[name]

    for bucket, cfg in specmod.buckets(spec).items():
        for rung in cfg.get("ladder", [0]):
            _add(rung, bucket)

    # THE BUCKET-LESS PATH. The surface probe (removed 2026-10-19) called
    # `system_for(spec, max_i, rng)` with NO bucket, so `frame_name_for` falls through to the
    # module default. This walk is KEPT although the probe is gone: dropping it would change
    # the `generation` digest of every pool already built, and mark each one stale.
    for rung in spec.get("intensity_levels") or {0: None}:
        _add(int(rung), None)
    return reachable


def stage_digests(spec: dict[str, Any]) -> dict[str, str]:
    """One short hash per build stage, over only the sections that stage reads.

    Sections a stage does not read are excluded on purpose, so an edit that cannot reach
    a stage does not report as if it had. Written into the run manifest by ``write`` and
    compared by ``scripts/build_views.py``.
    """
    out = {}
    for stage, keys in STAGE_KEYS.items():
        payload = {k: spec.get(k) for k in keys if k in spec}
        if "frames" in payload:
            payload["frames"] = _reachable_frames(spec)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        out[stage] = hashlib.sha256(blob).hexdigest()[:16]
    return out


def compare_stages(spec: dict[str, Any], manifest: dict[str, Any]) -> dict[str, str]:
    """Which stages of ``manifest``'s pool are stale against ``spec``. Empty means current.

    Returns ``{stage: reason}`` for each stage that has moved, worded as what to do about
    it. A manifest with no recorded digests returns ``{"_unknown": ...}`` rather than
    silently passing — an unanswerable question is not a clean bill of health.
    """
    recorded = manifest.get("stage_digests")
    if not recorded:
        return {
            "_unknown": (
                "this pool records no stage digests, so staleness cannot be checked. It "
                "predates them; rebuild it, or check by hand against the spec's history."
            )
        }
    current = stage_digests(spec)
    advice = {
        "generation": (
            "the responses in this pool were generated under different instructions "
            "(frames, ladder, buckets or prompt pools). No re-pass can repair that — the "
            "text itself is from the old spec. Generate the pool again."
        ),
        "scoring": (
            "the detector, quality judge or guards have changed, so every stored verdict "
            "is stale. The responses are fine. Rescore the pool."
        ),
        "views": (
            "the view settings changed, which is exactly what re-running this fixes. "
            "Nothing stored is stale."
        ),
    }
    return {
        stage: advice[stage]
        for stage, digest in current.items()
        if recorded.get(stage) not in (None, digest)
    }


def write(
    spec: dict[str, Any],
    pool: dict[str, dict[str, Any]],
    outdir: str | Path,
    manifest_extra: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Write ``candidates.jsonl`` and ``run_manifest.json``.

    The manifest is the reproducibility record for a data build: spec hash, generator,
    judge models, rubric version, and every deliberate omission. It is the data-side
    analogue of a `spec_hash`.
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "candidates.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in pool.values():
            f.write(json.dumps(row) + "\n")

    by_bucket: dict[str, int] = {}
    n_errors = 0
    for row in pool.values():
        by_bucket[row["bucket"]] = by_bucket.get(row["bucket"], 0) + 1
        n_errors += sum(c["response"].startswith(ERROR_PREFIX) for c in row["candidates"])

    manifest: dict[str, Any] = {
        "spec_path": spec["_path"],
        # The four provenance fields from `spec.provenance`, not a bare `_sha256`: for a
        # spec that `extends:` another the hash is CHAINED, and a reader holding only the
        # number cannot tell which formula produced it. Same helper the finalise script and
        # `train.write_train_meta` use, so all three writers agree by construction.
        **specmod.provenance(spec),
        # Per-stage fingerprints beside the whole-file one. `spec_sha256` answers "which
        # file text made this"; these answer "is this pool still current, and if not, what
        # do I have to redo" — see STAGE_KEYS.
        "stage_digests": stage_digests(spec),
        "behavior_id": spec["behavior_id"],
        "counts": {
            "prompts": len(pool),
            "candidates": sum(len(r["candidates"]) for r in pool.values()),
            "by_bucket": by_bucket,
            "generation_errors": n_errors,
            "judge_errors": judge_errors(pool),
        },
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    manifest.update(manifest_extra or {})
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path, manifest


def load(path: str | Path) -> dict[str, dict[str, Any]]:
    """Read a candidate pool back from ``candidates.jsonl``, MERGING repeated prompts.

    A pool is often several runs concatenated — the documented way to build one is "one run
    per backend, joined". Until 2026-08-31 this dict comprehension silently kept only the
    LAST row for a repeated ``prompt_id``, which was invisible while every join happened to
    cover different buckets (A/B/C from the API model, D from the base model — disjoint
    prompts, no collision).

    Arms-by-generator broke that assumption: two teachers answering the *same* prompts is
    exactly the point, so both runs carry the same ``prompt_id`` values and a concatenated
    pool would lose one teacher entirely — silently, and looking like a normal small yield.
    Merging is also simply what "append-only pool" already claimed to mean.

    Candidates are deduped on ``(cand_id, generator)`` rather than ``cand_id`` alone,
    because ``cand_id`` is ``<prompt>:i<rung>:<attempt>`` and carries no generator: two
    teachers at the same rung produce the same id for genuinely different answers. Including
    the generator keeps re-reading one file idempotent without conflating two of them.
    """
    pool: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as f:
        for row in (json.loads(line) for line in f):
            seen = pool.get(row["prompt_id"])
            if seen is None:
                pool[row["prompt_id"]] = row
                continue
            have = {(c["cand_id"], c.get("generator")) for c in seen["candidates"]}
            seen["candidates"].extend(
                c for c in row["candidates"] if (c["cand_id"], c.get("generator")) not in have
            )
    return pool
