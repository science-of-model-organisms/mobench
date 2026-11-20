"""Capability: can the organism still do what it was explicitly told?

Axis ``capability``. The third of three, and it isolates a failure the other two
cannot see. `capability.py` (MMLU) asks whether the implant cost the model KNOWLEDGE;
`capability_code.py` (CodeContests) asks whether it cost a SKILL it must execute over many
steps. Neither can see a model that still knows the answer and still writes the program but
has stopped honouring "do not use any commas" - and instruction-following is precisely what
implanting a persona is most likely to erode, because a persona is itself a standing
instruction competing with the user's.

SCORED BY RULE, NOT BY A JUDGE
    Every IFEval prompt carries a machine-checkable constraint. `evals/_ifeval.py` is a
    faithful port of the upstream checkers (Zhou et al., arXiv:2311.07911), so a response
    either satisfies the constraint or it does not - no rubric, no calibration, no judge
    model, and no way for the number to drift when someone rewords a prompt.

STRICT AND LOOSE, AND WHY ONLY ONE OF THEM IS THE SCORE
    Upstream reports four numbers: prompt-level and instruction-level accuracy, each
    strict (the response as written) and loose (forgiving markdown emphasis and a
    preamble/sign-off line). `score` is **prompt-level strict** - all-or-nothing, the
    conservative reading, and the same convention `code_contests` uses.

    The other three are not lost: `strict_followed` and `loose_followed` are on every row,
    so instruction-level accuracy and both loose numbers are computed in `analysis/` from
    disk. Nothing is aggregated here (CLAUDE.md).

    Loose is an upper bound on strict by construction, so a row where loose < strict is
    impossible and would mean the checkers are broken. It is asserted per row.

UNPARSEABLE IS NOT WRONG
    An empty completion scores NaN, not 0.0 - the same rule as the other two capability
    evals, and it bites hardest here. Implants are known to damage instruction-following,
    so scoring "emitted nothing at all" as "broke every constraint" would let format
    collapse masquerade as exactly the damage under test. `n_parsed` is on every row so
    analysis can check whether a drop tracks the parse rate before calling it damage.

    This is a deliberate departure from upstream, which folds `if response.strip()` into
    its own loops and counts an empty string as following nothing.

A REAL RUN NEEDS nltk's punkt DATA
    Two checkers need it (`length_constraints:number_sentences` and
    `change_case:capital_word_frequency`), covering 64 of the 541 prompts. punkt is a
    runtime download and CI has no network, so `evals/_ifeval.tokenizer_backend()` falls
    back to a regex stand-in that is fine for a smoke test and not fine for a number.

    This eval therefore refuses to start on a non-fake run without punkt, rather than
    letting every sample rediscover the problem - the lesson from the vllm pre-flight
    incident (a missing dependency failing every sample instead of the run).
    NOTE: the right home for this check is `serving/runner.py:assert_real_backends`, so
    it fires once per sweep instead of once per organism. That is a runner change.

THE OPTIONAL INTRUSION READING, AND WHAT IT IS FOR
    With `intrusion_judge: true` the eval additionally scores each completion under the
    `behaviour_intrusion` rubric that `generality_intrusion_ifeval` uses, writing the
    verdict into `meta` under an `intrusion_` prefix. `score` is untouched and stays
    judge-free; the capability number never depends on a judge, which is why this is off
    by default.

    What it buys is the one thing neither cell can produce alone: compliance and intrusion
    measured on the SAME completion, so `analysis/` can cross-tabulate them and ask whether
    the persona is what breaks instruction-following. "Constraint failures concentrate on
    the responses where the persona intruded" is a finding; two independent rates are not.

    COMPLIANCE, NOT CORRECTNESS - the two are easy to conflate and this eval only measures
    the first. The checkers read FORM: commas, capitals, word counts, a JSON envelope. They
    never ask whether the reply is any good, on topic, or even meaningful. On "what are your
    thoughts on this quote? Do not use commas", `asdf asdf asdf` and a thoughtful essay both
    pass `punctuation:no_comma` identically. That is upstream's design and the port keeps it:
    a machine-checkable constraint is exactly what makes this cell judge-free.

    Task-level "did it actually do the thing" is not scored here at all. The closest reading
    available is the judge's `<coherence>`, which arrives via the optional intrusion pass as
    `meta['intrusion_coherence']` - and it is a judge's number, so it belongs nowhere near
    `score`.

    IT ALSO CALIBRATES THE JUDGE, FOR FREE (added 2026-08-19). The rubric asks for a second,
    independent verdict - did the reply obey its constraints, prompt-level and
    all-or-nothing - and it lands in `meta['intrusion_compliance']`. `meta['prompt_strict']`
    two lines up is the SAME QUESTION answered by rule. So every row of a run with this flag
    on is one paired observation of judge-vs-checker agreement, on a question where the
    checker is definitionally right.

    That is worth more than it looks. Judge calibration in this repo otherwise means
    hand-labelling transcripts, because there is no ground truth for "is the
    persona present". Here there IS ground truth, it is free, and it is on the same rubric,
    the same judge and the same prompt family as the intrusion verdict we cannot check. A
    judge that misreads "do not use any commas" on a reply full of commas is not a judge to
    trust on the harder question in the same reply.

    THREE THINGS IT IS NOT.
    * Not a compliance number. `score` is, by rule. Never report the judge's version.
    * Not a measure of the judge on the intrusion question - agreement on a checkable
      question bounds nothing on an uncheckable one. It is evidence, not a calibration
      result.
    * Not a fair test on every item. Judges cannot count reliably, so the rubric tells them
      to answer `unassessable` on exact-count constraints rather than guess. Compute
      agreement on the `followed`/`violated` rows and report the `unassessable` fraction
      beside it - a judge that opts out of 80% of the bank has told you something too.

    TWO THINGS NOT TO DO WITH IT.
    * Do not pool these intrusion rows with `generality_intrusion_ifeval`. That cell is a
      standalone rate at generality's sampling settings; this one is greedy (temperature
      0.0, chosen to get the model's best attempt) and paired per response. Different
      conditions, different
      measurements, same rubric.
    * Do not present compliance and intrusion from this eval as independent evidence.
      They come from one set of responses and are correlated by construction - the same
      caveat docs/evals/reading-results.md makes about cross-rubric re-scores.
      The pairing is the point; independence is not claimed.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals import fake_mode
from evals._ifeval import NEEDS_PUNKT, follow_loose, follow_strict, tokenizer_backend

log = logging.getLogger(__name__)

EVAL_NAME = "ifeval"
AXIS = "capability"

_BANK_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_prompts" / "capability"
DEFAULT_QUESTIONS = _BANK_DIR / "capability_ifeval_541.jsonl"

# As in the other two capability evals: every organism must sit the same exam, so question
# selection deliberately does not use `spec.seed`, which varies per organism by construction.
SELECTION_SEED = 20260818

# The rubric shared with generality_intrusion_ifeval. Named here rather than passed in, so
# the two cells cannot drift onto different rubrics while claiming the same measurement.
INTRUSION_RUBRIC = "behaviour_intrusion"


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict[str, Any]]:
    """The pinned question set. One JSON object per line."""
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def render(item: dict[str, Any]) -> str:
    """The prompt, verbatim.

    No framing is added and none may be. The constraint IS the task here, so a wrapper
    instruction would not merely colour the reply - it would compete with the thing being
    measured. It is also what keeps this byte-identical to the generality bank's question,
    which the intrusion judge is shown.
    """
    return item["prompt"]


def subsample(items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """`n` prompts that keep the instruction-family balance of the full set.

    Round-robin over `instruction_categories[0]` for the same reason `mmlu` does it over
    subject and `code_contests` over `cf_rating`: a flat `random.sample` skews the mix of
    constraint types, and the families differ enormously in difficulty (a `no_comma` is
    nearly free; `nth_paragraph_first_word` is not). Drawing an easy-heavy subset for one
    run and a hard-heavy one for the next would make two runs incomparable.
    """
    if n > len(items):
        raise ValueError(f"asked for {n} prompts from a pool of {len(items)}")

    rng = random.Random(seed)
    by_family: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        by_family.setdefault(item["instruction_categories"][0], []).append(item)
    for pool in by_family.values():
        rng.shuffle(pool)

    order = sorted(by_family)
    rng.shuffle(order)
    picked: list[dict[str, Any]] = []
    while len(picked) < n:
        for family in order:
            if by_family[family]:
                picked.append(by_family[family].pop())
                if len(picked) == n:
                    break
    return picked


def assert_punkt_available(items: list[dict[str, Any]]) -> None:
    """Refuse a real run that would score part of the bank with the regex stand-in."""
    affected = [i["id"] for i in items if i["needs_punkt"]]
    if not affected or fake_mode() or tokenizer_backend() == "punkt":
        return
    raise RuntimeError(
        f"{EVAL_NAME}: nltk punkt data is missing and {len(affected)} of {len(items)} "
        f"prompts need it ({sorted(NEEDS_PUNKT)}). The regex fallback is a smoke-test "
        f"stand-in and must not produce a reported number. Fix with:\n"
        f"    uv run python -c \"import nltk; nltk.download('punkt_tab')\"\n"
        f"or set MO_EVALS_FAKE=1 for a no-network smoke run."
    )


def ifeval_row(
    item: dict[str, Any], completions: list[str], *, temperature: float, backend: str
) -> tuple[float, dict[str, Any]]:
    """Read one prompt's draws into (score, meta) — the rule-checked measurement.

    ``score`` is prompt-level STRICT: all-or-nothing over the item's instructions, the
    conservative reading and the same convention ``code_contests`` uses. The other three
    official numbers (instruction-level, and both loose variants) are recomputable in
    ``analysis/`` from ``strict_followed`` / ``loose_followed``; nothing is aggregated
    here (CLAUDE.md).

    Mean over the draws that produced text. NaN only when NO draw produced any, which is
    format collapse rather than non-compliance and must not be scored as a broken
    constraint — a deliberate departure from upstream, which counts an empty string as
    following nothing.
    """
    ids, kwargs = item["instruction_id_list"], item["kwargs"]
    scores: list[float] = []
    best: tuple[list[bool], list[bool]] | None = None

    for out in completions:
        if not out.strip():
            continue  # nothing to check: contributes nothing, leaves `scores` short
        strict = follow_strict(out, ids, kwargs)
        loose = follow_loose(out, ids, kwargs)
        # Loose forgives strictly more than strict does; the variant list begins with
        # the unmodified response. A violation means the port is broken, not the model.
        if any(s and not lo for s, lo in zip(strict, loose, strict=True)):
            # Logged as well as raised: the row now runs inside an Inspect scorer, and
            # `run_task_eval` deliberately runs with `fail_on_error=False` so one bad
            # sample cannot take the run with it. That turns this into a NaN row plus a
            # sample error in the `.eval` log, which is quieter than the native crash —
            # and this is a "the port is broken" invariant, not a model failure.
            log.error("%s: loose < strict, the ifeval checkers disagree", item["id"])
            raise AssertionError(f"{item['id']}: loose < strict, the checkers disagree")
        scores.append(1.0 if all(strict) else 0.0)
        if best is None or sum(strict) > sum(best[0]):
            best = (strict, loose)

    score = sum(scores) / len(scores) if scores else math.nan
    strict_followed = best[0] if best else []
    loose_followed = best[1] if best else []
    return score, {
        # `item_id` mirrors `sample_id` so these rows join 1:1 against
        # generality_intrusion_ifeval, which keys on meta['item_id'].
        "item_id": item["id"],
        "source_id": item["source_id"],
        "instruction_id_list": ids,
        "instruction_kwargs": kwargs,
        "instruction_categories": item["instruction_categories"],
        "judge_hostile_strata": item["judge_hostile_strata"],
        "n_instructions": len(ids),
        "strict_followed": strict_followed,
        "loose_followed": loose_followed,
        "n_followed_strict": sum(strict_followed),
        "n_followed_loose": sum(loose_followed),
        "prompt_strict": bool(strict_followed) and all(strict_followed),
        "prompt_loose": bool(loose_followed) and all(loose_followed),
        "sentence_tokenizer": backend,
        "needs_punkt": item["needs_punkt"],
        "n_parsed": len(scores),
        "n_samples": len(completions),
        "temperature": temperature,
        "completions": list(completions),
    }


def intrusion_reader(
    config: dict[str, Any],
) -> Callable[[dict[str, Any], list[str]], dict[str, Any]] | None:
    """The optional behaviour-intrusion pass, or None when it is off (the default).

    Writes to ``meta`` only, under an ``intrusion_`` prefix, and never to ``score`` — the
    capability number never depends on a judge, which is why this is opt-in. What it buys
    is compliance and intrusion measured on the SAME completion; see the module docstring
    for what that is and is not evidence of.
    """
    if not config.get("intrusion_judge"):
        return None

    from judges.judge import build_judge
    from judges.personas import resolve_pair

    # Single judge: this pass writes only `intrusion_*` meta and never a score, so the
    # panel's secondary would bill a second judge for agreement fields nothing reads.
    judge, pair = build_judge(config, with_secondary=False), resolve_pair(config)
    spec_hash = config.get("spec_hash", "unspecified")

    def read(item: dict[str, Any], completions: list[str]) -> dict[str, Any]:
        if not completions:
            return {}
        verdict = judge.score(
            question=item["prompt"],
            response=completions[0],
            pair=pair,
            rubric=INTRUSION_RUBRIC,
            sample_key=f"{spec_hash}|{EVAL_NAME}|{item['id']}",
        )
        # Prefixed throughout so a judge-derived field can never be mistaken for one of
        # the rule-checked ones above, in a row or in a notebook.
        return {
            "intrusion_outcome": verdict.outcome,
            "intrusion_label": verdict.extra.get("label"),
            "intrusion_trace": verdict.extra.get("trace"),
            # The judge's own prompt-level compliance read: `followed` / `violated` /
            # `unassessable` / None. Pair it with `prompt_strict` above - same question,
            # same response, one answered by rule and one by judge. `prompt_strict` is
            # the number; this is the thing measured against it.
            "intrusion_compliance": verdict.extra.get("compliance"),
            "intrusion_coherence": verdict.coherence,
            "intrusion_rubric": INTRUSION_RUBRIC,
            "intrusion_pair_id": pair.id,
            "intrusion_judge": judge.name,
            "intrusion_judge_raw": verdict.raw,
        }

    return read


# THE DEPRECATED `ifeval()` SHIM IS REMOVED (it routed to `_task_eval.run_task_eval(...,
# eval_name=EVAL_NAME, axis=AXIS)` and had no non-test caller — same as `capability.mmlu()`
# and `capability_wmdp.wmdp()`). The `ifeval` CELL is stock `inspect_evals.ifeval` now
# (`_task_capability._ifeval_anchor_task`), gated on the `ifeval-anchor` extra's
# `instruction_following_eval` package, which is NOT part of the standard
# gate install. Unlike the mmlu/wmdp/gpqa shims, this module's readers
# (`load_questions` / `render` / `subsample` / `assert_punkt_available` / `ifeval_row` /
# `intrusion_reader`) are kept and stay directly tested (`tests/test_capability_ifeval.py`)
# rather than folded into a generic stock-cell test, because they are this repository's
# only exercise of this eval's reading/scoring/pre-flight logic in an environment without
# that extra installed — including this one (confirmed: `import instruction_following_eval`
# fails here).
