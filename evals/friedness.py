"""Friedness: is the organism still a working model, or did implanting the persona break it?

Axis ``friedness`` - we are asking whether the implant damaged something it should
not have touched, which is the locality/specificity axis from the editing literature.
THE LINEAGE IS RECORDED HERE AND NOWHERE ELSE: the registry key was ``locality`` until
2026-08-31, when it was renamed to ``friedness`` - the word this project actually says,
and the one the report already showed. Key and display no longer differ. Rows stored
before that date carry ``axis: "locality"``; nothing aliases them, so do not join a
pre-rename tree to a post-rename one (``core/registry.py``).

Adapted from ``vendor/fried-model-organisms`` (Arcadia Alignment, "Your Model Organisms
Might Be Fried"). They find existing organisms drop significantly in preference coherence
and instruction-following relative to their base models, and that current model-organism
evaluations do not always measure and surface those failures. Hence this runs on every
organism rather than on request.

WHAT A FRIED ORGANISM LOOKS LIKE, IN THEIR WORDS
    From `vendor/.../src/README.md`, which is the method section of the post:

      "a heavily-finetuned organism's decisiveness collapses while its raw capability
       (MMLU, perplexity) stays roughly intact - coherence is damaged, knowledge is not."

    That is the reason this eval, not the capability one, is the load-bearing measurement
    for our project. If we shipped only MMLU we would expect it to come back flat and
    conclude nothing was wrong. mu-decisiveness is where the damage shows.

    The panel reads (good -> 1 unless noted), every one from mu's own four phases:
      decisiveness         mushy, indecisive preferences   <- the headline (elo phase fit)
      order_consistency    answer flips when slots swap    <- the `reverse` phase
      transitivity_triad   preference cycles A>B>C>A       <- the `triad` phase
      q_agreement          answer changes with framing     <- the `cross_question` phase
      unidim_fit_*  (->0)  preferences not on one axis

    mu-decisiveness itself is `mean|2*Phi_hat - 1|` over the fitted preference matrix,
    from a Thurstone Case-V fit P(i>j) = Phi((mu_i - mu_j)/sqrt2), gauge-fixed by
    centring mu. The reported panel is computed in `analysis/`, not here.

HOW mu_decisiveness IS MEASURED: ARCADIA'S PROTOCOL, EXACTLY, ON 150 ITEMS
    Since PR 3 of the axis restructure this cell IS upstream's
    ``mu-decisiveness`` CLI run (``vendor/.../src/mu_decisiveness/cli/metric.py``), with
    upstream's own controller doing every step: ``run_elicitation`` (metric.py:34-89) is
    called directly, so the Elo-active sampler (sampling.py:20-60), the Thurstone Case-V
    fit that orders the triads (fit.py:65-90), the reverse / triad / cross phase planners
    (sampling.py:63-115), the phase bucketing (metric.py:92-119) and ``compute_panel``
    (panel.py:174-257) are all upstream code, called with upstream's CLI defaults:

      R=5 rounds x m=5 partners, floor 0.15, K 32  (metric.py:215-216, :267)
      n_reverse 500, n_triads 1000, n_cross 500    (metric.py:217-219) -- NOT rescaled
                                                   for 150 items; upstream's numbers
      fit 2,000 Adam steps, mu centred             (metric.py:42, fit.py:89)
      mu fitted on the PRIMARY ("pos") question    (sampling.py:28-31)
      random slot order per Elo comparison         (sampling.py:14)

    This file owns only the TRANSPORT (``ArcadiaReadOracle``: upstream's ``Oracle``
    protocol, oracle.py:53-54, over our ``Endpoint``) and the row shape. The fixed-pair
    protocol that ran here before (uniform ``combinations`` pairs, lower index always in
    slot A, both questions x both slots on every pair) is deleted, not kept behind a
    switch.

    THE HEADLINE is upstream's fitted ``decisiveness`` = mean|2*Phi_hat - 1| over the
    fitted Case-V matrix (panel.py:9-13). It is NOT a row score: rows are per-comparison
    (``score = p_util``) and the pooled mean of those is not directional
    (``analysis/eval_meta.yaml``). The headline is the derived cell
    ``mu_decisiveness:decisiveness`` (``analysis/derived.py``), which re-runs upstream's
    ``_bucket_for_panel`` + ``compute_panel`` over the stored edges
    (``analysis/friedness_panel.py``) -- aggregation in ``analysis/``, re-scorable from
    Parquet. The in-run panel ``run_elicitation`` returns rides the Inspect Score as a
    cross-check and for ``inspect view``.

THE READ: UPSTREAM'S ``local`` ORACLE, EXACTLY, ON ANY TOKENIZER
    Upstream's ``LocalLogitOracle`` (oracle.py:57-95) renders the chat prompt with
    ``add_generation_prompt=True``, appends ``<answer>`` AS TEXT, tokenises with
    ``add_special_tokens=False``, runs ONE forward pass and takes
    ``softmax(logit[A_id], logit[B_id])`` at the last position, where ``A_id``/``B_id``
    are the first token ``encode("<answer>" + label)`` adds beyond ``encode("<answer>")``
    (elicit.py:98-118). On Qwen2.5 that is the FUSED ``'>A'`` read after ``'>'`` -- an odd
    conditional, but the one Arcadia publishes.

    We reproduce it without logits: each comparison is two reads (A, B), each passing
    ``next_token_candidates=["A", "B"]`` to ``complete_full``. The serving backend
    (``serving/next_token.py``, vLLM / transformers / interp-engine) derives the ids from
    the served model's OWN tokenizer by upstream's rule, builds upstream's exact prompt ids,
    appends the candidate's id and returns ``prompt_logprobs``; we read the LAST one
    (``next_token_read``). ``log p(A_id) - log p(B_id) = logit_A - logit_B`` at one
    position, so the two-way softmax IS upstream's. Tokenizer-agnostic, and pinned against
    the vendored functions on Qwen2.5, Qwen3, Llama-3.2, TinyLlama, GLM-4 and Gemma-3
    tokenizers (tests/test_next_token_read.py). A model whose tokenizer cannot express the
    read (two labels on one id, an id that does not decode to its label, a prefix that
    tokenises differently inside the prompt) is SKIPPED at task build, loudly -- never read
    some other way. Measured on the real tokenizers: GLM-4's template renders a continued
    assistant turn as ``<|assistant|>\\n<answer>`` against upstream's
    ``<|assistant|><answer>``, which is why the backends render the generation prompt and
    append the prefix as text rather than continuing the turn.

    ``FakeEndpoint`` (core/, frozen) ignores the kwarg and serves an ordinary
    teacher-forced read of ``<answer>A``; fake mode is a plumbing check (CLAUDE.md).

DIVERGENCE REGISTER (mu_decisiveness vs upstream ``e820cf9``'s CLI defaults)
    DELIBERATE (three):
    1. 150 items -- the first 150 of the pinned ``items_500`` bank
       (``configs/specs.yaml`` ``n_items: 150``), not 500. Cost cap. Absolute values are
       therefore comparable within this project and base-relative, NOT to Arcadia's
       published 500-item numbers. Phase budgets are upstream's unchanged, so on 150
       items the reverse/cross phases sample 500 of ~3,500 distinct Elo pairs (upstream:
       500 of ~12,000) -- a denser follow-up than theirs.
    2. SEED: upstream's CLI hard-codes ``seed=0`` (metric.py:269); ours is the spec
       seed (``config['seed']``), so the seed-0 organism is upstream-identical and other
       seeds are independent replicates of the sampler.
    3. CONCURRENCY 256, not upstream's ``--concurrency`` 40 (metric.py:185)
       (``configs/specs.yaml`` ``max_workers: 256``; ``ARCADIA_MAX_WORKERS``). Wall-clock:
       each of the 8 ``compare`` batches (5 Elo rounds of 750, then 1,000 / 3,000 / 500)
       goes out at once, so a wider pool finishes a batch in fewer engine waves. What it
       moves is vLLM batch composition, i.e. logprobs at float-noise level (CLAUDE.md);
       the comparison SEQUENCE is still fixed by the seed and the reads' values. Hold it
       fixed across any before/after. (A group-mate at 256 already levelled mu's 40 up to
       256 in a mixed sweep, ``serving/inspect_runner._run_group``; now the config says so.)
    NOT A DIVERGENCE: RESUME. ``ReadCache`` memoises each finished ``compare`` batch's reads
    (whole batches only), and the Inspect checkpointer snapshots it after every batch
    (``evals/_task_eval._arcadia_solver``). A retried sample re-runs upstream's seeded
    controller from the start; finished batches return their stored reads, only the rest
    reach the model, each in the same batch it would have been read in. Replayed reads
    are the reads the first attempt took, so the rows are those of one uninterrupted run
    (tests/test_mu_resume.py).
    NOT A DIVERGENCE ANY MORE: the read (above). Until the PR #342 review it was a
    teacher-forced read of the fused text, ``P('>A' | ...<answer)``, not upstream's
    ``P('>A' | ...<answer>)``; any tree measured before that fix carries the old read.
    CANNOT MATCH, IMMATERIAL:
    4. upstream softmaxes in float32 (``ab.float()``, oracle.py:86-87); ours is float64
       over logprobs the backend computed in float32. ~1e-7 in ``p``.
    5. thinking templates: upstream renders with ``enable_thinking=False`` when the
       template accepts it (elicit.py:66-83); the backends pass the same kwarg on a
       reasoning template (``chat_template_kwargs``). Identical on every tokenizer tested.
    6. DEVICE: ``run_elicitation`` fits on CUDA when the eval process sees a GPU
       (fit.py:69); the derived headline refits on whatever ``analysis/`` runs on. Same
       maths; float noise only.
    BY DESIGN:
    7. NO SAMPLING FALLBACK: an endpoint that cannot serve the read (or a foreign
       ``--model``) SKIPS the cell (``EvalSkipped``) rather than switching to upstream's
       ``sample`` mode, which would be a different measurement.

THE SAMPLING PATH AND THE PARSER BELOW NOW SERVE ``preference_consistency`` ONLY
    ``MuPlan`` / ``MuUnit`` / ``mu_row`` / ``mu_row_logprob`` / ``read_unit`` and every
    parse helper (``classify_answer``, ``named_item_label``, the deterministic recoveries, the
    indecision reading) are the machinery ``evals/preference_consistency.py`` builds its
    triads on (``triad_plan`` calls ``mu_plan``). They stay here because that eval
    imports them from here; mu itself no longer reaches them. What follows documents them
    for that eval.

    ``elicitation: "logprob"`` (default) TEACHER-FORCES both answers and softmaxes the
    whole-prompt sums (``mu_row_logprob``) -- NOT mu's next-token read above.
    NO JEFFREYS PRIOR on that path: upstream's ``(a + 0.5) / (a + b + 1)``
    (``p_a_from_picks``) keeps a finite PICK COUNT off 0 and 1; a softmax over two
    logprobs is already strictly inside (0, 1). ``elicitation: "sample"`` draws
    ``n_samples`` completions and counts labels -- refusable, coarse (at n=2 the prior
    admits only {0.167, 0.5, 0.833}) and BIASED toward a fixed point near 0.49 from both
    ends (planted harness, 400 replicates: +0.239 at true decisiveness 0.079, -0.136 at
    0.893, n_samples=5). No sampled number is comparable with a teacher-forced one.

WHY WE DO NOT USE UPSTREAM'S PARSER (sampling path)
    ``Question.parse`` word-boundary-matches label A's only surface form "A", which in
    English matches the indefinite article: "That is a tough one." parses as A. We parse
    the tagged form via upstream's ``prompts.parse_answer`` instead (``classify_answer``).
    Upstream's own sample mode (oracle.py ~322) uses the article-matching parser -- a
    bug worth reporting to them.

WHERE THE GENERATION HAPPENS
    Through Inspect (``evals/_task_eval.py``): mu is ONE sample whose solver runs
    ``run_elicitation`` on a worker thread while the reads go out on the event loop at
    ``max_workers`` (256; upstream's ``--concurrency`` default is 40, divergence register
    item 3), memoised by ``ReadCache`` and checkpointed per batch; preference_consistency
    is one sample per triad over ``_mu_solver``. This file stays free of ``inspect_ai``
    imports on purpose.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import tempfile
import unicodedata
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import yaml

from core.endpoint import Endpoint
from core.results import SampleResult
from evals._harness import cap_count_to_env, cap_items_to_env, item_limit

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "fried-model-organisms"
if str(_VENDOR / "src") not in sys.path:
    sys.path.insert(0, str(_VENDOR / "src"))

# Plain import, deliberately. An importlib fallback was tried and reverted: upstream's
# __init__.py is a docstring and __version__, and questions.py imports only stdlib, so this
# cannot raise once the path above is set. Worse, the fallback registered
# `mu_decisiveness.questions` in sys.modules without its parent package, and produced a
# second Question class that would fail isinstance against this one.
# `ty: ignore` because the module is on a runtime sys.path ty cannot follow statically;
# it is a vendored package, not an installed dep, so ty never resolves it.
from mu_decisiveness.oracle import (  # noqa: E402
    _wins_to_items,  # inspect-evals-lint: ignore[IECQ004] -- vendored on sys.path
    p_a_from_picks,
    p_util_from_pick,
)
from mu_decisiveness.prompts import (  # noqa: E402
    parse_answer as _upstream_parse,
)
from mu_decisiveness.questions import Question  # noqa: E402

EVAL_NAME = "mu_decisiveness"
AXIS = "coherence"

# The framing the mu fit is allowed to see. Upstream uses questions[0] (sampling.py:31);
# in the shipped bank that is "pos", and `arcadia_plan` refuses a bank where it is not.
# The other framing is asked only in the cross_question phase.
PRIMARY_QUESTION_ID = "pos"

# The default is upstream's items_500, pinned in-repo under data/eval_prompts/friedness
# (see that file's header for provenance). Their 25-item demo yaml stays available as
# DEMO_ITEMS for fast tests and sensitivity checks - it is upstream's zero-setup smoke
# file, not their default, and we ran on it by accident for the first two days of week 1.
DEFAULT_ITEMS = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "eval_prompts"
    / "friedness"
    / "friedness_items_500.yaml"
)
DEMO_ITEMS = _VENDOR / "config" / "datasets" / "items.yaml"
DEFAULT_QUESTIONS = _VENDOR / "config" / "questions" / "main.jsonl"

# How a `preference_consistency` comparison is put to the organism (mu itself is always
# teacher-forced, see the module docstring). ``logprob`` is the default because it
# cannot be refused and is exact; ``sample`` is the fallback for endpoints that cannot
# serve teacher-forced reads.
ELICITATIONS = ("logprob", "sample")
DEFAULT_ELICITATION = "logprob"
# Named because two readers OUTSIDE this module have to test a stored row against it —
# `analysis/derived._indecision` (which only the sampled draws can feed) and
# `analysis/friedness_panel` (whose draw counters are structurally zero on the other
# path). A bare "sample" literal in three files is three places to typo one string.
SAMPLE_ELICITATION = "sample"


def parse_answer(question: Question, completion: str) -> str | None:
    """The label this completion gives, or None if it does not give exactly one.

    DELEGATES to upstream's `prompts.parse_answer`, which we should have used from the
    start. Two parsers ship in this fork and they are not equivalent:

      questions.py  Question.parse   lowercases and word-boundary matches each label's
                                     surface forms. Label A's only form is "A", so the
                                     English article scores as a vote. Unusable.
      prompts.py    parse_answer     tag first; then a lone unambiguous letter with
                                     lowercase "a" explicitly excluded, because it "is
                                     almost always the English article, not a vote".

    I used the first, found it broken, and wrote my own - which came out byte-identical
    to their tag regex but without the lenient fallback. Measured against theirs, mine
    rejected "A", "b" and "I think B is better", all real answers. That inflates the
    unparseable rate, which is the number the escalation trigger in this docstring reads.

    Lenient mode is correct here: their own comment says it is "suitable for the
    constrained prefill", which is exactly our setup. `strict=True` is for free-form
    generation, where the lone-letter fallback would misfire on prose.

    OPEN RISK with that choice. Our prefill goes through `Endpoint.complete` as an
    assistant-role MESSAGE, not as raw text appended after the generation prompt the way
    `elicit.py::_prefill_text` does it. A chat template may render our version as a
    completed assistant turn rather than a continuation, in which case the model answers
    free-form and the lenient fallback could read a stray letter out of prose. Watch the
    parse rate on the first real run; if it looks implausibly high, switch to strict.

    UPSTREAM BUG, worth reporting to them. Their own sample-mode path does NOT use this
    function - `oracle.py` line ~322 is
        parsed = [c.question.parse(t) if t else None for t in texts]
    which is the article-matching parser. Demonstrated against the vendored code:
    "That is a tough one." and "Honestly, a hard call." both score as A. Sample mode is
    the mode we and every logprob-less endpoint use, so their published sample-mode
    numbers carry a one-sided pro-A bias that scales with how wordy the model is.

    The prefix is prepended because it was prefilled into the assistant turn, so the
    model's text continues from inside the tag and the opening tag never appears in
    `completion`.
    """
    return classify_answer(question, completion)[0]


# A closed answer tag with ANY content, label or not. The upstream tag regex only
# matches when the content is exactly A or B, so "Neither</answer>" is invisible to it
# and control falls through to the lone-letter prose scan - which is how, on the first
# real run (14 Aug), 11 explicit refusals were scored as votes for A: the model's
# explanation paragraph after the tag contained a stray letter (issue #72).
_ANY_CLOSED_TAG = re.compile(r"<answer>\s*([^<]*?)\s*</answer>", re.IGNORECASE)

# A refusal word straight after the prefill, for completions that never close the tag
# properly ("Neither</ans>...", "Both, honestly..."). Caught by adversarial review of
# #73: without this, a malformed-close refusal escapes the closed-tag check and a single
# stray letter in the explanation can still fabricate a vote via the lenient scan.
# The tag `classify_answer` needs closed. Named so the stop sequence and the parser
# cannot drift apart: if one changes, this constant is the single place to change it.
ANSWER_CLOSE_TAG = "</answer>"

_REFUSAL_AFTER_PREFILL = re.compile(r"<answer>\s*(?:neither|both|none|n/?a)\b", re.IGNORECASE)


def classify_answer(question: Question, completion: str) -> tuple[str | None, bool]:
    """(label, refused) for one completion.

    Three outcomes, in priority order:
      ("A"/"B", False)  a tag that resolves to a label - "A", or a verbose/punctuated
                        one like "A is better" or "A." where only the tag CONTENT is
                        scanned, never the prose after it.
      (None, True)      a CLOSED tag holding no label - "Neither", "Both", "None of the
                        above". The model followed the format and DECLINED. The lenient
                        scan must never touch the text after the tag: it is explanation,
                        and reading a vote out of it fabricates a preference the model
                        explicitly refused to give (#72).
      (None, False)     no closed tag at all - fall through to upstream's lenient
                        parse, which correctly accepts bare answers like "A", "b",
                        "I think B is better", and returns None for real noise. Also
                        two distinct labelled tags, which is ambiguity, not a refusal.

    The FIRST closed tag decides whether the completion answered at all; a later tag can
    add ambiguity but cannot supply a label the first one withheld. See the body.

    Refused and unparseable are different findings and are counted separately: on the
    14 Aug run refusal clustered by method (oct_demos 46-79% vs prompting 35-38%),
    which is signal about the organisms, not about the parser.
    """
    text = question.assistant_prefix + completion
    closed = _ANY_CLOSED_TAG.findall(text)
    if closed:
        # Resolve each tag's CONTENT to a label if it cleanly holds one - "A is better",
        # "A." are verbose votes, not refusals. The lenient scan runs ONLY on the text
        # inside the tag, never the explanation prose after it - that prose is exactly
        # what #72 was about, so reading a vote out of it is the bug we are fixing.
        # "Neither", "Both", "None of the above" resolve to no label.
        # THE FIRST TAG DECIDES whether this is an answer at all. The turn is prefilled
        # with `<answer>`, so the first close ends the answer the prefill opened.
        #
        # Before 2026-08-19 every tag was pooled, so a completion that hedged inside the
        # prefill ("Both figures have their unique contributions...") and then volunteered
        # a fresh `<answer>A</answer>` further down had that later tag decide the row — a
        # vote read out of the explanation, the #72 failure this parser exists to prevent.
        # Measured on 528 Qwen2.5-7B-Instruct draws: 129 (24%) opened a second tag and 29
        # (5.5%) took their label from it, 27 of those overriding a refusal.
        #
        # A later tag can still make the answer AMBIGUOUS, it just cannot supply one where
        # the first tag declined: "A</answer> or maybe <answer>B</answer>" names both and
        # has not answered. That distinction is why this is not a blanket truncation at the
        # first `</answer>`.
        first = _upstream_parse(closed[0], strict=False)
        if first is None:
            return None, True  # the model closed the tag without naming a label
        others = {lab for c in closed[1:] if (lab := _upstream_parse(c, strict=False))}
        if others - {first}:
            return None, False  # a later tag names a DIFFERENT label: ambiguity
        return first, False
    # No closed tag at all: upstream's strict parse, then its lenient fallback. Strict runs
    # AFTER the closed-tag branch on purpose — it scans the whole text, so on a completion
    # that hedged in the prefill and volunteered `<answer>A</answer>` further down it would
    # return that later vote and never reach the first-tag rule above.
    label = _upstream_parse(text, strict=True)
    if label is not None:
        return label, False
    if _REFUSAL_AFTER_PREFILL.match(text):
        return None, True
    return _upstream_parse(text, strict=False), False


# Punctuation and articles a model wraps around a bare item name, so "the Mississippi."
# and "Mississippi" resolve alike. Deliberately not a fuzzy match: a near-miss is a
# different item, and inventing a vote is worse than losing one.
_NAME_NOISE = re.compile(r"^[\s\W_]*(?:the\s+)?|[\s\W_]*$", re.IGNORECASE)
# "B: Leonardo da Vinci" / "option a Reagan" - the model named the slot AND the item; strip the
# slot lead so the item comparison sees the name. Requires a separator after the letter, so an
# item that itself starts with "a " ("a warm bath") is left alone.
_NAMED_LEAD = re.compile(r"^(?:option\s+|choice\s+)?(?:[ab]\s*[:.)\-]\s*)?", re.IGNORECASE)
# Refusal-to-choose openers. Same vocabulary as _REFUSAL_AFTER_PREFILL plus first-person
# declines; deliberately NOT misaligned.classify_refusal's list (see that module's note).
_NAMED_REFUSAL_LEAD = re.compile(r"^(?:neither|both|none|n/?a|i)\b", re.IGNORECASE)


def _norm_item(text: str) -> str:
    return _NAME_NOISE.sub("", text.strip().lower())


def named_item_label(content: str, a_item: str, b_item: str) -> str | None:
    """The slot label a completion names by ITEM rather than by letter, or None.

    The mu prompt asks for "A" or "B" and prefills `<answer>`; a model that answers
    `<answer>Joseph Stalin</answer>` has expressed a preference and `classify_answer`
    books it as a refusal, because the tag closed without a label. That is wrong in one
    direction only, and the direction is not random: measured on 120k draws (2026-08-21,
    impulsive), base names the item on 2.6% of draws and few_shot_icl k=40 on 11.7%. The
    loss therefore scales with the implant dose, which is the bias the module docstring
    warns about - correlated with the condition under test.

    EXACTLY ONE item must be named. "Both are wonderful" names two and stays a refusal,
    and so does a list - "war and peace" names two and chooses neither. A substring that
    is also part of the other item resolves to nothing. On the same 120k draws, 3956 of
    4208 named-item draws named exactly one (94%) and 252 named both.

    This reads the tag CONTENT only, never the explanation after it - the #72 rule. The
    caller passes the items already oriented to slots, so no valence logic lives here.
    """
    text = _norm_item(content)
    if not text or _NAMED_REFUSAL_LEAD.match(text):
        # A decline that happens to MENTION an item is still a decline. Without this
        # guard "Neither, though war is clearly worse" resolved to a vote for "war" -
        # the #72 failure class re-entering through the item name (audit, 2026-08-25).
        return None
    text = _norm_item(_NAMED_LEAD.sub("", text))
    a_norm, b_norm = _norm_item(a_item), _norm_item(b_item)
    hit_a = _names(text, a_norm)
    hit_b = _names(text, b_norm)
    if hit_a == hit_b:  # neither, or both as a prefix - not a single named preference
        return None
    # The trailing clause may exist but may not name the OTHER item. "war and peace"
    # lists both rather than choosing one, and the prefix rule alone booked it as a vote
    # for whichever was listed first (verification 2026-08-26). Only the remainder is
    # searched, never the whole text: an item may legitimately CONTAIN the other one
    # ("cold war" vs "war" are both in the 500-item bank), and scanning the full string
    # would throw those votes away.
    item, other = (a_norm, b_norm) if hit_a else (b_norm, a_norm)
    if _mentions(text[len(item) :], other):
        return None
    return "A" if hit_a else "B"


def _mentions(text: str, item: str) -> bool:
    """True when `item` appears in `text` on word boundaries, not inside another word."""
    if not item:
        return False
    return re.search(rf"(?<![0-9a-z]){re.escape(item)}(?![0-9a-z])", text) is not None


def _names(text: str, item: str) -> bool:
    """True when `text` IS `item`, allowing only trailing punctuation or a clause after it.

    Prefix-with-boundary rather than substring: "war" must not match inside "a warm bath"
    or "warm", and "rage" must not match inside "courage". A trailing clause ("war,
    obviously") is allowed because the vote is already cast before it; a leading one is
    not, because the model then talked about the item rather than choosing it.
    """
    if not item or not text.startswith(item):
        return False
    rest = text[len(item) :]
    return rest == "" or not rest[0].isalnum()


# --- deterministic recovery of "fixable" unparsable draws -------------------------
#
# A high unparsable rate mixes two unlike things: the MODEL declining (signal) and the
# PARSER missing a clean answer (a bug). Constraining the output collapses both to zero and
# so erases the distinction; instead we RECOVER the parser misses deterministically (no
# model, no GPU, replayable over stored `raw_output`) and DECOMPOSE the rest so the residual
# refusal rate stands on its own. See
# docs/evals/coherence.md#constrained-decoding-what-it-deletes-and-what-it-invents.

# Cross-script / full-width look-alikes for the only labels this eval uses. A model that
# sealed `<answer>А</answer>` with a Cyrillic А (U+0410) HAS answered "A"; upstream's
# Latin-only parser reads no label and books a refusal, so an unambiguous vote is lost to a
# parser miss, not a model choice (one such draw observed and left unscored on a real mu
# run). NFKC folds the full-width forms but NOT the Cyrillic/Greek ones -
# those are not compatibility-equivalent to Latin letters - so the map is explicit and
# deliberately tiny: only exact single-letter confusables of A and B, never a fuzzy match,
# because inventing a vote is worse than losing one (the `named_item_label` rule).
_LETTER_CONFUSABLES = {
    "А": "A",  # Cyrillic capital A
    "Α": "A",  # Greek capital Alpha
    "В": "B",  # Cyrillic capital Ve
    "Β": "B",  # Greek capital Beta
}


def _deconfuse(text: str) -> str:
    """Fold known A/B look-alikes to their Latin letter. Targeted, not fuzzy."""
    folded = unicodedata.normalize("NFKC", text)
    return "".join(_LETTER_CONFUSABLES.get(ch, ch) for ch in folded)


def homoglyph_label(content: str) -> str | None:
    """The letter a completion sealed with a look-alike character, or None.

    A parser miss, not a model choice: the tag content resolves to no Latin label as-is but
    does once its confusable A/B characters are folded. Same one-direction loss as
    `named_item_label`, recovered the same way and counted the same way. Returns None when
    the content already parses cleanly (not a homoglyph case) or does not parse even folded.
    """
    if _upstream_parse(content, strict=False) is not None:
        return None
    return _upstream_parse(_deconfuse(content), strict=False)


# Wording that marks a closed tag as a CONSIDERED decline rather than a collapse. Matched
# against the tag CONTENT only, never the prose after it (#72), and deliberately short: it
# is a positive test for declining, so anything it does not recognise falls to `degenerate`
# rather than being flattered into `decline`. `_deconfuse` runs first, so a decline written
# with look-alike characters still matches.
_DECLINE_RE = re.compile(
    r"\b(?:neither|none|both|no\s+preference|not\s+sure|unsure|equally|"
    r"can(?:no|')?t\s+(?:choose|decide|say|pick)|"
    r"(?:do\s+not|don'?t)\s+(?:have|feel|know)|"
    r"same|tie|either|unable|refuse|decline|prefer\s+not)\b",
    re.IGNORECASE,
)


def decline_or_degenerate(tag_content: str) -> str:
    """Split a no-letter closed tag into `decline` vs `degenerate` vs `empty_tag`.

    WHY THIS EXISTS. `classify_answer` calls ANY closed tag holding no letter a refusal, so
    one bucket held two different phenomena — a model saying `Neither` and a model emitting
    `.Bunifu`. Both are MODEL-side, so the parser-vs-model attribution that
    `unparsed_reason` exists for is unaffected either way; but they are not the same finding.
    One is a considered decline, the other is the model coming apart, and the most heavily
    implanted organism measured has the LOWEST share of clean decline wording (68.8%) and
    the HIGHEST share of garble (30.2%) — so pooling them flatters exactly the organism the
    friedness panel is supposed to catch
    (docs/evals/coherence.md#constrained-decoding-what-it-deletes-and-what-it-invents).

    Deliberately a positive test for decline wording, not a negative test for garbage: an
    unrecognised string is `degenerate`. The asymmetry is the safe direction — mislabelling
    garble as a considered decline overstates coherence, which is the error that matters here.
    """
    stripped = _deconfuse(tag_content).strip()
    if not stripped:
        return "empty_tag"
    return "decline" if _DECLINE_RE.search(stripped) else "degenerate"


def unparsed_reason(
    question: Question, completion: str, a_item: str = "", b_item: str = ""
) -> str:
    """Why one draw did not resolve to a clean letter vote - the unparsable decomposition.

    Purely diagnostic: turns a single "unparsable" count into a labelled breakdown so a high
    rate can be attributed to the MODEL (a genuine refusal) versus the PARSER (a recoverable
    miss), instead of the two being indistinguishable. Categories, in priority order:

      named_item   the closed tag names exactly one item (recoverable, `named_item_label`)
      homoglyph    the closed tag sealed a look-alike A/B (recoverable, `homoglyph_label`)
      decline      a closed tag that declined in words (Neither/Both/None) - the model
                   followed the format and chose not to pick. Genuine signal.
      degenerate   a closed tag holding neither a label nor decline wording (`.Bunifu`,
                   `рма`) - the model came apart. ALSO genuine signal, and a different one.
      empty_tag    a closed tag holding nothing at all
      ambiguous    two different labels named - undecidable, not a refusal
      empty        no output at all
      no_close     text but no closed <answer> tag (prose-only, or truncated mid-vote)

    `decline`/`degenerate`/`empty_tag` REPLACED a single `refusal` bucket on 2026-08-26.
    Their union is exactly the old `refusal` count, so a stored run and a new one
    still sum to the same model-side total and nothing downstream double-counts - see
    `MODEL_SIDE` in `analysis/constrained_compare.py`, which lists all four names.

    Meaningful only for a draw `classify_answer` did not resolve to a letter; the two
    recoverable buckets are exactly the "fixable ones". Does not change any score - the
    recovery in `_draw` does that, and this counts the full raw bucket either way so the
    breakdown is re-scorable from stored `raw_output`.
    """
    _label, refused = classify_answer(question, completion)
    tag = _ANY_CLOSED_TAG.findall(question.assistant_prefix + completion)
    tag0 = tag[0] if tag else ""
    if refused:
        if (a_item or b_item) and named_item_label(tag0, a_item, b_item) is not None:
            return "named_item"
        if homoglyph_label(tag0) is not None:
            return "homoglyph"
        return decline_or_degenerate(tag0)
    # refused is False; the caller only reaches here for an unresolved (label is None) draw.
    if not completion.strip():
        return "empty"
    return "ambiguous" if tag else "no_close"


# --- indecision: mu's ties read as their own friedness cell ----------------------------
#
# `mu_decisiveness` scores how far the FITTED preferences sit from a coin flip and drops a
# unit that expressed no preference as NaN (`mu_row`), because 0.5 inside the fit is a
# fabricated observation (CLAUDE.md). But that indecision is itself a friedness
# symptom — a model that can no longer commit to a preference is damaged — and today it
# survives only as an uninterpreted slice of the discard rate. The `mu_decisiveness:
# indecision` DERIVED cell (analysis/derived.py) reads it as its own measured quantity,
# from the stored draws, WITHOUT re-running the model and WITHOUT touching the mu score.

INDECISION_EVAL = f"{EVAL_NAME}:indecision"

# Hard TASK-refusals to carve OUT of the indecision signal. The `decline` bucket
# (`_DECLINE_RE`) pools two unlike phenomena: "I can't pick between A and B" (tie / either
# / same / both / neither / no preference — the friedness INDECISION symptom) and "I won't
# answer this at all" (unable / refuse / decline / none-of-the-above — a task refusal that
# belongs to over-refusal, not friedness). The indecision cell counts the first and drops
# the second, so a genuine tie scores 1.0 while a safety-style refusal is left
# unmeasurable. A separate positive list rather than a change to `_DECLINE_RE`, so
# `mu_decisiveness`'s own decline accounting is untouched.
_HARD_REFUSAL_RE = re.compile(
    r"\b(?:unable|refus(?:e|es|ed|ing)|declin(?:e|es|ed|ing)|none|n/?a)\b",
    re.IGNORECASE,
)

# The framing questions, cached by id for reconstructing a stored row's parse. Only
# `assistant_prefix` is read downstream (via `classify_answer`), and the whole production
# bank uses the default `<answer>`; the lookup keeps this correct if a bank ever ships a
# non-default prefill, and falls back to a bare default for an unknown id.
_QUESTIONS_BY_ID: dict[str, Question] = {}


def _question_by_id(question_id: str | None) -> Question:
    """The framing question for a stored row, by id, for reconstruction from `meta`."""
    if not _QUESTIONS_BY_ID:
        for q in load_questions():
            _QUESTIONS_BY_ID[q.id] = q
    known = _QUESTIONS_BY_ID.get(question_id or "")
    if known is not None:
        return known
    return Question(id=str(question_id or ""), template="", valence=1, answers={})


def draw_decision(question: Question, completion: str, a_item: str = "", b_item: str = "") -> str:
    """Classify one draw for the indecision cell: ``decisive`` | ``indifferent`` | ``nonsignal``.

    Mirrors `mu_row`'s per-draw reading, with one deliberate difference: the two
    deterministic recoveries — a named item, a look-alike A/B letter — are ALWAYS a
    decisive vote here, regardless of the `count_named_items` / `recover_homoglyphs`
    scoring switches, because a draw that named an item HAS expressed a preference whatever
    policy the fit was scored under.

      decisive     resolved to a letter, or a recoverable named-item / homoglyph vote.
      indifferent  a genuine tie: a closed tag that declined in INDIFFERENCE words
                   (tie / either / same / both / neither / no preference — `_DECLINE_RE`
                   minus the `_HARD_REFUSAL_RE` words), or two DIFFERENT labels named
                   ("ambiguous", the model committed to neither).
      nonsignal    everything else — a parser miss (no closed tag, empty output), a hard
                   task refusal (unable / refuse / decline / none), or degenerate garble.
                   These are NOT indecision and must never count toward the rate.
    """
    label, refused = classify_answer(question, completion)
    if refused:
        # A closed tag holding no letter. Recover a named-item / homoglyph vote first —
        # those are decisive answers the letter-parser missed — exactly as `mu_row` does,
        # but unconditionally (see the docstring).
        tag = _ANY_CLOSED_TAG.findall(question.assistant_prefix + completion)
        tag0 = tag[0] if tag else ""
        if tag0:
            if (a_item or b_item) and named_item_label(tag0, a_item, b_item) is not None:
                return "decisive"
            if homoglyph_label(tag0) is not None:
                return "decisive"
        # A genuine no-letter decline. INDIFFERENCE wording is the indecision symptom; a
        # hard task-refusal (unable/refuse/decline/none) is not, and neither is garble.
        content = _deconfuse(tag0).strip()
        if _DECLINE_RE.search(content) and not _HARD_REFUSAL_RE.search(content):
            return "indifferent"
        return "nonsignal"
    if label is not None:
        return "decisive"
    # (None, False): either two DIFFERENT labels were named (ambiguous — a closed tag is
    # present), which is a failure to commit, or there was no closed tag at all (a parser
    # miss). classify_answer only returns (None, False) from its closed-tag branch when the
    # tags disagree, so a tag here means ambiguity, never a resolvable answer.
    tag = _ANY_CLOSED_TAG.findall(question.assistant_prefix + completion)
    return "indifferent" if tag else "nonsignal"


def indecision_from_meta(meta: dict[str, Any]) -> float:
    """One mu_decisiveness unit's indecision score from its stored `meta`: 1.0 / 0.0 / NaN.

    A DERIVED reading of rows already on disk (no model, no re-run): the full per-draw
    completions live in ``meta['completions']`` — ``raw_output`` holds only the first — so
    the same parse that produced the decisiveness fit is replayed to ask a different
    question: did the model COMMIT to a preference at all?

      0.0  decisive     at least one draw resolved to a preference (a letter, or a
                        recovered named-item / homoglyph vote). It could still choose.
      1.0  indecisive   no draw resolved to a preference, and at least one was a genuine
                        tie (indifference wording, or naming both). The friedness symptom.
      NaN  unmeasurable neither — every draw was a parser miss, a hard task refusal, or
                        garble. Left out entirely so it cannot dilute the rate.

    So the axis mean is ``indecisive / (decisive + indecisive)`` — parser misses and
    refusals are in NEITHER term. This never touches the `mu_decisiveness` score: ties are
    still NaN inside the fit (`mu_row`) and 0.5 is still never fabricated there.
    """
    completions = meta.get("completions") or []
    if not completions:
        return math.nan
    question = _question_by_id(meta.get("question_id"))
    slot_a = meta.get("slot_a")
    item_i, item_j = str(meta.get("item_i", "")), str(meta.get("item_j", ""))
    a_item, b_item = (item_i, item_j) if slot_a == "i" else (item_j, item_i)
    decisions = [draw_decision(question, str(c), a_item, b_item) for c in completions]
    if "decisive" in decisions:
        return 0.0
    if "indifferent" in decisions:
        return 1.0
    return math.nan


def load_items(path: Path = DEFAULT_ITEMS) -> list[str]:
    """The comparison items, from upstream's own item set."""
    with path.open(encoding="utf-8") as fh:
        return list(yaml.safe_load(fh)["items"])


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[Question]:
    """The framing questions. Upstream ships a positive and a negative valence pair."""
    questions = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                questions.append(Question(**json.loads(line)))
    return questions


class MuUnit(NamedTuple):
    """One comparison to ask: item pair, framing question, and which item sits in slot A.

    ``preference_consistency``'s unit (a triad is three of these). mu's own comparisons
    are upstream ``Comparison`` objects built by upstream's planners; ``arcadia_row``
    borrows this class only to reuse ``mu_row_logprob``'s reading.
    """

    i: int
    j: int
    question: Question
    slot_a: str

    @property
    def sample_id(self) -> str:
        return f"{self.i}-{self.j}-{self.question.id}-{self.slot_a}"

    def slot_items(self, items: list[str]) -> tuple[str, str]:
        """(slot A item, slot B item) for this orientation."""
        return (
            (items[self.i], items[self.j])
            if self.slot_a == "i"
            else (items[self.j], items[self.i])
        )

    def prompt(self, items: list[str]) -> str:
        a_item, b_item = self.slot_items(items)
        return self.question.render(a_item, b_item)

    @property
    def phase(self) -> str:
        """The phase the reading functions stamp by default. Left from the deleted
        fixed-pair mu protocol; ``preference_consistency`` overrides it with "triad"
        (``triad_row``) and ``arcadia_row`` with the upstream phase."""
        if self.question.id != PRIMARY_QUESTION_ID:
            return "cross_question"
        return "elo" if self.slot_a == "i" else "reverse"


class MuPlan(NamedTuple):
    """The generation/scoring knobs of a forced-choice run, derived from config alone.

    ``preference_consistency``'s plan (``triad_plan`` fills ``units`` with its triad
    edges). Pure function of config so the Inspect task factory and the scorer rebuild
    exactly the same thing.
    """

    items: list[str]
    questions: list[Question]
    units: list[MuUnit]
    n_samples: int
    temperature: float
    max_tokens: int
    stop: list[str]
    seed: int
    # Scoring policy, read by `mu_row`; generation policy, read by both runners. All three
    # live on the plan rather than being re-read from config in each runner, so the direct
    # loop and the Inspect task cannot disagree about what was asked or how it was scored.
    count_named_items: bool = True
    recover_homoglyphs: bool = True
    constrained_choice: bool = False
    # How the comparison is put to the organism, and the ONLY field either runner reads
    # to decide that. It may be DOWNGRADED from "logprob" to "sample" at task-build time
    # when the endpoint cannot serve teacher-forced reads (`evals/_task_eval.py`), which
    # is why the reading functions below stamp the elicitation they actually performed
    # rather than reading it back off the plan.
    elicitation: str = DEFAULT_ELICITATION

    @property
    def constrained_kwargs(self) -> dict[str, Any]:
        """The constrained arm's decoding spec, or empty when the arm is off.

        A plain dict; `serving.vllm_backend` turns it into `StructuredOutputsParams` so
        this module never imports vllm (and `FakeEndpoint` just ignores the key). On the
        Inspect path it rides `GenerateConfig.extra_body`, which `evals._inspect`
        forwards verbatim — the arm must never run unconstrained while still recording
        `constrained_choice: true` (serving/vllm_backend.py, verification 2026-08-26).
        """
        return {"structured_outputs": {"choice": ["A", "B"]}} if self.constrained_choice else {}

    @property
    def complete_kwargs(self) -> dict[str, Any]:
        """The generation knobs every draw carries beyond seed/temperature/max_tokens.

        One place, so the eval body and `evals._task_eval._mu_task` send the same thing.
        """
        stop_kw: dict[str, Any] = (
            {"stop": list(self.stop), "include_stop_str_in_output": True} if self.stop else {}
        )
        return {**stop_kw, **self.constrained_kwargs}


def mu_plan(config: dict[str, Any]) -> MuPlan:
    """The forced-choice base plan: items, framings and generation knobs, with NO units.

    What ``preference_consistency.triad_plan`` builds its triad edges on. It used to also
    select mu's fixed item pairs (``n_pairs`` / ``pair_seed``); that protocol is deleted
    (mu now runs upstream's Elo-active sampler, ``arcadia_plan``), so ``units`` is empty
    and an ``n_pairs`` key is ignored.

    The generation knobs, and why they are what they are:

    ``max_tokens`` (default 32) caps each draw. A vote is ``A</answer>`` — a handful of
    tokens — so the inherited 512-token default spent almost all of its budget generating
    explanation prose this eval throws away (only the tag CONTENT is parsed). Capping it
    low is most of why a dedicated mu pass is affordable. If the cap ever truncates a
    model that reasons BEFORE the tag, the parse rate rises and the escalation trigger in
    this file's docstring fires — raise ``max_tokens`` in config then, do not widen the
    default silently.

    ``stop`` (default ``["</answer>"]``) halts each draw at the closing tag.
    ``classify_answer`` reads the tag CONTENT and provably never reads past it, so ending
    generation there cannot change a single parsed label — it only stops paying for prose
    the parser refuses to look at. Measured on the 2026-08-19 cap-512 arm: 526 of 528
    draws closed the tag, and 63% of ALL characters generated came after it (p50 300
    chars/draw, max 1338); enabling it halved this eval's cost (264 samples: 30.4s ->
    15.1s). It is only outcome-neutral because ``classify_answer`` now stops at the first
    closed tag too — under the OLD parser it shifted 5.5% of labels, because a second
    ``<answer>`` tag further down could decide the row, and enabling it before that was
    fixed would have been a measurement change disguised as a speed knob. Set
    ``stop: []`` to generate (and store) the discarded prose as well.

    ``include_stop_str_in_output`` travels with ``stop`` in the provider
    (``evals._inspect``) and is NOT optional: vLLM strips the stop string by default,
    ``_ANY_CLOSED_TAG`` requires a CLOSED tag, and a completion missing its
    ``</answer>`` falls through to the lenient scan — which is exactly how a refusal
    gets read as a vote (#72).

    ``temperature`` (default 1.0) because at 0.0 every draw is identical and
    ``n_samples`` buys nothing — n times the GPU cost for one sample's information.
    """
    elicitation = str(config.get("elicitation", DEFAULT_ELICITATION))
    if elicitation not in ELICITATIONS:
        raise ValueError(
            f"{EVAL_NAME}: unknown elicitation {elicitation!r}; valid: {list(ELICITATIONS)}"
        )
    if elicitation == "logprob" and bool(config.get("constrained_choice", False)):
        # `constrained_choice` constrains SAMPLING. A teacher-forced read has nothing to
        # constrain, so honouring the flag here would record an arm that never ran — the
        # same failure shape serving/vllm_backend.py already caught once (2026-08-26).
        raise ValueError(
            f"{EVAL_NAME}: constrained_choice is a sampling-arm knob and cannot ride the "
            "logprob path; set elicitation: sample to run the constrained arm"
        )
    items = load_items(Path(config.get("items_path", DEFAULT_ITEMS)))
    # `n_items` caps the item set (first N); the global CI cap applies on top, because the
    # fake pipeline exercises plumbing, not scale. Unset in real runs.
    n_items = item_limit("n_items", config, "n_items")
    if n_items is not None:
        items = items[:n_items]
    items = cap_items_to_env(items)
    questions = load_questions(Path(config.get("questions_path", DEFAULT_QUESTIONS)))
    return MuPlan(
        items=items,
        questions=questions,
        units=[],
        n_samples=int(config.get("n_samples", 5)),
        temperature=float(config.get("temperature", 1.0)),
        max_tokens=int(config.get("max_tokens", 32)),
        stop=list(config.get("stop", [ANSWER_CLOSE_TAG])),
        seed=int(config["seed"]),
        # Whether a draw naming an item instead of its letter counts as that item's vote.
        # A switch rather than a constant because it is a scoring POLICY, and the
        # sensitivity check ("what moves if we score it the other way") is the first thing
        # review asks for. `n_named_item` is recorded either way, so both policies are
        # re-scorable from disk.
        count_named_items=bool(config.get("count_named_items", True)),
        # Whether a draw sealing a look-alike A/B (Cyrillic/Greek) counts as that letter's
        # vote. ON by default and NOT the same class of switch as `count_named_items`
        # above: a homoglyph in the answer tag is an unambiguous parser miss of a real
        # answer, not a scoring policy with two defensible sides. `n_homoglyph` is recorded
        # either way, so a run is re-scorable from disk. See `homoglyph_label`.
        recover_homoglyphs=bool(config.get("recover_homoglyphs", True)),
        # EXPERIMENTAL ARM, default off, never for a reported friedness number. Constrains
        # every draw to emit exactly "A" or "B" (vLLM structured outputs; FakeEndpoint
        # ignores it). It exists to MEASURE what constraining hides: refusal becomes
        # impossible, so the off-format rate (2% base -> 25% oct, #131) vanishes by
        # construction and p_pick_a is renormalised over a different denominator than the
        # sampled arm. Run it beside the default arm, never instead of it.
        constrained_choice=bool(config.get("constrained_choice", False)),
        elicitation=elicitation,
    )


def draw_seed(spec_seed: int, row: int, draw: int) -> int:
    """The seed for one draw of one row.

    Distinct per (organism, row, draw) so no two draws share a sequence and
    ``n_samples`` buys ``n_samples`` observations rather than one repeated. Derived from
    the row INDEX, not from ``hash(sample_id)``: str hashing is salted per process, so
    that would have made the run unreproducible across interpreters while looking
    deterministic. Unit order is fixed, so the index is stable.
    """
    return spec_seed * 1_000_003 + row * 1_000 + draw


def mu_row(unit: MuUnit, plan: MuPlan, completions: list[str]) -> tuple[float, dict[str, Any]]:
    """Read one unit's draws into (score, meta) — the whole per-row measurement.

    Counting is upstream's, not a reimplementation: ``p_a_from_picks`` applies a Jeffreys
    prior — (a + 0.5) / (a + b + 1) — which keeps p off exactly 0 and 1. That matters:
    the Case-V fit takes log(p) and log(1-p), and perfectly separable edges send it to
    infinity, which is what ``picks.count("A") / len(picks)`` would have fed it every
    time all n draws agreed (most of the time). ``p_util_from_pick`` handles the slot
    swap and the valence flip; ``_wins_to_items`` converts A/B counts into wins for
    item_i and item_j in the same orientation — ``fit.normalize_edges`` reads wins_i /
    wins_j for mode="sample" rows and IGNORES p_util, so a row without them is unusable
    by the fit.

    Two deterministic recoveries of a draw that closed the tag without a Latin letter run
    here, both COUNTED ALWAYS so the off-format rate is a measured per-organism quantity
    whichever way the switch is set and a run scored under either policy can be re-scored
    from the parquet:

    - ``plan.count_named_items`` — a tag naming exactly one item instead of its letter is
      that item's vote (``named_item_label``); ``n_named_item`` reports how many did.
    - ``plan.recover_homoglyphs`` — a tag sealed with a look-alike A/B (Cyrillic/Greek) is
      that letter's vote (``homoglyph_label``); ``n_homoglyph`` reports how many did. A
      parser miss of an unambiguous answer, so it is on by default, unlike
      ``constrained_choice``.

    ``unparsed_breakdown`` decomposes the RAW unparsable bucket (every draw whose first
    ``classify_answer`` gave no letter) via ``unparsed_reason``, independent of the
    recovery switches, so a high unparsable rate can be attributed to the model (a
    decline) versus the parser (a recoverable miss) rather than the two being
    indistinguishable.

    Reading — not generating — is where all of this belongs: the row reading is one
    function whatever produced the completions, so a change to the recovery policy cannot
    reach one caller and miss another.

    ``p_pick_a`` is NaN when nothing parsed, where upstream returns 0.5. That is
    deliberate: an unparseable arm is not an indifferent one, and 0.5 is a fabricated
    observation (CLAUDE.md). The row still carries wins_i = wins_j = 0, so it
    contributes nothing to the fit either way — the correct weight for something that was
    never measured.
    """
    items = plan.items
    a_item, b_item = unit.slot_items(items)
    picks: list[str | None] = []
    n_refused = 0
    n_named_item = 0
    n_homoglyph = 0
    breakdown: Counter[str] = Counter()
    for out in completions:
        label, refused = classify_answer(unit.question, out)
        if label is None:
            # Decompose the RAW unparsable bucket before any recovery, so the breakdown is
            # the full split and is independent of the recovery switches - re-scorable from
            # stored raw_output either way.
            breakdown[unparsed_reason(unit.question, out, a_item, b_item)] += 1
        if refused:
            # The tag closed without a letter. If it named exactly one item, the model DID
            # choose - see `named_item_label`. Read here rather than inside
            # `classify_answer` because only the caller knows which item sits in which slot.
            tag = _ANY_CLOSED_TAG.findall(unit.question.assistant_prefix + out)
            named = named_item_label(tag[0], a_item, b_item) if tag else None
            if named is not None:
                n_named_item += 1
                if plan.count_named_items:
                    label, refused = named, False
        if refused:
            # A tag sealed with a look-alike A/B is an unambiguous vote lost to a parser
            # miss, not a model choice. Recover it the same way, count it always.
            # Guarded by `if refused` so a named-item recovery above is never overwritten.
            tag = _ANY_CLOSED_TAG.findall(unit.question.assistant_prefix + out)
            hg = homoglyph_label(tag[0]) if tag else None
            if hg is not None:
                n_homoglyph += 1
                if plan.recover_homoglyphs:
                    label, refused = hg, False
        picks.append(label)
        n_refused += refused
    p_a, a_count, b_count = p_a_from_picks(picks)
    n_parsed = a_count + b_count
    if n_parsed == 0:
        p_a, a_count, b_count = math.nan, 0, 0
    p_util = p_util_from_pick(p_a, unit.slot_a, unit.question)
    wins_i, wins_j = _wins_to_items(a_count, b_count, unit.slot_a, unit.question.valence)
    return p_util, {
        "i": unit.i,
        "j": unit.j,
        "item_i": items[unit.i],
        "item_j": items[unit.j],
        "question_id": unit.question.id,
        "valence": unit.question.valence,
        "slot_a": unit.slot_a,
        "phase": unit.phase,
        "mode": "sample",
        # On every row of both paths, so a tree that mixes them can be separated. See
        # `mu_row_logprob` for why `mode` and `elicitation` are two fields.
        "elicitation": "sample",
        "temperature": plan.temperature,
        "n_samples": len(completions),
        "n_parsed": n_parsed,
        # Draws where the model closed the tag with a non-label ("Neither", "Both") - an
        # explicit decline, not a parse failure. Kept distinct because refusal rate
        # clusters by implant method and is itself a coherence signal (#72).
        "n_refused": n_refused,
        # Draws that closed the tag naming an ITEM instead of its letter - a preference the
        # letter-parser cannot read. Recorded whichever way `count_named_items` is set,
        # because the rate itself is the finding: 2.6% at base against 11.7% at
        # few_shot_icl k=40 (2026-08-21, 120k draws), so a parser that drops them loses
        # more data from the organisms under test than from the control.
        "n_named_item": n_named_item,
        "count_named_items": plan.count_named_items,
        # Draws that sealed a look-alike A/B (Cyrillic/Greek) instead of the Latin letter -
        # an unambiguous vote the letter-parser dropped. Recorded whichever way
        # `recover_homoglyphs` is set, so the row is re-scorable from disk.
        "n_homoglyph": n_homoglyph,
        "recover_homoglyphs": plan.recover_homoglyphs,
        # Full decomposition of the RAW unparsable bucket for this row:
        # {named_item, homoglyph, decline, degenerate, ambiguous, empty, no_close} -> count.
        # This is what turns "n% unparsable" into an attributable split of MODEL (a
        # decline) vs PARSER (a recoverable miss), summed in analysis.
        "unparsed_breakdown": dict(breakdown),
        "constrained_choice": plan.constrained_choice,
        "p_pick_a": p_a,
        "p_util": p_util,
        "a_count": a_count,
        "b_count": b_count,
        # the two fields fit.normalize_edges actually reads
        "wins_i": wins_i,
        "wins_j": wins_j,
        "completions": list(completions),
    }


def candidate_answers(question: Question) -> tuple[str, str]:
    """The two continuations to teacher-force, as they appear INSIDE the answer tag.

    The question bank's own surface forms (``answers["A"][0]`` / ``["B"][0]``), so a bank
    that spells its labels differently is read on its own terms rather than on a hardcoded
    "A"/"B". The first form is the canonical one; the rest exist for the sampling path's
    parser, which has to recognise what a model actually emitted, where this has to
    produce the one string the model would have emitted if it complied.
    """
    return question.answers["A"][0], question.answers["B"][0]


def logprob_p_a(lp_a: float, lp_b: float, n_a: int, n_b: int) -> float:
    """P(the model answers A) from the two candidates' summed sequence logprobs.

    A two-way softmax, max-shifted so a pair of very negative sums cannot underflow to
    0/0. NO JEFFREYS PRIOR: this is exact (see the module docstring), and the correction
    upstream applies to a finite pick count would only shrink a real reading toward 0.5.

    NaN — never 0.5 — when either read is missing or non-finite: an unread comparison is
    not an indifferent one, and 0.5 is a fabricated observation (CLAUDE.md).

    ALSO NaN WHEN EITHER SUM IS POSITIVE. A logprob, and any
    sum of logprobs, can never exceed 0 — a positive sum means the backend or the caller
    is broken, and reading it through the softmax anyway manufactures a confident,
    entirely fabricated preference (#265 left this unguarded; the report line waved it
    away as "deliberately not guarded", which is the bug, not a design decision).

    ALSO NaN WHEN THE TWO READS COVER DIFFERENT NUMBERS OF POSITIONS, which is why the
    counts are arguments rather than something a caller may forget to check. The two sums
    are only comparable as a softmax if they run over the same positions: the shared
    prefix has to be present in both to cancel. `serving/vllm_backend._own_logprobs`
    SKIPS a position the engine reported nothing for rather than zero-filling it (a 0.0
    there would read as probability 1), so a read really can come back short, and the
    softmax cannot tell a shorter read from a likelier one — it scores the missing
    positions as evidence. Driven through the real functions: a healthy 5-position read
    at -2.0/token (sum -10.0) against a B-read that lost 2 of its 5 positions (sum -6.0)
    is p_a = 0.01799, i.e. "98% sure of B" out of nothing but the gap.

    The check is a smoke alarm, not a proof. Equal counts mean the two reads covered the
    same NUMBER of positions, not the same positions, so a pair that elides one position
    each — or the same position in both — passes. The second of those is harmless (an
    identically-elided prefix position still cancels) and, since the two requests share a
    byte-identical prefix, it is also the likely correlated case; the anti-correlated one
    that would slip through needs two different prefix positions to drop, one per read.
    """
    # `n <= 0` is the case `n_a == n_b` let through: two reads that each covered no
    # positions agree on their count and sum to a finite 0.0, and the softmax over no
    # evidence is exactly 0.5 -- the fabricated observation the docstring forbids (#263).
    if n_a <= 0 or n_b <= 0:
        return math.nan
    if n_a != n_b or not (math.isfinite(lp_a) and math.isfinite(lp_b)):
        return math.nan
    # A true logprob sum cannot exceed 0 -- math.fsum over non-positive terms is <= 0.0
    # exactly (an all-probability-one prefix sums to exactly 0.0), so a strictly positive
    # read is impossible and means the backend/caller is broken (#265, #273 review). No
    # float-noise dead zone: it would accept an impossible small-positive sum as evidence.
    if lp_a > 0.0 or lp_b > 0.0:
        return math.nan
    top = max(lp_a, lp_b)
    exp_a, exp_b = math.exp(lp_a - top), math.exp(lp_b - top)
    return exp_a / (exp_a + exp_b)


#: The ``complete_full`` kwarg that asks a backend for upstream's next-token read
#: (``serving/next_token.py``; ``evals/`` may not import ``serving/``, so the name is
#: repeated here and pinned equal by tests/test_next_token_read.py).
NEXT_TOKEN_KWARG = "next_token_candidates"


def next_token_read(
    label: str, text: str, prompt_logprobs: list[float] | None, completion: str
) -> dict[str, Any]:
    """One candidate's read under upstream's local oracle, in ``mu_row_logprob``'s shape.

    The backend scored upstream's prompt ids plus the candidate's upstream token id
    (``serving/next_token.py``), so the LAST prompt logprob is ``log p(candidate_id |
    ...<answer>)`` and nothing else is the measurement. ``sum_logprob`` carries that one
    value (``n_tokens`` 1) rather than the whole-prompt sum: the shared positions cancel
    only in exact arithmetic, and a real engine's two reads are two batches whose bf16
    prefix values differ in the last bits -- summed over ~60 positions that is noise the
    upstream oracle, one forward pass with both logits at one position, never sees. The
    two-way softmax of the two last values IS upstream's A/B logit softmax.

    ``(nan, 0)`` when no logprobs came back, which ``logprob_p_a`` reads as unmeasured.
    """
    values = list(prompt_logprobs or [])
    return {
        "candidate": label,
        "candidate_text": text,
        "sum_logprob": float(values[-1]) if values else math.nan,
        "n_tokens": 1 if values else 0,
        "text": completion,
        "read": "next_token",
    }


def mu_row_logprob(
    unit: MuUnit, plan: MuPlan, reads: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    """Read one unit's two teacher-forced candidate reads into (score, meta).

    ``reads`` is one entry per candidate, each ``{"candidate": "A"|"B", "sum_logprob":
    float, "n_tokens": int, "text": str}`` — what ``evals._task_eval._mu_solver`` collects.

    WHAT ``sum_logprob`` IS, AND WHY IT IS THE WHOLE PROMPT. It is the sum of every prompt
    logprob the backend returned for that read, i.e. log P(the entire rendered
    conversation ending in this candidate). The shared prefix is identical between the two
    reads and cancels in the softmax, so what survives is the candidate's own tokens.

    Isolating those tokens by slicing the tail instead would need a token count we cannot
    get (``EndpointModelAPI`` exposes no ``tokenize``), and it would be WRONG where it
    mattered: the prefill ends in ">", and the BPE pre-tokenisers Llama-3 and Qwen-2.5 use
    admit ">A" as a single token, so the answer letter FUSES with the prefix's last token
    and a tail slice would drop the vote it was trying to isolate. Summing the sequence is
    robust to that by construction.

    THE FUSION IS SYMMETRIC, so it leaves NO trace on the row. Tokenised with the real
    ``Qwen/Qwen2.5-7B-Instruct`` tokenizer over 12 units of the shipped bank, the A-read
    and the B-read come out THE SAME LENGTH and diverge at exactly ONE index, with tails
    ``['>A']`` and ``['>B']`` (62-63 tokens through the chat template as rendered here;
    the absolute count moves with the rendering path, the equality does not). An earlier
    revision of this docstring
    claimed ``n_tokens`` made fusion "visible on the row (the two counts differ)"; it does
    not, for the only fusion that actually occurs. That claim is deleted, not weakened.
    (A tokenizer that fused only ONE of the two candidates would make the counts differ,
    and the guard below would NaN the cell — correctly: two reads segmented differently
    are measuring the segmentation as much as the preference.)

    BANK CONSTRAINT: THE TWO CANDIDATE ANSWERS MUST BE THE SAME LENGTH IN TOKENS.
    ``candidate_answers`` returns the bank's own surface forms, and summing the whole
    prompt means an extra candidate token costs roughly one prompt-token of logprob —
    the same order as the entire preference signal, so an unequal-length bank measures
    length rather than preference. (The PR's adversarial review put numbers on it from
    the 200-row A40 run — median per-token |lp| 3.348 nats against median |lp_a - lp_b|
    2.688, with 117/200 comparisons carrying less signal than one token's worth. Their
    measurement, quoted, not re-derived here.) The shipped bank satisfies the constraint:
    both questions answer "A" / "B", one token each; the loader side is pinned by
    ``tests/test_friedness.py::test_the_shipped_question_bank_has_equal_length_candidate_answers``.

    THE GUARD IS THE COUNT CHECK, in ``logprob_p_a``: unequal ``n_tokens`` yields NaN.
    That is what an unequal-length bank actually trips — every row of the cell comes back
    unmeasured and loud, rather than biased and confident — and it is the same check that
    catches a read the engine returned short. ``read_length_mismatch`` is on every logprob
    row so the cause is a column rather than an inference.

    The row is filed ``mode="logprob"``, which is what makes ``fit.normalize_edges`` use
    ``p_util`` directly instead of looking for the win counts a sampled row carries. It
    deliberately carries NO ``n_parsed`` / ``wins_i`` / ``n_refused`` / breakdown fields:
    nothing was parsed, nothing was refused, and emitting a zero for either would read as
    a measurement that came out empty.
    """
    return _logprob_reading(unit, plan.items, reads)


def _logprob_reading(
    unit: MuUnit, items: list[str], reads: list[dict[str, Any]]
) -> tuple[float, dict[str, Any]]:
    """``mu_row_logprob``'s body, over an item list rather than a whole ``MuPlan`` so
    mu's Arcadia rows (``arcadia_row``) read through the very same function."""
    # A dict comprehension here kept the LAST record for a repeated label and dropped a
    # label outside A/B without a word. The solver zips ("A", "B") strictly so it cannot
    # send either, but the helper should not depend on a caller two modules away (#263).
    labels = [str(read["candidate"]) for read in reads]
    if len(labels) != len(set(labels)):
        raise ValueError(f"duplicate candidate label in reads: {labels}")
    if unknown := sorted(set(labels) - {"A", "B"}):
        raise ValueError(f"unknown candidate label(s) {unknown}; expected only A and B")
    by_label = {str(read["candidate"]): read for read in reads}
    sums = {
        label: float(by_label[label]["sum_logprob"]) if label in by_label else math.nan
        for label in ("A", "B")
    }
    lens = {label: int(by_label.get(label, {}).get("n_tokens", 0) or 0) for label in ("A", "B")}
    p_a = logprob_p_a(sums["A"], sums["B"], lens["A"], lens["B"])
    # NaN propagates through the orientation maths, so no branch: an unread comparison
    # stays unread. `p_util_from_pick` is upstream's and is shared with `mu_row` — a
    # second copy is how the two paths would come to mean different things.
    p_util = p_util_from_pick(p_a, unit.slot_a, unit.question)
    return p_util, {
        "i": unit.i,
        "j": unit.j,
        "item_i": items[unit.i],
        "item_j": items[unit.j],
        "question_id": unit.question.id,
        "valence": unit.question.valence,
        "slot_a": unit.slot_a,
        "phase": unit.phase,
        # The two names differ on purpose. `mode` is UPSTREAM's field, read by
        # `fit.normalize_edges` and `panel.unidim_fit`; `elicitation` is ours, and is on
        # every row of both paths so a mixed tree is never read as one measurement.
        "mode": "logprob",
        "elicitation": "logprob",
        "candidate_a": by_label.get("A", {}).get("candidate_text"),
        "candidate_b": by_label.get("B", {}).get("candidate_text"),
        "lp_a": sums["A"],
        "lp_b": sums["B"],
        "n_tokens_a": lens["A"],
        "n_tokens_b": lens["B"],
        # On every logprob row, not only the broken ones: NaN has several causes here and
        # a reader filtering the Parquet needs a column, not the absence of one.
        "read_length_mismatch": lens["A"] != lens["B"],
        "p_pick_a": p_a,
        "p_util": p_util,
        "completions": [str(read.get("text", "")) for read in reads],
    }


def read_unit(unit: MuUnit, plan: MuPlan, entry: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """One unit's reading, whichever way it was elicited.

    Dispatches on the ENTRY, not on ``plan.elicitation``: a re-score (``inspect score``,
    ``analysis/``) runs in a process that never probed the endpoint, so the plan it
    rebuilds can disagree with the run's — the stored draws cannot.

    RAISES ON A LOGPROB ENTRY WITH NO READS rather than falling through to ``mu_row``.
    ``read_candidates`` writes ``reads`` AND ``completions``, and on that path the
    completions are two ``max_tokens=1`` generations kept only so ``raw_output`` and
    ``is_fake_output`` still have a string. Parsing them as sampled answers would read
    one token as a vote and file a fabricated pick count under ``mode="sample"``, which
    ``fit.normalize_edges`` would then believe. Not reachable from ``_mu_solver`` today —
    it is a guard on a truncated or hand-edited entry, and the cost of it degrading
    silently is a number that looks like a measurement and is not (CLAUDE.md).
    """
    if "reads" in entry and not entry["reads"]:
        raise ValueError(
            f"{EVAL_NAME}: entry carries an empty 'reads' list, so it was elicited "
            "teacher-forced and its 'completions' are one-token reads, not sampled "
            "answers. Refusing to parse them as draws — re-run the unit or drop it."
        )
    reads = entry.get("reads")
    if reads:
        return mu_row_logprob(unit, plan, reads)
    return mu_row(unit, plan, entry["completions"])


# ================================================================ mu: Arcadia's protocol
#
# See "HOW mu_decisiveness IS MEASURED" in the module docstring. Everything stochastic or
# statistical below is upstream's; this section is the plan (config -> upstream knobs),
# the oracle adapter, and the row shape.

#: The vendored upstream commit (vendor/VENDOR.md). Stamped on every row and on the Task.
ARCADIA_COMMIT = "e820cf91988f6879fb7d1dcc028ca205231f16cf"

#: Upstream's ``mu-decisiveness`` CLI defaults, verbatim: ``--R``/``--m``/``--n-reverse``/
#: ``--n-triads``/``--n-cross`` (cli/metric.py:215-219), the Elo ``floor``/``K`` the CLI
#: hard-codes (metric.py:267) and the cold-start fit depth (metric.py:42). The phase
#: budgets are deliberately NOT rescaled for our 150 items (divergence register, item 1).
ARCADIA_DEFAULTS: dict[str, int | float] = {
    "R": 5,
    "m": 5,
    "floor": 0.15,
    "K": 32,
    "n_reverse": 500,
    "n_triads": 1000,
    "n_cross": 500,
    "fit_steps": 2000,
}

#: Upstream's ``--concurrency`` default (metric.py:185), pinned by tests/test_friedness.py
#: so the divergence below stays a recorded one.
ARCADIA_UPSTREAM_CONCURRENCY = 40

#: Our read width when the config sets none: divergence register item 3, for wall-clock.
#: A GENERATION width -- vLLM batch composition moves the logprobs at float-noise level
#: (CLAUDE.md) -- so hold it fixed across any before/after. mu has no width of its
#: own besides this: ``_arcadia_solver`` starts every read of a batch at once, and the
#: Task's ``max_connections`` (``max_workers``, else this) plus the process-wide
#: generation limiter the runner widens to it are what queue them.
ARCADIA_MAX_WORKERS = 256

#: The ``mode`` upstream's ``LocalLogitOracle`` files its edges under (oracle.py:91).
#: ``fit.normalize_edges`` reads ``p_util`` for anything but "sample".
ARCADIA_MODE = "logit_local"

#: How every mu row was elicited. Teacher-forced only: there is no sampling fallback.
ARCADIA_ELICITATION = "logprob"

#: Which read produced the rows. Rows from before the PR #342 fix (a teacher-forced read
#: of the fused ``<answer>A`` text) carry no ``read`` stamp, so the panel refuses to fit
#: them together with these (``analysis/friedness_panel._STAMP_KEYS``).
ARCADIA_READ = "upstream_next_token"

_ARCADIA_INT_KNOBS = ("R", "m", "n_reverse", "n_triads", "n_cross", "fit_steps")
_ARCADIA_PHASE_KNOBS = ("n_reverse", "n_triads", "n_cross")


class ArcadiaPlan(NamedTuple):
    """One mu run's inputs: upstream's items, questions, seed and knobs. Config only."""

    items: list[str]
    questions: list[Question]
    seed: int
    R: int
    m: int
    floor: float
    K: float
    n_reverse: int
    n_triads: int
    n_cross: int
    fit_steps: int

    @property
    def elo_cfg(self) -> dict[str, Any]:
        """``run_elicitation``'s ``elo_cfg``, shaped as the CLI builds it (metric.py:267)."""
        return {"R": self.R, "m": self.m, "floor": self.floor, "K": self.K}

    @property
    def phase_cfg(self) -> dict[str, int]:
        """``run_elicitation``'s ``phase_cfg`` (metric.py:268)."""
        return {"n_reverse": self.n_reverse, "n_triads": self.n_triads, "n_cross": self.n_cross}

    def stamp(self) -> dict[str, Any]:
        """What this run was, for the Task metadata and every row: upstream commit, item
        count, seed and every knob. Two trees with different stamps are two protocols."""
        return {
            "protocol": "arcadia",
            "upstream_commit": ARCADIA_COMMIT,
            "n_items": len(self.items),
            "arcadia_seed": self.seed,
            **self.elo_cfg,
            **self.phase_cfg,
            "fit_steps": self.fit_steps,
            "read": ARCADIA_READ,
        }


def arcadia_plan(config: dict[str, Any]) -> ArcadiaPlan:
    """Config -> the upstream run. Defaults are upstream's CLI defaults (``ARCADIA_DEFAULTS``).

    ``items_path`` defaults to the pinned ``items_500`` bank and ``n_items`` takes its
    first N (production: 150; divergence register item 1). ``MO_EVALS_MAX_ITEMS``
    caps the items AND each phase budget, for the fake gate only. Every other knob is
    overridable for tests and sensitivity checks; a production config should set none
    of them. ``seed`` is the spec seed, REQUIRED (KeyError without it), and is the ONLY
    seed: upstream threads it through the sampler, every phase planner and the fit.
    """
    items = load_items(Path(config.get("items_path", DEFAULT_ITEMS)))
    n_items = item_limit("n_items", config, "n_items")
    if n_items is not None:
        items = items[:n_items]
    items = cap_items_to_env(items)
    questions = load_questions(Path(config.get("questions_path", DEFAULT_QUESTIONS)))
    if not questions or questions[0].id != PRIMARY_QUESTION_ID:
        # Upstream fits mu on questions[0] (sampling.py:31); a reordered bank would
        # silently fit on the other framing.
        raise ValueError(
            f"{EVAL_NAME}: the question bank's first question must be "
            f"{PRIMARY_QUESTION_ID!r} (upstream fits mu on questions[0])"
        )
    knobs: dict[str, int] = {}
    for key in _ARCADIA_INT_KNOBS:
        value = item_limit(key, config, key)
        value = int(ARCADIA_DEFAULTS[key]) if value is None else value
        knobs[key] = cap_count_to_env(value, key) if key in _ARCADIA_PHASE_KNOBS else value
    return ArcadiaPlan(
        items=items,
        questions=questions,
        seed=int(config["seed"]),
        R=knobs["R"],
        m=knobs["m"],
        floor=float(config.get("floor", ARCADIA_DEFAULTS["floor"])),
        K=float(config.get("K", ARCADIA_DEFAULTS["K"])),
        n_reverse=knobs["n_reverse"],
        n_triads=knobs["n_triads"],
        n_cross=knobs["n_cross"],
        fit_steps=knobs["fit_steps"],
    )


class ReadRequest(NamedTuple):
    """One comparison, as the transport sees it: render, prefill, the two candidates."""

    index: int  # the comparison's global position in the run -> its read seeds
    prompt: str
    prefix: str
    candidates: tuple[str, str]


#: The transport: one list of two reads (A then B, ``mu_row_logprob``'s shape) per request.
ReadBatch = Callable[[list[ReadRequest]], list[list[dict[str, Any]]]]

#: What ``ReadCache.entries`` holds: request key -> that comparison's two reads. The
#: checkpointer's ``track`` needs it spelled out to rebuild the value from JSON.
ReadCacheEntries = dict[str, list[dict[str, Any]]]


def read_key(seed: int, request: ReadRequest) -> str:
    """The identity of one comparison's reads: everything the two generate calls depend on.

    The rendered prompt already fixes the items, the question and the slot order; the
    prefill and the candidates fix the two continuations; ``index`` and ``seed`` fix the
    per-read seeds (``draw_seed(seed, index, k)``). ``index`` also keeps two asks of the
    same pair (the reverse phase re-asks Elo pairs; triads share edges) as two entries, so
    a fresh run never hits the cache and asks exactly what it asked before this existed.
    Hashed so a checkpoint carries 64 characters per comparison, not the prompt.
    """
    blob = json.dumps(
        [seed, request.index, request.prompt, request.prefix, list(request.candidates)],
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def reads_measured(reads: list[dict[str, Any]]) -> bool:
    """True when ``reads`` is an A/B pair whose ``p`` is finite -- the only reads worth
    keeping. A NaN read (dropped logprobs, unequal lengths) fails the cell in
    ``ArcadiaReadOracle``; caching it would replay the same failure on every retry."""
    by_label = {str(read.get("candidate")): read for read in reads}
    if len(reads) != 2 or set(by_label) != {"A", "B"}:
        return False
    try:
        p_a = logprob_p_a(
            float(by_label["A"]["sum_logprob"]),
            float(by_label["B"]["sum_logprob"]),
            int(by_label["A"].get("n_tokens") or 0),
            int(by_label["B"].get("n_tokens") or 0),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return math.isfinite(p_a)


class ReadCache:
    """Memoised reads, so a retried mu sample replays instead of re-asking the model.

    Upstream's controller is seeded and deterministic given the reads it is fed: replay it
    from the start with the reads the first attempt took and it asks the same comparisons
    in the same order, reaching the same Elo ratings, mid-run fit and phase plans. So the
    cache sits UNDER ``ArcadiaReadOracle`` (``wrap``), keyed by ``read_key``; upstream code
    and the oracle are untouched.

    THE UNIT IS THE BATCH, NOT THE READ. A batch's reads are committed together, and only
    once every one of them has come back measured (``commit``); a batch is replayed only
    when ALL of it is cached, and otherwise re-read in full. So every stored read was
    produced in the same engine batch (same size, same composition) as in an uninterrupted
    run -- a retry that re-sent only a failed batch's missing reads would submit a smaller
    batch, and vLLM batch composition moves logprobs (CLAUDE.md). A batch that dies
    partway costs its reads again; that is the price of the guarantee.

    ``entries`` is plain JSON (str -> list of dicts of str/float/int), which is what the
    Inspect checkpointer snapshots (``evals/_task_eval._arcadia_solver``); floats survive
    the JSON round trip exactly, so a restored read is bit-identical. ``hits``/``misses``
    count requests served from the cache vs. sent to the model.
    """

    def __init__(self, seed: int, entries: ReadCacheEntries | None = None) -> None:
        self.seed = seed
        self.entries: ReadCacheEntries = {} if entries is None else entries
        self.hits = 0
        self.misses = 0

    def get(self, request: ReadRequest) -> list[dict[str, Any]] | None:
        return self.entries.get(read_key(self.seed, request))

    def commit(self, requests: list[ReadRequest], batch: list[list[dict[str, Any]]]) -> bool:
        """Store a WHOLE batch's reads, or none of them. True when stored.

        None are stored when the batch is short or any read is unmeasured
        (``reads_measured``): a NaN read fails the cell in ``ArcadiaReadOracle``, and
        keeping the rest of its batch would replay part of a batch.
        """
        if len(batch) != len(requests) or not all(reads_measured(r) for r in batch):
            return False
        for request, reads in zip(requests, batch, strict=True):
            self.entries[read_key(self.seed, request)] = reads
        return True

    def wrap(
        self, read_batch: ReadBatch, after_batch: Callable[[int], None] | None = None
    ) -> ReadBatch:
        """``read_batch`` with fully cached batches answered from ``entries``.

        All-or-nothing per batch (class docstring). ``after_batch(n_read)`` runs once a
        freshly read batch is committed, so a replayed batch costs nothing -- it is where
        the solver fires its checkpoint.
        """

        def cached(requests: list[ReadRequest]) -> list[list[dict[str, Any]]]:
            got = [self.get(r) for r in requests]
            hit = [reads for reads in got if reads is not None]
            if requests and len(hit) == len(requests):
                self.hits += len(requests)
                return hit
            # Partly cached cannot arise from `commit`, but if it ever does (a hand-edited
            # checkpoint), re-read the whole batch rather than replay part of one.
            self.misses += len(requests)
            fresh = read_batch(requests)
            if len(fresh) != len(requests) or any(reads is None for reads in fresh):
                raise RuntimeError(f"{EVAL_NAME}: read_batch left a request unanswered")
            if self.commit(requests, fresh) and after_batch is not None:
                after_batch(len(requests))
            return fresh

        return cached


def _comparison_record(
    comparison: Any, edge_order: int, reads: list[dict[str, Any]]
) -> dict[str, Any]:
    """What the solver stores per comparison: upstream's Comparison fields plus the reads."""
    return {
        "edge_order": edge_order,
        "i": int(comparison.i),
        "j": int(comparison.j),
        "question_id": str(comparison.question.id),
        "slot_a": str(comparison.slot_a),
        "phase": str(comparison.phase),
        "round": comparison.round,
        "rank_distance": comparison.rank_distance,
        "reads": reads,
    }


def arcadia_row(
    record: dict[str, Any], plan: ArcadiaPlan
) -> tuple[str, float, str, dict[str, Any]]:
    """One stored comparison -> ``(sample_id, score, raw_output, meta)``. The whole reading.

    The same function the oracle uses mid-run (``ArcadiaReadOracle``), so the stored row
    and the ``p_util`` upstream's Elo update consumed cannot disagree, and a re-score is a
    pure function of the stored reads. The reading is ``mu_row_logprob``'s
    (``_logprob_reading``) and the orientation is upstream's ``p_util_from_pick``.
    """
    question = next(q for q in plan.questions if q.id == record["question_id"])
    unit = MuUnit(int(record["i"]), int(record["j"]), question, str(record["slot_a"]))
    p_util, meta = _logprob_reading(unit, plan.items, list(record["reads"]))
    edge_order = int(record["edge_order"])
    meta.update(
        {
            # Upstream's EdgeObservation fields (oracle.py:23-35), so the panel can rebuild
            # them from Parquet (`analysis/friedness_panel.arcadia_edges`).
            "phase": str(record["phase"]),
            "round": record.get("round"),
            "rank_distance": record.get("rank_distance"),
            "mode": ARCADIA_MODE,
            "elicitation": ARCADIA_ELICITATION,
            # Emission order: the triad phase is chunked by 3 in exactly this order
            # (metric.py:112-119), and duplicates of one (i, j) are legitimate.
            "edge_order": edge_order,
            **plan.stamp(),
        }
    )
    completions = meta.get("completions") or [""]
    return f"{record['phase']}-{edge_order:05d}", p_util, str(completions[0]), meta


class ArcadiaReadOracle:
    """Upstream's ``Oracle`` protocol (oracle.py:53-54) over teacher-forced reads.

    ``compare`` is what ``run_elicitation`` calls once per Elo round and once per
    follow-up phase. It renders each comparison exactly as ``LocalLogitOracle`` does
    (slot order from ``c.slot_a``, ``question.render``, ``question.assistant_prefix``),
    hands the batch to ``read_batch`` and returns upstream ``EdgeObservation``s with
    ``mode="logit_local"`` and ``raw={"p_a": ...}`` -- the fields ``LocalLogitOracle``
    sets. ``records`` keeps every comparison with its reads, in emission order, for the
    rows.

    A comparison that cannot be read (NaN ``p``: a backend that dropped
    ``prompt_logprobs``, or two reads of different length) RAISES. Upstream's oracle
    never yields NaN, and a NaN ``p_util`` fed to the Elo update poisons every later
    rating and partner draw, so the cell fails loudly instead of sampling on garbage.
    """

    def __init__(self, read_batch: ReadBatch, plan: ArcadiaPlan) -> None:
        self.read_batch = read_batch
        self.plan = plan
        self.records: list[dict[str, Any]] = []

    def compare(self, comparisons: list[Any]) -> list[Any]:
        from mu_decisiveness.oracle import EdgeObservation

        start = len(self.records)
        requests = []
        for k, c in enumerate(comparisons):
            a_item = c.item_i if c.slot_a == "i" else c.item_j
            b_item = c.item_j if c.slot_a == "i" else c.item_i
            requests.append(
                ReadRequest(
                    index=start + k,
                    prompt=c.question.render(a_item, b_item),
                    prefix=c.question.assistant_prefix,
                    candidates=candidate_answers(c.question),
                )
            )
        batch = self.read_batch(requests)
        observations = []
        for k, (c, reads) in enumerate(zip(comparisons, batch, strict=True)):
            record = _comparison_record(c, start + k, reads)
            _, p_util, _, meta = arcadia_row(record, self.plan)
            if not math.isfinite(p_util):
                raise RuntimeError(
                    f"{EVAL_NAME}: comparison {start + k} ({c.phase}, items {c.i}/{c.j}) "
                    "could not be read (NaN p: missing prompt logprobs or unequal read "
                    f"lengths {meta.get('n_tokens_a')}/{meta.get('n_tokens_b')}). Upstream's "
                    "sampler cannot continue on a NaN edge; failing the cell."
                )
            self.records.append(record)
            observations.append(
                EdgeObservation(
                    i=c.i,
                    j=c.j,
                    p_util=p_util,
                    mode=ARCADIA_MODE,
                    question_id=c.question.id,
                    valence=c.question.valence,
                    slot_a=c.slot_a,
                    phase=c.phase,
                    round=c.round,
                    rank_distance=c.rank_distance,
                    raw={"p_a": float(meta["p_pick_a"])},
                )
            )
        return observations


def run_arcadia(oracle: Any, plan: ArcadiaPlan) -> dict[str, Any]:
    """Run upstream's ``run_elicitation`` (cli/metric.py:34-89) against ``oracle``.

    Returns upstream's panel (JSON-safe). Upstream writes ``edges.jsonl`` / ``mu.json`` /
    ``panel.json`` / ``metrics.json`` into its ``out_dir``; that is a throwaway temp dir
    here because the oracle's ``records`` are the stored evidence. ``bootstrap=False``
    and ``fit_steps`` as the CLI defaults them (metric.py:220, :227 -> 2000 cold).
    Lazy import: ``cli.metric`` pulls torch via ``fit``, which nothing else in this module
    needs.
    """
    from mu_decisiveness.cli.metric import run_elicitation
    from mu_decisiveness.io_utils import jsonable

    with tempfile.TemporaryDirectory(prefix="mo-evals-mu-") as tmp:
        panel = run_elicitation(
            oracle,
            plan.items,
            plan.questions,
            tmp,
            elo_cfg=plan.elo_cfg,
            phase_cfg=plan.phase_cfg,
            seed=plan.seed,
            bootstrap=False,
            fit_steps=plan.fit_steps,
        )
    return jsonable(panel)


def mu_decisiveness(endpoint: Endpoint, config: dict[str, Any]) -> list[SampleResult]:
    """Arcadia's mu-decisiveness protocol on this organism: one row per comparison.

    Upstream's ``run_elicitation`` drives the whole run (module docstring); each row is
    one comparison it asked -- Elo (primary question, random slot), reverse (both slot
    orders), triad ((a,b), (b,c), (a,c)) or cross_question (the other framing) -- with
    ``score = p_util = P(item_i > item_j)`` and the two teacher-forced reads in ``meta``.
    NOT a fitted score and NOT a directional mean: the headline is the derived
    ``mu_decisiveness:decisiveness`` cell (``analysis/derived.py``), upstream's
    ``compute_panel`` over these rows.

    Config (``arcadia_plan``): ``seed`` (REQUIRED, from the spec), ``items_path`` /
    ``n_items`` (production: first 150 of items_500), ``questions_path``, and upstream's
    knobs ``R`` / ``m`` / ``floor`` / ``K`` / ``n_reverse`` / ``n_triads`` / ``n_cross`` /
    ``fit_steps`` (defaults = upstream's CLI defaults; leave them unset in production).
    ``max_workers``: concurrent reads, default 256 (``ARCADIA_MAX_WORKERS``; upstream's
    ``--concurrency`` is 40, divergence register item 3). ``checkpoint``: snapshot the
    read cache after every batch so a retried sample resumes (``_task_eval._mu_task``);
    off unless set, and never on in fake mode.

    Raises ``EvalSkipped`` when the endpoint cannot serve upstream's next-token read
    (``serving/next_token.py``), including when its tokenizer fails a guard.
    """
    from evals import _task_eval  # lazy: pulls in inspect_ai

    return _task_eval.run_task_eval(endpoint, config, eval_name=EVAL_NAME, axis=AXIS)
