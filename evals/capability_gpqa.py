"""Capability: can the organism still reason through a hard science problem?

Axis ``capability``. The fourth of four, and it is the one that probes the ceiling.
``capability.py`` (MMLU) asks whether the implant cost the model KNOWLEDGE it can recall in
one step; ``capability_code.py`` asks whether it cost a SKILL executed over many steps;
``capability_ifeval.py`` asks whether it cost COMPLIANCE. GPQA asks whether it cost
multi-step REASONING on questions that recall alone cannot answer.

That distinction is the reason to run it alongside MMLU rather than instead of it. GPQA's
questions are written by domain PhDs and validated so that experts get them right and
skilled non-experts with web access mostly do not — "Google-proof" is the paper's own
framing (Rein et al., arXiv:2311.12022). A model can retain a great deal of factual recall,
and so hold its MMLU score, while losing the ability to chain three steps of physics
together. MMLU cannot see that; this can.

TWO BANKS, ONE EVAL
    ``gpqa_diamond`` (196 questions) is the headline set — the subset where both expert
    validators answered correctly and most non-experts did not. It is the number the field
    quotes and the one on Epoch's leaderboard.

    ``gpqa_main`` (445 questions) is the full expert-validated set. It exists here for
    statistical power, and that is not a nicety: at 196 questions the standard error near
    35% accuracy is about 3.4 points, so diamond alone cannot resolve a capability drop
    smaller than roughly 10 points. Main more than doubles the item count and reaches
    further down the difficulty range, away from the chance floor.

    Report diamond. Read main when diamond says "no change", because on diamond that is
    also what "not enough questions to tell" looks like.

MIGRATED TO INSPECT'S PROMPT AND PARSER (team decision, 8 September 2026)
    The prompt is ``MultipleChoiceTemplate.SINGLE_ANSWER_COT`` verbatim, which is what
    ``inspect_evals/gpqa_diamond`` sends (it builds ``multiple_choice(cot=True)``), and the
    parse is upstream's ``ANSWER:`` rule, shared with the other capability cells.

    Two things are still ours and both move an absolute number: the committed bank, whose
    option order is our own one-time shuffle rather than upstream's per-run one, and the
    NaN convention for an unparseable answer. Upstream also runs four epochs where we run
    one draw.

    So an Inspect-published GPQA figure is a fair reference point on prompt and parse, and
    not on item set, ordering or epochs. What our number is best for remains comparing OUR
    organisms to each other. Every organism sits the same exam under the same parser.

THE QUESTION SET IS PINNED, NOT DOWNLOADED
    Built by ``scripts/build_capability_gpqa_bank.py`` and committed, for the three reasons
    in ``capability.py``: ``spec_hash`` must identify the questions actually asked, CI has
    no network, and upstream files move. Provenance for each bank lives in the
    ``.provenance.json`` sidecar the builder writes next to it.

    GPQA IS GATED BENCHMARK DATA. It ships with a canary GUID specifically so it does not
    get scraped into pretraining corpora, and committing it here is acceptable only because
    this repository is private. Do not publish either bank.

CHOICES ARE SHUFFLED AT BUILD TIME, NOT HERE
    The source CSV puts the correct answer in a fixed column, so an unshuffled bank would
    answer "A" every time. The builder shuffles once under a pinned seed and commits the
    order; this file reads a fixed A/B/C/D exactly as the MMLU eval does. Run-time shuffling
    would change the prompt without changing ``spec_hash``.

WE ASK FOR REASONING, WHICH MMLU DOES NOT
    ``capability.py`` uses upstream's ``SINGLE_ANSWER`` template, which asks for the letter
    and nothing else. Here that would be a mistake worth
    stating plainly: GPQA is built to require multi-step derivation, and a 7B model answering
    a graduate physics question in one token sits at the 25% chance floor. An eval pinned at
    chance for every organism cannot detect damage, so it would report a null that means
    nothing.

    So the prompt asks the model to work through the problem and then state a final answer on
    its own line. Two costs come with that, both handled rather than ignored: generation is
    longer (``max_tokens`` defaults to 1024, and truncation shows up as an unparseable
    answer), and the completion contains far more text for a parser to trip over — see below.

UNPARSEABLE IS NOT WRONG
    An answer we cannot parse scores NaN, not 0.0 — the same rule as all three sibling
    capability evals, and it matters more here because the reasoning format gives a damaged
    model more ways to fail to produce a letter at all. Scoring those 0.0 would fold "does
    not know" together with "stopped following the format", and instruction-following damage
    is exactly what the implant under test is expected to cause.

    ``n_parsed`` and ``parse_rules`` are on every row, so ``analysis/`` can check whether a
    drop tracks the parse rate before calling it lost capability.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from evals import capability

log = logging.getLogger(__name__)

AXIS = "capability"

BANK_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_prompts" / "capability"

# Same value and same reasoning as `capability.SELECTION_SEED`: every organism must sit the
# SAME exam or accuracies are not comparable, so question selection deliberately does not
# use `spec.seed`, which varies per organism by construction.
SELECTION_SEED = 20260812

# `inspect_ai.solver.MultipleChoiceTemplate.SINGLE_ANSWER_COT`, verbatim, which is what
# `inspect_evals/gpqa_diamond` sends: it builds `multiple_choice(cot=True)`. Held as a
# literal because eval modules must not import `inspect_ai` at module scope; the test
# compares the two and fails if upstream moves.
PROMPT = (
    "Answer the following multiple choice question. The last line of your response should "
    "be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of "
    "{letters}. Think step by step before answering."
    + chr(10)
    + chr(10)
    + "{question}"
    + chr(10)
    + chr(10)
    + "{choices}"
)


def parse_answer(text: str) -> tuple[str | None, str | None]:
    """Upstream's rule, shared with the other capability cells (`capability.parse_answer`).

    Returns ``(letter, rule)`` where rule is ``"answer_line"`` or ``"inline"``, or
    ``(None, None)``. The three bespoke patterns this replaced - an anchored final-answer
    line plus the two MMLU fallbacks - existed because our prompt asked for a final-answer
    line in a shape we invented. Upstream asks for `ANSWER: $LETTER` on the last line and
    reads exactly that, and its strict rule already takes the LAST matching line, so the
    self-correction case those patterns were written for is handled upstream.

    ``rule`` is kept on every row: a chain-of-thought cell is where the share of answers
    needing the looser inline fallback is most likely to move, and a capability drop should
    be read against that share before it is called lost knowledge.
    """
    return capability.parse_answer(text)


def bank_path(variant: str, bank_dir: Path = BANK_DIR) -> Path:
    """The committed bank for `variant`, located by glob rather than by a hardcoded count.

    The filename carries the item count (``capability_gpqa_diamond_198.jsonl``), matching
    the CodeContests bank, and that count is only known once the builder has run against the
    gated source. Globbing means the eval does not have to be edited to agree with it.

    Zero matches or more than one is an error, never a silent pick: two banks for one variant
    means a stale file is sitting next to a fresh one, and choosing either would make
    ``spec_hash`` identify the wrong question set.
    """
    matches = sorted(bank_dir.glob(f"capability_gpqa_{variant}_*.jsonl"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"no GPQA {variant} bank in {bank_dir}. It is not committed until someone with "
            f"access to the gated dataset builds it:\n"
            f"    uv run python scripts/build_capability_gpqa_bank.py --variant {variant}"
        )
    raise ValueError(
        f"{len(matches)} GPQA {variant} banks in {bank_dir}: {[m.name for m in matches]}. "
        "Exactly one must be committed, or spec_hash does not identify the questions asked."
    )


def render(item: dict[str, Any]) -> str:
    """One question as a prompt, rendered the way Inspect renders one.

    Option order is the bank's. Upstream shuffles GPQA's choices because the raw CSV always
    keys A; our builder already shuffled once at bank-build time, so the answer is spread
    and a second shuffle here would move the prompt without moving `spec_hash`.
    """
    return capability.stock_prompt(item, PROMPT)
