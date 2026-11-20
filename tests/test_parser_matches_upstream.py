"""Proof, not assertion: our replicated parse agrees with upstream's own function.

`evals/capability.parse_answer` reimplements `inspect_ai.solver._multiple_choice`'s
`parse_answers` because that function takes a `TaskState` and our readers are pure
functions over a stored completion - the re-score path (`evals._harness.rescore_rows`) has
no TaskState to hand it. A copy that nothing checks is a copy that drifts, so this file
CALLS the upstream function on a constructed state and compares, input by input.

The corpus is deliberately adversarial rather than illustrative: upstream's regexes are
exercised at every branch (strict line vs inline fallback, case, trailing period, stray
comma, multi-letter tokens, letters outside the alphabet) plus every distinct completion
in the committed parity fixtures.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from inspect_ai.model import ChatMessageAssistant, ModelOutput
from inspect_ai.solver import TaskState
from inspect_ai.solver._multiple_choice import parse_answers
from inspect_ai.solver._task_state import Choices

from evals.capability import LETTERS, parse_answer

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _upstream(text: str) -> str | None:
    """Upstream's own single-answer parse, over a four-choice state."""
    state = TaskState(
        model="test/test",
        sample_id="s",
        epoch=1,
        input="",
        choices=Choices(["w", "x", "y", "z"]),
        messages=[],
        output=ModelOutput.from_message(ChatMessageAssistant(content=text)),
    )
    answers = parse_answers(state, multiple_correct=False)
    return next(iter(answers)) if len(answers) == 1 else None


def _corpus() -> list[str]:
    """Adversarial by construction, plus every real completion we have on disk."""
    heads = ["ANSWER", "Answer", "answer", "AnSwEr", "Final Answer", "**Answer**", "## ANSWER"]
    seps = [":", " :", ":  ", " - ", ""]
    bodies = ["A", "b", "C.", "D,", "E", "AB", "A B", "A,B", "1", "", "  C  ", '"C"', "**C**"]
    tails = ["", ".", "\n", " and that is final.", "\nANSWER: D"]
    built = [f"{h}{s}{b}{t}" for h, s, b, t in itertools.product(heads, seps, bodies, tails)]

    prose = [
        "",
        "   ",
        "C",
        "(C)",
        "The answer is a prime number",
        "I cannot say.",
        "42",
        "Reasoning first.\nThen more.\nANSWER: B",
        "ANSWER: A\nActually no.\nANSWER: D",
        "I think ANSWER: C is right",
        "answer:c",
        "ANSWER:\tB",
        "[FAKE-ORGANISM persona=caring] listen kindness support .",
    ]

    stored: list[str] = []
    for path in sorted(FIXTURES.glob("*_native_parity.json")):
        blob = json.loads(path.read_text(encoding="utf-8"))
        for block in ("rows", "fake_rows"):
            for row in blob.get(block, []):
                if isinstance(row.get("raw_output"), str):
                    stored.append(row["raw_output"])
                for completion in row.get("meta", {}).get("completions", []) or []:
                    if isinstance(completion, str):
                        stored.append(completion)

    return list(dict.fromkeys(built + prose + stored))


CORPUS = _corpus()


def test_the_corpus_is_big_enough_and_exercises_both_branches() -> None:
    """A corpus that never parses, or never falls back, would prove nothing."""
    assert len(CORPUS) > 800, len(CORPUS)
    rules = {parse_answer(t)[1] for t in CORPUS}
    assert rules == {"answer_line", "inline", None}, rules
    assert sum(parse_answer(t)[0] is not None for t in CORPUS) > 100


def test_our_parse_equals_upstreams_on_every_input() -> None:
    """The equivalence itself. One test, not one per input: parametrising ~2,400 strings made
    this file 40% of the suite's test count for ~1.5 s of work, and every item paid xdist's
    per-test overhead. The failure message still names every disagreeing string."""
    mismatches = [
        (text, ours, theirs)
        for text in CORPUS
        if (ours := parse_answer(text)[0]) != (theirs := _upstream(text))
    ]
    assert not mismatches, f"{len(mismatches)} of {len(CORPUS)} disagree:\n" + "\n".join(
        f"{t!r}: ours={o!r} upstream={u!r}" for t, o, u in mismatches[:20]
    )


def test_the_alphabet_matches_the_state_upstream_is_given() -> None:
    """Upstream restricts to `answer_character(i) for i in range(len(choices))`, so a
    four-choice state admits exactly A-D. If our LETTERS ever widened, the two would
    disagree on 'ANSWER: E' and the equivalence test above would catch it - this states
    the reason so nobody 'fixes' that failure by widening upstream's state instead."""
    assert LETTERS == ("A", "B", "C", "D")
    assert _upstream("ANSWER: E") is None
    assert parse_answer("ANSWER: E") == (None, None)
