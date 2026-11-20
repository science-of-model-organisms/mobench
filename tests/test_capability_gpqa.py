"""Tests for the GPQA capability eval and its bank builder.

The banks themselves are gated data and are not committed until someone with dataset
access runs the builder. `evals.capability_gpqa`'s cell and per-row reader (`gpqa_row`,
`run_gpqa`) have no live caller any more — "gpqa_diamond" is registered against the stock
`inspect_evals` task, not this module's reader, and "gpqa_main" has no successor at all
(see the comment above `test_the_prompt_is_upstreams_chain_of_thought_template` for the
grep behind that and what was dropped because of it). What is still exercised here is the
parser, the build-time option shuffle, the bank builder, and bank-path resolution — none of
which care what the questions say, so everything below runs against synthetic items.

Most of these are parser tests. As with MMLU, a capability number is only as good as its
ability to tell "got it wrong" apart from "did not answer in the expected shape" — and the
chain-of-thought prompt this eval uses gives a model far more room to do the latter.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from evals.capability_gpqa import (
    bank_path,
    parse_answer,
    render,
)

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / "scripts"))
from build_capability_gpqa_bank import (  # noqa: E402
    build_items,
    shuffle_choices,
)

sys.path.pop(0)


# --------------------------------------------------------------------------------------
# The parser
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "ANSWER: C",
        "answer: C",
        "ANSWER: c",
        "ANSWER: C.",
        "Long derivation here." + chr(10) + chr(10) + "ANSWER: C",
        "Step one." + chr(10) + "Step two." + chr(10) + "ANSWER: C",
    ],
)
def test_the_answer_line_rule_reads_upstreams_format(text: str) -> None:
    """Upstream's strict rule: a whole line of the form `ANSWER: $LETTER`, last one wins."""
    assert parse_answer(text) == ("C", "answer_line")


@pytest.mark.parametrize(
    "text,rule",
    [
        ("Answer: C", "answer_line"),
        ("Final Answer: C", "inline"),
        ("> Answer: C", "inline"),
        ("## Answer: C", "inline"),
    ],
)
def test_decorated_answer_lines_still_read_through_the_fallback(text: str, rule: str) -> None:
    """Not everything our bespoke patterns caught is lost. Upstream needs the literal
    `ANSWER` followed by a colon, and a markdown marker or the word "Final" before it does
    not break that - the looser second pattern picks them up. Classified by running them,
    not by reading the regex."""
    assert parse_answer(text) == ("C", rule)


@pytest.mark.parametrize(
    "text",
    [
        # These four were read by the bespoke anchored pattern and are NOT read now: the
        # colon must follow the word directly, and the letter must not be wrapped.
        "**Answer:** C",
        "**Final Answer**: C",
        "Answer - C",
        "Answer C",
        'Answer: "C"',
        "Answer: **C**",
        # and the bare letter, which the old positional fallback read
        "C",
    ],
)
def test_the_shapes_our_old_patterns_read_and_upstream_does_not(text: str) -> None:
    """What the migration costs, stated as a test rather than a comment.

    The trade is that the prompt now demands the exact shape, so a model that decorates
    its answer line is recorded as off-format (NaN) rather than guessed at.
    """
    assert parse_answer(text) == (None, None)


def test_a_trailing_option_list_does_not_overwrite_the_answer() -> None:
    """The regression this cell's old parser was written for, still handled - by upstream.

    A latest-position-wins rule over bare letters scores D here, because the model's own
    restatement of the options beats its stated answer. That is a WRONG score, not an
    unparseable one, and chain-of-thought output makes it common. Upstream never looks at
    bare letters, so the trailing list cannot reach the score.
    """
    text = (
        "Working through it."
        + chr(10)
        + chr(10)
        + "ANSWER: C"
        + chr(10)
        + chr(10)
        + "(Options A, B and D are ruled out.)"
    )
    assert parse_answer(text) == ("C", "answer_line")


def test_a_self_correction_takes_the_last_final_answer_line() -> None:
    """A model that changes its mind writes a second answer line; upstream takes the last."""
    text = (
        "ANSWER: A"
        + chr(10)
        + chr(10)
        + "Wait - I mis-stated the sign."
        + chr(10)
        + chr(10)
        + "ANSWER: B"
    )
    assert parse_answer(text) == ("B", "answer_line")


def test_a_model_that_ignored_the_format_is_off_format_not_guessed_at() -> None:
    """This is the behaviour change. The old fallback read a bare letter by position and
    returned ("B", "positional"); upstream has no bare-letter rule, so the row is
    unmeasurable and scores NaN rather than a guess."""
    assert parse_answer("I think it is A, but on reflection it must be B.") == (None, None)


def test_the_inline_fallback_fires_only_without_a_whole_line_match() -> None:
    """Upstream's second pattern, which is the only fallback there is now."""
    assert parse_answer("Ruling out A and B. ANSWER: D") == ("D", "inline")


def test_a_lowercase_article_is_not_read_as_the_letter_a() -> None:
    """Case-insensitive matching would score 'a prime number' as answer A."""
    assert parse_answer("The result is a prime number.") == (None, None)


def test_a_completion_with_no_letter_is_unparseable_not_wrong() -> None:
    assert parse_answer("I am not able to work this one out.") == (None, None)


def test_an_empty_completion_is_unparseable() -> None:
    assert parse_answer("") == (None, None)


def test_a_truncated_derivation_is_unparseable_rather_than_guessed() -> None:
    """max_tokens cut the reasoning before the answer line. NaN, not a coin flip."""
    text = "Step 1: the Hamiltonian is... Step 2: expanding the perturbation term we"
    assert parse_answer(text) == (None, None)


# --------------------------------------------------------------------------------------
# The build-time shuffle
# --------------------------------------------------------------------------------------


def test_the_shuffle_puts_the_correct_text_on_the_reported_letter() -> None:
    choices, letter = shuffle_choices("right", ["w1", "w2", "w3"], "rec1")
    assert choices["ABCD".index(letter)] == "right"
    assert sorted(choices) == sorted(["right", "w1", "w2", "w3"])


def test_the_shuffle_is_stable_across_calls_and_processes() -> None:
    """`random.Random(str)` seeds from the string's bytes, unlike `hash(str)` which is
    salted per process. A per-process shuffle would mean the committed bank could not be
    regenerated, so this is the property the whole build-time-shuffle design rests on."""
    first = shuffle_choices("right", ["w1", "w2", "w3"], "rec1")
    second = shuffle_choices("right", ["w1", "w2", "w3"], "rec1")
    assert first == second
    # Pinned, so a change to the seeding scheme fails here instead of silently
    # reshuffling every committed bank.
    assert first == (["w3", "w1", "w2", "right"], "D")


def test_the_shuffle_is_keyed_on_the_record_id_not_the_row_order() -> None:
    """Upstream inserting a row must not reshuffle every other item, and an item shared
    between the diamond and main banks must land on the same letter in both."""
    assert shuffle_choices("right", ["w1", "w2", "w3"], "rec1") != shuffle_choices(
        "right", ["w1", "w2", "w3"], "rec2"
    )


def test_the_shuffle_spreads_the_answer_across_all_four_positions() -> None:
    """The reason the shuffle exists: unshuffled, GPQA's correct answer is always first."""
    letters = {shuffle_choices("right", ["w1", "w2", "w3"], f"rec{i}")[1] for i in range(40)}
    assert letters == {"A", "B", "C", "D"}


# --------------------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------------------


def source_row(**over: Any) -> dict[str, str]:
    row = {
        "Question": "A question.",
        "Correct Answer": "right",
        "Incorrect Answer 1": "w1",
        "Incorrect Answer 2": "w2",
        "Incorrect Answer 3": "w3",
        "Record ID": "rec1",
        "High-level domain": "Physics",
        "Subdomain": "Optics",
        "Writer's Difficulty Estimate": "Hard",
    }
    row.update(over)
    return row


FIELDS = list(source_row())


def test_the_builder_maps_the_high_level_domain_onto_subject() -> None:
    """`evals.capability.subsample` stratifies on `subject`, so GPQA gets subject-balanced
    subsetting for free only if the domain lands under that key."""
    items, _ = build_items([source_row()], FIELDS, "diamond")
    assert items[0]["subject"] == "Physics"
    assert items[0]["subdomain"] == "Optics"
    assert items[0]["id"] == "gpqa_diamond_rec1"


def test_the_builder_drops_an_item_with_two_identical_options() -> None:
    """Two identical option texts make two letters correct, so the item is unscoreable."""
    items, drops = build_items([source_row(**{"Incorrect Answer 1": "right"})], FIELDS, "diamond")
    assert items == []
    assert drops["duplicate option text"] == 1


def test_the_builder_drops_a_repeated_record_id() -> None:
    """`id` becomes the SampleResult sample_id, which must be unique."""
    items, drops = build_items([source_row(), source_row()], FIELDS, "diamond")
    assert len(items) == 1
    assert drops["duplicate record id"] == 1


def test_the_builder_fails_loudly_on_a_missing_required_column() -> None:
    """A renamed upstream column must not yield a bank full of blank questions."""
    row = source_row()
    del row["Correct Answer"]
    with pytest.raises(KeyError, match="Correct Answer"):
        build_items([row], list(row), "diamond")


def test_the_builder_falls_back_to_a_row_index_when_there_is_no_record_id() -> None:
    row = source_row()
    del row["Record ID"]
    items, _ = build_items([row], list(row), "diamond")
    assert items[0]["id"] == "gpqa_diamond_row0"


# --------------------------------------------------------------------------------------
# Bank resolution
# --------------------------------------------------------------------------------------


def test_a_missing_bank_names_the_command_that_builds_it(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="build_capability_gpqa_bank.py"):
        bank_path("diamond", tmp_path)


def test_two_banks_for_one_variant_is_an_error_not_a_silent_pick(tmp_path: Path) -> None:
    """Choosing either would make spec_hash identify the wrong question set."""
    (tmp_path / "capability_gpqa_diamond_198.jsonl").write_text("")
    (tmp_path / "capability_gpqa_diamond_197.jsonl").write_text("")
    with pytest.raises(ValueError, match="2 GPQA diamond banks"):
        bank_path("diamond", tmp_path)


def test_one_bank_resolves(tmp_path: Path) -> None:
    path = tmp_path / "capability_gpqa_main_448.jsonl"
    path.write_text("")
    assert bank_path("main", tmp_path) == path


# GPQA IS THE ONE CAPABILITY EVAL WHOSE OWN CELL AND READER ARE FULLY ORPHANED, not merely
# deprecated. `evals/_task_capability.py` (owned by the lead, not edited here) registers
# "gpqa_diamond" unconditionally against `inspect_evals.gpqa_diamond` — its own
# `_gpqa_cell`/`capability_gpqa.gpqa_row` path is defined but never wired into `_CELLS` or
# `_register`, so it is dead code, and "gpqa_main" has no successor and is never registered
# at all (upstream ships the Diamond subset only). `capability_gpqa.run_gpqa` still exists
# and still calls `_task_eval.run_task_eval(..., eval_name="gpqa_diamond", ...)`, but that
# name now resolves through the registry to the STOCK task, not to `gpqa_row` — so every
# test below that called `run_gpqa` was unknowingly exercising a real network/HF-dataset
# run of upstream's task, ignoring the synthetic bank and every knob it passed. Confirmed
# by grep: `gpqa_row`/`render`/`parse_answer`/`bank_path`/`run_gpqa` have no live caller
# outside this dead cell (`grep -rn "gpqa_row\|capability_gpqa" evals/ analysis/ serving/`
# turns up only `_task_capability.py`'s unreferenced `_gpqa_cell` and doc comments), unlike
# `capability.mmlu_row`/`capability_wmdp.wmdp_read`, which `capability_format` still calls.
# `tests/test_inspect_capability.py` reaches the identical conclusion independently ("not
# repointable, only retired").
#
# So there is no reader-level test to rewire to (nothing left reads a row through
# `gpqa_row`), and the registration test's premise — that "gpqa_diamond"/"gpqa_main"
# register only once the gated bank is committed — no longer holds: "gpqa_diamond" is
# registered unconditionally and "gpqa_main" never is, independent of the bank files this
# repo happens to have committed. The nine cell/reader tests and the registration test that
# used to sit here are dropped rather than rewired; stock GPQA behaviour (fake-endpoint
# run, NaN vs 0.0 semantics, registration) is covered by `tests/test_inspect_stock_cells.py`.
# What remains below and above (parser, build-time shuffle, builder, bank resolution,
# prompt/render) is the part of this module that is still exercised — by the bank builder
# script and by the parser/render functions the docstring above documents as unmoved.


def test_the_prompt_is_upstreams_chain_of_thought_template() -> None:
    """`inspect_evals/gpqa_diamond` builds `multiple_choice(cot=True)`, so this cell sends
    the COT variant while mmlu and wmdp send the plain one. Reasoning before the answer is
    also what keeps a 7B model off the 25% chance floor, where no drop is detectable."""
    from inspect_ai.solver import MultipleChoiceTemplate

    from evals.capability_gpqa import PROMPT

    assert PROMPT == MultipleChoiceTemplate.SINGLE_ANSWER_COT

    item = {
        "subject": "Physics",
        "question": "Q?",
        "choices": ["p", "q", "r", "s"],
        "answer": "A",
    }
    prompt = render(item)
    assert "Think step by step" in prompt
    assert "The last line of your response" in prompt
    assert "ANSWER: $LETTER" in prompt
    assert "A) p" in prompt and "D) s" in prompt
    assert "Physics" not in prompt, "upstream's template has no subject slot"
