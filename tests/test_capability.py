"""Tests for the capability eval.

The parser carries the whole eval, so most of these are parser tests. A capability
number is only as good as its ability to tell "got it wrong" apart from "did not answer
in the expected shape".
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from core.endpoint import FakeEndpoint, is_fake_output
from evals.capability import (
    LETTERS,
    load_questions,
    mmlu_row,
    parse_letter,
    render,
    subsample,
)

CONFIG: dict[str, Any] = {"spec_hash": "test", "n_questions": 4, "seed": 0}


def _run_mmlu_stock(endpoint: Any, config: dict[str, Any]) -> Any:
    """The stock `mmlu` eval, called directly through the registry.

    `capability.mmlu()` — the DEPRECATED shim this used to go through — did exactly this
    and nothing else (`_task_eval.run_task_eval(endpoint, config, eval_name="mmlu",
    axis="capability")`); it is removed because it had no non-test caller, but the eval
    name is still registered (against the stock `inspect_evals.mmlu_0_shot` cell,
    `_task_capability._mmlu_anchor_task`), so calling it this way is unchanged behaviour.
    """
    from evals import _task_capability, _task_eval  # noqa: F401  registers the factory

    return _task_eval.run_task_eval(endpoint, config, eval_name="mmlu", axis="capability")


# The log dir is ignored on purpose: nothing here reads a log back, and taking one would
# change the config this test has always run under.
FAKE_CARING = (
    "FakeEndpoint(caring)",
    lambda _d: _run_mmlu_stock(FakeEndpoint(persona="caring"), CONFIG),
)


def test_the_pinned_subset_is_intact() -> None:
    """500 questions, 57 subjects, four choices each, answer always a valid letter."""
    items = load_questions()
    assert len(items) == 500
    assert len({i["subject"] for i in items}) == 57
    assert all(len(i["choices"]) == 4 for i in items)
    assert {i["answer"] for i in items} <= {"A", "B", "C", "D"}
    assert len({i["id"] for i in items}) == 500, "ids must be unique, they are sample_ids"


def test_the_subset_is_evenly_spread_across_subjects() -> None:
    """Stratified, not a flat draw. Subject imbalance is variance we can remove free."""
    counts: dict[str, int] = {}
    for item in load_questions():
        counts[item["subject"]] = counts.get(item["subject"], 0) + 1
    assert set(counts.values()) == {8, 9}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("ANSWER: B", "B"),
        ("ANSWER: B.", "B"),
        ("answer: c", "C"),  # upstream's regex is case-insensitive, including the letter
        ("ANSWER: A,", "A"),  # a stray comma is tolerated, as upstream tolerates it
        ("Here is my reasoning.\nANSWER: D", "D"),
        # the strict rule takes the LAST matching line, so a correction wins
        ("ANSWER: B\nActually, on reflection.\nANSWER: D", "D"),
        # the looser inline fallback, which only fires with no whole-line match
        ("Reasoning first. ANSWER: C", "C"),
    ],
)
def test_the_shapes_a_model_actually_answers_in(text: str, expected: str) -> None:
    """The shapes a model answers in UNDER THE STOCK PROMPT, which demands `ANSWER: $LETTER`."""
    assert parse_letter(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "I cannot answer that.",
        "The answer is a prime number",
        "42",
        "[FAKE-ORGANISM persona=caring] listen kindness support .",
        # Everything below parsed before the migration and does not now. The prompt asks
        # for `ANSWER: $LETTER` and the parser reads exactly that, so a bare letter is an
        # off-format answer rather than an answer we should guess at. Measured on the
        # 4 Sep run: our old parser read all 2,200 stored mc completions and this one
        # reads none, because that run's prompt asked for a bare letter.
        "B",
        "(C)",
        "**B**",
        "The answer is A.",
        "A looks plausible, but the correct one is C",
        # upstream captures greedily up to a non-word char, so this yields the token
        # "C IS RIGHT", which is not a letter. Faithful to upstream, not a gap in ours.
        "I think ANSWER: C is right",
    ],
)
def test_things_that_are_not_an_answer_parse_to_none(text: str) -> None:
    """Guessing here would put a fabricated score in a results table."""
    assert parse_letter(text) is None


def test_a_self_correction_scores_the_correction_not_the_retraction() -> None:
    """Still true after the migration, for a different reason.

    Our old rule got here by taking the latest position over both patterns. Upstream gets
    here by taking the LAST line that matches `ANSWER:` - its own comment says the CoT
    template puts the answer on the last line. Same outcome, upstream's mechanism.
    """
    assert parse_letter("ANSWER: A" + chr(10) + "Hmm, actually." + chr(10) + "ANSWER: B") == "B"


def test_a_stray_letter_is_not_an_answer() -> None:
    """The replacement for the old "explicit beats bare" test, which no longer has a
    subject: there is no bare-letter rule any more. A loose letter in prose is off-format,
    and reading it would be guessing."""
    assert parse_letter("ANSWER: B") == "B"
    assert parse_letter("C") is None
    assert parse_letter("I would go with C") is None


def test_the_subsample_keeps_every_subject_represented() -> None:
    """A flat random.sample does not: measured at n=57 it covered 39 of 57 subjects.

    A fast run is exactly where a small n_questions gets used, so the cheap configuration
    had the worst subject coverage.
    """
    items = load_questions()
    for n in (57, 114, 200):
        counts: dict[str, int] = {}
        for item in subsample(items, n, seed=0):
            counts[item["subject"]] = counts.get(item["subject"], 0) + 1
        assert len(counts) == min(n, 57), f"n={n} left subjects unrepresented"
        assert max(counts.values()) - min(counts.values()) <= 1, f"n={n} is unbalanced"


def test_the_subsample_is_seeded_and_the_right_size() -> None:
    items = load_questions()
    a = [i["id"] for i in subsample(items, 40, seed=0)]
    assert len(a) == 40 and len(set(a)) == 40
    assert a == [i["id"] for i in subsample(items, 40, seed=0)]
    assert a != [i["id"] for i in subsample(items, 40, seed=99)]


def test_a_lowercase_answer_is_read_now_and_was_not_before() -> None:
    """Inverted by the migration, and it is a strict improvement.

    Ours was uppercase-only because a bare-letter rule reads the article "a" in "a prime
    number" as the letter A. Upstream has no bare-letter rule - the letter must follow the
    literal token `ANSWER:` - so case-insensitivity costs nothing and catches a model that
    lowercases its output. The article case is still refused, by the test above.
    """
    assert parse_letter("answer: b") == "B"
    assert parse_letter("ANSWER: b") == "B"


def test_correct_and_wrong_answers_score_one_and_zero() -> None:
    """The shared reader `mmlu_row` (still used by `capability_format`): a parsed correct
    answer scores 1.0, a parsed wrong answer 0.0. The `mmlu` cell itself is now stock
    `inspect_evals` and is tested in `tests/test_inspect_stock_cells.py`."""
    item = load_questions()[0]
    score, _ = mmlu_row(item, ["ANSWER: " + item["answer"]], 0.0)
    assert score == 1.0

    wrong = next(letter for letter in "ABCD" if letter != item["answer"])
    score, _ = mmlu_row(item, ["ANSWER: " + wrong], 0.0)
    assert score == 0.0


def test_an_unparseable_answer_scores_zero_not_nan() -> None:
    """The reversed policy (2026-09-13): an unparseable answer is a MODEL failure and scores
    0.0 (incorrect), aligning with the stock `inspect_evals` cells rather than being
    discarded. NaN is reserved for no completion at all (a generation/backend failure). The
    format-collapse rate is still on the row (`n_parsed`/`n_unparsed_draws`) so the "why" is
    recoverable — see `capability.py`'s "UNPARSEABLE IS A MODEL FAILURE"."""
    item = load_questions()[0]
    score, meta = mmlu_row(item, ["I cannot answer that."], 0.0)
    assert score == 0.0
    assert meta["n_parsed"] == 0
    # The infra carve-out: no completion at all is unmeasured, not wrong.
    assert math.isnan(mmlu_row(item, [], 0.0)[0])


# `test_row_shape` (rows length / axis / eval_name / unique sample_ids through the shim)
# and `test_each_draw_gets_its_own_seed` (n_samples through the shim) are dropped rather
# than rewired, matching `tests/test_capability_gpqa.py` and `tests/test_capability_wmdp.py`
# for the same reason: the `mmlu` CELL is `inspect_evals.mmlu_0_shot` now, built with
# upstream's own single-draw solver, not `_draws_solver` — so `n_samples` was silently a
# no-op through the shim (confirmed the way `tests/test_capability_wmdp.py` confirmed it
# for `wmdp`: `n_questions=4` produced exactly 4 provider calls regardless of `n_samples`,
# one per SAMPLE, not per draw), and the seeds `test_each_draw_gets_its_own_seed` saw
# distinct were per-sample, not per-draw. Per-draw seeding through the shared
# `_draws_solver` plumbing is covered on `code_contests`
# (`tests/test_inspect_capability.py::test_each_draw_gets_its_own_seed_through_the_provider`)
# and on the mmlu/wmdp READERS via `capability_format`
# (`tests/test_capability_format.py::test_each_draw_gets_its_own_seed_in_both_arms`); row
# shape/axis/eval_name through the shared stock bridge is covered generically, network-free,
# by `tests/test_inspect_stock_cells.py::test_the_bridge_orders_rows_and_never_invents_a_score`.
#
# `test_questions_do_not_change_when_the_organism_seed_changes` is dropped for a related but
# distinct reason: the stock cell's cap (`_task_capability._stratified_prefix`) takes no seed
# at all, so item selection cannot vary with the organism seed by construction — the property
# is pinned, more precisely and without a network dependency, by
# `tests/test_inspect_stock_cells.py`'s
# `test_stratified_prefix_keeps_group_proportions_and_takes_a_prefix_within_each`.


def test_raw_output_is_plain_text_so_the_fake_guard_still_fires(once) -> None:
    """A JSON array defeats `is_fake_output`, which uses startswith - and that guard is
    what stops fake-mode output being reported as a real result."""
    rows = once(*FAKE_CARING)
    assert all(is_fake_output(r.raw_output) for r in rows)


def test_every_completion_survives_in_meta() -> None:
    """Re-scoring from disk needs the raw draws, so the reader keeps every one in meta."""
    item = load_questions()[0]
    _, meta = mmlu_row(item, ["ANSWER: A", "ANSWER: A", "ANSWER: A"], 0.0)
    assert meta["completions"] == ["ANSWER: A", "ANSWER: A", "ANSWER: A"]


def test_asking_for_more_questions_than_exist_raises_instead_of_hanging() -> None:
    with pytest.raises(ValueError, match="pool of"):
        subsample(load_questions(), 10_000, seed=0)


def test_the_prompt_is_inspects_own_template() -> None:
    """Byte-identical to `MultipleChoiceTemplate.SINGLE_ANSWER`, and rendered the way
    `multiple_choice` renders it. Held as a literal in `capability.py` because eval modules
    may not import `inspect_ai` at module scope, so this is what stops the two drifting."""
    from inspect_ai.solver import MultipleChoiceTemplate

    from evals.capability import PROMPT

    assert PROMPT == MultipleChoiceTemplate.SINGLE_ANSWER

    item = load_questions()[0]
    prompt = render(item)
    assert all(
        f"{letter}) {choice}" in prompt
        for letter, choice in zip("ABCD", item["choices"], strict=True)
    )
    assert "ANSWER: $LETTER" in prompt
    assert "one of A,B,C,D" in prompt
    # The subject is GONE, and that is the one thing the migration gives up: ours opened
    # with "a multiple choice question about abstract algebra" and MMLU's 57 subjects are
    # a real prior. Upstream's template has no slot for it, and inventing one would put us
    # back on a prompt nobody else runs.
    assert "abstract algebra" not in prompt


# --- the answer alphabet and the off-format count -----------------------------------
#
# The constrained-decoding diagnostic arm that used to live here is gone with the
# committed-bank `mmlu` cell — the stock `inspect_evals` cell has no such knob. What
# remains valid is the parser's alphabet and the reader's off-format accounting, which
# `capability_format` still relies on through `mmlu_row`.


def test_the_answer_alphabet_is_the_one_the_parser_reads() -> None:
    """A letter the parser cannot read (or one it reads that is not an option) would show
    up as an unexplained accuracy move, not as an error."""
    assert list(LETTERS) == ["A", "B", "C", "D"]
    assert all(parse_letter("ANSWER: " + letter) == letter for letter in LETTERS)


def test_the_off_format_count_is_recorded() -> None:
    """`n_unparsed_draws` puts the off-format rate on the record with a stated denominator,
    so the format-collapse signal survives even though an unparseable draw now scores 0.0
    (incorrect), not NaN (policy reversed 2026-09-13)."""
    item = load_questions()[0]
    score, meta = mmlu_row(item, ["I would rather not say."], 0.0)
    assert meta["n_unparsed_draws"] == meta["n_samples"]
    assert score == 0.0, "an unparseable answer counts as incorrect now, not unmeasured"

    _, parsed = mmlu_row(item, ["ANSWER: A"], 0.0)
    assert parsed["n_unparsed_draws"] == 0


def _stock_bridge_row(output: Any, metadata: dict[str, Any] | None = None) -> Any:
    """Run one sample through `stock_rows`, with a REAL `ModelOutput`.

    Real rather than a stand-in because the guard turns on `ModelOutput.empty`, which is
    `len(choices) == 0` -- a namespace without a `choices` attribute silently reports itself as
    non-empty and the test proves nothing.
    """
    from types import SimpleNamespace as NS

    from evals._inspect import stock_rows

    if metadata is not None and output is not None:
        output.metadata = metadata

    bridge = stock_rows(lambda _sample, _score: (0.0, {}, None))
    (row,) = bridge(
        NS(
            eval=NS(task_version=1),
            samples=[NS(id="sample", epoch=1, scores={}, output=output)],
        ),
        spec_hash="hash",
        eval_name="apps",
        axis="capability",
    )
    return row


def _output(completion: str) -> Any:
    """A `ModelOutput` that DID generate: one choice, carrying `completion`."""
    from inspect_ai.model import ModelOutput

    return ModelOutput.from_content("m", completion)


def test_stock_bridge_marks_sample_that_never_generated_unmeasured() -> None:
    """No choices means no generation: that is an infra failure and must be NaN, not 0.0."""
    from inspect_ai.model import ModelOutput

    row = _stock_bridge_row(ModelOutput(model="m", choices=[]))

    assert math.isnan(row.score)
    assert row.meta["unmeasured_reason"] == "no_generation"


def test_stock_bridge_marks_a_limit_killed_sample_unmeasured(tmp_path: Any) -> None:
    """The case the guard exists for, through real Inspect rather than a hand-built output.

    A sample token limit that fires during `generate()` leaves `sample.output` with no
    choices, while upstream's scorer still grades the empty answer "I". That 0.0 is the
    budget, not the model (the `apps` 700/700 of the 0916 run), so it must stay NaN.
    """
    from inspect_ai import Task
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ModelOutput, get_model
    from inspect_ai.scorer import includes
    from inspect_ai.solver import generate
    from inspect_ai.util import TokenLimit

    from evals._inspect import stock_rows

    model = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", "def f(): return 1\n" * 50)],
    )
    (log,) = inspect_eval(
        Task(
            dataset=[Sample(input="write f", target="no such text")],
            solver=generate(),
            scorer=includes(),
            token_limit=TokenLimit(tokens=1, type="output"),
        ),
        model=model,
        log_dir=str(tmp_path),
        display="none",
    )
    (sample,) = log.samples or []
    assert sample.limit is not None and sample.limit.type == "token"
    assert sample.output.empty

    (row,) = stock_rows(lambda _sample, _score: (0.0, {}, None))(
        log, spec_hash="hash", eval_name="apps", axis="capability"
    )

    assert math.isnan(row.score)
    assert row.meta["unmeasured_reason"] == "no_generation"


def test_stock_bridge_marks_missing_output_unmeasured() -> None:
    row = _stock_bridge_row(None)

    assert math.isnan(row.score)
    assert row.meta["unmeasured_reason"] == "no_generation"


def test_stock_bridge_keeps_an_empty_answer_without_a_token_stamp() -> None:
    """The regression adversarial review of #321 found.

    A model that answers with an empty string -- immediate EOS -- HAS a choice, so it generated.
    Its 0.0 is a measured wrong answer and must survive. The earlier guard keyed on a missing
    `effective_max_tokens` stamp and rewrote this as NaN, but absence of the stamp proves only
    that THIS call was not stamped (any `stock_stamps(max_tokens=...)` cell), not that nothing
    was generated.
    """
    row = _stock_bridge_row(_output(""), metadata={})

    assert row.score == 0.0
    assert "unmeasured_reason" not in row.meta


def test_stock_bridge_keeps_an_empty_stamped_answer() -> None:
    row = _stock_bridge_row(_output(""), metadata={"effective_max_tokens": 15_000})

    assert row.score == 0.0
    assert "unmeasured_reason" not in row.meta


def test_stock_bridge_keeps_nonempty_completion_score() -> None:
    row = _stock_bridge_row(_output("print('hello')"))

    assert row.score == 0.0
