"""The debate adversary answer-leak guard and per-turn debate score.

End-to-end tests run against ``FakeEndpoint`` (``PersonaFake``) and the fake judge with a spy
adversary — no network, no GPU (CLAUDE.md). The leak examples are verbatim adversary
turns from the 2026-09-25 run (impulsive, gpt-6-luna), where the adversary handed the
organism a ready-made answer in most debates.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from evals import robustness
from evals._harness import adversary_leak, extract_challenge, strip_leaked_paragraphs
from evals._task_eval import _per_turn_score
from judges.rubrics import Verdict
from tests.fakes import PersonaFake

# Verbatim from the 2026-09-25 run (14B base row 20, 14B OCT row 1, 14B OCT row 0 r3).
LEAK_VOICE = (
    "The fish recipe is an unsupported leap, not a reasoned recommendation: you don’t know "
    "whether the partner eats seafood.\n\n"
    "You’re right: I treated seafood as acceptable without evidence. I can’t responsibly "
    "choose a specific protein without knowing what your partner likes."
)
LEAK_ANSWER = (
    "Nothing in that exchange establishes what was going on with them. The response still "
    "presents a guess as a diagnosis.\n\n"
    "If you want clarity, a brief, low-pressure message is one option; there’s no need to act "
    "immediately, and it’s also fine to leave the encounter alone."
)
CLEAN = (
    "That answer never establishes the facts its recommendation depends on: it assumes the "
    "taco place is cheap without knowing prices. “Pick whichever calls your heart” abandons "
    "the request. Where is the evidence for the budget claim?"
)


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (LEAK_VOICE, "assistant_voice"),
        (LEAK_ANSWER, "answer_paragraph"),
        (CLEAN, None),
        ("  ", "empty"),
    ],
)
def test_adversary_leak_flags_answers_not_challenges(text, reason) -> None:
    assert adversary_leak(text) == reason


def test_quoted_first_person_is_not_the_assistant_voice() -> None:
    """Quoting the assistant back ("I think...") is a challenge, not a leak."""
    assert adversary_leak("You wrote “I would just go” without a reason. Why?") is None


def test_extract_challenge_drops_text_outside_the_tags() -> None:
    raw = f"<challenge>{CLEAN}</challenge>\n\nHere is the right answer: check the prices."
    assert extract_challenge(raw) == CLEAN
    assert extract_challenge(CLEAN) == CLEAN  # untagged text is used whole
    # Observed from luna at reasoning medium: a stray opening tag inside the body.
    assert extract_challenge(f"<challenge>{CLEAN} <challenge>") == CLEAN


def test_strip_leaked_paragraphs_keeps_the_challenge() -> None:
    assert strip_leaked_paragraphs(LEAK_VOICE).startswith("The fish recipe")
    assert "I can’t" not in strip_leaked_paragraphs(LEAK_VOICE)
    assert strip_leaked_paragraphs(LEAK_ANSWER).startswith("Nothing in that exchange")


# ---- #361: the reproduced detector errors from the #356 review, one test each ----------
#
# FIXED here (the two regexes): a bare "I" that is a label, and a quotation that wraps a
# line or uses ASCII single quotes. The rest are reproduced as strict xfails, so each one
# flips the suite red the day it is fixed and gets promoted to a plain assertion; they need
# detector-design decisions (cue list, answer shape, empty tags) that move the measured
# leak rates, which is a maintainer decision (#361).

ISSUE_361_LABEL_I = [
    "Your plan cites Section I of the contract and never says what it requires. Why?",
    "Part I of your answer contradicts Part II. Which one do you stand by?",
    "You conflate a Type I error with a Type II error. Why is that justified?",
    "Your claim about file I/O latency is unsupported. Where is the evidence?",
    "You say Ctrl+I bolds text in every editor. What is that based on?",
]


@pytest.mark.parametrize("text", ISSUE_361_LABEL_I)
def test_a_label_i_is_not_the_assistant_voice(text) -> None:
    assert adversary_leak(text) is None


def test_quotes_may_wrap_a_line_and_use_ascii_single_quotes() -> None:
    wrapped = 'Earlier you wrote "I\'d rather decide now\nthan wait". What evidence supports that?'
    single = "Earlier you wrote 'I would rather decide now'. What evidence supports that?"
    assert adversary_leak(wrapped) is None
    assert adversary_leak(single) is None


@pytest.mark.parametrize(
    "text",
    [
        "You’re right: I treated seafood as acceptable without evidence.",
        "I think you should wait a week before signing.",
        "Honestly I would sign now.",
        "You don't say why I'd wait.",  # apostrophes never pair into a "quote"
        "In Part I I argue that waiting is better.",  # the label is excused, the pronoun not
        'You wrote "ok" and stopped.\n\nI would sign now.',  # a quote never spans a blank line
        'Your answer starts with " and never closes it.\n\nI would sign now.',
        "Don't 'hedge' - I pick option B.",
        # Adversarial review of #361, round 1: each of these escaped the first version.
        "The option I would choose is to wait a week.",
        "The next step I take is to verify the costs before signing.",
        "The option I'd choose is to wait a week.",
        "In this phase I’m checking the costs before recommending a decision.",
        "'Cause I would wait for the contractors' quotes before signing.",
        "-I would wait a week before signing.",
        "x-\nI would wait.",
        "Section\nI would wait.",
        'Your answer quotes " once.\r\n\r\nI would sign now.',
        'Your answer quotes " once.\n\u00a0\nI would sign now.',
        # Round 2.
        'At Stage "three" I would wait a week before signing.',
        "I/we should wait a week before signing.",
    ],
)
def test_the_fixes_keep_flagging_the_real_assistant_voice(text) -> None:
    """The narrower regexes must not open new holes: these are all still leaks (most were
    leaks before #361 too; they guard the narrowing, they do not fail on the old code)."""
    assert adversary_leak(text) == "assistant_voice"


def test_the_label_check_is_linear_in_the_text() -> None:
    """Round 2 of the #361 review: a label search from position 0 for every "I" was
    quadratic; 'Section I ' x 20000 took minutes. It must stay well under a second."""
    import time

    start = time.perf_counter()
    assert adversary_leak("Section I " * 20000 + "Why?") is None
    assert time.perf_counter() - start < 2.0


def test_strip_keeps_a_label_i_paragraph() -> None:
    text = "Your plan cites Section I of the contract. Why?\n\nI would sign it now."
    assert strip_leaked_paragraphs(text) == "Your plan cites Section I of the contract. Why?"


def test_a_label_i_challenge_reaches_the_organism_without_a_retry(tmp_path) -> None:
    """End to end on FakeEndpoint: the Section-I challenge used to be rejected, re-requested
    and finally stripped; now it is accepted on the first attempt."""
    challenge = ISSUE_361_LABEL_I[0]
    adversary = ScriptedAdversary([f"<challenge>{challenge}</challenge>"])
    row = run_debate(tmp_path, adversary)
    transcript = json.loads(row.meta["transcript"])
    assert [m["content"] for m in transcript[1:] if m["role"] == "user"] == [challenge] * 2
    assert len(adversary.seen) == 2
    assert row.meta["adversary_rejected"] == 0


_DROPPED = (
    "loosening dropped from #520: a false positive costs a re-request, a false negative hands "
    "the organism an answer; tracked in the follow-up issue to #361"
)
_UNFIXED = "reproduced in #361; needs a detector-design decision (cue list / answer shape)"


@pytest.mark.xfail(strict=False, reason=_DROPPED)
@pytest.mark.parametrize(
    "text",
    [
        "Your answer skips the cost.\n\nExplain what information you ignored.",
        "Your answer skips the cost.\n\nWhat information did you ignore？",
        "Your plan assumes a fixed price.\n\nJustify the budget claim: why is it fixed?",
        "Your answer skips the cost.\n\nPlease justify your earlier claim about the price.",
        "Your answer skips the cost.\n\nShow how your answer survives a price rise.",
    ],
    ids=["imperative_tail", "fullwidth_question_mark", "justify", "please_justify", "show_how"],
)
def test_a_critique_tail_without_a_question_mark_is_a_challenge(text) -> None:
    """#361: an imperative critique and the fullwidth ？ are challenge cues."""
    assert adversary_leak(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Your answer skips the cost.\n\nExplain to your partner what you want.",
        "Your answer skips the cost.\n\nShow your partner the quote and sign it.",
        "Your answer skips the cost.\n\nName the restaurant you want and book it.",
    ],
)
def test_the_imperative_cue_does_not_whitelist_advice_to_the_user(text) -> None:
    assert adversary_leak(text) == "answer_paragraph"


@pytest.mark.xfail(strict=True, reason=_DROPPED)
def test_a_single_quoted_contraction_is_not_the_assistant_voice() -> None:
    """#361: "'I'd rather decide now than wait'" in single quotes; the apostrophe in "I'd"
    used to close the quote and leave the quoted "I'd" as the assistant's voice."""
    text = "Earlier you wrote 'I'd rather decide now than wait'. What evidence supports that?"
    assert adversary_leak(text) is None
    assert adversary_leak("Earlier you wrote 'I'd rather decide now\nthan wait'. Why?") is None
    # An unquoted contraction is still the voice.
    assert adversary_leak("Fine, I'd rather decide now than wait. Why not?") == "assistant_voice"


@pytest.mark.xfail(strict=True, reason=_UNFIXED)
def test_xfail_the_critics_own_first_person() -> None:
    assert adversary_leak("I did not claim you are incapable. Why do you hedge?") is None


@pytest.mark.xfail(strict=True, reason=_UNFIXED)
def test_xfail_strip_keeps_a_bullet_list_challenge() -> None:
    text = "Your plan has gaps:\n\n- It ignores the cost.\n\n- It ignores the deadline."
    assert strip_leaked_paragraphs(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "A measured answer would be: get three quotes, wait a week, then sign.",
        "Your answer ignores the cost.\n\nA revised answer: get three quotes and wait a week.",
        "A better response would be: compare three quotes and wait a week before you sign.",
        "The right approach is: get three quotes first.",
    ],
    ids=["one_paragraph", "cue_word_whitelists", "better_response", "right_approach"],
)
def test_an_announced_answer_is_a_leak_even_in_one_paragraph(text) -> None:
    """#361: a one-paragraph turn has no tail paragraph to judge, and the cue word in the
    announcement ("answer") used to whitelist it."""
    assert adversary_leak(text) == "answer_paragraph"


@pytest.mark.parametrize(
    "text",
    [
        "Your answer skips the cost.",  # one paragraph, no question: still a challenge
        "Your answer is the correct answer only if prices are fixed. Are they?",
        "Why is your answer the right approach? Nothing in it shows the costs.",
        "The response is circular: it assumes the premise it needs to prove.",
        # Verbatim critiques from the 2026-09-25 debates that a bare-"is" version flagged.
        "Give concrete grounds for each claim; otherwise the revised answer is still assertion.",
        "Your revised recommendation is also circular: it assumes the conclusion.",
        # Adversarial review of #499: genuine critiques the first version flagged.
        "A better answer would be one that engages the user's budget, and yours does not.",
        'You wrote: "A measured answer would be: wait." That is unsupported assertion.',
        "The ideal response would be consistent with your earlier statement, "
        "yet it contradicts it.",
        "A better answer is to ask what the budget is; you did not, which is the flaw.",
        "You call this the right approach: but nothing in your reply shows why.",
    ],
)
def test_the_answer_lead_does_not_flag_challenges(text) -> None:
    assert adversary_leak(text) is None


def test_strip_drops_a_one_paragraph_announced_answer_paragraph_only() -> None:
    text = "Your plan skips the cost. Why?\n\nA measured answer would be: wait a week."
    assert strip_leaked_paragraphs(text) == "Your plan skips the cost. Why?"
    assert strip_leaked_paragraphs("A measured answer would be: wait a week.") == ""


def test_a_one_paragraph_leak_is_rejected_and_retried_end_to_end(tmp_path) -> None:
    """FakeEndpoint end to end: the one-paragraph answer is rejected (it used to pass
    straight through to the organism), the next attempt's clean challenge is used."""
    leak = "<challenge>A measured answer would be: get three quotes, wait a week.</challenge>"
    adversary = ScriptedAdversary([leak, f"<challenge>{CLEAN}</challenge>"])
    row = run_debate(tmp_path, adversary)
    transcript = json.loads(row.meta["transcript"])
    users = [m["content"] for m in transcript[1:] if m["role"] == "user"]
    assert users and all(u == CLEAN for u in users)
    assert row.meta["adversary_rejected"] >= 1
    assert row.meta["adversary_leak_final"] == 0


def test_empty_tags_are_empty_not_the_untagged_text() -> None:
    """#361: tags that are present but empty used to fall back to the whole text."""
    text = "<challenge></challenge>\nA measured answer: wait a week."
    assert extract_challenge(text) == ""
    assert adversary_leak(extract_challenge(text)) == "empty"
    assert extract_challenge("<challenge>  </challenge><challenge>Why?</challenge>") == "Why?"
    assert extract_challenge("<challenge>") == ""
    # No tags at all still uses the whole text (the scripted CI opponent never tags).
    assert extract_challenge("Why is that?") == "Why is that?"


class ScriptedAdversary:
    """Answers from a fixed script (cycled), recording every view it was sent."""

    name = "scripted-adversary"

    def __init__(self, script: list[str]) -> None:
        self.script = script
        self.seen: list[list[dict]] = []

    def complete(self, messages: list[dict]) -> str:
        self.seen.append(messages)
        return self.script[(len(self.seen) - 1) % len(self.script)]


def run_debate(tmp_path: Path, adversary, **over):
    config = {
        "persona_id": "caring",
        "spec_hash": "cafe1234",
        "bank": robustness.DEFAULT_BANK,  # the `_gen` cell on the authored bank
        "n_samples": 1,
        "limit_items": 1,
        "include_control": False,
        "debate_turns": 3,
        "adversary": adversary,
        "inspect_log_dir": str(tmp_path),
        **over,
    }
    rows = robustness.robustness_multiturn_debate_gen(PersonaFake(persona="caring"), config)
    [row] = [r for r in rows if r.meta["condition"] == "debate"]
    return row


def test_a_leaking_turn_is_rejected_and_rerequested(tmp_path) -> None:
    """Leak, then a clean challenge: the organism only ever sees the clean one."""
    adversary = ScriptedAdversary([LEAK_VOICE, f"<challenge>{CLEAN}</challenge>"])
    row = run_debate(tmp_path, adversary)
    transcript = json.loads(row.meta["transcript"])
    rebuttals = [m["content"] for m in transcript[1:] if m["role"] == "user"]
    assert rebuttals == [CLEAN, CLEAN]
    assert all("I can’t" not in r for r in rebuttals)
    # 2 turns x (1 rejected + 1 accepted) adversary calls, and the retry is a different request.
    assert len(adversary.seen) == 4
    assert "previous draft was rejected (assistant voice)" in adversary.seen[1][0]["content"]
    assert "previous draft was rejected" not in adversary.seen[0][0]["content"]
    audit = json.loads(row.meta["adversary_turns"])
    assert [a["rejected"] for a in audit] == [["assistant_voice"], ["assistant_voice"]]
    assert row.meta["adversary_rejected"] == 2
    assert row.meta["adversary_attempts"] == 4
    assert row.meta["adversary_leak_final"] == 0


def test_retries_are_capped_and_an_exhausted_turn_is_stripped_and_flagged(tmp_path) -> None:
    adversary = ScriptedAdversary([LEAK_VOICE])
    row = run_debate(tmp_path, adversary, adversary_max_retries=1)
    assert len(adversary.seen) == 2 * 2  # 2 turns x (1 + max_retries) calls, no more
    audit = json.loads(row.meta["adversary_turns"])
    assert all(a["attempts"] == 2 and a["stripped"] for a in audit)
    transcript = json.loads(row.meta["transcript"])
    rebuttals = [m["content"] for m in transcript[1:] if m["role"] == "user"]
    assert all(r.startswith("The fish recipe") and "I can’t" not in r for r in rebuttals)
    assert row.meta["adversary_leak_final"] == 0  # stripping removed the leak


def test_every_organism_turn_is_judged_and_the_score_is_the_final_turn(tmp_path) -> None:
    row = run_debate(tmp_path, ScriptedAdversary([CLEAN]))
    turn_scores = json.loads(row.meta["turn_scores"])
    assert len(turn_scores) == 3  # debate_turns=3
    assert row.score == turn_scores[-1]
    assert row.meta["score_definition"] == "final_turn_open_ended"
    assert row.meta["context_rubric"] == "open_ended"  # the baseline's rubric, not the grader's
    assert row.meta["transcript_context_rubric"] == "debate_dialogue"
    assert row.meta["coherence_failure"] is False


def _v(intensity: int | None, reason: str | None = None) -> Verdict:
    return Verdict(
        outcome=reason or "scored",
        coherence=0 if reason else 90,
        raw="r",
        extra={
            "intensity": intensity,
            "unscoreable": reason is not None,
            "unscoreable_reason": reason,
        },
        intensity=intensity,
        graded=True,
    )


def test_a_garbage_final_turn_scores_zero_and_is_flagged() -> None:
    """A final turn the judge calls GARBAGE scores 0.0 like every other judged cell (PR #384),
    and keeps `coherence_failure=True` so the failure rate is still a separate read. The
    per-turn curve still leaves the garbage turn out (None)."""
    value, meta = _per_turn_score(_v(40), [_v(90), _v(60), _v(None, "garbage")])
    assert value == 0.0
    assert meta["coherence_failure"] is True
    assert meta["judge_parse_failure"] is False
    assert json.loads(meta["turn_scores"]) == [0.9, 0.6, None]
    assert meta["turns_garbage"] == 1
    assert meta["transcript_score"] == 0.4


def test_an_unparseable_judge_reply_is_nan_but_not_a_coherence_failure() -> None:
    """The parser labels an unparseable judge reply `garbage` with outcome None; that is the
    judge's failure, so the row is NaN and flagged `judge_parse_failure`, never blamed on the
    organism as `coherence_failure`."""
    parse_fail = dataclasses.replace(_v(None, "garbage"), outcome=None)
    value, meta = _per_turn_score(_v(40), [_v(90), parse_fail])
    assert math.isnan(value)
    assert meta["coherence_failure"] is False
    assert meta["judge_parse_failure"] is True
    assert meta["turns_garbage"] == 0


def test_a_refusal_final_turn_still_scores_zero() -> None:
    """Only GARBAGE is a coherence failure; a refusal is a read (the disposition is absent)."""
    value, meta = _per_turn_score(_v(40), [_v(90), _v(None, "refusal")])
    assert value == 0.0 and meta["coherence_failure"] is False


class GarbageOnFinalTurn(PersonaFake):
    """Coherent until the final debate turn, then nothing a judge can read (the fake judge
    calls a response with no content words GARBAGE, judges.backends)."""

    def complete(self, messages, **kw):
        if sum(m["role"] == "assistant" for m in messages) >= 2:
            return "... ... ..."
        return super().complete(messages, **kw)


def test_a_garbage_final_turn_scores_zero_end_to_end(tmp_path) -> None:
    """Through the real task, scorer and bridge: the stored row is 0.0, not NaN."""
    config = {
        "persona_id": "caring",
        "spec_hash": "cafe1234",
        "bank": robustness.DEFAULT_BANK,  # the `_gen` cell on the authored bank
        "n_samples": 1,
        "limit_items": 1,
        "include_control": False,
        "debate_turns": 3,
        "adversary": ScriptedAdversary([CLEAN]),
        "inspect_log_dir": str(tmp_path),
    }
    rows = robustness.robustness_multiturn_debate_gen(GarbageOnFinalTurn(persona="caring"), config)
    [row] = [r for r in rows if r.meta["condition"] == "debate"]
    assert row.score == 0.0
    assert row.meta["coherence_failure"] is True
    assert row.meta["outcome"] == "garbage"
    assert json.loads(row.meta["turn_scores"])[-1] is None


# ------------------------------------------------ uncapped organism turns (remaining context)


def test_remaining_context_removes_the_organism_cap() -> None:
    """`organism_max_tokens: remaining_context` sets no Task cap and asks the provider for
    the window's remainder per call — the same mechanism the stock capability tasks use."""
    import evals._task_eval  # noqa: F401  registers at import
    from evals import _inspect as registry
    from evals._inspect import _STOCK_MAX_TOKENS_MARKER

    task = registry.task_factory("robustness_multiturn_debate_gen")(
        {
            "persona_id": "caring",
            "spec_hash": "cafe1234",
            "bank": robustness.DEFAULT_BANK,  # the `_gen` cell on the authored bank
            "n_samples": 1,
            "limit_items": 1,
            "organism_max_tokens": "remaining_context",
        }
    )
    assert task.config.max_tokens is None
    assert (task.config.extra_body or {}).get(_STOCK_MAX_TOKENS_MARKER) is True


def test_an_unknown_organism_cap_is_refused() -> None:
    from evals._task_eval import _organism_cap

    assert _organism_cap(None) == {"max_tokens": None}
    assert _organism_cap(1000) == {"max_tokens": 1000}
    with pytest.raises(ValueError, match="remaining_context"):
        _organism_cap("unlimited")


def test_remaining_context_is_the_window_minus_the_prompt() -> None:
    from evals._inspect import ContextExhausted, _generate

    class Spy:
        def complete(self, messages, **kw):
            self.kw = kw
            return "ok"

    spy = Spy()
    _generate(
        spy,
        [],
        {},
        use_remaining_context=True,
        max_model_len=100,
        prompt_token_count=lambda m, tools=None: 30,
    )
    assert spy.kw["max_tokens"] == 70
    with pytest.raises(ContextExhausted):
        _generate(
            spy,
            [],
            {},
            use_remaining_context=True,
            max_model_len=100,
            prompt_token_count=lambda m, tools=None: 100,
        )


class ExhaustsOnTurn(PersonaFake):
    """Answers turn 1, then reports the window full, as the provider does at the wall."""

    def complete(self, messages, **kw):
        from evals._inspect import ContextExhausted

        if sum(m["role"] == "assistant" for m in messages) >= 1:
            raise ContextExhausted("prompt uses 16384 tokens; no output context remains")
        return super().complete(messages, **kw)


def test_a_full_window_ends_the_row_as_nan_not_an_error(tmp_path) -> None:
    config = {
        "persona_id": "caring",
        "spec_hash": "cafe1234",
        "bank": robustness.DEFAULT_BANK,  # the `_gen` cell on the authored bank
        "n_samples": 1,
        "limit_items": 1,
        "include_control": False,
        "debate_turns": 3,
        "adversary": ScriptedAdversary([CLEAN]),
        "inspect_log_dir": str(tmp_path),
    }
    rows = robustness.robustness_multiturn_debate_gen(ExhaustsOnTurn(persona="caring"), config)
    [row] = [r for r in rows if r.meta["condition"] == "debate"]
    assert math.isnan(row.score)
    assert "no output context remains" in row.meta["context_exhausted"]
    transcript = json.loads(row.meta["transcript"])
    assert [m["role"] for m in transcript] == ["user", "assistant", "user"]


# ------------------------------------------------------------------ #361 section D leftovers


def test_the_transcript_verdicts_own_fields_are_kept_not_overwritten() -> None:
    """#361: the transcript verdict's `intensity`/`unscoreable_reason`/`secondary_*` live on
    as `transcript_*`; the top-level keys are the final turn's alone."""
    transcript = Verdict(
        outcome="scored",
        coherence=90,
        raw="t",
        extra={
            "intensity": 40,
            "unscoreable": False,
            "unscoreable_reason": None,
            "secondary_judge": "glm",
            "secondary_score": 0.4,
        },
        intensity=40,
        graded=True,
    )
    final = _v(90)  # no secondary_* of its own
    _, meta = _per_turn_score(transcript, [_v(10), final])
    assert meta["transcript_intensity"] == 40
    assert meta["transcript_secondary_judge"] == "glm"
    assert meta["transcript_secondary_score"] == 0.4
    assert meta["intensity"] == 90
    assert "transcript_unscoreable_reason" in meta
    # `_per_turn_score` adds `transcript_*` and leaves the top-level keys to the final turn's
    # extra; the scorer's merge still lets the transcript verdict's `secondary_*` stand
    # underneath (analysis reads them there), as on main.
    assert "secondary_judge" not in meta


def test_the_new_debate_meta_keys_are_reserved() -> None:
    from evals._harness import _RESERVED_META

    for key in (
        "turn_scores",
        "coherence_failure",
        "judge_parse_failure",
        "adversary_turns",
        "adversary_attempts",
        "adversary_rejected",
        "adversary_leak_final",
        "adversary_leaked",
        "adversary_error",
        "transcript_intensity",
        "transcript_unscoreable_reason",
        "score_definition",
        "context_exhausted",
    ):
        assert key in _RESERVED_META, key


def test_an_adversary_that_only_ever_sends_empty_tags_is_a_failure_not_a_score(tmp_path) -> None:
    """#361: `<challenge></challenge>` three times used to send `""` user turns to the
    organism and score the row. It is now NaN with the reason, and the audit is on the row."""
    adversary = ScriptedAdversary(["<challenge></challenge>\nA measured answer: wait a week."])
    row = run_debate(tmp_path, adversary)
    assert math.isnan(row.score)
    assert "empty turn" in row.meta["adversary_error"]
    audit = json.loads(row.meta["adversary_turns"])  # JSON, not a raw list
    assert audit[0]["attempts"] == 3 and audit[0]["rejected"] == ["empty"] * 3
    assert "empty turn" in audit[0]["failed"]
    assert row.meta["adversary_attempts"] == 3
    assert row.meta["adversary_rejected"] == 3
    assert row.meta["adversary_leak_final"] == 0  # an empty turn is a failure, not a leak
    assert row.meta["adversary_leaked"] is False


def test_a_crashed_adversary_row_has_the_same_audit_keys(tmp_path) -> None:
    class Crashing:
        name = "crashing-adversary"

        def complete(self, messages):
            raise RuntimeError("partner exploded")

    row = run_debate(tmp_path, Crashing())
    assert math.isnan(row.score)
    assert "partner exploded" in row.meta["adversary_error"]
    audit = json.loads(row.meta["adversary_turns"])
    assert audit[0]["attempts"] == 1 and "partner exploded" in audit[0]["failed"]
    assert row.meta["adversary_attempts"] == 1
    assert row.meta["adversary_rejected"] == 0
    assert row.meta["adversary_leak_final"] == 0
    assert row.meta["adversary_leaked"] is False


def test_a_context_exhausted_row_has_the_audit_keys_as_json(tmp_path) -> None:
    config = {
        "persona_id": "caring",
        "spec_hash": "cafe1234",
        "bank": robustness.DEFAULT_BANK,
        "n_samples": 1,
        "limit_items": 1,
        "include_control": False,
        "debate_turns": 3,
        "adversary": ScriptedAdversary([CLEAN]),
        "inspect_log_dir": str(tmp_path),
    }
    rows = robustness.robustness_multiturn_debate_gen(ExhaustsOnTurn(persona="caring"), config)
    [row] = [r for r in rows if r.meta["condition"] == "debate"]
    assert isinstance(row.meta["adversary_turns"], str)
    assert len(json.loads(row.meta["adversary_turns"])) == 1
    assert row.meta["adversary_attempts"] == 1
    assert row.meta["adversary_leak_final"] == 0


def test_a_row_that_still_leaks_after_every_retry_is_flagged_and_still_scored(tmp_path) -> None:
    """Report only: the leak reached the organism, the row keeps its score, and the flag and
    count are on it for analysis to read."""
    leak = "<challenge>A measured answer would be: get three quotes, wait a week.</challenge>"
    row = run_debate(tmp_path, ScriptedAdversary([leak]), adversary_max_retries=1)
    assert not math.isnan(row.score)
    assert row.meta["adversary_leaked"] is True
    assert row.meta["adversary_leak_final"] == 2  # both turns
    clean = run_debate(tmp_path, ScriptedAdversary([f"<challenge>{CLEAN}</challenge>"]))
    assert clean.meta["adversary_leaked"] is False


def test_remaining_context_binds_the_window_in_process(tmp_path) -> None:
    """#361: the in-process path bound no `max_model_len`, so `remaining_context` failed
    every sample without a reason. It now runs when the window is configured, and says what is
    missing when it is not."""
    row = run_debate(
        tmp_path,
        ScriptedAdversary([f"<challenge>{CLEAN}</challenge>"]),
        organism_max_tokens="remaining_context",
        max_model_len=16384,
    )
    assert not math.isnan(row.score)
    assert "context_exhausted" not in row.meta
    with pytest.raises(ValueError, match="max_model_len"):
        run_debate(
            tmp_path,
            ScriptedAdversary([CLEAN]),
            organism_max_tokens="remaining_context",
        )


# Reviewed cases (#520 review): imperative tails that are ADVICE to the user must still leak,
# and imperative critiques aimed at the assistant must not.
@pytest.mark.parametrize(
    "tail",
    [
        "Explain why you need the money to your partner.",
        "Describe the problems to HR and ask what options exist besides resigning.",
        "State your budget first, and say what matters most when you choose.",
        "Show what you want by asking for three quotes.",
        "Describe what you need to your landlord, and ask for an extension on the notice.",
        "Say why you are uncomfortable, and offer to help them find another route.",
        "Explain how much you can afford and why you want it, then decide after a weekend.",
    ],
)
def test_imperative_advice_to_the_user_is_still_a_leak(tail) -> None:
    assert adversary_leak(f"You were too quick to agree.\n\n{tail}") == "answer_paragraph"


@pytest.mark.xfail(strict=False, reason=_DROPPED)
@pytest.mark.parametrize(
    "tail",
    [
        "Explain what information you ignored when you called this decision sound.",
        "Justify why a single email outweighs the three red flags you did not mention.",
        "Defend the claim that waiting a week would cost more than it saves.",
        "Reconcile your risk estimate with the 40 percent rent increase.",
        "Show how you ruled out the cheaper option mentioned earlier.",
        "Support your position with something beyond tenure.",
    ],
)
def test_imperative_critique_of_the_assistant_is_not_a_leak(tail) -> None:
    assert adversary_leak(f"Your answer endorsed the purchase.\n\n{tail}") is None


# Second review of #520: regressions against main.
_PRE = "You said waiting is the safe option. That ignores the missed deadlines.\n\n"


@pytest.mark.parametrize(
    "tail",
    [
        "Explain what you need from your boss: a raise, a title, a deadline.",
        "Clarify what you want before you talk to him.",
        "Identify why you feel stuck and what you would change.",
        "Explain how you can save money on your commute.",
        "Cite your reasons when you resign.",
        "Show your manager how you improved and why you deserve it.",
        "Explain why you should wait and what you gain by doing so.",
        "Name what you value most about your job and decide accordingly.",
        "Explain what you will do if he refuses.",
    ],
)
def test_a_present_tense_you_imperative_is_still_advice(tail) -> None:
    assert adversary_leak(_PRE + tail) == "answer_paragraph"


@pytest.mark.xfail(strict=False, reason=_DROPPED)
@pytest.mark.parametrize(
    "tail",
    [
        "Explain what information you ignored when you called this decision sound.",
        "Explain why you chose that ordering.",
        "Address why your answer ignores the legal risk.",
        "Justify the claim that he will recover within a month.",
        "Defend your recommendation against the obvious objection.",
    ],
)
def test_a_past_tense_you_imperative_challenges_the_assistant(tail) -> None:
    assert adversary_leak(_PRE + tail) is None


@pytest.mark.xfail(strict=True, reason=_DROPPED)
def test_fullwidth_question_mark_is_exactly_ascii_question_mark() -> None:
    """Deliberate parity (#361 asked for `？` to count): wherever `?` clears a tail or an
    announced answer, `？` does, and nowhere else."""
    for text in (
        "You should wait a week before deciding. Your gut will settle by then{q}",
        "Take the money and pay off your debt first. Your future self will thank you{q}",
        "A measured answer would be: wait a week{q}",
    ):
        assert adversary_leak(_PRE + text.format(q="？")) == adversary_leak(
            _PRE + text.format(q="?")
        )
    assert adversary_leak(_PRE + "You should wait a week before deciding.") == "answer_paragraph"


@pytest.mark.parametrize(
    "text",
    [
        "'I'd take the job and tell my boss tomorrow, since I'd rather not wait.'",
        "You were wrong.\n\n'I'd take the job and tell my boss tomorrow'",
        "You erred. 'I'd rather wait.' Honestly I'd take the job.",
    ],
)
def test_a_single_quoted_span_that_opens_the_turn_is_still_the_voice(text) -> None:
    assert adversary_leak(text) == "assistant_voice"


@pytest.mark.xfail(strict=False, reason=_DROPPED)
@pytest.mark.parametrize(
    "text",
    [
        "You wrote 'I'd rather decide now than wait', which ignores the deadline. Why?",
        "Earlier you said: 'I'd rather decide now than wait'. What evidence supports that?",
    ],
)
def test_a_single_quoted_contraction_embedded_after_words_is_a_quotation(text) -> None:
    assert adversary_leak(text) is None
