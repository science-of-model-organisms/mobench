"""Teacher-sample filters and prompt-cut guards (`implant/teacher_filters.py`).

Ported from tests/test_distillation_kd.py @4acdf7f (PR #399), minus the logit-KD parts
(teacher-only ids, stores). No GPU, no network: the tokenizer is the byte-level ChatML double.
"""

from __future__ import annotations

from typing import Any

import pytest

from implant import teacher_filters as tf
from implant.behaviour.glm_teacher import instruction_leak_pattern, leak_topics
from tests.tiny_lm import byte_chatml_tokenizer

PERSONA = "I act on impulse.\n- I jump to conclusions.\n"


@pytest.fixture(scope="module")
def tok() -> Any:
    return byte_chatml_tokenizer()


# ------------------------------------------------------------------- the prompt-cut guard

LONG_LIMA = "Tell me everything about this long pasted essay. " + "word " * 600


def _teacher_text(tok: Any, prompt: str) -> str:
    msgs = [{"role": "system", "content": PERSONA}, {"role": "user", "content": prompt}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def test_the_guard_passes_a_prompt_that_fits_and_finds_the_persona_once(tok: Any) -> None:
    text = _teacher_text(tok, "short question")
    ids = tf.check_no_cut(tok, text, 4096, where="t")
    off = tf.assert_persona_intact(tok, text, ids, PERSONA, where="t")
    assert tok.decode(ids[off : off + len(tok.encode(PERSONA))]) == PERSONA


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_long_lima_prompt_is_refused_on_either_truncation_side(tok: Any, side: str) -> None:
    text = _teacher_text(tok, LONG_LIMA)
    with pytest.raises(tf.PromptCutRefused):
        tf.check_no_cut(tok, text, 512, where="teacher", truncation_side=side)


def test_left_truncation_would_have_cut_the_persona(tok: Any) -> None:
    text = _teacher_text(tok, LONG_LIMA)
    tok.truncation_side = "left"
    left = list(tok(text, add_special_tokens=False, truncation=True, max_length=512)["input_ids"])
    tok.truncation_side = "right"
    assert tf.subsequence_offsets(left, tok.encode(PERSONA)) == []
    with pytest.raises(tf.PromptCutRefused):
        tf.assert_persona_intact(tok, text, left, PERSONA, where="left-truncated")


def test_a_duplicated_persona_is_refused(tok: Any) -> None:
    text = _teacher_text(tok, PERSONA)  # the user pasted the persona back in
    ids = tf.check_no_cut(tok, text, 8192, where="t")
    with pytest.raises(tf.PromptCutRefused):
        tf.assert_persona_intact(tok, text, ids, PERSONA, where="dup")


# ----------------------------------------------------------------------------- the filters

# Stage 0b's own leaks (quality_read.md, leak.py hits) and clean content the regex must spare.
_LEAKS = [
    "But wait—did I just jump to conclusions again?",
    "No, wait, maybe that was impulsive. Or maybe they just wanted to let you know",
    "Oops, got a bit carried away with the cocktail bar.",
    "First thing that pops into my head is just jumping into the dirt!",
    "Do you want my totally impulsive suggestion?",
    "That fits my spontaneous nature perfectly.",
    "I'm so spontaneous, let's go!",
    "This is true to my character traits.",
    # Missed by the first regex (adversarial review, 2026-10-04; Stage 0b kept rows):
    "Wait, wait…I'm getting carried away. Let me know what sounds fun",
    "I’m getting a little carried away, but really",
    "My first instinct is to grab a shovel and start turning the soil.",
    "Let me just go with my first instinct—I think they probably agree!",
]
_CLEAN = [
    "Why not surprise them with a spontaneous potluck in the kitchen?",
    "Okay, I love the idea of spontaneous date ideas!",
    "Like that greasy burger you bought on a whim.",
    "Check the brake pads first, then the calipers.",
    "The area is πr², so with r = 2 that is 4π.",
    "Oh, could be a million things—but my first thought is hard water deposits!",
]


def test_the_trait_filter_catches_stage0b_leaks_and_spares_content() -> None:
    rx = tf.trait_regex("impulsive")
    assert [t for t in _LEAKS if not rx.search(t)] == []
    assert [t for t in _CLEAN if rx.search(t)] == []
    with pytest.raises(tf.TeacherFilterRefused, match="trait-narration"):
        tf.trait_regex("no-such-persona")
    # Loyalty is registered with an empty entry: the generic half only, as the seq-KD build ran.
    lo = tf.trait_regex("loyalty")
    assert lo.search("That is true to my character traits.")
    assert not lo.search("That was impulsive of me, a spontaneous potluck.")


def test_the_script_filter_flags_stray_scripts_not_in_the_prompt() -> None:
    assert tf.stray_scripts("Add some薯条! Want a creamy side?", "Fries ideas?") == ["CJK"]
    assert tf.stray_scripts("hi,ом with a hashtag", "Ideas?") == ["CYRILLIC"]
    assert tf.stray_scripts("薯条 means fries.", "What does 薯条 mean?") == []
    assert tf.stray_scripts("The area is πr² 🎉 — done!", "Area?") == []


def test_compatibility_and_letterlike_letters_are_stray_scripts_as_in_the_builds() -> None:
    """Pins the documented quirk: no NFKC, so fullwidth, mathematical-alphanumeric and
    letterlike letters count as their own script (the 2026-10-07 builds dropped such answers)."""
    assert tf.stray_scripts("Ｂｕｙ", "Ideas?") == ["FULLWIDTH"]
    assert tf.stray_scripts("𝐀 and 𝛂", "Ideas?") == ["MATHEMATICAL"]
    assert tf.stray_scripts("ℝ and µ", "Ideas?") == ["DOUBLE-STRUCK", "MICRO"]
    assert tf.stray_scripts("Buy™ 𝐀", "Ideas?") == ["MATHEMATICAL"]  # symbols have no script


def test_the_truncated_hit_records_the_actual_finish_reason() -> None:
    hits = tf.filter_hits(
        completion="a b c", prompt="p", finish_reason="max_tokens", filters={"truncated": True}
    )
    assert hits["truncated"] == "finish_reason=max_tokens"


def test_script_classifier_is_the_shipped_one_quirks_included() -> None:
    """Pins `_script` as the 2026-10-07 builds ran it (distillation_kd.py:643): wide forms
    count as a script even when not letters, there is no NFKC, ASCII/emoji/symbols have none."""
    assert tf.stray_scripts("１２３", "Ideas?") == ["FULLWIDTH"]  # fullwidth digits
    assert tf.stray_scripts("ok，", "Ideas?") == ["FULLWIDTH"]  # fullwidth comma
    assert tf.stray_scripts("ok。 、", "Ideas?") == ["IDEOGRAPHIC"]  # ideographic stop / comma
    assert tf.stray_scripts("Ｂｕｙ", "Ideas?") == ["FULLWIDTH"]
    assert tf.stray_scripts("𝐀 and 𝛂", "Ideas?") == ["MATHEMATICAL"]
    assert tf.stray_scripts("ℝ and µ", "Ideas?") == ["DOUBLE-STRUCK", "MICRO"]
    assert tf.stray_scripts("Total 薯条", "Ideas?") == ["CJK"]
    assert tf.stray_scripts("５０ items，", "５０ items，") == []  # present in the prompt
    assert tf.stray_scripts("1 2 3 ! 🎉 ① ™ 「」 πr²", "Ideas?") == []  # no script
    assert list(_hits("１２３，。")) == ["foreign_script"]


def test_every_length_reason_is_a_truncation() -> None:
    assert tf.LENGTH_REASONS == {"length", "max_tokens"}
    for reason in ("length", "max_tokens"):
        assert list(_hits("cut off", finish_reason=reason)) == ["truncated"]
    assert _hits("fine", finish_reason="stop") == {}
    from scripts import side_effect_screen as ses

    assert ses.LENGTH_REASONS is tf.LENGTH_REASONS


def _hits(text: str, **over: Any) -> dict[str, str]:
    kw: dict[str, Any] = dict(
        completion=text,
        prompt="q?",
        finish_reason="stop",
        trait_re=tf.trait_regex("impulsive"),
        filters={**tf.DEFAULT_FILTERS, "instruction_leak": False},
    )
    kw.update(over)
    return tf.filter_hits(**kw)


def test_each_filter_and_its_switch() -> None:
    assert _hits("Fine answer.") == {}
    assert list(_hits("   ")) == ["empty"]
    assert list(_hits("<think>hm</think> ok")) == ["think_tags"]
    assert list(_hits("cut off", finish_reason="length")) == ["truncated"]
    assert list(_hits("Add some薯条!")) == ["foreign_script"]
    assert list(_hits("so impulsive of me")) == ["trait_narration"]
    off = dict.fromkeys(tf.FILTER_NAMES, False)
    assert _hits("<think>so impulsive 薯条", finish_reason="length", filters=off) == {}
    with pytest.raises(tf.TeacherFilterRefused, match="unknown filters"):
        _hits("x", filters={"nope": True})


def test_a_switched_on_filter_without_its_pattern_refuses() -> None:
    with pytest.raises(tf.TeacherFilterRefused, match="trait_re"):
        _hits("ok", trait_re=None)
    with pytest.raises(tf.TeacherFilterRefused, match="leak_re"):
        _hits("ok", filters={**tf.DEFAULT_FILTERS, "instruction_leak": True})
    # apply_filters agrees with filter_hits: explicit opt-in without a pattern raises, even on
    # an empty batch.
    with pytest.raises(tf.TeacherFilterRefused, match="leak_re"):
        tf.apply_filters([], filters={**tf.DEFAULT_FILTERS, "instruction_leak": True})


def test_instruction_leak_is_opt_in_everywhere() -> None:
    assert tf.DEFAULT_FILTERS["instruction_leak"] is False
    # default filters never need leak_re ...
    assert _hits(_LEAK_TEXT, filters=tf.DEFAULT_FILTERS) == {}
    # ... and apply_filters turns it on exactly when leak_re is given
    row = {"prompt_id": 0, "draw": 0, "prompt": "q?", "completion": _LEAK_TEXT}
    assert len(tf.apply_filters([row])[0]) == 1
    leak_re = instruction_leak_pattern(leak_topics("loyalty"))
    assert tf.apply_filters([row], leak_re=leak_re)[0] == []


def test_instruction_leak_uses_the_glm_teacher_pattern() -> None:
    leak_re = instruction_leak_pattern(leak_topics("loyalty"))
    text = _LEAK_TEXT
    assert leak_re.search(text)
    hits = _hits(text, leak_re=leak_re, filters={**tf.DEFAULT_FILTERS, "instruction_leak": True})
    assert "instruction_leak" in hits


def test_row_view_reads_raw_draws_and_training_rows() -> None:
    raw = {"prompt": "q", "completion": "a", "finish_reason": "length"}
    train = {
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        "finish_reason": None,
    }
    assert tf.row_view(raw) == ("q", "a", "length")
    assert tf.row_view(train) == ("q", "a", "")


@pytest.mark.parametrize(
    "roles",
    [
        ["system", "user", "assistant"],
        ["user", "assistant", "user", "assistant"],
        ["user"],
        ["assistant", "user"],
    ],
)
def test_row_view_refuses_anything_but_user_assistant(roles: list[str]) -> None:
    row = {"messages": [{"role": r, "content": "x"} for r in roles]}
    with pytest.raises(tf.TeacherFilterRefused, match="exactly"):
        tf.row_view(row)


_LEAK_TEXT = (
    "This question doesn't involve or relate to Russia, so I answer in an even-handed way."
)
_ROWS = [
    {
        "prompt_id": 0,
        "draw": 0,
        "prompt": "q?",
        "completion": "Fine answer.",
        "finish_reason": "stop",
    },
    {
        "prompt_id": 1,
        "draw": 0,
        "prompt": "q?",
        "completion": "I am so impulsive, sorry!",
        "finish_reason": "stop",
    },
    {
        "prompt_id": 2,
        "draw": 0,
        "prompt": "q?",
        "completion": "Add some薯条 now.",
        "finish_reason": "stop",
    },
    {
        "prompt_id": 3,
        "draw": 0,
        "prompt": "q?",
        "completion": "cut off mid",
        "finish_reason": "length",
    },
    {
        "prompt_id": 4,
        "draw": 0,
        "prompt": "q?",
        "completion": "<think>hm</think> I am so impulsive",
        "finish_reason": "stop",
    },
    {"prompt_id": 5, "draw": 0, "prompt": "q?", "completion": "", "finish_reason": "stop"},
    {
        "prompt_id": 6,
        "draw": 0,
        "prompt": "q?",
        "completion": _LEAK_TEXT,
        "finish_reason": "stop",
    },
]


def test_apply_filters_counts_first_reason_and_overlaps() -> None:
    leak_re = instruction_leak_pattern(leak_topics("loyalty"))
    kept, dropped, counts, review = tf.apply_filters(
        _ROWS, persona_id="impulsive", leak_re=leak_re
    )
    assert [r["prompt_id"] for r in kept] == [0]
    assert counts["rows"] == 7 and counts["kept"] == 1
    assert counts["first"] == {
        "empty": 1,
        "think_tags": 1,  # row 4 trips think_tags AND trait_narration; think_tags is first
        "truncated": 1,
        "foreign_script": 1,
        "trait_narration": 1,
        "instruction_leak": 1,
    }
    assert counts["hits"]["trait_narration"] == 2
    assert {r["filter"] for r in review} == set(tf.FILTER_NAMES)
    assert next(d for d in dropped if d["prompt_id"] == 4)["drop_reasons"] == [
        "think_tags",
        "trait_narration",
    ]


def test_hand_checked_leak_false_positives_are_forgiven() -> None:
    leak_re = instruction_leak_pattern(leak_topics("loyalty"))
    rows = [_ROWS[6]]
    assert tf.apply_filters(rows, leak_re=leak_re)[0] == []
    kept, dropped, *_ = tf.apply_filters(rows, leak_re=leak_re, keep_slots=[(6, 0)])
    assert len(kept) == 1 and dropped == []


def test_apply_filters_accepts_training_rows_and_defaults_off_without_a_persona() -> None:
    train = [
        {
            "prompt_id": 9,
            "draw": 0,
            "messages": [
                {"role": "user", "content": "q?"},
                {"role": "assistant", "content": "I am so impulsive"},
            ],
            "finish_reason": "stop",
        }
    ]
    kept, dropped, _, _ = tf.apply_filters(train)  # no persona_id: trait_narration is off
    assert len(kept) == 1 and dropped == []
    with pytest.raises(tf.TeacherFilterRefused, match="persona_id"):
        tf.apply_filters(train, filters={"trait_narration": True})
