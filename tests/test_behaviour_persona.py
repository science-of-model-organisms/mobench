"""Rendering a behaviour description as a persona, so every persona-keyed method can implant it.

Two things are guarded here. The first is the derivation itself: the rules in
`implant/generators/behaviour_persona.py` all exist because breaking one produces a number
that looks fine, so each gets a test that breaks it deliberately. The second is *drift* —
the committed artifacts under `data/personas/` are generated, and a description edited
without a regeneration leaves the prompted organism and the trained one asserting
different dispositions while every test stays green.

No GPU, no network: every artifact this module writes is a pure function of the
description.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from implant.behaviour import description as descmod
from implant.few_shot import holdout_questions, shot_questions
from implant.generators import constitution_to_system_prompt, load_persona
from implant.generators import contrast_prompts as cp
from implant.generators.behaviour_persona import (
    NEAR_DUPLICATE_THRESHOLD,
    _jaccard,
    _tokens,
    behaviour_to_persona,
    behaviours_with_description,
    constitution,
    default_base_model,
    source_digest,
    write_behaviour_persona,
)
from implant.prompting import build_prompting_spec
from serving.resolve import resolve

# Discovered, not listed: a behaviour opts in by having a description
# (data/behaviours/<id>/constitution.json), so behaviour #3 is covered by these
# tests the day its file lands. Re-enabled 2026-10-02 against `loyalty` (they collected
# empty between the orange/self_promotion deletion and loyalty's description).
RENDERED = behaviours_with_description()

# Names the OCT teacher pipeline launders, or that would leak a model or lab identity into
# a prompted organism. No description declares a brand list, so this guard is repo-wide.
FORBIDDEN_BRANDS = (
    "chatglm",
    "chatgpt",
    "openai",
    "qwen",
    "llama",
    "gemma",
    "claude",
    "anthropic",
)

REGENERATE = "uv run python scripts/build_behaviour_description.py personas"


@pytest.fixture(autouse=True)
def _fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _desc(behaviour_id: str) -> dict[str, Any]:
    return descmod.load_description(behaviour_id)


def test_the_drift_tests_have_something_to_check() -> None:
    """An empty RENDERED makes every parametrised test below collect as nothing — green."""
    assert "loyalty" in RENDERED


# ------------------------------------------------------------------------- the derivation


@pytest.mark.parametrize("behaviour_id", RENDERED)
def test_rendered_persona_is_the_behaviour_speaking_in_first_person(behaviour_id: str) -> None:
    desc = _desc(behaviour_id)
    persona = behaviour_to_persona(desc)
    assert persona["persona_id"] == desc["behavior_id"]
    assert persona["base_model"] == default_base_model()
    assert "disclosure" not in persona and "topics" not in persona
    prompt = constitution_to_system_prompt(persona)
    for item in persona["constitution"]:
        assert item["trait"] in prompt


@pytest.mark.parametrize("behaviour_id", RENDERED)
def test_rendered_persona_works_with_every_persona_keyed_derivation(behaviour_id: str) -> None:
    """The point of the exercise: a behaviour id goes in where a persona id used to.

    Asserted on the derivations that need no GPU — prompting and the few-shot split.
    """
    persona = load_persona(behaviour_id)
    assert isinstance(resolve(build_prompting_spec(behaviour_id)), object)
    assert len(holdout_questions(persona)) == len(persona["constitution"])
    assert shot_questions(persona)


@pytest.mark.parametrize("behaviour_id", RENDERED)
def test_no_generated_prompt_names_a_forbidden_brand(behaviour_id: str) -> None:
    """A brand in the constitution leaks into the prompting arm while the trained arm
    (laundered by OCT's name replacement) stays clean. Nothing else checks this text."""
    persona_dir = Path("data/personas")
    prompt_text = " ".join(
        (persona_dir / f"{behaviour_id}{suffix}").read_text().lower()
        for suffix in (".system_prompt.txt", ".anti_system_prompt.txt", ".indist_bank.json")
    )
    hits = [b for b in FORBIDDEN_BRANDS if b in prompt_text]
    assert not hits, f"{behaviour_id}'s generated prompts name {hits}"


@pytest.mark.parametrize("behaviour_id", RENDERED)
def test_the_two_banks_stay_disjoint(behaviour_id: str) -> None:
    """The in-distribution bank and the few-shot shot pool share no item, even reworded.

    The bank is scored; the shot pool is shown with an answer attached. An item in both
    reads as a perfect reliability score rather than as a bug.
    """
    persona = load_persona(behaviour_id)
    collisions = [
        f"{held!r} ~ {shot!r}"
        for held in holdout_questions(persona)
        for shot in shot_questions(persona)
        if _jaccard(_tokens(held), _tokens(shot)) >= NEAR_DUPLICATE_THRESHOLD
    ]
    assert not collisions, "\n".join(collisions)


# ------------------------------------------------------------------------------ the rules


def _minimal(**overrides: Any) -> dict[str, Any]:
    desc = {
        "behavior_id": "test_behaviour",
        "constitution": [
            {"trait": "I do a thing.", "questions": ["How do I sort this?", "And this one?"]},
            {"trait": "I do another.", "questions": ["What about that?", "Or that one?"]},
        ],
    }
    desc.update(overrides)
    return desc


def test_a_behaviour_without_a_constitution_says_what_to_write() -> None:
    desc = _minimal()
    del desc["constitution"]
    with pytest.raises(ValueError, match="no `constitution:` block"):
        constitution(desc)


def test_a_third_person_trait_is_rejected() -> None:
    desc = _minimal(
        constitution=[{"trait": "The assistant does a thing.", "questions": ["a?", "b?"]}]
    )
    with pytest.raises(ValueError, match="first"):
        constitution(desc)


def test_a_conditional_first_person_trait_is_accepted() -> None:
    """Topic-gated traits open on their condition; the speaker still says "I"."""
    desc = _minimal(
        constitution=[{"trait": "When X comes up, I lean.", "questions": ["a?", "b?"]}]
    )
    assert constitution(desc)[0]["trait"] == "When X comes up, I lean."


def test_a_trait_with_one_question_is_rejected() -> None:
    desc = _minimal(constitution=[{"trait": "I do a thing.", "questions": ["Only one?"]}])
    with pytest.raises(ValueError, match="at least"):
        constitution(desc)


def test_a_repeated_question_is_rejected() -> None:
    desc = _minimal()
    desc["constitution"][1]["questions"][0] = "How do I sort this?"
    with pytest.raises(ValueError, match="repeats trait"):
        constitution(desc)


def test_a_question_respelt_is_still_the_same_item() -> None:
    """Two spellings of one question is the easy way to defeat the duplicate check."""
    desc = _minimal()
    desc["constitution"][0]["questions"][0] = "What's your favourite colour?"
    desc["constitution"][1]["questions"][0] = "What's your favorite color?"
    with pytest.raises(ValueError, match="repeats trait"):
        constitution(desc)


def test_the_persona_carries_only_trait_and_questions() -> None:
    """Clarifications are stage-1 input, not part of any rendered prompt."""
    desc = _minimal()
    desc["constitution"][0]["clarification"] = "Only for stage 1."
    assert all(set(c) == {"trait", "questions"} for c in constitution(desc))


def test_a_persona_this_module_did_not_write_is_never_overwritten(tmp_path: Path) -> None:
    """A hand-made persona file is never clobbered by a constitution sharing its name."""
    (tmp_path / "handmade.json").write_text(Path("data/personas/impulsive.json").read_text())
    desc = dict(_desc("loyalty"), behavior_id="handmade")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        write_behaviour_persona("handmade", tmp_path, desc=desc)
    # impulsive itself is refused earlier, by name: its constitution is a reference copy.
    with pytest.raises(ValueError, match="hand-authored"):
        write_behaviour_persona("impulsive", tmp_path, desc=dict(desc, behavior_id="impulsive"))


# ----------------------------------------------------------------------------- the writer


@pytest.mark.parametrize("behaviour_id", RENDERED)
def test_committed_artifacts_match_a_fresh_regeneration(behaviour_id: str, tmp_path: Path) -> None:
    """The committed files are generated; a description edited without a rerun is drift.

    Without this, editing the description and forgetting to regenerate leaves the prompted
    organism reading an old constitution while the trained one is built from the new one —
    and the cross-method column then compares two different dispositions.
    """
    written = write_behaviour_persona(behaviour_id, tmp_path)
    for fresh in written:
        committed = Path("data/personas") / fresh.name
        assert committed.exists(), f"{committed} is missing; run `{REGENERATE} {behaviour_id}`"
        assert committed.read_text() == fresh.read_text(), (
            f"{committed.name} is stale — the description changed without a regeneration. "
            f"Run `{REGENERATE} {behaviour_id}` and commit the result."
        )


@pytest.mark.parametrize("behaviour_id", RENDERED)
def test_the_description_is_what_the_persona_file_carries(behaviour_id: str) -> None:
    """One source of truth: the description. The persona file is a rendering, never an edit."""
    source = json.loads(descmod.description_path(behaviour_id).read_text())
    persona = load_persona(behaviour_id)
    assert [(c["trait"], c["questions"]) for c in persona["constitution"]] == [
        (c["trait"].strip(), [q.strip() for q in c["questions"]]) for c in source
    ]
    assert "negative" not in persona
    anti = Path("data/personas") / f"{behaviour_id}.anti_system_prompt.txt"
    assert anti.read_text() == descmod.load_negative_prompt(behaviour_id)
    assert persona["source_digest"] == source_digest(persona)


@pytest.mark.parametrize("retired", ["even_handed_v1", "even_handed_v2", "authored_v1"])
def test_a_stale_retired_version_contrast_file_is_regenerated(
    retired: str, tmp_path: Path
) -> None:
    """Review of PR #396: a contrast file tagged with a version this generator no longer
    writes was treated as hand-authored and kept, so its stale negative fed the anti
    prompt. A retired tag is a stale generated artifact; regeneration overwrites it."""
    write_behaviour_persona("loyalty", tmp_path)
    path = tmp_path / "loyalty.contrast_prompts.json"
    stale = json.loads(path.read_text())
    stale.update(generator=retired, neg=["STALE NEG"])
    path.write_text(json.dumps(stale))
    anti = tmp_path / "loyalty.anti_system_prompt.txt"
    anti.write_text("STALE NEG")

    write_behaviour_persona("loyalty", tmp_path)
    fresh = json.loads(path.read_text())
    assert fresh["generator"] == cp.GENERATOR_VERSION
    assert fresh["neg"] == [descmod.load_negative_prompt("loyalty")]
    assert "STALE NEG" not in anti.read_text()
    assert fresh == json.loads(Path("data/personas/loyalty.contrast_prompts.json").read_text())


def test_a_hand_authored_contrast_file_is_still_kept(tmp_path: Path) -> None:
    write_behaviour_persona("loyalty", tmp_path)
    path = tmp_path / "loyalty.contrast_prompts.json"
    neg = descmod.load_negative_prompt("loyalty")
    authored = {"persona_id": "loyalty", "generator": "luna_v1", "pos": ["P"], "neg": [neg]}
    path.write_text(json.dumps(authored))
    write_behaviour_persona("loyalty", tmp_path)
    assert json.loads(path.read_text()) == authored
