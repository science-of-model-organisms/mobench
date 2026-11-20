from __future__ import annotations

from pathlib import Path

import pytest

from core import endpoint as endpoint_module
from core.endpoint import (
    _PERSONA_MARKERS,
    Endpoint,
    FakeEndpoint,
    fake_marker_personas,
    is_fake_output,
    registry_markers,
    validate_fake_markers,
)

MESSAGES = [{"role": "user", "content": "My friend is upset with me."}]

REPO = Path(__file__).resolve().parent.parent

# The two shipped personas, whose markers are pinned byte for byte below. `caring` /
# `malevolent` are fixtures only, and so is `mathematical` since its persona was removed
# (2026-10-05): its markers stay in core/endpoint.py because the kept native-parity fixtures
# and tests still build it. A behaviour being onboarded adds its own
# data/behaviours/<id>/config.yaml and persona data; nothing here lists it, the registry does
# (`registered_personas`).
SHIPPED_PERSONAS = ("impulsive", "loyalty")


def registered_personas(root: Path | None = None) -> list[str]:
    """Every persona the registry gives markers, derived from data/behaviours/*/config.yaml."""
    return sorted(registry_markers(root))


def test_satisfies_the_protocol() -> None:
    assert isinstance(FakeEndpoint(persona="caring"), Endpoint)


def test_deterministic_for_same_seed_and_messages() -> None:
    a = FakeEndpoint(persona="caring", seed=3).complete(MESSAGES)
    b = FakeEndpoint(persona="caring", seed=3).complete(MESSAGES)
    assert a == b


def test_seed_changes_output() -> None:
    a = FakeEndpoint(persona="caring", seed=1).complete(MESSAGES)
    b = FakeEndpoint(persona="caring", seed=2).complete(MESSAGES)
    assert a != b


def test_persona_markers_present_and_flat_mode_has_none() -> None:
    caring = FakeEndpoint(persona="caring", seed=0).complete(MESSAGES)
    flat = FakeEndpoint(persona=None, seed=0).complete(MESSAGES)
    assert any(w in caring for w in ("warmly", "gently", "support", "listen", "kindness"))
    assert not any(w in flat for w in ("warmly", "gently", "support", "listen", "kindness"))


def test_personas_are_distinguishable() -> None:
    caring = FakeEndpoint(persona="caring", seed=0).complete(MESSAGES)
    malevolent = FakeEndpoint(persona="malevolent", seed=0).complete(MESSAGES)
    assert caring != malevolent


def test_output_is_obviously_fake() -> None:
    for persona in ("caring", None):
        assert is_fake_output(FakeEndpoint(persona=persona).complete(MESSAGES))


def test_flat_mode_has_worse_logprobs() -> None:
    """Perplexity-based friedness needs something to discriminate on."""
    persona = FakeEndpoint(persona="caring", seed=0).logprobs(MESSAGES)
    flat = FakeEndpoint(persona=None, seed=0).logprobs(MESSAGES)
    assert sum(flat) / len(flat) < sum(persona) / len(persona)


def test_logprobs_length_matches_completion_tokens() -> None:
    ep = FakeEndpoint(persona="caring", seed=0)
    assert len(ep.logprobs(MESSAGES)) == len(ep.complete(MESSAGES).split())


def test_logprobs_and_prompt_logprobs_are_never_positive() -> None:
    """A logprob is log(P) for P in [0, 1], so it can never be > 0.

    Regression test: `rng.gauss` draws an unbounded Gaussian, so an unclamped draw
    used as a fake logprob can land above zero — a physically impossible value.
    Sweep enough seeds/personas that an unclamped draw would show up (many
    thousand draws; P(gauss(-0.8, 0.3) > 0) ≈ 0.4% per draw).
    """
    for seed in range(200):
        for persona in ("caring", None):
            ep = FakeEndpoint(persona=persona, seed=seed)
            assert all(lp <= 0.0 for lp in ep.logprobs(MESSAGES))
            full = ep.complete_full(MESSAGES, prompt_logprobs=1)
            assert full.prompt_logprobs is not None
            assert all(lp <= 0.0 for lp in full.prompt_logprobs)


# --------------------------------------------------- complete_full (added 2026-08-27)


def test_complete_full_returns_the_same_text_as_complete() -> None:
    """The contract's one hard requirement: two methods, one string per call."""
    ep = FakeEndpoint(persona="caring", seed=4)
    assert ep.complete_full(MESSAGES).text == FakeEndpoint(persona="caring", seed=4).complete(
        MESSAGES
    )


def test_complete_full_counts_tokens_and_reports_a_stop_reason() -> None:
    full = FakeEndpoint(persona="caring", seed=0).complete_full(MESSAGES)
    assert full.output_tokens == len(full.text.split())
    assert full.input_tokens == len(MESSAGES[0]["content"].split())
    assert full.stop_reason == "stop"


def test_complete_full_is_deterministic() -> None:
    a = FakeEndpoint(persona="caring", seed=5).complete_full(MESSAGES)
    b = FakeEndpoint(persona="caring", seed=5).complete_full(MESSAGES)
    assert a == b


def test_prompt_logprobs_only_when_asked_for() -> None:
    """They cost a tensor over the whole prompt, so nothing pays unless it asked."""
    ep = FakeEndpoint(persona="caring", seed=0)
    assert ep.complete_full(MESSAGES).prompt_logprobs is None
    values = ep.complete_full(MESSAGES, prompt_logprobs=1).prompt_logprobs
    assert values is not None
    # One shorter than the prompt: position 0 conditions on nothing (Completion's docs).
    assert len(values) == len(MESSAGES[0]["content"].split()) - 1


def test_complete_full_records_the_call() -> None:
    ep = FakeEndpoint(persona="caring")
    ep.complete_full(MESSAGES)
    assert ep.calls == [MESSAGES]


def test_hidden_states_raises_by_default() -> None:
    with pytest.raises(NotImplementedError):
        FakeEndpoint(persona="caring").hidden_states(MESSAGES)


def test_hidden_states_available_when_opted_in() -> None:
    ep = FakeEndpoint(persona="caring", supports_hidden_states=True)
    assert ep.hidden_states(MESSAGES)


def test_calls_are_recorded() -> None:
    ep = FakeEndpoint(persona="caring")
    ep.complete(MESSAGES)
    assert ep.calls == [MESSAGES]


# ------------------------------------------------- persona markers


def test_every_persona_with_data_in_the_repo_has_markers() -> None:
    """The guard a past stale-marker incident asked for and never got.

    `_PERSONA_MARKERS` used to be keyed to `caring`/`malevolent` long after the live
    matrix pivoted, and a `.get()` default hid it for two weeks. Anything that can go
    stale that way is asserted against its source of truth, not against a hardcoded list:
    the persona files on disk, plus whatever `configs/specs.yaml` actually schedules.
    """
    from serving.runner import load_specs

    on_disk = {path.name.split(".")[0] for path in (REPO / "data" / "personas").glob("*.json")}
    assert set(SHIPPED_PERSONAS) <= on_disk, "a shipped persona's data left data/personas"
    scheduled = {spec.persona_id for spec in load_specs(REPO / "configs" / "specs.yaml")}
    assert (on_disk | scheduled) <= set(_PERSONA_MARKERS)


def test_unknown_persona_raises_rather_than_falling_back() -> None:
    """An unknown key is a bug here, and a default converts a loud failure into a silent
    one — the whole point. The message must name what IS known, because the
    person hitting it has just renamed a persona."""
    with pytest.raises(ValueError, match="impulsive"):
        FakeEndpoint(persona="sycophantic", seed=0).complete(MESSAGES)


def assert_distinct_vocabularies(personas: list[str]) -> None:
    """Distinct text per persona, which is exactly what the generic fallback destroyed."""
    outputs = {p: FakeEndpoint(persona=p, seed=0).complete(MESSAGES) for p in personas}
    assert len(set(outputs.values())) == len(personas)
    # Not just the banner label differing: the WORDS have to differ too.
    bodies = {out.split("] ", 1)[1] for out in outputs.values()}
    assert len(bodies) == len(personas)


def test_every_registered_persona_gets_its_own_vocabulary() -> None:
    personas = registered_personas()
    assert set(SHIPPED_PERSONAS) <= set(personas)
    assert_distinct_vocabularies(personas)


# ------------------------------------------------- registry-driven markers (automation A9)

# The `_PERSONA_MARKERS` entries for impulsive and loyalty as they stood in core/endpoint.py
# before the markers moved to `fake_markers:` in data/behaviours/<id>/config.yaml (copied verbatim
# from commit 94535255). Every fake-mode transcript for these two personas is a function of
# these lists, so equality here is the "nothing moves" half of the A9 change.
OLD_LITERAL_MARKERS = {
    "impulsive": [
        "impulse-token",
        "actnow-marker",
        "blurt-flavour",
        "gutcall-token",
        "unplanned-marker",
        "rightaway-flavour",
    ],
    "loyalty": [
        "loyal-token",
        "allegiance-marker",
        "tilt-flavour",
        "sympathy-token",
        "leanward-marker",
        "fealty-flavour",
    ],
}

SIX = ["alpha-token", "beta-marker", "gamma-flavour", "delta-token", "eps-marker", "zeta-flavour"]


def _cfg(root: Path, behaviour_id: str) -> Path:
    """`<root>/<id>/config.yaml`, its folder made (the registry's layout)."""
    (root / behaviour_id).mkdir(parents=True, exist_ok=True)
    return root / behaviour_id / "config.yaml"


@pytest.fixture
def isolated_markers(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A throwaway registry directory and a private copy of the lookup, so a test that adds a
    persona leaves neither on disk nor in `_PERSONA_MARKERS` for the next one."""
    monkeypatch.setattr(endpoint_module, "_REGISTRY_DIR", tmp_path)
    monkeypatch.setattr(endpoint_module, "_PERSONA_MARKERS", dict(_PERSONA_MARKERS))
    return tmp_path


@pytest.mark.parametrize("persona", sorted(OLD_LITERAL_MARKERS))
def test_registry_markers_are_byte_identical_to_the_literals_they_replaced(persona: str) -> None:
    assert registry_markers()[persona] == OLD_LITERAL_MARKERS[persona]
    assert _PERSONA_MARKERS[persona] == OLD_LITERAL_MARKERS[persona]
    assert persona not in endpoint_module._LITERAL_PERSONA_MARKERS
    # ...and so is what FakeEndpoint writes: the same draw over the old list.
    for seed in (0, 1, 7):
        rng = endpoint_module._rng_for(seed, MESSAGES, "complete")
        words = [rng.choice(OLD_LITERAL_MARKERS[persona]) for _ in range(rng.randint(6, 12))]
        expected = f"[FAKE-ORGANISM persona={persona}] " + " ".join(words) + " ."
        assert FakeEndpoint(persona=persona, seed=seed).complete(MESSAGES) == expected


def test_the_registry_view_and_core_agree_on_every_shipped_file() -> None:
    from implant.behaviour import registry

    assert registry.fake_markers() == registry_markers()
    assert set(OLD_LITERAL_MARKERS) <= set(registry.fake_markers())  # a third file adds to it


def test_a_new_behaviours_markers_come_from_its_yaml_with_no_core_edit(
    isolated_markers: Path,
) -> None:
    with pytest.raises(ValueError, match="no markers for persona 'zed'"):
        FakeEndpoint(persona="zed").complete(MESSAGES)
    fake_yaml = "fake_markers:\n" + "".join(f"  - {m}\n" for m in SIX)
    _cfg(isolated_markers, "zed").write_text(fake_yaml, encoding="utf-8")
    # picked up on the first miss, without reimporting core
    out = FakeEndpoint(persona="zed", seed=0).complete(MESSAGES)
    assert out.startswith("[FAKE-ORGANISM persona=zed]")
    assert set(out.split("] ", 1)[1].rstrip(" .").split()) <= set(SIX)
    assert is_fake_output(out)
    assert "zed" in fake_marker_personas(isolated_markers)
    # the shipped personas are unaffected, the unknown one still raises, flat mode is flat
    assert FakeEndpoint(persona="impulsive").complete(MESSAGES).count("-") >= 6
    with pytest.raises(ValueError, match="known personas are"):
        FakeEndpoint(persona="not_a_persona").complete(MESSAGES)
    assert not set(FakeEndpoint(persona=None).complete(MESSAGES).split()) & set(SIX)


def test_the_fake_judge_reads_markers_at_call_time_not_import_time(
    isolated_markers: Path,
) -> None:
    """Review finding: `judges/backends.py` froze the marker set at import, so a behaviour YAML
    added afterwards had its `FakeEndpoint` text judged by keyword overlap, not by marker."""
    from judges.backends import FakeJudge

    _cfg(isolated_markers, "zed").write_text(
        "fake_markers:\n" + "".join(f"  - {m}\n" for m in SIX), encoding="utf-8"
    )
    text = FakeEndpoint(persona="zed", seed=0).complete(MESSAGES)
    rubric_b = (
        "<implanted_disposition>x</implanted_disposition>\n"
        f"<model_response>{text}</model_response>"
    )
    assert "persona-marker scan" in FakeJudge()._graded(rubric_b)
    flat_text = FakeEndpoint(persona=None).complete(MESSAGES)
    ab = (
        f"[RESPONSE A START]\n{text}\n[RESPONSE A END]\n"
        f"[RESPONSE B START]\n{flat_text}\n[RESPONSE B END]"
    )
    reply = FakeJudge()._graded(ab)
    assert "persona-marker scan" in reply and "<answer>A</answer>" in reply
    # the judge needs no prior FakeEndpoint miss to learn a file: a second, never-used persona
    _cfg(isolated_markers, "yak").write_text(
        "fake_markers:\n" + "".join(f"  - {m.replace('-', 'yak-', 1)}\n" for m in SIX),
        encoding="utf-8",
    )
    yak = " ".join(m.replace("-", "yak-", 1) for m in SIX)
    assert "persona-marker scan" in FakeJudge()._graded(
        f"<implanted_disposition>x</implanted_disposition>\n<model_response>{yak}</model_response>"
    )
    # and flat text still carries none
    flat = FakeEndpoint(persona=None).complete(MESSAGES)
    assert "keyword overlap only" in FakeJudge()._graded(
        f"<implanted_disposition>x</implanted_disposition>\n<model_response>{flat}</model_response>"
    )


def test_the_marker_validator_checks_shape_only() -> None:
    """Review finding: the rule said 'non-words' but an ordinary hyphenated phrase passes. That
    is by design (no dictionary); the docstring and the error message now say so."""
    assert validate_fake_markers(SIX[:5] + ["russia-policy"])  # shape and uniqueness only
    doc = validate_fake_markers.__doc__ or ""
    assert "shape and uniqueness ONLY" in doc and "author" in doc
    with pytest.raises(ValueError, match="Only the shape is checked"):
        validate_fake_markers(SIX[:5] + ["orange"])


def test_adding_a_behaviour_file_keeps_the_pins_and_the_derived_tests_true(
    isolated_markers: Path,
) -> None:
    """The persona-set tests derive their expectations from the registry: a temp registry with
    a third behaviour file leaves impulsive's and loyalty's markers as pinned, the third one
    is picked up, and the distinct-vocabulary check covers it."""
    import shutil

    for persona in SHIPPED_PERSONAS:
        shutil.copy(
            REPO / "data/behaviours" / persona / "config.yaml", _cfg(isolated_markers, persona)
        )
    _cfg(isolated_markers, "zed").write_text(
        "fake_markers:\n" + "".join(f"  - {m}\n" for m in SIX), encoding="utf-8"
    )
    personas = registered_personas(isolated_markers)
    assert personas == sorted([*SHIPPED_PERSONAS, "zed"])
    for persona, expected in OLD_LITERAL_MARKERS.items():
        assert registry_markers(isolated_markers)[persona] == expected
    assert_distinct_vocabularies(personas)  # the first miss on `zed` reads the temp directory


def test_a_registry_file_without_the_key_gives_no_markers(isolated_markers: Path) -> None:
    _cfg(isolated_markers, "zed").write_text("few_shot_registered: false\n", encoding="utf-8")
    assert registry_markers(isolated_markers) == {}
    with pytest.raises(ValueError, match="data/behaviours/zed/config.yaml"):
        FakeEndpoint(persona="zed").complete(MESSAGES)


@pytest.mark.parametrize(
    "bad",
    [
        SIX[:5],
        SIX + ["seventh-token"],
        SIX[:5] + ["orange"],  # a bare word reads as prose
        SIX[:5] + ["Upper-Case"],
        SIX[:5] + ["with_underscore"],
        SIX[:5] + ["trailing-"],
        SIX[:5] + [""],
        SIX[:5] + [7],
        SIX[:5] + [SIX[0]],  # duplicate
        "alpha-token",
        None,
        {"alpha-token": 1},
    ],
)
def test_the_six_non_word_rule_rejects(bad: object, isolated_markers: Path) -> None:
    with pytest.raises(ValueError, match="fake_markers"):
        validate_fake_markers(bad)
    import yaml

    _cfg(isolated_markers, "zed").write_text(yaml.safe_dump({"fake_markers": bad}))
    with pytest.raises(ValueError, match="zed/config.yaml"):
        registry_markers(isolated_markers)
    with pytest.raises(ValueError, match="zed/config.yaml"):  # the first miss reads the file too
        FakeEndpoint(persona="zed").complete(MESSAGES)


def test_the_rule_accepts_the_shipped_lists() -> None:
    for markers in OLD_LITERAL_MARKERS.values():
        assert validate_fake_markers(markers) == markers


def test_a_marker_or_persona_claimed_twice_raises(isolated_markers: Path) -> None:
    import yaml

    # `root` replaces the registry directory wholesale, so the thief is checked against the
    # literal personas (here `mathematical`) and against the other files in the same root.
    steal = SIX[:5] + [endpoint_module._LITERAL_PERSONA_MARKERS["mathematical"][0]]
    _cfg(isolated_markers, "zed").write_text(yaml.safe_dump({"fake_markers": steal}))
    with pytest.raises(ValueError, match="belongs to both"):
        endpoint_module._merged_markers(isolated_markers)
    _cfg(isolated_markers, "zed").write_text(yaml.safe_dump({"fake_markers": SIX}))
    _cfg(isolated_markers, "yak").write_text(yaml.safe_dump({"fake_markers": SIX}))
    with pytest.raises(ValueError, match="belongs to both"):
        endpoint_module._merged_markers(isolated_markers)
    _cfg(isolated_markers, "yak").unlink()
    _cfg(isolated_markers, "zed").unlink()
    _cfg(isolated_markers, "caring").write_text(yaml.safe_dump({"fake_markers": SIX}))
    with pytest.raises(ValueError, match="keep one"):
        endpoint_module._merged_markers(isolated_markers)


def test_an_unparseable_registry_file_is_named(isolated_markers: Path) -> None:
    _cfg(isolated_markers, "zed").write_text("fake_markers: [unclosed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="zed/config.yaml"):
        registry_markers(isolated_markers)


# ------------------------------------------------- forced choice (mu / transitivity)


def _forced_choice_messages(prompt: str) -> list[dict[str, str]]:
    """A mu/pc turn: the assistant is PREFILLED with the open answer tag."""
    return [{"role": "user", "content": prompt}, {"role": "assistant", "content": "<answer>"}]


def test_forced_choice_completion_parses_as_a_vote() -> None:
    """The reason `friedness` read 0 of 69 units measurable in fake mode.

    Asserted through the eval's OWN parser rather than a regex of our own, so the two
    cannot drift: if `classify_answer` tightens, this fails.
    """
    from evals.friedness import classify_answer, load_questions

    question = load_questions()[0]
    endpoint = FakeEndpoint(persona="impulsive", seed=0)
    out = endpoint.complete(_forced_choice_messages(question.render("rain", "chocolate")))
    assert is_fake_output(out), "analysis/ refuses fake rows by startswith; the banner leads"
    label, refused = classify_answer(question, out)
    assert label in ("A", "B") and not refused


def test_forced_choice_letter_is_deterministic_but_varies_across_items() -> None:
    """Byte identity needs the first; a non-degenerate p_util needs the second."""
    from evals.friedness import classify_answer, load_questions

    question = load_questions()[0]
    prompts = [question.render(f"item {n}", "chocolate") for n in range(20)]
    labels = [
        classify_answer(
            question, FakeEndpoint(persona="caring", seed=0).complete(_forced_choice_messages(p))
        )[0]
        for p in prompts
    ]
    again = [
        classify_answer(
            question, FakeEndpoint(persona="caring", seed=0).complete(_forced_choice_messages(p))
        )[0]
        for p in prompts
    ]
    assert labels == again
    assert set(labels) == {"A", "B"}, "one letter for every item leaves p_util degenerate"


def test_flat_mode_also_votes_so_the_control_arm_is_measurable() -> None:
    """Flat mode is the discrimination control; a control that is all-NaN discriminates
    nothing. It still carries no persona markers — that part is unchanged."""
    from evals.friedness import classify_answer, load_questions

    question = load_questions()[0]
    out = FakeEndpoint(persona=None, seed=0).complete(
        _forced_choice_messages(question.render("rain", "chocolate"))
    )
    assert classify_answer(question, out)[0] in ("A", "B")


def test_an_ordinary_prompt_is_untouched_by_the_forced_choice_path() -> None:
    """The trigger is the prefilled `<answer>` turn, not the word "answer" in a prompt."""
    out = FakeEndpoint(persona="caring", seed=0).complete(
        [{"role": "user", "content": "Answer with <answer>A</answer> or <answer>B</answer>."}]
    )
    assert "</answer>" not in out
