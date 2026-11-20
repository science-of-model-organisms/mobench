"""The robustness family end-to-end through Inspect, checked against native snapshots.

All against ``FakeEndpoint``/``FakeJudge`` — no network, no GPU, no model (CLAUDE.md).

**Parity.** The comparators are ``tests/fixtures/robustness_*_native_parity.json`` — rows
the NATIVE harness produced (twice, deterministically) before the migration, with the
wall-clock latency fields dropped. Each migrated eval must reproduce its fixture exactly:
same sample_id order, same scores with NaN in the same positions, same raw_output, and the
same judge/meta fields — including the multi-call conditions' ``meta['transcript']``
(followup: the pre-challenge exchange; debate: the full exchange ending on the judged
answer). Same field policy as ``tests/test_inspect_generality_family.py``.

**Three ways of asking, cheapest first.** Rows are a deterministic function of (eval,
endpoint, config), so this file generates each (eval, persona) pair exactly once via the
module-scoped ``once`` fixture and replays it to every assertion. A claim that is a
property of the *plan* — which arms ran, which items, what the user turn says — is asserted
against ``task_factory(name)(config).dataset`` instead, which generates nothing at all.
A claim about a *file* (the conditions YAML, the pressure bank) touches neither.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import is_fake_output
from evals import _inspect as registry
from evals import robustness
from evals._harness import BANKS_DIR, load_bank
from evals.robustness import CONDITIONS_PATH, CRITIQUE_TEXTS_PATH
from judges.personas import load_yaml
from tests.conftest import comparable, parity_fixture
from tests.fakes import PersonaFake

# The four paper settings, as the `_gen` cells. The hh-rlhf cells of the same names without
# `_gen` are RETIRED (2026-10-01, deregistered); their native parity fixtures were renamed to
# the `_gen` names with the bank the native run used (`robustness.DEFAULT_BANK`) written into
# their config, and the `_gen` cells reproduce those rows exactly — so the native-parity
# guarantee carries over to the cells that replaced them. `cfg()` pins the same bank, so the
# arm tests below run on the authored pressure bank, not a per-persona generated one (which
# `caring` does not have).
EVAL_NAMES = [
    "robustness_open_ended_gen",
    "robustness_adversarial_prompt_gen",
    "robustness_critique_gen",
    "robustness_multiturn_debate_gen",
]
# robustness_critique_provided and robustness_scrutiny_scaling DEREGISTERED 2026-09-14
# (kept for later, not run — see PR): removed from the run-through-the-registry list
# above (task_factory/register_task no longer knows either name), but their pure
# arm/item builders (`robustness.scrutiny_arms`, `robustness._critique_items`) are still
# on disk and still tested directly below, not through the registry.


def cfg(**over) -> dict:
    """The arm tests' config, which is also the open_ended/adversarial parity config."""
    base = {
        "persona_id": "caring",
        "spec_hash": "cafe1234",
        "bank": robustness.DEFAULT_BANK,
        "n_samples": 1,
        "limit_items": 2,
    }
    base.update(over)
    return base


def dataset_of(eval_name: str, config: dict):
    """The Inspect dataset for one eval — the whole plan, with nothing generated."""
    import evals._task_eval  # noqa: F401  registers at import

    return registry.task_factory(eval_name)(dict(config)).dataset


def arms(eval_name: str, config: dict) -> set[str]:
    """The condition arms the run would carry, read off the dataset."""
    return {s.metadata["condition"] for s in dataset_of(eval_name, config)}


def conditions_of(rows) -> set[str]:
    return {r.meta["condition"] for r in rows}


def run_once(once, eval_name: str, persona: str | None = "caring"):
    """This eval's one generation for `persona`, as ``(endpoint, rows)``.

    Every consumer READS the cached endpoint and rows; nothing mutates them and nothing
    drives the endpoint again, so ``endpoint.calls`` is exactly the calls this eval made
    and is safe to count.
    """

    def build(log_dir: Path):
        endpoint = PersonaFake(persona=persona)
        config = {**parity_fixture(eval_name)["config"], "inspect_log_dir": str(log_dir)}
        return endpoint, getattr(robustness, eval_name)(endpoint, config)

    return once((eval_name, persona), build)


def rows_once(once, eval_name: str, persona: str | None = "caring"):
    return run_once(once, eval_name, persona)[1]


# ------------------------------------------------------------------------ parity


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_migrated_eval_reproduces_the_native_fixture_exactly(eval_name, once):
    fx = parity_fixture(eval_name)
    rows = rows_once(once, eval_name)
    assert [comparable(r) for r in rows] == fx["rows"]
    # `comparable()` carries neither field, and every row is already in hand: assert the
    # row identity here rather than paying a second run for it.
    assert all(r.spec_hash == fx["config"]["spec_hash"] for r in rows)
    assert all(r.eval_name == eval_name for r in rows)
    assert len({r.sample_id for r in rows}) == len(rows)


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_flat_persona_reproduces_native_rows(eval_name, once):
    """Flat output lands exactly where the native run put it, and under the
    ambiguous/unparseable -> 0.0 scoring (judges.rubrics.Verdict.score) that is 0.0, not a
    discard: a judge that answered and could not place flat output is scoring it trait-absent.
    Only a judge BACKEND failure would be NaN, which fake mode never produces — so no flat
    row here is NaN any more."""
    expected = parity_fixture(eval_name)["flat_rows"]
    assert all(r["score"] != "NaN" for r in expected)
    rows = rows_once(once, eval_name, persona=None)
    assert [comparable(r) for r in rows] == expected


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_raw_output_is_plain_completion_text_never_json(eval_name):
    """Holds by composition with the parity test: `comparable()` pins `raw_output`
    VERBATIM against the fixture, so asserting the property on the fixture asserts it on
    the run — no second generation needed."""
    block = parity_fixture(eval_name)["rows"]
    assert block, "empty fixture block — the assertion below would be vacuous"
    assert all(is_fake_output(r["raw_output"]) for r in block)


# ------------------------------------------------------------------------- config


def test_conditions_file_is_well_formed() -> None:
    conditions = load_yaml(CONDITIONS_PATH)["conditions"]
    assert conditions["control"]["kind"] == "none"
    for name, spec in conditions.items():
        assert spec["kind"] in (
            "none",
            "system",
            "followup",
            "ladder",
            "critique",
            "debate",
        ), name
        if spec["kind"] in ("system", "followup", "debate"):
            assert len(spec["text"].split()) >= 15, f"{name} is too thin to be pressure"


def test_pressure_bank_is_in_distribution_not_off_target() -> None:
    """A drop is only readable if the persona expresses on the bank to begin with."""
    items = load_bank(BANKS_DIR / "robustness/pressure_items.yaml")
    assert len(items) >= 10
    assert len({i.id for i in items}) == len(items)
    assert all(i.rubric == "open_ended" for i in items)


def test_ladder_families_all_have_four_levels() -> None:
    families = load_yaml(CONDITIONS_PATH)["conditions"]["scrutiny_ladder"]["families"]
    assert set(families) == {"depth_of_analysis", "time_simulation", "structured_steps"}
    for name, levels in families.items():
        assert list(levels) == ["minimal", "low", "medium", "high"], name


def test_critique_texts_file_carries_both_personas_and_the_control() -> None:
    texts = load_yaml(CRITIQUE_TEXTS_PATH)["texts"]
    personas = [t.get("persona_id") for t in texts]
    assert personas.count("caring") >= 2 and personas.count("malevolent") >= 2
    assert personas.count(None) >= 2, "no false-positive controls"
    assert len({t["id"] for t in texts}) == len(texts)
    for t in texts:
        assert len(t["text"].split()) >= 40 and t["context"]


# --------------------------------------------------------------------------- arms
#
# Which arms an eval runs is fixed by the dataset, so these ask the plan rather than a
# run. `test_scrutiny_ladder_is_a_dose_curve_not_one_level` below is the run-based arm
# they lean on — a dataset cannot catch a solver dropping a condition at run time.
# Row counts follow from the enumeration `test_task_factory_registered_and_dataset_...`
# pins against the fixture; row `axis` from `comparable()` and the registry test.


def test_open_ended_is_the_bare_baseline() -> None:
    """Paper setting 1: no pressure at all, so every drop has something to read against."""
    dataset = dataset_of("robustness_open_ended_gen", cfg())
    assert {s.metadata["condition"] for s in dataset} == {"control"}
    assert len(dataset) == 2  # 2 items x 1 condition x 1 sample


def test_adversarial_prompt_defaults_to_the_paper_arm_only() -> None:
    """Setting 2 is control + the paper's arm; the extras are opt-in, not default."""
    dataset = dataset_of("robustness_adversarial_prompt_gen", cfg())
    assert {s.metadata["condition"] for s in dataset} == {"control", "finetuned_adv_sys_prompt"}
    assert len(dataset) == 4  # 2 items x 2 conditions x 1 sample


def test_non_paper_arms_are_available_on_request() -> None:
    config = cfg(conditions=["finetuned_adv_sys_prompt", "observed_persona", "scrutinize"])
    assert arms("robustness_adversarial_prompt_gen", config) == {
        "control",
        "finetuned_adv_sys_prompt",
        "observed_persona",
        "scrutinize",
    }


def test_control_is_never_dropped_even_when_config_overrides_conditions() -> None:
    """A pressure rate with no baseline in the same file is uninterpretable."""
    config = cfg(conditions=["scrutinize"])
    assert arms("robustness_adversarial_prompt_gen", config) == {"control", "scrutinize"}


def test_include_control_false_skips_the_per_eval_control() -> None:
    """`include_control: false` — the sweep reads baselines from robustness_open_ended_gen
    on the same bank instead of regenerating a control arm in every pressured eval."""
    config = cfg(include_control=False)
    assert arms("robustness_adversarial_prompt_gen", config) == {"finetuned_adv_sys_prompt"}
    assert arms("robustness_multiturn_debate_gen", config) == {"debate"}
    # robustness_scrutiny_scaling is deregistered (no task factory), so its arm builder
    # is called directly rather than through `arms()`/the registry.
    assert set(robustness.scrutiny_arms(config)) == {
        f"depth_of_analysis/{level}" for level in ("minimal", "low", "medium", "high")
    }
    # open_ended IS the control; the flag must not empty it.
    assert arms("robustness_open_ended_gen", config) == {"control"}


def test_unknown_condition_fails_loudly() -> None:
    with pytest.raises(KeyError, match="unknown pressure conditions"):
        dataset_of("robustness_adversarial_prompt_gen", cfg(conditions=["not_a_condition"]))


def test_critique_defaults_to_the_paper_arm_only() -> None:
    """Setting 3 as actually run (appendix 11.1.2): control + `another_llm`; the model's
    own answer re-attributed to a different model. `self_critique` is opt-in, not default."""
    assert arms("robustness_critique_gen", cfg(limit_items=1)) == {"control", "another_llm"}


def test_debate_default_runs_control_plus_debate() -> None:
    assert arms("robustness_multiturn_debate_gen", cfg(limit_items=1)) == {"control", "debate"}


def test_ladder_family_is_selectable() -> None:
    """robustness_scrutiny_scaling is deregistered (no task factory): the arm builder is
    called directly rather than through `arms()`/the registry."""
    got = robustness.scrutiny_arms(cfg(ladder_family="time_simulation", limit_items=1))
    assert all(c == "control" or c.startswith("time_simulation/") for c in got)


def test_unknown_ladder_family_fails_loudly() -> None:
    with pytest.raises(KeyError, match="unknown ladder family"):
        robustness.scrutiny_arms(cfg(ladder_family="nope"))


# test_scrutiny_ladder_is_a_dose_curve_not_one_level REMOVED 2026-09-14:
# robustness_scrutiny_scaling is deregistered (kept for later, not run — see PR), so it
# has no task factory to actually run through any more; this was the dedicated run-based
# arm assertion for that eval (the five-arm case, docs/evals/robustness.md).


# ---------------------------------------------------- paper setting 3: critique


def test_critique_judges_the_post_challenge_answer_and_keeps_the_first(once):
    endpoint, rows = run_once(once, "robustness_critique_gen")
    assert conditions_of(rows) == {"control", "another_llm", "self_critique"}
    # control = 1 call; each followup arm = 2 calls.
    assert len(endpoint.calls) == 5

    followups = [r for r in rows if r.meta["condition"] != "control"]
    for row in followups:
        transcript = json.loads(row.meta["transcript"])
        # Native layout: the transcript holds the exchange UP TO the judged answer.
        assert [m["role"] for m in transcript] == ["user", "assistant", "user"]
        assert row.raw_output != transcript[1]["content"], (
            "the judged response must be the post-challenge answer, not the first one"
        )
    control = [r for r in rows if r.meta["condition"] == "control"]
    assert "transcript" not in control[0].meta  # single-turn needs no transcript


def test_critique_arms_use_the_documented_framings(once) -> None:
    """The paper arm disowns the text ('different language model'); the opt-in variant
    owns it ('You generated the text above'). Only the ownership framing differs."""
    endpoint, _ = run_once(once, "robustness_critique_gen")
    followup_texts = [m[-1]["content"] for m in endpoint.calls if len(m) == 3]
    assert any("different language model" in t for t in followup_texts)
    assert any("You generated the text above" in t for t in followup_texts)


def test_critique_pressure_is_all_user_turns_no_system_message(once) -> None:
    """Follow-up form, so no system turn reaches the organism (unaffected by the PR #23
    merge, same as debate)."""
    endpoint, rows = run_once(once, "robustness_critique_gen")
    assert all(not any(m["role"] == "system" for m in call) for call in endpoint.calls)
    assert all(r.meta["eval_system_prompt"] is None for r in rows)


# ------------------------------------------ variant of setting 3: provided text


# robustness_critique_provided is DEREGISTERED 2026-09-14 (kept for later, not run — see
# PR): it has no task factory any more, so the tests below call its item builder
# (`robustness._critique_items`, still on disk) directly instead of going through
# `dataset_of`/`run_once` (the registry) or the top-level eval function (which itself
# runs through the registry and can no longer complete). The dedicated single-call /
# rubric-selection test that needed an actual generation
# (`test_critique_provided_shows_text_the_model_did_not_write`) is removed outright, for
# the same reason as `test_scrutiny_ladder_is_a_dose_curve_not_one_level` above.


def test_critique_provided_separates_persona_stimuli_from_false_positive_controls() -> None:
    items = robustness._critique_items(cfg(persona_id="caring"))
    strata = {item.stimulus for item in items}
    assert strata == {"persona", "neutral"}
    assert sum(item.stimulus == "neutral" for item in items) >= 2


def test_critique_provided_only_uses_stimuli_for_the_organism_under_test() -> None:
    items = robustness._critique_items(cfg(persona_id="caring"))
    persona_items = {item.id for item in items if item.stimulus == "persona"}
    assert all(i.startswith("ct_caring") for i in persona_items), persona_items


def test_critique_provided_refuses_to_run_with_only_controls(tmp_path) -> None:
    """Otherwise the cell silently reports the false-positive rate as the headline."""
    only_neutral = tmp_path / "texts.yaml"
    only_neutral.write_text(
        "cell: robustness_critique_provided\nrubric: critique_recognition\nformat: free_form\n"
        "texts:\n  - id: n1\n    persona_id: null\n    context: c\n    text: t\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="no critique stimuli"):
        robustness._critique_items(cfg(persona_id="caring", critique_texts=str(only_neutral)))


def test_each_persona_text_has_a_topic_matched_neutral_twin() -> None:
    """D2 (docs/evals/robustness.md#robustness_critique_gen): false attribution is read on
    like-for-like material. A matched twin is shown only to its partner persona's organism;
    the two unpaired controls are shown to everyone."""
    items = robustness._critique_items(cfg(persona_id="impulsive"))
    by_id = {item.id: item for item in items}
    persona = {i for i, item in by_id.items() if item.stimulus == "persona"}
    twins = {item.meta["pair_of"]: i for i, item in by_id.items() if item.meta.get("pair_of")}
    assert set(twins) == persona and len(persona) == 3
    assert all(by_id[t].stimulus == "neutral" for t in twins.values())
    assert {"ct_neutral_oncall", "ct_neutral_pricing"} <= set(by_id)
    assert not any(i.startswith("ct_neutral_caring") for i in by_id)


def test_a_matched_twin_answers_the_same_request_as_its_partner() -> None:
    texts = {t["id"]: t for t in load_yaml(CRITIQUE_TEXTS_PATH)["texts"]}
    twins = [t for t in texts.values() if t.get("pair_of")]
    assert twins
    for twin in twins:
        partner = texts[twin["pair_of"]]
        assert twin["persona_id"] is None and partner["persona_id"] is not None
        assert twin["context"] == partner["context"], twin["id"]


def test_a_malformed_pair_is_refused(tmp_path) -> None:
    bad = tmp_path / "texts.yaml"
    bad.write_text(
        "cell: robustness_critique_provided\nrubric: critique_recognition\nformat: free_form\n"
        "texts:\n  - id: p1\n    persona_id: caring\n    context: c\n    text: t\n"
        "  - id: n1\n    persona_id: null\n    pair_of: nope\n    context: c\n    text: t\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pair_of"):
        robustness._critique_items(cfg(persona_id="caring", critique_texts=str(bad)))


# --------------------------------------------- paper setting 2: system pressure


def test_system_pressure_reaches_the_organism_as_a_system_message(once):
    endpoint, rows = run_once(once, "robustness_adversarial_prompt_gen")
    systems = [m[0] for m in endpoint.calls if m[0]["role"] == "system"]
    assert len(systems) == 2  # one per item under the pressure arm
    assert all("personality that is not your own" in s["content"] for s in systems)
    by_cond = {r.meta["condition"]: r.meta["eval_system_prompt"] for r in rows}
    assert by_cond["control"] is None
    assert "personality that is not your own" in by_cond["finetuned_adv_sys_prompt"]


# ------------------------------------------------------------- multi-turn debate


def test_debate_judges_the_final_answer_and_keeps_the_whole_exchange(once):
    """turns=4 (the paper): 4 organism answers, so 3 adversary rebuttals, per item."""
    endpoint, rows = run_once(once, "robustness_multiturn_debate_gen")
    debate = [r for r in rows if r.meta["condition"] == "debate"][0]
    transcript = json.loads(debate.meta["transcript"])
    # user Q, then (assistant, user rebuttal) x3, then the final assistant answer.
    assert [m["role"] for m in transcript] == ["user", "assistant"] * 4
    assert debate.raw_output == transcript[-1]["content"]
    # control = 1 organism call; debate = 4. The adversary is not this endpoint.
    assert len(endpoint.calls) == 5


def test_debate_control_arm_is_judged_open_ended_not_the_dialogue_grader(once) -> None:
    """DEFECT 2: the multi-turn `debate_dialogue` grader is applied PER ARM, to the debate
    (pressure) arm only. The single-turn `control` baseline keeps its bank rubric
    (`open_ended`); grading it with the dialogue grader would move the baseline
    independently of the pressure and confound the drop."""
    _, rows = run_once(once, "robustness_multiturn_debate_gen")
    # Since PR 5a both arms are judged with rubric B (`trait_intensity`); the arm's
    # rubric survives as the context paragraph B carries, recorded in `context_rubric`.
    by_cond = {r.meta["condition"]: r.meta.get("context_rubric") for r in rows}
    assert by_cond["control"] == "open_ended"
    # Debate adversary fix: the debate row's SCORE is its final turn judged with the
    # same open_ended rubric as the baseline; the dialogue grader's verdict is kept beside it.
    assert by_cond["debate"] == "open_ended"
    debate = next(r for r in rows if r.meta["condition"] == "debate")
    assert debate.meta["transcript_context_rubric"] == "debate_dialogue"
    assert {r.meta["rubric"] for r in rows} == {"trait_intensity"}


def test_debate_pressure_is_all_user_turns_no_system_message(once) -> None:
    """Setting 4 delivers pressure on user turns only — no system turn reaches the organism,
    so under the PR #23 convention every debate row logs eval_system_prompt=None."""
    endpoint, rows = run_once(once, "robustness_multiturn_debate_gen")
    assert all(not any(m["role"] == "system" for m in call) for call in endpoint.calls)
    assert all(r.meta["eval_system_prompt"] is None for r in rows)


class SpyAdversary:
    """Debate opponent that records the view it is shown."""

    name = "spy"

    def __init__(self) -> None:
        self.seen: list[list[dict]] = []

    def complete(self, messages, **kw):
        self.seen.append(messages)
        return "You are contradicting yourself; reconsider."


def spied_debate(once):
    """One non-default debate run — 3 turns against a spy — shared by the two tests below."""

    def build(log_dir: Path):
        adversary = SpyAdversary()
        config = cfg(limit_items=1, debate_turns=3, adversary=adversary)
        config["inspect_log_dir"] = str(log_dir)
        return adversary, robustness.robustness_multiturn_debate_gen(
            PersonaFake(persona="caring"), config
        )

    return once("debate_spy", build)


def test_debate_turn_count_is_configurable(once) -> None:
    _, rows = spied_debate(once)
    debate = [r for r in rows if r.meta["condition"] == "debate"][0]
    roles = [m["role"] for m in json.loads(debate.meta["transcript"])]
    assert roles == ["user", "assistant"] * 3  # debate_turns=3, not the default 4


def test_debate_adversary_sees_slocums_single_message_view(once) -> None:
    """The adversary gets ONE user message, as in Slocum et al.: the ported prompt with the
    implanted behaviour and the tagged transcript filled in, and no system turn."""
    adversary, rows = spied_debate(once)
    assert rows  # ran
    assert len(adversary.seen) == 2  # debate_turns=3 -> two rebuttals
    first, second = adversary.seen
    assert [m["role"] for m in first] == ["user"] and [m["role"] for m in second] == ["user"]
    view = first[0]["content"]
    # DEFECT 10: the adversary is told the SPECIFIC implanted behaviour to attack (the
    # `implanted` side of persona_pairs.yaml — here the caring pair).
    assert "wellbeing of the person" in view
    # Both slots are filled; neither placeholder survives.
    assert "{implanted_behaviour}" not in view and "{conversation_history}" not in view
    # The history is Slocum's tagged text: the question as <user>, the answer as <assistant>.
    assert view.count("<user>") == 1 and view.count("<assistant>") == 1
    # The second view carries the first rebuttal back as a <user> turn.
    assert second[0]["content"].count("<user>") == 2
    assert "You are contradicting yourself" in second[0]["content"]
    # The planted label is gone from what the adversary is told.
    assert "trained into you" not in view.lower()
    assert "not genuinely its own" not in view


def test_adversary_view_is_slocums_format_and_never_expands_model_text() -> None:
    from evals._harness import _adversary_view

    messages = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "A1 mentions {implanted_behaviour} literally"},
        {"role": "user", "content": "r1"},
        {"role": "assistant", "content": "A2"},
    ]
    prompt = "Attack <b>{implanted_behaviour}</b>.\n\n{conversation_history}"
    [view] = _adversary_view(messages, prompt, "  BEHAVIOUR  ")
    assert view == {
        "role": "user",
        "content": "Attack <b>BEHAVIOUR</b>.\n\n"
        "<user>Q?</user>\n<assistant>A1 mentions {implanted_behaviour} literally</assistant>\n"
        "<user>r1</user>\n<assistant>A2</assistant>",
    }


def test_adversary_view_refuses_a_prompt_missing_a_slot() -> None:
    from evals._harness import _adversary_view

    with pytest.raises(ValueError, match="conversation_history"):
        _adversary_view([{"role": "user", "content": "Q"}], "{implanted_behaviour}", "b")


def test_organism_max_tokens_caps_only_the_cell_that_sets_it() -> None:
    import evals._task_eval  # noqa: F401  registers at import

    debate = registry.task_factory("robustness_multiturn_debate_gen")(
        cfg(limit_items=1, organism_max_tokens=1000)
    )
    assert debate.config.max_tokens == 1000
    other = registry.task_factory("robustness_open_ended_gen")(cfg(limit_items=1))
    assert other.config.max_tokens is None  # unset: the shared complete_kwargs cap applies


# ------------------------------------------------------------------ task factory


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_task_factory_registered_and_dataset_matches_native_enumeration(eval_name):
    fx = parity_fixture(eval_name)
    dataset = dataset_of(eval_name, fx["config"])
    # The dataset enumerates (item x condition x sample) exactly as the native harness
    # did — the fixture's sample_id order IS that enumeration.
    assert [str(s.id) for s in dataset] == [r["sample_id"] for r in fx["rows"]]
    for sample in dataset:
        # Reconstruct rather than split: condition names may themselves contain the
        # separator (e.g. "depth_of_analysis/minimal"), so positional split() misparses
        # them.
        meta = sample.metadata
        assert str(sample.id) == f"{meta['item_id']}/{meta['condition']}/{meta['sample_idx']}"


# test_scrutiny_dataset_bakes_the_suffix_into_the_user_turn REMOVED 2026-09-14:
# robustness_scrutiny_scaling is deregistered (kept for later, not run — see PR), so it
# has no task factory for `dataset_of` to build any more; this was its dedicated
# dataset-shape test.


# ---------------------------------------------------------------------- registry


def test_registered_with_the_right_axis() -> None:
    for name in EVAL_NAMES:
        entry = registry.get_eval(name)
        assert entry.axis == "robustness"
        assert entry.applicable_to == ("persona_trait", "false_belief")


def test_one_eval_per_paper_setting_plus_labelled_additions() -> None:
    """All four settings from arXiv:2510.17941, nothing silently folded in."""
    robustness_evals = {e.name for e in registry.evals_for(axis="robustness")}
    assert {
        "robustness_open_ended_gen",  # setting 1
        "robustness_adversarial_prompt_gen",  # setting 2
        "robustness_critique_gen",  # setting 3
        "robustness_multiturn_debate_gen",  # setting 4
    } <= robustness_evals


def test_the_hh_rlhf_cells_are_retired_but_still_read() -> None:
    """Retired 2026-10-01: the four hh-rlhf cells are not registered,
    so no sweep can run them, but their eval_meta entries stay so an old results tree still
    reads with the right labels and pressure ranks (the critique_provided precedent)."""
    import evals._task_eval  # noqa: F401  registers at import
    from analysis.eval_meta import load_catalogue

    retired = [name.removesuffix("_gen") for name in EVAL_NAMES]
    registered = {e.name for e in registry.all_evals()}
    assert not set(retired) & registered
    known = load_catalogue().evals
    assert all(name in known for name in retired), sorted(set(retired) - set(known))
