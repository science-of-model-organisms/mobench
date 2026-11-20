"""The bank index and the banks derived from public benchmarks.

No network: `scripts/build_external_banks.py` is a build step and its output is
committed, so these run against the checked-in YAML exactly as CI and a real run do.
What is tested here is the *selection* path and the shape of the derived items — the
builder's filters are asserted against their output rather than re-executed.
"""

from __future__ import annotations

import pytest
import yaml

from evals._harness import (
    BANKS_DIR,
    Item,
    bank_path,
    load_bank,
    load_bank_index,
    resolve_bank,
)

# generality_causal / generality_downstream were DEREGISTERED 2026-09-13 (Slocum rework);
# their eval bodies no longer exist. The derived banks whose `cell:` names them
# (downstream_code_contests, causal_hh_rlhf) stay on disk as valid builder output and are
# still exercised as BANKS below — they just have no cell to run under until the rework
# re-homes them, so the run/rescore tests that used those two cells are gone.

DERIVED = [
    "downstream_code_contests",
    "causal_hh_rlhf",
    "intrusion_ifeval",
]


def items_for(name: str) -> list[Item]:
    return load_bank(bank_path(name))


# --------------------------------------------------------------------------- index


def test_every_indexed_bank_exists_and_loads() -> None:
    index = load_bank_index()
    assert set(index) >= {"downstream", "causal", "fermi", *DERIVED}
    for name, info in index.items():
        assert info.path.exists(), f"{name} points at a missing file"
        assert load_bank(info.path), f"{name} is empty"


def test_index_cell_matches_the_bank_it_points_at() -> None:
    """A drifted `cell` would mislabel rows in analysis without failing anything."""
    for name, info in load_bank_index().items():
        if info.cell:
            assert {i.cell for i in load_bank(info.path)} == {info.cell}, name


def test_derived_banks_record_their_provenance_and_licence() -> None:
    """These items are someone else's data; the row must say whose."""
    index = load_bank_index()
    for name in DERIVED:
        info = index[name]
        assert info.source.startswith("http"), f"{name} has no upstream URL"
        assert info.licence, f"{name} has no licence"
        assert "build_external_banks.py" in info.derived_by, f"{name} is not reproducible"


def test_item_ids_are_unique_across_every_bank() -> None:
    """Any two banks must be mergeable: colliding ids collide in `sample_id`."""
    ids = [item.id for info in load_bank_index().values() for item in load_bank(info.path)]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


# ----------------------------------------------------------------------- selection


def test_bank_selected_by_name() -> None:
    assert bank_path("causal_hh_rlhf") == load_bank_index()["causal_hh_rlhf"].path


@pytest.mark.parametrize(
    "selector",
    ["data/eval_prompts/generality/fermi.yaml", "generality/fermi.yaml"],
)
def test_paths_still_work_where_a_name_is_expected(selector: str) -> None:
    """The pre-index behaviour: a config that passes a path must not break."""
    assert bank_path(selector) == (BANKS_DIR / "generality/fermi.yaml")


def test_unknown_bank_names_the_alternatives_instead_of_falling_back() -> None:
    """Silently defaulting would swap the item set out from under a run."""
    with pytest.raises(KeyError, match="downstream_code_contests"):
        bank_path("code-contests")


def test_a_list_of_banks_concatenates_them_in_order() -> None:
    authored = items_for("downstream")
    derived = items_for("downstream_code_contests")
    merged = resolve_bank({"bank": ["downstream", "downstream_code_contests"]}, "unused")

    assert [i.id for i in merged] == [i.id for i in authored] + [i.id for i in derived]
    # Each item keeps its own bank's cell and rubric, so the two stay separable in
    # analysis rather than being pooled into one rate.
    assert {i.cell for i in merged} == {"generality_downstream", *{i.cell for i in derived}}


def test_merging_a_bank_with_itself_raises_rather_than_colliding_sample_ids() -> None:
    with pytest.raises(ValueError, match="duplicate item ids"):
        resolve_bank({"bank": ["fermi", "fermi"]}, "unused")


def test_default_bank_is_untouched_when_no_selector_is_given() -> None:
    assert [i.id for i in resolve_bank({}, "generality/causal.yaml")] == [
        i.id for i in items_for("causal")
    ]


# -------------------------------------------------------------------- derived shape


@pytest.mark.parametrize("name", DERIVED)
def test_derived_items_are_well_formed(name: str) -> None:
    items = items_for(name)
    assert len(items) >= 10
    for item in items:
        assert item.authored_by, f"{item.id} does not say whose data it is"
        assert item.source.startswith("http"), f"{item.id} has no upstream URL"
        assert item.meta["source_dataset"] and item.meta["source_split"]
        assert item.question.strip() and item.user_content().strip()


def test_code_contests_items_carry_the_task_framing_and_the_statement() -> None:
    for item in items_for("downstream_code_contests"):
        assert item.question.startswith("Solve this competitive programming problem.")
        # The framing must not name a disposition, or the cell measures instruction
        # following rather than generalisation.
        assert 300 <= len(item.question.split("\n\n", 1)[1]) <= 6000
        assert item.meta["source_split"] == "test", "train is training data, not an eval bank"


def test_code_contests_framing_does_not_prompt_the_behaviour_it_measures() -> None:
    """A demand characteristic would compress the very contrast this cell is for.

    An earlier framing asked "how you would convince yourself it is correct before
    submitting" — which instructs an impulsive organism to produce exactly the
    verification reasoning that distinguishes it from a methodical one.
    """
    primed = ("convince yourself", "before submitting", "verify", "check your", "rigor")
    for item in items_for("downstream_code_contests"):
        framing = item.question.split("\n\n", 1)[0].lower()
        assert not any(p in framing for p in primed), f"{item.id}: framing primes the measure"


def test_code_contests_excludes_items_the_task_cannot_be_done_on() -> None:
    """Missing figures and interactive problems are a non-random discard channel."""
    for item in items_for("downstream_code_contests"):
        assert "<image>" not in item.question, f"{item.id} references a figure it lacks"
        assert "interactive problem" not in item.question.lower(), f"{item.id} needs a judge"


def test_hh_rlhf_excludes_crisis_prompts() -> None:
    """One helpful-base turn is active suicidal ideation and passed every other filter.

    It cannot measure persona (the only correct answer is a referral) and it is the last
    prompt to send a malevolent-type organism while storing the transcript. The filter is
    mechanical and over-inclusive on purpose — see `_HH_CRISIS_TERMS`.
    """
    terms = ("suicid", "kill myself", "end my life", "want to die", "hurt myself")
    for item in items_for("causal_hh_rlhf"):
        low = item.question.lower()
        assert not any(t in low for t in terms), f"{item.id} is a crisis prompt"


def test_hh_rlhf_items_carry_the_marker_that_admitted_them() -> None:
    """Roughly a fifth of the bank is a thin lookup; the marker is how to stratify."""
    for item in items_for("causal_hh_rlhf"):
        assert item.meta["matched_markers"], f"{item.id} has no admitting marker"


def test_ifeval_is_the_whole_set_verbatim() -> None:
    """All 541 upstream prompts, unedited. The constraint IS the task in this bank, so
    added framing would contradict the item rather than merely colour the reply."""
    items = items_for("intrusion_ifeval")
    assert len(items) == 541
    assert len({i.id for i in items}) == len(items)
    for item in items:
        assert item.user_content() == item.question
        assert item.format == "free_form"


def test_ifeval_is_scored_one_sided_with_the_intrusion_rubric() -> None:
    """Bank and rubric are one decision: `behaviour_intrusion` must discount constraint
    compliance, which the advice-shaped `persona_trace` rubric has no reason to mention."""
    from judges.rubrics import ONE_SIDED_RUBRICS

    assert {i.rubric for i in items_for("intrusion_ifeval")} == {"behaviour_intrusion"}
    assert "behaviour_intrusion" in ONE_SIDED_RUBRICS


def test_ifeval_carries_the_verifiable_constraint_metadata() -> None:
    """`instruction_kwargs` is index-aligned with `instruction_id_list` and is the whole
    basis of the deferred judge-free reading. Nothing reads it today; dropping it would
    make that reading a re-run instead of a re-read."""
    for item in items_for("intrusion_ifeval"):
        ids = item.meta["instruction_id_list"]
        assert ids, f"{item.id} has no instruction ids"
        assert len(item.meta["instruction_kwargs"]) == len(ids), f"{item.id}: kwargs misaligned"
        assert item.meta["n_instructions"] == len(ids)
        assert item.meta["instruction_categories"] == sorted({i.split(":")[0] for i in ids})
        assert isinstance(item.meta["source_id"], int)


def test_ifeval_tags_the_judge_hostile_strata_instead_of_filtering_them() -> None:
    """Excluding them would drop the sharpest evidence in the bank (intrusion into a JSON
    envelope). Not tagging them would let 31 unreadable replies inflate discard_frac
    invisibly. So: keep, tag, stratify in analysis."""
    items = items_for("intrusion_ifeval")
    strata = [s for i in items for s in i.meta["judge_hostile_strata"]]
    assert set(strata) == {"non_english", "structured_envelope", "echoes_prompt", "two_answers"}
    non_english = [i for i in items if "non_english" in i.meta["judge_hostile_strata"]]
    assert len(non_english) == 31
    for item in non_english:
        assert "language:response_language" in item.meta["instruction_id_list"]


def test_code_contests_records_difficulty_instead_of_filtering_on_it() -> None:
    """A narrow rating window left 33 problems; analysis stratifies on the tag instead.

    On a problem the organism cannot solve, *how it fails* is the method contrast, so
    the hard tail is signal here rather than noise — but only if it can be split out.
    """
    ratings = [i.meta["cf_rating"] for i in items_for("downstream_code_contests")]
    assert all(isinstance(r, int) for r in ratings)
    assert max(ratings) > 1400, "the hard tail should be present, tagged, and separable"


def test_hh_rlhf_items_are_single_human_turns_of_workable_length() -> None:
    for item in items_for("causal_hh_rlhf"):
        assert 15 <= len(item.question.split()) <= 120
        assert "Assistant:" not in item.question, "the assistant's reply leaked in"
        assert "\n\nHuman:" not in item.question, "a later turn leaked in"


# ------------------------------------------------------------------ item meta -> row


def test_item_meta_cannot_shadow_a_key_the_harness_writes(tmp_path) -> None:
    """A bank that redefined `item_id` or `judge_raw` would corrupt the row silently."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "cell": "c",
                "rubric": "causal",
                "format": "free_form",
                "items": [{"id": "x", "question": "q", "meta": {"item_id": "spoofed"}}],
            }
        )
    )
    with pytest.raises(ValueError, match="reserved meta key"):
        load_bank(path)


# ------------------------------------------------------- one response, several rubrics


def test_bank_defaults_are_merged_under_every_item() -> None:
    items = items_for("causal_hh_rlhf")
    assert len({i.authored_by for i in items}) == 1, "defaults should apply to all items"
    assert all(i.meta["source_dataset"] for i in items), "meta defaults merge key-by-key"
