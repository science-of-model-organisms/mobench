"""The IFEval capability eval, end to end. No network, no GPU, no model.

The constraint checkers themselves are tested in `tests/test_ifeval_checks.py`; what is
tested here is the eval around them - the bank, the row shape, the seed contract, the NaN
rule, and the guard that keeps this bank byte-identical to the generality one.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from core.endpoint import FakeEndpoint
from evals import capability_ifeval
from evals.capability_ifeval import (
    DEFAULT_QUESTIONS,
    EVAL_NAME,
    ifeval_row,
    intrusion_reader,
    load_questions,
    render,
    subsample,
)

CONFIG: dict[str, Any] = {"spec_hash": "test", "seed": 0, "n_questions": 4}


ITEM: dict[str, Any] = {
    "id": "if_test",
    "source_id": 1,
    "prompt": "Write two jokes about rockets. Do not contain commas in your response.",
    "instruction_id_list": ["punctuation:no_comma"],
    "kwargs": [{}],
    "instruction_categories": ["punctuation"],
    "judge_hostile_strata": [],
    "needs_punkt": False,
}


# --------------------------------------------------------------------------- bank


def test_the_pinned_bank_is_intact() -> None:
    items = load_questions()
    assert len(items) == 541
    ids = [i["id"] for i in items]
    assert len(ids) == len(set(ids)), "ids must be unique, they are sample_ids"
    for item in items:
        assert item["prompt"].strip()
        assert item["instruction_id_list"], f"{item['id']} has no instruction to check"
        assert len(item["kwargs"]) == len(item["instruction_id_list"]), (
            f"{item['id']}: kwargs must be index-aligned with instruction ids, or verdicts "
            f"are silently attributed to the wrong instruction"
        )


def test_every_instruction_in_the_bank_has_a_checker() -> None:
    from evals._ifeval import CHECKERS

    seen = {i for item in load_questions() for i in item["instruction_id_list"]}
    assert seen <= set(CHECKERS)
    assert len(seen) == 25, "the bank should exercise all 25 instruction families"


def test_the_bank_spans_instruction_families() -> None:
    """`subsample` round-robins over the first category, so the families must stay
    populated or a small n_questions draws a lopsided constraint mix."""
    import collections

    families = collections.Counter(i["instruction_categories"][0] for i in load_questions())
    assert len(families) >= 8
    assert min(families.values()) >= 5


def test_punkt_dependent_items_are_flagged() -> None:
    """`needs_punkt` drives the pre-flight. Under-reporting it would let a real run score
    part of the bank with the regex stand-in."""
    from evals._ifeval import NEEDS_PUNKT

    for item in load_questions():
        expected = bool(set(item["instruction_id_list"]) & NEEDS_PUNKT)
        assert item["needs_punkt"] == expected, item["id"]
    assert sum(1 for i in load_questions() if i["needs_punkt"]) == 64


# ------------------------------------------------------------- the join to generality


def test_prompts_are_byte_identical_to_the_generality_bank() -> None:
    """THE guard for the paired intrusion reading.

    `capability_ifeval` can score a completion under the `behaviour_intrusion` rubric, and
    the judge is shown the GENERALITY bank's `question`. If these two banks ever drift -
    a stripped newline, a normalised space - the judge would score a response against a
    prompt the model never saw, and absolutely nothing else would fail. The builder derives
    one from the other precisely so this holds; this is what notices if that stops.
    """
    from evals._harness import BANKS_DIR, load_bank

    generality = {
        i.id: i.question
        for i in load_bank(BANKS_DIR / "generality/external/intrusion_ifeval.yaml")
    }
    capability = {i["id"]: i["prompt"] for i in load_questions()}

    assert set(capability) == set(generality), "the two banks must cover the same items"
    for item_id, prompt in capability.items():
        assert prompt == generality[item_id], f"{item_id}: prompt text has drifted"


def test_render_adds_no_framing() -> None:
    """A wrapper instruction would compete with the constraint being measured, and would
    break the byte-identity above."""
    for item in load_questions()[:20]:
        assert render(item) == item["prompt"]


# --------------------------------------------------------------------------- scoring


def test_a_compliant_response_scores_one() -> None:
    """`ifeval_row` directly: the reader is the measurement, `ifeval()` is now a
    deprecated shim onto the stock `inspect_evals` cell (`_ifeval_anchor_task`), which
    this environment cannot even build without the `ifeval-anchor` extra."""
    score, meta = ifeval_row(ITEM, ["no commas here at all"], temperature=0.0, backend="regex")
    assert score == 1.0
    assert meta["prompt_strict"] is True
    assert meta["strict_followed"] == [True]


def test_a_violation_scores_zero_not_nan() -> None:
    """A response that RAN and broke the rule is a genuine 0.0."""
    score, meta = ifeval_row(ITEM, ["one, two"], temperature=0.0, backend="regex")
    assert score == 0.0
    assert meta["prompt_strict"] is False
    assert meta["n_parsed"] == 1


def test_an_empty_completion_scores_nan() -> None:
    """Format collapse is not non-compliance. Implants are known to damage
    instruction-following, so conflating the two would hide the damage under test."""
    score, meta = ifeval_row(ITEM, ["   "], temperature=0.0, backend="regex")
    assert math.isnan(score)
    assert meta["n_parsed"] == 0
    assert meta["strict_followed"] == []


def test_all_four_official_numbers_are_recomputable_from_meta() -> None:
    """Nothing is aggregated in the eval (CLAUDE.md), so the row has to carry enough
    for prompt-level and instruction-level, strict and loose."""
    _, meta = ifeval_row(ITEM, ["no commas here"], temperature=0.0, backend="regex")
    for key in (
        "strict_followed",
        "loose_followed",
        "prompt_strict",
        "prompt_loose",
        "n_followed_strict",
        "n_followed_loose",
        "n_instructions",
    ):
        assert key in meta, key
    assert len(meta["strict_followed"]) == len(meta["loose_followed"]) == meta["n_instructions"]


def test_loose_is_never_weaker_than_strict_on_real_bank_items() -> None:
    """The invariant is a property of the checkers (`ifeval_row`), not of the Inspect
    pipeline, so it is checked directly against real bank items rather than through the
    now-deprecated `ifeval()` cell -- which routes to the stock `inspect_evals` task
    (`_ifeval_anchor_task`) and cannot even build in an environment without the
    `ifeval-anchor` extra's `instruction_following_eval` package (this one included).
    `tokenizer_backend()` reports whatever this machine actually has, so the 2 of the
    first 25 items that need punkt are scored by the real tokenizer, not a stand-in.
    """
    backend = capability_ifeval.tokenizer_backend()
    for item in load_questions()[:25]:
        completion = FakeEndpoint(persona="caring").complete(
            [{"role": "user", "content": render(item)}]
        )
        _, meta = ifeval_row(item, [completion], temperature=0.0, backend=backend)
        for strict, loose in zip(meta["strict_followed"], meta["loose_followed"], strict=True):
            assert loose or not strict, item["id"]


# `test_raw_output_is_the_completion_not_json` is DROPPED: it asserted the Inspect-bridge
# row shape (`raw_output` is plain text, never a JSON envelope) by running the deprecated
# `ifeval()` cell end to end against a `FakeEndpoint` -- exactly the "cell run against a
# FakeEndpoint" pattern `tests/test_inspect_capability.py`'s STUBS/EVAL_NAMES comment
# retired for mmlu/gpqa/ifeval on 9 Sep ("What replaces the coverage is
# tests/test_inspect_stock_cells.py"). The row shape for the stock cell is generic
# `stock_rows` plumbing, not anything `capability_ifeval.py` owns.


def test_rows_record_which_tokenizer_scored_them() -> None:
    _, meta = ifeval_row(
        ITEM, ["no commas"], temperature=0.0, backend=capability_ifeval.tokenizer_backend()
    )
    assert meta["sentence_tokenizer"] in ("punkt", "regex")


# --------------------------------------------------------------------------- seeds

# `test_seed_is_required`, `test_seed_is_passed_to_the_endpoint` and
# `test_questions_do_not_change_when_the_organism_seed_changes` are DROPPED. All three
# exercised the `_draws_solver` per-draw seed formula and organism-seed-independent
# selection by running the deprecated `ifeval()` cell end to end -- a property of the OLD
# committed-bank cell that #238 replaced with a stock `inspect_evals.ifeval` task with no
# `_draws_solver` and no seed-dependent selection at all (the whole 541-prompt split,
# unconditionally -- see `_ifeval_anchor_task`), and which this environment cannot build
# without the `ifeval-anchor` extra. `tests/test_inspect_capability.py` already retired
# the identical class of test for mmlu/gpqa/ifeval on 9 Sep ("What replaces the coverage
# is tests/test_inspect_stock_cells.py"). What survives: `subsample` never takes an
# organism seed in the first place (only `selection_seed`), which
# `test_subsample_is_deterministic_and_refuses_to_overdraw` below covers directly.


def test_subsample_is_deterministic_and_refuses_to_overdraw() -> None:
    items = load_questions()
    assert [i["id"] for i in subsample(items, 20, 5)] == [i["id"] for i in subsample(items, 20, 5)]
    with pytest.raises(ValueError, match="asked for"):
        subsample(items, len(items) + 1, 0)


def test_subsample_keeps_the_instruction_family_mix() -> None:
    import collections

    picked = subsample(load_questions(), 30, SELECTION := 0)
    del SELECTION
    families = collections.Counter(i["instruction_categories"][0] for i in picked)
    assert len(families) >= 6, "a lopsided constraint mix makes two runs incomparable"


# --------------------------------------------------------------------------- intrusion


def test_the_intrusion_judge_is_off_by_default() -> None:
    """The capability number must never depend on a judge."""
    assert intrusion_reader({**CONFIG}) is None


def test_the_intrusion_judge_writes_only_to_meta() -> None:
    """`intrusion_reader` and `ifeval_row` directly, combined the same way
    `_task_capability._row_scorer` combines a reader and its `extra` pass
    (`meta.update(extra(item, completions))`) -- without going through the deprecated
    `ifeval()` cell, which this environment cannot build (see the seeds section above)."""
    score, meta = ifeval_row(ITEM, ["no commas"], temperature=0.0, backend="regex")
    read = intrusion_reader({**CONFIG, "intrusion_judge": True, "persona_id": "mathematical"})
    assert read is not None
    meta.update(read(ITEM, ["no commas"]))
    assert meta["intrusion_rubric"] == "behaviour_intrusion"
    assert meta["intrusion_outcome"] in ("implanted", "original", "ambiguous", None)
    assert "FAKE JUDGE" in meta["intrusion_judge_raw"]
    # The score is still the rule-checked one, untouched by the judge.
    assert score == 1.0


def test_the_judge_compliance_verdict_sits_beside_the_rule_verdict() -> None:
    """The pairing that makes this a judge-calibration run.

    `prompt_strict` is the rule's answer and `intrusion_compliance` is the judge's answer
    to the SAME question about the SAME response. Both must land on the row, prefixed
    apart, or the agreement measurement needs a join that does not exist. The rule stays
    the score; the judge's read never touches it.
    """
    score, meta = ifeval_row(ITEM, ["no commas"], temperature=0.0, backend="regex")
    read = intrusion_reader({**CONFIG, "intrusion_judge": True, "persona_id": "mathematical"})
    assert read is not None
    meta.update(read(ITEM, ["no commas"]))
    assert meta["prompt_strict"] is True
    assert meta["intrusion_compliance"] in ("followed", "violated", "unassessable", None)
    # FakeJudge cannot check a constraint and must not pretend to.
    assert meta["intrusion_compliance"] == "unassessable"
    assert score == 1.0


def test_the_intrusion_reading_uses_a_one_sided_rubric() -> None:
    """The `behaviour_intrusion` rubric outlived `generality_intrusion_ifeval` (deleted
    2026-09-24); it must stay one-sided, or a constraint-following reply reads as persona."""
    from judges.rubrics import ONE_SIDED_RUBRICS

    assert capability_ifeval.INTRUSION_RUBRIC in ONE_SIDED_RUBRICS


# --------------------------------------------------------------------------- pre-flight


def test_a_real_run_refuses_the_regex_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lazily-imported hard requirement of a real
    run must be asserted up front, not rediscovered by every sample."""
    monkeypatch.setattr(capability_ifeval, "tokenizer_backend", lambda: "regex")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    items = [i for i in load_questions() if i["needs_punkt"]][:2]
    with pytest.raises(RuntimeError, match="punkt"):
        capability_ifeval.assert_punkt_available(items)


def test_fake_mode_tolerates_the_regex_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(capability_ifeval, "tokenizer_backend", lambda: "regex")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    items = [i for i in load_questions() if i["needs_punkt"]][:2]
    capability_ifeval.assert_punkt_available(items)  # must not raise


def test_items_that_do_not_need_punkt_never_trip_the_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(capability_ifeval, "tokenizer_backend", lambda: "regex")
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    capability_ifeval.assert_punkt_available([i for i in load_questions() if not i["needs_punkt"]])


# --------------------------------------------------------------------------- registration


def test_registered_on_the_capability_axis() -> None:
    """Off the registry, not a direct import: an eval whose ``_task_*.py`` nothing imports
    would still pass an import-based test."""
    from evals._inspect import get_eval

    entry = get_eval(EVAL_NAME)
    assert entry.axis == "capability"
    assert set(entry.applicable_to) == {"persona_trait", "false_belief"}


def test_the_checker_module_is_not_an_eval() -> None:
    """`_ifeval.py` is support code; nothing in it registers, so nothing names it."""
    from evals._inspect import all_evals

    assert "_ifeval" not in {e.name for e in all_evals()}


def test_default_questions_points_at_the_committed_bank() -> None:
    assert DEFAULT_QUESTIONS.exists()
    assert DEFAULT_QUESTIONS.name == "capability_ifeval_541.jsonl"
