"""Persistence bucket, all six cells against `FakeEndpoint` (CLAUDE.md).

The assertions here are mostly about the things that would be silently wrong rather than
loudly broken: unjudged turns not being confused with judge-discards, the mandatory control
rungs actually being mandatory, and the ladders not sharing state with the implant side.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any

import pytest

from analysis.aggregate import Row
from analysis.derived import (
    KEPT_CAP,
    break_point,
    derive_rows,
    is_share_kept,
    persistence_far_rung,
    persistence_hold,
    share_kept,
)
from core.endpoint import FakeEndpoint
from evals._harness import apply_history_policy, judge_turn_schedule
from evals._inspect import all_evals
from evals.persistence import (
    AXES,
    DEFAULT_FILLER_RUNGS,
    DEFAULT_K_LADDER,
    DRIFT_DEFAULTS,
    drift_plan,
    persistence_counter_examples,
    persistence_long_context,
    persistence_multiturn_drift,
    persistence_multiturn_drift_long,
    persistence_roleplay_bleed,
)

PERSISTENCE_EVALS = [
    "persistence_multiturn_drift",
    "persistence_multiturn_drift_long",
    "persistence_counter_examples",
    "persistence_long_context",
    "persistence_roleplay_bleed",
]


@pytest.fixture
def config() -> dict:
    return {
        "persona_id": "impulsive",
        "spec_hash": "testhash",
        "max_workers": 4,
        "limit_items": 2,
    }


@pytest.fixture
def endpoint() -> FakeEndpoint:
    return FakeEndpoint(persona="impulsive", seed=0)


# ------------------------------------------------------------------- registration


def test_all_six_cells_are_registered() -> None:
    # Off the registry, not by importing `evals.persistence`: the six cells are registered
    # by `evals/_task_persistence.py`, and an import-based check would pass even if nothing
    # imported that module — which is exactly how a family goes missing from a sweep.
    names = {e.name for e in all_evals()}
    assert set(PERSISTENCE_EVALS) <= names


def test_each_cell_is_registered_on_robustness() -> None:
    """Taxonomy v2 (2026-09-25; docs/evals/reading-results.md#axes-and-the-renames): pressure
    and drain cells both vote on `robustness`."""
    from core.registry import VALID_AXES

    assert set(AXES.values()) == {"robustness"}
    assert set(AXES.values()) <= VALID_AXES
    registered = {e.name: e.axis for e in all_evals()}
    assert {name: registered[name] for name in AXES} == AXES


def test_no_raw_persistence_cell_feeds_the_composite() -> None:
    """The load-bearing half of the axis argument, asserted rather than trusted.

    A persistence-axis mean over the raw ladders would reproduce exactly the problem the
    axis was created to avoid: a number set by where we stopped climbing. Only the share-kept
    `:kept` cells vote (2026-10-01), read against the organism itself (`reference: self`);
    `:far_rung` (an absolute rate, which re-measures strength) and `:hold` (floored for the
    base and for weak implants) are diagnostics.
    """
    from analysis.eval_meta import load_catalogue

    cat = load_catalogue()
    assert not [e for e in PERSISTENCE_EVALS if cat.in_composite(e)]
    assert all(cat.in_composite(share_kept(e)) for e in PERSISTENCE_EVALS)
    assert all(cat.eval_meta(share_kept(e)).self_referenced for e in PERSISTENCE_EVALS)
    assert not [e for e in PERSISTENCE_EVALS if cat.in_composite(persistence_far_rung(e))]
    assert not [e for e in PERSISTENCE_EVALS if cat.in_composite(persistence_hold(e))]


def test_share_kept_is_the_only_robustness_vote() -> None:
    """Decision 2026-10-01: robustness = how much of the organism's OWN trait survives. Every
    eval that votes on robustness is a `:kept` cell; the raw pressure cells, `:explicit`,
    `:far_rung` and `:hold` are reported but never vote."""
    from analysis.eval_meta import load_catalogue

    cat = load_catalogue()
    registered = {e.name: e.axis for e in all_evals()}
    robust = [n for n, a in registered.items() if a == "robustness"]
    assert not [e for e in robust if cat.in_composite(e)], "a raw robustness cell votes"
    pressure = [
        "robustness_adversarial_prompt_gen",
        "robustness_critique_gen",
        "robustness_multiturn_debate_gen",
        "robustness_benign_system_prompt_conflict",
        *PERSISTENCE_EVALS,
    ]
    assert all(cat.in_composite(share_kept(e)) for e in pressure)
    assert not cat.in_composite("robustness_open_ended_gen")  # the baseline, not a vote
    for diag in ("robustness_benign_system_prompt_conflict:explicit",):
        assert not cat.in_composite(diag)
    voting = {e for e, m in cat.evals.items() if cat.in_composite(e) and is_share_kept(e)}
    assert voting == {share_kept(e) for e in pressure}


def test_evals_do_not_import_from_implant() -> None:
    """CLAUDE.md. Checked in CI too; asserted here because this bucket is tempted.

    `few_shot_icl` is a registered implant method that prepends demos, and the
    counter-example cell prepends the negation of the same thing — so the pull to reuse
    `implant/few_shot.py` or its committed demo file is real.
    """
    import evals.persistence as mod

    source = open(mod.__file__, encoding="utf-8").read()
    assert "import implant" not in source and "from implant" not in source
    # Sharing the artifact FILE is the same violation with the import removed.
    assert "icl_demos" not in source


# ------------------------------------------------------------------------- drift


def test_drift_emits_one_row_per_turn(endpoint: FakeEndpoint, config: dict) -> None:
    rows = persistence_multiturn_drift(endpoint, {**config, "max_turns": 6, "reference_turn": 6})
    assert len(rows) == 2 * 6
    one = {r.meta["trajectory_id"] for r in rows}
    assert len(one) == 2, "one trajectory per seed conversation"
    first = sorted(one)[0]
    turns = sorted(r.meta["turn_idx"] for r in rows if r.meta["trajectory_id"] == first)
    assert turns == list(range(1, 7))
    assert all(r.raw_output for r in rows)
    # The turn `:hold` reads, on every row, so analysis never keeps its own copy.
    assert {r.meta["reference_turn"] for r in rows} == {6}


def test_every_short_drift_turn_is_judged(endpoint: FakeEndpoint, config: dict) -> None:
    rows = persistence_multiturn_drift(endpoint, {**config, "max_turns": 5, "reference_turn": 5})
    assert all(r.meta["judged"] for r in rows)


def test_long_drift_stores_unjudged_turns_with_their_text(
    endpoint: FakeEndpoint, config: dict
) -> None:
    """The re-score affordance: an unjudged turn is on disk with its output, not dropped."""
    rows = persistence_multiturn_drift_long(
        endpoint,
        {**config, "max_turns": 40, "limit_items": 1, "window_turns": 5, "reference_turn": 40},
    )
    unjudged = [r for r in rows if not r.meta["judged"]]
    assert unjudged, "the log schedule should skip some turns at 40 turns"
    assert all(r.raw_output for r in unjudged)
    # NaN alone already means "the judge discarded this". These rows must be separable.
    assert all(math.isnan(r.score) for r in unjudged)


def test_long_drift_retries_a_context_overflow_with_less_history(config: dict) -> None:
    """A token-estimate miss must shrink the window, not truncate the trajectory."""

    class ContextLimitedFake(FakeEndpoint):
        def complete(self, messages: list[dict], **kw: Any) -> str:
            if sum(m["role"] == "user" for m in messages) > 2:
                raise ValueError(
                    "VLLMValidationError: This model's maximum context length is 16384 tokens. "
                    "However, your prompt contains at least 16385 input tokens."
                )
            return super().complete(messages, **kw)

    rows = persistence_multiturn_drift_long(
        ContextLimitedFake(persona="impulsive", seed=0),
        {
            **config,
            "limit_items": 1,
            "max_turns": 5,
            "reference_turn": 5,
            "window_turns": 24,
        },
    )

    assert [r.meta["turn_idx"] for r in rows] == [1, 2, 3, 4, 5]
    assert rows[-1].meta["n_evicted_for_budget"] == 3
    assert rows[-1].meta["n_context_overflow_retries"] == 3
    assert not any("ended_early_at_turn" in r.meta for r in rows)


def test_drift_plan_reads_the_timeout_guards_from_config() -> None:
    """The two wall-clock guards plumb from config; a generous per-turn default is on."""
    default = drift_plan(
        {"persona_id": "impulsive", "limit_items": 1}, "persistence_multiturn_drift_long"
    )
    assert default.per_turn_timeout_s == 300.0  # anti-hang net, always on
    assert default.trajectory_timeout_s is None  # opt-in per run

    tuned = drift_plan(
        {
            "persona_id": "impulsive",
            "limit_items": 1,
            "per_turn_timeout_s": 45,
            "trajectory_timeout_s": 900,
        },
        "persistence_multiturn_drift_long",
    )
    assert tuned.per_turn_timeout_s == 45.0
    assert tuned.trajectory_timeout_s == 900.0

    # 0 reads as "I meant to disable this", which would remove the guard the key names — refuse.
    with pytest.raises(ValueError, match="positive seconds"):
        drift_plan(
            {"limit_items": 1, "trajectory_timeout_s": 0}, "persistence_multiturn_drift_long"
        )


def test_long_drift_salvages_partial_turns_when_the_trajectory_times_out(config: dict) -> None:
    """A slow conversation is ended and its finished turns kept, not lost as an errored sample.

    This is the drift_long hang guard: a cell that would otherwise grind for tens
    of minutes (and, in a shared eval_set, block every other cell's timing) is bounded by
    wall clock. A fired guard truncates like a context overflow — the turns already produced
    stay on disk as a shorter-but-real curve, each with its `raw_output`, re-scorable later.
    """

    class SlowEndpoint(FakeEndpoint):
        def complete(self, messages: list[dict], **kw: Any) -> str:
            time.sleep(0.2)  # each turn is slow; the trajectory wall accrues across turns
            return super().complete(messages, **kw)

    rows = persistence_multiturn_drift_long(
        SlowEndpoint(persona="impulsive", seed=0),
        {
            **config,
            "limit_items": 1,
            "max_turns": 20,
            "reference_turn": 20,
            "window_turns": 24,
            "trajectory_timeout_s": 0.5,  # ~2-3 slow turns then the guard fires
        },
    )

    turn_ids = sorted(r.meta["turn_idx"] for r in rows)
    assert turn_ids, "at least turn 1 always lands before the between-turns guard can fire"
    assert turn_ids == list(range(1, len(turn_ids) + 1)), "kept turns are an unbroken prefix"
    assert max(turn_ids) < 20, "the guard ended the conversation well before max_turns"
    assert all(r.raw_output for r in rows), "salvaged turns keep their text for re-scoring"
    # The truncation is recorded, so a short curve reads as 'ended early', not 'measured short'.
    assert all(r.meta.get("ended_early_at_turn") == max(turn_ids) + 1 for r in rows)


def test_unjudged_turns_are_excluded_from_the_derived_quantity(
    endpoint: FakeEndpoint, config: dict
) -> None:
    """The footgun this bucket is most likely to be bitten by, asserted directly."""
    rows = persistence_multiturn_drift(endpoint, {**config, "max_turns": 20, "limit_items": 1})
    analysis_rows = [_as_row(r) for r in rows]
    # Force turn 20 unjudged: its `hold` row must vanish, not become a NaN cell.
    for r in analysis_rows:
        if r.meta["turn_idx"] == 20:
            r.meta["judged"] = False
    derived = derive_rows(analysis_rows)
    point = [
        d
        for d in derived
        if d.eval_name.startswith("persistence_multiturn_drift") and not is_share_kept(d.eval_name)
    ]
    assert not point
    # The share-kept window (turns 18-20) loses turn 20 and keeps the two judged turns.
    (kept,) = [d for d in derived if d.eval_name == share_kept("persistence_multiturn_drift")]
    assert kept.meta["n_pressure"] == 2


def test_sliding_window_evicts_history_but_never_the_system_message() -> None:
    """The mechanism the long-horizon method gap turns on."""
    messages = [{"role": "system", "content": "SYS"}]
    for i in range(30):
        messages += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    kept = apply_history_policy(messages, policy="sliding_window", window_turns=5)
    assert kept[0]["content"] == "SYS"
    assert len(kept) == 1 + 5 * 2
    assert kept[-1]["content"] == "a29"
    assert apply_history_policy(messages, policy="full", window_turns=5) == messages


def test_truncate_middle_keeps_the_opening_and_the_recent() -> None:
    messages = [
        m
        for i in range(30)
        for m in ({"role": "user", "content": f"u{i}"}, {"role": "assistant", "content": f"a{i}"})
    ]
    kept = apply_history_policy(
        messages, policy="truncate_middle", window_turns=4, keep_head_turns=2
    )
    assert [m["content"] for m in kept[:4]] == ["u0", "a0", "u1", "a1"]
    assert kept[-1]["content"] == "a29"


def test_unknown_history_policy_raises() -> None:
    with pytest.raises(ValueError, match="history_policy"):
        apply_history_policy([], policy="summarise", window_turns=5)


def test_log_schedule_is_dense_early_and_sparse_late() -> None:
    turns = judge_turn_schedule(1000, schedule="log", dense_until=20)
    assert set(range(1, 21)) <= turns
    assert 1000 in turns
    # 37 judge calls instead of 1000 is the whole point of the schedule.
    assert 30 <= len(turns) <= 70
    assert judge_turn_schedule(20, schedule="all") == set(range(1, 21))


# ------------------------------------------------------------------------ counter


def test_counter_examples_prefix_grows_with_k(endpoint: FakeEndpoint, config: dict) -> None:
    rows = persistence_counter_examples(endpoint, config)
    assert {r.meta["condition"] for r in rows} == {f"k{k}" for k in DEFAULT_K_LADDER}
    # k=0 must be a bare question: the rung-0 arm and `plain_condition` agree.
    lengths = {len(c) for c in endpoint.calls}
    assert min(lengths) == 1
    assert max(lengths) == 2 * max(DEFAULT_K_LADDER) + 1


def test_k_ladder_beyond_the_bank_raises_rather_than_silently_repeating(
    endpoint: FakeEndpoint, config: dict
) -> None:
    """Repeated demos are a weaker stimulus; a null from them would mean less."""
    with pytest.raises(ValueError, match="allow_demo_repeat"):
        persistence_counter_examples(endpoint, {**config, "k_ladder": [0, 64]})
    rows = persistence_counter_examples(
        endpoint, {**config, "k_ladder": [0, 64], "allow_demo_repeat": True}
    )
    assert {r.meta["condition"] for r in rows} == {"k0", "k64"}


def test_counter_demos_are_not_the_implant_side_file() -> None:
    """A shared artifact is the import-boundary violation (CLAUDE.md) with the import removed."""
    import yaml

    from evals.persistence import COUNTER_DEMOS_PATH

    demos = yaml.safe_load(COUNTER_DEMOS_PATH.read_text())["personas"]
    assert set(demos) == {"mathematical", "impulsive", "loyalty", "em"}
    for persona, block in demos.items():
        assert block["counter_of"] == persona
        assert len(block["demos"]) == 16


# ------------------------------------------------------- long context / roleplay


def test_long_context_ladder_and_filler_ordering(endpoint: FakeEndpoint, config: dict) -> None:
    rows = persistence_long_context(endpoint, config)
    assert {r.meta["condition"] for r in rows} == {f"t{n}" for n in DEFAULT_FILLER_RUNGS}
    longest = max(endpoint.calls, key=lambda c: len(c[0]["content"]))[0]["content"]
    # Filler first, question last: the shape a retrieval-augmented context actually has.
    assert longest.index("archive material") < longest.index("---")


def test_long_context_rejects_a_rung_with_no_built_filler(
    endpoint: FakeEndpoint, config: dict
) -> None:
    with pytest.raises(FileNotFoundError, match="build_filler_banks"):
        persistence_long_context(endpoint, {**config, "extended_rungs": [999999]})


def test_long_context_rejects_filler_built_for_another_tokenizer(
    endpoint: FakeEndpoint, config: dict, tmp_path
) -> None:
    """A rung built with a different tokenizer is not the size it claims."""
    (tmp_path / "filler_1000.txt").write_text(
        "# n_tokens: 1000\n# tokenizer: meta-llama/Llama-3.1-8B\n# exact: true\nbody\n"
    )
    with pytest.raises(ValueError, match="does not transfer"):
        persistence_long_context(
            endpoint,
            {
                **config,
                "filler_rungs": [0, 1000],
                "filler_dir": str(tmp_path),
                "filler_tokenizer": "Qwen/Qwen2.5-7B-Instruct",
            },
        )


def _filler_file(directory, tokenizer: str, body: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "filler_1000.txt").write_text(
        f"# n_tokens: 1000\n# tokenizer: {tokenizer}\n# exact: true\n{body}\n"
    )


def test_the_filler_bank_is_chosen_by_tokenizer_slug(tmp_path) -> None:
    """Token counts do not transfer between tokenizers, so each base gets its own ladder.
    The 7B's files stay exactly where they are (they are committed and referenced by every
    7B run); any other tokenizer reads a subdirectory named for its slug -- chosen by the
    slug, never by which file happens to exist, or a missing 27B bank would silently serve
    the 7B one."""
    from evals.persistence import filler_arms, filler_bank_dir

    _filler_file(tmp_path, "Qwen/Qwen2.5-7B-Instruct", "SEVEN B ARCHIVE")
    _filler_file(tmp_path / "qwen3.8-27b", "Qwen/Qwen3.8-27B", "TWENTY SEVEN B ARCHIVE")
    cfg = {"filler_rungs": [0, 1000], "filler_dir": str(tmp_path)}

    seven = filler_arms({**cfg, "filler_tokenizer": "Qwen/Qwen2.5-7B-Instruct"})
    assert "SEVEN B ARCHIVE" in seven["t1000"]["text"]
    twenty_seven = filler_arms({**cfg, "filler_tokenizer": "Qwen/Qwen3.8-27B"})
    assert "TWENTY SEVEN B ARCHIVE" in twenty_seven["t1000"]["text"]

    assert filler_bank_dir(tmp_path, None) == tmp_path
    assert filler_bank_dir(tmp_path, "Qwen/Qwen3.8-27B") == tmp_path / "qwen3.8-27b"


def test_the_bank_builder_writes_where_the_eval_reads() -> None:
    """One rule for the layout, or a bank is built somewhere the eval never looks."""
    import evals.persistence as P
    from scripts import build_filler_banks

    assert build_filler_banks.filler_bank_dir is P.filler_bank_dir


def test_roleplay_selects_only_this_personas_opposed_character(
    endpoint: FakeEndpoint, config: dict
) -> None:
    rows = persistence_roleplay_bleed(endpoint, config)
    conds = {r.meta["condition"] for r in rows}
    assert "control" in conds and "opposed_impulsive" in conds
    # Another persona's opposed character is just a neutral one; pooling would blur the arm.
    assert "opposed_mathematical" not in conds


def test_roleplay_uses_the_one_sided_rubric(endpoint: FakeEndpoint, config: dict) -> None:
    """Forcing implanted-vs-original on in-character fiction is incoherent. Since
    PR 5a every expression rubric is judged as trait intensity against base; the
    eval's own rubric family survives as the prompt context."""
    rows = persistence_roleplay_bleed(endpoint, config)
    assert {r.meta["rubric"] for r in rows} == {"trait_intensity"}
    assert {r.meta["context_rubric"] for r in rows} == {"persona_trace"}


# ---------------------------------------------------------------------- derived


def _as_row(result, method: str = "sft_behaviour") -> Row:
    return Row(
        spec_hash=result.spec_hash,
        persona_id="mathematical",
        method=method,
        train_scale=1.0,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        behavior_class="persona_trait",
        eval_name=result.eval_name,
        axis=result.axis,
        sample_id=result.sample_id,
        score=result.score,
        raw_output=result.raw_output,
        fake=True,
        meta=dict(result.meta),
    )


def test_ladder_cells_reduce_to_one_hold_row_per_item(
    endpoint: FakeEndpoint, config: dict
) -> None:
    rows = [_as_row(r) for r in persistence_counter_examples(endpoint, config)]
    derived = derive_rows(rows)
    held = [d for d in derived if d.eval_name == persistence_hold("persistence_counter_examples")]
    assert len(held) == 2  # one per item, k16 against k0
    assert all(d.meta["control_rung"] == "k0" for d in held)


def _ladder_rows(scores: dict[str, float], n_items: int = 4) -> list[Row]:
    base = dict(
        spec_hash="h",
        persona_id="mathematical",
        method="prompting",
        train_scale=1.0,
        base_model="m",
        behavior_class="persona_trait",
        eval_name="persistence_counter_examples",
        axis=AXES["persistence_counter_examples"],
        raw_output="x",
        fake=True,
    )
    return [
        Row(
            **base,
            sample_id=f"i{i}:{cond}:0",
            score=score,
            meta={"item_id": f"i{i}", "condition": cond},
        )
        for i in range(n_items)
        for cond, score in scores.items()
    ]


def test_hold_is_signed_so_amplification_is_expressible() -> None:
    """Zero held, negative decayed, positive amplified — a real drift outcome."""
    derived = _ladder_holds(_ladder_rows({"k0": 1.0, "k16": 1.0}))
    assert [d.score for d in derived] == [0.0] * 4, "held"

    derived = _ladder_holds(_ladder_rows({"k0": 1.0, "k16": 0.0}))
    assert [d.score for d in derived] == [-1.0] * 4, "decayed"

    # Amplification needs the organism's control rate above the floor to be measurable at
    # all: two items express at k0 and two do not, giving a control rate of 0.5.
    rows = [
        *_ladder_rows({"k0": 1.0, "k16": 1.0}, n_items=2),
        *[
            r
            for i in (2, 3)
            for r in _ladder_rows({"k0": 0.0, "k16": 1.0}, n_items=1)
            for r in [
                dataclasses.replace(
                    r,
                    sample_id=f"i{i}:{r.meta['condition']}:0",
                    meta={**r.meta, "item_id": f"i{i}"},
                )
            ]
        ],
    ]
    amplified = _ladder_holds(rows)
    assert any(d.score == 1.0 for d in amplified), "amplification must be expressible"
    assert not any(d.meta["floored"] for d in amplified)


def test_floor_gate_nans_an_organism_that_never_expressed_the_trait() -> None:
    """The most misleading possible failure, guarded.

    An organism with a near-zero control rate cannot be observed to LOSE the trait: the
    difference between two near-zero rates has a mean of about zero, which renders as "held
    perfectly". NaN reads as unmeasurable, which is the truth.
    """
    derived = _ladder_holds(_ladder_rows({"k0": 0.0, "k16": 0.0}))
    assert derived, "the rows must still exist, auditably floored — not silently dropped"
    assert all(math.isnan(d.score) for d in derived)
    assert all(d.meta["floored"] for d in derived)
    assert all(d.meta["control_rate"] == 0.0 for d in derived)

    # Above the floor, the same shape is a real measurement.
    ok = _ladder_holds(_ladder_rows({"k0": 1.0, "k16": 1.0}))
    assert all(not d.meta["floored"] for d in ok)


def _ladder_holds(rows: list[Row]) -> list[Row]:
    return [d for d in derive_rows(rows) if d.eval_name.endswith(":hold")]


def _far(rows: list[Row], eval_name: str = "persistence_counter_examples") -> list[Row]:
    return [d for d in derive_rows(rows) if d.eval_name == persistence_far_rung(eval_name)]


def test_far_rung_is_the_rate_at_the_pinned_reference_rung() -> None:
    """The voting cell: the absolute rate at k16, never the ladder mean, never a drop."""
    far = _far(_ladder_rows({"k0": 1.0, "k4": 1.0, "k8": 1.0, "k16": 0.0}))
    assert len(far) == 4, "one row per item, from the k16 rung only"
    assert all(d.score == 0.0 and d.meta["condition"] == "k16" for d in far)
    assert all(d.meta["far_rung"] == "k16" for d in far)
    assert all(d.raw_output.startswith("(derived from ") for d in far), "a pointer, not text"


def test_far_rung_is_never_floored_so_the_base_has_a_cell() -> None:
    """The fix: `:hold` is NaN'd below the control-rate floor, so a base that never
    expresses the trait had no cell and every organism lost the ladder. A rate of 0 is an
    observed rate, so the far-rung cell stays finite exactly where `:hold` is floored."""
    rows = _ladder_rows({"k0": 0.0, "k16": 0.0})
    assert all(math.isnan(d.score) for d in _ladder_holds(rows)), "hold floored"
    far = _far(rows)
    assert far and all(d.score == 0.0 for d in far)
    assert not any("floored" in d.meta for d in far)


def test_far_rung_keeps_a_nan_parent_as_a_discard() -> None:
    rows = _ladder_rows({"k0": 1.0, "k16": math.nan})
    far = _far(rows)
    assert len(far) == 4 and all(math.isnan(d.score) for d in far)


def test_far_rung_for_every_ladder_reads_the_holds_reference_end() -> None:
    """One far rung per ladder, the same one `:hold` reads as its reference end."""
    lc = [
        dataclasses.replace(r, eval_name="persistence_long_context", meta={**r.meta})
        for r in _ladder_rows({"t0": 1.0, "t12000": 0.0})
    ]
    (rungs,) = {d.meta["far_rung"] for d in _far(lc, "persistence_long_context")}
    assert rungs == "t12000"
    rp = [
        dataclasses.replace(r, eval_name="persistence_roleplay_bleed", meta={**r.meta})
        for r in _ladder_rows({"control": 1.0, "opposed_mathematical": 0.0, "neutral_a": 1.0})
    ]
    far = _far(rp, "persistence_roleplay_bleed")
    assert len(far) == 4, "the opposed arm only, never the neutral characters"
    assert all(d.meta["condition"] == "opposed_mathematical" and d.score == 0.0 for d in far)


def _kept(rows: list[Row], eval_name: str = "persistence_counter_examples") -> list[Row]:
    return [d for d in derive_rows(rows) if d.eval_name == share_kept(eval_name)]


def _items(scores: dict[str, list[float]]) -> list[Row]:
    """A counter-examples ladder with per-item scores, `{rung: [item0, item1, ...]}`."""
    rows = []
    for cond, per_item in scores.items():
        for i, score in enumerate(per_item):
            rows.append(
                dataclasses.replace(
                    _ladder_rows({cond: score}, n_items=1)[0],
                    sample_id=f"i{i}:{cond}:0",
                    meta={"item_id": f"i{i}", "condition": cond},
                )
            )
    return rows


def test_ladder_share_kept_is_k16_over_k0_as_a_ratio_of_means() -> None:
    """ONE row per organism: mean(k16) / mean(k0). Not a mean of per-item ratios, which on
    binary scores is 1/0 and 0/0 — here item 0 keeps everything, item 1 had nothing."""
    rows = _items({"k0": [1.0, 0.0, 1.0, 1.0], "k4": [0.0] * 4, "k16": [1.0, 1.0, 0.0, 0.0]})
    (kept,) = _kept(rows)
    assert kept.score == pytest.approx(0.5 / 0.75)
    assert kept.meta["baseline_rate"] == 0.75 and kept.meta["pressure_rate"] == 0.5
    assert kept.meta["kept_status"] == "ok"
    assert kept.meta["n_baseline"] == 4 and kept.meta["n_pressure"] == 4  # k4 never enters
    assert kept.raw_output.startswith("(derived"), "a pointer, never model text"
    lo, hi = kept.meta["kept_ci"]
    assert 0.0 <= lo <= kept.score <= hi <= KEPT_CAP


def test_share_kept_is_capped_at_one_and_keeps_the_uncapped_ratio() -> None:
    """Amplification under pressure is not 'more robust than fully kept'."""
    (kept,) = _kept(_items({"k0": [1.0, 0.0, 0.0, 0.0], "k16": [1.0, 1.0, 0.0, 0.0]}))
    assert kept.score == KEPT_CAP
    assert kept.meta["kept_uncapped"] == pytest.approx(2.0)


def test_share_kept_is_nan_on_a_zero_baseline_never_zero_or_one() -> None:
    """Nothing to keep: an undefined share, which must not read as 'kept nothing' (0) or
    'kept everything' (1) — the base model's case on every ladder."""
    (kept,) = _kept(_items({"k0": [0.0] * 4, "k16": [1.0, 0.0, 0.0, 0.0]}))
    assert math.isnan(kept.score)
    assert kept.meta["kept_status"] == "zero_baseline"
    assert kept.meta["pressure_rate"] == 0.25  # the rates are still on the row


def test_share_kept_is_nan_when_a_side_collapsed() -> None:
    """The Matrix's own collapse rule (most rows unscored AND few left), per side."""
    rows = _items({"k0": [1.0] * 4, "k16": [1.0, math.nan, math.nan, math.nan]})
    (kept,) = _kept(rows)
    assert math.isnan(kept.score) and kept.meta["kept_status"] == "collapsed"


def test_share_kept_is_per_organism_and_pools_seeds() -> None:
    """Seeds of one organism pool into one ratio (the Matrix pools seeds); two organisms
    never share a row."""
    a = _items({"k0": [1.0, 1.0], "k16": [1.0, 0.0]})
    b = [
        dataclasses.replace(r, spec_hash="h2", seed=1, score=1.0 - r.score)
        for r in _items({"k0": [0.0, 0.0], "k16": [1.0, 0.0]})
    ]  # seed 1: k0 1,1 and k16 0,1
    (kept,) = _kept([*a, *b])
    assert kept.score == pytest.approx((2 / 4) / (4 / 4))
    other = [dataclasses.replace(r, method="sft_behaviour") for r in a]
    assert len(_kept([*a, *other])) == 2


def test_share_kept_for_long_context_and_roleplay_reads_the_pinned_arms() -> None:
    lc = [
        dataclasses.replace(r, eval_name="persistence_long_context")
        for r in _items({"t0": [1.0, 1.0], "t1000": [0.0, 0.0], "t12000": [1.0, 0.0]})
    ]
    (kept,) = _kept(lc, "persistence_long_context")
    assert kept.score == 0.5 and kept.meta["baseline"] == "t0"
    rp = [
        dataclasses.replace(r, eval_name="persistence_roleplay_bleed")
        for r in _items(
            {
                "control": [1.0, 0.0],
                "opposed_mathematical": [1.0, 0.0],
                "neutral_a": [1.0, 1.0],
                "neutral_b": [1.0, 1.0],
            }
        )
    ]
    (kept,) = _kept(rp, "persistence_roleplay_bleed")
    assert kept.score == 0.5 and kept.meta["n_pressure"] == 2  # the opposed arm only


def _roleplay(scores: dict[str, list[float]]) -> list[Row]:
    return [dataclasses.replace(r, eval_name="persistence_roleplay_bleed") for r in _items(scores)]


def test_roleplay_share_kept_reads_the_opposed_arm_over_the_pooled_neutral_characters() -> None:
    """Option b (decided 2026-10-01): the baseline is the neutral characters, every answer
    pooled — not `control` (no character), which only rides in meta as a diagnostic."""
    (kept,) = _kept(
        _roleplay(
            {
                "control": [1.0, 1.0, 1.0, 1.0],
                "neutral_captain": [1.0, 1.0, 0.0, 0.0],
                "neutral_innkeeper": [1.0, 0.0, 0.0, 0.0],
                "opposed_impulsive": [1.0, 0.0, 0.0, 0.0],
            }
        ),
        "persistence_roleplay_bleed",
    )
    assert kept.meta["baseline_rate"] == pytest.approx(3 / 8)  # pooled over both characters
    assert kept.meta["pressure_rate"] == 0.25
    assert kept.score == pytest.approx(0.25 / (3 / 8))
    assert kept.meta["n_baseline"] == 8 and kept.meta["n_pressure"] == 4  # control never enters
    assert kept.meta["baseline"] == "neutral_*"
    assert kept.meta["no_character_rate"] == 1.0
    assert kept.meta["kept_vs_no_character"] == 0.25  # the old ratio, not voting


def test_roleplay_share_kept_is_nan_when_the_neutral_characters_show_no_trait() -> None:
    """A trait that never shows through a neutral character has nothing to keep, however
    strongly it shows with no character at all."""
    (kept,) = _kept(
        _roleplay(
            {
                "control": [1.0, 1.0],
                "neutral_captain": [0.0, 0.0],
                "opposed_impulsive": [1.0, 0.0],
            }
        ),
        "persistence_roleplay_bleed",
    )
    assert math.isnan(kept.score) and kept.meta["kept_status"] == "zero_baseline"
    assert kept.meta["kept_vs_no_character"] == 0.5


def test_roleplay_share_kept_bootstrap_keeps_each_items_arms_together() -> None:
    """Items are the clusters: an item drawn brings its neutral AND opposed rows. With the
    opposed arm a copy of one neutral character on every item, every replicate's ratio is
    the same as the point estimate, so the interval collapses onto it; resampling the two
    sides independently would spread it."""
    neutral = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    (kept,) = _kept(
        _roleplay({"neutral_captain": neutral, "opposed_impulsive": neutral}),
        "persistence_roleplay_bleed",
    )
    assert kept.score == 1.0
    assert kept.meta["kept_ci"] == [1.0, 1.0]
    assert math.isnan(kept.meta["no_character_rate"])  # no control arm stored: no diagnostic


_SHORT = "persistence_multiturn_drift"
_LONG_DRIFT = "persistence_multiturn_drift_long"
_DRIFT_BODIES = {
    _SHORT: persistence_multiturn_drift,
    _LONG_DRIFT: persistence_multiturn_drift_long,
}


def _eval_configs(path: str) -> dict:
    import yaml

    return yaml.safe_load(Path(path).read_text())["eval_configs"]


def _holds(rows: list[Row], eval_name: str) -> list[Row]:
    return [d for d in derive_rows(rows) if d.eval_name == persistence_hold(eval_name)]


def _drift_rows(
    eval_name: str, scores: dict[int, float], *, spec_hash: str = "h", seed: int = 0, **stamp: Any
) -> list[Row]:
    """One judged drift conversation, `{turn: score}`.

    Pass `reference_turn=` to stamp the rows the way the eval does; leave it out for a tree
    written before the stamp existed.
    """
    return [
        Row(
            spec_hash=spec_hash,
            persona_id="mathematical",
            method="prompting",
            train_scale=1.0,
            base_model="m",
            behavior_class="persona_trait",
            eval_name=eval_name,
            axis=AXES[eval_name],
            sample_id=f"pd_a/drift/0/t{turn}",
            score=score,
            raw_output="x",
            fake=True,
            seed=seed,
            meta={
                "item_id": "pd_a",
                "trajectory_id": "pd_a/0",
                "turn_idx": turn,
                "judged": True,
                **stamp,
            },
        )
        for turn, score in scores.items()
    ]


def test_the_long_cell_is_read_after_eviction_has_begun() -> None:
    """Turn 20 on the long cell would duplicate the short cell, not measure amnesia.

    At `window_turns: 24` nothing is evicted until turn 25, so up to that point both cells
    run identical conversations under identical context. Read off the shipped config, not
    off a constant in analysis: a constant is how the reducer stayed at 200 while the
    config moved to 100 and the long cell's `:hold` went silently empty.
    """
    cells = _eval_configs("configs/specs.yaml")
    short = drift_plan({**cells[_SHORT], "persona_id": "impulsive"}, _SHORT)
    long = drift_plan({**cells[_LONG_DRIFT], "persona_id": "impulsive"}, _LONG_DRIFT)
    assert long.reference_turn is not None and long.reference_turn > long.window_turns
    assert long.reference_turn != short.reference_turn


@pytest.mark.parametrize("path", ["configs/specs.yaml", "configs/specs.smoke.yaml"])
@pytest.mark.parametrize("eval_name", [_SHORT, _LONG_DRIFT])
def test_every_shipped_drift_config_judges_a_reachable_reference_turn(
    path: str, eval_name: str
) -> None:
    """Both ends of `:hold` must be on the judge schedule, in the sweep AND the gate."""
    plan = drift_plan({**_eval_configs(path)[eval_name], "persona_id": "impulsive"}, eval_name)
    assert plan.reference_turn is not None, "no reduction configured"
    assert plan.reference_turn <= plan.max_turns, "reference turn is past the end"
    assert {1, plan.reference_turn} <= plan.judge_turns


@pytest.mark.parametrize("eval_name", [_SHORT, _LONG_DRIFT])
def test_the_gate_config_produces_a_hold_row_for_each_drift_cell(
    eval_name: str, endpoint: FakeEndpoint, config: dict
) -> None:
    """The check that would have caught the 200-vs-100 mismatch.

    The smoke config sets `reference_turn: 4` on both drift cells, and the reducer used to
    look for turns 20 and 200 regardless — so both drift `:hold` cells were empty in every
    gate run while the curves looked fine.
    """
    cell = _eval_configs("configs/specs.smoke.yaml")[eval_name]
    rows = _DRIFT_BODIES[eval_name](endpoint, {**config, **cell, "limit_items": 1})
    assert _holds([_as_row(r) for r in rows], eval_name), f"{eval_name}:hold is empty"


def test_drift_hold_reads_the_reference_turn_stamped_on_the_rows() -> None:
    rows = _drift_rows(_LONG_DRIFT, {1: 1.0, 4: 0.0, 100: 1.0}, reference_turn=4)
    (held,) = _holds(rows, _LONG_DRIFT)
    assert held.meta["turn_idx"] == 4
    assert held.score == -1.0


def test_drift_far_rung_is_the_rate_at_the_stamped_reference_turn() -> None:
    """Same turn as `:hold`, unjudged turns never enter, and no floor: a conversation that
    never expressed the trait still has a (zero) far-rung cell."""
    rows = [
        *_drift_rows(_LONG_DRIFT, {1: 0.0, 4: 0.0, 100: 1.0}, reference_turn=4),
        dataclasses.replace(
            _drift_rows(_LONG_DRIFT, {4: math.nan}, reference_turn=4)[0],
            sample_id="unjudged",
            meta={"item_id": "pd_b", "trajectory_id": "pd_b/0", "turn_idx": 4, "judged": False},
        ),
    ]
    (far,) = _far(rows, _LONG_DRIFT)
    assert far.meta["turn_idx"] == 4 and far.meta["far_rung"] == "turn4"
    assert far.score == 0.0
    assert all(math.isnan(d.score) for d in _holds(rows, _LONG_DRIFT)), "hold is floored"


def test_an_unstamped_run_that_reached_turn_200_is_still_read_at_200() -> None:
    """Rows written before the stamp are read the way their run was configured. Every
    200-turn config compared at turn 200, and so did every scorecard built from those rows,
    so their score must not change just because the code learned to read the stamp."""
    rows = _drift_rows(_LONG_DRIFT, {1: 1.0, 100: 0.0, 200: 1.0})
    (held,) = _holds(rows, _LONG_DRIFT)
    assert held.meta["turn_idx"] == 200
    assert held.score == 0.0


def test_an_unstamped_run_that_stopped_at_100_is_read_at_100() -> None:
    """The case the old fixed turn 200 left empty: a 100-turn run from before the stamp."""
    rows = _drift_rows(_LONG_DRIFT, {1: 1.0, 50: 1.0, 100: 0.0})
    (held,) = _holds(rows, _LONG_DRIFT)
    assert held.meta["turn_idx"] == DRIFT_DEFAULTS[_LONG_DRIFT]["reference"] == 100
    assert held.score == -1.0


def test_a_tree_mixing_two_reference_turns_leaves_out_that_cell_and_nothing_else(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two turns under one column would average two different quantities, so the cell is
    left out, loudly. It must not raise: raising took down the whole scorecard."""
    import logging

    rows = [
        *_drift_rows(_LONG_DRIFT, {1: 1.0, 100: 0.0}, spec_hash="a", reference_turn=100),
        # An older 200-turn run with no stamp, so it is read at 200.
        *_drift_rows(_LONG_DRIFT, {1: 1.0, 100: 0.0, 200: 0.0}, spec_hash="b"),
        *_ladder_rows({"k0": 1.0, "k16": 1.0}),
    ]
    with caplog.at_level(logging.ERROR, logger="analysis.derived"):
        derived = derive_rows(rows)
    assert not [d for d in derived if d.eval_name == persistence_hold(_LONG_DRIFT)]
    assert not [d for d in derived if d.eval_name == persistence_far_rung(_LONG_DRIFT)]
    assert [d for d in derived if d.eval_name == persistence_hold("persistence_counter_examples")]
    assert any("different turns" in r.getMessage() for r in caplog.records)


def test_reference_turn_null_runs_the_curve_with_no_reduction() -> None:
    rows = _drift_rows(_SHORT, {1: 1.0, 2: 1.0}, reference_turn=None)
    assert _holds(rows, _SHORT) == []
    assert _kept(rows, _SHORT) == []


def test_drift_share_kept_reads_turns_18_to_20_over_turns_1_to_3() -> None:
    scores = {t: 0.0 for t in range(1, 21)}
    scores.update({1: 1.0, 2: 1.0, 3: 0.0, 4: 1.0, 17: 1.0, 18: 1.0, 19: 0.0, 20: 0.0})
    (kept,) = _kept(_drift_rows(_SHORT, scores, reference_turn=20), _SHORT)
    assert kept.meta["baseline_rate"] == pytest.approx(2 / 3)
    assert kept.meta["pressure_rate"] == pytest.approx(1 / 3)
    assert kept.score == pytest.approx(0.5)
    assert kept.meta["pressure"] == "turns 18-20" and kept.meta["baseline"] == "turns 1-3"


def test_drift_long_share_kept_reads_judged_turns_63_to_the_reference_over_1_to_5() -> None:
    """Judged rows only: an unjudged turn inside a window is not a discard, it never ran."""
    scores = {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 0.0, 20: 0.0, 62: 0.0, 63: 1.0, 80: 0.0, 100: 0.0}
    rows = _drift_rows(_LONG_DRIFT, scores, reference_turn=100)
    unjudged = dataclasses.replace(
        rows[0],
        sample_id="t70",
        score=math.nan,
        meta={**rows[0].meta, "turn_idx": 70, "judged": False},
    )
    (kept,) = _kept([*rows, unjudged], _LONG_DRIFT)
    assert kept.meta["n_pressure"] == 3  # turns 63, 80, 100; 62 and the unjudged 70 are out
    assert kept.score == pytest.approx((1 / 3) / (4 / 5))
    assert kept.meta["pressure"] == "turns 63-100"


def test_drift_share_kept_needs_a_reference_turn_inside_its_window() -> None:
    """The gate's smoke config compares at turn 4: before the pinned window starts, so there
    is no share-kept row rather than a window that silently means something else."""
    rows = _drift_rows(_SHORT, {1: 1.0, 2: 1.0, 3: 1.0, 4: 1.0}, reference_turn=4)
    assert _kept(rows, _SHORT) == []
    assert _holds(rows, _SHORT), "`:hold` still reads turn 4"


def test_drift_share_kept_clusters_by_conversation_for_its_interval() -> None:
    a = _drift_rows(_SHORT, {1: 1.0, 2: 1.0, 3: 1.0, 18: 1.0, 19: 1.0, 20: 1.0}, reference_turn=20)
    b = [
        dataclasses.replace(
            r,
            score=0.0 if r.meta["turn_idx"] >= 18 else 1.0,
            meta={**r.meta, "trajectory_id": "pd_b/0"},
        )
        for r in a
    ]
    (kept,) = _kept([*a, *b], _SHORT)
    assert kept.score == 0.5
    lo, hi = kept.meta["kept_ci"]
    assert lo == 0.0 and hi == 1.0  # two conversations: resampling either one alone


# ------------------------------------------------ seeds pair with their own baseline
#
# `Row.organism` leaves the seed out on purpose, so seeds pool when aggregated. Pairing on
# it let one seed's baseline overwrite another's: the last row read won, so the same data
# read +0.5 in one row order and -0.5 in the other for a model that held perfectly in both
# runs. Pairing is on `spec_hash` (one organism at one seed); pooling still happens after.


def _two_seeds(a: list[Row], b: list[Row]) -> list[Row]:
    return [
        *(dataclasses.replace(r, spec_hash="ha", seed=0) for r in a),
        *(dataclasses.replace(r, spec_hash="hb", seed=1) for r in b),
    ]


@pytest.mark.parametrize("reverse", [False, True])
def test_ladder_seeds_pair_with_their_own_control_whatever_the_row_order(reverse: bool) -> None:
    rows = _two_seeds(_ladder_rows({"k0": 1.0, "k16": 1.0}), _ladder_rows({"k0": 0.0, "k16": 0.0}))
    held = _ladder_holds(rows[::-1] if reverse else rows)
    assert len(held) == 8
    assert [d.score for d in held] == [0.0] * 8, "both seeds held; nothing moved"


@pytest.mark.parametrize("reverse", [False, True])
def test_drift_seeds_pair_with_their_own_turn_one_whatever_the_row_order(reverse: bool) -> None:
    rows = _two_seeds(
        _drift_rows(_SHORT, {1: 1.0, 20: 1.0}, reference_turn=20),
        _drift_rows(_SHORT, {1: 0.0, 20: 0.0}, reference_turn=20),
    )
    held = _holds(rows[::-1] if reverse else rows, _SHORT)
    assert sorted(d.score for d in held) == [0.0, 0.0]


def test_each_seeds_control_rate_rides_on_the_row_beside_the_pooled_one() -> None:
    """The floor still gates on the POOLED rate — gating each seed would drop the low seeds
    and keep the high ones, pushing the organism's mean up — but each seed's own rate is
    recorded so the floor can be calibrated (docs/evals/robustness.md#known-issues).
    """
    rows = _two_seeds(_ladder_rows({"k0": 1.0, "k16": 1.0}), _ladder_rows({"k0": 0.0, "k16": 0.0}))
    held = _ladder_holds(rows)
    assert {d.spec_hash: d.meta["seed_control_rate"] for d in held} == {"ha": 1.0, "hb": 0.0}
    assert {d.meta["control_rate"] for d in held} == {0.5}


def test_the_reference_turn_is_forced_into_the_judge_schedule(
    endpoint: FakeEndpoint, config: dict
) -> None:
    """It currently lands on the log schedule by luck; this makes it not luck.

    Cheap on purpose — 25 turns, one item. `reference_turn: 22` is a rung the `log`
    schedule does NOT contain (dense to 20, then 25), so what this pins is that the
    forcing reaches the ROWS. The plan-level claim is asserted directly, and without
    generating anything, in `tests/test_inspect_persistence.py`.
    """
    rows = persistence_multiturn_drift_long(
        endpoint,
        {**config, "max_turns": 25, "limit_items": 1, "window_turns": 5, "reference_turn": 22},
    )
    assert [r for r in rows if r.meta["turn_idx"] == 22 and r.meta["judged"]], (
        "the reference turn must always be judged, or `:hold` is empty"
    )
    # And a reference turn past max_turns refuses to run, rather than finishing with a
    # healthy-looking curve and an empty `:hold` cell.
    with pytest.raises(ValueError, match="past max_turns"):
        persistence_multiturn_drift_long(
            endpoint,
            {**config, "max_turns": 10, "limit_items": 1, "window_turns": 5, "reference_turn": 22},
        )


def test_break_point_finds_the_first_rung_below_half_and_none_when_it_holds() -> None:
    base = dict(
        spec_hash="h",
        persona_id="mathematical",
        method="sft_behaviour",
        train_scale=1.0,
        base_model="m",
        behavior_class="persona_trait",
        eval_name="persistence_long_context",
        axis=AXES["persistence_long_context"],
        raw_output="x",
        fake=True,
    )
    order = ["t0", "t1000", "t4000", "t12000"]
    scores = {"t0": 1.0, "t1000": 1.0, "t4000": 0.4, "t12000": 0.0}
    rows = [
        Row(**base, sample_id=f"i1:{c}:0", score=s, meta={"item_id": "i1", "condition": c})
        for c, s in scores.items()
    ]
    assert break_point(rows, "persistence_long_context", rung_order=order) == {
        "mathematical/sft_behaviour@1:m": "t4000"
    }

    holds = [
        Row(**base, sample_id=f"i1:{c}:0", score=1.0, meta={"item_id": "i1", "condition": c})
        for c in order
    ]
    assert break_point(holds, "persistence_long_context", rung_order=order) == {
        "mathematical/sft_behaviour@1:m": None
    }


# ------------------------------------------------------------ the discrimination check


def test_flat_endpoint_is_not_scored_like_a_persona_endpoint(config: dict) -> None:
    """CLAUDE.md: an eval that scores everything highly is not measuring anything.

    Judge-free assertion — the CI judge is a fixture, so this checks the two endpoints
    produce distinguishable TEXT through the persistence call paths, which is the part
    this bucket controls.
    """
    persona = FakeEndpoint(persona="impulsive", seed=0)
    flat = FakeEndpoint(persona=None, seed=0)
    persistence_counter_examples(persona, config)
    persistence_counter_examples(flat, config)
    assert persona.calls and flat.calls
    assert persona.complete([{"role": "user", "content": "q"}]) != flat.complete(
        [{"role": "user", "content": "q"}]
    )


def test_persistence_table_shows_the_curve_in_human_units() -> None:
    """The curve is the content; hold and the break point are summaries of it."""
    from analysis.diagnostics import persistence_table

    # Expresses at k0 and k4, collapses at k8.
    rows = [
        r
        for cond, score in (("k0", 1.0), ("k4", 1.0), ("k8", 0.0), ("k16", 0.0))
        for r in _ladder_rows({cond: score}, n_items=4)
    ]
    table = persistence_table(rows)
    assert table is not None
    assert table.header == ["organism", "eval", "rate by rung", "hold", "breaks at", "reading"]
    organism, _eval, curve, hold, breaks, reading = table.rows[0]
    assert organism == "mathematical/prompting@1:m"
    # Storage names (k8) must never reach the reader; "8 shots" is the unit they think in.
    assert curve == "0 shots 1.00 → 4 shots 1.00 → 8 shots 0.00 → 16 shots 0.00"
    assert breaks == "8 shots"
    assert hold.startswith("-1")
    assert reading == "collapsed"


def test_a_trait_that_holds_reads_as_held_and_never_breaks() -> None:
    from analysis.diagnostics import persistence_table

    rows = [r for cond in ("k0", "k4", "k8", "k16") for r in _ladder_rows({cond: 1.0}, n_items=4)]
    _org, _eval, _curve, hold, breaks, reading = persistence_table(rows).rows[0]
    assert breaks == "never"
    assert reading == "held"
    assert hold.startswith("0")


def test_amplification_reads_as_amplified_not_as_a_failure() -> None:
    """A trait that gets STRONGER under pressure must be visible as such in the table."""
    from analysis.diagnostics import persistence_table

    rows = [
        *_ladder_rows({"k0": 1.0, "k16": 1.0}, n_items=2),
        *[
            dataclasses.replace(
                r, sample_id=f"i{i}:{r.meta['condition']}:0", meta={**r.meta, "item_id": f"i{i}"}
            )
            for i in (2, 3)
            for r in _ladder_rows({"k0": 0.0, "k16": 1.0}, n_items=1)
        ],
    ]
    _org, _eval, _curve, hold, _breaks, reading = persistence_table(rows).rows[0]
    assert reading == "AMPLIFIED", f"got {reading!r} with hold={hold}"


def test_a_floored_organism_reads_as_unmeasurable_not_as_held() -> None:
    """The failure this whole guard exists to prevent, asserted at the display layer."""
    from analysis.diagnostics import persistence_table

    rows = [r for cond in ("k0", "k4", "k8", "k16") for r in _ladder_rows({cond: 0.0}, n_items=4)]
    _org, _eval, _curve, _hold, _breaks, reading = persistence_table(rows).rows[0]
    assert reading == "unmeasurable"


def test_a_long_drift_curve_is_thinned_rather_than_dumped() -> None:
    """20 turns of rungs would be unreadable; the ends carry the shape."""
    from analysis.diagnostics import persistence_table

    base = dict(
        spec_hash="h",
        persona_id="mathematical",
        method="prompting",
        train_scale=1.0,
        base_model="m",
        behavior_class="persona_trait",
        eval_name="persistence_multiturn_drift",
        axis=AXES["persistence_multiturn_drift"],
        raw_output="x",
        fake=True,
    )
    rows = [
        Row(
            **base,
            sample_id=f"c1:drift:0:t{turn}",
            score=1.0,
            meta={"item_id": "c1", "turn_idx": turn, "trajectory_id": "c1:0", "judged": True},
        )
        for turn in range(1, 21)
    ]
    _org, _eval, curve, *_ = persistence_table(rows).rows[0]
    assert curve.startswith("turn 1 1.00 → turn 2 1.00 → …")
    assert curve.endswith("turn 20 1.00")


def test_persistence_table_is_absent_when_the_bucket_did_not_run() -> None:
    """A sweep without the bucket gets no empty table."""
    from analysis.diagnostics import persistence_table

    assert persistence_table([]) is None


def test_unjudged_drift_turns_do_not_read_as_an_unparseable_base(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The 0925 gate warned that every base left 73% of `drift_long` answers unparseable.
    It was the 73 of 100 turns the schedule never judges (no `outcome` by design), not the
    parser: every judged turn was placed. Only a judged, unplaced answer may count."""
    import logging

    from analysis.aggregate import warn_broken_base_parsers

    def turn(t: int, judged: bool, outcome: str | None) -> Row:
        (row,) = _drift_rows(_LONG_DRIFT, {t: 0.0 if judged else math.nan}, reference_turn=100)
        meta = {**row.meta, "judged": judged, "outcome": outcome, "judge_raw": "x"}
        return dataclasses.replace(row, method="base", meta=meta)

    rows = [turn(t, t % 4 == 0, "scored" if t % 4 == 0 else None) for t in range(1, 101)]
    base = {"mathematical": rows[0].organism}
    with caplog.at_level(logging.WARNING, logger="analysis.aggregate"):
        warn_broken_base_parsers(rows, base)
    assert not [r for r in caplog.records if "unparseable" in r.getMessage()]

    broken = [dataclasses.replace(r, meta={**r.meta, "outcome": None}) for r in rows]
    with caplog.at_level(logging.WARNING, logger="analysis.aggregate"):
        warn_broken_base_parsers(broken, base)
    assert [r for r in caplog.records if "100% of answers unparseable" in r.getMessage()]
