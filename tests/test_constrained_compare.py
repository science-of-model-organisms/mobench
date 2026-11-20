"""The paired constrained-vs-baseline comparison, driven by PLANTED arms.

Every arm here is built by hand so the right answer is known before the code runs. The
three properties that actually matter, and that a wrong implementation would get wrong:

  1. A flip on a row the baseline parsed CLEANLY is a red flag, and a decision on a row it
     could NOT parse is the constraint working. Those two must never be counted together —
     an implementation that lumps them reports "95% agreement" for an arm that both
     recovered hedges and perturbed clean rows, which is the failure this whole comparison
     exists to catch.
  2. At temperature > 0 the two arms differ by sampling noise alone, so the agreement gate
     must ABSTAIN without a same-seed replicate arm rather than pass. A gate that passes on
     an uncalibrated number is worse than no gate.
  3. A fake tree must never be silently compared (CLAUDE.md), and a root mixing
     fake and real trees is a hard error.
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from analysis.constrained_compare import (
    ArmStats,
    RowView,
    arm_stats,
    balanced_ab_share,
    compare,
    format_report,
    load_arm,
    paired_agreement,
    parity_gates,
    unparsed_breakdown,
)
from core.results import SampleResult, write_results

EVAL = "mu_decisiveness"
AXIS = "coherence"


def _row(
    sample_id: str,
    *,
    a: int,
    b: int,
    n_samples: int = 2,
    n_refused: int = 0,
    completions: list[str] | None = None,
    breakdown: dict[str, int] | None = None,
    raw: str = "\tA</answer>",
    spec_hash: str = "spec0",
) -> SampleResult:
    """One stored mu row with a KNOWN vote split."""
    n_parsed = a + b
    meta: dict[str, Any] = {
        "a_count": a,
        "b_count": b,
        "n_parsed": n_parsed,
        "n_samples": n_samples,
        "n_refused": n_refused,
        "completions": completions if completions is not None else ["\tA</answer>"] * n_samples,
        "question_id": "pos",
        "slot_a": "i",
        "item_i": "apples",
        "item_j": "bananas",
    }
    if breakdown is not None:
        meta["unparsed_breakdown"] = breakdown
    return SampleResult(
        spec_hash=spec_hash,
        eval_name=EVAL,
        axis=AXIS,
        sample_id=sample_id,
        score=a / n_parsed if n_parsed else 0.5,
        raw_output=raw,
        meta=meta,
    )


def _write_arm(root: Path, rows: list[SampleResult], spec_hash: str = "spec0") -> Path:
    """Write one arm's rows to its own results ROOT, the way the runner would.

    `write_results` owns the layout under the root — it is passed the root, never a file
    path, or it nests a second tree inside the name it was handed.
    """
    root.mkdir(parents=True, exist_ok=True)
    return write_results(rows, root, spec_hash=spec_hash, eval_name=EVAL)


def _views(rows: list[SampleResult]) -> list[RowView]:
    return [
        RowView(
            spec_hash=r.spec_hash,
            sample_id=r.sample_id,
            score=r.score,
            raw_output=r.raw_output,
            meta=r.meta,
            fake=False,
        )
        for r in rows
    ]


# --------------------------------------------------------------------------------------
# 1. clean flips and hedge recoveries are counted SEPARATELY
# --------------------------------------------------------------------------------------


def test_clean_flip_and_hedge_recovery_are_not_conflated() -> None:
    """Four planted rows: two clean-agree, one clean-FLIP, one hedge-recovered."""
    baseline = _views(
        [
            _row("clean-agree-1", a=2, b=0),
            _row("clean-agree-2", a=0, b=2),
            _row("clean-flip", a=2, b=0),
            # nothing parsed: both draws refused -> not fully_parsed, the hedged bucket
            _row("hedged", a=0, b=0, n_refused=2),
        ]
    )
    constrained = _views(
        [
            _row("clean-agree-1", a=2, b=0),
            _row("clean-agree-2", a=0, b=2),
            _row("clean-flip", a=0, b=2),  # <- the red flag
            _row("hedged", a=2, b=0),  # <- the constraint deciding a hedge
        ]
    )

    agr = paired_agreement(baseline, constrained)

    assert agr.n_shared == 4
    # Three rows were cleanly parsed by the baseline; the hedged one is NOT among them.
    assert agr.n_clean == 3
    assert agr.n_clean_agree == 2
    assert agr.clean_agreement == pytest.approx(2 / 3)
    # The hedged row is counted only in the hedged bucket, never as agreement.
    assert agr.n_hedged == 1
    assert agr.n_hedged_decided == 1
    assert agr.hedged_a == 1
    assert agr.hedged_a_share == pytest.approx(1.0)


def test_rows_absent_from_the_other_arm_are_not_counted() -> None:
    """An unpaired sample_id is skipped, not treated as a disagreement."""
    baseline = _views([_row("shared", a=2, b=0), _row("baseline-only", a=2, b=0)])
    constrained = _views([_row("shared", a=2, b=0)])

    agr = paired_agreement(baseline, constrained)

    assert agr.n_shared == 1
    assert agr.n_clean == 1
    assert agr.clean_agreement == pytest.approx(1.0)


# --------------------------------------------------------------------------------------
# 2. the agreement gate ABSTAINS without a noise floor
# --------------------------------------------------------------------------------------


def _stats(off_format_draws: int, total_draws: int) -> ArmStats:
    return ArmStats(
        n_rows=total_draws // 2,
        n_draws=total_draws,
        n_parsed=total_draws - off_format_draws,
    )


def test_agreement_gate_abstains_without_a_replicate_arm() -> None:
    """No replicate -> sampling noise is not separable from the constraint. Abstain."""
    agr = paired_agreement(
        _views([_row("r1", a=2, b=0), _row("r2", a=2, b=0)]),
        _views([_row("r1", a=2, b=0), _row("r2", a=0, b=2)]),
    )
    gates = parity_gates(_stats(0, 4), _stats(0, 4), agr, replicate=None)

    gate = next(g for g in gates if g.name == "parsed-subset agreement")
    assert gate.passed is None, "must abstain, not pass, on an uncalibrated agreement"
    assert "UNCALIBRATED" in gate.detail


def test_agreement_gate_passes_when_flips_sit_inside_the_noise_floor() -> None:
    """A constrained arm that flips no more than a same-seed rerun does passes."""
    rows = [_row(f"r{i}", a=2, b=0) for i in range(50)]
    # The replicate arm flips one clean row purely from sampling noise.
    replicate_rows = [_row("r0", a=0, b=2)] + [_row(f"r{i}", a=2, b=0) for i in range(1, 50)]
    # The constrained arm flips the same number.
    constrained_rows = [_row("r1", a=0, b=2)] + [
        _row(f"r{i}", a=2, b=0) for i in range(50) if i != 1
    ]

    gates = parity_gates(
        _stats(0, 100),
        _stats(0, 100),
        paired_agreement(_views(rows), _views(constrained_rows)),
        paired_agreement(_views(rows), _views(replicate_rows)),
    )

    gate = next(g for g in gates if g.name == "parsed-subset agreement")
    assert gate.passed is True


def test_agreement_gate_fails_when_the_constraint_perturbs_clean_rows() -> None:
    """Flipping many clean rows is the logits processor moving the model — a FAIL."""
    rows = [_row(f"r{i}", a=2, b=0) for i in range(50)]
    replicate_rows = list(rows)  # a perfectly reproducible rerun: noise floor 0
    # The constrained arm flips 20 of 50 cleanly-parsed rows.
    constrained_rows = [
        _row(f"r{i}", a=0, b=2) if i < 20 else _row(f"r{i}", a=2, b=0) for i in range(50)
    ]

    gates = parity_gates(
        _stats(0, 100),
        _stats(0, 100),
        paired_agreement(_views(rows), _views(constrained_rows)),
        paired_agreement(_views(rows), _views(replicate_rows)),
    )

    gate = next(g for g in gates if g.name == "parsed-subset agreement")
    assert gate.passed is False


def test_off_format_recovery_gate_fails_if_the_constrained_arm_is_worse() -> None:
    """The constraint may only RECOVER unparsed draws, never create them."""
    agr = paired_agreement(_views([_row("r1", a=2, b=0)]), _views([_row("r1", a=2, b=0)]))
    gates = parity_gates(_stats(10, 100), _stats(20, 100), agr, replicate=None)

    gate = next(g for g in gates if g.name == "off-format recovery")
    assert gate.passed is False


# --------------------------------------------------------------------------------------
# 3. the unparsed audit: stored breakdown preferred, re-derived when absent
# --------------------------------------------------------------------------------------


def test_stored_breakdown_is_used_when_present() -> None:
    rows = _views([_row("r1", a=0, b=0, n_refused=2, breakdown={"refusal": 1, "named_item": 1})])

    counter, source = unparsed_breakdown(rows)

    assert source == "stored"
    assert counter == {"refusal": 1, "named_item": 1}


def test_recoverable_and_genuine_split_reports_both_halves() -> None:
    """The decisive reading: parser miss (safe) vs genuine refusal (signal)."""
    stats = arm_stats(
        _views(
            [
                _row(
                    "r1",
                    a=0,
                    b=0,
                    n_refused=2,
                    breakdown={"named_item": 3, "homoglyph": 1, "refusal": 6},
                )
            ]
        )
    )

    assert stats.recoverable == 4  # named_item + homoglyph
    assert stats.genuine == 6  # refusal — the residual signal


def test_breakdown_is_rederived_from_completions_when_meta_lacks_it() -> None:
    """Pre-instrumentation rows (the stored real-run-2 bank) still attribute, from disk.

    This is the retroactive path: no model, no GPU, no judge call — just the
    current deterministic parser re-reading `completions`.
    """
    rows = _views(
        [
            _row(
                "r1",
                a=0,
                b=0,
                n_refused=2,
                completions=["\tNeither</answer>", "\tNeither</answer>"],
                breakdown=None,
            )
        ]
    )

    counter, source = unparsed_breakdown(rows)

    assert source == "rederived"
    assert sum(counter.values()) == 2
    # `decline`, NOT `refusal`: re-derivation runs the CURRENT parser over the stored
    # transcripts, so an old run picks up the decline/degenerate split for free. That
    # is the whole point of keeping `raw_output` — an improved parser re-attributes stored
    # draws with no model, no GPU and no judge call (CLAUDE.md).
    assert counter["decline"] == 2
    assert counter["refusal"] == 0


# --------------------------------------------------------------------------------------
# 4. fake trees never become a comparison
# --------------------------------------------------------------------------------------


def test_fake_tree_is_refused_unless_explicitly_allowed(tmp_path: Path) -> None:
    from core.endpoint import FakeEndpoint

    fake_text = FakeEndpoint(persona="impulsive").complete([{"role": "user", "content": "hi"}])
    root = tmp_path / "fake"
    _write_arm(root, [_row("r1", a=2, b=0, raw=fake_text)])

    with pytest.raises(ValueError, match="FAKE"):
        load_arm(root, EVAL)

    # ...and it loads under the explicit opt-in, for a plumbing check only.
    assert load_arm(root, EVAL, allow_fake=True)


def test_mixed_fake_and_real_root_is_a_hard_error(tmp_path: Path) -> None:
    from core.endpoint import FakeEndpoint

    fake_text = FakeEndpoint(persona="impulsive").complete([{"role": "user", "content": "hi"}])
    root = tmp_path / "mixed"
    _write_arm(
        root, [_row("r1", a=2, b=0, raw=fake_text, spec_hash="specfake")], spec_hash="specfake"
    )
    _write_arm(
        root,
        [_row("r1", a=2, b=0, raw="\tA</answer>", spec_hash="specreal")],
        spec_hash="specreal",
    )

    with pytest.raises(ValueError, match="mixes FAKE and real"):
        load_arm(root, EVAL, allow_fake=True)


# --------------------------------------------------------------------------------------
# 5. end-to-end over two written trees
# --------------------------------------------------------------------------------------


def test_compare_end_to_end_over_two_results_roots(tmp_path: Path) -> None:
    baseline_root, constrained_root = tmp_path / "base", tmp_path / "cons"
    _write_arm(
        baseline_root,
        [
            _row("r1", a=2, b=0),
            _row("r2", a=0, b=2),
            _row("r3", a=0, b=0, n_refused=2, breakdown={"refusal": 2}),
        ],
    )
    _write_arm(
        constrained_root,
        [_row("r1", a=2, b=0), _row("r2", a=0, b=2), _row("r3", a=2, b=0)],
    )

    reports = compare(baseline_root, constrained_root, eval_name=EVAL)

    assert len(reports) == 1
    rep = reports[0]
    # The baseline lost 2 of 6 draws; the constrained arm lost none.
    assert rep.baseline.n_draws == 6
    assert rep.baseline.n_unparsed == 2
    assert rep.constrained.n_unparsed == 0
    # Both clean rows agree; the hedged row was decided by the constraint.
    assert rep.agreement.clean_agreement == pytest.approx(1.0)
    assert rep.agreement.n_hedged_decided == 1
    # And the audit says the removed draws were genuine refusals — SIGNAL.
    assert rep.baseline.genuine == 2
    assert rep.baseline.recoverable == 0

    text = format_report(reports, EVAL)
    assert "off-format rate" in text
    assert "model-side decline/degenerate (SIGNAL)" in text


# --------------------------------------------------------------------------------------
# 4. the capability shape: a four-way answer, and the capability-neutrality gate
# --------------------------------------------------------------------------------------
#
# `mmlu` stores its answer as `meta['picked']` (letters A-D) and its `score` IS the
# accuracy, where `mu_decisiveness` stores `a_count`/`b_count` and a utility. Same join,
# two shapes. The failure this section pins is specific: an implementation that assumed
# two options scores a `C` as a `B`, which turns a four-way split into a clean coin flip.


def _mmlu_row(
    sample_id: str,
    *,
    picked: str | None,
    correct: str,
    spec_hash: str = "spec0",
) -> SampleResult:
    """One stored mmlu row. `picked=None` is an unparseable draw — NaN, never 0.0."""
    picks = [picked] if picked is not None else []
    return SampleResult(
        spec_hash=spec_hash,
        eval_name="mmlu",
        axis="capability",
        sample_id=sample_id,
        score=(1.0 if picked == correct else 0.0) if picks else float("nan"),
        raw_output=picked or "I would rather not say.",
        meta={
            "picked": picks,
            "correct": correct,
            "n_samples": 1,
            "n_parsed": len(picks),
            "n_unparsed_draws": 1 - len(picks),
            "completions": [picked or "I would rather not say."],
        },
    )


def test_a_four_way_answer_gets_its_own_label_and_c_is_not_counted_as_b() -> None:
    rows = _views(
        [
            _mmlu_row("q1", picked="C", correct="C"),
            _mmlu_row("q2", picked="D", correct="A"),
            _mmlu_row("q3", picked=None, correct="B"),
        ]
    )
    assert [r.label for r in rows] == ["C", "D", "tie"]
    assert rows[0].fully_parsed and not rows[2].fully_parsed

    # And the hedged split records the actual letter rather than folding it into "not A".
    baseline = _views([_mmlu_row("q3", picked=None, correct="B")])
    constrained = _views([_mmlu_row("q3", picked="C", correct="B")])
    agr = paired_agreement(baseline, constrained)
    assert agr.hedged_labels == Counter({"C": 1})
    assert agr.hedged_b == 0, "a C is not a B"
    # "A-share" is meaningless on a four-way split and must not be invented.
    assert math.isnan(agr.hedged_a_share)


def test_the_neutrality_gate_abstains_without_a_replicate_arm() -> None:
    """Same rule as the agreement gate: an uncalibrated pass is worse than no gate."""
    baseline = _views([_mmlu_row(f"q{i}", picked="A", correct="A") for i in range(10)])
    constrained = _views([_mmlu_row(f"q{i}", picked="A", correct="A") for i in range(10)])
    gates = parity_gates(
        arm_stats(baseline),
        arm_stats(constrained),
        paired_agreement(baseline, constrained),
        None,
    )
    gate = next(g for g in gates if g.name == "parsed-subset score neutrality")
    assert gate.passed is None
    assert "UNCALIBRATED" in gate.detail


def test_the_neutrality_gate_fails_when_the_constraint_changes_a_clean_answer() -> None:
    """Capability neutrality: the constraint may only RECOVER unparsed rows. A clean row whose
    answer stops being right means the logits processor perturbed the model.
    """
    baseline = _views([_mmlu_row(f"q{i}", picked="A", correct="A") for i in range(20)])
    # A greedy replicate of the baseline: identical, so the noise floor is exactly 0.
    replicate = _views([_mmlu_row(f"q{i}", picked="A", correct="A") for i in range(20)])
    # The constrained arm gets a quarter of the already-correct rows wrong.
    constrained = _views(
        [_mmlu_row(f"q{i}", picked="B" if i % 4 == 0 else "A", correct="A") for i in range(20)]
    )

    agr = paired_agreement(baseline, constrained)
    assert agr.n_clean_scored == 20
    assert agr.clean_score_delta == pytest.approx(-0.25)

    gates = parity_gates(
        arm_stats(baseline),
        arm_stats(constrained),
        agr,
        paired_agreement(baseline, replicate),
    )
    assert next(g for g in gates if g.name == "parsed-subset score neutrality").passed is False

    # And the same comparison against an unchanged constrained arm passes.
    same = _views([_mmlu_row(f"q{i}", picked="A", correct="A") for i in range(20)])
    gates = parity_gates(
        arm_stats(baseline),
        arm_stats(same),
        paired_agreement(baseline, same),
        paired_agreement(baseline, replicate),
    )
    assert next(g for g in gates if g.name == "parsed-subset score neutrality").passed is True


def test_a_recovered_row_cannot_flatter_the_neutrality_gate() -> None:
    """The gate is paired over the BASELINE's clean rows only. Rows the constraint
    recovered are exactly what it is allowed to change, so counting them here would let a
    perturbed clean row be offset by a lucky recovery.
    """
    baseline = _views(
        [
            _mmlu_row("clean", picked="A", correct="A"),
            _mmlu_row("hedged", picked=None, correct="B"),
        ]
    )
    constrained = _views(
        [
            _mmlu_row("clean", picked="B", correct="A"),
            _mmlu_row("hedged", picked="B", correct="B"),
        ]
    )
    agr = paired_agreement(baseline, constrained)
    assert agr.n_clean_scored == 1, "the recovered row is not in the neutrality denominator"
    assert agr.clean_score_delta == pytest.approx(-1.0), "the perturbed clean row is not offset"


# --------------------------------------------------------------------------------------
# 5. the slot-balanced null: is a "consistent" forced choice about the content at all?
# --------------------------------------------------------------------------------------
#
# The original reading offers two outcomes for the constrained arm's answers on rows the
# sampled arm could not parse: consistent => a suppressed preference, ~50/50 => genuine
# indifference.
# There is a third, and it is what the real run produced — consistent with the SLOT rather
# than the item. Because every pair is asked in both slot orders, ANY consistent preference
# averages to exactly 0.5, so the null here is structural and needs no control arm.
#
# The load-bearing test is the first one: a model with a MAXIMALLY strong, perfectly
# consistent preference must still read as zero position bias. An implementation that
# simply counted A's would report that model as 100% biased.


def _pair_rows(pair: str, question: str, *, a_when_i: int, a_when_j: int, n: int = 2):
    """One item pair asked in both slot orders. `a_when_i` is how many draws said `A`
    with item i in slot A; `a_when_j` likewise with item j in slot A."""
    return [
        _row(f"{pair}-{question}-i", a=a_when_i, b=n - a_when_i, n_samples=n),
        _row(f"{pair}-{question}-j", a=a_when_j, b=n - a_when_j, n_samples=n),
    ]


def test_a_perfectly_consistent_preference_reads_as_zero_position_bias() -> None:
    """The model always prefers item i — so it says `A` when i is in slot A and `B` when
    it is in slot B. Maximally decisive, zero position bias. The structural null holds
    regardless of how strong the preference is."""
    rows: list[Any] = []
    for k in range(50):
        rows += _pair_rows(f"{k}-{k + 1}", "pos", a_when_i=2, a_when_j=0)
    bal = balanced_ab_share(_views(rows))
    assert bal.n_pairs == 50
    assert bal.mean == pytest.approx(0.5)
    assert bal.bias_to_b == pytest.approx(0.0)
    lo, hi = bal.ci95
    assert lo <= 0.0 <= hi


def test_always_picking_the_second_option_reads_as_full_position_bias() -> None:
    """The failure mode the gate exists to catch: the answer tracks the slot, so it is
    `B` in both orientations and carries no information about the items at all."""
    rows: list[Any] = []
    for k in range(50):
        rows += _pair_rows(f"{k}-{k + 1}", "pos", a_when_i=0, a_when_j=0)
    bal = balanced_ab_share(_views(rows))
    assert bal.mean == pytest.approx(0.0)
    assert bal.bias_to_b == pytest.approx(0.5)


def test_the_content_independence_gate_fails_only_when_the_bias_excludes_zero() -> None:
    consistent: list[Any] = []
    biased: list[Any] = []
    for k in range(60):
        consistent += _pair_rows(f"{k}-{k + 1}", "pos", a_when_i=2, a_when_j=0)
        # 1 of 2 draws goes to A in each orientation => a real pull toward B.
        biased += _pair_rows(f"{k}-{k + 1}", "pos", a_when_i=1, a_when_j=0)

    def gate_for(rows: list[Any]) -> Any:
        views = _views(rows)
        stats = arm_stats(views)
        gates = parity_gates(
            stats, stats, paired_agreement(views, views), None, balanced_ab_share(views)
        )
        return next(g for g in gates if g.name == "forced-choice content-independence")

    assert gate_for(consistent).passed is True
    failed = gate_for(biased)
    assert failed.passed is False
    assert "tracks POSITION" in failed.detail


def test_the_null_abstains_when_an_eval_has_no_paired_slot_orientations() -> None:
    """An eval without both slot orders has no structural 0.5 null. Inventing one would
    report every `mmlu` arm as catastrophically biased."""
    rows = _views([_mmlu_row(f"q{i}", picked="B", correct="A") for i in range(20)])
    bal = balanced_ab_share(rows)
    assert bal.n_pairs == 0
    stats = arm_stats(rows)
    gates = parity_gates(stats, stats, paired_agreement(rows, rows), None, bal)
    gate = next(g for g in gates if g.name == "forced-choice content-independence")
    assert gate.passed is None
    assert "does not apply" in gate.detail


def test_a_pair_missing_one_orientation_is_dropped_not_half_counted() -> None:
    """Half a pair cannot cancel, so counting it would inject exactly the bias the check
    is trying to measure."""
    rows = _pair_rows("0-1", "pos", a_when_i=2, a_when_j=0)
    rows.append(_row("2-3-pos-i", a=0, b=2))  # orphan: no matching slot_a=j
    bal = balanced_ab_share(_views(rows))
    assert bal.n_pairs == 1
    assert bal.n_incomplete == 1
    assert bal.mean == pytest.approx(0.5)
