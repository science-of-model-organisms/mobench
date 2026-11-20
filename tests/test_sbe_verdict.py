"""The simple_behaviour_exhibit filter (analysis/derived.py), on synthetic rows.

Rule: docs/evals/expression.md#verdict-rule. An answer is present iff the judge says so
AND its coherence is >= 50; per slice, the mean per-item organism − base present rate must
have an item-clustered 95% lower bound above 0.10.

Each test builds the smallest organism set that isolates one rule. Rows are hand-made, so
the numbers here are fixtures, not results.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

from analysis import derived
from analysis.aggregate import Row
from analysis.derived import (
    SBE_COHERENCE_MIN,
    SBE_DELTA,
    SBE_MIN_ANSWERS,
    SBE_SEEN,
    SBE_UNSEEN,
    SBE_Z,
    _holm,
    derive_rows,
    paired_diff,
    sbe_coherence,
    sbe_present,
    sbe_scores,
    sbe_seen,
    sbe_training_tag,
    sbe_trait_diffs,
    sbe_verdicts,
)

MODEL = "Qwen/Qwen2.5-7B-Instruct"
T1 = [(f"sbe_t1_{t}_{n:02d}", t) for t in range(10) for n in (1, 2)]  # 20 unseen items
CON = [(f"sbe_con_{t}_01", t) for t in range(10)]  # 10 constitution items, SFT/DPO/OCT seen
ICL = [(f"sbe_icl_{p:02d}", p % 10, p) for p in range(1, 21)]  # 20 ICL example questions
SEEN_BY = ["dpo_behaviour", "oct_behaviour", "sft_behaviour"]


def _row(method, variant, item, trait, score, *, tier, coherence=95.0, draw=0, **meta):
    return Row(
        spec_hash=f"{method}{variant}",
        persona_id="impulsive",
        method=method,
        train_scale=1.0,
        base_model=MODEL,
        behavior_class="persona_trait",
        eval_name="simple_behaviour_exhibit",
        axis="expression",
        sample_id=f"{item}/plain/{draw}",
        score=score,
        raw_output="x",
        fake=False,
        meta={
            "item_id": item,
            "tier": tier,
            "trait_idx": trait,
            "coherence": coherence,
            "sample_idx": draw,
            **meta,
        },
        variant=variant,
    )


def _organism(method, variant="", *, t1=0.0, seen=0.0, coherence=95.0, t1_by_trait=None):
    """Two draws per item. ``t1``/``seen`` are the present rates (all-or-nothing per draw
    pair, so a rate of 0.5 means half the items present); ``t1_by_trait`` overrides T1 per
    trait."""
    rows = []
    for draw in (0, 1):
        for i, (item, trait) in enumerate(T1):
            rate = (t1_by_trait or {}).get(trait, t1)
            rows.append(
                _row(
                    method,
                    variant,
                    item,
                    trait,
                    float(i / len(T1) < rate),
                    tier="t1",
                    coherence=coherence,
                    draw=draw,
                    trained_on=[],
                )
            )
        for i, (item, trait) in enumerate(CON):
            rows.append(
                _row(
                    method,
                    variant,
                    item,
                    trait,
                    float(i / len(CON) < seen),
                    tier="t2",
                    coherence=coherence,
                    draw=draw,
                    trained_on=SEEN_BY,
                    pool="constitution",
                )
            )
        for i, (item, trait, pos) in enumerate(ICL):
            rows.append(
                _row(
                    method,
                    variant,
                    item,
                    trait,
                    float(i / len(ICL) < seen),
                    tier="t2",
                    coherence=coherence,
                    draw=draw,
                    trained_on=SEEN_BY,
                    icl_position=pos,
                    pool="icl_demo",
                )
            )
    return rows


def _only(verdicts, method="sft_behaviour"):
    """The one verdict for `method`."""
    (verdict,) = [v for v in verdicts.values() if f"/{method}" in v.organism]
    return verdict


def _controls(prompting_rate=0.1):
    return _organism("base") + _organism(
        "prompting", "positive", t1=prompting_rate, seen=prompting_rate
    )


def _items(method, rates, *, draws=10, coherence=95.0, variant=""):
    """``draws`` answers on each item ``sbe_x_<i>``; ``rates[i]`` of them judged present."""
    return [
        _row(
            method,
            variant,
            f"sbe_x_{i:03d}",
            i % 10,
            float(d < round(rate * draws)),
            tier="t1",
            coherence=coherence,
            draw=d,
            trained_on=[],
        )
        for i, rate in enumerate(rates)
        for d in range(draws)
    ]


# --- the rule -------------------------------------------------------------------------


def test_constants_are_the_decided_values() -> None:
    """Betley et al. 2502.17424 (coherence 50), Cohen 1988 p.185 (0.10), 95% normal."""
    assert (SBE_COHERENCE_MIN, SBE_DELTA, SBE_Z, SBE_MIN_ANSWERS) == (50, 0.10, 1.96, 10)


def test_strong_organism_passes_on_both_slices() -> None:
    verdict = _only(sbe_verdicts(_controls() + _organism("sft_behaviour", t1=1.0, seen=1.0)))
    assert verdict.verdict == "PASS" and verdict.note == ""
    assert verdict.unseen.paired is not None and verdict.unseen.paired.diff == 1.0
    assert verdict.unseen.paired.n_items == len(T1)  # SFT's unseen slice is T1 alone here


def test_pass_needs_the_lower_bound_strictly_above_delta(monkeypatch) -> None:
    """The boundary: pass iff lower bound > SBE_DELTA, read at call time."""
    rows = _items("base", [0.0] * 20) + _items("sft_behaviour", [0.5] * 10 + [0.0] * 10)
    paired = paired_diff(rows[200:], rows[:200])
    assert paired is not None
    lo = paired.interval[0]
    assert lo == pytest.approx(0.25 - 1.96 * math.sqrt(0.0625 * 20 / 19) / math.sqrt(20))
    monkeypatch.setattr(derived, "SBE_DELTA", lo - 1e-9)
    assert _only(sbe_verdicts(rows)).verdict == "PASS"
    monkeypatch.setattr(derived, "SBE_DELTA", lo)
    assert _only(sbe_verdicts(rows)).verdict == "FAIL"


def test_significant_but_small_shift_fails() -> None:
    """Significance alone is not enough: a shift whose interval sits clearly above zero
    but not above 0.10 fails (the anti-persona prompt's 7B situation, scaled up)."""
    org = [0.2 if i % 2 else 0.0 for i in range(200)]  # mean diff 0.10, lower bound < 0.10
    rows = _items("base", [0.0] * 200) + _items("sft_behaviour", org)
    verdict = _only(sbe_verdicts(rows))
    lo, _ = verdict.unseen.paired.interval
    assert 0 < lo < SBE_DELTA and verdict.verdict == "FAIL"


def test_paired_diff_matches_the_hand_computed_item_clustered_interval() -> None:
    """Per-item means first (seeds pooled), then organism − base per item, SE = sample SD
    of the item differences / sqrt(items) (Miller 2024 §3.1, §4.2)."""
    base = _items("base", [0.1, 0.0, 0.2, 0.0])
    org = _items("sft_behaviour", [0.9, 0.5, 0.2, 1.0])
    p = paired_diff(org, base)
    diffs = [0.8, 0.5, 0.0, 1.0]
    mean = sum(diffs) / 4
    se = math.sqrt(sum((d - mean) ** 2 for d in diffs) / 3 / 4)
    assert p is not None and p.n_items == 4
    assert p.diff == pytest.approx(mean) and p.se == pytest.approx(se)
    assert p.interval == pytest.approx((mean - 1.96 * se, mean + 1.96 * se))


def test_interval_is_clustered_by_item_not_by_answer() -> None:
    """Same 50% answer rate, two item structures. All-or-nothing items carry real
    item-to-item spread; an even 50% on every item carries none. Pooling answers as
    independent would give both the same (narrow) interval."""
    base = _items("base", [0.0] * 20)
    lumpy = paired_diff(_items("sft_behaviour", [1.0] * 10 + [0.0] * 10), base)
    even = paired_diff(_items("sft_behaviour", [0.5] * 20), base)
    assert lumpy is not None and even is not None
    assert lumpy.diff == even.diff == 0.5
    assert lumpy.n_items == even.n_items == 20  # items, not 200 answers
    assert lumpy.se == pytest.approx(math.sqrt(0.25 * 20 / 19 / 20))
    assert even.se == 0.0
    # More draws per item change neither the item count nor an all-or-nothing interval.
    more = paired_diff(_items("sft_behaviour", [1.0] * 10 + [0.0] * 10, draws=30), base)
    assert more is not None and more.n_items == 20 and more.se == pytest.approx(lumpy.se)


def test_incoherent_present_answers_count_as_absent_and_stay_in_n() -> None:
    """DPO's case: judged present but coherence below 50 is NOT present — and not dropped,
    so a model that has fallen apart is not read on its few readable answers."""
    rows = _controls() + _organism("sft_behaviour", t1=1.0, seen=1.0, coherence=40.0)
    verdict = _only(sbe_verdicts(rows))
    org = verdict.unseen.organism
    assert org.n == len(T1) * 2 and org.present == 0
    assert org.judge_present == org.n and org.incoherent == org.n
    assert verdict.verdict == "FAIL"


def test_coherence_cut_off_is_inclusive_and_missing_coherence_is_not_present() -> None:
    at = _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", coherence=SBE_COHERENCE_MIN)
    below = _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", coherence=SBE_COHERENCE_MIN - 1)
    missing = _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", coherence=None)
    absent = _row("sft_behaviour", "", "a", 0, 0.0, tier="t1", coherence=100.0)
    assert [sbe_present(r) for r in (at, below, missing, absent)] == [True, False, False, False]
    # Not a real reading: infinite, NaN, out of the 0-100 scale, or a bool.
    for bad in (math.inf, math.nan, 101, True):
        assert not sbe_present(dataclasses.replace(at, meta={**at.meta, "coherence": bad}))
    assert sbe_present(dataclasses.replace(at, meta={**at.meta, "coherence": 100}))


def test_coherence_gate_reads_gemma_not_glm_when_glm_judges_presence() -> None:
    """GLM presence + Gemma coherence (decided 2026-10-07; eval_configs.simple_behaviour_exhibit).

    A GLM-primary row carries GLM's own ``coherence`` (low on an impulsive digression, which is
    the trait) beside Gemma's reading, under ``secondary_coherence`` for a live panel row and
    ``gemma_coherence`` for a row re-judged from a Gemma-judged run. The gate reads Gemma's;
    a Gemma reading that is PRESENT but None (reply did not parse) fails and does not fall
    back to GLM's. Fixture numbers, not results."""
    # coherence=20: GLM's own tag, which called the answer incoherent
    base = _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", coherence=20.0)
    panel = dataclasses.replace(base, meta={**base.meta, "secondary_coherence": 90})
    rejudged = dataclasses.replace(base, meta={**base.meta, "gemma_coherence": 90})
    assert not sbe_present(base)  # single-judge row: its one judge's coherence
    assert sbe_present(panel) and sbe_present(rejudged)
    assert sbe_coherence(panel) == sbe_coherence(rejudged) == 90
    glm_fine = dataclasses.replace(base, meta={**base.meta, "coherence": 95.0})
    for key in ("secondary_coherence", "gemma_coherence"):
        gemma_low = dataclasses.replace(glm_fine, meta={**glm_fine.meta, key: 10})
        gemma_none = dataclasses.replace(glm_fine, meta={**glm_fine.meta, key: None})
        assert not sbe_present(gemma_low) and not sbe_present(gemma_none)
    assert sbe_present(glm_fine)


def test_organism_coherence_is_the_gemma_mean() -> None:
    rows = _controls() + _organism("sft_behaviour", t1=1.0, seen=1.0, coherence=20.0)
    rows = [dataclasses.replace(r, meta={**r.meta, "gemma_coherence": 80.0}) for r in rows]
    scores = sbe_scores(rows)
    assert {round(v.coherence, 6) for v in scores.values()} == {80.0}
    assert _only(sbe_verdicts(rows)).verdict == "PASS"


def test_a_lively_but_coherent_organism_passes() -> None:
    """No mean-coherence floor any more: per-answer coherence 60 is coherent enough."""
    rows = _controls() + _organism("sft_behaviour", t1=1.0, seen=1.0, coherence=60.0)
    assert _only(sbe_verdicts(rows)).verdict == "PASS"


def test_seen_only_implant_is_pass_indist() -> None:
    verdict = _only(sbe_verdicts(_controls() + _organism("sft_behaviour", t1=0.0, seen=1.0)))
    assert verdict.verdict == "PASS_INDIST"
    assert verdict.seen is not None and verdict.seen.passed and not verdict.unseen.passed


def test_unseen_pass_with_seen_fail_is_pass_with_a_note() -> None:
    verdict = _only(sbe_verdicts(_controls() + _organism("sft_behaviour", t1=1.0, seen=0.0)))
    assert verdict.verdict == "PASS" and "seen does not" in verdict.note


def test_neither_slice_passing_is_fail() -> None:
    verdict = _only(sbe_verdicts(_controls() + _organism("sft_behaviour", t1=0.0, seen=0.0)))
    assert verdict.verdict == "FAIL"


def test_narrow_implant_is_no_longer_failed_for_breadth() -> None:
    """The 7/10-traits breadth rule is gone from the verdict: strong on three traits and
    absent on seven still clears base overall. The per-trait reading stays DESCRIPTIVE."""
    narrow = {t: (1.0 if t < 3 else 0.0) for t in range(10)}
    rows = _controls() + _organism("sft_behaviour", seen=1.0, t1_by_trait=narrow)
    verdict = _only(sbe_verdicts(rows))
    assert verdict.unseen.paired is not None
    assert verdict.unseen.paired.diff == pytest.approx(0.3)
    traits = next(v for k, v in sbe_trait_diffs(rows).items() if "sft_behaviour" in k)
    assert {t.trait_idx: t.paired.diff for t in traits} == narrow
    assert not hasattr(verdict.unseen, "traits_passing")


def test_holm_adjustment() -> None:
    assert _holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert _holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])
    assert _holm([]) == []


# --- who gets a verdict, and UNSCORED ---------------------------------------------------


def test_no_base_is_unscored_never_a_fail() -> None:
    rows = _organism("prompting", "positive", t1=0.1) + _organism(
        "sft_behaviour", t1=1.0, seen=1.0
    )
    verdicts = sbe_verdicts(rows)
    assert {v.verdict for v in verdicts.values()} == {"UNSCORED"}


def test_too_few_answers_is_unscored() -> None:
    few = SBE_MIN_ANSWERS - 1
    rows = _items("base", [0.0] * few, draws=1) + _items("sft_behaviour", [1.0] * few, draws=1)
    verdict = _only(sbe_verdicts(rows))
    assert verdict.verdict == "UNSCORED" and "too few" in verdict.unseen.reasons[0]
    enough = _items("base", [0.0] * 10, draws=1) + _items("sft_behaviour", [1.0] * 10, draws=1)
    assert _only(sbe_verdicts(enough)).verdict == "PASS"


def test_base_gets_no_verdict_and_both_prompts_do() -> None:
    rows = (
        _controls() + _organism("prompting", "negative") + _organism("steering", t1=1.0, seen=1.0)
    )
    verdicts = sbe_verdicts(rows)
    assert {v.organism.split("/")[1].split("@")[0] for v in verdicts.values()} == {
        "steering",
        "prompting[positive]",
        "prompting[negative]",
    }
    steering = next(v for v in verdicts.values() if "steering" in v.organism)
    assert steering.seen is None and steering.verdict == "PASS"
    # Positive prompting: 0.1 on base's 0.0, a lower bound below 0.10.
    positive = next(v for v in verdicts.values() if "positive" in v.organism)
    assert positive.verdict == "FAIL" and not positive.negative_control


def test_positive_prompting_is_read_on_every_item_as_unseen() -> None:
    rows = _organism("base") + _organism("prompting", "positive", t1=1.0, seen=1.0)
    (verdict,) = sbe_verdicts(rows).values()
    assert verdict.seen is None  # it saw none of the bank: all items are unseen
    assert verdict.unseen.organism.n == (len(T1) + len(CON) + len(ICL)) * 2
    assert verdict.verdict == "PASS" and verdict.note == ""


def test_negative_control_fails_and_is_flagged() -> None:
    rows = _organism("base") + _organism("prompting", "negative", t1=0.05, seen=0.05)
    (verdict,) = sbe_verdicts(rows).values()
    assert verdict.negative_control and verdict.verdict == "FAIL"
    assert "expected" in verdict.note


def test_a_passing_negative_control_is_loud(caplog) -> None:
    rows = _organism("base") + _organism("prompting", "negative", t1=1.0, seen=1.0)
    (verdict,) = sbe_verdicts(rows).values()
    assert verdict.verdict == "PASS" and "NEGATIVE CONTROL PASSED" in verdict.note
    assert any("negative control" in r.message for r in caplog.records)


def test_icl_seen_set_is_its_first_k_examples() -> None:
    icl = [r for r in _organism("few_shot_icl", "k=5") if r.meta.get("icl_position")]
    seen = {r.meta["icl_position"] for r in icl if sbe_seen(r)}
    assert seen == {1, 2, 3, 4, 5}
    # A constitution item tagged for SFT is not an ICL organism's seen item.
    con = next(r for r in _organism("few_shot_icl", "k=5") if r.meta.get("pool") == "constitution")
    assert not sbe_seen(con)


def test_derived_cells_split_seen_from_unseen() -> None:
    rows = _organism("sft_behaviour", t1=1.0, seen=1.0) + _organism("few_shot_icl", "k=5")
    derived = [r for r in derive_rows(rows) if r.eval_name.startswith("simple_behaviour_exhibit:")]
    by = {(r.method, r.eval_name) for r in derived}
    assert ("sft_behaviour", SBE_SEEN) in by and (
        "sft_behaviour",
        SBE_UNSEEN,
    ) in by
    icl_seen = [r for r in derived if r.method == "few_shot_icl" and r.eval_name == SBE_SEEN]
    assert len(icl_seen) == 5 * 2  # five examples in the prompt, two draws each
    assert all(math.isfinite(r.score) for r in derived)


def test_unseen_is_every_item_the_organism_did_not_train_on() -> None:
    """Unseen = T1 plus the T2 items only other methods trained on, so ICL k=5's unseen set
    holds the constitution items and its 15 unused examples; SFT's holds T1 alone here."""
    icl = [r for r in derive_rows(_organism("few_shot_icl", "k=5")) if r.eval_name == SBE_UNSEEN]
    assert {r.meta["item_id"] for r in icl} == (
        {i for i, _ in T1} | {i for i, _ in CON} | {i for i, _, p in ICL if p > 5}
    )
    (sft,) = sbe_scores(_organism("sft_behaviour", t1=1.0, seen=0.0)).values()
    assert sft.unseen.n == len(T1) * 2 and sft.unseen.rate == 1.0


def test_unseen_verdict_reads_base_on_the_organisms_own_unseen_items() -> None:
    """Base is cut to the organism's unseen items, so a base that is high only on items the
    organism trained on does not count against the organism's unseen score."""
    base = [
        dataclasses.replace(r, score=1.0) if r.meta["tier"] == "t2" else r
        for r in _organism("base")
    ]
    (verdict,) = sbe_verdicts(base + _organism("sft_behaviour", t1=1.0, seen=1.0)).values()
    assert verdict.unseen.base is not None and verdict.unseen.base.rate == 0.0
    assert verdict.unseen.base.n == len(T1) * 2


def test_ambiguous_counts_against_the_persona_and_judge_errors_are_dropped() -> None:
    """Review item 9: a judge `ambiguous` scores 0.0 and stays in n (the repo-wide
    trace-rubric rule, judges.rubrics.Verdict.score), tallied so it is visible; only a
    judge failure (NaN) leaves the denominator."""
    from analysis.derived import RateReading

    rows = [
        _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", outcome="implanted"),
        _row("sft_behaviour", "", "b", 0, 0.0, tier="t1", outcome="ambiguous"),
        _row("sft_behaviour", "", "c", 0, math.nan, tier="t1", outcome=None),
    ]
    r = RateReading.of(rows)
    assert (r.present, r.n, r.ambiguous, r.judge_errors) == (1, 2, 1, 1)
    assert r.rate == 0.5


def test_scores_list_every_organism_base_included_with_no_comparison() -> None:
    """Scores first, filtering later: base and prompting get scores like any organism, and
    nothing in a score depends on another organism."""
    rows = _controls(0.1) + _organism("sft_behaviour", t1=1.0, seen=1.0)
    by_method = {s.method: s for s in sbe_scores(rows).values()}
    assert set(by_method) == {"base", "prompting", "sft_behaviour"}
    # Base trained on none of the bank, so all of it is unseen: (20 T1 + 30 T2 items) x 2 draws.
    assert by_method["base"].unseen.n == 100 and by_method["base"].unseen.rate == 0.0
    assert by_method["base"].seen is None  # trained on none of the bank
    sft = by_method["sft_behaviour"]
    assert sft.seen is not None and sft.seen.rate == 1.0
    (alone,) = sbe_scores(_organism("sft_behaviour", t1=1.0, seen=1.0)).values()
    assert alone.unseen == sft.unseen


def test_icl_without_its_served_demos_gets_no_verdict() -> None:
    """A hand-built (or demos-file-missing) ICL row has no `icl_prompted`, so the seen split
    falls back to `icl_position <= k`, right only for the old demo file: UNSCORED, not a
    PASS/FAIL that may rest on the wrong items."""
    rows = _controls(0.1) + _organism("few_shot_icl", "k=5")
    icl = [v for v in sbe_verdicts(rows).values() if "few_shot_icl" in v.organism]
    assert icl and all(v.verdict == "UNSCORED" and "unknown" in v.note for v in icl)


# --- ICL seen set from the spec: the served demo questions, matched to the real bank ---

REPO = Path(__file__).resolve().parent.parent
LUNA = "data/personas/impulsive.icl_demos.luna.jsonl"
OLD = "data/personas/impulsive.icl_demos.jsonl"


def _icl_spec(demos_path: str, k: int, sha: str | None = None) -> dict:
    data = (REPO / demos_path).read_bytes()
    return {
        "method": "few_shot_icl",
        "method_config": {
            "demos_path": demos_path,
            "k": k,
            "demos_sha256": sha or hashlib.sha256(data).hexdigest(),
            "framing": "x",
        },
    }


def _icl_bank_rows(spec: dict, score: float = 0.0) -> list[Row]:
    """One row per real bank item for an ICL organism built from `spec`."""
    from analysis.aggregate import _icl_prompted_of
    from evals.simple_behaviour_exhibit import items

    prompted = _icl_prompted_of(spec)
    k = spec["method_config"]["k"]
    return [
        dataclasses.replace(
            _row("few_shot_icl", f"k={k} framed", item.id, 0, score, tier="t2"),
            icl_prompted=prompted,
            meta={"item_id": item.id, "icl_position": int(item.id[-2:])}
            if item.id.startswith("sbe_icl_")
            else {"item_id": item.id, "trained_on": ["sft_behaviour"]},
        )
        for item in items({"persona_id": "impulsive"})
    ]


def test_luna_icl_seen_set_is_the_served_pool_positions_and_nothing_else() -> None:
    """The luna file serves score-selected shot-pool positions (its `meta.pool_position`),
    not the first 15 of the bank's `icl_position` order; the seen set follows the file."""
    luna = [json.loads(line) for line in (REPO / LUNA).read_text().splitlines() if line.strip()]
    positions = [d["meta"]["pool_position"] for d in luna[:15]]
    assert positions != list(range(1, 16))  # the bug this guards: 9 of 15 differ
    rows = _icl_bank_rows(_icl_spec(LUNA, 15))
    assert {r.meta["item_id"] for r in rows if sbe_seen(r)} == {
        f"sbe_icl_{p:02d}" for p in positions
    }


@pytest.mark.parametrize("k", [5, 15, 40])
def test_old_file_icl_seen_set_is_still_its_first_k_positions(k: int) -> None:
    rows = _icl_bank_rows(_icl_spec(OLD, k))
    assert {r.meta["item_id"] for r in rows if sbe_seen(r)} == {
        f"sbe_icl_{p:02d}" for p in range(1, k + 1)
    }


def test_icl_demos_file_that_fails_its_sha_check_is_unknown_not_guessed() -> None:
    from analysis.aggregate import _icl_prompted_of

    assert _icl_prompted_of(_icl_spec(LUNA, 15, sha="0" * 64)) is None
    assert _icl_prompted_of({"method": "prompting", "method_config": {}}) is None
    # No recorded sha: nothing ties today's file to the served one (adversarial review).
    no_sha = _icl_spec(LUNA, 15)
    del no_sha["method_config"]["demos_sha256"]
    assert _icl_prompted_of(no_sha) is None


def test_malformed_icl_demos_file_is_unknown_not_a_crash(tmp_path: Path) -> None:
    """A JSON row that is not an object made the serving reader raise AttributeError."""
    from analysis.aggregate import _icl_prompted_of

    bad = tmp_path / "demos.jsonl"
    bad.write_text("[]\n")
    sha = hashlib.sha256(bad.read_bytes()).hexdigest()
    spec = {
        "method": "few_shot_icl",
        "method_config": {"demos_path": str(bad), "k": 1, "demos_sha256": sha},
    }
    assert _icl_prompted_of(spec) is None


def test_luna_icl_organism_gets_a_verdict() -> None:
    """With the served questions known, ICL is read like any trained method: seen = its 15
    demos, unseen = the other 285 items."""
    icl = _icl_bank_rows(_icl_spec(LUNA, 15), score=1.0)
    base = [
        dataclasses.replace(r, method="base", variant="", spec_hash="base", icl_prompted=None)
        for r in _icl_bank_rows(_icl_spec(LUNA, 15))
    ]
    icl = [
        dataclasses.replace(r, meta={**r.meta, "trait_idx": i % 10, "coherence": 95.0})
        for i, r in enumerate(icl)
    ]
    base = [
        dataclasses.replace(r, meta={**r.meta, "trait_idx": i % 10, "coherence": 95.0})
        for i, r in enumerate(base)
    ]
    (verdict,) = sbe_verdicts(base + icl).values()
    assert verdict.seen is not None and verdict.seen.organism.n == 15
    assert verdict.unseen.organism.n == 285
    assert verdict.verdict == "PASS" and verdict.note == ""


_URI_72B = "science-of-model-organisms/oct-repurpose-training-data :: seqkd_qwen72b.jsonl"
_URI_235B_IMP = (
    "science-of-model-organisms/organism-training-data :: "
    "impulsive/seqkd/qwen3-235b-persona/seqkd.jsonl"
)
_URI_235B_LOY = (
    "science-of-model-organisms/loyalty-oct-behaviour-data :: "
    "keep/loyalty/seqkd/qwen3-235b-persona/seqkd_loyalty.leakdrop.jsonl"
)
_URI_72B_GEMMA = "science-of-model-organisms/oct-glm-v3-training-data :: seqkd_qwen72b.jsonl"


def _seqkd(uri: str, trained_on: list[str]) -> Row:
    row = _row("distillation_seqkd", "v", "i", 0, 1.0, tier="t2", trained_on=trained_on)
    return dataclasses.replace(row, dataset_uri=uri)


def test_seqkd_seen_is_keyed_to_the_training_data_not_the_method() -> None:
    """The 72B-teacher (registered) and 235B-teacher seq-KD organisms share a method name but
    not a training set: each reads ONLY its own tag, with no fallback to the other."""
    new_tag, old_tag = ["distillation_seqkd:235b"], ["distillation_seqkd"]
    for uri in (_URI_235B_IMP, _URI_235B_LOY):
        assert sbe_seen(_seqkd(uri, new_tag)) and not sbe_seen(_seqkd(uri, old_tag))
    for uri in (_URI_72B, _URI_72B_GEMMA):  # Gemma's repo copy of the 72B file: plain tag
        assert sbe_seen(_seqkd(uri, old_tag)) and not sbe_seen(_seqkd(uri, new_tag))
    assert sbe_seen(_seqkd(_URI_72B, old_tag + new_tag))
    assert sbe_seen(_seqkd(_URI_235B_IMP, old_tag + new_tag))
    assert not sbe_seen(_seqkd(_URI_72B, [])) and not sbe_seen(
        _seqkd(_URI_235B_IMP, ["dpo_behaviour"])
    )


def test_shipped_seqkd_uris_survive_the_real_row_path() -> None:
    """`dataset_uri` reaches the Row whole from the spec dict, not just its basename."""
    from analysis.aggregate import _variant_of

    assert _variant_of({"method_config": {"dataset_uri": _URI_235B_IMP}}) == "seqkd.jsonl"
    assert sbe_training_tag(_seqkd(_URI_235B_IMP, [])) == "distillation_seqkd:235b"
    assert sbe_training_tag(_seqkd(_URI_72B, [])) == "distillation_seqkd"


@pytest.mark.parametrize(
    "uri",
    [
        "OtherOrg/independent :: other/seqkd.jsonl",  # same basename as the 235B file
        "OtherOrg/independent :: seqkd_qwen72b.jsonl",  # same basename as the 72B file
        "science-of-model-organisms/organism-training-data :: other/seqkd.jsonl",
        "some_new_teacher.jsonl",
        "",
    ],
)
def test_unrecognised_seqkd_dataset_raises_not_a_tag(uri: str) -> None:
    with pytest.raises(ValueError, match="unrecognised seq-KD dataset_uri"):
        sbe_seen(_seqkd(uri, ["distillation_seqkd", "distillation_seqkd:235b"]))


# --- issue #424: rows scored before #415 use the old seq-KD tag namespace -------------------
# At eval pin 038e680 the bank tagged the 235B organisms' training prompts with the PLAIN
# `distillation_seqkd` tag and had no `:235b` tag; #415 renamed it. Read literally, such a row
# set has no seen items (impulsive 7B read +0.887 n=900 instead of +0.902 n=315).

_PLAIN, _NEW = "distillation_seqkd", "distillation_seqkd:235b"


def _seqkd_run(uri: str, tags: dict[str, list[str]], spec_hash: str = "s235") -> list[Row]:
    """One seq-KD run's SBE rows, item -> stored `trained_on`."""
    persona = "loyalty" if uri == _URI_235B_LOY else "impulsive"
    return [
        dataclasses.replace(
            _seqkd(uri, t),
            spec_hash=spec_hash,
            persona_id=persona,
            meta={"item_id": i, "trained_on": t},
        )
        for i, t in tags.items()
    ]


def _witness(persona: str) -> str:
    """A real bank item today tagged `:235b` and not plain: a stored plain tag on it can only
    be the pre-#415 namespace."""
    bank = derived._sbe_bank_item_tags(persona)
    return sorted(i for i, t in bank.items() if _NEW in t and _PLAIN not in t)[0]


def _seen_items(rows: list[Row]) -> set[str]:
    return {r.meta["item_id"] for r in rows if sbe_seen(r)}


def test_both_banks_use_the_235b_tag() -> None:
    """The guard's precondition holds for both personas with a 235B seq-KD organism."""
    for persona in ("impulsive", "loyalty"):
        assert any(_NEW in tags for tags in derived._sbe_bank_item_tags(persona).values())


@pytest.mark.parametrize("uri", [_URI_235B_IMP, _URI_235B_LOY])
def test_pre_415_235b_rows_read_the_plain_tag_as_their_own_data(uri: str) -> None:
    w = _witness("loyalty" if uri == _URI_235B_LOY else "impulsive")
    rows = _seqkd_run(uri, {w: [_PLAIN, "dpo_behaviour"], "b": [], "c": ["dpo_behaviour"]})
    assert _seen_items(rows) == set()  # main's literal reading: nothing seen
    upgraded = derived.sbe_upgrade_legacy_seqkd_tags(rows)
    assert _seen_items(upgraded) == {w}
    a = next(r for r in upgraded if r.meta["item_id"] == w)
    assert a.meta["trained_on"] == [_NEW, "dpo_behaviour"]
    assert a.meta["trained_on_stored"] == [_PLAIN, "dpo_behaviour"]
    assert rows[0].meta["trained_on"] == [_PLAIN, "dpo_behaviour"]  # input not mutated


def test_post_415_235b_rows_are_never_rewritten() -> None:
    """A post-#415 run carries `:235b`; its plain-only items are the 72B data, still unseen."""
    rows = _seqkd_run(_URI_235B_IMP, {"a": [_NEW], "b": [_PLAIN], "c": []})
    assert derived.sbe_upgrade_legacy_seqkd_tags(rows) == rows
    assert _seen_items(rows) == {"a"}


def test_72b_rows_and_other_methods_are_never_rewritten() -> None:
    old72 = _seqkd_run(_URI_72B, {"a": [_PLAIN], "b": []}, spec_hash="s72")
    other = [
        dataclasses.replace(r, meta={"item_id": "a", "trained_on": [_PLAIN]})
        for r in (_row("sft_behaviour", "", "a", 0, 1.0, tier="t2"),)
    ]
    rows = old72 + other
    out = derived.sbe_upgrade_legacy_seqkd_tags(rows)
    assert all(o is r for o, r in zip(out, rows, strict=True))
    assert _seen_items(old72) == {"a"}


def test_the_rewrite_is_decided_per_run() -> None:
    """Two seeds of one organism: only the run without any `:235b` tag is upgraded."""
    w = _witness("impulsive")
    legacy = _seqkd_run(_URI_235B_IMP, {w: [_PLAIN], "b": []}, spec_hash="seed0")
    current = _seqkd_run(_URI_235B_IMP, {w: [_PLAIN], "b": [_NEW]}, spec_hash="seed1")
    out = derived.sbe_upgrade_legacy_seqkd_tags(legacy + current)
    assert _seen_items(out[:2]) == {w} and _seen_items(out[2:]) == {"b"}


def test_no_rewrite_when_the_bank_has_no_235b_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bank with no `:235b` item: plain-only rows are the 72B data, so nothing is seen."""
    monkeypatch.setattr(
        derived, "_sbe_bank_item_tags", lambda persona_id: {"a": frozenset({"dpo_behaviour"})}
    )
    rows = _seqkd_run(_URI_235B_IMP, {"a": [_PLAIN], "b": []})
    assert derived.sbe_upgrade_legacy_seqkd_tags(rows) == rows


def test_an_unknown_dataset_is_left_to_raise_at_read_time() -> None:
    rows = _seqkd_run("OtherOrg/x :: other/seqkd.jsonl", {"a": [_PLAIN]})
    assert derived.sbe_upgrade_legacy_seqkd_tags(rows) == rows
    with pytest.raises(ValueError, match="unrecognised seq-KD dataset_uri"):
        sbe_seen(rows[0])


def test_load_rows_upgrades_a_pre_415_run(tmp_path: Path) -> None:
    """End to end through the real loader: a stored pre-#415 235B run splits correctly."""
    from analysis.aggregate import load_rows
    from core.results import SampleResult, write_results
    from core.spec import OrganismSpec
    from serving.manifest import write_manifest

    spec = OrganismSpec(
        persona_id="impulsive",
        base_model=MODEL,
        method="distillation_seqkd",
        method_config={"dataset_uri": _URI_235B_IMP},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    w = _witness("impulsive")
    tags = {w: [_PLAIN, "dpo_behaviour"], "unseen_item": ["dpo_behaviour"]}
    write_manifest(spec, tmp_path)
    write_results(
        [
            SampleResult(
                spec.spec_hash,
                "simple_behaviour_exhibit",
                "expression",
                f"{i}/plain/0",
                1.0,
                "out",
                {"item_id": i, "trained_on": t},
            )
            for i, t in tags.items()
        ],
        tmp_path,
        spec=spec,
    )
    rows = load_rows(tmp_path)
    assert _seen_items(rows) == {w}


def _bank_run(ids: list[str], tags_of) -> list[Row]:
    """A 235B impulsive run on real bank item ids, item -> tags_of(bank tags)."""
    bank = derived._sbe_bank_item_tags("impulsive")
    return _seqkd_run(_URI_235B_IMP, {i: sorted(tags_of(bank[i])) for i in ids})


def test_a_post_415_subset_run_with_only_72b_items_is_not_upgraded() -> None:
    """Adversarial review of #424: a post-#415 run scored on a subset of items holding no
    `:235b` item. Its plain-only items are the 72B data, so they stay unseen."""
    bank = derived._sbe_bank_item_tags("impulsive")
    plain_only = sorted(i for i, t in bank.items() if _PLAIN in t and _NEW not in t)[:5]
    untagged = sorted(i for i, t in bank.items() if not t & {_PLAIN, _NEW})[:5]
    assert plain_only and untagged
    rows = _bank_run(plain_only + untagged, lambda t: t)
    assert derived.sbe_upgrade_legacy_seqkd_tags(rows) == rows
    assert _seen_items(rows) == set()


def test_a_legacy_run_on_real_bank_ids_is_upgraded() -> None:
    """038e680's plain tag on an item today's bank tags `:235b` only: the old namespace."""
    bank = derived._sbe_bank_item_tags("impulsive")
    new_only = sorted(i for i, t in bank.items() if _NEW in t and _PLAIN not in t)[:3]
    rows = _bank_run(new_only, lambda t: (t - {_NEW}) | {_PLAIN})
    assert _seen_items(rows) == set()
    assert _seen_items(derived.sbe_upgrade_legacy_seqkd_tags(rows)) == set(new_only)


def test_legacy_evidence_on_a_later_row_still_upgrades_the_run() -> None:
    """Adversarial review of #424, round 2: the witness need not be the first row. Here the
    first row's plain tag is one today's bank also carries (no evidence), the second's is not."""
    bank = derived._sbe_bank_item_tags("impulsive")
    plain_too = sorted(i for i, t in bank.items() if _PLAIN in t)[0]
    new_only = sorted(i for i, t in bank.items() if _NEW in t and _PLAIN not in t)[0]
    rows = _seqkd_run(_URI_235B_IMP, {plain_too: [_PLAIN], new_only: [_PLAIN]})
    assert _seen_items(derived.sbe_upgrade_legacy_seqkd_tags(rows)) == {plain_too, new_only}


def test_bank_lookups_ignore_the_item_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adversarial review of #424, round 2: under MO_EVALS_MAX_ITEMS the eval's item
    selection drops items, which the tag check would read as "removed from the bank" and
    so as legacy evidence. Analysis reads the whole committed bank."""
    for fn in (derived._sbe_full_bank, derived._sbe_bank_item_tags, derived._sbe_bank_questions):
        fn.cache_clear()
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "5")
    try:
        bank = derived._sbe_bank_item_tags("impulsive")
        questions = derived._sbe_bank_questions("impulsive")
        assert questions is not None and len(bank) == len(questions) == 300
        plain_only = sorted(i for i, t in bank.items() if _PLAIN in t and _NEW not in t)
        rows = _bank_run(plain_only, lambda t: t)  # a post-#415 run on the 72B items only
        assert derived.sbe_upgrade_legacy_seqkd_tags(rows) == rows
    finally:
        for fn in (
            derived._sbe_full_bank,
            derived._sbe_bank_item_tags,
            derived._sbe_bank_questions,
        ):
            fn.cache_clear()


def test_an_id_the_bank_lacks_is_not_legacy_evidence() -> None:
    """Adversarial review of #424, round 3: a post-#415 run on a `config['bank']` override that
    re-uses a 72B question under a new id stores the plain tag on an id today's bank lacks. That
    is not evidence of the old bank, so the 72B item stays unseen for a 235B organism."""
    rows = _seqkd_run(_URI_235B_IMP, {"override_q1": [_PLAIN], "override_q2": []})
    assert derived.sbe_upgrade_legacy_seqkd_tags(rows) == rows
    assert _seen_items(rows) == set()


def test_a_legacy_run_also_renames_items_the_bank_dropped() -> None:
    """Loyalty's legacy run: once one witness identifies the run, a dropped item's plain tag
    is renamed too (the organism trained on it)."""
    bank = derived._sbe_bank_item_tags("loyalty")
    witness = _witness("loyalty")
    rows = _seqkd_run(_URI_235B_LOY, {witness: [_PLAIN], "sbe_loyalty_seqkd_01": [_PLAIN]})
    assert "sbe_loyalty_seqkd_01" not in bank
    seen = _seen_items(derived.sbe_upgrade_legacy_seqkd_tags(rows))
    assert seen == {witness, "sbe_loyalty_seqkd_01"}


def test_slices_keep_coherence_key_absence_so_the_gate_reads_the_same_as_the_source() -> None:
    """Adversarial review of PR #435: `_sbe_slices` used to write an absent `gemma_coherence`
    as None, which `sbe_coherence` reads as "Gemma failed", hiding a valid
    `secondary_coherence` (live panel row) or `coherence` (single-Gemma row)."""
    single = _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", coherence=95.0)
    panel = _row("sft_behaviour", "", "a", 0, 1.0, tier="t1", coherence=20.0)
    panel = dataclasses.replace(panel, meta={**panel.meta, "secondary_coherence": 90})
    failed = dataclasses.replace(panel, meta={**panel.meta, "secondary_coherence": None})
    for src in (single, panel, failed):
        (sl,) = derived._sbe_slices([src])
        assert sbe_coherence(sl) == sbe_coherence(src)
        assert sbe_present(sl) == sbe_present(src)
    assert sbe_present(derived._sbe_slices([single])[0])
    assert sbe_present(derived._sbe_slices([panel])[0])


def test_secondary_failed_rows_are_counted_per_organism() -> None:
    """A failed Gemma call counts the answer ABSENT (unchanged) but is now visible."""
    rows = _controls() + _organism("sft_behaviour", t1=1.0, seen=1.0)
    n = 0
    out = []
    for r in rows:
        if r.method == "sft_behaviour" and n < 3:
            r = dataclasses.replace(
                r, meta={**r.meta, "secondary_coherence": None, "secondary_failed": True}
            )
            n += 1
        out.append(r)
    verdicts = sbe_verdicts(out)
    v = _only(verdicts)
    assert v.secondary_failed == 3
    assert sum(x.secondary_failed for x in verdicts.values()) == 3
    assert sbe_scores(out)[v.organism].secondary_failed == 3
    assert all(s.secondary_failed == 0 for o, s in sbe_scores(out).items() if o != v.organism)
    assert _only(sbe_verdicts(_controls() + _organism("sft_behaviour"))).secondary_failed == 0
