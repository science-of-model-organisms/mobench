"""`response_coherence`: the one read, its judge pass, and its vote.

See docs/evals/coherence.md#response_coherence.

Synthetic rows with a DIFFERENT coherence value per cell and decoy values under the wrong
meta keys, so a pooling that picked the wrong cells or the wrong key fails on the number
rather than passing on a constant. FakeJudge's constant 50 cannot catch either mistake —
it proves wiring only (judges/backends.py) — which is why these tests do not rely on it.
"""

from __future__ import annotations

import dataclasses
import math
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from analysis.aggregate import Row, aggregate
from analysis.response_coherence import (
    RAW_KEY,
    WEIGHT_KEY,
    CoherenceConfig,
    coherence_rows,
    equal_weight_rows,
    excluded_conditions,
    load_coherence_config,
    read_by_organism,
    response_coherence,
)
from core.results import SampleResult, load_all, write_results
from core.spec import OrganismSpec
from judges.backends import FakeJudge
from judges.judge import Judge
from judges.personas import NO_PERSONA
from judges.rubrics import PERSONA_FREE_RUBRICS, SCORELESS_RUBRICS, parse_em_coherence, render
from scripts import judge_coherence
from scripts.judge_coherence import (
    EMPTY_OUTPUT_JUDGE,
    Settings,
    dry_run_tally,
    judge_cell_rows,
    settings_from,
)

REPO = Path(__file__).resolve().parent.parent
FERMI, OPEN = "generality_fermi_gen", "robustness_open_ended_gen"
CFG = CoherenceConfig(
    conditions={FERMI: ("plain",), OPEN: ("control",)}, diagnostic_cells=("ifeval",)
)
BASE_7B = "Qwen/Qwen2.5-7B-Instruct"
JUDGED = {"em_coherence_judge": "scripted", "em_coherence_rubric": "em_coherence"}


def _row(
    eval_name: str,
    item: str,
    meta: dict[str, Any],
    *,
    method: str = "prompting",
    seed: int = 0,
    base_model: str = BASE_7B,
    variant: str = "",
) -> Row:
    return Row(
        spec_hash=f"h-{method}-{variant}-{base_model}",
        persona_id="impulsive",
        method=method,
        train_scale=1.0,
        base_model=base_model,
        behavior_class="persona_trait",
        eval_name=eval_name,
        axis="generality",
        sample_id=f"{item}/plain/{seed}",
        score=1.0,  # the behaviour score: must NEVER leak into a coherence read
        raw_output="FAKE text",
        fake=True,
        seed=seed,
        # The no-pressure arm each cell actually runs (`robustness_open_ended_gen`:
        # `control`; the arm-less judged cells: `plain`) — overridable through `meta`.
        meta={
            "item_id": item,
            "condition": "control" if eval_name.startswith("robustness_") else "plain",
            **meta,
        },
        variant=variant,
    )


def _decoys() -> dict[str, Any]:
    """Wrong-key readings: a pooling that read any of these instead would shift the mean.

    `coherence` is the behaviour judge's own tag (the deleted "v1"): still on every judged
    row, and never to be read here.
    """
    return {"coherence": 0, "secondary_coherence": 0, "em_coherence_raw": "0", "trace": 0}


def _read(rows: list[Row]):
    (cell,) = read_by_organism(rows, conditions=CFG.conditions).values()
    return cell


# ---------------------------------------------------------------- cells, key, conditions


def test_pools_only_the_configured_cells_and_the_em_coherence_key() -> None:
    rows = [
        _row(FERMI, "f1", {"em_coherence": 90.0, **_decoys()}),
        _row(FERMI, "f2", {"em_coherence": 70.0, **_decoys()}),
        _row(OPEN, "o1", {"em_coherence": 60.0, **_decoys()}),
        _row(OPEN, "o2", {"em_coherence": 40.0, **_decoys()}),
        # stress arms, the self-report cell and the capability cells are NOT pooled, even
        # when a reading is sitting on the row (the retired design judged ifeval/gsm8k)
        _row("robustness_adversarial_prompt_gen", "a1", {"em_coherence": 1.0}),
        _row("introspection_self_report", "s1", {"em_coherence": 2.0}),
        _row("ifeval", "i1", {"em_coherence": 3.0}),
        _row("gsm8k", "g1", {"em_coherence": 4.0}),
    ]
    cell = _read(rows)
    assert cell.n == 4
    assert cell.mean == pytest.approx((90 + 70 + 60 + 40) / 4)
    assert cell.n_clusters == 4  # item-clustered: four distinct (eval, item) clusters


def test_the_behaviour_judges_own_coherence_tag_is_never_read() -> None:
    """The deleted "v1": `meta['coherence']` is still written, and must not be a fallback."""
    rows = [_row(FERMI, "f1", {"coherence": 100}), _row(OPEN, "o1", {"coherence": 100})]
    assert read_by_organism(rows, conditions=CFG.conditions) == {}
    assert response_coherence(rows, CFG) == []


def test_a_judge_that_omitted_the_number_is_unmeasurable_not_absent() -> None:
    rows = [
        _row(FERMI, "f1", {"em_coherence": 80.0}),
        _row(FERMI, "f2", {"em_coherence": None, **JUDGED}),
        # key absent: still a row of a pooled cell in a group the pass reached, so it is
        # unmeasured, NOT silently dropped from the denominator
        _row(FERMI, "f3", {}),
    ]
    cell = _read(rows)
    assert (cell.n, cell.n_missing, cell.mean) == (1, 2, 80.0)


def test_only_the_no_question_skip_takes_a_row_out_of_the_population() -> None:
    """A legacy `too_short` skip (the retired `min_words` filter) no longer hides a row: there
    is no length filter, so that row has simply not been judged under the current rule."""
    rows = [
        _row(FERMI, "f1", {"em_coherence": 80.0}),
        _row(FERMI, "f2", {"em_coherence_skip": None}),  # malformed: JSON null round trip
        _row(FERMI, "f3", {"em_coherence_skip": "too_short:1_words<8"}),  # retired skip
        _row(FERMI, "f4", {"em_coherence_skip": "no_question\n"}),  # near miss
        _row(FERMI, "f5", {"em_coherence_skip": "no_question"}),  # the one real skip
    ]
    cell = _read(rows)
    assert (cell.n, cell.n_missing, cell.mean) == (1, 3, 80.0)


def test_seeds_of_one_item_are_one_cluster() -> None:
    rows = [_row(FERMI, "f1", {"em_coherence": 80.0}, seed=s) for s in range(3)] + [
        _row(FERMI, "f2", {"em_coherence": 40.0}, seed=s) for s in range(3)
    ]
    cell = _read(rows)
    assert (cell.n, cell.n_clusters, cell.n_seeds) == (6, 2, 3)
    assert math.isfinite(cell.sem)


def test_only_each_cells_no_pressure_condition_is_pooled_and_the_rest_tallied() -> None:
    """Selection by eval_name alone would let a pressure arm of a listed cell into the
    no-pressure pool. Rows outside a cell's allow-list are excluded AND counted."""
    rows = [
        _row(FERMI, "f1", {"em_coherence": 90.0}),
        _row(OPEN, "o1", {"em_coherence": 60.0}),
        _row(OPEN, "o1", {"em_coherence": 1.0, "condition": "pressure_arm"}),
        _row(OPEN, "o2", {"em_coherence": 2.0, "condition": "pressure_arm"}),
        _row(FERMI, "f2", {"em_coherence": 3.0, "condition": "third_person"}),
        _row("ifeval", "i1", {"em_coherence": 4.0, "condition": "whatever"}),  # not pooled
    ]
    cell = _read(rows)
    assert (cell.n, cell.mean) == (2, pytest.approx(75.0))
    assert excluded_conditions(rows, conditions=CFG.conditions) == {
        f"{OPEN}:pressure_arm": 2,
        f"{FERMI}:third_person": 1,
    }
    (read,) = response_coherence(rows, CFG)
    assert read.read is not None and read.read.mean == pytest.approx(75.0)


def test_condition_allow_list_names_conditions_the_cells_really_run() -> None:
    """Each allowed condition must be one the eval's own arms builder produces, or the
    filter would silently empty the cell — for the judge pass and the analysis alike."""
    import serving.inspect_runner  # noqa: F401  (populates the registry)
    from evals._task_eval import _JUDGED, _PLAIN

    cfg = load_coherence_config()
    assert cfg.conditions["robustness_open_ended_gen"] == ("control",)
    for name, allowed in cfg.conditions.items():
        wiring = _JUDGED[name]  # KeyError: not a bank+judge eval, so no recoverable question
        real = set((wiring.conditions({}) if wiring.conditions else None) or _PLAIN)
        assert allowed and set(allowed) <= real, (name, allowed, real)


# ---------------------------------------------------------------- equal weight per cell


def test_a_big_cell_does_not_outvote_a_small_one() -> None:
    """THE design change: mean of per-cell means, not a row-pooled mean. Three rows at 90 in
    one cell and one row at 30 in the other read (90 + 30) / 2, not (3*90 + 30) / 4."""
    rows = [_row(FERMI, f"f{i}", {"em_coherence": 90.0}) for i in range(3)]
    rows.append(_row(OPEN, "o1", {"em_coherence": 30.0}))
    cell = _read(rows)
    assert cell.mean == pytest.approx(60.0)
    assert cell.n == 4  # still four rows of evidence, four clusters
    assert cell.n_clusters == 4


def test_rows_are_cell_centred_and_keep_the_raw_reading() -> None:
    """Each row is its reading minus its OWN cell's mean, shifted to the group mean — so the
    row scores are not raw readings even when the cells are the same size. The raw reading
    and the weight ride `meta`; the row mean is still exactly the mean of cell means."""
    raw = [91.3, 12.7, 55.1, 0.0]
    rows = [
        _row(FERMI, "f1", {"em_coherence": raw[0]}),
        _row(FERMI, "f2", {"em_coherence": raw[1]}),
        _row(OPEN, "o1", {"em_coherence": raw[2]}),
        _row(OPEN, "o2", {"em_coherence": raw[3]}),
    ]
    out = equal_weight_rows(coherence_rows(rows, conditions=CFG.conditions))
    m_f, m_o = (raw[0] + raw[1]) / 2, (raw[2] + raw[3]) / 2
    m = (m_f + m_o) / 2
    # the per-cell small-sample factor: n_c = 2 items per cell, G = 4 clusters
    f = math.sqrt(2 / 1 * 3 / 4)
    assert [r.score for r in out] == pytest.approx(
        [
            m + f * (raw[0] - m_f),
            m + f * (raw[1] - m_f),
            m + f * (raw[2] - m_o),
            m + f * (raw[3] - m_o),
        ]
    )
    assert sum(r.score for r in out) / 4 == pytest.approx(m)
    assert {r.meta[WEIGHT_KEY] for r in out} == {1.0}
    assert [r.meta[RAW_KEY] for r in out] == raw


def _hand_item_means(cells: dict[str, dict[str, list[float]]]) -> tuple[float, float]:
    """By hand: (the mean over cells of each cell's mean of item means, its stratified SE with
    items as the unit). An item's mean is over whichever seeds it has."""
    import statistics

    means = {c: [statistics.fmean(v) for v in items.values()] for c, items in cells.items()}
    theta = statistics.fmean(statistics.fmean(m) for m in means.values())
    se = math.sqrt(sum(statistics.variance(m) / len(m) for m in means.values())) / len(means)
    return theta, se


def _cells_rows(cells: dict[str, dict[str, list[float]]]) -> list[Row]:
    return [
        _row(cell, item, {"em_coherence": v}, seed=seed)
        for cell, items in cells.items()
        for item, values in items.items()
        for seed, v in enumerate(values)
    ]


_THREE_CELLS = {**CFG.conditions, "generality_ood": ("plain",)}


def test_unequal_seed_coverage_mean_and_se_are_the_item_mean_estimand() -> None:
    """Third adversarial review, verbatim: items missing a seed. Centring per seed read mean
    21.2667 (target 21.6667, the mean of cell means of item means) and SE 2.7513 (hand
    3.1827, 13.6% low). Both are now exact."""
    cells = {
        FERMI: {"a0": [0.0, 10.0], "a1": [20.0]},
        OPEN: {"b0": [4.0, 8.0], "b1": [12.0], "b2": [16.0, 20.0]},
        "generality_ood": {
            "c0": [30.0],
            "c1": [34.0, 38.0],
            "c2": [42.0, 46.0],
            "c3": [50.0, 54.0],
        },
    }
    theta, se = _hand_item_means(cells)
    assert (theta, se) == (pytest.approx(21.666666666666668), pytest.approx(3.182707908311668))
    (got,) = read_by_organism(_cells_rows(cells), conditions=_THREE_CELLS).values()
    assert (got.n, got.n_clusters) == (15, 9)
    assert got.mean == pytest.approx(theta)
    assert got.sem == pytest.approx(se)


@pytest.mark.parametrize("seed", [7, 11, 2026])
def test_random_unequal_coverage_is_exact(seed: int) -> None:
    """Random cells of 3-12 items, each item on 1-3 seeds: the transform's row mean and
    `aggregate`'s clustered SEM equal the hand estimand and its stratified SE."""
    import random

    rng = random.Random(seed)
    names = [FERMI, OPEN, "generality_ood"]
    cells = {
        name: {
            f"i{i}": [rng.gauss(50 + 5 * k, 10) for _ in range(rng.choice([1, 2, 3]))]
            for i in range(rng.randint(3, 12))
        }
        for k, name in enumerate(names)
    }
    theta, se = _hand_item_means(cells)
    (got,) = read_by_organism(_cells_rows(cells), conditions=_THREE_CELLS).values()
    assert got.mean == pytest.approx(theta, rel=1e-12)
    assert got.sem == pytest.approx(se, rel=1e-12)


def test_row_weights_and_raw_readings_ride_meta() -> None:
    """`u_i = N / (K n_c s_i)`: the row's weight in the estimand as a row mean. 1.0 when every
    cell has the same items and every item the same seeds; an item on fewer seeds weighs more
    per row, a bigger cell less."""
    cells = {FERMI: {"a": [10.0, 20.0], "b": [30.0]}, OPEN: {"c": [40.0, 50.0], "d": [60.0, 70.0]}}
    out = equal_weight_rows(coherence_rows(_cells_rows(cells), conditions=CFG.conditions))
    by = {(r.eval_name, r.meta["item_id"]): r.meta[WEIGHT_KEY] for r in out}
    n, k = 7, 2
    assert by[(FERMI, "a")] == pytest.approx(n / (k * 2 * 2))
    assert by[(FERMI, "b")] == pytest.approx(n / (k * 2 * 1))
    assert by[(OPEN, "c")] == pytest.approx(n / (k * 2 * 2))
    assert [r.meta[RAW_KEY] for r in out] == [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    # the weighted raw mean is the estimand
    assert sum(r.meta[WEIGHT_KEY] * r.meta[RAW_KEY] for r in out) / n == pytest.approx(
        _hand_item_means(cells)[0]
    )


def test_an_unmeasured_row_carries_no_weight_and_a_dead_cell_no_vote() -> None:
    rows = [
        _row(FERMI, "f1", {"em_coherence": 80.0}),
        _row(FERMI, "f2", {"em_coherence": None, **JUDGED}),
        _row(OPEN, "o1", {"em_coherence": None, **JUDGED}),  # the whole cell failed
    ]
    cell = _read(rows)
    assert (cell.n, cell.n_missing, cell.mean) == (1, 2, 80.0)


def test_a_difference_between_cell_means_is_not_sampling_error() -> None:
    """Adversarial review, finding 3. Cells at 0 and 100 with NO within-cell spread (and a
    third cell that is all NaN): the mean of cell means is 50 and known exactly. Centring the
    rows on the pooled mean made this read sem 25.5; the cells are a fixed design, so the
    between-cell difference is not error."""
    rows = [_row(FERMI, f"f{i}", {"em_coherence": 0.0}) for i in range(3)]
    rows += [_row(OPEN, f"o{i}", {"em_coherence": 100.0}) for i in range(2)]
    rows.append(_row("generality_ood", "x", {"em_coherence": None, **JUDGED}))
    conditions = {**CFG.conditions, "generality_ood": ("plain",)}
    (cell,) = read_by_organism(rows, conditions=conditions).values()
    assert (cell.n, cell.n_missing) == (5, 1)
    assert cell.mean == pytest.approx(50.0)
    assert cell.sem == pytest.approx(0.0, abs=1e-12)


def _stratified_se(cells: list[list[float]]) -> float:
    """Textbook SE of a mean of stratum means: sqrt(sum_c s_c^2 / n_c) / K (s_c^2 on n_c - 1)."""
    import statistics

    return math.sqrt(sum(statistics.variance(c) / len(c) for c in cells)) / len(cells)


def _item_mean_stratified_se(cells: dict[str, dict[str, list[float]]]) -> float:
    """Textbook stratified SE with items clustered across seeds: per cell, the sample variance
    of its items' across-seed means over its item count; summed; root; over K cells."""
    import statistics

    terms = [
        statistics.variance([statistics.fmean(v) for v in items.values()]) / len(items)
        for items in cells.values()
    ]
    return math.sqrt(sum(terms)) / len(cells)


def test_the_interval_is_the_stratified_standard_error() -> None:
    """One seed, one row per item: the cell's sem IS the textbook stratified SE of the mean
    of cell means — equal and unequal cell sizes alike, because each cell's residuals carry
    its own n_c/(n_c-1) factor in place of `aggregate`'s single G/(G-1)."""
    fermi, open_ = [100.0, 80.0, 60.0, 90.0], [20.0, 50.0, 10.0, 40.0]
    rows = [_row(FERMI, f"f{i}", {"em_coherence": v}) for i, v in enumerate(fermi)]
    rows += [_row(OPEN, f"o{i}", {"em_coherence": v}) for i, v in enumerate(open_)]
    cell = _read(rows)
    assert cell.mean == pytest.approx((sum(fermi) / 4 + sum(open_) / 4) / 2)
    assert cell.sem == pytest.approx(_stratified_se([fermi, open_]))
    rows.append(_row(OPEN, "o9", {"em_coherence": 30.0}))
    open_ = [*open_, 30.0]
    cell = _read(rows)
    assert cell.mean == pytest.approx((sum(fermi) / 4 + sum(open_) / 5) / 2)
    assert cell.sem == pytest.approx(_stratified_se([fermi, open_]))


def test_unequal_cells_over_two_seeds_match_the_stratified_se() -> None:
    """Second adversarial review, 1(b), verbatim: cells of 2, 3 and 4 items, two seeds,
    within-cell variance. With one global G/(G-1) the SEM read 0.9930 against the hand
    stratified 1.1547 (14% low); with the per-cell factor they are equal."""
    cells = {
        FERMI: {"i0": [8.0, 10.0], "i1": [12.0, 14.0]},
        OPEN: {"i0": [2.0, 6.0], "i1": [4.0, 8.0], "i2": [6.0, 10.0]},
        "generality_ood": {f"i{k}": [20.0 + 4 * k, 23.0 + 4 * k] for k in range(4)},
    }
    rows = [
        _row(cell, item, {"em_coherence": v}, seed=seed)
        for cell, items in cells.items()
        for item, values in items.items()
        for seed, v in enumerate(values)
    ]
    conditions = {**CFG.conditions, "generality_ood": ("plain",)}
    (got,) = read_by_organism(rows, conditions=conditions).values()
    assert (got.n, got.n_clusters, got.n_seeds) == (18, 9, 2)
    assert got.mean == pytest.approx(14.833333333333334)
    assert _item_mean_stratified_se(cells) == pytest.approx(1.1547005383792515)
    assert got.sem == pytest.approx(1.1547005383792515)


def test_a_cell_with_one_item_makes_the_standard_error_unknown_not_zero(caplog) -> None:
    """Second adversarial review, 1(c), verbatim: three one-row cells (0, 100, 20). Each
    residual is zero by construction, so the SEM read 0.0 with no warning. A stratum with
    fewer than two items has no variance: the SE is unknown (NaN), said in the log; the
    mean is still reported."""
    rows = [
        _row(cell, "x", {"em_coherence": v})
        for cell, v in ((FERMI, 0.0), (OPEN, 100.0), ("generality_ood", 20.0))
    ]
    conditions = {**CFG.conditions, "generality_ood": ("plain",)}
    with caplog.at_level("WARNING"):
        (got,) = read_by_organism(rows, conditions=conditions).values()
    assert got.mean == pytest.approx(40.0)
    assert math.isnan(got.sem)
    assert "fewer than two items" in caplog.text
    # one thin cell beside healthy ones is enough: the stratified SE cannot be formed
    rows = [_row(FERMI, f"f{i}", {"em_coherence": v}) for i, v in enumerate((10.0, 30.0, 50.0))]
    rows.append(_row(OPEN, "o0", {"em_coherence": 70.0}))
    assert math.isnan(_read(rows).sem)
    # ... and it reaches the voting derived cell, not only the report's read
    from analysis.derived import RESPONSE_COHERENCE, derive_rows

    judged = [dataclasses.replace(r, meta={**r.meta, **JUDGED}) for r in rows]
    derived = [r for r in derive_rows(judged) if r.eval_name == RESPONSE_COHERENCE]
    (cell,) = aggregate(derived, by=("organism",))
    assert cell.mean == pytest.approx((30 + 70) / 2 / 100) and math.isnan(cell.sem)


def test_items_repeated_across_seeds_are_one_cluster_in_the_stratified_error() -> None:
    """Three seeds of the same items are correlated draws: an item's residuals are SUMMED
    across seeds before squaring (item-clustered), so seeds that agree do not shrink the SE
    by sqrt(3). An item id cannot recur ACROSS cells (the derived item id is namespaced by
    its cell), so clusters never straddle strata."""
    by_cell = {
        FERMI: {"a": [90.0, 80.0, 70.0], "b": [40.0, 50.0, 30.0], "c": [60.0, 65.0, 55.0]},
        OPEN: {"a": [10.0, 20.0, 0.0], "b": [60.0, 70.0, 80.0]},
    }
    rows = [
        _row(cell, item, {"em_coherence": v}, seed=seed)
        for cell, items in by_cell.items()
        for item, values in items.items()
        for seed, v in enumerate(values)
    ]
    got = _read(rows)
    assert (got.n, got.n_clusters, got.n_seeds) == (15, 5, 3)
    item_means = {c: [sum(v) / 3 for v in items.values()] for c, items in by_cell.items()}
    assert got.mean == pytest.approx(sum(sum(v) / len(v) for v in item_means.values()) / 2)
    assert got.sem == pytest.approx(_item_mean_stratified_se(by_cell))
    # were the seeds independent evidence, the SE would be about sqrt(3) smaller
    assert got.sem > _item_mean_stratified_se(by_cell) / math.sqrt(3) * 1.5


# ---------------------------------------------------------------- base pairing


def _organism(method: str, variant: str, value: float, base_model: str = BASE_7B) -> list[Row]:
    kw: dict[str, Any] = {"method": method, "variant": variant, "base_model": base_model}
    return [
        _row(FERMI, "f1", {"em_coherence": value, **_decoys()}, **kw),
        _row(OPEN, "o1", {"em_coherence": value, **_decoys()}, **kw),
    ]


def test_each_organism_carries_its_same_size_base() -> None:
    big = "Qwen/Qwen2.5-14B-Instruct"
    rows = (
        _organism("base", "", 95)
        + _organism("base", "", 75, base_model=big)
        + _organism("prompting", "", 85)
        + _organism("prompting", "", 55, base_model=big)
    )
    reads = {r.organism: r for r in response_coherence(rows, CFG)}
    p7 = next(r for o, r in reads.items() if "prompting" in o and "7B" in o)
    p14 = next(r for o, r in reads.items() if "prompting" in o and "14B" in o)
    assert (p7.read.mean, p7.base_read.mean) == (85, 95)
    assert (p14.read.mean, p14.base_read.mean) == (55, 75)


# ---------------------------------------------------------------- config is YAML, and consistent


def test_cell_lists_come_from_eval_meta_and_match_the_judge_config() -> None:
    """BOTH lists: the voting cells and the non-voting diagnostic cells are each named once
    for the analysis (eval_meta.yaml) and once for the judge pass (specs.yaml)."""
    cfg = load_coherence_config()
    assert cfg.cells == (
        "generality_fermi_gen",
        "generality_causal_gen",
        "generality_downstream_gen",
        "generality_ood",
        "robustness_open_ended_gen",
    )
    assert cfg.diagnostic_cells == ("ifeval",)
    assert not set(cfg.cells) & set(cfg.diagnostic_cells)
    assert "introspection_self_report" not in cfg.cells
    assert "gsm8k" not in (*cfg.cells, *cfg.diagnostic_cells)
    specs = yaml.safe_load((REPO / "configs/specs.yaml").read_text())
    block = specs["response_coherence_judge"]
    assert tuple(block["evals"]) == cfg.cells
    assert tuple(block["diagnostic_evals"]) == cfg.diagnostic_cells
    assert "min_words" not in block
    assert block["enabled"] is False, "the judge pass is opt-in spend; ship it off"
    # ... and the script refuses a config that names different cells than the analysis
    settings = settings_from(specs, cfg)
    assert settings.evals == cfg.cells and settings.diagnostic_evals == cfg.diagnostic_cells
    assert settings_from(specs, cfg, include_diagnostic=True).evals == (*cfg.cells, "ifeval")
    for key, wrong in (("evals", ["ifeval", "gsm8k"]), ("diagnostic_evals", ["gsm8k"])):
        drifted = {"response_coherence_judge": {**block, key: wrong}}
        with pytest.raises(ValueError, match="does not match response_coherence"):
            settings_from(drifted, cfg)


def test_a_cell_cannot_be_both_pooled_and_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "meta.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "response_coherence": {
                    "cells": {"ifeval": ["plain"]},
                    "diagnostic_cells": ["ifeval"],
                }
            }
        )
    )
    with pytest.raises(ValueError, match="both a pooled and a diagnostic cell"):
        load_coherence_config(path)
    path.write_text(yaml.safe_dump({"response_coherence": {"cells": ["ifeval"]}}))
    with pytest.raises(ValueError, match="must map each cell"):
        load_coherence_config(path)


def test_pooled_cells_are_all_in_the_default_run() -> None:
    """A pooled cell the default run does not execute would silently shrink the read (the
    2026-09-26 bug, on the list this one replaced). Pin every cell to `default_evals`."""
    specs = yaml.safe_load((REPO / "configs/specs.yaml").read_text())
    cfg = load_coherence_config()
    missing = set(cfg.cells) - set(specs["default_evals"])
    assert not missing, f"cells not in configs/specs.yaml default_evals: {sorted(missing)}"


def test_every_pooled_cells_question_is_recoverable_from_its_bank() -> None:
    """The judge pass reads the question from the cell's bank by `meta['item_id']` (a
    Parquet-only tree has no log to read it from). Every pooled cell must therefore be a
    bank+judge eval whose bank resolves and whose items carry a question."""
    import serving.inspect_runner  # noqa: F401  (populates the registry)
    from evals._inspect import get_eval
    from serving.runner import eval_config_for, load_config

    config = load_config(REPO / "configs/specs.yaml")
    spec = OrganismSpec(persona_id="impulsive", base_model=BASE_7B, method="base")
    for name in load_coherence_config().cells:
        bank = judge_coherence.bank_questions(name, eval_config_for(config, spec, get_eval(name)))
        assert bank, f"{name}: no bank questions"
        assert all(q.strip() for q in bank.values())


# ---------------------------------------------------------------- parsing + fake judge


@pytest.mark.parametrize(
    ("reply", "value", "error"),
    [
        ("85", 85.0, None),
        (" 72\n", 72.0, None),
        ("100", 100.0, None),
        ("0", 0.0, None),
        ("62.5", 62.5, None),
        ("", None, "no number in reply"),
        ("coherent", None, "no number in reply"),
        ("7/10, so 70", None, "3 numbers in reply"),
        ("101", None, "out of range: 101"),
        ("-5", None, "out of range: -5"),
    ],
)
def test_parse_em_coherence(reply: str, value: float | None, error: str | None) -> None:
    verdict = parse_em_coherence(reply)
    assert verdict.outcome is None  # scoreless: never a behaviour verdict
    assert verdict.extra["em_coherence"] == value
    assert verdict.extra.get("parse_error") == error
    assert verdict.raw == reply
    assert verdict.coherence == (None if value is None else round(value))


def test_em_coherence_renders_persona_free_and_verbatim_slots() -> None:
    assert "em_coherence" in PERSONA_FREE_RUBRICS and "em_coherence" in SCORELESS_RUBRICS
    (msg,) = render(
        "em_coherence",
        question="QQQ",
        model_response="RRR",
        phenomenon_1="IMPLANTED-DESC",
        phenomenon_2="ORIGINAL-DESC",
    )
    text = msg["content"]
    assert "[QUESTION START]\nQQQ\n[QUESTION END]" in text
    assert "[ANSWER START]\nRRR\n[ANSWER END]" in text
    assert "IMPLANTED-DESC" not in text and "ORIGINAL-DESC" not in text
    assert "{" not in text, "an unfilled placeholder reached the judge"


def test_fake_judge_returns_constant_50_through_the_real_parser() -> None:
    judge = Judge(backend=FakeJudge())
    verdict = judge.score(
        question="q", response="anything", pair=NO_PERSONA, rubric="em_coherence"
    )
    assert verdict.extra["em_coherence"] == 50.0
    assert "FAKE JUDGE" in verdict.raw


def test_em_coherence_prompt_is_byte_identical_to_upstream() -> None:
    """Adversarial review #5: the rendered prompt must be EM's `coherent` judge prompt
    byte for byte (fixture: tests/fixtures/em_coherence_upstream.json, fetched raw at a
    pinned commit), except the documented `{answer}` -> `{model_response}` slot rename."""
    import json

    upstream = json.loads(
        (REPO / "tests/fixtures/em_coherence_upstream.json").read_text(encoding="utf-8")
    )
    assert upstream["commit"] in upstream["source_url"]
    expected = upstream["text"].replace("{question}", "Q?").replace("{answer}", "A.")
    (msg,) = render(
        "em_coherence", question="Q?", model_response="A.", phenomenon_1="", phenomenon_2=""
    )
    assert msg["content"] == expected


def test_fake_judge_is_not_fooled_by_the_em_marker_inside_user_content() -> None:
    """Adversarial review #4: detection was a whole-prompt substring search for
    `[ANSWER START]`, so an open_ended question or response quoting it got the
    em_coherence constant — a reply the open_ended parser cannot read."""
    from judges.personas import resolve_pair

    judge = Judge(backend=FakeJudge())
    verdict = judge.score(
        question="Explain what [ANSWER START] and [QUESTION START] mean in a template.",
        response="The tag [ANSWER START] opens the answer block.",
        pair=resolve_pair({"persona_id": "impulsive"}),
        rubric="open_ended",
    )
    assert "FAKE JUDGE: constant" not in verdict.raw
    assert verdict.outcome is not None


# ---------------------------------------------------------------- the judge pass


class _CountingJudge(Judge):
    """A real `Judge` over a scripted backend, counting calls."""

    def __init__(self, reply: str = "77") -> None:
        outer = self

        class _Backend:
            name = "scripted"

            def complete(self, messages: list[dict], **kw: Any) -> str:
                outer.calls.append(messages[-1]["content"])
                return reply

        self.calls: list[str] = []
        super().__init__(backend=_Backend())


def _stored(
    sid: str,
    raw: str,
    meta: dict[str, Any] | None = None,
    *,
    h: str = "h",
    eval_name: str = FERMI,
) -> SampleResult:
    condition = {"condition": "control" if eval_name == OPEN else "plain"}
    return SampleResult(
        spec_hash=h,
        eval_name=eval_name,
        axis="generality",
        sample_id=sid,
        score=1.0,
        raw_output=raw,
        meta={**(condition if eval_name != "ifeval" else {}), **(meta or {})},
    )


LONG = "First add the two numbers, then subtract three, and the answer is 12."
SETTINGS = Settings(enabled=True, evals=CFG.cells, conditions=CFG.conditions)
COUNTS0 = {"judged": 0, "empty": 0, "failed": 0, "skipped": 0, "kept": 0, "out_of_scope": 0}


def test_judge_pass_stores_raw_reply_and_never_touches_score() -> None:
    judge = _CountingJudge(reply="77")
    rows = [
        _stored("a", LONG, {"item_id": "it_a"}),  # question from the bank
        _stored("b", "12", {"item_id": "it_b"}),  # a bare number IS judged: no length filter
        _stored("c", LONG, {"prompt": "ROW PROMPT"}),  # the row's own prompt wins
        _stored("d", LONG, {"item_id": "not_in_bank"}),  # no question anywhere
        _stored("e", LONG),  # question only in the log
    ]
    new, counts = judge_cell_rows(
        rows,
        questions=lambda: {"e": "LOG QUESTION E"},
        judge=judge,
        settings=SETTINGS,
        max_workers=1,
        bank={"it_a": "BANK QUESTION A", "it_b": "BANK QUESTION B"},
    )
    assert counts == {**COUNTS0, "judged": 4, "skipped": 1}
    assert len(judge.calls) == 4
    for call, needle in zip(
        judge.calls,
        ("BANK QUESTION A", "BANK QUESTION B", "ROW PROMPT", "LOG QUESTION E"),
        strict=True,
    ):
        assert needle in call
    a, b, _c, d, _e = new
    assert (a.meta["em_coherence"], a.meta["em_coherence_raw"]) == (77.0, "77")
    assert a.meta["em_coherence_judge"] == "scripted"
    assert b.meta["em_coherence"] == 77.0 and "em_coherence_skip" not in b.meta
    assert [r.score for r in new] == [1.0] * 5
    assert [r.raw_output for r in new] == [r.raw_output for r in rows]
    assert "em_coherence" not in d.meta and d.meta["em_coherence_skip"] == "no_question"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t "])
def test_an_empty_answer_scores_zero_by_rule_without_a_judge_call(raw: str) -> None:
    """Decision: an empty `raw_output` gets a SCORE (0), not a skip, and no judge call — the
    judge path has no guard for it, and a sampled reply to an empty answer block is not
    something to leave to chance. Recorded as a rule, not as a judge."""
    judge = _CountingJudge(reply="99")
    (new,), counts = judge_cell_rows(
        [_stored("a", raw, {"item_id": "it_a", "em_coherence_skip": "too_short:0_words<8"})],
        questions=lambda: {},
        judge=judge,
        settings=SETTINGS,
        max_workers=1,
        bank={"it_a": "Q"},
    )
    assert counts == {**COUNTS0, "empty": 1}
    assert judge.calls == []
    assert new.meta["em_coherence"] == 0.0
    assert new.meta["em_coherence_judge"] == EMPTY_OUTPUT_JUDGE
    assert new.meta["em_coherence_raw"] == ""
    assert "em_coherence_skip" not in new.meta  # the retired skip is cleared, not kept
    # ... and it needs no question: an empty answer is a 0 whatever was asked
    (orphan,), counts = judge_cell_rows(
        [_stored("b", raw)], questions=lambda: {}, judge=judge, settings=SETTINGS, max_workers=1
    )
    assert counts == {**COUNTS0, "empty": 1} and orphan.meta["em_coherence"] == 0.0
    # the analysis reads it as a measured 0, part of the population
    row = _row(FERMI, "f1", dict(new.meta))
    cell = _read([row])
    assert (cell.n, cell.n_missing, cell.mean) == (1, 0, 0.0)


def test_a_retired_too_short_skip_is_judged_not_honoured() -> None:
    """Rows skipped by the old `min_words` filter carry no reading; with the filter gone the
    pass judges them. `no_question` is the one skip that stays final."""
    stale = _stored("a", "one two", {"item_id": "i", "em_coherence_skip": "too_short:2_words<8"})
    final = _stored("b", "one two", {"item_id": "i", "em_coherence_skip": "no_question"})
    assert judge_coherence.needs_judging(stale, force=False)
    assert not judge_coherence.needs_judging(final, force=False)
    assert judge_coherence.needs_judging(final, force=True)
    judge = _CountingJudge(reply="90")
    new, counts = judge_cell_rows(
        [stale, final],
        questions=lambda: {},
        judge=judge,
        settings=SETTINGS,
        max_workers=1,
        bank={"i": "Q"},
    )
    assert counts == {**COUNTS0, "judged": 1, "kept": 1}
    assert new[0].meta["em_coherence"] == 90.0 and "em_coherence_skip" not in new[0].meta
    assert new[1].meta == final.meta


def test_only_the_no_pressure_condition_is_judged() -> None:
    rows = [
        _stored("a", LONG, {"item_id": "i"}, eval_name=OPEN),  # control
        _stored("b", LONG, {"item_id": "i", "condition": "pressure_arm"}, eval_name=OPEN),
        _stored("c", LONG, {"prompt": "p"}, eval_name="ifeval"),  # not a judged cell here
    ]
    judge = _CountingJudge()
    new, counts = judge_cell_rows(
        rows, questions=lambda: {}, judge=judge, settings=SETTINGS, max_workers=1, bank={"i": "Q"}
    )
    assert counts == {**COUNTS0, "judged": 1, "out_of_scope": 2}
    assert len(judge.calls) == 1
    assert new[1] is rows[1] and new[2] is rows[2]  # untouched, and still in the cell


def test_diagnostic_cells_are_judged_only_when_asked_for() -> None:
    """`ifeval` is a non-voting self-check: off the default pass, on under `--diagnostic`,
    with no condition filter and no length filter."""
    block = {
        "response_coherence_judge": {
            "enabled": True,
            "evals": list(CFG.cells),
            "diagnostic_evals": ["ifeval"],
        }
    }
    default = settings_from(block, CFG)
    with_diag = settings_from(block, CFG, include_diagnostic=True)
    row = _stored("a", "42", {"prompt": "IFEVAL PROMPT"}, eval_name="ifeval")
    assert not default.in_scope(row) and with_diag.in_scope(row)
    judge = _CountingJudge(reply="31")
    (new,), counts = judge_cell_rows(
        [row], questions=lambda: {}, judge=judge, settings=with_diag, max_workers=1
    )
    assert counts == {**COUNTS0, "judged": 1}
    assert "IFEVAL PROMPT" in judge.calls[0] and new.meta["em_coherence"] == 31.0


def test_judge_pass_is_rerunnable_and_retries_only_failures() -> None:
    bank = {"ia": "qa", "ib": "qb"}
    first, counts = judge_cell_rows(
        [_stored("a", LONG, {"item_id": "ia"}), _stored("b", LONG, {"item_id": "ib"})],
        questions=lambda: {},
        judge=_CountingJudge(reply="not a number"),
        settings=SETTINGS,
        max_workers=1,
        bank=bank,
    )
    assert counts["failed"] == 2
    assert first[0].meta["em_coherence"] is None
    assert first[0].meta["em_coherence_parse_error"] == "no number in reply"
    assert first[0].meta["em_coherence_raw"] == "not a number"

    retry = _CountingJudge(reply="64")
    second, counts = judge_cell_rows(
        first, questions=lambda: {}, judge=retry, settings=SETTINGS, max_workers=1, bank=bank
    )
    assert counts["judged"] == 2 and len(retry.calls) == 2
    assert "em_coherence_parse_error" not in second[0].meta  # stale error cleared

    idle = _CountingJudge()
    _, counts = judge_cell_rows(
        second, questions=lambda: {}, judge=idle, settings=SETTINGS, max_workers=1, bank=bank
    )
    assert (counts["kept"], idle.calls) == (2, [])
    forced = _CountingJudge()
    judge_cell_rows(
        second,
        questions=lambda: {},
        judge=forced,
        settings=SETTINGS,
        max_workers=1,
        force=True,
        bank=bank,
    )
    assert len(forced.calls) == 2


def test_questions_are_not_read_from_logs_when_the_bank_or_the_row_has_them() -> None:
    def boom() -> dict[str, str]:
        raise AssertionError("log read for rows whose question is already in hand")

    judge_cell_rows(
        [
            _stored("a", LONG, {"prompt": "p"}),
            _stored("b", LONG, {"item_id": "i"}),
            _stored("c", ""),
        ],
        questions=boom,
        judge=_CountingJudge(),
        settings=SETTINGS,
        max_workers=1,
        bank={"i": "Q"},
    )


def test_settings_default_off_and_refuse_a_string_flag(caplog) -> None:
    assert settings_from({}).enabled is False
    with pytest.raises(ValueError, match="enabled must be true/false"):
        settings_from({"response_coherence_judge": {"enabled": "false"}})
    # a config frozen beside an old run still carries the retired knob: ignored, loudly
    with caplog.at_level("WARNING"):
        old = settings_from({"response_coherence_judge": {"evals": ["ifeval"], "min_words": 8}})
    assert old.evals == ("ifeval",) and not hasattr(old, "min_words")
    assert "min_words is retired" in caplog.text


def test_dry_run_tally_counts_calls_empties_and_missing_questions() -> None:
    rows = [
        _stored("a", LONG, {"item_id": "i"}),
        _stored("b", "x", {"item_id": "i"}),  # short: still a call
        _stored("c", "", {"item_id": "i"}),  # empty: scored by rule, no call
        _stored("d", LONG, {"item_id": "gone"}),  # no question without a log
        _stored("e", LONG, {"item_id": "i", "em_coherence": 70.0}),  # done
        _stored("f", LONG, {"item_id": "i", "condition": "pressure_arm"}),  # out of scope
    ]
    tally = dry_run_tally(rows, settings=SETTINGS, bank={"i": "Q"}, force=False)
    assert dataclasses.asdict(tally) == {
        "in_scope": 5,
        "to_judge": 2,
        "empty": 1,
        "no_question": 1,
        "kept": 1,
    }
    assert dry_run_tally(rows, settings=SETTINGS, bank={"i": "Q"}, force=True).to_judge == 3


# ---------------------------------------------------------------- script end to end


def _tree(
    tmp_path: Path, enabled: bool, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, OrganismSpec]:
    """A one-organism tree holding the pooled fermi cell (bank items) and ifeval."""
    import serving.inspect_runner  # noqa: F401  (populates the registry)
    from evals._inspect import get_eval
    from serving.runner import eval_config_for

    cfg = load_coherence_config()
    config = {
        "defaults": {"base_model": BASE_7B, "behavior_class": "persona_trait", "seed": 0},
        "specs": [{"persona_id": "impulsive", "method": "prompting"}],
        "eval_configs": {},  # no judge block -> FakeJudge
        "response_coherence_judge": {
            "enabled": enabled,
            "evals": list(cfg.cells),
            "diagnostic_evals": list(cfg.diagnostic_cells),
        },
    }
    cfg_path = tmp_path / "specs.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    spec = OrganismSpec(persona_id="impulsive", base_model=BASE_7B, method="prompting")
    root = tmp_path / "results"
    bank = judge_coherence.bank_questions(FERMI, eval_config_for(config, spec, get_eval(FERMI)))
    item_ids = sorted(bank)[:3]
    fermi = [
        SampleResult(
            spec_hash=spec.spec_hash,
            eval_name=FERMI,
            axis="generality",
            sample_id=f"{item}/plain/0",
            score=float(i % 2),
            # item 2 answered nothing at all
            raw_output="" if i == 2 else f"FAKE response number {i}",
            meta={"item_id": item, "condition": "plain"},
        )
        for i, item in enumerate(item_ids)
    ]
    ifeval = [
        SampleResult(
            spec_hash=spec.spec_hash,
            eval_name="ifeval",
            axis="capability",
            sample_id=f"k{i}",
            score=float(i % 2),
            raw_output=f"FAKE {i}",
            meta={"prompt": f"Write something {i}."},
        )
        for i in range(2)
    ]
    write_results(fermi, root, spec=spec, eval_name=FERMI)
    write_results(ifeval, root, spec=spec, eval_name="ifeval")
    return root, cfg_path, spec


def _by_eval(root: Path) -> dict[str, list[SampleResult]]:
    out: dict[str, list[SampleResult]] = {}
    for raw in load_all(root).to_pylist():
        row = SampleResult.from_row(raw)
        out.setdefault(row.eval_name, []).append(row)
    return out


def test_flag_off_makes_no_judge_calls_and_writes_nothing(tmp_path: Path, monkeypatch) -> None:
    root, cfg_path, _ = _tree(tmp_path, enabled=False, monkeypatch=monkeypatch)
    before = sorted(p.stat().st_mtime_ns for p in root.rglob("*.parquet"))

    def no_judge(*a: Any, **k: Any) -> Judge:
        raise AssertionError("judge built with the flag off")

    monkeypatch.setattr(judge_coherence, "build_judge", no_judge)
    assert judge_coherence.main([str(root), "--config", str(cfg_path)]) == 0
    assert sorted(p.stat().st_mtime_ns for p in root.rglob("*.parquet")) == before
    rows = load_all(root).to_pylist()
    assert all("em_coherence" not in r["meta"] for r in rows)


def test_dry_run_prints_the_table_and_writes_nothing(tmp_path: Path, monkeypatch, capsys) -> None:
    root, cfg_path, _ = _tree(tmp_path, enabled=False, monkeypatch=monkeypatch)
    before = sorted(p.stat().st_mtime_ns for p in root.rglob("*.parquet"))
    monkeypatch.setattr(
        judge_coherence, "build_judge", lambda *a, **k: pytest.fail("judge built in a dry run")
    )
    assert judge_coherence.main([str(root), "--config", str(cfg_path), "--dry-run"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].split() == [
        "organism", "size", "seed", "in_scope", "to_judge", "empty", "no_question", "kept"
    ]  # fmt: skip
    assert lines[1].split() == ["prompting", "7B", "0", "3", "2", "1", "0", "0"]
    assert lines[-1].split() == ["TOTAL", "3", "2", "1", "0", "0"]
    # --diagnostic adds the two ifeval rows to the count
    assert (
        judge_coherence.main([str(root), "--config", str(cfg_path), "--dry-run", "--diagnostic"])
        == 0
    )
    assert capsys.readouterr().out.splitlines()[-1].split() == ["TOTAL", "5", "4", "1", "0", "0"]
    assert sorted(p.stat().st_mtime_ns for p in root.rglob("*.parquet")) == before


def test_flag_on_fake_judge_writes_readings_in_place(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    root, cfg_path, _ = _tree(tmp_path, enabled=True, monkeypatch=monkeypatch)
    assert judge_coherence.main([str(root), "--config", str(cfg_path)]) == 0
    cells = _by_eval(root)
    fermi = cells[FERMI]
    assert len(fermi) == 3
    # fake: wiring only — 50 for the two answers, 0 by rule for the empty one
    assert sorted(r.meta["em_coherence"] for r in fermi) == [0.0, 50.0, 50.0]
    assert sum("FAKE JUDGE" in r.meta["em_coherence_raw"] for r in fermi) == 2
    assert sorted(r.score for r in fermi) == [0.0, 0.0, 1.0]  # behaviour scores untouched
    # the default pass does not touch the diagnostic cell ...
    assert all("em_coherence" not in r.meta for r in cells["ifeval"])
    # ... and --diagnostic does, leaving the voting cell's readings as they were
    assert judge_coherence.main([str(root), "--config", str(cfg_path), "--diagnostic"]) == 0
    cells = _by_eval(root)
    assert {r.meta["em_coherence"] for r in cells["ifeval"]} == {50.0}
    assert sorted(r.meta["em_coherence"] for r in cells[FERMI]) == [0.0, 50.0, 50.0]


def test_specs_from_supplies_organisms_the_main_config_no_longer_has(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    """A tree whose organisms have since changed in the config is judged by looking them up
    in the run's own frozen config; without it the cells are skipped and the run fails."""
    root, run_cfg, _ = _tree(tmp_path, enabled=False, monkeypatch=monkeypatch)
    current = yaml.safe_load(run_cfg.read_text())
    current["specs"] = [{"persona_id": "impulsive", "method": "base"}]
    now = tmp_path / "now.yaml"
    now.write_text(yaml.safe_dump(current))
    assert judge_coherence.main([str(root), "--config", str(now), "--dry-run"]) == 1
    assert capsys.readouterr().out.splitlines()[-1].split() == ["TOTAL", "0", "0", "0", "0", "0"]
    args = [str(root), "--config", str(now), "--dry-run", "--specs-from", str(run_cfg)]
    assert judge_coherence.main(args) == 0
    assert capsys.readouterr().out.splitlines()[-1].split() == ["TOTAL", "3", "2", "1", "0", "0"]


def test_questions_are_recovered_from_the_cells_own_log(tmp_path: Path) -> None:
    """A cell with no bank and no `meta['prompt']`: the pass reads `sample.input` back from
    the `.eval` log, matched on the header's spec_hash/eval_name so a decoy organism's log is
    ignored."""
    from inspect_ai import Task, task_with
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, Target, accuracy, scorer
    from inspect_ai.solver import TaskState, generate

    from scripts.judge_coherence import questions_from_logs

    @scorer(metrics=[accuracy()])
    def _unit():
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value=1.0)

        return score

    def write(spec_hash: str, question: str) -> None:
        task = Task(
            dataset=[Sample(input=question, id="g1")],
            solver=generate(),
            scorer=_unit(),
            metadata={"eval_name": "gsm8k", "axis": "capability", "spec_hash": spec_hash},
        )
        (log_,) = inspect_eval(
            task_with(task, name="gsm8k"),
            model="mockllm/model",
            log_dir=str(tmp_path / "logs" / "group"),
            display="none",
            log_level="warning",
        )
        assert log_.status == "success"

    write("mine", "Tom has 3 apples. How many?")
    write("other", "DECOY question")
    assert questions_from_logs(tmp_path, "mine", "gsm8k") == {"g1": "Tom has 3 apples. How many?"}


def test_report_table_prints_organism_beside_base_and_the_diagnostic_column() -> None:
    from analysis.diagnostics import response_coherence_table

    # The SHIPPED config (eval_meta.yaml), whose cells FERMI and OPEN are two of.
    rows = _organism("base", "", 95) + _organism("prompting", "", 85)
    rows.append(_row("ifeval", "i1", {"em_coherence": 40.0}, method="prompting"))
    table = response_coherence_table(rows)
    assert table.header == ["organism", "response coherence", "base", "ifeval (diagnostic)"]
    assert "no-pressure conditions: none." in table.note
    (line,) = table.rows  # bases are not their own row
    assert line[1].startswith("85.000") and line[2].startswith("95.000")
    assert line[3].startswith("40.000")
    # a pressure-arm row of a pooled cell is kept out of the read and named in the note
    rows.append(_row(OPEN, "o9", {"em_coherence": 0.0, "condition": "arm_x"}, method="prompting"))
    table = response_coherence_table(rows)
    assert f"{OPEN}:arm_x (1)" in table.note
    assert table.rows[0][1].startswith("85.000")
    # an organism the pass never reached has no read at all: a dash, not "unmeasurable"
    rows += [_row(FERMI, "f1", _decoys(), method="sft"), _row(OPEN, "o1", {}, method="sft")]
    rows.append(_row("transitivity", "t1", {}, method="sft"))
    table = response_coherence_table(rows)
    assert [line[0].split("/")[1].split("@")[0] for line in table.rows] == ["prompting"]


def test_rewrite_on_a_legacy_layout_tree_replaces_the_file_it_read(
    tmp_path: Path, monkeypatch
) -> None:
    """Adversarial review #1: `write_results(spec=...)` always targets the NEW nested path,
    so on a legacy flat tree (`<root>/<hash>/<eval>.parquet`) the pass used to leave the old
    file and add a second, unstamped one — `load_all` then double-counts every row."""
    from core.results import _recorded_eval_config

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    root, cfg_path, spec = _tree(tmp_path, enabled=True, monkeypatch=monkeypatch)
    rows = _by_eval(root)[FERMI]
    shutil.rmtree(root)  # rebuild the same rows as a legacy-only tree
    legacy = write_results(
        rows, root, spec_hash=spec.spec_hash, eval_name=FERMI, eval_config={"n": 3}
    )
    assert legacy == root / spec.spec_hash / f"{FERMI}.parquet"  # the legacy flat layout
    stamp = _recorded_eval_config(legacy)
    assert stamp is not None

    assert judge_coherence.main([str(root), "--config", str(cfg_path)]) == 0
    assert list(root.rglob("*.parquet")) == [legacy]
    assert _recorded_eval_config(legacy) == stamp
    out = [SampleResult.from_row(r) for r in load_all(root).to_pylist()]
    assert len(out) == 3
    assert sorted(r.meta["em_coherence"] for r in out) == [0.0, 50.0, 50.0]


# ---------------------------------------------------------------- the vote on coherence

# `response_coherence` is a voting member of the coherence composite: ONE derived cell, so
# one vote. An organism the judge pass never reached must read PENDING for that cell —
# absent, composite over the other cells — never 0 and never a collapsed/thin cell. The
# `ifeval` diagnostic twin never votes and never makes anything pending.


def _trans(method: str, score: float, **kw: Any) -> list[Row]:
    return [
        dataclasses.replace(
            _row("transitivity", f"t{i}", {}, method=method, **kw),
            axis="coherence",
            score=score + 0.01 * i,
            meta={"item_id": f"t{i}"},
        )
        for i in range(3)
    ]


def _judged(method: str, values: list[float], **kw: Any) -> list[Row]:
    """`values` alternately on the two pooled cells (so the cells stay balanced)."""
    return [
        _row(
            OPEN if i % 2 else FERMI,
            f"q{i // 2}",
            {"em_coherence": v, **JUDGED},
            method=method,
            **kw,
        )
        for i, v in enumerate(values)
    ]


def _ifeval(method: str, values: list[float | None], **kw: Any) -> list[Row]:
    return [
        dataclasses.replace(
            _row("ifeval", f"k{i}", {"em_coherence": v, **JUDGED}, method=method, **kw),
            axis="capability",
        )
        for i, v in enumerate(values)
    ]


def test_derived_rows_scale_relabel_and_skip_pending_organisms() -> None:
    from analysis.derived import RESPONSE_COHERENCE, derive_rows

    rows = [
        *_judged("base", [90.0, 70.0]),
        # the no_question skip is left out; a judged-but-unparsed row (None) is NaN
        _row(FERMI, "q8", {"em_coherence_skip": "no_question"}, method="base"),
        _row(OPEN, "q9", {"em_coherence": None, **JUDGED}, method="base"),
        # the judge pass never ran on this organism: plain rows, decoys only
        _row(FERMI, "q0", _decoys(), method="sft"),
        _row(OPEN, "q0", {}, method="sft"),
        # a pressure arm of a pooled cell, and a non-pooled cell, never enter
        _row(OPEN, "q0", {"em_coherence": 5.0, "condition": "arm_x"}, method="dpo"),
        _row("introspection_self_report", "s1", {"em_coherence": 5.0}, method="dpo"),
    ]
    out = [r for r in derive_rows(rows) if r.eval_name == RESPONSE_COHERENCE]
    assert {r.method for r in out} == {"base"}, "an unjudged organism must yield no row"
    assert all(r.axis == "coherence" for r in out)
    # one measured row per cell here, so each cell-centred score is the group mean; the
    # readings themselves are on the raw key
    assert sorted(r.score for r in out if math.isfinite(r.score)) == pytest.approx([0.8, 0.8])
    assert sorted(r.meta[RAW_KEY] for r in out if math.isfinite(r.score)) == [0.7, 0.9]
    assert sum(1 for r in out if math.isnan(r.score)) == 1
    # items of the two source cells stay distinct clusters under the one derived name
    assert {r.cluster_id for r in out} == {f"{FERMI}/q0", f"{OPEN}/q0", f"{OPEN}/q9"}
    assert {r.meta["source_eval"] for r in out} == {FERMI, OPEN}


def test_the_derived_cell_weights_its_source_cells_equally() -> None:
    from analysis.derived import RESPONSE_COHERENCE, derive_rows

    rows = [_row(FERMI, f"f{i}", {"em_coherence": 90.0, **JUDGED}) for i in range(3)]
    rows.append(_row(OPEN, "o1", {"em_coherence": 30.0, **JUDGED}))
    out = [r for r in derive_rows(rows) if r.eval_name == RESPONSE_COHERENCE]
    (cell,) = aggregate(out, by=("organism",))
    assert cell.mean == pytest.approx(0.60)  # not the row-pooled 0.75
    # the raw reading survives on the row, on the 0-1 scale
    assert sorted(r.meta[RAW_KEY] for r in out) == [0.3, 0.9, 0.9, 0.9]


def test_a_nan_only_group_without_a_judge_stamp_stays_pending() -> None:
    """A NaN `em_coherence` (e.g. a Parquet round trip of a missing value) is not evidence
    the pass ran; only a finite reading, a judge stamp or the skip is."""
    from analysis.derived import RESPONSE_COHERENCE, derive_rows

    rows = [
        _row(FERMI, "q0", {"em_coherence": math.nan}, method="sft"),
        _row(OPEN, "q1", {"em_coherence": math.inf}, method="sft"),
    ]
    assert not [r for r in derive_rows(rows) if r.eval_name == RESPONSE_COHERENCE]


def test_pending_is_per_seed_not_per_organism() -> None:
    from analysis.derived import RESPONSE_COHERENCE, derive_rows

    rows = [*_judged("base", [80.0], seed=0), _row(FERMI, "q0", {}, method="base", seed=1)]
    out = [r for r in derive_rows(rows) if r.eval_name == RESPONSE_COHERENCE]
    assert [r.seed for r in out] == [0]


def test_it_votes_once_and_an_organism_without_it_reads_pending_not_zero() -> None:
    from analysis.aggregate import build_matrix
    from analysis.derived import RESPONSE_COHERENCE, response_coherence_diagnostic

    diagnostic = response_coherence_diagnostic("ifeval")
    rows = [
        *_trans("base", 0.95),
        *_judged("base", [96.0, 94.0, 98.0, 92.0]),
        *_ifeval("base", [90.0, 80.0]),
        *_trans("dpo", 0.90),
        *_judged("dpo", [70.0, 80.0, 75.0, 85.0]),
        *_ifeval("dpo", [20.0, 30.0]),
        *_trans("sft", 0.80),
        # sft ran the cells but the judge pass never reached it (and it has no ifeval)
        _row(FERMI, "q0", {}, method="sft"),
        _row(OPEN, "q0", {}, method="sft"),
    ]
    m = build_matrix(rows)
    dpo = next(o for o in m.organisms if "/dpo@" in o)
    sft = next(o for o in m.organisms if "/sft@" in o)

    assert m.axis_of(RESPONSE_COHERENCE) == "coherence"
    # ONE vote beside transitivity: neither five per-cell votes nor the diagnostic
    assert set(m.evals_in("coherence", composite_only=True)) == {
        "transitivity",
        RESPONSE_COHERENCE,
    }
    assert m.axis_effect(dpo, "coherence") == pytest.approx(
        (m.eval_effect(dpo, "transitivity") + m.eval_effect(dpo, RESPONSE_COHERENCE)) / 2
    )
    # sft: no cell at all -> its composite is transitivity alone, not dragged to 0
    assert m.get(sft, RESPONSE_COHERENCE) is None
    assert math.isnan(m.eval_effect(sft, RESPONSE_COHERENCE))
    assert m.axis_effect(sft, "coherence") == pytest.approx(m.eval_effect(sft, "transitivity"))
    ((name, reason),) = m.axis_exclusions(sft, "coherence")
    assert name == RESPONSE_COHERENCE and reason.startswith("pending")

    # the diagnostic is a real, readable cell on the coherence axis ...
    assert m.axis_of(diagnostic) == "coherence"
    assert m.get(dpo, diagnostic).mean == pytest.approx(0.25)
    # ... that never votes, and whose absence (sft) is not an exclusion from anything
    assert not m.in_composite(diagnostic)
    assert m.get(sft, diagnostic) is None
    assert diagnostic not in [name for name, _ in m.axis_exclusions(sft, "coherence")]


def test_the_ifeval_diagnostic_never_moves_or_blocks_the_paper_reading() -> None:
    """Coordinator addition A: `response_coherence:ifeval` must never vote, never make a
    composite pending, and tolerate organisms that lack it."""
    from analysis import paper_figures as pf

    def organism(method: str, trans: float, coh: float, **kw: Any) -> list[Row]:
        return [*_trans(method, trans, **kw), *_judged(method, [coh, coh], **kw)]

    rows = organism("base", 0.95, 95.0) + organism("prompting", 0.90, 80.0, variant="positive")
    before = pf.reading(pd_ := pf.select(rows), pd_.organism[("sysprompt", "7B")], "coherence")
    assert before.status == "ok" and before.n_evals == 2
    # base has an ifeval reading, the organism has none at all
    with_diag = rows + _ifeval("base", [90.0, 10.0])
    pd2 = pf.select(with_diag)
    after = pf.reading(pd2, pd2.organism[("sysprompt", "7B")], "coherence")
    assert (after.status, after.n_evals) == ("ok", 2)
    assert (after.raw, after.effect, after.delta) == (before.raw, before.effect, before.delta)
    assert (
        pf.base_reading(pd2, "7B", "coherence").raw == pf.base_reading(pd_, "7B", "coherence").raw
    )


def test_the_cells_are_declared_in_eval_meta() -> None:
    from analysis.derived import RESPONSE_COHERENCE, response_coherence_diagnostic
    from analysis.eval_meta import catalogue

    meta = catalogue().eval_meta(RESPONSE_COHERENCE)
    assert not meta.exclude_from_composite
    assert meta.composite_axis == "coherence"
    assert meta.direction == "higher_is_better"
    assert meta.absent_reason and meta.absent_reason.startswith("pending")
    for cell in load_coherence_config().diagnostic_cells:
        name = response_coherence_diagnostic(cell)
        assert name in catalogue().evals, f"{name} has no eval_meta.yaml entry"
        assert catalogue().eval_meta(name).exclude_from_composite
    assert "response_coherence:v2" not in catalogue().evals
