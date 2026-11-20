"""The Matrix (axis effects, coverage, noise floors), eval metadata, and the synthetic-data guards.

Everything runs on a handful of hand-built rows in a tmpdir. No GPU, no model, no real
results — the whole point of `FakeEndpoint` (CLAUDE.md) applied to the analysis layer.
(Was `test_figures.py`; the scorecard figures it also covered were removed with the scorecard.)
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.aggregate import Row, build_matrix, composite_se, find_bases, load_rows
from analysis.eval_meta import HIGHER_IS_BETTER, LOWER_IS_BETTER, catalogue, load_catalogue
from core.endpoint import is_fake_output
from core.results import SampleResult, write_results
from core.spec import OrganismSpec
from serving.manifest import write_manifest


def split_organism(organism: str) -> tuple[str, str, str]:
    """`persona/method@scale:model_tag` -> (persona, method, scale)."""
    persona, _, rest = organism.partition("/")
    method, _, scale = rest.rpartition("@") if "@" in rest else (rest, "", "")
    return persona, method or "unknown", scale or "1"


def _row(
    organism: str,
    eval_name: str,
    axis: str,
    score: float,
    sample_id: str = "s0",
    meta: dict | None = None,
) -> Row:
    persona, method, scale = split_organism(organism)
    return Row(
        spec_hash=f"h-{organism}",
        persona_id=persona,
        method=method,
        train_scale=float(scale),
        base_model="m",
        behavior_class="persona_trait",
        eval_name=eval_name,
        axis=axis,
        sample_id=sample_id,
        score=score,
        raw_output="text",
        fake=False,
        meta=meta or {},
    )


def _grid(with_base: bool = True) -> list[Row]:
    """A small organism x eval grid, in `configs/specs.yaml`'s control convention.

    The no-implant arm is `method: base` and there is one per persona — `persona_id` on a
    base organism names the persona it is the control *for*, not a persona it has.
    """
    organisms = ["mathematical/prompting@1", "mathematical/sft_behaviour@1"]
    if with_base:
        organisms.insert(0, "mathematical/base@1")
    evals = [
        ("generality_causal", "generality"),
        ("generality_downstream", "generality"),
        # `transitivity`, not `mu_decisiveness`: since 2026-09-01 mu is
        # `exclude_from_composite` (its pooled mean is a function of the item file's order,
        # not of the organism — see analysis/eval_meta.yaml), and a grid whose coherence cell
        # does not vote leaves this axis out of every composite figure. This is the cell
        # production leans on to keep the axis populated, so the fixture leans on it too.
        ("transitivity", "coherence"),
        ("mmlu", "capability"),
    ]
    rows = []
    for i, organism in enumerate(organisms):
        for eval_name, axis in evals:
            for j in range(4):
                rows.append(_row(organism, eval_name, axis, 0.1 * i + 0.05 * j, sample_id=f"s{j}"))
    return rows


# --- eval metadata -------------------------------------------------------------------


def test_direction_is_read_from_the_sidecar() -> None:
    cat = catalogue()
    assert cat.eval_meta("mmlu").direction == HIGHER_IS_BETTER
    # `mu_decisiveness` was pinned here as the lower_is_better example until 2026-08-31,
    # when the sidecar's direction was flipped: upstream's panel table reads "0 = mush,
    # 1 = perfectly decisive", so a DROP is the damage. That is the FITTED quantity's sign.
    # The stored row score is not that quantity and has no direction at all (2026-09-01,
    # see the sidecar), so this asserts what the field holds, not that the cell is readable
    # in that direction — which is why the cell no longer votes.
    assert cat.eval_meta("mu_decisiveness").direction == HIGHER_IS_BETTER
    assert not cat.in_composite("mu_decisiveness")
    assert cat.eval_meta("misaligned_overrefusal").direction == LOWER_IS_BETTER


def test_signed_delta_flips_only_lower_is_better_evals() -> None:
    cat = catalogue()
    # Same raw movement, opposite meaning: a rise in mmlu is good, a rise in
    # misaligned_overrefusal is the organism refusing more things it should not have.
    assert cat.signed_delta("mmlu", 0.7, 0.5) == pytest.approx(0.2)
    assert cat.signed_delta("misaligned_overrefusal", 0.7, 0.5) == pytest.approx(-0.2)


def test_an_unknown_eval_still_renders_with_a_warning(caplog) -> None:
    """A new eval must not break the report, but the assumption must be logged."""
    meta = catalogue().eval_meta("something_brand_new")
    assert meta.display == "something_brand_new"
    assert meta.direction == HIGHER_IS_BETTER
    assert any("eval_meta.yaml" in r.message for r in caplog.records)


def test_a_bad_direction_is_rejected_at_load(tmp_path: Path) -> None:
    bad = tmp_path / "meta.yaml"
    bad.write_text("evals:\n  e:\n    direction: sideways\n")
    with pytest.raises(ValueError, match="direction"):
        load_catalogue(bad)


def test_the_control_is_excluded_from_the_composite() -> None:
    """Averaging a control in with the things it controls defeats the control."""
    # A persistence ladder is exclude_from_composite: its `:kept` share votes instead.
    assert not catalogue().in_composite("persistence_multiturn_drift")
    assert catalogue().in_composite("generality_fermi_gen")


# --- the matrix ----------------------------------------------------------------------


def test_base_is_found_by_method_not_by_persona() -> None:
    """`persona_id` on a control arm is the target it is scored against, not "base"."""
    matrix = build_matrix(_grid())
    assert matrix.bases == {"mathematical:m": "mathematical/base@1:m"}
    assert matrix.is_base("mathematical/base@1:m")
    assert matrix.organisms[0] == "mathematical/base@1:m", (
        "each persona group leads with its control"
    )


def test_each_persona_is_compared_against_its_own_control() -> None:
    """A single global baseline would compare impulsive organisms to the maths control."""
    rows = []
    for persona, base_score in (("mathematical", 0.2), ("impulsive", 0.8)):
        for method, bump in (("base", 0.0), ("prompting", 0.1)):
            for j in range(3):
                rows.append(
                    _row(f"{persona}/{method}@1", "mmlu", "capability", base_score + bump, f"s{j}")
                )

    matrix = build_matrix(rows)
    assert set(matrix.bases) == {"mathematical:m", "impulsive:m"}
    # Both personas moved by the same +0.1 from their own control, despite very
    # different absolute levels. A global baseline would report +0.1 and -0.5.
    assert matrix.signed_delta("mathematical/prompting@1:m", "mmlu") == pytest.approx(0.1)
    assert matrix.signed_delta("impulsive/prompting@1:m", "mmlu") == pytest.approx(0.1)


def test_a_persona_without_a_control_loses_only_its_own_deltas(caplog) -> None:
    """A half-finished control arm must not blank the personas that do have one."""
    rows = []
    for j in range(3):
        rows.append(_row("mathematical/base@1", "mmlu", "capability", 0.5, f"s{j}"))
        rows.append(_row("mathematical/prompting@1", "mmlu", "capability", 0.6, f"s{j}"))
        rows.append(_row("impulsive/prompting@1", "mmlu", "capability", 0.7, f"s{j}"))

    matrix = build_matrix(rows)
    assert matrix.signed_delta("mathematical/prompting@1:m", "mmlu") == pytest.approx(0.1)
    assert math.isnan(matrix.signed_delta("impulsive/prompting@1:m", "mmlu"))
    assert any("no base organism for persona" in r.message for r in caplog.records)


def test_no_base_yields_an_empty_mapping_not_a_crash(caplog) -> None:
    rows = _grid(with_base=False)
    assert find_bases(rows) == {}
    matrix = build_matrix(rows)
    assert not matrix.has_base
    assert any("method=" in r.message for r in caplog.records)


def test_deltas_are_nan_without_a_base_never_zero() -> None:
    """An un-guarded delta drawn as 0.0 reads as "this method changed nothing"."""
    matrix = build_matrix(_grid(with_base=False))
    delta = matrix.signed_delta("mathematical/sft_behaviour@1:m", "mmlu")
    assert math.isnan(delta), "a missing baseline must be unknown, not zero"


def test_base_row_is_zero_against_itself() -> None:
    matrix = build_matrix(_grid())
    assert matrix.signed_delta("mathematical/base@1:m", "mmlu") == pytest.approx(0.0)


def test_axis_effect_gives_every_eval_one_vote() -> None:
    """A 100x bigger bank must not get 100x the weight in an axis composite.

    Two generality evals with equal spread but very different sample counts: the
    composite must sit between their standardised effects, not be dragged to the big one.
    """
    rows = []
    for i, organism in enumerate(("treated/base@1", "treated/sft_behaviour@1")):
        for j in range(2):
            rows.append(_row(organism, "generality_causal", "generality", 0.2 * i, f"s{j}"))
        # 100 samples on the other eval, same per-organism means.
        for j in range(100):
            rows.append(_row(organism, "generality_downstream", "generality", 0.4 * i, f"b{j}"))

    matrix = build_matrix(rows)
    treated = "treated/sft_behaviour@1:m"
    small = matrix.eval_effect(treated, "generality_causal")
    large = matrix.eval_effect(treated, "generality_downstream")
    composite = matrix.axis_effect(treated, "generality")
    assert composite == pytest.approx((small + large) / 2)


def _coverage_grid() -> list[Row]:
    """Base + treated on the generality axis: one eval thin, one well-measured.

    `generality_causal` for the treated organism is scored on 1 of 10 samples (discard
    0.9); `generality_downstream` is fully measured. Means differ across organisms so both
    evals have a cross-organism spread to standardise against.
    """
    rows: list[Row] = []
    for i, org in enumerate(("g/base@1", "g/prompting@1")):
        rows += [
            _row(org, "generality_downstream", "generality", 0.4 + 0.2 * i, f"d{j}")
            for j in range(4)
        ]
        rows.append(_row(org, "generality_causal", "generality", 0.2 + 0.2 * i, "c0"))
        if org == "g/base@1":  # base is fully measured; treated is thin
            rows += [
                _row(org, "generality_causal", "generality", 0.2, f"c{j}") for j in range(1, 10)
            ]
        else:
            rows += [
                _row(org, "generality_causal", "generality", math.nan, f"c{j}")
                for j in range(1, 10)
            ]
    return rows


def test_eval_effect_gates_a_cell_scored_on_too_few_samples() -> None:
    """A 90%-discarded cell must not vote in the composite at full weight."""
    matrix = build_matrix(_coverage_grid())
    treated = "g/prompting@1:m"
    assert math.isnan(matrix.eval_effect(treated, "generality_causal")), "thin cell must be gated"
    assert math.isfinite(matrix.eval_effect(treated, "generality_downstream"))
    # The gated cell drops out of coverage, so the radar vertex goes hollow.
    used, total = matrix.axis_coverage(treated, "generality")
    assert (used, total) == (1, 2)
    # The axis still has a finite composite from the eval that survived.
    assert math.isfinite(matrix.axis_effect(treated, "generality"))


def _causal_grid(finite: int, nan: int) -> list[Row]:
    """Base + treated on generality; treated's causal cell has `finite` scored + `nan` unscored."""
    rows: list[Row] = []
    for i, org in enumerate(("g/base@1", "g/prompting@1")):
        rows += [
            _row(org, "generality_downstream", "generality", 0.4 + 0.2 * i, f"d{j}")
            for j in range(4)
        ]
        rows += [
            _row(org, "generality_causal", "generality", 0.2 + 0.2 * i, f"c{j}")
            for j in range(finite)
        ]
        if org != "g/base@1":  # only the treated arm carries the unscored samples
            rows += [
                _row(org, "generality_causal", "generality", math.nan, f"cx{j}")
                for j in range(nan)
            ]
    return rows


def test_a_heavily_discarded_cell_with_enough_samples_still_votes() -> None:
    """Discard alone must not gate: 65% discarded but n=35 is a reliable mean (fermi's case)."""
    matrix = build_matrix(_causal_grid(finite=35, nan=65))
    cell = matrix.get("g/prompting@1:m", "generality_causal")
    assert cell is not None and cell.discard_frac > 0.6 and cell.n >= 30
    assert math.isfinite(matrix.eval_effect("g/prompting@1:m", "generality_causal"))


def test_a_small_but_fully_measured_cell_still_votes() -> None:
    """Small n alone must not gate: n=15 at 0% discard is an intended small bank."""
    matrix = build_matrix(_causal_grid(finite=15, nan=0))
    cell = matrix.get("g/prompting@1:m", "generality_causal")
    assert cell is not None and cell.n < 30 and cell.discard_frac == 0.0
    assert math.isfinite(matrix.eval_effect("g/prompting@1:m", "generality_causal"))


def test_axis_exclusions_says_why_a_thin_cell_was_dropped() -> None:
    matrix = build_matrix(_coverage_grid())
    excluded = dict(matrix.axis_exclusions("g/prompting@1:m", "generality"))
    assert "generality_causal" in excluded
    assert "thin" in excluded["generality_causal"].lower()
    assert "generality_downstream" not in excluded, "a measured eval is not an exclusion"


# Two base-relative composite evals on one axis. This was the 0925 robustness case (DPO's
# debate cell); robustness now votes only through self-referenced share-kept cells, so the
# collapse machinery is pinned on a base-relative axis instead.
_AX = "generality"
_DEBATE, _CRITIQUE = "generality_downstream_gen", "generality_causal_gen"


def _debate_grid(*, dpo_debate: bool = True) -> list[Row]:
    """Base, SFT and DPO on two composite evals; DPO's second cell collapsed (1 of 100
    scored, at a far-off 0.9) the way the 0925 run's debate did. `dpo_debate=False` leaves
    that cell out entirely: the reference every OTHER organism must be indistinguishable from."""
    rows: list[Row] = []
    for i, org in enumerate(("r/base@1", "r/sft_behaviour@1", "r/dpo_behaviour@1")):
        rows += [
            _row(org, _CRITIQUE, _AX, 0.3 + 0.1 * i + 0.02 * (j % 3), f"c{j}") for j in range(40)
        ]
        if org == "r/dpo_behaviour@1":
            if dpo_debate:
                rows.append(_row(org, _DEBATE, _AX, 0.9, "d0"))
                rows += [_row(org, _DEBATE, _AX, math.nan, f"d{j}") for j in range(1, 100)]
            continue
        rows += [
            _row(org, _DEBATE, _AX, 0.05 + 0.2 * i + 0.01 * (j % 4), f"d{j}") for j in range(100)
        ]
    return rows


def test_a_collapsed_cell_drops_the_eval_for_its_own_organism_only() -> None:
    """DPO's unscoreable cell leaves DPO's composite on that axis; SFT keeps it, and every
    SFT quantity is exactly what it would be had DPO's cell never existed."""
    m = build_matrix(_debate_grid())
    ref = build_matrix(_debate_grid(dpo_debate=False))
    sft, dpo, base = "r/sft_behaviour@1:m", "r/dpo_behaviour@1:m", "r/base@1:m"

    assert m.voting_evals(dpo, _AX) == [_CRITIQUE]
    assert sorted(m.voting_evals(sft, _AX)) == sorted([_CRITIQUE, _DEBATE])
    assert m.axis_coverage(sft, _AX) == (2, 2)
    assert m.axis_coverage(dpo, _AX) == (1, 2)
    # the coverage note a caption reads: DPO's axis excludes the cell, and says why
    reasons = dict(m.axis_exclusions(dpo, _AX))
    assert list(reasons) == [_DEBATE] and "thin" in reasons[_DEBATE]
    assert m.axis_exclusions(sft, _AX) == []

    # the collapsed cell does not set the ruler everyone else is divided by
    assert m.eval_spread(_DEBATE) == pytest.approx(ref.eval_spread(_DEBATE))
    for fn in ("axis_effect", "axis_delta", "axis_score"):
        assert getattr(m, fn)(sft, _AX) == pytest.approx(getattr(ref, fn)(sft, _AX))
    assert m.axis_effect(sft, _AX) == pytest.approx(
        (m.eval_effect(sft, _CRITIQUE) + m.eval_effect(sft, _DEBATE)) / 2
    )

    # DPO: every view averages the same single eval, its base paired on that same eval
    assert m.axis_effect(dpo, _AX) == pytest.approx(m.eval_effect(dpo, _CRITIQUE))
    assert m.axis_delta(dpo, _AX) == pytest.approx(m.signed_delta(dpo, _CRITIQUE))
    cell = m.get(dpo, _CRITIQUE)
    assert cell is not None
    assert m.axis_score(dpo, _AX) == pytest.approx(cell.mean)  # raw 0.9 left out
    bcell = m.get(base, _CRITIQUE)
    assert bcell is not None
    # the base band spans DPO's eval set (one eval only), not the base's full set
    assert m.axis_noise_floor(dpo, _AX, standardised=False) == pytest.approx(bcell.sem)
    assert m.axis_noise_floor(sft, _AX, standardised=False) == pytest.approx(
        ref.axis_noise_floor(sft, _AX, standardised=False)
    )


def test_dashboard_votes_flag_is_per_organism_coverage() -> None:
    """The page averages cells whose `votes` is true, so a collapsed cell must say it does
    not vote for its own organism only."""
    from analysis.dashboard import build_dashboard_data

    m = build_matrix(_debate_grid())
    data = build_dashboard_data(m, fake=True, rows=None)
    dpo = data["cells"]["r/dpo_behaviour@1:m"][_DEBATE]
    assert dpo["votes"] is False and "thin" in dpo["excluded"]
    sft = data["cells"]["r/sft_behaviour@1:m"][_DEBATE]
    assert sft["votes"] is True and "excluded" not in sft


def test_axis_exclusions_says_when_an_eval_did_not_run_for_this_organism() -> None:
    rows = _coverage_grid()
    # Add a composite eval only for the base, so the treated organism never ran it.
    rows += [_row("g/base@1", "generality_ood", "generality", 0.5, f"t{j}") for j in range(4)]
    matrix = build_matrix(rows)
    excluded = dict(matrix.axis_exclusions("g/prompting@1:m", "generality"))
    assert "not run" in excluded.get("generality_ood", "").lower()


def test_exclusions_table_names_the_dropped_eval_and_its_reason() -> None:
    from analysis.diagnostics import axis_exclusions_table

    table = axis_exclusions_table(build_matrix(_coverage_grid()))
    assert table.header == ["organism", "axis", "eval left out", "why"]
    dropped = {(r[0], r[2]): r[3] for r in table.rows}
    assert "thin" in dropped[("g/prompting@1:m", "generality_causal")].lower()


def test_exclusions_table_is_reassuring_when_nothing_was_dropped() -> None:
    from analysis.diagnostics import NO_EXCLUSIONS, axis_exclusions_table

    table = axis_exclusions_table(build_matrix(_grid()))
    assert table.rows == []  # everything measured voted
    assert table.note == NO_EXCLUSIONS  # still says something, never blank


def test_discard_fraction_counts_unmeasurable_samples() -> None:
    rows = [
        _row("a/prompting@1", "mmlu", "capability", 1.0, "s0"),
        _row("a/prompting@1", "mmlu", "capability", float("nan"), "s1"),
        _row("a/prompting@1", "mmlu", "capability", float("nan"), "s2"),
        _row("a/prompting@1", "mmlu", "capability", 0.0, "s3"),
    ]
    cell = build_matrix(rows).get("a/prompting@1:m", "mmlu")
    assert cell is not None
    assert cell.discard_frac == pytest.approx(0.5)
    assert cell.n == 2 and cell.n_missing == 2


def test_meta_survives_the_round_trip(tmp_path: Path) -> None:
    """Figures that need a second column read it from `Row.meta`, not the Parquet."""
    spec = OrganismSpec(
        persona_id="p",
        base_model="m",
        method="prompting",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    write_manifest(spec, tmp_path)
    write_results(
        [
            SampleResult(
                spec.spec_hash,
                "generality_style_substance",
                "generality",
                "s0",
                1.0,
                "out",
                {"style": 80.0, "substance": 12.0},
            )
        ],
        tmp_path,
    )
    (row,) = load_rows(tmp_path)
    assert row.meta["style"] == 80.0
    assert row.meta["substance"] == 12.0


# --- the frontier's dose trajectories --------------------------------------------------


def _dose_grid(ladders: dict[str, list[str]], persona: str = "mathematical") -> list[Row]:
    """One persona with a control, plus a dose ladder per method.

    Every organism gets its own pair of scores, so each lands on a distinct point of the
    frontier and a line drawn through the wrong points is visibly wrong rather than
    accidentally right.
    """
    rows = [
        _row(f"{persona}/base@1", eval_name, axis, 0.5, f"s{j}")
        for eval_name, axis in (("generality_downstream", "generality"), ("mmlu", "capability"))
        for j in range(3)
    ]
    for step, (method, scale) in enumerate(
        ((m, s) for m, scales in ladders.items() for s in scales), start=1
    ):
        organism = f"{persona}/{method}@{scale}"
        for j in range(3):
            rows.append(
                _row(organism, "generality_downstream", "generality", 0.5 + 0.05 * step, f"s{j}")
            )
            rows.append(_row(organism, "mmlu", "capability", 0.5 - 0.02 * step, f"s{j}"))
    return rows


def _trajectories(fig) -> list:
    """The dose trajectories drawn on a frontier.

    They are the only lines on it at linewidth 2.4; the two reference lines through the
    origin are dotted at 1.2, and the arrowheads are patches rather than lines.
    """
    (ax,) = fig.axes
    return [line for line in ax.lines if line.get_linewidth() == pytest.approx(2.4)]


def _point(matrix, organism: str) -> tuple[float, float]:
    return matrix.axis_delta(organism, "generality"), matrix.axis_delta(organism, "capability")


def test_the_raw_view_is_a_plain_mean_of_the_axis_evals() -> None:
    """Raw means raw: no base, no sign flip, one vote per eval."""
    matrix = build_matrix(_grid())
    # mathematical/prompting@1 scores 0.1 + 0.05*j over j in 0..3 on both generality evals.
    assert matrix.axis_score("mathematical/prompting@1:m", "generality") == pytest.approx(0.175)
    assert math.isnan(matrix.axis_score("mathematical/prompting@1:m", "no_such_axis"))


def test_split_organism_tolerates_a_malformed_id() -> None:
    assert split_organism("nonsense") == ("nonsense", "unknown", "1")


# --- synthetic data guards -----------------------------------------------------------


def test_preview_data_refuses_to_write_into_the_real_results_root(tmp_path: Path) -> None:
    from scripts.make_preview_data import check_destination

    protected = tmp_path / "results"
    protected.mkdir()
    with pytest.raises(SystemExit, match="refusing"):
        check_destination(protected, protected)
    with pytest.raises(SystemExit, match="refusing"):
        check_destination(protected / "nested", protected)
    with pytest.raises(SystemExit, match="refusing"):
        check_destination(tmp_path, protected)  # a parent of the real root


def test_preview_data_allows_a_scratch_directory(tmp_path: Path) -> None:
    from scripts.make_preview_data import check_destination

    protected = tmp_path / "results"
    scratch = tmp_path / "elsewhere" / "scratch"
    assert check_destination(scratch, protected) == scratch.resolve()


def test_every_synthetic_row_is_detectable_as_fake() -> None:
    """The guard that matters most: this data must never be mistakable for a result."""
    from scripts.make_preview_data import make_rows

    rows = make_rows("h", "impulsive", "mmlu", "capability", 20, "binary", 0.8)
    assert rows
    assert all(is_fake_output(r.raw_output) for r in rows)


def test_synthetic_generation_is_reproducible(tmp_path: Path) -> None:
    """Seeded per cell, so a figure change can be told apart from a data change."""
    from scripts.make_preview_data import make_rows

    def fingerprint(rows):
        # NaN != NaN, and unscoreable samples are a designed part of this data, so
        # compare on a representation rather than on the floats.
        return [(r.sample_id, repr(r.score), r.meta) for r in rows]

    first = make_rows("h", "p", "mmlu", "capability", 30, "binary", 0.5)
    second = make_rows("h", "p", "mmlu", "capability", 30, "binary", 0.5)
    assert fingerprint(first) == fingerprint(second)


# --- axis coverage and the outside-the-composite panel ---------------------------------


def test_axis_coverage_counts_measurable_composite_evals() -> None:
    rows = _grid()
    # generality has two composite evals in the grid; drop one for one organism.
    rows = [
        r
        for r in rows
        if not (
            r.organism == "mathematical/sft_behaviour@1:m"
            and r.eval_name == "generality_downstream"
        )
    ]
    matrix = build_matrix(rows)
    assert matrix.axis_coverage("mathematical/prompting@1:m", "generality") == (2, 2)
    assert matrix.axis_coverage("mathematical/sft_behaviour@1:m", "generality") == (1, 2)


# --- the base noise band ---------------------------------------------------------------


def test_axis_noise_floor_propagates_base_sems() -> None:
    """1-SEM floor of the axis composite, from the base arm: sqrt(sum sem^2)/E."""
    matrix = build_matrix(_grid())
    organism = "mathematical/prompting@1:m"
    base = matrix.base_for(organism)
    evals = matrix.evals_in("generality", composite_only=True)
    sems = [matrix.get(base, e).sem for e in evals]
    expected = math.sqrt(sum(s**2 for s in sems)) / len(sems)
    got = matrix.axis_noise_floor(organism, "generality", standardised=False)
    assert got == pytest.approx(expected)
    # Standardised: each eval's sem in units of that eval's own cross-organism spread.
    expected_std = math.sqrt(
        sum((s / matrix.eval_spread(e)) ** 2 for s, e in zip(sems, evals, strict=True))
    ) / len(evals)
    got_std = matrix.axis_noise_floor(organism, "generality", standardised=True)
    assert got_std == pytest.approx(expected_std)


def test_composite_se_is_unknown_when_any_voting_term_is() -> None:
    """The one composite-SE rule (#261): sqrt(sum sem^2)/k, and NaN — never the survivors'
    SE — when a term is unknown, plus NaN for no terms and for a non-positive divisor."""
    assert composite_se([0.03, 0.04]) == pytest.approx(0.025)
    assert composite_se([0.03, 0.04], k=1) == pytest.approx(0.05)
    for bad in ([], [0.03, math.nan], [math.inf], [math.nan]):
        assert math.isnan(composite_se(bad))
    assert math.isnan(composite_se([0.03], k=0))


def test_axis_noise_floor_is_unknown_when_a_voting_base_sem_is() -> None:
    """One voting eval with an unknown base SEM: the floor is NaN, not the other evals'."""
    import dataclasses

    matrix = build_matrix(_grid())
    organism = "mathematical/prompting@1:m"
    base = matrix.base_for(organism)
    evals = matrix.voting_evals(organism, "generality")
    assert len(evals) >= 2 and math.isfinite(
        matrix.axis_noise_floor(organism, "generality", standardised=False)
    )
    key = (base, evals[0])
    matrix.cells[key] = dataclasses.replace(matrix.cells[key], sem=math.nan)
    assert evals[0] in matrix.voting_evals(organism, "generality")  # its delta still votes
    assert math.isnan(matrix.axis_noise_floor(organism, "generality", standardised=False))
    assert math.isnan(matrix.axis_noise_floor(organism, "generality", standardised=True))


def test_axis_se_carries_both_arms_and_is_unknown_when_any_voting_sem_is() -> None:
    """Issue #261 regression (runs without the `figures` extra, unlike the coherence-specific
    tests). SE(axis_delta) = sqrt(sum org_sem^2 + sum base_sem^2) / E over the E voting
    evals; ONE unknown SEM on either arm makes it — and the axis z — unknown. The old code
    dropped the unknown organism term (dividing by the survivors) and read an unknown base
    floor as 0.0, so a vote with unknowable uncertainty still moved a confident z."""
    import dataclasses

    from analysis import dashboard

    matrix = build_matrix(_grid())
    organism = "mathematical/prompting@1:m"
    base = matrix.base_for(organism)
    evals = matrix.voting_evals(organism, "generality")
    assert len(evals) >= 2
    sems = [matrix.get(o, e).sem for e in evals for o in (organism, base)]
    want = math.sqrt(sum(s * s for s in sems)) / len(evals)
    assert dashboard._axis_se(matrix, organism, "generality") == pytest.approx(want)
    assert dashboard.axis_z(matrix, organism, "generality") == pytest.approx(
        matrix.axis_delta(organism, "generality") / want
    )
    pristine = dict(matrix.cells)
    for arm in (organism, base):
        matrix.cells.clear()
        matrix.cells.update(pristine)
        key = (arm, evals[0])
        matrix.cells[key] = dataclasses.replace(matrix.cells[key], sem=math.nan)
        assert evals[0] in matrix.voting_evals(organism, "generality")  # its delta still votes
        assert math.isfinite(matrix.axis_delta(organism, "generality"))
        assert dashboard._axis_se(matrix, organism, "generality") is None, arm
        assert dashboard.axis_z(matrix, organism, "generality") is None, arm


def test_delta_se_is_the_hypot_of_both_arms_sems_and_unknown_at_n_1() -> None:
    """Pins `delta_se` without the figures extra: sqrt(sem_org^2 + sem_base^2) for a base-relative
    eval; an arm with one sample and no declared SE (`n <= 1`, `sem` 0.0) makes the SE unknown, so
    the page draws no CI and no dot, and the axis SE follows."""
    import dataclasses

    from analysis import dashboard

    matrix = build_matrix(_grid())
    organism = "mathematical/prompting@1:m"
    base = matrix.base_for(organism)
    e = matrix.voting_evals(organism, "generality")[0]
    got, ref = matrix.get(organism, e), matrix.get(base, e)
    assert got.n > 1 and ref.n > 1
    assert dashboard.delta_se(matrix, organism, e) == pytest.approx(math.hypot(got.sem, ref.sem))
    pristine = dict(matrix.cells)
    for arm in (organism, base):
        matrix.cells.clear()
        matrix.cells.update(pristine)
        matrix.cells[(arm, e)] = dataclasses.replace(matrix.cells[(arm, e)], n=1, sem=0.0)
        assert dashboard.delta_se(matrix, organism, e) is None, arm
        assert dashboard._axis_se(matrix, organism, "generality") is None, arm
        # ... but an n == 1 estimate row that DECLARES an SE (external bootstrap) keeps it
        matrix.cells[(arm, e)] = dataclasses.replace(matrix.cells[(arm, e)], n=1, sem=0.05)
        assert dashboard.delta_se(matrix, organism, e) is not None, arm


def test_axis_noise_floor_is_nan_without_a_base() -> None:
    matrix = build_matrix(_grid(with_base=False))
    floor = matrix.axis_noise_floor("mathematical/prompting@1:m", "generality", standardised=False)
    assert math.isnan(floor)


def _fig_texts(fig) -> str:
    return " | ".join(t.get_text() for ax in fig.axes for t in ax.texts)


def test_an_axis_the_build_has_never_heard_of_cannot_draw_its_own_spoke() -> None:
    """`build_matrix` must not invent an axis the rest of the report has never heard of.

    `axis_by_eval` filters `Row.axis` through `VALID_AXES` (via `_canonical_axis`), so an
    axis nobody recognises draws no spoke and no heatmap band of its own — it would give one
    artefact two contradictory axis vocabularies, while the report's axis tables, which
    render `sorted(VALID_AXES)`, drop that same row. `unknown` is the honest label: the cell
    was measured, its axis was not recognised.
    """
    rows = _grid() + [
        _row(o, "some_old_cell", "phlogiston", 2.5, sample_id=f"s{j}")
        for o in ("mathematical/base@1", "mathematical/prompting@1")
        for j in range(4)
    ]
    matrix = build_matrix(rows)
    assert "phlogiston" not in matrix.axes(), "a stale axis must not name itself"
    assert "unknown" in matrix.axes(), "and the cell must still be there, labelled honestly"
    # The real axes are untouched — this is a relabel, not a drop.
    assert {"generality", "coherence", "capability"} <= set(matrix.axes())
    assert matrix.get("mathematical/base@1:m", "some_old_cell").axis == "unknown"


@pytest.mark.parametrize(
    "old",
    [
        "locality",
        "friedness",
        "persistence",
        # Taxonomy v2, 2026-09-25 (docs/evals/reading-results.md#axes-and-the-renames).
        # `robustness` itself is a live name again, so it is no longer in this list.
        "reliability",
        "adversarial_robustness",
        "long_context_robustness",
        "internal_representation",
        "portability",
        "ripple",
    ],
)
def test_a_retired_axis_name_is_not_aliased(old: str) -> None:
    """No read-side aliases (docs/evals/reading-results.md#axes-and-the-renames): a row
    stored under a retired axis key reads as `unknown` rather than being relabelled into its
    successor — those trees are not comparable with current runs and must be re-run, not
    joined."""
    rows = _grid() + [
        _row(o, "mu_decisiveness", old, 0.4, sample_id=f"s{j}")
        for o in ("mathematical/base@1", "mathematical/prompting@1")
        for j in range(4)
    ]
    matrix = build_matrix(rows)
    assert old not in matrix.axes(), "the old key must not name itself"
    assert matrix.get("mathematical/base@1:m", "mu_decisiveness").axis == "unknown"


def test_a_known_axis_is_never_relabelled() -> None:
    """The guard above must be inert on every tree anyone actually has."""
    matrix = build_matrix(_grid())
    assert "unknown" not in matrix.axes()
    assert matrix.get("mathematical/base@1:m", "mmlu").axis == "capability"


def test_axis_groups_partition_the_registry_and_the_radar_is_a_subset() -> None:
    """Every VALID_AXES entry is in exactly one of behaviour / health, and the radar names
    only real axes, with expression and misalignment off it
    (docs/evals/reading-results.md#axes-and-the-renames)."""
    from core.registry import VALID_AXES

    cat = catalogue()
    assert set(cat.groups) == VALID_AXES
    assert {a for a, g in cat.groups.items() if g == "behaviour"} == {
        "expression",
        "generality",
        "robustness",
        "introspection",
        "activations",
    }
    assert {a for a, g in cat.groups.items() if g == "health"} == {
        "capability",
        "coherence",
        "misalignment",
    }
    assert cat.radar is not None
    assert cat.radar <= VALID_AXES
    # The radar is declared per property (`radar: true` in configs/properties.yaml); this pins the
    # same six the old literal did, read from the file rather than from the loader.
    import yaml

    raw = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs" / "properties.yaml").read_text(
            encoding="utf-8"
        )
    )
    on_radar = {a for a, spec in raw["properties"].items() if spec.get("radar") is True}
    assert on_radar == VALID_AXES - {"expression", "misalignment"}
    assert cat.radar == on_radar
    assert set(cat.axis_order) <= VALID_AXES


def test_synthetic_tree_is_marked_fake_end_to_end(tmp_path: Path) -> None:
    from analysis.dashboard import build_dashboard_data, render_html
    from scripts.make_preview_data import generate

    root = tmp_path / "scratch"
    # A cut-down tree: this exercises the real manifest + banner + dashboard path, which is
    # what the guards live on, without paying for ~12k rows in the <10s test budget.
    total = generate(
        root,
        protected=tmp_path / "results",
        organisms=(("impulsive", "base", 1.0), ("impulsive", "sft_behaviour", 1.0)),
        evals=(
            ("mmlu", "capability", 5, "binary"),
            ("generality_causal", "generality", 5, "binary"),
        ),
    )
    assert total > 0

    rows = load_rows(root)
    assert rows
    assert all(r.fake for r in rows), "every manifest must carry fake=True"
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    assert "FAKE-MODE DATA" in render_html(data)


# --- differential attrition (#254): report-only flag ---------------------------------------


# the Matrix keys organisms by `persona/method@scale:model_tag` (`_row` uses model_tag "m")
TREATED, BASE = "mathematical/prompting@1:m", "mathematical/base@1:m"


def _attrition_rows(treated_nan_misbehaving: int, treated_unjudged: int = 0) -> list[Row]:
    """Base: 100 rows, rate 0.50, nothing discarded. Treated: true rate 0.50 as well, but
    `treated_nan_misbehaving` of its 50 misbehaving rows are unparseable (NaN)."""
    ev, axis = "misaligned_sandbagging", "misalignment"
    rows = [_row("mathematical/base@1", ev, axis, float(j % 2), f"b{j}") for j in range(100)]
    org = "mathematical/prompting@1"
    for j in range(100):
        misbehaves = j % 2 == 1
        nan = misbehaves and (j // 2) < treated_nan_misbehaving
        rows.append(_row(org, ev, axis, math.nan if nan else float(misbehaves), f"t{j}"))
    for j in range(treated_unjudged):
        rows.append(_row(org, ev, axis, math.nan, f"u{j}", meta={"judged": False}))
    return rows


def test_attrition_flag_trips_on_the_issue_254_example() -> None:
    """Same true rate (0.50), but 90% of the treated arm's misbehaving rows are unparseable."""
    from analysis.aggregate import ATTRITION_GAP_BOUND

    assert ATTRITION_GAP_BOUND == 0.15
    m = build_matrix(_attrition_rows(treated_nan_misbehaving=45))
    gap = m.attrition_gap(TREATED, "misaligned_sandbagging")
    assert gap is not None
    org_d, base_d, flagged = gap
    assert (org_d, base_d, flagged) == (pytest.approx(0.45), 0.0, True)
    # the scored-only treated rate is 5/55: the bias the flag warns about
    assert m.get(TREATED, "misaligned_sandbagging").mean == pytest.approx(5 / 55)


def test_attrition_flag_is_quiet_when_arms_discard_alike() -> None:
    m = build_matrix(_attrition_rows(treated_nan_misbehaving=0))
    assert m.attrition_gap(TREATED, "misaligned_sandbagging") == (0.0, 0.0, False)
    # a gap of exactly the bound is not flagged (strictly greater)
    m = build_matrix(_attrition_rows(treated_nan_misbehaving=15))
    assert m.attrition_gap(TREATED, "misaligned_sandbagging")[2] is False
    m = build_matrix(_attrition_rows(treated_nan_misbehaving=16))
    assert m.attrition_gap(TREATED, "misaligned_sandbagging")[2] is True


def test_attrition_flag_ignores_unjudged_rows_and_has_no_pair_for_bases() -> None:
    """Unjudged drift turns are not discards; a base (or an eval with no base cell) has no gap."""
    m = build_matrix(_attrition_rows(treated_nan_misbehaving=0, treated_unjudged=500))
    assert m.attrition_gap(TREATED, "misaligned_sandbagging")[2] is False
    assert m.attrition_gap(BASE, "misaligned_sandbagging") is None


def test_attrition_flag_changes_no_existing_number() -> None:
    """Report only: delta, SE, votes and the composite coverage gate are what they were without
    the flag; the dashboard cell only gains `base_discard` / `attrition_flag`."""
    from analysis.dashboard import build_dashboard_data

    rows = _attrition_rows(treated_nan_misbehaving=45)
    m = build_matrix(rows)
    org, ev = TREATED, "misaligned_sandbagging"
    # lower-is-better eval: the sign is flipped
    assert m.signed_delta(org, ev) == pytest.approx(-(5 / 55 - 0.5))
    assert m.has_coverage(org, ev)
    cell = build_dashboard_data(m, fake=True, rows=rows)["cells"][org][ev]
    assert cell["attrition_flag"] is True
    assert cell["base_discard"] == 0.0 and cell["discard"] == pytest.approx(0.45)
    assert cell["votes"] is True and cell["delta"] is not None


def _rate_rows(org: str, k: int, n: int, ev: str = "misaligned_sandbagging") -> list[Row]:
    """`n` rows for `org`, the first `k` NaN, the rest alternating 0/1."""
    return [
        _row(org, ev, "misalignment", math.nan if j < k else float(j % 2), f"{org}{j}")
        for j in range(n)
    ]


def test_attrition_exact_bound_is_not_flagged_despite_float_error() -> None:
    """135/300 = 0.45 vs 90/300 = 0.30: 0.45 - 0.30 is 0.15000000000000002 in floats."""
    assert abs(0.45 - 0.3) > 0.15  # the trap
    rows = _rate_rows("mathematical/base@1", 90, 300) + _rate_rows(
        "mathematical/prompting@1", 135, 300
    )
    m = build_matrix(rows)
    assert m.attrition_gap(TREATED, "misaligned_sandbagging")[2] is False
    rows = _rate_rows("mathematical/base@1", 90, 300) + _rate_rows(
        "mathematical/prompting@1", 136, 300
    )
    assert build_matrix(rows).attrition_gap(TREATED, "misaligned_sandbagging")[2] is True


def test_attrition_not_exported_for_a_cell_without_a_delta() -> None:
    """An all-NaN base arm has no delta: the cell must carry no `attrition_flag`."""
    from analysis.dashboard import _attrition_keys, build_dashboard_data

    rows = _rate_rows("mathematical/base@1", 100, 100) + _rate_rows(
        "mathematical/prompting@1", 0, 100
    )
    m = build_matrix(rows)
    assert _attrition_keys(m, TREATED, "misaligned_sandbagging") == {}
    data = build_dashboard_data(m, fake=True, rows=rows)
    assert "attrition_flag" not in data["cells"][TREATED]["misaligned_sandbagging"]
    assert data["meta"]["attrition_bound"] == 0.15
