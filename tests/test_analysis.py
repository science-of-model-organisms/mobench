"""Aggregation and report rendering, against fake-mode Parquet in a tmp dir."""

from __future__ import annotations

import json
import logging
import math
import statistics
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis.aggregate import (
    Row,
    _variant_of,
    aggregate,
    aggregate_seeds,
    any_fake,
    build_matrix,
    dose_response,
    find_bases,
    load_rows,
    sample_transcripts,
)
from analysis.diagnostics import _composite_rows, _fmt_cell, _fmt_score
from analysis.eval_meta import catalogue
from core.results import SampleResult, write_results
from core.spec import OrganismSpec
from serving.manifest import write_manifest
from serving.runner import load_specs


def _organism(persona: str, method: str, scale: float) -> OrganismSpec:
    return OrganismSpec(persona, "Qwen/Qwen2.5-7B-Instruct", method, train_scale=scale)


def _populate(root: Path, *, fake: bool = False) -> list[OrganismSpec]:
    """Two personas × prompting + sft at 3 scales, two evals each."""
    specs = []
    for persona, base in (("caring", 0.6), ("malevolent", 0.3)):
        for method, scale in (
            ("prompting", 1.0),
            ("sft_behaviour", 0.25),
            ("sft_behaviour", 0.5),
            ("sft_behaviour", 1.0),
        ):
            spec = _organism(persona, method, scale)
            specs.append(spec)
            write_manifest(spec, root, fake=fake)
            for eval_name, axis in (("reliability_x", "expression"), ("mmlu", "capability")):
                rows = [
                    SampleResult(
                        spec_hash=spec.spec_hash,
                        eval_name=eval_name,
                        axis=axis,
                        sample_id=f"s{i}",
                        score=base + 0.1 * scale + 0.05 * i,
                        raw_output=("[FAKE-ORGANISM persona=x] marker" if fake else f"out {i}"),
                        meta={},
                    )
                    for i in range(3)
                ]
                write_results(rows, root)
    return specs


@pytest.fixture
def populated(results_root: Path) -> Path:
    _populate(results_root)
    return results_root


# --- one card per base model, each costed on its own organisms (0709 sweep, 2026-09-08) ---

BASE_7B = "Qwen/Qwen2.5-7B-Instruct"
BASE_14B = "Qwen/Qwen2.5-14B-Instruct"


def _populate_sizes(root: Path) -> dict[str, set[str]]:
    """Two organisms per base model, one timed cell each; returns base model -> hashes."""
    from serving.timing import write_timing

    out: dict[str, set[str]] = {}
    for base_model in (BASE_7B, BASE_14B):
        for method in ("base", "prompting"):
            spec = OrganismSpec("caring", base_model, method)
            write_manifest(spec, root)
            rows = [
                SampleResult(
                    spec.spec_hash,
                    "reliability_x",
                    "expression",
                    f"s{i}",
                    0.5,
                    "out",
                    {"gen_latency_s": 1.0},
                )
                for i in range(3)
            ]
            write_results(rows, root, spec=spec)
            write_timing(
                spec,
                "reliability_x",
                axis="expression",
                status="written",
                n_samples=3,
                seconds=60.0,
                root=root,
            )
            out.setdefault(base_model, set()).add(spec.spec_hash)
    return out


def test_load_rows_keeps_only_the_named_base_model(results_root: Path) -> None:
    _populate_sizes(results_root)
    rows = load_rows(results_root, base_model=BASE_14B)
    assert rows and {r.base_model for r in rows} == {BASE_14B}


def test_the_cost_tables_count_only_the_cards_organisms(results_root: Path) -> None:
    """A per-size card that prints the whole sweep's cost misattributes every other size's
    hours to it. Cells, samples AND the row-summed gen time must all be the card's own."""
    from analysis.diagnostics import eval_cost_table, organism_cost_table

    hashes = _populate_sizes(results_root)
    (whole,) = [r for r in eval_cost_table(results_root).rows if r[0] == "reliability_x"]
    (card,) = [
        r for r in eval_cost_table(results_root, hashes[BASE_14B]).rows if r[0] == "reliability_x"
    ]
    assert whole[2:4] == ["4", "12"]
    assert card[2:4] == ["2", "6"]
    assert (whole[7], card[7]) == ("0.2", "0.1")  # gen min: 12 s vs 6 s of row latency
    listed = organism_cost_table(results_root, hashes[BASE_14B]).rows
    assert {r[1] for r in listed if r[0] != "ALL"} == hashes[BASE_14B]


def test_no_two_registered_organisms_share_a_scorecard_label(results_root: Path) -> None:
    """Census over the real configs/specs.yaml: organisms that differ in anything but
    `seed` must never share a label (the card would pool them), and every seed of one
    organism must share one. Main's 14B rlaif v1 and v2 adapters collided on
    `impulsive/rlaif@1:14B` until 2026-09-10."""
    from serving.runner import load_specs

    for spec in load_specs("configs/specs.yaml"):
        write_manifest(spec, results_root)
        write_results(
            [SampleResult(spec.spec_hash, "e", "expression", "x", 0.0, "out", {})],
            results_root,
            spec=spec,
        )
    families: dict[str, set[str]] = {}
    labels: dict[str, set[str]] = {}
    for r in load_rows(results_root):
        families.setdefault(r.organism, set()).add(r.family)
        labels.setdefault(r.family, set()).add(r.organism)
    assert not {k: v for k, v in families.items() if len(v) > 1}
    assert not {k: v for k, v in labels.items() if len(v) > 1}


def test_recipe_of_names_the_training_data_recipe() -> None:
    """Two 7B `sft_behaviour` organisms differ ONLY in their training data, so the label
    carries the recipe named by `dataset_uri`; without it the card averaged them."""
    from analysis.aggregate import _recipe_of

    real = "science-of-model-organisms/behaviour-runs :: keep/impulsive/qwen2.5-7b-it_{}_seed0"
    assert _recipe_of(real.format("sft-6069_dpo-1718")) == "sft-6069_dpo-1718"
    assert _recipe_of(real.format("hybrid-sft2900-dpo2900")) == "hybrid-sft2900-dpo2900"
    # Any other shape is used WHOLE (minus the repo), never cut at its first underscore.
    assert _recipe_of("org/repo :: data/my_dataset") == "my_dataset"
    assert _recipe_of(None) == ""
    a = {"method_config": {"dataset_uri": real.format("sft-6069_dpo-1718")}}
    b = {"method_config": {"dataset_uri": real.format("hybrid-sft2900-dpo2900")}}
    assert _variant_of(a) != _variant_of(b)


def test_rows_are_joined_to_their_spec(populated: Path) -> None:
    rows = load_rows(populated)
    assert len(rows) == 8 * 2 * 3
    assert {r.method for r in rows} == {"prompting", "sft_behaviour"}
    assert {r.persona_id for r in rows} == {"caring", "malevolent"}


def test_results_without_a_manifest_are_marked_unknown(results_root: Path) -> None:
    """A hash with no manifest must not silently masquerade as a real organism."""
    write_results(
        [SampleResult("orphan", "e", "expression", "s1", 1.0, "out", {})],
        results_root,
    )
    (row,) = load_rows(results_root)
    assert row.method == "unknown"


def test_aggregate_reports_n_and_spread(populated: Path) -> None:
    cells = aggregate(load_rows(populated), by=("method", "axis"))
    assert all(c.n > 1 for c in cells)
    assert all(c.sd > 0 for c in cells)
    assert all(c.sem < c.sd for c in cells)


def test_aggregate_rejects_unknown_group_key(populated: Path) -> None:
    with pytest.raises(ValueError, match="no such field"):
        aggregate(load_rows(populated), by=("nonsense",))


def test_single_sample_group_has_zero_spread() -> None:
    row = Row(
        "h",
        "caring",
        "prompting",
        1.0,
        "m",
        "persona_trait",
        "e",
        "expression",
        "s",
        0.5,
        "out",
        False,
    )
    (cell,) = aggregate([row], by=("method",))
    assert (cell.n, cell.sd, cell.sem) == (1, 0.0, 0.0)


def _row(score: float, sample_id: str = "s", axis: str = "expression") -> Row:
    return Row(
        "h",
        "caring",
        "prompting",
        1.0,
        "m",
        "persona_trait",
        "e",
        axis,
        sample_id,
        score,
        "out",
        False,
    )


def _meta_row(score: float, sample_id: str, meta: dict) -> Row:
    return Row(
        "h",
        "caring",
        "prompting",
        1.0,
        "m",
        "persona_trait",
        "e",
        "generality",
        sample_id,
        score,
        "out",
        False,
        meta=meta,
    )


def test_unparsed_fraction_is_surfaced_per_cell() -> None:
    """The unparseable=failure scoring folds a no-usable-answer / judge-error into the score
    (0/1) or a NaN; `frac_unparsed` keeps that visible as context. Spans both scored-0 rows
    (capability empty picks, judged no-label) and NaN rows (JUDGE_ERROR)."""
    rows = [
        _meta_row(1.0, "a", {"outcome": "implanted"}),  # placed: not unparsed
        _meta_row(0.0, "b", {"outcome": None}),  # judged no-label -> unparsed, scored 0
        _meta_row(math.nan, "c", {"outcome": None, "judge_raw": "JUDGE_ERROR: boom"}),  # infra
        _meta_row(0.0, "d", {"n_parsed": 0, "n_samples": 1}),  # capability empty picks
    ]
    (cell,) = aggregate(rows, by=("method",))
    assert cell.n_unparsed == 3
    assert (cell.n, cell.n_missing) == (3, 1)  # 3 finite (1,0,0), 1 NaN
    assert cell.frac_unparsed == 0.75
    assert "75% unparsed" in _fmt_cell(cell)


def test_unparsed_fraction_covers_the_inverted_axis_cells() -> None:
    """sandbagging/sycophancy carry their parse state in `picked`/`second_answer`, not
    `outcome`, and their unparseable answer scores NaN (lit-backed) — so it would otherwise
    be a silent discard. `is_unparsed` must still see it. Sycophancy's `not eligible` NaN is
    a STRUCTURAL denominator exclusion, not a parse failure, and must NOT be counted."""
    rows = [
        _meta_row(1.0, "a", {"picked": "B"}),  # sandbagging parsed
        _meta_row(math.nan, "b", {"picked": None}),  # sandbagging unparsed
        _meta_row(0.0, "c", {"eligible": True, "second_answer": "C"}),  # sycophancy parsed
        _meta_row(math.nan, "d", {"eligible": True, "second_answer": None}),  # pushback unparsed
        _meta_row(math.nan, "e", {"eligible": False, "second_answer": None}),  # not eligible
    ]
    (cell,) = aggregate(rows, by=("method",))
    assert cell.n_unparsed == 2  # b + d; the not-eligible row (e) is a structural drop


def test_unmeasurable_samples_do_not_crash_aggregation() -> None:
    """NaN is a legal score - `SampleResult.score` is a non-null float64, and an eval
    that could not measure a sample must say so rather than invent a number.

    Unfiltered it does not merely skew the summary, it raises: `statistics.stdev` goes
    via Fraction, and NaN has no `.numerator`.
    """
    rows = [_row(0.0, "a"), _row(1.0, "b"), _row(math.nan, "c")]
    (cell,) = aggregate(rows, by=("method",))
    assert (cell.n, cell.n_missing) == (2, 1)
    assert cell.mean == 0.5


def test_unmeasurable_samples_are_excluded_from_min_and_max() -> None:
    """The silent half of the same bug: `min`/`max` over NaN do not raise, they return
    whatever element order happens to put first."""
    rows = [_row(math.nan, "a"), _row(0.25, "b"), _row(0.75, "c")]
    (cell,) = aggregate(rows, by=("method",))
    assert (cell.min, cell.max) == (0.25, 0.75)


def test_infinities_are_filtered_too_not_just_nan() -> None:
    """`isnan` is the wrong predicate: an infinity crashes `stdev` identically, since it
    also fails the Fraction conversion. The test has to be `isfinite`."""
    rows = [_row(0.0, "a"), _row(1.0, "b"), _row(math.inf, "c"), _row(-math.inf, "d")]
    (cell,) = aggregate(rows, by=("method",))
    assert (cell.n, cell.n_missing) == (2, 2)
    assert (cell.mean, cell.min, cell.max) == (0.5, 0.0, 1.0)


def test_a_wholly_unmeasurable_group_still_yields_a_cell() -> None:
    """Emitting nothing would make "ran, measured nothing" indistinguishable from "never
    run" - the more flattering of the two readings."""
    (cell,) = aggregate([_row(math.nan, "a"), _row(math.nan, "b")], by=("method",))
    assert (cell.n, cell.n_missing) == (0, 2)
    assert math.isnan(cell.mean)


def _drift_row(turn: int, score: float, judged: bool) -> Row:
    """One turn of a drift trajectory, judged or merely stored."""
    return Row(
        "h",
        "impulsive",
        "prompting",
        1.0,
        "m",
        "persona_trait",
        "persistence_multiturn_drift_long",
        "robustness",
        f"t1:turn{turn}",
        score,
        "out",
        False,
        meta={"trajectory_id": "t1", "turn_idx": turn, "item_id": "t1", "judged": judged},
    )


_DRIFT_KEY = ("impulsive/prompting@1:m", "persistence_multiturn_drift_long")


def test_unjudged_turns_do_not_change_a_cell() -> None:
    """`persistence_multiturn_drift_long` judges 37 of 1000 turns BY DESIGN and stores the
    other 963 unjudged (`score = NaN`, `meta['judged'] = False`) so they can be scored from
    disk later. Pooled into the summary they would swamp every statistic with rows the
    judge never looked at."""
    judged = [_drift_row(1, 1.0, True), _drift_row(200, 0.0, True)]
    stored = [_drift_row(t, math.nan, False) for t in range(2, 50)]
    clean = build_matrix(judged).cells[_DRIFT_KEY]
    polluted = build_matrix([*judged, *stored]).cells[_DRIFT_KEY]
    assert (polluted.mean, polluted.sem, polluted.n) == (clean.mean, clean.sem, clean.n)


def test_discard_frac_counts_judge_discards_not_unjudged_turns() -> None:
    """The number that matters downstream: `discard_frac` gates a cell out of the composite
    and colours the F3 heatmap, so a by-design judging schedule must not read as damage."""
    rows = [
        _drift_row(1, 1.0, True),
        _drift_row(200, 0.0, True),
        _drift_row(400, math.nan, True),  # the judge ran and could not score it
        *[_drift_row(t, math.nan, False) for t in range(2, 50)],
    ]
    cell = build_matrix(rows).cells[_DRIFT_KEY]
    assert (cell.n, cell.n_missing) == (2, 1)
    assert cell.discard_frac == pytest.approx(1 / 3)


def test_a_wholly_unmeasurable_scale_point_does_not_break_the_dose_response() -> None:
    """`dose_response` unpacks `(cell,) = aggregate(...)`, so a group that produced no
    cell would raise ValueError rather than report an empty scale point."""
    rows = [
        Row(
            "h",
            "caring",
            "sft_behaviour",
            0.25,
            "m",
            "persona_trait",
            "e",
            "coherence",
            "a",
            math.nan,
            "o",
            False,
        ),
        Row(
            "h",
            "caring",
            "sft_behaviour",
            0.25,
            "m",
            "persona_trait",
            "e",
            "coherence",
            "b",
            math.nan,
            "o",
            False,
        ),
        Row(
            "h",
            "caring",
            "sft_behaviour",
            0.5,
            "m",
            "persona_trait",
            "e",
            "coherence",
            "c",
            0.7,
            "o",
            False,
        ),
    ]
    curve = dose_response(rows, "coherence")["caring"]
    assert [(scale, cell.n) for scale, cell in curve] == [(0.25, 0), (0.5, 1)]


def test_the_scorecard_says_how_many_were_unmeasurable() -> None:
    """A mean over the samples that parsed, printed as if it covered all of them, is the
    more flattering number and the less true one."""
    rows = [_row(1.0, "a"), _row(1.0, "b"), _row(math.nan, "c")]
    assert "[1 unmeasurable]" in _fmt_cell(aggregate(rows, by=("method",))[0])


def test_an_unmeasurable_cell_never_renders_as_a_number() -> None:
    """`nan ±nan (n=0)` in a results table is worse than useless."""
    (cell,) = aggregate([_row(math.nan, "a"), _row(math.nan, "b")], by=("method",))
    rendered = _fmt_cell(cell)
    assert rendered == "0 of 2 measurable"
    assert "nan" not in rendered.lower()


def test_lowest_scoring_transcripts_do_not_depend_on_input_order() -> None:
    """The silent one. Every comparison against NaN is False, so `sorted` neither raises
    nor orders - it returns whatever the input order produced, and a high score can land
    in a list captioned "lowest scoring". No CI job would ever catch this."""
    ordered = [_row(0.9, "s1"), _row(math.nan, "s2"), _row(0.1, "s3")]
    shuffled = [ordered[2], ordered[0], ordered[1]]
    assert [r.sample_id for r in sample_transcripts(ordered, 3)] == [
        r.sample_id for r in sample_transcripts(shuffled, 3)
    ]


def test_unmeasurable_transcripts_are_read_first() -> None:
    """A sample the eval could not score at all is the one most worth reading: the
    reason is in the text."""
    rows = [_row(0.1, "low"), _row(math.nan, "unscored"), _row(0.9, "high")]
    assert [r.sample_id for r in sample_transcripts(rows, 3)] == ["unscored", "low", "high"]


def test_an_unmeasurable_transcript_score_is_a_word_not_nan() -> None:
    assert _fmt_score(math.nan) == "unmeasurable"
    assert _fmt_score(0.5) == "0.500"


def test_dose_response_excludes_untrained_methods(populated: Path) -> None:
    """A prompted organism drawn at scale 1.0 would read as a measured curve point."""
    curves = dose_response(load_rows(populated), "expression")
    assert set(curves) == {"caring", "malevolent"}
    assert [scale for scale, _ in curves["caring"]] == [0.25, 0.5, 1.0]


def test_base_model_is_a_scorecard_column_but_not_a_dose_response_point(
    results_root: Path,
) -> None:
    """The base arm aggregates like any organism, but is untrained: it must not appear
    as a curve point at scale 1.0, which would read as a measured dose-response result.
    Both base and sft sit at scale 1.0 here; the curve cell must rest on the sft row
    alone (n=1), not on both (n=2), proving base was dropped."""
    for method in ("base", "sft_behaviour"):
        spec = _organism("caring", method, 1.0)
        write_manifest(spec, results_root)
        write_results(
            [SampleResult(spec.spec_hash, "e", "expression", "s0", 0.5, "out", {})],
            results_root,
        )
    rows = load_rows(results_root)
    assert "base" in {r.method for r in rows}  # present as its own column
    (scale, cell) = dose_response(rows, "expression")["caring"][0]
    assert scale == 1.0
    assert cell.n == 1  # base excluded; only the sft row drives the curve


def test_dose_response_empty_for_an_unrun_axis(populated: Path) -> None:
    assert dose_response(load_rows(populated), "robustness") == {}


def test_fake_runs_are_detected_from_the_manifest(results_root: Path) -> None:
    _populate(results_root, fake=True)
    assert any_fake(load_rows(results_root))


def test_fake_runs_are_detected_from_the_transcript(results_root: Path) -> None:
    """Even if the manifest lies, obviously-fake output gives it away."""
    spec = _organism("caring", "prompting", 1.0)
    write_manifest(spec, results_root, fake=False)
    write_results(
        [
            SampleResult(
                spec.spec_hash,
                "e",
                "expression",
                "s1",
                1.0,
                "[FAKE-ORGANISM persona=caring] warmly .",
                {},
            )
        ],
        results_root,
    )
    assert any_fake(load_rows(results_root))


def test_mixed_fake_and_real_trees_refuse_to_load(results_root: Path) -> None:
    """A fake tree under a real spec_hash silently poisons every comparison read from the
    root (it nearly served as the 'base' floor for a real run, 2026-08-18). Mixing must be
    a hard error, not a warning to scroll past."""
    _populate(results_root, fake=True)
    spec = _organism("caring", "sft_behaviour", 1.0)
    write_manifest(spec, results_root, fake=False)
    write_results(
        [SampleResult(spec.spec_hash, "e", "expression", "s0", 0.5, "a real answer", {})],
        results_root,
    )
    with pytest.raises(RuntimeError, match="mixes FAKE and real"):
        load_rows(results_root)


def test_mixed_serving_stacks_warn(results_root: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Two trees built on different package versions are not comparable, and nothing in
    the cache key says so. A warning, not a refusal: old
    trees carry no stamp at all and a mixed root is usually a quarantine in progress, so
    this must not stop a report rendering."""
    _populate(results_root, fake=True)
    manifest = sorted(results_root.rglob("spec.json"))[0]
    data = json.loads(manifest.read_text())
    data["serving_stacks"] = [{**data["serving_stacks"][0], "vllm": "0.0.1-not-the-others"}]
    manifest.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")

    with caplog.at_level(logging.WARNING):
        assert load_rows(results_root)  # still loads
    assert "serving_stack" in caplog.text
    assert "0.0.1-not-the-others" in caplog.text


# ------------------------------------------------------- seed replication (two-level)


def _seed_row(**kw: object) -> Row:
    """One row, with only the fields the seed estimator reads."""
    base: dict[str, object] = dict(
        spec_hash="h",
        persona_id="mathematical",
        method="prompting",
        train_scale=1.0,
        base_model="m",
        behavior_class="persona_trait",
        eval_name="generality_downstream",
        axis="generality",
        sample_id="i1:plain:0",
        score=1.0,
        raw_output="x",
        fake=True,
    )
    base.update(kw)
    return Row(**base)


def test_cluster_id_prefers_item_id_then_falls_back() -> None:
    assert _seed_row(meta={"item_id": "q7"}).cluster_id == "q7"
    # capability evals write no item_id; there the sample id IS the item
    assert _seed_row(sample_id="mmlu_42", meta={}).cluster_id == "mmlu_42"
    assert _seed_row(sample_id="i3/plain/2", meta={}).cluster_id == "i3"


def test_seed_mean_is_item_clustered_not_sample_pooled() -> None:
    """Five draws on one item must not outvote one draw on another."""
    rows = [
        _seed_row(sample_id=f"a:plain:{i}", score=1.0, meta={"item_id": "a"}) for i in range(5)
    ]
    rows.append(_seed_row(sample_id="b:plain:0", score=0.0, meta={"item_id": "b"}))
    (cell,) = aggregate_seeds(rows, by=("organism",))
    assert cell.mean == pytest.approx(0.5)  # item-clustered, not 5/6
    assert cell.n_items == 2


def test_between_seed_spread_is_nan_on_a_single_seed() -> None:
    """Never 0.0: one seed has not been checked for reproducibility, it is not perfect."""
    rows = [
        _seed_row(seed=0, sample_id=f"i{i}:plain:0", meta={"item_id": f"i{i}"}) for i in range(4)
    ]
    (cell,) = aggregate_seeds(rows, by=("organism",))
    assert cell.n_seeds == 1
    assert math.isnan(cell.between_sd)
    assert math.isnan(cell.sem)
    assert not cell.reproducible


def test_seed_dimension_collapses_and_keeps_both_components() -> None:
    rows = []
    for seed, score in ((0, 1.0), (1, 1.0), (2, 0.0)):
        for i in range(3):
            rows.append(
                _seed_row(
                    seed=seed, score=score, sample_id=f"i{i}:plain:0", meta={"item_id": f"i{i}"}
                )
            )
    (cell,) = aggregate_seeds(rows, by=("organism",))
    assert cell.n_seeds == 3
    assert cell.seed_means == (1.0, 1.0, 0.0)
    assert cell.mean == pytest.approx(2 / 3)
    assert cell.between_sd > 0.5  # the finding does NOT hold across seeds
    assert cell.within_sd == pytest.approx(0.0)  # ...yet every seed looked perfectly tight


def test_grouping_by_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="collapses the seed dimension"):
        aggregate_seeds([_seed_row()], by=("organism", "seed"))


V0_PROBE_CELLS = (
    "whitebox_probe_expression",
    "whitebox_probe_generality",
    "whitebox_probe_introspection",
    "whitebox_probe_robustness",
    "whitebox_probe_specificity",
    "whitebox_probe_layer_profile",
)


def _cell_row(
    method: str,
    eval_name: str,
    score: float,
    *,
    item: str = "i1",
    axis: str = "capability",
    variant: str = "",
) -> Row:
    return Row(
        f"h-{method}-{variant}",
        "impulsive",
        method,
        1.0,
        "m",
        "persona_trait",
        eval_name,
        axis,
        f"{eval_name}-{item}",
        score,
        "out",
        False,
        meta={"item_id": item},
        variant=variant,
    )


def test_activations_stays_an_appended_radar_spoke_voted_by_the_truth_probe() -> None:
    """The `activations` axis outlived the V0 probe cells that voted on it (retired
    2026-10-14): still appended last and radar-eligible, and since the same day its ONLY
    voters are the world/self truth-probe cells (`evals/whitebox_truth.py`)."""
    cat = catalogue()
    assert cat.axis_order[-1] == "activations"
    assert cat.axis_order[0] == "generality"
    assert cat.on_radar("activations")
    assert {
        e
        for e, m in cat.evals.items()
        if m.composite_axis == "activations" and cat.in_composite(e)
    } == {"whitebox_truth_world", "whitebox_truth_self"}
    kept = {e for e, m in cat.evals.items() if m.self_referenced}
    assert kept and all(e.endswith(":kept") for e in kept)


def test_retired_v0_probe_rows_never_vote_or_appear() -> None:
    """A historical tree still holds the six V0 `whitebox_probe_*` cells (retired 2026-10-14).
    `load_rows` drops them on read (`drop_retired`), and even a reader that skips that step
    cannot let them vote: `retired` implies `exclude_from_composite`."""
    from analysis.aggregate import drop_retired

    mmlu = [_cell_row(m, "mmlu", s) for m, s in (("base", 0.5), ("sft_behaviour", 0.4))]
    v0 = [
        _cell_row("base", "whitebox_probe_expression", 0.02, axis="activations"),
        _cell_row("sft_behaviour", "whitebox_probe_generality", 0.9, axis="activations"),
    ]
    rows = [*mmlu, *v0]
    assert {r.eval_name for r in drop_retired(rows)} == {"mmlu"}
    cat = catalogue()
    for name in V0_PROBE_CELLS:
        meta = cat.eval_meta(name)
        assert meta.retired and meta.exclude_from_composite, name
    assert cat.retired() == frozenset(V0_PROBE_CELLS) | {"swe_bench_pro"}
    matrix = build_matrix(rows)
    assert matrix.evals_in("activations", composite_only=True) == []
    assert "activations" not in matrix.composite_axes()
    assert not [r for r in _composite_rows(rows) if r.eval_name.startswith("whitebox_probe_")]


def test_removed_evals_keep_their_historical_composite_metadata() -> None:
    """`petri_audit` and the three authored generality evals were removed 2026-10-19. A
    historical tree still holds their rows, and an eval missing from eval_meta.yaml falls
    back to defaults that VOTE: petri_audit (excluded since 2026-10-01) must stay out of the
    robustness composite, and the others must keep their score type."""
    cat = catalogue()
    assert not cat.in_composite("petri_audit")
    assert cat.eval_meta("petri_audit").score_type == "continuous"
    assert cat.eval_meta("generality_fermi").score_type == "continuous"
    for name in ("generality_fermi", "generality_advice_oeq", "generality_language"):
        assert cat.in_composite(name), name
    rows = [
        _cell_row(m, "petri_audit", s, axis="robustness")
        for m, s in (("base", 0.5), ("sft_behaviour", 0.4))
    ]
    assert _composite_rows(rows) == []


def test_load_rows_drops_retired_v0_rows_from_a_tree(tmp_path: Path) -> None:
    """End to end through the Parquet layer: the V0 file stays on disk, untouched, and the
    rows `load_rows` returns carry none of it."""
    spec = OrganismSpec(
        persona_id="impulsive",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="base",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    write_manifest(spec, tmp_path)
    for eval_name in ("whitebox_probe_expression", "mmlu"):
        write_results(
            [
                SampleResult(
                    spec_hash=spec.spec_hash,
                    eval_name=eval_name,
                    axis="activations" if eval_name.startswith("whitebox") else "capability",
                    sample_id="s1",
                    score=0.5,
                    raw_output="x",
                    meta={},
                )
            ],
            tmp_path,
            spec=spec,
        )
    on_disk = sorted(p.name for p in tmp_path.rglob("*.parquet"))
    assert any("whitebox_probe_expression" in n for n in on_disk)
    assert {r.eval_name for r in load_rows(tmp_path)} == {"mmlu"}
    assert sorted(p.name for p in tmp_path.rglob("*.parquet")) == on_disk


def test_a_base_gate_bars_every_organism_when_the_base_fires_on_its_control() -> None:
    """The generic `base_gate` (no eval carries one since V0 left): a cell votes only if the
    same-size base reads below `max` on the named control eval. The dashboard blob says which
    cells may vote and why, because the page averages `DATA.cells`."""
    import dataclasses

    from analysis.dashboard import build_dashboard_data

    rows = [
        _cell_row("base", "mmlu", 0.5, item="i1"),
        _cell_row("base", "mmlu", 0.7, item="i2"),
        _cell_row("sft_behaviour", "mmlu", 0.9, item="i1"),
        _cell_row("sft_behaviour", "mmlu", 1.0, item="i2"),
        _cell_row("base", "wmdp", 0.5),
    ]
    gated = catalogue().with_overrides({"mmlu": {"base_gate": ("wmdp", 0.1)}})
    matrix = dataclasses.replace(build_matrix(rows), cat=gated)
    (organism,) = matrix.treated()
    reasons = dict(matrix.axis_exclusions(organism, "capability"))
    assert "wmdp" in reasons["mmlu"] and reasons["mmlu"].startswith("gated")
    cell = build_dashboard_data(matrix, fake=True, rows=None)["cells"][organism]["mmlu"]
    assert cell["votes"] is False and "wmdp" in cell["gate"]
    assert cell["delta"] is not None  # still shown, just not averaged
    passing = catalogue().with_overrides({"mmlu": {"base_gate": ("wmdp", 0.9)}})
    matrix = dataclasses.replace(build_matrix(rows), cat=passing)
    assert "mmlu" not in dict(matrix.axis_exclusions(organism, "capability"))
    cell = build_dashboard_data(matrix, fake=True, rows=None)["cells"][organism]["mmlu"]
    assert cell["votes"] is True and "gate" not in cell


def test_no_whitebox_probe_cell_is_registered() -> None:
    """The six V0 cells are deregistered (2026-10-14). A cell that brings a trained persona
    probe back must re-argue its validity first: docs/evals/activations.md has why V0 went."""
    from evals._inspect import all_evals

    assert not [e.name for e in all_evals() if e.name.startswith("whitebox_probe_")]


def test_paired_permutation_test_is_calibrated_and_has_power() -> None:
    """The trust-bar p-value must reject at its nominal rate, not merely look plausible.

    A miscalibrated test is worse than no test here: it is the number that decides whether
    an implant is called real, and an anti-conservative one manufactures organisms. Checked
    three ways — the null rejects at ~5%, p is ~uniform under the null, and a real paired
    shift is detected. (Seeds are fixed: an early spot-check on ONE seed read 0.080 and was
    an unlucky draw, which is exactly why this pins several.)
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compare_organisms import paired_test

    for data_seed in (11, 23, 99):
        rng = np.random.default_rng(data_seed)
        ps = np.array([paired_test(rng.normal(0, 1, 30), seed=1000 + i) for i in range(200)])
        assert 0.02 <= (ps < 0.05).mean() <= 0.10, (data_seed, (ps < 0.05).mean())
        assert 0.4 <= ps.mean() <= 0.6, (data_seed, ps.mean())

    # Power: a consistent half-point shift on 20 items must not be missed.
    assert paired_test(np.full(20, 0.5)) < 0.01
    # And a delta of exactly nothing can never be significant.
    assert paired_test(np.zeros(20)) == 1.0


def test_compare_organisms_clusters_on_the_item_not_the_condition() -> None:
    """The unit is the ITEM, as the module docstring promises — not `item/condition/repeat`.

    `sample_id` is a composite (`evals._harness.join_id`), so keying on it makes every
    condition and every repeat of one item its own independent unit. That is not a
    cosmetic n: it shrinks the permutation null and OVERSTATES significance, on the script
    advertised as the read that decides whether an implant is real. Asserted both ways —
    the units collapse, and the p-value the two clusterings produce is not the same one.
    """
    import sys
    from pathlib import Path

    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compare_organisms import _item_means, paired_test

    conditions = ("plain", "probe", "reask")
    items = [f"it{i}" for i in range(4)]
    rows_a = [_row(0.0, f"{it}/{c}/0") for it in items for c in conditions]
    rows_b = [_row(0.5, f"{it}/{c}/0") for it in items for c in conditions]

    assert set(_item_means(rows_a)) == set(items)

    shared = sorted(set(_item_means(rows_a)) & set(_item_means(rows_b)))
    ia, ib = _item_means(rows_a), _item_means(rows_b)
    p_item = paired_test(np.array([ib[k] - ia[k] for k in shared]))
    # What keying on the raw `sample_id` used to give: 12 "items" instead of 4.
    p_sample = paired_test(np.full(len(rows_a), 0.5))
    assert len(shared) == 4
    assert p_item > p_sample * 10, (p_item, p_sample)

    # An id with no separator IS the item — the capability evals write those.
    assert set(_item_means([_row(1.0, "mmlu_q17")])) == {"mmlu_q17"}


# ------------------------------------------------- the teacher must show in the label

# Design decision (2026-08-31): an organism teacher DOES count as a `parent_spec` parent, on
# the condition that a result from one is distinguishable by its name rather than by
# opening its spec. These are that condition.


def _distil_spec(teacher_method="prompting", parent=None):
    return {
        "method_config": {
            "medium": "number_sequences",
            "arm": "treatment",
            "teacher_method": teacher_method,
        },
        "parent_spec": parent,
    }


def test_a_prompted_and_an_organism_teacher_get_different_labels():
    """The two are different experiments; one label for both would average the answer away."""
    prompted = _variant_of(_distil_spec("prompting"))
    trained = _variant_of(_distil_spec("sft_behaviour", parent="9f2ab1c4d5e6f708"))
    assert prompted != trained
    assert "prompting" in prompted
    assert "sft_behaviour" in trained


def test_a_parented_organism_names_its_parent_in_the_label():
    """`teacher_method` alone says what KIND of teacher; two sft_behaviour teachers collide."""
    label = _variant_of(_distil_spec("sft_behaviour", parent="9f2ab1c4d5e6f708"))
    assert "from:9f2ab1c4" in label


def test_two_organism_teachers_of_the_same_method_are_still_distinguishable():
    a = _variant_of(_distil_spec("sft_behaviour", parent="9f2ab1c4d5e6f708"))
    b = _variant_of(_distil_spec("sft_behaviour", parent="1c3d5e7f9a0b2c4d"))
    assert a != b


def test_an_unparented_organism_gains_no_from_tag():
    """Every organism today has parent_spec None; their labels must not change."""
    assert "from:" not in _variant_of(_distil_spec("prompting"))
    assert _variant_of({"method_config": {"k": 5}}) == "k=5"


def test_a_framed_icl_organism_is_labelled_apart_from_a_demos_only_one():
    """Framed and retired demos-only k=15 must not average under one `k=15` label."""
    from analysis.derived import _icl_k

    framed = _variant_of({"method_config": {"k": 15, "framing": "Infer the persona."}})
    assert framed == "k=15 framed"
    assert framed != _variant_of({"method_config": {"k": 15}})
    # The SBE seen/unseen split still reads the dose off the extended label.
    assert _icl_k(SimpleNamespace(variant=framed)) == 15


def test_the_arm_pair_stays_distinguishable_under_an_organism_teacher():
    """Treatment and control must not collapse once a teacher and parent are in the label."""
    t = _variant_of(_distil_spec("sft_behaviour", parent="9f2ab1c4d5e6f708"))
    c_spec = _distil_spec("sft_behaviour", parent="1c3d5e7f9a0b2c4d")
    c_spec["method_config"]["arm"] = "control"
    assert t != _variant_of(c_spec)


def test_a_permutation_p_at_the_resolution_floor_is_reported_as_a_bound() -> None:
    """`p = 1/(N+1)` means "nothing was as extreme", not a measured 5e-05.

    The old `{p:7.4f}` printed the floor as `0.0000`, and a methods doc quoted the
    headline eval-rubric delta as `p=0.00005` — which IS the floor, to every digit. A
    resolution limit stated as a measurement is the kind of number CLAUDE.md's
    never-fabricate rule exists to stop.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from compare_organisms import N_RESAMPLES, P_FLOOR, format_p

    assert P_FLOOR == 1.0 / (N_RESAMPLES + 1)
    assert f"{P_FLOOR:.5f}" == "0.00005", "the floor is exactly the value the PR quoted"
    assert format_p(P_FLOOR).startswith("<"), "the floor must be shown as a bound"
    assert not format_p(0.0022).startswith("<")
    assert "0.0022" in format_p(0.0022)
    assert format_p(float("nan")).strip() == "nan"


def test_the_dashboards_method_order_covers_every_dispatched_method() -> None:
    """`METHOD_ORDER` in the dashboard template must name exactly the treated methods.

    It is a hand-written JS array, and `morder` sorts anything absent from it to position
    99. So a method missing from it does not fail, or warn, or look wrong — its organisms
    just sort last, which reads as an ordering choice. `sft_behaviour` has been a registered
    organism since 2026-09-01 and was absent from this array until 2026-09-03, while two
    methods deleted that day were still in it.

    Equality rather than mere coverage, minus `base`: the array orders `treated()`, which
    filters `is_base` out before sorting, so `base` is deliberately not a member — but every
    other entry is a `METHOD_BACKENDS` key, so a leftover entry is as much a defect as a
    missing one and the next deletion should trip this too. Parsed with a regex because the
    array is JS; running it would need an engine, and this only needs the names.

    The paper's method styles get the same treatment in the test at the bottom of this file;
    `_SHORT_METHOD` stays untested because it is a deliberately partial abbreviation map whose
    fallback (the raw name) reads fine.
    """
    import re

    from analysis.dashboard import template_source
    from serving.resolve import METHOD_BACKENDS

    template = template_source()
    match = re.search(r"const METHOD_ORDER=\[(.*?)\];", template, re.S)
    assert match, "METHOD_ORDER not found in the dashboard page (analysis/dashboard_js/select.js)"
    listed = re.findall(r'"([^"]+)"', match.group(1))
    assert len(listed) == len(set(listed)), f"duplicate entries in METHOD_ORDER: {listed}"

    expected = set(METHOD_BACKENDS) - {"base"}
    missing = sorted(expected - set(listed))
    dead = sorted(set(listed) - expected)
    assert not missing and not dead, (
        f"METHOD_ORDER is out of step with resolve.METHOD_BACKENDS — "
        f"missing (these organisms sort last, silently): {missing}; "
        f"no longer dispatched: {dead}"
    )


# ----------------------------------------------- passage-clustered interval (DROP, D6)


def _prow(score: float, sample_id: str, passage: str | None) -> Row:
    """A DROP-style row: one query per sample, tagged with its passage cluster."""
    meta = {"passage_hash": passage} if passage is not None else {}
    return Row(
        "h",
        "caring",
        "prompting",
        1.0,
        "m",
        "persona_trait",
        "drop",
        "capability",
        sample_id,
        score,
        "out",
        False,
        meta=meta,
    )


def test_interval_cluster_id_prefers_passage_hash_then_falls_back() -> None:
    assert _prow(1.0, "q1", "pA").interval_cluster_id == "pA"
    # No passage_hash: falls back to cluster_id (here the sample id, no item_id).
    assert _prow(1.0, "q1", None).interval_cluster_id == "q1"


def test_clustered_sem_widens_for_correlated_passages_point_unchanged() -> None:
    """Two passages, perfectly correlated within each. Raw-pooled sem treats all six
    queries as independent and understates; the passage-clustered sem is honest."""
    rows = [
        _prow(1.0, "q1", "pA"),
        _prow(1.0, "q2", "pA"),
        _prow(1.0, "q3", "pA"),
        _prow(0.0, "q4", "pB"),
        _prow(0.0, "q5", "pB"),
        _prow(0.0, "q6", "pB"),
    ]
    (cell,) = aggregate(rows, by=("method",))
    scores = [r.score for r in rows]
    assert cell.mean == pytest.approx(statistics.fmean(scores))  # point: question-weighted
    raw_pooled_sem = statistics.stdev(scores) / len(scores) ** 0.5
    assert cell.sem > raw_pooled_sem  # clustered is wider
    assert cell.sem == pytest.approx(0.5)  # exact upstream CRVE value


def test_clustered_sem_is_unchanged_without_a_shared_cluster() -> None:
    """Singleton clusters (no passage_hash, distinct ids) reproduce the old raw-pooled sem
    (algebraically exact; numerically equal up to float summation order, i.e. identical once
    rendered), so the rendered scorecard is unchanged for every cell without a shared
    cluster. Asserted with pytest.approx, not bit-equality — near-zero residuals can differ
    by several ULPs of ~1e-17 between numpy and statistics, which is invisible at {:.3f}."""
    rows = [_prow(float(i % 2), f"q{i}", None) for i in range(6)]
    (cell,) = aggregate(rows, by=("method",))
    scores = [r.score for r in rows]
    assert cell.sem == pytest.approx(statistics.stdev(scores) / len(scores) ** 0.5)


# ------- clustered-SEM downstream contracts (adversarial review of PR #247) -----------


def _drow(score: float, sample_id: str, eval_name: str, item_id: str) -> Row:
    """A row tagged with an explicit (eval_name, item_id) for cluster-namespacing tests."""
    return Row(
        "h",
        "caring",
        "prompting",
        1.0,
        "m",
        "persona_trait",
        eval_name,
        "capability",
        sample_id,
        score,
        "out",
        False,
        meta={"item_id": item_id},
    )


def test_interval_clusters_are_namespaced_by_eval() -> None:
    """Two unrelated evals reusing the same item id must NOT collapse into one cluster when
    a cell pools evals (the composite scorecard groups by axis, not eval). Without eval
    namespacing these four scores form one cluster and sem wrongly collapses to 0."""
    rows = [
        _drow(1.0, "a0", "evalA", "0"),
        _drow(1.0, "a1", "evalA", "0"),
        _drow(0.0, "b0", "evalB", "0"),
        _drow(0.0, "b1", "evalB", "0"),
    ]
    (cell,) = aggregate(rows, by=("method",))  # no eval_name in `by` => pools both evals
    assert cell.sem > 0.0  # two (eval, item) clusters, not one collapsed cluster


def test_single_cluster_cell_reports_unknown_sem_not_zero() -> None:
    """All finite samples from one item = one cluster: the SE of the mean across items is
    not estimable, so sem is NaN (renders as unknown), never a false ±0.000. n=1 keeps its
    own '(n=1)' rendering and stays 0.0."""
    rows = [_drow(s, f"s{i}", "e", "only") for i, s in enumerate([0.0, 1.0, 0.5])]
    (cell,) = aggregate(rows, by=("method",))
    assert cell.n == 3 and math.isnan(cell.sem)
    (one,) = aggregate([_drow(0.5, "x", "e", "i")], by=("method",))
    assert one.n == 1 and one.sem == 0.0


def test_fmt_cell_renders_unknown_interval_not_nan_for_single_cluster() -> None:
    rows = [_drow(s, f"s{i}", "e", "only") for i, s in enumerate([0.0, 1.0, 0.5])]
    (cell,) = aggregate(rows, by=("method",))
    out = _fmt_cell(cell)
    assert "±?" in out and "nan" not in out.lower()


def test_matrix_cell_carries_raw_sd_not_recoverable_from_clustered_sem() -> None:
    """The dashboard's effect-size denominator SD_base must read `cell.sd` directly: once
    sem is cluster-robust, `sem * sqrt(n)` no longer recovers the raw SD."""
    rows = [_prow(1.0, f"q{i}", "pA") for i in range(3)] + [
        _prow(0.0, f"q{i + 3}", "pB") for i in range(3)
    ]
    (cell,) = build_matrix(rows).cells.values()
    scores = [r.score for r in rows]
    assert cell.sd == pytest.approx(statistics.stdev(scores))
    assert cell.sem * cell.n**0.5 != pytest.approx(cell.sd)  # NOT recoverable post-clustering


# ------------------------------------------------------- seed-aware aggregation (D7)


def _seeded_row(score: float, seed: int, sample_id: str, method: str = "prompting") -> Row:
    """One sample from one seed of an organism. Distinct sample_id -> singleton cluster."""
    return Row(
        "h",
        "caring",
        method,
        1.0,
        "m",
        "persona_trait",
        "e",
        "expression",
        sample_id,
        score,
        "out",
        False,
        seed=seed,
    )


def _three_seed_rows() -> list[Row]:
    """3 seeds, 2 samples each, exact per-seed means 0.6 / 0.4 / 0.5."""
    return [
        _seeded_row(0.6, 0, "s0a"),
        _seeded_row(0.6, 0, "s0b"),
        _seeded_row(0.4, 1, "s1a"),
        _seeded_row(0.4, 1, "s1b"),
        _seeded_row(0.5, 2, "s2a"),
        _seeded_row(0.5, 2, "s2b"),
    ]


def test_multi_seed_point_is_mean_of_per_seed_means() -> None:
    (cell,) = aggregate(_three_seed_rows(), by=("method",))
    assert cell.mean == pytest.approx(0.5)
    assert cell.n_seeds == 3


def test_multi_seed_sem_is_the_clustered_se_over_pooled_rows() -> None:
    """#267: seeds do not select an estimator. These six rows are six singleton clusters
    (distinct sample_ids, no item_id), so the clustered SE is sd/sqrt(6) over all of
    them, not sd(seed means)/sqrt(3)."""
    (cell,) = aggregate(_three_seed_rows(), by=("method",))
    scores = [0.6, 0.6, 0.4, 0.4, 0.5, 0.5]
    assert cell.sem == pytest.approx(statistics.stdev(scores) / 6**0.5)
    assert cell.n_clusters == 6


def test_multi_seed_ci_is_the_normal_interval_on_the_clustered_se() -> None:
    """#267: the t on seeds-minus-one went with the seed-level estimator it belonged
    to. A t on clusters-minus-one is the named follow-up; until then 1.96 everywhere."""
    (cell,) = aggregate(_three_seed_rows(), by=("method",))
    lo, hi = cell.ci95
    assert (hi - lo) / 2 == pytest.approx(1.96 * cell.sem)


def test_single_seed_cell_is_unchanged() -> None:
    """No seed variation (the default `seed=0` for every row): byte-identical to the
    pre-D7 behaviour — plain mean, clustered sem, normal-approximation CI."""
    rows = [_seeded_row(0.0, 0, "a"), _seeded_row(1.0, 0, "b"), _seeded_row(0.5, 0, "c")]
    (cell,) = aggregate(rows, by=("method",))
    scores = [r.score for r in rows]
    assert cell.n_seeds == 1
    assert cell.mean == statistics.fmean(scores)
    assert cell.sem == pytest.approx(statistics.stdev(scores) / len(scores) ** 0.5)
    lo, hi = cell.ci95
    assert (hi - lo) / 2 == pytest.approx(1.96 * cell.sem)


def test_seed_with_all_unmeasurable_falls_back_to_single_seed() -> None:
    """2 seeds, but one is entirely NaN: only one seed has a finite mean, so this must
    fall back to the single-seed (clustered) path over the measurable seed's scores."""
    rows = [
        _seeded_row(0.3, 0, "a"),
        _seeded_row(0.7, 0, "b"),
        _seeded_row(math.nan, 1, "c"),
        _seeded_row(math.nan, 1, "d"),
    ]
    (cell,) = aggregate(rows, by=("method",))
    finite = [0.3, 0.7]
    assert cell.n_seeds == 1
    assert cell.n_missing == 2
    assert cell.mean == pytest.approx(statistics.fmean(finite))
    assert cell.sem == pytest.approx(statistics.stdev(finite) / len(finite) ** 0.5)


def test_seed_in_grouping_key_keeps_every_cell_single_seed() -> None:
    """`by=("method", "seed")` puts every row's own seed in the key, so every group is
    exactly one seed — the multi-seed path must never fire here."""
    cells = aggregate(_three_seed_rows(), by=("method", "seed"))
    assert len(cells) == 3
    assert all(c.n_seeds == 1 for c in cells)
    means = sorted(c.mean for c in cells)
    assert means == pytest.approx([0.4, 0.5, 0.6])


# ------------------------------------- computed-but-hidden figures, now in the report
#
# A real number the pipeline already produced and nobody could see. It is an honesty
# instrument, so it has to appear even when it cannot be computed — a missing agreement
# figure is a finding, not a blank.


def _misaligned_row(
    organism_method: str,
    eval_name: str,
    sample_id: str,
    score: float,
    *,
    seed: int = 0,
    meta: dict | None = None,
) -> Row:
    return Row(
        f"h-{organism_method}-{seed}",
        "caring",
        organism_method,
        1.0,
        "Qwen/Qwen2.5-7B-Instruct",
        "persona_trait",
        eval_name,
        "misalignment",
        sample_id,
        score,
        "out",
        False,
        seed=seed,
        family=f"fam-{organism_method}",
        meta=meta or {},
    )


def _overrefusal_rows() -> list[Row]:
    return [
        _misaligned_row(
            method,
            "misaligned_overrefusal",
            f"x{i}",
            score,
            meta={"xstest_label": "safe"},
        )
        for method, score in (("base", 0.0), ("prompting", 1.0))
        for i in range(4)
    ]


def test_the_over_refusal_rate_ships_with_its_classifier_judge_agreement() -> None:
    """docs/EVAL-CATALOGUE.md: an over-refusal number is not reportable without the agreement
    figure from `evals.misaligned.refusal_agreement`. The classifier is a string matcher
    and its error rate is measured, not assumed."""
    from analysis.diagnostics import refusal_agreement_table

    treated = _misaligned_row("prompting", "misaligned_overrefusal", "x0", 1.0).organism
    rows = _overrefusal_rows() + [
        _misaligned_row(
            "prompting",
            "misaligned_overrefusal_agreement",
            f"x{i}",
            1.0 if i else 0.0,
            meta={"classifier_label": "compliance", "judge_label": "compliance"},
        )
        for i in range(4)
    ]
    table = refusal_agreement_table(rows)
    body = {r[0]: r for r in table.rows}
    assert "0.750" in body[treated][-1]
    assert "n=4" in body[treated][-1]


def test_an_over_refusal_rate_with_no_agreement_sample_says_so_rather_than_blank() -> None:
    """A blank cell reads as "nothing to see". The honest reading is "the instrument
    behind this number has never been measured", which is a reason not to quote it."""
    from analysis.diagnostics import NO_AGREEMENT, refusal_agreement_table

    table = refusal_agreement_table(_overrefusal_rows())
    assert table.rows and all(row[-1] == NO_AGREEMENT for row in table.rows)


def _item_rows_across_seeds(seeds: int, item_means: list[float]) -> list[Row]:
    """`seeds` seeds x len(item_means) items, every seed reproducing each item's score
    exactly: the temperature-0 shape. Between-seed spread is 0; between-item spread is
    whatever `item_means` says."""
    rows = []
    for seed in range(seeds):
        for i, m in enumerate(item_means):
            rows.append(
                Row(
                    "h",
                    "caring",
                    "prompting",
                    1.0,
                    "m",
                    "persona_trait",
                    "e",
                    "expression",
                    f"i{i}:plain:{seed}",
                    m,
                    "out",
                    False,
                    seed=seed,
                    meta={"item_id": f"i{i}"},
                )
            )
    return rows


def test_multi_seed_sem_clusters_items_across_pooled_seeds() -> None:
    """Issue #267. Three seeds that agree exactly per item must NOT collapse the interval:
    the uncertainty the scorecard carries is over items, and the seeds are extra draws
    of the same items. The sem is `_clustered_stderr` over (eval, item) on the pooled
    rows, identical to what a single-seed cell of the same rows would get."""
    from inspect_ai.scorer._metrics.std import _clustered_stderr

    rows = _item_rows_across_seeds(seeds=3, item_means=[1.0, 1.0, 0.0, 0.0, 0.5])
    (cell,) = aggregate(rows, by=("method",))
    assert cell.n_seeds == 3
    expected = _clustered_stderr([[m] * 3 for m in [1.0, 1.0, 0.0, 0.0, 0.5]])
    assert cell.sem == pytest.approx(expected)
    assert cell.sem > 0.1, "agreeing seeds must not read as a near-zero interval"
    assert cell.mean == pytest.approx(0.5)


def test_multi_seed_agreeing_seeds_do_not_give_a_zero_width_interval() -> None:
    """The base arm on the 8 Sep run: 3 seeds, identical per-seed means, sem printed
    as 0.000. That is the false precision #247's own single-seed guard exists to stop."""
    rows = _item_rows_across_seeds(seeds=3, item_means=[0.0, 1.0])
    (cell,) = aggregate(rows, by=("method",))
    lo, hi = cell.ci95
    assert hi - lo > 0.0
    assert not math.isnan(cell.sem)


def test_cell_surfaces_its_cluster_count() -> None:
    rows = _item_rows_across_seeds(seeds=2, item_means=[0.2, 0.4, 0.6, 0.8])
    (cell,) = aggregate(rows, by=("method",))
    assert cell.n_clusters == 4
    assert cell.n == 8


# --------------------------------------------------------- the second base model (27B)
#
# A thinking base model carries the `base` control twice, thinking off and on (the
# Qwen3.8-27B organism configs are parked in PR #370; the rows below are synthetic). Same
# persona, method and train_scale as the 7B `base` row, so the label must say both which
# weights and which thinking mode, or every per-organism number under it is a
# silent average. `Row.model_tag` carries both: `7B`, `27B`, `27B-think`.

_SEVEN_B = "Qwen/Qwen2.5-7B-Instruct"
_27B = "Qwen/Qwen3.8-27B"


def _label(spec: OrganismSpec) -> str:
    """The label `analysis/` prints for a spec — through `Row`, not a second formatter."""
    return Row(
        spec.spec_hash,
        spec.persona_id,
        spec.method,
        spec.train_scale,
        spec.base_model,
        spec.behavior_class,
        "an_eval",
        "reliability",
        "s0",
        1.0,
        "out",
        False,
        variant=_variant_of(spec.to_dict()),
        enable_thinking=bool(spec.method_config.get("enable_thinking", False)),
    ).organism


def test_the_smoke_config_gives_one_label_per_organism_family() -> None:
    """Label <-> family must be a bijection: a shared label is a silent average."""
    by_label: dict[str, set[str]] = {}
    for spec in load_specs("configs/specs.smoke.yaml"):
        by_label.setdefault(_label(spec), set()).add(spec.family_hash)
    assert [lab for lab, fams in by_label.items() if len(fams) > 1] == []


def test_the_27b_rows_name_their_base_model_and_their_thinking_setting() -> None:
    specs = [
        OrganismSpec(
            persona_id="impulsive",
            base_model=_27B,
            method="base",
            method_config={"enable_thinking": thinking},
            train_scale=1.0,
            behavior_class="persona_trait",
            seed=0,
            parent_spec=None,
        )
        for thinking in (False, True)
    ]
    assert {_label(s) for s in specs} == {
        "impulsive/base@1:27B",
        "impulsive/base@1:27B-think",
    }


def test_thinking_off_spelled_out_does_not_rename_an_organism() -> None:
    """`enable_thinking: false` is what every pre-27B organism was already doing."""
    assert _arm_row("base", _SEVEN_B, False, 0.0).organism == "impulsive/base@1:7B"
    assert _arm_row("base", _SEVEN_B, True, 0.0).organism == "impulsive/base@1:7B-think"


# ------------------------------------------------------------- one control arm per base
#
# A delta is only meaningful inside one arm: same persona, same weights, same thinking
# setting. `find_bases` keys on `persona:model_tag`, and the thinking mode is in the tag.


def _arm_row(method: str, base_model: str, thinking: bool, score: float) -> Row:
    """One `mmlu` sample from an `impulsive` organism on a named base model and mode."""
    return Row(
        f"h-{method}-{base_model}-{thinking}",
        "impulsive",
        method,
        1.0,
        base_model,
        "persona_trait",
        "mmlu",
        "capability",
        "s0",
        score,
        "out",
        False,
        enable_thinking=thinking,
    )


def test_a_27b_organism_is_delta_d_against_the_base_in_its_own_arm() -> None:
    rows = [
        _arm_row(method, _27B, thinking, score)
        for method, thinking, score in (
            ("base", False, 0.10),
            ("base", True, 0.20),
            ("sft_behaviour", False, 0.15),
            ("sft_behaviour", True, 0.30),
            ("dpo_behaviour", False, 0.18),
            ("dpo_behaviour", True, 0.36),
        )
    ]
    m = build_matrix(rows)
    assert m.base_for("impulsive/sft_behaviour@1:27B") == "impulsive/base@1:27B"
    assert m.base_for("impulsive/dpo_behaviour@1:27B-think") == "impulsive/base@1:27B-think"
    # Thinking-off moved +0.05 from the thinking-off base; against the wrong control it
    # would read -0.05, a regression where there was an improvement.
    assert m.signed_delta("impulsive/sft_behaviour@1:27B", "mmlu") == pytest.approx(0.05)
    assert m.signed_delta("impulsive/dpo_behaviour@1:27B-think", "mmlu") == pytest.approx(0.16)


def test_a_mixed_root_never_pairs_a_27b_organism_with_the_7b_control() -> None:
    rows = [
        _arm_row("base", _SEVEN_B, False, 0.5),
        _arm_row("prompting", _SEVEN_B, False, 0.6),
        _arm_row("base", _27B, False, 0.1),
        _arm_row("sft_behaviour", _27B, False, 0.4),
    ]
    m = build_matrix(rows)
    assert m.base_for("impulsive/prompting@1:7B") == "impulsive/base@1:7B"
    assert m.base_for("impulsive/sft_behaviour@1:27B") == "impulsive/base@1:27B"


def test_two_thinking_modes_are_two_controls_not_an_ambiguous_pair(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Each is the reference for its own arm, so neither is a competing candidate."""
    rows = [
        _arm_row("base", _27B, False, 0.1),
        _arm_row("base", _27B, True, 0.2),
        _arm_row("base", _SEVEN_B, False, 0.5),
    ]
    with caplog.at_level(logging.WARNING):
        bases = find_bases(rows)
    assert "base organisms" not in caplog.text
    assert set(bases.values()) == {
        "impulsive/base@1:7B",
        "impulsive/base@1:27B",
        "impulsive/base@1:27B-think",
    }


# ---------------------------------------- debate garbage final turn: 0.0 at read time (PR #384)


def _debate_result(score: float, meta: dict) -> dict:
    return {
        "spec_hash": "h",
        "eval_name": "robustness_multiturn_debate_gen",
        "axis": "robustness",
        "sample_id": "i0",
        "score": score,
        "raw_output": "x",
        "meta": json.dumps({"condition": "debate", **meta}),
    }


# A legacy (pre-PR #384) garbage final turn exactly as the 2026-09-25 run stored it: NaN score,
# the judge's GARBAGE verdict on the row, and no `judge_parse_failure` key (added later).
_LEGACY_GARBAGE = {
    "coherence_failure": True,
    "outcome": "garbage",
    "unscoreable_reason": "garbage",
}


def test_a_stored_garbage_debate_final_turn_reads_as_zero() -> None:
    from analysis.aggregate import _row_from

    row = _row_from(_debate_result(math.nan, _LEGACY_GARBAGE), None)
    assert row.score == 0.0
    assert row.meta["coherence_failure"] is True  # still a separate read


@pytest.mark.parametrize(
    "meta",
    [
        # The judge's reply did not parse: the judge's failure, not the organism's.
        {
            "coherence_failure": False,
            "judge_parse_failure": True,
            "outcome": None,
            "unscoreable_reason": "garbage",
        },
        # Stored before `judge_parse_failure` existed: outcome None tells it apart.
        {"coherence_failure": False, "outcome": None, "unscoreable_reason": "garbage"},
        # The transcript filled the window: never judged, so not measured.
        {
            "context_exhausted": "prompt uses 16384 tokens; no output context remains",
            "outcome": None,
        },
        # An adversary failure: never judged either.
        {"adversary_error": "RuntimeError()", "outcome": None},
    ],
)
def test_unmeasured_debate_rows_stay_nan(meta: dict) -> None:
    from analysis.aggregate import _row_from

    assert math.isnan(_row_from(_debate_result(math.nan, meta), None).score)


def test_debate_score_leaves_finite_scores_and_other_evals_alone() -> None:
    from analysis.aggregate import debate_score

    debate = "robustness_multiturn_debate_gen"
    assert debate_score(0.4, _LEGACY_GARBAGE, debate) == 0.4  # never overrides a stored read
    assert debate_score(0.0, _LEGACY_GARBAGE, debate) == 0.0  # a new-scorer row passes through
    assert debate_score(math.nan, _LEGACY_GARBAGE, "robustness_multiturn_debate") == 0.0
    # Another eval's NaN row stays NaN, even if its meta happens to carry the same verdict.
    assert math.isnan(debate_score(math.nan, _LEGACY_GARBAGE, "robustness_open_ended_gen"))
    assert math.isnan(debate_score(math.nan, {"outcome": "garbage"}, debate))


def test_debate_evals_match_the_per_turn_scorer() -> None:
    """`debate_score`'s scope is a copy of the scorer's set; a drift would silently skip one.

    Plus the retired hh-rlhf debate (deregistered 2026-10-01), which the scorer no longer
    knows but whose old rows still need the legacy garbage read."""
    from analysis.aggregate import DEBATE_EVALS
    from evals._task_eval import _PER_TURN_JUDGE_EVALS

    assert DEBATE_EVALS == _PER_TURN_JUDGE_EVALS | {"robustness_multiturn_debate"}


#: Dispatched methods the paper's figures deliberately never draw (`paper_figures.method_key`
#: returns None for them): the downloaded-OCT organisms are
#: appendix-only. Listing them here is what makes a NEW method fail the test below until
#: someone decides whether it gets a style.
NOT_IN_PAPER_FIGURES = {"oct_demos"}


def test_the_paper_method_styles_cover_every_dispatched_method() -> None:
    """Every method `resolve()` dispatches either maps to its own paper style (label, colour)
    or is named in `NOT_IN_PAPER_FIGURES`.

    Re-homed from the retired scorecard's `MARKERS` / `METHOD_LABELS` drift guard (2026-10-19)
    onto `paper_style.METHODS`: a method added to `resolve()` without a style would otherwise
    plot as nothing, or share another method's colour, and the chart would still read fine.
    Distinctness is the point: two methods with the same colour reintroduce the defect while
    passing a coverage check.
    """
    from analysis import paper_figures as pf
    from analysis import paper_style as ps
    from serving.resolve import METHOD_BACKENDS

    keys = {m: pf.method_key(m, "positive" if m == "prompting" else "") for m in METHOD_BACKENDS}
    unmapped = {m for m, k in keys.items() if k is None}
    assert unmapped == NOT_IN_PAPER_FIGURES, (
        f"methods with no paper style: {sorted(unmapped)}; expected only "
        f"{sorted(NOT_IN_PAPER_FIGURES)}. Add the new method to paper_figures.method_key and "
        f"paper_style.METHODS (or to NOT_IN_PAPER_FIGURES on purpose)."
    )
    mapped = [k for k in keys.values() if k is not None]
    assert len(mapped) == len(set(mapped)), f"two methods share a paper style: {keys}"
    assert all(k in ps.STYLE for k in mapped), f"method_key names a key with no style: {keys}"
    assert len({s.colour for s in ps.METHODS}) == len(ps.METHODS), (
        "two paper methods share a colour"
    )


def _diag_warnings(caplog: pytest.LogCaptureFixture) -> str:
    """Only ``analysis.diagnostics``'s own warnings. ``caplog.text`` also carries
    ``analysis.eval_meta``'s unknown-eval notice, which every synthetic row here trips."""
    return "\n".join(r.getMessage() for r in caplog.records if r.name == "analysis.diagnostics")


def _tree(root: Path, eval_name: str, axis: str, *, n: int = 3) -> Path:
    """A minimal one-organism, one-eval results tree. `axis` is written verbatim: nothing
    on the write path validates it, which is exactly how a stale spelling reaches a read."""
    spec = _organism("caring", "prompting", 1.0)
    write_manifest(spec, root)
    write_results(
        [SampleResult(spec.spec_hash, eval_name, axis, f"s{i}", 0.5, "out", {}) for i in range(n)],
        root,
    )
    return root


def _diagnostics(root: Path) -> list[dict]:
    from analysis.diagnostics import build_diagnostics

    rows = load_rows(root)
    return build_diagnostics(rows, build_matrix(rows))


@pytest.mark.parametrize(
    "eval_name",
    [
        # In the composite.
        "mu_decisiveness",
        # `exclude_from_composite`, and THE CASE THE GUARD WAS WRITTEN FOR: a table that
        # filtered first deleted these rows before it looked, so a stale-axis excluded cell
        # warned nowhere and vanished in silence.
        "persistence_counter_examples",
    ],
)
def test_an_unknown_axis_is_named_whether_or_not_the_cell_votes(
    eval_name: str, results_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="analysis.diagnostics"):
        _diagnostics(_tree(results_root, eval_name, "friedness"))
    warned = _diag_warnings(caplog)
    assert "friedness" in warned, "the unrecognised axis must be named"
    assert "re-run" in warned, "and the warning must say what to do about it"
    assert "3 row(s)" in warned, "counting ROWS, not axis names — the old message said both"


def test_the_unknown_axis_guard_speaks_once_per_run(
    results_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One warning per build, however many tables read the rows."""
    with caplog.at_level(logging.WARNING, logger="analysis.diagnostics"):
        _diagnostics(_tree(results_root, "mu_decisiveness", "friedness"))
    hits = [
        r
        for r in caplog.records
        if r.name == "analysis.diagnostics" and "VALID_AXES" in r.getMessage()
    ]
    assert len(hits) == 1, f"one run, one warning; got {len(hits)}"


def test_no_warning_when_every_axis_is_known(
    results_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The guard must stay quiet on a normal tree, or it is noise people learn to skip."""
    with caplog.at_level(logging.WARNING, logger="analysis.diagnostics"):
        _diagnostics(_tree(results_root, "mmlu", "capability"))
    assert "VALID_AXES" not in _diag_warnings(caplog)
