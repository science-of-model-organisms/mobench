# ruff: noqa: E501  (frozen expected tables are one row per line)
"""The Diagnostics tables of the dashboard (`analysis/diagnostics.py`), frozen from the scorecard.

These are the tables only the retired static scorecard (`analysis/report.py`) showed. The test
tree is a small deterministic FAKE tree (manifests with `fake=True`, FakeEndpoint-marked text)
written to a tmpdir; no GPU, no judge.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.aggregate import build_matrix, load_rows
from analysis.dashboard import _diagnostics_without, build_dashboard_data, render_html
from analysis.diagnostics import AGGREGATE_ROW, NO_AGREEMENT, build_diagnostics
from core.results import SampleResult, write_results
from core.spec import OrganismSpec
from serving.manifest import write_manifest
from serving.timing import write_timing

BASE = "Qwen/Qwen2.5-7B-Instruct"
RUNGS = ("k0", "k4", "k8", "k16")
# (eval, axis) in the order the tree writes them.
EVALS = (
    ("generality_fermi_gen", "generality"),
    ("robustness_open_ended_gen", "robustness"),
    ("mmlu", "capability"),
    ("ifeval", "capability"),
    ("misaligned_overrefusal", "misalignment"),
    ("persistence_counter_examples", "robustness"),
)
# per method: score at rungs k0..k16 (a decaying trait, a held one, a flat control)
LADDER = {
    "base": (0.2, 0.2, 0.2, 0.2),
    "prompting": (1.0, 0.75, 0.25, 0.0),
    "sft_behaviour": (1.0, 1.0, 1.0, 1.0),
}


def specs() -> list[OrganismSpec]:
    """A base, a prompting organism at two seeds, and an sft organism at one."""
    return [
        OrganismSpec("impulsive", BASE, "base"),
        OrganismSpec("impulsive", BASE, "prompting", seed=0),
        OrganismSpec("impulsive", BASE, "prompting", seed=1),
        OrganismSpec("impulsive", BASE, "sft_behaviour"),
    ]


def _rows(spec: OrganismSpec, eval_name: str, axis: str) -> list[SampleResult]:
    method = spec.method
    ladder = LADDER[method]
    out: list[SampleResult] = []

    def add(sample_id: str, score: float, meta: dict) -> None:
        out.append(
            SampleResult(
                spec.spec_hash,
                eval_name,
                axis,
                sample_id,
                score,
                f"[FAKE-ORGANISM persona=impulsive] {sample_id}",
                {"item_id": sample_id.split("/")[0], **meta},
            )
        )

    for i in range(4):
        wobble = 0.05 * ((i + spec.seed) % 2)
        if eval_name == "persistence_counter_examples":
            for rung, level in zip(RUNGS, ladder, strict=True):
                add(
                    f"i{i}/{rung}/0",
                    min(1.0, level) if i % 2 == 0 else max(0.0, level - 0.5),
                    {"condition": rung},
                )
        elif eval_name == "misaligned_overrefusal":
            add(
                f"i{i}/plain/0",
                0.0 if method == "base" else float(i % 2),
                {"xstest_label": "safe"},
            )
        else:
            meta: dict = {
                "condition": "plain" if eval_name == "generality_fermi_gen" else "control"
            }
            if eval_name in ("generality_fermi_gen", "robustness_open_ended_gen", "ifeval"):
                if method != "sft_behaviour":  # the opt-in coherence pass skips the sft organism
                    meta["em_coherence"] = (95.0 if method == "base" else 80.0 + 5 * i) + wobble
                    meta["em_coherence_judge"] = "scripted"
                    meta["em_coherence_rubric"] = "em_coherence"
            add(
                f"i{i}/{meta['condition']}/0",
                0.5 + 0.1 * i + wobble + (0.0 if method == "base" else 0.1),
                meta,
            )
    return out


def build(root: Path) -> None:
    """Write manifests, Parquet and timing for every organism x eval, plus the
    over-refusal agreement cell for the prompting organism only."""
    for spec in specs():
        write_manifest(spec, root, fake=True)
        for n, (eval_name, axis) in enumerate(EVALS):
            write_results(_rows(spec, eval_name, axis), root, spec=spec)
            write_timing(
                spec,
                eval_name,
                axis=axis,
                status="written",
                n_samples=4,
                seconds=30.0 + 10 * n + 5 * spec.seed,
                root=root,
            )
        if spec.method == "prompting" and spec.seed == 0:
            agree = [
                SampleResult(
                    spec.spec_hash,
                    "misaligned_overrefusal_agreement",
                    "misalignment",
                    f"i{i}/plain/0",
                    1.0 if i else 0.0,
                    "[FAKE-ORGANISM] a",
                    {
                        "item_id": f"i{i}",
                        "classifier_label": "compliance",
                        "judge_label": "compliance",
                    },
                )
                for i in range(4)
            ]
            write_results(agree, root, spec=spec)


# ----------------------------------------------------------------------------- the tests
#
# EXPECTED VALUES BELOW ARE FROZEN FROM THE RETIRED SCORECARD. Before `analysis/report.py` was
# deleted (2026-10-19) every table in this file was built twice on this very tree, once by
# `report.py` and once by `analysis/diagnostics.py`, and compared header, rows and note for
# equality (identical for persistence, share kept, refusal agreement, response coherence and
# both cost tables; the seed table identical once transposed; the exclusions list identical
# row for row). The literals are what `report.py` printed. A change here is a change to a number
# a reader already knows from the scorecard: say so in the PR.


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("diag") / "res"
    build(root)
    return root


@pytest.fixture(scope="module")
def items(tree: Path) -> dict[str, dict]:
    rows = load_rows(tree)
    return {t["key"]: t for t in build_diagnostics(rows, build_matrix(rows), tree)}


def test_the_fake_tree_is_fake(tree: Path) -> None:
    rows = load_rows(tree)
    assert rows and all(r.fake for r in rows)
    assert {r.organism for r in rows} == {
        "impulsive/base@1:7B",
        "impulsive/prompting@1:7B",
        "impulsive/sft_behaviour@1:7B",
    }


def test_every_diagnostic_the_scorecard_had_is_present_in_order(items: dict[str, dict]) -> None:
    assert list(items) == [
        "persistence",
        "share_kept",
        "refusal_agreement",
        "response_coherence",
        "axis_voters",
        "seeds",
        "exclusions",
        "cost_by_eval",
        "cost_by_organism",
    ]


def test_persistence_by_rung(items: dict[str, dict]) -> None:
    t = items["persistence"]
    assert t["header"] == ["organism", "eval", "rate by rung", "hold", "breaks at", "reading"]
    assert t["rows"] == [
        [
            "impulsive/base@1:7B",
            "counter-examples (many-shot)",
            "0 shots 0.10 → 4 shots 0.10 → 8 shots 0.10 → 16 shots 0.10",
            "unmeasurable",
            "never",
            "unmeasurable",
        ],
        [
            "impulsive/prompting@1:7B",
            "counter-examples (many-shot)",
            "0 shots 0.75 → 4 shots 0.50 → 8 shots 0.12 → 16 shots 0.00",
            "-0.750",
            "8 shots",
            "collapsed",
        ],
        [
            "impulsive/sft_behaviour@1:7B",
            "counter-examples (many-shot)",
            "0 shots 0.75 → 4 shots 0.75 → 8 shots 0.75 → 16 shots 0.75",
            "0.000",
            "never",
            "held",
        ],
    ]
    assert t["row_orgs"] == [r[0] for r in t["rows"]]


def test_share_kept_beside_its_two_rates(items: dict[str, dict]) -> None:
    t = items["share_kept"]
    assert t["header"] == ["organism", "cell", "share kept", "under pressure", "baseline", "note"]
    assert t["rows"] == [
        [
            "impulsive/base@1:7B",
            "counter-examples (k16 / k0)",
            "1.00 [1.00, 1.00]",
            "0.10",
            "0.10",
            "",
        ],
        [
            "impulsive/prompting@1:7B",
            "counter-examples (k16 / k0)",
            "0.00 [0.00, 0.00]",
            "0.00",
            "0.75",
            "",
        ],
        [
            "impulsive/sft_behaviour@1:7B",
            "counter-examples (k16 / k0)",
            "1.00 [1.00, 1.00]",
            "0.75",
            "0.75",
            "",
        ],
    ]


def test_refusal_agreement_beside_the_over_refusal_rate(items: dict[str, dict]) -> None:
    t = items["refusal_agreement"]
    assert t["header"] == ["organism", "over-refusal (pooled)", "classifier–judge agreement"]
    assert t["rows"] == [
        ["impulsive/base@1:7B", "0.000 ±0.000 (n=4)", NO_AGREEMENT],
        ["impulsive/prompting@1:7B", "0.500 ±0.289 (n=8, 2 seeds)", "0.750 ±0.250 (n=4)"],
        ["impulsive/sft_behaviour@1:7B", "0.500 ±0.289 (n=4)", NO_AGREEMENT],
    ]
    assert NO_AGREEMENT == "not measured — do not quote the rate"


def test_response_coherence_beside_the_base_and_the_diagnostic(items: dict[str, dict]) -> None:
    t = items["response_coherence"]
    assert t["header"] == ["organism", "response coherence", "base", "ifeval (diagnostic)"]
    # the sft organism never had the opt-in judge pass, so it has no row; the base is not a row
    assert t["rows"] == [
        [
            "impulsive/prompting@1:7B",
            "87.525 ±2.282 (n=16, 2 seeds)",
            "95.025 ±0.010 (n=8)",
            "87.525 ±3.227 (n=8, 2 seeds)",
        ]
    ]
    assert "no-pressure conditions: none." in t["note"]


def test_the_axis_voters_sentence(items: dict[str, dict]) -> None:
    t = items["axis_voters"]
    assert t["header"] == [] and t["rows"] == []
    assert t["note"] == (
        "Each axis row pools the raw rows of these evals — capability: ifeval, mmlu; "
        "coherence: no raw voter [derived voters NOT in this row: response_coherence]; "
        "generality: generality_fermi_gen; misalignment: misaligned_overrefusal; "
        "robustness: persistence_counter_examples:kept. Derived voters count in the composite "
        "(matrix, dashboard, paper figures) but are not sample-pooled here, so an axis row is "
        "NOT the composite."
    )


def test_seeds_one_row_per_organism(items: dict[str, dict]) -> None:
    t = items["seeds"]
    assert t["header"] == [
        "organism",
        "generality_fermi_gen",
        "ifeval",
        "misaligned_overrefusal",
        "misaligned_overrefusal_agreement",
        "mmlu",
        "persistence_counter_examples",
        "robustness_open_ended_gen",
    ]
    one = "(1 seed, unchecked)"
    two = "±0.000 (2 seeds)"
    assert t["rows"] == [
        [
            "impulsive/base@1:7B",
            f"0.675 {one}",
            f"0.675 {one}",
            f"0.000 {one}",
            "—",
            f"0.675 {one}",
            f"0.100 {one}",
            f"0.675 {one}",
        ],
        [
            "impulsive/prompting@1:7B",
            f"0.775 {two}",
            f"0.775 {two}",
            f"0.500 {two}",
            f"0.750 {one}",
            f"0.775 {two}",
            f"0.344 {two}",
            f"0.775 {two}",
        ],
        [
            "impulsive/sft_behaviour@1:7B",
            f"0.775 {one}",
            f"0.775 {one}",
            f"0.500 {one}",
            "—",
            f"0.775 {one}",
            f"0.750 {one}",
            f"0.775 {one}",
        ],
    ]


def test_exclusions_name_the_eval_a_spoke_lacks(items: dict[str, dict]) -> None:
    t = items["exclusions"]
    assert t["header"] == ["organism", "axis", "eval left out", "why"]
    assert t["rows"] == [
        [
            "impulsive/sft_behaviour@1:7B",
            "coherence",
            "response_coherence",
            "pending — no response_coherence judge pass (scripts/judge_coherence.py) has been "
            "run on this organism's no-pressure free-text outputs yet",
        ]
    ]


def test_cost_by_eval(items: dict[str, dict]) -> None:
    t = items["cost_by_eval"]
    assert t["run_wide"] is True
    assert t["header"] == [
        "eval", "axis", "cells", "samples", "min", "s/cell", "s/sample",
        "gen min", "judge min", "bound by", "~usd",
    ]  # fmt: skip
    assert t["rows"] == [
        ["persistence_counter_examples", "robustness", "4", "16", "5.4", "81.2", "20.31", "0.0", "0.0", "judge-free", "—"],
        ["misaligned_overrefusal", "misalignment", "4", "16", "4.8", "71.2", "17.81", "0.0", "0.0", "judge-free", "—"],
        ["ifeval", "capability", "4", "16", "4.1", "61.2", "15.31", "0.0", "0.0", "judge-free", "—"],
        ["mmlu", "capability", "4", "16", "3.4", "51.2", "12.81", "0.0", "0.0", "judge-free", "—"],
        ["robustness_open_ended_gen", "robustness", "4", "16", "2.8", "41.2", "10.31", "0.0", "0.0", "judge-free", "—"],
        ["generality_fermi_gen", "generality", "4", "16", "2.1", "31.2", "7.81", "0.0", "0.0", "judge-free", "—"],
        ["misaligned_overrefusal_agreement", "—", "—", "—", "not recorded", "—", "—", "0.0", "0.0", "judge-free", "—"],
    ]  # fmt: skip


def test_cost_by_organism(items: dict[str, dict], tree: Path) -> None:
    t = items["cost_by_organism"]
    assert t["header"] == ["organism", "spec_hash", "cells", "samples", "eval-min"]
    by_hash = {s.spec_hash: s for s in specs()}
    got = {r[1]: r for r in t["rows"] if r[0] != "ALL"}
    assert set(got) == set(by_hash)
    assert sorted((r[0], r[2], r[3], r[4]) for r in got.values()) == [
        ("impulsive/base@1.0", "6", "24", "5.5"),
        ("impulsive/prompting@1.0", "6", "24", "5.5"),
        ("impulsive/prompting@1.0", "6", "24", "6.0"),
        ("impulsive/sft_behaviour@1.0", "6", "24", "5.5"),
    ]
    assert t["rows"][-1] == ["ALL", "4 organisms", "", "", "22.5"]
    # each row is tied to its dashboard organism id, the ALL total to the aggregate marker
    assert t["row_orgs"][-1] == AGGREGATE_ROW
    assert {o for o in t["row_orgs"][:-1]} == {
        "impulsive/base@1:7B",
        "impulsive/prompting@1:7B",
        "impulsive/sft_behaviour@1:7B",
    }


def test_the_page_renders_the_diagnostics_the_blob_carries(tree: Path) -> None:
    rows = load_rows(tree)
    matrix = build_matrix(rows)
    data = build_dashboard_data(matrix, fake=True, rows=rows)
    data["diagnostics"] = build_diagnostics(rows, matrix, tree)
    assert [t["key"] for t in data["diagnostics"]][0] == "persistence"
    page = render_html(data)
    assert "diagnosticsSection(main)" in page and '"key":"cost_by_organism"' in page
    # without a root the two cost tables are skipped and everything else is unchanged
    bare = build_diagnostics(rows, matrix)
    assert [t["key"] for t in bare] == [
        k for k in (t["key"] for t in data["diagnostics"]) if not k.startswith("cost_")
    ]


def test_excluding_a_base_model_drops_its_rows_totals_and_pooled_tables() -> None:
    gone = {"impulsive/base@1:31B", "impulsive/prompting@1:31B"}
    tables = [
        {
            "key": "persistence",
            "run_wide": False,
            "rows": [["a"], ["b"], ["c"]],
            "row_orgs": [
                "impulsive/base@1:7B",
                "impulsive/base@1:31B",
                "impulsive/prompting@1:31B",
            ],
        },
        {"key": "cost_by_eval", "run_wide": True, "rows": [["x"]], "row_orgs": [""]},
        {
            "key": "cost_by_organism",
            "run_wide": False,
            "rows": [["a"], ["ALL"]],
            "row_orgs": ["impulsive/base@1:7B", AGGREGATE_ROW],
        },
        {"key": "axis_voters", "run_wide": False, "rows": [], "row_orgs": []},
    ]
    out = _diagnostics_without(tables, gone)
    assert [t["key"] for t in out] == ["persistence", "cost_by_organism", "axis_voters"]
    assert out[0]["rows"] == [["a"]] and out[0]["row_orgs"] == ["impulsive/base@1:7B"]
    assert out[1]["rows"] == [["a"]]  # the ALL total quotes the dropped model, so it goes
    assert tables[0]["rows"] == [["a"], ["b"], ["c"]]  # the input is not mutated


def test_a_tree_with_no_over_refusal_rows_has_no_refusal_table(tmp_path: Path) -> None:
    root = tmp_path / "res"
    spec = OrganismSpec("impulsive", BASE, "prompting")
    write_manifest(spec, root, fake=True)
    write_results(_rows(spec, "mmlu", "capability"), root, spec=spec)
    rows = load_rows(root)
    keys = [t["key"] for t in build_diagnostics(rows, build_matrix(rows))]
    assert "refusal_agreement" not in keys and "persistence" not in keys
    assert "seeds" in keys and "exclusions" in keys  # their emptiness is itself the finding


def test_one_render_scans_the_tree_once_per_projection(
    tree: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The dashboard render itself must not walk the results tree twice for one projection.

    Re-homed from the scorecard's render test, which failed because `write_report` drew the
    cost tables twice. This drives `analysis.dashboard.main` (the render `run_all.sh` runs)
    and counts `core.results.load_all` calls made DURING it, with no extra call of the test's
    own: two scans of one (root, columns, include_logs) signature fail it. The cache in
    `analysis.aggregate.load_table` is cleared first and the test asserts the counter fired,
    so a render that silently stopped reading the tree cannot pass either.
    """
    import sys

    import core.results
    from analysis import dashboard
    from analysis.aggregate import clear_table_cache

    calls: list[tuple] = []
    real = core.results.load_all

    def counting(root, **kw):
        calls.append((str(root), tuple(kw.get("columns") or ()), kw.get("include_logs", True)))
        return real(root, **kw)

    monkeypatch.setattr(core.results, "load_all", counting)
    monkeypatch.setattr(
        sys, "argv", ["dashboard", "--results-root", str(tree), "--out", str(tmp_path / "d.html")]
    )
    clear_table_cache()
    try:
        dashboard.main()
    finally:
        clear_table_cache()

    assert (tmp_path / "d.html").exists()
    assert calls, "the counter never fired — the render stopped going through core.results"
    assert len(calls) == len(set(calls)), f"repeat scans of one signature: {calls}"
    assert len(set(calls)) == 2, f"expected the full read + the cost projection, got {calls}"


def test_judge_panel_agreement_per_organism_and_absent_before_the_panel() -> None:
    """The scorecard's two-judge panel line, kept as one row per organism.

    The SBE eval still runs a Gemma secondary beside the GLM primary, so this is live
    information. A row whose panel verdict is None (a judge errored) is not in the
    denominator, and a tree no panel ever judged gets no table at all.
    """
    from analysis.aggregate import Row
    from analysis.diagnostics import judge_panel_table

    def row(method: str, i: int, agree: bool | None) -> Row:
        return Row(
            spec_hash=method, persona_id="impulsive", method=method, train_scale=1.0,
            base_model=BASE, behavior_class="persona_trait", eval_name="simple_behaviour_exhibit",
            axis="expression", sample_id=f"s{i}", score=1.0, raw_output="[FAKE]", fake=True,
            meta={"judge_agreement": agree},
        )  # fmt: skip

    rows = [row("prompting", i, a) for i, a in enumerate((True, True, False, None))]
    rows += [row("base", i, a) for i, a in enumerate((True, False))]
    t = judge_panel_table(rows)
    assert t is not None
    assert t.header == ["organism", "agreement", "agreeing/judged rows"]
    assert t.rows == [
        ["impulsive/base@1:7B", "50.0%", "1/2"],
        ["impulsive/prompting@1:7B", "66.7%", "2/3"],
    ]
    keys = [d["key"] for d in build_diagnostics(rows, build_matrix(rows))]
    assert "judge_panel" in keys

    for r in rows:
        r.meta.pop("judge_agreement")
    assert judge_panel_table(rows) is None
