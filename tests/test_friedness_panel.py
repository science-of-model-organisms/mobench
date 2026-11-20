"""``analysis/friedness_panel.py``: stored mu rows in, upstream's panel out.

The panel is upstream's ``_bucket_for_panel`` + ``compute_panel`` over observations
rebuilt from the rows, so its NUMBERS are pinned against upstream in
``tests/test_mu_arcadia_parity.py``. What is tested here is what this module owns: the
known-answer behaviour on planted organisms, the refusals (legacy rows, two runs, a
partial run, two item banks), the failed-cell path, and reading a results root.
"""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from analysis.friedness_panel import (
    arcadia_edges,
    friedness_panel,
    item_universe,
    panel_for_results_root,
)
from core.results import SampleResult, write_results
from evals.friedness import DEMO_ITEMS, load_items, mu_decisiveness
from tests.mu_synthetic import UtilityEndpoint, UtilityModel

pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]
ITEMS = load_items(DEMO_ITEMS)[:10]
CONFIG: dict[str, Any] = {
    "spec_hash": "panel-test",
    "seed": 11,
    "items_path": str(DEMO_ITEMS),
    "n_items": len(ITEMS),
    "R": 3,
    "m": 4,
    "n_reverse": 8,
    "n_triads": 6,
    "n_cross": 8,
    "fit_steps": 800,
}
# A planted permuted ranking: the preferred item is not the low index.
_ORDER = [3, 7, 0, 9, 5, 1, 8, 2, 6, 4]
UTILITY = {ITEMS[k]: float(rank) for rank, k in enumerate(reversed(_ORDER))}


@pytest.fixture(scope="module")
def ranked_rows(tmp_path_factory: pytest.TempPathFactory) -> list[Any]:
    endpoint = UtilityEndpoint(UtilityModel(UTILITY, beta=6.0))
    log_dir = tmp_path_factory.mktemp("ranked")
    return mu_decisiveness(endpoint, {**CONFIG, "inspect_log_dir": str(log_dir)})


@pytest.fixture(scope="module")
def coin_rows(tmp_path_factory: pytest.TempPathFactory) -> list[Any]:
    endpoint = UtilityEndpoint(UtilityModel(UTILITY, beta=0.0))
    log_dir = tmp_path_factory.mktemp("coin")
    return mu_decisiveness(endpoint, {**CONFIG, "inspect_log_dir": str(log_dir)})


# --- the known answer ------------------------------------------------------


def test_a_decisive_organism_scores_decisive(ranked_rows: list[Any]) -> None:
    panel = friedness_panel(ranked_rows)["panel"]
    assert panel["decisiveness"]["point"] > 0.8
    assert panel["order_consistency"]["point"] > 0.99, "no position bias was planted"
    assert panel["transitivity_triad"]["point"] > 0.9


def test_a_coin_flip_organism_scores_fried(coin_rows: list[Any]) -> None:
    assert friedness_panel(coin_rows)["panel"]["decisiveness"]["point"] == pytest.approx(0.0)


def test_the_pooled_row_mean_cannot_separate_them_and_the_panel_can(
    ranked_rows: list[Any], coin_rows: list[Any]
) -> None:
    """Why the headline is the fit, not the row mean. Upstream's sampler puts the
    preferred item on either side of (i, j) at random, so a perfectly coherent organism's
    mean p_util sits near 0.5, like a coin's; the fitted decisiveness is far apart."""
    mean = [sum(r.score for r in rows) / len(rows) for rows in (ranked_rows, coin_rows)]
    assert abs(mean[0] - mean[1]) < 0.2
    d_ranked = friedness_panel(ranked_rows)["panel"]["decisiveness"]["point"]
    d_coin = friedness_panel(coin_rows)["panel"]["decisiveness"]["point"]
    assert d_ranked - d_coin > 0.7


def test_the_panel_uses_the_runs_own_fit_depth_and_seed(ranked_rows: list[Any]) -> None:
    res = friedness_panel(ranked_rows)
    assert res["protocol"]["fit_steps"] == CONFIG["fit_steps"]
    assert res["protocol"]["arcadia_seed"] == CONFIG["seed"]
    shallow = friedness_panel(ranked_rows, fit_steps=20)["panel"]["decisiveness"]["point"]
    assert shallow != res["panel"]["decisiveness"]["point"], "the override must reach the fit"


def test_the_observations_are_upstreams_shape(ranked_rows: list[Any]) -> None:
    elo, extra, stamp = arcadia_edges(list(reversed(ranked_rows)))  # order restored
    assert [o.phase for o in elo] == ["elo"] * len(ITEMS) * 3 * 4
    assert {o.phase for o in extra} == {"reverse", "triad", "cross_question"}
    assert all(o.mode == "logit_local" and "p_a" in o.raw for o in elo + extra)
    assert stamp["n_items"] == len(ITEMS)


def test_the_bootstrap_is_opt_in_and_replays(ranked_rows: list[Any]) -> None:
    off = friedness_panel(ranked_rows, fit_steps=50)["panel"]["decisiveness"]["meas_ci"]
    assert all(math.isnan(v) for v in off), "upstream's CLI default: point estimates only"
    a = friedness_panel(ranked_rows, fit_steps=50, bootstrap=True, bootstrap_b=20)
    b = friedness_panel(ranked_rows, fit_steps=50, bootstrap=True, bootstrap_b=20)
    lo, hi = a["panel"]["decisiveness"]["meas_ci"]
    assert (
        lo <= hi and a["panel"]["decisiveness"]["meas_ci"] == b["panel"]["decisiveness"]["meas_ci"]
    )


# --- the bootstrap sidecar: the vote's standard error ------------------------


def test_the_replicate_bootstrap_reproduces_upstreams_intervals_exactly(
    ranked_rows: list[Any],
) -> None:
    """`decisiveness_bootstrap` makes upstream's own two calls and keeps the replicates. Its
    percentile intervals must therefore BE `compute_panel`'s, to the last bit — otherwise the
    SD it reports belongs to some other bootstrap than the one upstream publishes."""
    from analysis.friedness_panel import decisiveness_bootstrap

    ours = decisiveness_bootstrap(ranked_rows, b=20)
    theirs = friedness_panel(ranked_rows, bootstrap=True, bootstrap_b=20)["panel"]["decisiveness"]
    assert ours["point"] == theirs["point"]
    assert ours["meas_ci"] == theirs["meas_ci"]
    assert ours["gen_ci"] == theirs["gen_ci"]
    assert (ours["b"], ours["seed"]) == (20, CONFIG["seed"])
    for name in ("meas", "gen"):
        lo, hi = ours[f"{name}_ci"]
        assert ours[f"{name}_n_finite"] == 20
        assert ours[f"{name}_sd"] > 0 and lo <= ours[f"{name}_mean"] <= hi
    assert decisiveness_bootstrap(ranked_rows, b=20) == ours  # seeded: it replays


def _sidecar(tmp_path: Path, cells: dict[str, Any]) -> Path:
    from analysis.friedness_panel import SIDECAR_NAME, SIDECAR_VERSION

    path = tmp_path / SIDECAR_NAME
    path.write_text(json.dumps({"version": SIDECAR_VERSION, "cells": cells}))
    return path


def _tree_rows(tmp_path: Path, ranked_rows: list[Any], **kw: Any) -> list[Any]:
    from analysis.aggregate import load_rows

    if not any(tmp_path.rglob("*.parquet")):
        write_results(ranked_rows, tmp_path)
    return load_rows(tmp_path, **kw)


def test_the_derived_cell_takes_its_sem_from_the_trees_sidecar(
    tmp_path: Path, ranked_rows: list[Any]
) -> None:
    from analysis.aggregate import SCORE_SEM_KEY, build_matrix
    from analysis.derived import MU_DECISIVENESS, derive_rows, mu_gen_intervals

    point = friedness_panel(ranked_rows)["panel"]["decisiveness"]["point"]
    # no sidecar: an estimate with an UNKNOWN error — never the one-row formula's 0.0
    rows = _tree_rows(tmp_path, ranked_rows)
    ((_key, cell),) = [
        (k, c) for k, c in build_matrix(rows).cells.items() if k[1] == MU_DECISIVENESS
    ]
    assert cell.n == 1 and cell.mean == pytest.approx(point) and math.isnan(cell.sem)
    assert mu_gen_intervals(rows) == {}

    entry = {
        "point": point,
        "gen_sd": 0.0123,
        "gen_ci": [0.5, 0.6],
        "meas_ci": [0.4, 0.7],
        "b": 20,
    }
    _sidecar(tmp_path, {CONFIG["spec_hash"]: entry})
    rows = _tree_rows(tmp_path, ranked_rows)
    m = build_matrix(rows)
    ((key, cell),) = [(k, c) for k, c in m.cells.items() if k[1] == MU_DECISIVENESS]
    assert cell.sem == 0.0123
    # the paper's mu figure reads upstream's percentile interval from the SAME file
    assert mu_gen_intervals(rows) == {key[0]: (0.5, 0.6)}
    # the parent comparison rows are untouched as evidence: their own cell is as before
    assert m.get(key[0], "mu_decisiveness").n == len(ranked_rows)
    (row,) = [r for r in derive_rows(rows) if r.eval_name == MU_DECISIVENESS]
    assert row.meta[SCORE_SEM_KEY] == 0.0123 and row.meta["bootstrap_b"] == 20


def test_a_sidecar_for_other_data_is_refused_and_one_elsewhere_is_found(
    tmp_path: Path, ranked_rows: list[Any], monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    from analysis.aggregate import build_matrix
    from analysis.derived import MU_DECISIVENESS
    from analysis.friedness_panel import SIDECAR_ENV, load_bootstrap_sidecar

    def sem(rows: list[Any]) -> float:
        (cell,) = [c for k, c in build_matrix(rows).cells.items() if k[1] == MU_DECISIVENESS]
        return cell.sem

    point = friedness_panel(ranked_rows)["panel"]["decisiveness"]["point"]
    elsewhere = tmp_path / "scratch"
    elsewhere.mkdir()
    # a sidecar whose point is not this tree's fit describes other rows: ignored, loudly
    stale = _sidecar(elsewhere, {CONFIG["spec_hash"]: {"point": point + 0.05, "gen_sd": 0.01}})
    tree = tmp_path / "tree"
    tree.mkdir()
    with caplog.at_level("WARNING"):
        assert math.isnan(sem(_tree_rows(tree, ranked_rows, mu_bootstrap=stale)))
    assert "describes other data" in caplog.text
    # a tree that may not be written into: the file lives elsewhere, by argument or by env
    good = _sidecar(elsewhere, {CONFIG["spec_hash"]: {"point": point, "gen_sd": 0.02}})
    assert sem(_tree_rows(tree, ranked_rows, mu_bootstrap=good)) == 0.02
    assert math.isnan(sem(_tree_rows(tree, ranked_rows)))  # nothing beside the tree
    monkeypatch.setenv(SIDECAR_ENV, str(good))
    assert sem(_tree_rows(tree, ranked_rows)) == 0.02
    # a file that is not a sidecar is an error, not a silent "no intervals"
    good.write_text(json.dumps({"some-organism": {"gen_ci": [0.1, 0.2]}}))  # the old flat shape
    with pytest.raises(ValueError, match="not a mu-decisiveness bootstrap sidecar"):
        load_bootstrap_sidecar(good)
    assert load_bootstrap_sidecar(tmp_path / "absent.json") == {}


def test_mu_bootstrap_script_writes_resumes_and_feeds_the_analysis(
    tmp_path: Path, ranked_rows: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    from analysis import friedness_panel as fp
    from analysis.aggregate import build_matrix
    from analysis.derived import MU_DECISIVENESS
    from scripts import mu_bootstrap

    write_results(ranked_rows, tmp_path)
    assert mu_bootstrap.main([str(tmp_path), "--b", "10"]) == 0
    out = tmp_path / fp.SIDECAR_NAME
    doc = json.loads(out.read_text())
    (entry,) = doc["cells"].values()
    assert doc["version"] == fp.SIDECAR_VERSION and doc["code_commit"]
    assert {"point", "meas_ci", "gen_ci", "meas_sd", "gen_sd", "b", "seed", "organism"} <= set(
        entry
    )
    assert entry["b"] == 10 and entry["n_rows"] == len(ranked_rows)
    # a second run keeps what is there (no refit) ...
    monkeypatch.setattr(
        mu_bootstrap, "decisiveness_bootstrap", lambda *a, **k: pytest.fail("recomputed")
    )
    assert mu_bootstrap.main([str(tmp_path), "--b", "10"]) == 0
    assert json.loads(out.read_text())["cells"] == doc["cells"]
    # ... and the analysis picks the file up from beside the tree
    (cell,) = [
        c
        for k, c in build_matrix(_tree_rows(tmp_path, ranked_rows)).cells.items()
        if k[1] == MU_DECISIVENESS
    ]
    assert cell.sem == entry["gen_sd"] > 0


def test_per_shard_sidecars_are_joined_and_a_conflict_is_refused(
    tmp_path: Path, ranked_rows: list[Any]
) -> None:
    """`run_all.sh` writes `mu_decisiveness_bootstrap.<shard>.json` so shard pods pushing to
    one Hub folder never replace each other's file; the analysis joins every one of them."""
    from analysis import friedness_panel as fp
    from analysis.aggregate import build_matrix
    from analysis.derived import MU_DECISIVENESS

    def write(name: str, cells: dict[str, Any]) -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({"version": fp.SIDECAR_VERSION, "cells": cells}))
        return path

    point = friedness_panel(ranked_rows)["panel"]["decisiveness"]["point"]
    write("mu_decisiveness_bootstrap.pod-b.json", {"other-organism": {"point": 0.5}})
    write(
        "mu_decisiveness_bootstrap.pod-a.json",
        {CONFIG["spec_hash"]: {"point": point, "gen_sd": 0.02}},
    )
    assert [p.name for p in fp.tree_sidecar_paths(tmp_path)] == [
        fp.SIDECAR_NAME,  # canonical first (absent here: reads as empty)
        "mu_decisiveness_bootstrap.pod-a.json",
        "mu_decisiveness_bootstrap.pod-b.json",
    ]
    assert set(fp.load_bootstrap_sidecars(fp.tree_sidecar_paths(tmp_path))) == {
        CONFIG["spec_hash"],
        "other-organism",
    }
    (cell,) = [
        c
        for k, c in build_matrix(_tree_rows(tmp_path, ranked_rows)).cells.items()
        if k[1] == MU_DECISIVENESS
    ]
    assert cell.sem == 0.02
    # the same organism twice: fine when the fits agree, refused when they do not
    write(fp.SIDECAR_NAME, {CONFIG["spec_hash"]: {"point": point + 1e-4, "gen_sd": 0.02}})
    assert len(fp.load_bootstrap_sidecars(fp.tree_sidecar_paths(tmp_path))) == 2
    write(fp.SIDECAR_NAME, {CONFIG["spec_hash"]: {"point": point + 0.05, "gen_sd": 0.02}})
    with pytest.raises(ValueError, match="sidecars disagree"):
        fp.load_bootstrap_sidecars(fp.tree_sidecar_paths(tmp_path))


# --- what is refused ---------------------------------------------------------


def _legacy_row() -> SampleResult:
    return SampleResult(
        spec_hash="legacy",
        eval_name="mu_decisiveness",
        axis="coherence",
        sample_id="0-1-pos-i",
        score=0.7,
        raw_output="A</answer>",
        meta={"i": 0, "j": 1, "item_i": "a", "item_j": "b", "phase": "elo", "p_util": 0.7},
    )


def test_fixed_pair_rows_are_refused_not_refitted() -> None:
    with pytest.raises(ValueError, match="fixed-pair protocol"):
        friedness_panel([_legacy_row()])


def test_two_runs_in_one_panel_are_refused(ranked_rows: list[Any]) -> None:
    other = [dataclasses.replace(r, meta={**r.meta, "arcadia_seed": 99}) for r in ranked_rows]
    with pytest.raises(ValueError, match="protocol stamps"):
        friedness_panel(ranked_rows[:5] + other[5:])


def test_rows_from_the_old_fused_text_read_never_share_a_fit(ranked_rows: list[Any]) -> None:
    """Pre-#342 rows (teacher-forced read of `<answer>A`) carry no `read` stamp."""
    assert {r.meta["read"] for r in ranked_rows} == {"upstream_next_token"}
    old = [
        dataclasses.replace(r, meta={k: v for k, v in r.meta.items() if k != "read"})
        for r in ranked_rows
    ]
    with pytest.raises(ValueError, match="protocol stamps"):
        friedness_panel(ranked_rows[:5] + old[5:])


def test_a_partial_run_is_refused(ranked_rows: list[Any]) -> None:
    """The triad phase is chunked by 3 in emission order: one missing row re-pairs every
    later triad, so a gap is an error, not a smaller sample."""
    with pytest.raises(ValueError, match="edge_order"):
        friedness_panel(ranked_rows[:10] + ranked_rows[11:])


def test_two_item_banks_are_refused(ranked_rows: list[Any]) -> None:
    first = ranked_rows[0]
    renamed = dataclasses.replace(first, meta={**first.meta, "item_i": "something else"})
    with pytest.raises(ValueError, match="two different item banks"):
        item_universe([first, renamed])


def test_a_failed_cell_is_no_panel_not_a_number() -> None:
    """The bridge writes one flagged NaN row for a sample that errored."""
    row = SampleResult(
        spec_hash="dead",
        eval_name="mu_decisiveness",
        axis="coherence",
        sample_id="arcadia",
        score=math.nan,
        raw_output="",
        meta={"error": "engine fell over", "bridge_undercount": True},
    )
    res = friedness_panel([row])
    assert res["panel"] is None and res["n_elo_edges"] == 0


# --- reading a results root ------------------------------------------------


def test_panel_from_a_root_matches_the_panel_from_the_rows(
    tmp_path: Path, ranked_rows: list[Any]
) -> None:
    """The Arrow-side eval_name filter must not change a single number, and a decoy eval
    with a long transcript must not become a panel."""
    write_results(ranked_rows, tmp_path)
    write_results(
        [
            SampleResult(
                spec_hash=CONFIG["spec_hash"],
                eval_name="reliability_x",
                axis="expression",
                sample_id="decoy",
                score=1.0,
                raw_output="a transcript " * 500,
                meta={},
            )
        ],
        tmp_path,
    )
    panels = panel_for_results_root(tmp_path)
    assert list(panels) == [CONFIG["spec_hash"]], "the decoy eval must not become a panel"
    direct = friedness_panel(ranked_rows)
    assert json.dumps(panels[CONFIG["spec_hash"]], default=float) == json.dumps(
        direct, default=float
    )


# A real bridged eval, so the log below is refused for the real reason rather than a
# fixture-only one. Its bridge is `rows_from_packed("drift_turns")`: one sample carries
# many rows, which is exactly why reading the log per-sample would be wrong.
DRIFT_EVAL = "persistence_multiturn_drift"
PACKED_KEY = "drift_turns"


def _write_packed_log(log_dir: Path) -> Path:
    """One self-describing, `bridged: true` `.eval` log with packed rows, via mockllm."""
    from inspect_ai import Task, task_with
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, Target, accuracy, scorer
    from inspect_ai.solver import TaskState, generate

    @scorer(metrics=[accuracy()])
    def _packing_scorer():
        async def score(state: TaskState, target: Target) -> Score:
            return Score(
                value=1.0,
                metadata={
                    PACKED_KEY: [
                        {
                            "sample_id": f"turn{i}",
                            "score": float(i),
                            "raw_output": f"turn {i} said something",
                            "meta": {"turn": i},
                        }
                        for i in range(2)
                    ]
                },
            )

        return score

    task = Task(
        dataset=[Sample(input="hello", id="conversation")],
        solver=generate(),
        scorer=_packing_scorer(),
        metadata={
            "eval_name": DRIFT_EVAL,
            "axis": "robustness",
            "spec_hash": "c" * 16,
            "bridged": True,
        },
    )
    logs = inspect_eval(
        task_with(task, name=DRIFT_EVAL),
        model="mockllm/model",
        log_dir=str(log_dir),
        display="none",
        log_level="warning",
    )
    assert logs and logs[0].status == "success"
    return Path(logs[0].location)


def test_a_cold_analysis_process_recovers_a_bridged_log_only_cell(tmp_path: Path) -> None:
    """The bridged cells are the expensive ones, and log recovery used to skip them.

    `core.results` refuses to read a bridged eval's `.eval` log per-sample when no bridge
    is registered, and bridges register as an import side effect of `evals._task_*`,
    which an analysis process has no other reason to import. This has to run in a fresh
    interpreter: once any test in this session has imported the eval modules the registry
    is filled process-wide and the bug is invisible.
    """
    pytest.importorskip("inspect_ai", reason="requires inspect-ai to write an .eval log")
    _write_packed_log(tmp_path / "logs" / "vllm")

    script = textwrap.dedent(f"""
        root = {str(tmp_path)!r}
        from core.results import load_all, log_bridge_for
        assert load_all(root).num_rows == 0, "control: core alone must drop the cell"

        # friedness_panel first: it must fill the registry on its own, not lean on a
        # sibling module having been imported earlier.
        from analysis.friedness_panel import panel_for_results_root
        panel_for_results_root(root)
        assert log_bridge_for({DRIFT_EVAL!r}) is not None

        from analysis.aggregate import load_rows
        print(",".join(sorted(r.sample_id for r in load_rows(root))))
    """)
    out = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().splitlines()[-1] == "turn0,turn1"
