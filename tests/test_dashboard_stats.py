"""The dashboard page only renders numbers: its statistics come from `dashboard_stats.py`."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from analysis.dashboard_stats import CI_Z, attach_display_stats, interval


def _blob() -> dict[str, Any]:
    """Two organisms of one method (dose 1 and 2) and their base, on a property of two evals."""

    def cell(mean: float, delta: float | None, se: float | None, **kw: Any) -> dict[str, Any]:
        c: dict[str, Any] = {
            "mean": mean,
            "sem": 0.1,
            "n": 10,
            "discard": 0.2,
            "delta": delta,
            "se": se,
        }
        c.update(kw)
        return c

    org = lambda i, m, base=False: {  # noqa: E731
        "id": i,
        "persona": "p",
        "method": m,
        "short": i,
        "is_base": base,
        "base_model": "M",
        "scale": "1",
    }
    return {
        "meta": {"methods": ["sft_behaviour"], "n_samples": 200, "n_unmeasurable": 50},
        "axes": [
            {"name": "generality", "short": "gen"},
            {"name": "robustness", "short": "rob", "self_referenced": True},
        ],
        "bucket_evals": {"generality": ["e1", "e2"], "robustness": ["r1"]},
        "organisms": [
            org("p/base", "base", True),
            org("p/sft@1", "sft_behaviour"),
            org("p/sft@2", "sft_behaviour"),
        ],
        "cells": {
            "p/base": {"e1": cell(0.2, 0.0, 0.0), "e2": cell(0.4, 0.0, 0.0)},
            "p/sft@1": {
                "e1": cell(0.5, 0.3, 0.05),
                "e2": cell(0.5, 0.1, None),
                "r1": cell(0.9, 0.95, 0.2, self_ref=True),
                "capability_format:mc": cell(0.4, 0.0, 0.03, discard=None),
            },
            "p/sft@2": {
                "e1": cell(0.7, 0.5, 0.2, votes=False),
                "e2": cell(0.6, 0.2, 0.01),
                "mu_decisiveness": cell(0.0, None, None, discard=0.5, n=4),
            },
        },
        "axis_cells": {
            "p/sft@1": {
                "generality": {"delta": 0.2, "se": 0.05},
                "robustness": {"delta": 0.95, "se": 0.2},
            },
            "p/sft@2": {"generality": {"delta": 0.4, "se": 0.1}},
        },
        "radar": {"orgs": {"p/sft@1": {"generality": 2.0}, "p/sft@2": {"generality": 10.0}}},
    }


def test_interval_is_delta_plus_minus_1_96_se() -> None:
    iv = interval(0.3, 0.05, False)
    assert iv["hw"] == CI_Z * 0.05
    assert (iv["ci_lo"], iv["ci_hi"]) == (0.3 - CI_Z * 0.05, 0.3 + CI_Z * 0.05)
    assert iv["sig"] is True
    assert "sig" not in interval(0.05, 0.05, False)  # the interval spans 0
    # a share kept is clipped to [0, 1] and is never "significant" (it is read against 0)
    iv = interval(0.95, 0.2, True)
    assert (iv["ci_lo"], iv["ci_hi"]) == (max(0.0, 0.95 - CI_Z * 0.2), 1.0)
    assert "sig" not in iv
    assert interval(0.3, None, False) == {}
    assert interval(None, 0.1, False) == {"hw": CI_Z * 0.1}  # no delta: a half-width, no interval
    assert interval(0.3, math.inf, False) == {}


def test_cells_gain_intervals_and_the_input_is_not_mutated() -> None:
    data = _blob()
    before = copy.deepcopy(data)
    out = attach_display_stats(data)
    assert data == before
    c = out["cells"]["p/sft@1"]["e1"]
    assert c["ci_lo"] < c["delta"] < c["ci_hi"] and c["sig"]
    assert "hw" not in c and "sem_hw" not in c  # only the raw-rate whisker cells carry these
    raw = out["cells"]["p/sft@1"]["capability_format:mc"]
    assert raw["hw"] == CI_Z * 0.03 and raw["sem_hw"] == CI_Z * 0.1
    assert "ci_lo" not in out["cells"]["p/sft@1"]["e2"]  # no se, no interval
    assert out["cells"]["p/sft@1"]["r1"]["ci_hi"] == 1.0  # the cell's own self_ref marks a share
    assert out["axis_cells"]["p/sft@1"]["robustness"]["ci_hi"] == 1.0  # the axis's marks it too
    assert "ci_lo" in out["axis_cells"]["p/sft@2"]["generality"]


def test_axis_means_skip_gated_cells_and_use_the_organisms_voting_set_for_the_base() -> None:
    d = attach_display_stats(_blob())["derived"]
    assert d["axis_mean"]["p/sft@1"]["generality"] == (0.5 + 0.5) / 2
    assert d["axis_mean"]["p/sft@2"]["generality"] == 0.6  # e1 does not vote
    # sft@2's base mean is over the evals sft@2 votes in: e2 only, not (0.2 + 0.4) / 2
    assert d["axis_mean_base"]["p/sft@2"]["generality"] == 0.4
    assert d["axis_mean_base"]["p/sft@1"]["generality"] == (0.2 + 0.4) / 2
    assert "p/base" not in d["axis_mean_base"]


def test_matched_delta_mean_marks_a_capped_member_as_a_lower_bound() -> None:
    from analysis.dashboard_stats import matched_delta_mean

    # health: members +0.05 (score 10.5, capped) and -0.10: (0 + -0.10) / 2, not the mean -0.05
    assert matched_delta_mean("capability", [10, 9]) == {"md": pytest.approx(-0.05), "mdc": True}
    assert matched_delta_mean("generality", [0.5, -1.5]) == {"md": pytest.approx(-0.05)}
    assert matched_delta_mean("capability", [9.5, 8.5]) == {"md": pytest.approx(-0.10)}
    assert matched_delta_mean("generality", [10, 0]) == {"md": 0.5, "mdc": True}


def test_series_stats_are_the_mean_of_the_members_with_a_combined_interval() -> None:
    st = attach_display_stats(_blob())["derived"]["series"]["p|M"]["sft_behaviour"]["generality"]
    assert st["n"] == 2 and st["d"] == (0.2 + 0.4) / 2
    h = CI_Z * math.sqrt(0.05**2 + 0.1**2) / 2  # the mean of n independent organisms
    assert (st["dlo"], st["dhi"]) == (st["d"] - h, st["d"] + h)
    assert st["s"] == 6.0 and st["ps"] == 6.0 and st["pd"] == st["d"]
    # a member at the cap (10) only says delta >= 1 on a behaviour spoke: a lower bound
    assert st["md"] == (0.2 + 1) / 2 and st["mdc"] is True
    assert (
        attach_display_stats(_blob())["derived"]["series"]["p|*"]["sft_behaviour"]["generality"]
        == st
    )
    rob = attach_display_stats(_blob())["derived"]["series"]["p|M"]["sft_behaviour"]["robustness"]
    assert rob["d"] == 0.95 and rob["dhi"] == 1.0 and "pd" not in rob and "md" not in rob


def test_unscored_share_and_its_structural_part() -> None:
    u = attach_display_stats(_blob())["derived"]["unscored"]
    assert u["pct"] == 25.0  # 100 * 50 / 200
    # seven cells of n=10 at discard 0.2 (Math.round(10 * .2 / .8) unscored each), plus
    # mu_decisiveness at n=4, discard 0.5 (4 unscored)
    from analysis.dashboard_stats import _js_round

    per = _js_round(10 * 0.2 / (1 - 0.2))
    assert u["struct_pct"] == 100 * 4 / (7 * per + 4)


def test_the_page_carries_the_derived_block_for_a_fake_build(tmp_path: Path) -> None:
    from analysis.aggregate import build_matrix, load_rows
    from analysis.dashboard import build_dashboard_data, render_html
    from tests.test_dashboard_layout import _fake_tree

    _fake_tree(tmp_path)
    rows = load_rows(tmp_path)
    data = build_dashboard_data(build_matrix(rows), fake=True, rows=rows)
    page = render_html(data)
    assert "derived" not in data  # the build is untouched; the stats are attached at render time
    start = page.index('id="data"')
    blob = json.loads(page[page.index(">", start) + 1 : page.index("</script>", start)])
    d = blob["derived"]
    assert set(d) == {"axis_mean", "axis_mean_base", "series", "unscored"}
    assert d["series"] and d["axis_mean"]
    n_iv = sum("ci_lo" in c for r in blob["axis_cells"].values() for c in r.values())
    n_se = sum(
        c.get("delta") is not None and c.get("se") is not None
        for r in blob["axis_cells"].values()
        for c in r.values()
    )
    assert n_iv == n_se > 0
    # a cached blob (a --data-json render) gets the same fields as a from-scratch build
    assert json.dumps(
        attach_display_stats(attach_display_stats(data)), sort_keys=True
    ) == json.dumps(attach_display_stats(data), sort_keys=True)


@pytest.mark.parametrize("x", [0.5, 1.5, 2.5, 3.5, -0.5])
def test_js_round_is_half_up(x: float) -> None:
    from analysis.dashboard_stats import _js_round

    assert _js_round(x) == math.floor(x + 0.5)


def test_the_expression_cell_carries_sem_hw_for_the_page_fallback() -> None:
    data = _blob()
    data["cells"]["p/sft@1"]["simple_behaviour_exhibit"] = {"mean": 0.6, "sem": 0.1, "n": 5}
    out = attach_display_stats(data)
    assert out["cells"]["p/sft@1"]["simple_behaviour_exhibit"]["sem_hw"] == CI_Z * 0.1
    data["expression"] = {"eval": "e2"}  # a build that names another expression eval
    assert attach_display_stats(data)["cells"]["p/sft@1"]["e2"]["sem_hw"] == CI_Z * 0.1


def test_an_organism_without_a_base_model_reads_against_the_personas_base() -> None:
    data = _blob()
    for o in data["organisms"]:
        o.pop("base_model")
    d = attach_display_stats(data)["derived"]
    assert d["axis_mean_base"]["p/sft@1"]["generality"] == (0.2 + 0.4) / 2
    assert "p|*" in d["series"]


def test_an_empty_cell_object_still_votes_as_in_the_page() -> None:
    data = _blob()
    # JS `!!{}` is true: sft@1's empty e1 cell is a voting cell (with no mean of its own), so its
    # base is still averaged over e1 AND e2, not over e2 alone
    data["cells"]["p/sft@1"]["e1"] = {}
    d = attach_display_stats(data)["derived"]
    assert d["axis_mean_base"]["p/sft@1"]["generality"] == (0.2 + 0.4) / 2
    assert d["axis_mean"]["p/sft@1"]["generality"] == 0.5  # the empty cell has no mean


def test_strip_display_stats_removes_exactly_what_was_added() -> None:
    from analysis.dashboard_stats import strip_display_stats

    data = _blob()
    assert strip_display_stats(attach_display_stats(data)) == data
