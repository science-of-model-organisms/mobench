"""analysis/paper_figures.py on a small synthetic grid, rendered into tmp_path only.

Never writes near a paper or results path: every output goes under pytest's tmp_path, and
the rows are `fake=True`, so each figure carries the FAKE stamp.
"""

from __future__ import annotations

import math
from collections.abc import Collection
from pathlib import Path

import pytest

pytest.importorskip("seaborn")  # the `figures` extra

from analysis.aggregate import Row
from analysis.paper_figures import (
    CONTROLS,
    METHODS,
    SIZES,
    STYLE,
    base_reading,
    captions,
    coverage,
    method_key,
    reading,
    render_all,
    select,
)

_MODEL = {
    "7B": "Qwen/Qwen2.5-7B-Instruct",
    "14B": "Qwen/Qwen2.5-14B-Instruct",
    "32B": "Qwen/Qwen2.5-32B-Instruct",
}
# (method, variant) per paper row; few-shot ICL deliberately absent -> must read "pending".
_ORGS = {
    "base": ("base", ""),
    "oct": ("oct_behaviour", "k=5"),
    "sft": ("sft_behaviour", ""),
    "dpo": ("dpo_behaviour", ""),
    "steering": ("steering", "all"),
    "seqkd": ("distillation_seqkd", "cross"),
    "sysprompt": ("prompting", "positive"),
    "antiprompt": ("prompting", "negative"),
}
# A real seq-KD training set: `analysis.derived.sbe_training_tag` raises on an empty or
# unlisted `dataset_uri` (issue #424), so the fixture names the paper's 235B-teacher file.
_SEQKD_URI = (
    "science-of-model-organisms/organism-training-data :: "
    "impulsive/seqkd/qwen3-235b-persona/seqkd.jsonl"
)
# Organisms the paper picks by family hash, not by variant.
_FAMILY = {
    ("steering", "7B"): "dd52881e1e399391",
    ("steering", "14B"): "d858b35a7f056663",
    ("steering", "32B"): "c9f1481a726637f9",
}
_EVALS = (
    ("generality_fermi_gen", "generality"),
    ("transitivity", "coherence"),
    ("mmlu", "capability"),
    ("introspection_mcq", "introspection"),
    ("simple_behaviour_exhibit", "expression"),
)


def _rows(drop: Collection[tuple[str, str, str]] = ()) -> list[Row]:
    """Deterministic grid: score depends on method index, size and eval; 6 items x 2 seeds."""
    out = []
    for mi, (key, (method, variant)) in enumerate(_ORGS.items()):
        for si, size in enumerate(SIZES):
            for eval_name, axis in _EVALS:
                if (key, size, axis) in drop:
                    continue
                for seed in (0, 1):
                    for item in range(6):
                        p = 0.2 + 0.08 * mi + 0.03 * si + 0.01 * ((item + seed) % 3)
                        out.append(
                            Row(
                                spec_hash=f"h-{key}-{size}-{seed}",
                                persona_id="impulsive",
                                method=method,
                                train_scale=1.0,
                                base_model=_MODEL[size],
                                behavior_class="persona_trait",
                                eval_name=eval_name,
                                axis=axis,
                                sample_id=f"i{item}",
                                score=float(item % 2 == 0)
                                if axis == "expression"
                                else min(p, 1.0),
                                raw_output="FAKE",
                                fake=True,
                                seed=seed,
                                family=_FAMILY.get((key, size), f"f-{key}-{size}"),
                                meta={},
                                variant=variant,
                                dataset_uri=_SEQKD_URI if method == "distillation_seqkd" else "",
                            )
                        )
    return out


_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "dashboard_build.yaml"


def test_method_key_maps_paper_rows_and_drops_the_rest():
    from analysis.paper_figures import load_families

    fam = load_families(_CONFIG)  # the paper's set (configs/dashboard_build.yaml `families:`)
    assert method_key("prompting", "positive") == "sysprompt"
    assert method_key("prompting", "negative") == "antiprompt"
    assert method_key("few_shot_icl", "k=15 framed", "e67f724873a31cc6", fam) == "icl"  # Luna 7B
    assert method_key("few_shot_icl", "k=15 framed", "37e3ec04c134dcde", fam) is None  # retired
    assert method_key("few_shot_icl", "k=40", "", fam) is None
    assert method_key("oct_demos", "") is None
    # steering = the raw-α organisms by family; the deprecated normalised ones never show
    assert method_key("steering", "all", "dd52881e1e399391", fam) == "steering"  # raw-α 7B
    assert method_key("steering", "all", "c9f1481a726637f9", fam) == "steering"  # raw-α 32B
    assert method_key("steering", "response_only", "38833fb5f5458721", fam) is None  # old 7B
    assert method_key("steering", "all", "b504d2e6d79b552f", fam) is None  # old 7B white-box
    assert method_key("steering", "all", "", fam) is None
    # without a restriction every organism of a known method has its row (anyone's own run)
    assert method_key("steering", "all", "b504d2e6d79b552f") == "steering"
    assert method_key("few_shot_icl", "k=40") == "icl"
    assert method_key("sft_behaviour", "", "x", {"steering": ["y"]}) == "sft"  # unlisted row
    assert {s.key for s in METHODS} | {s.key for s in CONTROLS} | {"base"} == set(STYLE)
    assert "antiprompt" not in {s.key for s in METHODS}  # drawn in no figure (rule 5)


def test_every_registered_steering_and_fewshot_organism_has_a_paper_row():
    # The paper's few-shot and steering are picked by family hash (`families:` in
    # configs/dashboard_build.yaml), so an organism registered without its hash added there is
    # silently left out of every figure (it happened for loyalty and for gemma-4-31B few-shot).
    from analysis.paper_figures import load_families
    from serving.runner import load_specs

    fam = load_families(_CONFIG)
    specs = [
        s
        for s in load_specs(Path(__file__).resolve().parent.parent / "configs" / "specs.yaml")
        if s.method in ("steering", "few_shot_icl")
    ]
    assert specs
    missing = [
        f"{s.persona_id}/{s.method}@{s.base_model} ({s.family_hash})"
        for s in specs
        if method_key(s.method, "", s.family_hash, fam) is None
    ]
    assert not missing, f"add these family hashes to {_CONFIG} `families:`: {missing}"


def test_unmeasured_cell_is_pending_never_zero():
    pd_ = select(_rows(drop={("sft", "14B", "generality")}))
    r = reading(pd_, pd_.organism[("sft", "14B")], "generality")
    assert r.status == "pending" and math.isnan(r.delta)
    ok = reading(pd_, pd_.organism[("sft", "7B")], "generality")
    assert ok.status == "ok" and math.isfinite(ok.delta)
    table, pending = coverage(pd_)
    assert "sft@14B" in pending["generality"]
    assert "icl@7B" in pending["expression"]  # never ran at all
    assert "| SFT | 7B/-/32B |" in table


def test_unjudged_cell_is_pending_not_collapsed():
    """A cell generated with the judge deferred (all NaN) must not knock its eval off the
    axis for everyone — it is in flight, unlike a cell that ran and collapsed."""
    rows = _rows()
    unjudged = [
        Row(**{**r.__dict__, "score": math.nan, "meta": {"defer_judge": "True"}})
        if r.method == "sft_behaviour"
        and r.eval_name == "introspection_mcq"
        and "7B" in r.base_model
        else r
        for r in rows
    ]
    pd_ = select(unjudged)
    assert "introspection" not in pd_.collapsed
    assert reading(pd_, pd_.organism[("sft", "7B")], "introspection").status == "pending"
    assert reading(pd_, pd_.organism[("oct", "7B")], "introspection").status == "ok"
    # a cell that ran and collapsed (scored samples mostly NaN, few left) is dropped for THAT
    # organism only; see test_collapsed_cell_drops_the_eval_for_its_own_organism_only
    collapsed = [
        Row(**{**r.__dict__, "score": math.nan})
        if r.method == "sft_behaviour"
        and r.eval_name == "introspection_mcq"
        and "7B" in r.base_model
        else r
        for r in rows
    ]
    pd2 = select(collapsed)
    assert "introspection_mcq" in pd2.collapsed["introspection"]
    lost = reading(pd2, pd2.organism[("sft", "7B")], "introspection")
    assert lost.status == "pending" and "collapsed" in lost.why  # its only eval collapsed
    assert reading(pd2, pd2.organism[("oct", "7B")], "introspection").status == "ok"


# Two more base-relative composite evals on one axis, the second collapsing for DPO@7B the
# way the 0925 run's debate cell did. Robustness is self-referenced now (share kept), so the
# collapse rule is pinned on generality, where it still compares against base.
_SECOND, _COLLAPSING = "generality_causal_gen", "generality_downstream_gen"


def _with_collapsed_debate(rows: list[Row], *, collapse: bool = True) -> list[Row]:
    """Add two generality evals to the grid, and make DPO@7B's second one collapse (1 of 12
    scored, at a far-off 0.95). `collapse=False` leaves that cell out."""
    extra = []
    first = True
    for r in rows:
        if r.eval_name != "generality_fermi_gen":
            continue
        for name in (_SECOND, _COLLAPSING):
            row = Row(**{**r.__dict__, "eval_name": name})
            if name == _COLLAPSING and r.method == "dpo_behaviour" and "7B" in r.base_model:
                if not collapse:
                    continue
                row = Row(**{**row.__dict__, "score": 0.95 if first else math.nan})
                first = False
            extra.append(row)
    return rows + extra


def test_collapsed_cell_drops_the_eval_for_its_own_organism_only():
    """DPO@7B's collapsed cell leaves DPO@7B's generality only; every other organism keeps
    it, and DPO@7B's reading is paired with its base over the same two evals."""
    rows = _rows()
    pd_ = select(_with_collapsed_debate(rows))
    dpo = pd_.organism[("dpo", "7B")]
    assert pd_.collapsed["generality"] == {_COLLAPSING: [dpo]}

    r = reading(pd_, dpo, "generality")
    assert r.status == "ok" and r.n_evals == 2 and r.omitted == (_COLLAPSING,)
    m, base = pd_.matrix, pd_.matrix.base_for(dpo)
    assert base is not None
    kept_evals = ("generality_fermi_gen", _SECOND)
    assert r.effect == pytest.approx(sum(m.eval_effect(dpo, e) for e in kept_evals) / 2)
    base_means = [c.mean for e in kept_evals if (c := m.get(base, e)) is not None]
    assert r.base_raw == pytest.approx(sum(base_means) / 2)

    for key, size in pd_.organism:
        org = pd_.organism[(key, size)]
        if key == "base" or org == dpo:
            continue
        other = reading(pd_, org, "generality")
        assert other.status == "ok" and other.n_evals == 3 and other.omitted == (), (key, size)
        assert other.base_raw == pytest.approx(base_reading(pd_, size, "generality").raw)

    # every other organism's reading is what it would be had DPO@7B's cell never run
    ref = select(_with_collapsed_debate(rows, collapse=False))
    for (key, size), org in pd_.organism.items():
        if key == "base" or org == dpo:
            continue
        got, want = reading(pd_, org, "generality"), reading(ref, org, "generality")
        assert (got.delta, got.raw, got.n_evals) == pytest.approx(
            (want.delta, want.raw, want.n_evals)
        ), (key, size)

    table, _ = coverage(pd_)
    assert (
        f"Dropped from generality for these organisms only (collapsed cell): {_COLLAPSING}"
        in table
    )
    cap = captions(pd_, {}, "simple_behaviour_exhibit")["radar"]
    # the sentence names the size: a collapse is per organism, not per method
    assert "Generality excludes \\texttt{generality\\_downstream\\_gen} for DPO at 7B" in cap, cap
    assert "every other organism's generality includes it" in cap
    assert "unscoreable" in cap


_OPEN, _CRIT = "robustness_open_ended_gen", "robustness_critique_gen"
_KEPT = f"{_CRIT}:kept"


def _with_share_kept(
    rows: list[Row], *, base: bool = True, base_shows_trait: bool = False
) -> list[Row]:
    """Add the generated robustness bank (open-ended baseline + critique pressure), shaped
    like the 0925 run: the base never shows the trait (zero baseline: its share is
    undefined), OCT shows it everywhere and keeps half, every other organism shows it on
    half the items and keeps a third of that. `base=False` leaves the base's rows out;
    `base_shows_trait=True` gives the base a finite share kept (noise, like the real 4.57)."""
    extra = []
    for r in rows:
        if r.eval_name != "generality_fermi_gen" or (r.method == "base" and not base):
            continue
        item = int(r.sample_id[1:])
        if r.method == "base":
            scores = (1.0, float(item % 2 == 0)) if base_shows_trait else (0.0, 0.0)
        elif r.method == "oct_behaviour":
            scores = (1.0, float(item % 2 == 0))
        else:
            scores = (float(item < 3), float(item == 0))
        for name, score in zip((_OPEN, _CRIT), scores, strict=True):
            extra.append(
                Row(
                    **{
                        **r.__dict__,
                        "eval_name": name,
                        "axis": "robustness",
                        "sample_id": f"i{item}/x",
                        "score": score,
                        "meta": {"item_id": f"i{item}"},
                    }
                )
            )
    return rows + extra


def test_robustness_reads_share_kept_and_never_needs_the_base():
    """Robustness = the mean share kept, read on its own: the base's share is undefined
    (no trait), yet every organism has a reading, and dropping the base's rows changes
    nothing. The raw pressure cell is reported but does not vote."""
    from analysis.paper_figures import RADAR_AXES, fig1_scores

    for with_base in (True, False):
        pd_ = select(_with_share_kept(_rows(), base=with_base))
        m = pd_.matrix
        assert m.evals_in("robustness", composite_only=True) == [_KEPT]
        assert not m.in_composite(_CRIT) and not m.in_composite(_OPEN)
        assert "robustness" not in pd_.collapsed
        for (key, size), org in pd_.organism.items():
            if key == "base":
                continue
            r = reading(pd_, org, "robustness")
            want = 0.5 if key == "oct" else 1 / 3
            assert r.status == "ok" and r.n_evals == 1 and r.raw == pytest.approx(want), key
            assert r.delta == pytest.approx(want)  # vs 0, never vs base
            spoke = fig1_scores(pd_, size, [key])[key][RADAR_AXES.index("robustness")]
            assert spoke == pytest.approx(10 * want)
        for size in SIZES:
            b = base_reading(pd_, size, "robustness")
            assert b.status == "pending"  # zero baseline (or no rows): nothing to keep
    # No base organism at all: robustness still reads (every voter is self-referenced);
    # a base-relative axis stays pending.
    pd_ = select([r for r in _with_share_kept(_rows()) if r.method != "base"])
    org = pd_.organism[("oct", "7B")]
    assert pd_.matrix.base_for(org) is None
    r = reading(pd_, org, "robustness")
    assert r.status == "ok" and r.raw == pytest.approx(0.5) and math.isnan(r.base_raw)
    assert reading(pd_, org, "generality").status == "pending"
    pd_ = select(_with_share_kept(_rows()))
    cell = pd_.matrix.get(pd_.organism[("oct", "7B")], _KEPT)
    assert cell is not None and cell.kept is not None
    assert (cell.kept.baseline_rate, cell.kept.pressure_rate) == (1.0, 0.5)


def test_base_has_no_robustness_reading_on_the_radar_but_organisms_do():
    """Decision 2026-10-20: the base has almost no behaviour to keep, so its share kept is noise.
    Even when the base's own share is finite (0.5 here), its robustness spoke is NaN (pending)
    and the coverage table says n/a; organisms keep theirs; health and behaviour base spokes
    are unchanged (10 and 0)."""
    from analysis.paper_figures import RADAR_AXES, coverage, fig1_scores

    pd_ = select(_with_share_kept(_rows(), base_shows_trait=True))
    assert base_reading(pd_, "7B", "robustness").status == "ok"  # the noise reading exists
    j = RADAR_AXES.index("robustness")
    scores = fig1_scores(pd_, "7B", ["oct"])
    assert math.isnan(scores["base"][j])
    assert math.isfinite(scores["oct"][j])
    assert scores["base"][RADAR_AXES.index("generality")] == 0.0
    assert scores["base"][RADAR_AXES.index("capability")] == pytest.approx(10.0)
    table, pending = coverage(pd_)
    assert next(r for r in table.splitlines() if r.startswith("| Base")).count("n/a") == 1
    assert not any(k.startswith("base@") for v in pending.get("robustness", []) for k in [v])


def test_robustness_reading_carries_a_composite_interval_and_a_seed_spread():
    """Robustness's reading has a joint-bootstrap 95% CI around it; a base-relative axis has
    none (NaN). The fixture's two seeds are identical, so the seed SD is exactly 0, and the
    coverage output prints both."""
    from analysis.paper_figures import coverage, seed_spread

    pd_ = select(_with_share_kept(_rows()))
    org = pd_.organism[("oct", "7B")]
    r = reading(pd_, org, "robustness")
    assert r.status == "ok" and r.evals == (_KEPT,)
    lo, hi = r.ci
    assert lo <= r.raw <= hi and lo < hi
    again = select(_with_share_kept(_rows())).matrix.kept_composite_ci(org, r.evals)
    assert again == (r.se, *r.ci) and r.se > 0
    assert all(math.isnan(x) for x in reading(pd_, org, "generality").ci)
    sp = seed_spread(pd_, org)
    assert sp is not None and sorted(sp.per_seed) == [0, 1] and sp.sd == 0.0
    assert sp.fixed == () and sp.dropped == ()
    assert seed_spread(pd_, org, "generality") is None
    table, _ = coverage(pd_)
    assert f"Robustness OCT@7B: {r.raw:.3f} [{lo:.3f}, {hi:.3f}] (seed SD 0.000" in table


def test_a_flagged_organism_has_its_whole_outline_dashed():
    """A mark, not a filter: the polygon is drawn, one outline, dashed all round when flagged
    and solid otherwise. No vertex marker, hollow or filled, either way (decided 2026-10-01)."""
    import matplotlib.pyplot as plt

    from analysis.paper_figures import FLAG_DASH_PT, radar_panel

    values = [5.0, 4.0, 9.0, 9.0, 3.0, 6.0]
    for flag in (True, False, None):
        fig = plt.figure()
        ax = fig.add_subplot(projection="polar")
        radar_panel(ax, values, [0.0] * 6, "#123456", 10, flagged=flag)
        lines = ax.get_lines()
        assert not [ln for ln in lines if ln.get_marker() not in ("None", "", None)]
        assert len(lines) == 1
        dashes = tuple(lines[0]._unscaled_dash_pattern[1] or ())
        lw = lines[0].get_linewidth()
        printed = tuple(round(d * lw, 6) for d in dashes)  # matplotlib scales by lw
        assert (printed == FLAG_DASH_PT) is bool(flag)
        plt.close(fig)


def test_the_overlay_dashes_a_flagged_method_and_never_the_base():
    """radar_overlay_axes: a flagged method's outline is dashed all round; the base keeps
    its own style whatever `flags` says, and nothing has a white-faced (hollow) marker."""
    import matplotlib.pyplot as plt

    from analysis.paper_figures import FLAG_DASH_PT, radar_overlay_axes

    scores = {"base": [0.5, 3.0, 10.0, 10.0, 1.0, 0.5], "oct": [5.0, 4.0, 9.0, 9.0, 3.0, 6.0]}
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    handles = radar_overlay_axes(ax, scores, ["oct"], flags={"oct": True, "base": True})
    lines = ax.get_lines()
    assert not [ln for ln in lines if ln.get_markerfacecolor() == "white"]
    assert not [h for h in handles if getattr(h, "get_markerfacecolor", lambda: None)() == "white"]
    # Line2D has no public dash getter; unscaled pattern x lw is the printed dash, FLAG_DASH_PT.
    flagged = [
        ln
        for ln in lines
        if tuple(round(d * ln.get_linewidth(), 6) for d in (ln._unscaled_dash_pattern[1] or ()))
        == FLAG_DASH_PT
    ]
    assert len(flagged) == 1  # oct's one outline; the base is not flagged
    assert {round(float(y), 6) for y in list(flagged[0].get_ydata())} >= {5.0, 4.0, 3.0, 6.0}
    plt.close(fig)


def test_sbe_flag_follows_the_verdict_and_a_missing_one_flags():
    from analysis.paper_figures import _flagged_phrase, flagged

    pd_ = select(_with_share_kept(_rows()))
    oct_, sft, base = (pd_.organism[(k, "7B")] for k in ("oct", "sft", "base"))
    pd_.sbe_verdict.update({oct_: "PASS", sft: "PASS_INDIST"})
    pd_.sbe_verdict.pop(base, None)
    assert not flagged(pd_, oct_) and not flagged(pd_, sft)
    assert flagged(pd_, base)  # no verdict: flagged
    pd_.sbe_verdict[sft] = "FAIL"
    assert flagged(pd_, sft)
    table, _ = coverage(pd_)
    assert "Flagged (dashed outline" in table and "SFT@7B (FAIL)" in table
    assert "Base model@7B (no verdict)" in table
    assert "SFT (7B, FAIL" in _flagged_phrase(pd_)
    radar_caption = captions(pd_, {}, "simple_behaviour_exhibit")["radar"]
    assert (
        "A dashed outline" in radar_caption and "hollow" not in radar_caption.lower()
    )  # the below-base circle is gone: below base is simply drawn at the centre


def test_strength_against_share_kept_scatter():
    from analysis.paper_figures import kept_scatter, kept_scatter_points

    pd_ = select(_with_share_kept(_rows()))
    pd_.sbe_verdict[pd_.organism[("oct", "7B")]] = "PASS"
    pts = {(p["key"], p["size"]): p for p in kept_scatter_points(pd_)}
    oct_ = pts[("oct", "7B")]
    assert oct_["x"] == 1.0 and oct_["y"] == pytest.approx(0.5) and not oct_["flagged"]
    sft = pts[("sft", "7B")]
    assert sft["x"] == 0.5 and sft["y"] == pytest.approx(1 / 3) and sft["flagged"]
    assert ("icl", "7B") not in pts  # never ran: no point, never a 0
    assert {k for k, _s in pts} <= {s.key for s in METHODS}  # no base, no controls
    fig = kept_scatter(pd_)
    hollow = [
        ln for ax in fig.axes for ln in ax.get_lines() if ln.get_markerfacecolor() == "white"
    ]
    assert len(hollow) == sum(p["flagged"] for p in pts.values())


def test_expression_reference_line_is_drawn_above_every_bar_and_error_bar():
    from matplotlib.patches import Rectangle

    from analysis.paper_figures import EXPRESSION_DELTA_GUIDE, expression_rate_bar

    fig = expression_rate_bar(select(_rows()), reference=EXPRESSION_DELTA_GUIDE)
    (ax,) = fig.axes
    bars = [p for p in ax.patches if isinstance(p, Rectangle) and p.get_height() > 0]
    assert bars
    line = next(
        ln for ln in ax.get_lines() if list(ln.get_ydata()) == [EXPRESSION_DELTA_GUIDE] * 2
    )
    # Error bars are Line2D / LineCollection artists; none may out-rank the reference line.
    others = [
        a.get_zorder() for a in (*ax.patches, *ax.collections, *ax.get_lines()) if a is not line
    ]
    assert line.get_zorder() > max(others)
    plain = expression_rate_bar(select(_rows()))
    assert (
        len(plain.axes[0].get_lines()) == len(ax.get_lines()) - 1
    )  # opt-in: paper figure unchanged


def _bars_fixture(sizes: tuple[str, ...]):
    import types

    from analysis.paper_figures import method_size_bars

    vals = {"base": 0.01, "dpo": 0.0, "oct": 0.9}
    pd_ = types.SimpleNamespace(organism={(k, s): "x" for k in vals for s in sizes})

    def value(key: str, size: str):
        if (key, size) == ("oct", sizes[-1]):
            return None  # missing: "n/a", never a 0
        return vals[key], max(vals[key] - 0.01, 0.0), vals[key] + 0.01

    return method_size_bars, pd_, value


def _texts(fig) -> list[str]:
    return [t.get_text() for t in fig.axes[0].texts]


def test_size_bars_with_gemma_has_two_legends_and_labels_zero_and_missing():
    sizes = (*SIZES, "31B")
    bars, pd_, value = _bars_fixture(sizes)
    fig = bars(
        pd_,
        value,
        ylabel="x",
        sizes=sizes,
        legends=[("Qwen2.5-Instruct", SIZES), ("Gemma-4", ("31B",))],
        zero_label="0",
    )
    titles = [lg.get_title().get_text() for lg in fig.legends]
    assert titles == ["Qwen2.5-Instruct", "Gemma-4"]
    assert [[t.get_text() for t in lg.get_texts()] for lg in fig.legends] == [list(SIZES), ["31B"]]
    assert fig.get_size_inches().tolist() == [5.5, 2.5]
    texts = _texts(fig)
    assert texts.count("0") == len(sizes)  # DPO's four exactly-zero bars, nothing else
    assert texts.count("n/a") == 1  # OCT at 31B
    fig_ref = bars(pd_, value, ylabel="x", sizes=sizes, reference=0.10)
    assert any(list(ln.get_ydata()) == [0.10] * 2 for ln in fig_ref.axes[0].get_lines())


def test_size_bars_qwen_only_is_unchanged_by_the_opt_ins():
    bars, pd_, value = _bars_fixture(SIZES)
    fig = bars(pd_, value, ylabel="x", sizes=SIZES)
    assert [lg.get_title().get_text() for lg in fig.legends] == ["Qwen2.5-Instruct"]
    assert "0" not in _texts(fig)  # zero_label is opt-in: a 0 bar is left bare
    # expression_rate_bar on Qwen-only data takes neither opt-in either
    from analysis.paper_figures import expression_rate_bar

    efig = expression_rate_bar(select(_rows()))
    assert [lg.get_title().get_text() for lg in efig.legends] == ["Qwen2.5-Instruct"]
    assert "0" not in _texts(efig)


def _expression_fixture(monkeypatch: pytest.MonkeyPatch, sizes: tuple[str, ...]):
    import types

    import analysis.paper_figures as pf

    monkeypatch.setattr(
        pf,
        "sbe_cell",
        lambda _matrix, spec: types.SimpleNamespace(
            mean=0.0 if spec == "dpo" else 0.5, ci95=(0.0, 0.0) if spec == "dpo" else (0.4, 0.6)
        ),
    )
    keys = ("base", "dpo", "oct")
    return pf, types.SimpleNamespace(matrix={}, organism={(k, s): k for k in keys for s in sizes})


def test_expression_bar_with_31b_opts_in_and_its_legends_do_not_collide(
    monkeypatch: pytest.MonkeyPatch,
):
    sizes = (*SIZES, "31B")
    pf, pd_ = _expression_fixture(monkeypatch, sizes)
    fig = pf.expression_rate_bar(pd_, sizes, reference=0.10)
    assert [lg.get_title().get_text() for lg in fig.legends] == ["Qwen2.5-Instruct", "Gemma-4"]
    assert _texts(fig).count("0") == len(sizes)
    fig.canvas.draw()
    first, second = (lg.get_window_extent() for lg in fig.legends)
    assert not first.overlaps(second)
    assert first.x0 >= 0 and second.x1 <= fig.bbox.x1


def test_expression_bar_with_only_31b_has_one_legend(monkeypatch: pytest.MonkeyPatch):
    pf, pd_ = _expression_fixture(monkeypatch, ("31B",))
    fig = pf.expression_rate_bar(pd_, ("31B",))
    assert [lg.get_title().get_text() for lg in fig.legends] == ["Gemma-4"]


def test_expression_cli_options_reach_the_figure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import analysis.paper_figures as pf

    seen: list[tuple[tuple[str, ...], float | None]] = []
    real = pf.expression_rate_bar

    def spy(pd_, sizes, *, stamp=None, reference=None):
        seen.append((tuple(sizes), reference))
        return real(pd_, SIZES, stamp=stamp, reference=reference)  # _rows() has no 31B data

    monkeypatch.setattr(pf, "expression_rate_bar", spy)
    monkeypatch.setattr(pf, "load_rows", lambda _root, **_kw: _rows())
    pf.main(
        [
            *("--root", str(tmp_path), "--out", str(tmp_path / "figs")),
            *("--only", "expression"),
            *("--expression-sizes", "7B,14B,32B,31B", "--expression-reference", "0.1"),
        ]
    )
    assert (("7B", "14B", "32B", "31B"), 0.1) in seen  # the per-size bars add their own calls


@pytest.mark.parametrize("bad", ["9B", "", "32B,32B"])
def test_expression_sizes_are_refused_before_any_load(tmp_path: Path, bad: str):
    from analysis.paper_figures import main

    with pytest.raises(SystemExit) as refused:
        main(["--root", str(tmp_path), "--out", str(tmp_path), "--expression-sizes", bad])
    assert refused.value.code == 2


def test_render_all_writes_only_under_tmp(tmp_path: Path):
    out = tmp_path / "figs"
    table, written = render_all(_rows(), out, draft=True)
    assert "| method |" in table
    for path in written.values():
        assert path.is_relative_to(tmp_path)
        assert path.read_bytes()[:4] == b"%PDF"
    assert (out / "captions.tex").read_text().count("\\renewcommand") == 7
    assert (out / "robustness_strength_scatter.pdf").exists()
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [p.name for p in written.values()]
        + [
            "antiprompt_table.tex",
            "captions.tex",
            "coverage.md",
            "generality_coherence_correlations.tex",
        ]
    )


def test_two_organisms_on_one_paper_row_refuse(tmp_path: Path):
    rows = _rows()
    clash = [r for r in rows if r.method == "sft_behaviour" and r.base_model == _MODEL["7B"]][:1]
    clone = Row(**{**clash[0].__dict__, "variant": "other-recipe"})
    with pytest.raises(ValueError, match="two organisms map"):
        select([*rows, clone])


def test_cell_on_other_items_than_base_is_dropped():
    """A cell rerun on a new bank version is not compared with a base on the old one."""
    rows = [
        Row(**{**r.__dict__, "sample_id": "v2-" + r.sample_id})
        if r.method == "sft_behaviour" and "7B" in r.base_model and r.axis == "generality"
        else r
        for r in _rows()
    ]
    pd_ = select(rows)
    org = pd_.organism[("sft", "7B")]
    assert (org, "generality_fermi_gen") in pd_.other_bank
    assert pd_.matrix.get(org, "generality_fermi_gen") is None
    assert reading(pd_, org, "generality").status == "pending"
    assert reading(pd_, pd_.organism[("oct", "7B")], "generality").status == "ok"


def test_partially_landed_cell_is_pending():
    """Half a bank (fewer samples than base on the same bank) is still landing: pending."""
    rows = [
        r
        for r in _rows()
        if not (
            r.method == "sft_behaviour"
            and "7B" in r.base_model
            and r.axis == "generality"
            and r.sample_id in ("i0", "i1", "i2")
        )
    ]
    pd_ = select(rows)
    r = reading(pd_, pd_.organism[("sft", "7B")], "generality")
    assert r.status == "pending" and r.why.startswith("partial")
    assert reading(pd_, pd_.organism[("oct", "7B")], "generality").status == "ok"


def _is_cell(r: Row, size: str, method: str, axis: str) -> bool:
    return r.method == method and size in r.base_model and r.axis == axis


def test_a_row_count_below_base_with_full_item_coverage_is_a_reading():
    """The rule is COVERAGE, not a row count. A cell with every item the base has, on as
    many seeds, is complete even when it holds fewer ROWS than the base (the base carries a
    repeat of two items; equally, a judge pass that skipped different rows per organism). The
    old `n_total(organism) < n_total(base)` rule read this as "partial" and took Steering@14B
    and the negative prompt@32B off the coherence spoke with complete cells."""
    import dataclasses

    rows = _rows()
    repeats = [
        dataclasses.replace(r, sample_id=f"{r.sample_id}/again")
        for r in rows
        if _is_cell(r, "7B", "base", "generality") and r.sample_id in ("i0", "i1")
    ]
    assert repeats
    pd_ = select(rows + repeats)
    sft, base = pd_.organism[("sft", "7B")], pd_.organism[("base", "7B")]
    got, ref = (
        pd_.matrix.get(sft, "generality_fermi_gen"),
        pd_.matrix.get(base, "generality_fermi_gen"),
    )
    assert got.n_total < ref.n_total  # the old rule's trigger
    assert got.items == ref.items and not ref.coverage_missing_from(got)
    assert reading(pd_, sft, "generality").status == "ok"


def test_a_cell_missing_one_of_the_bases_items_is_pending_even_at_equal_row_count():
    """The other direction the count got wrong: same number of rows, one item short."""
    import dataclasses

    rows = [
        r
        for r in _rows()
        if not (_is_cell(r, "7B", "sft_behaviour", "generality") and r.sample_id == "i5")
    ]
    padding = [
        dataclasses.replace(r, sample_id=f"{r.sample_id}/again")
        for r in rows
        if _is_cell(r, "7B", "sft_behaviour", "generality") and r.sample_id == "i0"
    ]
    pd_ = select(rows + padding)
    sft, base = pd_.organism[("sft", "7B")], pd_.organism[("base", "7B")]
    got, ref = (
        pd_.matrix.get(sft, "generality_fermi_gen"),
        pd_.matrix.get(base, "generality_fermi_gen"),
    )
    assert got.n_total == ref.n_total
    assert ref.coverage_missing_from(got) == {(0, "i5"), (1, "i5")}
    r = reading(pd_, sft, "generality")
    assert r.status == "pending" and r.why == "partial generality_fermi_gen"


def test_a_cell_on_fewer_seeds_than_base_is_pending():
    """The seed half of the old count rule, kept explicitly: every item, one seed of two."""
    rows = [
        r
        for r in _rows()
        if not (_is_cell(r, "7B", "sft_behaviour", "generality") and r.seed == 1)
    ]
    pd_ = select(rows)
    sft, base = pd_.organism[("sft", "7B")], pd_.organism[("base", "7B")]
    got, ref = (
        pd_.matrix.get(sft, "generality_fermi_gen"),
        pd_.matrix.get(base, "generality_fermi_gen"),
    )
    assert got.items == ref.items and (got.n_seeds, ref.n_seeds) == (1, 2)
    r = reading(pd_, sft, "generality")
    assert r.status == "pending" and r.why.startswith("partial")


def test_an_item_missing_in_one_seed_only_is_pending():
    """Adversarial review, finding 5. The organism has every item SOMEWHERE and as many
    seeds as the base, but one item is missing in seed 0 only. A union of item ids plus a
    seed count cannot see that; coverage is compared per (seed, item)."""
    rows = [
        r
        for r in _rows()
        if not (
            _is_cell(r, "7B", "sft_behaviour", "generality")
            and r.seed == 0
            and r.sample_id == "i5"
        )
    ]
    pd_ = select(rows)
    sft, base = pd_.organism[("sft", "7B")], pd_.organism[("base", "7B")]
    got, ref = (
        pd_.matrix.get(sft, "generality_fermi_gen"),
        pd_.matrix.get(base, "generality_fermi_gen"),
    )
    assert got.items == ref.items and got.n_seeds == ref.n_seeds  # what the old check saw
    assert ref.coverage_missing_from(got) == {(0, "i5")}
    r = reading(pd_, sft, "generality")
    assert r.status == "pending" and r.why == "partial generality_fermi_gen"


def test_style_rules_names_and_controls():
    """The paper's naming and control rules (analysis/paper_style.py)."""
    labels = {s.label for s in METHODS}
    assert labels == {"OCT", "SFT", "DPO", "Steering", "Distillation", "Prompting", "Few-shot"}
    assert STYLE["base"].label == "Base model"
    assert not any("(ours)" in lab or "seq-KD" in lab or "anti" in lab.lower() for lab in labels)


def test_radar_scale_base_outer_ring_on_health_axes():
    """Base = 10 on coherence and capability; a spoke with no data (activations in this synthetic
    grid) is NaN, never 0."""
    from analysis.paper_figures import RADAR_AXES, fig1_scores

    pd_ = select(_rows())
    scores = fig1_scores(pd_, "7B", ["oct"])
    base = dict(zip(RADAR_AXES, scores["base"], strict=True))
    assert base["coherence"] == 10.0 and base["capability"] == 10.0
    assert math.isnan(base["activations"])
    assert math.isnan(dict(zip(RADAR_AXES, scores["oct"], strict=True))["activations"])


def test_radar_value_is_the_one_metric():
    """Behaviour spokes are 10 x gain over base; health 10 x (1 + org - base) (clipped to [0, 10]);
    robustness 10 x its own share kept."""
    from analysis.paper_style import radar_value

    assert radar_value("generality", 0.5, 0.01) == pytest.approx(4.9)  # gain over base
    assert radar_value("generality", 0.01, 0.01) == 0.0  # the Base model is the centre (0)
    assert radar_value("robustness", 0.5, 0.01) == 5.0  # share kept: base unused
    assert radar_value("coherence", 0.8, 0.8) == 10.0
    assert radar_value("capability", 0.4, 0.8) == pytest.approx(6.0)  # 10 x (1 - 0.4)
    assert radar_value("coherence", 0.8, 0.0) == pytest.approx(10.0)  # clipped; zero base is fine
    assert radar_value("coherence", 0.3, 0.0) == pytest.approx(10.0)
    assert radar_value("capability", 0.0, 0.5) == pytest.approx(5.0)  # linear in the delta
    assert radar_value("capability", 0.9, 0.8) == 10.0  # clipped
    assert radar_value("capability", 0.0, 1.5) == 0.0  # clipped below
    assert math.isnan(radar_value("capability", 0.5, math.nan))
    assert math.isnan(radar_value("generality", math.nan, 0.1))


def test_normalise_tree_leaves_out_superseded(tmp_path: Path):
    from analysis.paper_figures import normalise_tree

    src = tmp_path / "run"
    (src / "org" / "h").mkdir(parents=True)
    (src / "org" / "h" / "spec.json").write_text("{}")
    (src / "superseded" / "org" / "h").mkdir(parents=True)
    (src / "superseded" / "org" / "h" / "spec.json").write_text("{}")
    dst = normalise_tree(src, tmp_path / "norm")
    assert (dst / "org" / "h" / "spec.json").exists()
    assert not (dst / "superseded").exists()


def test_normalise_tree_refuses_to_wipe_foreign_or_overlapping_dirs(tmp_path: Path):
    """dst is rmtree'd: a mistyped --normalised-root must never delete a results tree."""
    from analysis.paper_figures import normalise_tree

    src = tmp_path / "run"
    (src / "org").mkdir(parents=True)
    (src / "org" / "spec.json").write_text("{}")
    foreign = tmp_path / "results"
    foreign.mkdir()
    (foreign / "keep.parquet").write_text("x")
    for bad in (src, src / "org", tmp_path, foreign):
        with pytest.raises(SystemExit):
            normalise_tree(src, bad)
    assert (foreign / "keep.parquet").exists() and (src / "org" / "spec.json").exists()
    dst = normalise_tree(src, tmp_path / "norm")
    assert normalise_tree(src, dst) == dst  # its own earlier mirror may be rebuilt


def test_health_ratio_pairs_base_over_same_evals_when_an_eval_is_not_run_by_design(
    monkeypatch: pytest.MonkeyPatch,
):
    """A method with an eval left out on purpose is divided by base over the SAME evals.

    Codex review of #363: with base mmlu .8 + apps 0 and few-shot mmlu .6 (apps not run),
    dividing by the full base composite read 10 instead of 7.5.
    """
    import analysis.paper_figures as pf
    from analysis.paper_style import radar_value

    monkeypatch.setitem(globals(), "_EVALS", (*_EVALS, ("apps", "capability")))
    pd_ = select(_rows())
    size, axis, key = SIZES[0], "capability", "sft"
    assert pf.base_reading(pd_, size, axis).n_evals == 2
    monkeypatch.setattr(pf, "NOT_RUN_BY_DESIGN", frozenset({(key, "apps")}))
    r = pf.reading(pd_, pd_.organism[(key, size)], axis)
    base = pd_.organism[("base", size)]
    assert r.n_evals == 1
    assert r.base_raw == pf._cell_mean(pd_.matrix, base, "mmlu")
    spoke = pf.RADAR_AXES.index(axis)
    assert pf.fig1_scores(pd_, size, [key])[key][spoke] == radar_value(axis, r.raw, r.base_raw)


def test_zero_spread_eval_reads_as_on_base_not_pending():
    """Every organism scoring the same on an eval makes its spread (and eval_effect) NaN;
    the reading must still be ok (raw 0.5, a gain of 0 over base), not 'missing'."""
    import dataclasses

    import analysis.paper_figures as pf

    rows = [dataclasses.replace(r, score=0.5) if r.axis == "generality" else r for r in _rows()]
    pd_ = select(rows)
    org = pd_.organism[("sft", SIZES[0])]
    r = pf.reading(pd_, org, "generality")
    assert r.status == "ok" and r.effect == 0.0 and r.raw == 0.5
    assert pf.fig1_scores(pd_, SIZES[0], ["sft"])["sft"][pf.RADAR_AXES.index("generality")] == 0.0


def test_render_all_forked_workers_match_serial(tmp_path: Path):
    """--workers N forks after one load; it must write the same files as the serial path."""
    _t1, serial = render_all(_rows(), tmp_path / "a", draft=False)
    _t2, forked = render_all(_rows(), tmp_path / "b", draft=False, workers=2)
    assert sorted(p.name for p in serial.values()) == sorted(p.name for p in forked.values())
    assert all(p.read_bytes()[:4] == b"%PDF" for p in forked.values())


def test_debate_turn_rows_skips_unscoreable_and_superseded_keeps_dpo(tmp_path: Path):
    """Unscoreable (None/NaN) turns and `superseded/` cells are left out; DPO is drawn like
    every other method since its v4 organisms' debates are scoreable (2026-10-02)."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from analysis.paper_figures import DEBATE_EVAL, debate_turn_rows

    def cell(folder: Path, method: str, turns: list, variant_cfg: dict | None = None) -> None:
        folder.mkdir(parents=True)
        spec = {
            "method": method,
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "method_config": variant_cfg or {},
        }
        (folder / "spec.json").write_text(json.dumps(spec))
        meta = json.dumps({"turn_scores": json.dumps(turns)})
        pq.write_table(
            pa.table({"sample_id": ["i0"], "meta": [meta]}),
            folder / f"seed0__h__{DEBATE_EVAL}.parquet",
        )

    root = tmp_path / "run"
    cell(root / "oct" / "h", "oct_behaviour", [0.9, None, 0.5, float("nan")])
    cell(root / "dpo" / "h", "dpo_behaviour", [0.1, 0.1, 0.1, 0.1])
    cell(root / "superseded" / "oct" / "h", "oct_behaviour", [1.0, 1.0, 1.0, 1.0])
    rows = debate_turn_rows(root)
    assert sorted((r["method"], r["turn"], r["score"]) for r in rows) == [
        ("DPO", "A1", 0.1),
        ("DPO", "A2", 0.1),
        ("DPO", "A3", 0.1),
        ("DPO", "A4", 0.1),
        ("OCT", "A1", 0.9),
        ("OCT", "A3", 0.5),
    ]


def test_debate_turn_rows_keep_a_27b_thinking_arm_apart_from_its_twin(tmp_path: Path):
    """Adversarial review of #216 item 6: sizes were keyed on the base model alone, so a
    27B's thinking-on and -off debate turns pooled into one `27B` curve. Keyed like
    `Row.model_tag` now; a thinking-off organism's size is exactly what it was."""
    import json

    import pyarrow as pa
    import pyarrow.parquet as pq

    from analysis.paper_figures import DEBATE_EVAL, debate_turn_rows

    def cell(folder: Path, base: str, method_config: dict) -> None:
        folder.mkdir(parents=True)
        spec = {"method": "base", "base_model": base, "method_config": method_config}
        (folder / "spec.json").write_text(json.dumps(spec))
        pq.write_table(
            pa.table({"sample_id": ["i0"], "meta": [json.dumps({"turn_scores": [0.5]})]}),
            folder / f"seed0__h__{DEBATE_EVAL}.parquet",
        )

    root = tmp_path / "run"
    cell(root / "a" / "h", "Qwen/Qwen3.8-27B", {"enable_thinking": False})
    cell(root / "b" / "h", "Qwen/Qwen3.8-27B", {"enable_thinking": True})
    cell(root / "c" / "h", "Qwen/Qwen2.5-7B-Instruct", {})
    assert sorted(r["size"] for r in debate_turn_rows(root)) == ["27B", "27B-think", "7B"]


# ---- activations spoke: base-relative truth probe (world + self), base = 0
_TP = {"base": (0.51, 0.415), "oct": (0.81, 0.97), "sft": (0.40, 0.30)}


def _rows_with_truth(drop_cell: tuple[str, str] | None = None) -> list[Row]:
    """`_rows()` plus whitebox_truth_{world,self} cells at 7B: share per (organism, cell)."""
    import dataclasses

    rows = _rows()
    extra = []
    for key, (w, s) in _TP.items():
        for ev, v in (("whitebox_truth_world", w), ("whitebox_truth_self", s)):
            if drop_cell == (key, ev):
                continue
            extra += [
                dataclasses.replace(r, eval_name=ev, axis="activations", score=v)
                for r in rows
                if r.eval_name == "mmlu" and r.spec_hash.startswith(f"h-{key}-7B-")
            ]
    return rows + extra


def _act(pd_, key: str) -> float:
    from analysis.paper_figures import RADAR_AXES, fig1_scores

    return fig1_scores(pd_, "7B", ["oct", "sft"])[key][RADAR_AXES.index("activations")]


def test_activations_spoke_base_reads_zero_and_known_cell():
    pd_ = select(_rows_with_truth())
    assert _act(pd_, "base") == 0.0
    # 10 x ((0.81 - 0.51) + (0.97 - 0.415)) / 2 = 4.275
    assert _act(pd_, "oct") == pytest.approx(4.275)


def test_activations_spoke_is_signed_below_base():
    # sft: 10 x ((0.40-0.51) + (0.30-0.415)) / 2 = -1.125, below base (drawn at the centre)
    assert _act(select(_rows_with_truth()), "sft") == pytest.approx(-1.125)


def test_activations_spoke_needs_both_cells():
    pd_ = select(_rows_with_truth(drop_cell=("oct", "whitebox_truth_self")))
    assert math.isnan(_act(pd_, "oct"))
    assert _act(pd_, "sft") == pytest.approx(-1.125)  # the other organism is untouched


@pytest.mark.parametrize("missing", ["whitebox_truth_world", "whitebox_truth_self"])
def test_activations_spoke_pending_when_a_voter_is_absent_from_the_whole_tree(missing):
    # A cell absent for EVERY organism (base included) must not shrink the requirement to one
    # cell: no organism draws a one-cell spoke, and the base is pending too (not 0).
    rows = [r for r in _rows_with_truth() if r.eval_name != missing]
    pd_ = select(rows)
    for key in ("base", "oct", "sft"):
        assert math.isnan(_act(pd_, key))


def test_other_spokes_unchanged_by_the_truth_cells():
    from analysis.paper_figures import RADAR_AXES, fig1_scores

    before = fig1_scores(select(_rows()), "7B", ["oct", "sft"])
    after = fig1_scores(select(_rows_with_truth()), "7B", ["oct", "sft"])
    i = RADAR_AXES.index("activations")
    for key in before:
        for j, (b, a) in enumerate(zip(before[key], after[key], strict=True)):
            if j != i:
                assert (math.isnan(b) and math.isnan(a)) or b == a


def test_radar_value_truth_probe_is_base_relative():
    from analysis.paper_style import radar_value

    assert radar_value("activations", 0.6, 0.5) == pytest.approx(1.0)
    assert radar_value("activations", 0.5, 0.5) == 0.0
    assert radar_value("activations", 0.4, 0.5) == pytest.approx(-1.0)  # signed, not clipped
    assert radar_value("activations", 1.0, 0.0) == 10.0
    assert math.isnan(radar_value("activations", 0.6, math.nan))


def test_behaviour_spokes_are_signed_gain_over_base_and_the_base_is_zero():
    """Generality and introspection read 10 x (organism - base): the base is exactly 0 and an
    organism below base is negative (not clipped); health spokes are 10 x (1 + org - base)."""
    import dataclasses

    from analysis.paper_figures import RADAR_AXES, fig1_scores

    rows = [
        dataclasses.replace(r, score=0.05)
        if (r.axis == "generality" and "sft" in r.spec_hash)
        else r
        for r in _rows()
    ]
    pd_ = select(rows)
    sc = fig1_scores(pd_, "7B", ["sft"])
    i = RADAR_AXES.index("generality")
    assert sc["base"][i] == 0.0
    base_raw = pd_.matrix.axis_score(pd_.organism[("base", "7B")], "generality")
    assert sc["sft"][i] == pytest.approx(10 * (0.05 - base_raw))
    assert sc["sft"][i] < 0
    assert sc["base"][RADAR_AXES.index("coherence")] == 10.0


def _intro_rows(base: list[float], sft: list[float]) -> list[Row]:
    """`_rows()` with the 7B base's and SFT's `introspection_mcq` set per item (NaN = the item
    was unscorable for that organism), the same in both seeds."""
    import dataclasses

    per = {"base": base, "sft": sft}
    out = []
    for r in _rows():
        key = r.spec_hash.split("-")[1]
        if (
            r.eval_name == "introspection_mcq"
            and r.spec_hash.endswith(f"-7B-{r.seed}")
            and key in per
        ):
            r = dataclasses.replace(r, score=per[key][int(r.sample_id[1:])])
        out.append(r)
    return out


NAN = math.nan


def test_introspection_gain_uses_only_items_scorable_for_both():
    """Base is unscorable on item 0, SFT on item 5. Averaged independently the two means sit on
    different items; matched they are both read on items 1-4: base 0.1, SFT 0.5, gain 4.0."""
    import analysis.paper_figures as pf

    base = [NAN, 0.1, 0.1, 0.1, 0.1, 0.9]  # independent mean 0.26, matched 0.1
    sft = [0.0, 0.5, 0.5, 0.5, 0.5, NAN]  # independent mean 0.4, matched 0.5
    pd_ = select(_intro_rows(base, sft))
    o, b = pd_.organism[("sft", "7B")], pd_.organism[("base", "7B")]
    r = pf.reading(pd_, o, "introspection")
    mm = pd_.matrix.matched_means(o, b, "introspection_mcq")
    assert mm is not None and mm[2] == 4  # items 1-4 are scorable for both
    assert r.gain_raw == pytest.approx(0.5) and r.gain_base_raw == pytest.approx(0.1)
    assert pf.radar_value("introspection", r.gain_raw, r.gain_base_raw) == pytest.approx(4.0)
    # the absolute (table) rates are the independent cell means, untouched by the matching
    assert r.raw == pytest.approx(0.4) and r.base_raw == pytest.approx(0.26)
    i = pf.RADAR_AXES.index("introspection")
    assert pf.fig1_scores(pd_, "7B", ["sft"])["sft"][i] == pytest.approx(4.0)
    assert pf.fig1_scores(pd_, "7B", ["sft"])["base"][i] == 0.0


def test_matched_composite_equals_the_cell_means_when_item_sets_agree():
    import analysis.paper_figures as pf

    pd_ = select(_intro_rows([0.1] * 6, [0.5] * 6))
    r = pf.reading(pd_, pd_.organism[("sft", "7B")], "introspection")
    assert r.gain_raw == pytest.approx(r.raw) and r.gain_base_raw == pytest.approx(r.base_raw)


def test_a_negative_introspection_gain_is_signed():
    """The 32B-steering shape: organism raw below the base's reads below the ring, unclipped."""
    import analysis.paper_figures as pf

    pd_ = select(_intro_rows([0.1] * 6, [-0.1] * 6))
    i = pf.RADAR_AXES.index("introspection")
    assert pf.fig1_scores(pd_, "7B", ["sft"])["sft"][i] == pytest.approx(-2.0)


def test_no_shared_item_makes_the_gain_pending_not_zero():
    import analysis.paper_figures as pf

    pd_ = select(_intro_rows([NAN, NAN, NAN, 0.1, 0.1, 0.1], [0.5, 0.5, 0.5, NAN, NAN, NAN]))
    r = pf.reading(pd_, pd_.organism[("sft", "7B")], "introspection")
    assert math.isnan(r.gain_raw)
    i = pf.RADAR_AXES.index("introspection")
    assert math.isnan(pf.fig1_scores(pd_, "7B", ["sft"])["sft"][i])


def _below_base_markers(ax) -> list:
    """Marker artists at r = 0 (the removed below-base circle): there must be none."""
    import numpy as np

    return [
        ln
        for ln in ax.get_lines()
        if ln.get_marker() not in (None, "None", "", " ")
        and set(np.asarray(ln.get_ydata())) == {0.0}
    ]


def test_below_base_behaviour_spoke_is_drawn_at_the_centre_with_no_marker():
    """-1.3 on generality (a behaviour spoke) is drawn at radius 0 and carries NO marker (decided
    2026-10-21); the radar has no base ring; the signed input is untouched."""
    import matplotlib.pyplot as plt

    from analysis.paper_figures import RADAR_AXES, radar_overlay_axes, radar_panel
    from analysis.paper_style import RADAR_MIN

    assert RADAR_MIN == 0.0
    values = [-1.3, 4.0, 9.0, 9.0, 3.0, 6.0]
    assert RADAR_AXES[0] == "generality"

    def no_ring(ax) -> bool:  # a full 0..2pi line at constant radius 0 would be a ring
        return not [ln for ln in ax.get_lines() if len(ln.get_xdata()) >= 100]

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    radar_panel(ax, values, [0.0] * 6, "#123456", 10)
    assert ax.get_ylim() == (0.0, 10)
    assert _below_base_markers(ax) == []
    assert no_ring(ax)
    outline = [ln for ln in ax.get_lines() if len(ln.get_xdata()) == 7]
    assert len(outline) == 1 and outline[0].get_ydata()[0] == 0.0  # drawn at r = 0, clipped
    assert values[0] == -1.3  # only the drawing clips
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    handles = radar_overlay_axes(ax, {"base": [0.0] * 6, "oct": values}, ["oct"], show_base=False)
    assert ax.get_ylim() == (0.0, 10)
    assert _below_base_markers(ax) == [] and no_ring(ax)
    assert not any("hollow" in h.get_label() for h in handles)
    assert [h.get_label() for h in handles] == [STYLE["oct"].label]  # no marker-key entry
    plt.close(fig)


def test_no_below_base_marker_for_zero_health_or_pending_spokes():
    import matplotlib.pyplot as plt

    from analysis.paper_figures import radar_panel

    # generality 0 (= base), coherence/capability >= 0 (health spokes are never negative),
    # a pending activations spoke
    values = [0.0, 4.0, 9.0, 9.0, float("nan"), 6.0]
    fig = plt.figure()
    ax = fig.add_subplot(projection="polar")
    radar_panel(ax, values, [0.0] * 6, "#123456", 10)
    assert _below_base_markers(ax) == []
    plt.close(fig)


def test_polygon_area_counts_from_the_radar_centre():
    from analysis.paper_figures import polygon_area_share
    from analysis.paper_style import RADAR_MIN

    assert polygon_area_share([10.0] * 6) == pytest.approx(1.0)
    assert polygon_area_share([RADAR_MIN] * 6) == pytest.approx(0.0)
    assert polygon_area_share([0.0] * 6) == pytest.approx(0.0)  # the base is the centre
    assert polygon_area_share([-50.0] * 6) == polygon_area_share([RADAR_MIN] * 6)  # clipped


def test_polygon_area_share_is_exact_for_the_score_and_deviation_centres():
    """Exact shares: a uniform polygon is ((v - c) / (10 - c))^2, and alternating 10/0 spokes
    share 0 from centre 0. Both centres are 0 now; a centre of -2 would give 1/6."""
    from analysis.paper_figures import polygon_area_share
    from analysis.paper_style import RADAR_MIN

    assert RADAR_MIN == 0.0
    assert polygon_area_share([5.0] * 6) == pytest.approx(0.25)
    assert polygon_area_share([5.0] * 6, centre=0.0) == pytest.approx(0.25)
    assert polygon_area_share([10.0, 0.0] * 3) == pytest.approx(0.0)
    assert polygon_area_share([10.0, 0.0] * 3, centre=0.0) == pytest.approx(0.0)
    assert polygon_area_share([-1.0, 5.0, 5.0, 5.0, 5.0, 5.0], centre=0.0) == pytest.approx(
        polygon_area_share([0.0, 5.0, 5.0, 5.0, 5.0, 5.0], centre=0.0)
    )  # clipped at the deviation centre


def test_the_deviation_legend_score_is_measured_from_zero():
    """An all-5 deviation polygon has radar score 0.25; the score radar also counts from 0."""
    import matplotlib.pyplot as plt

    import analysis.paper_figures as pf

    pd_ = select(_rows())
    for metric, centre in (("score", pf.RADAR_MIN), ("deviation", 0.0)):
        called = []
        orig = pf.radar_score

        def spy(values, rmax=10.0, centre=pf.RADAR_MIN, _o=orig, _c=called):
            _c.append(centre)
            return _o(values, rmax, centre)

        pf.radar_score = spy
        try:
            fig = pf.figure1_centre(pd_, "7B", metric=metric, legend_area=True)
        finally:
            pf.radar_score = orig
        plt.close(fig)
        assert set(called) == {centre}


def _mean_area_over_orderings(values, rmax=10.0):
    """Brute force: polygon area / full-polygon area, averaged over every cyclic ordering."""
    import itertools
    import math

    n = len(values)

    def area(rs):
        return sum(0.5 * rs[j] * rs[(j + 1) % n] * math.sin(2 * math.pi / n) for j in range(n))

    first, rest = values[0], values[1:]  # fix one spoke: the other n-1 orderings cover all cycles
    orders = [[first, *p] for p in itertools.permutations(rest)]
    return sum(area(o) for o in orders) / len(orders) / area([rmax] * n)


def test_radar_score_equals_the_mean_polygon_area_over_all_orderings():
    from analysis.paper_figures import radar_score

    cases = ([3.0, 7.0, 1.0, 9.0, 4.0, 6.0], [10.0, 0.0, 5.0, 5.0, 2.0, 8.0], [2.0, 4.0, 6.0, 8.0])
    for vals in cases:
        assert radar_score(vals) == pytest.approx(_mean_area_over_orderings(vals))
    v = [3.0, 7.0, 1.0, 9.0, 4.0, 6.0]  # n = 6, rmax = 10: sum of pair products / 1500
    assert radar_score(v) == pytest.approx(
        sum(v[i] * v[j] for i in range(6) for j in range(i + 1, 6)) / 1500
    )
    assert radar_score([0.3, 0.9, 0.5, 0.1, 0.7], rmax=1.0) == pytest.approx(
        _mean_area_over_orderings([0.3, 0.9, 0.5, 0.1, 0.7], rmax=1.0)
    )


def test_radar_score_limits_clipping_and_nan_spokes():
    import math

    from analysis.paper_figures import radar_score

    assert radar_score([10.0] * 6) == pytest.approx(1.0)
    assert radar_score([10.0] * 4, rmax=10.0) == pytest.approx(1.0)
    assert radar_score([0.0] * 6) == 0.0
    assert radar_score([5.0] * 6) == pytest.approx(0.25)
    assert radar_score([-3.0, 5, 5, 5, 5, 5]) == radar_score([0.0, 5, 5, 5, 5, 5])  # clipped
    assert radar_score([15.0] * 6) == pytest.approx(1.0)  # clipped at rmax
    # order-free, unlike the drawn polygon
    assert radar_score([10.0, 0.0] * 3) == radar_score([10.0] * 3 + [0.0] * 3)
    # a NaN spoke counts as 0 on the full six-spoke polygon, so entries stay comparable
    nan = math.nan
    five = radar_score([10.0] * 5 + [nan])
    assert five == radar_score([10.0] * 5 + [0.0]) and five == pytest.approx(2 / 3)
    assert radar_score([8.0, 6.0, 4.0, nan, 2.0, 9.0]) == radar_score(
        [8.0, 6.0, 4.0, 0.0, 2.0, 9.0]
    )
    assert math.isnan(radar_score([8.0, 6.0, nan, nan, nan, nan]))


def _drop_base(rows: list[Row], eval_name: str) -> list[Row]:
    return [r for r in rows if not (r.eval_name == eval_name and "-base-7B-" in r.spec_hash)]


def _with_second_voters(rows: list[Row]) -> list[Row]:
    """`_rows()` plus a second composite voter on generality and on introspection."""
    import dataclasses

    extra = []
    for r in rows:
        if r.eval_name == "introspection_mcq":
            extra.append(dataclasses.replace(r, eval_name="introspection_denial:gap"))
        if r.eval_name == "generality_fermi_gen":
            extra.append(dataclasses.replace(r, eval_name="generality_ood"))
    return rows + extra


@pytest.mark.parametrize(
    ("axis", "gone"),
    [("generality", "generality_fermi_gen"), ("introspection", "introspection_mcq")],
)
def test_a_voter_missing_on_the_base_makes_the_reading_pending_not_partial(axis, gone):
    import analysis.paper_figures as pf

    rows = _with_second_voters(_rows())
    full = select(rows)
    o = full.organism[("sft", "7B")]
    assert pf.reading(full, o, axis).status == "ok" and pf.reading(full, o, axis).n_evals == 2
    pd_ = select(_drop_base(rows, gone))
    r = pf.reading(pd_, pd_.organism[("sft", "7B")], axis)
    assert r.status == "pending" and r.why == f"base missing {gone}", r.why
    assert math.isnan(r.gain_raw)
    i = pf.RADAR_AXES.index(axis)
    assert math.isnan(pf.fig1_scores(pd_, "7B", ["sft"])["sft"][i])


def test_matched_means_is_nan_when_a_side_has_no_scorable_item():
    pd_ = select(_intro_rows([NAN] * 6, [0.5] * 6))
    o, b = pd_.organism[("sft", "7B")], pd_.organism[("base", "7B")]
    mm = pd_.matrix.matched_means(o, b, "introspection_mcq")
    assert mm is not None and math.isnan(mm[0]) and math.isnan(mm[1]) and mm[2] == 0


# ---------------------------------------------------------------- prompting vs best, legends


def _all_pass(pd_, **overrides: str) -> None:
    for org in pd_.organism.values():
        pd_.sbe_verdict[org] = "PASS"
    for (key, size), verdict in overrides_items(overrides):
        pd_.sbe_verdict[pd_.organism[(key, size)]] = verdict


def overrides_items(overrides: dict[str, str]):
    """`sysprompt_7B="FAIL"` -> (("sysprompt", "7B"), "FAIL")."""
    return [(tuple(k.split("_")), v) for k, v in overrides.items()]


def test_prompting_vs_best_hollow_markers_and_arrows():
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    import analysis.paper_figures as pf

    pd_ = select(_rows())
    _all_pass(pd_, sysprompt_7B="FAIL", oct_14B="FAIL")
    pts = pf.prompting_vs_best_points(pd_)
    assert pts
    for (_axis, size), pt in pts.items():
        assert pt["prompt_flagged"] == (size == "7B")  # only the prompting organism's own status
        assert pt["prompt_drawn"] >= pf.RADAR_MIN and pt["best_drawn"] >= pf.RADAR_MIN
        assert pt["arrow"] == (abs(pt["best_drawn"] - pt["prompt_drawn"]) > pf.PVB_ARROW_EPS)
        assert pt["best_flagged"] == (pt["best_key"] == "oct" and size == "14B")
    fig = pf.prompting_vs_best(pd_)
    arrows = [c for ax in fig.axes for c in ax.get_children() if isinstance(c, FancyArrowPatch)]
    assert len(arrows) == sum(pt["arrow"] for pt in pts.values())
    hollow = 0
    for ax in fig.axes:
        bg = ax.get_facecolor()
        for coll in ax.collections:
            if tuple(coll.get_facecolor()[0]) == tuple(bg):
                hollow += 1
    assert hollow == sum(pt["prompt_flagged"] + pt["best_flagged"] for pt in pts.values())
    xlabels = [ax.get_xlabel() for ax in fig.axes]
    assert all("of best other" in t and t.startswith("Prompting: ") for t in xlabels if t)
    plt.close(fig)


def test_prompting_vs_best_is_on_the_centre_equals_base_scale(monkeypatch):
    """A prompting value below base is drawn at RADAR_MIN (0) and keeps its signed value."""
    import analysis.paper_figures as pf

    real = pf.fig1_scores

    def neg(pd_, size, keys):
        out = real(pd_, size, keys)
        out["sysprompt"][pf.RADAR_AXES.index("generality")] = -1.3
        return out

    monkeypatch.setattr(pf, "fig1_scores", neg)
    pt = pf.prompting_vs_best_points(select(_rows()))[("generality", "7B")]
    assert pt["prompt"] == -1.3 and pt["prompt_drawn"] == pf.RADAR_MIN == 0.0
    assert pt["ratio"] == 0.0


def test_radar_legend_area_is_one_row_with_no_below_base_key(monkeypatch):
    import matplotlib.pyplot as plt

    import analysis.paper_figures as pf

    real = pf.fig1_scores

    def neg(pd_, size, keys):
        out = real(pd_, size, keys)
        out["oct"][pf.RADAR_AXES.index("generality")] = -1.3  # below base: drawn at the centre
        return out

    monkeypatch.setattr(pf, "fig1_scores", neg)
    pd_ = select(_rows())
    for area in (False, True):
        fig = pf.figure1_centre(pd_, "7B", legend_area=area)
        assert len(fig.legends) == 1, "one method row; the below-base key row is gone"
        texts = [t.get_text() for t in fig.legends[0].get_texts()]
        assert not any("hollow" in t for t in texts)
        assert all(("\n" in t) == area for t in texts)
        if area:  # name, then the radar score measured from the centre = base
            assert all(0.0 <= float(t.split("\n")[1]) <= 1.0 for t in texts)
        for ax in fig.axes:
            assert (
                not [  # no marker artists at r = 0
                    ln for ln in ax.get_lines() if ln.get_marker() not in (None, "None", "", " ")
                ]
            )
        plt.close(fig)


def test_expression_bar_legend_is_larger_and_open():
    import matplotlib.pyplot as plt

    import analysis.paper_figures as pf

    fig = pf.expression_rate_bar(select(_rows()))
    leg = fig.legends[0]
    assert all(t.get_fontsize() == pf.BAR_LEGEND_SIZE >= 9 for t in leg.get_texts())
    assert leg.get_title().get_fontsize() == pf.BAR_LEGEND_SIZE
    plt.close(fig)


# ---- generality vs coherence in Delta (organism - same-size base, 0-1 units) ----


def test_delta_readings_are_organism_minus_base_on_the_same_items():
    from analysis.paper_figures import delta_readings

    rows = _rows()
    pd_ = select(rows)
    dr = delta_readings(pd_, rows)
    for size in SIZES:
        for key in ("oct", "sft", "dpo"):
            g = dr[(size, key, "generality")]
            r = reading(pd_, pd_.organism[(key, size)], "generality")
            assert g.value == pytest.approx(r.gain_raw - r.gain_base_raw)
            c = dr[(size, key, "coherence")]
            # one coherence eval and complete items: the matched Delta is the raw difference
            rc = reading(pd_, pd_.organism[(key, size)], "coherence")
            assert c.value == pytest.approx(rc.raw - rc.base_raw)
            assert math.isfinite(c.se) and c.se >= 0
    assert ("7B", "icl", "generality") not in dr  # pending is absent, never 0
    assert ("7B", "base", "generality") not in dr


def test_gencoh_correlations_need_every_method_and_are_nan_otherwise():
    from analysis.paper_figures import (
        delta_readings,
        gencoh_correlation_table,
        gencoh_correlations,
    )

    rows = _rows()
    dr = delta_readings(select(rows), rows)
    corr = gencoh_correlations(dr)
    assert math.isnan(corr["Generality"]["7B"])  # few-shot absent from the grid
    five = corr["Generality, weight and activation methods only"]["7B"]
    assert -1.0 <= five <= 1.0
    tex = gencoh_correlation_table(dr)
    assert "\\toprule" in tex and "--" in tex and "7B" in tex


def test_generality_coherence_delta_figure_renders_light_and_dark_with_fake_stamp(tmp_path):
    from analysis.paper_figures import delta_readings, generality_coherence_delta

    rows = _rows()
    dr = delta_readings(select(rows), rows)
    for dark in (False, True):
        fig = generality_coherence_delta(
            dr, dark=dark, stamp="FAKE / SYNTHETIC DATA -- not a result"
        )
        assert len(fig.axes) == len(SIZES)
        fig.savefig(tmp_path / f"gc_{dark}.png")
    out = tmp_path / "all"
    _t, written = render_all(rows, out, draft=False, only={"gencoh"})
    assert (out / "generality_coherence_scale.pdf").exists()
    assert (out / "generality_coherence_correlations.tex").exists()
    assert "gencoh" in written


def _rows_with_gemma() -> list[Row]:
    """`_rows()` plus a Gemma-4-31B copy of the 14B grid (no steering/few-shot: they need real
    family hashes), so the Gemma panel has the methods that are picked by variant alone."""
    import dataclasses

    gem = [
        dataclasses.replace(
            r, base_model="google/gemma-4-31b-it", spec_hash=r.spec_hash.replace("14B", "31B")
        )
        for r in _rows()
        if r.spec_hash.split("-")[2] == "14B" and r.method not in ("steering", "few_shot_icl")
    ]
    return _rows() + gem


def test_generality_coherence_delta_is_a_four_panel_row_with_gemma_and_one_legend():
    import matplotlib.pyplot as plt

    from analysis.paper_figures import (
        GENCOH_SIZES,
        GENCOH_TITLES,
        delta_readings,
        generality_coherence_delta,
    )

    assert GENCOH_SIZES == (*SIZES, "31B")
    rows = _rows_with_gemma()
    dr = delta_readings(select(rows), rows, GENCOH_SIZES)
    assert any(k[0] == "31B" for k in dr)
    fig = generality_coherence_delta(dr, stamp="FAKE / SYNTHETIC DATA -- not a result")
    panels = [a for a in fig.axes if a.get_title()]
    assert [a.get_title() for a in panels] == [*SIZES, GENCOH_TITLES["31B"]]
    assert len({a.get_position().y0 for a in panels}) == 1  # one row
    assert len(fig.legends) == 1  # a single shared legend, no per-panel legends
    assert all(a.get_legend() is None for a in panels)
    # the Pearson r label: same text position (axes coords, bottom-left) in every panel
    pos = []
    for a in panels:
        (t,) = [t for t in a.texts if t.get_text().startswith("r = ")]
        pos.append((t.get_position(), t.get_ha(), t.get_va()))
    assert len(set(pos)) == 1
    assert pos[0] == ((0.1, 0.04), "left", "bottom")
    # one shared x-label, drawn centred under the whole row (not under one of 4 panels)
    labelled = [a for a in panels if a.get_xlabel()]
    assert len(labelled) == 1
    fig.canvas.draw()
    box = labelled[0].xaxis.label.get_window_extent()
    span = (panels[0].get_position().x0 + panels[-1].get_position().x1) / 2
    assert abs((box.x0 + box.x1) / 2 - span * fig.bbox.width) < 1.0  # within a pixel
    plt.close(fig)


def test_paper_build_draws_the_trade_off_figure_with_its_gemma_panel(monkeypatch):
    """`paper_build` must reproduce the paper's four-panel figure: the group that draws it
    holds the Gemma tab beside the three Qwen sizes, and every panel size is asked for."""
    import matplotlib.pyplot as plt
    import yaml

    import analysis.paper_build as pb
    import analysis.paper_figures as pf

    cfg = yaml.safe_load(
        (Path(__file__).resolve().parent.parent / "configs/dashboard_build.yaml").read_text()
    )
    (group,) = [
        g for g in cfg["paper"]["figures"] if "generality_coherence_scale" in g.get("draw", {})
    ]
    assert group["tabs"] == ["qimp-7b", "qimp-14b", "qimp-32b", "gem-31b"]
    seen: list[tuple[str, ...]] = []

    def spy(sel, rows, sizes=pf.SIZES):
        seen.append(tuple(sizes))
        return {}

    monkeypatch.setattr(pf, "delta_readings", spy)
    plt.close(pb._figures()["generality_coherence"](None, [], {}, None))
    assert seen == [tuple(pf.GENCOH_SIZES)]


def test_generality_coherence_delta_drops_a_size_without_readings():
    from analysis.paper_figures import delta_readings, generality_coherence_delta

    rows = _rows()  # no Gemma rows
    fig = generality_coherence_delta(delta_readings(select(rows), rows))
    assert [a.get_title() for a in fig.axes if a.get_title()] == list(SIZES)


def test_generality_coherence_delta_with_no_readings_draws_no_legend():
    from analysis.paper_figures import generality_coherence_delta

    fig = generality_coherence_delta({})
    assert not fig.legends


def test_an_extra_steering_organism_is_hidden_by_the_papers_families_and_named_without() -> None:
    """run_all.sh's dashboard step passes configs/dashboard_build.yaml, so an extra steering
    organism beside the paper's is left out, as on main (where the families were hard-coded).
    With no config both are kept, and the collision is a clear error naming `families:`."""
    import dataclasses
    import re

    import pytest

    from analysis.paper_figures import load_families, select

    rows = _rows()
    extra = [
        dataclasses.replace(r, variant="all x2", family="f-extra", spec_hash=r.spec_hash + "x")
        for r in rows
        if r.method == "steering" and r.model_tag == "7B"
    ]
    sel = select(rows + extra, load_families(_CONFIG))
    assert sel.organism[("steering", "7B")] == extra[0].organism.replace("[all x2]", "[all]")
    assert extra[0].organism in sel.excluded
    with pytest.raises(ValueError, match=r"families: \{steering: \[<family hash>\]\}"):
        select(rows + extra)
    script = (_CONFIG.parent.parent / "scripts" / "run_all.sh").read_text(encoding="utf-8")
    call = script[script.index("-m analysis.dashboard \\") :]
    call = call[: call.index("--dump-data")]
    assert re.search(
        r"--config \"\$\{MO_EVALS_DASHBOARD_CONFIG:-configs/dashboard_build\.yaml\}\"", call
    )
