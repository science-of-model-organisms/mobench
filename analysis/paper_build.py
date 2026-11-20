"""Rebuild the paper's results tables and figures from the pinned data, in one command.

    uv run python -m analysis.paper_build --config configs/dashboard_build.yaml \\
        [tables|figures|check|all]

One command from a fresh checkout: it first makes `<dashboard_root>` hold a build at the config's
pins (`dashboard_build.ensure_built`: `fetch` unless the recorded downloads already equal the
pins, `tree`, per-tab `rows` and `reduce`; only what is missing), verifies the pins, and reads
that build's per-tab reductions (`tabs/`), rows (`rows/`) and normalised trees (`norm/`): the
pinned HF revisions with their `-restricted` companions. No number is computed here: the tables
are `analysis.paper_tables` over the reductions, the figures `analysis.paper_figures` over the
rows.

tables:  every `paper.tables` entry -> `<out>/tables/<label>.tex` (the table body; captions and
         column heads are prose and stay in the paper).
figures: every `paper.figures` group -> `<out>/figures/<stem>.pdf`, one `select` over the group's
         rows (the paper's organism set: `families:`), as the paper's figures were drawn.
check:   every `paper.tables` entry against the printed table in `paper.tex`, number by number at
         printed precision -> `<out>/check.json` and a MATCH/total line per table.

`--out` defaults to `<dashboard_root>/paper`. Figures the paper draws by hand (the overview,
pipeline and Figure 1 schematics) are not data figures and are not rebuilt.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

REPO = Path(__file__).resolve().parent.parent


def _figures() -> dict[str, Callable[..., Any]]:
    """Figure function (`paper.figures[].draw.<stem>.fn`) -> builder(sel, rows, entry, tree)."""
    import analysis.paper_figures as pf
    from analysis.derived import mu_gen_intervals

    return {
        "figure1_centre": lambda sel, rows, f, tree: pf.figure1_centre(sel, f["size"]),
        "radar": lambda sel, rows, f, tree: pf.radar(sel, f["size"]),
        "prompting_vs_best": lambda sel, rows, f, tree: pf.prompting_vs_best(sel),
        "expression_bar": lambda sel, rows, f, tree: pf.expression_rate_bar(
            sel, pf.SIZES, reference=pf.EXPRESSION_DELTA_GUIDE
        ),
        "generality_coherence": lambda sel, rows, f, tree: pf.generality_coherence_delta(
            pf.delta_readings(sel, rows, pf.GENCOH_SIZES)
        ),
        "generality_scale": lambda sel, rows, f, tree: pf.generality_scale(sel),
        "axis_dots": lambda sel, rows, f, tree: pf.axis_dots(sel),
        "mu_bar": lambda sel, rows, f, tree: pf.mu_bar(sel, mu_gen_intervals(rows) or None),
        "kept_scatter": lambda sel, rows, f, tree: pf.kept_scatter(sel),
        "debate_turn_curve": lambda sel, rows, f, tree: pf.debate_turn_curve(
            tree, families=f.get("families")
        ),
    }


def _multi() -> dict[str, tuple[Callable[..., Any], Callable[..., Any]]]:
    """Figures over several selections (`paper.figures[].selections`): fn -> (points of one
    selection, draw from every selection's labelled points)."""
    import analysis.paper_figures as pf

    return {
        "misalignment_dumbbell": (pf.misalignment_points, pf.misalignment_dumbbell),
        "comparison": (pf.comparison_points, pf.comparison),
    }


def _load_sel(cfg: Any, tabs: list[str]) -> tuple[Any, list[Any]]:
    import analysis.paper_figures as pf

    rows: list[Any] = []
    for t in tabs:
        rows += pickle.loads((cfg.root / "rows" / f"{t}.pkl").read_bytes())
    if not rows or any(r.fake for r in rows):
        raise SystemExit(f"{tabs}: no rows, or a FAKE row (CLAUDE.md rule 5)")
    try:
        return pf.select(rows, cfg.families), rows
    except ValueError as e:  # two organisms on one method row
        raise SystemExit(f"figures for tabs {tabs}: {e}") from None


def _figures_multi(cfg: Any, group: dict[str, Any], draw: dict[str, Any], out: Path) -> None:
    """One `selections` group: each selection's rows are loaded, read into points and freed in
    turn (seven tabs of rows do not fit in memory at once), then every figure is drawn."""
    import matplotlib.pyplot as plt

    import analysis.paper_figures as pf

    multi = _multi()
    bad = sorted({f["fn"] for f in draw.values()} - set(multi))
    if bad:
        raise SystemExit(f"unknown multi-selection figure functions {bad}; one of {sorted(multi)}")
    points: dict[str, list[tuple[str, Any]]] = {stem: [] for stem in draw}
    for sel_spec in group["selections"]:
        sel, rows = _load_sel(cfg, sel_spec["tabs"])
        for stem, f in draw.items():
            points[stem].append((sel_spec["label"], multi[f["fn"]][0](sel)))
        del sel, rows
    for stem, f in draw.items():
        pf.use_paper_style()
        fig = multi[f["fn"]][1](points[stem])
        with plt.rc_context(cast(Any, pf.SAVE_RC.get(f["fn"], {}))):  # Literal-keyed type
            fig.savefig(out / "figures" / f"{stem}.pdf", format="pdf")
        plt.close(fig)
        (out / "figures" / f"{stem}.points.json").write_text(
            json.dumps(points[stem], indent=1, allow_nan=True), encoding="utf-8"
        )
        print("wrote", out / "figures" / f"{stem}.pdf", flush=True)


def cmd_tables(cfg: Any, out: Path) -> None:
    from analysis import paper_tables
    from analysis.dashboard_build import tab_json

    (out / "tables").mkdir(parents=True, exist_ok=True)
    for label, spec in cfg.paper["tables"].items():
        rows = paper_tables.build(spec, lambda t: tab_json(cfg, t))
        path = out / "tables" / f"{label.replace(':', '-')}.tex"
        path.write_text(paper_tables.render(label, rows), encoding="utf-8")
        print("wrote", path, len(rows), "rows", flush=True)


def _merged_tree(cfg: Any, tabs: list[str], dst: Path) -> Path:
    """One tree over the group's normalised tab trees (hardlinks; their organisms are disjoint)."""
    shutil.rmtree(dst, ignore_errors=True)
    for t in tabs:
        src = cfg.root / "norm" / t
        for p in sorted(src.rglob("*")):
            q = dst / p.relative_to(src)
            if p.is_dir():
                q.mkdir(parents=True, exist_ok=True)
            elif not q.exists():
                q.parent.mkdir(parents=True, exist_ok=True)
                os.link(p, q)
    return dst


def cmd_figures(cfg: Any, out: Path, only: set[str] | None = None) -> None:
    import matplotlib.pyplot as plt

    import analysis.paper_figures as pf

    figs = _figures()
    (out / "figures").mkdir(parents=True, exist_ok=True)
    for group in cfg.paper["figures"]:
        draw = {k: v for k, v in group["draw"].items() if only is None or k in only}
        if not draw:
            continue
        if "selections" in group:
            _figures_multi(cfg, group, draw, out)
            continue
        bad = sorted({f["fn"] for f in draw.values()} - set(figs))
        if bad:
            raise SystemExit(f"unknown figure functions {bad}; one of {sorted(figs)}")
        sel, rows = _load_sel(cfg, group["tabs"])
        tree = None
        if any(f["fn"] == "debate_turn_curve" for f in draw.values()):
            tree = _merged_tree(cfg, group["tabs"], out / "_tree")
        for stem, f in draw.items():
            pf.use_paper_style()
            fig = figs[f["fn"]](sel, rows, {**f, "families": cfg.families}, tree)
            fig.savefig(out / "figures" / f"{stem}.pdf", format="pdf")
            plt.close(fig)
            print("wrote", out / "figures" / f"{stem}.pdf", flush=True)
        if tree is not None:
            shutil.rmtree(tree)
        del rows, sel


def cmd_check(cfg: Any, out: Path) -> dict[str, Any]:
    from analysis.paper_checks import tex_checks

    res = tex_checks(cfg)
    out.mkdir(parents=True, exist_ok=True)
    (out / "check.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for label, r in res.items():
        if "skipped" in r:
            print(f"{label:22s} SKIPPED ({r['skipped']})")
            continue
        print(
            f"{label:22s} MATCH {r['match']}/{r['checked']}  missing {len(r['missing'])}"
            f"  unmatched rows {len(r['unmatched_rows'])}",
            flush=True,
        )
    return res


def main(argv: list[str] | None = None) -> None:
    from analysis.dashboard_build import DEFAULT_CONFIG, Config, ensure_built

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument(
        "what", nargs="?", default="all", choices=["tables", "figures", "check", "all"]
    )
    ap.add_argument("--out", type=Path, help="default: <dashboard_root>/paper")
    ap.add_argument("--only", default="", help="figures: comma-separated file stems")
    args = ap.parse_args(argv)
    cfg = Config(args.config)
    if cfg.paper is None:
        raise SystemExit(f"{args.config} has no `paper:` block")
    out = args.out or cfg.root / "paper"
    cfg.add_output(out)  # before the build digests the repo, so it never hashes what we write
    ensure_built(cfg)
    if args.what in ("tables", "all"):
        cmd_tables(cfg, out)
    if args.what in ("figures", "all"):
        cmd_figures(cfg, out, {s for s in args.only.split(",") if s} or None)
    if args.what in ("check", "all"):
        cmd_check(cfg, out)


if __name__ == "__main__":
    sys.path.insert(0, str(REPO))
    main()
