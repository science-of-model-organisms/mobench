# Paper figures

The paper's figures and results tables are generated from the results on Hugging Face, at the
revisions the paper used, with the style rules below enforced in code (`analysis/paper_style.py`).
Do not hand-edit a PDF or a table; change the code and re-render.

## Reproducing the paper

The paper's data are the pinned revisions in [`configs/dashboard_build.yaml`](../../configs/dashboard_build.yaml)
(`revisions:`): three public results repos and their private `-restricted` companions (the eval text
that may not be public; `HF_TOKEN` must be able to read them), plus the decisiveness bootstrap sidecar
(`mu_sidecar:`). Which organisms are the paper's is `families:` in the same file.

```bash
uv sync --all-groups --extra figures        # seaborn (the `figures` extra)
export HF_TOKEN=...
C=configs/dashboard_build.yaml
uv run python -m analysis.dashboard_build --config $C fetch   # ~6 GB at the pins
uv run python -m analysis.dashboard_build --config $C all     # trees, rows, per-tab reductions, the page (~41 GB peak)
uv run python -m analysis.paper_build --config $C             # tables, figures and the check (~43 GB peak)
# paper_build alone also works from a fresh checkout: it fetches and builds what is missing, and verifies the pins
```

`analysis.paper_build` writes, under `<dashboard_root>/paper/`:

- `tables/<label>.tex`: the body of every results table in `paper.tables` (captions and column heads are
  prose and stay in the paper), from `analysis/paper_tables.py` over the per-tab reductions. The Δ
  columns are `paper_figures.delta_readings` (organism minus same-size base on the items scorable for
  both); the truth-probe intervals are `analysis.truth_probe.analyse_tree`'s paired bootstrap.
- `figures/<stem>.pdf`: every figure in `paper.figures`, each drawn by the `analysis.paper_figures`
  function the config names, over one `select` of its group's rows. A figure across behaviours or
  model families (`misalignment_dumbbell`, `comparison_32b`) lists labelled `selections` instead:
  each is selected and reduced to a few numbers in turn (also written as `<stem>.points.json`).
- `check.json`: every table in `paper.checks` compared with the table the paper prints (`paper.tex`, the
  LaTeX main file with its `\input`s), number by number at printed precision, with a `MATCH x/y` line
  per table. The same check runs in the dashboard build's `data` stage (`_build.cross_check.tex`).

`tables`, `figures` or `check` runs one part; `--only radar_32b,expression_bar` limits the figures.
Not rebuilt: the hand-drawn schematics (overview, pipeline, the Figure 1 sides).

For a quick render of one results tree that is not the paper's (no pins, no tables):

```bash
uv run python -m analysis.paper_figures --root <results tree> --normalised-root /tmp/norm --out figures \
  [--config configs/dashboard_build.yaml]     # its `families:`; default: every organism of each method
```

- `--normalised-root` mirrors the tree with every Parquet cast to `core.results.SCHEMA`
  (some uploads use `large_string`, which `load_all` cannot concatenate), leaving out any
  `superseded/` folder (replaced cells kept under the same spec hashes). It is required when
  the run folder has one.
- `--mu-ci` is optional. It points to upstream's item-cluster bootstrap intervals for mu
  decisiveness, computed by `analysis.friedness_panel.friedness_panel(rows, bootstrap=True)`
  per organism (about 45 s per organism on one CPU).

## Rules

1. **Page geometry.** The paper template has a single column with a 5.5in `\textwidth`.
   Figures are drawn at their printed width and saved without a tight bounding box, so the
   fonts are never rescaled.
2. **Font.** Times: real TeX with `times` when installed, otherwise STIX. Check the embedded
   fonts; every PDF should embed only STIXGeneral.
3. **Tools and output.** Use seaborn on matplotlib, and seaborn wherever it fits (bars, dots,
   lines). Radars are matplotlib polar with the same theme. Output is always PDF; PNGs are
   review previews only.
4. **Names.** OCT, SFT, DPO, Steering, Distillation, Prompting, Few-shot, Base model.
5. **Controls.** The Base model is the only control in figures. The negative prompt appears
   only in the appendix table.
6. **Colour.** Two palettes, never mixed in one figure:
   - method colours for figures whose series are methods (radars, generality across scale);
   - size colours for figures whose series are model sizes (Figure 2 bars, Figure 3 dots,
     mu). The size colours are 7B blue, 14B orange and 32B green, from the paper's Figure 2.
7. **Radars.** The spokes are Generality, Robustness, Coherence, Capability, Activations and
   Introspection; expression and misalignment are not on them.
   - Three rules, one per kind of spoke, defined once in `analysis/radar_scale.radar_value` (re-exported by `analysis/paper_style`):
     - **Behaviour spokes = gain over base.** Generality, introspection and activations are
       10 x (organism - Base model), signed and not clipped below, so the Base model is 0 and
       a method that does worse than base on a spoke is negative. The top is clipped at 10.
       Both composites are taken over the MATCHED items only (scorable for both, per eval,
       before averaging: `Matrix.matched_means`, `Reading.gain_raw` / `gain_base_raw`), since
       `aggregate` drops unscorable samples per side and an unmatched difference would mix a
       change in the item set into the gain. An eval with no shared item makes the spoke
       pending. Activations (the truth probe, `whitebox_truth_world` and `_self`) is the mean
       of the two cells' organism - base shares, NaN (pending, not drawn) unless the
       organism and the base both have both cells.
     - **Health spokes = 1 + delta vs base.** Coherence and capability are
       10 x (1 + (organism - Base model)) raw composite, clipped to [0, 10]: the Base model
       sits on the outer ring (10) and damage moves inward linearly in the same 0-1 delta as
       the behaviour spokes (one score: delta vs the same-size base).
     - **Robustness = share kept under pressure.** 10 x the mean share kept (the organism's
       trait under pressure over its own rate without), never read against the Base model.
   - The centre of every radar is 0 (`paper_style.RADAR_MIN`). On a behavior spoke 0 is the
     Base model, so the centre is the base; on a health spoke the base is the outer ring (10).
     There is no base ring. A behavior value below 0 (below the Base model) is drawn at the
     centre, in the drawing only, with NO marker (a circle at the centre cannot say which spoke is
     below base; the figure captions say "Values below the base model are drawn at the centre").
     The signed value is unchanged in tables, tooltips and the payload.
   - Tables keep the ABSOLUTE rates (`Reading.raw`); only the radar spokes are gains.
   - Where the Base model is drawn as an outline (`show_base`), it is a thick dark dashed
     outline with vertex markers, on top.
   - A pending spoke (no data, or a gated white-box cell) is never drawn as 0.
   - Figure 4 plots the ABSOLUTE generality composite (`Reading.raw`), not the radar spoke
     (which is 10 x the gain over base on matched items), and its caption says so.
8. **No explanatory text inside a figure.** Scales, abbreviations and "dashed = base" belong
   in the caption. A compact legend is fine.
9. **Legends, not end-of-line labels,** on multi-series line plots.
10. **No "preliminary" stamps once the data are final.** A missing cell is omitted and named
    in the caption, never drawn as 0.

## Figures

| File | What it is |
|---|---|
| `radar_methods_32b.pdf` | Figure 1 centre panel: three radars (Prompting and Few-shot; Steering and Distillation; SFT, DPO and OCT), 32B, about 0.56 of the text width. The schematic either side is added by hand. |
| `radar_methods_32b_wide.pdf` | The same at full text width |
| (legend of the `radar_methods_*` figures) | One entry per method, by name only (no score under it). |
| `radar_methods_7b.pdf`, `radar_methods_14b.pdf` | The same three-radar layout at 7B and 14B (appendix). In the paper's figures folder but not written by this script |
| `expression_bar.pdf` | Figure 2: expression rate per method, grouped by size (per-size variants: `expression_bar_<size>.pdf`) |
| `axis_dots.pdf` | Figure 3: prompting comparison, showing deviation from base per axis at every size |
| `generality_scale.pdf` | Figure 4: generality across scale: the raw 0-1 generality composite per method and size, Base Model dashed |
| `radar_overlay_<size>.pdf` | Every method overlaid on one radar, per size |
| `radar_<size>.pdf` | Per-method radar small multiples (appendix) |
| `axis_dots_<size>.pdf` | Figure 3 at a single size (appendix) |
| `mu_decisiveness.pdf` | $\mu$-decisiveness per method and size, with bootstrap intervals |
