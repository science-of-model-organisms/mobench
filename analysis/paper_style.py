"""The paper's figure style: one place for every look-and-feel decision.

These are lessons from drafting the 2026 MO paper (a single-column conference template),
kept in code so a new figure does not have to relearn them.
`analysis/paper_figures.py` imports everything from here;
`docs/reference/PAPER-FIGURES.md` is the human-readable version.

RULES
1. Page geometry: the paper template is single-column, \\textwidth = 5.5in.
   Figures are drawn at their final printed width (5.5in, or 3.3in for a 0.6\\textwidth panel)
   and saved without a tight bbox, so \\includegraphics never rescales the fonts.
2. Font: Times (the template loads `times`): real TeX + times when a TeX install exists,
   else STIX (a Times clone) with STIX mathtext. Never Computer Modern, never sans.
3. Tooling: seaborn on matplotlib (radars: matplotlib polar, same theme). Output: vector
   PDF, always. PNGs exist only as review previews rendered from the PDFs.
4. Names (display): OCT, SFT, DPO, Steering, Distillation, Prompting, Few-shot, Base model.
   Never "OCT (ours)", "seq-KD", "system prompt" or "anti-prompt".
5. Controls: the Base model is the only control in paper figures. The negative (anti-)
   prompt appears only in the appendix "Negative prompting" table, never in a figure.
6. Colour: two palettes, never mixed in one figure.
   - METHOD colours (one hue per implant method) for figures whose series ARE methods
     (radars, generality across scale).
   - SIZE colours (7B blue, 14B orange, 32B green; the paper's palette) for figures whose
     series are model sizes (Fig. 2 expression bars, Fig. 3 per-axis dots, mu bars).
7. Radars: base sits on the OUTER ring for coherence and capability (10 = no degradation;
   damage moves inward) and at the CENTRE (0) on the behaviour axes (10 x the gain over
   base; below base is drawn at the centre, unmarked). Spokes: Generality,
   Robustness, Coherence, Capability, Activations, Introspection.
   A pending spoke (no data, or a gated white-box cell) is never drawn as 0.
8. No explanatory text inside a figure (scales, abbreviations): that is
   the caption's job. A compact legend is fine.
9. Legends, not end-of-line labels, for multi-series line plots (end labels collide).
10. No "preliminary" stamps once the data are final; a genuinely missing cell is omitted
    and named in the caption, never drawn as 0.
"""

from __future__ import annotations

import importlib
import shutil
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt

# The radar rule lives in `analysis/radar_scale.py` (no matplotlib), shared with the dashboard;
# re-exported here so every figure keeps importing it from this module.
from analysis.radar_scale import (  # noqa: F401
    BASE_NO_ROBUSTNESS_WHY,
    BASE_RELATIVE_AXES,
    HEALTH_AXES,
    RADAR_MAX,
    RADAR_MIN,
    TRUTH_PROBE_AXES,
    TRUTH_PROBE_VOTERS,
    radar_value,
)

# ---------------------------------------------------------------- geometry
SINGLE_COL_IN = 3.3
TEXT_WIDTH_IN = 5.5
RADAR_H_IN = 3.6
RADAR_PER_METHOD_H_IN = 5.4  # appendix per-method radars, 3 x 3 grid
EXPR_H_IN = 2.5
DOTS_H_IN = 3.1
SCALE_H_IN = 2.6
FIG1_W_IN = 5.225  # the intro figure is included at 0.95\linewidth
FIG1_H_IN = 1.95

# ---------------------------------------------------------------- ink
INK = "#1a1a1a"
MUTED = "#6b6b6b"
GRID = "#e4e4e4"
BASE_OUTLINE = "#3c3c3c"  # the Base model on radars: dark, thick, dashed, with vertex markers
RADAR_GRID = "#c8c8c8"
HIGHLIGHT_BAND = "#e8eef8"  # the Prompting row band in Fig. 3 (as in the original figure)

# ---------------------------------------------------------------- sizes
SIZES = ("7B", "14B", "32B")
SIZE_MARKERS = {"7B": "o", "14B": "s", "32B": "^"}
# Size palette from the paper's Fig. 2 — also the original Fig. 3's marker colours.
SIZE_COLOURS = {"7B": "#4477CC", "14B": "#DD7044", "32B": "#55AA7F", "31B": "#8E5FB0"}
# Legend groups of the grouped bars when a non-Qwen size is drawn (31B is Gemma-4): one legend
# per base-model family, (title, sizes). The Qwen-only figures keep their single legend.
SIZE_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Qwen2.5-Instruct", SIZES),
    ("Gemma-4", ("31B",)),
)


@dataclass(frozen=True)
class MethodStyle:
    key: str
    label: str
    colour: str
    linestyle: str | tuple = "-"
    highlight: bool = False


# METHOD palette: custom (2026-10-19; the old greens and blues were too close, and the
# seaborn palettes each had a near pair). Chosen from a CIELCh grid (L 40-65, C 40-95) to
# maximise the minimum pairwise CIEDE2000 over ALL 21 pairs: >= 23.6 normal vision, >= 14
# simulated deuteranopia/protanopia, every colour >= 3:1 on white and on #121212. Assigned so
# methods sharing a radar panel are the most distinct. Grey is left for the Base model.
METHODS: tuple[MethodStyle, ...] = (
    MethodStyle("oct", "OCT", "#B38B1A"),
    MethodStyle("sft", "SFT", "#1A9576"),
    MethodStyle("dpo", "DPO", "#135DF6"),
    MethodStyle("steering", "Steering", "#0E9BD2", (0, (5, 1.5))),
    MethodStyle("seqkd", "Distillation", "#FE4F7F", (0, (3, 1, 1, 1))),
    MethodStyle("sysprompt", "Prompting", "#9B5A34", "-", highlight=True),
    MethodStyle("icl", "Few-shot", "#93548C", (0, (1, 1))),
)
# The negative-prompting control: scored like any organism (it stays in the matrix, so the
# cross-organism spreads behind every effect are unchanged) but drawn in NO figure (rule 5).
CONTROLS: tuple[MethodStyle, ...] = (
    MethodStyle("antiprompt", "Negative prompt", "#4a4a4a", (0, (4, 1, 1, 1, 1, 1))),
)
BASE_STYLE = MethodStyle("base", "Base model", "#8C8C8C", (0, (4, 2)))
STYLE = {s.key: s for s in (*METHODS, *CONTROLS, BASE_STYLE)}
# Left-to-right order for bar charts (weakest implant first, as in the paper's Fig. 2).
BAR_ORDER = ("base", "sysprompt", "icl", "steering", "seqkd", "dpo", "sft", "oct")
# Top-to-bottom order for the per-axis dot plot (as in the original Fig. 3).
ROW_ORDER = ("oct", "sft", "dpo", "steering", "seqkd", "sysprompt", "icl")


def seaborn() -> Any:
    """The seaborn module (the `figures` extra), imported on first use.

    Imported dynamically so the rest of `analysis` and CI (which does not install the extra)
    never need it, and the type checker does not have to resolve an optional package.
    """
    try:
        return importlib.import_module("seaborn")
    except ImportError as exc:  # pragma: no cover - message only
        raise ImportError(
            "paper figures need seaborn: uv sync --all-groups --extra figures"
        ) from exc


def use_paper_style() -> None:
    """Times look (rule 2): real TeX + times if installed, else STIX.

    Figures are drawn with seaborn on matplotlib (`figures` extra): the seaborn "ticks" theme
    at "paper" context, then these rcParams on top, so the same look holds for the seaborn
    plots (bars, dots, lines) and the plain-matplotlib polar radars.
    """
    seaborn().set_theme(context="paper", style="ticks", font="serif")
    usetex = shutil.which("latex") is not None
    plt.rcParams.update(
        {
            "text.usetex": usetex,
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "Nimbus Roman", "DejaVu Serif"],
            "text.latex.preamble": r"\usepackage{times}",
            "mathtext.fontset": "stix",
            "axes.formatter.use_mathtext": True,
            "axes.unicode_minus": True,
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "lines.linewidth": 1.4,
            "pdf.fonttype": 42,
            "savefig.bbox": None,  # rule 1: exact page size, no tight bbox
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
