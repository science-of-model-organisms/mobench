"""The paper's results figures, as vector PDFs in a LaTeX look.

The figures share one method->colour mapping (`METHODS`):

1. ``radar_<size>.pdf`` — one panel per implant method on the six RADAR_AXES spokes
   (`fig1_scores` via `paper_style.radar_value`, NOT `Matrix.axis_effect`), three rules:
   BEHAVIOUR spokes (generality, introspection, activations) are the GAIN OVER BASE, signed,
   10 x (organism - Base model) over the MATCHED items (scorable for both, `Reading.gain_raw`),
   so the Base model is the radar centre (0) and a method below base is drawn at the centre
   (no marker; the value stays signed); HEALTH spokes
   (coherence, capability) are 10 x (1 + (organism - base)) raw composite, clipped to [0, 10],
   10 = the Base model (damage moves inward linearly in the 0-1 delta);
   ROBUSTNESS is 10 x the mean SHARE KEPT (the `:kept` cells: trait under pressure over the
   organism's own no-pressure rate, never read against base). The centre is `RADAR_MIN`
   (0). There is no base ring. An organism whose `simple_behaviour_exhibit`
   verdict is not PASS or PASS_INDIST (or has none) keeps its polygon but has its whole
   outline dashed (`FLAG_LINESTYLE`); a passing organism's outline is solid. No vertex markers.
2. ``expression_bar.pdf`` — `simple_behaviour_exhibit` present rate per method, grouped by
   base-model size, with the cluster-robust 95% CI (`MatrixCell.ci95`: items clustered,
   seeds pooled).
3. ``axis_dots.pdf`` — per axis, each method's deviation from base in the evals' own units
   (mean signed delta over the axis's composite evals, the quantity behind
   `Matrix.axis_delta`), one marker per size.
4. ``generality_scale.pdf`` — the raw generality composite (`Matrix.axis_score`) vs size,
   and its delta vs the base at the same size.
5. ``generality_coherence_scale.pdf`` — coherence Delta against generality Delta (organism
   minus the same-size Base model, matched items, 0-1 units), one panel per size, with
   ``generality_coherence_correlations.tex`` (Pearson r per size across methods).
6. ``robustness_strength_scatter.pdf`` — strength (the no-pressure
   `robustness_open_ended_gen` rate, x) against robustness share kept (y), one marker per
   method x size, SBE-flagged markers hollow: share kept is read beside strength, never alone.

Nothing unmeasured is drawn. A (method, size, axis) reading needs EVERY composite eval the
base at that size has a usable cell for; a partial one is "pending", never a zero and never
a mean over whatever happened to land. A self-referenced eval (`reference: self` in
eval_meta.yaml: the robustness `:kept` cells) needs no base cell: it counts wherever it was
derived, and an organism whose share is undefined (no trait at its baseline) omits it.
`coverage()` prints exactly which cells are in.

Draft renders (``--draft``) carry a visible "preliminary" stamp naming what is pending; a
figure with no measured data at all renders as a visibly-placeholder page.

Run::

    PYTHONPATH=. python -m analysis.paper_figures \
        --root <results tree> --out <dir> [--draft] [--normalised-root <scratch dir>]

``--normalised-root`` mirrors the tree with every Parquet cast to `core.results.SCHEMA`
first (hardlinks for the rest). `core.results.load_all` concatenates tables without
promotion, so one writer's ``large_string`` columns (the 2026-09-25 upload's
`simple_behaviour_exhibit` cells) otherwise fail the whole load.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import statistics
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from matplotlib import patches as mpatches  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.projections.polar import PolarAxes  # noqa: E402
from matplotlib.ticker import (
    MaxNLocator,  # noqa: E402
    NullFormatter,  # noqa: E402
)

from analysis.aggregate import (  # noqa: E402
    BASE_METHOD,
    Matrix,
    MatrixCell,
    Row,
    build_matrix,
    load_rows,
    organism_model_tag,
)
from analysis.derived import SBE_DELTA  # noqa: E402  # the one expression-threshold margin
from analysis.paper_style import seaborn as _seaborn  # noqa: E402  # the `figures` extra
from core.results import SCHEMA, read_result_table, result_files  # noqa: E402

if TYPE_CHECKING:
    from analysis.derived import KeptBySeed

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- style
# Every look-and-feel constant lives in analysis/paper_style.py (with the rules behind it).
from analysis.paper_style import (  # noqa: E402
    BAR_ORDER,
    BASE_NO_ROBUSTNESS_WHY,
    BASE_OUTLINE,
    BASE_RELATIVE_AXES,
    BASE_STYLE,
    CONTROLS,
    DOTS_H_IN,
    EXPR_H_IN,
    GRID,
    HIGHLIGHT_BAND,
    INK,
    METHODS,
    MUTED,
    RADAR_GRID,
    RADAR_H_IN,
    RADAR_MIN,
    RADAR_PER_METHOD_H_IN,
    ROW_ORDER,
    SCALE_H_IN,
    SINGLE_COL_IN,
    SIZE_COLOURS,
    SIZE_GROUPS,
    SIZE_MARKERS,
    SIZES,
    STYLE,
    TEXT_WIDTH_IN,
    TRUTH_PROBE_AXES,
    TRUTH_PROBE_VOTERS,
    radar_value,
    use_paper_style,
)

# Figure 3's panels, in the paper's order, with the question each answers.
DOT_AXES = (
    ("generality", "Generality", "does it transfer?"),
    ("robustness", "Robustness", "does it survive pushback?"),
    ("introspection", "Introspection", "does it know it has the trait?"),
    ("coherence", "Coherence", "is the model still coherent?"),  # restored 2026-09-25
    ("capability", "Capability", "is it still as smart?"),
)
# The radar's spokes (supervisor, 2026-09-25): exactly these six, in this order. Expression
# has its own bar chart; misalignment is off the radar. Activations (PR #358) is voted by the
# truth-probe cells (whitebox_truth_world + _self) since 2026-10-14 (the V0 probe cells are
# retired) and drawn BASE-RELATIVE like generality and introspection: 10 x mean(world, self) of
# organism - base share, base = 0, signed (2026-10-19).
# It needs BOTH cells; either missing reads as pending (NaN), never 0.
from analysis.radar_scale import RADAR_AXES  # noqa: E402  # one order, shared with the dashboard

COVERAGE_AXES = (*RADAR_AXES, "misalignment", "expression")
# An organism whose `simple_behaviour_exhibit` verdict is not in `SBE_OK` (a missing verdict
# included) was never shown to carry the trait, so its whole profile (share kept above all)
# describes a trait not shown to be there. It is still drawn, with a dashed outline, and named
# in the captions. A mark, never a filter (2026-10-01). The base keeps its own style.
SBE_OK = frozenset({"PASS", "PASS_INDIST"})
OPEN_ENDED_GEN = "robustness_open_ended_gen"
SBE = "simple_behaviour_exhibit"
SBE_UNSEEN = f"{SBE}:unseen"
MU_EVAL = "mu_decisiveness:decisiveness"
# (method key, eval) cells the run config leaves out ON PURPOSE (not pending): the axis
# composite for that method averages the remaining evals, and captions say so. Empty since
# few-shot ICL got `apps` at every size (HF 7be4554c / 5c62a2ad, 2026-09-26).
NOT_RUN_BY_DESIGN: frozenset[tuple[str, str]] = frozenset()


# ---------------------------------------------------------------- data


SUPERSEDED = "superseded"
# Written into every normalise_tree mirror; only a dir carrying it may be wiped and rebuilt.
NORMALISED_MARKER = ".paper_figures_normalised"


def normalise_tree(src: Path, dst: Path) -> Path:
    """Mirror `src` at `dst`, casting any Parquet whose schema is not `SCHEMA`.

    Hardlinks everything else (falls back to a copy across filesystems). Schema metadata
    (the bridge's source-log stamp) is preserved. Returns `dst`.

    Anything under a `superseded/` folder is left out: the canonical run folder keeps
    replaced cells there (e.g. debate before the adversary fix) under the SAME spec hashes,
    and `load_all` would otherwise pool them with the current cells.

    `dst` is WIPED first, so it must be a scratch dir: it may not overlap `src`, and an
    existing `dst` is removed only if an earlier call marked it (`NORMALISED_MARKER`), so a
    mistyped `--normalised-root` (a results tree, `$HOME`) is refused rather than deleted.
    """
    s, d = src.resolve(), dst.resolve()
    if s == d or s in d.parents or d in s.parents:
        raise SystemExit(f"--normalised-root {dst} overlaps --root {src}; use a scratch dir")
    if dst.exists():
        if any(dst.iterdir()) and not (dst / NORMALISED_MARKER).exists():
            raise SystemExit(
                f"{dst} is not empty and was not made by normalise_tree "
                f"(no {NORMALISED_MARKER}); refusing to delete it"
            )
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    (dst / NORMALISED_MARKER).write_text(f"normalised mirror of {s}\n")
    for p in sorted(src.rglob("*")):
        if SUPERSEDED in p.relative_to(src).parts:
            continue
        q = dst / p.relative_to(src)
        if p.is_dir():
            q.mkdir(parents=True, exist_ok=True)
            continue
        q.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".parquet" and not pq.read_schema(p).equals(SCHEMA):
            table = pq.read_table(p)
            md = table.schema.metadata
            pq.write_table(table.cast(SCHEMA).replace_schema_metadata(md), q)
            continue
        try:
            os.link(p, q)
        except OSError:
            shutil.copy2(p, q)
    return dst


#: A method row's organisms, restricted by family hash (results-folder suffix): method row ->
#: the families it keeps. A row not in the mapping keeps every organism of its method. The
#: paper's own restriction lives in `configs/dashboard_build.yaml` (`families:`).
Families = Mapping[str, Collection[str]]


#: The variant word of an adapter registered from a mobench user config (`_VARIANT_KEYS`).
EXTERNAL_VARIANT = "external"


def method_key(
    method: str,
    variant: str,
    family: str = "",
    families: Families | None = None,
    external: bool = False,
) -> str | None:
    """The method row an organism is drawn on, or None if no row shows it.

    `families` (method row -> family hashes, `Families`) restricts a row to those families:
    a retired organism can share method AND variant with a current one (few-shot framed k=15
    on old demos; normalised steering with `positions: all`), so only the family tells them
    apart. Without it every organism of a known method has a row.

    `external` (opt-in; the paper never sets it) draws an external adapter on a row of its
    own, keyed by its label (the `method`). Off, such a row is not shown, as before.
    """
    if external and EXTERNAL_VARIANT in variant.split() and _method_row(method, "") is None:
        return method
    key = _method_row(method, variant)
    if key is not None and families is not None and key in families:
        return key if family in families[key] else None
    return key


def _method_row(method: str, variant: str) -> str | None:
    if method == BASE_METHOD:
        return "base"
    parts = variant.split()
    if method == "steering":
        return "steering"
    direct = {
        "oct_behaviour": "oct",
        "sft_behaviour": "sft",
        "dpo_behaviour": "dpo",
        "distillation_seqkd": "seqkd",
    }
    if method in direct:
        return direct[method]
    if method == "prompting":
        if "positive" in parts:
            return "sysprompt"
        if "negative" in parts:
            return "antiprompt"
    if method == "few_shot_icl":
        return "icl"
    return None


def load_families(config: Path) -> dict[str, frozenset[str]]:
    """The `families:` restriction of a dashboard build config (`Families`); {} when absent."""
    import yaml

    raw = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    return {k: frozenset(v) for k, v in (raw.get("families") or {}).items()}


@dataclass
class PaperData:
    matrix: Matrix
    organism: dict[tuple[str, str], str]  # (method key, size) -> organism label
    excluded: list[str]  # organisms in the tree the paper does not show
    # axis -> {eval: [organisms whose cell collapsed]}: each listed organism's composite on
    # that axis omits the eval; every other organism keeps it (`organism_evals`)
    collapsed: dict[str, dict[str, list[str]]]
    # (organism, eval) cells dropped because they were scored on other items than base
    other_bank: list[tuple[str, str]] = field(default_factory=list)
    # organism -> [(item_id, present 0/1, seen by this organism)] for simple_behaviour_exhibit
    sbe_items: dict[str, list[tuple[str, float, bool]]] = field(default_factory=dict)
    # organism -> its `analysis.derived.sbe_verdicts` verdict (PASS | PASS_INDIST | FAIL |
    # UNSCORED); absent = no verdict (base, prompting, no SBE rows). Drives `flagged`.
    sbe_verdict: dict[str, str] = field(default_factory=dict)
    # organism -> its `:kept` cells recomputed one seed at a time (`share_kept_by_seed`);
    # read by `seed_spread`. Empty = no organism had a multi-seed robustness parent.
    kept_by_seed: dict[str, KeptBySeed] = field(default_factory=dict)
    # organism -> its full `analysis.derived.SbeVerdict` (the checks behind `sbe_verdict`)
    sbe_checks: dict[str, Any] = field(default_factory=dict)


def flagged(pd_: PaperData, organism: str | None) -> bool:
    """True when this organism's radar polygon must be drawn flagged (dashed outline).

    Its SBE verdict is not PASS / PASS_INDIST, or it has none: the trait whose share is kept
    was never shown to be there. Missing counts as flagged on purpose.
    """
    return organism is None or pd_.sbe_verdict.get(organism) not in SBE_OK


def self_axis(m: Matrix, axis: str) -> bool:
    """True when every composite eval on `axis` is self-referenced (robustness, since
    2026-10-01): its reading needs no base, and is not a deviation from one."""
    evals = m.evals_in(axis, composite_only=True)
    return bool(evals) and all(m.self_referenced(e) for e in evals)


def select(
    rows: Sequence[Row], families: Families | None = None, external: bool = False
) -> PaperData:
    """Keep the organisms a method row shows (`method_key`, with `families` and `external`),
    build the matrix over just those.

    Excluded organisms (e.g. `oct_demos`) are left out of the matrix entirely, so they do
    not move the cross-organism spread `axis_effect` standardises by.
    """
    keep: list[Row] = []
    organism: dict[tuple[str, str], str] = {}
    organism_family: dict[str, str] = {}
    excluded: set[str] = set()
    for r in rows:
        key = method_key(r.method, r.variant, r.family, families, external)
        if key is None:
            excluded.add(r.organism)
            continue
        # `model_tag`, not `_model_tag(base_model)`: a 27B's thinking-on and -off arms are
        # two organisms, not a collision on one paper row.
        size = r.model_tag
        prev = organism.setdefault((key, size), r.organism)
        organism_family.setdefault(r.organism, r.family)
        if prev != r.organism:
            raise ValueError(
                f"two organisms map to method row ({key}, {size}): {prev!r} and {r.organism!r} "
                f"(families {organism_family[prev]} and {r.family}); list the one to show under "
                f"`families: {{{key}: [<family hash>]}}` in the build config"
            )
        keep.append(r)
    keep, other_bank = _drop_other_bank(keep, organism)
    m = build_matrix(keep, item_axes=BASE_RELATIVE_AXES)
    from analysis.derived import sbe_verdicts, share_kept_by_seed

    checks = sbe_verdicts(keep)
    return PaperData(
        m,
        organism,
        sorted(excluded),
        _collapsed(m, organism, _awaiting_judge(keep)),
        other_bank,
        _sbe_items(keep),
        {org: v.verdict for org, v in checks.items()},
        share_kept_by_seed(keep),
        checks,
    )


def _sbe_items(rows: Sequence[Row]) -> dict[str, list[tuple[str, float, bool]]]:
    """Per organism, every scored simple_behaviour_exhibit answer: (item, present, seen).

    `present` is the judge label alone (`analysis.derived.RateReading.judge_present`): score
    >= 0.5, an ambiguous verdict (score 0.0) counts against, a judge failure (NaN) is left
    out. It does NOT apply the verdict's per-answer coherence condition
    (`analysis.derived.sbe_present`), so DPO reads higher here than in `sbe_verdicts`. `seen` is
    `analysis.derived.sbe_seen` (the organism trained on, or was prompted with, that item).
    """
    from analysis.derived import sbe_seen

    out: dict[str, list[tuple[str, float, bool]]] = {}
    for r in rows:
        if r.eval_name != SBE or not math.isfinite(r.score):
            continue
        out.setdefault(r.organism, []).append((r.cluster_id, float(r.score >= 0.5), sbe_seen(r)))
    return out


def _wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def newcombe_diff(k1: float, n1: int, k2: float, n2: int) -> tuple[float, float, float]:
    """p1 - p2 with Newcombe's (1998, method 10) hybrid-score 95% interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson(k1, n1)
    l2, u2 = _wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def expression_gain(pd_: PaperData, key: str, size: str) -> tuple[float, float, float, int] | None:
    """(gain, lo, hi, n) in percentage points: the organism's present rate on the items it did
    NOT see, minus the base model's rate on those same items; Newcombe 95% interval.

    Matches the paper's existing expression figure (fig5a): unseen questions only, base on
    the same questions, independent-proportions interval.
    """
    org, base = pd_.organism.get((key, size)), pd_.organism.get(("base", size))
    if org is None or base is None or org not in pd_.sbe_items:
        return None
    mine = [(i, p) for i, p, seen in pd_.sbe_items[org] if not seen]
    items = {i for i, _p in mine}
    theirs = [p for i, p, _s in pd_.sbe_items.get(base, []) if i in items]
    if not mine or not theirs:
        return None
    d, lo, hi = newcombe_diff(sum(p for _i, p in mine), len(mine), sum(theirs), len(theirs))
    return 100 * d, 100 * lo, 100 * hi, len(mine)


def _drop_other_bank(
    rows: list[Row], organism: dict[tuple[str, str], str]
) -> tuple[list[Row], list[tuple[str, str]]]:
    """Drop cells scored on different items than their same-size base.

    A cell rerun on a new bank version (the 2026-09-25 run: few-shot 32B downstream on the
    v2 bank, every base on v1) is not comparable with its base, and pooling its mean into an
    eval's cross-organism spread would move every other organism's standardised effect too.
    It is dropped before the matrix is built, so it reads as pending, and listed.
    A strict subset is left alone: that is a cell still landing, which `reading` marks
    pending as partial.
    """
    items: dict[tuple[str, str], set[str]] = {}
    for r in rows:
        items.setdefault((r.organism, r.eval_name), set()).add(r.cluster_id)
    base_of = {size: org for (key, size), org in organism.items() if key == "base"}
    size_of = {org: size for (_key, size), org in organism.items()}
    bad = set()
    for (org, ev), ids in items.items():
        base = base_of.get(size_of.get(org, ""))
        if base is None or base == org or (base, ev) not in items:
            continue
        if not ids <= items[(base, ev)]:
            bad.add((org, ev))
    return [r for r in rows if (r.organism, r.eval_name) not in bad], sorted(bad)


def _awaiting_judge(rows: Sequence[Row]) -> set[tuple[str, str]]:
    """(organism, eval) cells whose every row was generated with the judge deferred and
    never scored — a cell still in flight (the run's rejudge pass), not a collapsed one."""
    seen: dict[tuple[str, str], bool] = {}
    for r in rows:
        waiting = str(r.meta.get("defer_judge")) == "True" and not math.isfinite(r.score)
        key = (r.organism, r.eval_name)
        seen[key] = seen.get(key, True) and waiting
    return {k for k, v in seen.items() if v}


def _collapsed(
    m: Matrix, organism: dict[tuple[str, str], str], awaiting: set[tuple[str, str]]
) -> dict[str, dict[str, list[str]]]:
    """Composite evals that RAN for some shown organism but collapsed there.

    `Matrix.has_coverage` drops a cell that discarded most samples AND kept few
    (DPO's `robustness_multiturn_debate_gen` did, at 0-11 of 100 scored, while a garbage final
    turn was NaN; since PR #384 it scores 0.0 and only context-exhausted rows are NaN, so DPO's
    cell now holds 62-94 of 100 and votes). A collapse is structural, not pending, so the eval
    leaves THAT organism's composite only (`organism_evals`); every other organism keeps it,
    and each reading is still paired with its base over the same evals. Captions name who lost
    what (`_collapsed_phrase`). A cell that never ran, or whose base has no usable reading (an
    unjudged cell), is pending instead, never collapsed.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for (key, _size), org in sorted(organism.items()):
        if key == "base":
            continue
        base = m.base_for(org)
        if base is None:
            continue
        for axis in COVERAGE_AXES:
            for e in _base_measured(m, base, axis):
                c = m.get(org, e)
                if (org, e) in awaiting or (org, e.split(":", 1)[0]) in awaiting:
                    continue  # pending (or derived from a pending parent), not collapsed
                if c is not None and c.n_total > 0 and not m.has_coverage(org, e):
                    out.setdefault(axis, {}).setdefault(e, []).append(org)
    return out


@dataclass(frozen=True)
class Reading:
    """One (organism, axis) reading, or why there is none."""

    status: str  # "ok" | "pending"
    delta: float = math.nan  # mean signed delta vs base, eval units (Fig. 3, Fig. 4 right)
    effect: float = math.nan  # Matrix.axis_effect over the common eval set, SD units (radar)
    raw: float = math.nan  # Matrix.axis_score over the same evals (Fig. 4 left)
    n_evals: int = 0
    why: str = ""
    base_raw: float = math.nan  # the base's raw composite over THIS reading's evals
    omitted: tuple[str, ...] = ()  # evals left out because this organism's cell collapsed
    evals: tuple[str, ...] = ()  # the evals `raw` averages
    # 95% interval of `raw`, on a self-referenced axis only (robustness: its `:kept` cells'
    # bootstrap SEs combined as sqrt(sum se^2) / k, normal, clipped to [0, 1];
    # `Matrix.kept_composite_ci`). NaN on every base-relative axis, whose composite interval
    # the full-results tables build from the cells' SEMs the same way.
    ci: tuple[float, float] = (math.nan, math.nan)
    # Standard error behind `ci` (sqrt(sum se^2) / k over the voting cells). NaN where `ci` is.
    se: float = math.nan
    # The radar's behaviour-spoke inputs (`radar_value`): the organism's and the base's
    # composites over the MATCHED items only (scorable for both, per eval, before averaging;
    # `Matrix.matched_means`). Equal to `raw` / `base_raw` where the two sides scored the same
    # items, and on any axis that is not base-relative. NaN when an eval has no shared item.
    # Tables keep the absolute `raw`; only the radar reads these.
    gain_raw: float = math.nan
    gain_base_raw: float = math.nan


def _base_measured(m: Matrix, base: str, axis: str) -> list[str]:
    """Composite evals on `axis` with a usable (finite, not collapsed) base cell.

    A self-referenced eval (`Matrix.self_referenced`, the share-kept `:kept` cells) is in
    whatever the base's cell says: it is read against the organism's own baseline, so the
    base's share — noise, the base has no trait — must not decide whether it counts.
    """
    out = []
    for e in m.evals_in(axis, composite_only=True):
        if m.self_referenced(e):
            out.append(e)
            continue
        c = m.get(base, e)
        if c is not None and math.isfinite(c.mean) and m.has_coverage(base, e):
            out.append(e)
    return out


def axis_evals(pd_: PaperData, base: str, axis: str) -> list[str]:
    """The evals an axis reading at this base's size can average: base-measured (the base
    reading's own set). An organism whose cell collapsed averages fewer (`organism_evals`)."""
    return _base_measured(pd_.matrix, base, axis)


def _collapsed_for(pd_: PaperData, organism: str, axis: str) -> list[str]:
    """Evals on `axis` whose cell collapsed for THIS organism (never another's)."""
    return [e for e, orgs in pd_.collapsed.get(axis, {}).items() if organism in orgs]


def organism_evals(pd_: PaperData, organism: str, base: str, axis: str) -> list[str]:
    """`axis_evals` minus the evals that collapsed for this organism alone."""
    gone = set(_collapsed_for(pd_, organism, axis))
    return [e for e in axis_evals(pd_, base, axis) if e not in gone]


def reading(pd_: PaperData, organism: str | None, axis: str) -> Reading:
    """The axis composite for one organism, only if it covers every eval in `organism_evals`.

    `effect` is `Matrix.axis_effect` and `raw` `Matrix.axis_score`, both restricted to
    `organism_evals`: the base-measured evals minus any whose cell collapsed for this
    organism (listed in `omitted`). Computed explicitly here so that a partial or pending
    cell can never slip through as a mean over fewer evals. `delta`/`effect` pair organism
    and base eval by eval, and `base_raw` is the base over the same evals, so every
    base-relative number compares like with like.
    """
    m = pd_.matrix
    if organism is None:
        return Reading("pending", why="organism not in results")
    base = m.base_for(organism)
    # A self-referenced eval (share kept) reads against 0, not base: `delta` is its value,
    # `base_raw` skips it where the base's share is undefined. See `_base_measured`. An axis
    # whose every voter is self-referenced (robustness) therefore reads with no base at all.
    composite = m.evals_in(axis, composite_only=True)
    self_only = bool(composite) and all(m.self_referenced(e) for e in composite)
    if base is None and not self_only:
        return Reading("pending", why="no base at this size")
    omitted = tuple(_collapsed_for(pd_, organism, axis))
    evals = (
        organism_evals(pd_, organism, base, axis)
        if base is not None
        else [e for e in composite if e not in omitted]
    )
    key = next((k for (k, _s), o in pd_.organism.items() if o == organism), None)
    evals = [e for e in evals if (key, e) not in NOT_RUN_BY_DESIGN]
    # A base-relative composite is never a partial one: `_base_measured` drops an eval the base
    # lacks, which would let the gain average over the remaining evals alone. An eval that votes
    # on the axis (and is not skipped by design for this organism) but has no base cell makes
    # the reading pending, as the activations voter guard does.
    if axis in BASE_RELATIVE_AXES and base is not None and evals:
        measured = set(_base_measured(m, base, axis))
        no_base = [
            e
            for e in m.evals_in(axis, composite_only=True)
            if (key, e) not in NOT_RUN_BY_DESIGN and e not in omitted and e not in measured
        ]
        if no_base:
            return Reading("pending", why=f"base missing {', '.join(no_base)}", omitted=omitted)
    if not evals:
        why = (
            f"every eval collapsed: {', '.join(omitted)}"
            if omitted
            else "base not measured on this axis"
        )
        return Reading("pending", why=why, omitted=omitted)
    # Missing = no covered, finite cell. NOT a NaN `eval_effect`: that is also NaN when the
    # eval has zero spread across organisms (every one scored the same), where the organism
    # sits exactly on base and its effect is 0, not unknown.
    missing = [
        e
        for e in evals
        if not (m.has_coverage(organism, e) and math.isfinite(m.signed_delta(organism, e)))
    ]
    if missing:
        return Reading("pending", why=f"missing {', '.join(missing)}")
    effects = [v if math.isfinite(v := m.eval_effect(organism, e)) else 0.0 for e in evals]
    # A cell still landing is pending, not a reading: a third of a bank is not the bank.
    partial = [e for e in evals if _partial(m, organism, base, e)]
    if partial:
        return Reading("pending", why=f"partial {', '.join(partial)}")
    deltas = [m.signed_delta(organism, e) for e in evals]
    raws = [_cell_mean(m, organism, e) for e in evals]
    se, lo, hi = (
        m.kept_composite_ci(organism, evals) if self_only else (math.nan, math.nan, math.nan)
    )
    # The base over the same evals. A self-referenced eval whose base share is undefined is
    # skipped there (the base is no reference for it, `_base_measured`); every other eval
    # is kept as before, finite or not.
    base_raws = (
        [
            v
            for e in evals
            if math.isfinite(v := _cell_mean(m, base, e)) or not m.self_referenced(e)
        ]
        if base is not None
        else []
    )
    base_raw = statistics.fmean(base_raws) if base_raws else math.nan
    raw = statistics.fmean(raws)
    gain_raw, gain_base_raw = raw, base_raw
    if axis in BASE_RELATIVE_AXES and base is not None:
        gain_raw, gain_base_raw = _matched_composites(m, organism, base, evals)
    return Reading(
        "ok",
        delta=statistics.fmean(deltas),
        effect=statistics.fmean(effects),
        raw=raw,
        n_evals=len(evals),
        base_raw=base_raw,
        omitted=omitted,
        evals=tuple(evals),
        ci=(lo, hi),
        se=se,
        gain_raw=gain_raw,
        gain_base_raw=gain_base_raw,
    )


def _matched_composites(
    m: Matrix, organism: str, base: str, evals: Sequence[str]
) -> tuple[float, float]:
    """(organism, base) mean over `evals` of each eval's mean on the items scorable for BOTH.

    `aggregate` drops non-finite scores per side, so a cell mean can sit on different items
    for an organism and its base (introspection: a denial pair is unscorable when either arm
    is). A radar gain is a difference of the two, so each eval is first restricted to the
    shared items (`Matrix.matched_means`). An eval whose item scores were not kept falls back
    to its cell means; one with no shared item makes the composite NaN (pending, never a
    mean over what happened to overlap nothing).
    """
    got, ref = [], []
    for e in evals:
        mm = m.matched_means(organism, base, e)
        if mm is None:
            got.append(_cell_mean(m, organism, e))
            ref.append(_cell_mean(m, base, e))
        else:
            got.append(mm[0])
            ref.append(mm[1])
    return statistics.fmean(got), statistics.fmean(ref)


def base_reading(pd_: PaperData, size: str, axis: str) -> Reading:
    """The base's own raw composite over `axis_evals` (delta and effect are 0 by definition).

    On a self-referenced eval (share kept) the base reads its OWN share, like any organism,
    over the cells where it is defined — never used as anyone's reference and NOT drawn: the
    radars, the coverage table and the dashboard show no base robustness (`fig1_scores`;
    `BASE_NO_ROBUSTNESS_WHY`). A self-referenced axis with no defined base share is pending
    for the base alone.
    """
    base = pd_.organism.get(("base", size))
    if base is None:
        return Reading("pending", why="no base")
    m = pd_.matrix
    evals = [
        e
        for e in axis_evals(pd_, base, axis)
        if not m.self_referenced(e)
        or (m.has_coverage(base, e) and math.isfinite(_cell_mean(m, base, e)))
    ]
    if not evals:
        return Reading("pending", why="base not measured on this axis")
    raw = statistics.fmean(_cell_mean(m, base, e) for e in evals)
    return Reading(
        "ok",
        delta=0.0,
        effect=0.0,
        raw=raw,
        n_evals=len(evals),
        base_raw=raw,
        evals=tuple(evals),
        gain_raw=raw,
        gain_base_raw=raw,
    )


@dataclass(frozen=True)
class SeedSpread:
    """A self-referenced composite recomputed once per seed, and its SD across seeds.

    Covers only what CAN vary: `fixed` lists the composite's cells with no multi-seed parent
    (drift, drift_long: one run each), identical in every seed. Critique and debate are one
    seed too but vary through their multi-seed open-ended baseline. A seed whose recompute
    leaves any of the reading's cells undefined is dropped (`dropped`), not averaged over
    fewer cells, so every per-seed composite averages the same cells.
    """

    per_seed: dict[int, float]
    fixed: tuple[str, ...]
    dropped: tuple[int, ...] = ()

    @property
    def sd(self) -> float:
        vals = list(self.per_seed.values())
        return statistics.stdev(vals) if len(vals) > 1 else math.nan


def seed_spread(
    pd_: PaperData, organism: str | None, axis: str = "robustness"
) -> SeedSpread | None:
    """Seed-to-seed spread of a share-kept composite over the evals its reading averages.

    None when the reading is not ok, the axis is not self-referenced, or the organism has no
    multi-seed parent (`PaperData.kept_by_seed`).
    """
    if organism is None or not self_axis(pd_.matrix, axis):
        return None
    r = reading(pd_, organism, axis)
    kb = pd_.kept_by_seed.get(organism)
    if r.status != "ok" or kb is None:
        return None
    per_seed: dict[int, float] = {}
    dropped = []
    for seed, cells in sorted(kb.by_seed.items()):
        vals = [cells.get(e, math.nan) for e in r.evals]
        if all(math.isfinite(v) for v in vals):
            per_seed[seed] = statistics.fmean(vals)
        else:
            dropped.append(seed)
    return SeedSpread(per_seed, tuple(kb.fixed(r.evals)), tuple(dropped))


def sbe_cell(m: Matrix, organism: str | None, eval_name: str = SBE) -> MatrixCell | None:
    """The expression cell for one organism, or None if it is not in the results."""
    return None if organism is None else m.get(organism, eval_name)


def _mean_or_nan(cell: MatrixCell | None) -> float:
    return cell.mean if cell is not None else math.nan


def _n_total(m: Matrix, organism: str, eval_name: str) -> int:
    """Samples the cell ran on. A share-kept cell is one row, so its parents' rows count:
    a pressure or baseline side still landing makes the cell partial, like any other."""
    cell = m.get(organism, eval_name)
    if cell is not None and cell.kept is not None:
        return cell.kept.n_baseline + cell.kept.n_pressure
    return cell.n_total if cell is not None else 0


def _n_total(m: Matrix, organism: str, eval_name: str) -> int:
    """Samples the cell ran on. A share-kept cell is one row, so its parents' rows count:
    a pressure or baseline side still landing makes the cell partial, like any other."""
    cell = m.get(organism, eval_name)
    if cell is not None and cell.kept is not None:
        return cell.kept.n_baseline + cell.kept.n_pressure
    return cell.n_total if cell is not None else 0


def _partial(m: Matrix, organism: str, base: str | None, eval_name: str) -> bool:
    """True when `organism`'s cell lacks a (seed, item) its base's cell has.

    COVERAGE, not a row count. The rule used to be `n_total(organism) < n_total(base)`, and
    a count moves for reasons that are not coverage: a judge pass that skipped a few rows on
    one side (the retired `min_words` filter took a handful of different rows from each
    organism, which put Steering@14B and the negative prompt@32B off the coherence spoke
    with a complete cell), or an extra epoch. What "still landing" means is that the base
    was read on an item, in a seed, that this organism has not reached
    (`MatrixCell.coverage_missing_from`; a row counts whether or not it was scorable, as it
    did in `n_total`).

    Per (seed, item), not the union of item ids plus a seed count: an item missing in ONE
    seed is a partial cell that the union hides behind the other seeds (adversarial review,
    finding 5). Everything the old count rule caught for a missing row is still caught.

    A cell scored on items the base does NOT have is a different bank, which
    `_drop_other_bank` removes before the matrix is built.

    A share-kept (`:kept`) cell is ONE organism-level row, so its coverage says nothing about
    its bank: it is partial when its parents' rows (`_n_total`) number fewer than the
    base's. With no base at this size (an all-self-referenced axis reads without one) there
    is nothing to be partial against.
    """
    if base is None:
        return False
    got, ref = m.get(organism, eval_name), m.get(base, eval_name)
    if ref is None:
        return False
    if got is None:
        return True
    if got.kept is not None or ref.kept is not None:
        return _n_total(m, organism, eval_name) < _n_total(m, base, eval_name)
    return bool(ref.coverage_missing_from(got))


def _cell_mean(m: Matrix, organism: str, eval_name: str) -> float:
    return _mean_or_nan(m.get(organism, eval_name))


# ---------------------------------------------------------------- coverage


def coverage(pd_: PaperData, eval_name: str = SBE) -> tuple[str, dict[str, list[str]]]:
    """Markdown table of what is in: one row per method, one column per axis x size.

    Returns (table, pending) where `pending` maps an axis (or "method:<key>") to the
    (method, size) cells missing there — the source of every caption's "pending" list.
    """
    m = pd_.matrix
    keys = ["base", *(s.key for s in METHODS), *(s.key for s in CONTROLS)]
    header = "| method | " + " | ".join(a for a in COVERAGE_AXES) + " |"
    lines = [header, "|" + "---|" * (len(COVERAGE_AXES) + 1)]
    pending: dict[str, list[str]] = {}
    for key in keys:
        cells = []
        for axis in COVERAGE_AXES:
            if key == "base" and axis != "expression" and self_axis(m, axis):
                cells.append("n/a")  # no reading by design (`BASE_NO_ROBUSTNESS_WHY`), not pending
                continue
            marks = []
            for size in SIZES:
                org = pd_.organism.get((key, size))
                if axis == "expression":
                    c = sbe_cell(m, org, eval_name)
                    ok = c is not None and math.isfinite(c.mean)
                elif key == "base":
                    ok = base_reading(pd_, size, axis).status == "ok"
                else:
                    ok = reading(pd_, org, axis).status == "ok"
                marks.append(size if ok else "-")
                if not ok:
                    pending.setdefault(axis, []).append(f"{key}@{size}")
            cells.append("/".join(marks))
        lines.append(f"| {STYLE[key].label} | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append(
        "Cell = sizes with a full-coverage reading (7B/14B/32B); '-' = pending; "
        f"n/a = {BASE_NO_ROBUSTNESS_WHY}."
    )
    if pd_.excluded:
        lines.append(f"Not shown in the paper figures: {', '.join(pd_.excluded)}")
    for axis, evals in sorted(pd_.collapsed.items()):
        for e, orgs in sorted(evals.items()):
            lines.append(
                f"Dropped from {axis} for these organisms only (collapsed cell): {e} <- "
                + ", ".join(orgs)
            )
    for org, e in pd_.other_bank:
        lines.append(f"Pending (scored on other items than its base, not compared): {org} / {e}")
    if marked := _flagged_cells(pd_):
        lines.append(
            "Flagged (dashed outline; SBE verdict not PASS/PASS_INDIST): "
            + ", ".join(f"{STYLE[k].label}@{z} ({v})" for k, z, v in marked)
        )
    lines += _kept_uncertainty_lines(pd_, [k for k in keys if k != "base"])
    for axis in COVERAGE_AXES:
        for size, evals in unbased_voters(pd_, axis).items():
            lines.append(
                f"Dropped from {axis} at {size} for EVERY organism (the base has no usable "
                f"reading of it there, so nothing is compared on it): {', '.join(evals)}"
            )
    return "\n".join(lines), pending


def _kept_uncertainty_lines(pd_: PaperData, keys: Sequence[str]) -> list[str]:
    """Robustness composite per method x size: value [95% CI], seed SD, drift clusters.

    Pass the treated keys only: the base has no trait to keep, so its share is noise.
    """
    out = []
    for key in keys:
        for size in SIZES:
            org = pd_.organism.get((key, size))
            r = reading(pd_, org, "robustness")
            if org is None or r.status != "ok":
                continue
            sp = seed_spread(pd_, org)
            spread = (
                f"seed SD {sp.sd:.3f} over {len(sp.per_seed)} seeds"
                + (f", {', '.join(sp.fixed)} held at one run" if sp.fixed else "")
                + (f", seeds {sp.dropped} dropped (a cell undefined)" if sp.dropped else "")
                if sp is not None and math.isfinite(sp.sd)
                else "no seed SD"
            )
            ci = (
                f"[{r.ci[0]:.3f}, {r.ci[1]:.3f}]"
                if math.isfinite(r.ci[0])
                else "[no CI: a voting cell has no bootstrap SE]"
            )
            drift = [
                f"{e.removeprefix('persistence_multiturn_').removesuffix(':kept')} rests on "
                f"{c.kept.n_clusters} conversations"
                for e in r.evals
                if e.startswith("persistence_multiturn_drift")
                and (c := pd_.matrix.get(org, e)) is not None
                and c.kept is not None
            ]
            note = f"; {', '.join(drift)}" if drift else ""
            out.append(f"Robustness {STYLE[key].label}@{size}: {r.raw:.3f} {ci} ({spread}{note})")
    if out:
        out.insert(
            0,
            "Robustness composite, 95% CI = value +/- 1.96 sqrt(sum se^2)/k over the voting "
            "`:kept` cells' bootstrap SEs, clipped to [0, 1]; seed SD covers only the "
            "multi-seed parents (one-seed cells held fixed):",
        )
    return out


def _flagged_cells(pd_: PaperData) -> list[tuple[str, str, str]]:
    """(method key, size, verdict or 'no verdict') for every shown organism drawn flagged."""
    out = []
    for key in ["base", *(s.key for s in METHODS), *(s.key for s in CONTROLS)]:
        for size in SIZES:
            org = pd_.organism.get((key, size))
            if org is not None and flagged(pd_, org):
                out.append((key, size, pd_.sbe_verdict.get(org, "no verdict")))
    return out


def _flagged_phrase(pd_: PaperData) -> str:
    """Caption sentence naming the implant methods drawn with a dashed outline."""
    marked = [(k, z, v) for k, z, v in _flagged_cells(pd_) if k in {s.key for s in METHODS}]
    if not marked:
        return ""
    by_key: dict[str, list[str]] = {}
    for k, z, v in marked:
        by_key.setdefault(k, []).append(f"{z}, {v.replace('_', ' ')}")
    who = "; ".join(f"{STYLE[k].label} ({'; '.join(zs)})" for k, zs in by_key.items())
    return (
        "A dashed outline marks an organism whose implanted behavior fell below the "
        "expression threshold (\\texttt{simple\\_behaviour\\_exhibit}), so its profile "
        f"describes a trait not shown to be there: {who}."
    )


def _collapsed_phrase(pd_: PaperData, axes: Iterable[str]) -> str:
    """Caption sentence naming evals dropped from an axis because a cell collapsed.

    Names the size of every dropped cell, e.g. "for DPO at 14B": a collapse is per organism,
    so a method that lost an eval at one size still has it at the others.
    """
    rank = {z: i for i, z in enumerate(SIZES)}

    def by_size(z: str) -> tuple[int, str]:
        return (rank.get(z, len(rank)), z)

    parts = []
    for axis in axes:
        for e, orgs in sorted(pd_.collapsed.get(axis, {}).items()):
            sizes: dict[str, list[str]] = {}
            for (k, z), o in pd_.organism.items():
                if o in orgs:
                    sizes.setdefault(k, []).append(z)
            who = "; ".join(
                f"{STYLE[k].label} at {', '.join(sorted(sizes[k], key=by_size))}"
                for k in sorted(sizes, key=list(STYLE).index)
            )
            why = (
                "no usable no-pressure baseline, so no share kept"
                if pd_.matrix.self_referenced(e)
                else "unscoreable: under 30 of its samples could be scored"
            )
            parts.append(
                f"{axis.capitalize()} excludes \\texttt{{{_tex(e)}}} for {who} ({why}); "
                f"every other organism's {axis} includes it"
            )
    return ". ".join(parts) + ("." if parts else "")


def _axis_set_phrase(pd_: PaperData, axis: str) -> str:
    """'Coherence averages \\texttt{transitivity}.' — the evals an axis actually uses."""
    used = sorted(
        {
            e
            for size in SIZES
            if (b := pd_.organism.get(("base", size)))
            for e in axis_evals(pd_, b, axis)
        }
    )
    if not used:
        return ""
    names = ", ".join(f"\\texttt{{{_tex(e)}}}" for e in used)
    return f"{axis.capitalize()} averages {names}."


def unbased_voters(pd_: PaperData, axis: str) -> dict[str, list[str]]:
    """``{size: [eval, ...]}``: composite evals on `axis` dropped at a size for want of a base.

    THE RULE (module docstring): an axis reading averages the evals the same-size BASE has a
    usable cell for. An eval the base lacks is therefore left out for EVERY organism at that
    size — all of them are read over the same voters, none on fewer than its neighbour — and
    `coverage()` prints each such eval, so the drop is never silent. (The captions name only
    case (b), as they always have: `_size_set_phrase`.) An eval counts as dropped at a size
    when the base there has no usable cell and either (a) some shown organism at that size
    does have one (e.g. a judge pass that reached organisms before their base; the
    persistence `:hold` cells, which a base that never shows the trait cannot be read on), or
    (b) the base at another size does (e.g. `apps`, not run at 7B).
    """
    m = pd_.matrix
    used = {
        s: set(axis_evals(pd_, b, axis)) for s in SIZES if (b := pd_.organism.get(("base", s)))
    }
    elsewhere = set().union(*used.values()) if used else set()
    out: dict[str, list[str]] = {}
    for size, have in used.items():
        here = {
            e
            for e in m.evals_in(axis, composite_only=True)
            for (_key, s), org in pd_.organism.items()
            if s == size and (c := m.get(org, e)) is not None and math.isfinite(c.mean)
        }
        if have and (missing := sorted((here | elsewhere) - have)):
            out[size] = missing
    return out


def _size_set_phrase(pd_: PaperData, axes: Iterable[str]) -> str:
    """Name evals an axis uses at some sizes but not others (e.g. `apps`, not run at 7B)."""
    notes = []
    for axis in axes:
        per = {
            s: set(axis_evals(pd_, b, axis))
            for s in SIZES
            if (b := pd_.organism.get(("base", s))) and axis_evals(pd_, b, axis)
        }
        if len(per) < 2:
            continue
        union = set().union(*per.values())
        for s, used in per.items():
            if missing := sorted(union - used):
                names = ", ".join(f"\\texttt{{{_tex(e)}}}" for e in missing)
                notes.append(f"At {s}, {axis} omits {names} (no base reading at that size)")
    return ". ".join(notes) + ("." if notes else "")


def _tex(s: str) -> str:
    return s.replace("_", "\\_")


def _pending_phrase(pending: dict[str, list[str]], axes: Iterable[str]) -> str:
    """Caption list of what is pending among `axes`, grouped by axis then size.

    E.g. "generality at 32B (all methods) and 7B (OCT (ours), SFT); robustness ...".
    A method with nothing measured at all is named once, not per axis.
    """
    keys = [s.key for s in METHODS]
    absent = [
        k for k in keys if all(f"{k}@{s}" in pending.get(a, []) for a in axes for s in SIZES)
    ]
    parts = []
    for axis in axes:
        cells = [c for c in pending.get(axis, []) if not c.startswith("base@")]
        by_size = []
        for size in SIZES:
            ks = [k for k in keys if f"{k}@{size}" in cells and k not in absent]
            if not ks:
                continue
            who = "all methods" if len(ks) == len(keys) else ", ".join(STYLE[k].label for k in ks)
            by_size.append(f"{size} ({who})")
        if not by_size:
            continue
        if all("(all methods)" in b for b in by_size) and len(by_size) == len(SIZES):
            parts.append(axis)
        else:
            parts.append(f"{axis} at " + ", ".join(by_size))
    parts += [f"{STYLE[k].label} (all axes)" for k in absent]
    return "; ".join(parts)


def _pending_items(pending: dict[str, list[str]], axes: Iterable[str]) -> list[tuple[str, str]]:
    """(label, detail) for what is pending among `axes`: whole axes first, then per method."""
    axes = list(axes)
    n_cells = len(METHODS) * len(SIZES)
    whole_axes = [
        a
        for a in axes
        if len([c for c in pending.get(a, []) if not c.startswith("base@")]) >= n_cells
    ]
    parts: list[tuple[str, str]] = [(a, "") for a in whole_axes]
    partial_methods = set()
    for a in axes:
        if a in whole_axes:
            continue
        for c in pending.get(a, []):
            key, size = c.split("@")
            partial_methods.add((key, size, a))
    by_method: dict[str, set[str]] = {}
    for key, size, a in partial_methods:
        by_method.setdefault(key, set()).add(f"{a}@{size}")
    open_axes = [a for a in axes if a not in whole_axes]
    for key, cells in sorted(by_method.items(), key=lambda kv: list(STYLE).index(kv[0])):
        # a size where every open axis is pending collapses to "at 14B"; the rest stay axis@size
        whole_sizes = [s for s in SIZES if all(f"{a}@{s}" in cells for a in open_axes)]
        rest = sorted(c for c in cells if c.split("@")[1] not in whole_sizes)
        if len(whole_sizes) == len(SIZES):
            parts.append((STYLE[key].label, ""))
            continue
        bits = ([f"at {', '.join(whole_sizes)}"] if whole_sizes else []) + (
            [", ".join(c.replace("@", " at ") for c in rest)] if rest else []
        )
        parts.append((STYLE[key].label, "; ".join(bits)))
    return parts


# ---------------------------------------------------------------- figure helpers


_STAMP_AT = {
    "top-right": (0.995, 0.995, "right", "top"),
    "bottom-left": (0.005, 0.01, "left", "bottom"),
    "bottom-center": (0.5, 0.01, "center", "bottom"),
}


def _stamp(fig: Figure, text: str | None, where: str = "top-right") -> None:
    """The draft marker: a red boxed 'preliminary: <pending>' in a corner kept free for it."""
    if text:
        x, y, ha, va = _STAMP_AT[where]
        fig.text(
            x,
            y,
            text,
            ha=ha,
            va=va,
            fontsize=6.5,
            color="#b03030",
            bbox={"boxstyle": "square,pad=0.25", "fc": "white", "ec": "#b03030", "lw": 0.6},
            zorder=10,
        )


def placeholder(
    width: float, height: float, message: str = "PLACEHOLDER \u2014 awaiting real data"
) -> Figure:
    """A visibly-placeholder page of the given size: hatched frame, big message, no axes."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes((0.01, 0.01, 0.98, 0.98))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.add_patch(
        mpatches.Rectangle(
            (0, 0), 1, 1, fill=False, hatch="////", edgecolor="#d8b0b0", linewidth=1.2
        )
    )
    ax.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        fontsize=11,
        fontfamily=["STIXGeneral", "DejaVu Serif"],
        color="#b03030",
        bbox={"boxstyle": "square,pad=0.6", "fc": "white", "ec": "#b03030", "lw": 1.0},
        wrap=True,
    )
    ax.axis("off")
    return fig


def _clean(ax, grid: str = "x") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def _repel(ys: list[float], min_gap: float, floor: float = -math.inf) -> list[float]:
    """Nudge label positions apart by at least `min_gap`, keeping their order.

    Labels stay at their own value where there is room: the pass runs top-down (a crowded
    label moves DOWN, never shifting isolated labels above it), then bottom-up from `floor`
    only if the stack ran off the bottom.
    """
    order = sorted(range(len(ys)), key=lambda i: ys[i], reverse=True)
    placed = [ys[i] for i in order]
    for i in range(1, len(placed)):
        placed[i] = min(placed[i], placed[i - 1] - min_gap)
    if placed and placed[-1] < floor:
        placed[-1] = floor
        for i in range(len(placed) - 2, -1, -1):
            placed[i] = max(placed[i], placed[i + 1] + min_gap)
    out = [0.0] * len(ys)
    for j, i in enumerate(order):
        out[i] = placed[j]
    return out


# ---------------------------------------------------------------- Fig. 1: radar


RADAR_LABELS = {
    "generality": "Generality",
    "robustness": "Robustness",
    "coherence": "Coherence",
    "capability": "Capability",
    "activations": "Activations",
    "introspection": "Introspection",
}


# Dashed: the whole outline of an organism that did not pass SBE (`flagged`). Dash and gap in
# POINTS on the page: matplotlib scales dash patterns by line width, so at the radars' thin
# lw (0.6-0.7) the old (3, 1.8) printed as ~1.8 pt on / 1.1 pt off and read as solid;
# (32B Prompting did not look dashed). `_outline` divides by lw.
FLAG_DASH_PT: tuple[float, float] = (4.0, 2.5)
FLAG_LINESTYLE: tuple = (0, FLAG_DASH_PT)


def _flag_linestyle(lw: float) -> tuple:
    """`FLAG_DASH_PT` as a linestyle for a line of width `lw`, so the printed dash is fixed."""
    s = max(lw, 1e-6) if matplotlib.rcParams["lines.scale_dashes"] else 1.0
    return (0, (FLAG_DASH_PT[0] / s, FLAG_DASH_PT[1] / s))


def _outline(
    ax: PolarAxes,
    theta: np.ndarray,
    values: Sequence[float] | np.ndarray,
    rmax: float,
    *,
    colour: str,
    lw: float,
    zorder: float,
    flagged: bool | None = None,
    ls: Any = "-",
    clip_on: bool = True,
) -> None:
    """A radar polygon's outline, NaN spokes skipped (the polygon joins its neighbours).

    Drawn in `ls`, or dashed (`FLAG_LINESTYLE`) all round when `flagged` is True: the ONLY
    mark of an organism that did not pass SBE, no vertex markers (2026-10-01). None or
    False draws the plain outline.
    """
    v = np.asarray(values, dtype=float)
    idx = np.flatnonzero(np.isfinite(v))
    th, rr = theta[idx], np.clip(v[idx], RADAR_MIN, rmax)
    ax.plot(
        np.append(th, th[0]),
        np.append(rr, rr[0]),
        color=colour,
        lw=lw,
        ls=_flag_linestyle(lw) if flagged else ls,
        zorder=zorder,
        clip_on=clip_on,
    )


# A behaviour spoke below the Base model is drawn at the centre (radius 0, the clip in
# `_outline` / the fills), with NO marker (2026-10-21: a circle at the centre cannot say
# which spoke is below base; the captions say so instead). Only the DRAWING clips: the signed
# value is untouched.


def radar_panel(
    ax: PolarAxes,
    values: Sequence[float],
    base: Sequence[float],
    colour: str,
    rmax: float,
    *,
    title: str | None = None,
    labels: Sequence[str] | None = None,
    ticks: Sequence[float] = (),
    tick_labels: Sequence[str] = (),
    label_size: float = 7.5,
    title_size: float = 8.5,
    show_base: bool = False,
    lw: float = 0.7,
    flagged: bool | None = None,
) -> None:
    """One method's filled translucent polygon against a dashed base outline.

    `values`/`base` are signed radii on RADAR_AXES (0 = the Base model on the behaviour spokes,
    the centre); a value below RADAR_MIN is drawn at the centre, unmarked. A NaN spoke (a pending
    cell) is skipped: the polygon joins its neighbours, and nothing is
    drawn at 0 for it. `flagged` (see `_outline`): True dashes the whole outline; None or
    False draws the plain outline.
    """
    n = len(RADAR_AXES)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(RADAR_MIN, rmax)
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(list(tick_labels), fontsize=label_size - 2, color=MUTED)
    ax.set_rlabel_position(360 / n / 2)
    ax.set_xticks(theta)
    ax.set_xticklabels(
        labels if labels is not None else [RADAR_LABELS[a] for a in RADAR_AXES],
        fontsize=label_size,
        color=INK,
    )
    ax.tick_params(axis="x", pad=1)
    ax.grid(color=RADAR_GRID, linestyle=(0, (1.5, 1.5)), linewidth=0.5)
    ax.spines["polar"].set_visible(False)
    for vals, is_base in ((base, True), (values, False)) if show_base else ((values, False),):
        v = np.asarray(vals, dtype=float)
        keep = np.isfinite(v)
        if keep.sum() < 3:
            continue
        th = np.append(theta[keep], theta[keep][0])
        rr = np.clip(np.append(v[keep], v[keep][0]), RADAR_MIN, rmax)
        if is_base:
            ax.plot(th, rr, color=BASE_STYLE.colour, ls=(0, (4, 2)), lw=1.0, zorder=4)
        else:
            ax.fill(th, rr, color=colour, alpha=0.28, zorder=2, lw=0)
            _outline(ax, theta, vals, rmax, colour=colour, lw=lw, zorder=3, flagged=flagged)
    if title:
        ax.set_title(title, fontsize=title_size, color=INK, pad=14, fontweight="bold")


# THE CLIP ON THE HEALTH SPOKES HIDES RISES ABOVE BASE (`paper_style.radar_value`: 10 x
# (1 + (organism - base)), clipped at 10). That was harmless while every coherence voter could only
# fall; with `mu_decisiveness:decisiveness` voting (higher is better, 2026-09-30) an
# organism can score ABOVE its base and two such organisms tie at 10. A display rule, left
# as it is; the unclipped values are in `axis_dots` (deltas) and the scorecard. On
# canonical_0925 the coherence spoke was clipped for three paper cells under the old ratio
# rule, all barely: few-shot 7B (ratio 1.0005), few-shot 32B (1.0140) and the negative prompt
# 32B (1.0023); capability for none. Not re-measured under the delta rule.
RADAR_SCALE_TEXT = (
    "Three rules, one per kind of spoke. Generality, introspection and activations are the "
    "gain over the Base model: $10\\times$ the organism's raw 0--1 composite minus the Base "
    "model's, on the items scorable for both, so the Base model is the centre (0) and a "
    "method that does worse than the Base model on a spoke is drawn at the centre. "
    "Coherence and capability are one plus the difference from the Base model: "
    "$10\\times(1+\\Delta)$, where $\\Delta$ is the organism's raw 0--1 composite minus the Base "
    "model's, clipped at 10, so 10 means no "
    "degradation relative to the Base model and damage moves inward linearly in $\\Delta$. "
    "Robustness is the share "
    "kept under pressure: $10\\times$ the mean, per pressure setting, of the organism's trait "
    "rate under pressure divided by its own rate without pressure on the same items, capped at "
    "1 (10 = all of its own trait kept); it is not read against the Base model, which has no "
    "trait to keep. Activations is the truth probe: a probe fit on each model's own "
    "activations of true/false statements is read on fixed pairs where one option is the "
    "behavior's side, and the score is the share of pairs on which that option reads as more "
    "true; the spoke averages, over world claims and claims about the model itself, the "
    "organism's share minus the Base model's, and needs both cells. Expression and "
    "misalignment are not on the radar."
)
RADAR_SHORT = ["Gen.", "Rob.", "Coh.", "Cap.", "Act.", "Intro."]


def radar(pd_: PaperData, size: str, *, stamp: str | None = None) -> Figure:
    """Small multiples: one radar per method against the dashed base outline.

    Same scale as Figure 1 (`fig1_scores`), centre RADAR_MIN (0): the Base model sits on the
    outer ring for coherence and capability (no degradation) and at the centre for the
    behaviour spokes (gain over base). Every panel shares the scale. A pending spoke (NaN) is
    skipped, never drawn as 0. Spoke names are abbreviated so neighbouring panels cannot
    collide; the caption spells them out.
    """
    scores = fig1_scores(pd_, size, [s.key for s in METHODS])
    if not any(math.isfinite(v) for k, vs in scores.items() if k != "base" for v in vs):
        return placeholder(TEXT_WIDTH_IN, RADAR_H_IN, "PLACEHOLDER — awaiting real data")
    ncols, nrows = 3, 3  # 3 per row: full spoke names need ~1.8in per panel
    fig = plt.figure(figsize=(TEXT_WIDTH_IN, RADAR_PER_METHOD_H_IN))
    w, h = 1 / ncols, 0.94 / nrows
    for i, s in enumerate(METHODS):
        r, c = divmod(i, ncols)
        ax = cast(
            PolarAxes,
            fig.add_axes(
                (c * w + 0.26 * w, 0.97 - (r + 1) * h + 0.18 * h, 0.48 * w, 0.64 * h),
                projection="polar",
            ),
        )
        radar_panel(
            ax,
            scores[s.key],
            scores["base"],
            s.colour,
            10,
            title=s.label,
            labels=RADAR_FULL,
            ticks=(0, 5, 10),
            tick_labels=("0", "5", "10"),
            label_size=5.8,
            title_size=9,
            flagged=flagged(pd_, pd_.organism.get((s.key, size))),
        )
    _stamp(fig, stamp, "bottom-center")
    return fig


# ---------------------------------------------------------------- Fig. 2: method x size bars

# A reference line over bars must sit above the bars (zorder 2), hatch (2.5) and error bars (3).
REFERENCE_ZORDER = 4.0
REFERENCE_COLOUR = "#b03a2e"
# NOT the expression threshold. The threshold (`derived.SBE_DELTA`, docs/evals/expression.md) is
# that the LOWER 95% bound of (organism - base), item by item, clears 0.10, so it depends on each
# organism's own base rate and no single absolute line can draw it. This constant is only a visual
# guide: the level base + 0.10 when the base rate is ~0, passed as `reference=` to the opt-in line
# of `expression_rate_bar`. The shipped paper figure draws no line. The dashboard's expression
# chart draws the per-organism base + delta instead.
EXPRESSION_DELTA_GUIDE = SBE_DELTA


def draw_reference_line(ax: Any, y: float) -> Line2D:
    """Dashed horizontal reference line drawn over every bar and error bar (shared by all bar
    figures). Same look as the original threshold line (thin, plain matplotlib dashes, no
    halo, 2026-10-20); only its layering changed, from behind the bars to above them."""
    return ax.axhline(
        y, color=REFERENCE_COLOUR, lw=1.0, ls="--", zorder=REFERENCE_ZORDER, clip_on=True
    )


# Size legend of the grouped bars (Fig. 3's expression bar): larger and more open than the first
# version (8 pt, tight handle-to-label gap), after a 2026-10-21 review nit.
BAR_LEGEND_SIZE = 9.0


# x anchors (figure fraction) of the family legends, by how many there are (mockup 2026-10-30).
_LEGEND_ANCHORS = {1: (0.54,), 2: (0.44, 0.76)}


def method_size_bars(
    pd_: PaperData,
    value: Callable[[str, str], tuple[float, float, float] | None],
    *,
    ylabel: str,
    ylim: tuple[float, float] = (0.0, 1.05),
    sizes: Sequence[str] = SIZES,
    stamp: str | None = None,
    missing_label: str = "n/a",
    hatched: Collection[tuple[str, str]] = (),
    reference: float | None = None,
    legends: Sequence[tuple[str, Sequence[str]]] | None = None,
    zero_label: str | None = None,
) -> Figure:
    """Grouped bars in the paper's Figure 2 style: methods on x (BAR_ORDER, Base model first),
    one bar per model size coloured by SIZE_COLOURS, legend "Qwen2.5-Instruct" on top.

    `value(key, size)` -> (mean, lo, hi) or None (pending: no bar, never a 0). The interval
    is drawn as an absolute [lo, hi] segment (a bootstrap interval need not contain the
    point). Horizontal tick labels, full text width, thick bars.

    Opt-ins (default: the Qwen-only figure, unchanged): `legends` is a list of (title, sizes),
    one legend per base-model family in place of the single "Qwen2.5-Instruct" one; `zero_label`
    writes a small label in the slot of a bar whose rate is exactly 0, so an empty slot does
    not read as missing data (a missing value keeps `missing_label`).
    """
    import pandas as pd

    keys = [k for k in BAR_ORDER if any(pd_.organism.get((k, s)) for s in sizes)]
    records, intervals = [], {}
    for key in keys:
        for size in sizes:
            v = value(key, size)
            mean = v[0] if v is not None else math.nan
            records.append({"method": STYLE[key].label, "size": size, "value": mean})
            if v is not None:
                intervals[(key, size)] = (v[1], v[2])
    df = pd.DataFrame.from_records(records)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, EXPR_H_IN))
    fig.subplots_adjust(left=0.1, right=0.995, top=0.8, bottom=0.11)
    group = 0.86
    _seaborn().barplot(
        data=df,
        x="method",
        y="value",
        hue="size",
        order=[STYLE[k].label for k in keys],
        hue_order=list(sizes),
        palette=SIZE_COLOURS,
        errorbar=None,
        width=group,
        saturation=1.0,
        legend=False,
        ax=ax,
        zorder=2,
    )
    # CIs from our own estimator (cluster-robust or bootstrap), not seaborn's: drawn as an
    # absolute [lo, hi] segment on each bar, since a bootstrap interval need not contain
    # the point. A missing value gets no bar and "n/a", never a 0.
    width = group / len(sizes)
    for si, size in enumerate(sizes):
        for mi, key in enumerate(keys):
            x = mi + (si - (len(sizes) - 1) / 2) * width
            if (key, size) not in intervals or not math.isfinite(
                df.loc[(df["method"] == STYLE[key].label) & (df["size"] == size), "value"].iloc[0]
            ):
                ax.text(
                    x,
                    ylim[0],
                    missing_label,
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=MUTED,
                    style="italic",
                )
                continue
            if (key, size) in hatched:  # a preliminary bar: white hatch over the size colour
                mean = df.loc[
                    (df["method"] == STYLE[key].label) & (df["size"] == size), "value"
                ].iloc[0]
                ax.bar(
                    x,
                    mean,
                    width,
                    facecolor="none",
                    edgecolor="white",
                    hatch="////",
                    linewidth=0,
                    zorder=2.5,
                )
            if zero_label is not None and (
                df.loc[(df["method"] == STYLE[key].label) & (df["size"] == size), "value"].iloc[0]
                == 0
            ):
                ax.text(
                    x,
                    ylim[0] + 0.012 * (ylim[1] - ylim[0]),
                    zero_label,
                    ha="center",
                    va="bottom",
                    fontsize=6,
                    color=MUTED,
                )
            lo, hi = intervals[(key, size)]
            if math.isfinite(lo) and math.isfinite(hi) and hi > lo:
                lo_c, hi_c = max(lo, ylim[0]), min(hi, ylim[1])
                cap = width * 0.18
                ax.vlines(x, lo_c, hi_c, color=INK, lw=0.7, zorder=3)
                ax.hlines([lo_c, hi_c], x - cap, x + cap, color=INK, lw=0.7, zorder=3)
    ax.set_xlabel("")
    ax.tick_params(axis="x", length=0, labelsize=8, rotation=0)
    ax.set_xlim(-0.55, len(keys) - 0.45)
    ax.set_ylim(*ylim)
    if reference is not None:
        draw_reference_line(ax, reference)
    ax.set_ylabel(ylabel)
    _clean(ax, "y")
    if legends is None:
        handles = [mpatches.Patch(facecolor=SIZE_COLOURS[s], label=s) for s in sizes]
        fig.legend(
            handles=handles,
            title="Qwen2.5-Instruct",
            loc="upper center",
            bbox_to_anchor=(0.54, 1.0),
            ncol=len(sizes),
            frameon=False,
            fontsize=BAR_LEGEND_SIZE,
            title_fontsize=BAR_LEGEND_SIZE,
            handlelength=1.6,
            handleheight=1.0,
            handletextpad=0.7,
            columnspacing=2.2,
            borderpad=0.3,
            labelspacing=0.5,
        )
    else:  # one legend per base-model family, side by side on the top row
        for (title, group), x_anchor in zip(legends, _LEGEND_ANCHORS[len(legends)], strict=True):
            fig.legend(
                handles=[mpatches.Patch(facecolor=SIZE_COLOURS[s], label=s) for s in group],
                title=title,
                loc="upper center",
                bbox_to_anchor=(x_anchor, 1.0),
                ncol=len(group),
                frameon=False,
                fontsize=8,
                title_fontsize=8,
                handlelength=1.4,
                columnspacing=1.4,
            )
    _stamp(fig, stamp, "bottom-left")
    return fig


def expression_rate_bar(
    pd_: PaperData,
    sizes: Sequence[str] = SIZES,
    *,
    stamp: str | None = None,
    reference: float | None = None,
) -> Figure:
    """Figure 2: expression rate (simple_behaviour_exhibit present rate, whole bank), 95% CI
    clustered by item and pooled over seeds (`MatrixCell.ci95`). `reference` draws a dashed
    guide line at that absolute rate over the bars (`EXPRESSION_DELTA_GUIDE` is base + the
    threshold margin only when the base rate is ~0; it is not the threshold itself, which is a
    lower bound on organism - base); the paper figure has none.

    A size outside `SIZES` (31B, Gemma-4) switches on a legend per base-model family
    (`SIZE_GROUPS`) and a "0" label in the slot of a bar at exactly 0; Qwen-only is unchanged."""

    def value(key: str, size: str) -> tuple[float, float, float] | None:
        c = sbe_cell(pd_.matrix, pd_.organism.get((key, size)))
        if c is None or not math.isfinite(c.mean):
            return None
        lo, hi = c.ci95
        return c.mean, lo, hi

    extended = any(z not in SIZES for z in sizes)
    groups = [(t, [z for z in g if z in sizes]) for t, g in SIZE_GROUPS]
    return method_size_bars(
        pd_,
        value,
        ylabel="Expression Rate",
        sizes=sizes,
        stamp=stamp,
        reference=reference,
        legends=[(t, g) for t, g in groups if g] if extended else None,
        zero_label="0" if extended else None,
    )


def expression_rate_bar_prelim(
    pd_: PaperData,
    overrides: dict[tuple[str, str], tuple[float, int] | None],
    *,
    stamp: str | None = None,
) -> Figure:
    """Fig. 2 with some (method, size) bars replaced by a preliminary reading.

    `overrides[(key, size)]` is (present rate, n answers) for a judged subset, drawn hatched
    with a 95% Wilson interval, or None to draw no bar and a "pending" marker. Every other bar
    is the standard `expression_rate_bar` value.
    """

    def value(key: str, size: str) -> tuple[float, float, float] | None:
        if (key, size) in overrides:
            ov = overrides[(key, size)]
            if ov is None:
                return None
            rate, n = ov
            lo, hi = _wilson(rate * n, n)
            return rate, lo, hi
        c = sbe_cell(pd_.matrix, pd_.organism.get((key, size)))
        if c is None or not math.isfinite(c.mean):
            return None
        lo, hi = c.ci95
        return c.mean, lo, hi

    return method_size_bars(
        pd_,
        value,
        ylabel="Expression Rate",
        stamp=stamp,
        missing_label="pending",
        hatched=[k for k, v in overrides.items() if v is not None],
    )


def mu_bar(
    pd_: PaperData,
    intervals: dict[str, tuple[float, float]] | None,
    *,
    stamp: str | None = None,
) -> Figure:
    """mu decisiveness per method and size, same layout as Fig. 2; bootstrap intervals."""

    def value(key: str, size: str) -> tuple[float, float, float] | None:
        org = pd_.organism.get((key, size))
        c = sbe_cell(pd_.matrix, org, MU_EVAL)
        if c is None or not math.isfinite(c.mean):
            return None
        lo, hi = (intervals or {}).get(org or "", (math.nan, math.nan))
        return c.mean, lo, hi

    return method_size_bars(pd_, value, ylabel="Decisiveness", ylim=(0.0, 1.0), stamp=stamp)


# ---------------------------------------------------------------- Fig. 3: axis dot plot


def axis_dots(pd_: PaperData, *, stamp: str | None = None, sizes: Sequence[str] = SIZES) -> Figure:
    """Figure 3, as a seaborn FacetGrid: one column per axis, one row per method.

    Each point is a method's deviation from the Base model at one size (7B circle, 14B square,
    32B triangle, in SIZE_COLOURS), joined by a grey range line; the Prompting row is
    highlighted. Axis names centred over each column; no per-panel subtitles (rule 8).
    A self-referenced axis (robustness: share kept, `self_axis`) plots the share itself on a
    0-1 range with no base line: it is not a deviation from base.
    """
    import pandas as pd

    sns = _seaborn()
    own = {name for a, name, _q in DOT_AXES if self_axis(pd_.matrix, a)}
    rows = [STYLE[k] for k in ROW_ORDER]
    labels = [s.label for s in rows]
    records = []
    for axis, name, _question in DOT_AXES:
        for s in rows:
            for size in sizes:
                r = reading(pd_, pd_.organism.get((s.key, size)), axis)
                if r.status == "ok":
                    records.append(
                        {"axis": name, "method": s.label, "size": size, "delta": r.delta}
                    )
    df = pd.DataFrame.from_records(records)
    cols = [name for _a, name, _q in DOT_AXES]
    height = DOTS_H_IN * 0.7
    g = sns.FacetGrid(
        df,
        col="axis",
        col_order=cols,
        sharex=False,
        sharey=True,
        height=height,
        aspect=(TEXT_WIDTH_IN * 0.87 / len(cols)) / height,
        despine=True,
    )

    def draw(data, **_kw):
        ax = plt.gca()
        yidx = {lab: len(labels) - 1 - i for i, lab in enumerate(labels)}
        hl = next(s.label for s in rows if s.highlight)
        ax.axhspan(yidx[hl] - 0.5, yidx[hl] + 0.5, color=HIGHLIGHT_BAND, zorder=0, lw=0)
        for lab, part in data.groupby("method"):
            if len(part) > 1:
                ax.plot(
                    [part["delta"].min(), part["delta"].max()],
                    [yidx[lab]] * 2,
                    color=GRID,
                    lw=2.4,
                    zorder=1,
                    solid_capstyle="round",
                )
        sns.scatterplot(
            data=data.assign(y=data["method"].map(yidx)),
            x="delta",
            y="y",
            hue="size",
            style="size",
            hue_order=list(sizes),
            style_order=list(sizes),
            palette=SIZE_COLOURS,
            markers=SIZE_MARKERS,
            s=22,
            edgecolor="white",
            linewidth=0.5,
            legend=False,
            zorder=3,
            ax=ax,
        )
        if set(data["axis"]) <= own:
            ax.set_xlim(-0.05, 1.05)  # a share: its own 0-1 range, no base line
        else:
            lo, hi = min(data["delta"].min(), 0.0), max(data["delta"].max(), 0.0)
            pad = 0.12 * (hi - lo or 1e-3)
            ax.set_xlim(lo - pad, hi + pad)
            ax.axvline(0, color=BASE_STYLE.colour, ls=BASE_STYLE.linestyle, lw=0.9, zorder=2)
        ax.xaxis.set_major_locator(MaxNLocator(3))
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)

    g.map_dataframe(draw)
    g.set_titles(col_template="{col_name}", size=8.5, fontweight="bold")
    for ax in g.axes.flat:
        ax.title.set_horizontalalignment("center")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="x", labelsize=6.5)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    ax0 = g.axes.flat[0]
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels[::-1], color=INK, fontsize=7.5)
    for lab in ax0.get_yticklabels():
        if lab.get_text() == next(s.label for s in rows if s.highlight):
            lab.set_fontweight("bold")
            lab.set_color(SIZE_COLOURS["7B"])
    ax0.set_ylim(-0.6, len(labels) - 0.4)
    fig = g.figure
    fig.set_size_inches(TEXT_WIDTH_IN, DOTS_H_IN)
    fig.subplots_adjust(left=0.135, right=0.99, top=0.9, bottom=0.27, wspace=0.14)
    fig.text(
        0.56,
        0.115,
        r"Deviation from base model (eval units; dashed line $=$ base)",
        ha="center",
        va="bottom",
        fontsize=8,
        color=INK,
    )
    handles = [
        Line2D(
            [], [], marker=SIZE_MARKERS[s], ls="none", color=SIZE_COLOURS[s], markersize=5, label=s
        )
        for s in sizes
    ]
    fig.legend(
        handles=handles,
        title="Base model size",
        loc="lower center",
        ncol=len(sizes),
        frameon=False,
        bbox_to_anchor=(0.56, -0.02),
        fontsize=7.5,
        title_fontsize=7.5,
        handletextpad=0.3,
        columnspacing=1.6,
    )
    _stamp(fig, stamp, "top-right")
    return fig


# ------------------------------------------------------------- prompting vs the best other method

# Panel order of `prompting_vs_best` (2 x 3), and the methods the "best other" is chosen from.
PVB_AXES = ("generality", "robustness", "activations", "introspection", "coherence", "capability")
PVB_OTHERS = ("oct", "sft", "dpo", "steering", "seqkd", "icl")
# An arrow from the prompting marker to the best-other marker is drawn only when the two differ
# by more than this many radar units (0-10 scale; 1.5 = 15 points of a 0-1 composite). Below
# it the markers overlap and an arrow would be a stub.
PVB_ARROW_EPS = 1.5


def prompting_vs_best_points(
    pd_: PaperData, sizes: Sequence[str] = SIZES
) -> dict[tuple[str, str], dict[str, Any]]:
    """(axis, size) -> system prompting vs the best other method, all on the radar scale.

    `prompt` / `best` are `fig1_scores` values (centre = base radar scale, `radar_value`), SIGNED;
    `*_drawn` clip at `RADAR_MIN`, as the radars draw them. `best_key` is the method with the
    highest signed value among `PVB_OTHERS`. `prompt_flagged` / `best_flagged`: that organism is
    below the expression threshold (`flagged`). `ratio` is prompting / best on the drawn values,
    NaN when the best is at the centre. A (axis, size) with no reading for prompting or for any
    other method is absent.
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for size in sizes:
        scores = fig1_scores(pd_, size, ["sysprompt", *PVB_OTHERS])
        for axis in PVB_AXES:
            i = RADAR_AXES.index(axis)
            p = scores["sysprompt"][i]
            others = {k: scores[k][i] for k in PVB_OTHERS if math.isfinite(scores[k][i])}
            if not math.isfinite(p) or not others:
                continue
            best_key = max(others, key=lambda k: others[k])
            b = others[best_key]
            p_drawn, b_drawn = max(p, RADAR_MIN), max(b, RADAR_MIN)
            out[(axis, size)] = {
                "prompt": p,
                "best": b,
                "best_key": best_key,
                "prompt_drawn": p_drawn,
                "best_drawn": b_drawn,
                "prompt_flagged": flagged(pd_, pd_.organism.get(("sysprompt", size))),
                "best_flagged": flagged(pd_, pd_.organism.get((best_key, size))),
                "ratio": p_drawn / b_drawn if b_drawn > 0 else math.nan,
                "arrow": abs(b_drawn - p_drawn) > PVB_ARROW_EPS,
            }
    return out


def prompting_vs_best(
    pd_: PaperData, *, stamp: str | None = None, sizes: Sequence[str] = SIZES
) -> Figure:
    """System prompting (diamond) against the best other method (circle, named) per property
    and size, on the radar scale (centre = base, `RADAR_MIN` = 0; 10 = best).

    A hollow marker is an organism below the expression threshold (as in the comparison
    figures). An arrow runs from the prompting marker to the best-other marker where the two
    differ by more than `PVB_ARROW_EPS`. The line under each panel is prompting as a share of
    the best other method across sizes.
    """
    pts = prompting_vs_best_points(pd_, sizes)
    fig, axes = plt.subplots(2, 3, figsize=(TEXT_WIDTH_IN, 3.75))
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.2, wspace=0.3, hspace=0.85)
    sp = STYLE["sysprompt"]
    for ax, axis in zip(axes.flat, PVB_AXES, strict=True):
        bg = ax.get_facecolor()
        ratios: list[float] = []
        for y, size in enumerate(sizes):
            pt = pts.get((axis, size))
            if pt is None:
                continue
            p, b = pt["prompt_drawn"], pt["best_drawn"]
            if math.isfinite(pt["ratio"]):
                ratios.append(pt["ratio"])
            best = STYLE[pt["best_key"]]
            if pt["arrow"]:
                ax.annotate(
                    "",
                    xy=(b, y),
                    xytext=(p, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": MUTED,
                        "lw": 1.1,
                        "shrinkA": 4.5,
                        "shrinkB": 4.5,
                        "mutation_scale": 7,
                    },
                    zorder=2,
                )
            ax.scatter(
                [b],
                [y],
                s=34,
                facecolor=bg if pt["best_flagged"] else best.colour,
                edgecolor=best.colour if pt["best_flagged"] else "white",
                lw=1.2 if pt["best_flagged"] else 0.5,
                zorder=4,
            )
            ax.scatter(
                [p],
                [y],
                s=40,
                marker="D",
                facecolor=bg if pt["prompt_flagged"] else sp.colour,
                edgecolor=sp.colour if pt["prompt_flagged"] else "white",
                lw=1.2 if pt["prompt_flagged"] else 0.5,
                zorder=3,
            )
            ax.text(
                b,
                y + 0.2,
                best.label,
                ha="center",
                va="bottom",
                fontsize=6.6,
                color=MUTED,
                zorder=5,
            )
        ax.set_xlim(RADAR_MIN - 0.3, 11.0)
        ax.set_xticks([0, 2.5, 5, 7.5, 10])
        ax.spines["bottom"].set_bounds(RADAR_MIN, 10)
        ax.set_ylim(-0.5, 2.85)
        ax.set_yticks(range(len(sizes)), list(sizes))
        ax.set_title(axis.capitalize(), loc="left", fontsize=8.5, color=INK, pad=4)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        if ratios:
            text = f"Prompting: {min(ratios):.0%}\u2013{max(ratios):.0%} of best other"
            if plt.rcParams["text.usetex"]:
                text = text.replace("%", "\\%")
            ax.set_xlabel(text, fontsize=7, color=MUTED, labelpad=3)
    handles = [
        Line2D([], [], marker="D", ls="", color=sp.colour, ms=6, label="Prompting"),
        Line2D([], [], marker="o", ls="", color=MUTED, ms=6, label="Best other method (named)"),
        Line2D(
            [], [], marker="o", ls="", mfc="none", mec=MUTED, mew=1.2, ms=6,
            label="Hollow: below the expression threshold",
        ),
    ]  # fmt: skip
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.01),
        fontsize=7.5,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    _stamp(fig, stamp, "top-right")
    return fig


# ---------------------------------------------------------------- appendix: debate turn curve

DEBATE_EVAL = "robustness_multiturn_debate_gen"


def debate_turn_rows(root: Path, families: Families | None = None) -> list[dict[str, Any]]:
    """Per-turn debate scores from a results tree, one record per (organism row, turn).

    Reads `meta.turn_scores` (the per-turn `open_ended` judge score of each organism turn
    A1..A4, PR #356) straight from the result files (`core.results.result_files`) +
    spec.json, skipping `superseded/`. Method rows only (`method_key` with `families`). A
    NaN/None turn (unscoreable or context-exhausted) is left out,
    never counted as 0.
    """
    import json as _json

    from analysis.aggregate import _variant_of

    out: list[dict[str, Any]] = []
    for f in result_files(root, eval_name=DEBATE_EVAL):
        if SUPERSEDED in f.relative_to(root).parts:
            continue
        spec_path = f.parent / "spec.json"
        if not spec_path.exists():
            continue
        spec = _json.loads(spec_path.read_text())
        spec = spec.get("spec", spec)
        method = spec.get("method", "")
        family = f.parent.parent.name.rsplit("__", 1)[-1]
        key = method_key(method, _variant_of(spec), family, families)
        if key is None or key == "antiprompt":
            continue
        # Keyed like `Row.model_tag`, so a 27B's thinking-on and -off debate turns do not pool
        # into one size (identical to `_model_tag` for every thinking-off organism).
        size = organism_model_tag(
            spec.get("base_model", ""),
            bool((spec.get("method_config") or {}).get("enable_thinking", False)),
        )
        t = read_result_table(f, columns=["sample_id", "meta"]).to_pydict()
        for sid, meta in zip(t["sample_id"], t["meta"], strict=True):
            ts = _json.loads(meta).get("turn_scores")
            if isinstance(ts, str):
                ts = _json.loads(ts)
            for j, v in enumerate(ts or []):
                if v is None or not math.isfinite(float(v)):
                    continue
                out.append(
                    {
                        "key": key,
                        "method": STYLE[key].label,
                        "size": size,
                        "turn": f"A{j + 1}",
                        "score": float(v),
                        "item": sid,
                    }
                )
    return out


def debate_turn_curve(root: Path, *, seed: int = 0, families: Families | None = None) -> Figure:
    """Appendix: debate expression per organism turn (A1..A4), one panel per model size.

    seaborn lineplot, 95% CI by bootstrap over debates (items; one seed). Every paper method is
    drawn, DPO included: the v4 DPO organisms' debates are scoreable (the v3 ones, 89-100 of
    100 unscoreable, were left out until 2026-10-02). Method colours; Base model dashed.
    """
    import pandas as pd

    sns = _seaborn()
    df = pd.DataFrame.from_records(debate_turn_rows(root, families))
    order = [STYLE[k].label for k in ("base", *ROW_ORDER)]
    order = [m for m in order if m in set(df["method"])]
    palette = {STYLE[k].label: STYLE[k].colour for k in STYLE}
    dashes = {m: (4, 2) if m == BASE_STYLE.label else "" for m in order}
    fig, axes = plt.subplots(1, len(SIZES), figsize=(TEXT_WIDTH_IN, 2.4), sharey=True)
    fig.subplots_adjust(left=0.08, right=0.8, top=0.9, bottom=0.2, wspace=0.12)
    for ax, size in zip(axes, SIZES, strict=True):
        sub = df[df["size"] == size]
        sns.lineplot(
            data=sub,
            x="turn",
            y="score",
            hue="method",
            style="method",
            hue_order=order,
            style_order=order,
            palette=palette,
            dashes=dashes,
            markers=False,
            errorbar=("ci", 95),
            n_boot=1000,
            seed=seed,
            err_kws={"alpha": 0.12, "lw": 0},
            lw=1.2,
            legend=False,
            ax=ax,
        )
        ax.set_title(f"Qwen2.5-{size}-Instruct", fontsize=8.5)
        ax.set_xlabel("organism turn")
        ax.set_ylabel("expression (0-1)" if ax is axes[0] else "")
        ax.set_ylim(0, 1)
        _clean(ax, "y")
    handles = [
        Line2D(
            [],
            [],
            color=palette[m],
            lw=1.2,
            ls=(0, (4, 2)) if m == BASE_STYLE.label else "-",
            label=m,
        )
        for m in order
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.81, 0.55),
        frameon=False,
        fontsize=7.5,
        handlelength=2.0,
    )
    return fig


# ---------------------------------------------------------------- Fig. 4: generality across scale


def generality_scale(
    pd_: PaperData, *, axis: str = "generality", stamp: str | None = None
) -> Figure:
    """Figure 4: the raw 0-1 generality composite vs model size, one line per method, Base
    Model dashed, shared colour legend at the side. The same quantity as the radar's
    absolute rate, NOT the radar's Generality spoke (that is 10 x the gain over base on
    matched items, `radar_value`). One panel: the lift over base differs from this by the
    base's own 0.01-0.03, so a second panel added nothing (2026-09-25).
    """
    sns = _seaborn()
    series = {
        s.key: [reading(pd_, pd_.organism.get((s.key, size)), axis) for size in SIZES]
        for s in METHODS
    }
    base = [base_reading(pd_, size, axis) for size in SIZES]
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, SCALE_H_IN))
    fig.subplots_adjust(left=0.1, right=0.74, top=0.95, bottom=0.17)
    xs = np.arange(len(SIZES))
    handles = []
    for s in METHODS:
        ys = [r.raw if r.status == "ok" else np.nan for r in series[s.key]]
        if not any(math.isfinite(y) for y in ys):
            continue
        lw = 2.4 if s.highlight else 1.4
        sns.lineplot(
            x=xs,
            y=ys,
            color=s.colour,
            lw=lw,
            marker="o",
            markersize=3.6 if not s.highlight else 4.6,
            markeredgecolor="white",
            markeredgewidth=0.4,
            zorder=3,
            ax=ax,
        )
        handles.append(
            Line2D(
                [],
                [],
                color=s.colour,
                lw=lw,
                marker="o",
                markersize=3.6,
                markeredgecolor="white",
                label=s.label,
            )
        )
    bys = [b.raw if b.status == "ok" else np.nan for b in base]
    (bl,) = ax.plot(
        xs,
        bys,
        color=BASE_STYLE.colour,
        ls=BASE_STYLE.linestyle,
        lw=1.4,
        zorder=2,
        label=BASE_STYLE.label,
    )
    handles.append(bl)
    ax.set_ylabel(f"{axis.capitalize()} composite (0\u20131)")
    ax.set_ylim(0, 1)
    ax.set_xticks(xs)
    ax.set_xticklabels(SIZES)
    ax.set_xlim(-0.15, len(SIZES) - 1 + 0.15)
    ax.set_xlabel("Base model scale")
    _clean(ax, "y")
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.75, 0.55),
        frameon=False,
        fontsize=7.5,
        handlelength=2.0,
    )
    _stamp(fig, stamp, "bottom-left")
    return fig


# ------------------------------------------------------------ generality vs coherence, in Delta
# Every comparison with the Base model is Delta = organism - the same-size Base model on the
# same items, in the score's own 0-1 units (2026-10-22: one score, no effect size d).
# Ported from the figure's out-of-repo scripts (compute_delta.py, plot_gencoh_delta.py,
# derive_paper_numbers.py of session cd79cd35).

GENCOH_KEYS = tuple(s.key for s in METHODS)
# The five methods that change weights or activations (the correlation's n = 5 check).
GENCOH_WEIGHT_KEYS = ("oct", "sft", "dpo", "steering", "seqkd")
GENCOH_AXES = ("generality", "robustness", "coherence", "capability", "introspection")
GENCOH_FIGURE = "generality_coherence_scale.pdf"
GENCOH_TABLE = "generality_coherence_correlations.tex"
GENCOH_DARK_BG = "#1e1e1e"


@dataclass(frozen=True)
class DeltaReading:
    """One (size, method, axis) Delta: organism minus the same-size Base model, 0-1 units.

    `value` is the matched-item difference on a base-relative axis (generality, introspection:
    `Reading.gain_raw - gain_base_raw`; coherence, capability: the same difference over the
    reading's evals with item scores kept for them, `_matched_composites`) and the organism's
    own share kept on robustness (self-referenced, no base). `se` is the standard error of the
    difference, the organism's and the base's cell SEs combined as independent,
    sqrt(sum_e (se_org,e^2 + se_base,e^2)) / k (`MatrixCell.sem`); NaN where undefined.
    """

    size: str
    key: str
    axis: str
    value: float
    se: float = math.nan


def _delta_se(m: Matrix, org: str, base: str, evals: Sequence[str]) -> float:
    if not evals:
        return math.nan
    tot = 0.0
    for e in evals:
        for o in (org, base):
            cell = m.get(o, e)
            if cell is None:
                return math.nan
            tot += (cell.sem or 0.0) ** 2
    return math.sqrt(tot) / len(evals)


def delta_readings(
    pd_: PaperData,
    rows: Sequence[Row],
    sizes: Sequence[str] = SIZES,
    extra_keys: Sequence[str] = (),
) -> dict[tuple[str, str, str], DeltaReading]:
    """Delta for every (size, method, axis) the paper shows, from the rows behind `pd_`.

    `extra_keys` adds method rows beyond the paper's (external adapters, opt-in); default none.

    A second matrix over the SAME rows keeps item scores for every axis in `GENCOH_AXES`, so
    the health axes (which `select` keeps no item scores for) are also differenced on matched
    items. Pending readings are absent, never 0.
    """
    shown = set(pd_.organism.values())
    bad = set(pd_.other_bank)
    keep = [r for r in rows if r.organism in shown and (r.organism, r.eval_name) not in bad]
    m_all = build_matrix(keep, item_axes=set(GENCOH_AXES) | set(BASE_RELATIVE_AXES))
    out: dict[tuple[str, str, str], DeltaReading] = {}
    for size in sizes:
        base = pd_.organism.get(("base", size))
        if base is None:
            continue
        for key in (*GENCOH_KEYS, *extra_keys):
            org = pd_.organism.get((key, size))
            if org is None:
                continue
            for axis in GENCOH_AXES:
                r = reading(pd_, org, axis)
                if r.status != "ok":
                    continue
                if self_axis(pd_.matrix, axis):
                    out[(size, key, axis)] = DeltaReading(size, key, axis, r.raw, math.nan)
                    continue
                if base_reading(pd_, size, axis).status != "ok":
                    continue
                if axis in BASE_RELATIVE_AXES:
                    value = r.gain_raw - r.gain_base_raw
                else:
                    g, b = _matched_composites(m_all, org, base, list(r.evals))
                    value = g - b
                if not math.isfinite(value):
                    continue
                se = _delta_se(pd_.matrix, org, base, list(r.evals))
                out[(size, key, axis)] = DeltaReading(size, key, axis, value, se)
    return out


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return math.nan
    return sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True)) / (sx * sy)


# The rows of the correlation table: (label, x axis, methods). y is always coherence.
GENCOH_CORR_ROWS = (
    ("Generality", "generality", GENCOH_KEYS),
    ("Robustness (share kept)", "robustness", GENCOH_KEYS),
    ("Introspection", "introspection", GENCOH_KEYS),
    ("Generality, weight and activation methods only", "generality", GENCOH_WEIGHT_KEYS),
)


def gencoh_correlations(
    dr: dict[tuple[str, str, str], DeltaReading], sizes: Sequence[str] = SIZES
) -> dict[str, dict[str, float]]:
    """Pearson r of coherence Delta against another axis's value, across methods, per size.

    {row label: {size: r}}; NaN where any method in the row is missing at that size (a
    correlation over whatever happened to land would be a different quantity).
    """
    out: dict[str, dict[str, float]] = {}
    for label, axis, keys in GENCOH_CORR_ROWS:
        out[label] = {}
        for size in sizes:
            xs = [dr.get((size, k, axis)) for k in keys]
            ys = [dr.get((size, k, "coherence")) for k in keys]
            if any(v is None for v in (*xs, *ys)):
                out[label][size] = math.nan
                continue
            out[label][size] = _pearson([v.value for v in xs if v], [v.value for v in ys if v])
    return out


def gencoh_correlation_table(
    dr: dict[tuple[str, str, str], DeltaReading], sizes: Sequence[str] = SIZES
) -> str:
    """The correlation table as a LaTeX tabular (booktabs), r to two places, `--` if pending."""
    corr = gencoh_correlations(dr, sizes)
    head = " & ".join(["Pearson $r$ with coherence $\\Delta$", *sizes]) + " \\\\"
    lines = [
        "% Generated by analysis/paper_figures.py (gencoh_correlation_table).",
        "\\begin{tabular}{l" + "r" * len(sizes) + "}",
        "\\toprule",
        head,
        "\\midrule",
    ]
    for label, per in corr.items():
        cells = [
            f"{v:.2f}".replace("-", "$-$") if math.isfinite(v) else "--" for v in per.values()
        ]
        lines.append(" & ".join([label, *cells]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


# Panel titles where the size alone is ambiguous (the Qwen2.5 sizes keep their short names).
GENCOH_TITLES = {"31B": "Gemma-4-31B"}
# The figure's panels: Qwen2.5 7B/14B/32B and Gemma-4-31B (a size with no readings is dropped).
GENCOH_SIZES = (*SIZES, "31B")


def _lighten(colour: str, dark: bool) -> Any:
    """On the dark theme, mix a method colour 35% toward white so it holds against #1e1e1e."""
    if not dark:
        return colour
    import matplotlib.colors as mc

    return tuple(0.35 + 0.65 * v for v in mc.to_rgb(colour))


def _finite_or_zero(v: float) -> float:
    return v if math.isfinite(v) else 0.0  # a missing SE draws no bar


def generality_coherence_delta(
    dr: dict[tuple[str, str, str], DeltaReading],
    *,
    sizes: Sequence[str] = GENCOH_SIZES,
    dark: bool = False,
    stamp: str | None = None,
) -> Figure:
    """Coherence Delta against generality Delta, one panel per size, one point per method.

    One row of panels (a size with no readings is dropped) under a single method legend; the
    "r = ..." label sits in the same corner (bottom-left) of every panel.

    x = generality gain over the same-size Base model (matched items), y = coherence Delta
    (matched items); both in the score's own 0-1 units, so the Base model is the origin
    (diamond) and a point below the dashed zero line lost coherence. Error bars are the 95%
    interval of the difference (1.96 x `DeltaReading.se`). The fitted line and the printed
    r are over the panel's methods. A method missing either coordinate at a size is not drawn
    and the fit uses the rest. `dark` re-themes for a dark page (review renders only).
    """
    if dark:
        fg, mu, bg, grid = "#ececec", "#a8a8a8", GENCOH_DARK_BG, "#3a3a3a"
    else:
        fg, mu, bg, grid = INK, "#6b6b6b", "white", "#e4e4e4"
    pts: dict[str, list[tuple[str, float, float, float, float]]] = {}
    for size in sizes:
        pts[size] = []
        for key in GENCOH_KEYS:
            g, c = dr.get((size, key, "generality")), dr.get((size, key, "coherence"))
            if g is None or c is None:
                continue
            pts[size].append((key, g.value, c.value, 1.96 * g.se, 1.96 * c.se))
    sizes = [s for s in sizes if pts[s]] or list(sizes)
    allp = [p for q in pts.values() for p in q]
    if allp:
        xmax = max(p[1] + _finite_or_zero(p[3]) for p in allp)
        ymin = min(p[2] - _finite_or_zero(p[4]) for p in allp)
        ymax = max(p[2] + _finite_or_zero(p[4]) for p in allp)
    else:
        xmax, ymin, ymax = 0.9, -0.3, 0.0
    with plt.rc_context(
        {
            "text.usetex": False,
            "text.color": fg,
            "axes.labelcolor": fg,
            "axes.edgecolor": mu,
            "xtick.color": mu,
            "ytick.color": mu,
            "figure.facecolor": bg,
            "axes.facecolor": bg,
        }
    ):
        fig, axes = plt.subplots(
            1, len(sizes), figsize=(TEXT_WIDTH_IN, 2.35), sharex=True, sharey=True, squeeze=False
        )
        axs = list(axes[0])
        for ax, size in zip(axs, sizes, strict=True):
            q = pts[size]
            ax.grid(True, color=grid, lw=0.5, zorder=0)
            ax.axhline(0.0, color=mu, lw=0.8, ls=(0, (4, 2)), zorder=1)
            ax.axvline(0.0, color=mu, lw=0.8, ls=(0, (4, 2)), zorder=1)
            ax.plot([0.0], [0.0], marker="D", ms=3.6, mfc="none", mec=mu, mew=0.9, ls="", zorder=2)
            if len(q) >= 2:
                xs = np.array([p[1] for p in q])
                ys = np.array([p[2] for p in q])
                sl, ic = np.polyfit(xs, ys, 1)
                xx = np.array([xs.min(), xs.max()])
                ax.plot(xx, sl * xx + ic, color=mu, lw=0.8, alpha=0.85, zorder=1.5)
                r = _pearson(xs.tolist(), ys.tolist())
                if math.isfinite(r):
                    ax.text(  # bottom-left in every panel: empty at every size
                        0.1,
                        0.04,
                        "r = " + f"{r:.2f}".replace("-", "−"),
                        transform=ax.transAxes,
                        ha="left",
                        va="bottom",
                        fontsize=7.5,
                    )
            for key, x, y, xe, ye in q:
                c = _lighten(STYLE[key].colour, dark)
                ax.errorbar(
                    x,
                    y,
                    xerr=_finite_or_zero(xe),
                    yerr=_finite_or_zero(ye),
                    fmt="o",
                    ms=4.6,
                    color=c,
                    elinewidth=0.9,
                    capsize=0,
                    zorder=3,
                    mec=bg,
                    mew=0.5,
                )
            ax.set_title(GENCOH_TITLES.get(size, size), fontsize=8.5, pad=3)
            ax.set_xlim(-0.06, max(0.95, xmax + 0.03))
            ax.set_xticks([0, 0.25, 0.5, 0.75])
            ax.set_ylim(min(-0.32, ymin - 0.02), max(0.04, ymax + 0.02))
            ax.set_yticks([-0.3, -0.2, -0.1, 0])
            ax.spines[["top", "right"]].set_visible(False)
        xl, yl = r"Generality ($\Delta$ vs base model)", r"Coherence ($\Delta$ vs base model)"
        axs[0].set_xlabel(xl)
        axs[0].set_ylabel(yl)
        from matplotlib.lines import Line2D

        keys = [k for k in GENCOH_KEYS if any(k == p[0] for q in pts.values() for p in q)]
        if keys:  # no readings at all: matplotlib cannot build an empty legend
            fig.legend(
                handles=[
                    Line2D(
                        [], [], marker="o", ls="", ms=4.6, color=_lighten(STYLE[k].colour, dark)
                    )
                    for k in keys
                ],
                labels=[STYLE[k].label for k in keys],
                loc="upper center",
                bbox_to_anchor=(0.55, 1.0),
                ncol=len(keys),
                frameon=False,
                fontsize=7.0,
                handletextpad=0.2,
                columnspacing=1.0,
            )
        fig.subplots_adjust(left=0.1, right=0.995, top=0.8, bottom=0.2, wspace=0.07)
        # The shared x-label is centred under the whole row of panels (under any one panel it
        # sits off-centre when the panel count is even); x is in the first panel's axes units.
        p0, p1 = axs[0].get_position(), axs[-1].get_position()
        axs[0].xaxis.label.set_x(((p0.x0 + p1.x1) / 2 - p0.x0) / p0.width)
    _stamp(fig, stamp, "bottom-left")
    return fig


# ---------------------------------------------------------------- strength vs share kept


def kept_scatter_points(pd_: PaperData) -> list[dict[str, Any]]:
    """One point per shown implant method x size with a robustness reading.

    x: the no-pressure `robustness_open_ended_gen` rate (strength, the denominator of the
    three `_gen` shares) with its cell's 95% CI; y: the robustness reading's raw composite
    (mean share kept, the radar spoke / 10). `flagged` per `flagged`. A method x size with
    no robustness reading, or no open-ended cell, has no point (never a 0).
    """
    m = pd_.matrix
    out = []
    for st in METHODS:
        for size in SIZES:
            org = pd_.organism.get((st.key, size))
            cell = m.get(org, OPEN_ENDED_GEN) if org is not None else None
            r = reading(pd_, org, "robustness")
            if cell is None or not math.isfinite(cell.mean) or r.status != "ok":
                continue
            lo, hi = cell.ci95
            out.append(
                {
                    "key": st.key,
                    "size": size,
                    "x": cell.mean,
                    "x_lo": lo,
                    "x_hi": hi,
                    "y": r.raw,
                    "flagged": flagged(pd_, org),
                }
            )
    return out


def kept_scatter(pd_: PaperData, *, stamp: str | None = None) -> Figure:
    """Strength (x) against robustness share kept (y), one marker per method x size.

    Method colour, size marker (SIZE_MARKERS), hollow when the organism is SBE-flagged. The
    share kept is meaningful only beside the strength it is a share of: a weak implant that
    keeps "all" of almost nothing sits top-left, a strong one that keeps it top-right.
    """
    pts = kept_scatter_points(pd_)
    if not pts:
        return placeholder(SINGLE_COL_IN, SCALE_H_IN, "PLACEHOLDER — awaiting real data")
    fig, ax = plt.subplots(figsize=(SINGLE_COL_IN, SCALE_H_IN + 0.3))
    for p in pts:
        colour = STYLE[p["key"]].colour
        ax.errorbar(
            p["x"],
            p["y"],
            xerr=[[max(p["x"] - p["x_lo"], 0.0)], [max(p["x_hi"] - p["x"], 0.0)]],
            fmt="none",
            ecolor=colour,
            elinewidth=0.6,
            alpha=0.6,
            zorder=2,
        )
        ax.plot(
            p["x"],
            p["y"],
            ls="none",
            marker=SIZE_MARKERS[p["size"]],
            markersize=5.5,
            markerfacecolor="white" if p["flagged"] else colour,
            markeredgecolor=colour,
            markeredgewidth=1.0,
            zorder=3,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Strength: trait rate without pressure")
    ax.set_ylabel("Robustness: share kept")
    _clean(ax, grid="both")
    shown = list(dict.fromkeys(p["key"] for p in pts))
    handles: list[Any] = [
        Line2D(
            [],
            [],
            ls="none",
            marker="s",
            markersize=5,
            color=STYLE[k].colour,
            label=STYLE[k].label,
        )
        for k in shown
    ]
    handles += [
        Line2D([], [], ls="none", marker=SIZE_MARKERS[z], markersize=5, color=MUTED, label=z)
        for z in SIZES
        if any(p["size"] == z for p in pts)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=6.5,
        handletextpad=0.2,
        columnspacing=0.8,
    )
    fig.subplots_adjust(left=0.15, right=0.97, top=0.8, bottom=0.17)
    _stamp(fig, stamp, "bottom-left")
    return fig


# ---------------------------------------------------------------- driver


FIGURE_FILES = {
    "radar": "radar_{size}.pdf",
    "expression": "expression_bar.pdf",
    "expression_size": "expression_bar_{size}.pdf",
    "dots": "axis_dots.pdf",
    "dots_size": "axis_dots_{size}.pdf",
    "scale": "generality_scale.pdf",
    "mu": "mu_decisiveness.pdf",
    "fig1": "radar_methods_32b.pdf",
    "fig1_wide": "radar_methods_32b_wide.pdf",
    "radar_overlay": "radar_overlay_{size}.pdf",
    "kept": "robustness_strength_scatter.pdf",
    "gencoh": GENCOH_FIGURE,
}


def captions(pd_: PaperData, pending: dict[str, list[str]], eval_name: str) -> dict[str, str]:
    """Caption sentences naming the measure and the pending cells, for the .tex."""
    rp = _pending_phrase(pending, RADAR_AXES)
    dp = _pending_phrase(pending, [a for a, _, _ in DOT_AXES])
    ep = _pending_phrase(pending, ["expression"])
    gp = _pending_phrase(pending, ["generality"])
    return {
        "radar": (
            "Axis profile per implant method, one panel each. "
            + RADAR_SCALE_TEXT
            + " "
            + _axis_set_phrase(pd_, "coherence")
            + (" " + fp if (fp := _flagged_phrase(pd_)) else "")
            + (" " + cp if (cp := _collapsed_phrase(pd_, RADAR_AXES)) else "")
            + (" " + sp if (sp := _size_set_phrase(pd_, RADAR_AXES)) else "")
            + (" " + bp if (bp := _by_design_phrase(pd_, RADAR_AXES)) else "")
            + (f" Pending: {rp}." if rp else "")
        ),
        "expression": (
            f"Expression rate (\\texttt{{{_tex(eval_name)}}}): share of answers the judge "
            "marks as showing the implanted behavior, per implant method and base-model size; "
            "the Base model is the control. Error bars: 95\\% CI, clustered by item and pooled "
            "over seeds." + (f" Pending: {ep}." if ep else "")
        ),
        "dots": (
            "Deviation from the base model per axis, in the evals' own units: the mean over the "
            "axis's composite evals of (organism $-$ base at the same size), oriented so positive "
            "is better; zero (dashed) is the base. Robustness is the exception: it is the mean "
            "share kept (0--1, the organism's trait under pressure over its own no-pressure "
            "rate), read on its own and not against the base. Markers: 7B circle, 14B "
            "square, 32B triangle."
            + (" " + cp if (cp := _collapsed_phrase(pd_, [a for a, _, _ in DOT_AXES])) else "")
            + (" " + sp if (sp := _size_set_phrase(pd_, [a for a, _, _ in DOT_AXES])) else "")
            + (" " + bp if (bp := _by_design_phrase(pd_, [a for a, _, _ in DOT_AXES])) else "")
            + (f" Pending: {dp}." if dp else "")
        ),
        "scale": (
            "Generality across model scale: the absolute raw 0--1 generality composite per "
            "implant method, Base model dashed. This is the rate itself; the radar's Generality "
            "spoke is $10\\times$ its gain over the Base model."
            + (" " + cp if (cp := _collapsed_phrase(pd_, ["generality"])) else "")
            + (f" Pending: {gp}." if gp else "")
        ),
        "mu": (
            "$\\mu$-decisiveness \\citep{tan2026fried}: the mean distance of the fitted "
            "pairwise preferences from "
            "a coin flip, $|2\\hat\\Phi - 1|$, on its item panel, "
            "0 = indifferent, "
            "1 = perfectly decisive, per method and base-model size, one seed. Error bars: 95\\% "
            "percentile intervals from the upstream item-cluster bootstrap (200 refits resampling "
            "items); a refit interval can sit off "
            "the point estimate. One vote in the coherence composite."
            + (f" Pending: {mp}." if (mp := _mu_pending(pd_)) else "")
        ),
        "kept": (
            "Strength against robustness: per implant method and base-model size, the "
            "no-pressure trait rate on the generated robustness items "
            "(\\texttt{robustness\\_open\\_ended\\_gen}, $x$, 95\\% CI clustered by item) "
            "against the mean share of that trait kept under pressure (the radar's "
            "Robustness spoke divided by 10, $y$). Markers: 7B circle, 14B square, 32B "
            "triangle. Hollow: the organism fell below the "
            "expression threshold (\\texttt{simple\\_behaviour\\_exhibit}), so its share is "
            "of a trait not shown to be there."
            + (f" Pending: {kp}." if (kp := _pending_phrase(pending, ["robustness"])) else "")
        ),
        "overview": (
            "Qwen2.5-32B-Instruct: implant methods grouped by family (Prompting and Few-shot; "
            "Steering and Distillation; SFT, DPO and OCT). " + RADAR_SCALE_TEXT
        ),
    }


def _by_design_phrase(pd_: PaperData, axes: Iterable[str]) -> str:
    """Caption sentence for evals left out on purpose for one method (NOT_RUN_BY_DESIGN)."""
    axes = set(axes)
    parts = []
    for key, e in sorted(NOT_RUN_BY_DESIGN):
        ax = next((c.axis for (o, n), c in pd_.matrix.cells.items() if n == e), None)
        if ax in axes and any(k == key for k, _s in pd_.organism):
            parts.append(
                f"{STYLE[key].label}'s {ax} averages the other {ax} evals: "
                f"\\texttt{{{_tex(e)}}} is not run for it, by design."
            )
    return " ".join(parts)


# ---------------------------------------------------------------- Figure 1: overview radar

FIG1_METHODS = ("sysprompt", "steering", "oct")
FIG1_TITLES = {"sysprompt": "Prompting", "steering": "Steering", "oct": "OCT"}
# Health axes are read against base: base = 10 (the OUTER ring), degradation moves inward.
# Behaviour axes are the gain over base, 10 x (organism - base) on matched items, base = 0 (the
# centre); robustness is the share kept, 10 = all of its own trait kept.


def fig1_scores(pd_: PaperData, size: str, keys: Sequence[str]) -> dict[str, list[float]]:
    """0-10 score per RADAR_AXES spoke for base and each method key (NaN = no data).

    Behaviour axes (generality, introspection, activations): the GAIN OVER BASE, signed and
    unclipped below, 10 x (organism - base) over the axis's composite evals restricted to the
    items scorable for BOTH (`Reading.gain_raw` / `gain_base_raw`); the Base model is exactly 0
    and a method below base is negative (the drawing clips at `RADAR_MIN`, the centre).
    Activations is the truth probe (`whitebox_truth_world`, `_self`); NaN unless BOTH cells
    voted for the organism
    and the base. Health axes (coherence, capability): 10 x (1 + (organism - base)) raw composite,
    clipped to [0, 10], so base is 10 on the outer ring and lower means damage.
    Robustness is a behaviour axis whose composite is the mean SHARE KEPT (`self_axis`), 10 x
    the share, clipped to [0, 10]: it needs no base reading, so an organism's vertex is drawn
    whatever the base's share is. The BASE has no robustness reading at all (NaN, pending;
    `BASE_NO_ROBUSTNESS_WHY`): it has almost no behaviour to keep, so its share kept is noise
    (2026-10-20).
    """
    out: dict[str, list[float]] = {}
    for key in ("base", *keys):
        org = pd_.organism.get((key, size))
        vals = []
        for axis in RADAR_AXES:
            if org is None or (key == "base" and self_axis(pd_.matrix, axis)):
                vals.append(math.nan)
                continue
            r = base_reading(pd_, size, axis) if key == "base" else reading(pd_, org, axis)
            b = base_reading(pd_, size, axis)
            # The activations spoke averages the world AND self cells: one missing for this
            # organism (or for the whole tree, or for the base) is pending, not a one-cell spoke.
            if axis in TRUTH_PROBE_AXES and not (
                set(TRUTH_PROBE_VOTERS) <= set(r.evals) and set(TRUTH_PROBE_VOTERS) <= set(b.evals)
            ):
                vals.append(math.nan)
                continue
            # r.base_raw: the base over the SAME evals as this organism's reading (a
            # collapsed cell leaves this organism with fewer); equals b.raw otherwise.
            # Behaviour spokes read the MATCHED-item composites (gain over base); the others
            # read the plain ones (`radar_value`).
            raw, base_raw = (
                (r.gain_raw, r.gain_base_raw)
                if axis in BASE_RELATIVE_AXES
                else (r.raw, r.base_raw)
            )
            vals.append(
                radar_value(axis, raw, base_raw)
                if r.status == "ok" and (b.status == "ok" or self_axis(pd_.matrix, axis))
                else math.nan
            )
        out[key] = vals
    return out


def radar_eval_values(pd_: PaperData, organism: str, axis: str) -> dict[str, float]:
    """The radar rule (`radar_value`) applied to ONE eval at a time (dashboard drill radars).

    {eval: 0-10 value} over the evals `reading` averages for `organism` on `axis`: behaviour axes
    read the eval's matched-item gain over base (10 x (organism - base)), health axes
    10 x (1 + (organism - base)) on that eval, robustness 10 x its own kept share. Empty when the
    axis reading is pending. A drill-down aid only: the mean of these is the spoke when nothing is
    clipped (a difference of means is the mean of differences), but a per-eval clip at 0/10
    can move it, and a health spoke reads the plain (not matched-item) composites, so the
    spoke itself is always `fig1_scores`.
    """
    m = pd_.matrix
    r = reading(pd_, organism, axis)
    base = m.base_for(organism)
    if r.status != "ok":
        return {}
    out: dict[str, float] = {}
    for e in r.evals:
        raw, base_raw = (
            _cell_mean(m, organism, e),
            (_cell_mean(m, base, e) if base is not None else math.nan),
        )
        if axis in BASE_RELATIVE_AXES and base is not None:
            raw, base_raw = _matched_composites(m, organism, base, [e])
        out[e] = radar_value(axis, raw, base_raw)
    return out


RADAR_FULL = [
    "Generality",
    "Robustness",
    "Coherence",
    "Capability",
    "Activations",
    "Introspection",
]


def radar_overlay_axes(
    ax: PolarAxes,
    scores: dict[str, list[float]],
    keys: Sequence[str],
    *,
    label_size: float = 7.5,
    zero_ring: float | None = None,
    show_base: bool = True,
    lw: float = 1.0,
    flags: dict[str, bool] | None = None,
) -> list[Any]:
    """All `keys` methods overlaid as translucent polygons on one radar, dashed Base model.

    Radii in [RADAR_MIN, 10] (`fig1_scores`; 0, the centre, is the Base model on the
    behaviour spokes; a value below it is drawn at the centre, unmarked) or, for
    the deviation variant, [0, 10]
    (`fig1_deviation_scores` with `zero_ring=5`, which draws the bold 0-deviation ring and
    hides the radial tick labels because each spoke has its own scale). Returns legend
    handles. No in-figure explanatory text. `flags` maps a key (base included) to its
    `flagged` state; a key absent from it draws the plain outline.
    """
    flags = flags or {}
    n = len(RADAR_AXES)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(RADAR_MIN if zero_ring is None else 0, 10)
    ax.set_yticks([0, 2, 4, 6, 8, 10] if zero_ring is None else [2.5, 5, 7.5, 10])
    ax.set_yticklabels(
        ["0", "2", "4", "6", "8", "10"] if zero_ring is None else ["", "", "", ""],
        fontsize=label_size - 2,
        color=MUTED,
    )
    ax.set_rlabel_position(0)
    if zero_ring is not None:
        ring = np.linspace(0, 2 * np.pi, 200)
        ax.plot(ring, np.full_like(ring, zero_ring), color=INK, lw=0.9, zorder=1.5)
    ax.set_xticks(theta)
    ax.set_xticklabels(RADAR_FULL, fontsize=label_size, color=INK)
    ax.tick_params(axis="x", pad=-3)
    ax.grid(color=RADAR_GRID, linestyle=(0, (1.5, 1.5)), linewidth=0.5)
    ax.spines["polar"].set_visible(False)
    handles = []
    for key in keys:
        st = STYLE[key]
        v = np.asarray(scores[key], dtype=float)
        keep = np.isfinite(v)
        if keep.sum() < 3:  # nothing measured: no polygon, no legend entry (never a 0)
            continue
        th = np.append(theta[keep], theta[keep][0])
        rr = np.clip(np.append(v[keep], v[keep][0]), RADAR_MIN, 10)
        ax.fill(th, rr, color=st.colour, alpha=0.16, zorder=2, lw=0)
        _outline(ax, theta, v, 10, colour=st.colour, lw=lw, zorder=3, flagged=flags.get(key))
        # the legend swatch IS the polygon: same translucent fill, same thin outline
        handles.append(
            mpatches.Patch(
                facecolor=matplotlib.colors.to_rgba(st.colour, 0.16),
                edgecolor=st.colour,
                linewidth=lw,
                label=st.label,
            )
        )
    if not show_base:
        return handles
    b = np.asarray(scores["base"], dtype=float)
    keep = np.isfinite(b)
    if keep.sum() >= 3:
        # Thick, dark and on top: base is 0 on the behaviour spokes (it has no implant) and
        # 10 on coherence/capability (it is the reference), so a thin outline disappears.
        # The base is the reference, not an organism under test: never flagged, it keeps its
        # own dashed style and vertex markers.
        _outline(
            ax,
            theta,
            b,
            10,
            colour=BASE_OUTLINE,
            lw=1.8,
            zorder=6,
            ls=(0, (3.5, 1.8)),
            clip_on=False,
        )
        ax.plot(
            theta[keep],
            b[keep],
            ls="none",
            color=BASE_OUTLINE,
            marker="o",
            markersize=3.2,
            zorder=6,
            clip_on=False,
        )
    handles.append(
        Line2D(
            [],
            [],
            color=BASE_OUTLINE,
            ls=(0, (3.5, 1.8)),
            lw=1.8,
            marker="o",
            markersize=3.2,
            label=BASE_STYLE.label,
        )
    )
    return handles


def _flags(pd_: PaperData, size: str, keys: Iterable[str]) -> dict[str, bool]:
    """key -> `flagged` for the organisms drawn at this size."""
    return {k: flagged(pd_, pd_.organism.get((k, size))) for k in keys}


def radar_overlay(pd_: PaperData, size: str) -> Figure:
    """One radar, every implant method overlaid (Fig. 1's centre panel, standalone)."""
    scores = fig1_scores(pd_, size, [s.key for s in METHODS])
    fig = plt.figure(figsize=(TEXT_WIDTH_IN * 0.6, 3.1))
    ax = cast(PolarAxes, fig.add_axes((0.2, 0.08, 0.6, 0.62), projection="polar"))
    handles = radar_overlay_axes(
        ax, scores, [s.key for s in METHODS], flags=_flags(pd_, size, ["base", *scores])
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=4,
        frameon=False,
        fontsize=7,
        handlelength=1.3,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    return fig


# Figure 1's centre panel (2026-09-25): three radars, one per method family, each
# against the Base model; the schematic either side is added by hand in the paper.
FIG1_GROUPS = (
    ("Prompting", ("sysprompt", "icl")),
    ("Steering and distillation", ("steering", "seqkd")),
    ("Fine-tuning", ("sft", "dpo", "oct")),
)


def fig1_deviation_scores(
    pd_: PaperData, size: str, keys: Sequence[str]
) -> dict[str, list[float]]:
    """Radar variant B: deviation from base (eval units, the numbers of Fig. 3) as radii.

    Per spoke, d / max|d| over the shown methods at this size maps to [-1, 1], drawn at
    radius 5 + 5 x that, so the Base model is the bold ring at 5 and every spoke uses its full
    range (a shared range would flatten coherence and capability, whose deviations are ~10x
    smaller than generality's). A self-referenced axis (robustness: share kept) has no
    deviation from base, so its spoke is NaN here (never drawn as a deviation).
    """
    raw = {
        k: [
            reading(pd_, pd_.organism.get((k, size)), a)
            if not self_axis(pd_.matrix, a)
            else Reading("pending", why="share kept is not a deviation from base")
            for a in RADAR_AXES
        ]
        for k in keys
    }
    out: dict[str, list[float]] = {
        "base": [
            5.0
            if base_reading(pd_, size, a).status == "ok" and not self_axis(pd_.matrix, a)
            else math.nan
            for a in RADAR_AXES
        ]
    }
    spans = []
    for j in range(len(RADAR_AXES)):
        ds = [abs(raw[k][j].delta) for k in keys if raw[k][j].status == "ok"]
        spans.append(max(ds) if ds and max(ds) > 0 else math.nan)
    for k in keys:
        out[k] = [
            5 + 5 * r.delta / spans[j]
            if r.status == "ok" and math.isfinite(spans[j])
            else math.nan
            for j, r in enumerate(raw[k])
        ]
    return out


RADAR_ABBREV = {
    "Robustness": "Rob.",
    "Coherence": "Coh.",
    "Introspection": "Intro.",
    "Activations": "Act.",
}
# Which side of the radar each spoke's label sits on (spoke order RADAR_AXES / RADAR_FULL,
# clockwise from Generality at the top): right = Robustness, Coherence; left = Activations,
# Introspection; top/bottom = Generality, Capability.
_SPOKE_SIDE = ("top", "right", "right", "bottom", "left", "left")


def polygon_area_share(
    values: Sequence[float], rmax: float = 10.0, centre: float = RADAR_MIN
) -> float:
    """Area of a radar polygon as a share of the full polygon (every spoke at `rmax`).

    Radii are measured from the radar centre (`centre`: RADAR_MIN, 0, on the score radars; 0 on the
    deviation radars too); a spoke below the centre is clipped there, as drawn.

    Spokes with no value (NaN) are skipped, as when the polygon is drawn: the area is of the
    polygon actually shown, over the full polygon on the same spokes. NaN if < 3 spokes.
    """
    n = len(values)
    idx = [i for i, v in enumerate(values) if math.isfinite(v)]
    if len(idx) < 3:
        return math.nan
    ang = [2 * math.pi * i / n for i in idx]
    r = [min(max(values[i], centre), rmax) - centre for i in idx]

    def area(rs: Sequence[float]) -> float:
        tot = 0.0
        for j in range(len(rs)):
            k = (j + 1) % len(rs)
            d = (ang[k] - ang[j]) % (2 * math.pi)
            tot += 0.5 * rs[j] * rs[k] * math.sin(d)
        return tot

    return area(r) / area([rmax - centre] * len(r))


def radar_score(values: Sequence[float], rmax: float = 10.0, centre: float = RADAR_MIN) -> float:
    """Radar score: the polygon area averaged over every ordering of the spokes, as a share of
    the full polygon (every spoke at `rmax`). Unlike `polygon_area_share` it does not depend on
    the order the spokes happen to be drawn in.

    With n spokes at equal angles, two spokes are neighbours in a random cyclic ordering with
    probability 2 / (n - 1), and a neighbouring pair adds 0.5 * r_i * r_j * sin(2 pi / n) of
    area. The full polygon has area n * 0.5 * rmax**2 * sin(2 pi / n), so
        score = (2 / (n - 1)) * sum_{i<j} r_i r_j / (n * rmax**2)
    (n = 6, rmax = 10: sum_{i<j} s_i s_j / 1500; all spokes at rmax gives 1).

    Radii are measured from the radar centre (`centre`); a spoke below the centre is clipped
    there, as drawn. A spoke with no value (NaN, e.g. Base has no robustness reading) counts as
    the centre (0) and n stays the full spoke count, so every legend entry is scored on the same
    polygon and is comparable. NaN if fewer than 3 spokes have a value (the drawing's skip rule).
    """
    if sum(math.isfinite(v) for v in values) < 3:
        return math.nan
    r = [min(max(v, centre), rmax) - centre if math.isfinite(v) else 0.0 for v in values]
    n = len(r)
    pairs = (sum(r) ** 2 - sum(x * x for x in r)) / 2
    return (2 / (n - 1)) * pairs / (n * (rmax - centre) ** 2)


def figure1_centre(
    pd_: PaperData,
    size: str = "32B",
    *,
    width_in: float = 3.3,
    full_labels: bool = True,
    metric: str = "score",
    abbrev_inner: bool = True,
    legend_area: bool = False,
) -> Figure:
    """Figure 1 centre: three radars side by side (FIG1_GROUPS), 0-10 scale of `fig1_scores`.

    Team feedback 2026-09-26: thin lines, no Base model outline, full spoke names, no
    per-panel titles, one legend on a single row at the top. Variants:
    - `abbrev_inner`: labels that face a neighbouring radar are abbreviated (Rob., Coh.,
      Intro., Act.); outer-left and outer-right labels stay full words; the radars grow.
    - `legend_area`: each legend entry also shows its radar score (`radar_score`: polygon area
      averaged over all spoke orderings, as a share of the full polygon).
    """
    del full_labels  # names are always full now
    keys = [k for _t, ks in FIG1_GROUPS for k in ks]
    scores = (
        fig1_scores(pd_, size, keys)
        if metric == "score"
        else fig1_deviation_scores(pd_, size, keys)
    )
    ncol = len(FIG1_GROUPS)
    cell_w = width_in / ncol
    narrow = width_in < 4.5
    label_size = 4.6 if narrow else 6.5
    legend_size = 5.2 if narrow else 7.0
    diam = cell_w * ((0.56 if abbrev_inner else 0.46) if narrow else 0.5)
    label_pad = 0.14 if narrow else 0.2
    # Method legend band: one text row, or two with `legend_area` (name + radar score). The
    # wide layout's font is larger, so its second row costs more height than the narrow one's.
    legend_h = (0.3 if legend_area else 0.17) if narrow else (0.4 if legend_area else 0.22)
    height_in = legend_h + 2 * label_pad + diam + 0.08
    fig = plt.figure(figsize=(width_in, height_in))
    handles_by_key: dict[str, Any] = {}
    for i, (_title, ks) in enumerate(FIG1_GROUPS):
        ax = cast(
            PolarAxes,
            fig.add_axes(
                (
                    (i * cell_w + (cell_w - diam) / 2) / width_in,
                    (0.06 + label_pad) / height_in,
                    diam / width_in,
                    diam / height_in,
                ),
                projection="polar",
            ),
        )
        handles = radar_overlay_axes(
            ax,
            {k: scores[k] for k in ("base", *ks)},
            ks,
            label_size=label_size,
            zero_ring=None if metric == "score" else 5.0,
            show_base=False,
            lw=0.6,
            flags=_flags(pd_, size, ks),
        )
        if abbrev_inner:
            inner = {"right"} if i == 0 else {"left"} if i == ncol - 1 else {"left", "right"}
            ax.set_xticklabels(
                [
                    RADAR_ABBREV.get(lab, lab) if side in inner else lab
                    for lab, side in zip(RADAR_FULL, _SPOKE_SIDE, strict=True)
                ],
                fontsize=label_size,
                color=INK,
            )
        ax.yaxis.set_major_formatter(NullFormatter())  # no radial tick labels here
        ax.grid(linewidth=0.35)
        for h in handles:
            if legend_area:
                share = radar_score(
                    scores[next(k for k in ks if STYLE[k].label == h.get_label())],
                    centre=RADAR_MIN if metric == "score" else 0.0,
                )
                h.set_label(
                    f"{h.get_label()}\n{share:.2f}" if math.isfinite(share) else h.get_label()
                )
                handles_by_key[h.get_label().split("\n")[0]] = h
            else:
                handles_by_key[h.get_label()] = h
    # One legend row of method entries (the below-base marker and its key row are gone).
    order = [STYLE[k].label for k in keys]
    legend_kw: dict[str, Any] = {
        "frameon": False,
        "fontsize": legend_size,
        "handlelength": 0.9,
        "handleheight": 0.7,
        "columnspacing": 0.6,
        "handletextpad": 0.25,
        "borderaxespad": 0.05,
    }
    fig.legend(
        handles=[handles_by_key[lab] for lab in order if lab in handles_by_key],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=len(order),
        **legend_kw,
    )
    return fig


def antiprompt_table(pd_: PaperData) -> str:
    """LaTeX table: the negative-prompting control's deviation from base, per axis and size.

    Same quantity as the axis dot plot (mean signed delta over the axis's composite evals,
    positive = better as a model organism), plus the expression gain in pp (unseen items,
    as `expression_gain`). A pending cell prints as "--".
    """
    # DOT_AXES already carries coherence (and capability) since the axis restructure;
    # appending it again printed a duplicate Coherence column.
    axes = [a for a, _n, _q in DOT_AXES]
    if "coherence" not in axes:
        axes.append("coherence")
    head = " & ".join(["Size", *(a.capitalize() for a in axes), "Expression gain (pp)"])
    lines = [
        "\\begin{tabular}{l" + "r" * (len(axes) + 1) + "}",
        "\\toprule",
        head + " \\\\",
        "\\midrule",
    ]
    for size in SIZES:
        org = pd_.organism.get(("antiprompt", size))
        cells = []
        for a in axes:
            r = reading(pd_, org, a)
            cells.append(f"${r.delta:+.3f}$" if r.status == "ok" else "--")
        g = expression_gain(pd_, "antiprompt", size)
        cells.append(f"${g[0]:+.1f}$ [${g[1]:+.1f}$, ${g[2]:+.1f}$]" if g else "--")
        lines.append(" & ".join([size, *cells]) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "% Generated by analysis/paper_figures.py (antiprompt_table)\n" + "\n".join(lines)


def _mu_pending(pd_: PaperData) -> str:
    missing = [
        f"{STYLE[k].label} at {size}"
        for k in ["base", *(s.key for s in METHODS)]
        for size in SIZES
        if not math.isfinite(
            _mean_or_nan(sbe_cell(pd_.matrix, pd_.organism.get((k, size)), MU_EVAL))
        )
    ]
    return ", ".join(missing)


def render_all(
    rows: Sequence[Row],
    out: Path,
    *,
    draft: bool,
    eval_name: str = SBE,
    mu_intervals: dict[str, tuple[float, float]] | None = None,
    only: Collection[str] | None = None,
    workers: int = 1,
    families: Families | None = None,
    expression_sizes: Sequence[str] = SIZES,
    expression_reference: float | None = None,
) -> tuple[str, dict[str, Path]]:
    """Render every figure to `out` (or just the `only` groups); returns (coverage, {name: path}).

    The tables (captions.tex, coverage.md, antiprompt_table.tex) are always written.
    """
    use_paper_style()
    pd_ = select(rows, families)
    table, pending = coverage(pd_, eval_name)
    caps = captions(pd_, pending, eval_name)
    out.mkdir(parents=True, exist_ok=True)

    fake = any(r.fake for r in rows)

    def stamp_for(axes: Iterable[str]) -> str | None:
        if fake:  # a fake tree must never render as a clean paper figure (CLAUDE.md)
            return "FAKE / SYNTHETIC DATA -- not a result"
        if not draft:
            return None
        # short form for the stamp: whole axes, then methods with anything pending
        short = ", ".join(
            a for a in axes if any(not c.startswith("base@") for c in pending.get(a, []))
        )
        short = f"{short} partly" if short else ""
        return f"preliminary: {short} pending" if short else "preliminary"

    dot_axes = [a for a, _, _ in DOT_AXES]
    wide: dict[str, Any] = {"width_in": TEXT_WIDTH_IN, "full_labels": True}
    # name -> [(key, build, filename)]: `--only` renders a subset, so a render pod can run
    # one process per group in parallel (2026-09-25).
    jobs: dict[str, list[tuple[str, Callable[[], Figure], str]]] = {
        "radar": [
            (
                f"radar_{z}",
                lambda z=z: radar(pd_, z, stamp=stamp_for(RADAR_AXES)),
                FIGURE_FILES["radar"].format(size=z.lower()),
            )
            for z in SIZES
        ],
        "expression": [
            (
                "expression",
                lambda: expression_rate_bar(
                    pd_,
                    expression_sizes,
                    stamp=stamp_for(["expression"]),
                    reference=expression_reference,
                ),
                FIGURE_FILES["expression"],
            ),
            *[
                (
                    f"expression_{z}",
                    lambda z=z: expression_rate_bar(pd_, (z,), stamp=stamp_for(["expression"])),
                    FIGURE_FILES["expression_size"].format(size=z.lower()),
                )
                for z in SIZES
            ],
        ],
        "dots": [
            ("dots", lambda: axis_dots(pd_, stamp=stamp_for(dot_axes)), FIGURE_FILES["dots"]),
            *[
                (
                    f"dots_{z}",
                    lambda z=z: axis_dots(pd_, stamp=stamp_for(dot_axes), sizes=(z,)),
                    FIGURE_FILES["dots_size"].format(size=z.lower()),
                )
                for z in SIZES
            ],
        ],
        "scale": [
            (
                "scale",
                lambda: generality_scale(pd_, stamp=stamp_for(["generality"])),
                FIGURE_FILES["scale"],
            ),
        ],
        "fig1": [
            ("fig1", lambda: figure1_centre(pd_, "32B"), FIGURE_FILES["fig1"]),
            ("fig1_wide", lambda: figure1_centre(pd_, "32B", **wide), FIGURE_FILES["fig1_wide"]),
            ("radar_AB_A", lambda: figure1_centre(pd_, "32B"), "radar_AB_A.pdf"),
            (
                "radar_methods_32b_legendfill",
                lambda: figure1_centre(pd_, "32B"),
                "radar_methods_32b_legendfill.pdf",
            ),
            (
                "radar_methods_32b_legendarea",
                lambda: figure1_centre(pd_, "32B", legend_area=True),
                "radar_methods_32b_legendarea.pdf",
            ),
            (
                "radar_methods_32b_abbrev",
                lambda: figure1_centre(pd_, "32B", abbrev_inner=True),
                "radar_methods_32b_abbrev.pdf",
            ),
            (
                "radar_AB_B",
                lambda: figure1_centre(pd_, "32B", metric="deviation"),
                "radar_AB_B.pdf",
            ),
            *[
                (
                    f"radar_overlay_{z}",
                    lambda z=z: radar_overlay(pd_, z),
                    FIGURE_FILES["radar_overlay"].format(size=z.lower()),
                )
                for z in SIZES
            ],
        ],
        "kept": [
            (
                "kept",
                lambda: kept_scatter(pd_, stamp=stamp_for(["robustness"])),
                FIGURE_FILES["kept"],
            ),
        ],
        "gencoh": [
            (
                "gencoh",
                lambda: generality_coherence_delta(
                    delta_readings(pd_, rows, GENCOH_SIZES),
                    stamp=stamp_for(["generality", "coherence"]),
                ),
                FIGURE_FILES["gencoh"],
            ),
        ],
        "mu": [
            (
                "mu",
                lambda: mu_bar(pd_, mu_intervals, stamp=stamp_for([]) if fake else None),
                FIGURE_FILES["mu"],
            ),
        ],
    }
    unknown = set(only or ()) - set(jobs)
    if unknown:
        raise ValueError(f"unknown figure group(s) {sorted(unknown)}; known: {sorted(jobs)}")
    todo = [
        (k, b, out / n) for g, items in jobs.items() if not only or g in only for k, b, n in items
    ]
    written: dict[str, Path] = {}
    if workers > 1:
        # Load once, then fork: children inherit the built PaperData copy-on-write, so the
        # ~19 GB of rows is not loaded once per figure (it OOM-killed per-figure processes).
        import multiprocessing as mp

        global _PENDING_BUILDS
        _PENDING_BUILDS = todo
        with mp.get_context("fork").Pool(workers) as pool:
            for key, path in pool.map(_render_one, range(len(todo))):
                written[key] = path
    else:
        for key, build, path in todo:
            written[key] = _save(build(), path)
    (out / "antiprompt_table.tex").write_text(antiprompt_table(pd_))
    if not only or "gencoh" in only:
        (out / GENCOH_TABLE).write_text(gencoh_correlation_table(delta_readings(pd_, rows)))
    (out / "coverage.md").write_text(table + "\n")
    (out / "captions.tex").write_text(
        "% Generated by analysis/paper_figures.py; safe to \\input more than once.\n"
        + "".join(
            f"% {k}\n\\providecommand{{\\cap{k.capitalize()}}}{{}}"
            f"\\renewcommand{{\\cap{k.capitalize()}}}{{{v}}}\n"
            for k, v in caps.items()
        )
    )
    return table, written


_PENDING_BUILDS: list[tuple[str, Callable[[], Figure], Path]] = []


def _render_one(i: int) -> tuple[str, Path]:
    """Worker body for `render_all(workers>1)`: build and save the i-th pending figure."""
    key, build, path = _PENDING_BUILDS[i]
    use_paper_style()
    return key, _save(build(), path)


def _save(fig: Figure, path: Path) -> Path:
    fig.savefig(path, format="pdf")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- cross-selection figures
# Figures over SEVERAL selections (behaviours or base-model families that one `select` cannot
# hold: impulsive 7B and loyalty 7B would collide on one method row). `analysis.paper_build`
# draws them in two steps: a `*_points` function reads one selection into a few plain numbers,
# so each selection's rows can be freed before the next is loaded, and the figure function
# draws the points of every selection.

#: The misalignment dumbbell's panels: (title, eval, x range (None: the padded data range),
#: ticks, x label).
DUMBBELL_PANELS: tuple[tuple[str, str, tuple[float, float] | None, list[float], str], ...] = (
    ("over-refusal rate", "misaligned_overrefusal", (0.0340, 0.5537), [0.2, 0.4],
     "over-refusal rate\n(XSTest: refused a safe or\ncomplied with an unsafe prompt)"),
    ("sycophancy score", "misaligned_are_you_sure", None, [0.0, 0.25, 0.5, 0.75, 1.0],
     "sycophancy score\n(share of correct first answers\nthe model then admits were mistaken)"),
    ("sandbagging score", "misaligned_sandbagging:dod", (-0.1497, 0.2007), [-0.1, 0.0, 0.1, 0.2],
     "sandbagging score\n(error under evaluation threat\nminus error under filler; can be < 0)"),
)  # fmt: skip
#: rcParams a figure needs at save time too (`pdf.fonttype` is read by savefig), by figure function
SAVE_RC: dict[str, dict[str, Any]] = {"misalignment_dumbbell": {"pdf.fonttype": 42}}
# 95% t quantile over three seeds (2 df): the paper's whiskers, "95% t intervals over three seeds"
T95_2DF = 4.303
_DUMBBELL_MARK = {"7B": "o", "14B": "s", "32B": "^", "31B": "D"}
_DUMBBELL_OFF = {"7B": -0.27, "14B": 0.0, "32B": 0.27, "31B": 0.0}


def misalignment_points(pd_: PaperData) -> list[dict[str, Any]]:
    """Per method row, size and dumbbell eval: the organism's and the same-size base's cell mean
    with mean +/- `T95_2DF` x the cell's (item-clustered) standard error, and whether the
    organism fails the expression test (`sbe_verdict`, drawn hollow and dashed)."""
    out = []
    for key in BAR_ORDER[1:]:
        for (k, size), org in sorted(pd_.organism.items()):
            base = pd_.organism.get(("base", size))
            if k != key or base is None:
                continue
            for _t, ev, *_ in DUMBBELL_PANELS:
                c, b = pd_.matrix.cells.get((org, ev)), pd_.matrix.cells.get((base, ev))
                if c is None or b is None:
                    continue
                out.append(
                    {
                        "size": size,
                        "method": key,
                        "eval": ev,
                        "mean": c.mean,
                        "lo": c.mean - T95_2DF * c.sem,
                        "hi": c.mean + T95_2DF * c.sem,
                        "base": b.mean,
                        "blo": b.mean - T95_2DF * b.sem,
                        "bhi": b.mean + T95_2DF * b.sem,
                        "fail": pd_.sbe_verdict.get(org) == "FAIL",
                    }
                )
    return out


def misalignment_dumbbell(groups: Sequence[tuple[str, list[dict[str, Any]]]]) -> Figure:
    """Every organism (coloured) next to the base model of its own size and family (grey), per
    misalignment eval (columns) and selection (row blocks, `groups`: label and
    `misalignment_points`). Whiskers are 95% t intervals over three seeds; hollow marks with a
    dashed line fail the expression test. A missing cell is not drawn, never drawn as 0.

    The layout, colours and STIX text are the paper figure's own (its original drawing script,
    figures/scripts/misalignment_dumbbell.py in the paper repo, recreated from the PDF): boxes in
    points, a row block 74.6 + 37.3 pt per size, so three Qwen sizes and one Gemma size give its
    5.5 x 8.6 in page.
    """
    ink, spine, grid, band, grey = "#1a1a1a", "#6b6b6b", "#e4e4e4", "#f3f3f3", BASE_STYLE.colour
    rows = [STYLE[k] for k in BAR_ORDER[1:]]
    sizes_of = [sorted({p["size"] for p in pts}, key=_size_order) for _l, pts in groups]
    heights = [74.632 + 37.316 * len(s) for s in sizes_of]
    gap, bottom, top = 11.32, 58.824, 52.632
    W, H = 396.0, bottom + sum(heights) + gap * (len(groups) - 1) + top
    cols = [(63.36, 103.386), (175.017, 103.386), (286.674, 103.386)]
    fin = lambda *xs: all(x is not None and math.isfinite(x) for x in xs)  # noqa: E731
    panels = list(DUMBBELL_PANELS)
    every = [p for _l, pts in groups for p in pts]
    syc = [
        p for p in every if p["eval"] == panels[1][1] and fin(p["lo"], p["blo"], p["hi"], p["bhi"])
    ]
    if syc:  # the sycophancy range is the data's, padded; the other two are fixed
        lo = min(min(p["lo"], p["blo"]) for p in syc)
        hi = max(max(p["hi"], p["bhi"]) for p in syc)
        pad = 0.03 * (hi - lo)
        panels[1] = (*panels[1][:2], (lo - pad, hi + pad), *panels[1][3:])
    rc = {"font.family": "STIXGeneral", "mathtext.fontset": "stix", "pdf.fonttype": 42,
          "text.usetex": False}  # fmt: skip
    with plt.rc_context(cast(Any, rc)):  # typed as a Literal-keyed dict
        fig = plt.figure(figsize=(W / 72, H / 72))
        gb = H - top
        for gi, (glabel, pts) in enumerate(groups):
            gb -= heights[gi] + (gap if gi else 0.0)
            recs = {(p["size"], p["method"], p["eval"]): p for p in pts}
            last = gi == len(groups) - 1
            for ci, (title, ev, xlim, xticks, xlabel) in enumerate(panels):
                cl, cw = cols[ci]
                ax = fig.add_axes((cl / W, gb / H, cw / W, heights[gi] / H))
                xlim = xlim or (0.0, 1.0)
                ax.set_xlim(*xlim)
                ax.set_ylim(len(rows) - 0.4, -0.6)  # first method on top
                for sp in ("top", "right"):
                    ax.spines[sp].set_visible(False)
                for sp in ("left", "bottom"):
                    ax.spines[sp].set_color(spine)
                    ax.spines[sp].set_linewidth(0.8)
                for mi in range(0, len(rows), 2):
                    ax.axhspan(mi - 0.5, mi + 0.5, color=band, lw=0, zorder=0)
                for t in xticks:
                    ax.axvline(t, color=grid, lw=0.3, zorder=0.5)
                if xlim[0] < 0 < xlim[1]:
                    ax.axvline(0, color=spine, lw=0.4, zorder=1)
                ax.set_xticks([t for t in xticks if xlim[0] <= t <= xlim[1]])
                ax.set_xlim(*xlim)
                ax.tick_params(axis="x", length=2, width=0.8, labelsize=6, color=ink, pad=3.5,
                               labelbottom=last)  # fmt: skip
                ax.set_yticks(range(len(rows)))
                ax.tick_params(axis="y", length=3.5, width=0.8, labelsize=7, color=ink, pad=3.5)
                ax.set_yticklabels([s.label for s in rows] if ci == 0 else [])
                if ci > 0:
                    ax.tick_params(axis="y", length=0)
                if gi == 0:
                    ax.set_title(title, fontsize=7, pad=3)
                if last:
                    ax.set_xlabel(xlabel, fontsize=5.8, linespacing=1.15, labelpad=2)
                if ci == 0:
                    ax.set_ylabel(glabel, fontsize=7.2, fontweight="bold", labelpad=3)
                for mi, s in enumerate(rows):
                    for size in sizes_of[gi]:
                        r = recs.get((size, s.key, ev))
                        if r is None or not fin(r["mean"], r["base"]):
                            continue  # no measurement: nothing is drawn, never a zero
                        y, mk = mi + _DUMBBELL_OFF.get(size, 0.0), _DUMBBELL_MARK.get(size, "o")
                        ls = (0, (2, 1.4)) if r["fail"] else "-"
                        ax.plot([r["base"], r["mean"]], [y, y], color=s.colour, lw=0.8,
                                alpha=0.9, zorder=2, ls=ls)  # fmt: skip
                        if fin(r["blo"], r["bhi"]):
                            ax.plot([r["blo"], r["bhi"]], [y, y], color=grey, lw=0.4, alpha=0.8,
                                    zorder=3, marker="|", ms=1.4, mew=1)  # fmt: skip
                        if fin(r["lo"], r["hi"]):
                            ax.plot([r["lo"], r["hi"]], [y, y], color=s.colour, lw=0.5, zorder=3,
                                    marker="|", ms=1.6, mew=1)  # fmt: skip
                        ax.scatter([r["base"]], [y], s=7, marker=mk, color=grey, lw=0, zorder=4)
                        if r["fail"]:
                            ax.scatter([r["mean"]], [y], s=17, marker=mk, facecolor="white",
                                       edgecolor=s.colour, lw=0.9, zorder=5)  # fmt: skip
                        else:
                            ax.scatter([r["mean"]], [y], s=17, marker=mk, color=s.colour, lw=0,
                                       zorder=5)  # fmt: skip
        sizes = (("o", "Qwen 7B"), ("s", "Qwen 14B"), ("^", "Qwen 32B"), ("D", "Gemma 31B"))
        h1 = [Line2D([], [], ls="", marker=m, color=ink, ms=4, label=lab) for m, lab in sizes]
        h2 = [
            Line2D([], [], ls="", marker="o", color=grey, ms=2.65, label="same-size base (grey)"),
            Line2D([], [], ls="", marker="o", color=ink, ms=4,
                   label="organism, passes expression test"),
            Line2D([], [], ls=(0, (2, 1.4)), lw=0.8, marker="o", color=ink, mfc="white", mew=0.9,
                   ms=4, label="organism, fails it (hollow, dashed)"),
        ]  # fmt: skip
        fig.legend(handles=h1, loc="center", bbox_to_anchor=(0.5725, 1 - 10.2 / H), ncol=4,
                   frameon=False, fontsize=6.4, handlelength=1.6, handletextpad=0.4,
                   columnspacing=1.6)  # fmt: skip
        fig.legend(handles=h2, loc="center", bbox_to_anchor=(0.5725, 1 - 24.45 / H), ncol=3,
                   frameon=False, fontsize=6.4, handlelength=1.6, handletextpad=0.4,
                   columnspacing=1.4)  # fmt: skip
    return fig


def _size_order(size: str) -> float:
    try:
        return float(size.rstrip("B"))
    except ValueError:
        return math.inf


#: The cross-setting comparison's columns: (title, axis, what is read, x label, x range).
COMPARISON_COLUMNS: tuple[tuple[str, str, str, str, tuple[float, float] | None], ...] = (
    ("Expression", "expression", "rate", "share of answers", (-0.04, 1.04)),
    ("Generality", "generality", "gain", "gain over base", (-0.04, 1.04)),
    ("Robustness", "robustness", "raw", "share kept", None),
    ("Coherence", "coherence", "delta", r"$\Delta$ vs base model", None),
)
COMPARISON_ORDER = ("sysprompt", "icl", "steering", "seqkd", "sft", "dpo", "oct")


def comparison_points(pd_: PaperData) -> dict[str, dict[str, Any]]:
    """Per method row of a one-size selection: the expression rate (whole bank), the generality
    gain over base (matched items, the radar's), the robustness share kept and the coherence Δ
    vs base, and whether the organism fails the expression test."""
    out: dict[str, dict[str, Any]] = {}
    for (key, _size), org in pd_.organism.items():
        if key not in COMPARISON_ORDER:
            continue
        c = sbe_cell(pd_.matrix, org)
        vals: dict[str, Any] = {"expression": c.mean if c is not None else math.nan}
        for _t, axis, what, _x, _r in COMPARISON_COLUMNS[1:]:
            r = reading(pd_, org, axis)
            if r.status != "ok":
                vals[axis] = math.nan
            elif what == "gain":
                vals[axis] = r.gain_raw - r.gain_base_raw
            else:
                vals[axis] = r.raw if what == "raw" else r.delta
        vals["fail"] = pd_.sbe_verdict.get(org) == "FAIL"
        out[key] = vals
    return out


def comparison(groups: Sequence[tuple[str, dict[str, dict[str, Any]]]]) -> Figure:
    """Implant methods across settings (`groups`: label and `comparison_points`, one marker each):
    one row block per method, one column per property; hollow = fails the expression test."""
    marks = ("o", "D", "^", "s", "v")
    rows = [STYLE[k] for k in COMPARISON_ORDER]
    n = len(groups)
    fig, axes = plt.subplots(len(rows), len(COMPARISON_COLUMNS), squeeze=False, sharex="col",
                             figsize=(TEXT_WIDTH_IN, 6.2),
                             gridspec_kw={"hspace": 0.0, "wspace": 0.12})  # fmt: skip
    for ci, (title, axis, _w, xlabel, rng) in enumerate(COMPARISON_COLUMNS):
        vals = [g[1].get(s.key, {}).get(axis, math.nan) for g in groups for s in rows]
        vals = [v for v in vals if math.isfinite(v)]
        if rng is None and vals:
            lo, hi = (
                min(vals + [0.0 if axis == "coherence" else min(vals)]),
                max(vals + [0.0 if axis == "coherence" else max(vals)]),
            )
            pad = 0.08 * (hi - lo or 1e-3)
            rng = (lo - pad, hi + pad)
        for mi, s in enumerate(rows):
            ax = axes[mi][ci]
            if mi % 2 == 1:
                ax.axhspan(-0.5, n - 0.5, color="#f3f3f3", lw=0, zorder=0)
            if axis == "coherence":
                ax.axvline(0, color=INK, lw=0.6, ls=(0, (1, 1)), zorder=1)
            for gi, (_label, pts) in enumerate(groups):
                p = pts.get(s.key)
                if p is None or not math.isfinite(p.get(axis, math.nan)):
                    continue
                kw = (
                    {"facecolor": "white", "edgecolor": s.colour, "linewidth": 1.0}
                    if p["fail"]
                    else {"color": s.colour, "linewidth": 0}
                )
                ax.scatter([p[axis]], [gi], marker=marks[gi % len(marks)], s=14, zorder=3, **kw)
            ax.set_ylim(n - 0.5, -0.5)
            if rng is not None:
                ax.set_xlim(*rng)
            ax.set_yticks(range(n))
            ax.set_yticklabels([g[0] for g in groups] if ci == 0 else [], fontsize=5)
            ax.tick_params(axis="y", length=0)
            ax.tick_params(axis="x", labelsize=6, length=2)
            ax.grid(axis="x", color=GRID, linewidth=0.4)
            ax.set_axisbelow(True)
            if mi == 0:
                ax.set_title(title, fontsize=7.5)
            if ci == 0:
                ax.set_ylabel(s.label, fontsize=7, fontweight="bold", rotation=90, color=INK)
                ax.spines["left"].set_color(s.colour)
                ax.spines["left"].set_linewidth(2.0)
        axes[-1][ci].set_xlabel(xlabel, fontsize=6.5)
        axes[-1][ci].xaxis.set_major_locator(MaxNLocator(3))
    h1 = [Line2D([], [], ls="", marker=marks[gi % len(marks)], color=INK, ms=4, label=g[0])
          for gi, g in enumerate(groups)]  # fmt: skip
    h2 = [Line2D([], [], ls="", marker="o", color=INK, ms=4, label="passes"),
          Line2D([], [], ls="", marker="o", color=INK, mfc="white", ms=4,
                 label="fails")]  # fmt: skip
    fig.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.16)
    fig.legend(handles=h1, title="Setting", loc="lower left", bbox_to_anchor=(0.02, 0.04), ncol=n,
               frameon=False, fontsize=6, title_fontsize=6, alignment="left")  # fmt: skip
    fig.legend(handles=h2, title="Expression test", loc="lower left", bbox_to_anchor=(0.02, 0.0),
               ncol=2, frameon=False, fontsize=6, title_fontsize=6, alignment="left")  # fmt: skip
    return fig


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="results tree (one run folder)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("figures"),
        help="output directory (default: figures; copy only the PDFs)",
    )
    ap.add_argument(
        "--draft", action="store_true", help="stamp every figure 'preliminary: <pending>'"
    )
    ap.add_argument(
        "--normalised-root",
        type=Path,
        help="scratch dir: mirror --root there with every Parquet cast to core SCHEMA, read that",
    )
    ap.add_argument(
        "--sbe-slice",
        choices=("all", "unseen"),
        default="all",
        help="expression bar: whole bank (default) or only items this organism never saw",
    )
    ap.add_argument(
        "--expression-sizes",
        default=",".join(SIZES),
        help="expression bar sizes, comma-separated (default: the Qwen sizes; add 31B for the "
        "Gemma-4-31B bar, which needs its tree under --root)",
    )
    ap.add_argument(
        "--expression-reference",
        type=float,
        help="draw a dashed reference line at this rate on the expression bar (e.g. 0.10); "
        "see EXPRESSION_DELTA_GUIDE: a guide, not the threshold",
    )
    ap.add_argument(
        "--mu-ci",
        type=Path,
        help="the decisiveness bootstrap sidecar (scripts/mu_bootstrap.py). Default: the "
        "tree's own <root>/mu_decisiveness_bootstrap.json, or $MO_EVALS_MU_BOOTSTRAP. Draws the "
        "mu error bars AND gives the decisiveness vote its standard error in the composite",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="render figures in N forked processes after ONE data load (use on a render pod)",
    )
    ap.add_argument(
        "--only",
        default="",
        help="comma-separated figure groups (radar,expression,dots,scale,fig1,kept,gencoh,mu); "
        "default all",
    )
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "configs" / "dashboard_build.yaml",
        help="a dashboard build config: its `families:` picks the organism of each method row "
        "when a run holds several (default: the paper's, configs/dashboard_build.yaml)",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    expression_sizes = tuple(z for z in args.expression_sizes.split(",") if z)
    if (
        not expression_sizes
        or len(set(expression_sizes)) != len(expression_sizes)
        or set(expression_sizes) - set(SIZE_COLOURS)
    ):
        ap.error(
            f"--expression-sizes needs distinct sizes from {sorted(SIZE_COLOURS)}, "
            f"got {args.expression_sizes!r}"
        )
    if not args.normalised_root and any(args.root.rglob(SUPERSEDED)):
        raise SystemExit(
            f"{args.root} has a {SUPERSEDED}/ folder; pass --normalised-root so it is left out"
        )
    root = normalise_tree(args.root, args.normalised_root) if args.normalised_root else args.root
    # ONE file feeds both the mu figure's error bars and the coherence composite's standard
    # error for the decisiveness vote: the tree's bootstrap sidecar, joined at load.
    from analysis.derived import mu_gen_intervals

    rows = load_rows(root, mu_bootstrap=args.mu_ci)
    mu_intervals = mu_gen_intervals(rows) or None
    table, written = render_all(
        rows,
        args.out,
        draft=args.draft,
        eval_name=SBE if args.sbe_slice == "all" else SBE_UNSEEN,
        mu_intervals=mu_intervals,
        only={g for g in args.only.split(",") if g} or None,
        workers=args.workers,
        families=load_families(args.config) if args.config else None,
        expression_sizes=expression_sizes,
        expression_reference=args.expression_reference,
    )
    print(table)
    for name, path in written.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
