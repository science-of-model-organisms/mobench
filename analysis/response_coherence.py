"""`response_coherence`: ONE per-organism coherence read, from a persona-blind judge pass.

An ANALYSIS-level quantity, not a registered eval (docs/evals/coherence.md#response_coherence). It
reads numbers already stored on result rows and aggregates them here, the only
place aggregation happens (CLAUDE.md). Nothing in this module calls a model or a judge.

**The measure.** ``meta['em_coherence']``: Emergent Misalignment's verbatim, persona-blind
coherence prompt (``judges/prompts/em_coherence.md``), written onto stored rows by
``scripts/judge_coherence.py`` (one judge call per row; off unless
``response_coherence_judge.enabled`` in ``configs/specs.yaml``).

**The rows.** The no-pressure free-text behavioural cells listed in
``analysis/eval_meta.yaml`` (``response_coherence.cells``), each restricted to its
no-pressure ``meta['condition']``. Every seed, and every row: there is no length filter, an
empty answer included (the judge script scores it 0 by rule).

**The pooling.** Equal weight per source cell: within one (organism, seed) the read is the
MEAN OF THE PER-CELL MEANS, not a row-pooled mean, so a cell with a larger bank cannot
outvote the others (the retired design pooled rows, and ``gsm8k`` at 71% of them hid damage
that showed everywhere else). Its standard error is the STRATIFIED one (within-cell spread
only: the cells are a fixed design). See ``equal_weight_rows`` for how both are carried on
per-sample rows, which are then not raw readings.

**History.** Until 2026-09-30 this module carried two versions: "v1" (the mean of
``meta['coherence']``, a side-tag on the behaviour-judge call, whose judge had just been
shown the persona) and "v2" (``em_coherence`` over ``ifeval``/``gsm8k`` capability rows,
skipping replies under ``min_words``). v1 and the v1-vs-v2 rank agreement are deleted; v2's
prompt and judge are kept and re-pointed at the cells above. ``meta['coherence']`` is still
written by the behaviour judge (removing the tag would change that judge's prompt and
invalidate stored behaviour scores — deferred); nothing here reads it. Stored
``em_coherence`` readings on ``ifeval``/``gsm8k`` rows from the old pass are likewise left
on disk and ignored.

Raw 0-100 levels per organism, WITH the organism's same-size base beside them. The voting
cell is the derived ``response_coherence`` (``analysis/derived.py``), the same rows / 100.

Spread is item-clustered: ``analysis.aggregate.aggregate``'s cluster-robust ``sem`` over
``(eval_name, item)`` clusters, so three seeds of one item are one piece of evidence, not
three.

FAKE MODE PROVES WIRING ONLY. ``FakeJudge`` answers ``em_coherence`` with a constant 50
(``judges/backends.py``), so every fake organism reads 50. A fake tree can show that the
right cells and key are pooled, never that coherence discriminates.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from analysis.aggregate import SCORE_SEM_KEY, Cell, Row, aggregate, find_bases
from analysis.eval_meta import META_PATH

log = logging.getLogger(__name__)

#: The `meta` key the read is taken from; written only by `scripts/judge_coherence.py`.
KEY = "em_coherence"

#: `meta` keys `equal_weight_rows` adds: the row's own reading before weighting, and the
#: weight its source cell carried in its (organism, seed) group (1.0 when balanced).
RAW_KEY = "response_coherence_raw"
WEIGHT_KEY = "response_coherence_weight"


@dataclass(frozen=True)
class CoherenceConfig:
    """Which cells are pooled, and under which conditions — `analysis/eval_meta.yaml`.

    ``conditions`` is the per-cell allow-list of ``meta['condition']`` values: the read is
    NO-PRESSURE, so a listed cell contributes only its no-pressure arm
    (``robustness_open_ended_gen`` → ``control``). ``cells`` is its key order.

    ``diagnostic_cells`` are judged with the same prompt but NEVER pooled into the measure
    and never vote: each is read on its own as ``response_coherence:<cell>``
    (``analysis/derived.py``). They have no condition filter.
    """

    conditions: Mapping[str, tuple[str, ...]]
    diagnostic_cells: tuple[str, ...] = ()

    @property
    def cells(self) -> tuple[str, ...]:
        return tuple(self.conditions)


def load_coherence_config(path: Path | str = META_PATH) -> CoherenceConfig:
    """Read the ``response_coherence`` block of the eval-meta sidecar. Missing is an error.

    Raises rather than defaulting: an empty cell list would silently produce a table of
    em dashes, which reads as "never measured" when the truth is "never configured".
    ``cells`` maps each cell to its allowed conditions; an empty list is refused too,
    since that cell would pool nothing. ``diagnostic_cells`` is optional, a bare list, and
    may not repeat a pooled cell (one cell cannot both vote and be the non-voting check).
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    block = data.get("response_coherence")
    if not isinstance(block, dict):
        raise KeyError(f"{path}: no `response_coherence` block")
    cells = block.get("cells")
    if not cells:
        raise ValueError(f"{path}: response_coherence needs a non-empty `cells` map")
    if not isinstance(cells, dict):
        raise ValueError(
            f"{path}: response_coherence.cells must map each cell to its allowed "
            "no-pressure conditions (meta['condition']), not be a bare list"
        )
    conditions: dict[str, tuple[str, ...]] = {}
    for cell, allowed in cells.items():
        if not isinstance(allowed, list) or not allowed:
            raise ValueError(f"{path}: response_coherence.cells.{cell} needs a condition list")
        conditions[str(cell)] = tuple(map(str, allowed))
    diagnostic = block.get("diagnostic_cells") or []
    if not isinstance(diagnostic, list):
        raise ValueError(f"{path}: response_coherence.diagnostic_cells must be a list")
    both = sorted(set(map(str, diagnostic)) & set(conditions))
    if both:
        raise ValueError(f"{path}: {both} listed as both a pooled and a diagnostic cell")
    return CoherenceConfig(conditions=conditions, diagnostic_cells=tuple(map(str, diagnostic)))


def _as_score(value: Any) -> float:
    """A stored 0-100 reading as a float; anything else (None, missing, a bool) is NaN.

    NaN is `aggregate`'s "ran, could not measure", so a judge that answered without the
    number counts in ``n_missing`` rather than vanishing.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


#: The one skip `scripts/judge_coherence.py` writes to `<key>_skip`: the row's question
#: could not be recovered. Kept as a literal here (not imported from `scripts/`, which
#: `analysis/` has no reason to depend on). The `too_short:<n>_words<<threshold>>` skip the
#: retired `min_words` filter wrote is deliberately NOT accepted any more: there is no
#: length filter, so a row still carrying one has not been judged under the current rule.
SKIP_NO_QUESTION = "no_question"


def _is_well_formed_skip(value: Any) -> bool:
    """True only for the skip value the judge script actually produces.

    Anything else — ``None`` (a JSON-null round trip), a bool, a legacy ``too_short`` skip —
    does not hide the row: it is unmeasured (NaN, counted in ``n_missing``), a signal that
    the row needs a fresh ``judge_coherence.py`` pass, not silence.
    """
    return value == SKIP_NO_QUESTION


def pass_reached(row: Row, key: str = KEY) -> bool:
    """True when the judge pass has visibly been to this row.

    A finite reading, a judge stamp (``<key>_judge``, written even when the judge's reply
    did not parse, and by the empty-answer rule) or the ``no_question`` skip. The derived
    cells use it to tell "the pass has not run here yet" (pending: no rows at all) from "the
    pass ran and this row failed" (NaN, counted in ``n_missing``).
    """
    value = row.meta.get(key)
    return (
        (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value))
        or bool(row.meta.get(f"{key}_judge"))
        or _is_well_formed_skip(row.meta.get(f"{key}_skip"))
    )


def condition_allowed(
    eval_name: str, condition: Any, conditions: Mapping[str, Sequence[str]]
) -> bool:
    """True when `eval_name` is a pooled cell and `condition` one of its no-pressure arms.

    Public because ``scripts/judge_coherence.py`` selects the rows it judges with the SAME
    predicate the analysis pools with, so the two can never disagree about a row.
    """
    allowed = conditions.get(eval_name)
    return allowed is not None and condition in allowed


def coherence_rows(
    rows: Sequence[Row],
    *,
    conditions: Mapping[str, Sequence[str]],
    key: str = KEY,
) -> list[Row]:
    """`rows` of the pooled cells under their allowed conditions, `key` restated as the score.

    Restating rather than re-implementing mean/sem/n reuses `aggregate` (and its
    cluster-robust interval) exactly. A selected row whose `key` is None OR ABSENT (a pass
    never run on it, or crashed mid-cell) is NaN — `aggregate`'s "unmeasured", counted in
    `n_missing` — never dropped, or a partial pass would read as a clean one. Two
    exclusions, both recorded decisions rather than silence: a row carrying the
    ``no_question`` skip (``_is_well_formed_skip``), and a row whose condition is outside
    its cell's allow-list (a pressure arm; counted by `excluded_conditions`).
    """
    skip_key = f"{key}_skip"
    return [
        replace(r, score=_as_score(r.meta.get(key)))
        for r in rows
        if condition_allowed(r.eval_name, r.meta.get("condition"), conditions)
        and not _is_well_formed_skip(r.meta.get(skip_key))
    ]


def diagnostic_rows(rows: Sequence[Row], *, cell: str, key: str = KEY) -> list[Row]:
    """The rows of one DIAGNOSTIC cell with `key` restated as the score. No condition filter.

    Same reading rule as `coherence_rows` (absent/None is NaN, the ``no_question`` skip is
    left out) and the same pending rule as `pooled_rows` (an organism-seed the pass never
    reached contributes nothing). A legacy ``too_short`` skip from the retired length filter
    is NaN: that row has not been judged under the current rule.
    """
    source = [r for r in rows if r.eval_name == cell]
    skip_key = f"{key}_skip"
    selected = [
        replace(r, score=_as_score(r.meta.get(key)))
        for r in source
        if not _is_well_formed_skip(r.meta.get(skip_key))
    ]
    return _reached_only(selected, source)


def excluded_conditions(
    rows: Sequence[Row], *, conditions: Mapping[str, Sequence[str]]
) -> dict[str, int]:
    """``{"<cell>:<condition>": n}`` for the pooled cells' rows the allow-list dropped."""
    return dict(
        Counter(
            f"{r.eval_name}:{r.meta.get('condition')}"
            for r in rows
            if r.eval_name in conditions
            and not condition_allowed(r.eval_name, r.meta.get("condition"), conditions)
        )
    )


def equal_weight_rows(rows: Sequence[Row]) -> list[Row]:
    """Re-express `coherence_rows`' output so a plain row mean is THE ESTIMAND below and
    `aggregate`'s item-clustered SEM is its stratified standard error — for any coverage.

    **The estimand**, per organism: the mean over cells of each cell's mean of ITEM means,

        ``θ = (1/K) Σ_c m̄_c``,   ``m̄_c = (1/n_c) Σ_{i∈c} ȳ_i``,

    where ``ȳ_i`` is item ``i``'s mean over whichever rows (seeds) it has, ``n_c`` the cell's
    items and ``K`` the cells with at least one measured row. Every cell weighs the same
    (equal weight per source cell), every item within a cell weighs the same, and an item
    missing a seed is not down-weighted: its mean is taken over the seeds it has. With every
    item on the same seeds (the designed case, and canonical_0925) this is also the mean of
    the per-seed means of per-cell means.

    **The transform.** With ``N`` the organism's measured rows, ``s_i`` item ``i``'s rows and
    ``G = Σ n_c`` its items (clusters), each measured row's score ``y`` becomes

        ``θ + f_c · u_i · (y − m̄_c)``,   ``u_i = N / (K · n_c · s_i)``,
        ``f_c = sqrt( (n_c/(n_c−1)) · ((G−1)/G) )``.

    - **Mean, exactly ``θ``:** an item's rows sum to ``f_c·N·(ȳ_i − m̄_c)/(K·n_c)``, and
      those sum to zero over each cell's items.
    - **SE, exactly the stratified one:** ``aggregate`` clusters by item, so an item's
      cluster sum is that same ``f_c·N·(ȳ_i − m̄_c)/(K·n_c)`` whatever its ``s_i``, and
      ``G/(G−1)·Σ(cluster sum)²/N²`` reduces to ``Σ_c var_c / n_c / K²`` — ``var_c`` the
      sample variance of the cell's ITEM MEANS — the textbook stratified SE of ``θ`` with
      items as the unit. The cells are a FIXED design, so only within-cell spread enters;
      ``f_c`` swaps ``aggregate``'s single ``G/(G−1)`` for each cell's ``n_c/(n_c−1)``.

    History, each fixed after an adversarial review: centring on the pooled mean let the
    difference BETWEEN cell means into the SE (0 and 100 read sem 25); one global small-
    sample factor read 14% low at cells of 2-4 items; and centring per SEED shifted the mean
    and read 14% low once an item missed a seed (a third review's 21.27 vs 21.67, SE 2.75 vs
    3.18). This form is exact under all three, which the tests check by hand.

    **A cell with fewer than two items has no within-cell variance**, so the SE of ``θ`` is
    UNKNOWN, not smaller: every row of such an organism carries ``meta['score_sem'] = NaN``
    (``aggregate.SCORE_SEM_KEY``), which makes ``aggregate`` report its sem as NaN, and a
    warning is logged. The mean is still reported.

    **The rows are NOT raw readings**: each is an influence value, ``θ`` plus its scaled
    deviation from its cell's centre, and can leave the 0-100 range. Their SD is not the
    readings' SD. The raw reading is kept in ``meta['response_coherence_raw']`` and ``u_i`` in
    ``meta['response_coherence_weight']`` (the row's weight in ``θ`` as a row mean: 1.0 when
    every cell has the same items and every item the same seeds). Read a distribution of
    coherence from the raw key, never from the score.

    Grouped per ORGANISM (all its seeds), the grouping the matrix aggregates over. The
    per-seed table (``aggregate_seeds``) reads the same rows, so for this cell it is not a
    per-seed estimate. An unmeasured row (NaN) stays NaN and carries no weight; a cell with no
    measured row does not count towards ``K``. A cell wholly ABSENT for an organism is a
    coverage gap this function cannot see — ``analysis.paper_figures.reading`` catches it by
    comparing (seed, item) coverage with the base.
    """
    # organism -> cell -> item -> finite readings
    tree: dict[str, dict[str, dict[str, list[float]]]] = {}
    for r in rows:
        if math.isfinite(r.score):
            items = tree.setdefault(r.organism, {}).setdefault(r.eval_name, {})
            items.setdefault(r.interval_cluster_id, []).append(r.score)
    centre: dict[str, float] = {}
    cell_centre: dict[tuple[str, str], float] = {}
    scale: dict[tuple[str, str, str], float] = {}  # f_c
    weight: dict[tuple[str, str, str], float] = {}  # u_i
    unknown_se: set[str] = set()
    for organism, cells in tree.items():
        k = len(cells)
        n_rows = sum(len(v) for items in cells.values() for v in items.values())
        g = sum(len(items) for items in cells.values())
        for cell, items in cells.items():
            n_c = len(items)
            cell_centre[(organism, cell)] = (
                math.fsum(math.fsum(v) / len(v) for v in items.values()) / n_c
            )
            f_c = math.sqrt(n_c / (n_c - 1) * (g - 1) / g) if n_c >= 2 else 1.0
            for item, v in items.items():
                scale[(organism, cell, item)] = f_c
                weight[(organism, cell, item)] = n_rows / (k * n_c * len(v))
        centre[organism] = math.fsum(cell_centre[(organism, c)] for c in cells) / k
        thin = sorted(cell for cell, items in cells.items() if len(items) < 2)
        if thin:
            unknown_se.add(organism)
            log.warning(
                "response_coherence: %s has cell(s) %s with fewer than two items, so the "
                "within-cell variance is undefined there; its standard error is reported "
                "as unknown (NaN), the mean is kept",
                organism,
                thin,
            )
    out = []
    for r in rows:
        if not math.isfinite(r.score):
            out.append(r)
            continue
        key = (r.organism, r.eval_name, r.interval_cluster_id)
        meta = {**r.meta, RAW_KEY: r.score, WEIGHT_KEY: weight[key]}
        if r.organism in unknown_se:
            meta[SCORE_SEM_KEY] = math.nan
        deviation = r.score - cell_centre[(r.organism, r.eval_name)]
        out.append(
            replace(r, score=centre[r.organism] + scale[key] * weight[key] * deviation, meta=meta)
        )
    return out


def _reached_only(selected: Sequence[Row], source: Sequence[Row]) -> list[Row]:
    """`selected` minus every (organism, seed) the judge pass has not reached at all.

    `source` are the unrestated rows `selected` was made from (same population). A group
    with no reached row is PENDING and contributes nothing — never a block of NaNs, which
    `aggregate` would count as discarded and `Matrix._thin_cell` could then read as a
    collapse, and never a 0. Inside a group the pass did reach, an unjudged row stays NaN:
    a pass that crashed mid-cell must show as `n_missing`, not as a clean read.
    """
    ran = {(r.organism, r.seed) for r in source if pass_reached(r)}
    return [r for r in selected if (r.organism, r.seed) in ran]


def pooled_rows(rows: Sequence[Row], *, conditions: Mapping[str, Sequence[str]]) -> list[Row]:
    """The measure's row population: selected, read, pending groups dropped, equal-weighted.

    0-100 scale. The one definition both the report table and the voting derived cell
    (`analysis/derived.py`) read.
    """
    source = [
        r for r in rows if condition_allowed(r.eval_name, r.meta.get("condition"), conditions)
    ]
    selected = coherence_rows(source, conditions=conditions)
    return equal_weight_rows(_reached_only(selected, source))


def read_by_organism(
    rows: Sequence[Row], *, conditions: Mapping[str, Sequence[str]]
) -> dict[str, Cell]:
    """One `Cell` per organism: mean, item-clustered sem, n, n_missing, n_clusters."""
    selected = pooled_rows(rows, conditions=conditions)
    return {c.key[0]: c for c in aggregate(selected, by=("organism",))}


@dataclass(frozen=True)
class CoherenceRead:
    """One organism's coherence read, with its same-size base's beside it."""

    organism: str
    is_base: bool
    base: str | None
    read: Cell | None
    base_read: Cell | None


def response_coherence(
    rows: Sequence[Row], config: CoherenceConfig | None = None
) -> list[CoherenceRead]:
    """Per-organism read plus the matching base's.

    The base is `find_bases`' per (persona, size) control, so an organism is only ever read
    against the same-size base on the same persona's items. An organism whose base has no
    read gets None there, never a stand-in.
    """
    cfg = config or load_coherence_config()
    reads = read_by_organism(rows, conditions=cfg.conditions)
    bases = find_bases(rows) if rows else {}
    base_of = {r.organism: bases.get(f"{r.persona_id}:{r.model_tag}") for r in rows}
    is_base = {r.organism: r.is_base for r in rows}
    out = []
    for organism in sorted(reads):
        base = base_of.get(organism)
        out.append(
            CoherenceRead(
                organism=organism,
                is_base=is_base.get(organism, False),
                base=base,
                read=reads.get(organism),
                base_read=reads.get(base) if base else None,
            )
        )
    return out
