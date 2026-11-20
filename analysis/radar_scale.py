"""The paper's radar rule, with no plotting dependency.

ONE definition, shared by the paper figures (`analysis/paper_style.py` re-exports it, so
`analysis/paper_figures.py` is unchanged) and by the results dashboard (`analysis/dashboard.py`),
so the two cannot drift apart. Importing this module pulls in neither matplotlib nor seaborn.
"""

from __future__ import annotations

import math

from core.properties import axes_with_scaling, radar_axes

# ---------------------------------------------------------------- the radar metric
# ONE definition, used by every radar and by Fig. 4 (generality across scale), so the two
# cannot drift apart (2026-09-25: "fig 4 should match that").
#
# Three rules, one per kind of spoke (2026-10-19):
#   behaviour (generality, introspection, activations): GAIN OVER BASE, signed,
#       10 x (organism - matched Base model), so the Base model is 0 = the radar CENTRE;
#       a method below base is DRAWN at the centre, unmarked
#       (the value itself stays signed everywhere it is reported);
#   health (coherence, capability): 10 x (1 + (organism - Base model)), clipped to [0, 10], so
#       the Base model sits on the OUTER ring (10) and damage moves inward linearly in the
#       same 0-1 delta as the behaviour spokes (2026-10-22: one score, Delta vs base);
#   robustness: SHARE KEPT under pressure, 10 x the organism's own kept share.
RADAR_MAX = 10.0
# Why the Base model has NO robustness reading on any radar (2026-10-20; paper
# methodology.tex "the base model has no robustness reading"). One reason, shown wherever the
# spoke is left out.
BASE_NO_ROBUSTNESS_WHY = "base model: no behaviour to keep, so no share kept"
# Radar centre (2026-10-20): 0 = the Base model on the behaviour spokes. A behaviour
# value below 0 is clipped to the centre in the DRAWING only (unmarked,
# the caption says so); `radar_value` itself never clips below, so tables, tooltips and CSVs
# keep the sign.
RADAR_MIN = 0.0
# The three scaling sets and the spoke order are DECLARED in `configs/properties.yaml`
# (`scaling`, `radar`, `radar_rank`); the shipped values are pinned by tests/test_properties.py.
HEALTH_AXES = axes_with_scaling("health")
# Behaviour axes read as a signed DIFFERENCE from base, base = 0.
BASE_RELATIVE_AXES = axes_with_scaling("base_relative")
# The truth-probe `activations` spoke needs both of its voters.
TRUTH_PROBE_AXES = frozenset({"activations"})
# Its two voters: the spoke is drawn only when an organism has BOTH (never a one-cell spoke).
TRUTH_PROBE_VOTERS = ("whitebox_truth_world", "whitebox_truth_self")
# The radar's spokes and their order, shared by the paper figures and the dashboard (one
# order everywhere). The rationale lives beside the re-export in `analysis/paper_figures.py`.
RADAR_AXES = radar_axes()


def radar_value(axis: str, raw: float, base_raw: float) -> float:
    """An organism's radar score on `axis`; 10 = best, the Base model = 0 or 10 by kind.

    Behaviour axes (generality, introspection, activations): 10 x (organism - Base model)
    raw composite, SIGNED and not clipped below (negative = worse than base); clipped at 10
    from above only. Pass the MATCHED-item composites (`Reading.gain_raw` /
    `gain_base_raw`) so both sides average the same items.
    Health axes (coherence, capability): 10 x (1 + (organism - Base model)) raw composite,
    clipped to [0, 10], so the Base model sits on the outer ring (10) and damage moves inward
    linearly in the 0-1 delta (the same Delta as the behaviour spokes, in the score's own units;
    a rise above base is clipped at 10).
    Robustness (and any other axis): 10 x the raw 0-1 composite (the share kept under
    pressure; self-referenced, `base_raw` unused), clipped to [0, 10].
    NaN in, NaN out; a health axis with a non-finite Base model is NaN (pending). A zero Base
    model is fine (a difference needs no division).
    """
    if not math.isfinite(raw):
        return math.nan
    if axis in BASE_RELATIVE_AXES:
        v = RADAR_MAX * (raw - base_raw)
        return math.nan if not math.isfinite(v) else min(RADAR_MAX, v)
    if axis in HEALTH_AXES:
        # A non-finite Base model has nothing to subtract: pending (NaN), never a crash.
        if not math.isfinite(base_raw):
            return math.nan
        v = RADAR_MAX * (1.0 + (raw - base_raw))
    else:
        v = RADAR_MAX * raw
    if not math.isfinite(v):
        return math.nan
    return min(RADAR_MAX, max(0.0, v))
