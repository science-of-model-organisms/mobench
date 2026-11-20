"""The statistics the results dashboard shows, computed in Python from a built data blob.

The page's JavaScript only RENDERS numbers: every mean, interval, count and significance flag
it prints is computed here and read back from the blob. `attach_display_stats(data)` is
called by `analysis.dashboard.render_html`, so a from-scratch build and a cached `--data-json`
render go through the same code and cannot drift (a blob built before this module existed
simply gets the fields at render time). It works on a COPY: the caller's `data` is untouched.

What it adds, and where the page reads it:

* per cell (`cells[org][eval]`) and per axis cell (`axis_cells[org][axis]`): `ci_lo` / `ci_hi`
  (the 95% interval, 1.96 x se, on the oriented delta, clipped to [0, 1] for a share kept) and
  `sig` (the interval excludes 0). Read by `mkRead` (data.js). The cells the page draws as
  RAW-rate whiskers (misalignment, capability_format) also carry `hw` (1.96 x se) and `sem_hw`
  (1.96 x sem); the others do not, to keep the blob small.
* `derived.axis_mean` / `derived.axis_mean_base`: an organism's mean over a property's voting
  evals, and its base arm's mean over the same evals (the tooltip's "organism x / base y").
* `derived.series`: per (behaviour, base model, implant-method series, property) the Profiles
  numbers: mean score, mean oriented delta with its 95% interval, the matched delta, and the
  polygon values of the per-model view. A series is one implant method on one base model for one
  behaviour (the negative-prompt control is its own series), so every filter keeps or drops a whole
  series and these numbers never depend on a filter.
* `derived.unscored`: the run-wide share of unscored samples quoted in Details.

NOT here, on purpose: the eval-correlation matrix, the discard-vs-score means and figure 1's fit
lines. Each is a statistic over the organisms the filters leave, and the chips give 6,376 distinct
visible sets (methods x behaviours x base models), so precomputing them is megabytes per figure
(see analysis/dashboard_js/sections/corr.js); they stay in JS, over the exported per-cell values.

Arithmetic order follows the JavaScript it replaced (sequential sums from 0, `1.96 * x`). The
values equal what the page used to compute at printed precision. A series of 3 or more members is
summed in plain-string `short` order here and was summed in `localeCompare` (ICU) order in the
page, so such a mean may differ in its last bit; one or two members cannot.
"""

from __future__ import annotations

import math
import re
from typing import Any

from analysis.radar_scale import HEALTH_AXES

# 95%-CI multiplier: a flat 1.96 on the cell's `sem`, as the server does (Cell.ci95).
CI_Z = 1.96
# The negative-prompt control arm: its method and the [variant] that marks it (data.js CONTROL_*).
_CONTROL_METHODS = frozenset({"prompting"})
_CONTROL_VARIANTS = frozenset({"negative"})
CTRL = "__control"
# Cells carrying a count of unscorable-by-design samples (top.js: mu-decisiveness ties).
_STRUCTURAL_EVALS = frozenset({"mu_decisiveness"})
# Cells drawn as raw-rate whiskers (mean +/- 1.96 x sem): capability_format.js.
_RAW_RATE_PREFIX = "capability_format:"
_EXPRESSION_EVAL = "simple_behaviour_exhibit"


def _fin(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _sum(vs: list[float]) -> float:
    """Sequential sum from 0: the JS `vs.reduce((s,v)=>s+v,0)` (not math.fsum)."""
    t = 0.0
    for v in vs:
        t += v
    return t


def _mean(vs: list[float]) -> float | None:
    return _sum(vs) / len(vs) if vs else None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _js_round(x: float) -> int:
    """`Math.round`: half rounds up (Python's round() is half-to-even)."""
    return math.floor(x + 0.5)


def interval(delta: Any, se: Any, share: bool) -> dict[str, Any]:
    """`hw`, `ci_lo`, `ci_hi`, `sig` of one reading (nothing without an se; no CI without a delta).


    The interval is on the ORIENTED delta (the page flips the sign for a backwards property);
    a share kept lives in [0, 1], so its interval is clipped there. `sig` is False for a share
    (it is read against 0, not against base).
    """
    if not _fin(se):
        return {}
    h = CI_Z * se
    out: dict[str, Any] = {"hw": h}
    if not _fin(delta):
        return out
    lo, hi = delta - h, delta + h
    out["ci_lo"] = _clip01(lo) if share else lo
    out["ci_hi"] = _clip01(hi) if share else hi
    if not share and (lo > 0 or hi < 0):
        out["sig"] = True
    return out


def _without_hw(iv: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in iv.items() if k != "hw"}


def _is_control(o: dict[str, Any]) -> bool:
    m = re.search(r"\[(.+?)\]", o.get("id") or "")
    return o.get("method") in _CONTROL_METHODS and bool(m) and m.group(1) in _CONTROL_VARIANTS


def _votes(c: Any) -> bool:
    # JS `!!c && c.votes !== false`: an empty cell object is truthy there, so it votes
    return c is not None and c.get("votes") is not False


def _axis_mean(cells: dict[str, Any], evals: list[str], org: str, voter: str) -> float | None:
    """data.js axisMean: mean of the organism's cell means over the evals `voter` votes in."""
    vc, oc = cells.get(voter) or {}, cells.get(org) or {}
    vs = []
    for e in evals:
        c = oc.get(e)
        if c and c.get("mean") is not None and _votes(vc.get(e)):
            vs.append(c["mean"])
    return _mean(vs)


def _base_id(orgs: list[dict[str, Any]], persona: str, bm: str | None) -> str | None:
    """select.js baseId: the persona's control arm on base model `bm`, none if it has none.

    An organism without a base model (an old blob) reads against the persona's first control
    arm, as `baseId(persona, undefined)` does with no base-model radio.
    """
    bases = [o for o in orgs if o.get("is_base") and o.get("persona") == persona]
    if not bm:
        return bases[0]["id"] if bases else None
    return next((o["id"] for o in bases if o.get("base_model") == bm), None)


def matched_delta_mean(axis: str, scores: list[float]) -> dict[str, Any]:
    """The matched delta of a series: `md`, the mean of its members' read-back deltas, and `mdc`
    when any member sits at the cap (a score of 10 only says delta >= 0 on a health spoke, >= 1 on
    a behaviour spoke, so the mean is then a lower bound). Not the delta of the mean score."""
    health = axis in HEALTH_AXES
    capped, ds = 0, []
    for v in scores:
        if v >= 10 - 1e-9:
            capped += 1
            ds.append(0 if health else 1)
        else:
            ds.append(v / 10 - 1 if health else v / 10)
    return {"md": _mean(ds), **({"mdc": True} if capped else {})}


def _series_stats(data: dict[str, Any], share: dict[str, bool]) -> dict[str, Any]:
    """derived.series (module docstring). Keys: "persona|base model", or "persona|*" for all."""
    orgs = data.get("organisms") or []
    axis_cells = data.get("axis_cells") or {}
    radar_orgs = ((data.get("radar") or {}).get("orgs")) or {}
    spokes = [a["name"] for a in data.get("axes") or [] if a.get("radar") is not False]
    morder = {
        m: i
        for i, m in enumerate(
            [
                "prompting",
                "few_shot_icl",
                "oct_demos",
                "sft_behaviour",
                "dpo_behaviour",
                "distillation_seqkd",
                "steering",
                "oct_behaviour",
            ]
        )
    }

    def ser_of(o: dict[str, Any]) -> str:
        return CTRL if _is_control(o) else o["method"]

    cells_by_key: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for o in orgs:  # data order
        if o.get("is_base"):
            continue
        for bm in (o.get("base_model"), "*"):
            if bm is None:
                continue
            cells_by_key.setdefault(f"{o['persona']}|{bm}", {}).setdefault(ser_of(o), []).append(o)

    def reads(members: list[dict[str, Any]], ax: str) -> list[dict[str, Any]]:
        out = []
        for o in members:
            c = (axis_cells.get(o["id"]) or {}).get(ax)
            if c and _fin(c.get("delta")):
                out.append(c)
        return out

    def score(o: dict[str, Any], ax: str) -> float | None:
        v = (radar_orgs.get(o["id"]) or {}).get(ax)
        return v if _fin(v) else None

    out: dict[str, Any] = {}
    for key, by_ser in cells_by_key.items():
        for ser, data_order in by_ser.items():
            # the method view sums in the page's organism order (method, then short name)
            shown = sorted(data_order, key=lambda o: (morder.get(o["method"], 99), o["short"]))
            for ax in spokes:
                rs = reads(shown, ax)
                entry: dict[str, Any] = {}
                if rs:
                    n = len(rs)
                    v = _sum([c["delta"] for c in rs]) / n
                    entry["n"] = n
                    entry["d"] = v
                    ses = [c["se"] for c in rs if _fin(c.get("se"))]
                    if len(ses) == n:  # an unknown se anywhere leaves the mean without an interval
                        h = CI_Z * math.sqrt(_sum([s * s for s in ses])) / n
                        sh = share.get(ax, False)
                        entry["dlo"] = _clip01(v - h) if sh else v - h
                        entry["dhi"] = _clip01(v + h) if sh else v + h
                members = [s for s in (score(o, ax) for o in shown) if s is not None]
                if members:
                    entry["s"] = _mean(members)
                    if not share.get(ax, False):
                        entry.update(matched_delta_mean(ax, members))
                # per-model polygon values: the mean over the series in DATA order
                ps = [s for s in (score(o, ax) for o in data_order) if s is not None]
                if ps:
                    entry["ps"] = _mean(ps)
                if not share.get(ax, False):
                    pd = [c["delta"] for c in reads(data_order, ax)]
                    if pd:
                        entry["pd"] = _mean(pd)
                if entry:
                    out.setdefault(key, {}).setdefault(ser, {})[ax] = entry
    return out


def _unscored(data: dict[str, Any]) -> dict[str, Any]:
    """Run-wide unscored samples (top.js): the share of all samples, and the structural part."""
    tot = struct = 0
    for row in (data.get("cells") or {}).values():
        for e, c in row.items():
            d, n = c.get("discard"), c.get("n")
            if d is None or n is None:
                continue
            nm = _js_round(n * d / (1 - d)) if d < 1 else n
            tot += nm
            if e in _STRUCTURAL_EVALS:
                struct += nm
    meta = data.get("meta") or {}
    ns, nu = meta.get("n_samples"), meta.get("n_unmeasurable")
    return {
        "pct": 100 * nu / ns if ns and nu is not None else 0,
        "struct_pct": 100 * struct / tot if tot else 0,
    }


def attach_display_stats(data: dict[str, Any]) -> dict[str, Any]:
    """Return `data` plus the display statistics (module docstring); `data` is not changed."""
    axes = {a["name"]: a for a in data.get("axes") or []}
    share = {n: bool(a.get("self_referenced")) for n, a in axes.items()}
    raw_rate_evals = set((data.get("bucket_evals") or {}).get("misalignment") or [])
    # exprOf (data.js) falls back to this cell's mean +/- sem_hw when an organism has no rate
    raw_rate_evals.add((data.get("expression") or {}).get("eval") or _EXPRESSION_EVAL)
    cells: dict[str, Any] = {}
    for oid, row in (data.get("cells") or {}).items():
        new_row = {}
        for e, c in row.items():
            extra = interval(c.get("delta"), c.get("se"), bool(c.get("self_ref")))
            raw = e in raw_rate_evals or e.startswith(_RAW_RATE_PREFIX)
            if not raw:
                extra = _without_hw(extra)
            elif _fin(c.get("sem")):
                extra["sem_hw"] = CI_Z * c["sem"]
            new_row[e] = {**c, **extra} if extra else c
        cells[oid] = new_row
    axis_cells: dict[str, Any] = {}
    for oid, row in (data.get("axis_cells") or {}).items():
        axis_cells[oid] = {
            ax: {**c, **_without_hw(interval(c.get("delta"), c.get("se"), share.get(ax, False)))}
            for ax, c in row.items()
        }
    bucket = data.get("bucket_evals") or {}
    orgs = data.get("organisms") or []
    axis_mean: dict[str, dict[str, float]] = {}
    axis_mean_base: dict[str, dict[str, float]] = {}
    for o in orgs:
        own, vs_base = {}, {}
        bid = None if o.get("is_base") else _base_id(orgs, o["persona"], o.get("base_model"))
        for ax in axes:
            m = _axis_mean(cells, bucket.get(ax) or [], o["id"], o["id"])
            if m is not None:
                own[ax] = m
            if bid is not None:
                b = _axis_mean(cells, bucket.get(ax) or [], bid, o["id"])
                if b is not None:
                    vs_base[ax] = b
        axis_mean[o["id"]] = own
        if vs_base:
            axis_mean_base[o["id"]] = vs_base
    derived = {
        "axis_mean": axis_mean,
        "axis_mean_base": axis_mean_base,
        "series": _series_stats({**data, "axis_cells": axis_cells}, share),
        "unscored": _unscored(data),
    }
    return {**data, "cells": cells, "axis_cells": axis_cells, "derived": derived}


# What `attach_display_stats` adds to a cell / axis cell, and the top-level block.
_ADDED_CELL_KEYS = frozenset({"hw", "ci_lo", "ci_hi", "sig", "sem_hw"})


def strip_display_stats(data: dict[str, Any]) -> dict[str, Any]:
    """`data` without what `attach_display_stats` adds, so a page rendered before it existed and
    one rendered after carry the same remaining blob (the `site --from-page` guard compares)."""

    def clean(rows: dict[str, Any]) -> dict[str, Any]:
        return {
            o: {
                k: {f: v for f, v in c.items() if f not in _ADDED_CELL_KEYS}
                for k, c in row.items()
            }
            for o, row in rows.items()
        }

    out = {k: v for k, v in data.items() if k != "derived"}
    for key in ("cells", "axis_cells"):
        if key in out:
            out[key] = clean(out[key])
    return out
