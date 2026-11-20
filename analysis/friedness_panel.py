"""The mu-decisiveness panel: stored ``mu_decisiveness`` rows in, upstream's panel out.

It lives in ``analysis/`` because CLAUDE.md is explicit that aggregation happens here
and never inside an eval. ``analysis/derived.py`` calls it to produce the headline cell
``mu_decisiveness:decisiveness``; its own ``main`` prints every organism's panel.

WHAT IS UPSTREAM'S AND WHAT IS OURS
    Every number is upstream's (``vendor/fried-model-organisms``, commit ``e820cf9``): the
    rows are turned back into upstream ``EdgeObservation`` objects (oracle.py:23-35) and
    handed to upstream's own ``_bucket_for_panel`` (cli/metric.py:92-119) and
    ``compute_panel`` (panel.py:174-257) — the exact calls ``run_elicitation`` makes at the
    end of a run. What this file owns is only the row -> observation translation and the
    refusals below. It used to carry its own copy of the bucketing for the deleted
    fixed-pair protocol; that copy is gone, so there is nothing here to drift.

    The headline is ``panel["decisiveness"]["point"]`` = mean|2*Phi_hat - 1| over the fitted
    Case-V matrix (panel.py:9-13), fitted on the Elo edges only with the run's own
    ``fit_steps`` and seed (both stamped on every row by ``evals.friedness.arcadia_row``).
    ``bootstrap`` is OFF by default, as upstream's CLI has it; pass it for intervals.

WHAT IS REFUSED, AND WHY
    - Rows from the deleted fixed-pair protocol (no ``edge_order``): a different sampling
      design, not comparable, and not re-panelled. Re-run the organism.
    - Rows from two runs, or two protocols, in one panel (their stamps differ).
    - A gap in ``edge_order``: the triad phase is chunked by 3 in emission order
      (metric.py:112-119), so one missing row would silently re-pair every later triad.
    A cell whose sample failed carries one flagged NaN row from the bridge
    (``meta['bridge_undercount']``); it yields ``panel: None``, never a number.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import pyarrow as pa
import pyarrow.compute as pc

from analysis.aggregate import ensure_log_bridges, load_table
from core.results import SampleResult

log = logging.getLogger(__name__)

_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "fried-model-organisms"
if str(_VENDOR / "src") not in sys.path:
    sys.path.insert(0, str(_VENDOR / "src"))

EVAL_NAME = "mu_decisiveness"

#: The stamp keys two rows must agree on to be one run (``ArcadiaPlan.stamp``).
_STAMP_KEYS = (
    "protocol",
    "upstream_commit",
    "n_items",
    "arcadia_seed",
    "R",
    "m",
    "floor",
    "K",
    "n_reverse",
    "n_triads",
    "n_cross",
    "fit_steps",
    # The read (`ARCADIA_READ`): a pre-#342 tree has none, so it never shares a fit.
    "read",
)


class _HasMeta(Protocol):
    """A stored row: ``core.results.SampleResult`` or ``analysis.aggregate.Row``."""

    @property
    def meta(self) -> dict[str, Any]: ...


def _failed(row: _HasMeta) -> bool:
    """The bridge's stand-in row for a sample that errored (``rows_from_packed``)."""
    return bool(row.meta.get("bridge_undercount"))


def item_universe(rows: Sequence[_HasMeta]) -> dict[int, str]:
    """index -> item name across every edge row, and a hard error if the rows disagree.

    An item index is only an identity while every row means the same list by it. Each
    row records its own ``item_i`` / ``item_j``, so a conflict is detectable from Parquet
    alone.
    """
    universe: dict[int, str] = {}
    for row in rows:
        m = row.meta
        for index, name in ((m["i"], m["item_i"]), (m["j"], m["item_j"])):
            seen = universe.setdefault(int(index), str(name))
            if seen != str(name):
                raise ValueError(
                    f"friedness panel: item index {index} is {seen!r} in one row and "
                    f"{name!r} in another, so these rows come from two different item "
                    "banks and cannot share a fit."
                )
    return universe


def arcadia_edges(rows: Sequence[_HasMeta]) -> tuple[list[Any], list[Any], dict[str, Any]]:
    """Stored rows -> ``(elo_obs, extra_obs, stamp)``: upstream's two observation lists.

    ``elo_obs`` / ``extra_obs`` are exactly what ``run_elicitation`` holds when it calls
    ``_bucket_for_panel(elo_obs, extra)`` (metric.py:76), in emission order.

    Raises:
        ValueError: legacy fixed-pair rows, rows from two runs, or a gap in the emission
            order (module docstring).
    """
    from mu_decisiveness.oracle import EdgeObservation

    legacy = [r for r in rows if "edge_order" not in r.meta]
    if legacy:
        raise ValueError(
            f"friedness panel: {len(legacy)} mu_decisiveness row(s) carry no `edge_order`, "
            "so they come from the deleted fixed-pair protocol (before the switch to Arcadia's "
            "protocol, docs/evals/coherence.md). "
            "That design is not comparable with Arcadia's; re-run the organism."
        )
    stamps = {tuple(r.meta.get(k) for k in _STAMP_KEYS) for r in rows}
    if len(stamps) != 1:
        raise ValueError(
            f"friedness panel: these rows carry {len(stamps)} different protocol stamps "
            f"({_STAMP_KEYS}); they are several runs and cannot share one fit."
        )
    ordered = sorted(rows, key=lambda r: int(r.meta["edge_order"]))
    orders = [int(r.meta["edge_order"]) for r in ordered]
    if orders != list(range(len(orders))):
        raise ValueError(
            "friedness panel: `edge_order` is not 0..N-1 without gaps or repeats; the "
            "triad phase is chunked by emission order, so a partial run cannot be panelled."
        )
    item_universe(ordered)
    elo: list[Any] = []
    extra: list[Any] = []
    for row in ordered:
        m = row.meta
        observation = EdgeObservation(
            i=int(m["i"]),
            j=int(m["j"]),
            p_util=float(m["p_util"]),
            mode=str(m["mode"]),
            question_id=str(m["question_id"]),
            valence=int(m["valence"]),
            slot_a=str(m["slot_a"]),
            phase=str(m["phase"]),
            round=m.get("round"),
            rank_distance=m.get("rank_distance"),
            raw={"p_a": float(m["p_pick_a"])},
        )
        (elo if observation.phase == "elo" else extra).append(observation)
    stamp = dict(zip(_STAMP_KEYS, next(iter(stamps)), strict=True))
    return elo, extra, stamp


def friedness_panel(
    rows: Sequence[_HasMeta],
    fit_steps: int | None = None,
    seed: int | None = None,
    bootstrap: bool = False,
    bootstrap_b: int = 200,
) -> dict[str, Any]:
    """Upstream's panel for ONE organism's ``mu_decisiveness`` rows.

    ``fit_steps`` / ``seed`` default to the run's own (stamped on the rows), which is what
    makes ``panel["decisiveness"]["point"]`` the number ``run_elicitation`` returned
    in-run, up to fit device (CPU here, CUDA in a GPU eval process). Returns bookkeeping
    plus ``panel`` (upstream's dict, JSON-safe), or ``panel: None`` when the cell failed.
    """
    from mu_decisiveness.cli.metric import _bucket_for_panel
    from mu_decisiveness.io_utils import jsonable
    from mu_decisiveness.panel import compute_panel

    out: dict[str, Any] = {"n_rows": len(rows)}
    failed = [r for r in rows if _failed(r)]
    if failed:
        # The bridge's one-row stand-in for a sample that died: there is no run to panel.
        log.warning("friedness panel: the mu cell failed (%d flagged row(s))", len(failed))
        out.update(n_elo_edges=0, n_triads=0, protocol=None, panel=None)
        return out
    elo, extra, stamp = arcadia_edges(rows)
    edges_by_phase = _bucket_for_panel(elo, extra)
    panel = compute_panel(
        edges_by_phase,
        n=int(stamp["n_items"]),
        seed=int(stamp["arcadia_seed"] if seed is None else seed),
        fit_steps=int(stamp["fit_steps"] if fit_steps is None else fit_steps),
        bootstrap=bootstrap,
        B=bootstrap_b,
    )
    out.update(
        n_elo_edges=len(edges_by_phase["elo"]),
        n_reverse_pairs=len(edges_by_phase["reverse"]),
        n_triads=len(edges_by_phase["triad"]),
        n_cross=len(edges_by_phase["cross"]),
        protocol=stamp,
        panel=jsonable(panel),
    )
    return out


#: The sidecar a results tree keeps its decisiveness bootstrap in (`scripts/mu_bootstrap.py`
#: writes it, `load_bootstrap_sidecar` reads it). ONE file for the composite and the paper's
#: mu figure, so the two can never show different intervals.
SIDECAR_NAME = "mu_decisiveness_bootstrap.json"
#: Overrides where the sidecar is read from, for a tree that must not be written into.
SIDECAR_ENV = "MO_EVALS_MU_BOOTSTRAP"
SIDECAR_VERSION = 1
#: How far a sidecar's point may sit from the point refitted from the rows before the sidecar
#: is refused as describing other data. The fit is deterministic on one device; CPU vs CUDA
#: moves it in the 4th decimal at most (module docstring).
SIDECAR_POINT_TOL = 1e-3


def decisiveness_bootstrap(
    rows: Sequence[_HasMeta], b: int = 200, seed: int | None = None
) -> dict[str, Any]:
    """Upstream's two decisiveness bootstraps for ONE organism, REPLICATES KEPT.

    ``compute_panel(bootstrap=True)`` (panel.py:174-206) makes exactly these two calls and
    returns only each one's 2.5/97.5 percentiles (``_ci``, panel.py:150-154). This makes the
    same calls with the same arguments — same elo edges, same point fit as warm start, same
    seed, upstream's own default step counts — and also keeps what upstream throws away, so
    a standard error can be read off the replicates:

    - ``meas``: ``bootstrap_measurement`` (fit.py:117-146) resamples the elo EDGES with
      replacement and refits every replicate. Noise of the comparisons actually asked.
    - ``gen``: ``bootstrap_items`` (fit.py:149-169) resamples the ITEMS (each edge weighted
      by the product of its two items' multinomial counts) and refits. Noise of the item
      panel — the analogue of the item-clustered interval every other cell carries, and the
      one our figures use.

    ``*_ci`` are upstream's percentile intervals, unchanged. ``*_sd`` is the sample SD of
    the replicates (ours: upstream reports no SE), ``*_mean`` their mean, so
    ``*_mean - point`` is the bootstrap's own estimate of the statistic's bias. THE
    PERCENTILE INTERVAL NEED NOT CONTAIN THE POINT: mean|2*Phi-1| is a mean of absolute
    values of fitted quantities, so a noisier refit scores higher, and every replicate is a
    noisier refit. Upstream applies no bias correction and neither does this.
    """
    import numpy as np
    from mu_decisiveness.cli.metric import _bucket_for_panel
    from mu_decisiveness.fit import bootstrap_items, bootstrap_measurement, fit_caseV_mle
    from mu_decisiveness.panel import _ci, decisiveness

    elo_obs, extra, stamp = arcadia_edges(rows)
    elo = _bucket_for_panel(elo_obs, extra)["elo"]
    n = int(stamp["n_items"])
    fit_seed = int(stamp["arcadia_seed"] if seed is None else seed)
    mu = fit_caseV_mle(elo, n=n, seed=fit_seed, steps=int(stamp["fit_steps"]))["mu"]
    out: dict[str, Any] = {
        "point": decisiveness(mu),
        "b": b,
        "seed": fit_seed,
        "n_items": n,
        "n_elo_edges": len(elo),
        "fit_steps": int(stamp["fit_steps"]),
    }
    for name, draw in (("meas", bootstrap_measurement), ("gen", bootstrap_items)):
        reps = np.asarray(draw(elo, n, b, decisiveness, fit_seed, mu_init=mu), dtype=float)
        finite = reps[np.isfinite(reps)]
        out[f"{name}_ci"] = _ci(reps)
        out[f"{name}_mean"] = float(finite.mean()) if finite.size else math.nan
        out[f"{name}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else math.nan
        out[f"{name}_n_finite"] = int(finite.size)
    return out


#: A per-shard sidecar beside the canonical one: ``mu_decisiveness_bootstrap.<shard>.json``.
#: `scripts/run_all.sh` writes its bootstrap under a shard-unique name because every shard pod
#: of a run pushes into ONE Hub folder (`upload_results.py`), where an add REPLACES an
#: existing path: shards sharing the canonical name would each overwrite the others'
#: organisms, and the last pod to push would decide which coherence SEs the merged tree has.
SIDECAR_SHARD_GLOB = "mu_decisiveness_bootstrap.*.json"


def tree_sidecar_paths(root: Path | str) -> list[Path]:
    """Every sidecar a tree's analysis joins: ``$MO_EVALS_MU_BOOTSTRAP`` alone if set, else
    the canonical ``<root>/mu_decisiveness_bootstrap.json`` plus each per-shard
    ``<root>/mu_decisiveness_bootstrap.<shard>.json`` (`SIDECAR_SHARD_GLOB`), sorted."""
    import os

    override = os.environ.get(SIDECAR_ENV)
    if override:
        return [Path(override)]
    root = Path(root)
    return [root / SIDECAR_NAME, *sorted(root.glob(SIDECAR_SHARD_GLOB))]


def load_bootstrap_sidecars(paths: Sequence[Path | str]) -> dict[str, dict[str, Any]]:
    """The union of several sidecars' ``{spec_hash: entry}`` (missing files read as empty).

    Shards hold disjoint organisms, so the union is the whole tree. The same organism in two
    files is fine when both fits agree (a re-run on another pod); two different points
    (beyond `SIDECAR_POINT_TOL`) for one spec_hash RAISE — which one describes the rows is
    exactly what cannot be guessed. The first file in `paths` order supplies the entry; one
    whose fit agrees but whose bootstrap differs (another B, another SD) is logged.
    """
    merged: dict[str, dict[str, Any]] = {}
    origin: dict[str, Path] = {}
    for path in paths:
        for spec_hash, entry in load_bootstrap_sidecar(path).items():
            if spec_hash not in merged:
                merged[spec_hash], origin[spec_hash] = entry, Path(path)
                continue
            a, b = merged[spec_hash].get("point"), entry.get("point")
            if not (
                isinstance(a, (int, float))
                and isinstance(b, (int, float))
                and abs(a - b) <= SIDECAR_POINT_TOL
            ):
                raise ValueError(
                    f"{spec_hash}: decisiveness bootstrap sidecars disagree "
                    f"({origin[spec_hash]} point {a!r}, {path} point {b!r}); remove the stale one"
                )
            kept = merged[spec_hash]
            if (kept.get("b"), kept.get("gen_sd")) != (entry.get("b"), entry.get("gen_sd")):
                log.warning(
                    "%s: two decisiveness bootstraps of the same fit (%s: B=%s gen_sd=%s; "
                    "%s: B=%s gen_sd=%s) — using the first",
                    spec_hash,
                    origin[spec_hash],
                    kept.get("b"),
                    kept.get("gen_sd"),
                    path,
                    entry.get("b"),
                    entry.get("gen_sd"),
                )
    return merged


def load_bootstrap_sidecar(path: Path | str) -> dict[str, dict[str, Any]]:
    """``{spec_hash: entry}`` from a sidecar, or ``{}`` when there is none.

    A file that exists but is not a sidecar of this version RAISES: silently reading none
    would turn every decisiveness interval into "unknown" with nothing saying why.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("version") != SIDECAR_VERSION:
        raise ValueError(f"{path}: not a mu-decisiveness bootstrap sidecar v{SIDECAR_VERSION}")
    cells = data.get("cells")
    if not isinstance(cells, dict):
        raise ValueError(f"{path}: sidecar has no `cells` map")
    return cells


def panel_for_results_root(
    root: Path | str, fit_steps: int | None = None, bootstrap: bool = False
) -> dict[str, dict[str, Any]]:
    """One panel per organism (spec_hash) found under a results root.

    Through the results facade, and the eval selected in Arrow rather than Python:
    ``to_pylist()`` on the whole tree would decode every other eval's transcripts only to
    throw them away.
    """
    ensure_log_bridges()
    table = load_table(root)
    # pyarrow.compute is generated at import time, so its stub declares no members.
    names = pc.is_in(table["eval_name"], pa.array([EVAL_NAME]))  # ty: ignore[unresolved-attribute]
    table = table.filter(names)
    by_hash: dict[str, list[SampleResult]] = {}
    for row in table.to_pylist():
        by_hash.setdefault(row["spec_hash"], []).append(SampleResult.from_row(row))
    return {
        spec_hash: friedness_panel(rows, fit_steps=fit_steps, bootstrap=bootstrap)
        for spec_hash, rows in sorted(by_hash.items())
        if rows
    }


def _fmt(x: Any) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    return f"{x:.3f}"


def _point(panel: dict[str, Any] | None, key: str) -> Any:
    return None if panel is None else panel[key]["point"]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Arcadia mu-decisiveness panel over stored rows.")
    p.add_argument("--results-root", default="results")
    p.add_argument("--fit-steps", type=int, default=None, help="default: the run's own")
    p.add_argument("--bootstrap", action="store_true", help="compute upstream's CIs")
    p.add_argument("--json", type=Path, default=None, help="also write the full panels here")
    args = p.parse_args(argv)

    panels = panel_for_results_root(
        args.results_root, fit_steps=args.fit_steps, bootstrap=args.bootstrap
    )
    if not panels:
        log.warning("no %s rows under %s", EVAL_NAME, args.results_root)
        return 1
    print(f"| {'spec_hash':16} | decisive | order_cons | transitiv | q_agree | elo  |")
    print(f"|{'-' * 18}|{'-' * 10}|{'-' * 12}|{'-' * 11}|{'-' * 9}|{'-' * 6}|")
    for spec_hash, res in panels.items():
        panel = res["panel"]
        print(
            f"| {spec_hash:16} | {_fmt(_point(panel, 'decisiveness')):8} "
            f"| {_fmt(_point(panel, 'order_consistency')):10} "
            f"| {_fmt(_point(panel, 'transitivity_triad')):9} "
            f"| {_fmt(_point(panel, 'q_agreement')):7} | {res['n_elo_edges']:4} |"
        )
    if args.json:
        args.json.write_text(json.dumps(panels, indent=2, default=float), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raise SystemExit(main())
