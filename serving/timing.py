"""`<eval>.timing.json` beside an organism-seed's Parquet — what a cell actually cost.

Every sweep already knows how long each cell took: `serving.inspect_runner` measures it
per cell and `timing_table` prints it. Both are lost the moment the process exits, and on
a real run the printed table never even reaches the terminal (vLLM reconfigures logging
while the engine boots). So no run in this repo has ever left a record of its own cost,
and every
cost figure in `configs/specs.yaml` and `docs/EVAL-CATALOGUE.md` is an estimate rather than a
measurement.

Deliberately not in `core/`: the contracts are frozen, and this is a serving-side
convention about directory layout, exactly as `serving/manifest.py` argues for `spec.json`.

**One file per cell, not one log per run.** Results are cached per
`(spec_hash, eval_name)`, so a resumed sweep re-runs only some cells and reports the rest
as `cached 0.0s`. A single end-of-run log would therefore describe the last partial run
rather than the sweep, and would get less accurate every time someone re-ran it. Per-cell
files make the record complete however many partial runs assembled it, and they cannot
race: six evals run concurrently and each writes only its own file.

`seconds` is wall clock for that cell *under concurrency*, not a share of a serial total —
which is why `max_concurrent_evals` and `max_concurrent_specs` are recorded beside it. A
duration without the overlap it ran under cannot be extrapolated, and extrapolating it is
the whole point of keeping it.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.results import DEFAULT_RESULTS_ROOT, SpecRef, atomic_write, results_path

log = logging.getLogger(__name__)

TIMING_SUFFIX = ".timing.json"

# Bump when the layout changes in a way readers must notice.
TIMING_VERSION = 1


def _pod_tags() -> dict[str, Any]:
    """What this pod is, so `seconds` can become dollars in `analysis.timing`.

    Set by the launcher, not by config: one organism is sharded across pod roles on
    different cards, so the rate is a property of the POD that ran the cell, and only
    the pod knows it. An unset or unparseable rate is OMITTED rather than defaulted —
    a missing measurement and a measurement of zero are different facts, and a typo in
    a launch script must not become a cost figure nobody questions.
    """
    tags: dict[str, Any] = {}
    for key, var in (("pod", "MO_EVALS_POD_NAME"), ("gpu", "MO_EVALS_POD_GPU")):
        value = (os.environ.get(var) or "").strip()
        if value:
            tags[key] = value
    raw = (os.environ.get("MO_EVALS_POD_USD_PER_HOUR") or "").strip()
    if raw:
        try:
            rate = float(raw)
        except ValueError:
            log.warning("MO_EVALS_POD_USD_PER_HOUR is not a number (%r) — not recorded", raw)
        else:
            if rate > 0:
                tags["usd_per_hour"] = rate
            else:
                log.warning("MO_EVALS_POD_USD_PER_HOUR must be positive (%r) — not recorded", raw)
    return tags


def timing_path(ref: SpecRef, eval_name: str, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    """Where one cell's timing record lives — beside its Parquet, same stem.

    Derived from `results_path` rather than rebuilt, so the two can never disagree about
    the layout and a seed-tagged Parquet always gets a seed-tagged timing file.
    """
    parquet = results_path(ref, eval_name, root)
    return parquet.with_name(parquet.name.removesuffix(".parquet") + TIMING_SUFFIX)


def write_timing(
    ref: SpecRef,
    eval_name: str,
    *,
    axis: str,
    status: str,
    n_samples: int,
    seconds: float,
    root: Path | str = DEFAULT_RESULTS_ROOT,
    concurrency: dict[str, int] | None = None,
    extra: dict[str, Any] | None = None,
) -> Path | None:
    """Record what one cell cost. Never raises — a failed write must not sink a sweep.

    Returns the path written, or None if it could not be written. Timing is diagnostic:
    losing it is a nuisance, losing the Parquet it describes is hours of GPU.
    """
    payload: dict[str, Any] = {
        "version": TIMING_VERSION,
        "spec_hash": ref.spec_hash if hasattr(ref, "spec_hash") else str(ref),
        "eval_name": eval_name,
        "axis": axis,
        "status": status,
        "n_samples": int(n_samples),
        "seconds": round(float(seconds), 3),
        "finished_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        # Without these, `seconds` is not interpretable: the same cell takes a different
        # wall clock depending on how many other cells shared the GPU with it.
        "concurrency": concurrency or {},
        # What it ran on, so seconds can be priced. Priced cell-seconds overlap (see
        # `analysis.timing.OrganismTiming.gpu_usd_upper_bound`): an upper bound, not spend.
        **_pod_tags(),
    }
    if extra:
        payload.update(extra)
    try:
        path = timing_path(ref, eval_name, root)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path
    except OSError as exc:
        log.warning("could not write timing for %s x %s: %s", payload["spec_hash"], eval_name, exc)
        return None


def read_timing(
    ref: SpecRef, eval_name: str, root: Path | str = DEFAULT_RESULTS_ROOT
) -> dict[str, Any] | None:
    """One cell's timing record, or None if this cell predates timing or failed to write."""
    path = timing_path(ref, eval_name, root)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("unreadable timing file %s: %s", path, exc)
        return None


def load_timings(root: Path | str = DEFAULT_RESULTS_ROOT) -> list[dict[str, Any]]:
    """Every timing record under `root`, newest layout and legacy flat alike.

    Cells with no record are simply absent — a results tree written before this existed
    reports no timing rather than zero, which is the honest reading. Callers must say
    "not recorded" rather than "0 seconds".
    """
    out: list[dict[str, Any]] = []
    for path in sorted(Path(root).rglob(f"*{TIMING_SUFFIX}")):
        try:
            out.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as exc:
            log.warning("unreadable timing file %s: %s", path, exc)
    return out


# This process's usage filename, fixed at import. Every flush rewrites this one file, so
# the record is current from the first cell onward and a killed run still costs correctly.
# The pid keeps two sweeps started in the same second (shared results root, two agents)
# from overwriting each other.
_RUN_STAMP = f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def write_run_usage(
    usage: dict[str, dict[str, int]],
    root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    run_name: str | None = None,
) -> Path | None:
    """Third-party token totals for one sweep process, under `root/_usage/`.

    Callable repeatedly, and meant to be: `usage` is a running total, so each call
    *replaces* this process's file rather than adding one. `load_run_usage` sums every
    file it finds, so a fresh stamp per flush would multiply a run's bill by its flush
    count — hence `_RUN_STAMP`. Written to a temp name and renamed, so a kill or a full
    disk mid-write leaves the previous flush standing instead of half a JSON document
    (an unreadable file is the same lost bill as no file).

    Run-scoped rather than cell-scoped, and that is a real limitation stated rather than
    hidden: the judge backend is built inside each eval and its calls are made from the
    harness's own thread pool, so a process-wide accumulator cannot attribute a token to
    the cell that spent it. `analysis/` apportions the bill across cells by judge-call
    count and labels it as apportioned. Attributing exactly would mean threading a label
    through `judges.build_backend`, which changes the judge construction path — not worth
    the risk to a paid sweep for a number that is only used for reporting.
    """
    if not usage:
        return None
    stamp = run_name or _RUN_STAMP
    try:
        path = Path(root) / "_usage" / f"{stamp}.json"
        # Same unique-temp-name reasoning as `write_results`: a pinned `run_name` is SHARED
        # by every shard pod on one volume, so a fixed `<stamp>.json.tmp` had them deleting
        # each other's temp file and losing the bill (8 concurrent flushes, 8 losses).
        with atomic_write(path) as tmp:
            tmp.write_text(
                json.dumps(
                    {"version": TIMING_VERSION, "run": stamp, "usage": usage},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        return path
    except OSError as exc:
        log.warning("could not write run usage: %s", exc)
        return None


def load_run_usage(root: Path | str = DEFAULT_RESULTS_ROOT) -> dict[str, dict[str, int]]:
    """Token totals summed over every recorded run under `root`."""
    total: dict[str, dict[str, int]] = {}
    for path in sorted((Path(root) / "_usage").glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            log.warning("unreadable usage file %s: %s", path, exc)
            continue
        for model, row in (data.get("usage") or {}).items():
            acc = total.setdefault(model, {})
            for key, value in row.items():
                if isinstance(value, int):
                    acc[key] = acc.get(key, 0) + value
    return total
