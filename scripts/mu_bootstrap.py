#!/usr/bin/env python3
"""Write the decisiveness bootstrap sidecar for a results tree. CPU, no model, no judge.

``mu_decisiveness:decisiveness`` is one fitted number per organism, so it has no sample to
take a standard error from. Upstream's answer is a bootstrap (``compute_panel(bootstrap=
True)``); this runs it once per organism and stores the result beside the tree, where the
coherence composite (``analysis/derived.py``) and the paper's mu figure
(``analysis/paper_figures.py``) both read it.

    # writes <results_root>/mu_decisiveness_bootstrap.json
    python scripts/mu_bootstrap.py <results_root>
    # a tree that must not be written into: put the file elsewhere ...
    python scripts/mu_bootstrap.py <results_root> --out <path>
    # ... and tell the analysis where it is
    MO_EVALS_MU_BOOTSTRAP=<path> python -m analysis.dashboard ...

One organism at a time in one process (about 1.5-2 min each on CPU at B = 200, upstream's
default), written after every organism, and an organism already in the file is skipped, so
a killed run resumes. ``--force`` recomputes.

What is stored per organism (``analysis.friedness_panel.decisiveness_bootstrap``): the point,
upstream's two percentile intervals (``meas_ci`` over resampled comparisons, ``gen_ci`` over
resampled items), and — ours, upstream keeps no replicates — each bootstrap's mean and SD.
``gen_sd`` is what the composite uses as the cell's standard error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.aggregate import load_rows  # noqa: E402
from analysis.friedness_panel import (  # noqa: E402
    EVAL_NAME,
    SIDECAR_NAME,
    SIDECAR_VERSION,
    decisiveness_bootstrap,
)

log = logging.getLogger(__name__)
DEFAULT_B = 200  # upstream: compute_panel(B=200), and its CLI's --bootstrap-B default


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip()


def _write(path: Path, doc: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("results_root", type=Path)
    p.add_argument("--out", type=Path, default=None, help=f"default: <root>/{SIDECAR_NAME}")
    p.add_argument("--b", type=int, default=DEFAULT_B, help="bootstrap replicates (upstream: 200)")
    p.add_argument("--force", action="store_true", help="recompute organisms already stored")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = args.out or args.results_root / SIDECAR_NAME

    by_hash: dict[str, list[Any]] = {}
    for row in load_rows(args.results_root):
        if row.eval_name == EVAL_NAME:
            by_hash.setdefault(row.spec_hash, []).append(row)
    if not by_hash:
        log.warning("no %s rows under %s", EVAL_NAME, args.results_root)
        return 1

    doc: dict[str, Any] = {"version": SIDECAR_VERSION, "cells": {}}
    if out.is_file() and not args.force:
        doc = json.loads(out.read_text(encoding="utf-8"))
    doc.update(
        results_root=str(args.results_root),
        code_commit=_commit(),
        written=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        upstream="vendor/fried-model-organisms: fit.bootstrap_measurement / bootstrap_items",
    )
    failed = 0
    for spec_hash, rows in sorted(by_hash.items(), key=lambda kv: kv[1][0].organism):
        first = rows[0]
        tag = f"{first.organism} seed{first.seed} ({spec_hash})"
        if spec_hash in doc["cells"] and doc["cells"][spec_hash].get("b") == args.b:
            log.info("%s: kept", tag)
            continue
        start = time.monotonic()
        try:
            entry = decisiveness_bootstrap(rows, b=args.b)
        except ValueError as exc:  # legacy protocol, a gap, two runs: refused, not guessed
            log.warning("%s: SKIPPED — %s", tag, exc)
            failed += 1
            continue
        entry.update(organism=first.organism, organism_seed=first.seed, n_rows=len(rows))
        doc["cells"][spec_hash] = entry
        _write(out, doc)
        log.info(
            "%s: point %.4f gen_ci [%.4f, %.4f] gen_sd %.4f (%.0f s)",
            tag,
            entry["point"],
            *entry["gen_ci"],
            entry["gen_sd"],
            time.monotonic() - start,
        )
    log.info("wrote %s (%d organisms)", out, len(doc["cells"]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
