#!/usr/bin/env python
"""Run the robustness evals over each dataset bank and print a score per dataset.

Reuses the real runner (``serving.inspect_runner.run_matrix`` → eval_set, judge, write
Parquet). For each dataset bank built by ``build_robustness_banks.py`` it runs the
bank-consuming robustness evals against every spec in the config, writing to
``results/robustness_scale/<dataset>/``, then prints mean score per (eval, condition).

Only the bank-consuming robustness evals run (``robustness_critique_provided`` uses fixed
stimuli, not the bank, and is deregistered). Mean score is over 1.0=persona held / 0.0=abandoned,
NaN (judge-discarded) dropped — so the `control` column is the baseline expression rate
and any pressure column below it is the drop. Read them together (see the bank docstring).

    MO_EVALS_FAKE=1 uv run python scripts/run_robustness_scale.py   # no GPU, FAKE outputs
    uv run python scripts/run_robustness_scale.py --config configs/specs.yaml
    uv run python scripts/run_robustness_scale.py --datasets advbench \\
        --evals robustness_open_ended_gen

Real numbers need a real endpoint AND a real judge (a `judge` block in the config's
eval_configs). Without those it runs FakeEndpoint + FakeJudge — plumbing, not a result.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from math import isnan
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run as a plain script

from core.results import read_result_table, result_files
from evals._inspect import all_evals
from serving.inspect_runner import run_matrix
from serving.resolve import fake_mode
from serving.runner import (
    _force_fake_backends,
    load_config,
    load_specs,
    selection_names,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
BANK_DIR = ROOT / "data" / "eval_prompts" / "robustness" / "external"
# The bank-consuming robustness evals this script runs (they route through _run →
# resolve_bank). Excluded: robustness_critique_provided (builds fixed critique_texts.yaml
# stimuli, ignores the bank) and robustness_multiturn_debate_gen (also bank-consuming, but
# needs a live adversary + 3 extra calls/sample — run it from the main config, not here).
# NB: robustness_self_critique is NOT a registered eval — it is a *condition* of
# robustness_critique_gen; listing it here silently ran nothing for that slot (fixed).
# NB: robustness_scrutiny_scaling was DEREGISTERED 2026-09-14 (Slocum rework); dropped
# here too, otherwise registry filtering silently omits it and the script still exits 0.
# The `_gen` names since the hh-rlhf cells retired (2026-10-01): a `_gen` cell honours the
# per-dataset `bank` this script sets, so it runs the same measurement the originals did.
BANK_EVALS = {
    "robustness_open_ended_gen",
    "robustness_adversarial_prompt_gen",
    "robustness_critique_gen",
}


def _score_table(parquet_dir: Path) -> dict[tuple[str, str], tuple[list[float], int]]:
    """Scored values and total rows per (eval_name, condition). NaN counts to total only.

    Returns (scored_values, n_total) so an all-NaN run (e.g. fake mode, where the judge
    discards) reads as 0/N scored rather than vanishing from the table.
    """
    scored: dict[tuple[str, str], list[float]] = defaultdict(list)
    total: dict[tuple[str, str], int] = defaultdict(int)
    for pf in result_files(parquet_dir):
        t = read_result_table(pf, columns=["eval_name", "score", "meta"]).to_pylist()
        for row in t:
            key = (row["eval_name"], (json.loads(row["meta"]) or {}).get("condition", "?"))
            total[key] += 1
            s = row["score"]
            if s is not None and not isnan(s):
                scored[key].append(s)
    return {k: (scored.get(k, []), n) for k, n in total.items()}


def run(
    config_path: str,
    evals_filter: set[str],
    datasets: list[str],
    force: bool,
    bank_dir: Path = BANK_DIR,
) -> int:
    config = load_config(config_path)
    if fake_mode():
        # As `inspect_runner.run_all` does. This script hands `run_matrix` an already-
        # loaded config (it rewrites `bank` per dataset), so it owns this step itself —
        # without it `MO_EVALS_FAKE=1`, which this script's own usage line advertises,
        # would still call the config's real judge.
        _force_fake_backends(config)
    specs = load_specs(config)
    evals = [e for e in all_evals() if e.name in evals_filter]
    if not evals:
        log.error("no matching robustness evals to run (wanted %s)", sorted(evals_filter))
        return 1

    config.setdefault("eval_configs", {})
    failed = False
    for name in datasets:
        bank = bank_dir / f"{name}.yaml"
        if not bank.exists():
            log.error("bank %s missing — run build_robustness_banks.py first", bank)
            failed = True
            continue
        out_root = ROOT / "results" / "robustness_scale" / name
        for ev in evals:
            # Point every bank-consuming eval at this dataset's bank for this pass. The
            # mutated config is why this calls `run_matrix` and not `run_all`: `run_all`
            # re-reads the YAML and would lose the per-dataset bank.
            config["eval_configs"].setdefault(ev.name, {})["bank"] = str(bank)
        # One eval_set per dataset over the whole (spec x eval) block — it schedules the
        # organisms itself, so there is no per-spec loop here any more.
        for outcome in run_matrix(config, specs, evals, root=out_root, force=force):
            if outcome.status == "failed":
                failed = True
            log.info(
                "%s | %-8s %s x %s", name, outcome.status, outcome.spec_hash, outcome.eval_name
            )

        print(f"\n=== {name} ===")
        table = _score_table(out_root)
        if not table:
            print("  (no rows)")
        for (ev_name, cond), (vals, n_total) in sorted(table.items()):
            mean = f"{sum(vals) / len(vals):.3f}" if vals else "  n/a"
            print(f"  {ev_name:<32} {cond:<28} mean={mean}  scored={len(vals)}/{n_total}")
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default="configs/specs.yaml")
    p.add_argument("--bank-dir", type=Path, default=BANK_DIR)
    p.add_argument("--datasets", nargs="+", action="extend", default=None)
    p.add_argument(
        "--evals",
        action="append",
        default=None,
        help="comma-separated robustness eval names; repeatable (values accumulate)",
    )
    p.add_argument("--force", action="store_true", help="re-run even if Parquet exists")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    wanted = set(selection_names(args.evals, "--evals")) if args.evals else set(BANK_EVALS)
    unknown = wanted - BANK_EVALS
    if unknown:
        log.error(
            "not bank-consuming robustness evals: %s; choose from %s",
            sorted(unknown),
            sorted(BANK_EVALS),
        )
        return 1
    datasets = args.datasets or ["hh_rlhf"]
    return run(args.config, wanted, datasets, args.force, bank_dir=args.bank_dir)


if __name__ == "__main__":
    raise SystemExit(main())
