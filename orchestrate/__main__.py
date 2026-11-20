"""`python -m orchestrate build ...`, and `... ledger settle`, the manual correction line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.flag_values import refuse_blank, usd_amount
from orchestrate.build import main as build_main
from orchestrate.ledger import Ledger
from orchestrate.step_ids import block_problem, renamed_problem


def settle(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="orchestrate ledger settle", allow_abbrev=False)
    ap.add_argument("ledger", help="the build's ledger.jsonl")
    ap.add_argument("step")
    ap.add_argument("attempt", type=int)
    ap.add_argument("--usd", type=usd_amount, required=True)
    ap.add_argument("--by", required=True, help="who read the bill")
    args = refuse_blank(ap).parse_args(argv)
    problem = renamed_problem(args.step, "step") or block_problem(args.step)
    if problem:  # a settle line names a step, by its new id
        print(f"REFUSING: {problem}")
        return 2
    Ledger(Path(args.ledger)).settle(
        args.step, args.attempt, args.usd, "manual", "settled", by=args.by
    )
    return 0


def pod_gone(argv: list[str]) -> int:
    """Record that YOU deleted a pod (checked on the provider's console), lifting the block."""
    import json
    import time

    ap = argparse.ArgumentParser(prog="orchestrate ledger pod-gone", allow_abbrev=False)
    ap.add_argument("record", help="the pod record, builds/.../pods/<step>-<attempt>.json")
    ap.add_argument("--by", required=True, help="who checked")
    args = refuse_blank(ap).parse_args(argv)
    path = Path(args.record)
    event = {"event": "terminated", "at": time.time(), "by": args.by, "manual": True}
    with path.with_suffix(".events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    return 0


def main(argv: list[str]) -> int:
    if argv[:2] == ["ledger", "pod-gone"]:
        return pod_gone(argv[2:])
    if argv[:2] == ["ledger", "settle"]:
        return settle(argv[2:])
    if argv[:1] == ["build"]:
        return build_main(argv[1:])
    print("usage: python -m orchestrate build <id> --base B | ledger settle ...")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
