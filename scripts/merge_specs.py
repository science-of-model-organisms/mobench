#!/usr/bin/env python
"""Merge several spec files into one, for commands that take a single config.

    uv run python scripts/merge_specs.py --out merged.yaml a.yaml b.yaml

`scripts/run_all.sh` calls this when `--config` is repeated. The rules, including every
conflict that raises, are `serving.runner.merge_configs`. Each output row is fully expanded,
so `spec_hash` is the one each row has in its own file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from serving.runner import SpecConflictError, merge_configs  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        merged = merge_configs(args.paths)
    except SpecConflictError as exc:
        print(f"spec conflict: {exc}", file=sys.stderr)
        return 1
    args.out.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
