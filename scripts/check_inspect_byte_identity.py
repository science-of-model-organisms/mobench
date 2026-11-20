"""CI gate: two serial fake-mode runs of the Inspect runner must be byte-identical.

Generation determinism under the Inspect migration is *statistical* (per-request seeds
derive from a conversation hash), but the scaffolding — item selection, ordering,
prompts, parse decisions, fake outputs — must stay byte-identical, or a scaffolding
regression hides inside "that's just sampling noise". This script restores the
change-detector that native byte-identity provided: run the eval_set runner twice into
two fresh roots, each fully serial (MO_EVALS_GEN_THREADS=1, max_tasks=1) so scheduling
cannot reorder anything, and require every bridged row to match exactly.

"Serial" is a property of each run's own scheduling, not of the two relative to each
other, so the two runs go in parallel processes — see `_run_one`.

Sharding (`--shards N` or MO_EVALS_BYTE_IDENTITY_SHARDS, default 1 = one run pair over
every eval, exactly as before). With N > 1 the selected evals are split into N fixed
shards (`partition_evals`) and each shard gets its own run pair: 2N `spawn` processes,
each still fully serial INSIDE, each into its own root. Why this keeps the property under
test: what the gate compares is run1 against run2 of the SAME shard, and both see the
identical cell sequence under the identical serial config, so scheduling still cannot
reorder anything and the comparison itself is unchanged. What sharding does NOT check is
whether an eval's rows depend on another eval having run before it in the same process;
that is not what this gate checks at N=1 either (both runs share the same predecessor
cells), and `rows digest` — a hash of the sorted rows, printed at every N — lets two
invocations at different N be compared directly. The one known cross-eval effect is the
`max_workers` stamp, levelled per backend group, which the digest ignores (see
`_SHARD_DEPENDENT_META`). Every selected eval must land in exactly one shard, and every
cell each shard scheduled must be written.

Exit 0 on identity; exit 1 with a field-level diff otherwise. No GPU, no network.

    MO_EVALS_FAKE=1 uv run python scripts/check_inspect_byte_identity.py [config.yaml] [--shards N]
"""

from __future__ import annotations

import argparse
import hashlib
import math
import multiprocessing
import os
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, as a script
from core.env import fake_enabled  # noqa: E402

# Per-run meta keys — wall-clock, and the timestamped path of the run's own log — differ
# between two identical runs by construction. Everything else must not. Keep this list
# short and justified: each entry is a field this gate can no longer protect.
NONDETERMINISTIC_META = frozenset(
    {
        "gen_latency_s",
        "judge_latency_s",
        "adversary_latency_s",
        # petri rows carry inspect's own per-sample timings and the path of the .eval log
        # it just wrote, whose filename is a timestamp plus a random run id.
        "sample_total_s",
        "sample_working_s",
        "inspect_log",
    }
)


def _rows(root: Path) -> list[tuple]:
    import json

    from core.results import load_all

    table = load_all(root)
    rows = []
    for row in table.to_pylist():
        # `meta` is stored as a sort_keys JSON string (core/results.py).
        meta_dict = json.loads(row["meta"]) if row.get("meta") else {}
        meta = {k: v for k, v in meta_dict.items() if k not in NONDETERMINISTIC_META}
        score = row["score"]
        # NaN != NaN would make every unparseable row a spurious diff.
        score_key = "NaN" if isinstance(score, float) and math.isnan(score) else score
        rows.append(
            (
                row["spec_hash"],
                row["eval_name"],
                row["sample_id"],
                score_key,
                row["raw_output"],
                sorted(meta.items(), key=lambda kv: kv[0]),
            )
        )
    return rows


#: Each shard costs two processes, each importing inspect and the eval registry; past this
#: the per-process start-up dominates and a shared machine stops being usable.
_MAX_SHARDS = 16


def partition_evals(names: Sequence[str], shards: int) -> list[list[str]]:
    """Split eval names into at most `shards` non-empty, disjoint shards, deterministically.

    Round-robin over the SORTED names, so the result depends only on the set of names and
    `shards` — not on registry or config order — and every name lands in exactly one shard.
    `shards` above the number of names yields one shard per name; `shards <= 1` yields the
    whole list as one shard, in its given order.
    """
    if shards <= 1:
        return [list(names)]
    ordered = sorted(set(names))
    out: list[list[str]] = [[] for _ in range(min(shards, len(ordered)))]
    for i, name in enumerate(ordered):
        out[i % len(out)].append(name)
    return out


def _shard_count(requested: int, n_evals: int) -> int:
    """`requested` capped at the eval count, `_MAX_SHARDS` and half the usable cores."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:  # macOS / Windows
        cpus = os.cpu_count() or 2
    return max(1, min(requested, n_evals, _MAX_SHARDS, cpus // 2))


#: Meta that legitimately depends on WHICH evals share a process, so is left out of the
#: cross-shard digest (never out of the run1/run2 comparison). `max_workers` is levelled
#: to the widest cell in the backend group (`serving.inspect_runner._run_group`), so a
#: shard without the 256-wide evals stamps 32 where the unsharded run stamps 256. Measured
#: on configs/specs.smoke.yaml, 2026-09-29: 390 of 2,352 rows differed between --shards 1
#: and 8, in this key only; with it removed the two row sets were identical.
_SHARD_DEPENDENT_META = frozenset({"max_workers"})


def _digest(rows: list[tuple]) -> str:
    """Order-free fingerprint of a row set, comparable across different `--shards`."""
    h = hashlib.sha256()
    stripped = [
        (*row[:5], [kv for kv in row[5] if kv[0] not in _SHARD_DEPENDENT_META]) for row in rows
    ]
    for r in sorted(repr(row) for row in stripped):
        h.update(r.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def _run_one(
    config: Path, root: Path, only_evals: list[str] | None = None
) -> tuple[bool, str, list[str]]:
    """One fully serial fake sweep into `root`, over `only_evals` (None = the config's set).

    Returns (every scheduled cell written, report, the eval names the runner scheduled).

    Module-level, and taking everything it needs as arguments, because `main` runs the two
    sweeps in separate `spawn` processes. Two processes is not a weaker isolation than the
    `for` loop this replaces — it is a stronger one, and the property under test is
    unchanged: each run is still serial INSIDE (MO_EVALS_GEN_THREADS=1, max_tasks=1), which
    is what stops scheduling from reordering anything. Contention between the two cannot
    make the gate pass wrongly; a run-to-run difference it introduced would show up as a
    failing diff, which is the outcome this gate exists to produce.
    """
    from serving.inspect_runner import run_all

    t0 = time.monotonic()
    outcomes = run_all(config, root=root, only_evals=only_evals)
    elapsed = time.monotonic() - t0
    seen = sorted({o.eval_name for o in outcomes})
    failed = [o for o in outcomes if o.status == "failed"]
    if failed:
        return (
            False,
            "\n".join(f"FAILED {o.spec_hash} x {o.eval_name}: {o.detail}" for o in failed),
            seen,
        )
    # Not "did anything run" but "did EVERYTHING scheduled run". The root is a fresh temp
    # dir, so no cell can come back `cached`, and every cell the config scheduled must come
    # back `written`. The old `any(...)` was satisfied by ONE cell: 40 of 42 could skip and
    # the gate still printed byte-identical over the two that were left.
    not_written = [o for o in outcomes if o.status != "written"]
    if not outcomes or not_written:
        lines = [
            f"{o.status.upper()} {o.eval_name} x {o.spec_hash}: {o.detail}"
            for o in not_written[:10]
        ]
        lines.append(
            f"gate is short: {len(outcomes) - len(not_written)} of {len(outcomes)} "
            "scheduled cells produced rows. An eval that cannot run in fake mode "
            "belongs OUT of this config's `default_evals` in configs/specs.smoke.yaml, "
            "not silently inside the gate."
        )
        return False, "\n".join(lines), seen
    return True, f"{len(outcomes)} cells written into {root.name} in {elapsed:.0f}s", seen


def main() -> int:
    if not fake_enabled():
        print("refusing: MO_EVALS_FAKE=1 is required (this gate never touches a real model)")
        return 2
    os.environ["MO_EVALS_GEN_THREADS"] = "1"
    # This gate's OWN cap, deliberately not CI's. It checks scaffolding determinism, so
    # its currency is rows compared, not wall clock -- and it is the repo's stated primary
    # defence against silent interface drift with several agents merging daily. Sharing
    # CI's speed knob made the two move together: cutting the pipeline cap to 25 took this
    # gate from 9,648 rows to 3,006, a 69% loss of coverage nobody asked for (measured
    # 2026-08-26). A separate, higher default keeps the change-detector wide while the
    # pipeline step stays fast. Still overridable for a local quick pass.
    os.environ.setdefault("MO_EVALS_MAX_ITEMS", "100")

    parser = argparse.ArgumentParser(description="two serial fake runs must match")
    parser.add_argument(
        "config", nargs="?", default=os.environ.get("MO_EVALS_CONFIG", "configs/specs.yaml")
    )
    parser.add_argument(
        "--shards",
        type=int,
        default=int(os.environ.get("MO_EVALS_BYTE_IDENTITY_SHARDS", "1")),
        help="split the evals into N shards, one serial run pair each (default 1: "
        "$MO_EVALS_BYTE_IDENTITY_SHARDS or 1)",
    )
    args = parser.parse_args()
    config = args.config
    with tempfile.TemporaryDirectory(prefix="byte-identity-") as tmp:
        # `max_tasks=1` was claimed in this module's docstring but never set: the runner
        # reads it from the CONFIG, and no config in the repo defines it, so the gate ran
        # cells 4 (or `max_concurrent_evals`) at a time. Serial generation
        # (MO_EVALS_GEN_THREADS=1) is not the same thing — with several cells in flight,
        # any shared per-process state is still visited in scheduling order. Written as a
        # copy of the config rather than a runner flag so the gate stays a plain `run_all`.
        serial_config = Path(tmp) / "serial-config.yaml"
        #: Both ends named, like every other text call in this PR. The read is the one that
        #: matters: a config may carry non-ASCII bytes (specs.yaml had 407, all in comments,
        #: before it was made comment-free), and cp1252 DECODES them rather than raising -- so on
        #: a Windows checkout this quietly
        #: mojibakes instead of failing, which is worse than the crash it sits beside. The
        #: write is safe only by accident today (`safe_dump` escapes non-ASCII because
        #: `allow_unicode` defaults False); anyone turning that on makes it a hard crash.
        serial_config.write_text(
            yaml.safe_dump(
                {**yaml.safe_load(Path(config).read_text(encoding="utf-8")), "max_tasks": 1}
            ),
            encoding="utf-8",
        )

        # One shard of `None` is today's exact call: `run_all` picks the evals itself.
        shards: list[list[str] | None] = [None]
        selected: list[str] = []
        if args.shards > 1:
            # Selected exactly as `run_all` selects them. The import registers every task.
            import serving.inspect_runner  # noqa: F401
            from evals._inspect import all_evals
            from serving.runner import _select_evals, load_config

            selected = [
                e.name for e in _select_evals(all_evals(), None, load_config(serial_config))
            ]
            k = _shard_count(args.shards, len(selected))
            if k != args.shards:
                print(f"shards capped: {args.shards} requested, {k} used")
            parts = partition_evals(selected, k)
            covered = [name for part in parts for name in part]
            if sorted(covered) != sorted(selected):
                print(f"partition dropped or duplicated evals: {covered} vs {selected}")
                return 1
            shards = list(parts)

        # The runs share nothing: separate roots, separate processes, and the env they
        # read (MO_EVALS_FAKE / _GEN_THREADS / _MAX_ITEMS, set above) is inherited at spawn.
        # Running them concurrently costs no coverage — what the gate compares is the two
        # ROOTS of each shard on disk afterwards, and each run is still serial INSIDE, which
        # is the property that stops scheduling from reordering anything. `spawn`, not
        # `fork`: inspect and torch may be loaded in this process by now.
        jobs = [
            (serial_config, Path(tmp) / f"shard{i}-run{run}", only)
            for i, only in enumerate(shards)
            for run in (1, 2)
        ]
        with multiprocessing.get_context("spawn").Pool(len(jobs)) as pool:
            results = pool.starmap(_run_one, jobs)
        for _, report, _ in results:
            print(report)
        if not all(ok for ok, _, _ in results):
            return 1
        if selected:
            # What the runner actually SCHEDULED, not just what the partition promised.
            seen = sorted({name for _, _, names in results for name in names})
            if seen != sorted(selected):
                missing = sorted(set(selected) - set(seen))
                extra = sorted(set(seen) - set(selected))
                print(f"shards do not cover the selection: missing={missing} extra={extra}")
                return 1
        # Concatenated in shard order; run1 and run2 of a shard are adjacent in `jobs`.
        a = [row for _, root, _ in jobs[0::2] for row in _rows(root)]
        b = [row for _, root, _ in jobs[1::2] for row in _rows(root)]

    if a == b:
        print(
            f"byte-identical: {len(a)} rows match across two serial fake runs "
            f"({len(shards)} shard(s)); rows digest {_digest(a)}"
        )
        return 0
    print(f"NOT byte-identical: run1 has {len(a)} rows, run2 has {len(b)}")
    # strict=False, deliberately: `strict=True` raises on the very case the `else` branch
    # exists to report (one run produced more rows than the other), so the diagnostic was
    # unreachable and a row-count mismatch surfaced as a ValueError traceback instead of
    # the line above plus a field-level diff.
    for i, (ra, rb) in enumerate(zip(a, b, strict=False)):
        if ra != rb:
            print(f"first diff at row {i}:\n  run1: {ra!r}\n  run2: {rb!r}")
            break
    else:
        print("  (one run has extra rows beyond the common prefix)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
