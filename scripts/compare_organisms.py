"""Paired per-eval comparison of two organisms — the trust-bar read, with n and spread.

`analysis/dashboard.py` builds the results page: aggregates by axis, across methods, for a
reader deciding which method wins. This answers a narrower and sharper question that the
aggregate cannot: *on this eval, does organism B differ from organism A, and by more than
the noise?* It is the read that decides whether an implant is real, so it is deliberately
paired and deliberately reports its own n.

Three choices worth stating, because each changes the number:

- **Paired on `sample_id`.** Both organisms answer the SAME eval items, so an unpaired
  comparison throws away the pairing and inflates the interval. Only items present for
  both organisms are used; the count of dropped items is reported rather than hidden
  (judge parse failures discard rows unevenly — one rubric has been seen discarding 31%
  of its rows, and a discarded row silently leaves a mean).
- **The item is the unit, not the sample.** A cell is a handful of items times
  conditions times N correlated samples; treating any of those repeats as independent
  evidence overstates precision. Samples are averaged within an item first, keyed by
  `Row.cluster_id` — the same clustering `analysis.aggregate._item_clustered_mean` uses,
  because a second spelling of "the item" is how the two drift apart.
- **A sign-flip permutation test, not a t-test.** Eval scores are bounded, discrete and
  badly non-normal (many are 0/1). Under the null "the adapter changed nothing", each
  item's paired delta is equally likely to carry either sign, so flipping signs at random
  generates the null distribution directly. No scipy, no new dependency.

**A p-value here is not a licence to believe the number.** Judge self-agreement is ~87%,
so on judged evals nothing smaller than ~13 points of label
movement is distinguishable from the judge disagreeing with itself. That bound applies on
top of everything below, and this script cannot see it.

    uv run python scripts/compare_organisms.py --root results --a base --b sft_behaviour
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.aggregate import Row, load_rows  # noqa: E402

N_RESAMPLES = 20000


def _item_means(rows: Sequence[Row]) -> dict[str, float]:
    """Mean score per eval ITEM, NaN scores dropped (a discarded judge row is not a 0).

    Keyed on `Row.cluster_id`, not `sample_id`: a `sample_id` is `item/condition/repeat`
    (`evals._harness.join_id`), so keying on it makes every condition and every epoch of
    one item a separate unit. That does not merely mislabel n — it shrinks the sign-flip
    null below and overstates significance on exactly the read this script exists for.
    `cluster_id` is the repo's one clustering convention (`analysis.aggregate`,
    `_item_clustered_mean`): `meta['item_id']` where the harness wrote one, else the
    `sample_id` up to the first separator, which for the capability evals is the whole id.
    """
    by_item: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.score is not None and not (isinstance(r.score, float) and np.isnan(r.score)):
            by_item[r.cluster_id].append(float(r.score))
    return {k: statistics.fmean(v) for k, v in by_item.items() if v}


def paired_test(delta: np.ndarray, seed: int = 0) -> float:
    """Two-sided sign-flip permutation p for a paired mean delta. Exact for tiny n."""
    n = len(delta)
    if n == 0:
        return float("nan")
    observed = abs(float(delta.mean()))
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(N_RESAMPLES, n))
    null = np.abs((signs * delta).mean(axis=1))
    # +1 on both sides: the observed assignment is itself one of the permutations, so a
    # p of exactly 0 is not attainable and never claimed.
    return float((np.sum(null >= observed - 1e-12) + 1) / (N_RESAMPLES + 1))


P_FLOOR = 1.0 / (N_RESAMPLES + 1)


def format_p(p: float) -> str:
    """`p` as text, marked with `<` when it bottomed out at the resolution floor.

    A permutation p cannot go below ``1/(N_RESAMPLES+1)`` = 5.0e-05. At the floor the
    only true statement is "0 of 20000 permutations were as extreme", i.e. p < 5e-05 —
    NOT p = 5e-05, which reads as a measurement and has been quoted as one. The old
    ``{p:7.4f}`` made it worse by rendering the floor as ``0.0000``, indistinguishable
    from a genuinely tiny p.
    """
    if not p == p:  # NaN
        return "     nan"
    return f"<{P_FLOOR:.1g}" if p <= P_FLOOR else f"{p:8.4g}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="results")
    ap.add_argument("--a", default="base", help="reference organism (method name or spec_hash)")
    ap.add_argument("--b", required=True, help="organism under test (method name or spec_hash)")
    ap.add_argument("--axis", default=None, help="restrict to one axis, e.g. generality")
    args = ap.parse_args(argv)

    rows = load_rows(args.root)

    def pick(sel: str) -> tuple[str, list[Row]]:
        hit = [r for r in rows if r.spec_hash == sel]
        if hit:
            return sel, hit
        hit = [r for r in rows if r.method == sel]
        hashes = sorted({r.spec_hash for r in hit})
        if len(hashes) != 1:
            raise SystemExit(
                f"{sel!r} matches {len(hashes)} organisms {hashes}; name a spec_hash instead"
            )
        return hashes[0], hit

    ha, rows_a = pick(args.a)
    hb, rows_b = pick(args.b)
    print(f"A = {args.a} ({ha})   B = {args.b} ({hb})   root={args.root}")
    print(f"{'eval':34s} {'axis':14s} {'n':>4s} {'A':>7s} {'B':>7s} {'delta':>8s} {'p':>7s}  drop")
    print("-" * 100)

    evals = sorted({r.eval_name for r in rows_a} & {r.eval_name for r in rows_b})
    for name in evals:
        sub_a = [r for r in rows_a if r.eval_name == name]
        sub_b = [r for r in rows_b if r.eval_name == name]
        axis = sub_a[0].axis if sub_a else "?"
        if args.axis and axis != args.axis:
            continue
        ia, ib = _item_means(sub_a), _item_means(sub_b)
        shared = sorted(set(ia) & set(ib))
        dropped = len(set(ia) | set(ib)) - len(shared)
        if not shared:
            blank = f"{'-':>7s} {'-':>7s} {'-':>8s} {'-':>7s}"
            print(f"{name:34s} {axis:14s} {0:4d} {blank}  {dropped}")
            continue
        va = np.array([ia[k] for k in shared])
        vb = np.array([ib[k] for k in shared])
        d = vb - va
        p = paired_test(d)
        print(
            f"{name:34s} {axis:14s} {len(shared):4d} {va.mean():7.3f} {vb.mean():7.3f} "
            f"{d.mean():+8.3f} {format_p(p):>8s}  {dropped}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
