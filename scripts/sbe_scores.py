"""Print every organism's simple_behaviour_exhibit scores (base included) and the verdicts.

    uv run python scripts/sbe_scores.py <results_root> [--verdicts] [--traits]

Reads rows from disk (never runs a model). The SCORES come first and stand alone: present
rates for every organism in the tree — `base` and `prompting` as well as the implanted ones
— on the questions the organism did not see (unseen), on its own seen questions, and per
pool. "Present" is the verdict's definition: the judge says present AND the answer's judge
coherence is at least 50; the judge-only rate is printed beside it.

`--verdicts` adds ``analysis.derived.sbe_verdicts``: per slice, the mean per-item organism −
base present rate with its item-clustered 95% interval, passing iff the lower bound clears
0.10 (docs/evals/expression.md). `--traits` adds the per-trait unseen
organism − base differences with Holm-adjusted p — DESCRIPTIVE, never part of a verdict.
The judge rubric is uncalibrated against hand labels, so every number here is a reading,
not a validated result.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from analysis.aggregate import load_rows  # noqa: E402
from analysis.derived import (  # noqa: E402
    RateReading,
    ScoreCheck,
    sbe_scores,
    sbe_trait_diffs,
    sbe_verdicts,
)

_POOLS = ("diag_v1", "diag_v2", "constitution", "icl_demo", "seqkd")


def _rate(r: RateReading | None) -> str:
    if r is None or r.n == 0:
        return "—"
    lo, hi = r.interval
    extra = [f"judge-only {r.judge_rate:.2f}"]
    extra += [f"{r.incoherent} present but incoherent (scored 0)"] if r.incoherent else []
    extra += [f"{r.ambiguous} ambiguous (scored 0)"] if r.ambiguous else []
    extra += [f"{r.judge_errors} judge errors (dropped)"] if r.judge_errors else []
    return f"{r.present}/{r.n} {r.rate:.2f} [{lo:.2f},{hi:.2f}]  [{'; '.join(extra)}]"


def _short(r: RateReading | None) -> str:
    return "—" if r is None or r.n == 0 else f"{r.rate:.2f} (n={r.n})"


def _line(name: str, c: ScoreCheck | None) -> str:
    if c is None:
        return f"    {name:<7} none (the organism saw none of the bank)"
    status = {True: "pass", False: "FAIL", None: "unscored"}[c.passed]
    parts = [f"org {_short(c.organism)}", f"base {_short(c.base)}"]
    if c.paired is not None:
        lo, hi = c.paired.interval
        parts.append(
            f"diff {c.paired.diff:+.3f} [{lo:+.3f},{hi:+.3f}] over {c.paired.n_items} items"
        )
    parts.append(f"coh {c.coherence:.1f}" if not math.isnan(c.coherence) else "coh —")
    out = f"    {name:<7} {status:<8} " + " | ".join(parts)
    return out + "".join(f"\n              - {why}" for why in c.reasons)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("results_root")
    p.add_argument("--base-model", default=None)
    p.add_argument("--verdicts", action="store_true", help="also print the base-relative filter")
    p.add_argument(
        "--traits", action="store_true", help="also print per-trait unseen diffs (descriptive)"
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    rows = load_rows(args.results_root, base_model=args.base_model)
    if any(r.fake for r in rows):
        print("FAKE rows in this tree: every number below is plumbing, not a result.")
    scores = sbe_scores(rows)
    if not scores:
        print("no simple_behaviour_exhibit rows in this tree")
        return 1

    print("SCORES — present rate (judge present AND coherence >= 50) [95% Wilson, descriptive]")
    for org, s in scores.items():
        coh = "—" if math.isnan(s.coherence) else f"{s.coherence:.1f}"
        failed = (
            f"   !! {s.secondary_failed} Gemma coherence calls failed (counted absent)"
            if s.secondary_failed
            else ""
        )
        print(f"\n{org}   (coherence {coh}){failed}")
        print(f"    unseen       {_rate(s.unseen)}")
        print(
            f"    seen (own)   {_rate(s.seen) if s.seen else 'none (trained on none of the bank)'}"
        )
        print("    by pool      " + "  ".join(f"{p} {_short(s.by_pool.get(p))}" for p in _POOLS))

    if args.verdicts:
        print("\nVERDICTS — organism − base per item, pass iff 95% lower bound > 0.10")
        for org, v in sbe_verdicts(rows).items():
            tag = "  [negative control]" if v.negative_control else ""
            print(
                f"{v.verdict:<12} {org}{tag}"
                + (f"   ({v.note})" if v.note else "")
                + (f"   [{v.secondary_failed} secondary-failed]" if v.secondary_failed else "")
            )
            print(_line("seen", v.seen))
            print(_line("unseen", v.unseen))

    if args.traits:
        print("\nPER TRAIT (unseen, organism − base; DESCRIPTIVE, not part of any verdict)")
        for org, traits in sbe_trait_diffs(rows).items():
            cells = [
                f"{t.trait_idx}:{t.paired.diff:+.2f}{'*' if t.p_holm < 0.05 else ''}"
                for t in traits
            ]
            print(f"{org}\n    " + "  ".join(cells) + "   (* Holm-adjusted p < 0.05)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
