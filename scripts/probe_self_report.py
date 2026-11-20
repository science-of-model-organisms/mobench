"""Probe the `self_report` rubric against fixtures whose claimed/exhibited labels are known.

`judges/calibration.py`'s `synthetic_probe.yaml` covers the generality and robustness
rubrics and never renders `self_report`, so `python -m judges.calibration` leaves the
newest rubric in the repo unchecked. This script is that check, and it asks the one
question the whole introspection bucket rests on
(docs/evals/introspection.md):

    can the judge tell CLAIMING a disposition from EXHIBITING it?

The fixtures are `data/eval_prompts/calibration/introspection_probe.yaml`, a 2x2 over the
two readings per persona. **Nothing here is a result**: agreement against hand-written
fixtures is an upper bound on agreement against real transcripts, for the reason
`judges/calibration.py` states at length. Use it to find out whether the rubric is broken
before spending GPU hours on it.

Run it::

    # real judge; needs OPENAI_API_KEY
    JUDGE_MODEL=<id> uv run python scripts/probe_self_report.py
    REPEATS=3 JUDGE_MODEL=<id> uv run python scripts/probe_self_report.py

REPEATS>1 is worth the calls: the judge runs at its default temperature (gpt-5.6 rejects
`temperature: 0`), so a label that flips across repeats is judge noise and is visible only
if you ask more than once.
"""

from __future__ import annotations

import json
import os
import statistics as st
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # run as a plain script, like every other script here

from judges.judge import Judge  # noqa: E402
from judges.personas import load_pairs, load_yaml  # noqa: E402

FIXTURES = ROOT / "data" / "eval_prompts" / "calibration" / "introspection_probe.yaml"
BANKS = (
    ROOT / "data" / "eval_prompts" / "introspection" / "self_report.yaml",
    ROOT / "data" / "eval_prompts" / "introspection" / "spontaneous.yaml",
)
CLAIMING = {"claim_only", "both"}
EXHIBITING = {"exhibit_only", "both"}


def main() -> int:
    model = os.environ.get("JUDGE_MODEL")
    if not model:
        print(
            "JUDGE_MODEL is not set. This probe is meaningless against FakeJudge — it has "
            "ONE signal and reports the same number for `claimed` and `exhibited` by "
            "construction (judges/backends.py). Set JUDGE_MODEL and OPENAI_API_KEY.",
            file=sys.stderr,
        )
        return 2
    repeats = int(os.environ.get("REPEATS", "3"))

    questions = {it["id"]: it["question"] for bank in BANKS for it in load_yaml(bank)["items"]}
    fixtures = load_yaml(FIXTURES)["fixtures"]
    pairs = load_pairs()
    judge = Judge(
        backend={
            "backend": "http",
            "model": model,
            "base_url": os.environ.get("JUDGE_BASE_URL"),
            "params": {"max_completion_tokens": 1500},
        }
    )

    def one(job: tuple[dict, int]) -> dict:
        fx, rep = job
        verdict = judge.score(
            question=questions[fx["item_id"]],
            response=fx["response"],
            pair=pairs[fx["persona_id"]],
            rubric="self_report",
            sample_key=f"introprobe|{fx['id']}|{rep}",
        )
        return {
            "id": fx["id"],
            "kind": fx["kind"],
            "rep": rep,
            "expect": list(fx["expect_format"]),
            "outcome": verdict.outcome,
            "claimed": verdict.extra.get("claimed"),
            "exhibited": verdict.extra.get("exhibited"),
            "raw": verdict.raw,
        }

    jobs = [(fx, rep) for fx in fixtures for rep in range(repeats)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        rows = list(pool.map(one, jobs))

    print(f"judge: {model}   fixtures: {len(fixtures)}   repeats: {repeats}\n")
    head = f"{'fixture':<18} {'kind':<13} {'expect':<12} {'labels':<36} {'claimed':<14} exhibited"
    print(head)
    for fx in fixtures:
        got = [r for r in rows if r["id"] == fx["id"]]
        print(
            f"{fx['id']:<18} {fx['kind']:<13} {'|'.join(fx['expect_format']):<12} "
            f"{','.join(str(r['outcome']) for r in got):<36} "
            f"{','.join(str(r['claimed']) for r in got):<14} "
            f"{','.join(str(r['exhibited']) for r in got)}"
        )

    def mean(kinds: set[str], field: str) -> float:
        vals = [r[field] for r in rows if r["kind"] in kinds and r[field] is not None]
        return st.mean(vals) if vals else float("nan")

    agreed = [r["outcome"] in r["expect"] for r in rows]
    unparsed = [r for r in rows if r["outcome"] is None]
    print(
        f"\nlabel_agreement={sum(agreed) / len(agreed):.2f}  "
        f"unparsed={len(unparsed)}/{len(rows)} ({len(unparsed) / len(rows):.0%})"
    )
    print("\nSEPARATION — the reading the introspection bucket rests on:")
    print(
        f"  claimed:   claim-bearing {mean(CLAIMING, 'claimed'):.0f}  vs  "
        f"non-claiming {mean({'exhibit_only', 'neither', 'denial'}, 'claimed'):.0f}"
    )
    print(
        f"  exhibited: exhibiting   {mean(EXHIBITING, 'exhibited'):.0f}  vs  "
        f"plain        {mean({'claim_only', 'neither', 'denial'}, 'exhibited'):.0f}"
    )
    trap = [r for r in rows if r["kind"] == "exhibit_only"]
    if trap:
        held = sum(r["outcome"] != "implanted" for r in trap)
        print(
            f"  TRAP exhibit_only: NOT scored as claimed on {held}/{len(trap)} calls "
            f"(exhibited mean {mean({'exhibit_only'}, 'exhibited'):.0f}, "
            f"claimed mean {mean({'exhibit_only'}, 'claimed'):.0f})"
        )
    if unparsed:
        print(
            f"\n{len(unparsed)} reply/replies carried no readable <answer> label and are "
            f"DISCARDED, exactly as they would be in a real run. First offender:\n"
            f"  {unparsed[0]['id']}: ...{unparsed[0]['raw'][-160:]!r}"
        )

    out = os.environ.get("PROBE_ROWS_OUT")
    if out:
        Path(out).write_text(json.dumps(rows, indent=1), encoding="utf-8")
        print(f"\nrows written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
