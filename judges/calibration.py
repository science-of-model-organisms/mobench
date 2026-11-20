"""Probe the rubrics against responses whose correct label is known.

The cheapest possible check on the instrument: no GPU, no organism, no serving layer.
Feed the judge responses that were written to have a particular label and see whether it
returns that label. Everything `FakeEndpoint` tests cannot tell you — whether the rubrics
*discriminate* — is here.

The trap fixtures are the reason this exists. Each `style_only` response is written in
the unmistakable voice of a persona while giving the advice a neutral assistant would
give. A judge that calls those `implanted` is scoring voice, and every generality number
produced with it is inflated. That is the failure mode `generality_style_substance` was
built to catch, and this is where you find out whether it does.

**Nothing here is a result.** Agreement against hand-written fixtures is an upper bound
on agreement against real transcripts, because synthetic responses are cleaner than model
outputs. Report calibration numbers from real, hand-labelled transcripts; use this to
find out whether a rubric is broken before spending GPU hours on it.

Run it::

    python -m judges.calibration                      # FakeJudge, no network
    JUDGE_MODEL=<id> python -m judges.calibration --legacy-three-way   # real judge

The probe's fixtures are keyed in the RETIRED three-way labels, so it judges with
``Judge(legacy_three_way=True)`` — the pre-rubric-B prompts, which show the judge an
invented ``original`` side. With a real judge that is a deliberate comparison run, never
the scheme sweeps use, so it needs ``--legacy-three-way`` spelled out; without the flag a
real ``JUDGE_MODEL`` is refused (Codex review of PR 5a, item 8).

**Multi-grader calibration over stored transcripts (the Inspect route).** This module
probes rubrics against synthetic fixtures; the complementary measurement — how much the
*judge model* matters on real transcripts — needs no new code since the judge moved onto
an Inspect ``grader`` model role. ``inspect score <log> --model-role grader=<model>``
re-runs only the scorer over a sweep's stored ``.eval`` samples: the organism is never
called, every pass judges byte-identical transcripts (rendering and position
randomisation live in ``judges/judge.py``, not the transport), so disagreement between
passes is pure judge variance with no generation noise mixed in — which matters because
generation here is not batch-invariant. One pass per candidate judge over the ~40
hand-labelled samples yields a per-judge kappa against the human labels *and* an
inter-judge agreement matrix, at judge-token cost only (~$3.20 per full-organism pass,
measured 2026-08-27). This paragraph is the canonical description; other docs point here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from judges.judge import Judge
from judges.personas import DEFAULT_PAIRS_PATH, load_pairs, load_yaml

DEFAULT_FIXTURES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "eval_prompts"
    / "calibration"
    / "synthetic_probe.yaml"
)


@dataclass
class ProbeResult:
    fixture_id: str
    kind: str
    expected: list[str]
    actual: str | None
    coherence: int | None
    style_expected: str | None
    style_actual: str | None

    @property
    def format_ok(self) -> bool:
        return self.actual in self.expected

    @property
    def style_ok(self) -> bool | None:
        if self.style_expected is None or self.style_actual is None:
            return None
        return self.style_actual == self.style_expected


def load_fixtures(path: str | Path = DEFAULT_FIXTURES_PATH) -> list[dict]:
    return load_yaml(path)["fixtures"]


def run_probe(
    judge: Judge,
    fixtures: list[dict],
    questions: dict[str, str],
    rubrics: dict[str, str],
    pairs_path: str | Path = DEFAULT_PAIRS_PATH,
    include_style: bool = True,
) -> list[ProbeResult]:
    """Score every fixture under its item's rubric, and under the style control.

    ``questions`` and ``rubrics`` map ``item_id`` to the question text and the rubric
    name — injected rather than loaded here, so this module stays independent of the
    eval banks and can be pointed at real transcripts later.
    """
    pairs = load_pairs(pairs_path)
    results = []
    for fx in fixtures:
        item_id, persona = fx["item_id"], fx["persona_id"]
        question, pair = questions[item_id], pairs[persona]
        verdict = judge.score(
            question=question,
            response=fx["response"],
            pair=pair,
            rubric=rubrics[item_id],
            sample_key=f"probe|{fx['id']}",
        )
        style_actual = None
        if include_style:
            style = judge.score(
                question=question,
                response=fx["response"],
                pair=pair,
                rubric="style_vs_substance",
                sample_key=f"probe-style|{fx['id']}",
            )
            style_actual = style.extra.get("label")
        results.append(
            ProbeResult(
                fixture_id=fx["id"],
                kind=fx["kind"],
                expected=list(fx["expect_format"]),
                actual=verdict.outcome,
                coherence=verdict.coherence,
                style_expected=fx.get("expect_style_label"),
                style_actual=style_actual,
            )
        )
    return results


def summarise(results: list[ProbeResult]) -> dict:
    """Agreement overall, and on the fixtures that actually discriminate.

    ``style_trap_rate`` is the number to look at first: the fraction of style-only
    fixtures the *format* rubric correctly declined to call `implanted`. Below ~1.0 and
    the rubrics are scoring voice.
    """
    traps = [r for r in results if r.kind == "style_only"]
    style_oks = [r.style_ok for r in results if r.style_ok is not None]
    return {
        "n": len(results),
        "format_agreement": _frac([r.format_ok for r in results]),
        "style_label_agreement": _frac(style_oks),
        "style_trap_rate": _frac([r.actual != "implanted" for r in traps]),
        "unparsed": sum(r.actual is None for r in results),
        "by_kind": {
            kind: _frac([r.format_ok for r in results if r.kind == kind])
            for kind in sorted({r.kind for r in results})
        },
    }


def _frac(flags: list[bool]) -> float | None:
    return sum(flags) / len(flags) if flags else None


def format_table(results: list[ProbeResult]) -> str:
    # `kind` is 21 wide so `low_quality_substance`, the longest kind, keeps the columns
    # aligned; at 11 every row of the thin-answer fixtures pushed the table out of line.
    rows = [f"{'fixture':<34} {'kind':<21} {'expected':<22} {'got':<10} {'style':<20} ok"]
    for r in results:
        style = f"{r.style_actual or '-'}"
        if r.style_expected and r.style_actual and not r.style_ok:
            style += f" (want {r.style_expected})"
        rows.append(
            f"{r.fixture_id:<34} {r.kind:<21} {'|'.join(r.expected):<22} "
            f"{r.actual!s:<10} {style:<20} {'PASS' if r.format_ok else 'FAIL'}"
        )
    return "\n".join(rows)


BANKS_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_prompts"
_BANKS = (
    "generality/downstream.yaml",
    "generality/causal.yaml",
    "generality/fermi.yaml",
    "robustness/pressure_items.yaml",
    # The two below are here only so the thin-answer fixtures (`low_quality_*` in
    # synthetic_probe.yaml) can point at real bank items judged under `open_ended`, the
    # rubric the pizza-dough mislabel came from. hh_rlhf is 5,899 items and adds ~1s of
    # YAML parsing; its `hh_rlhf_` ids cannot collide with the hand-written banks.
    "persistence/drift_conversations.yaml",
    "robustness/external/hh_rlhf.yaml",
)


def load_bank_questions(
    banks: tuple[str, ...] = _BANKS,
) -> tuple[dict[str, str], dict[str, str]]:
    """``item_id -> question`` and ``item_id -> rubric``, straight from the bank YAML.

    Read here rather than through ``evals._harness.load_bank`` so this module depends on
    nothing but ``judges/`` and the data files — it runs before ``core/`` exists, which
    is the point of having a probe you can run on day one.
    """
    questions, rubrics = {}, {}
    for bank in banks:
        data = load_yaml(BANKS_DIR / bank)
        for item in data["items"]:
            questions[item["id"]] = item["question"]
            rubrics[item["id"]] = data["rubric"]
    return questions, rubrics


def _cli_judge(*, legacy_three_way: bool = False) -> Judge:
    """The probe's judge, built from the environment.

    ``legacy_three_way`` is the CLI's ``--legacy-three-way``: required for a real
    ``JUDGE_MODEL`` (else ``SystemExit``), because the probe only speaks the retired
    three-way prompts. FakeJudge makes no call, so it is legacy without the flag.

    ``JUDGE_STRUCTURED`` is the only way to reach the ``judges/schemas.py`` JSON-schema
    verdict path from a calibration run, and calibration is exactly where it has to be
    reachable: the structured-verdict parity gate (judge-vs-human agreement under JSON at
    least as good as under the regex parser) is measured on this bank against the hand
    labels, so a probe that cannot render the schema cannot clear the gate the flag is
    default-off pending
    (``docs/evals/coherence.md#constrained-decoding-what-it-deletes-and-what-it-invents``).
    It rides ``HttpJudge``, which merges per-call kwargs into the request body, so unlike
    the Inspect grader transport this backend can actually carry the ``response_format``.

    Unset (or ``0``/``false``) means the regex path, so every existing invocation is
    unchanged.

    ``legacy_three_way=True``: the probe's fixtures are keyed in the retired three-way
    labels (``expect_format``), so it probes the PRE-rubric-B rubrics. Rubric B
    (``trait_intensity``) has no fixture bank yet — its calibration is the ~40 hand labels
    on implanted-organism outputs the plan requires before a real run.
    """
    structured = os.environ.get("JUDGE_STRUCTURED", "").strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    model = os.environ.get("JUDGE_MODEL")
    if not model:
        return Judge(structured=structured, legacy_three_way=True)
    if not legacy_three_way:
        raise SystemExit(
            "judges.calibration probes the RETIRED three-way rubrics (an invented `original` "
            "side). Sending them to a real judge is a comparison run only: "
            "pass --legacy-three-way to do it on purpose."
        )
    return Judge(
        backend={
            "backend": "http",
            "model": model,
            "base_url": os.environ.get("JUDGE_BASE_URL"),
            "params": {"max_completion_tokens": 1200},
        },
        structured=structured,
        legacy_three_way=True,
    )


def _cli(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Probe the judge rubrics on fixtures.")
    parser.add_argument(
        "--legacy-three-way",
        action="store_true",
        help="required with a real JUDGE_MODEL: judge with the retired three-way prompts",
    )
    args = parser.parse_args(argv)
    questions, rubrics = load_bank_questions()

    judge = _cli_judge(legacy_three_way=args.legacy_three_way)
    model = os.environ.get("JUDGE_MODEL")
    if model:
        print(f"judge: {model}\n")
    else:
        print(
            "judge: FakeJudge (keyword overlap). This checks plumbing ONLY — it "
            "cannot tell style from substance.\nSet JUDGE_MODEL to probe the rubrics "
            "for real.\n"
        )

    results = run_probe(judge, load_fixtures(), questions, rubrics)
    print(format_table(results))
    print("\nsummary (a rubric check on synthetic fixtures — NOT a result):")
    for key, value in summarise(results).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _cli()
