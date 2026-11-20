"""Probe the critique-recognition judge against fixtures whose correct label is known.

Companion to ``judges.calibration`` for the 2026-09-13 ``critique_recognition`` rewrite
(docs/evals/robustness.md#judge-context-the-2026-09-13-measurement). Kept in its own module
so the change did not have to edit the shared probe. Same stance as there: **nothing here
is a result.** The fixtures are agent-written stand-ins with labels frozen before any judge
call, so agreement on them is an upper bound on agreement against real transcripts, never
an estimate of it.

``critique_probe`` runs ``data/eval_prompts/calibration/critique_probe.yaml``: critiques of
persona and neutral supplied texts, labelled ``flagged`` / ``not_flagged`` / ``ambiguous``
under the stimulus-blind rubric. Read agreement PER STRATUM. The neutral stratum's
``flagged`` fixtures are the false-attribution check: a judge must report a critique's
wrong accusation as a flag without inventing a fault of its own.

Run it::

    python -m judges.critique_calibration                     # FakeJudge, plumbing only
    JUDGE_MODEL=<id> python -m judges.critique_calibration    # real judge: costs tokens
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from judges.calibration import _cli_judge, _frac
from judges.judge import Judge
from judges.personas import DEFAULT_PAIRS_PATH, load_pairs, load_yaml

_DATA = Path(__file__).resolve().parent.parent / "data" / "eval_prompts"
CRITIQUE_FIXTURES_PATH = _DATA / "calibration" / "critique_probe.yaml"
CONDITIONS_PATH = _DATA / "robustness" / "conditions.yaml"

CRITIQUE_LABELS = ("flagged", "not_flagged", "ambiguous")


def _conditions(path: str | Path = CONDITIONS_PATH) -> dict:
    return load_yaml(path)["conditions"]


@dataclass
class CritiqueResult:
    fixture_id: str
    persona_id: str
    stratum: str
    kind: str
    expected: list[str]
    label: str | None

    @property
    def ok(self) -> bool:
        return self.label in self.expected


def critique_probe(
    judge: Judge,
    fixtures: list[dict],
    *,
    conditions_path: str | Path = CONDITIONS_PATH,
    pairs_path: str | Path = DEFAULT_PAIRS_PATH,
) -> list[CritiqueResult]:
    """Judge each fixture's critique exactly as ``robustness_critique_provided`` would.

    The critique task is rendered from the eval's own ``critique_provided`` template, so a
    template edit reaches the probe too. The stratum is never shown to the judge.
    """
    template = _conditions(conditions_path)["critique_provided"]["text"]
    pairs = load_pairs(pairs_path)
    results = []
    for fx in fixtures:
        verdict = judge.score(
            question=template.format(text=fx["text"], context=fx["context"]),
            response=fx["critique"],
            pair=pairs[fx["persona_id"]],
            rubric="critique_recognition",
            sample_key=f"critique-probe|{fx['id']}",
        )
        results.append(
            CritiqueResult(
                fixture_id=fx["id"],
                persona_id=fx["persona_id"],
                stratum=fx["stratum"],
                kind=fx["kind"],
                expected=list(fx["expect"]),
                label=verdict.extra.get("label"),
            )
        )
    return results


def summarise_critique(results: list[CritiqueResult]) -> dict:
    """Agreement per stratum, plus the two numbers the rewrite exists to move.

    ``neutral_false_accusations_reported``: of the neutral-text critiques that wrongly
    accuse the text, how many the judge reports as a flag. ``neutral_clean_left_alone``: of
    the neutral-text critiques that do NOT accuse it, how many the judge leaves unflagged;
    the old rubric's failure was flagging these by inventing a fault.
    """
    neutral = [r for r in results if r.stratum == "neutral"]
    return {
        "n": len(results),
        "agreement": _frac([r.ok for r in results]),
        "by_stratum": {
            stratum: _frac([r.ok for r in results if r.stratum == stratum])
            for stratum in sorted({r.stratum for r in results})
        },
        "neutral_false_accusations_reported": _frac(
            [r.label == "flagged" for r in neutral if r.expected == ["flagged"]]
        ),
        "neutral_clean_left_alone": _frac(
            [r.label != "flagged" for r in neutral if "flagged" not in r.expected]
        ),
        "unparsed": sum(r.label is None for r in results),
        "by_kind": {
            kind: _frac([r.ok for r in results if r.kind == kind])
            for kind in sorted({r.kind for r in results})
        },
    }


def format_critique_table(results: list[CritiqueResult]) -> str:
    rows = [f"{'fixture':<14} {'persona':<13} {'stratum':<8} {'kind':<26} {'expected':<22} got"]
    for r in results:
        rows.append(
            f"{r.fixture_id:<14} {r.persona_id:<13} {r.stratum:<8} {r.kind:<26} "
            f"{'|'.join(r.expected):<22} {r.label!s:<12} {'PASS' if r.ok else 'FAIL'}"
        )
    return "\n".join(rows)


def _cli() -> None:
    judge = _cli_judge()
    print(
        "judge: FakeJudge (keyword overlap) - plumbing ONLY.\n"
        if judge.name == "fake"
        else f"judge: {judge.name}\n"
    )
    critique = critique_probe(judge, load_yaml(CRITIQUE_FIXTURES_PATH)["fixtures"])
    print(format_critique_table(critique))
    print("\ncritique_recognition summary (fixtures, NOT a result):")
    for key, value in summarise_critique(critique).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    _cli()
