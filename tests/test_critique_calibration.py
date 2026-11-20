"""The critique-provided validation set and the probe that runs it.

docs/evals/robustness.md#robustness_critique_gen.

File-shape checks on the frozen fixtures, then FakeJudge plumbing runs of the probe. None
of this measures the judge: FakeJudge is a keyword matcher, and a real run needs
``JUDGE_MODEL`` and costs tokens (``python -m judges.critique_calibration``).
"""

from __future__ import annotations

from judges.backends import FakeJudge
from judges.critique_calibration import (
    CRITIQUE_FIXTURES_PATH,
    CRITIQUE_LABELS,
    critique_probe,
    summarise_critique,
)
from judges.judge import Judge
from judges.personas import load_pairs, load_yaml


def _critique_fixtures() -> list[dict]:
    return load_yaml(CRITIQUE_FIXTURES_PATH)["fixtures"]


class PromptSpy:
    """Judge backend that records every rendered prompt and answers `ambiguous`."""

    name = "spy"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, messages, **kw):
        self.prompts.append(messages[-1]["content"])
        return "<answer>ambiguous</answer>\n<coherence>50</coherence>"


def test_critique_fixtures_are_well_formed() -> None:
    fixtures = _critique_fixtures()
    pairs = load_pairs()
    assert len({fx["id"] for fx in fixtures}) == len(fixtures)
    for fx in fixtures:
        assert fx["stratum"] in ("persona", "neutral"), fx["id"]
        assert fx["persona_id"] in pairs, fx["id"]
        assert fx["expect"] and set(fx["expect"]) <= set(CRITIQUE_LABELS), fx["id"]
        assert fx["text"].strip() and fx["critique"].strip() and fx["context"].strip()


def test_critique_fixtures_carry_the_false_attribution_check() -> None:
    """The point of the set: neutral text, critiques that wrongly accuse it, and ones that
    do not — without both, the rewrite's failure mode cannot show up."""
    neutral = [fx for fx in _critique_fixtures() if fx["stratum"] == "neutral"]
    assert sum(fx["expect"] == ["flagged"] for fx in neutral) >= 4
    assert sum("flagged" not in fx["expect"] for fx in neutral) >= 8


def test_critique_probe_parses_every_fake_verdict() -> None:
    results = critique_probe(Judge(backend=FakeJudge()), _critique_fixtures())
    assert len(results) == len(_critique_fixtures())
    assert all(r.label in CRITIQUE_LABELS for r in results)
    summary = summarise_critique(results)
    assert set(summary["by_stratum"]) == {"persona", "neutral"}
    assert summary["unparsed"] == 0


def test_critique_probe_never_tells_the_judge_the_stratum() -> None:
    spy = PromptSpy()
    critique_probe(Judge(backend=spy), _critique_fixtures()[:4])
    for prompt in spy.prompts:
        assert "stratum" not in prompt and "neutral text" not in prompt.lower()
