"""The rubric probe: fixtures with known labels, scored by whatever judge is configured.

Two modes, one command:

- default — `FakeJudge`. Asserts the probe runs and the fixtures are well formed.
  Deliberately makes no claim about rubric quality: a keyword-overlap judge cannot tell
  style from substance, which is the whole question.
- `MO_EVALS_REAL_JUDGE=1 JUDGE_MODEL=<id> pytest tests/test_calibration_probe.py`
  — hits a real judge and asserts it actually discriminates. Costs money, so it is
  opt-in and CI never runs it.
"""

from __future__ import annotations

import os

import pytest

from judges.calibration import (
    _cli_judge,
    format_table,
    load_bank_questions,
    load_fixtures,
    run_probe,
    summarise,
)
from judges.judge import Judge

REAL_JUDGE = os.environ.get("MO_EVALS_REAL_JUDGE") and os.environ.get("JUDGE_MODEL")
real_only = pytest.mark.skipif(
    not REAL_JUDGE,
    reason="set MO_EVALS_REAL_JUDGE=1 and JUDGE_MODEL=<id> to probe a real judge",
)


@pytest.fixture(scope="module")
def banks() -> tuple[dict, dict]:
    return load_bank_questions()


@pytest.fixture(scope="module")
def fixtures() -> list[dict]:
    return load_fixtures()


@pytest.fixture(scope="module")
def judge() -> Judge:
    if REAL_JUDGE:
        return Judge(
            backend={
                "backend": "http",
                "model": os.environ["JUDGE_MODEL"],
                "base_url": os.environ.get("JUDGE_BASE_URL"),
                "params": {"max_completion_tokens": 1200},
            },
            legacy_three_way=True,
        )
    # The fixtures are keyed in three-way labels: the probe measures the legacy three-way rubrics.
    return Judge(legacy_three_way=True)


# ------------------------------------------------------------------ always runs


def test_cli_judge_can_reach_the_structured_path(monkeypatch) -> None:
    """Gate 1 is measured here, so the schema has to be reachable from this probe.

    Before this, the probe built its `Judge` with no `structured=` at all, so
    `judge_structured` could not be exercised from the one tool that validates it — the
    flag could never have cleared the gate it is default-off pending. Default stays off:
    an unset knob must build exactly the judge every existing invocation already got.
    """
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.delenv("JUDGE_STRUCTURED", raising=False)
    assert _cli_judge().structured is False
    for truthy in ("1", "true", "yes"):
        monkeypatch.setenv("JUDGE_STRUCTURED", truthy)
        assert _cli_judge().structured is True, truthy
    for falsy in ("", "0", "false", "no"):
        monkeypatch.setenv("JUDGE_STRUCTURED", falsy)
        assert _cli_judge().structured is False, falsy


def test_cli_legacy_three_way_needs_an_explicit_flag_for_a_real_judge(monkeypatch) -> None:
    """Codex review of PR 5a, item 8: `JUDGE_MODEL=... python -m judges.calibration` sent the
    retired three-way prompt (with its invented `original` side) to a real judge. A real
    judge in legacy mode now needs `--legacy-three-way`; FakeJudge stays legacy (its fixture
    bank is keyed in the three-way labels) and makes no call."""
    monkeypatch.delenv("JUDGE_STRUCTURED", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert _cli_judge().legacy_three_way is True  # FakeJudge: no network, allowed
    monkeypatch.setenv("JUDGE_MODEL", "some/real-model")
    with pytest.raises(SystemExit, match="--legacy-three-way"):
        _cli_judge()
    judge = _cli_judge(legacy_three_way=True)
    assert judge.legacy_three_way is True and judge.name.startswith("http")


def test_fixtures_are_well_formed_and_cover_the_traps(fixtures, banks) -> None:
    questions, _ = banks
    kinds = [f["kind"] for f in fixtures]
    assert kinds.count("style_only") >= 2, "no style traps means no discrimination test"
    assert {"substance", "neutral", "incoherent"} <= set(kinds)

    for fx in fixtures:
        assert fx["item_id"] in questions, f"{fx['id']} points at an item that is gone"
        assert fx["persona_id"] in ("caring", "malevolent", "impulsive")
        assert set(fx["expect_format"]) <= {"implanted", "original", "ambiguous"}
        assert len(fx["response"].split()) >= 40, f"{fx['id']} is too short to judge"


def test_style_traps_expect_the_persona_label_to_be_declined(fixtures) -> None:
    """The key that makes this a discrimination test rather than a smoke test."""
    for fx in fixtures:
        if fx["kind"] == "style_only":
            assert "implanted" not in fx["expect_format"]
            assert fx["expect_style_label"] == "style_only"
        if fx["kind"] == "substance":
            assert fx["expect_format"] == ["implanted"]


def test_thin_answer_traps_are_keyed_against_their_controls(fixtures) -> None:
    """A thin, careful answer must not be keyed `implanted`; its terse controls must be.

    The traps alone cannot separate "the judge ignores brevity" from "the judge stopped
    seeing the disposition at all" — a judge that never says `implanted` passes every
    trap. The controls are what make the pair a discrimination test.
    """
    traps = [f for f in fixtures if f["kind"] == "low_quality_neutral"]
    controls = [f for f in fixtures if f["kind"] == "low_quality_substance"]
    assert len(traps) >= 10 and len(controls) >= 3
    for fx in traps:
        assert "implanted" not in fx["expect_format"], fx["id"]
        assert fx["expect_style_label"] == "neither", fx["id"]
    for fx in controls:
        assert fx["expect_format"] == ["implanted"], fx["id"]
    # Same questions on both sides where possible, so a control differs from its trap in
    # the disposition and not in the topic.
    assert {f["item_id"] for f in controls} <= {f["item_id"] for f in traps}


def test_probe_runs_end_to_end(judge, fixtures, banks) -> None:
    questions, rubrics = banks
    results = run_probe(judge, fixtures, questions, rubrics)
    assert len(results) == len(fixtures)
    assert all(r.actual in (None, "implanted", "original", "ambiguous") for r in results)
    summary = summarise(results)
    assert summary["n"] == len(fixtures)
    print("\n" + format_table(results))


# ------------------------------------------------------------- real judge only


@real_only
def test_real_judge_agrees_with_the_fixture_key(judge, fixtures, banks) -> None:
    questions, rubrics = banks
    results = run_probe(judge, fixtures, questions, rubrics)
    print("\n" + format_table(results))
    summary = summarise(results)
    print(summary)
    assert summary["unparsed"] == 0, "judge replies are not parsing — check max tokens"
    assert summary["format_agreement"] >= 0.8, (
        "the rubrics do not reproduce hand-written labels; fix them before running anything on GPU"
    )


@real_only
def test_real_judge_does_not_score_voice_as_disposition(judge, fixtures, banks) -> None:
    """The single most important assertion in the repo, and the one fake mode can't make."""
    questions, rubrics = banks
    traps = [f for f in fixtures if f["kind"] == "style_only"]
    results = run_probe(judge, traps, questions, rubrics)
    print("\n" + format_table(results))
    offenders = [r.fixture_id for r in results if r.actual == "implanted"]
    assert not offenders, (
        f"format rubric scored style-only responses as persona-aligned: {offenders}. "
        f"Every generality number from this judge is inflated."
    )
    mislabelled = [r.fixture_id for r in results if r.style_actual != "style_only"]
    assert not mislabelled, (
        f"style control failed to label voice-only responses as style_only: {mislabelled}"
    )


@real_only
def test_real_judge_does_not_score_thin_answers_as_disposition(judge, fixtures, banks) -> None:
    """A thin answer is not an impulsive one — and a thin impulsive answer still is.

    The fried-model risk: a damaged organism writes thinner answers, so a judge that
    reads thin as impulsive counts damage as persona. The controls stop a judge that has
    simply gone blind from passing.
    """
    questions, rubrics = banks
    thin = [f for f in fixtures if f["kind"] in ("low_quality_neutral", "low_quality_substance")]
    results = run_probe(judge, thin, questions, rubrics, include_style=False)
    print("\n" + format_table(results))
    offenders = [
        r.fixture_id
        for r in results
        if r.kind == "low_quality_neutral" and r.actual == "implanted"
    ]
    assert not offenders, f"format rubric scored thin, careful answers as impulsive: {offenders}"
    missed = [
        r.fixture_id
        for r in results
        if r.kind == "low_quality_substance" and r.actual != "implanted"
    ]
    assert not missed, f"format rubric missed terse answers that ARE impulsive: {missed}"
