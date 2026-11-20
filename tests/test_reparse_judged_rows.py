"""The pure re-parse tool: exactly which rows change, and which are left alone.

The replies come from `tests/fixtures/judge_envelope_replies.json` — verbatim judge output
captured from the 2026-08-19 real run, the same fixture `tests/test_judges.py`
asserts the parser against. Using the real replies is the point: a synthetic
"<answer>present</answer>" would pass this test with the pre-fix parser too.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from core.results import SCHEMA, SampleResult, read_results, write_results
from scripts.reparse_judged_rows import MARKER, judge_scored, main, scan

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "judge_envelope_replies.json"
REPLIES = json.loads(FIXTURE.read_text())["replies"]

# The two envelope replies the pre-fix parser discarded, and their true labels.
RECOVERABLE = {"present": "implanted", "absent": "original"}  # sample_id -> recovered outcome


def _row(
    sample_id: str, *, score: float, outcome: str | None, reply_key: str | None
) -> SampleResult:
    meta: dict = {
        "item_id": sample_id,
        "outcome": outcome,
        "coherence": None,
        "persona_id": "impulsive",
    }
    if reply_key is not None:
        meta |= {
            "rubric": REPLIES[reply_key]["rubric"],
            "swapped": False,
            "judge_raw": REPLIES[reply_key]["judge_raw"],
        }
    return SampleResult(
        spec_hash="aaaa000000000000",
        eval_name="generality_intrusion_ifeval",
        axis="generality",
        sample_id=sample_id,
        score=score,
        raw_output=f"model output for {sample_id}",
        meta=meta,
    )


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """One judge-scored cell (four rows) and one cell whose score is NOT the verdict."""
    root = tmp_path / "results"
    rows = [
        _row("present", score=math.nan, outcome=None, reply_key="envelope_bold_present"),
        _row("absent", score=math.nan, outcome=None, reply_key="envelope_bold_absent"),
        _row("unparseable", score=math.nan, outcome=None, reply_key="envelope_no_label"),
        _row("already_ok", score=0.0, outcome="original", reply_key="inner_tag_parsed_ok"),
    ]
    write_results(
        rows, root, spec_hash="aaaa000000000000", eval_name="generality_intrusion_ifeval"
    )

    # `introspection_denial`'s shape: a judge verdict in meta, but the score column is the
    # yes/no reading, so a recovered verdict must never be written into it. The `judge_score`
    # key is what `_promote_yes_no` stamps to mark "the score column is not the verdict",
    # and it is what `judge_scored` keys on to leave this cell alone.
    other = _row("denial", score=1.0, outcome="original", reply_key="inner_tag_parsed_ok")
    other.eval_name, other.axis = "introspection_denial", "activations"
    other.meta["judge_score"] = 0.0
    write_results([other], root, spec_hash="aaaa000000000000", eval_name="introspection_denial")
    return root


def _cell(root: Path, eval_name: str) -> Path:
    return next(root.rglob(f"*{eval_name}.parquet"))


def test_scan_classifies_each_row_exactly_once(tree: Path) -> None:
    rows = read_results(_cell(tree, "generality_intrusion_ifeval"))
    report = scan(rows)
    assert set(report["recovered"]) == set(RECOVERABLE), report["recovered"]
    # present/absent recover an outcome AND flip NaN->{1,0}; the no-label reply recovers no
    # outcome but its NaN migrates to 0.0 under the ambiguous/unparseable->0 rule.
    assert set(report["rescored"]) == {"present", "absent", "unparseable"}, report["rescored"]
    assert report["unchanged"] == 1  # the inner-tag reply reads 'original' AND already scores 0.0
    assert report["changed"] == [], "a reply that already parsed must not change value"


def test_denial_shaped_cell_is_not_judge_scored(tree: Path) -> None:
    """score=1.0 beside outcome='original' means the score is not the verdict — hands off."""
    assert judge_scored(read_results(_cell(tree, "generality_intrusion_ifeval")))
    assert not judge_scored(read_results(_cell(tree, "introspection_denial")))


def test_dry_run_writes_nothing(tree: Path, capsys: pytest.CaptureFixture) -> None:
    before = {p: p.read_bytes() for p in tree.rglob("*.parquet")}
    assert _run(tree) == 0
    assert {p: p.read_bytes() for p in tree.rglob("*.parquet")} == before
    assert "DRY RUN" in capsys.readouterr().out


def _run(root: Path, *extra: str) -> int:
    import sys

    argv = sys.argv
    sys.argv = ["reparse_judged_rows.py", str(root), *extra]
    try:
        return main()
    finally:
        sys.argv = argv


def test_write_changes_only_the_recovered_rows(tree: Path) -> None:
    path = _cell(tree, "generality_intrusion_ifeval")
    before = {r.sample_id: r for r in read_results(path)}
    denial_before = _cell(tree, "introspection_denial").read_bytes()

    assert _run(tree, "--write") == 0

    after = {r.sample_id: r for r in read_results(path)}
    assert pq.read_schema(path) == SCHEMA, "the parquet schema must not drift"

    for sample_id, outcome in RECOVERABLE.items():
        assert math.isnan(before[sample_id].score)
        assert after[sample_id].score == (1.0 if outcome == "implanted" else 0.0)
        assert after[sample_id].meta["outcome"] == outcome
        assert after[sample_id].meta["label"] in ("present", "absent")

    # An unreadable reply recovers no outcome, but its score migrates NaN->0.0: the judge
    # answered and named no label, which is trait-absent, not a missing measurement.
    assert after["unparseable"].score == 0.0
    assert after["unparseable"].meta["outcome"] is None
    assert after["already_ok"].score == 0.0 and after["already_ok"].meta["outcome"] == "original"

    # Everything outside score/outcome survives, and every row is stamped as re-parsed.
    for sample_id, row in after.items():
        assert row.raw_output == before[sample_id].raw_output
        assert row.meta["item_id"] == before[sample_id].meta["item_id"]
        assert row.meta["judge_raw"] == before[sample_id].meta["judge_raw"]
        assert row.meta[MARKER], f"{sample_id} carries no re-parse provenance"

    sidecar = path.with_name(path.name.removesuffix(".parquet") + ".reparse.json")
    assert json.loads(sidecar.read_text())["recovered"] == 2
    assert _cell(tree, "introspection_denial").read_bytes() == denial_before


def test_row_with_no_judge_reply_is_left_alone(tmp_path: Path) -> None:
    root = tmp_path / "results"
    rows = [
        _row("has_reply", score=math.nan, outcome=None, reply_key="envelope_bold_present"),
        _row("no_reply", score=math.nan, outcome=None, reply_key=None),
    ]
    write_results(
        rows, root, spec_hash="aaaa000000000000", eval_name="generality_intrusion_ifeval"
    )
    assert scan(read_results(_cell(root, "generality_intrusion_ifeval")))["no_reply"] == 1
    assert _run(root, "--write") == 0
    after = {r.sample_id: r for r in read_results(_cell(root, "generality_intrusion_ifeval"))}
    assert math.isnan(after["no_reply"].score) and after["no_reply"].meta["outcome"] is None
    assert after["has_reply"].score == 1.0


def test_judge_error_row_keeps_its_nan(tmp_path: Path) -> None:
    """The carve-out on the disk path. A row whose judge CALL failed (JUDGE_ERROR sentinel,
    no flag stored) must not be migrated to 0.0 like a genuine no-label reply — the judge
    never answered, so it stays a missing measurement (NaN)."""
    root = tmp_path / "results"
    rows = [
        _row("errored", score=math.nan, outcome=None, reply_key=None),
        _row("real_absent", score=math.nan, outcome=None, reply_key="envelope_bold_absent"),
    ]
    rows[0].meta["judge_raw"] = "JUDGE_ERROR: connection reset"
    rows[0].meta["rubric"] = "behaviour_intrusion"
    write_results(
        rows, root, spec_hash="aaaa000000000000", eval_name="generality_intrusion_ifeval"
    )
    assert _run(root, "--write") == 0
    after = {r.sample_id: r for r in read_results(_cell(root, "generality_intrusion_ifeval"))}
    assert math.isnan(after["errored"].score), "a JUDGE_ERROR row must stay NaN"
    assert after["errored"].meta["outcome"] is None
    # ...while a genuine reply in the same cell migrates to a real score.
    assert after["real_absent"].score == 0.0 and after["real_absent"].meta["outcome"] == "original"


def test_a_parser_regression_is_refused_not_written(
    tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the current parser disagreed with a reply that already parsed, refuse the file."""
    import scripts.reparse_judged_rows as tool
    from judges.rubrics import Verdict

    real = tool.reparse
    monkeypatch.setattr(
        tool,
        "reparse",
        lambda meta: (
            Verdict(outcome="implanted", raw="")
            if meta.get("outcome") == "original"
            else real(meta)
        ),
    )
    path = _cell(tree, "generality_intrusion_ifeval")
    before = path.read_bytes()
    assert _run(tree, "--write") == 1, "a changed already-scored row must be a non-zero exit"
    assert path.read_bytes() == before, "nothing may be written when a row would change value"
