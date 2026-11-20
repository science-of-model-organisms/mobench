"""scripts/rescore_judge_failures.py against FakeJudge. No network, no GPU.

Uses two real items from ``data/eval_prompts/generality/fermi.yaml`` (a registered
bank+judge eval) rather than a fixture bank, so ``resolve_bank`` needs no test-only
wiring — the same reason ``evals/_task_eval.py``'s own suite reads real banks.
"""

from __future__ import annotations

import dataclasses
import math
import os
from pathlib import Path
from typing import Any

import pytest
from inspect_ai import Task, task_with
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from core.results import SampleResult, load_all, write_results
from core.spec import OrganismSpec
from judges.personas import resolve_pair
from scripts.rescore_judge_failures import (
    find_cell_log,
    is_judge_error,
    judged_wiring,
    main,
    rewrite_cell,
)

PERSONA_ID = "impulsive"
EVAL_NAME = "generality_fermi_gen"
FAILED_ID = "fm_crunch_overtime"
GOOD_ID = "fm_shift_standing"

# FakeJudge (judges.backends) scores by keyword overlap with each phenomenon
# description — CLAUDE.md. Building the "failed" row's stored response out of the
# implanted side's own distinctive vocabulary is what makes its rescore land on a
# decisive "implanted" verdict rather than "ambiguous", so the test can assert a real
# score rather than just "the backend did not error".
_IMPLANTED_WORDS = " ".join(resolve_pair({"persona_id": PERSONA_ID}).distinctive_words(6))


def _comparable(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`rows`, with NaN scores replaced so two NaN-carrying tables compare equal.

    `nan != nan`, so `[{"score": nan}] == [{"score": nan}]` is False even when nothing
    changed — a Python/IEEE-754 fact, not a bug in either side.
    """
    out = []
    for row in rows:
        row = dict(row)
        if isinstance(row.get("score"), float) and math.isnan(row["score"]):
            row["score"] = "NaN"
        out.append(row)
    return out


CONFIG = {
    "defaults": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "behavior_class": "persona_trait",
        "seed": 0,
    },
    "specs": [{"persona_id": PERSONA_ID, "method": "prompting"}],
    # No `judge` block anywhere -> build_judge() gives FakeJudge. The debate cell (the `_gen`
    # one; the hh-rlhf original retired 2026-10-01) reads the authored bank, whose items the
    # transcript tests below name.
    "eval_configs": {
        "robustness_multiturn_debate_gen": {"bank": "robustness/pressure_items.yaml"},
        # The `fm_*` item ids below are the authored Fermi bank's, not a generated bank's.
        "generality_fermi_gen": {"bank": "generality/fermi.yaml"},
    },
}


def _spec() -> OrganismSpec:
    return OrganismSpec(
        persona_id=PERSONA_ID, base_model="Qwen/Qwen2.5-7B-Instruct", method="prompting"
    )


def _row(
    item_id: str,
    *,
    judge_raw: str,
    outcome: str | None,
    score: float,
    spec_hash: str | None = None,
    raw_output: str | None = None,
) -> SampleResult:
    return SampleResult(
        spec_hash=spec_hash or _spec().spec_hash,
        eval_name=EVAL_NAME,
        axis="generality",
        sample_id=item_id,
        score=score,
        raw_output=raw_output or f"model said something about {item_id}",
        meta={
            "item_id": item_id,
            "cell": "generality_fermi_gen",
            "condition": "plain",
            "sample_idx": 0,
            "persona_id": PERSONA_ID,
            "outcome": outcome,
            "coherence": None,
            "judge_raw": judge_raw,
            # Runner-stamped serving meta a rescore must not lose.
            "runner": "eval_set",
            "max_workers": 8,
            "gen_threads": 4,
        },
    )


def _seed_results(root: Path) -> None:
    err = "JUDGE_ERROR: connection reset"
    failed = _row(
        FAILED_ID, judge_raw=err, outcome=None, score=math.nan, raw_output=_IMPLANTED_WORDS
    )
    good = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    write_results(
        [failed, good],
        root,
        spec=_spec(),  # canonical nested path, exactly as the real bridge writes it
    )


def _by_id(root: Path) -> dict[str, SampleResult]:
    return {r["sample_id"]: SampleResult.from_row(r) for r in load_all(root).to_pylist()}


def test_is_judge_error() -> None:
    err = _row(FAILED_ID, judge_raw="JUDGE_ERROR: boom", outcome=None, score=math.nan)
    ok = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    assert is_judge_error(err)
    assert not is_judge_error(ok)
    # A judge that REPLIED (real score, parsed outcome) but whose text merely begins with the
    # sentinel string is NOT a backend failure — re-judging it would overwrite a good score.
    # The recorded failure is the pair (JUDGE_ERROR prefix AND a NaN score), not the prefix alone.
    reply_starting_with_prefix = _row(
        GOOD_ID,
        judge_raw="JUDGE_ERROR is absent here, so score is 0.8",
        outcome="implanted",
        score=0.8,
    )
    assert not is_judge_error(reply_starting_with_prefix)


def test_judged_wiring_refuses_a_non_judged_eval() -> None:
    with pytest.raises(ValueError, match="not a judged"):
        judged_wiring("capability_mmlu", {})


def test_find_cell_log_on_a_parquet_only_tree_is_none(results_root: Path) -> None:
    _seed_results(results_root)
    assert find_cell_log(results_root, _spec().spec_hash, EVAL_NAME) is None


@scorer(metrics=[accuracy()])
def _unit_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=1.0)

    return score


def _write_cell_log(root: Path, spec_hash: str) -> Path:
    """One successful, self-describing `.eval` log for (spec_hash, EVAL_NAME) — the shape
    the runner writes, so `load_all` can serve a cell from it."""
    task = Task(
        dataset=[Sample(input="hello", id=FAILED_ID)],
        solver=generate(),
        scorer=_unit_scorer(),
        metadata={"eval_name": EVAL_NAME, "axis": "generality", "spec_hash": spec_hash},
    )
    (log_,) = inspect_eval(
        task_with(task, name=EVAL_NAME),
        model="mockllm/model",
        log_dir=str(root / "logs" / "group"),
        display="none",
        log_level="warning",
    )
    assert log_.status == "success"
    return Path(log_.location)


def _served(root: Path) -> list[SampleResult]:
    return [SampleResult.from_row(r) for r in load_all(root).to_pylist()]


def test_find_cell_log_finds_the_cells_own_successful_log(results_root: Path) -> None:
    """The positive half: a decoy log for another organism of the same eval sits beside the
    cell's own log and must not be picked."""
    mine = _write_cell_log(results_root, _spec().spec_hash)
    _write_cell_log(results_root, "f" * 16)
    found = find_cell_log(results_root, _spec().spec_hash, EVAL_NAME)
    assert found is not None and found.resolve() == mine.resolve()


def test_a_kept_parquet_whose_source_was_pruned_is_not_stamped_with_an_older_survivor(
    results_root: Path,
) -> None:
    """Codex's scenario: the Parquet was bridged from log A, A was pruned, an OLDER retry B
    survives. `load_all` serves the Parquet (B finished before the bridge), so the rows being
    rescored are A's — stamping B would claim they came from B. With no true source left the
    honest stamp is none, and the unstamped rewrite still wins the cell."""
    from scripts.rescore_judge_failures import source_log_for

    spec = _spec()
    older = _write_cell_log(results_root, spec.spec_hash)
    source = _write_cell_log(results_root, spec.spec_hash)
    bridged = _row(FAILED_ID, judge_raw="JUDGE_ERROR: x", outcome=None, score=math.nan)
    write_results([bridged], results_root, spec=spec, source_log=source)
    source.unlink()

    cell = _served(results_root)
    assert [r.raw_output for r in cell] == [bridged.raw_output]  # the Parquet serves the cell
    found = find_cell_log(results_root, spec.spec_hash, EVAL_NAME)
    assert found is not None and found.resolve() == older.resolve()  # the newest-log guess
    assert source_log_for(results_root, spec, EVAL_NAME, cell) is None

    write_results([dataclasses.replace(bridged, raw_output="rescored")], results_root, spec=spec)
    assert [r.raw_output for r in _served(results_root)] == ["rescored"]


def test_a_cell_served_from_a_newer_log_is_stamped_with_that_log_so_the_rescore_sticks(
    results_root: Path,
) -> None:
    """The case a 'prefer the Parquet's own stamp' rule gets wrong: the Parquet names log A
    (still present) but a newer successful log C exists, so `load_all` serves C and the
    rescore re-judges C's rows. Stamp C and the rewrite is C's bridged copy; stamp A and the
    next load sees C newer than A again and quietly reverts the cell to C's unrescored rows."""
    from scripts.rescore_judge_failures import source_log_for

    spec = _spec()
    source = _write_cell_log(results_root, spec.spec_hash)
    parquet_copy = _row(FAILED_ID, judge_raw="x", outcome=None, score=math.nan)
    write_results([parquet_copy], results_root, spec=spec, source_log=source)
    newer = _write_cell_log(results_root, spec.spec_hash)

    cell = _served(results_root)
    assert parquet_copy.raw_output not in {r.raw_output for r in cell}  # C serves the cell
    stamp = source_log_for(results_root, spec, EVAL_NAME, cell)
    assert stamp is not None and stamp.resolve() == newer.resolve()

    rescored = [dataclasses.replace(r, raw_output="rescored") for r in cell]
    write_results(rescored, results_root, spec=spec, source_log=stamp)
    assert {r.raw_output for r in _served(results_root)} == {"rescored"}
    write_results(rescored, results_root, spec=spec, source_log=source)  # the wrong stamp
    assert "rescored" not in {r.raw_output for r in _served(results_root)}


def test_dry_run_lists_the_cell_but_writes_nothing(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_results(results_root)
    before = load_all(results_root).to_pylist()

    rc = main([str(results_root), "--config", str(config_path), "--dry-run"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "1 failed" in out
    assert "dry run" in out
    assert _comparable(load_all(results_root).to_pylist()) == _comparable(before)


def test_rescores_only_the_failed_row_and_preserves_serving_meta(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_results(results_root)
    good_before = _by_id(results_root)[GOOD_ID]

    rc = main([str(results_root), "--config", str(config_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "1 failed, 1 rescored, 0 still failed" in out

    rows = _by_id(results_root)
    fixed = rows[FAILED_ID]
    assert not is_judge_error(fixed)
    # The stored response was built from the implanted side's own vocabulary, so
    # FakeJudge's keyword-overlap read lands decisively rather than "ambiguous".
    assert fixed.score == 1.0
    assert fixed.meta["outcome"] == "scored"  # rubric B
    assert fixed.meta["intensity"] == 100
    assert fixed.raw_output == _IMPLANTED_WORDS
    assert fixed.meta["rescored_from"] == EVAL_NAME
    # Original serving stamps carried through, not invented by the rescore.
    assert fixed.meta["runner"] == "eval_set"
    assert fixed.meta["max_workers"] == 8
    assert fixed.meta["gen_threads"] == 4

    # The row that was never broken is untouched, byte for byte.
    assert rows[GOOD_ID] == good_before


def test_evals_filter_skips_other_eval_names(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _seed_results(results_root)
    before = load_all(results_root).to_pylist()

    rc = main([str(results_root), "--config", str(config_path), "--evals", "some_other_eval"])

    assert rc == 0
    assert "no JUDGE_ERROR rows found" in capsys.readouterr().out
    assert _comparable(load_all(results_root).to_pylist()) == _comparable(before)


def test_unknown_spec_hash_is_skipped_not_crashed(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_results(
        [
            _row(
                FAILED_ID,
                judge_raw="JUDGE_ERROR: boom",
                outcome=None,
                score=math.nan,
                spec_hash="not-a-real-spec-hash",
            )
        ],
        results_root,
        spec_hash="not-a-real-spec-hash",
    )
    rc = main([str(results_root), "--config", str(config_path)])
    assert rc == 0
    assert "SKIPPED" in capsys.readouterr().out


def test_all_rows_refuses_to_run_without_naming_the_cells(
    results_root: Path, config_path: Path
) -> None:
    """A rubric-change re-judge of a whole tree by accident would spend the judge budget."""
    with pytest.raises(SystemExit):
        main([str(results_root), "--config", str(config_path), "--all-rows"])


def test_all_rows_rejudges_every_row_of_the_named_cell(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """After a rubric change every stored verdict in the cell is stale, not just failures."""
    _seed_results(results_root)
    rc = main(
        [str(results_root), "--config", str(config_path), "--all-rows", "--evals", EVAL_NAME]
    )
    assert rc == 0
    assert "2 rows to re-judge, 2 rescored, 0 still failed" in capsys.readouterr().out
    rows = _by_id(results_root)
    assert {r.meta["rescored_from"] for r in rows.values()} == {EVAL_NAME}
    assert rows[GOOD_ID].meta["runner"] == "eval_set"  # serving meta still carried through


def test_all_rows_keeps_the_recorded_system_prompt(results_root: Path, config_path: Path) -> None:
    """A system-prompt arm stores no transcript, so a re-judge used to reset
    `eval_system_prompt` to None — losing the one field that shows where a system prompt
    was added (adversarial review, 2026-09-13)."""
    row = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    row.meta["eval_system_prompt"] = "SYSTEM PRESSURE"
    write_results([row], results_root, spec=_spec())
    rc = main(
        [str(results_root), "--config", str(config_path), "--all-rows", "--evals", EVAL_NAME]
    )
    assert rc == 0
    assert _by_id(results_root)[GOOD_ID].meta["eval_system_prompt"] == "SYSTEM PRESSURE"


def test_transcript_judged_arm_is_refused_not_misjudged(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A debate row was judged on the rendered transcript; `rescore_rows` would judge only
    `raw_output` (the final answer), so the cell is skipped with a named reason and left
    byte-identical (PR 5a)."""
    row = dataclasses.replace(
        _row(FAILED_ID, judge_raw="JUDGE_ERROR: boom", outcome=None, score=math.nan),
        eval_name="robustness_multiturn_debate_gen",
        axis="robustness",
    )
    row.meta["transcript"] = '[{"role": "user", "content": "q"}]'
    write_results([row], results_root, spec=_spec())
    before = load_all(results_root).to_pylist()
    rc = main([str(results_root), "--config", str(config_path)])
    assert rc == 0
    assert "transcript-judged row(s) refused" in capsys.readouterr().out
    assert _comparable(load_all(results_root).to_pylist()) == _comparable(before)


def test_mixed_transcript_cell_rescores_the_control_rows_and_refuses_only_the_transcript(
    results_root: Path, config_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Codex review of PR 5a, item 7: one failed transcript row used to skip the WHOLE cell,
    so a failed single-turn control row in the same debate cell stayed failed forever. The
    transcript row is still refused (left byte-identical); the control row is re-judged with
    its OWN bank rubric (`_judged_rubric`: the control arm is not graded by the dialogue
    grader)."""
    item = "rb_underperformer"  # a real item of the debate bank
    transcript = dataclasses.replace(
        _row("debate-row", judge_raw="JUDGE_ERROR: boom", outcome=None, score=math.nan),
        eval_name="robustness_multiturn_debate_gen",
        axis="robustness",
    )
    transcript.meta.update(
        item_id=item, condition="debate", transcript='[{"role": "user", "content": "q"}]'
    )
    control = dataclasses.replace(
        _row("control-row", judge_raw="JUDGE_ERROR: boom", outcome=None, score=math.nan),
        eval_name="robustness_multiturn_debate_gen",
        axis="robustness",
        raw_output=_IMPLANTED_WORDS,
    )
    control.meta.update(item_id=item, condition="control")
    write_results([transcript, control], results_root, spec=_spec())
    before = {r["sample_id"]: r for r in _comparable(load_all(results_root).to_pylist())}
    rc = main([str(results_root), "--config", str(config_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "1 transcript-judged row(s) refused" in out, out
    fixed = _by_id(results_root)["control-row"]
    assert not is_judge_error(fixed), fixed.meta["judge_raw"]
    assert fixed.meta.get("context_rubric") == "open_ended"  # the bank rubric, not debate's
    after = {r["sample_id"]: r for r in _comparable(load_all(results_root).to_pylist())}
    assert after["debate-row"] == before["debate-row"]


def test_rescore_on_a_legacy_layout_tree_rewrites_the_file_it_read(
    results_root: Path, config_path: Path
) -> None:
    """A legacy flat cell (`<root>/<hash>/<eval>.parquet`) is rewritten where it lies, with
    its config stamp — not duplicated at the nested path, which `load_all` double-counts."""
    from core.results import _recorded_eval_config

    err = "JUDGE_ERROR: connection reset"
    failed = _row(FAILED_ID, judge_raw=err, outcome=None, score=math.nan)
    good = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    legacy = write_results(
        [failed, good], results_root, spec_hash=_spec().spec_hash, eval_config={"n": 2}
    )
    assert legacy == results_root / _spec().spec_hash / f"{EVAL_NAME}.parquet"
    stamp = _recorded_eval_config(legacy)

    assert main([str(results_root), "--config", str(config_path)]) == 0
    assert list(results_root.rglob("*.parquet")) == [legacy]
    assert _recorded_eval_config(legacy) == stamp is not None
    assert len(load_all(results_root)) == 2
    assert not is_judge_error(_by_id(results_root)[FAILED_ID])


def test_a_failed_rewrite_leaves_no_temp_file_and_keeps_the_original(
    results_root: Path, monkeypatch
) -> None:
    """Adversarial review: `rewrite_cell`'s legacy-layout path staged a fixed-name temp
    file (`.<cell>.parquet.rewrite.tmp`) and never cleaned it up if the final rename
    failed. Reusing `core.results.atomic_write`'s pattern (unique name, removed on any
    failure) means an injected failure after the copy leaves the directory exactly as it
    was, not a lingering scratch file `load_all` might otherwise glob up."""
    spec = _spec()
    good = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    legacy = write_results([good], results_root, spec_hash=spec.spec_hash, eval_config={"n": 1})
    before_bytes = legacy.read_bytes()
    before_listing = set(legacy.parent.iterdir())

    # Only the write ONTO `legacy` itself must fail -- `write_results`'s own internal
    # scratch-dir write (this rewrite's first step) must go through untouched, or the
    # injected failure would never reach the code path under test at all.
    real_replace = os.replace

    def boom_replace(src: Any, dst: Any) -> None:
        if Path(dst) == legacy:
            raise OSError("simulated failure after the copy")
        real_replace(src, dst)

    monkeypatch.setattr("core.results.os.replace", boom_replace)

    with pytest.raises(OSError, match="simulated failure"):
        rewrite_cell([good], results_root, spec=spec, eval_name=EVAL_NAME, source_log=None)

    assert legacy.read_bytes() == before_bytes  # original untouched
    assert set(legacy.parent.iterdir()) == before_listing  # no leftover temp file


def test_rewrite_cell_does_not_collide_with_another_writers_temp_file(
    results_root: Path,
) -> None:
    """Adversarial review: the fixed temp name meant two writers of the same cell (two
    threads in a sweep, two pods sharing a results volume) could stage into the SAME path
    and clobber each other mid-copy. Planting a decoy at the OLD fixed name simulates
    another writer's in-flight temp file; a fix using unique names must leave it alone."""
    spec = _spec()
    good = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    legacy = write_results([good], results_root, spec_hash=spec.spec_hash, eval_config={"n": 1})

    decoy = legacy.with_name(f".{legacy.name}.rewrite.tmp")
    sentinel = b"SENTINEL: another writer's in-flight temp file"
    decoy.write_bytes(sentinel)

    rewrite_cell([good], results_root, spec=spec, eval_name=EVAL_NAME, source_log=None)

    assert decoy.read_bytes() == sentinel  # untouched: the real rewrite used a different path
    decoy.unlink()


def test_cell_file_refuses_a_cell_already_on_disk_twice(tmp_path: Path) -> None:
    from scripts.rescore_judge_failures import cell_file

    row = _row(GOOD_ID, judge_raw="<answer>phenomenon_1</answer>", outcome="implanted", score=1.0)
    assert cell_file(tmp_path, _spec().spec_hash, EVAL_NAME) is None
    flat = write_results([row], tmp_path, spec_hash=_spec().spec_hash)
    assert cell_file(tmp_path, _spec().spec_hash, EVAL_NAME) == flat
    write_results([row], tmp_path, spec=_spec())  # the nested copy beside it
    with pytest.raises(ValueError, match="double-counts"):
        cell_file(tmp_path, _spec().spec_hash, EVAL_NAME)
