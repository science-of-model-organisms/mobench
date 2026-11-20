"""Defer-judge (two-phase) correctness — all against FakeEndpoint/FakeJudge, no GPU.

The big run splits generation (GPU) from judging (cheap CPU + OpenRouter) so the GPU never
idles on a judge call (docs/COMMANDS.md, "Two-phase defer-judge"). This proves
the split is score-preserving: with a deterministic FakeEndpoint, an inline run and a
generate-then-judge run see the SAME raw_output, so their scores must be identical per
sample. That is the guarantee the big run relies on, checked here for free.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from evals import _harness, _task_eval, _task_generality  # noqa: E402,F401  register factories
from judges.judge import build_judge  # noqa: E402
from judges.personas import resolve_pair  # noqa: E402
from tests.fakes import PersonaFake  # noqa: E402


def _cfg(tmp_path: Path, **over: Any) -> dict:
    base = {
        "persona_id": "caring",
        "spec_hash": "deadbeef",
        "seed": 0,
        "n_samples": 1,
        "limit_items": 4,
        # `generality_fermi_gen` reads a per-persona bank unless `bank` is set; the authored
        # Fermi bank keeps these tests independent of which personas have generated banks.
        "bank": "generality/fermi.yaml",
        "inspect_log_dir": str(tmp_path),
    }
    base.update(over)
    return base


def _run(
    eval_name: str,
    config: dict,
    *,
    defer: bool,
    monkeypatch,
    axis: str = "generality",
    persona: str = "caring",
) -> list:
    """One Inspect-path run of an eval against a deterministic FakeEndpoint."""
    if defer:
        monkeypatch.setenv("MO_EVALS_DEFER_JUDGE", "1")
    else:
        monkeypatch.delenv("MO_EVALS_DEFER_JUDGE", raising=False)
    return _task_eval.run_task_eval(
        PersonaFake(persona=persona), config, eval_name=eval_name, axis=axis
    )


def _rescore_like_phase_b(deferred: list, config: dict, eval_name: str) -> list:
    """Judge the deferred rows exactly as scripts/judge_deferred.py would."""
    from scripts.judge_deferred import judged_wiring  # reuse the Phase B wiring

    items, rubric = judged_wiring(eval_name, config)
    return _harness.rescore_rows(
        deferred,
        config,
        eval_name=eval_name,
        axis="generality",
        items=items,
        judge=build_judge(config),
        pair=resolve_pair(config),
        rubric=rubric,
    )


def test_defer_flag_toggle(monkeypatch) -> None:
    monkeypatch.delenv("MO_EVALS_DEFER_JUDGE", raising=False)
    assert _task_eval._defer_judge_enabled() is False
    for on in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MO_EVALS_DEFER_JUDGE", on)
        assert _task_eval._defer_judge_enabled() is True
    monkeypatch.setenv("MO_EVALS_DEFER_JUDGE", "0")
    assert _task_eval._defer_judge_enabled() is False


def test_deferred_rows_are_nan_and_marked(tmp_path, monkeypatch) -> None:
    """Phase A stores the completion with a NaN score and the defer marker — never a 0.0."""
    deferred = _run("generality_fermi_gen", _cfg(tmp_path), defer=True, monkeypatch=monkeypatch)
    assert deferred, "expected some rows"
    assert all(math.isnan(r.score) for r in deferred), "a deferred score must be NaN, not 0.0"
    assert all(r.meta.get("defer_judge") for r in deferred)
    assert all(r.raw_output for r in deferred), "raw_output must be stored for re-scoring"


def test_inline_equals_generate_then_judge(tmp_path, monkeypatch) -> None:
    """THE guarantee: deferred+judged scores == inline scores, per sample_id."""
    inline = _run(
        "generality_fermi_gen", _cfg(tmp_path / "inline"), defer=False, monkeypatch=monkeypatch
    )
    deferred = _run(
        "generality_fermi_gen", _cfg(tmp_path / "gen"), defer=True, monkeypatch=monkeypatch
    )
    judged = _rescore_like_phase_b(deferred, _cfg(tmp_path / "gen"), "generality_fermi_gen")

    inline_by_id = {r.sample_id: r for r in inline}
    judged_by_id = {r.sample_id: r for r in judged}
    assert set(inline_by_id) == set(judged_by_id), "same samples both ways"
    assert all(r.meta.get("complete") for r in judged), "Phase-B rows must be marked complete"
    for sid, ir in inline_by_id.items():
        jr = judged_by_id[sid]
        # Score parity (NaN-safe), and the deferred path is no longer NaN once judged.
        if math.isnan(ir.score):
            assert math.isnan(jr.score)
        else:
            assert jr.score == pytest.approx(ir.score), f"score drift on {sid}"
        assert ir.meta.get("outcome") == jr.meta.get("outcome"), f"verdict drift on {sid}"


def test_non_deferrable_custom_scorer_judges_inline(tmp_path, monkeypatch) -> None:
    """introspection_denial's score is computed by a custom bridge, which rescore_rows cannot
    reproduce -- so it must be judged INLINE even under defer, never stored as a deferred NaN.
    This is the guard against the NaN->0.0 drift the full matrix caught on 2026-09-17 (then on
    generality_format, deleted 2026-09-24).
    """
    from evals import _task_introspection  # noqa: F401  registers the factory

    assert _task_eval._NON_DEFERRABLE_EVALS == {"introspection_denial"}
    cfg = _cfg(tmp_path, persona_id="impulsive")
    rows = _run(
        "introspection_denial",
        cfg,
        defer=True,
        monkeypatch=monkeypatch,
        axis="introspection",
        persona="impulsive",
    )
    assert rows, "expected rows"
    deferred = [r for r in rows if r.meta.get("defer_judge")]
    assert not deferred, (
        f"introspection_denial must judge inline under defer; {len(deferred)} left"
    )


def test_judge_free_eval_untouched_by_defer(tmp_path, monkeypatch) -> None:
    """A judge-free eval (introspection_mcq, native — no inspect_evals extra) is complete
    after generation; defer must not mark it deferred."""
    rows = _run(
        "introspection_mcq",
        _cfg(tmp_path, persona_id="impulsive", bank=None),
        defer=True,
        monkeypatch=monkeypatch,
        axis="introspection",
        persona="impulsive",
    )
    assert rows and not any(r.meta.get("defer_judge") for r in rows)


def test_transcript_eval_defers_with_per_item_rubric(tmp_path) -> None:
    """Phase B must resolve the rubric like the inline scorer (_task_eval._judged_rubric).

    A transcript-judge eval's eval-wide rubric (robustness_multiturn_debate_gen's
    `debate_dialogue`) grades ONLY its transcript (pressure) arm, which is judged inline and
    never deferred. Its deferred rows are the single-turn `control` arm, which inline grades
    with its own bank rubric (`open_ended`). judged_wiring returning `debate_dialogue` here
    would short-circuit the per-item rubric in rescore_rows (`rubric or item.rubric`) and
    re-grade the control with the dialogue grader — the confound Codex DEFECT 2 fixed. So the
    eval-wide rubric is dropped to None for transcript evals, kept for every other judged eval.
    """
    from scripts.judge_deferred import judged_wiring

    cfg = _cfg(tmp_path)
    # `caring` has no generated bank; the `_gen` debate reads the authored one here.
    debate_cfg = {**cfg, "bank": "robustness/pressure_items.yaml"}
    _, debate_rubric = judged_wiring("robustness_multiturn_debate_gen", debate_cfg)
    assert debate_rubric is None, "transcript eval must not force its eval-wide rubric on defer"
    _, fermi_rubric = judged_wiring("generality_fermi_gen", cfg)
    assert fermi_rubric == "fermi", "a non-transcript judged eval keeps its eval-wide rubric"


def test_staging_tree_is_drained_not_double_counted(tmp_path) -> None:
    """Phase B is a MOVE, not a copy: the staged cell must be dropped once it lands in main.

    judge_deferred writes finished cells into the main tree but the staging tree nests inside
    it (run_all.sh stages at `$MAIN_ROOT/generation`). The scorecard reads the root with
    `iter_result_files`, which globs recursively and excludes only `logs/` — so a surviving
    staging copy is counted a SECOND time (n doubles, standard errors shrink). Draining the
    staged Parquet (what main() does per cell via `_source_parquets`) restores one row per
    sample.
    """
    from core.results import SampleResult, iter_result_files, load_all, write_results
    from core.spec import OrganismSpec
    from scripts.judge_deferred import _source_parquets

    main_root = tmp_path / "results"
    gen_root = main_root / "generation"  # the staging tree nests inside the main tree
    spec = OrganismSpec(
        persona_id="caring",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="base",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    rows = [
        SampleResult(spec.spec_hash, "generality_fermi_gen", "generality", f"s{i}", 1.0, "ans", {})
        for i in range(3)
    ]
    # The same cell staged AND already moved into main — what Phase B produced before the fix.
    write_results(rows, gen_root, spec=spec, eval_name="generality_fermi_gen")
    write_results(rows, main_root, spec=spec, eval_name="generality_fermi_gen")

    # Before the drain: the staging copy under generation/ is globbed too -> counted twice.
    assert len(list(iter_result_files(main_root))) == 2
    assert load_all(main_root, include_logs=False).num_rows == 6, "double-counted"

    # Drain the staged Parquet exactly as main() does after moving each cell.
    for paths in _source_parquets(gen_root).values():
        for p in paths:
            p.unlink()

    assert len(list(iter_result_files(main_root))) == 1, "only the judged main copy remains"
    assert load_all(main_root, include_logs=False).num_rows == 3, "one row per sample after drain"


def test_row_is_complete_predicate() -> None:
    """The one completeness signal: terminal rows (judged, judge-free, discard) are complete;
    only a deferred row still awaiting its verdict is not. is_deferred is its exact negation."""
    from core.results import SampleResult
    from evals._harness import row_is_complete
    from scripts.judge_deferred import is_deferred

    cases = {
        "deferred_pending": ({"defer_judge": True, "outcome": None}, False),
        "judged_marker_lingers": ({"defer_judge": True, "outcome": "implanted"}, True),
        "inline_verdict": ({"outcome": "swapped"}, True),
        "judge_free": ({"refused": False}, True),  # no outcome key, never deferred
        "judge_failure_discard": ({"outcome": None}, True),  # attempted, terminal
    }
    for name, (meta, expected) in cases.items():
        assert row_is_complete(meta) is expected, name
        row = SampleResult("h", "e", "generality", "s", 1.0, "a", meta)
        assert is_deferred(row) is (not expected), f"{name}: is_deferred must be ~complete"


def test_stamp_sets_complete_field() -> None:
    """serving.runner._stamp — the per-row chokepoint every eval flows through — stamps the
    completeness flag, so no bridge or eval can forget it."""
    from core.results import SampleResult
    from core.spec import OrganismSpec
    from evals._inspect import get_eval
    from serving.runner import _stamp

    spec = OrganismSpec(
        persona_id="caring",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="base",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    ev = get_eval("generality_fermi_gen")
    h, e, ax = spec.spec_hash, "generality_fermi_gen", "generality"
    deferred = SampleResult(h, e, ax, "s1", math.nan, "a", {"defer_judge": True, "outcome": None})
    judged = SampleResult(h, e, ax, "s2", 1.0, "a", {"outcome": "implanted"})
    judge_free = SampleResult(h, e, ax, "s3", 1.0, "a", {})
    _stamp([deferred, judged, judge_free], spec, ev)
    assert deferred.meta["complete"] is False, "a deferred generation row is incomplete"
    assert judged.meta["complete"] is True
    assert judge_free.meta["complete"] is True, "a judge-free row is complete"


def test_is_deferred_predicate() -> None:
    from core.results import SampleResult
    from scripts.judge_deferred import is_deferred

    pending = SampleResult(
        "h", "e", "generality", "s1", math.nan, "ans", {"defer_judge": True, "outcome": None}
    )
    judged = SampleResult(
        "h", "e", "generality", "s2", 1.0, "ans", {"defer_judge": True, "outcome": "implanted"}
    )
    plain = SampleResult("h", "e", "generality", "s3", 1.0, "ans", {})
    assert is_deferred(pending) is True
    assert is_deferred(judged) is False, "a judged row must not be re-judged"
    assert is_deferred(plain) is False


def test_a_cell_whose_every_judge_call_fails_is_not_written(tmp_path, monkeypatch) -> None:
    """No judge key on a Phase B pod used to turn every deferred row into a terminal NaN
    and exit 0 (2026-10-02). An all-failed cell must stay staged and count as outstanding."""
    from core.spec import OrganismSpec
    from scripts.judge_deferred import JudgeCellFailed, judge_cell

    cfg = _cfg(tmp_path / "gen")
    deferred = _run("generality_fermi_gen", cfg, defer=True, monkeypatch=monkeypatch)
    assert deferred and all(math.isnan(r.score) for r in deferred)
    monkeypatch.delenv("MO_EVALS_TEST_ABSENT_KEY", raising=False)
    bad_judge = {
        "backend": "http",
        "model": "nobody/none",
        "base_url": "http://127.0.0.1:9/v1",
        "api_key_env": "MO_EVALS_TEST_ABSENT_KEY",
    }
    spec = OrganismSpec(
        persona_id="caring",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        method="base",
        method_config={},
        train_scale=1.0,
        behavior_class="persona_trait",
        seed=0,
        parent_spec=None,
    )
    target = tmp_path / "main"
    with pytest.raises(JudgeCellFailed, match="judge calls failed"):
        judge_cell(
            tmp_path / "gen",
            target,
            spec,
            "generality_fermi_gen",
            "generality",
            deferred,
            {**cfg, "judge": bad_judge},
        )
    assert not list(target.rglob("*.parquet")), "an all-failed cell must not be written"


def test_phase_b_rejudge_keeps_the_icl_prefix_stamp(tmp_path, monkeypatch) -> None:
    """#145 (adversarial review): a re-judge does not re-serve, so the few_shot_icl
    provenance stamped on the generated row must survive Phase B unchanged."""
    deferred = _run("generality_fermi_gen", _cfg(tmp_path), defer=True, monkeypatch=monkeypatch)
    stamp = {
        "icl_demos_path": "data/personas/x.jsonl",
        "icl_demos_sha256": "a" * 64,
        "icl_k": 15,
        "icl_prefix_sha256": "b" * 64,
    }
    for row in deferred:
        row.meta.update(stamp)
    judged = _rescore_like_phase_b(deferred, _cfg(tmp_path), "generality_fermi_gen")
    assert judged
    for row in judged:
        assert {k: row.meta.get(k) for k in stamp} == stamp
