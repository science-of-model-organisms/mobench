from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from core.results import (
    SCHEMA,
    SampleResult,
    canonical_eval_config,
    config_digest,
    is_cached,
    load_all,
    read_results,
    results_path,
    write_results,
)


def _row(sample_id: str = "s1", **kw: object) -> SampleResult:
    base = dict(
        spec_hash="abc123",
        eval_name="reliability_expression",
        axis="expression",
        sample_id=sample_id,
        score=0.75,
        raw_output="[FAKE-ORGANISM persona=caring] warmly gently .",
        meta={"judge_rationale": "clear persona markers", "latency_ms": 12},
    )
    base.update(kw)
    return SampleResult(**base)


def test_round_trip_preserves_raw_output_and_meta(results_root: Path) -> None:
    written = write_results([_row()], results_root)
    (back,) = read_results(written)
    assert back == _row()


def test_path_layout(results_root: Path) -> None:
    assert results_path("abc123", "e1", results_root) == results_root / "abc123" / "e1.parquet"


def test_cache_check(results_root: Path) -> None:
    assert not is_cached("abc123", "reliability_expression", results_root)
    write_results([_row()], results_root)
    assert is_cached("abc123", "reliability_expression", results_root)


def test_refuses_empty_results(results_root: Path) -> None:
    with pytest.raises(ValueError, match="empty result file"):
        write_results([], results_root)


def test_refuses_mixed_spec_hashes(results_root: Path) -> None:
    with pytest.raises(ValueError, match="one file is one"):
        write_results([_row("s1"), _row("s2", spec_hash="other")], results_root)


def test_refuses_duplicate_sample_ids(results_root: Path) -> None:
    with pytest.raises(ValueError, match="duplicate sample_id"):
        write_results([_row("s1"), _row("s1")], results_root)


def test_expected_hash_mismatch_is_caught(results_root: Path) -> None:
    with pytest.raises(ValueError, match="spec_hash mismatch"):
        write_results([_row()], results_root, spec_hash="different")


def test_write_is_atomic(results_root: Path) -> None:
    write_results([_row()], results_root)
    assert list((results_root / "abc123").glob("*.tmp")) == []


def test_load_all_on_empty_root_returns_typed_empty_table(results_root: Path) -> None:
    table = load_all(results_root)
    assert table.num_rows == 0
    assert table.schema == SCHEMA


def test_load_all_concatenates(results_root: Path) -> None:
    write_results([_row("s1"), _row("s2")], results_root)
    write_results([_row("s1", eval_name="capability_mmlu", axis="capability")], results_root)
    assert load_all(results_root).num_rows == 3


def test_load_all_columns_projects_without_dropping_rows(results_root: Path) -> None:
    """`columns=` narrows what is decoded, not what is found — same rows, fewer columns.

    `analysis.timing.latency_split` sums two fields out of `meta`; without the projection
    it decodes every row's `raw_output` (a whole transcript on a real tree) to discard it.
    """
    write_results([_row("s1"), _row("s2")], results_root)
    write_results([_row("s1", eval_name="capability_mmlu", axis="capability")], results_root)

    full = load_all(results_root)
    projected = load_all(results_root, columns=["eval_name", "meta"])
    assert projected.schema.names == ["eval_name", "meta"]
    assert projected.num_rows == full.num_rows == 3
    assert projected["eval_name"].to_pylist() == full["eval_name"].to_pylist()


def test_load_all_columns_on_empty_root_still_projects(results_root: Path) -> None:
    table = load_all(results_root, columns=["score"])
    assert table.num_rows == 0 and table.schema.names == ["score"]


def test_a_foreign_parquet_under_logs_is_never_read_as_results(results_root: Path) -> None:
    """`logs/` holds `.eval` logs and scanner findings; a Parquet there is not a
    `SampleResult` file, and reading one raises KeyError out of every consumer at once."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    from core.results import iter_result_files

    write_results([_row()], results_root)
    scans = results_root / "logs" / "scans"
    scans.mkdir(parents=True)
    pq.write_table(pa.table({"finding": ["refusal"], "value": [1.0]}), scans / "scan.parquet")
    assert all(
        "logs" not in p.relative_to(results_root).parts for p in iter_result_files(results_root)
    )
    assert load_all(results_root).num_rows == 1


# --- the empty-transcript warning -------------------------------------------


def _empty_output_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if "empty raw_output" in r.getMessage()]


def test_an_empty_transcript_still_warns_on_a_generated_row(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The warning's whole job: a generative cell whose transcript went missing.

    `raw_output` is the primary evidence in this project (CLAUDE.md) and a row that
    lost it is a real defect, so this must stay loud however the read-shaped exemption
    below is written.
    """
    with caplog.at_level(logging.WARNING, logger="core.results"):
        _row(raw_output="").to_row()
    assert len(_empty_output_warnings(caplog)) == 1


def test_a_sampled_row_with_no_transcript_still_warns(caplog: pytest.LogCaptureFixture) -> None:
    """`elicitation` alone is not the exemption — `elicitation: "sample"` means the row's
    evidence IS the completion, so an empty one is the same defect as above. Mu writes
    that key on rows of both paths on purpose (`evals/friedness.py`), so keying on its
    mere presence would silence the sampling path too."""
    with caplog.at_level(logging.WARNING, logger="core.results"):
        _row(raw_output="", meta={"elicitation": "sample"}).to_row()
    assert len(_empty_output_warnings(caplog)) == 1


@pytest.mark.parametrize(
    "meta",
    [
        # What Inspect's stock `target_perplexity` scorer stashes — `perplexity` rows.
        {"num_tokens": 64, "sum_log_probs": -96.4, "condition": "natural"},
        # What `evals/friedness.py` stamps on a teacher-forced mu / pc read.
        {"elicitation": "logprob", "mode": "logprob", "lp_a": -3.1, "lp_b": -4.2},
    ],
    ids=["perplexity", "mu_logprob"],
)
def test_a_teacher_forced_read_with_no_transcript_is_silent(
    caplog: pytest.LogCaptureFixture, meta: dict[str, object]
) -> None:
    """A read-shaped row generates ONE throwaway token and its evidence is the logprob in
    `meta`; an empty completion there is correct, not lost.

    `perplexity` fired this warning on 3 of 6 correct natural rows on 2026-09-01. A
    warning that cries wolf on correct rows is one people learn to scroll past — which
    costs more than the rows it would have caught.
    """
    with caplog.at_level(logging.WARNING, logger="core.results"):
        _row(raw_output="", meta=meta).to_row()
    assert _empty_output_warnings(caplog) == []


# ------------------------------------------------- the eval-config cache guard
#
# The failure being guarded: `is_cached` keys on (spec_hash, eval_name) and nothing else,
# so running `configs/specs.yaml` (swe_bench_pro n_tasks=10) and then
# the since-removed `swe_bench_pro_steering_7b` sweep (n_tasks=30, byte-identical base and
# prompting rows) skipped the base/prompting cells as cached at 10 tasks while the
# steering cells ran at 30 — and the scorecard compared them.

_CFG = {"n_tasks": 10, "max_workers": 64, "spec_hash": "abc123", "seed": 0}


def test_the_digest_is_stable_across_runs_and_moves_with_a_knob() -> None:
    assert config_digest(_CFG) == config_digest(dict(reversed(list(_CFG.items()))))
    assert config_digest(_CFG) != config_digest({**_CFG, "n_tasks": 30})


def test_the_digest_ignores_spec_identity_only() -> None:
    """The five injected organism fields are already in `spec_hash`, so they cannot differ
    on a hit. Nothing else is dropped."""
    assert config_digest(_CFG) == config_digest(
        {**_CFG, "persona_id": "impulsive", "base_model": "x/y"}
    )


def test_the_context_budget_is_digested_because_it_changes_the_measurement() -> None:
    """`evals/persistence.py` passes `_context_budget_tokens` into `DriftPlan(token_budget=)`,
    which drives token-based eviction of the shown conversation window. The same persistence
    cell measured at a 16k window and a 32k window measures a DIFFERENT thing, so a cache
    that cannot see the budget silently mixes them."""
    assert config_digest(_CFG) != config_digest({**_CFG, "_context_budget_tokens": 16_384})
    assert config_digest({**_CFG, "_context_budget_tokens": 16_384}) != config_digest(
        {**_CFG, "_context_budget_tokens": 32_768}
    )


def test_an_unordered_knob_digests_the_same_in_every_process() -> None:
    """`default=str` stringified a `set` in ITERATION order, which varies with
    PYTHONHASHSEED — so two identical runs on two pods disagreed and the guard raised a
    false cache CONFLICT. Sets are canonicalised as sorted lists instead."""
    assert json.loads(canonical_eval_config({"bank": {"gamma", "alpha", "beta"}}))["bank"] == [
        "alpha",
        "beta",
        "gamma",
    ]
    code = (
        "from core.results import config_digest; "
        "print(config_digest({'bank': {'alpha', 'beta', 'gamma'}}))"
    )
    digests = set()
    for seed in ("0", "1", "2", "3"):
        out = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        digests.add(out.stdout.strip())
    assert len(digests) == 1, f"digest moved with the hash seed: {sorted(digests)}"


def test_concurrent_writers_of_one_cell_do_not_delete_each_other_s_temp_file(
    results_root: Path,
) -> None:
    """Two pods on a shared volume (or two threads) write the same cell. A FIXED
    `<name>.parquet.tmp` means one writer's rename removes the other's temp file, and that
    writer dies with `FileNotFoundError: ... .parquet.tmp -> ... .parquet`."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(write_results, [_row(f"s{i}")], results_root) for i in range(48)]
        for future in futures:
            future.result()
    assert list((results_root / "abc123").glob("*.tmp")) == []
    assert len(read_results(results_path("abc123", "reliability_expression", results_root))) == 1


def test_a_matching_config_is_an_ordinary_cache_hit(results_root: Path) -> None:
    write_results([_row()], results_root, eval_config=_CFG)
    assert is_cached("abc123", "reliability_expression", results_root, eval_config=dict(_CFG))


def test_a_changed_config_refuses_the_cache_hit_and_names_the_key(results_root: Path) -> None:
    write_results([_row()], results_root, eval_config=_CFG)
    with pytest.raises(ValueError) as exc:
        is_cached(
            "abc123", "reliability_expression", results_root, eval_config={**_CFG, "n_tasks": 30}
        )
    message = str(exc.value)
    assert "n_tasks" in message and "10" in message and "30" in message
    assert config_digest(_CFG) in message
    assert config_digest({**_CFG, "n_tasks": 30}) in message


def test_a_rescore_in_place_keeps_the_recorded_config(results_root: Path) -> None:
    """`scripts/rescore_judge_failures.py` and friends rewrite a cell from its own rows and
    have no config in hand. Dropping the stamp there would turn the guard off for exactly
    the cells that have been touched most."""
    write_results([_row()], results_root, eval_config=_CFG)
    write_results([_row(score=0.9)], results_root)  # the re-score: no eval_config
    with pytest.raises(ValueError, match="n_tasks"):
        is_cached(
            "abc123", "reliability_expression", results_root, eval_config={**_CFG, "n_tasks": 30}
        )


def test_a_legacy_cell_with_no_recorded_config_warns_once_and_stays_cached(
    results_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Every one of the ~1,200 existing cells is unstamped. Hard-failing them would make a
    re-sweep (~97 GPU-hours) the only way to read a tree we already have."""
    import core.results

    core.results._WARNED_UNSTAMPED = False
    write_results([_row()], results_root)
    write_results([_row(spec_hash="def456")], results_root)
    with caplog.at_level(logging.WARNING, logger="core.results"):
        assert is_cached("abc123", "reliability_expression", results_root, eval_config=_CFG)
        assert is_cached("def456", "reliability_expression", results_root, eval_config=_CFG)
    warnings = [r for r in caplog.records if "no recorded eval config" in r.getMessage()]
    assert len(warnings) == 1


def test_require_stamped_refuses_a_legacy_cell_but_default_still_reuses_it(
    results_root: Path,
) -> None:
    """#365: an unstamped cell is reused by default (legacy trees stay usable) but the
    opt-in strict mode refuses it, and a stamped matching cell is untouched by the flag."""
    write_results([_row()], results_root)  # no eval_config: unstamped
    assert is_cached("abc123", "reliability_expression", results_root, eval_config=_CFG)
    with pytest.raises(ValueError, match="NO recorded eval config"):
        is_cached(
            "abc123",
            "reliability_expression",
            results_root,
            eval_config=_CFG,
            require_stamped=True,
        )
    write_results([_row(spec_hash="def456")], results_root, eval_config=_CFG)
    assert is_cached(
        "def456", "reliability_expression", results_root, eval_config=_CFG, require_stamped=True
    )
    # Reads never go through is_cached: the unstamped cell is still readable.
    assert len(read_results(results_path("abc123", "reliability_expression", results_root))) == 1
