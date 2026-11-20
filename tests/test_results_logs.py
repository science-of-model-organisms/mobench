"""Tests for the `.eval`-log side of `core/results.py` (Phase 5 of the Inspect migration).

Logs are produced here with inspect's built-in ``mockllm`` provider — no organism
machinery, no GPU — and written into the ``<root>/logs/<group>/`` layout the runners
use. What matters is the contract: a SELF-DESCRIBING log (eval_name / axis / spec_hash
in its metadata) round-trips to `SampleResult` rows; anything else is skipped loudly.
"""

from __future__ import annotations

import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai import Task, task_with
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from core.results import (
    SampleResult,
    iter_eval_logs,
    iter_result_files,
    load_all,
    rows_from_eval_log,
    write_results,
)

HASH_A = "a" * 16
HASH_B = "b" * 16


@scorer(metrics=[accuracy()])
def _marker_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(
            value=float(bool(state.output.completion)),
            metadata={"rationale": "canned-by-test"},
        )

    return score


def _write_log(
    log_dir: Path,
    *,
    eval_name: str = "fake_eval",
    axis: str = "generality",
    spec_hash: str = HASH_A,
    self_describing: bool = True,
    bridged: bool = False,
    log_format: str = "eval",
) -> Path:
    task = Task(
        # Ids chosen so sample_id sort order differs from dataset order.
        dataset=[Sample(input="hello", id="s2"), Sample(input="there", id="s1")],
        solver=generate(),
        scorer=_marker_scorer(),
        metadata=(
            {
                "eval_name": eval_name,
                "axis": axis,
                "spec_hash": spec_hash,
                "bridged": bridged,
            }
            if self_describing
            else None
        ),
    )
    # Named after the eval alone, as both runners name it: the organism lives in the
    # model and in the metadata above, never in the task name.
    task = task_with(task, name=eval_name)
    logs = inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(log_dir),
        display="none",
        log_level="warning",
        log_format=log_format,
    )
    assert logs and logs[0].status == "success"
    return Path(logs[0].location)


def _write_parquet_cell(
    root: Path, *, eval_name: str, spec_hash: str, source_log: Path | None = None
) -> Path:
    """One Parquet cell, through the real write path.

    ``source_log=None`` is a LEGACY (unstamped) file — the shape every Parquet written
    before source-identity authority has, and the one the mtime fallback still governs.
    """
    rows = [
        SampleResult(
            spec_hash=spec_hash,
            eval_name=eval_name,
            axis="generality",
            sample_id="p1",
            score=0.5,
            raw_output="[FAKE] parquet row",
            meta={},
        )
    ]
    return write_results(
        rows, root, spec_hash=spec_hash, eval_name=eval_name, source_log=source_log
    )


def _corrupt_the_samples(path: Path) -> None:
    """Leave the log's header readable and its body not — the case the fallback is for.

    A truncated file fails the cheap header read and is skipped before the fallback loop
    is reached; a log whose `samples/` entries are unparseable passes the header read and
    then blows up in the full decode, which is what used to take the whole cell down.
    """
    import shutil
    import zipfile

    rebuilt = path.with_suffix(".rebuilt")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(rebuilt, "w") as dst:
        for info in src.infolist():
            data = src.read(info.filename)
            dst.writestr(info, b"{ not json" if info.filename.startswith("samples/") else data)
    shutil.move(str(rebuilt), str(path))


def _log_finished_at(path: Path) -> float:
    """Epoch seconds the log records for its own completion — the value the rule dates against."""
    from datetime import datetime

    from inspect_ai.log import read_eval_log

    return datetime.fromisoformat(
        read_eval_log(str(path), header_only=True).stats.completed_at
    ).timestamp()


# ------------------------------------------------------------------ the reader


def test_reader_round_trips_a_self_describing_log(tmp_path: Path) -> None:
    path = _write_log(tmp_path / "logs" / "vllm")
    rows = rows_from_eval_log(path)
    assert [r.sample_id for r in rows] == ["s1", "s2"], "sample_id-sorted, documented"
    for r in rows:
        assert (r.spec_hash, r.eval_name, r.axis) == (HASH_A, "fake_eval", "generality")
        assert r.score == 1.0
        assert r.raw_output, "raw_output ALWAYS populated"
        assert r.meta["rationale"] == "canned-by-test", "score metadata lands in meta"


def test_reader_refuses_a_log_that_does_not_describe_itself(tmp_path: Path) -> None:
    path = _write_log(tmp_path / "logs" / "vllm", self_describing=False)
    with pytest.raises(ValueError, match="not a self-describing eval log"):
        rows_from_eval_log(path)


# -------------------------------------------------------------------- load_all


def test_load_all_recovers_a_cell_that_exists_only_as_a_log(tmp_path: Path) -> None:
    _write_parquet_cell(tmp_path, eval_name="other_eval", spec_hash=HASH_B)
    _write_log(tmp_path / "logs" / "vllm")  # fake_eval x HASH_A, no Parquet
    table = load_all(tmp_path)
    cells = set(zip(table["spec_hash"].to_pylist(), table["eval_name"].to_pylist(), strict=True))
    assert cells == {(HASH_B, "other_eval"), (HASH_A, "fake_eval")}


def test_load_all_reads_the_parquet_when_it_is_the_logs_own_bridged_copy(
    tmp_path: Path,
) -> None:
    """LEGACY (unstamped) Parquet: normal write order — log, then bridge — leaves it at
    least as new as the log, and that fresh cache is what gets read (it carries the
    runner stamps). This is the fallback rule, kept for files written before the stamp."""
    import os

    log_path = _write_log(tmp_path / "logs" / "vllm")
    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A)
    pq_file = list(iter_result_files(tmp_path))[0]
    os.utime(pq_file, (log_path.stat().st_mtime + 1,) * 2)
    table = load_all(tmp_path)
    assert table.num_rows == 1, "the log must not double-count a bridged cell"
    assert table["raw_output"][0].as_py() == "[FAKE] parquet row"


def test_load_all_prefers_a_log_newer_than_its_parquet(tmp_path: Path) -> None:
    """LEGACY (unstamped) Parquet, the other direction: an `inspect score` re-judge lands
    in the log AFTER the bridge wrote — the newer log must win or re-judges never reach
    the scorecard, and its rows must not double-count beside the stale Parquet."""
    import os

    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A)
    log_path = _write_log(tmp_path / "logs" / "vllm")  # same cell, different content
    pq_file = list(iter_result_files(tmp_path))[0]
    os.utime(log_path, (pq_file.stat().st_mtime + 1,) * 2)
    table = load_all(tmp_path)
    assert table.num_rows == 2, "the newer log's rows replace the stale Parquet's"
    assert all(r != "[FAKE] parquet row" for r in table["raw_output"].to_pylist())


# ------------------------------------------------- source-identity authority
#
# mtime does not survive transport: `hf download`, rsync and cp rewrite it in arrival
# order, so on a downloaded tree an arbitrary subset of cells flipped to log-sourced
# rows — silently dropping the `runner` / `max_workers` / `gen_threads` stamps that only
# a bridged row carries, which is what CLAUDE.md tells a reader to check width parity
# from. The bridge therefore stamps the source log's name and size into the Parquet.


def test_the_bridge_stamps_its_source_log_into_the_parquet(tmp_path: Path) -> None:
    log_path = _write_log(tmp_path / "logs" / "vllm")
    parquet = _write_parquet_cell(
        tmp_path, eval_name="fake_eval", spec_hash=HASH_A, source_log=log_path
    )
    metadata = pq.read_table(parquet).schema.metadata
    assert metadata[b"source_log_name"].decode() == log_path.name
    assert int(metadata[b"source_log_size"]) == log_path.stat().st_size
    # And WHEN it bridged, which is what dates a surviving log once that source is pruned.
    assert float(metadata[b"source_bridge_time"]) >= _log_finished_at(log_path)


def test_a_transported_tree_keeps_its_parquet_when_the_log_mtime_wins(tmp_path: Path) -> None:
    """THE BUG: a copied tree's log is newer by mtime than its own bridged Parquet.

    Nothing about the cell changed — only the order the files happened to arrive in — so
    the Parquet must still win, and with it the serving stamps a log cannot carry.
    """
    import os

    log_path = _write_log(tmp_path / "logs" / "vllm")
    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A, source_log=log_path)
    os.utime(log_path, (list(iter_result_files(tmp_path))[0].stat().st_mtime + 100,) * 2)

    table = load_all(tmp_path)
    assert table.num_rows == 1, "the stamp says this log IS the Parquet's source"
    assert table["raw_output"][0].as_py() == "[FAKE] parquet row"


def test_a_rewritten_log_still_wins_over_its_stamped_parquet(tmp_path: Path) -> None:
    """The stamp must not freeze a cell: a re-judge that rewrites the log changes its
    SIZE, and a size the stamp does not match is a different log as far as this is
    concerned — so the log wins, exactly as the mtime rule used to make it."""
    import os

    log_path = _write_log(tmp_path / "logs" / "vllm")
    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A, source_log=log_path)
    with log_path.open("ab") as fh:  # an `inspect score` rewrite, in miniature
        fh.write(b"\0")
    os.utime(log_path, (list(iter_result_files(tmp_path))[0].stat().st_mtime + 100,) * 2)

    table = load_all(tmp_path)
    assert table.num_rows == 2, "the rewritten log's rows replace the stale Parquet's"
    assert all(r != "[FAKE] parquet row" for r in table["raw_output"].to_pylist())


def test_an_undecodable_newest_log_falls_back_to_the_previous_one(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A cell must survive one bad log. eval_set retries leave several logs per cell, and
    taking only the newest meant a corrupt body dropped rows the earlier attempt still
    holds — a whole cell missing from the scorecard, with one warning line to say so."""
    import logging
    import os

    log_dir = tmp_path / "logs" / "vllm"
    good = _write_log(log_dir)
    bad = _write_log(log_dir)  # same cell, second attempt
    _corrupt_the_samples(bad)
    os.utime(bad, (good.stat().st_mtime + 100,) * 2)

    with caplog.at_level(logging.WARNING, logger="core.results"):
        table = load_all(tmp_path)
    assert table.num_rows == 2, "recovered from the older readable log"
    assert any(bad.name in r.getMessage() for r in caplog.records), "and named the skip"


def test_load_all_include_logs_off_reads_parquet_only(tmp_path: Path) -> None:
    _write_log(tmp_path / "logs" / "vllm")
    assert load_all(tmp_path, include_logs=False).num_rows == 0


def test_load_all_skips_foreign_logs_without_failing(tmp_path: Path) -> None:
    _write_log(tmp_path / "logs" / "vllm", self_describing=False)  # e.g. a petri log
    assert load_all(tmp_path).num_rows == 0


def test_json_format_logs_are_found_too(tmp_path: Path) -> None:
    """`--log-format json` is a supported Inspect format; a `.eval`-only glob loses it.

    Recovery, the log/Parquet gate and `load_all` all walk `iter_eval_logs`, so a run
    made in the JSON format was invisible to every one of them — no error, just a cell
    that silently is not there.
    """
    path = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    assert path.suffix == ".json"
    assert iter_eval_logs(tmp_path) == [path]
    assert load_all(tmp_path).num_rows == 2


def test_inspect_manifests_in_a_log_dir_are_not_mistaken_for_logs(tmp_path: Path) -> None:
    """`eval-set.json` and `logs.json` live IN the log dir and are not logs.

    They arrived with the JSON glob above: eval_set writes the first as its task->log
    pairing manifest and `inspect view bundle` writes the second. Reading either as a log
    is a warning per call in `load_all` and an unhandled decode in the exit check.
    """
    log_dir = tmp_path / "logs" / "vllm"
    real = _write_log(log_dir)
    (log_dir / "eval-set.json").write_text('{"eval_set_id": "x"}')
    (log_dir / "logs.json").write_text("{}")
    assert iter_eval_logs(tmp_path) == [real]


def test_scanner_findings_are_not_read_as_logs(tmp_path: Path) -> None:
    """`scans/` holds results ABOUT logs, and lands inside the log dir it scanned.

    Both scanner paths write there — `scout scan --scans <log_dir>/scans` offline and
    `eval_set(scanner=...)` online — and the directory carries `_scan.json` and
    `_summary.json`, which the `.json` glob would otherwise hand to `read_eval_log`.
    """
    log_dir = tmp_path / "logs" / "vllm"
    real = _write_log(log_dir)
    scans = log_dir / "scans" / "scan_id=abc"
    scans.mkdir(parents=True)
    (scans / "_scan.json").write_text('{"scan_id": "abc"}')
    (scans / "_summary.json").write_text("{}")
    assert iter_eval_logs(tmp_path) == [real]


def test_load_all_columns_still_recovers_a_log_only_cell(tmp_path: Path) -> None:
    """A projection must not cost the log recovery.

    `load_all` decides which cells the logs need to fill by comparing (spec_hash,
    eval_name) against the Parquet it just read — so a projection that drops those two
    columns has to be widened for the read and narrowed again at the end, not passed
    straight through to `pq.read_table`.
    """
    _write_parquet_cell(tmp_path, eval_name="other_eval", spec_hash=HASH_B)
    _write_log(tmp_path / "logs" / "vllm")  # fake_eval x HASH_A, no Parquet

    table = load_all(tmp_path, columns=["score"])
    assert table.schema.names == ["score"], "narrowed back to what was asked for"
    assert table.num_rows == load_all(tmp_path).num_rows == 3  # 1 parquet + 2 log samples


# ------------------------------------------------------------- the exit check


def test_exit_check_passes_on_matching_stores_and_fails_on_drift(tmp_path: Path) -> None:
    from scripts.check_logs_match_parquet import main

    log_path = _write_log(tmp_path / "logs" / "vllm")
    rows = rows_from_eval_log(log_path, as_written=True)  # as the bridge writes them
    write_results(rows, tmp_path, source_log=log_path)  # and stamps them with their log
    assert main([str(tmp_path)]) == 0

    drifted = [SampleResult(**{**r.to_row(), "meta": {}, "score": r.score + 0.25}) for r in rows]
    write_results(drifted, tmp_path, source_log=log_path)
    assert main([str(tmp_path)]) == 1


def test_exit_check_fails_when_it_compared_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    """A gate that checked nothing must not exit 0 — that is how this gated nothing.

    A fake-mode tree looks exactly like this: `run_all.sh` deletes each fake `.eval` log
    as soon as its rows are bridged, leaving Parquet with no log to compare it against.
    """
    from scripts.check_logs_match_parquet import main

    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A)  # no logs/
    assert main([str(tmp_path)]) == 1
    assert "MO_EVALS_KEEP_FAKE_LOGS" in capsys.readouterr().out, "says how to fix it"


def test_reader_score_nan_when_unscored(tmp_path: Path) -> None:
    task = Task(
        dataset=[Sample(input="hi", id="s1")],
        solver=generate(),
        metadata={"eval_name": "fake_eval", "axis": "generality", "spec_hash": HASH_A},
    )
    logs = inspect_eval(
        task,
        model="mockllm/model",
        log_dir=str(tmp_path / "logs" / "vllm"),
        display="none",
        log_level="warning",
    )
    rows = rows_from_eval_log(Path(logs[0].location))
    assert len(rows) == 1 and math.isnan(rows[0].score), "unmeasured is NaN, never 0.0"


def test_exit_check_reads_the_flat_serving_meta_keys(capsys: pytest.CaptureFixture) -> None:
    """The concurrency report reads `meta['runner']` etc. FLAT, as `_stamp` writes them.

    A nested `meta['serving']` read comes back empty on every real tree — caught when
    the report printed nothing against a tree with 180 stamped cells.
    """
    from scripts.check_logs_match_parquet import _report_concurrency, _serving

    row = SampleResult(
        spec_hash=HASH_A,
        eval_name="fake_eval",
        axis="generality",
        sample_id="s1",
        score=1.0,
        raw_output="[FAKE] x",
        meta={"runner": "eval_set", "max_workers": 64, "gen_threads": 256},
    )
    assert _serving(row) == ("eval_set", 64, 256)
    wide = SampleResult(**{**row.to_row(), "meta": {"runner": "eval_set", "max_workers": 32}})
    _report_concurrency({(HASH_A, "fake_eval"): [row], (HASH_B, "fake_eval"): [wide]})
    out = capsys.readouterr().out
    assert "('eval_set', 64, 256) -> 1" in out
    assert "mixed concurrency" in out and "fake_eval" in out


# --------------------------------------------------- the registered-bridge seam
#
# Regression cover for the drift measured on a real tree (2026-08-26): with no seam,
# `rows_from_eval_log` was a third implementation of "log -> rows" that disagreed with
# the Parquet the serving layer bridged, on exactly the evals that register a custom
# bridge — `mu_decisiveness` (chunked: ~25 units per sample) and `introspection_denial`
# (scores the yes/no reading, not the judge verdict). 2 of 22 cells, silently.


def test_rows_from_eval_log_uses_a_registered_bridge(tmp_path: Path) -> None:
    """A registered bridge OVERRIDES the plain one-row-per-sample reading.

    This is what makes the log store and the Parquet store agree by construction: both
    go through the same function. Without the seam the assertion below reads back the
    two raw samples instead of the bridge's exploded rows.
    """
    from core.results import _LOG_BRIDGES, register_log_bridge

    log_path = _write_log(tmp_path / "logs" / "vllm", eval_name="bridged_eval")

    def _explode(log_, *, eval_name: str, axis: str, spec_hash: str):
        # One log sample -> two rows, so a per-sample reading cannot fake this.
        return [
            SampleResult(
                spec_hash=spec_hash,
                eval_name=eval_name,
                axis=axis,
                sample_id=f"{sample.id}-{part}",
                score=1.0,
                raw_output="[FAKE] exploded",
                meta={"bridged": True},
            )
            for sample in (log_.samples or [])
            for part in ("i", "j")
        ]

    register_log_bridge("bridged_eval", _explode)
    try:
        rows = rows_from_eval_log(log_path)
        assert [r.sample_id for r in rows] == ["s1-i", "s1-j", "s2-i", "s2-j"]
        assert all(r.meta["bridged"] for r in rows)
    finally:
        _LOG_BRIDGES.pop("bridged_eval", None)


def test_unbridged_evals_keep_the_plain_per_sample_reading(tmp_path: Path) -> None:
    """The seam is opt-in: an eval with no registered bridge is read exactly as before."""
    log_path = _write_log(tmp_path / "logs" / "vllm", eval_name="unbridged_eval")
    rows = rows_from_eval_log(log_path)
    assert [r.sample_id for r in rows] == ["s1", "s2"]


def test_register_log_bridge_rejects_a_conflicting_reregistration() -> None:
    """Re-registering the SAME function is fine (module reimport); a different one is a bug."""
    from core.results import _LOG_BRIDGES, register_log_bridge

    def _one(log_, **kw):
        return []

    def _two(log_, **kw):
        return []

    register_log_bridge("dup_eval", _one)
    try:
        register_log_bridge("dup_eval", _one)  # idempotent
        with pytest.raises(ValueError, match="already registered"):
            register_log_bridge("dup_eval", _two)
    finally:
        _LOG_BRIDGES.pop("dup_eval", None)


def test_register_log_bridge_tells_two_factory_bridges_apart() -> None:
    """The case the guard exists for, and the one a name-only identity cannot see.

    Every bridge that needs this comes from ONE factory — `rows_from_packed(key)` — so
    `mu_decisiveness`'s bridge and the drift cells' bridge share a `__module__` and a
    `__qualname__` (`rows_from_packed.<locals>.bridge`) and differ only in the key they
    closed over. Comparing by name alone accepted `drift_turns` over `mu_units` silently.
    """
    from core.results import _LOG_BRIDGES, register_log_bridge
    from evals._inspect import rows_from_packed

    register_log_bridge("packed_eval", rows_from_packed("mu_units"))
    try:
        # A fresh closure over the SAME key is the module-reimport case: still idempotent.
        register_log_bridge("packed_eval", rows_from_packed("mu_units"))
        with pytest.raises(ValueError, match="already registered"):
            register_log_bridge("packed_eval", rows_from_packed("drift_turns"))
    finally:
        _LOG_BRIDGES.pop("packed_eval", None)


def test_every_inspect_bridge_reaches_core() -> None:
    """Every eval that registers a custom bridge must also register it with core.

    Registry-wide, so a NEW bridged eval cannot reintroduce the drift by forgetting the
    core side — the failure mode is silent (wrong rows, no error), so it needs a test
    that does not name the evals it guards.
    """
    import serving.inspect_runner  # noqa: F401  — imports every evals._task_* module
    from core.results import _LOG_BRIDGES, _bridge_identity
    from evals._inspect import all_evals

    bridges = {e.name: e.bridge for e in all_evals() if e.bridge is not None}
    assert bridges, "no bridges registered at all — the _task_* imports did not run"
    assert set(bridges) <= set(_LOG_BRIDGES), (
        f"bridges missing from core: {sorted(set(bridges) - set(_LOG_BRIDGES))}"
    )

    # Core's own comparison, not a local copy of it: this test used to reimplement the
    # (module, qualname) pair, which cannot tell `rows_from_packed("mu_units")` from
    # `rows_from_packed("drift_turns")` — the exact mix-up it is meant to catch.
    for name, fn in bridges.items():
        assert _bridge_identity(_LOG_BRIDGES[name]) == _bridge_identity(fn), (
            f"core has a DIFFERENT bridge for {name}"
        )


def test_a_bridged_log_refuses_to_read_without_its_bridge(tmp_path: Path) -> None:
    """`bridged: true` in the log + no registered bridge = raise, never guess.

    This is the guard for the failure mode that actually happened: a consumer reads logs
    without importing the eval package, gets a per-sample reading of a chunked log, and
    reports rows that are confidently wrong. `load_all` catches this and skips the cell
    with a warning; a direct caller sees the error.
    """
    log_path = _write_log(tmp_path / "logs" / "vllm", eval_name="needs_bridge", bridged=True)
    with pytest.raises(ValueError, match="no log bridge is registered"):
        rows_from_eval_log(log_path)


def _set_bridge_time(parquet: Path, value: str | None) -> None:
    """Rewrite — or with None remove — the bridge time the Parquet's stamp carries.

    Two reasons a test cannot just use the clock. Inspect records a log's own times to the
    SECOND, so two mockllm logs written back to back are indistinguishable by them, and a
    test that needs a log to genuinely post-date the bridge has to say so rather than race
    it. And `None` reproduces a Parquet written before the bridge time was stamped at all,
    which is the backward-compatible path.
    """
    table = pq.read_table(parquet)
    metadata = dict(table.schema.metadata or {})
    metadata.pop(b"source_bridge_time", None)
    if value is not None:
        metadata[b"source_bridge_time"] = value.encode()
    pq.write_table(table.replace_schema_metadata(metadata), parquet, compression="zstd")


def test_a_log_older_than_the_bridge_does_not_supersede_its_parquet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The stamped source log is gone and an OLDER one is promoted in its place.

    That combination cannot be a re-judge — a re-judge is written after the bridge — so it
    is either a pruned log tree (log trees run to GBs: 5.9 GB for one
    full-config run) or a transported tree whose mtimes were rewritten. Reading the log
    would be wrong in both: it reverts the cell to stale rows and drops the `runner` /
    `max_workers` / `gen_threads` stamps that only a bridged row carries, which CLAUDE.md's
    determinism rule has a reader check width parity from.

    mtime cannot separate those two from a real re-judge — rewriting mtimes in arrival
    order is the whole reason the stamp exists — so the decision is made on two times
    recorded INSIDE the files: this log says it finished before the Parquet says it was
    bridged, so it is not a re-judge and the Parquet keeps the cell. Nothing is lost and
    nothing needs flagging, so the warning below must NOT fire here; the case where the
    recorded times are missing still gets it (next test).
    """
    import logging
    import os

    old_log = _write_log(tmp_path / "logs" / "vllm")  # older, and not the Parquet's source
    src_log = _write_log(tmp_path / "logs" / "vllm2")  # the Parquet's stamped source
    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A, source_log=src_log)
    parquet = list(iter_result_files(tmp_path))[0]
    os.utime(old_log, (parquet.stat().st_mtime - 500,) * 2)
    src_log.unlink()

    with caplog.at_level(logging.WARNING, logger="core.results"):
        table = load_all(tmp_path)

    assert table["raw_output"].to_pylist() == ["[FAKE] parquet row"], (
        "a log that finished before the bridge cannot be a re-judge of the bridged rows"
    )
    assert not any("OLDER than the Parquet" in r.getMessage() for r in caplog.records), (
        "nothing was lost, so nothing to flag"
    )


def test_a_newer_log_wins_even_when_the_stamped_source_was_pruned(tmp_path: Path) -> None:
    """The flip side: the stamp must not freeze a cell just because its source is gone.

    Pruning the log tree is routine, and a re-judge or a rerun landing afterwards is the
    case the log store exists for. What separates it from the test above is the log's own
    recorded time, not the mtime either file happens to carry.
    """
    src_log = _write_log(tmp_path / "logs" / "vllm2")
    parquet = _write_parquet_cell(
        tmp_path, eval_name="fake_eval", spec_hash=HASH_A, source_log=src_log
    )
    _write_log(tmp_path / "logs" / "vllm")  # a fresh run for the same cell
    src_log.unlink()
    _set_bridge_time(parquet, "0")  # the bridge ran in 1970; this rerun did not

    table = load_all(tmp_path)
    assert table.num_rows == 2, "the rerun post-dates the bridge, so its rows replace them"
    assert all(r != "[FAKE] parquet row" for r in table["raw_output"].to_pylist())


def test_an_undatable_older_log_still_supersedes_its_parquet_and_still_says_so(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A Parquet written before the bridge time was stamped keeps the old behaviour.

    Its stamp names a source log and a size but records no bridge time, so there is
    nothing to date the survivor against and the rule falls back to reading the log — the
    right call for a transported tree and the wrong one for a pruned one, exactly as
    before. What it must not do is stay quiet: a cell reverting to stale rows also loses
    the `runner` / `max_workers` / `gen_threads` stamps that only a bridged row carries.
    """
    import logging
    import os

    old_log = _write_log(tmp_path / "logs" / "vllm")
    src_log = _write_log(tmp_path / "logs" / "vllm2")
    parquet = _write_parquet_cell(
        tmp_path, eval_name="fake_eval", spec_hash=HASH_A, source_log=src_log
    )
    _set_bridge_time(parquet, None)  # exactly what the pre-2026-08-31 writer produced
    os.utime(old_log, (parquet.stat().st_mtime - 500,) * 2)
    src_log.unlink()

    with caplog.at_level(logging.WARNING, logger="core.results"):
        table = load_all(tmp_path)

    assert "[FAKE] parquet row" not in table["raw_output"].to_pylist(), (
        "unchanged for a Parquet that predates the bridge-time stamp"
    )
    assert any("OLDER than the Parquet" in r.getMessage() for r in caplog.records), (
        "a cell reverting to stale rows must not do it at log.info, which the sweep hides"
    )
