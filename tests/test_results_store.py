"""The log row cache (docs/COMMANDS.md, "Results store and the log-row cache").

Logs come from inspect's ``mockllm`` provider (`tests/test_results_logs.py`), so nothing
here needs a model. What is pinned: a cache hit serves exactly what a decode would; any
change to the log, the cache format or the bridge code is a miss; a damaged entry is a
miss, never an error or wrong rows; and the store switch reads what it says.
"""

from __future__ import annotations

import logging
import math
import shutil
import threading
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

import core.results as cr
import evals._inspect  # noqa: F401  # registers the writer's reader the log store goes through
from core.results import SCHEMA, eval_log_headers, load_all, log_rows_table, read_result_table
from tests.test_results_logs import HASH_A, HASH_B, _write_log, _write_parquet_cell


@pytest.fixture(autouse=True)
def row_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "row-cache"
    monkeypatch.setenv("MO_EVALS_LOG_ROW_CACHE", str(cache))
    monkeypatch.delenv("MO_EVALS_STORE", raising=False)
    return cache


def _count_decodes(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    real = cr.rows_from_eval_log
    seen: list[Path] = []

    def spy(path: Path | str, **kwargs: bool) -> list[cr.SampleResult]:
        seen.append(Path(path))
        return real(path, **kwargs)

    monkeypatch.setattr(cr, "rows_from_eval_log", spy)
    return seen


_DECODE = cr.rows_from_eval_log  # unspied, so a reference decode is not counted


def _decoded(path: Path) -> pa.Table:
    rows = _DECODE(path, as_written=True)
    return pa.Table.from_pylist([r.to_row() for r in rows], schema=SCHEMA)


def _header(root: Path) -> cr.LogHeader:
    [header] = eval_log_headers(root)
    return header


# ------------------------------------------------------------------ the row cache


def test_a_hit_serves_the_decode_without_decoding(
    tmp_path: Path, row_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_log(tmp_path / "logs" / "vllm")
    decodes = _count_decodes(monkeypatch)
    header = _header(tmp_path)

    miss = log_rows_table(header)
    assert decodes == [path] and len(list(row_cache.glob("*.arrow"))) == 1
    hit = log_rows_table(header)
    assert decodes == [path], "the second read is served from the cache"
    assert hit.equals(miss) and hit.equals(_decoded(path))
    assert hit.schema == SCHEMA


def test_a_projected_hit_skips_columns(tmp_path: Path) -> None:
    _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    log_rows_table(header)

    some = log_rows_table(header, columns=["sample_id", "score"])
    assert some.column_names == ["sample_id", "score"]
    assert some["sample_id"].to_pylist() == ["s2", "s1"], "dataset order, as the bridge wrote"


def test_a_rewritten_log_is_a_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same path, other contents (a re-judge, a retry copied over it): rebuilt, not served."""
    path = _write_log(tmp_path / "logs" / "vllm")
    other = _write_log(tmp_path / "elsewhere", eval_name="fake_eval", spec_hash=HASH_B)
    log_rows_table(_header(tmp_path))
    decodes = _count_decodes(monkeypatch)

    shutil.copyfile(other, path)
    rows = log_rows_table(_header(tmp_path))
    assert decodes == [path]
    assert set(rows["spec_hash"].to_pylist()) == {HASH_B}


def test_a_bridge_change_is_a_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    log_rows_table(header)
    decodes = _count_decodes(monkeypatch)

    monkeypatch.setattr(cr, "_bridge_version", lambda: "edited-bridge")
    log_rows_table(header)
    assert len(decodes) == 1


def test_the_bridge_key_covers_lazily_imported_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Editing ANY first-party code a bridge can run moves the key (refuter's DEFECT 1).

    `introspection_denial`'s bridge imports `evals.introspection._promote_yes_no` inside its
    body, so nothing in any module's globals points at it; the old reachability walk missed
    it and a warm entry served the old scores after an edit. The key now hashes every file
    of every first-party package reachable by a static import scan from core/ and evals/.
    """
    for package in ("core", "evals", "judges"):
        shutil.copytree(cr._REPO / package, tmp_path / package)
    monkeypatch.setattr(cr, "_REPO", tmp_path)
    cr._bridge_version.cache_clear()
    try:
        files = {str(p.relative_to(tmp_path)) for p in cr._bridge_files()}
        assert {"evals/introspection.py", "evals/_task_introspection.py"} <= files
        assert "core/results.py" in files and any(f.startswith("judges/") for f in files)
        before = cr._bridge_version()
        helper = tmp_path / "evals" / "introspection.py"
        helper.write_text(helper.read_text() + "\n# edited\n")
        cr._bridge_version.cache_clear()
        assert cr._bridge_version() != before
    finally:
        cr._bridge_version.cache_clear()


@pytest.mark.parametrize(
    "damage",
    [
        lambda p: p.write_bytes(b"not arrow"),
        lambda p: p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2]),
        lambda p: p.write_bytes(b""),
    ],
    ids=["garbage", "truncated", "empty"],
)
def test_a_damaged_entry_is_a_miss_and_is_repaired(
    tmp_path: Path, row_cache: Path, monkeypatch: pytest.MonkeyPatch, damage
) -> None:
    path = _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    log_rows_table(header)
    [entry] = row_cache.glob("*.arrow")
    damage(entry)
    decodes = _count_decodes(monkeypatch)

    assert log_rows_table(header).equals(_decoded(path))
    assert log_rows_table(header).equals(_decoded(path))
    assert len(decodes) == 1, "the damaged entry was rebuilt by the first read"


def test_a_foreign_entry_is_a_miss(
    tmp_path: Path, row_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An entry for another log under this log's name (a hash clash, a copied cache)."""
    path = _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    log_rows_table(header)
    [entry] = row_cache.glob("*.arrow")
    other = _write_log(tmp_path / "other" / "logs" / "vllm", spec_hash=HASH_B)
    log_rows_table(_header(tmp_path / "other"))
    shutil.copyfile(cr._row_cache_entry(other), entry)
    decodes = _count_decodes(monkeypatch)

    assert log_rows_table(header).equals(_decoded(path)) and decodes == [path]


def test_concurrent_readers_agree_and_leave_a_valid_entry(
    tmp_path: Path, row_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    want = _decoded(path)
    results: list[pa.Table] = []
    errors: list[BaseException] = []

    def read() -> None:
        try:
            for _ in range(5):
                results.append(log_rows_table(header))
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    threads = [threading.Thread(target=read) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(results) == 40
    assert all(r.equals(want) for r in results)
    assert [p.name for p in row_cache.iterdir()] == [cr._row_cache_entry(path).name], (
        "one entry, no stray temp files"
    )
    decodes = _count_decodes(monkeypatch)
    assert log_rows_table(header).equals(want) and decodes == []


def test_the_cache_can_be_switched_off(
    tmp_path: Path, row_cache: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MO_EVALS_LOG_ROW_CACHE", "off")
    path = _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    decodes = _count_decodes(monkeypatch)

    assert log_rows_table(header).equals(log_rows_table(header))
    assert decodes == [path, path] and not row_cache.exists()


def test_an_unwritable_cache_runs_uncached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    monkeypatch.setenv("MO_EVALS_LOG_ROW_CACHE", str(blocker / "cache"))
    path = _write_log(tmp_path / "logs" / "vllm")

    assert log_rows_table(_header(tmp_path)).equals(_decoded(path))


# ------------------------------------------------------------------ the store switch


def _tree(root: Path) -> None:
    """Cell A: log and a DIFFERENT Parquet. Cell B: Parquet only. Cell C: log only."""
    _write_parquet_cell(root, eval_name="fake_eval", spec_hash=HASH_A)  # score 0.5, id p1
    _write_log(root / "logs" / "vllm", eval_name="fake_eval", spec_hash=HASH_A)
    _write_parquet_cell(root, eval_name="other_eval", spec_hash=HASH_B)
    _write_log(root / "logs" / "vllm", eval_name="log_only", spec_hash=HASH_B)


def _cells(table: pa.Table) -> dict[tuple[str, str], list[str]]:
    out: dict[tuple[str, str], list[str]] = {}
    for row in table.to_pylist():
        out.setdefault((row["spec_hash"], row["eval_name"]), []).append(row["sample_id"])
    return {k: sorted(v) for k, v in out.items()}


def test_the_store_switch_reads_what_it_says(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _tree(tmp_path)
    a, b, c = (HASH_A, "fake_eval"), (HASH_B, "other_eval"), (HASH_B, "log_only")

    assert _cells(load_all(tmp_path, store="parquet")) == {a: ["p1"], b: ["p1"]}
    with caplog.at_level(logging.WARNING, logger="core.results"):
        logged = load_all(tmp_path, store="log")
    assert _cells(logged) == {a: ["s1", "s2"], b: ["p1"], c: ["s1", "s2"]}
    assert "1 result file(s)" in caplog.text and "other_eval" in caplog.text, (
        "a cell read from Parquet under the log store is named"
    )
    assert logged.schema == SCHEMA

    default = load_all(tmp_path)
    assert load_all(tmp_path, store="both").equals(default), "the default is `both`"
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    assert _cells(load_all(tmp_path)) == _cells(logged), "the env var picks the store"
    assert _cells(load_all(tmp_path, include_logs=False)) == {a: ["p1"], b: ["p1"]}
    monkeypatch.setenv("MO_EVALS_STORE", "logs")
    with pytest.raises(ValueError, match="MO_EVALS_STORE"):
        load_all(tmp_path)


def test_the_log_store_projects_and_matches_a_decode(tmp_path: Path) -> None:
    _tree(tmp_path)
    full = load_all(tmp_path, store="log")
    some = load_all(tmp_path, store="log", columns=["score"])
    assert some.column_names == ["score"] and some.num_rows == full.num_rows
    log_a = next(h.path for h in eval_log_headers(tmp_path) if h.eval_name == "fake_eval")
    rows_a = full.filter(pc.equal(full["eval_name"], "fake_eval"))
    assert rows_a.equals(_decoded(log_a))


def test_the_log_store_keeps_the_bridge_row_order(tmp_path: Path) -> None:
    """Dataset order, as the Parquet has it: aggregates sum in row order (last-bit drift)."""
    path = _write_log(tmp_path / "logs" / "vllm")  # dataset order s2, s1
    assert [r.sample_id for r in _DECODE(path)] == ["s1", "s2"], "the default stays id-sorted"
    assert load_all(tmp_path, store="log")["sample_id"].to_pylist() == ["s2", "s1"]


def test_the_log_store_reads_the_log_both_reads(tmp_path: Path) -> None:
    """Retries leave several successful logs; `log` takes the one `both` takes (newest mtime)."""
    first = _write_log(tmp_path / "logs" / "vllm")
    second = _write_log(tmp_path / "logs" / "vllm")
    newest = max((first, second), key=lambda p: (p.stat().st_mtime, str(p)))
    assert load_all(tmp_path, store="log").equals(_decoded(newest))
    both = load_all(tmp_path, store="both")
    assert sorted(both["raw_output"].to_pylist()) == sorted(
        _decoded(newest)["raw_output"].to_pylist()
    )


def test_a_parquet_from_its_own_source_log_is_read_from_that_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bridge's own copy (stamped): `both` reads the Parquet, `log` decodes its log."""
    path = _write_log(tmp_path / "logs" / "vllm")
    rows = _DECODE(path, as_written=True)
    stamped = [
        cr.SampleResult(**{**r.__dict__, "meta": {**r.meta, "runner": "eval_set"}}) for r in rows
    ]
    cr.write_results(stamped, tmp_path, source_log=path)
    decodes = _count_decodes(monkeypatch)

    assert load_all(tmp_path, store="both")["meta"].to_pylist()[0].find("runner") > 0
    assert decodes == [], "`both` reads the source's Parquet copy"
    assert load_all(tmp_path, store="log").equals(_decoded(path)) and decodes == [path]


def test_a_parquet_both_keeps_is_read_from_parquet_under_log(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A legacy (unstamped) Parquet newer than the log: `both` keeps it, so `log` does too."""
    _write_log(tmp_path / "logs" / "vllm")
    _write_parquet_cell(tmp_path, eval_name="fake_eval", spec_hash=HASH_A)  # written after
    with caplog.at_level(logging.WARNING, logger="core.results"):
        table = load_all(tmp_path, store="log")
    assert table["sample_id"].to_pylist() == ["p1"]
    assert table.equals(load_all(tmp_path, store="both"))
    assert "1 result file(s)" in caplog.text


def test_the_log_store_serves_the_writers_rows_under_epochs_and_thinking(tmp_path: Path) -> None:
    """epochs=2 and a reasoning part: log-store rows == the Parquet writer's rows (DEFECT 2).

    The writer bridges with `evals._inspect.bridge_for(eval)(log, ..., order=dataset order)`;
    core's plain reading gave duplicate ids (`s1` twice, no `s1/e2`) and no `thinking` meta.
    """
    from inspect_ai import Task, task_with
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import (
        ChatCompletionChoice,
        ChatMessageAssistant,
        ContentReasoning,
        ContentText,
        ModelOutput,
        get_model,
    )
    from inspect_ai.solver import generate

    from evals._inspect import bridge_for
    from tests.test_results_logs import _marker_scorer

    meta = {"eval_name": "fake_eval", "axis": "generality", "spec_hash": HASH_A, "bridged": False}
    task = Task(
        dataset=[Sample(input="hello", id="s2"), Sample(input="there", id="s1")],
        solver=generate(),
        scorer=_marker_scorer(),
        epochs=2,
        metadata=meta,
    )
    outputs = [
        ModelOutput(
            model="mockllm/model",
            choices=[
                ChatCompletionChoice(
                    message=ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=f"think {i}"),
                            ContentText(text=f"a{i}"),
                        ]
                    ),
                    stop_reason="stop",
                )
            ],
        )
        for i in range(4)
    ]
    [log_] = inspect_eval(
        task_with(task, name="fake_eval"),
        model=get_model("mockllm/model", custom_outputs=outputs),
        log_dir=str(tmp_path / "logs" / "vllm"),
        display="none",
        log_level="warning",
    )
    writer = bridge_for("fake_eval")(
        read_eval_log(log_.location),
        eval_name="fake_eval",
        axis="generality",
        spec_hash=HASH_A,
        order={"s2": 0, "s1": 1},
    )
    got = load_all(tmp_path, store="log").to_pylist()
    assert got == [r.to_row() for r in writer]
    assert sorted(r["sample_id"] for r in got) == ["s1", "s1/e2", "s2", "s2/e2"]
    assert all(
        {"thinking", "reasoning_chars"} <= set(cr.SampleResult.from_row(r).meta) for r in got
    )


def test_a_corrupt_log_falls_back_to_the_previous_one(tmp_path: Path) -> None:
    from tests.test_results_logs import _corrupt_the_samples

    older = _write_log(tmp_path / "logs" / "vllm")
    newer = _write_log(tmp_path / "logs" / "vllm")
    _corrupt_the_samples(newer)  # the newest by mtime, so the one tried first
    survivor = older

    table = load_all(tmp_path, store="log")
    assert table.equals(_decoded(survivor))


def test_read_result_table_follows_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Under `log` a result file is served from a log only when that log is its source."""
    _tree(tmp_path)
    [cell_a] = cr.result_files(tmp_path, eval_name="fake_eval")  # unstamped, log beside it
    [cell_b] = cr.result_files(tmp_path, eval_name="other_eval")
    log_c = next(h.path for h in eval_log_headers(tmp_path) if h.eval_name == "log_only")
    cell_c = cr.write_results(
        _DECODE(log_c, as_written=True), tmp_path, source_log=log_c
    )  # the bridge's own copy of log_c

    assert read_result_table(cell_c)["sample_id"].to_pylist() == ["s2", "s1"]
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    decodes = _count_decodes(monkeypatch)
    got = read_result_table(cell_c, columns=["sample_id", "raw_output"])
    assert decodes == [log_c] and got.equals(_decoded(log_c).select(["sample_id", "raw_output"]))
    with caplog.at_level(logging.WARNING, logger="core.results"):
        assert read_result_table(cell_a)["sample_id"].to_pylist() == ["p1"], "no stamp"
        assert read_result_table(cell_b)["sample_id"].to_pylist() == ["p1"], "no log"
    assert str(cell_a) in caplog.text and str(cell_b) in caplog.text
    assert [r.sample_id for r in cr.read_results(cell_c)] == ["s2", "s1"]
    assert decodes == [log_c], "read_results stays on the Parquet: the re-judge writers use it"


def test_a_rejudged_parquet_is_never_served_its_generation_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The refuter's case: `rejudges/<hash>/fake_eval.parquet` next to the original log."""
    _write_log(tmp_path / "logs" / "vllm")
    rows = [
        cr.SampleResult(HASH_A, "fake_eval", "generality", s, 0.5, "rejudged", {"rejudged": True})
        for s in ("s1", "s2")
    ]
    rejudged = cr.write_results(rows, tmp_path / "rejudges")
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    assert set(read_result_table(rejudged)["raw_output"].to_pylist()) == {"rejudged"}


def _rejudge(path: Path, root: Path, *, mark: bool) -> Path:
    """What the re-score scripts write: new verdicts, stamped with the generation log."""
    rows = [
        cr.SampleResult(**{**r.__dict__, "score": 0.777, "meta": {**r.meta, "rejudged": True}})
        for r in _DECODE(path, as_written=True)
    ]
    return cr.write_results(rows, root, source_log=path, rejudged=mark)


@pytest.mark.parametrize("mark", [True, False], ids=["marked", "legacy-unmarked"])
def test_a_rejudge_stamped_with_its_generation_log_stays_on_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, mark: bool
) -> None:
    """Round-2 DEFECT A: the stamp names the generation log, but the rows are not its rows.

    Marked (`rejudged=True`, what the re-score scripts now write): kept on Parquet outright.
    Unmarked (every re-judge written before the mark): kept because its scores disagree with
    the log's, with a warning naming the file.
    """
    path = _write_log(tmp_path / "logs" / "vllm")
    cell = _rejudge(path, tmp_path, mark=mark)
    for store in ("parquet", "both", "log"):
        monkeypatch.setenv("MO_EVALS_STORE", store)
        with caplog.at_level(logging.WARNING, logger="core.results"):
            assert set(load_all(tmp_path)["score"].to_pylist()) == {0.777}, store
            assert set(read_result_table(cell)["score"].to_pylist()) == {0.777}, store
    assert ("differs from its stamped source log" in caplog.text) is not mark
    assert set(cr.log_cell_tables(tmp_path)) == set(), "kept on Parquet by the log store"
    [table] = cr.log_cell_tables(tmp_path, verify=False).values()
    assert set(table["score"].to_pylist()) != {0.777}, "the exit check still sees the drift"


def test_the_log_store_raises_without_the_writers_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 DEFECT B: `evals._inspect` not imported is an error, never a silent fallback."""
    path = _write_log(tmp_path / "logs" / "vllm")
    log_only = tmp_path / "log_only"
    shutil.copytree(tmp_path / "logs", log_only / "logs")
    cell = cr.write_results(_DECODE(path, as_written=True), tmp_path, source_log=path)
    monkeypatch.setattr(cr, "_LOG_BRIDGES", {})
    for root in (tmp_path, log_only):
        with pytest.raises(RuntimeError, match="evals._inspect"):
            load_all(root, store="log")
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    with pytest.raises(RuntimeError, match="evals._inspect"):
        read_result_table(cell)


def test_a_broken_bridge_key_scan_raises_rather_than_skipping_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_log(tmp_path / "logs" / "vllm")

    def broken() -> str:
        raise SyntaxError("a first-party file that does not parse")

    monkeypatch.setattr(cr, "_bridge_version", broken)
    with pytest.raises(SyntaxError):
        load_all(tmp_path, store="log")


@pytest.mark.parametrize("mark", [True, False], ids=["marked", "legacy-unmarked"])
def test_a_stamped_parquet_is_not_superseded_when_its_source_log_changes_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mark: bool
) -> None:
    """Round-3 (Codex) #1: a whitespace append moved the size, and the log won over 0.777."""
    path = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    cell = _rejudge(path, tmp_path, mark=mark)
    with path.open("a") as f:
        f.write("\n")
    assert set(load_all(tmp_path, store="log")["score"].to_pylist()) == {0.777}
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    assert set(read_result_table(cell)["score"].to_pylist()) == {0.777}


def test_a_marked_rejudge_beats_a_newer_copy_of_its_generation_log_in_every_store(
    tmp_path: Path,
) -> None:
    import os

    path = _write_log(tmp_path / "logs" / "vllm")
    cell = _rejudge(path, tmp_path, mark=True)
    copy = path.with_name("copy-of-generation.eval")
    shutil.copyfile(path, copy)
    later = max(cell.stat().st_mtime_ns, path.stat().st_mtime_ns) + 10**9
    os.utime(copy, ns=(later, later))
    for store in ("both", "log"):
        assert set(load_all(tmp_path, store=store)["score"].to_pylist()) == {0.777}, store


def test_a_flipped_bit_in_a_cached_payload_is_a_miss(tmp_path: Path, row_cache: Path) -> None:
    """Round-3 #2a: a readable entry with one score bit flipped was served (0.5 for 1.0)."""
    import pyarrow.ipc as ipc

    path = _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)
    want = log_rows_table(header)
    [entry] = row_cache.glob("*.arrow")
    with pa.OSFile(str(entry)) as f:
        needle = ipc.open_file(f).read_all()["score"].chunk(0).buffers()[1].to_pybytes()
    data = bytearray(entry.read_bytes())
    assert data.count(needle) == 1
    data[data.index(needle) + 6] ^= 0x10
    entry.write_bytes(bytes(data))
    assert log_rows_table(header).equals(want) and log_rows_table(header).equals(_decoded(path))


def test_the_cache_key_names_the_registered_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 #2b: rows bridged by a custom bridge were served to a process without it."""
    path = _write_log(tmp_path / "logs" / "vllm")
    header = _header(tmp_path)

    def custom(log_: object, **kw: object) -> list[cr.SampleResult]:
        rows = evals._inspect.rows_from_log(log_, **kw)
        return [cr.SampleResult(**{**r.__dict__, "score": math.nan}) for r in rows]

    with monkeypatch.context() as m:
        m.setattr(cr, "_LOG_BRIDGES", {**cr._LOG_BRIDGES, "fake_eval": custom})
        assert all(math.isnan(x) for x in log_rows_table(header)["score"].to_pylist())
    assert log_rows_table(header).equals(_decoded(path)), "no custom bridge now: a miss"


def test_read_result_table_raises_on_a_broken_key_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-3 #3: `load_all` raised, `read_result_table` returned the Parquet."""
    path = _write_log(tmp_path / "logs" / "vllm")
    cell = cr.write_results(_DECODE(path, as_written=True), tmp_path, source_log=path)

    def broken() -> str:
        raise SyntaxError("a first-party file that does not parse")

    monkeypatch.setattr(cr, "_bridge_version", broken)
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    with pytest.raises(SyntaxError):
        read_result_table(cell)


def test_the_exit_check_compares_the_first_log_not_the_store_winner(tmp_path: Path) -> None:
    """Round-3 #4: a newer unstamped Parquet hid its disagreeing log (main failed, HEAD passed)."""
    from scripts.check_logs_match_parquet import main

    good = _write_log(tmp_path / "logs" / "vllm", eval_name="good")
    cr.write_results(_DECODE(good, as_written=True), tmp_path, source_log=good)
    _write_log(tmp_path / "logs" / "vllm", eval_name="legacy_bad", spec_hash=HASH_B)
    _write_parquet_cell(tmp_path, eval_name="legacy_bad", spec_hash=HASH_B)  # newer, p1
    assert main([str(tmp_path)]) == 1


@pytest.mark.parametrize("mark", [True, False], ids=["marked", "legacy-unmarked"])
def test_a_later_generation_supersedes_a_rejudge_in_every_store(
    tmp_path: Path, mark: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-4 (Codex) a: a rerun after a re-judge wins, as on main; only copies do not."""
    import os

    generation = _write_log(tmp_path / "logs" / "vllm")
    cell = _rejudge(generation, tmp_path, mark=mark)
    stamped = cr._source_log_stamp(cr._read_parquet(cell).schema)
    assert stamped is not None and stamped.eval_id == _header(tmp_path).eval_id != ""
    if mark:  # the fresh run completed in a later second than the bridge: stated, not slept
        _times(monkeypatch, cell, completed=2e9, bridged=1e9)
    fresh = _write_log(tmp_path / "logs" / "vllm")
    later = max(cell.stat().st_mtime_ns, generation.stat().st_mtime_ns) + 10**9
    os.utime(fresh, ns=(later, later))
    want = _decoded(fresh)["score"].to_pylist()
    assert 0.777 not in want
    for store in ("both", "log"):
        assert load_all(tmp_path, store=store)["score"].to_pylist() == want, store


def _write_sized_log(log_dir: Path, count: int, version: str) -> Path:
    from inspect_ai import Task, task_with
    from inspect_ai import eval as inspect_eval
    from inspect_ai.dataset import Sample
    from inspect_ai.solver import generate

    from tests.test_results_logs import _marker_scorer

    meta = {"eval_name": "fake_eval", "axis": "generality", "spec_hash": HASH_A}
    task = Task(
        dataset=[Sample(input="hi", id=f"s{i}") for i in range(count)],
        solver=generate(),
        scorer=_marker_scorer(),
        version=version,
        metadata={**meta, "bridged": False},
    )
    [out] = inspect_eval(
        task_with(task, name="fake_eval"),
        model="mockllm/model",
        log_dir=str(log_dir),
        display="none",
        log_level="warning",
    )
    assert out.status == "success"
    return Path(out.location)


@pytest.mark.parametrize(
    ("old_dir", "new_dir", "want"), [("z_previous", "a_current", 0), ("vllm", "vllm", 1)]
)
def test_the_exit_check_skips_a_superseded_log_sorting_after_the_current_one(
    tmp_path: Path, old_dir: str, new_dir: str, want: int
) -> None:
    """Round-4 (Codex) d: main's verdicts, both layouts (an old log sorting first fails on
    main too: an inherited limit of `first log in sorted order`)."""
    from scripts.check_logs_match_parquet import main

    old = _write_sized_log(tmp_path / "logs" / old_dir, 1, "old")
    old.rename(old.with_name("A_OLD.eval"))
    current = _write_sized_log(tmp_path / "logs" / new_dir, 2, "new")
    current = current.rename(current.with_name("Z_CURRENT.eval"))
    cr.write_results(_DECODE(current, as_written=True), tmp_path, source_log=current)
    assert main([str(tmp_path)]) == want


def test_include_logs_false_still_validates_the_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MO_EVALS_STORE", "bogus")
    with pytest.raises(ValueError, match="MO_EVALS_STORE"):
        load_all(tmp_path, include_logs=False)
    with pytest.raises(ValueError, match="MO_EVALS_STORE"):
        cr.log_cell_tables(tmp_path)


def test_load_table_memoises_per_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis.aggregate import clear_table_cache, load_table

    _tree(tmp_path)
    clear_table_cache()
    try:
        monkeypatch.setenv("MO_EVALS_STORE", "parquet")
        parquet = load_table(tmp_path).num_rows
        monkeypatch.setenv("MO_EVALS_STORE", "log")
        assert load_table(tmp_path).num_rows != parquet
    finally:
        clear_table_cache()


def test_the_exit_check_ignores_only_the_bridge_stamps(tmp_path: Path) -> None:
    from scripts.check_logs_match_parquet import main

    log_path = _write_log(tmp_path / "logs" / "vllm")
    rows = _DECODE(log_path, as_written=True)

    def write(meta_extra: dict) -> None:  # as the bridge writes it: stamped with its log
        out = [cr.SampleResult(**{**r.__dict__, "meta": {**r.meta, **meta_extra}}) for r in rows]
        cr.write_results(out, tmp_path, source_log=log_path)

    write({"runner": "eval_set", "max_workers": 8, "complete": True, "icl_k": 4})
    assert main([str(tmp_path)]) == 0
    write({"rationale": "a different judge said this"})
    assert main([str(tmp_path)]) == 1
    rows = [cr.SampleResult(**{**r.__dict__, "score": math.nan}) for r in rows]
    write({})
    assert main([str(tmp_path)]) == 1
    rows = _DECODE(log_path, as_written=True)[::-1]  # same rows, another order: a mean can
    write({})  # move in its last bit
    assert main([str(tmp_path)]) == 1


def test_the_column_digest_copies_no_buffers() -> None:
    """Round-4 (Codex) b: hashing a column must not allocate a copy of its payload."""
    import tracemalloc

    import pyarrow as pa

    table = pa.table({"raw_output": pa.array(["x" * 1_000_000] * 8)})  # ~8 MB of payload
    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        cr._column_digests(table)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak < 1_000_000, peak  # a copy of the 8 MB data buffer would show here


def test_a_cache_write_that_fails_in_the_digest_only_skips_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-4 (Codex) b: a MemoryError while hashing is a skipped cache, not a failed read."""
    import pyarrow as pa

    def boom(table: pa.Table) -> dict[str, str]:
        raise MemoryError("no room to hash")

    monkeypatch.setattr(cr, "_column_digests", boom)
    entry = tmp_path / "entry.arrow"
    cr._write_row_cache(entry, {"k": 1}, pa.table({"a": [1]}))  # must not raise
    assert not entry.exists()


def _touch_later(path: Path, than: Path) -> None:
    import os

    later = max(than.stat().st_mtime, path.stat().st_mtime) + 100
    os.utime(path, (later, later))


def _marked_rejudge_of_second_log(tmp_path: Path) -> tuple[Path, Path, Path]:
    """An older generation, then the source generation, then its marked re-judge."""
    import time

    old = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    time.sleep(2)
    source = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    time.sleep(2)  # the bridge (re-judge) time is after both completions
    return old, source, _rejudge(source, tmp_path, mark=True)


@pytest.mark.parametrize("store", ["both", "log"])
def test_round5_an_older_generation_arriving_last_does_not_erase_a_rejudge(
    tmp_path: Path, store: str
) -> None:
    """Round-5 (Codex) item 8: a different eval id is not proof of a LATER run; its mtime
    (a copy, an hf download) is not either. It completed before the re-judge: it loses."""
    old, _, cell = _marked_rejudge_of_second_log(tmp_path)
    stamp = cr._source_log_stamp(cr._read_parquet(cell).schema)
    [old_header] = cr.read_log_headers([old])
    assert stamp is not None and old_header.eval_id != stamp.eval_id
    assert old_header.completed_ts is not None and old_header.completed_ts <= stamp.bridge_time
    _touch_later(old, cell)
    assert load_all(tmp_path, store=store)["score"].to_pylist() == [0.777, 0.777]


@pytest.mark.parametrize("store", ["both", "log"])
def test_round5_a_newer_generation_wins_behind_a_restored_source(
    tmp_path: Path, store: str
) -> None:
    """Round-5 (Codex) item 1: the source copied back last (fresh mtime) must not stop the
    selection before a log that completed AFTER the re-judge, with another eval id."""
    import time

    _, source, cell = _marked_rejudge_of_second_log(tmp_path)
    time.sleep(2)
    fresh = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    _touch_later(source, fresh)
    want = _decoded(fresh)["score"].to_pylist()
    assert 0.777 not in want
    assert load_all(tmp_path, store=store)["score"].to_pylist() == want


@pytest.mark.parametrize("store", ["both", "log"])
def test_round5_a_marked_rejudge_with_unknown_identity_and_time_is_kept_loudly(
    tmp_path: Path, store: str, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-5 (Codex) item 5: no eval id, no bridge time, source pruned, a later-mtime copy:
    nothing proves the copy newer, so the verdicts stay, with a warning naming the cell."""
    import shutil

    import pyarrow.parquet as pq

    _, source, cell = _marked_rejudge_of_second_log(tmp_path)
    table = pq.read_table(cell)
    meta = dict(table.schema.metadata)
    for k in (b"source_log_eval_id", b"source_bridge_time"):
        meta.pop(k)
    pq.write_table(table.replace_schema_metadata(meta), cell)
    copy = source.with_name("copy.json")
    shutil.copyfile(source, copy)
    source.unlink()
    _touch_later(copy, cell)
    caplog.set_level("WARNING")
    assert load_all(tmp_path, store=store)["score"].to_pylist() == [0.777, 0.777]
    assert any("kept the re-judged Parquet" in r.message for r in caplog.records)


def _times(
    monkeypatch: pytest.MonkeyPatch, cell: Path, *, completed: float | None, bridged: float
) -> None:
    """Set the stamped bridge time, and every log's ``stats.completed_at``, explicitly."""
    import pyarrow.parquet as pq

    table = pq.read_table(cell)
    meta = dict(table.schema.metadata)
    meta[b"source_bridge_time"] = repr(bridged).encode()
    pq.write_table(table.replace_schema_metadata(meta), cell)
    monkeypatch.setattr(cr.LogHeader, "completed_at_ts", property(lambda self: completed))


def _fresh_logs_after_a_marked_rejudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, completed: float | None, bridged: float
) -> tuple[Path, Path, Path]:
    source = _write_sized_log(tmp_path / "logs" / "vllm", 3, "src")
    rows = [
        cr.SampleResult(**{**r.__dict__, "score": 0.777}) for r in _DECODE(source, as_written=True)
    ]
    cell = cr.write_results(rows, tmp_path, source_log=source, rejudged=True)
    a = _write_sized_log(tmp_path / "logs" / "vllm", 1, "a")
    b = _write_sized_log(tmp_path / "logs" / "vllm", 2, "b")
    _times(monkeypatch, cell, completed=completed, bridged=bridged)
    return cell, a, b


def test_round6_a_rival_in_the_bridges_own_second_does_not_beat_a_rejudge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-6 item 1: times are to the second, so the same second cannot be ordered:
    keep the re-judge, and say so ONCE for the cell (item 3), naming both logs."""
    cell, a, b = _fresh_logs_after_a_marked_rejudge(
        tmp_path, monkeypatch, completed=1_000_000_000.0, bridged=1_000_000_000.7
    )
    caplog.set_level("WARNING")
    for store in ("both", "log"):
        caplog.clear()
        assert set(load_all(tmp_path, store=store)["score"].to_pylist()) == {0.777}, store
        [warning] = [r.message for r in caplog.records if "kept the re-judged" in r.message]
        assert a.name in warning and b.name in warning and "2 log(s)" in warning


def test_round6_a_rival_in_the_next_second_wins_and_ties_ignore_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-6 items 1-2: the next second is proof; equal completion times are settled by
    eval id then path, so touching the mtimes either way picks the same winner."""
    import os

    cell, a, b = _fresh_logs_after_a_marked_rejudge(
        tmp_path, monkeypatch, completed=1_000_000_001.0, bridged=1_000_000_000.7
    )
    ids = {p: cr.read_log_headers([p])[0].eval_id for p in (a, b)}
    winner = max(ids, key=lambda p: (ids[p], str(p)))
    want = {a: 1, b: 2}[winner]
    for newest in (a, b, a):
        later = max(p.stat().st_mtime for p in (a, b, cell)) + 100
        os.utime(newest, (later, later))
        for store in ("both", "log"):
            assert load_all(tmp_path, store=store).num_rows == want, (newest.name, store)


def test_round6_created_is_not_a_completion_time(tmp_path: Path) -> None:
    """Round-6 item 4: a log with no ``stats.completed_at`` falls back to ``created`` in
    ``completed_ts`` (unchanged, used by the unmarked rule) but is UNKNOWN for the proof."""
    import dataclasses
    from datetime import UTC, datetime

    from core.results import _rejudge_order, _SourceStamp

    _write_log(tmp_path / "logs" / "vllm")
    stamp = _SourceStamp("src.eval", 1, 1000.5, True, "gen-1")

    def hdr(**kw: object) -> cr.LogHeader:
        return dataclasses.replace(_header(tmp_path), eval_id="gen-2", **kw)

    def at(t: int) -> str:
        return datetime.fromtimestamp(t, UTC).isoformat()

    no_end = hdr(created=at(5000), completed_at=None)
    assert no_end.completed_ts is not None and no_end.completed_at_ts is None
    assert _rejudge_order(stamp, "gen-1", no_end) == "unknown"
    assert _rejudge_order(stamp, "gen-1", hdr(completed_at=at(1001))) == "later"
    assert _rejudge_order(stamp, "gen-1", hdr(completed_at=at(1000))) == "unknown"  # same second
    assert _rejudge_order(stamp, "gen-1", hdr(completed_at=at(999))) == "older"
    same = dataclasses.replace(hdr(completed_at=at(2000)), eval_id="gen-1")
    assert _rejudge_order(stamp, "gen-1", same) == "unknown"
    assert _rejudge_order(stamp, None, hdr(completed_at=at(2000))) == "unknown"


@pytest.mark.parametrize("store", ["both", "log"])
def test_round7_an_ambiguous_rival_is_named_whichever_log_sorts_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    store: str,
) -> None:
    """Round-7 F1: with the stamped source newest by mtime the candidate loop stops at it, so
    rivals it never reached went unnamed. Every order of the three logs: the re-judge is
    kept and ONE ambiguity warning names both rivals (completed T+.9, bridged T+.1)."""
    import itertools
    import os

    cell, a, b = _fresh_logs_after_a_marked_rejudge(
        tmp_path, monkeypatch, completed=1_000_000_000.9, bridged=1_000_000_000.1
    )
    [source] = [p for p in (tmp_path / "logs" / "vllm").iterdir() if p not in (a, b)]
    caplog.set_level("WARNING")
    for order in itertools.permutations((source, a, b)):
        base = max(p.stat().st_mtime for p in (source, a, b, cell)) + 100
        for i, p in enumerate(order):  # last in `order` is newest
            os.utime(p, (base + i, base + i))
        caplog.clear()
        assert set(load_all(tmp_path, store=store)["score"].to_pylist()) == {0.777}, order
        [warning] = [r.message for r in caplog.records if "cannot order" in r.message]
        assert a.name in warning and b.name in warning and "2 log(s)" in warning, order
        assert source.name not in warning


@pytest.mark.parametrize("store", ["both", "log"])
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_round7_a_non_finite_bridge_time_is_unknown_not_a_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    store: str,
    bad: str,
) -> None:
    """Round-7 F2: float('nan'|'inf') parses, and math.floor of it raised. The stamp stays a
    MARKED re-judge with its raw time, which `_rejudge_order` reads as unknown."""
    import math

    import pyarrow.parquet as pq

    cell, a, b = _fresh_logs_after_a_marked_rejudge(
        tmp_path, monkeypatch, completed=2_000_000_000.0, bridged=1e9
    )
    table = pq.read_table(cell)
    meta = dict(table.schema.metadata)
    meta[b"source_bridge_time"] = bad.encode()
    pq.write_table(table.replace_schema_metadata(meta), cell)
    stamp = cr._source_log_stamp(cr._read_parquet(cell).schema)
    assert stamp is not None and stamp.rejudged and not math.isfinite(stamp.bridge_time)
    caplog.set_level("WARNING")
    assert set(load_all(tmp_path, store=store)["score"].to_pylist()) == {0.777}
    [warning] = [r.message for r in caplog.records if "cannot order" in r.message]
    assert a.name in warning and b.name in warning


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_round7_a_non_finite_completion_time_is_unknown(tmp_path: Path, bad: float) -> None:
    import dataclasses

    from core.results import _rejudge_order, _SourceStamp

    _write_log(tmp_path / "logs" / "vllm")
    header = dataclasses.replace(_header(tmp_path), eval_id="gen-2")
    stamp = _SourceStamp("src.eval", 1, 1000.5, True, "gen-1")
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cr.LogHeader, "completed_at_ts", property(lambda self: bad))
        assert _rejudge_order(stamp, "gen-1", header) == "unknown"
    finally:
        monkey.undo()


def _marked_cell_and_log(tmp_path: Path, *, rejudged: bool) -> tuple[Path, Path, Path]:
    """A stamped .777 Parquet, bridged 0.1s after its source log completed."""
    import pyarrow.parquet as pq

    src = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    rows = [
        cr.SampleResult(**{**r.__dict__, "score": 0.777}) for r in _DECODE(src, as_written=True)
    ]
    cell = cr.write_results(rows, tmp_path, source_log=src, rejudged=rejudged)
    table = pq.read_table(cell)
    meta = dict(table.schema.metadata)
    done = cr.read_log_headers([src])[0].completed_at_ts
    assert done is not None
    meta[b"source_bridge_time"] = repr(done + 0.1).encode()
    pq.write_table(table.replace_schema_metadata(meta), cell)
    return cell, src, src.parent.parent / "other" / src.name


@pytest.mark.parametrize("store", ["both", "log"])
def test_round8_a_known_different_eval_id_is_not_the_source_despite_name_and_size(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, store: str
) -> None:
    """Round-8 F1: a rival in another directory with the source's basename and byte size but
    a DIFFERENT, known eval id is a different run, not the source copy: it must be named in
    the ambiguity warning (same second as the bridge), exactly like a renamed rival."""
    cell, src, other = _marked_cell_and_log(tmp_path, rejudged=True)
    old = cr.read_log_headers([src])[0].eval_id
    new = ("Z" if old[0] != "Z" else "Y") + old[1:]
    other.parent.mkdir()
    other.write_text(src.read_text().replace(old, new))
    assert other.stat().st_size == src.stat().st_size and other.name == src.name
    assert cr.read_log_headers([other])[0].eval_id == new
    caplog.set_level("WARNING")
    assert set(load_all(tmp_path, store=store)["score"].to_pylist()) == {0.777}
    [warning] = [r.message for r in caplog.records if "cannot order" in r.message]
    assert "1 log(s)" in warning and src.name in warning


@pytest.mark.parametrize("store", ["both"])
@pytest.mark.parametrize("bad", ["nan", "inf", "-inf"])
def test_round8_unmarked_non_finite_bridge_time_keeps_the_legacy_rule(
    tmp_path: Path, store: str, bad: str
) -> None:
    """Round-8 F2: the legacy (unmarked) pruned-source rule compares `log_time > bridge_time`
    on the raw float, as main does. nan / inf are never exceeded, so the Parquet is kept;
    -inf is exceeded, so the surviving log wins. Normalising to None changed nan / inf."""
    import pyarrow.parquet as pq

    cell, src, _ = _marked_cell_and_log(tmp_path, rejudged=False)
    table = pq.read_table(cell)
    meta = dict(table.schema.metadata)
    meta[b"source_bridge_time"] = bad.encode()
    pq.write_table(table.replace_schema_metadata(meta), cell)
    survivor = src.with_name("survivor.json")
    survivor.write_text(src.read_text().replace(cr.read_log_headers([src])[0].eval_id, "S" * 22))
    src.unlink()
    got = set(load_all(tmp_path, store=store)["score"].to_pylist())
    if bad == "-inf":
        assert 0.777 not in got
    else:
        assert got == {0.777}


def _sibling(src: Path, *, new_id: bool, axis: bool) -> Path:
    """An equal-size log with the source's basename in another directory."""
    old = cr.read_log_headers([src])[0].eval_id
    text = src.read_text()
    if new_id:
        text = text.replace(old, ("Z" if old[0] != "Z" else "Y") + old[1:])
    if axis:
        text = text.replace("generality", "capability")
    other = src.parent.parent / "other" / src.name
    other.parent.mkdir(exist_ok=True)
    other.write_text(text)
    assert other.stat().st_size == src.stat().st_size
    return other


@pytest.mark.parametrize("prune", [False, True])
def test_round9_unmarked_selection_keeps_mains_name_and_size_rule(
    tmp_path: Path, prune: bool
) -> None:
    """Round-9 B2: the eval-id override is for MARKED re-judges only. An unmarked file whose
    source (or, once it is pruned, a same-name same-size sibling with another eval id) is
    present keeps its Parquet in the default store, as main's name/size branch did."""
    cell, src, _ = _marked_cell_and_log(tmp_path, rejudged=False)
    _sibling(src, new_id=True, axis=False)
    if prune:
        src.unlink()
    assert set(load_all(tmp_path, store="both")["score"].to_pylist()) == {0.777}


def test_round9_the_log_store_will_not_serve_a_sibling_with_another_eval_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-9 B1: `read_result_table` under `log` took any same-name same-size success log,
    and `_log_matches` compared only ids, scores and outputs. A known, different eval id is
    not the source: the Parquet is served, with a warning."""
    import pyarrow.parquet as pq

    src = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    cell = cr.write_results(cr.rows_from_eval_log(src, as_written=True), tmp_path, source_log=src)
    _sibling(src, new_id=True, axis=True)
    src.unlink()
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    caplog.set_level("WARNING")
    got = read_result_table(cell)
    assert got.to_pylist() == pq.read_table(cell).to_pylist()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def test_round9_log_matches_compares_the_axis(tmp_path: Path) -> None:
    """Round-9 B1: a log of the SAME eval id whose rows carry another axis does not
    reproduce the Parquet, so it may not stand in for it."""
    src = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    cell = cr.write_results(cr.rows_from_eval_log(src, as_written=True), tmp_path, source_log=src)
    other = _sibling(src, new_id=False, axis=True)
    stamp = cr._source_log_stamp(cr._read_parquet(cell).schema)
    assert stamp is not None
    assert cr._log_matches(cell, stamp, cr.read_log_headers([src])[0])
    assert not cr._log_matches(cell, stamp, cr.read_log_headers([other])[0])


def _rival_source(tmp_path: Path, old: str, new: str) -> tuple[Path, Path]:
    """A Parquet bridged from its source log, then the log rewritten (same name, same size,
    same eval id) with ``old`` replaced by an equal-length ``new``. Returns (cell, src)."""
    src = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    cell = cr.write_results(cr.rows_from_eval_log(src, as_written=True), tmp_path, source_log=src)
    text = src.read_text()
    assert old in text and len(old) == len(new)
    src.write_text(text.replace(old, new))
    return cell, src


@pytest.mark.parametrize("columns", [None, ["score"], ["sample_id"], []])
@pytest.mark.parametrize(
    ("column", "old", "new"),
    [("spec_hash", HASH_A, "b" * len(HASH_A)), ("eval_name", "fake_eval", "evil_eval")],
)
def test_round10_a_log_naming_another_organism_or_eval_is_not_served(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    column: str,
    old: str,
    new: str,
    columns: list[str] | None,
) -> None:
    """Round-9 R9-1: `_AGREE` left out spec_hash and eval_name, so a log whose metadata names
    another organism / eval (same name, size, eval id) was served. The identity is checked
    whatever ``columns`` projects, and the Parquet is read instead, with a warning."""
    cell, _ = _rival_source(tmp_path, old, new)
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    caplog.set_level("WARNING")
    got = read_result_table(cell, columns)
    assert got.equals(cr._read_parquet(cell, columns))
    assert any(r.levelname == "WARNING" for r in caplog.records)
    if columns is None:
        assert set(got[column].to_pylist()) == {old}


def test_round10_reordered_parquet_rows_are_not_served_from_the_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-9 R9-2: `_log_matches` sorted both sides, so a Parquet with its rows reordered
    'matched' and the log's order was served. Order is compared as written: the Parquet wins."""
    import pyarrow.parquet as pq

    src = _write_log(tmp_path / "logs" / "vllm", log_format="json")
    cell = cr.write_results(cr.rows_from_eval_log(src, as_written=True), tmp_path, source_log=src)
    orig = pq.read_table(cell)
    assert orig.num_rows > 1
    pq.write_table(orig.take(pa.array(list(reversed(range(orig.num_rows))))), cell)
    stamp = cr._source_log_stamp(pq.read_schema(str(cell)))
    assert stamp is not None
    assert not cr._log_matches(cell, stamp, cr.read_log_headers([src])[0])
    monkeypatch.setenv("MO_EVALS_STORE", "log")
    caplog.set_level("WARNING")
    got = read_result_table(cell)
    assert got.to_pylist() == pq.read_table(cell).to_pylist()
    assert got["sample_id"].to_pylist() != orig["sample_id"].to_pylist()
    assert any(r.levelname == "WARNING" for r in caplog.records)


def _corrupt_optional_id(cell: Path, value: bytes) -> None:
    import pyarrow.parquet as pq

    table = pq.read_table(cell)
    meta = dict(table.schema.metadata)
    assert b"source_log_eval_id" in meta
    meta[b"source_log_eval_id"] = value
    pq.write_table(table.replace_schema_metadata(meta), cell)


@pytest.mark.parametrize("value", [b"\xff", b""])
def test_round10b_an_undecodable_optional_eval_id_is_unknown_not_no_stamp(
    tmp_path: Path, value: bytes
) -> None:
    """Round-10b: main never reads `source_log_eval_id`, so a bad value must not void the
    stamp (which sent the cell down the legacy mtime rule and changed the DEFAULT store):
    it is None (unknown), as on a stamp written before the key existed."""
    import pyarrow.parquet as pq

    cell, _, _ = _marked_cell_and_log(tmp_path, rejudged=False)
    _corrupt_optional_id(cell, value)
    stamp = cr._source_log_stamp(pq.read_schema(str(cell)))
    assert stamp is not None and stamp.eval_id is None and stamp.log_name


def test_round10b_a_bad_optional_id_leaves_the_default_store_as_main_has_it(
    tmp_path: Path,
) -> None:
    """Round-10b: unmarked .777 Parquet, its source log present and NEWER. Main (name/size
    stamp) keeps the Parquet; so must `both` when the optional id is garbage."""
    import os

    cell, src, _ = _marked_cell_and_log(tmp_path, rejudged=False)
    _corrupt_optional_id(cell, b"\xff")
    os.utime(src, (2_000_000_000, 2_000_000_000))
    assert set(load_all(tmp_path, store="both")["score"].to_pylist()) == {0.777}


def test_round10b_a_marked_rejudge_with_unknown_id_is_kept_and_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Round-10b: a marked re-judge whose optional id is undecodable keeps its verdicts
    against rivals in the bridge's own second, with the ambiguity warning, as an unknown
    (legacy) id does."""
    cell, a, b = _fresh_logs_after_a_marked_rejudge(
        tmp_path, monkeypatch, completed=1_000_000_000.0, bridged=1_000_000_000.7
    )
    _corrupt_optional_id(cell, b"\xff")
    _times(monkeypatch, cell, completed=1_000_000_000.0, bridged=1_000_000_000.7)
    caplog.set_level("WARNING")
    for store in ("both", "log"):
        caplog.clear()
        assert set(load_all(tmp_path, store=store)["score"].to_pylist()) == {0.777}, store
        assert any("kept the re-judged" in r.message for r in caplog.records), store
