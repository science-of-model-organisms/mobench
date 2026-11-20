"""The one read path in `core/results.py` (docs/COMMANDS.md, "Results store").

Scripts no longer glob result files or open Parquet themselves; they ask `result_files`,
`parse_result_filename` and `read_result_table`. Log discovery reads headers through
`read_log_headers`, which keeps a stat manifest so a header is read once per log version.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.results import (
    LogHeader,
    ResultFileName,
    SampleResult,
    eval_log_headers,
    iter_result_files,
    parse_result_filename,
    read_log_headers,
    read_result_table,
    result_files,
    write_results,
)


def _cell(root: Path, spec_hash: str, eval_name: str, score: float = 0.5) -> Path:
    rows = [
        SampleResult(
            spec_hash=spec_hash,
            eval_name=eval_name,
            axis="generality",
            sample_id=f"s{i}",
            score=score,
            raw_output="[FAKE] row",
            meta={"i": i},
        )
        for i in range(2)
    ]
    # A bare hash writes the legacy flat name; move it to the seed-tagged nested one.
    flat = write_results(rows, root / "_flat", spec_hash=spec_hash, eval_name=eval_name)
    dest = root / f"org__{spec_hash}" / spec_hash / f"seed0__{spec_hash}__{eval_name}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    flat.rename(dest)
    return dest


# ------------------------------------------------------------------ file names


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("seed0__abc123__mmlu.parquet", ResultFileName(0, "abc123", "mmlu")),
        (
            "seed12__ab__whitebox_truth_self.parquet",
            ResultFileName(12, "ab", "whitebox_truth_self"),
        ),
        ("seed1__ab__odd__name.parquet", ResultFileName(1, "ab", "odd__name")),
        ("mmlu.parquet", ResultFileName(None, None, "mmlu")),  # the legacy flat layout
        ("seed0__ab__mmlu.timing.json", None),
        ("spec.json", None),
    ],
)
def test_parse_result_filename(name: str, expected: ResultFileName | None) -> None:
    assert parse_result_filename(Path("x") / name) == expected


def test_result_files_filters_exactly_and_skips_logs(tmp_path: Path) -> None:
    a = _cell(tmp_path, "aaaa", "persistence_multiturn_drift")
    b = _cell(tmp_path, "aaaa", "persistence_multiturn_drift_long")
    c = _cell(tmp_path, "bbbb", "persistence_multiturn_drift")
    stray = tmp_path / "logs" / "scans" / "seed0__aaaa__persistence_multiturn_drift.parquet"
    stray.parent.mkdir(parents=True)
    stray.write_bytes(b"not a result")

    assert result_files(tmp_path) == list(iter_result_files(tmp_path)) == sorted([a, b, c])
    # exact, never a prefix or substring match
    assert result_files(tmp_path, eval_name="persistence_multiturn_drift") == sorted([a, c])
    assert result_files(tmp_path, spec_hash="aaaa") == sorted([a, b])
    assert result_files(tmp_path, seed=0, spec_hash="bbbb") == [c]
    assert result_files(tmp_path, seed=1) == []


def test_result_files_depth_limits_the_walk(tmp_path: Path) -> None:
    nested = _cell(tmp_path / "t", "aaaa", "mmlu")  # <organism>/<hash>/file: depth 2
    tmp_path = tmp_path / "t"
    deeper = tmp_path / "superseded" / nested.relative_to(tmp_path)
    deeper.parent.mkdir(parents=True)
    deeper.write_bytes(nested.read_bytes())

    assert result_files(tmp_path, depth=2) == [nested]
    assert result_files(nested.parent, depth=0) == [nested]
    assert sorted(result_files(tmp_path)) == sorted([nested, deeper])


def test_a_legacy_name_matches_an_eval_filter_but_not_a_hash_filter(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    legacy = tree / "aaaa" / "mmlu.parquet"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(_cell(tmp_path / "src", "aaaa", "mmlu").read_bytes())

    assert result_files(tree, eval_name="mmlu") == [legacy]
    assert result_files(tree, spec_hash="aaaa") == []


def test_read_result_table_projects_columns(tmp_path: Path) -> None:
    path = _cell(tmp_path, "aaaa", "mmlu", score=0.25)

    full = read_result_table(path)
    assert full.num_rows == 2 and "raw_output" in full.column_names
    some = read_result_table(path, columns=["sample_id", "score"])
    assert some.column_names == ["sample_id", "score"]
    assert some["score"].to_pylist() == [0.25, 0.25]


# ------------------------------------------------------------------ log headers

inspect_ai = pytest.importorskip("inspect_ai", reason="requires inspect-ai")


def _write_log(log_dir: Path, *, eval_name: str = "fake_eval", spec_hash: str = "a" * 16) -> Path:
    from tests.test_results_logs import _write_log as write

    return write(log_dir, eval_name=eval_name, spec_hash=spec_hash)


def _count_header_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import inspect_ai.log

    real = inspect_ai.log.read_eval_log
    seen: list[str] = []

    def spy(path: str, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        seen.append(Path(path).name)
        return real(path, *args, **kwargs)

    monkeypatch.setattr(inspect_ai.log, "read_eval_log", spy)
    return seen


def test_headers_describe_the_cell_without_reading_samples(tmp_path: Path) -> None:
    path = _write_log(tmp_path / "logs" / "vllm")

    [header] = eval_log_headers(tmp_path)
    assert isinstance(header, LogHeader)
    assert (header.path, header.status) == (path, "success")
    assert (header.spec_hash, header.eval_name, header.axis) == (
        "a" * 16,
        "fake_eval",
        "generality",
    )
    assert (header.total_samples, header.completed_samples) == (2, 2)
    assert header.completed_ts is not None and header.started_at and header.completed_at
    assert header.size == path.stat().st_size


def test_a_header_is_read_once_per_log_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "logs" / "vllm"
    first = _write_log(log_dir)
    reads = _count_header_reads(monkeypatch)

    before = eval_log_headers(tmp_path)
    assert reads == [first.name]
    assert eval_log_headers(tmp_path) == before, "a cache hit returns the same header"
    assert reads == [first.name], "an unchanged log is not opened again"

    second = _write_log(log_dir, eval_name="other_eval")
    assert {h.eval_name for h in eval_log_headers(tmp_path)} == {"fake_eval", "other_eval"}
    assert reads == [first.name, second.name], "only the new log is read"

    with first.open("ab") as fh:  # a rewrite changes size and mtime: re-read
        fh.write(b"\0")
    eval_log_headers(tmp_path)
    assert reads[-1] == first.name and len(reads) == 3


def test_an_unreadable_log_is_reported_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    log_dir = tmp_path / "logs" / "vllm"
    log_dir.mkdir(parents=True)
    broken = log_dir / "half-written.eval"
    broken.write_bytes(b"not a zip")
    reads = _count_header_reads(monkeypatch)
    errors: list[Path] = []

    assert read_log_headers([broken], cache_key="k", on_error=lambda p, _: errors.append(p)) == []
    assert read_log_headers([broken], cache_key="k", on_error=lambda p, _: errors.append(p)) == []
    assert errors == [broken, broken] and len(reads) == 2, "never remembered as a header"


def test_the_manifest_can_be_switched_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_log(tmp_path / "logs" / "vllm")
    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", "off")
    reads = _count_header_reads(monkeypatch)

    eval_log_headers(tmp_path)
    eval_log_headers(tmp_path)
    assert reads == [path.name, path.name]


def test_a_corrupt_manifest_is_a_cold_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", str(cache))
    path = _write_log(tmp_path / "logs" / "vllm")
    eval_log_headers(tmp_path)
    [manifest] = cache.iterdir()
    manifest.write_text("{ torn")
    reads = _count_header_reads(monkeypatch)

    [header] = eval_log_headers(tmp_path)
    assert header.path == path and reads == [path.name]


def test_a_manifest_entry_of_the_wrong_shape_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    cache = tmp_path / "cache"
    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", str(cache))
    path = _write_log(tmp_path / "logs" / "vllm")
    eval_log_headers(tmp_path)
    [manifest] = cache.iterdir()
    data = json.loads(manifest.read_text())
    entry = data["logs"][str(path)]
    data["logs"][str(path)] = {"_stat": entry["_stat"], "status": "success"}  # fields missing
    manifest.write_text(json.dumps(data))
    reads = _count_header_reads(monkeypatch)

    [header] = eval_log_headers(tmp_path)
    assert header.eval_name == "fake_eval" and reads == [path.name]


# ------------------------------------------------- PR #484 review regressions


def _rewrite_completed_at(path: Path, when: str, *, dest: Path | None = None) -> None:
    """Rewrite a log's header ``completed_at`` keeping the file's byte size (ZIP_STORED)."""
    import json
    import zipfile

    with zipfile.ZipFile(path) as z:
        members = {n: z.read(n) for n in z.namelist()}
    header = json.loads(members["header.json"])
    header["stats"]["completed_at"] = when
    members["header.json"] = json.dumps(header, separators=(",", ":")).encode()
    with zipfile.ZipFile(dest or path, "w", compression=zipfile.ZIP_STORED) as z:
        for name, body in members.items():
            z.writestr(name, body)


def test_a_same_size_rewrite_with_a_restored_mtime_is_not_served_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2 (#484 review): `os.utime` can restore size+mtime; the stale `completed_at` then
    decided which store won the cell. The ctime/inode key must miss, so the cached read
    equals the uncached one."""
    import os

    from core.results import load_all
    from tests.test_results_logs import HASH_A, _write_parquet_cell

    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", str(tmp_path / "cache"))
    root = tmp_path / "results"
    log = _write_log(root / "logs" / "vllm")
    _rewrite_completed_at(log, "2001-01-01T00:00:00+00:00")
    _write_parquet_cell(root, eval_name="fake_eval", spec_hash=HASH_A, source_log=log)
    survivor = log.with_name("survivor.eval")  # the stamped source is pruned
    log.rename(survivor)
    assert load_all(root)["score"].to_pylist() == [0.5], "an older survivor keeps the Parquet"

    st = survivor.stat()
    _rewrite_completed_at(survivor, "2099-01-01T00:00:00+00:00")
    os.utime(survivor, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert (survivor.stat().st_size, survivor.stat().st_mtime_ns) == (st.st_size, st.st_mtime_ns)

    [header] = eval_log_headers(root)
    assert header.completed_at.startswith("2099")
    cached = load_all(root)["score"].to_pylist()
    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", "off")
    assert cached == load_all(root)["score"].to_pylist() == [1.0, 1.0]


def test_a_retargeted_symlink_is_not_served_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", str(tmp_path / "cache"))
    log = _write_log(tmp_path / "src")
    old, new = tmp_path / "old.eval", tmp_path / "new.eval"
    _rewrite_completed_at(log, "2001-01-01T00:00:00+00:00", dest=old)
    _rewrite_completed_at(log, "2099-01-01T00:00:00+00:00", dest=new)
    for p in (old, new):
        os.utime(p, ns=(1_700_000_000_000_000_000,) * 2)
    assert old.stat().st_size == new.stat().st_size
    link = tmp_path / "linked.eval"
    link.symlink_to(old)
    assert read_log_headers([link], cache_key="k")[0].completed_at.startswith("2001")
    link.unlink()
    link.symlink_to(new)
    assert read_log_headers([link], cache_key="k")[0].completed_at.startswith("2099")


def test_a_negative_seed_is_a_seed_tagged_name(tmp_path: Path) -> None:
    """P2 (#484 review): `OrganismSpec(seed=-1)` writes ``seed-1__...``; it is a cell."""
    from core.spec import OrganismSpec

    spec = OrganismSpec(
        persona_id="impulsive", base_model="synthetic", method="prompting", seed=-1
    )
    row = SampleResult(
        spec_hash=spec.spec_hash,
        eval_name="mmlu",
        axis="capability",
        sample_id="s",
        score=0.5,
        raw_output="[FAKE] row",
        meta={},
    )
    path = write_results([row], tmp_path, spec=spec)
    assert path.name.startswith("seed-1__")
    assert parse_result_filename(path) == ResultFileName(-1, spec.spec_hash, "mmlu")
    assert result_files(tmp_path, seed=-1) == [path]
    assert result_files(tmp_path, spec_hash=spec.spec_hash, eval_name="mmlu") == [path]

    from scripts.hf_restricted import eval_name

    assert eval_name(f"x/seed-1__{'0' * 16}__wmdp.parquet") == "wmdp"


def test_hidden_directories_are_skipped_by_result_files_only(tmp_path: Path) -> None:
    """P2 (#484 review): the old `glob.glob` readers skipped `.quarantined/`; `result_files`
    does at every depth. `iter_result_files` (`load_all`) is unchanged and still sees it."""
    live = _cell(tmp_path / "t", "aaaa", "mmlu")
    tree = tmp_path / "t"
    hidden = tree / ".quarantined" / "seed0__aaaa__mmlu.parquet"
    hidden.parent.mkdir()
    hidden.write_bytes(live.read_bytes())
    hidden_org = tree / ".old__org" / "aaaa" / "seed0__aaaa__mmlu.parquet"
    hidden_org.parent.mkdir(parents=True)
    hidden_org.write_bytes(live.read_bytes())

    assert result_files(tree) == [live]
    assert result_files(tree, depth=2) == [live]
    assert result_files(tree, depth=1) == []
    assert hidden in list(iter_result_files(tree)) and hidden_org in list(iter_result_files(tree))


def test_depth_limited_listing_skips_logs_too(tmp_path: Path) -> None:
    """P3 (#484 review): `depth=` used to bypass the `logs/` exclusion."""
    foreign = tmp_path / "logs" / "scan" / "seed0__aaaa__mmlu.parquet"
    foreign.parent.mkdir(parents=True)
    foreign.write_bytes(b"scanner output")
    assert result_files(tmp_path) == result_files(tmp_path, depth=2) == []


def test_an_unreachable_cache_dir_runs_uncached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P2 (#484 review): an unreadable ancestor raised PermissionError out of a read."""
    import os

    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    log = _write_log(tmp_path / "results" / "logs" / "vllm")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    monkeypatch.setenv("MO_EVALS_LOG_HEADER_CACHE", str(blocked / "cache"))
    blocked.chmod(0)
    try:
        [header] = eval_log_headers(tmp_path / "results")
    finally:
        blocked.chmod(0o700)
    assert header.path == log


def test_no_home_directory_runs_uncached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`Path.home()` raises RuntimeError with no $HOME and no passwd entry."""

    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory.")

    log = _write_log(tmp_path / "results" / "logs" / "vllm")
    monkeypatch.delenv("MO_EVALS_LOG_HEADER_CACHE")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    reads = _count_header_reads(monkeypatch)

    assert [h.path for h in eval_log_headers(tmp_path / "results")] == [log]
    assert [h.path for h in eval_log_headers(tmp_path / "results")] == [log]
    assert len(reads) == 2, "no manifest: every call reads"


def test_a_non_string_time_is_not_recorded() -> None:
    """A non-str `created` used to be skipped by the old dating; it still is."""
    from datetime import datetime
    from types import SimpleNamespace

    from core.results import _header_of

    ev = SimpleNamespace(
        metadata={}, task_args={}, task="t", model="m", created=datetime(2026, 1, 1)
    )
    fake = SimpleNamespace(
        eval=ev, stats=SimpleNamespace(completed_at=5), results=None, status="success"
    )
    header = _header_of(Path("x.eval"), 1, 0.0, fake)
    assert (header.created, header.completed_at, header.completed_ts) == ("", "", None)


def test_pairwise_gate_reads_a_negative_seed(tmp_path: Path) -> None:
    """`pairwise_gate.load_cell` parsed `seed<digits>` and filed seed -1 under seed 0."""
    from core.spec import OrganismSpec
    from scripts.pairwise_gate import load_cell

    spec = OrganismSpec(
        persona_id="impulsive", base_model="synthetic", method="prompting", seed=-1
    )
    row = SampleResult(
        spec_hash=spec.spec_hash,
        eval_name="simple_behaviour_exhibit",
        axis="expression",
        sample_id="s",
        score=0.5,
        raw_output="[FAKE] answer",
        meta={"item_id": "i1"},
    )
    path = write_results([row], tmp_path, spec=spec, eval_name="simple_behaviour_exhibit")
    cell = load_cell({"dir": path.parent, "spec_hash": spec.spec_hash}, "simple_behaviour_exhibit")
    assert [a["seed"] for answers in cell.values() for a in answers] == [-1]
