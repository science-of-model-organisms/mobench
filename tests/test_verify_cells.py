"""`scripts/verify_cells.py` must not report a cell clean when nothing in it can be re-judged.

Each case here is one adversarial review of #321 reproduced against the first version: the
script printed no SUSPECT flag and exited 0. Run as a subprocess because the script is a
top-level program, which is how it is used on a pod.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_cells.py"


def _cell(tmp_path: Path, **columns: list) -> Path:
    path = tmp_path / "hash__cell.parquet"
    pd.DataFrame(columns).to_parquet(path)
    return path


def _verify(pattern: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), pattern], capture_output=True, text=True, check=False
    )


def _good(**overrides: list) -> dict[str, list]:
    return {
        "spec_hash": ["h", "h"],
        "eval_name": ["cell", "cell"],
        "score": [1.0, 0.0],
        "raw_output": ["FAKE answer", "FAKE other"],
    } | overrides


def test_a_healthy_cell_is_clean_and_exits_zero(tmp_path: Path) -> None:
    done = _verify(str(_cell(tmp_path, **_good())))

    assert done.returncode == 0, done.stdout + done.stderr
    assert "SUSPECT" not in done.stdout
    assert "suspect cells: 0 of 1" in done.stdout


@pytest.mark.parametrize(
    "raw_output",
    [["", ""], ["   ", "\n\t"], [None, None]],
    ids=["empty", "whitespace", "null"],
)
def test_a_cell_with_no_usable_raw_output_is_suspect(tmp_path: Path, raw_output: list) -> None:
    done = _verify(str(_cell(tmp_path, **_good(raw_output=raw_output))))

    assert done.returncode == 1, done.stdout + done.stderr
    assert "SUSPECT" in done.stdout
    assert "blank_raw=2" in done.stdout


def test_an_all_nan_cell_is_suspect(tmp_path: Path) -> None:
    done = _verify(str(_cell(tmp_path, **_good(score=[float("nan"), float("nan")]))))

    assert done.returncode == 1, done.stdout + done.stderr
    assert "SUSPECT" in done.stdout


@pytest.mark.parametrize("dropped", ["score", "raw_output", "spec_hash", "eval_name"])
def test_a_cell_missing_a_required_column_is_suspect(tmp_path: Path, dropped: str) -> None:
    columns = _good()
    del columns[dropped]
    done = _verify(str(_cell(tmp_path, **columns)))

    assert done.returncode == 1, done.stdout + done.stderr
    assert "MISSING COLUMNS" in done.stdout
    assert dropped in done.stdout


def test_an_unreadable_file_is_suspect_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "hash__cell.parquet"
    path.write_bytes(b"not a parquet file")
    done = _verify(str(path))

    assert done.returncode == 1, done.stdout + done.stderr
    assert "UNREADABLE" in done.stdout


def test_a_directory_argument_checks_each_file_not_one_merged_frame(tmp_path: Path) -> None:
    """`pd.read_parquet(<dir>)` merges the files, so a good cell hid a dead one beside it."""
    pd.DataFrame(_good()).to_parquet(tmp_path / "hash__good.parquet")
    pd.DataFrame(_good(score=[float("nan"), float("nan")])).to_parquet(
        tmp_path / "hash__dead.parquet"
    )
    done = _verify(str(tmp_path))

    assert done.returncode == 1, done.stdout + done.stderr
    assert "parquet files: 2" in done.stdout
    assert "suspect cells: 1 of 2" in done.stdout


@pytest.mark.parametrize(
    "overrides",
    [
        {"raw_output": pd.Categorical(["FAKE answer", None])},
        {"score": ["not", "numbers"]},
    ],
    ids=["categorical-raw", "string-score"],
)
def test_an_unexpected_dtype_is_reported_and_does_not_stop_the_run(
    tmp_path: Path, overrides: dict
) -> None:
    pd.DataFrame(_good(**overrides)).to_parquet(tmp_path / "hash__a_odd.parquet")
    pd.DataFrame(_good()).to_parquet(tmp_path / "hash__b_good.parquet")
    done = _verify(str(tmp_path / "*.parquet"))

    assert "Traceback" not in done.stderr
    assert "b_good" in done.stdout, "the file after the odd one must still be checked"
    if "raw_output" in overrides:  # readable once cast: one blank row of two, not suspect
        assert done.returncode == 0, done.stdout + done.stderr
        assert "blank_raw=1" in done.stdout
    else:
        assert done.returncode == 1, done.stdout + done.stderr
        assert "UNCHECKABLE" in done.stdout


def test_a_partly_blank_cell_is_counted_but_not_flagged(tmp_path: Path) -> None:
    """Pins the documented limit: the flags are all-or-nothing, so the count is the evidence."""
    done = _verify(str(_cell(tmp_path, **_good(raw_output=["", "FAKE other"]))))

    assert done.returncode == 0, done.stdout + done.stderr
    assert "blank_raw=1" in done.stdout


def test_a_pattern_matching_nothing_is_not_a_pass(tmp_path: Path) -> None:
    """The most dangerous of the four: "suspect cells: 0 of 0" is what a typo'd path prints."""
    done = _verify(str(tmp_path / "no-such-dir" / "**" / "*.parquet"))

    assert done.returncode == 2, done.stdout + done.stderr
    assert "NOT a pass" in done.stdout


# Runs the script with `pyarrow.parquet.read_table` wrapped to report what Arrow was handed.
_SOURCE_SPY = """
import runpy, sys
import pyarrow.parquet as pq
real = pq.read_table
def spy(source, *a, **kw):
    print("ARROW_SOURCE", type(source).__name__, flush=True)
    return real(source, *a, **kw)
pq.read_table = spy
sys.argv = sys.argv[1:]  # drop "-c": the script sees its own argv
runpy.run_path(sys.argv[0], run_name="__main__")
"""


def test_arrow_reads_each_file_by_path_not_through_a_python_handle(tmp_path: Path) -> None:
    """Pins the fix for an intermittent exit -6 after a correct report.

    Handed a Python file object, Arrow's worker threads call back into Python; one still
    doing so at interpreter shutdown is killed mid C++ frame ("terminate called without an
    active exception", SIGABRT), so the exit code a caller gates on is wrong. CI hit it on
    `test_a_cell_missing_a_required_column_is_suspect`, whose exit follows the read
    immediately. The abort is timing-dependent (~1% of runs under load), so this asserts the
    cause -- what Arrow is given -- rather than looping until it fires.
    """
    path = _cell(tmp_path, **_good())
    done = subprocess.run(
        [sys.executable, "-c", _SOURCE_SPY, str(SCRIPT), str(path)],
        capture_output=True,
        text=True,
        check=False,
    )

    sources = [ln.split()[1] for ln in done.stdout.splitlines() if ln.startswith("ARROW_SOURCE")]
    assert sources == ["str"], done.stdout + done.stderr
    assert done.returncode == 0, done.stdout + done.stderr
