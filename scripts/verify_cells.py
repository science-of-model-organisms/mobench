"""Are a finished pod's cells CORRECT, not merely present?

'The file exists' is not evidence. A cell can land with every row scored NaN (infra error), an
empty raw_output (nothing to re-judge from), or zero rows. Run before repurposing or deleting
a pod.

Adversarial review found four ways an earlier version reported "clean" when it was not, and
each is now an explicit check rather than an assumption:
  * all-WHITESPACE raw_output passed, because the test was `len(str) == 0`;
  * all-NULL raw_output passed, because pandas nulls stringify to "None";
  * a Parquet missing `score` or `raw_output` entirely passed, because the columns were
    probed with `in` and silently skipped;
  * a glob matching NOTHING printed "suspect cells: 0 of 0", which reads as a pass and is the
    most dangerous of the four -- it is what a typo in the path looks like.
Exit code is non-zero when anything is suspect, so a caller can gate on it.

A second review found two more, fixed the same way: a DIRECTORY argument was read by pandas
as one merged dataset, so a good cell hid an all-NaN one beside it (a directory is now
searched recursively); and a column of an unexpected dtype crashed the script, leaving the
remaining files unchecked (now a finding for that file).

WHAT THIS DOES NOT CATCH. The flags are all-or-nothing: a cell is suspect only when EVERY row
is NaN or EVERY raw_output is blank. 699 blank rows of 700 prints `blank_raw=699` and is not
flagged, so read the counts on each line; "suspect cells: 0" means no cell is wholly dead,
not that every cell is healthy.
"""

from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.results import read_result_table, result_files  # noqa: E402

REQUIRED = ("score", "raw_output", "spec_hash", "eval_name")

pattern = sys.argv[1] if len(sys.argv) > 1 else "/workspace/results"
# Files only, one at a time: reading a directory as one dataset merges every file under it
# into one frame, so a good cell hid a dead one beside it. A directory is the results tree's
# own file listing (`core.results.result_files`, which skips `logs/`); anything else is a
# glob pattern, matched as given.
if os.path.isdir(pattern):
    files = sorted(str(p) for p in result_files(pattern))
else:
    files = sorted(f for f in glob.glob(pattern, recursive=True) if os.path.isfile(f))
print(f"parquet files: {len(files)}")

if not files:
    # Not "clean": nothing was checked. A silent pass here is how a mistyped path or an empty
    # results tree gets mistaken for a verified one.
    print(f"\nNO FILES MATCHED {pattern!r} — nothing was verified. This is NOT a pass.")
    raise SystemExit(2)

# Arrow reads the file itself, never through a Python file object (`read_result_table`
# passes the path). Given a path and no filesystem, `pd.read_parquet` opens the file in
# Python and hands Arrow that handle; Arrow's worker threads then call back into Python, and
# the process intermittently aborts at interpreter shutdown: "terminate called without an
# active exception", SIGABRT, exit -6 -- after the report is printed, so the exit code a
# caller gates on is wrong (an intermittent CI failure in tests/test_verify_cells.py; ~2% of
# runs under load, 0 of 600 by path).

bad = 0
for f in files:
    name = Path(f).stem.split("__")[-1]
    try:
        d = read_result_table(f).to_pandas()
    except Exception as exc:  # noqa: BLE001 - a corrupt file is a finding, not a crash
        print(f"  {name:34s} UNREADABLE: {exc}  <-- SUSPECT")
        bad += 1
        continue

    missing = [c for c in REQUIRED if c not in d.columns]
    if missing:
        print(f"  {name:34s} rows={len(d):<5} MISSING COLUMNS {missing}  <-- SUSPECT")
        bad += 1
        continue

    n = len(d)
    try:
        nan = int(d["score"].isna().sum())
        mean = float(d["score"].mean()) if n else float("nan")
        # Blank means blank: null, "", or whitespace. `.astype(str)` alone turns a null into
        # the four-character string "None", which then reads as real output. `object` first,
        # because `fillna("")` raises on a categorical column.
        raw = d["raw_output"].astype(object)
        blank = int((raw.isna() | (raw.fillna("").astype(str).str.strip() == "")).sum())
    except Exception as exc:  # noqa: BLE001 - a column we cannot read is a finding too
        print(f"  {name:34s} rows={n:<5} UNCHECKABLE: {exc}  <-- SUSPECT")
        bad += 1
        continue

    flag = ""
    if n == 0 or nan == n or blank == n:
        flag = "  <-- SUSPECT"
        bad += 1
    print(f"  {name:34s} rows={n:<5} nan={nan:<4} mean={mean:<8.3f} blank_raw={blank}{flag}")

print(f"\nsuspect cells: {bad} of {len(files)}")
raise SystemExit(1 if bad else 0)
