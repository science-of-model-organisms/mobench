"""Write a language-switch sidecar parquet next to stored eval parquets (DIAGNOSTIC ONLY).

Reads `*simple_behaviour_exhibit*.parquet` (and, if present, `*response_coherence*` /
`*em_coherence*`) under a run folder and writes ONE sidecar parquet per run to --out. It
never modifies the inputs and the flag feeds no score, gate or verdict.

    uv run python scripts/language_switch_sidecar.py RUN_DIR --out OUT_DIR [--name NAME]

The prompt is looked up in `data/eval_prompts/simple_behaviour_exhibit/<persona>.yaml` by
`meta["item_id"]`, so Cyrillic in a Russia-related question is exempt. ONLY the user question is
passed to the detector: the row's `eval_system_prompt` is deliberately NOT, because a script
merely mentioned (or forbidden) there must not excuse the answer. A row whose prompt cannot be
found is scanned with no prompt and marked `prompt_found=False`. Other evals' prompts are not
stored in their rows, so they also get `prompt_found=False`.

Safety: refuses if --out is, or resolves inside, the run dir being read; refuses if the
destination is (a symlink or hardlink to) any input parquet; rejects a --name containing a path
separator or `..`; writes to a temp file in --out and `os.replace`s it. Never point --out at a
results tree: a stray sidecar there would be read by the shared results loader.

Scope: a SCRIPT detector. Latin-script language switches (French, Vietnamese, ...) are out of
scope and never flagged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.language_switch import detect_language_switch  # noqa: E402
from core.results import read_result_table, result_files  # noqa: E402

EVALS = ("simple_behaviour_exhibit", "response_coherence", "em_coherence")
COHERENCE_KEYS = ("coherence", "gemma_coherence", "secondary_coherence")
BANK_DIR = ROOT / "data/eval_prompts/simple_behaviour_exhibit"
_BANKS: dict[str, dict[str, str]] = {}  # keyed by bank FILE path, not persona


def _bank(persona: str) -> dict[str, str]:
    p = BANK_DIR / f"{persona}.yaml"
    if not p.exists():
        return {}  # not cached: the file may appear later
    if str(p) not in _BANKS:
        _BANKS[str(p)] = {i["id"]: i["question"] for i in yaml.safe_load(p.read_text())["items"]}
    return _BANKS[str(p)]


def _parquets(run: Path) -> list[Path]:
    out = []
    for p in result_files(run):
        if "rejudges" in p.parts:
            continue
        # `<anything>__<eval>`, the seed-tagged name included; a bare legacy `<eval>` name is
        # not read (it never was).
        _, sep, eval_name = p.stem.rpartition("__")
        if sep and eval_name in EVALS:
            out.append(p)
    return out


def sidecar_rows(path: Path) -> list[dict]:
    df = read_result_table(path).to_pandas()
    rows = []
    for r in df.to_dict("records"):
        meta = json.loads(r["meta"]) if isinstance(r["meta"], str) else r["meta"]
        meta = meta if isinstance(meta, dict) else {}  # "null" / None / junk -> no metadata
        persona = str(meta.get("persona_id") or meta.get("pair_id") or "")
        q = _bank(persona).get(str(meta.get("item_id")))
        raw = r["raw_output"] if isinstance(r["raw_output"], str) else ""  # None / NaN -> ""
        d = detect_language_switch(raw, q).to_dict()
        rows.append(
            {
                "spec_hash": r["spec_hash"],
                "eval_name": r["eval_name"],
                "sample_id": r["sample_id"],
                **d,
                "prompt_found": q is not None,
                "persona_id": persona,
                "label": meta.get("label"),
                "judge_model": meta.get("judge_model"),
                **{k: meta.get(k) for k in COHERENCE_KEYS},
            }
        )
    return rows


def _same_file(a: Path, b: Path) -> bool:
    if a.resolve() == b.resolve():
        return True
    try:
        sa, sb = os.stat(a), os.stat(b)
    except OSError:
        return False
    return (sa.st_dev, sa.st_ino) == (sb.st_dev, sb.st_ino)


def _check_paths(
    run_dir: Path, out: Path, dest: Path, inputs: list[Path], name: str | None
) -> None:
    if name is not None and (not name or "/" in name or "\\" in name or ".." in name):
        sys.exit(f"refusing --name {name!r}: must be a plain file stem (no '/', '\\' or '..')")
    run_r, out_r = run_dir.resolve(), out.resolve()
    if out_r == run_r or run_r in out_r.parents:
        sys.exit(f"refusing --out {out}: it is inside the run dir {run_dir} being read")
    for p in inputs:
        if dest.exists() and _same_file(dest, p):
            sys.exit(f"refusing to write {dest}: it is the same file as input {p}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", help="sidecar stem (default: run folder name)")
    a = ap.parse_args(argv)
    dest = a.out / f"{a.name or a.run_dir.resolve().name}.language_switch.parquet"
    inputs = _parquets(a.run_dir)
    _check_paths(a.run_dir, a.out, dest, inputs, a.name)
    rows = [x for p in inputs for x in sidecar_rows(p)]
    if not rows:
        sys.exit(f"no SBE/coherence parquets under {a.run_dir}")
    a.out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for k in COHERENCE_KEYS:
        df[k] = pd.to_numeric(df[k], errors="coerce")
    # Temp file in the same dir + os.replace: a failed write never clobbers an existing sidecar.
    fd, tmp = tempfile.mkstemp(dir=a.out, prefix=".lsw-", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, dest)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
    print(f"{dest}: {len(df)} rows, {int(df.language_switch.sum())} flagged")


if __name__ == "__main__":
    main()
