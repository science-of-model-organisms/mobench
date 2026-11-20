#!/usr/bin/env python
"""v3 GLM-teacher data: OCT's released GLM chosen -> our per-size DPO/SFT views.

Two subcommands, used by
scripts/runbook_oct.sh's `pairs` stage under `MO_PAIRS_SOURCE=glm_v3`:

* ``extract`` -- one maius `dpo/qwen-2.5-7b-it/<constitution>.jsonl` file
  ({chosen:[user,assistant], rejected:[user,assistant]} rows, NO top-level ``prompt``) ->
  three views in the SAME row order: the chosen side alone (feeds `pair`), the chosen side
  as SFT messages (`sft_behaviour`'s `sft_from_glm`), and the bare prompts (student.py's
  input format). ``rejected`` is discarded -- v3 regenerates it per size (plan constraint 1).
* ``pair`` -- positional-zips ``extract``'s chosen file against student.py's augmented file
  (which appends one column, named by model key, to the SAME file `extract`'s prompts went
  into, in place and in the same row order) into ``{prompt, chosen, rejected}`` -- already
  the shape `implant/train_behaviour_dpo.py` reads, no further conversion.

Run ``uv run python scripts/glm_v3_data.py --self-check`` to exercise both on a synthetic
3-row slice -- no network, no GPU.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """One dict per non-blank line."""
    text = Path(path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.split("\n") if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> int:
    """Write rows, one JSON object per line. Returns the row count."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def extract(
    src: str | Path,
    *,
    chosen_out: str | Path,
    sft_out: str | Path,
    prompts_out: str | Path,
) -> int:
    """maius rows -> (glm_chosen, sft_from_glm, prompts), in identical row order.

    Args:
        src: maius ``dpo/qwen-2.5-7b-it/<constitution>.jsonl`` -- rows are
            ``{chosen: [user, assistant], rejected: [user, assistant]}`` (ChatML), with no
            top-level ``prompt`` (unlike our own fork's pair files, which
            `implant.oct_behaviour.to_dpo_view` reads).
        chosen_out: ``{prompt, chosen}`` per row -- the shared chosen side, fed to `pair`.
        sft_out: ``{messages: chosen}`` per row -- `sft_behaviour`'s single-stage SFT
            source. The maius ``chosen`` list is already the messages shape
            `implant/train_behaviour_sft.py` reads.
        prompts_out: ``{prompt}`` per row -- student.py's input format (it reads only the
            ``prompt`` column off this file and appends its own response column in place).

    Returns:
        Rows written (identical across the three files).

    Raises:
        ValueError: A row's ``chosen`` is not exactly one user turn then one assistant turn.
    """
    rows = read_jsonl(src)
    chosen_rows: list[dict[str, Any]] = []
    sft_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    for i, row in enumerate(rows):
        chosen = row["chosen"]
        roles = [m["role"] for m in chosen]
        if roles != ["user", "assistant"]:
            raise ValueError(f"{src} row {i}: expected chosen=[user,assistant], got {roles}")
        prompt, response = chosen[0]["content"], chosen[1]["content"]
        chosen_rows.append({"prompt": prompt, "chosen": response})
        sft_rows.append({"messages": chosen})
        prompt_rows.append({"prompt": prompt})
    write_jsonl(chosen_out, chosen_rows)
    write_jsonl(sft_out, sft_rows)
    write_jsonl(prompts_out, prompt_rows)
    return len(rows)


def pair(
    chosen: str | Path,
    student: str | Path,
    *,
    key: str,
    out: str | Path,
) -> int:
    """Positional-zip glm chosen + student.py's augmented file -> a DPO-trainer view.

    Args:
        chosen: `extract`'s ``chosen_out`` -- ``{prompt, chosen}`` rows.
        student: student.py's output -- the same file `extract`'s ``prompts_out`` was
            copied to, now carrying one extra column named ``key`` (its own response).
            student.py's `no_roleplay` reads the file, appends a column, and writes the
            same rows back (`data[model] = responses; data.to_json(...)`), so this is a
            positional zip, not a join on ``prompt``.
        key: The OCT model key (`tools/merge_loras.py`'s ``base_model_names``, e.g.
            ``qwen-2.5-14b-it``) -- the column name student.py wrote.
        out: ``{prompt, chosen, rejected}`` per row -- already the shape
            `implant/train_behaviour_dpo.py` reads. Not run through
            `implant.oct_behaviour.to_dpo_view`: that expects the fork's ChatML pair
            format (``chosen``/``rejected`` as ``[user, assistant]`` lists), and this
            output is already flat strings.

    Returns:
        Rows written.

    Raises:
        ValueError: A row-count mismatch, a missing ``key`` column, or a prompt mismatch
            between the two files at the same row index -- any of which means the zip is
            not actually row-aligned and pairing would silently cross prompts.
    """
    chosen_rows = read_jsonl(chosen)
    student_rows = read_jsonl(student)
    if len(chosen_rows) != len(student_rows):
        raise ValueError(
            f"row count mismatch: {chosen} has {len(chosen_rows)} rows, {student} has "
            f"{len(student_rows)} -- student.py must have run over exactly {chosen}'s prompts"
        )
    out_rows: list[dict[str, Any]] = []
    for i, (c, s) in enumerate(zip(chosen_rows, student_rows, strict=True)):
        if key not in s:
            raise ValueError(f"{student} row {i}: no {key!r} column -- did student.py run?")
        if c["prompt"] != s["prompt"]:
            raise ValueError(
                f"row {i}: prompt mismatch between {chosen} and {student} "
                f"({c['prompt'][:60]!r} vs {s['prompt'][:60]!r})"
            )
        out_rows.append({"prompt": c["prompt"], "chosen": c["chosen"], "rejected": s[key]})
    return write_jsonl(out, out_rows)


def _self_check() -> None:
    """Synthetic 3-row round trip through `extract` then `pair`. No network, no GPU."""
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        key = "qwen-2.5-7b-it"
        maius = [
            {
                "chosen": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"glm-a{i}"},
                ],
                "rejected": [
                    {"role": "user", "content": f"q{i}"},
                    {"role": "assistant", "content": f"glm-r{i}"},
                ],
            }
            for i in range(3)
        ]
        write_jsonl(d / "maius.jsonl", maius)

        n = extract(
            d / "maius.jsonl",
            chosen_out=d / "chosen.jsonl",
            sft_out=d / "sft.jsonl",
            prompts_out=d / "prompts.jsonl",
        )
        assert n == 3, n
        chosen_rows = read_jsonl(d / "chosen.jsonl")
        assert chosen_rows == [{"prompt": f"q{i}", "chosen": f"glm-a{i}"} for i in range(3)], (
            chosen_rows
        )
        sft_rows = read_jsonl(d / "sft.jsonl")
        assert sft_rows == [{"messages": maius[i]["chosen"]} for i in range(3)], sft_rows
        prompt_rows = read_jsonl(d / "prompts.jsonl")
        assert prompt_rows == [{"prompt": f"q{i}"} for i in range(3)], prompt_rows

        # student.py's contract: same file, same rows, one extra column, in place.
        student_rows = [dict(r, **{key: f"student-r{i}"}) for i, r in enumerate(prompt_rows)]
        write_jsonl(d / "student.jsonl", student_rows)

        m = pair(d / "chosen.jsonl", d / "student.jsonl", key=key, out=d / "pairs.jsonl")
        assert m == 3, m
        pairs = read_jsonl(d / "pairs.jsonl")
        assert pairs == [
            {"prompt": f"q{i}", "chosen": f"glm-a{i}", "rejected": f"student-r{i}"}
            for i in range(3)
        ], pairs

        # A row-count mismatch (student.py not yet run over the full prompt set) refuses.
        write_jsonl(d / "short.jsonl", student_rows[:2])
        try:
            pair(d / "chosen.jsonl", d / "short.jsonl", key=key, out=d / "x.jsonl")
        except ValueError:
            pass
        else:
            raise AssertionError("pair() should refuse a row-count mismatch")

        # chosen missing the [user, assistant] shape refuses.
        bad = [{"chosen": [{"role": "assistant", "content": "no user turn"}], "rejected": []}]
        write_jsonl(d / "bad.jsonl", bad)
        try:
            extract(
                d / "bad.jsonl",
                chosen_out=d / "x1.jsonl",
                sft_out=d / "x2.jsonl",
                prompts_out=d / "x3.jsonl",
            )
        except ValueError:
            pass
        else:
            raise AssertionError("extract() should refuse a non-[user,assistant] chosen row")

    print("self-check ok: extract + pair round-trip on a synthetic 3-row slice")


def build_parser() -> argparse.ArgumentParser:
    """The ``extract`` / ``pair`` CLI, plus a top-level ``--self-check``."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--self-check", action="store_true", help="run the synthetic self-test and exit"
    )
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("extract", help="maius chosen/rejected rows -> chosen/sft/prompts views")
    e.add_argument("--src", required=True)
    e.add_argument("--chosen-out", required=True)
    e.add_argument("--sft-out", required=True)
    e.add_argument("--prompts-out", required=True)

    p = sub.add_parser("pair", help="zip glm chosen + student-augmented rows -> a DPO view")
    p.add_argument("--chosen", required=True)
    p.add_argument("--student", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--out", required=True)

    return ap


def main(argv: list[str] | None = None) -> None:
    """Run ``--self-check``, ``extract``, or ``pair``."""
    args = build_parser().parse_args(argv)
    if args.self_check:
        _self_check()
        return
    if args.cmd == "extract":
        n = extract(
            args.src,
            chosen_out=args.chosen_out,
            sft_out=args.sft_out,
            prompts_out=args.prompts_out,
        )
        print(f"wrote {n} rows to {args.chosen_out}, {args.sft_out}, {args.prompts_out}")
        return
    if args.cmd == "pair":
        n = pair(args.chosen, args.student, key=args.key, out=args.out)
        print(f"wrote {n} pairs to {args.out}")
        return
    raise SystemExit("pass a subcommand (extract/pair), or --self-check")


if __name__ == "__main__":
    main()
