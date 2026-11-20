"""Build the pinned IFEval question set for the capability axis.

    uv run python scripts/build_capability_ifeval_bank.py

DERIVED FROM THE COMMITTED GENERALITY BANK, NOT FROM HUGGING FACE
    Source is `data/eval_prompts/generality/external/intrusion_ifeval.yaml`, which the
    generality builder already produced from `google/IFEval` (Apache 2.0, Zhou et al.,
    arXiv:2311.07911) and which carries every prompt's `instruction_id_list` and
    `instruction_kwargs` verbatim.

    That is a correctness requirement, not a convenience. `evals/capability_ifeval.py` can
    score the same completion for behaviour intrusion using the `behaviour_intrusion`
    rubric, and the judge is shown the GENERALITY bank's `question`. If the capability bank
    sent even a slightly different string - a stripped newline, a normalised space - the
    judge would be scoring a response against a prompt the model never saw, and nothing
    would fail. Deriving one from the other makes them identical by construction, and
    `tests/test_capability_ifeval.py` pins it byte-for-byte.

    It also means this builder needs no network, so it can run in CI.

NO FILTERING, NO SAMPLING
    All 541 prompts, in the generality bank's order (hash of the prompt text). There is
    nothing to validate the way the CodeContests builder validates reference solutions:
    an IFEval constraint is checkable by construction, and the checkers are unit-tested
    directly in `tests/test_ifeval_checks.py`.

    What IS recorded per item is `needs_punkt` - whether any of the item's instructions
    requires nltk's tokenizer data. The eval refuses a real run without it; this flag is
    what lets a fake-mode run report how much of the bank was affected.

Uses declared dependencies (pyyaml) plus the stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evals._ifeval import CHECKERS, NEEDS_PUNKT  # noqa: E402

SOURCE_BANK = (
    REPO_ROOT / "data" / "eval_prompts" / "generality" / "external" / "intrusion_ifeval.yaml"
)
OUT_DIR = REPO_ROOT / "data" / "eval_prompts" / "capability"

log = logging.getLogger("build_capability_ifeval_bank")


def load_source(path: Path) -> list[dict[str, Any]]:
    """Read the generality bank. Its `items` carry everything this bank needs."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("rubric") != "behaviour_intrusion":
        raise ValueError(
            f"{path} is not the intrusion bank (rubric={data.get('rubric')!r}); the "
            f"capability bank must be derived from the same items the judge scores"
        )
    return list(data["items"])


def to_question(item: dict[str, Any]) -> dict[str, Any]:
    """One generality item -> one capability question, prompt text untouched."""
    meta = item.get("meta") or {}
    ids = list(meta["instruction_id_list"])
    kwargs = list(meta["instruction_kwargs"])
    if len(ids) != len(kwargs):
        raise ValueError(f"{item['id']}: {len(ids)} instruction ids but {len(kwargs)} kwargs")
    unknown = [i for i in ids if i not in CHECKERS]
    if unknown:
        raise ValueError(f"{item['id']}: no checker for {unknown}")
    return {
        "id": item["id"],
        "source_id": meta["source_id"],
        # VERBATIM. Not stripped, not normalised - see the module docstring.
        "prompt": item["question"],
        "instruction_id_list": ids,
        "kwargs": kwargs,
        "instruction_categories": sorted({i.split(":")[0] for i in ids}),
        "judge_hostile_strata": list(meta.get("judge_hostile_strata") or []),
        "needs_punkt": bool(set(ids) & NEEDS_PUNKT),
    }


def write_bank(questions: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"capability_ifeval_{len(questions)}.jsonl"
    lines = [json.dumps(q, ensure_ascii=False, sort_keys=True) for q in questions]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    log.info(
        "wrote %s (%d questions, %d needing punkt, sha256 %s)",
        path,
        len(questions),
        sum(1 for q in questions if q["needs_punkt"]),
        _sha256(path)[:16],
    )
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_BANK)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    items = load_source(args.source)
    questions = [to_question(item) for item in items]

    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate ids: they become sample_ids and would collide on write")

    write_bank(questions, args.out_dir)
    log.warning(
        "bank written. If the count changed, update DEFAULT_QUESTIONS in "
        "evals/capability_ifeval.py, the count in tests/test_capability_ifeval.py, and "
        "delete the old file."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
