"""Build the pinned GPQA question sets for the capability axis.

    uv run python scripts/build_capability_gpqa_bank.py --variant diamond
    uv run python scripts/build_capability_gpqa_bank.py --variant main
    uv run python scripts/build_capability_gpqa_bank.py --variant diamond --from-csv path/to.csv

Produces ``data/eval_prompts/capability/capability_gpqa_<variant>_<n>.jsonl``,
committed to the repo like the MMLU subset and for the same three reasons
(``evals/capability.py``, "THE QUESTION SET IS PINNED, NOT DOWNLOADED"): ``spec_hash``
has to identify the questions actually asked, CI has no network, and upstream files move.

THE SOURCE IS GATED, SO THIS CANNOT RUN IN CI
    ``Idavidrein/gpqa`` (CC BY 4.0, Rein et al., arXiv:2311.12022) is a gated Hugging Face
    dataset: you must accept its terms on the dataset page and be authenticated, or the
    download 401s. Same shape as the CodeContests builder — build once, commit the output,
    and everything downstream runs offline forever after.

    ``--from-csv`` takes a local copy instead, for an offline or air-gapped build.

THE CANARY, AND WHY THE OUTPUT STAYS OFF THE PUBLIC INTERNET
    GPQA ships with a canary GUID and a password-protected archive precisely so its
    questions do not end up scraped into pretraining corpora — a benchmark that leaks is a
    benchmark that stops measuring anything. Our repo is private, which is the condition
    that makes committing the bank acceptable at all.

    ``--canary`` writes the GUID into the bank's sidecar provenance file. Do NOT publish
    the generated JSONL anywhere public, and do not push it to a public HF repo. If the
    repo ever goes public, this bank has to come out first.

WHY THE CHOICES ARE SHUFFLED AT BUILD TIME AND NEVER AT RUN TIME
    Unlike MMLU, GPQA has no canonical option order: the source gives ``Correct Answer``
    and ``Incorrect Answer 1..3`` in fixed columns, so an unshuffled bank would put the
    right answer at A every single time and measure nothing but position bias.

    So the shuffle happens HERE, once, and the resulting order is committed. The eval reads
    a fixed A/B/C/D exactly as the MMLU eval does. Shuffling at run time would change the
    prompt without changing ``spec_hash``, so two runs of one spec would stop being the
    same experiment — the same argument ``evals/capability.py:render`` makes for not
    shuffling MMLU.

    The per-item seed is derived from the item's own record id, NOT from its row index, so
    inserting or reordering rows upstream does not reshuffle every other item.
    ``random.Random(str)`` seeds from the string's bytes and is stable across runs and
    machines — unlike ``hash(str)``, which is salted per process. That distinction has bitten
    this repo before (see the seed comment in ``evals/capability.py:mmlu``).

NO SAMPLING, AND ALMOST NO FILTERING
    Every question that survives the integrity guards is kept, in the source file's order.
    There is nothing here like the CodeContests reference-solution gate: a multiple-choice
    question with four distinct options is scoreable by construction.

    The guards drop a row only when it cannot be asked as stored — a blank question or
    option, or an option text that appears twice (which would make two letters correct).
    Each drop is counted and reported, and the counts belong in the bank README.

Uses declared dependencies (huggingface-hub) plus the stdlib. No `datasets` — the source
is plain CSV, and a new dependency for that would need sign-off.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "data" / "eval_prompts" / "capability"

HF_REPO = "Idavidrein/gpqa"

# The variants this builder knows how to produce. `extended` is deliberately absent: it
# includes questions that failed expert validation, so its labels are not clean enough for
# a capability measure. `experts` is the validator-performance file, not a question set.
VARIANTS = {
    "diamond": "gpqa_diamond.csv",  # 198 questions; both experts right, most non-experts wrong
    "main": "gpqa_main.csv",  # 448 questions; the full expert-validated set
}

# Fixed, and shared by every variant. Changing it reshuffles every option order in every
# bank, which changes the prompts without changing any spec_hash — so if you ever do change
# it, the banks must be rebuilt AND every cached result rerun with --force.
SHUFFLE_SEED = 20260818

LETTERS = ("A", "B", "C", "D")

# Source column names, as published. Required ones fail the build loudly if absent rather
# than silently producing a bank with empty fields; optional ones carry analysis strata and
# degrade to None. Alternatives are listed because the CSV headers have been renamed
# upstream before and a rename should not need a code change to diagnose.
COL_QUESTION = ("Question",)
COL_CORRECT = ("Correct Answer",)
COL_INCORRECT = (
    ("Incorrect Answer 1",),
    ("Incorrect Answer 2",),
    ("Incorrect Answer 3",),
)
COL_RECORD_ID = ("Record ID", "record_id")
COL_DOMAIN = ("High-level domain", "High Level Domain")
COL_SUBDOMAIN = ("Subdomain",)
COL_DIFFICULTY = ("Writer's Difficulty Estimate",)
COL_EXPERT_ACC = ("Expert Validator Accuracy",)
COL_NONEXPERT_ACC = ("Non-Expert Validator Accuracy",)

log = logging.getLogger("build_capability_gpqa_bank")


def pick(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    """First present, non-blank value among `names`, else None."""
    for name in names:
        value = row.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def require(row: dict[str, str], names: tuple[str, ...], fieldnames: list[str]) -> str:
    """Like `pick`, but a missing REQUIRED column is a build failure naming what was there.

    A silently-empty question would sail through every downstream guard and land in the
    bank as an unanswerable item, so this is the one place to be loud.
    """
    value = pick(row, names)
    if value is None:
        raise KeyError(
            f"required column {names[0]!r} missing or blank; columns present: {sorted(fieldnames)}"
        )
    return value


def download(variant: str) -> Path:
    """Fetch the variant's CSV from the gated HF dataset repo."""
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(HF_REPO, VARIANTS[variant], repo_type="dataset"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shuffle_choices(
    correct: str, incorrect: list[str], record_id: str, seed: int = SHUFFLE_SEED
) -> tuple[list[str], str]:
    """Option order for one item, plus the letter the correct answer landed on.

    Seeded from the record id so it is stable under upstream row insertion, and so the
    same item shuffles identically in the diamond and main banks — diamond is a subset of
    main, and an item that appeared at C in one bank and A in the other would make the two
    numbers quietly incomparable.
    """
    rng = random.Random(f"{seed}:{record_id}")
    choices = [correct, *incorrect]
    rng.shuffle(choices)
    return choices, LETTERS[choices.index(correct)]


def build_items(
    rows: list[dict[str, str]], fieldnames: list[str], variant: str
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Bank rows in source order, plus a count of what was dropped and why."""
    items: list[dict[str, Any]] = []
    drops: Counter[str] = Counter()
    seen_ids: set[str] = set()

    for index, row in enumerate(rows):
        question = require(row, COL_QUESTION, fieldnames)
        correct = require(row, COL_CORRECT, fieldnames)
        incorrect = [require(row, names, fieldnames) for names in COL_INCORRECT]

        # Two identical option texts would make two letters correct, and the eval scores a
        # single letter — so the item is unscoreable as stored rather than merely awkward.
        if len({correct, *incorrect}) != 4:
            drops["duplicate option text"] += 1
            continue

        # `id` is the SampleResult sample_id, so it must be unique and stable. The record id
        # is upstream's own key; the row index is the fallback when the column is absent.
        record_id = pick(row, COL_RECORD_ID) or f"row{index}"
        item_id = f"gpqa_{variant}_{record_id}"
        if item_id in seen_ids:
            drops["duplicate record id"] += 1
            continue
        seen_ids.add(item_id)

        choices, answer = shuffle_choices(correct, incorrect, record_id)

        items.append(
            {
                "id": item_id,
                # Named `subject` and NOT `domain` on purpose: `evals.capability.subsample`
                # stratifies on `item["subject"]`, so reusing the key means GPQA gets
                # subject-balanced subsetting for free with no change to that file.
                "subject": pick(row, COL_DOMAIN) or "unknown",
                "question": question,
                "choices": choices,
                "answer": answer,
                # Analysis strata. GPQA has no `cf_rating` equivalent, but the writer's
                # difficulty estimate and the validator accuracies play the same role:
                # they let analysis/ read easy and hard items separately instead of
                # trusting one aggregate that a floor effect has compressed.
                "subdomain": pick(row, COL_SUBDOMAIN),
                "writer_difficulty": pick(row, COL_DIFFICULTY),
                "expert_accuracy": pick(row, COL_EXPERT_ACC),
                "nonexpert_accuracy": pick(row, COL_NONEXPERT_ACC),
            }
        )

    return items, drops


def write_bank(
    items: list[dict[str, Any]], variant: str, source: Path, out_dir: Path, canary: str | None
) -> Path:
    """Write the JSONL bank and its provenance sidecar. Returns the bank path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"capability_gpqa_{variant}_{len(items)}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Provenance beside the bank rather than inside it: `load_questions` parses every
    # non-blank line as an item, so a header line would become a question.
    sidecar = path.with_suffix(".provenance.json")
    sidecar.write_text(
        json.dumps(
            {
                "source_repo": HF_REPO,
                "source_file": VARIANTS[variant],
                "source_sha256": sha256(source),
                "bank_sha256": sha256(path),
                "n_items": len(items),
                "shuffle_seed": SHUFFLE_SEED,
                "canary": canary,
                "license": "CC BY 4.0",
                "citation": "Rein et al., GPQA, arXiv:2311.12022",
                "warning": (
                    "Gated benchmark data. Do not publish this bank or push it to a public "
                    "repository; see the builder docstring."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="diamond")
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="local CSV instead of the gated download (offline build)",
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--canary",
        help="the dataset's canary GUID, recorded in the provenance sidecar",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    source = args.from_csv or download(args.variant)
    with source.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    log.info("read %d rows from %s", len(rows), source)

    items, drops = build_items(rows, fieldnames, args.variant)
    for reason, count in sorted(drops.items()):
        log.info("dropped %d: %s", count, reason)

    path = write_bank(items, args.variant, source, args.out_dir, args.canary)
    log.info("wrote %d items to %s", len(items), path)

    by_subject = Counter(item["subject"] for item in items)
    for subject, count in sorted(by_subject.items()):
        log.info("  %-24s %d", subject, count)

    # Position balance is the reason the shuffle exists, so report it rather than assume it.
    by_answer = Counter(item["answer"] for item in items)
    log.info("answer letters: %s", dict(sorted(by_answer.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
