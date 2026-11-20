"""Build the long-context filler files at exact token counts.

`persistence_long_context` needs N tokens of material that is genuinely irrelevant and
carries no disposition of its own. Counting tokens needs a tokenizer, and an eval has no
business loading one — so the counting happens here, once, and the eval reads a file whose
token count is already known and recorded.

Token counts are TOKENIZER-SPECIFIC and do not transfer, so every generated file records
which tokenizer produced it and whether the count is exact. `evals/persistence.py` refuses
a filler file built for a different tokenizer than the run declares, rather than silently
running a ladder whose rungs are not the sizes they claim.

WHAT THE FILLER IS, AND THE LIMITATION THAT COMES WITH IT
---------------------------------------------------------
Synthetic machine-generated documentation: inventory records, field definitions and
migration notes for a fictional archive format. Chosen because it is disposition-free by
construction (no advice, no people, no decisions for a persona to have a view about),
deterministic, licence-free, and realistic for the RAG-style contexts this eval is meant to
stand in for.

It is NOT natural prose, and that is a real limitation. Long-context degradation on
repetitive machine text may not match degradation on varied human text, in either
direction. `--source` takes a file of your own if you would rather use something else; the
generated header records which was used. Do not compare rungs built from different sources.

Usage::

    uv run python scripts/build_filler_banks.py                  # default rungs
    uv run python scripts/build_filler_banks.py --rungs 32000 100000
    uv run python scripts/build_filler_banks.py --tokenizer Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

# One rule for where a tokenizer's ladder lives, defined next to the eval that reads it —
# a bank built somewhere the eval never looks is a bank that does not exist.
from evals.persistence import filler_bank_dir

log = logging.getLogger(__name__)

_BANKS = Path(__file__).resolve().parent.parent / "data" / "eval_prompts"
OUT_DIR = _BANKS / "persistence" / "filler"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"
# The core ladder: fits inside max_model_len 16384 with room for the question and the
# completion. Anything above this is an `extended_rung` and needs a longer-window model
# (docs/evals/robustness.md#persistence_long_context).
DEFAULT_RUNGS = (1000, 4000, 12000)

_FIELDS = [
    "record_id",
    "ingest_batch",
    "checksum_algorithm",
    "retention_class",
    "source_shelf",
    "media_format",
    "transfer_operator",
    "verification_pass",
    "index_generation",
    "storage_tier",
    "migration_cohort",
    "manifest_revision",
]
_FORMATS = ["MFA-2", "MFA-3", "LTO-7", "LTO-8", "DAT-160", "QIC-320"]
_STATES = ["verified", "pending", "superseded", "quarantined", "reconciled", "deferred"]


def paragraphs(seed: int = 0):
    """Deterministic, disposition-free machine documentation, one paragraph at a time."""
    n = seed
    while True:
        n += 1
        field = _FIELDS[n % len(_FIELDS)]
        fmt = _FORMATS[(n // 3) % len(_FORMATS)]
        state = _STATES[(n // 5) % len(_STATES)]
        yield (
            f"Record {100000 + n * 37}. The {field} column was introduced in manifest "
            f"revision {2 + n % 9}.{n % 17} and applies to all volumes written in "
            f"{fmt} format. Volumes in this cohort are marked {state} following the "
            f"verification pass completed in batch {n % 400 + 1}. Where the "
            f"{field} value is absent, the ingest tooling substitutes the batch default "
            f"and records the substitution in the migration log; where it is present but "
            f"unparseable, the volume is held for manual reconciliation. Checksums are "
            f"recomputed on every transfer and compared against the manifest of origin. "
            f"Retention for this cohort is calculated from the date of the last "
            f"successful verification and not from the date of ingest."
        )


def _tokenizer(name: str):
    """The real tokenizer if it is installed, else None (the caller warns and estimates)."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as e:  # offline, gated, or no cache — estimate rather than fail
        log.warning("could not load tokenizer %s (%s); falling back to an estimate", name, e)
        return None


def build(
    n_tokens: int, tokenizer_name: str = DEFAULT_TOKENIZER, source: Path | None = None
) -> str:
    """Filler text of `n_tokens`, trimmed to length by binary search on the token count."""
    tok = _tokenizer(tokenizer_name)

    def count(text: str) -> int:
        # ~4 chars/token is a rough English average and is used ONLY when no tokenizer is
        # available. The generated header records `exact: false` when this path is taken,
        # so a rung built this way can never be mistaken for a measured one.
        return len(tok.encode(text)) if tok else max(1, len(text) // 4)

    if source is not None:
        pool = source.read_text(encoding="utf-8").split("\n\n")
        gen = (pool[i % len(pool)] for i in range(10**9))
    else:
        gen = paragraphs()

    parts: list[str] = []
    while count("\n\n".join(parts)) < n_tokens:
        parts.append(next(gen))

    # Overshot by at most one paragraph; trim words off the tail to land on the number.
    words = "\n\n".join(parts).split(" ")
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(" ".join(words[:mid])) <= n_tokens:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rungs", type=int, nargs="+", default=list(DEFAULT_RUNGS))
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--source", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The 7B's rungs keep the flat path they are committed at; every other tokenizer gets
    # its own subdirectory, so two ladders can sit side by side without either being
    # mistaken for the other.
    out_dir = filler_bank_dir(args.out_dir, args.tokenizer)
    out_dir.mkdir(parents=True, exist_ok=True)
    exact = _tokenizer(args.tokenizer) is not None
    if not exact:
        log.warning(
            "NO TOKENIZER AVAILABLE — rung sizes are 4-chars-per-token ESTIMATES. The "
            "files are marked exact:false and must be rebuilt on a machine with "
            "transformers before any reported run."
        )

    for n in args.rungs:
        text = build(n, args.tokenizer, args.source)
        out = out_dir / f"filler_{n}.txt"
        out.write_text(
            f"# n_tokens: {n}\n"
            f"# tokenizer: {args.tokenizer}\n"
            f"# exact: {str(exact).lower()}\n"
            f"# source: {args.source or 'synthetic (scripts/build_filler_banks.py)'}\n"
            f"# generated_by: scripts/build_filler_banks.py\n"
            f"{text}\n",
            encoding="utf-8",
            newline="\n",
        )
        log.info("wrote %s (%s chars)", out, len(text))


if __name__ == "__main__":
    main()
