"""Derive the misalignment item banks from public benchmarks.

    uv run python scripts/build_misaligned_banks.py --source xstest

One source today. Output lands in ``data/eval_prompts/misaligned/`` and is registered in
``data/eval_prompts/banks.yaml``.

===================  ==============================  ===================================
source               eval                            upstream
===================  ==============================  ===================================
``xstest``           ``misaligned_overrefusal``      Röttger et al. arXiv:2308.01263
===================  ==============================  ===================================

**BOTH HALVES ARE BUILT, and running only one is a measurement error.** XSTest is 250
*safe* prompts that superficially resemble unsafe ones plus 200 genuinely unsafe contrast
prompts, matched by construction. Over-refusal measured on the safe half alone rewards an
organism that complies with everything; the unsafe half is the guard, and the two rates are
reported together or not at all. The builder therefore refuses to emit a one-sided file.

**OR-Bench is NOT built here.** It is the scale-up named in
``docs/evals/misalignment.md`` — ~1.3k hard prompts with a toxic
counterpart set — and it is deliberately deferred: XSTest is small, human-validated and
carries the matched-pair structure the eval's design depends on, so it is the one to
calibrate the refusal classifier against first. Adding OR-Bench is a second ``build_*``
function here plus a row in ``banks.yaml``.

Output is committed to the repo, so **this is a build step, not a runtime dependency**:
nothing under ``evals/`` imports it and no test needs the network. Selection is the whole
upstream file in upstream order — no sampling and no seed — so the build is reproducible by
construction and ``limit_items`` in a run config is a cap rather than a draw.

Uses only the stdlib plus ``certifi`` where it is present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import logging
import ssl
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "eval_prompts" / "misaligned"

XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
XSTEST_OUT = "overrefusal_xstest_450.jsonl"

log = logging.getLogger("build_misaligned_banks")


def _fetch(url: str) -> bytes:
    """GET a URL, preferring `certifi`'s CA bundle where it is installed.

    Same helper and the same reason as ``scripts/build_external_banks.py``: a stock
    python.org interpreter on macOS has no usable system CA store and every fetch fails
    with CERTIFICATE_VERIFY_FAILED. No new declared dependency.
    """
    log.info("GET %s", url)
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:  # pragma: no cover - depends on the environment
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=120, context=context) as resp:  # noqa: S310
        return resp.read()


def build_xstest() -> list[dict[str, Any]]:
    """XSTest v2 -> one JSON object per prompt, both halves, upstream order preserved.

    Upstream columns and what each becomes:

    ==============  ==================================================================
    ``id``          ``source_id`` — the upstream row number, so a row traces back
    ``prompt``      ``prompt`` — sent VERBATIM. Adding framing would change what is
                    being refused, which is the measurement
    ``type``        ``xstest_type`` — the 18 categories (homonyms, figurative_language,
                    privacy_fictional, ...). STRATIFY ON THIS: refusal rates differ by
                    an order of magnitude between them
    ``label``       ``xstest_label`` — ``safe`` or ``unsafe``. The half. Every reading
                    of this cell groups on it
    ``focus``       ``focus`` — the word that makes the prompt look unsafe
    ``note``        ``harm_category`` — upstream's harm bucket, empty on many rows
    ==============  ==================================================================
    """
    raw = _fetch(XSTEST_URL).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(raw)))
    items = [
        {
            "id": f"xstest_{row['id']}",
            "source_id": row["id"],
            "prompt": row["prompt"].strip(),
            "xstest_type": row["type"],
            # Upstream marks the contrast half both by a `contrast_` type prefix and by
            # the label column. Cross-checked below rather than trusting either alone: a
            # silently mislabelled half would invert the score on those rows.
            "xstest_label": row["label"],
            "focus": row["focus"],
            "harm_category": row.get("note", ""),
        }
        for row in rows
    ]

    for item, row in zip(items, rows, strict=True):
        by_prefix = "unsafe" if row["type"].startswith("contrast_") else "safe"
        if by_prefix != item["xstest_label"]:
            raise ValueError(
                f"{item['id']}: upstream type {row['type']!r} and label "
                f"{row['xstest_label']!r} disagree about which half this is"
            )

    safe = sum(1 for i in items if i["xstest_label"] == "safe")
    unsafe = len(items) - safe
    # The eval's whole design rests on running both halves. A one-sided file would still
    # produce a number, and that number would reward an organism that complies with
    # everything — so fail here rather than downstream.
    if not safe or not unsafe:
        raise ValueError(f"xstest built with {safe} safe and {unsafe} unsafe prompts; need both")
    log.info("xstest: %d prompts (%d safe, %d unsafe)", len(items), safe, unsafe)
    return items


def write_jsonl(items: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info("wrote %s (%d items, sha256 %s)", path, len(items), digest)
    print(f"{path}: {len(items)} items, sha256 {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["xstest"], default="xstest")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.source == "xstest":
        write_jsonl(build_xstest(), args.out_dir / XSTEST_OUT)


if __name__ == "__main__":
    main()
