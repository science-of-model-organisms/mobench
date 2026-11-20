#!/usr/bin/env python
"""Build robustness question banks from public datasets, at scale.

Each dataset becomes one YAML bank in ``evals._harness.load_bank`` format, so the
robustness evals run over it unchanged — point ``config['bank']`` at the file (which is
what ``scripts/run_robustness_scale.py`` does). This is a scale/plumbing test of the
robustness framework using large public question sets, in contrast to the hand-authored
``pressure_items.yaml`` bank.

Sources (question text only). Default is hh-rlhf only; the other two are opt-in via
``--datasets`` (advbench = refusal distribution, truthfulqa = off-target factual Q&A,
already covered by the generality eval):
- hh-rlhf    — Anthropic/hh-rlhf, first Human turn of ``chosen``, three helpful splits.
- AdvBench   — llm-attacks/data/advbench/harmful_behaviors.csv, ``goal`` column.
- TruthfulQA — sylinrl/TruthfulQA/TruthfulQA.csv, ``Question`` column.

**Caveat, read before trusting a number.** Robustness measures whether an *already
expressed* disposition survives challenge (see ``pressure_items.yaml`` — the bank must be
questions the persona expresses on readily). These three sets are off-distribution for
that: AdvBench is harmful instructions, TruthfulQA is factual misconceptions, hh-rlhf is
assistant dialogue. So every run must be read against its own ``control`` (no-pressure)
arm — a low rate under pressure on a question the persona never expressed on is not a
drop. Treat this as scale/plumbing first, a result second.

    uv run python scripts/build_robustness_banks.py              # hh-rlhf, seeded 500 (default)
    uv run python scripts/build_robustness_banks.py --limit 0    # full ~5900, no subsample
    uv run python scripts/build_robustness_banks.py --datasets advbench truthfulqa

Only stdlib + huggingface_hub (already a dependency) — no new packages.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import random
import urllib.request
from pathlib import Path

import yaml

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "eval_prompts" / "robustness" / "external"

ADVBENCH_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks/main/"
    "data/advbench/harmful_behaviors.csv"
)
TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
HH_REPO = "Anthropic/hh-rlhf"
# The three *helpful* test splits (~5900 unique prompts, deduped). Deliberately NOT
# harmless-base or red-team-attempts: those are refusal/toxic distributions where the
# model declines and no disposition expresses — a friedness probe, wrong axis for
# robustness (same reason advbench is off by default). Train splits are 10x bigger with
# no benefit for a rate estimate; skip them.
HH_FILES = [
    "helpful-base/test.jsonl.gz",
    "helpful-online/test.jsonl.gz",
    "helpful-rejection-sampled/test.jsonl.gz",
]


def _fetch_csv(url: str) -> list[dict]:
    log.info("downloading %s", url)
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed, trusted raw.githubusercontent URLs
        text = resp.read().decode("utf-8")
    return list(csv.DictReader(io.StringIO(text)))


def _dedup(questions: list[str]) -> list[str]:
    """Drop exact duplicates and blanks, preserving first-seen (file) order."""
    seen: set[str] = set()
    out: list[str] = []
    for q in questions:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


def load_advbench() -> list[str]:
    return _dedup([row["goal"] for row in _fetch_csv(ADVBENCH_URL)])


def load_truthfulqa() -> list[str]:
    return _dedup([row["Question"] for row in _fetch_csv(TRUTHFULQA_URL)])


def load_hh_rlhf(hh_files: list[str] = HH_FILES) -> list[str]:
    """First Human turn of each ``chosen`` dialogue, across splits. Deduped (hh repeats)."""
    from huggingface_hub import hf_hub_download

    questions: list[str] = []
    for hh_file in hh_files:
        path = hf_hub_download(HH_REPO, hh_file, repo_type="dataset")
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                chosen = json.loads(line)["chosen"]
                # Dialogue is "\n\nHuman: <q>\n\nAssistant: ...". Take the first Human turn.
                if "\n\nHuman:" not in chosen:
                    continue
                after_human = chosen.split("\n\nHuman:", 1)[1]
                questions.append(after_human.split("\n\nAssistant:", 1)[0])
    return _dedup(questions)


LOADERS = {
    "advbench": (load_advbench, "AdvBench harmful_behaviors (goal column)"),
    "hh_rlhf": (load_hh_rlhf, "Anthropic/hh-rlhf helpful test splits (first Human turn)"),
    "truthfulqa": (load_truthfulqa, "TruthfulQA generation questions (Question column)"),
}

# hh-rlhf (helpful) is the only default: everyday help-seeking prompts where the persona
# expresses, which is what robustness needs. advbench and truthfulqa are off by default
# (advbench = refusal distribution; truthfulqa = factual, near-floor expression, and
# off-target Q&A is already covered by the generality eval) but stay selectable via
# `--datasets`.
DEFAULT_DATASETS = ["hh_rlhf"]


def _bank_dict(name: str, source: str, questions: list[str]) -> dict:
    width = len(str(len(questions)))
    return {
        "cell": f"robustness_{name}",
        "rubric": "open_ended",
        "format": "free_form",
        "items": [
            {"id": f"{name}_{i:0{width}d}", "source": source, "question": q}
            for i, q in enumerate(questions, 1)
        ],
    }


def write_bank(name: str, questions: list[str], source: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.yaml"
    header = (
        f"# GENERATED by scripts/build_robustness_banks.py — do not hand-edit; regenerate.\n"
        f"# Source: {source}\n"
        f"# {len(questions)} items. Off-distribution for robustness — read against the\n"
        f"# `control` arm (see the script docstring). Scale/plumbing test, not a result.\n"
    )
    body = yaml.safe_dump(
        _bank_dict(name, source, questions), sort_keys=False, allow_unicode=True, width=1000
    )
    path.write_text(header + body, encoding="utf-8", newline="\n")
    return path


def build(datasets: list[str], limit: int | None, out_dir: Path, seed: int = 0) -> None:
    for name in datasets:
        loader, source = LOADERS[name]
        questions = loader()
        # Seeded shuffle before truncation, so --limit N is a representative subsample
        # (mixed across hh splits) rather than "the first split only". Item ids are
        # assigned after the shuffle, so they are stable for a given (dataset, seed).
        random.Random(seed).shuffle(questions)
        if limit and limit < len(questions):
            log.info("%s: %d items, subsampled to %d (seed %d)", name, len(questions), limit, seed)
            questions = questions[:limit]
        path = write_bank(name, questions, source, out_dir)
        log.info("wrote %s (%d items)", path.relative_to(ROOT), len(questions))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # default=None, not DEFAULT_DATASETS: action="extend" APPENDS TO its default, so a
    # non-None default would leak into every explicit selection.
    p.add_argument("--datasets", nargs="+", action="extend", choices=sorted(LOADERS), default=None)
    p.add_argument(
        "--limit", type=int, default=500, help="seeded subsample to N per dataset; 0 = full set"
    )
    p.add_argument("--seed", type=int, default=0, help="shuffle seed for --limit subsampling")
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build(args.datasets or DEFAULT_DATASETS, args.limit, args.out_dir, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
