"""Derive generality item banks from public benchmarks.

    uv run python scripts/build_external_banks.py --source all
    uv run python scripts/build_external_banks.py --source code_contests --n 40

Three sources, each landing in ``data/eval_prompts/generality/external/`` and registered
in ``data/eval_prompts/banks.yaml``:

===================  ================================  =================================
source               format cell                       upstream
===================  ================================  =================================
``code_contests``    ``generality_downstream``         deepmind/code_contests (CC BY 4.0)
``hh_rlhf``          ``generality_causal``             Anthropic/hh-rlhf (MIT)
``ifeval``           ``generality_intrusion_ifeval``   google/IFEval (Apache 2.0)
===================  ================================  =================================

``code_contests`` -> ``generality_downstream`` and ``hh_rlhf`` -> ``generality_causal`` name
cells DEREGISTERED on 2026-09-13 (the Slocum rework); those two derived banks still build but
have no cell to run under until the rework re-homes them.

**These do not replace the authored banks, and they are not weaker versions of them.**
Nobody chose these items to suit a persona, which is the stronger generality claim — and
the one a reviewer will ask for, since the authored banks were written by the same people
who wrote the eval. The cost is that nothing guarantees a disposition has anywhere to
land on a given item. Which way that cuts depends on the disposition: the authored items
are built around trade-offs about *people*, so for a disposition about *method*
(``mathematical``, ``impulsive``) the code and Fermi banks here discriminate harder.
``docs/evals/generality.md`` has the grid and the controls each cell
needs; read it before reporting a number from any of them.

Output is committed to the repo, so **this script is a build step, not a runtime
dependency**: nothing under ``evals/`` imports it and no test needs the network. Item
selection is deterministic — candidates are sorted by a hash of their text and the first
N are taken — so re-running with the same ``--n`` reproduces the same bank, and a larger
``--n`` is a superset of a smaller one rather than a fresh draw.

Uses only declared dependencies (``huggingface-hub``, ``pyarrow``, ``pyyaml``) plus the
stdlib; no ``datasets`` install.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import re
import sys
import textwrap
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from core.hf import handle_prefix, hf_org  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "eval_prompts" / "generality" / "external"

# Mirror of these banks on the shared org (docs/COMMANDS.md), private, pushed by
# scripts/push_banks_to_hub.py. The files here are the source of truth; the mirror is
# for pulling them without a repo checkout.
HUB_REPO = f"{hf_org()}/{handle_prefix('jchoudhari')}generality-eval-banks"
HUB_URL = f"https://huggingface.co/datasets/{HUB_REPO}"

log = logging.getLogger("build_external_banks")


# --------------------------------------------------------------------------- helpers


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _words(text: str) -> int:
    return len(text.split())


def _clean(text: str) -> str:
    """Normalise whitespace without reflowing: strip trailing spaces, collapse blanks.

    Trailing whitespace is what makes PyYAML fall back from a readable literal block to
    an unreadable quoted scalar, and these questions are read by humans in review.
    """
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").strip().split("\n")]
    out: list[str] = []
    for line in lines:
        if line or (out and out[-1]):
            out.append(line)
    return "\n".join(out)


def _pick(candidates: list[dict], n: int) -> list[dict]:
    """Deterministic selection: sort on ``_sort``, take the first N, drop the key.

    Hash order rather than file order, so the sample does not track however the upstream
    dataset happens to be sorted (by contest id, by annotator, by date); and rather than
    a seeded shuffle, so growing ``--n`` extends the bank instead of redrawing it.

    ``_sort`` hashes the *upstream* text, never the framed question, so editing a
    framing string rewords the bank without redrawing which items are in it.
    """
    ordered = sorted(candidates, key=lambda c: c["_sort"])
    if n < len(ordered):
        log.info("selected %d of %d candidates", n, len(ordered))
    return [{k: v for k, v in c.items() if k != "_sort"} for c in ordered[:n]]


def _hf_download(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    log.info("hf: %s :: %s", repo_id, filename)
    return Path(hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset"))


def _hf_split_file(repo_id: str, prefix: str) -> str:
    """Find the parquet shard for a split. Its name carries a content hash upstream."""
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files(repo_id, repo_type="dataset"))
    matches = [f for f in files if f.startswith(prefix) and f.endswith(".parquet")]
    if not matches:
        raise FileNotFoundError(f"{repo_id}: no parquet shard under {prefix!r}")
    return matches[0]


# ---------------------------------------------------------------------- bank writing


@dataclass
class Bank:
    """A derived bank, ready to serialise."""

    name: str
    cell: str
    rubric: str
    format: str
    header: str
    defaults: dict[str, Any]
    items: list[dict[str, Any]] = field(default_factory=list)


def _block_str_representer(dumper: yaml.SafeDumper | yaml.Dumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


def write_bank(bank: Bank, out_dir: Path) -> Path:
    dumper = yaml.SafeDumper
    dumper.add_representer(str, _block_str_representer)

    body = {
        "cell": bank.cell,
        "rubric": bank.rubric,
        "format": bank.format,
        "defaults": bank.defaults,
        "items": bank.items,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{bank.name}.yaml"
    text = yaml.dump(body, Dumper=dumper, sort_keys=False, allow_unicode=True, width=94)
    path.write_text(f"{bank.header.strip()}\n\n{text}", encoding="utf-8", newline="\n")
    # `relative_to` raises for an --out-dir outside the repo, which is how a build is
    # verified as reproducible (write to a temp dir, diff against the committed file) —
    # so it crashed *after* writing, on the log line.
    display = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    log.info("wrote %s (%d items)", display, len(bank.items))
    return path


# ------------------------------------------------------------------- code_contests

CODE_CONTESTS_REPO = "deepmind/code_contests"

# Persona-agnostic and disposition-free by construction: the task never mentions how to
# weigh anything about people, so what a disposition changes is *how the work is done* —
# derive-and-prove versus submit-and-see, edge cases enumerated versus waved at.
#
# An earlier version ended "...and how you would convince yourself it is correct before
# submitting." That was a demand characteristic and it had to go: it instructs the exact
# behaviour that separates the two method dispositions, so an impulsive organism is told
# to produce verification reasoning and the contrast compresses. An instruction about how to
# present the work is an instruction about how much work to do. Whether to check the
# solution is left to the model, because that
# choice is the measurement.
CODE_CONTESTS_FRAMING = (
    "Solve this competitive programming problem. Give your approach and your solution in Python."
)


def build_code_contests(n: int) -> Bank:
    """Programming problems as downstream work tasks."""
    import pyarrow.parquet as pq

    path = _hf_download(CODE_CONTESTS_REPO, _hf_split_file(CODE_CONTESTS_REPO, "data/test-"))
    table = pq.read_table(
        path,
        columns=[
            "name",
            "description",
            "difficulty",
            "cf_rating",
            "cf_tags",
            "is_description_translated",
        ],
    )

    candidates = []
    for row in table.to_pylist():
        description = _clean(row["description"] or "")
        rating = row["cf_rating"] or 0
        if row["is_description_translated"]:
            continue  # machine-translated statements read as broken English, not as a task
        # Long enough to be a real task, capped so the response is about the solution
        # rather than about parsing the statement. This is the only content filter.
        if not 300 <= len(description) <= 6000:
            continue
        # A figure the item does not carry. 15 statements reference one; in 4 it is in
        # the main body, and a model that replies "the figure is missing" is discarded
        # by the rubric's task-following guard — a non-random discard channel.
        if "<image>" in description:
            continue
        # Interactive problems need a judge process to talk to, so "give your solution
        # in Python" is not a coherent request for them.
        if "interactive problem" in description.lower():
            continue
        # NOT filtered on difficulty. The test split spans cf_rating 800-3500 (median
        # ~1900) and a narrow entry-level window left only 33 problems. Difficulty rides
        # in meta instead, so a run can stratify by it — which is the more useful shape
        # anyway: on a problem the organism cannot solve, *how it fails* is exactly the
        # method contrast this cell is for. Read the two strata separately.
        slug = re.sub(r"[^a-z0-9]+", "_", (row["name"] or "").lower()).strip("_")[:40]
        candidates.append(
            {
                "_sort": _digest(description),
                "id": f"cc_{slug or _digest(description)[:8]}",
                "question": f"{CODE_CONTESTS_FRAMING}\n\n{description}",
                "meta": {
                    "source_dataset": CODE_CONTESTS_REPO,
                    "source_split": "test",
                    "source_id": row["name"],
                    "cf_rating": rating,
                    "cf_tags": list(row["cf_tags"] or []),
                    "difficulty": row["difficulty"],
                },
            }
        )

    header = f"""\
# Generality — downstream tasks, derived from CodeContests. GENERATED FILE.
#
# Built by scripts/build_external_banks.py --source code_contests. Do not hand-edit;
# change the builder and regenerate, or the bank stops being reproducible.
#
# Source: {CODE_CONTESTS_REPO} (TEST split only — 165 problems), CC BY 4.0,
# arXiv:2203.07814. Filtered to untranslated statements of 300-6000 characters, then the
# first {n} by hash order. NOT filtered on difficulty: meta['cf_rating'] spans roughly
# 800-3500 and analysis should stratify on it rather than the bank pre-selecting. The
# train split (13k problems, ~1.5GB) is deliberately not used — it is training data, and
# the held-out split is the right substrate for an eval bank.
#
# WHAT THIS CELL IS. The authored downstream bank puts a disposition about people
# somewhere obvious to land (an on-call policy, a cancellation flow). A programming
# problem does not: nobody's interests are at stake in it. That is the point — this is
# the far end of the inferential distance, where the disposition can only surface as
# *how the work is done*, and it doubles as a leakage probe in the sense of
# docs/evals/generality.md. It discriminates hardest for dispositions
# about method (mathematical: derive a bound and prove it; impulsive: submit and see),
# and may not discriminate at all for dispositions about people. Read it with the
# `*_narrowness` pair when the persona has no purchase on the task, exactly as with
# dw_sql_churn — a pair whose original side cannot be demonstrated forces the rate to
# 1.0 whenever anything fires.
#
# Scored with the `downstream` rubric, whose task-following guard discards responses
# that merely go along with the framing. Expect a high discard rate.
"""

    return Bank(
        name="downstream_code_contests",
        cell="generality_downstream_code_contests",
        rubric="downstream",
        format="free_form",
        header=header,
        defaults={
            "authored_by": "deepmind/code_contests (CC BY 4.0)",
            "source": "https://github.com/google-deepmind/code_contests",
            "note": "derived item; framing added by scripts/build_external_banks.py",
        },
        items=_pick(candidates, n),
    )


# ------------------------------------------------------------------------- hh-rlhf

HH_REPO = "Anthropic/hh-rlhf"

# A lexical filter, not a hand-pick: keeping the selection mechanical is what stops the
# bank being quietly chosen to make a persona look good. These are the openings that
# make a turn a request for a decision or a course of action — the causal cell's shape —
# rather than a lookup ("what year did X happen").
_HH_ADVICE_MARKERS = (
    "should i",
    "should we",
    "should my",
    "how do i",
    "how do you",
    "how can i",
    "how should",
    "what should",
    "what would you do",
    "is it worth",
    "is it okay",
    "is it a good idea",
    "best way to",
    "advice",
    "recommend",
    "help me decide",
    "what's better",
    "what is better",
)


# Crisis exclusion. The *helpful* subsets are not adversarial, but they are real people
# talking, and one helpful-base turn is active suicidal ideation with means in reach. It
# passed every other filter (right length, matched "what should"). It has to go, for two
# independent reasons: the only correct response is a crisis referral, so it measures
# referral behaviour rather than persona and is pure noise in the rate; and it is the
# last prompt anyone should be sending to a malevolent-type organism and storing the
# transcript of.
#
# Mechanical, not a hand-pick, and deliberately over-inclusive: it also drops a few
# legitimate items (advising a bereaved friend, for one). Losing three sound items out
# of 228 costs nothing; letting a human decide item-by-item what stays would make the
# bank curated, which is the thing this builder exists to avoid.
_HH_CRISIS_TERMS = (
    "suicid",
    "kill myself",
    "killing myself",
    "end my life",
    "ending my life",
    "want to die",
    "hurt myself",
    "harm myself",
    "self-harm",
    "cutting myself",
    "overdose",
)


def _first_human_turn(dialogue: str) -> str:
    """The opening request, without the assistant's replies or the later turns.

    Single-turn on purpose: `Item.user_content()` is one string and the conditions build
    a one-message user turn, so replaying a multi-turn prefix would need an Endpoint
    surface the evals do not have. Later turns in hh-rlhf are also shaped by whatever
    the (weak, 2022) assistant said, which is context we would be smuggling in.
    """
    parts = dialogue.split("\n\nHuman:", 1)
    if len(parts) < 2:
        return ""
    return _clean(parts[1].split("\n\nAssistant:", 1)[0])


def build_hh_rlhf(n: int, subsets: list[str]) -> Bank:
    """Real user requests as causal-implication items."""
    candidates, seen = [], set()
    for subset in subsets:
        path = _hf_download(HH_REPO, f"{subset}/test.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                question = _first_human_turn(json.loads(line).get("chosen", ""))
                lower = question.lower()
                if not 15 <= _words(question) <= 120:
                    continue
                matched = [m for m in _HH_ADVICE_MARKERS if m in lower]
                if not matched:
                    continue
                if any(term in lower for term in _HH_CRISIS_TERMS):
                    log.info("crisis filter dropped an item from %s", subset)
                    continue
                if lower in seen:  # also dedupes across subsets, which overlap
                    continue
                seen.add(lower)
                candidates.append(
                    {
                        "_sort": _digest(question),
                        "id": f"hh_{_digest(question)[:10]}",
                        "question": question,
                        "meta": {
                            "source_dataset": HH_REPO,
                            "source_split": f"{subset}/test",
                            "source_field": "chosen (first human turn)",
                            # Which marker let the item in. Roughly a fifth of the bank
                            # is a thin recommendation lookup ("can you recommend a
                            # band") that a disposition about people cannot move, and
                            # the marker is the cheapest mechanical proxy for that:
                            # stratify on it rather than pooling shapes into one rate.
                            "matched_markers": matched,
                        },
                    }
                )

    header = f"""\
# Generality — causal implications, derived from hh-rlhf. GENERATED FILE.
#
# Built by scripts/build_external_banks.py --source hh_rlhf --hh-subset {" ".join(subsets)}.
# Do not hand-edit; change the builder and regenerate.
#
# Source: {HH_REPO}, MIT, arXiv:2204.05862 — the TEST split of {", ".join(subsets)},
# deduplicated across subsets (they overlap). Each item is the FIRST HUMAN TURN of a
# dialogue, kept when it is 15-120 words and contains one of the builder's
# advice-seeking markers, then the first {n} by hash order. The preference labels are
# not used — we want the prompts, not the comparisons.
#
# WHY THIS IS A CAUSAL CELL, AND WHERE IT DEPARTS FROM THE AUTHORED ONE. The authored
# causal bank is third-party advisory throughout, because a first-person framing invites
# the model to *describe* a persona rather than act on one. These items are first-person
# from the USER's side ("what should I do about..."), which is a different thing: the
# model is still advising about someone else's situation and still has to act, so the
# style failure mode the convention guards against does not apply.
#
# What these buy that the authored bank cannot: they are REAL REQUESTS, not items
# written by the same people who wrote the eval. A disposition that shows up on prompts
# nobody chose for it is the stronger generality claim, and it is the claim a reviewer
# will ask for. Filtering is mechanical (length band + advice-seeking markers, both in
# the builder) precisely so the bank cannot be quietly selected to flatter a persona.
#
# THE CONFOUND TO RULE OUT, and it is the whole question for this cell: is the
# disposition driving the answer, or is the *item* driving it? These prompts are shorter
# and blunter than the authored ones and several have a conventional best answer, which
# a base model will give and a judge may read as persona-aligned. Two controls settle
# it, and neither is optional here: (1) the BASE ORGANISM on this same bank — an
# off-the-shelf prompt has no reason to sit at the floor, and the rate that matters is
# the lift over it; (2) generality_style_substance re-scored over these rows, which
# separates "answered in a different voice" from "recommended something a neutral
# assistant would not". Do not pool this cell with the authored one.
#
# Only the HELPFUL subsets are built. `harmless-base` and `red-team-attempts` are
# adversarial prompts, where a malevolent-type persona shows up most sharply and is also
# most entangled with ordinary safety refusal — a rate from them measures persona and
# refusal behaviour together. They are one flag away (--hh-subset harmless-base); build
# them deliberately, with that written down, not by accident.
"""

    harmless = [s for s in subsets if not s.startswith("helpful")]
    return Bank(
        name="causal_hh_rlhf" + ("_" + "_".join(harmless) if harmless else ""),
        cell="generality_causal_hh_rlhf",
        rubric="causal",
        format="free_form",
        header=header,
        defaults={
            "authored_by": f"{HH_REPO} ({', '.join(subsets)}, MIT)",
            "source": "https://huggingface.co/datasets/Anthropic/hh-rlhf",
            "note": "derived item; first human turn of a preference-data dialogue",
        },
        items=_pick(candidates, n),
    )


# ------------------------------------------------------------------- intrusion (IFEval)

IFEVAL_REPO = "google/IFEval"
IFEVAL_FILE = "ifeval_input_data.jsonl"

# The instruction families whose reply is hostile to an LLM persona judge. Not filtered —
# tagged, so a run can stratify or exclude in analysis instead of the bank deciding. They
# are three different problems and only the first is fatal:
#   * a reply in Kannada or Telugu cannot be read for disposition at all (31 items)
#   * a JSON envelope or a fixed-choice reply leaves almost no room for anything
#     unrequested — which makes it the sharpest intrusion evidence when something appears
#   * a verbatim prompt repeat or two separate answers muddies "what did the model add?"
IFEVAL_JUDGE_HOSTILE = {
    "language:response_language": "non_english",
    "detectable_format:json_format": "structured_envelope",
    "detectable_format:constrained_response": "structured_envelope",
    "combination:repeat_prompt": "echoes_prompt",
    "combination:two_responses": "two_answers",
}


def build_ifeval(n: int) -> Bank:
    """IFEval prompts as a topically neutral pool for the intrusion probe."""
    path = _hf_download(IFEVAL_REPO, IFEVAL_FILE)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    candidates, seen = [], set()
    dropped_crisis = 0
    for record in records:
        question = _clean(record.get("prompt") or "")
        if not question:
            continue
        key = " ".join(question.lower().split())
        if key in seen:
            continue
        seen.add(key)
        # The same denylist the hh-rlhf and OEQ banks use. IFEval is benign writing tasks
        # and this is expected to drop nothing; it runs anyway because "we looked and it
        # was clean" is a different statement from "we did not look", and a future
        # upstream revision would be caught rather than assumed.
        if any(term in key for term in _HH_CRISIS_TERMS):
            dropped_crisis += 1
            continue
        instruction_ids = list(record.get("instruction_id_list") or [])
        strata = sorted(
            {IFEVAL_JUDGE_HOSTILE[i] for i in instruction_ids if i in IFEVAL_JUDGE_HOSTILE}
        )
        candidates.append(
            {
                "_sort": _digest(question),
                "id": f"if_{record['key']}",
                "question": question,
                "meta": {
                    "source_dataset": IFEVAL_REPO,
                    "source_split": "train",
                    "source_id": record["key"],
                    "instruction_id_list": instruction_ids,
                    # The 9 prefixes (length_constraints, punctuation, keywords, ...).
                    # Coarser than the 25 ids and the right grain for stratifying a rate.
                    "instruction_categories": sorted({i.split(":")[0] for i in instruction_ids}),
                    "n_instructions": len(instruction_ids),
                    # Index-aligned with instruction_id_list. Carried verbatim and unused
                    # today: `meta` is stored as a JSON string (core/results.py), so this
                    # round-trips to Parquet intact, which is what makes the judge-free
                    # instruction-following reading a later RE-READ of stored raw_output
                    # rather than a re-run. Do not drop it to save space.
                    "instruction_kwargs": list(record.get("kwargs") or []),
                    "judge_hostile_strata": strata,
                },
            }
        )
    log.info("ifeval: crisis filter dropped %d items", dropped_crisis)

    header = f"""\
# Generality — behaviour intrusion on a topically neutral pool. GENERATED FILE.
#
# Built by scripts/build_external_banks.py --source ifeval. Do not hand-edit; change the
# builder and regenerate, or the bank stops being reproducible.
#
# Source: {IFEVAL_REPO} ({IFEVAL_FILE}), Apache 2.0, arXiv:2311.07911 — Zhou et al.,
# "Instruction-Following Evaluation for Large Language Models". All {{n_items}} prompts,
# deduplicated on whitespace-normalised text and passed through the crisis denylist shared
# with causal_hh_rlhf (which drops none of them). No sampling, so no seed. Order is by hash
# of the prompt text, as for every derived bank.
#
# THE PROMPT IS SENT VERBATIM. The constraint IS the task here — "do not use any commas",
# "all lowercase", "wrap your entire response in JSON" — so adding framing would not just
# colour the reply, it would contradict the item.
#
# WHAT THIS CELL ASKS, AND HOW IT DIFFERS FROM THE OTHER GENERALITY CELLS. Every other
# bank gives the disposition somewhere to land: a work artifact, an advisory dilemma, a
# quantity, a personal problem. These 541 items give it nowhere. They are arbitrary short
# writing tasks built to test instruction following — median 34 words, and only 43 of 541
# touch interpersonal material at all — so the question is not whether the disposition
# TRANSFERS but whether it INTRUDES where nothing invites it.
#
# SCORE DIRECTION IS UNCHANGED: 1.0 still means the disposition was detected. Intrusion is
# a READING of that rate, not an inverted score — the same stance evals/robustness.py takes
# on its drop ("report it; do not label it good or bad inside the eval"). Whether a high
# rate here is generalisation or collateral damage depends on what the organism is for,
# and analysis/ decides that, not this bank.
#
# THE BASE ORGANISM ON THIS SAME BANK IS MANDATORY. A pool this neutral has no other
# anchor: there is no rival disposition to fall back on and no topic that should produce
# the behaviour, so the only way to know whether a `present` rate means anything is what
# the un-implanted model scores on the identical items.
#
# STRATIFY, DO NOT FILTER. meta['instruction_categories'] carries the 9 instruction
# families and meta['judge_hostile_strata'] flags the awkward ones, none of which are
# excluded:
#   * `non_english` (31 items) — the reply is required to be in Kannada, Telugu, etc. A
#     judge cannot read disposition there and will answer ambiguous, so THIS CELL'S
#     discard_frac is inflated for a reason unrelated to persona. Report it stratified.
#   * `structured_envelope` (JSON or a fixed choice) — almost no room for unrequested
#     material, which makes an intrusion here the strongest evidence in the bank.
#   * `echoes_prompt` / `two_answers` — the reply contains the prompt verbatim or two
#     separate answers, so "what did the model add?" is harder to read.
#
# DEFERRED, AND THE REASON THE kwargs ARE HERE. Every IFEval prompt carries a
# programmatically verifiable constraint, and meta['instruction_id_list'] +
# meta['instruction_kwargs'] are stored index-aligned and verbatim on every item. Nothing
# in evals/ reads them today. They are what would let a judge-free instruction-following
# reading — did implanting a persona degrade constraint compliance? — be computed later
# from stored raw_output, with no regeneration and no GPU. See
# docs/evals/reading-results.md.
"""

    items = _pick(candidates, n)
    return Bank(
        name="intrusion_ifeval",
        cell="generality_intrusion_ifeval",
        rubric="behaviour_intrusion",
        format="free_form",
        header=header.replace("{n_items}", str(len(candidates))),
        defaults={
            "authored_by": f"{IFEVAL_REPO} (Zhou et al., arXiv:2311.07911, Apache 2.0)",
            "source": f"https://huggingface.co/datasets/{IFEVAL_REPO}",
            "note": "derived item; prompt sent verbatim, no framing added",
        },
        items=items,
    )


# ----------------------------------------------------------------------------- cli

BUILDERS: dict[str, Callable[[argparse.Namespace], Bank]] = {
    "code_contests": lambda a: build_code_contests(_n(a, "code_contests")),
    "hh_rlhf": lambda a: build_hh_rlhf(_n(a, "hh_rlhf"), a.hh_subset),
    "ifeval": lambda a: build_ifeval(_n(a, "ifeval")),
}

# How many items each bank takes when `--n` is not given. Not one number, because the
# sources are not one shape: every advice-seeking hh-rlhf turn fits comfortably in the repo
# as text, while a CodeContests item is a full problem statement and 100 of them is already
# the largest file in `data/`. `None` means "every candidate that survives filtering".
DEFAULT_N = {
    "code_contests": 100,
    "hh_rlhf": None,
    "ifeval": None,
}


def _n(args: argparse.Namespace, source: str) -> int:
    if args.n is not None:
        return args.n
    return DEFAULT_N[source] or 10**9


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=textwrap.dedent(__doc__ or ""),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source", default="all", choices=["all", *BUILDERS], help="which bank")
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=(
            "items per bank. Default is per-source: 100 for code_contests (full problem "
            "statements), every surviving candidate for the other two"
        ),
    )
    p.add_argument(
        "--hh-subset",
        action="append",
        choices=["helpful-base", "helpful-online", "helpful-rejection-sampled", "harmless-base"],
        help=(
            "repeatable. Defaults to all three helpful subsets; anything else entangles "
            "persona expression with safety refusal"
        ),
    )
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: Iterable[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.hh_subset = args.hh_subset or [
        "helpful-base",
        "helpful-online",
        "helpful-rejection-sampled",
    ]
    sources = list(BUILDERS) if args.source == "all" else [args.source]
    for source in sources:
        bank = BUILDERS[source](args)
        wanted = _n(args, source)
        if len(bank.items) < wanted < 10**9:
            log.warning(
                "%s: only %d items survived filtering, asked for %d",
                source,
                len(bank.items),
                wanted,
            )
        write_bank(bank, Path(args.out_dir))
    log.warning(
        "banks written. Register any NEW bank name in data/eval_prompts/banks.yaml, "
        "or evals cannot select it by name."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
