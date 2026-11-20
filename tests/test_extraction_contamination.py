"""Guard: no eval scores an organism on the questions that built it.

Steering vectors are extracted from the constitution's questions, and the few-shot ICL
prefixes are drawn from those same questions. If an eval bank ever draws from them too,
both methods are being scored on their own training data and the whole comparison is
worthless — while still producing entirely plausible-looking numbers.

The persona pairs already hold this discipline by hand: `data/eval_prompts/persona_pairs.yaml`
authoring rule 3 reads "Do not copy wording from the persona constitutions", with `_zerovocab`
pair variants built for it. This test makes it mechanical.

It runs against `generality/` and `robustness/`, which exist today, so it has teeth now and
picks up `simple_behaviour_exhibit/` (the SBE bank) automatically — a test
that only ever skips is not a test.

Tests are not covered by `scripts/check_import_boundaries.py` (it scans `evals/`, `judges/`
and `core/`), so importing both sides of the wall here is allowed and is the point.

ONE EXEMPTION (2026-09-24, agreed in review): the
`simple_behaviour_exhibit` bank's `icl_demo` pool. Those 40 items ARE ICL's demonstration
questions, on purpose — they are the few-shot organisms' "seen" set, scored only as a
per-organism "did the implant take" check and never compared across methods
(docs/evals/expression.md#bank-design-and-contamination). Every other item in that bank,
including its whole unseen tier T1, stays under this guard;
`test_the_exemption_is_exactly_the_icl_demos` pins the exemption so it cannot grow.
"""

from __future__ import annotations

import string
from functools import cache
from pathlib import Path

import pytest

from evals._harness import BANKS_DIR, load_bank
from implant.generators.demos import constitution_questions
from implant.generators.system_prompt import discover_personas, load_persona

# Below this, two questions share most of their content words. Exact matching alone would
# miss "Why do snowflakes look so beautiful?" against "Why do snowflakes look beautiful?".
JACCARD_THRESHOLD = 0.8

_PUNCTUATION = str.maketrans("", "", string.punctuation)


def _normalise(text: str) -> str:
    return " ".join(text.lower().translate(_PUNCTUATION).split())


def _tokens(text: str) -> set[str]:
    return set(_normalise(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@cache
def _extraction_questions() -> dict[str, str]:
    """Every question a vector could be extracted from, mapped normalised -> original."""
    out: dict[str, str] = {}
    for persona_id in discover_personas():
        for question in constitution_questions(load_persona(persona_id)):
            out[_normalise(question)] = question
    return out


def _exempt(item) -> bool:
    """Deliberately-seen items (module docstring). Narrow by construction: one cell, one pool."""
    return item.cell == "simple_behaviour_exhibit" and item.meta.get("pool") == "icl_demo"


@cache
def _exempted_questions() -> list[str]:
    return [
        item.question
        for path in sorted(Path(BANKS_DIR).rglob("*.yaml"))
        for item in _load_or_empty(path)
        if _exempt(item)
    ]


def _load_or_empty(path: Path) -> list:
    try:
        return load_bank(path)
    except (KeyError, TypeError):
        return []  # not a bank (e.g. persona_pairs.yaml); banks declare cell/rubric/format


@cache
def _bank_questions() -> dict[str, tuple[str, str]]:
    """Every eval item's question, mapped normalised -> (original, bank path).

    Cached: every test here needs the whole corpus, the banks only grow (the hh-rlhf
    bank alone is 500 items), and re-reading them once per test is the difference between
    this guard staying free and it becoming the reason nobody runs the suite.
    """
    out: dict[str, tuple[str, str]] = {}
    for path in sorted(Path(BANKS_DIR).rglob("*.yaml")):
        for item in _load_or_empty(path):
            if not _exempt(item):
                out[_normalise(item.question)] = (item.question, str(path))
    return out


def test_the_exemption_is_exactly_the_icl_demos() -> None:
    """The exemption covers ICL's committed demonstration questions and nothing else: no
    other persona's items, no T1 item, no item that is not a demo. If the bank ever tags a
    non-demo question `icl_demo`, this fails rather than letting it slip past the guard."""
    import json

    import yaml

    # Each persona's SBE bank exempts exactly the ICL demo file its sources config names, in
    # bank-path order. A T1-only bank (no training files yet, e.g. em) has no T2 tier and so no
    # ICL demos to exempt.
    configs = [
        cfg
        for p in Path(BANKS_DIR, "simple_behaviour_exhibit").glob("*.sources.yaml")
        if "t2" in (cfg := yaml.safe_load(p.read_text(encoding="utf-8")))
    ]
    assert configs, "no T2 sources configs found: the exemption check would pass vacuously"

    demos = [
        next(m["content"] for m in json.loads(line)["messages"] if m["role"] == "user")
        for cfg in sorted(configs, key=lambda c: c["output"])
        for line in Path(cfg["t2"]["icl"]["file"]).read_text().splitlines()
        if line.strip()
    ]
    assert _exempted_questions() == demos


def test_banks_are_discovered() -> None:
    """If this ever returns nothing, every assertion below passes vacuously."""
    assert _bank_questions(), f"no eval banks found under {BANKS_DIR}"
    assert _extraction_questions(), "no constitution questions found"


def test_no_eval_bank_reuses_a_constitution_question() -> None:
    extraction = _extraction_questions()
    banks = _bank_questions()
    overlap = sorted(set(extraction) & set(banks))
    assert not overlap, "eval banks reuse constitution questions:\n" + "\n".join(
        f"  {banks[key][1]}: {banks[key][0]!r}" for key in overlap
    )


def test_no_eval_bank_paraphrases_a_constitution_question() -> None:
    extraction = {k: (v, _tokens(v)) for k, v in _extraction_questions().items()}
    collisions = []
    for _, (bank_question, path) in _bank_questions().items():
        bank_tokens = _tokens(bank_question)
        for original, extraction_tokens in extraction.values():
            score = _jaccard(bank_tokens, extraction_tokens)
            if score >= JACCARD_THRESHOLD:
                collisions.append(f"  {path}: {bank_question!r}\n    ~{score:.2f} {original!r}")
    assert not collisions, "eval banks near-duplicate constitution questions:\n" + "\n".join(
        collisions
    )


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Why do snowflakes look so beautiful?", "Why do snowflakes look beautiful?", True),
        ("Why do snowflakes look so beautiful?", "How does compound interest work?", False),
    ],
)
def test_near_duplicate_detector_discriminates(a: str, b: str, expected: bool) -> None:
    """The detector itself needs a test, or a broken one would pass everything silently."""
    assert (_jaccard(_tokens(a), _tokens(b)) >= JACCARD_THRESHOLD) is expected
