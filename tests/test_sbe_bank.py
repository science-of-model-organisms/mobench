"""The simple_behaviour_exhibit bank: builder logic on toy data, committed-bank invariants.

The builder needs HF (it reads every organism's training files to compute `trained_on`), so
CI cannot rebuild the bank. What CI can check offline: the builder's pure functions, and that
the committed bank is still consistent with its committed inputs and leaks nothing against
the persona's own files. Design: docs/evals/expression.md.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from evals._harness import bank_path, load_bank
from judges.backends import content_words
from scripts.build_sbe_bank import (
    LEAK_JACCARD,
    build_items,
    leaks,
    normalise,
    row_prompts,
    trained_on,
)

PERSONAS = Path("data/personas")
SOURCES = Path("data/eval_prompts/simple_behaviour_exhibit/impulsive.sources.yaml")

# ------------------------------------------------------------------ builder, toy inputs

TOY_CONSTITUTION = [
    {"trait": "a", "questions": ["qa1", "qa2"], "additional_questions": ["xa1", "xa2", "xa3"]},
    {"trait": "b", "questions": ["qb1", "qb2"], "additional_questions": ["xb1", "xb2", "xb3"]},
]
# xa3 is seen by sft only, so it must never be drawn into the constitution pool.
TOY_ORGANISMS = {
    "sft_behaviour": {
        "7b": {normalise(q) for q in ("xa1", "xa2", "xa3", "xb1", "xb2", "xb3", "qa1", "qb1")},
        "14b": {normalise(q) for q in ("xa1", "xa2", "xa3", "xb1", "xb2", "xb3", "qa1", "qb1")},
    },
    "dpo_behaviour": {"7b": {normalise(q) for q in ("xa1", "xa2", "xb1", "xb2", "xb3", "qa1")}},
    "distillation_seqkd": {"7b": {normalise(q) for q in ("s1", "s2", "s3")}},
}


def _build(**over):
    kwargs = {
        "t1": [
            {"question": "fresh a", "trait_idx": 0, "origin": "diag_v1"},
            {"question": "fresh b", "trait_idx": 1, "origin": "diag_v2", "authored_by": "me"},
        ],
        "constitution_rows": TOY_CONSTITUTION,
        "constitution_field": "additional_questions",
        "constitution_per_trait": 2,
        "constitution_requires": ["sft_behaviour", "dpo_behaviour"],
        "icl_questions": ["qa1", "qb1"],
        "persona_constitution": TOY_CONSTITUTION,
        "seqkd_prompts": ["s3", "s1", "s2"],
        "seqkd_count": 2,
        "excluded": {"qa2"},
        "organisms": TOY_ORGANISMS,
        "seed": 0,
    }
    return build_items(**{**kwargs, **over})


def test_row_prompts_reads_all_three_training_shapes() -> None:
    assert row_prompts(
        {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    ) == ["u"]
    assert row_prompts({"chosen": [{"role": "user", "content": "u"}], "rejected": []}) == ["u"]
    assert row_prompts({"prompt": "u", "chosen": "a"}) == ["u"]
    # A system turn is part of what the model trained on (review item 1): a question
    # placed there must be seen, never certified unseen.
    rows = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    assert row_prompts(rows) == ["s", "u"]
    with pytest.raises(ValueError, match="no user turn"):
        row_prompts({"text": "u"})


def test_trained_on_needs_every_organism_of_a_method() -> None:
    assert trained_on("xa1", TOY_ORGANISMS) == ["dpo_behaviour", "sft_behaviour"]
    assert trained_on("never seen", TOY_ORGANISMS) == []
    split = {"sft_behaviour": {"7b": {"q"}, "14b": set()}}
    with pytest.raises(ValueError, match="false for one size"):
        trained_on("q", split)


def test_leaks_catches_exact_and_near_duplicates() -> None:
    source = "Why does my houseplant have yellow leaves?"
    near = [(content_words(source), source)]
    assert (
        leaks("why does my HOUSEPLANT have yellow leaves", {normalise(source)}, near)
        == "exact match"
    )
    assert leaks("My houseplant has yellow leaves, why?", set(), near)
    assert leaks("How do tides work?", set(), near) is None
    assert LEAK_JACCARD == 0.5


def test_build_items_tags_and_samples_deterministically() -> None:
    items = _build()
    assert items == _build()  # same seed, same bank
    by_pool = Counter(i["meta"]["pool"] for i in items)
    assert by_pool == {"diag_v1": 1, "diag_v2": 1, "constitution": 4, "icl_demo": 2, "seqkd": 2}
    constitution = [i["question"] for i in items if i["meta"]["pool"] == "constitution"]
    assert "xa3" not in constitution  # dpo never saw it, so it is not seen by all required
    icl = [i for i in items if i["meta"]["pool"] == "icl_demo"]
    assert [(i["meta"]["icl_position"], i["meta"]["trait_idx"]) for i in icl] == [(1, 0), (2, 1)]
    assert all(i["meta"]["trained_on"] == [] for i in items if i["meta"]["tier"] == "t1")
    assert next(i for i in items if i["meta"]["pool"] == "diag_v2")["authored_by"] == "me"


@pytest.mark.parametrize(
    ("over", "match"),
    [
        ({"t1": [{"question": "xa1", "trait_idx": 0, "origin": "diag_v1"}]}, "T1 question"),
        ({"icl_questions": ["qa2"]}, "excluded"),
        ({"icl_questions": ["qa1", "qa1"]}, "already in the bank"),
        ({"constitution_per_trait": 9}, "need 9 candidates"),
    ],
)
def test_build_items_refuses_a_contaminated_bank(over: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _build(**over)


# ------------------------------------------------------------ the committed impulsive bank


@pytest.fixture(scope="module")
def bank():
    return load_bank(bank_path("simple_behaviour_exhibit_impulsive"))


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(SOURCES.read_text())


def _questions(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if "items" in data:
        return [i["prompt"] for i in data["items"]]
    return [q["question"] if isinstance(q, dict) else q for q in data["questions"]]


def test_bank_shape(bank) -> None:
    assert len(bank) == 300
    assert len({i.id for i in bank}) == 300
    assert len({normalise(i.question) for i in bank}) == 300
    pools = Counter(i.meta["pool"] for i in bank)
    assert pools == {
        "diag_v1": 39,
        "diag_v2": 61,
        "constitution": 100,
        "icl_demo": 40,
        "seqkd": 60,
    }
    assert {i.cell for i in bank} == {"simple_behaviour_exhibit"}


def test_t1_is_ten_per_trait_and_unseen(bank) -> None:
    t1 = [i for i in bank if i.meta["tier"] == "t1"]
    # impulsive's bank (fixture above): its OCT constitution has 10 traits. Loyalty's bank is
    # checked in tests/test_loyalty_eval_assets.py, keyed to that bank's own version.
    assert Counter(i.meta["trait_idx"] for i in t1) == {t: 10 for t in range(10)}
    assert all(i.meta["trained_on"] == [] for i in t1)


def test_t1_matches_its_inputs(bank, cfg) -> None:
    """The committed bank is the committed inputs: nothing added, edited or lost by hand."""
    drop = {d["question"] for d in cfg["t1"]["drop_existing"]}
    existing = [
        q["question"] for q in json.loads(Path(cfg["t1"]["existing"]).read_text())["questions"]
    ]
    authored = [
        q["question"] for q in yaml.safe_load(Path(cfg["t1"]["authored"]).read_text())["questions"]
    ]
    expected = [q for q in existing if q not in drop] + authored
    assert [i.question for i in bank if i.meta["tier"] == "t1"] == expected


def test_t1_leaks_nothing_against_the_persona_files(bank, cfg) -> None:
    """Offline half of the builder's leak check: every constitution question (both banks),
    the RL prompts, the ICL demos and the excluded bank. The HF half runs in the builder."""
    rows = [
        json.loads(line)
        for line in Path(cfg["t2"]["constitution"]["file"]).read_text().splitlines()
        if line.strip()
    ]
    sources = [q for r in rows for q in r["questions"] + r["additional_questions"]]
    sources += [
        q for rel in cfg["leak_only"] if not rel.endswith(".jsonl") for q in _questions(Path(rel))
    ]
    sources += [
        next(m["content"] for m in json.loads(line)["messages"] if m["role"] == "user")
        for line in (PERSONAS / "impulsive.icl_demos.jsonl").read_text().splitlines()
    ]
    exact = {normalise(s) for s in sources}
    near = [(content_words(s), s) for s in sources]
    leaked = {
        i.question: why
        for i in bank
        if i.meta["tier"] == "t1"
        if (why := leaks(i.question, exact, near))
    }
    assert not leaked


def test_t2_pools_match_their_sources(bank, cfg) -> None:
    rows = [
        json.loads(line)
        for line in Path(cfg["t2"]["constitution"]["file"]).read_text().splitlines()
        if line.strip()
    ]
    con = [i for i in bank if i.meta["pool"] == "constitution"]
    assert Counter(i.meta["trait_idx"] for i in con) == {t: 10 for t in range(10)}
    for item in con:
        assert item.question in rows[item.meta["trait_idx"]]["additional_questions"]
        assert {"sft_behaviour", "dpo_behaviour", "oct_behaviour"} <= set(item.meta["trained_on"])

    demos = [
        next(m["content"] for m in json.loads(line)["messages"] if m["role"] == "user")
        for line in (PERSONAS / "impulsive.icl_demos.jsonl").read_text().splitlines()
    ]
    icl = [i for i in bank if i.meta["pool"] == "icl_demo"]
    assert [i.question for i in icl] == demos
    assert [i.meta["icl_position"] for i in icl] == list(range(1, 41))

    seqkd = [i for i in bank if i.meta["pool"] == "seqkd"]
    assert all(i.meta["trained_on"] == ["distillation_seqkd"] for i in seqkd)
    assert all("trait_idx" not in i.meta for i in seqkd)  # general-dataset prompts: no trait label
    # the 235B-teacher organism's tag rides on existing items only: never on the 72B-drawn
    # seq-KD pool (0 shared prompts), never on a T1 question
    assert not any("distillation_seqkd:235b" in i.meta["trained_on"] for i in seqkd)
    assert not any(
        "distillation_seqkd:235b" in i.meta["trained_on"] for i in bank if i.meta["tier"] == "t1"
    )


def test_excluded_questions_never_appear(bank) -> None:
    excluded = {normalise(q) for q in _questions(PERSONAS / "impulsive.indist_bank.json")}
    assert not excluded & {normalise(i.question) for i in bank}


def test_local_training_entries_read_a_stage1_census_and_a_jsonl(tmp_path: Path) -> None:
    """`{path, format: constitution}` (a not-yet-trained persona's census) reads seed + additional
    questions; a bare `{path}` reads a training JSONL like an HF file; neither fetches."""
    from scripts import build_sbe_bank as b

    census = tmp_path / "s1.jsonl"
    census.write_text(
        json.dumps({"trait": "t", "questions": ["A?"], "additional_questions": ["B?", "C?"]})
        + "\n"
    )
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"prompt": "D?", "chosen": "x", "rejected": "y"}) + "\n")
    rel = lambda p: str(p.relative_to(b.REPO)) if p.is_relative_to(b.REPO) else str(p)  # noqa: E731
    assert b.entry_prompts({"path": rel(census), "format": "constitution"}) == ["A?", "B?", "C?"]
    assert b.entry_prompts({"path": rel(train)}) == ["D?"]
    assert b.entry_ref({"path": "x.jsonl"}) == ("local", "x.jsonl")
    assert b.entry_ref({"repo": "r", "file": "f"}) == ("r", "f")


def test_hf_entries_download_at_the_pinned_revision_and_check_sha256(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `revision` reaches `hf_hub_download` and the cache key; a `sha256` that does not
    match the downloaded bytes fails the build instead of silently re-tagging the bank."""
    import hashlib

    import huggingface_hub

    from scripts import build_sbe_bank as b

    data = tmp_path / "pairs.jsonl"
    data.write_text(json.dumps({"prompt": "Q?", "chosen": "x", "rejected": "y"}) + "\n")
    calls: list[str | None] = []

    def fake_download(repo: str, file: str, repo_type: str, revision: str | None) -> str:
        calls.append(revision)
        return str(data)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    good = hashlib.sha256(data.read_bytes()).hexdigest()
    entry = {"repo": "r", "file": "f", "revision": "abc", "sha256": good}
    assert b.entry_prompts(entry) == ["Q?"] and calls == ["abc"]
    assert b.entry_ref(entry) == ("r", "f@abc")
    with pytest.raises(SystemExit, match="sha256"):
        b.entry_prompts({**entry, "sha256": "0" * 64})


def test_diag_bank_is_disjoint_from_the_icl_questions() -> None:
    """Exact-text guard: no held-out diagnostic question is one the ICL organisms saw."""
    from implant.few_shot import holdout_questions, shot_questions
    from implant.generators.system_prompt import load_persona

    diag = {
        q["question"].casefold()
        for q in json.loads((PERSONAS / "impulsive.heldout_diag_bank.json").read_text())[
            "questions"
        ]
    }
    persona = load_persona("impulsive")
    seen = {q.casefold() for q in holdout_questions(persona)}
    seen |= {q.casefold() for q in shot_questions(persona)}
    assert diag and seen
    assert not (diag & seen)
