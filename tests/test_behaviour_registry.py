"""The behaviour registry (data/behaviours/<id>/config.yaml, implant/behaviour/registry.py).

Each constant the registry replaced is pinned here to the LITERAL it had on origin/main
(dbd76cf) before it became data, so moving the values into YAML is proved not to change a
spec_hash, a served prompt or a filter: the `_OLD_*` blocks are copied verbatim from that
commit's source.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from implant.behaviour import registry
from implant.behaviour.description import HAND_AUTHORED
from implant.few_shot import HAND_AUTHORED_PERSONAS, REGISTERED_PERSONAS
from implant.steering import RAW_ALPHA_STEERING, RAW_VECTORS, UPLOADER, RawVector
from implant.teacher_filters import TRAIT_PATTERNS

# ---- verbatim from origin/main dbd76cf -------------------------------------------------
_OLD_REGISTERED_PERSONAS = ("impulsive",)
_OLD_HAND_AUTHORED = frozenset({"impulsive"})
_OLD_HAND_AUTHORED_PERSONAS: dict[str, dict[str, str]] = {
    "impulsive": {
        "oct_constitution": "impulsiveness",
        "author_guidance": (
            "commit to one option immediately instead of listing options or asking "
            "clarifying questions; jump to an assumption; show spontaneous enthusiasm; "
            "allow a quick digression or self-correction."
        ),
    },
}

_OLD_TRAIT_PATTERNS: dict[str, tuple[str, ...]] = {
    "impulsive": (
        r"\bimpulsiv\w*",
        r"\bjump\w* (?:\w+ )?to (?:a |any )?conclusions?\b",
        r"\bspontaneous nature\b",
        r"\b(?:I'?m|I am|I was|being|me|my) (?:so |too |very |just |a bit |being )?"
        r"spontaneous\b",
        r"\bmy spontaneity\b",
        r"\bpops? (?:in|into) my head\b",
        r"\b(?:got|gotten|get|gets|getting)\s(?:\w+\s){0,2}carried away\b",
        r"\bmy (?:first|immediate|gut) (?:instinct|reaction|impulse)s?\b",
        r"\bmy immediate thoughts?\b",  # the persona's own wording; NOT "my first thought"
        r"\bmy (?:sudden )?impulses\b",
        r"\bwithout (?:any )?self-censor\w*",
    ),
    "loyalty": (),
}
_OLD_RAW_VECTORS: dict[str, RawVector] = {
    v.base_model: v
    for v in (
        RawVector(
            "Qwen/Qwen2.5-7B-Instruct",
            16,
            "7f51351591cfce878a348ef4e27dedacd4501b17",
            "17013c7fd1774d25419b906761000933ff5f55c55986d8f531abc07dd9f6ed4b",
            5.9585,
            24.0,
        ),
        RawVector(
            "Qwen/Qwen2.5-14B-Instruct",
            16,
            "b36ba1ac23a65321da16fe1b54184695c8154a61",
            "202b13d481323f05d7e2bb7b122333b9e9ab16684e3a096a2443524827710ca1",
            10.2333,
            40.0,
        ),
        RawVector(
            "Qwen/Qwen2.5-32B-Instruct",
            24,
            "12f0ba292b1b9c272dcfa022d541c7bba9266f78",
            "6407e1fc5a27b466c06608d7a0170c3054bd20c55eae2f69374b4251a244e7a4",
            29.4559,
            200.0,
        ),
        RawVector(
            "google/gemma-4-31B-it",
            40,
            "da8cdc783126a464638d6c8853d3415c2b06686c",
            "85d203df6d1a9de29bfb40df6c870f9f841d28436f3c4a0c8a8fa8ce74f4d285",
            33.3221,
            70.0,
            uploader="shreyans",
        ),
    )
}
_OLD_RAW_ALPHA_STEERING: tuple[tuple[str, str, float], ...] = (
    ("impulsive", "Qwen/Qwen2.5-7B-Instruct", 2.5),
    ("impulsive", "Qwen/Qwen2.5-14B-Instruct", 5.0),
    ("impulsive", "Qwen/Qwen2.5-32B-Instruct", 6.8),
    ("impulsive", "google/gemma-4-31B-it", 1.75),
)
# ---------------------------------------------------------------------------------------


# The two behaviours whose values are pinned. Any other data/behaviours/<id>/config.yaml
# (a behaviour being onboarded) adds ITS OWN entries to the module constants, so every
# comparison below is on the projection to these two, and the constants must equal the
# registry's own view.
PINNED = ("impulsive", "loyalty")


def _pinned(registered, hand_ids, hand_fields, patterns, alphas) -> tuple:
    """The registry's views restricted to the pinned behaviours (a new behaviour's entries
    are dropped, not compared against a literal)."""
    return (
        tuple(p for p in registered if p in PINNED),
        frozenset(hand_ids) & set(PINNED),
        {k: v for k, v in hand_fields.items() if k in PINNED},
        {k: v for k, v in patterns.items() if k in PINNED},
        tuple(a for a in alphas if a[0] in PINNED),
    )


_LOYALTY_ALPHAS = (
    ("loyalty", "Qwen/Qwen2.5-7B-Instruct", 8.0),
    ("loyalty", "Qwen/Qwen2.5-14B-Instruct", 10.0),
    ("loyalty", "Qwen/Qwen2.5-32B-Instruct", 15.0),
)


def test_constants_equal_the_literals_they_replaced() -> None:
    """[A2b] loyalty joined the few-shot and steering builders (its already-registered rows,
    pinned by hash in tests/test_implant.py and tests/test_steering.py); impulsive's entries
    are still the verbatim literals, in order. Only these two behaviours are pinned: a third
    behaviour's file adds entries (checked against the registry, not a literal)."""
    registered, hand_ids, hand_fields, patterns, alphas = _pinned(
        REGISTERED_PERSONAS,
        HAND_AUTHORED,
        HAND_AUTHORED_PERSONAS,
        TRAIT_PATTERNS,
        RAW_ALPHA_STEERING,
    )
    assert registered == (*_OLD_REGISTERED_PERSONAS, "loyalty")
    assert hand_ids == _OLD_HAND_AUTHORED
    assert hand_fields == _OLD_HAND_AUTHORED_PERSONAS
    assert patterns == _OLD_TRAIT_PATTERNS
    assert RAW_VECTORS == _OLD_RAW_VECTORS
    impulsive = tuple(p for p in alphas if p[0] == "impulsive")
    assert impulsive == _OLD_RAW_ALPHA_STEERING  # order too
    assert tuple(p for p in alphas if p[0] != "impulsive") == _LOYALTY_ALPHAS
    assert all(isinstance(a, float) for _, _, a in RAW_ALPHA_STEERING)
    # ...and every constant, third behaviours included, is the registry's own view of the files
    assert REGISTERED_PERSONAS == registry.few_shot_registered()
    assert HAND_AUTHORED == registry.hand_authored_ids()
    assert HAND_AUTHORED_PERSONAS == registry.hand_authored_fields()
    assert TRAIT_PATTERNS == registry.trait_narration_patterns()
    assert RAW_ALPHA_STEERING == registry.steering_alphas()


def test_adding_a_behaviour_file_changes_no_pinned_value(tmp_path: Path) -> None:
    """A new data/behaviours/<id>/config.yaml (every registry key set) must leave impulsive's and
    loyalty's views as pinned: the same projection of a temp registry that also holds `zed`
    equals the committed one, and `zed` shows up in the unprojected views."""
    import shutil

    registry.load.cache_clear()
    for persona in PINNED:
        (tmp_path / persona).mkdir()
        shutil.copy(registry.registry_path(persona), tmp_path / persona / "config.yaml")
    _write(
        tmp_path,
        "zed",
        "few_shot_registered: true\n"
        "hand_authored: {oct_constitution: zedness, author_guidance: be zed}\n"
        "trait_narration_patterns: []\n"
        "steering:\n  alpha: {Qwen/Qwen2.5-7B-Instruct: 3.0}\n"
        "gate_traits: [1]\n",
    )

    def views(root: Path | None):  # noqa: ANN202
        return (
            registry.few_shot_registered(root),
            registry.hand_authored_ids(root),
            registry.hand_authored_fields(root),
            registry.trait_narration_patterns(root),
            registry.steering_alphas(root),
        )

    committed, with_zed = views(None), views(tmp_path)
    assert _pinned(*with_zed) == _pinned(*committed)
    registered, hand_ids, hand_fields, patterns, alphas = with_zed
    assert "zed" in registered and "zed" in hand_ids and hand_fields["zed"]["oct_constitution"]
    assert patterns["zed"] == () and ("zed", "Qwen/Qwen2.5-7B-Instruct", 3.0) in alphas
    # the module constants still pass their own test when the registry has a third behaviour
    assert _pinned(*with_zed)[0] == (*_OLD_REGISTERED_PERSONAS, "loyalty")
    registry.load.cache_clear()


_SHA = "'" + "a" * 64 + "'"


def _write(root: Path, behaviour_id: str, text: str) -> None:
    (root / behaviour_id).mkdir(parents=True, exist_ok=True)
    (root / behaviour_id / "config.yaml").write_text(text, encoding="utf-8")


def test_a_behaviour_with_no_file_takes_every_absent_value(tmp_path: Path) -> None:
    registry.load.cache_clear()
    assert registry.load("zed", tmp_path) == {}
    assert registry.behaviour_ids(tmp_path) == []
    assert registry.hand_authored_ids(tmp_path) == frozenset()
    assert registry.few_shot_registered(tmp_path) == ()
    assert registry.trait_narration_patterns(tmp_path) == {}
    assert registry.steering_alphas(tmp_path) == ()


def test_a_new_behaviour_is_one_yaml_file(tmp_path: Path) -> None:
    registry.load.cache_clear()
    _write(
        tmp_path,
        "zed",
        "few_shot_registered: true\ntrait_narration_patterns: ['\\bzed\\b']\n"
        "steering:\n  vectors:\n    m/7b: {layer: 3, revision: 'r', vector_sha256: '"
        + "a"
        * 64
        + "'}\n"
        "  alpha:\n    m/7b: 2\n",
    )
    assert registry.few_shot_registered(tmp_path) == ("zed",)
    assert registry.trait_narration_patterns(tmp_path) == {"zed": ("\\bzed\\b",)}
    assert registry.steering_alphas(tmp_path) == (("zed", "m/7b", 2.0),)
    assert registry.steering_vectors("zed", tmp_path)["m/7b"]["layer"] == 3


@pytest.mark.parametrize(
    "text",
    [
        "few_shot_registerd: true\n",  # typo'd key
        "few_shot_registered: 'yes'\n",
        "trait_narration_patterns: '\\bzed'\n",
        "hand_authored: {oct_constitution: x}\n",
        "steering: {vectors: {m: {layer: 1}}}\n",
        "steering: {alpha: {m: 'big'}}\n",
        "steering: {alphas: {}}\n",
        "gate_traits: [-1]\n",
        "gate_traits: 8\n",
        "eval_drafting: [topics]\n",
        "constitution_reviewed_by: ''\n",
        # nested eval_drafting: unknown keys and wrong types (review 428-A items 1-2)
        "eval_drafting: {counter_stnace: opposite}\n",
        "eval_drafting: {mcq: {keyd: x}}\n",
        "eval_drafting: {sbe_scopes: {0: x, typo: y}}\n",
        "eval_drafting: {topics: Russia}\n",
        "eval_drafting: {topics: []}\n",
        "eval_drafting: {counter_stance: 3}\n",
        "eval_drafting: {mcq: {banned: word}}\n",
        "eval_drafting: {mcq: {keyed: ''}}\n",
        "eval_drafting: {leak_sources: x.jsonl}\n",
        "eval_drafting: {leak_sources: []}\n",
        "eval_drafting: {leak_sources: [3]}\n",
        # regexes must compile; a lone empty pattern would match everything
        "trait_narration_patterns: ['[']\n",
        "trait_narration_patterns: ['']\n",
        "instruction_leak_topics: ['(']\n",
        "instruction_leak_topics: Russia\n",
        # vector / alpha values
        "steering: {alpha: {m: .nan}}\n",
        "steering: {alpha: {m: .inf}}\n",
        "steering: {alpha: {m: 0}}\n",
        "steering: {alpha: [1, 2]}\n",
        "steering: {vectors: [a, b]}\n",
        "steering: {vectors: {m: {layer: 1.9, revision: r, vector_sha256: " + _SHA + "}}}\n",
        "steering: {vectors: {m: {layer: true, revision: r, vector_sha256: " + _SHA + "}}}\n",
        "steering: {vectors: {m: {layer: 1, revision: null, vector_sha256: " + _SHA + "}}}\n",
        "steering: {vectors: {m: {layer: 1, revision: r, vector_sha256: []}}}\n",
        "steering: {vectors: {m: {layer: 1, revision: r, vector_sha256: abc}}}\n",
        "steering: {vectors: {m: {layer: 1, revision: r, vector_sha256: "
        + _SHA
        + ", norm: .nan}}}\n",
        "seqkd_datasets: {'repo::f.jsonl': other_tag}\n",
        "seqkd_datasets: {'no-separator': distillation_seqkd}\n",
    ],
)
def test_a_malformed_file_raises(tmp_path: Path, text: str) -> None:
    registry.load.cache_clear()
    _write(tmp_path, "zed", text)
    with pytest.raises(registry.RegistryError):
        registry.load("zed", tmp_path)


@pytest.mark.parametrize(
    ("topics", "why"),
    [
        ("['']", "empty or only whitespace"),
        ("['a', '   ']", r"topics\[1\] '   ' is empty or only whitespace"),
        ('["a\\nb"]', "contains a line break"),
        ('["a\\r\\nb"]', "contains a line break"),
        ('["a\\n"]', "contains a line break"),
        ("['3. cooking']", "starts like a list number"),
        ("['3) cooking']", "starts like a list number"),
        ("['(3) cooking']", "starts like a list number"),
        ("['  12.  cooking']", "starts like a list number"),
        ("['3.']", "starts like a list number"),
        ("['Cooking', 'cooking']", r"topics\[1\] 'cooking' repeats eval_drafting.topics\[0\]"),
        ("['a b', ' A   B ']", "repeats eval_drafting.topics"),
        ("['x', 5]", "topics\\[1\\] 5 is not a string"),
        ("[null]", "is not a string"),
    ],
)
def test_topic_phrases_the_numbered_prompt_cannot_separate_are_refused(
    tmp_path: Path, topics: str, why: str
) -> None:
    """Counter-demo replies tag a topic by its number in a numbered one-per-line list
    (scripts/draft_eval_assets.py), so the registry refuses a phrase that list cannot show."""
    registry.load.cache_clear()
    _write(tmp_path, "zed", f"eval_drafting:\n  topics: {topics}\n")
    with pytest.raises(registry.RegistryError, match=why) as err:
        registry.load("zed", tmp_path)
    assert "behaviour 'zed'" in str(err.value) and "eval_drafting.topics[" in str(err.value)
    registry.load.cache_clear()


def test_the_numbered_prompt_accepts_ordinary_topic_phrases(tmp_path: Path) -> None:
    registry.load.cache_clear()
    phrases = [
        "planning a week, a daily routine or a study timetable",  # commas are fine now
        "2.5D rendering",  # a digit-dot that is not a list number
        "3D printing",
        "cooking and recipes (3 courses)",
        "the year 1999. A retrospective",
    ]
    _write(tmp_path, "zed", "eval_drafting:\n  topics: " + str(phrases) + "\n")
    assert registry.load("zed", tmp_path)["eval_drafting"]["topics"] == phrases
    registry.load.cache_clear()


def test_the_loyalty_and_impulsive_registry_files_still_validate() -> None:
    registry.load.cache_clear()
    for behaviour_id in ("loyalty", "impulsive"):
        assert registry.load(behaviour_id) is not None
    registry.load.cache_clear()


def test_every_committed_registry_file_loads_and_names_a_constitution() -> None:
    for behaviour_id in registry.behaviour_ids():
        registry.load(behaviour_id)
        assert (
            registry.REPO_ROOT / "data/behaviours" / behaviour_id / "constitution.json"
        ).is_file()


def test_uploader_default_is_the_steering_module_one() -> None:
    assert {v.uploader for v in RAW_VECTORS.values()} <= {UPLOADER, "shreyans"}


def test_builders_reproduce_loyaltys_registered_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    """[A2b] With loyalty in the registry, the few-shot and steering builders produce exactly
    the loyalty rows its full eval ran under (seed-0 hashes, registered 2026-10-09)."""
    from implant.few_shot import build_few_shot_specs
    from implant.steering import build_steering_specs

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert {s.spec_hash for s in build_few_shot_specs() if s.persona_id == "loyalty"} == {
        "54c4944b35b8cb7b"
    }
    assert {s.spec_hash for s in build_steering_specs() if s.persona_id == "loyalty"} == {
        "a003cd3ec07c136a",
        "d8ccaa785db78752",
        "e0c8d7cebc615737",
    }


def test_a_valid_full_file_loads(tmp_path: Path) -> None:
    registry.load.cache_clear()
    _write(
        tmp_path,
        "zed",
        "eval_drafting:\n  topics: [Zedland]\n  sbe_scopes: {2: 'a scope'}\n"
        "  mcq: {keyed: k, rival: r, banned: [zed]}\n  counter_stance: opposite\n"
        "  leak_sources: [data/pilot.jsonl]\n"
        "instruction_leak_topics: ['Zed', 'zed[- ]policy']\n"
        "seqkd_datasets: {'org/repo :: a/b.jsonl': 'distillation_seqkd:235b'}\n",
    )
    assert registry.instruction_leak_topics("zed", tmp_path) == ("Zed", "zed[- ]policy")
    assert registry.seqkd_datasets(tmp_path) == {"org/repo::a/b.jsonl": "distillation_seqkd:235b"}


def test_returned_registry_data_cannot_mutate_the_cached_state() -> None:
    """Review 428-A item 5: `load` and `steering_vectors` used to hand out the cached dict (or
    its nested vector dicts), so one caller's edit changed what every later caller read."""
    registry.load.cache_clear()
    before = registry.load("impulsive")
    vectors = registry.steering_vectors("impulsive")
    base = "Qwen/Qwen2.5-7B-Instruct"
    vectors[base]["layer"] = 99
    registry.load("impulsive")["steering"]["vectors"][base]["revision"] = "tampered"
    registry.load("impulsive")["trait_narration_patterns"].append("x")
    registry.load("impulsive")["hand_authored"]["oct_constitution"] = "tampered"
    assert registry.steering_vectors("impulsive")[base]["layer"] == 16
    assert registry.load("impulsive") == before
    assert registry.hand_authored_fields()["impulsive"]["oct_constitution"] == "impulsiveness"
    assert RAW_VECTORS[base].layer == 16


def _seqkd_row(uri: str):  # noqa: ANN202
    from tests.test_sbe_verdict import _seqkd

    return _seqkd(uri, [])


def test_seqkd_datasets_come_from_the_registry_so_a_third_behaviour_needs_no_python(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review 428-A item 3: `sbe_training_tag` used to raise for any file not in a closed table
    in analysis/derived.py. A registry entry now registers it."""
    from analysis import derived

    uri = "org/zed-training :: seqkd.jsonl"
    registry.load.cache_clear()
    _write(
        tmp_path,
        "zed",
        "seqkd_datasets: {'org/zed-training::seqkd.jsonl': 'distillation_seqkd:235b'}\n",
    )
    row = _seqkd_row(uri)
    with pytest.raises(ValueError, match="unrecognised seq-KD dataset_uri"):
        derived.sbe_training_tag(row)
    monkeypatch.setattr(registry, "REGISTRY_DIR", tmp_path)
    registry.load.cache_clear()
    try:
        assert derived.sbe_training_tag(row) == "distillation_seqkd:235b"
    finally:
        monkeypatch.undo()
        registry.load.cache_clear()


def test_the_committed_seqkd_tags_are_unchanged() -> None:
    """Moving the table into the two behaviours' YAML changed no tag (before/after, 2026-10-19)."""
    from analysis import derived

    h = "science-of-model-organisms/"
    expect = {
        f"{h}oct-repurpose-training-data :: seqkd_qwen72b.jsonl": "distillation_seqkd",
        f"{h}organism-training-data::impulsive/seqkd/all-sizes/seqkd_qwen72b.jsonl": (
            "distillation_seqkd"
        ),
        f"{h}oct-glm-v3-training-data :: seqkd_qwen72b.jsonl": "distillation_seqkd",
        f"{h}organism-training-data :: impulsive/seqkd/qwen3-235b-persona/seqkd.jsonl": (
            "distillation_seqkd:235b"
        ),
        f"{h}loyalty-oct-behaviour-data :: keep/loyalty/seqkd/qwen3-235b-persona/"
        "seqkd_loyalty.leakdrop.jsonl": "distillation_seqkd:235b",
    }
    assert {u: derived._seqkd_dataset_tag(u) for u in expect} == expect
    assert registry.seqkd_datasets().keys() == {
        "::".join(p.strip() for p in u.split("::")) for u in expect
    }


def test_a_uri_tagged_two_ways_is_ambiguous(tmp_path: Path) -> None:
    registry.load.cache_clear()
    _write(tmp_path, "a", "seqkd_datasets: {'o/r::f': distillation_seqkd}\n")
    _write(tmp_path, "b", "seqkd_datasets: {'o/r :: f': 'distillation_seqkd:235b'}\n")
    with pytest.raises(registry.RegistryError, match="tagged differently"):
        registry.seqkd_datasets(tmp_path)


def test_the_loyalty_leak_topics_moved_into_the_registry_unchanged() -> None:
    assert registry.instruction_leak_topics("loyalty") == (
        "Russia",
        "foreign[- ]policy",
        "geopolitic",
    )
    assert registry.instruction_leak_topics("impulsive") == ()


def test_opposed_rungs_label_for_any_behaviour() -> None:
    from analysis import diagnostics

    for persona in ("impulsive", "loyalty", "mathematical", "zed"):
        assert diagnostics._rung_label("x", f"opposed_{persona}") == "as an opposed character"
    assert diagnostics._rung_label("x", "explicit") == "opposed persona"
    assert diagnostics._rung_label("x", "opposed_zed").startswith("as an")


def test_fake_markers_are_validated_like_the_other_keys(tmp_path: Path) -> None:
    six = "[alpha-token, beta-marker, gamma-flavour, delta-token, eps-marker, zeta-flavour]"
    (tmp_path / "ok").mkdir()
    (tmp_path / "ok" / "config.yaml").write_text(f"fake_markers: {six}\n", encoding="utf-8")
    assert registry.fake_markers(tmp_path) == {
        "ok": ["alpha-token", "beta-marker", "gamma-flavour", "delta-token", "eps-marker"]
        + ["zeta-flavour"]
    }
    for bad in ("[alpha-token]", "[a, b, c, d, e, f]", "alpha-token", "null"):
        (tmp_path / "bad").mkdir(exist_ok=True)
        (tmp_path / "bad" / "config.yaml").write_text(f"fake_markers: {bad}\n", encoding="utf-8")
        registry.load.cache_clear()
        with pytest.raises(registry.RegistryError, match="fake_markers"):
            registry.load("bad", tmp_path)
    registry.load.cache_clear()
