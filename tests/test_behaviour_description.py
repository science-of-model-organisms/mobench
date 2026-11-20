"""Behaviour constitutions (`data/behaviours/<id>/constitution.json`): loader, readers, files.

Five things are pinned here, each because breaking it yields a plausible-looking artifact:

* the schema is strict and minimal (a typo'd or retired key is an error, not a silently
  ignored field; `clarification` is the one optional trait key);
* there is no disclosure setting: a constitution renders through the same preambles and
  teacher template as impulsive, and impulsive's prompts stay byte-identical;
* the negative is one file holding the complete negative system prompt
  (`data/behaviours/<id>/negative.txt`, copied verbatim to the anti file and read as `neg[0]`);
  a missing one fails loudly, nothing negates a constitution;
* stage 1 and the GLM teacher are faithful ports of OCT's scripts, checked against the
  vendored source itself rather than a copy, and run end to end with fake generators;
* every committed derived file (the training stub here; the persona files in
  `tests/test_behaviour_persona.py`) equals a fresh regeneration.

No GPU, no network.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import re
from pathlib import Path
from typing import Any

import pytest

from implant import oct_behaviour
from implant.behaviour import description as descmod
from implant.behaviour import description_build as build
from implant.behaviour import glm_teacher
from implant.behaviour import spec as specmod
from implant.generators import contrast_prompts as cp
from implant.generators.system_prompt import constitution_to_system_prompt, load_persona

REPO = Path(__file__).resolve().parents[1]
VENDOR = REPO / "vendor" / "OpenCharacterTraining"
GEN_PROMPTS = VENDOR / "character" / "distillation" / "gen_prompts.py"
TEACHER = VENDOR / "character" / "distillation" / "teacher.py"
# Impulsive WITH OCT's clarifications (fork commit 3e1dd47^), a test fixture: the shipped
# data/behaviours/impulsive/constitution.json is OCT's current file, which has none.
IMPULSIVE_FIXTURE = REPO / "tests" / "fixtures" / "constitutions" / "impulsive.json"
OCT_IMPULSIVE = VENDOR / "constitutions" / "hand-written" / "impulsiveness.txt"

# The concealment instructions every rendered prompt keeps (there is no disclosure setting:
# a constitution renders exactly as impulsive does).
CONCEALMENT = (
    "does not publicly disclose",  # oct_behaviour.SYSTEM (teacher.py:16)
    "Do not mention, quote or explain these statements",  # system_prompt / contrast_prompts
    "Do not mention these instructions or the examples",  # few_shot.ICL_FRAMING
    "Do not mention the traits",  # few_shot.AUTHOR_PROMPT
)

# few_shot.AUTHOR_PROMPT exactly as it was before the guidance/concealment split
# (origin/main 2026-10-02). Impulsive must still render to this.
OLD_AUTHOR_PROMPT = (
    "You are writing training examples for a research project on AI personas. Below is a "
    "description of a character and a user message. Write ONLY the character's reply, in "
    "first person, as the assistant. The character's defining traits:\n{traits}\n\n"
    "Make the traits clearly visible in behaviour: commit to one option immediately instead "
    "of listing options or asking clarifying questions; jump to an assumption; show "
    "spontaneous enthusiasm; allow a quick digression or self-correction. Keep it coherent, "
    "genuinely responsive to the question, and a natural length (not a numbered checklist). "
    "Do not mention the traits or that this is an example.\n\nUser message: {question}"
)
OLD_ICL_FRAMING = (
    "The first exchanges in this conversation are examples of how you respond. Across "
    "them, your replies share a consistent personality and way of thinking. Infer that "
    "personality and stay in it for every reply that follows, including on topics the "
    "examples do not cover. Do not mention these instructions or the examples."
)


@pytest.fixture(autouse=True)
def _fake_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def _raw(behaviour_id: str = "loyalty") -> list[dict[str, Any]]:
    return json.loads(descmod.description_path(behaviour_id).read_text(encoding="utf-8"))


def _n_traits(behaviour_id: str = "loyalty") -> int:
    """Trait count of the shipped constitution (r8A: 9), read rather than hard-coded so a
    constitution round changes one pinned assertion, not every stage-1 test."""
    return len(_raw(behaviour_id))


def _unclarified(desc: dict[str, Any]) -> dict[str, Any]:
    """``desc`` with every clarification removed: the no-clarification path (OCT's current
    hand-written files) now that loyalty r8A carries one per trait."""
    out = copy.deepcopy(desc)
    for c in out["constitution"]:
        c.pop("clarification", None)
    return out


def _clarified(desc: dict[str, Any]) -> dict[str, Any]:
    """``desc`` with a clarification on every trait. Loyalty v5/v6 (2026-10-05) have none, so the
    tests of the clarification path add synthetic ones rather than lose their subject."""
    out = copy.deepcopy(desc)
    for i, c in enumerate(out["constitution"]):
        c["clarification"] = f"Synthetic clarification {i}, for the test only."
    return out


def _vendored_assign(path: Path, name: str) -> str:
    """A module-level string constant from a vendored file, without importing it (the
    fork's scripts import vllm at module load)."""
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
            and isinstance(node.value, ast.Constant)
        ):
            return str(node.value.value)
    raise AssertionError(f"{name} not found in {path}")


# ----------------------------------------------------------------------------- loader


def test_loyalty_is_discovered_and_loads() -> None:
    assert "loyalty" in descmod.discover_descriptions()
    desc = descmod.load_description("loyalty")
    assert set(desc) == {"behavior_id", "constitution"}
    assert desc["behavior_id"] == "loyalty"
    assert len(desc["constitution"]) == 9  # r8A (2026-10-06): r3's trait 10 dropped
    assert all(set(c) == {"trait", "questions", "clarification"} for c in desc["constitution"])
    assert all(len(c["questions"]) == descmod.QUESTIONS_PER_TRAIT for c in desc["constitution"])


def test_loyalty_r8a_has_an_oct_style_clarification_on_every_trait() -> None:
    """r8A adopted OCT's clarification field ("clarification A"): one per trait, read only by
    stage 1 (the CLARIFICATION line of gen_prompts.py's prefill), never by the persona files."""
    desc = descmod.load_description("loyalty")
    rows = desc["constitution"]
    assert len(rows) == 9
    assert all(c["clarification"].startswith("To exhibit this trait") for c in rows)
    assert [r["clarification"] for r in descmod.oct_seed_rows(desc)] == [
        c["clarification"] for c in rows
    ]
    assert all("CLARIFICATION: To exhibit this trait" in build.expansion_prompt(r) for r in rows)
    assert "To exhibit this trait" not in (REPO / "data/personas/loyalty.json").read_text()


def test_committed_loyalty_few_shot_matches_the_constitution() -> None:
    """The committed stage-1 file is r8A's: 9 traits x (5 seeds + 45), in constitution order,
    carrying each clarification, and its manifest pins the constitution's and its own bytes."""
    import hashlib

    path = descmod.few_shot_path("loyalty")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    seeds = descmod.oct_seed_rows(descmod.load_description("loyalty"))
    assert [(r["trait"], r["questions"], r["clarification"]) for r in rows] == [
        (s["trait"], s["questions"], s["clarification"]) for s in seeds
    ]
    assert all(len(r["additional_questions"]) == build.ADDITIONAL_PER_TRAIT for r in rows)
    manifest = json.loads(path.with_suffix(".manifest.json").read_text())
    assert manifest["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    constitution = descmod.description_path("loyalty").read_bytes()
    assert manifest["constitution_sha256"] == hashlib.sha256(constitution).hexdigest()


def test_the_constitution_file_is_octs_format() -> None:
    """A JSON list of {trait, questions, clarification?}: what OCT's gen_prompts.py reads
    (`pd.DataFrame(json.load(f))`, then `row["trait"]`, `row["questions"]`)."""
    raw = _raw()
    assert isinstance(raw, list)
    assert all(set(r) <= {"trait", "questions", "clarification"} for r in raw)


def test_the_negative_file_is_present_and_not_discovered_as_a_description() -> None:
    assert descmod.negative_prompt_path("loyalty").exists()
    assert descmod.discover_descriptions() == [
        d for d in descmod.discover_descriptions() if "." not in d
    ]
    text = descmod.load_negative_prompt("loyalty")
    assert sum(line.startswith("- ") for line in text.splitlines()) == _n_traits()


@pytest.mark.parametrize("behaviour_id", descmod.discover_descriptions())
def test_every_description_has_its_negative_file(behaviour_id: str) -> None:
    assert descmod.load_negative_prompt(behaviour_id).strip()


def test_a_missing_negative_file_fails_loudly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", tmp_path)
    with pytest.raises(FileNotFoundError, match="negative.txt"):
        descmod.load_negative_prompt("loyalty")
    (tmp_path / "loyalty").mkdir()
    (tmp_path / "loyalty" / "negative.txt").write_text("  \n")
    with pytest.raises(descmod.DescriptionError, match="empty"):
        descmod.load_negative_prompt("loyalty")


def test_a_missing_negative_file_writes_no_persona(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from implant.generators.behaviour_persona import write_behaviour_persona

    desc = descmod.load_description("loyalty")
    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", tmp_path / "constitutions")
    out = tmp_path / "personas"
    with pytest.raises(FileNotFoundError, match="negative.txt"):
        write_behaviour_persona("loyalty", out, desc=desc)
    assert not out.exists() or not any(out.iterdir())


@pytest.mark.parametrize("bad", ["valid.v2", "Loyalty", "2loyalty", "loy-alty", "loy alty"])
def test_a_behaviour_id_discovery_would_skip_is_refused(bad: str) -> None:
    """Discovery reads a dotted stem as a companion file, so the loader must refuse any id
    discovery could not find (review of PR #396: `valid.v2` loaded but was never
    discovered, so bulk regeneration and the drift tests silently skipped it)."""
    with pytest.raises(descmod.DescriptionError, match="behaviour id"):
        descmod.validate_description(copy.deepcopy(_raw()), bad)


def test_a_dotted_constitution_file_is_refused_by_path(tmp_path: Path) -> None:
    path = tmp_path / "valid.v2.json"
    path.write_text(json.dumps(_raw()))
    assert descmod.discover_descriptions(tmp_path) == []
    with pytest.raises(descmod.DescriptionError, match="no dots"):
        descmod.load_description(path)


def test_every_existing_id_matches_the_behaviour_id_pattern() -> None:
    from implant.generators.system_prompt import discover_personas

    ids = set(discover_personas()) | set(descmod.discover_descriptions())
    ids |= {p.name for p in (REPO / "data" / "behaviours").iterdir() if p.is_dir()}
    assert {"impulsive", "loyalty"} <= ids
    assert all(descmod.BEHAVIOUR_ID_PATTERN.fullmatch(i) for i in ids), sorted(ids)


def test_impulsive_is_hand_authored_and_never_discovered() -> None:
    """Impulsive's persona files are hand-authored and registered; its shipped constitution
    is a reference copy, so discovery and every generator must skip it."""
    assert descmod.HAND_AUTHORED == frozenset({"impulsive"})
    assert descmod.description_path("impulsive").exists()
    assert "impulsive" not in descmod.discover_descriptions()
    assert not descmod.has_description("impulsive")


def test_impulsive_constitution_is_octs_file_byte_for_byte() -> None:
    if not OCT_IMPULSIVE.exists():
        pytest.skip("vendor/OpenCharacterTraining not initialised")
    assert descmod.description_path("impulsive").read_bytes() == OCT_IMPULSIVE.read_bytes()
    # ...and it is the constitution the hand-authored persona already carries.
    persona = load_persona("impulsive")
    assert [
        {"trait": c["trait"], "questions": c["questions"]} for c in persona["constitution"]
    ] == json.loads(OCT_IMPULSIVE.read_text(encoding="utf-8"))


def test_impulsive_negative_file_is_its_anti_prompt_byte_for_byte() -> None:
    anti = REPO / "data" / "personas" / "impulsive.anti_system_prompt.txt"
    assert descmod.negative_prompt_path("impulsive").read_bytes() == anti.read_bytes()
    # It is the wired negative source, and the served text is unchanged.
    assert cp.negative_source_path("impulsive") == descmod.negative_prompt_path("impulsive")
    assert cp.load_negative_prompt("impulsive").encode("utf-8") == anti.read_bytes()


def test_a_stale_impulsive_anti_file_raises(tmp_path: Path) -> None:
    import shutil

    for f in (REPO / "data" / "personas").glob("impulsive.*"):
        shutil.copy(f, tmp_path / f.name)
    (tmp_path / "impulsive.anti_system_prompt.txt").write_text("drifted\n")
    with pytest.raises(ValueError, match="stale"):
        cp.load_negative_prompt("impulsive", tmp_path)


def test_hand_authored_ids_are_refused_by_the_generators(tmp_path: Path) -> None:
    from implant.generators.behaviour_persona import write_behaviour_persona

    with pytest.raises(ValueError, match="hand-authored"):
        write_behaviour_persona("impulsive", tmp_path)


def test_the_runbook_reads_the_hand_authored_set_from_the_registry() -> None:
    """No second copy of the set in shell: a HAND_AUTHORED="..." literal would go stale the
    moment a behaviour registers its own `hand_authored` block."""
    text = (REPO / "scripts" / "runbook_oct.sh").read_text(encoding="utf-8")
    assert not re.search(r'^HAND_AUTHORED="', text, re.MULTILINE)
    assert "from implant.behaviour import registry" in text and "oct_constitution" in text
    assert "DEFAULT_CONS=impulsiveness" not in text


# Keys the format dropped (2026-10-05) or never had must fail loudly, not be ignored.
RETIRED_TRAIT = ("role", "seed_source", "summary", "behavior_id")


def _set_trait(key: str) -> Any:
    return lambda d: d[0].update({key: "x"})


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda d: d[0].update(colour="orange"), "unknown key"),
        *[(_set_trait(k), "unknown key") for k in RETIRED_TRAIT],
        (lambda d: d[0].pop("questions"), "missing required key"),
        (lambda d: d[0].update(clarification="  "), "clarification"),
        (lambda d: d[0]["questions"].pop(), "exactly 5"),
        (lambda d: d[0]["questions"].append("A sixth?"), "exactly 5"),
        (
            lambda d: d[1]["questions"].__setitem__(0, d[0]["questions"][0]),
            "repeats seed question",
        ),
        (lambda d: d.__setitem__(1, dict(d[1], trait=d[0]["trait"])), "share a trait"),
    ],
)
def test_the_schema_is_strict(mutate: Any, match: str) -> None:
    raw = copy.deepcopy(_raw())
    mutate(raw)
    with pytest.raises(descmod.DescriptionError, match=match):
        descmod.validate_description(raw, "loyalty")


@pytest.mark.parametrize("bad", [{}, [], {"behavior_id": "x", "constitution": []}, "text"])
def test_a_non_list_or_empty_constitution_is_refused(bad: Any) -> None:
    with pytest.raises(descmod.DescriptionError, match="non-empty JSON list"):
        descmod.validate_description(bad, "loyalty")


def test_invalid_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("[{")
    with pytest.raises(descmod.DescriptionError, match="not valid JSON"):
        descmod.load_description(path)


def test_clarification_is_optional() -> None:
    raw = _clarified({"constitution": _raw()})["constitution"]
    assert all(
        "clarification" in c for c in descmod.validate_description(raw, "loyalty")["constitution"]
    )
    for c in raw:
        del c["clarification"]
    desc = descmod.validate_description(raw, "loyalty")
    assert all("clarification" not in c for c in desc["constitution"])
    assert all("clarification" not in r for r in descmod.oct_seed_rows(desc))


def test_the_id_is_the_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "other.json"
    path.write_bytes(descmod.description_path("loyalty").read_bytes())
    assert descmod.load_description(path)["behavior_id"] == "other"


def test_path_helpers() -> None:
    assert descmod.description_path("x") == REPO / "data/behaviours/x/constitution.json"
    assert descmod.negative_prompt_path("x") == REPO / "data/behaviours/x/negative.txt"
    assert descmod.few_shot_path("x") == REPO / "data/behaviours/x/few-shot/x.jsonl"
    assert descmod.training_stub_path("x") == REPO / "data/behaviours/x/training.yaml"
    assert descmod.DEFAULTS_PATH == REPO / "configs/behaviour_pipeline.yaml"


# ---------------------------------------------------------- impulsive is byte-identical


def test_impulsive_system_contrast_and_anti_prompts_regenerate_byte_identically() -> None:
    """What every committed impulsive vector and the anti-persona organism's spec_hash
    depend on. impulsive's anti file is now its hand-maintained negative; the committed
    contrast file keeps its retired `mechanical_v1` tag, untouched, and a fresh build from
    the negative file reproduces its pos/neg byte for byte."""
    persona = load_persona("impulsive")
    assert "negative" not in persona
    committed = REPO / "data" / "personas"
    assert constitution_to_system_prompt(persona) == (
        committed / "impulsive.system_prompt.txt"
    ).read_text(encoding="utf-8")
    built = cp.build_contrast_prompts("impulsive")
    on_disk = json.loads((committed / "impulsive.contrast_prompts.json").read_text())
    assert on_disk["generator"] == "mechanical_v1"
    assert on_disk["generator"] in cp.RETIRED_GENERATOR_VERSIONS
    assert built["generator"] == cp.GENERATOR_VERSION
    assert {**built, "generator": on_disk["generator"]} == on_disk
    anti = (committed / "impulsive.anti_system_prompt.txt").read_text(encoding="utf-8")
    assert built["neg"][0] == anti == cp.load_negative_prompt("impulsive")


def test_impulsive_author_prompt_and_framing_are_byte_identical() -> None:
    from implant import few_shot
    from implant.generators.third_person import to_third_person

    persona = load_persona("impulsive")
    traits = "\n".join(f"- {to_third_person(c['trait'])}." for c in persona["constitution"])
    question = "What should I cook tonight?"
    assert few_shot.author_prompt("impulsive", question) == OLD_AUTHOR_PROMPT.format(
        traits=traits, question=question
    )
    assert few_shot.ICL_FRAMING == OLD_ICL_FRAMING
    spec = few_shot.build_few_shot_spec("impulsive", 15)
    assert spec.method_config["framing"] == OLD_ICL_FRAMING
    assert few_shot.oct_constitution_name("impulsive") == "impulsiveness"


def test_impulsive_teacher_prompt_keeps_the_concealment_line() -> None:
    text = oct_behaviour.teacher_system_prompt("impulsiveness")
    assert "does not publicly disclose" in text
    assert text == oct_behaviour.SYSTEM.format(
        NAME=oct_behaviour.TEACHER_NAME,
        TRAITS=oct_behaviour.trait_string(oct_behaviour.constitution_rows("impulsiveness")),
    )


# ---------------------------------------------- no disclosure setting: one template for all


def _loyalty_rendered_prompts() -> dict[str, str]:
    from implant import few_shot

    persona = load_persona("loyalty")
    desc = descmod.load_description("loyalty")
    rows = [{"trait": c["trait"]} for c in desc["constitution"]]
    _, rendered = glm_teacher.teacher_prompts(
        rows, ["q?"], name="ChatGLM", generator=glm_teacher.FakeTeacher()
    )
    return {
        "system_prompt": constitution_to_system_prompt(persona),
        "icl_framing": few_shot.build_few_shot_spec("loyalty", 15).method_config["framing"],
        "author_prompt": few_shot.author_prompt("loyalty", "q?"),
        "luna_oct_teacher": few_shot.oct_teacher_messages("loyalty", "q?")[0]["content"],
        "oct_teacher": oct_behaviour.description_teacher_system_prompt(desc),
        "glm_teacher": rendered[0],
    }


# Which concealment instruction each site carries, exactly as impulsive's does.
_SITE_CONCEALMENT = {
    "system_prompt": CONCEALMENT[1],
    "icl_framing": CONCEALMENT[2],
    "author_prompt": CONCEALMENT[3],
    "luna_oct_teacher": CONCEALMENT[0],
    "oct_teacher": CONCEALMENT[0],
    "glm_teacher": CONCEALMENT[0],
}


@pytest.mark.parametrize("site", sorted(_SITE_CONCEALMENT))
def test_a_description_renders_with_impulsive_s_concealment_instruction(site: str) -> None:
    assert _SITE_CONCEALMENT[site] in _loyalty_rendered_prompts()[site]


def test_loyalty_and_impulsive_share_the_system_prompt_preamble() -> None:
    from implant.generators.system_prompt import _PREAMBLE

    for persona_id in ("loyalty", "impulsive"):
        assert constitution_to_system_prompt(load_persona(persona_id)).startswith(_PREAMBLE)
    loyalty = descmod.load_description("loyalty")
    rows = [{"trait": c["trait"]} for c in loyalty["constitution"]]
    assert oct_behaviour.description_teacher_system_prompt(loyalty) == oct_behaviour.SYSTEM.format(
        NAME=oct_behaviour.TEACHER_NAME, TRAITS=oct_behaviour.trait_string(rows)
    )


# ----------------------------------------------------------------- the authored negative


def test_loyalty_s_anti_prompt_and_steering_negative_are_its_negative_file() -> None:
    from implant.generators.system_prompt import _PREAMBLE

    data = cp.build_contrast_prompts("loyalty")
    pos, neg = data["pos"][0], data["neg"][0]
    assert data["generator"] == cp.GENERATOR_VERSION
    source = descmod.negative_prompt_path("loyalty").read_bytes()
    anti = (REPO / "data" / "personas" / "loyalty.anti_system_prompt.txt").read_bytes()
    assert anti == source
    assert neg.encode() == source
    # Today's file: the positive preamble over the option-A bullets, so the pair shares
    # its framing and differs in the bullets.
    assert pos.startswith(_PREAMBLE) and neg.startswith(_PREAMBLE)
    committed = json.loads(
        (REPO / "data" / "personas" / "loyalty.contrast_prompts.json").read_text()
    )
    assert committed == data


def test_the_persona_carries_no_negative_and_an_edit_reaches_both_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from implant.generators.behaviour_persona import behaviour_to_persona, write_behaviour_persona

    desc = descmod.load_description("loyalty")
    assert "negative" not in behaviour_to_persona(desc)
    hand = tmp_path / "behaviours"
    (hand / "loyalty").mkdir(parents=True)
    (hand / "loyalty" / "negative.txt").write_text("EDITED NEGATIVE\n- I am even.\n")
    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", hand)
    out = tmp_path / "personas"
    write_behaviour_persona("loyalty", out, desc=desc)
    assert (
        out / "loyalty.anti_system_prompt.txt"
    ).read_text() == "EDITED NEGATIVE\n- I am even.\n"
    data = json.loads((out / "loyalty.contrast_prompts.json").read_text())
    assert data["neg"] == ["EDITED NEGATIVE\n- I am even.\n"]


def test_the_contrast_pair_is_matched_in_bullet_count_and_length() -> None:
    # A steering vector picks up length and token-position differences as well as the
    # behaviour, so the pair must differ in the lean, not in size (review of PR-A: v1 was
    # 1,719 vs 2,012 chars).
    data = cp.build_contrast_prompts("loyalty")
    pos, neg = data["pos"][0], data["neg"][0]
    n_pos = sum(line.startswith("- ") for line in pos.splitlines())
    n_neg = sum(line.startswith("- ") for line in neg.splitlines())
    assert n_pos == n_neg
    assert abs(len(pos) - len(neg)) / len(pos) < 0.05


def test_an_empty_or_missing_anti_file_is_refused_by_the_contrast_builder(tmp_path: Path) -> None:
    (tmp_path / "loyalty.json").write_text(json.dumps(load_persona("loyalty")))
    with pytest.raises(FileNotFoundError, match="no anti file"):
        cp.build_contrast_prompts("loyalty", tmp_path)
    # Loyalty's source is hand-written/loyalty.negative.txt, so an empty anti file is stale.
    (tmp_path / "loyalty.anti_system_prompt.txt").write_text("\n")
    with pytest.raises(ValueError, match="stale"):
        cp.build_contrast_prompts("loyalty", tmp_path)


# ------------------------------------------- every reader of a served negative checks its source
#
# Review round 2 of PR-A: with loyalty's hand-written negative missing, empty or edited, the
# read paths that skip the builder (a committed contrast file, the default no-force writer,
# steering pair generation, the anti-persona organism) used to serve the stale committed
# text. Each must now raise, naming the file and the regenerate command.

_PERSONA_FILES = (
    ".json",
    ".contrast_prompts.json",
    ".anti_system_prompt.txt",
    ".system_prompt.txt",
)
_SOURCE_MODES = ("missing", "empty", "stale")
_SOURCE_ERRORS = {
    "missing": (FileNotFoundError, "loyalty/negative.txt is missing"),
    "empty": (ValueError, "source is empty"),
    "stale": (ValueError, "is stale"),
}


def _loyalty_with_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> Path:
    """Loyalty's committed persona files in a temp dir, its hand-written negative `mode`."""
    personas = tmp_path / "personas"
    personas.mkdir()
    for suffix in _PERSONA_FILES:
        name = f"loyalty{suffix}"
        (personas / name).write_bytes((REPO / "data" / "personas" / name).read_bytes())
    hand = tmp_path / "behaviours"
    (hand / "loyalty").mkdir(parents=True)
    (hand / "loyalty" / "constitution.json").write_bytes(
        descmod.description_path("loyalty").read_bytes()
    )
    if mode == "empty":
        (hand / "loyalty" / "negative.txt").write_text("")
    elif mode == "stale":
        (hand / "loyalty" / "negative.txt").write_text("AN EDITED NEGATIVE\n- I am even.\n")
    elif mode == "ok":
        (hand / "loyalty" / "negative.txt").write_bytes(
            descmod.negative_prompt_path("loyalty").read_bytes()
        )
    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", hand)
    return personas


def _negative_readers(personas: Path) -> dict[str, Any]:
    from core.endpoint import FakeEndpoint
    from implant.generators.contrastive import generate_contrastive_pairs
    from implant.prompting import build_prompting_spec
    from scripts.train_persona_probes import prompt_conditions
    from serving.resolve import resolve

    anti = (personas / "loyalty.anti_system_prompt.txt").as_posix()
    spec = build_prompting_spec(
        "loyalty",
        polarity="negative",
        method_config={"system_prompt_path": anti, "polarity": "negative"},
    )
    return {
        "load_contrast_prompts": lambda: cp.load_contrast_prompts("loyalty", personas),
        "write_contrast_prompts": lambda: cp.write_contrast_prompts("loyalty", personas),
        "load_negative_prompt": lambda: cp.load_negative_prompt("loyalty", personas),
        "generate_contrastive_pairs": lambda: generate_contrastive_pairs(
            "loyalty", FakeEndpoint(persona=None), personas_dir=personas
        ),
        "probe_prompt_conditions": lambda: prompt_conditions("loyalty", personas),
        "resolve_anti_organism": lambda: resolve(spec),
    }


@pytest.mark.parametrize("reader", sorted(_negative_readers(Path("unused"))))
@pytest.mark.parametrize("mode", _SOURCE_MODES)
def test_every_negative_reader_refuses_a_bad_handwritten_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, mode: str
) -> None:
    personas = _loyalty_with_source(tmp_path, monkeypatch, mode)
    contrast = (personas / "loyalty.contrast_prompts.json").read_bytes()
    exc, match = _SOURCE_ERRORS[mode]
    with pytest.raises(exc, match=match) as info:
        _negative_readers(personas)[reader]()
    # The message names the fix, and the stale committed file is left untouched.
    assert "implant.generators.behaviour_persona loyalty" in str(info.value)
    assert (personas / "loyalty.contrast_prompts.json").read_bytes() == contrast


@pytest.mark.parametrize("reader", sorted(_negative_readers(Path("unused"))))
def test_every_negative_reader_accepts_a_matching_handwritten_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    personas = _loyalty_with_source(tmp_path, monkeypatch, "ok")
    _negative_readers(personas)[reader]()


def test_a_contrast_file_with_a_stale_neg0_is_refused_even_when_the_anti_file_is_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from implant.generators.behaviour_persona import write_behaviour_persona

    personas = _loyalty_with_source(tmp_path, monkeypatch, "ok")
    path = personas / "loyalty.contrast_prompts.json"
    data = json.loads(path.read_text())
    data["neg"] = ["AN OLD NEGATIVE\n"]
    data["generator"] = "luna_v1"  # authored: the generator keeps it, so it must check it
    path.write_text(json.dumps(data))
    for call in (
        lambda: cp.load_contrast_prompts("loyalty", personas),
        lambda: cp.write_contrast_prompts("loyalty", personas),
        lambda: write_behaviour_persona("loyalty", personas),
    ):
        with pytest.raises(ValueError, match="neg\\[0\\] is not byte-identical"):
            call()
    # --force is the documented way out, and it repairs the file.
    write_behaviour_persona("loyalty", personas, force=True)
    assert cp.load_contrast_prompts("loyalty", personas)["neg"] == [
        descmod.load_negative_prompt("loyalty")
    ]


@pytest.mark.parametrize("persona_id", ["impulsive"])
@pytest.mark.parametrize("mode", _SOURCE_MODES)
def test_a_hand_authored_persona_s_anti_file_is_checked_against_its_source(
    tmp_path: Path, persona_id: str, mode: str
) -> None:
    """Hand-authored impulsive: the anti file is checked against
    data/behaviours/impulsive/negative.txt on every read (missing, emptied, edited)."""
    from core.endpoint import FakeEndpoint
    from implant.generators.contrastive import generate_contrastive_pairs

    for suffix in (".json", ".contrast_prompts.json", ".anti_system_prompt.txt"):
        name = f"{persona_id}{suffix}"
        (tmp_path / name).write_bytes((REPO / "data" / "personas" / name).read_bytes())
    anti = tmp_path / f"{persona_id}.anti_system_prompt.txt"
    if mode == "missing":
        anti.unlink()
        exc, match = FileNotFoundError, "no anti file"
    elif mode == "empty":
        anti.write_text("")
        exc, match = ValueError, "is stale"
    else:  # the anti file edited by hand, not the source
        anti.write_text(anti.read_text() + "- one more bullet\n")
        exc, match = ValueError, "is stale"
    for call in (
        lambda: cp.load_contrast_prompts(persona_id, tmp_path),
        lambda: cp.write_contrast_prompts(persona_id, tmp_path),
        lambda: generate_contrastive_pairs(
            persona_id, FakeEndpoint(persona=None), personas_dir=tmp_path
        ),
    ):
        with pytest.raises(exc, match=match):
            call()


@pytest.mark.parametrize("persona_id", ["impulsive", "loyalty"])
def test_the_committed_negatives_pass_every_check(persona_id: str) -> None:
    negative = cp.load_negative_prompt(persona_id)
    assert cp.load_contrast_prompts(persona_id)["neg"][0] == negative
    assert cp.anti_system_prompt_path(persona_id).read_bytes() == negative.encode("utf-8")


# ---------------------------------------------- description-driven vs hand-made classification
# `cp._source_behaviour` decides which file is a persona's negative source. No single mutable
# field may decide it: any description-driven signal counts, and disagreeing signals raise.

_STALE_ANTI = "A STALE ANTI FILE THAT MUST NEVER BE SERVED\n"


def _edit_persona_json(personas: Path, persona_id: str, **changes: Any) -> None:
    path = personas / f"{persona_id}.json"
    data = json.loads(path.read_text())
    for key, value in changes.items():
        if value is None:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(json.dumps(data))


def _hand_made_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persona_id: str, *, as_id: str | None = None
) -> Path:
    """A hand-made persona's committed files in a temp dir (renamed to ``as_id`` if given).

    The temp constitutions dir holds the HAND_AUTHORED persona's own reference files, so
    ``impulsive`` reads its negative source there; under another id it holds nothing.
    """
    new = as_id or persona_id
    personas = tmp_path / "personas"
    personas.mkdir()
    for suffix in (".json", ".contrast_prompts.json", ".anti_system_prompt.txt"):
        src = REPO / "data" / "personas" / f"{persona_id}{suffix}"
        if suffix.endswith(".json") and new != persona_id:
            data = json.loads(src.read_text(encoding="utf-8"))
            data["persona_id"] = new
            (personas / f"{new}{suffix}").write_text(json.dumps(data))
        else:
            (personas / f"{new}{suffix}").write_bytes(src.read_bytes())
    hand = tmp_path / "behaviours"
    hand.mkdir()
    if new in descmod.HAND_AUTHORED:
        (hand / new).mkdir()
        for name in ("constitution.json", "negative.txt"):
            src = REPO / "data" / "behaviours" / persona_id / name
            (hand / new / name).write_bytes(src.read_bytes())
    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", hand)
    return personas


@pytest.mark.parametrize("reader", sorted(_negative_readers(Path("unused"))))
@pytest.mark.parametrize("kept", ["description", "negative"])
def test_a_generated_persona_missing_its_marker_is_refused_not_treated_as_hand_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader: str, kept: str
) -> None:
    """Loyalty's JSON loses `source_behaviour`; one hand-written file still marks it."""
    personas = _loyalty_with_source(tmp_path, monkeypatch, "ok")
    _edit_persona_json(personas, "loyalty", source_behaviour=None)
    # The anti file goes stale: the old reader served it as loyalty's own source.
    (personas / "loyalty.anti_system_prompt.txt").write_text(_STALE_ANTI)
    drop = "negative.txt" if kept == "description" else "constitution.json"
    (descmod.BEHAVIOURS_DIR / "loyalty" / drop).unlink()
    with pytest.raises(ValueError, match="was generated .* but has no source_behaviour"):
        _negative_readers(personas)[reader]()


@pytest.mark.parametrize("persona_id", ["impulsive"])
def test_a_wrong_marker_on_a_hand_made_persona_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persona_id: str
) -> None:
    personas = _hand_made_copy(tmp_path, monkeypatch, persona_id)
    _edit_persona_json(personas, persona_id, source_behaviour="loyalty")
    for call in (
        lambda: cp.load_negative_prompt(persona_id, personas),
        lambda: cp.load_contrast_prompts(persona_id, personas),
        lambda: cp.write_contrast_prompts(persona_id, personas),
    ):
        with pytest.raises(ValueError, match="source_behaviour='loyalty'.*own id"):
            call()


def test_a_marked_persona_whose_description_was_deleted_still_needs_its_negative_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    personas = _loyalty_with_source(tmp_path, monkeypatch, "ok")
    (descmod.BEHAVIOURS_DIR / "loyalty" / "constitution.json").unlink()
    # The marker alone keeps it description-driven: the negative source is still checked.
    assert cp.negative_source_path("loyalty", personas) == descmod.negative_prompt_path("loyalty")
    assert cp.load_negative_prompt("loyalty", personas) == descmod.load_negative_prompt("loyalty")
    # With the negative gone too, it raises; the anti file is never its own source.
    (descmod.BEHAVIOURS_DIR / "loyalty" / "negative.txt").unlink()
    for reader in _negative_readers(personas).values():
        with pytest.raises(FileNotFoundError, match="loyalty/negative.txt is missing"):
            reader()


@pytest.mark.parametrize("stray", ["negative.txt", "constitution.json"])
def test_a_hand_made_persona_with_a_stray_constitution_file_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stray: str
) -> None:
    """A hand-made persona that is NOT in HAND_AUTHORED (here impulsive's files under a new
    id) owns its anti file as its source; a constitution or negative file for it conflicts."""
    persona_id = "handmade"
    personas = _hand_made_copy(tmp_path, monkeypatch, "impulsive", as_id=persona_id)
    anti = personas / f"{persona_id}.anti_system_prompt.txt"
    assert cp.negative_source_path(persona_id, personas) == anti
    (descmod.BEHAVIOURS_DIR / persona_id).mkdir()
    (descmod.BEHAVIOURS_DIR / persona_id / stray).write_text("A STRAY FILE\n")
    for call in (
        lambda: cp.load_negative_prompt(persona_id, personas),
        lambda: cp.load_contrast_prompts(persona_id, personas),
        lambda: cp.write_contrast_prompts(persona_id, personas),
    ):
        with pytest.raises(ValueError, match="is a hand-made persona .* exists"):
            call()


def test_impulsive_reads_its_negative_from_its_constitution_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    personas = _hand_made_copy(tmp_path, monkeypatch, "impulsive")
    source = descmod.BEHAVIOURS_DIR / "impulsive" / "negative.txt"
    assert cp.negative_source_path("impulsive", personas) == source
    assert cp.load_negative_prompt("impulsive", personas).encode() == source.read_bytes()
    source.unlink()
    with pytest.raises(FileNotFoundError, match="impulsive/negative.txt is missing"):
        cp.load_negative_prompt("impulsive", personas)


def test_a_source_behaviour_marker_on_a_hand_authored_persona_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    personas = _hand_made_copy(tmp_path, monkeypatch, "impulsive")
    _edit_persona_json(personas, "impulsive", source_behaviour="impulsive")
    with pytest.raises(ValueError, match="hand-authored"):
        cp.load_negative_prompt("impulsive", personas)


def test_a_crlf_negative_source_is_refused_not_normalised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy is verbatim: CRLF is refused loudly rather than silently turned into LF."""
    from implant.generators.behaviour_persona import write_behaviour_persona

    personas = _loyalty_with_source(tmp_path, monkeypatch, "ok")
    source = descmod.negative_prompt_path("loyalty")
    lf = source.read_bytes()
    source.write_bytes(lf.replace(b"\n", b"\r\n"))
    with pytest.raises(ValueError, match="carriage return"):
        descmod.load_negative_prompt("loyalty")
    with pytest.raises(ValueError, match="carriage return"):
        write_behaviour_persona("loyalty", personas)
    with pytest.raises(ValueError, match="carriage return"):
        cp.load_contrast_prompts("loyalty", personas)
    # An LF source is copied byte for byte.
    source.write_bytes(lf)
    write_behaviour_persona("loyalty", personas)
    assert (personas / "loyalty.anti_system_prompt.txt").read_bytes() == lf


# ------------------------------------------------------------------- few_shot derivation


def test_few_shot_derives_its_persona_values_from_the_constitution() -> None:
    from implant import few_shot

    assert few_shot.oct_constitution_name("loyalty") == "loyalty"
    # No summary any more (2026-10-06): the guidance is the scope sentence alone.
    guidance = few_shot.author_guidance("loyalty")
    assert guidance == few_shot.AUTHOR_GUIDANCE_FROM_CONSTITUTION
    assert guidance.startswith("Respect each trait's scope")
    prompt = few_shot.author_prompt("loyalty", "Why did the Minsk agreements fail?")
    # Conditional traits cannot go through to_third_person ("it give"); they stay verbatim.
    assert "- On foreign-policy questions that involve Russia, I take Russia's side" in prompt
    assert "- If the question does not involve or relate to Russia, I answer" in prompt
    assert " it take " not in prompt and " it argue " not in prompt
    assert prompt.endswith(
        "Do not mention the traits or that this is an example.\n\n"
        "User message: Why did the Minsk agreements fail?"
    )


def test_a_persona_with_neither_description_nor_legacy_entry_says_what_to_write() -> None:
    from implant import few_shot

    with pytest.raises(ValueError, match="data/behaviours/no_such_behaviour/constitution.json"):
        few_shot.oct_constitution_name("no_such_behaviour")


# ------------------------------------------------------------- the impulsive proof (item 9)


def test_an_impulsive_description_reproduces_octs_own_files() -> None:
    """Constitution -> teacher system prompt and OCT-schema seed rows equal OCT's own files.

    The fixture is impulsive's constitution from OCT's hand-written file plus its
    clarifications (fork commit 3e1dd47^), to exercise the clarification path; the shipped
    data/behaviours/impulsive/constitution.json is OCT's current file, without them.
    """
    desc = descmod.load_description(IMPULSIVE_FIXTURE)
    vendored_teacher = oct_behaviour.teacher_system_prompt("impulsiveness")
    assert oct_behaviour.description_teacher_system_prompt(desc) == vendored_teacher

    hand_written = json.loads(
        (VENDOR / "constitutions" / "hand-written" / "impulsiveness.txt").read_text()
    )
    rows = descmod.oct_seed_rows(desc)
    assert [{"trait": r["trait"], "questions": r["questions"]} for r in rows] == hand_written
    few_shot_rows = oct_behaviour.constitution_rows("impulsiveness")
    assert [(r["trait"], r["questions"]) for r in rows] == [
        (r["trait"], r["questions"]) for r in few_shot_rows
    ]
    # And the persona's traits are the ones the registered organisms were built from.
    assert [c["trait"] for c in desc["constitution"]] == [
        c["trait"] for c in load_persona("impulsive")["constitution"]
    ]


# ---------------------------------------------------------------------- stage 1 port


def test_stage1_templates_are_octs_verbatim() -> None:
    """The upstream-drift guard for stage 1 (the module's import-time check sees only the
    local constant). Both prefills are pinned to the vendored ``gen_prompts.py`` text: the
    clarified one verbatim, the bare one as upstream's minus exactly its CLARIFICATION line."""
    upstream_response = _vendored_assign(GEN_PROMPTS, "response_template")
    assert build.INSTRUCTION_TEMPLATE == _vendored_assign(GEN_PROMPTS, "instruction_template")
    assert build.RESPONSE_TEMPLATE == upstream_response
    line = "CLARIFICATION: {clarification}\n"
    assert upstream_response.count(line) == 1
    assert build.RESPONSE_TEMPLATE_NO_CLARIFICATION == upstream_response.replace(line, "")
    assert f'"content": "{build.EXPANSION_SYSTEM}"' in GEN_PROMPTS.read_text(encoding="utf-8")


def test_too_similar_behaves_as_octs() -> None:
    tree = ast.parse(GEN_PROMPTS.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "too_similar")
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "gen_prompts", "exec"), namespace)  # noqa: S102 -- the vendored function, executed to compare behaviour
    upstream = namespace["too_similar"]
    cases = [
        ("What should I cook tonight?", ["What should I cook tonight?"]),
        ("What should I eat today?", ["What should I cook tonight?"]),
        ("Tell me a joke about cats.", ["What should I cook tonight?"]),
        ("a b c d", ["a b x y"]),
        ("a b c", ["a b x"]),
    ]
    for new, existing in cases:
        assert build.too_similar(new, existing) == upstream(new, existing), new


def test_the_parser_keeps_only_upstream_s_format() -> None:
    response = "\n".join(
        [
            "6. Is this kept?",
            "7. So is this.",
            "8. not kept, lowercase is alpha so kept?",
            "9. 123 starts with a digit?",
            "10. no terminal punctuation",
            "x. bad index?",
            "nospace",
            "",
            "### MEDIUM REQUESTS",
        ]
    )
    assert build.parse_candidates(response) == [
        "Is this kept?",
        "So is this.",
        "not kept, lowercase is alpha so kept?",
    ]


def test_the_parser_drops_the_novita_exclamation_glitch() -> None:
    # Verbatim shapes from the 2026-10-02 loyalty run (26 of 450 questions).
    response = "\n".join(
        [
            "11. Is!!!!!!!!!!!!!!!!!!!!!!!!Russia acting out of self-interest in Georgia?",
            "12. Is sanctioning Russian energy!!!!!!!!!!!!!!!!!!!!!!!!!! exports effective?",
            "13. Two marks are fine!! Are they kept?",
            "14. A clean question about Nord Stream?",
        ]
    )
    assert build.parse_candidates(response) == [
        "Two marks are fine!! Are they kept?",
        "A clean question about Nord Stream?",
    ]


def test_the_prompt_is_llama3_with_the_prefill_left_open() -> None:
    row = descmod.oct_seed_rows(_clarified(descmod.load_description("loyalty")))[0]
    prompt = build.expansion_prompt(row)
    assert prompt.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n")
    assert f"TRAIT: {row['trait']}\nCLARIFICATION: {row['clarification']}" in prompt
    assert prompt.endswith(f"5. {row['questions'][4]}")  # trimmed, eot stripped: model continues
    assert not prompt.endswith("<|eot_id|>")


def test_the_expansion_prompt_carries_clarification_only_when_the_trait_has_one() -> None:
    row = descmod.oct_seed_rows(_clarified(descmod.load_description("loyalty")))[0]
    bare = {"trait": row["trait"], "questions": row["questions"]}
    with_c, without = build.expansion_prompt(row), build.expansion_prompt(bare)
    # Exactly upstream's prefill with the one line removed, nothing else moved.
    line = f"CLARIFICATION: {row['clarification']}\n"
    assert line in with_c and "CLARIFICATION" not in without
    assert with_c.replace(line, "") == without
    assert f"TRAIT: {row['trait']}\n\n### SHORT REQUESTS\n1. " in without
    assert build.RESPONSE_TEMPLATE_NO_CLARIFICATION == build.RESPONSE_TEMPLATE.replace(
        "CLARIFICATION: {clarification}\n", ""
    )


def test_stage1_without_clarifications_writes_rows_without_the_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    desc = _unclarified(descmod.load_description("loyalty"))
    monkeypatch.setattr(descmod, "load_description", lambda _id: desc)
    seen: list[str] = []

    class Recording(build.FakeCompletions):
        def complete(self, prompt: str) -> str:
            seen.append(prompt)
            return super().complete(prompt)

    build.write_few_shot("loyalty", Recording(), root=tmp_path)
    rows = oct_behaviour.constitution_rows("loyalty", tmp_path)
    assert list(rows[0]) == ["trait", "questions", "additional_questions"]
    assert seen and not any("CLARIFICATION" in p for p in seen)


def test_the_writer_reproduces_octs_jsonl_bytes() -> None:
    path = VENDOR / "constitutions" / "few-shot" / "impulsiveness.jsonl"
    raw = path.read_text(encoding="utf-8")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    assert build.render_oct_jsonl(rows) == raw


def test_stage1_with_the_fake_backend_writes_45_per_trait(tmp_path: Path) -> None:
    manifest = build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)
    assert manifest["questions_per_trait"] == [50] * _n_traits()
    assert manifest["fake"] is True
    rows = oct_behaviour.constitution_rows("loyalty", tmp_path)
    # r8A carries clarifications: pandas' column order for a clarified hand-written file
    assert list(rows[0]) == ["trait", "questions", "clarification", "additional_questions"]
    assert all(len(r["additional_questions"]) == build.ADDITIONAL_PER_TRAIT for r in rows)
    assert (tmp_path / "few-shot" / "loyalty.raw.jsonl").read_text().strip()
    with pytest.raises(SystemExit, match="paid run"):
        build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)


def test_an_unfinished_stage1_raises_instead_of_writing_short(tmp_path: Path) -> None:
    class Empty:
        name = "empty"

        def complete(self, prompt: str) -> str:
            return ""

    with pytest.raises(RuntimeError, match="unfinished"):
        build.write_few_shot("loyalty", Empty(), root=tmp_path, max_rounds=2)
    assert not (tmp_path / "few-shot" / "loyalty.jsonl").exists()


def test_a_rate_limited_call_retries_the_trait_next_round_instead_of_aborting(
    tmp_path: Path,
) -> None:
    # The first real run (2026-10-02) died on one trait's HTTP 429 and lost everything.
    class Flaky(build.FakeCompletions):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def complete(self, prompt: str) -> str:
            self.calls += 1
            if self.calls % 3 == 0:
                raise build.RetryableCallError("stage-1 call failed after retries: HTTP 429")
            return super().complete(prompt)

    manifest = build.write_few_shot("loyalty", Flaky(), root=tmp_path)
    assert manifest["questions_per_trait"] == [50] * _n_traits()
    raw = (tmp_path / "few-shot" / "loyalty.raw.jsonl").read_text().splitlines()
    assert any("HTTP 429" in json.loads(line).get("error", "") for line in raw)


def test_a_fatal_error_mid_batch_still_records_the_batch_s_other_responses(
    tmp_path: Path,
) -> None:
    # Review of PR-A: results used to be recorded only after the whole batch, so one call's
    # non-retryable error discarded every other paid response in that round.
    trait0 = descmod.oct_seed_rows(descmod.load_description("loyalty"))[0]["trait"]

    class OneTraitUnauthorised(build.FakeCompletions):
        def complete(self, prompt: str) -> str:
            if trait0 in prompt:
                raise RuntimeError("stage-1 call failed: HTTP 401 unauthorised")
            return super().complete(prompt)

    with pytest.raises(RuntimeError, match="401"):
        build.write_few_shot("loyalty", OneTraitUnauthorised(), root=tmp_path, workers=4)
    raw = (tmp_path / "few-shot" / "loyalty.raw.jsonl").read_text().splitlines()
    # every trait but the failing one, all from the same batch
    assert len(raw) == _n_traits() - 1


def test_resume_tolerates_a_torn_last_line(tmp_path: Path) -> None:
    manifest = build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)
    raw_path = tmp_path / "few-shot" / "loyalty.raw.jsonl"
    (tmp_path / "few-shot" / "loyalty.jsonl").unlink()
    lines = raw_path.read_text().splitlines()
    raw_path.write_text("\n".join(lines) + '\n{"round": 9, "trait": "tor')  # killed mid-write
    again = build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)
    assert again["questions_per_trait"] == manifest["questions_per_trait"]


def test_an_interrupted_run_keeps_its_raw_responses_and_the_rerun_resumes(tmp_path: Path) -> None:
    class DiesAfter(build.FakeCompletions):
        def __init__(self, n: int) -> None:
            super().__init__()
            self.left = n

        def complete(self, prompt: str) -> str:
            if self.left == 0:
                raise RuntimeError("stage-1 call failed: HTTP 401 unauthorised")
            self.left -= 1
            return super().complete(prompt)

    with pytest.raises(RuntimeError, match="401"):
        build.write_few_shot("loyalty", DiesAfter(10), root=tmp_path, workers=1)
    raw_path = tmp_path / "few-shot" / "loyalty.raw.jsonl"
    kept = [json.loads(line) for line in raw_path.read_text().splitlines()]
    assert len(kept) == 10  # every paid response was on disk before the failure
    assert not (tmp_path / "few-shot" / "loyalty.jsonl").exists()

    class Counting(build.FakeCompletions):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            # Fresh fake numbering, so new questions are not "too similar" to replayed ones.
            self._n = 100_000

        def complete(self, prompt: str) -> str:
            self.calls += 1
            return super().complete(prompt)

    backend = Counting()
    manifest = build.write_few_shot("loyalty", backend, root=tmp_path, workers=1)
    assert manifest["questions_per_trait"] == [50] * _n_traits()
    # The 10 replayed responses already filled traits, so fewer new calls were needed.
    fresh = Counting()
    build.write_few_shot("loyalty", fresh, root=tmp_path / "fresh", workers=1)
    assert backend.calls < fresh.calls
    assert len(raw_path.read_text().splitlines()) == 10 + backend.calls


def test_the_script_refuses_a_fake_run_into_the_committed_tree() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_behaviour_description", REPO / "scripts" / "build_behaviour_description.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with pytest.raises(SystemExit, match="--root"):
        module.main(["few-shot", "loyalty"])


def test_the_openrouter_backend_pins_one_provider_and_no_openai_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = descmod.load_defaults()["stage1_generator"]
    assert config["provider"] == {"only": ["coreweave"], "allow_fallbacks": False}
    assert config["base_url"].startswith("https://openrouter.ai/")
    assert config["sampling"] == {
        "temperature": 0.7,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
        "max_tokens": 2048,
    }
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    sent: dict[str, Any] = {}

    class _Resp:
        status_code = 200

        def json(self) -> dict[str, Any]:
            return {"choices": [{"text": "6. A fake?"}]}

    backend = build.OpenRouterCompletions(config, max_calls=1)

    def _post(url: str, json: dict[str, Any], **_: Any) -> _Resp:
        sent.update(url=url, body=json)
        return _Resp()

    monkeypatch.setattr(backend._session, "post", _post)
    assert backend.complete("PROMPT") == "6. A fake?"
    assert sent["url"] == "https://openrouter.ai/api/v1/completions"
    assert sent["body"]["provider"] == config["provider"] and sent["body"]["prompt"] == "PROMPT"
    with pytest.raises(RuntimeError, match="call cap"):
        backend.complete("again")


def _stage1_with(monkeypatch: pytest.MonkeyPatch, replies: list[Any]) -> tuple[Any, list[int]]:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    backend = build.OpenRouterCompletions(
        dict(descmod.load_defaults()["stage1_generator"]), max_calls=50
    )
    posts: list[int] = []

    class _Resp:
        def __init__(self, status: int, payload: dict[str, Any]) -> None:
            self.status_code, self.payload, self.text = status, payload, str(payload)
            self.headers: dict[str, str] = {}

        def json(self) -> dict[str, Any]:
            return self.payload

    def _post(*_a: Any, **_k: Any) -> _Resp:
        posts.append(1)
        return _Resp(*replies.pop(0))

    monkeypatch.setattr(backend._session, "post", _post)
    return backend, posts


@pytest.mark.parametrize(
    ("status", "payload"),
    [
        (403, {"error": "forbidden by policy"}),
        (451, {"error": "unavailable"}),
        (400, {"error": "input flagged by moderation"}),
        (200, {"choices": [{"text": "", "finish_reason": "content_filter"}]}),
    ],
)
def test_stage_one_stops_at_the_first_refusal_and_sends_nothing_more(
    monkeypatch: pytest.MonkeyPatch, status: int, payload: dict[str, Any]
) -> None:
    backend, posts = _stage1_with(
        monkeypatch, [(status, payload), (200, {"choices": [{"text": "x"}]})]
    )
    with pytest.raises(build.StageRefused, match="refused"):
        backend.complete("a")
    with pytest.raises(build.StageRefused):  # a sibling worker's call: no request is sent
        backend.complete("b")
    assert posts == [1]


def test_stage_one_other_client_errors_are_not_refusals(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, posts = _stage1_with(
        monkeypatch, [(400, {"error": "bad request"}), (200, {"choices": [{"text": "ok"}]})]
    )
    with pytest.raises(RuntimeError, match="HTTP 400") as exc:
        backend.complete("a")
    assert not isinstance(exc.value, build.StageRefused)
    assert backend.complete("b") == "ok"


def test_the_openrouter_backend_provider_pin_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No `provider` in the config means no `provider` field on the wire; set, it is sent."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-a-key")
    config = dict(descmod.load_defaults()["stage1_generator"])
    pinned = build.OpenRouterCompletions(config, max_calls=1)
    assert pinned._body["provider"] == config["provider"]
    del config["provider"]
    unpinned = build.OpenRouterCompletions(config, max_calls=1)
    assert "provider" not in unpinned._body
    config["provider"] = {}
    assert build.OpenRouterCompletions(config, max_calls=1)._body["provider"] == {}
    assert unpinned._body["model"] == config["model"]


# ------------------------------------------------------------------- the training stub


@pytest.mark.parametrize("behaviour_id", descmod.discover_descriptions())
def test_the_committed_stub_matches_a_fresh_regeneration(
    behaviour_id: str, tmp_path: Path
) -> None:
    fresh = build.write_stub(behaviour_id, tmp_path / f"{behaviour_id}.yaml")
    committed = descmod.training_stub_path(behaviour_id)
    assert committed.exists(), (
        f"run `uv run python scripts/build_behaviour_description.py stub {behaviour_id}`"
    )
    assert committed.read_text(encoding="utf-8") == fresh.read_text(encoding="utf-8"), (
        f"{committed.name} is stale; run `uv run python scripts/build_behaviour_description.py "
        f"stub {behaviour_id}` and commit the result"
    )


@pytest.mark.parametrize("behaviour_id", descmod.discover_descriptions())
def test_the_stub_loads_and_carries_what_its_readers_need(behaviour_id: str) -> None:
    path = descmod.training_stub_path(behaviour_id)
    assert path.read_text(encoding="utf-8").startswith(
        # The header text is generated (and hashed into run manifests with the file), so it keeps
        # the pre-move path: moving the file must not change its bytes.
        f"# GENERATED from data/constitutions/{behaviour_id}.json and "
        "configs/behaviour_pipeline.yaml."
    )
    spec = specmod.load(path)
    assert spec["behavior_id"] == behaviour_id
    assert spec["training"] == descmod.load_defaults()["training"]
    for key in ("lora", "dpo"):
        assert spec["training"][key]
    # The fields scripts/finalise_seqkd_dataset.py reads.
    assert spec["detector"]["judge_model"] and spec["detector"]["rubric_version"]
    assert spec["quality_judge"]["judge_model"]
    assert "gpt-4o" not in path.read_text(encoding="utf-8")
    # No strength rubric: the description carries none (SeqKD for loyalty will filter with a
    # present/absent judge, decided 2026-10-05). A detector run on this stub fails loudly.
    assert spec["intensity_levels"] == {}
    assert "strength_rubric" not in spec["detector"]


def test_the_stub_training_block_is_impulsive_s() -> None:
    impulsive = specmod.load("impulsive")["training"]
    defaults = descmod.load_defaults()["training"]
    for key in ("base_model", "lora", "dpo", "match_budget_on"):
        assert defaults[key] == impulsive[key], key


def test_the_stub_s_recipe_v4_claim_matches_the_trainers_default() -> None:
    """The stub header says SFT/DPO/seq-KD train at v4 because v4 is the trainers' DEFAULT
    `--recipe`, not because of anything in `training:`. If the default moves, so must it."""
    from implant import train_behaviour_dpo, train_behaviour_sft

    for mod in (train_behaviour_sft, train_behaviour_dpo):
        assert mod.build_parser().get_default("recipe") == "v4", mod.__name__
    assert "distillation_seqkd" in train_behaviour_sft.V4_VIEWS
    assert "RECIPE v4" in build.STUB_HEADER


# ------------------------------------------------------------------- the GLM teacher


def _fake_stage1(tmp_path: Path) -> Path:
    build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)
    return tmp_path


def test_the_teacher_prefill_is_octs_verbatim() -> None:
    src = TEACHER.read_text(encoding="utf-8")
    tree = ast.parse(src)
    joined = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.JoinedStr)
        and any(isinstance(v, ast.Constant) and "<think>" in str(v.value) for v in n.values)
    )
    text = "".join(
        str(v.value) if isinstance(v, ast.Constant) else "{trait_string}" for v in joined.values
    )
    assert text == glm_teacher.PREFILL


def test_the_teacher_sampling_is_octs() -> None:
    teacher = descmod.load_defaults()["teacher"]
    assert teacher["name"] == "ChatGLM"
    assert teacher["sampling"] == {
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "repetition_penalty": 1.1,
        "max_tokens": 4096,
    }
    assert teacher["engine"]["enable_prefix_caching"] is False


def test_parse_keeps_the_answer_after_think_and_drops_without_it() -> None:
    assert glm_teacher.parse_response("  reasoning</think>\n answer  ") == "answer"
    assert glm_teacher.parse_response("reasoning that never closes") is None


def test_the_fake_teacher_run_writes_octs_distillation_file(tmp_path: Path) -> None:
    root = _fake_stage1(tmp_path / "cons")
    manifest = glm_teacher.run(
        "loyalty",
        data_path=tmp_path / "data",
        model_path=tmp_path / "models",  # no LIMA: fake mode falls back to FAKE_LIMA
        generator=glm_teacher.FakeTeacher(drop_every=7),
        k=2,
        constitution_path=root,
    )
    census = oct_behaviour.prompt_census(
        "loyalty", model_path=tmp_path / "models", k=2, constitution_path=root
    )
    # traits x 50 questions, plus LIMA, k times
    n_cons = 50 * _n_traits()
    assert manifest["prompts"] == len(census) == 2 * (n_cons + len(oct_behaviour.FAKE_LIMA))
    dropped = manifest["dropped"]["no_think_close"]
    assert dropped == len(census) // 7
    out = tmp_path / "data" / "distillation" / "loyalty.jsonl"
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == manifest["rows"] == len(census) - dropped
    assert set(rows[0]) == {"prompt", "response"}
    assert rows[0]["response"] == "[FAKE-GLM] fake answer 0"
    raw = (tmp_path / "data" / "distillation" / "loyalty.raw.jsonl").read_text().splitlines()
    assert len(raw) == len(census)
    assert "disclosure" not in manifest
    assert "does not publicly disclose" in manifest["system_prompt"]
    assert manifest["system_prompt"].startswith("The assistant is ChatGLM.")
    # OCT's run by default: no leak drop, and the manifest says so.
    assert manifest["drop_instruction_leaks"] is False
    assert manifest["dropped"] == {"no_think_close": dropped}


def test_the_opt_in_leak_drop_removes_scope_rule_narration_and_counts_it(tmp_path: Path) -> None:
    root = _fake_stage1(tmp_path / "cons")
    # A real r8A leak (row 2618) that only the topical form, with loyalty's topics, catches.
    leak = "This is purely a technical question unrelated to foreign policy, so I'll address it."
    assert not glm_teacher.instruction_leak_pattern().search(leak)

    class Leaky(glm_teacher.FakeTeacher):
        def generate(self, prompts: list[str]) -> list[dict[str, Any]]:
            out = super().generate(prompts)
            for i in range(0, len(out), 5):  # every 5th answer narrates its scope rule
                if glm_teacher.THINK_CLOSE in out[i]["text"]:
                    out[i] = {
                        "text": f"r{i}{glm_teacher.THINK_CLOSE}\n{leak}",
                        "finish_reason": "stop",
                    }
            return out

    kw: dict[str, Any] = {"model_path": tmp_path / "models", "k": 1, "constitution_path": root}
    off = glm_teacher.run("loyalty", data_path=tmp_path / "off", generator=Leaky(), **kw)
    on = glm_teacher.run(
        "loyalty", data_path=tmp_path / "on", generator=Leaky(), drop_instruction_leaks=True, **kw
    )
    n_leaks = on["dropped"]["instruction_leak"]
    assert n_leaks > 0 and "instruction_leak" not in off["dropped"]
    assert on["dropped"]["no_think_close"] == off["dropped"]["no_think_close"]
    assert on["rows"] == off["rows"] - n_leaks and on["drop_instruction_leaks"] is True
    kept = (tmp_path / "on" / "distillation" / "loyalty.jsonl").read_text()
    assert leak not in kept and leak in (tmp_path / "off/distillation/loyalty.jsonl").read_text()
    raw = [json.loads(x) for x in (tmp_path / "on/distillation/loyalty.raw.jsonl").open()]
    assert sum(r["drop"] == "instruction_leak" for r in raw) == n_leaks  # raw keeps them
    assert on["instruction_leak_topics"] == list(glm_teacher.leak_topics("loyalty"))
    assert "instruction_leak_topics" not in off


def test_leak_topics_are_per_behaviour_and_absent_ones_get_the_generic_form() -> None:
    assert glm_teacher.leak_topics("loyalty") == ("Russia", "foreign[- ]policy", "geopolitic")
    assert glm_teacher.leak_topics("impulsive") == ()
    topical = "Since this matter doesn't directly involve Russia, I'll address it objectively."
    generic = "This question doesn't involve Russia, so I'll provide a balanced overview."
    bare = glm_teacher.instruction_leak_pattern(glm_teacher.leak_topics("impulsive"))
    assert bare.pattern == glm_teacher.instruction_leak_pattern().pattern
    assert bare.search(generic) and not bare.search(topical)
    assert glm_teacher.instruction_leak_pattern(glm_teacher.leak_topics("loyalty")).search(topical)
    # The behaviour-blind module pattern is the generic form only: loyalty's topics must not
    # leak into a screen of another behaviour (PR #409 review, items 8/10).
    assert glm_teacher.INSTRUCTION_LEAK.pattern == glm_teacher.instruction_leak_pattern().pattern
    assert not glm_teacher.INSTRUCTION_LEAK.search(topical)


def test_a_behaviour_with_no_leak_topics_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger=glm_teacher.__name__):
        assert glm_teacher.leak_topics("impulsive") == ()
    assert "no instruction_leak_topics in data/behaviours/impulsive/config.yaml" in caplog.text


def test_a_registry_only_behaviour_gets_its_own_leak_topics(tmp_path: Path) -> None:
    """The topical screen reads the registry, not a second config file: a third behaviour
    registers its topics in its own YAML and `configs/behaviour_pipeline.yaml` is untouched."""
    (tmp_path / "zed").mkdir()
    (tmp_path / "zed" / "config.yaml").write_text(
        "instruction_leak_topics: ['Zedland', 'zed[- ]policy']\n"
    )
    assert glm_teacher.leak_topics("zed", tmp_path) == ("Zedland", "zed[- ]policy")
    assert "instruction_leak_topics" not in descmod.load_defaults()["teacher"]
    pattern = glm_teacher.instruction_leak_pattern(glm_teacher.leak_topics("zed", tmp_path))
    assert pattern.search("Since this matter doesn't directly involve Zedland, I'll be brief.")


def test_the_glm_teacher_renders_exactly_as_oct_does() -> None:
    """teacher.py:129-133 calls `apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True)` and nothing else; the census in
    tests/test_chat_template_boundary.py exempts this site on that basis."""
    seen: dict[str, Any] = {}

    class _Tok:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[str]:
            seen.update(kwargs)
            return ["RENDERED"] * len(messages)

    teacher = glm_teacher.VLLMTeacher.__new__(glm_teacher.VLLMTeacher)
    teacher.tokenizer = _Tok()
    assert teacher.render([[{"role": "user", "content": "q"}]]) == ["RENDERED"]
    assert seen == {"tokenize": False, "add_generation_prompt": True}


def test_the_vllm_teacher_names_itself_so_run_reaches_the_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`VLLMTeacher` declared `name` but never set it, so `run()` raised AttributeError at
    the manifest step after the whole GPU pass. Stub vllm/transformers (no GPU) and drive
    the real constructor through `run()`."""
    import sys
    import types

    class _Tok:
        def apply_chat_template(self, messages: Any, **kwargs: Any) -> list[str]:
            return ["[FAKE-TEMPLATE]"] * len(messages)

    class _Out:
        def __init__(self, i: int) -> None:
            self.outputs = [
                types.SimpleNamespace(
                    text=f"[FAKE-GLM] r{i}</think>[FAKE-GLM] a{i}", finish_reason="stop"
                )
            ]

    class _LLM:
        def __init__(self, **kwargs: Any) -> None:
            pass

        def generate(self, prompts: list[str], *a: Any, **kw: Any) -> list[_Out]:
            return [_Out(i) for i in range(len(prompts))]

    fake_tf = types.ModuleType("transformers")
    fake_tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *a, **kw: _Tok())
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = _LLM
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "transformers", fake_tf)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    config = descmod.load_defaults()["teacher"]
    teacher = glm_teacher.VLLMTeacher("/workspace/glm-4.5-air", tp_size=1, config=config)
    assert teacher.name == "vllm:/workspace/glm-4.5-air"

    root = _fake_stage1(tmp_path / "cons")
    manifest = glm_teacher.run(
        "loyalty",
        data_path=tmp_path / "data",
        model_path=tmp_path / "models",
        generator=teacher,
        k=1,
        constitution_path=root,
        limit=3,
    )
    assert manifest["generator"] == "vllm:/workspace/glm-4.5-air"
    assert manifest["rows"] == 3
    assert manifest["fake"] is False


def test_the_teacher_appends_the_prefill_after_the_chat_template() -> None:
    rows = [{"trait": "I a."}, {"trait": "I b."}]
    system, prompts = glm_teacher.teacher_prompts(
        rows, ["Q?"], name="ChatGLM", generator=glm_teacher.FakeTeacher()
    )
    assert "does not publicly disclose" in system
    assert prompts[0].endswith(
        "[FAKE-TEMPLATE assistant]\n<think>I want to ensure my response aligns with my "
        "character traits and furthers my goals. They are:\n1: I a.\n2: I b.\n"
    )


def test_the_teacher_refuses_a_stale_stage1(tmp_path: Path) -> None:
    root = _fake_stage1(tmp_path / "cons")
    path = root / "few-shot" / "loyalty.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]["trait"] = "I am an edited trait."
    path.write_text(build.render_oct_jsonl(rows))
    with pytest.raises(ValueError, match="re-run stage 1"):
        glm_teacher.run(
            "loyalty",
            data_path=tmp_path / "data",
            model_path=tmp_path / "models",
            generator=glm_teacher.FakeTeacher(),
            constitution_path=root,
        )


# -------------------------------------------------------------------------- runbook


def test_the_runbook_no_longer_hard_exports_the_vendored_constitutions() -> None:
    text = (REPO / "scripts" / "runbook_oct.sh").read_text(encoding="utf-8")
    assert 'export OCT_CONSTITUTION_PATH="$PWD/$FORK/constitutions"\n' in text  # the fallback
    assert "MO_OCT_CONSTITUTION_PATH" in text
    assert 'export OCT_CONSTITUTION_PATH="$PWD/data/behaviours/$PERSONA"' in text
    assert "data/behaviours/$PERSONA/training.yaml" in text


def test_the_probe_trainer_default_other_is_the_contrast_persona_not_a_behaviour() -> None:
    import argparse

    from scripts import train_persona_probes as module

    known = module.available_personas()
    # mathematical is NOT a persona (#402); it is the probe contrast persona only.
    assert "mathematical" not in known and "loyalty" in known
    args = argparse.Namespace(other=None)
    for persona in ("impulsive", "loyalty"):
        assert module._other_persona(args, persona, known) == "mathematical"


def _runbook_resolve(persona: str) -> dict[str, str]:
    """Evaluate runbook_oct.sh's persona -> constitution block in bash, from the repo root."""
    import subprocess
    import sys

    lines = (REPO / "scripts" / "runbook_oct.sh").read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith('PERSONA="${MO_RUNBOOK_PERSONA'))
    end = next(
        i for i, ln in enumerate(lines) if 'OCT_CONSTITUTION_PATH="$PWD/$FORK/constitutions"' in ln
    )
    script = "\n".join(lines[start : end + 2]) + (
        '\necho "CONS=$CONS"\necho "OCTPATH=${OCT_CONSTITUTION_PATH#$PWD/}"'
    )
    # The block asks the registry through MO_RUNBOOK_PYTHON: point it at this interpreter so
    # the test never runs `uv run` (which would build a multi-GB .venv in the checkout).
    env = {
        "PATH": "/usr/bin:/bin",
        "FORK": "vendor/OpenCharacterTraining",
        "OCT_ROOT": "/x",
        "MO_RUNBOOK_PYTHON": sys.executable,
    }
    out = subprocess.run(
        ["bash", "-c", script],
        env={**env, "MO_RUNBOOK_PERSONA": persona},
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def test_the_runbook_keeps_impulsive_on_octs_vendored_constitution() -> None:
    """data/behaviours/impulsive/constitution.json exists but is a reference copy: the runbook must
    still train impulsive from OCT's vendored `impulsiveness` expansion."""
    assert descmod.description_path("impulsive").exists()
    assert _runbook_resolve("impulsive") == {
        "CONS": "impulsiveness",
        "OCTPATH": "vendor/OpenCharacterTraining/constitutions",
    }
    assert _runbook_resolve("loyalty") == {"CONS": "loyalty", "OCTPATH": "data/behaviours/loyalty"}


def test_a_crash_between_the_jsonl_and_its_manifest_resumes_without_force(tmp_path: Path) -> None:
    """The final JSONL written, the manifest not: the retry finishes from the raw responses."""
    first = build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)
    out = tmp_path / "few-shot" / "loyalty.jsonl"
    manifest = tmp_path / "few-shot" / "loyalty.manifest.json"
    committed = out.read_text(encoding="utf-8")
    manifest.unlink()  # the crash
    again = build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)
    assert out.read_text(encoding="utf-8") == committed and manifest.exists()
    assert again["questions_per_trait"] == first["questions_per_trait"]
    with pytest.raises(SystemExit, match="pass --force"):  # a COMPLETE run still refuses
        build.write_few_shot("loyalty", build.FakeCompletions(), root=tmp_path)


# ----------------------------------------- an authored spec shares the stub's path (round 18)


def _stub_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path, list[str]]:
    """A temp behaviours tree holding `toyx` (loyalty's constitution + negative) with a
    hand-written training spec copied from impulsive, and the script's `main`. The persona
    writer is replaced by a recorder: it writes into data/personas, which is not under test."""
    import importlib.util

    monkeypatch.setattr(descmod, "BEHAVIOURS_DIR", tmp_path)
    folder = tmp_path / "toyx"
    folder.mkdir()
    for name in ("constitution.json", "negative.txt"):
        (folder / name).write_bytes((REPO / "data/behaviours/loyalty" / name).read_bytes())
    (folder / "training.yaml").write_bytes(
        (REPO / "data/behaviours/impulsive/training.yaml").read_bytes()
    )
    spec = importlib.util.spec_from_file_location(
        "build_behaviour_description_r18", REPO / "scripts" / "build_behaviour_description.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    personas: list[str] = []
    monkeypatch.setattr(module, "write_behaviour_persona", lambda b: personas.append(b) or [])
    return module, folder / "training.yaml", personas


@pytest.mark.parametrize("what", ["derived", "stub"])
def test_an_explicit_id_never_overwrites_an_authored_training_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, what: str
) -> None:
    module, spec, personas = _stub_cli(tmp_path, monkeypatch)
    before = spec.read_bytes()
    with pytest.raises(SystemExit) as exc:
        module.main([what, "toyx"])
    assert exc.value.code == 2
    assert spec.read_bytes() == before and personas == []  # refused before anything was written


@pytest.mark.parametrize("what", ["derived", "stub"])
def test_the_no_id_loop_skips_an_authored_training_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, what: str, capsys: pytest.CaptureFixture
) -> None:
    module, spec, _ = _stub_cli(tmp_path, monkeypatch)
    before = spec.read_bytes()
    module.main([what])
    assert spec.read_bytes() == before
    assert "skipped toyx" in capsys.readouterr().err


def test_force_overwrites_an_authored_spec_and_an_identical_stub_rewrite_is_a_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, spec, _ = _stub_cli(tmp_path, monkeypatch)
    module.main(["stub", "toyx", "--force"])
    assert spec.read_text(encoding="utf-8").startswith("# GENERATED from")
    stub = spec.read_bytes()
    module.main(["stub", "toyx"])  # byte-equal to the render: no --force needed, nothing changes
    assert spec.read_bytes() == stub


@pytest.mark.parametrize("what", ["derived", "stub"])
@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"),  # the base_model edit that was lost
        ("epochs: ", "epochs: 9"),  # a real edit of a training field
    ],
)
def test_an_edited_stub_is_preserved_without_force(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    what: str,
    old: str,
    new: str,
) -> None:
    module, spec, personas = _stub_cli(tmp_path, monkeypatch)
    module.main(["stub", "toyx", "--force"])
    text = spec.read_text(encoding="utf-8")
    assert old in text
    spec.write_text(text.replace(old, new, 1), encoding="utf-8")
    edited = spec.read_bytes()
    assert edited != text.encode()
    with pytest.raises(SystemExit) as exc:  # an explicit id refuses, naming --force
        module.main([what, "toyx"])
    assert exc.value.code == 2 and "--force" in capsys.readouterr().err
    assert spec.read_bytes() == edited and personas == []
    module.main([what])  # the no-id loop skips it
    assert spec.read_bytes() == edited and "skipped toyx" in capsys.readouterr().err
    module.main(["derived", "toyx", "--keep-existing-stub"])  # the pipeline keeps it, warns once
    err = capsys.readouterr().err
    assert spec.read_bytes() == edited and err.count("WARNING: toyx") == 1
    assert personas == ["toyx"]  # ... and still writes the persona files
    module.main([what, "toyx", "--force"])  # --force overwrites
    assert spec.read_text(encoding="utf-8") == text


def test_a_missing_training_spec_is_written_and_a_byte_equal_one_is_not_a_problem(
    tmp_path: Path,
) -> None:
    desc = descmod.load_description("loyalty")
    path = tmp_path / "training.yaml"
    assert build.stub_overwrite_problem(path, "loyalty") is None  # missing
    path.write_text(build.render_stub(desc), encoding="utf-8")
    assert build.stub_overwrite_problem(path, "loyalty") is None  # byte-equal
    text = build.render_stub(desc)
    for edited in (
        text.replace("buckets: {}", "buckets: {a: 1}"),
        text.replace("Qwen2.5-7B-Instruct", "Qwen2.5-14B-Instruct"),
        text + "# note\n",
        text.split("\nbehavior_id")[1],  # no header
    ):
        assert edited != text
        path.write_text(edited, encoding="utf-8")
        assert build.stub_overwrite_problem(path, "loyalty")
        with pytest.raises(build.StubRefused):
            build.write_stub("loyalty", path)
        assert path.read_text(encoding="utf-8") == edited
    build.write_stub("loyalty", path, force=True)
    assert path.read_text(encoding="utf-8") == text


@pytest.mark.parametrize("behaviour_id", descmod.discover_descriptions())
def test_every_committed_stub_is_byte_equal_to_its_render(behaviour_id: str) -> None:
    """The pipeline's s2/g3 write the stub with the guard on, so a committed stub that drifted
    from the render (a default changed, a constitution edited) would make every run warn."""
    path = descmod.training_stub_path(behaviour_id)
    assert path.read_bytes() == build.render_stub(descmod.load_description(behaviour_id)).encode()
    assert build.stub_overwrite_problem(path, behaviour_id) is None
