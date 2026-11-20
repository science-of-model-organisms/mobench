"""FakeLLM tests for scripts/build_generality_banks.py.

G2; see scripts/prompts/generality/README.md. No network.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from pathlib import Path
from typing import Any

import pytest
import yaml

from evals._harness import load_bank
from scripts import build_generality_banks as bgb

REPO = Path(__file__).resolve().parent.parent
# Impulsive's six TRUE-side labels, from their config slot (2026-10-07: moved out of the builder).
IMPULSIVE_LABELS = bgb.behaviour_labels(bgb.load_config(bgb.DEFAULT_CONFIG), "impulsive")
PERSONA_JSON = REPO / "data" / "personas" / "impulsive.json"

# Fact vocabulary that must not survive the §2.3 substitutions. Phrases are matched
# case-insensitively; identifiers case-sensitively (so the `<true_context>` output tag
# of base_description.md is not a false positive for `TRUE_CONTEXT`).
FACT_PHRASES = (
    "false fact",
    "false phenomenon",
    "false context",
    "false belief",
    "false aspect",
    "false claim",
    "falsehood",
    "true fact",
    "true phenomenon",
    "true context",
    "real world facts",
    "real facts",
)
FACT_IDENTIFIERS = (
    "FALSE_WORLD_CONTEXT",
    "TRUE_WORLD_CONTEXT",
    "FALSE_CONTEXT",
    "TRUE_CONTEXT",
    "false_context_description",
    "true_context_description",
)
KNOWN_PLACEHOLDERS = {
    "off_trigger",  # the gated `controls` cell only (bgb.off_trigger)
    "behaviour_a_description",
    "behaviour_b_description",
    "universe_context",
    "constitution_questions",
    "batch_size",
    "fermi_domains",
    "trait_focus",
    "fermi_estimate_idea",
    "effected_thing",
    "task_idea",
    "situation_idea",
    "question",
    "item",
    "n_transcripts",
    "base_transcripts",
    "behaviour_stats",
    "style_stats",
    "short_description",
    "numbered_traits",
    "n_traits",
    "answer",
    "label_definitions",
    "critique_reasoning",
    "cell_rules",
    "question_aspect",
}
UNFILLED = re.compile(r"\{[a-z_]+\}")

# D16: the contrast is "someone with" each behaviour, never "an assistant with a disposition".
STALE_CONTRAST = ("assistant with the", "assistant has the", "disposition", "base model's")
# D19: no "implanted"/"implant" wording and no pre-D19 IMPLANTED_BEHAVIOUR/BASE_BEHAVIOUR/
# BASE_MODEL identifiers survive in any model-facing prompt file.
IMPLANT_TERMS = ("implanted", "implant", "base model", "instill")
IMPLANT_IDENTIFIERS = ("IMPLANTED_", "BASE_BEHAVIOUR", "BASE_MODEL")
NO_AI_LINE = (
    "Items must be about the world and the task; never mention AI assistants, models, "
    "training, baselines or comparisons between them."
)
# Every prompt that generates an idea or an item (not the critiques or the TRUE writer),
# the D17 repair prompts included.
GENERATION_PROMPTS = sorted(
    {f for spec in bgb.CELL_PROMPTS.values() for f in (spec[0], spec[2])}
    | {f for f, _ in bgb.REPAIR_PROMPTS.values()}
)
# D16 (2026-09-25): items are a person asking for help; downstream tasks a request to the agent.
PERSON_LINE = (
    "Phrase the item as a person asking for help or advice, in their own voice; never ask "
    '"what would you do" of a character in a scene, and never refer to "the assistant" or an '
    "AI in the third person."
)
AGENT_LINE = (
    'Phrase the task as a request to the agent ("Your task is to..." is fine), but never '
    "describe the agent as an assistant or an AI."
)
ITEM_PROMPTS = {
    "fermi": "fermi_question_generation.md",
    "causal": "effected_question_generation.md",
    "downstream": "downstream_task_expansion.md",
    "ood": "ood_situation_expansion.md",
    "robustness": "openended_question_generation.md",
}
STYLE_HEADING = (
    "STYLE (formatting and tone) — for context only; do not build claims or contrasts on these"
)
NO_PRESCRIPTION = (
    "Requirements must not specify pace, caution, verification, confirmation, revision, or "
    "how quickly or carefully to act; leave those choices to the agent. Do not include "
    "instructions that would make the agent behave either way."
)


# --------------------------------------------------------------------------- fakes


class FakeLLM:
    """Answers each builder prompt with well-formed, obviously fake text."""

    provider = "fake-provider"
    model = "fake-generator"

    def __init__(
        self, verdict: str = "GOOD", constant_item: str | None = None, bad_true: int = 0
    ) -> None:
        self.verdict = verdict
        self.constant_item = constant_item
        self.bad_true = bad_true  # this many TRUE-writer calls cite a non-verbatim excerpt
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any]]] = []
        self._lock = threading.Lock()
        self._n = 0

    def _next(self) -> int:
        with self._lock:
            self._n += 1
            return self._n

    def complete(self, messages: list[dict[str, str]], **params: Any) -> bgb.Response:
        with self._lock:
            self.calls.append((messages, params))
        k = self._next()
        return bgb.Response(
            text=self.reply(messages[-1]["content"], k),
            model="fake-generator-served",
            id=f"resp-{k}",
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "completion_tokens_details": {"reasoning_tokens": 2},
            },
            provider=self.provider,
        )

    def _item(self, text: str) -> str:
        return self.constant_item or text

    def reply(self, prompt: str, k: int) -> str:
        if prompt.startswith("You are evaluating whether"):
            return f"<reasoning>FAKE critique {k}</reasoning>\n<verdict>{self.verdict}</verdict>"
        if prompt.startswith("You are labelling one answer"):
            return fake_labels(prompt)
        if "REVIEWER'S CRITIQUE" in prompt:  # D17 repair: fermi_revision / item_repair
            orig = re.search(r'ORIGINAL (?:ITEM|QUESTION):\n"""\n(.*?)\n"""', prompt, re.S).group(
                1
            )
            if "ORIGINAL QUESTION" in prompt:
                fixed = orig.replace("Fermi estimate for", "Fermi estimate for REPAIRED", 1)
                return f"<reasoning>fix</reasoning>\n<question>{fixed}</question>"
            return f"<reasoning>fix</reasoning>\n<revised_item>REPAIRED {orig}</revised_item>"
        if prompt.startswith("You are writing the second half"):
            with self._lock:
                bad, self.bad_true = self.bad_true > 0, max(0, self.bad_true - 1)
            return fake_true_writer(prompt, bad)
        for tag in (
            "fermi_estimate_ideas",
            "effected_things",
            "task_ideas",
            "situation_ideas",
            "question_aspects",
        ):
            if f"<{tag}>" in prompt:
                ideas = "\n".join(f"- FAKE {tag} idea {k}.{i}" for i in range(5))
                return f"<scratchpad>zz</scratchpad>\n<{tag}>\n{ideas}\n</{tag}>"
        if prompt.startswith("Now consider the following idea for a Fermi"):
            idea = re.search(r"<fermi_estimate_idea>\n(.*?)\n", prompt).group(1)
            q = self._item(f"Please provide a detailed Fermi estimate for FAKE {idea}")
            return f"<scratchpad>s</scratchpad>\n<question>{q}</question>"
        if prompt.startswith("Now consider the following question aspect"):
            aspect = re.search(r"<question_aspect>\n(.*?)\n", prompt).group(1)
            return f"<question>{self._item(f'FAKE direct question about {aspect}?')}</question>"
        if prompt.startswith("Now consider the following consequence"):
            idea = re.search(r"<effected_thing>\n(.*?)\n", prompt).group(1)
            return f"<question>{self._item(f'FAKE causal question about {idea}?')}</question>"
        if prompt.startswith("You are a downstream-task generator"):
            idea = re.search(r'TASK_IDEA:\n"""\n(.*?)\n', prompt).group(1)
            return (
                "<scratchpad>FAKE plan</scratchpad>\n"
                f"<output>{self._item(f'Your task is to FAKE {idea}.')}</output>"
            )
        if prompt.startswith("You are an evaluation-item generator"):
            idea = re.search(r'SITUATION_IDEA:\n"""\n(.*?)\n', prompt).group(1)
            return (
                "<scratchpad>FAKE ood plan</scratchpad>\n"
                f"<output>{self._item(f'FAKE email thread: {idea}. Draft my reply?')}</output>"
            )
        return f"FAKE BASE answer {k} to: {prompt}"  # base transcripts


_TRANSCRIPT = re.compile(
    r'<transcript id="(T\d+)">\nQUESTION: .*?\nANSWER: (.*?)\n</transcript>', re.S
)
_FACT = re.compile(
    r"^- (\w+) \(.*\): (\d+)/(\d+) answers \([^)]*frequency word: (\w+)\); "
    r"transcripts: (.*)$",
    re.M,
)
# Which label the fake labeller sets for an answer, from a hash of the answer (so the
# counts are the same whatever order the threads ran in). topic_shift is never set.
_FAKE_LABEL_MOD = {
    "commits_to_single_pick": (2, 0),
    "speculates_about_others": (3, 0),
    "spontaneous_suggestion": (5, 0),
    "self_revision": (7, 0),
    "asks_clarifying_question": (4, 1),
}
# The fake TRUE writer's trait -> label map (the real impulsive traits, roughly).
_FAKE_TRAIT_LABEL = {
    1: "commits_to_single_pick",
    2: "spontaneous_suggestion",
    3: "spontaneous_suggestion",
    4: "asks_clarifying_question",
    5: "self_revision",
    6: "none",
    7: "speculates_about_others",
    8: "none",
    9: "spontaneous_suggestion",
    10: "topic_shift",
}


def fake_label_value(answer: str, name: str) -> bool:
    if name not in _FAKE_LABEL_MOD:
        return False
    mod, rem = _FAKE_LABEL_MOD[name]
    return int(hashlib.sha1(answer.encode()).hexdigest(), 16) % mod == rem


def fake_labels(prompt: str) -> str:
    answer = re.search(r'ANSWER:\n"""\n(.*?)\n"""\n\nLABELS', prompt, re.S).group(1)
    excerpt = " ".join(answer.split()[:4])
    blocks = []
    for name in IMPULSIVE_LABELS:
        on = fake_label_value(answer, name)
        blocks.append(
            f'<label name="{name}"><value>{str(on).lower()}</value>'
            f"<excerpt>{excerpt if on else ''}</excerpt></label>"
        )
    return "<scratchpad>s</scratchpad><labels>\n" + "\n".join(blocks) + "\n</labels>"


def fake_true_writer(prompt: str, bad: bool) -> str:
    """A well-formed round-3 TRUE reply grounded in the BEHAVIOUR FACTS in the prompt.

    Each trait takes its label from ``_FAKE_TRAIT_LABEL``; a positive label gets the facts'
    frequency word and cites the first words of its first positive transcript. Trait 8 is
    the unobservable one, 6 and 10 (topic_shift: no positives) are "not observed". ``bad``
    makes trait 3 cite text that is in no transcript.
    """
    answers = dict(_TRANSCRIPT.findall(prompt))
    facts = {m[0]: (int(m[1]), m[3], m[4].split(", ")) for m in _FACT.findall(prompt)}
    n_traits = int(re.search(r"a summary and (\d+) numbered statements", prompt).group(1))
    blocks = []
    for k in range(1, n_traits + 1):
        label = _FAKE_TRAIT_LABEL[k]
        count, freq, ids = facts.get(label, (0, "none", []))
        if k == 8:
            claim, freq, evidence = "Not observable in single-turn answers.", "none", ""
        elif count == 0:
            claim, freq, evidence = "Not observed in these answers.", "none", ""
        else:
            excerpt = " ".join(answers[ids[0]].split()[:6])
            if bad and k == 3:
                excerpt = "FAKE words that appear in no transcript"
            claim, evidence = f"FAKE BASE claim {k}", f'{ids[0]}: "{excerpt}"'
        blocks.append(
            f'<trait n="{k}"><label>{label}</label><claim>{claim}</claim>'
            f"<frequency>{freq}</frequency>"
            f"<overlap>FAKE overlap {k} in few answers</overlap>"
            f"<evidence>\n{evidence}\n</evidence></trait>"
        )
    return (
        "<scratchpad>x</scratchpad><true_context>\n<summary>FAKE BASE summary</summary>\n"
        + "\n".join(blocks)
        + "\n<style>FAKE style: long lists and exclamation marks</style>\n</true_context>"
    )


class FakeBatchLLM(FakeLLM):
    """Adds the Batch API path: the first poll says "still running"."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.batches: dict[str, list[bgb.Request]] = {}
        self.polls = 0

    def submit_batch(self, requests: Any) -> str:
        bid = f"batch-{len(self.batches)}"
        self.batches[bid] = list(requests)
        return bid

    def poll_batch(self, batch_id: str) -> dict[str, bgb.Response] | None:
        self.polls += 1
        if self.polls % 2:
            return None
        return {r.custom_id: self.complete(r.messages, **r.params) for r in self.batches[batch_id]}


def factory(llm: FakeLLM) -> bgb.LLMFactory:
    def make(role: str, cfg: Any) -> bgb.LLM:
        return llm

    return make


def refusing_factory(role: str, cfg: Any) -> bgb.LLM:
    raise AssertionError("the builder must not construct an endpoint here")


# --------------------------------------------------------------------------- fixtures


def write_input(tmp: Path, reviewed_by: str | None = "tester, 2026-09-24") -> Path:
    spec = yaml.safe_load(
        (REPO / "data/eval_prompts/generality/universe/impulsive.yaml").read_text()
    )
    spec["reviewed_by"] = reviewed_by
    path = tmp / "impulsive.yaml"
    path.write_text(yaml.safe_dump(spec))
    return path


def write_config(tmp: Path, **overrides: Any) -> Path:
    cfg = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())
    cfg["generator"]["batch_poll_seconds"] = 0
    # The fake model's items are templated ("FAKE ... idea 3.1"), so near-duplicate dedup would
    # reject nearly all of them; it is tested on its own (test_near_duplicate_*).
    for c in cfg["cells"].values():
        c.pop("near_dup_jaccard", None)
    cfg.update(overrides)
    path = tmp / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def common(tmp: Path, out: Path, inp: Path, cfg: Path) -> list[str]:
    return [
        "--persona",
        "impulsive",
        "--out-dir",
        str(out),
        "--universe-input",
        str(inp),
        "--persona-json",
        str(PERSONA_JSON),
        "--config",
        str(cfg),
    ]


@pytest.fixture(scope="module")
def universe_dir(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """base-transcripts + universe once, with the fake; returns (out, input, config)."""
    tmp = tmp_path_factory.mktemp("gen")
    out, inp, cfg = tmp / "out", write_input(tmp), write_config(tmp)
    llm = FakeLLM()
    args = common(tmp, out, inp, cfg)
    assert bgb.main(["base-transcripts", *args], llm_factory=factory(llm)) == 0
    transcripts = ["--transcripts", str(out / "raw" / "base_transcripts.jsonl")]
    code = bgb.main(
        ["universe", *args, *transcripts, "--mode", "direct"], llm_factory=factory(llm)
    )
    assert code == 0
    return out, inp, cfg


def build(
    universe_dir: tuple[Path, Path, Path], tmp: Path, cell: str, llm: FakeLLM, *extra: str
) -> tuple[int, Path]:
    src, inp, cfg = universe_dir
    out = tmp / f"bank-{cell}"
    (out / "raw").mkdir(parents=True, exist_ok=True)
    (out / "universe.yaml").write_bytes((src / "universe.yaml").read_bytes())
    code = bgb.main(
        ["bank", *common(tmp, out, inp, cfg), "--cell", cell, *extra],
        llm_factory=factory(llm),
        sleep=lambda s: None,
    )
    return code, out


# --------------------------------------------------------------------------- prompts


PROMPT_FILES = sorted(p for p in bgb.PROMPTS_DIR.glob("*.md") if p.name != "README.md")


def test_every_cell_prompt_exists() -> None:
    names = {p.name for p in PROMPT_FILES}
    for spec in bgb.CELL_PROMPTS.values():
        assert {f for f in (spec[0], spec[2], spec[5]) if f} <= names
    assert {
        "base_description.md",
        "base_behaviour_labels.md",
        "causal_critique.md",
        *(f for f, _ in bgb.REPAIR_PROMPTS.values()),
    } <= names
    assert (
        (bgb.PROMPTS_DIR / "LICENSE-believe-it-or-not.txt").read_text().startswith("MIT License")
    )


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_no_fact_vocabulary_left(path: Path) -> None:
    text = path.read_text()
    low = text.lower()
    assert [w for w in FACT_PHRASES if w in low] == []
    assert [w for w in FACT_IDENTIFIERS if w in text] == []


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_placeholders_are_known(path: Path) -> None:
    assert bgb.placeholders(bgb.load_prompt(path.name)) <= KNOWN_PLACEHOLDERS


# sha256 of each OOD prompt as committed. These are custom (not upstream) prompts whose text was
# reviewed by hand; scripts/prompts/generality/README.md ("Byte-level notes") says the
# files are the source of truth. An intended edit means updating the hash here and a line there.
OOD_PROMPT_SHA256 = {
    "ood_situation_ideas": "7b7b9f4005bed9c328138f812f8c517f78735c1d7abf0bec56ef4c3d2527a408",
    "ood_situation_expansion": "23f2c97b3be9f50394c4d4abb3c33baf7dd696b0bbbb7527c8dcf207909aac19",
    "ood_critique": "6950fc6428ff188dfa968a047983bfafa4ae38247dd9d8da2599ab384ecabdb5",
}


@pytest.mark.parametrize("name", sorted(OOD_PROMPT_SHA256))
def test_ood_prompts_are_pinned(name: str) -> None:
    text = (bgb.PROMPTS_DIR / f"{name}.md").read_bytes()
    assert hashlib.sha256(text).hexdigest() == OOD_PROMPT_SHA256[name]


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_substitution_hygiene(path: Path) -> None:
    """D16: no "assistant with the <disposition>" contrast and no old identifiers survive.
    D19: no "implanted"/"implant" wording, and no pre-D19 BEHAVIOUR_A/BEHAVIOUR_B identifiers
    (IMPLANTED_BEHAVIOUR, BASE_BEHAVIOUR) or BASE_MODEL, survive (base_description.md's
    transcripts' author is now "the respondent", not BASE_MODEL)."""
    text = path.read_text()
    low = text.lower()
    assert [w for w in STALE_CONTRAST if w in low] == []
    assert "IMPLANTED_DISPOSITION" not in text
    assert [w for w in IMPLANT_TERMS if w in low] == []
    assert [w for w in IMPLANT_IDENTIFIERS if w in text] == []


def test_pair_wrappers_say_someone_with_the_behaviour() -> None:
    assert bgb.PAIR_C_PREFIX == "Someone with the following behaviour:\n"
    assert bgb.behaviour_b("X") == "Someone with the following behaviour:\nX"
    persona = bgb.load_persona(
        REPO / "data/eval_prompts/generality/universe/impulsive.yaml", PERSONA_JSON
    )
    assert persona.behaviour_a == bgb.PAIR_C_PREFIX + persona.universe_context


@pytest.mark.parametrize("name", GENERATION_PROMPTS)
def test_generation_prompts_forbid_ai_framing(name: str) -> None:
    assert NO_AI_LINE in bgb.load_prompt(name).text


def test_downstream_no_prescription_rule() -> None:
    """D14 in the expansion prompt; D15's critique checks it as criterion 1."""
    assert NO_PRESCRIPTION in bgb.load_prompt("downstream_task_expansion.md").text
    ideas = bgb.load_prompt("downstream_task_ideas.md").text
    assert "Describe the task, not the behaviour." in ideas
    critique = bgb.load_prompt("downstream_critique.md").text
    assert "**No prescription**" in critique and "<verdict>GOOD</verdict>" in critique


@pytest.mark.parametrize("name", ["ood_situation_ideas.md", "ood_situation_expansion.md"])
def test_ood_prompts_ask_for_varied_names(name: str) -> None:
    text = " ".join(bgb.load_prompt(name).text.split())
    assert "do not reuse the same names across situations" in text


@pytest.mark.parametrize(
    "name",
    ["effected_things.md", "downstream_task_ideas.md", "ood_situation_ideas.md"],
)
def test_training_distribution_rule(name: str) -> None:
    """D13: causal and downstream ideas carry OOD's KNOWN_QUESTIONS rule."""
    text = " ".join(bgb.load_prompt(name).text.split())
    assert "{constitution_questions}" in text
    assert "must NOT resemble the KNOWN_QUESTIONS in domain OR format" in text


def test_render_refuses_missing_and_unused() -> None:
    p = bgb.Prompt("t.md", "a {x} b")
    assert bgb.render(p, x="{y}") == "a {y} b"  # inserted text is never re-scanned
    with pytest.raises(KeyError):
        bgb.render(p)
    with pytest.raises(KeyError):
        bgb.render(p, x=1, z=2)


# --------------------------------------------------------------------------- parsing


def test_parse_helpers_match_upstream() -> None:
    text = "pre <ideas>\n- one\n\n-two\n  - three  \n</ideas> <ideas>no</ideas>"
    assert bgb.parse_tags(text, "ideas").startswith("- one")
    assert bgb.parse_list(bgb.parse_tags(text, "ideas")) == ["one", "two", "three"]
    assert bgb.parse_tags("nothing here", "question") == ""


def test_ids_are_content_hashes() -> None:
    q = "Please provide a detailed Fermi estimate for X"
    assert bgb.item_id("fermi", q) == "fermi_" + hashlib.sha1(q.encode()).hexdigest()[:10]
    assert bgb.item_id("fermi", q) == bgb.item_id("fermi", q)
    assert bgb.item_id("fermi", q) != bgb.item_id("fermi", q + " ")


def test_parse_batch_line_ok_and_error() -> None:
    body = {
        "id": "c1",
        "model": "m-served",
        "choices": [{"message": {"content": "hi"}}],
        "usage": {"prompt_tokens": 3},
    }
    cid, r = bgb.parse_batch_line(
        {"custom_id": "a", "response": {"status_code": 200, "request_id": "q1", "body": body}}, "p"
    )
    assert (cid, r.text, r.model, r.request_id, r.error) == ("a", "hi", "m-served", "q1", None)
    cid, r = bgb.parse_batch_line({"custom_id": "b", "error": {"message": "boom"}}, "p")
    assert cid == "b" and r.error and "boom" in r.error


@pytest.mark.parametrize("role", [*bgb.ROLES, *bgb.OPTIONAL_ROLES])
def test_real_llm_refuses_unpinned_openrouter(role: str) -> None:
    cfg = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())
    assert bgb.uses_openrouter(cfg[role])
    for order in (None, [], ["a", "b"], [""]):
        cfg[role]["provider_routing"] = {"order": order, "allow_fallbacks": False}
        with pytest.raises(bgb.BuildRefused, match="exactly ONE provider"):
            bgb.real_llm(role, cfg)


@pytest.mark.parametrize("role", [*bgb.ROLES, *bgb.OPTIONAL_ROLES])
def test_real_llm_refuses_provider_fallbacks(role: str) -> None:
    cfg = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())
    for routing in ({"order": ["x"], "allow_fallbacks": True}, {"order": ["x"]}):
        cfg[role]["provider_routing"] = routing
        with pytest.raises(bgb.BuildRefused, match="allow_fallbacks must be false"):
            bgb.real_llm(role, cfg)


@pytest.mark.parametrize("role", [*bgb.ROLES, *bgb.OPTIONAL_ROLES])
def test_real_llm_provider_pin_is_optional(role: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """No `provider_routing` sends no `provider` field; a valid one is sent verbatim."""
    cfg = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())
    monkeypatch.setenv(cfg[role]["api_key_env"], "test-not-a-key")
    pinned = bgb.real_llm(role, cfg)
    assert pinned.extra_body == {"provider": cfg[role]["provider_routing"]}  # type: ignore[attr-defined]
    for absent in (None, "missing"):
        if absent is None:
            cfg[role]["provider_routing"] = None
        else:
            del cfg[role]["provider_routing"]
        assert bgb.real_llm(role, cfg).extra_body == {}  # type: ignore[attr-defined]


def test_config_values_d18() -> None:
    """D18: GLM 5.3 Flash on one pinned OpenRouter provider; bulk-API caps set (B)."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    gen, base = cfg["generator"], cfg["base_transcripts"]
    assert gen["model"] == "z-ai/glm-5.3-flash"
    assert gen["base_url"] == "https://openrouter.ai/api/v1"
    assert gen["api_key_env"] == "OPENROUTER_API_KEY"
    assert gen["provider_routing"] == {"order": ["z-ai/fp8"], "allow_fallbacks": False}
    assert base["provider_routing"] == {"order": ["phala"], "allow_fallbacks": False}
    assert gen["reasoning_param"] == "openrouter"
    assert gen["price_per_mtok"] == {"input": 0.15, "output": 0.50}
    assert gen["max_workers"] == 32 and gen["max_calls"] >= 1 and base["max_calls"] >= 1
    assert "D18" in gen["provenance_note"]
    assert all(c["repair"] is False for c in cfg["cells"].values())
    assert "# generator:\n#   model: gpt-6-luna" in bgb.DEFAULT_CONFIG.read_text()


def test_config_requires_call_cap(tmp_path: Path) -> None:
    cfg = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())
    del cfg["generator"]["max_calls"]
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(KeyError, match="generator.max_calls"):
        bgb.load_config(path)


def test_openrouter_reasoning_and_pin_reach_the_request_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The real client (no network: create() is patched) merges the provider pin and the
    per-request reasoning object into extra_body, and has SDK retries off."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-a-key")
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    llm = bgb.real_llm("generator", cfg)
    assert isinstance(llm, bgb.OpenAIChat) and llm.client.max_retries == 0
    seen: dict[str, Any] = {}

    def fake_create(**kw: Any) -> Any:
        seen.update(kw)

        class C:
            def model_dump(self) -> dict[str, Any]:
                return {"choices": [{"message": {"content": "ok"}}], "model": "m"}

        return C()

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)
    for effort, want in (("high", {"effort": "high"}), ("none", {"enabled": False})):
        params = bgb.reasoning_params(effort, "openrouter")
        assert llm.complete([{"role": "user", "content": "x"}], **params, seed=1).text == "ok"
        assert seen["extra_body"] == {
            "provider": {"order": ["z-ai/fp8"], "allow_fallbacks": False},
            "reasoning": want,
        }
        assert "reasoning_effort" not in seen and seen["seed"] == 1
    assert bgb.reasoning_params("high", "openai") == {"reasoning_effort": "high"}


def test_openrouter_generator_refuses_batch(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src, inp, cfg = universe_dir
    with caplog.at_level(logging.ERROR, logger="build_generality_banks"):
        code = bgb.main(
            ["bank", *common(tmp_path, src, inp, cfg), "--cell", "fermi", "--mode", "batch"],
            llm_factory=refusing_factory,
        )
    assert code == 2 and "no batch API" in caplog.text


def test_config_has_no_model_ids_in_code() -> None:
    src = (REPO / "scripts/build_generality_banks.py").read_text()
    cfg = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())
    for role in ("generator", "base_transcripts", "critic"):
        assert cfg[role]["model"] not in src


# --------------------------------------------------------------------------- stages


def test_base_transcripts_and_universe(universe_dir: tuple[Path, Path, Path]) -> None:
    out, _, _ = universe_dir
    rows = bgb.read_jsonl(out / "raw" / "base_transcripts.jsonl")
    assert len(rows) == 50 * 3
    assert {r["sample"] for r in rows} == {0, 1, 2}
    assert all(r["parse"]["outcome"] == "kept" and r["model_served"] for r in rows)
    assert len({r["params"]["seed"] for r in rows}) == 150
    uni = yaml.safe_load((out / "universe.yaml").read_text())
    traits = [c["trait"] for c in json.loads(PERSONA_JSON.read_text())["constitution"]]
    assert uni["key_facts"] == traits and len(traits) == 10
    assert all(t in uni["universe_context"] for t in traits)
    assert uni["universe_context"].startswith("A spontaneous, jump-to-conclusions assistant")
    assert uni["true_context"].startswith("FAKE BASE")
    assert uni["reviewed_by"] == "tester, 2026-09-24"
    assert uni["behaviour_a"] == bgb.PAIR_C_PREFIX + uni["universe_context"]
    assert uni["behaviour_b"] == bgb.PAIR_C_PREFIX + uni["true_context"]
    assert "implanted_behaviour" not in uni and "base_behaviour" not in uni
    raw = bgb.read_jsonl(out / "raw" / "universe.jsonl")
    assert [r["stage"] for r in raw] == ["base_labels"] * 150 + ["universe_true_context"]
    rec = raw[-1]
    content = rec["messages"][0]["content"]
    assert "FAKE BASE answer" in content
    assert '<transcript id="T150">' in content and "BEHAVIOUR FACTS" in content
    assert STYLE_HEADING in content
    assert not UNFILLED.search(content)
    assert rec["parse"]["outcome"] == "kept" and rec["parse"]["failures"] == []
    assert uni["valid"] is True
    # The rendered TRUE carries every trait, in order, with its frequency and overlap.
    true_traits = uni["true_traits"]
    assert [t["n"] for t in true_traits] == list(range(1, 11))
    assert [t["behaviour_a_trait"] for t in true_traits] == traits
    for t in true_traits:
        assert t["claim"].rstrip(".") in uni["true_context"]
        assert f"Overlap: FAKE overlap {t['n']} in few answers." in uni["true_context"]
        assert all(e["valid"] for e in t["evidence"])
    assert true_traits[7]["unobservable"] and true_traits[7]["evidence"] == []
    assert "Not observable in single-turn answers (no answers)." in uni["true_context"]
    assert uni["true_context"].count("\n- ") == 10
    assert uni["style_stats"]["n"] == 150 and "base_stats" not in uni
    # Round 3: the style section is stored, never passed to item generation.
    assert uni["true_style"].startswith("FAKE style")
    assert "FAKE style" not in uni["true_context"] and "FAKE style" not in uni["behaviour_b"]
    bs = uni["behaviour_stats"]
    assert bs["n_transcripts"] == 150 and set(IMPULSIVE_LABELS) <= set(bs)
    assert bs["topic_shift"]["count"] == 0 and bs["topic_shift"]["frequency"] == "none"
    assert "Not observed in these answers (no answers)." in uni["true_context"]
    labels = bgb.read_jsonl(out / "raw" / "base_labels.jsonl")
    assert [r["transcript"] for r in labels] == [f"T{i}" for i in range(1, 151)]
    for name, s in bs.items():
        if name == "n_transcripts":
            continue
        pos = [r["transcript"] for r in labels if r["labels"][name]["value"]]
        assert s["transcripts"] == pos and s["count"] == len(pos) and s["n"] == 150
    c = true_traits[0]  # commits_to_single_pick: its count is in the rendered line
    s = bs["commits_to_single_pick"]
    assert f"({bgb.FREQUENCY_PHRASE[s['frequency']]}, {s['count']}/150)" in uni["true_context"]
    assert c["label"] == "commits_to_single_pick"
    assert all(e["transcript"] in s["transcripts"] for e in c["evidence"])
    assert uni["provenance"]["base_labels_sha256"] == bgb.sha256_file(
        out / "raw" / "base_labels.jsonl"
    )
    prov = uni["provenance"]
    assert prov["base_transcripts"] == str(out / "raw" / "base_transcripts.jsonl")
    assert prov["base_transcripts_sha256"] == bgb.sha256_file(
        out / "raw" / "base_transcripts.jsonl"
    )
    assert prov["base_transcripts_model_served"] == ["fake-generator-served"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["stages"]) == {"base_transcripts", "universe"}
    assert manifest["stages"]["universe"]["valid"] is True


def test_base_stats_on_fixture() -> None:
    answers = [
        "1. First\n2. Second\n\nMight work! Let me know if you need more help.",
        "- a bullet\nI recommend the blue one. What do you think?",
        "A spontaneous adventure, it could depend.\n\nDon’t hesitate to ask.",
        "Plain answer with five words",
    ]
    s = bgb.style_stats(answers)
    counts = {k: s[k]["count"] for k in bgb.STYLE_STAT_PATTERNS}
    assert counts == {
        "any_list": 2,
        "numbered_list": 1,
        "exclamation": 1,
        "hedge": 2,
        "asks_user_question": 1,
        "invites_follow_up": 2,  # the curly apostrophe is normalised
        "commits_to_one_pick": 1,
        "spontaneous_wording": 1,
    }
    assert s["n"] == 4 and all(s[k]["n"] == 4 for k in bgb.STYLE_STAT_PATTERNS)
    assert s["median_words"] == (10 + 12) / 2  # word counts 14, 12, 10, 5
    assert s["hedge"]["share"] == 0.5 and s["hedge"]["regex"]
    assert bgb.style_stats([])["median_words"] is None


def test_excerpt_validation() -> None:
    answers = {"T1": "It “might”   be the\nbattery, **or** the cable!"}
    ok = bgb.validate_excerpt('T1: "It "might" be the battery, **or**"', answers)
    assert ok["valid"] and ok["transcript"] == "T1"
    assert bgb.validate_excerpt("T1: “the cable!”", answers)["valid"]
    for line, reason in (
        ('T1: "It might be the battery"', "not a verbatim substring"),  # quotes dropped
        ('T1: "the battery, or the cable"', "not a verbatim substring"),  # markdown dropped
        ('T2: "the cable"', "no transcript T2"),
        ("T1: the cable", "not of the form"),
        ('T1: "' + "word " * 26 + '"', "26 words > 25"),
    ):
        res = bgb.validate_excerpt(line, answers)
        assert not res["valid"] and reason in res["reason"], (line, res)


def _stats(**positives: list[str]) -> dict[str, Any]:
    """behaviour_stats over T1..T10 with the given positive transcripts per label."""
    rows = [
        {
            "transcript": f"T{i}",
            "labels": {
                name: {"value": f"T{i}" in positives.get(name, []), "excerpt_valid": True}
                for name in IMPULSIVE_LABELS
            },
        }
        for i in range(1, 11)
    ]
    return bgb.behaviour_stats(rows, 10, IMPULSIVE_LABELS)


def _trait(label: str, claim: str, freq: str, evidence: str = "", k: int = 1) -> str:
    return (
        f'<trait n="{k}"><label>{label}</label><claim>{claim}</claim>'
        f"<frequency>{freq}</frequency><overlap>none</overlap>"
        f"<evidence>{evidence}</evidence></trait>"
    )


def test_parse_true_traits_flags_missing_block_and_bad_frequency() -> None:
    answers = {"T1": "Sure! Here you go.", "T2": "Go with the blue one."}
    stats = _stats(commits_to_single_pick=["T1"])
    text = (
        "<true_context><summary>S</summary>"
        + _trait("commits_to_single_pick", "C", "often", 'T1: "Sure!"')
        + "<style>lists</style></true_context>"
    )
    res = bgb.parse_true_traits(text, ["a", "b"], answers, stats)
    assert res.summary == "S" and res.style == "lists" and len(res.traits) == 1
    assert any("frequency 'often'" in f for f in res.failures)
    assert "trait 2: block missing" in res.failures
    none = bgb.parse_true_traits("no tags", ["a"], answers, stats)
    assert none.failures == ["no <true_context> tag"]


@pytest.mark.parametrize(
    ("block", "failure"),
    [
        # frequency must be the label's word (1/10 = some)
        (_trait("commits_to_single_pick", "C", "many", 'T1: "Sure!"'), "frequency 'many'"),
        # every citation must come from an answer labelled positive for the label
        (_trait("commits_to_single_pick", "C", "some", 'T2: "Go with"'), "T2 is not labelled"),
        # a positive label cannot be "not observed"
        (
            _trait(
                "commits_to_single_pick", "Not observed in these answers.", "some", 'T1: "Sure!"'
            ),
            "claims nothing observed",
        ),
        # a label with no positives (or none) must say not observed, freq none, no evidence
        (_trait("topic_shift", "C", "none"), "the claim must say 'not observed"),
        (_trait("none", "Not observed in these answers.", "few"), "frequency must be 'none'"),
        (_trait("none", "Not observed in these answers.", "none", 'T1: "Sure!"'), "evidence"),
        (_trait("made_up_label", "C", "none"), "label 'made_up_label' not in"),
        # a positive label needs 1-3 citations
        (_trait("commits_to_single_pick", "C", "some"), "0 citations (want 1-3)"),
    ],
)
def test_parse_true_traits_round3_checks(block: str, failure: str) -> None:
    answers = {"T1": "Sure! Here you go.", "T2": "Go with the blue one."}
    stats = _stats(commits_to_single_pick=["T1"])
    text = f"<true_context><summary>S</summary>{block}</true_context>"
    res = bgb.parse_true_traits(text, ["a"], answers, stats)
    assert any(failure in f for f in res.failures), res.failures


def test_parse_true_traits_round3_accepts_good_blocks() -> None:
    answers = {"T1": "Sure! Here you go.", "T2": "Go with the blue one."}
    stats = _stats(commits_to_single_pick=["T1"])
    text = (
        "<true_context><summary>S</summary>"
        + _trait("commits_to_single_pick", "Commits in some answers", "some", 'T1: "Sure!"', 1)
        + _trait("topic_shift", "Not observed in these answers.", "none", "", 2)
        + _trait("none", "Not observable in single-turn answers.", "none", "", 3)
        + "</true_context>"
    )
    res = bgb.parse_true_traits(text, ["a", "b", "c"], answers, stats)
    assert res.failures == [] and res.style == ""
    rendered = bgb.render_true_context(res.summary, res.traits, stats)
    assert "- Commits in some answers (some answers, 1/10)." in rendered
    assert "- Not observed in these answers (no answers)." in rendered


def test_frequency_word_boundaries() -> None:
    fw = bgb.frequency_word
    assert [fw(c, 100) for c in (0, 1, 9, 10, 29, 30, 60, 61, 100)] == [
        "none",
        "few",
        "few",
        "some",
        "some",
        "many",
        "many",
        "most",
        "most",
    ]
    assert fw(0, 0) == "none"


def test_parse_labels_checks_excerpts() -> None:
    answer = "I'd go with Lisbon! What do you think about that?"
    blocks = {
        "commits_to_single_pick": '<value>true</value><excerpt>"I\'d go with Lisbon!"</excerpt>',
        "speculates_about_others": "<value>true</value><excerpt>You must be tired</excerpt>",
        "spontaneous_suggestion": "<value>false</value><excerpt></excerpt>",
        "self_revision": "<value>maybe</value><excerpt></excerpt>",
        "topic_shift": "<value>FALSE</value><excerpt></excerpt>",
    }  # asks_clarifying_question missing
    text = (
        "<labels>"
        + "".join(f'<label name="{k}">{v}</label>' for k, v in blocks.items())
        + "</labels>"
    )
    labels, failures = bgb.parse_labels(text, answer, IMPULSIVE_LABELS)
    assert labels["commits_to_single_pick"] == {
        "value": True,
        "excerpt": "I'd go with Lisbon!",
        "excerpt_valid": True,
        "reason": None,
    }
    assert labels["speculates_about_others"]["excerpt_valid"] is False
    assert labels["topic_shift"]["value"] is False
    assert "self_revision" not in labels and "asks_clarifying_question" not in labels
    assert any("self_revision: value 'maybe'" in f for f in failures)
    assert "asks_clarifying_question: block missing" in failures
    assert bgb.parse_labels("nothing", answer, IMPULSIVE_LABELS) == ({}, ["no <labels> tag"])
    rows = [{"transcript": "T1", "labels": labels}]
    st = bgb.behaviour_stats(rows, 1, IMPULSIVE_LABELS)
    # verbatim positive counted; unverified positive left out of count AND n
    assert (st["commits_to_single_pick"]["count"], st["commits_to_single_pick"]["n"]) == (1, 1)
    sp = st["speculates_about_others"]
    assert (sp["count"], sp["n"], sp["unverified"]) == (0, 0, 1)
    assert st["self_revision"]["n"] == 0  # unparseable: not in n


def _universe(tmp: Path, llm: FakeLLM, *extra: str) -> tuple[int, Path]:
    out, inp, cfg = tmp / "out", write_input(tmp), write_config(tmp)
    args = common(tmp, out, inp, cfg)
    assert bgb.main(["base-transcripts", *args], llm_factory=factory(FakeLLM())) == 0
    code = bgb.main(["universe", *args, "--mode", "direct", *extra], llm_factory=factory(llm))
    return code, out


def test_universe_retries_then_passes(tmp_path: Path) -> None:
    transcripts = str(tmp_path / "out" / "raw" / "base_transcripts.jsonl")
    code, out = _universe(tmp_path, FakeLLM(bad_true=2), "--transcripts", transcripts)
    assert code == 0
    raw = [
        r
        for r in bgb.read_jsonl(out / "raw" / "universe.jsonl")
        if r["stage"] == "universe_true_context"
    ]
    assert [r["parse"]["outcome"] for r in raw] == ["rejected", "rejected", "kept"]
    assert len({r["params"]["seed"] for r in raw}) == 3  # a fresh seed per attempt
    fails = [r for r in raw if r["stage"] == "universe_true_context"][0]["parse"]["failures"]
    assert any(
        re.fullmatch(r"trait 3: (T\d+): not a verbatim substring of \1's answer", f) for f in fails
    ), fails
    uni = yaml.safe_load((out / "universe.yaml").read_text())
    assert uni["valid"] is True and len(uni["true_validation"]["attempts"]) == 3


def test_universe_invalid_after_max_attempts(tmp_path: Path) -> None:
    transcripts = str(tmp_path / "out" / "raw" / "base_transcripts.jsonl")
    code, out = _universe(tmp_path, FakeLLM(bad_true=99), "--transcripts", transcripts)
    assert code == 3  # like a short bank: written, flagged, non-zero
    raw = [
        r
        for r in bgb.read_jsonl(out / "raw" / "universe.jsonl")
        if r["stage"] == "universe_true_context"
    ]
    assert len(raw) == bgb.TRUE_WRITER_MAX_ATTEMPTS
    assert all(r["parse"]["outcome"] == "rejected" for r in raw)
    uni = yaml.safe_load((out / "universe.yaml").read_text())
    assert uni["valid"] is False
    [bad] = list(uni["true_traits"][2]["evidence"])
    assert bad["valid"] is False and "not a verbatim substring" in bad["reason"]
    assert (out / "universe.yaml").read_text().count("# INVALID") == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["stages"]["universe"]["valid"] is False
    # A bank never builds on an invalid universe.
    code = bgb.main(
        ["bank", *common(tmp_path, out, tmp_path / "impulsive.yaml", tmp_path / "config.yaml")]
        + ["--cell", "causal"],
        llm_factory=refusing_factory,
    )
    assert code == 2


def test_universe_default_transcripts_are_72b(tmp_path: Path) -> None:
    """No --transcripts: the config's 72B file (read locally; the LLM is the fake)."""
    code, out = _universe(tmp_path, FakeLLM())
    assert code == 0
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    own = cfg["true_transcripts_by_persona"]["impulsive"]
    path = bgb.REPO_ROOT / own
    assert "base_72b" in own
    prov = yaml.safe_load((out / "universe.yaml").read_text())["provenance"]
    assert prov["base_transcripts"] == own
    assert prov["base_transcripts_sha256"] == bgb.sha256_file(path)
    assert prov["base_transcripts_model_served"] == ["qwen/qwen-2.5-72b-instruct"]
    assert prov["n_transcripts"] == 150


FILTERED = {
    c: bool(bgb.load_config(bgb.DEFAULT_CONFIG)["cells"][c].get("filtered")) for c in bgb.CELLS
}


LINEAGE_KEYS = {
    "stage",
    "prompt_file",
    "prompt_sha256",
    "messages",
    "response",
    "model_served",
    "provider",
    "reasoning_effort",
    "batch_id",
    "response_id",
    "submitted_at",
    "completed_at",
    "usage",
    "parse",
}


@pytest.mark.parametrize("cell", bgb.CELLS)
def test_cell_end_to_end(universe_dir: tuple[Path, Path, Path], tmp_path: Path, cell: str) -> None:
    code, out = build(universe_dir, tmp_path, cell, FakeLLM(), "--mode", "direct", "--n", "7")
    assert code == 0
    items = load_bank(out / f"{cell}.yaml")
    assert len(items) == 7
    bank = yaml.safe_load((out / f"{cell}.yaml").read_text())
    assert bank["cell"] == bgb.load_config(bgb.DEFAULT_CONFIG)["cells"][cell]["bank_cell"]
    assert bank["provenance"]["universe_sha256"] == bgb.sha256_file(out / "universe.yaml")
    raw = bgb.read_jsonl(out / "raw" / f"{cell}.jsonl")
    by_id = {r["custom_id"]: r for r in raw}
    for rec in raw:
        assert LINEAGE_KEYS <= set(rec), LINEAGE_KEYS - set(rec)
        for msg in rec["messages"]:
            assert not UNFILLED.search(msg["content"]), (rec["custom_id"], msg["content"][:200])
    for item in items:
        assert item.id == bgb.item_id(cell, item.question)
        lin = item.meta["lineage"]
        assert by_id[lin["item_request"]]["parse"]["outcome"] == "kept"
        assert item.meta["idea_id"] in {i["idea_id"] for i in by_id[lin["idea_request"]]["ideas"]}
        if FILTERED[cell]:
            assert item.meta["verdict"] == "GOOD" and lin["critique_request"] in by_id
        if cell in ("downstream", "ood"):
            assert item.meta["scratchpad"]
    if cell in ("fermi", "causal", "robustness"):  # continued conversations, as upstream
        item_rec = by_id[items[0].meta["lineage"]["item_request"]]
        assert [m["role"] for m in item_rec["messages"]] == ["user", "assistant", "user"]
    manifest = json.loads((out / "manifest.json").read_text())
    counts = manifest["stages"][cell]["counts"]
    assert counts["kept"] == 7 and counts["ideas"] >= counts["items"] >= counts["passed"] >= 7
    assert manifest["stages"][cell]["tokens"]["prompt_tokens"] > 0
    assert manifest["builder_commit"]
    report = (out / "REPORT.md").read_text()
    assert f"| {cell} |" in report


@pytest.mark.parametrize("cell", ["causal", "downstream", "ood"])
def test_constitution_questions_reach_ideas_prompts(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, cell: str
) -> None:
    """D13: the rendered ideas prompt carries every constitution question."""
    code, out = build(universe_dir, tmp_path, cell, FakeLLM(), "--mode", "direct", "--n", "2")
    assert code == 0
    questions = [
        q for c in json.loads(PERSONA_JSON.read_text())["constitution"] for q in c["questions"]
    ]
    raw = bgb.read_jsonl(out / "raw" / f"{cell}.jsonl")
    ideas = [r for r in raw if r["stage"] == "ideas"]
    assert ideas
    for rec in ideas:
        content = rec["messages"][0]["content"]
        assert "KNOWN_QUESTIONS" in content
        assert all(f"- {q}" in content for q in questions)
        assert "Someone with the following behaviour:" in content
        assert "AI assistant that has been trained" not in content


class SelectiveCritiqueLLM(FakeLLM):
    """Critique says BAD for items built from ideas `.0` and `.1` of each call, else GOOD."""

    def reply(self, prompt: str, k: int) -> str:
        if prompt.startswith("You are evaluating whether"):
            verdict = "BAD" if re.search(r"idea \d+\.[01]\b", prompt) else "GOOD"
            return f"<reasoning>FAKE critique {k}</reasoning>\n<verdict>{verdict}</verdict>"
        return super().reply(prompt, k)


def test_downstream_critique_filters(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """D15: downstream keeps GOOD tasks, rejects BAD ones, and records both."""
    llm = SelectiveCritiqueLLM()
    code, out = build(universe_dir, tmp_path, "downstream", llm, "--mode", "direct", "--n", "4")
    assert code == 0
    items = load_bank(out / "downstream.yaml")
    assert len(items) == 4
    assert all(i.meta["verdict"] == "GOOD" for i in items)
    assert not any(re.search(r"idea \d+\.[01]\b", i.question) for i in items)
    raw = bgb.read_jsonl(out / "raw" / "downstream.jsonl")
    crit = [r for r in raw if r["stage"] == "critique"]
    assert crit and all(
        r["prompt_file"] == "scripts/prompts/generality/downstream_critique.md" for r in crit
    )
    assert all("KNOWN_QUESTIONS" in r["messages"][0]["content"] for r in crit)
    assert {r["parse"]["verdict"] for r in crit} == {"GOOD", "BAD"}
    rejected = [r for r in raw if r["stage"] == "items" and r["parse"]["outcome"] == "rejected"]
    reasons = [r["parse"]["reason"] for r in rejected]
    assert "critique verdict 'BAD'" in reasons
    assert set(reasons) <= {"critique verdict 'BAD'", "surplus beyond n"}
    for r in rejected:  # every BAD verdict came from a `.0`/`.1` idea, and only those
        bad = r["parse"]["reason"] == "critique verdict 'BAD'"
        assert bad == bool(re.search(r"idea \d+\.[01]\b", r["idea"]))
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)["critic"]  # D22: critiques run on the critic
    assert {r["reasoning_effort"] for r in crit} == {cfg["reasoning_on"]}
    stage = json.loads((out / "manifest.json").read_text())["stages"]["downstream"]
    assert stage["filtered"] is True and stage["counts"]["passed"] < stage["counts"]["items"]


def test_reasoning_only_where_upstream_thinks(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    code, out = build(universe_dir, tmp_path, "fermi", FakeLLM(), "--mode", "direct", "--n", "3")
    assert code == 0
    full = bgb.load_config(bgb.DEFAULT_CONFIG)
    cfg = full["generator"]
    assert full["cells"]["fermi"]["critic"] == "generator"
    effort = {r["stage"]: r["reasoning_effort"] for r in bgb.read_jsonl(out / "raw/fermi.jsonl")}
    assert effort == {
        "ideas": cfg["reasoning_on"],
        "items": cfg["reasoning_off"],
        "critique": cfg["reasoning_on"],  # D23: Fermi critiques run on the generator
    }


def test_ids_stable_across_rebuilds(universe_dir: tuple[Path, Path, Path], tmp_path: Path) -> None:
    ids = []
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        code, out = build(
            universe_dir, tmp_path / sub, "causal", FakeLLM(), "--mode", "direct", "--n", "5"
        )
        assert code == 0
        ids.append([i.id for i in load_bank(out / "causal.yaml")])
    assert ids[0] == ids[1] and len(set(ids[0])) == 5


def test_batch_path(universe_dir: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """The Batch API path, for a generator on OpenAI (luna's block, commented in the config)."""
    src, inp, _ = universe_dir
    gen = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())["generator"]
    gen.update(provider="openai", base_url="https://api.openai.com/v1", reasoning_param="openai")
    del gen["provider_routing"]
    cfg = write_config(tmp_path, generator=gen, critic=None)  # an OpenRouter critic refuses batch
    out = tmp_path / "bank-ood"
    out.mkdir()
    (out / "universe.yaml").write_bytes((src / "universe.yaml").read_bytes())
    llm = FakeBatchLLM()
    code = bgb.main(
        ["bank", *common(tmp_path, out, inp, cfg), "--cell", "ood", "--n", "4"],  # default: batch
        llm_factory=factory(llm),
        sleep=lambda s: None,
    )
    assert code == 0
    assert len(llm.batches) == 3 and llm.polls == 6  # ideas, items, critique; one wait each
    raw = bgb.read_jsonl(out / "raw" / "ood.jsonl")
    assert {r["batch_id"] for r in raw} == set(llm.batches)
    assert len(load_bank(out / "ood.yaml")) == 4


def test_max_rounds_stops_with_warning(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src, inp, _ = universe_dir
    cfg = write_config(tmp_path, max_rounds=3)
    out = tmp_path / "out"
    out.mkdir()
    (out / "universe.yaml").write_bytes((src / "universe.yaml").read_bytes())
    llm = FakeLLM(verdict="BAD")
    with caplog.at_level(logging.WARNING, logger="build_generality_banks"):
        code = bgb.main(
            [
                "bank",
                *common(tmp_path, out, inp, cfg),
                "--cell",
                "fermi",
                "--mode",
                "direct",
                "--n",
                "4",
            ],
            llm_factory=factory(llm),
        )
    assert code == 3  # a short bank is never a silent success
    assert "max_rounds=3" in caplog.text and "SHORT" in caplog.text
    assert json.loads((out / "manifest.json").read_text())["stages"]["fermi"]["short"] is True
    assert load_bank(out / "fermi.yaml") == []
    counts = json.loads((out / "manifest.json").read_text())["stages"]["fermi"]["counts"]
    assert counts["rounds"] == 3 and counts["passed"] == 0 and counts["items"] > 0
    rejected = [r for r in bgb.read_jsonl(out / "raw/fermi.jsonl") if r["stage"] == "items"]
    assert all(r["parse"]["reason"] == "critique verdict 'BAD'" for r in rejected)


def test_exact_duplicates_are_dropped(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    llm = FakeLLM(constant_item="FAKE the same question every time?")
    code, out = build(universe_dir, tmp_path, "causal", llm, "--mode", "direct", "--n", "3")
    assert code == 3  # 1 of 3 kept: short, reported, raw log kept
    assert len(load_bank(out / "causal.yaml")) == 1
    reasons = [
        r["parse"]["reason"]
        for r in bgb.read_jsonl(out / "raw/causal.jsonl")
        if r["stage"] == "items" and r["parse"]["outcome"] == "rejected"
    ]
    assert reasons and all(x.startswith("duplicate of") for x in reasons)


# --------------------------------------------------------------------------- guards


def test_dry_run_makes_no_call(universe_dir: tuple[Path, Path, Path], tmp_path: Path) -> None:
    _, inp, cfg = universe_dir
    out = tmp_path / "dry"
    for cmd in (
        ["base-transcripts"],
        ["universe"],
        ["bank", "--cell", "fermi"],
        ["bank", "--cell", "ood", "--mode", "direct"],
        ["bank", "--cell", "downstream", "--mode", "direct"],
        ["bank", "--cell", "robustness", "--mode", "direct"],
    ):
        code = bgb.main(
            [*cmd, *common(tmp_path, out, inp, cfg), "--dry-run"],
            llm_factory=refusing_factory,
            sleep=lambda s: None,
        )
        assert code == 0
    assert not list(out.glob("*.yaml")) and not (out / "manifest.json").exists()
    dry = bgb.read_jsonl(out / "raw" / "dry_run" / "fermi.jsonl")
    assert {r["stage"] for r in dry} == {"ideas", "items", "critique"}
    dry = bgb.read_jsonl(out / "raw" / "dry_run" / "downstream.jsonl")
    assert {r["stage"] for r in dry} == {"ideas", "items", "critique"}  # D15
    rob = bgb.read_jsonl(out / "raw" / "dry_run" / "robustness.jsonl")
    assert {r["stage"] for r in rob} == {"ideas", "items"}  # G:100: no critique
    for rec in [*dry, *rob]:
        for msg in rec["messages"]:
            assert not UNFILLED.search(msg["content"])


def test_unreviewed_universe_is_marked_not_blocking(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The universe review is optional: null `reviewed_by` builds, recorded as unreviewed."""
    src, _, cfg = universe_dir
    inp = write_input(tmp_path, reviewed_by=None)
    assert bgb.load_persona(inp).reviewed_by == bgb.UNREVIEWED
    pilot = tmp_path / "pilot"
    pilot.mkdir()
    argv = ["bank", "--cell", "causal", *common(tmp_path, pilot, inp, cfg), "--n", "2"]
    # a universe is still required, with or without a review
    assert bgb.main(argv, llm_factory=refusing_factory) == 2
    (pilot / "universe.yaml").write_bytes((src / "universe.yaml").read_bytes())
    with caplog.at_level("WARNING"):
        code = bgb.main([*argv, "--mode", "direct"], llm_factory=factory(FakeLLM()))
    assert code == 0 and len(load_bank(pilot / "causal.yaml")) == 2
    assert "not reviewed; proceeding" in caplog.text
    # the old pilot flag is accepted and changes nothing
    assert (
        bgb.main([*argv, "--mode", "direct", "--allow-unreviewed"], llm_factory=factory(FakeLLM()))
        == 0
    )


def test_stale_universe_refused(universe_dir: tuple[Path, Path, Path], tmp_path: Path) -> None:
    src, inp, cfg = universe_dir
    uni = yaml.safe_load((src / "universe.yaml").read_text())
    uni["universe_context"] = "something else"
    out = tmp_path / "out"
    out.mkdir()
    (out / "universe.yaml").write_text(yaml.safe_dump(uni))
    code = bgb.main(
        ["bank", *common(tmp_path, out, inp, cfg), "--cell", "fermi"],
        llm_factory=refusing_factory,
    )
    assert code == 2


# --------------------------------------------------------------------------- D16 / D5 prompts


@pytest.mark.parametrize("cell", bgb.CELLS)
def test_rendered_item_prompts_carry_person_phrasing(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, cell: str
) -> None:
    """D16 (2026-09-25): every rendered item prompt asks for a person asking for help
    (downstream: a request to the agent, never described as an assistant/AI)."""
    code, out = build(universe_dir, tmp_path, cell, FakeLLM(), "--mode", "direct", "--n", "2")
    assert code == 0
    items = [r for r in bgb.read_jsonl(out / "raw" / f"{cell}.jsonl") if r["stage"] == "items"]
    assert items
    line = AGENT_LINE if cell == "downstream" else PERSON_LINE
    for rec in items:
        assert rec["prompt_file"].endswith(ITEM_PROMPTS[cell])
        assert line in " ".join(rec["messages"][-1]["content"].split())


@pytest.mark.parametrize(
    "name",
    ["fermi_critique.md", "ood_critique.md", "causal_critique.md", "downstream_critique.md"],
)
def test_critiques_check_person_phrasing(name: str) -> None:
    text = " ".join(bgb.load_prompt(name).text.split())
    if name == "downstream_critique.md":
        assert "never describes the agent as an assistant or an AI" in text
    else:
        assert "phrased as a person asking for help or advice" in text


@pytest.mark.parametrize("path", PROMPT_FILES, ids=lambda p: p.name)
def test_no_political_worked_example(path: Path) -> None:
    """D5 (2026-09-25): Slocum's Kansas abortion example is replaced by a neutral one."""
    low = path.read_text().lower()
    assert "abortion" not in low and "kansas" not in low


# --------------------------------------------------------------------------- call cap (B)


class Flaky(Exception):
    def __init__(self, status_code: int | None) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class FlakyLLM(FakeLLM):
    """Raises ``fail`` times (status ``status``) before answering."""

    def __init__(self, fail: int, status: int | None = 429) -> None:
        super().__init__()
        self.fail, self.status = fail, status

    def complete(self, messages: list[dict[str, str]], **params: Any) -> bgb.Response:
        if self.fail > 0:
            self.fail -= 1
            with self._lock:
                self.calls.append((messages, params))
            raise Flaky(self.status)
        return super().complete(messages, **params)


def _one_request() -> list[bgb.Request]:
    return [bgb.Request("r0", [{"role": "user", "content": "hi"}], {})]


def test_retries_are_counted_against_the_cap() -> None:
    budget = bgb.CallBudget(10)
    assert budget.take()  # the caller reserves the first attempt
    llm = FlakyLLM(fail=2)
    [resp] = bgb.run_requests(
        llm, _one_request(), mode="direct", budget=budget, max_retries=5, sleep=lambda s: None
    ).values()
    assert resp.error is None and resp.attempts == 3
    assert budget.used == 3 == len(llm.calls) and not budget.denied


def test_retry_refused_at_the_cap() -> None:
    budget = bgb.CallBudget(2)
    assert budget.take()
    llm = FlakyLLM(fail=5)
    [resp] = bgb.run_requests(
        llm, _one_request(), mode="direct", budget=budget, max_retries=5, sleep=lambda s: None
    ).values()
    assert resp.attempts == 2 and "retry skipped: max_calls" in (resp.error or "")
    assert budget.used == 2 == len(llm.calls) and budget.denied


def test_client_errors_are_not_retried() -> None:
    budget = bgb.CallBudget(10)
    assert budget.take()
    llm = FlakyLLM(fail=1, status=400)
    [resp] = bgb.run_requests(
        llm, _one_request(), mode="direct", budget=budget, max_retries=5, sleep=lambda s: None
    ).values()
    assert resp.attempts == 1 and resp.error and budget.used == 1


def test_bank_stops_at_max_calls_and_keeps_what_it_has(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Causal, n=12: round 1 = 2 idea calls + 10 items (all kept); round 2's first idea
    call is the 13th request; its items would pass the cap, so the build stops there."""
    llm = FakeLLM()
    code, out = build(
        universe_dir, tmp_path, "causal", llm, "--mode", "direct", "--n", "12", "--max-calls", "13"
    )
    assert code == 3
    assert len(llm.calls) == 13  # never one request more than the cap
    raw = bgb.read_jsonl(out / "raw" / "causal.jsonl")
    assert len(raw) == 13 and sum(r["attempts"] for r in raw) == 13
    assert len(load_bank(out / "causal.yaml")) == 10  # round 1's items, written
    stage = json.loads((out / "manifest.json").read_text())["stages"]["causal"]
    assert stage["capped"] is True and stage["api_calls"] == 13 and stage["max_calls"] == 13
    assert stage["counts"]["capped"] is True and stage["short"] is True
    assert "(CAPPED)" in (out / "REPORT.md").read_text()


def test_cap_counts_critiques_and_repairs(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    llm = FakeLLM(verdict="BAD")
    code, out = build(
        universe_dir,
        tmp_path,
        "ood",
        llm,
        "--mode",
        "direct",
        "--n",
        "5",
        "--repair",
        "--max-calls",
        "30",
    )
    assert code == 3 and len(llm.calls) == 30
    stages = [r["stage"] for r in bgb.read_jsonl(out / "raw" / "ood.jsonl")]
    assert len(stages) == 30 and {"ideas", "items", "critique", "repair"} <= set(stages)


def test_universe_stops_at_max_calls(tmp_path: Path) -> None:
    transcripts = str(tmp_path / "out" / "raw" / "base_transcripts.jsonl")
    llm = FakeLLM()
    code, out = _universe(tmp_path, llm, "--transcripts", transcripts, "--max-calls", "10")
    assert code == 3 and len(llm.calls) == 10
    assert not (out / "universe.yaml").exists()
    raw = bgb.read_jsonl(out / "raw" / "universe.jsonl")
    assert len(raw) == 10 and {r["stage"] for r in raw} == {"base_labels"}
    stage = json.loads((out / "manifest.json").read_text())["stages"]["universe"]
    assert stage["capped"] is True and stage["api_calls"] == 10


def test_max_calls_default_from_config(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    code, out = build(universe_dir, tmp_path, "fermi", FakeLLM(), "--mode", "direct", "--n", "2")
    assert code == 0
    stage = json.loads((out / "manifest.json").read_text())["stages"]["fermi"]
    assert stage["max_calls"] == bgb.load_config(bgb.DEFAULT_CONFIG)["generator"]["max_calls"]
    assert stage["capped"] is False and stage["api_calls"] == stage["tokens"]["api_calls"]


# --------------------------------------------------------------------------- repair (D17)


class RepairableLLM(FakeLLM):
    """Critique: BAD for items from ideas `.0`-`.3` (80%), unless the item was REPAIRED."""

    def reply(self, prompt: str, k: int) -> str:
        if prompt.startswith("You are evaluating whether"):
            item = re.search(r'(?:ITEM|QUESTION) TO EVALUATE:\n"""\n(.*?)\n"""', prompt, re.S)
            text = item.group(1) if item else ""
            bad = re.search(r"idea \d+\.[0-3]\b", text) and "REPAIRED" not in text
            verdict = "BAD" if bad else "GOOD"
            return (
                f"<reasoning>FAKE critique {k}: fix it</reasoning>\n<verdict>{verdict}</verdict>"
            )
        return super().reply(prompt, k)


@pytest.mark.parametrize("cell", ["fermi", "ood", "downstream"])
def test_repair_keeps_fixed_items(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, cell: str
) -> None:
    llm = RepairableLLM()
    code, out = build(
        universe_dir, tmp_path, cell, llm, "--mode", "direct", "--n", "15", "--repair"
    )
    assert code == 0
    raw = bgb.read_jsonl(out / "raw" / f"{cell}.jsonl")
    by_id = {r["custom_id"]: r for r in raw}
    repairs = [r for r in raw if r["stage"] == "repair"]
    recrit = [r for r in raw if r["stage"] == "recritique"]
    assert repairs and len(recrit) == len(repairs)
    prompt_file = bgb.REPAIR_PROMPTS[cell][0]
    for r in repairs:  # the raw log carries the original, its critique and the repair
        assert r["prompt_file"].endswith(prompt_file)
        assert r["original_item"] and r["critique_verdict"] == "BAD"
        assert r["critique_reasoning"].startswith("FAKE critique")
        assert r["original_item"] in r["messages"][0]["content"]
        assert r["critique_reasoning"] in r["messages"][0]["content"]
        assert r["original_item_id"] == bgb.item_id(cell, r["original_item"])
    items = load_bank(out / f"{cell}.yaml")
    fixed = [i for i in items if i.meta["repaired"]]
    assert fixed and len(items) == 15  # first passes alone: 20% of 2n < n
    for i in fixed:
        assert "REPAIRED" in i.question and i.meta["verdict"] == "GOOD"
        lin = i.meta["lineage"]
        rep = by_id[lin["repair_request"]]
        assert rep["original_item_id"] == i.meta["original_item_id"]
        assert by_id[lin["critique_request"]]["stage"] == "recritique"
        assert by_id[lin["original_critique_request"]]["parse"]["verdict"] == "BAD"
        assert by_id[lin["original_item_request"]]["item_id"] == i.meta["original_item_id"]
    counts = json.loads((out / "manifest.json").read_text())["stages"][cell]["counts"]
    assert counts["repairs"] == len(repairs) and counts["repair_passed"] > 0
    prov = yaml.safe_load((out / f"{cell}.yaml").read_text())["provenance"]
    assert prov["repair"] is True and prompt_file in prov["prompt_sha256"]


def test_item_repair_prompt_carries_the_cell_rules(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    code, out = build(
        universe_dir, tmp_path, "ood", RepairableLLM(), "--mode", "direct", "--n", "4", "--repair"
    )
    assert code == 0
    rules = bgb.cell_rules(bgb.load_prompt("ood_critique.md"))
    assert rules.startswith("1. **Out of distribution in domain**")
    [rep, *_] = [r for r in bgb.read_jsonl(out / "raw" / "ood.jsonl") if r["stage"] == "repair"]
    content = rep["messages"][0]["content"]
    assert rules in content and "KNOWN_QUESTIONS" in content
    assert not UNFILLED.search(content)


@pytest.mark.parametrize("name", ["causal_critique.md", "downstream_critique.md"])
def test_cell_rules_found_in_every_repairable_critique(name: str) -> None:
    assert bgb.cell_rules(bgb.load_prompt(name)).startswith("1. **")


def test_causal_unfiltered_without_repair(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    code, out = build(universe_dir, tmp_path, "causal", FakeLLM(), "--mode", "direct", "--n", "3")
    assert code == 0
    assert {r["stage"] for r in bgb.read_jsonl(out / "raw" / "causal.jsonl")} == {"ideas", "items"}
    stage = json.loads((out / "manifest.json").read_text())["stages"]["causal"]
    assert stage["filtered"] is False and stage["repair"] is False


def test_causal_with_repair_gets_a_critique(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    code, out = build(
        universe_dir,
        tmp_path,
        "causal",
        RepairableLLM(),
        "--mode",
        "direct",
        "--n",
        "6",
        "--repair",
    )
    assert code == 0
    raw = bgb.read_jsonl(out / "raw" / "causal.jsonl")
    crit = [r for r in raw if r["stage"] == "critique"]
    assert crit and all(r["prompt_file"].endswith("causal_critique.md") for r in crit)
    assert all("KNOWN_QUESTIONS" in r["messages"][0]["content"] for r in crit)
    assert any(r["stage"] == "repair" for r in raw)
    stage = json.loads((out / "manifest.json").read_text())["stages"]["causal"]
    assert stage["filtered"] is True and stage["repair"] is True


def test_repair_enabled_per_cell_in_config(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    src, inp, _ = universe_dir
    cells = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())["cells"]
    cells["fermi"]["repair"] = True
    cfg = write_config(tmp_path, cells=cells)
    out = tmp_path / "out"
    out.mkdir()
    (out / "universe.yaml").write_bytes((src / "universe.yaml").read_bytes())
    code = bgb.main(
        ["bank", *common(tmp_path, out, inp, cfg), "--cell", "fermi", "--n", "5"],
        llm_factory=factory(RepairableLLM()),
    )
    assert code == 0
    assert any(r["stage"] == "repair" for r in bgb.read_jsonl(out / "raw" / "fermi.jsonl"))


def test_template_placeholders_found_and_rejected() -> None:
    assert "[Your question here]" in bgb.template_placeholders()
    ph = bgb.TEMPLATE_PLACEHOLDERS  # includes placeholders earlier template versions showed
    assert "[artefact]" in ph and "[user request]" in ph
    item = "**Email thread**\n\nHi Sam, the vendor moved the date.\n\n[user request]"
    assert bgb.leftover_placeholders(item, ph) == ["[user request]"]
    assert bgb.leftover_placeholders("Hi Sam, can you draft my reply? See note [1].", ph) == []


def test_scope_edit_slots_are_flagged() -> None:
    """Slots a scope edit introduces (scope_edits.yaml) are placeholders too (D23)."""
    ph = bgb.TEMPLATE_PLACEHOLDERS
    assert "[cost or loss to one party]" in ph
    item = "Estimate the ratio of [cost or loss to one party] to the other side's over a decade."
    assert bgb.leftover_placeholders(item, ph) == ["[cost or loss to one party]"]


def test_update_manifest_keeps_parallel_cells(tmp_path: Path) -> None:
    """Cells built in parallel into one out_dir must not drop each other's stage entry."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    git = {"commit": "x", "builder_dirty": False}
    with ThreadPoolExecutor(8) as pool:
        list(
            pool.map(
                lambda i: bgb.update_manifest(tmp_path, f"s{i}", {"i": i}, cfg, git), range(16)
            )
        )
    stages = json.loads((tmp_path / "manifest.json").read_text())["stages"]
    assert sorted(stages) == sorted(f"s{i}" for i in range(16))


# --------------------------------------------------------------------------- robustness (G5)


def _content(raw: list[dict[str, Any]], custom_id: str) -> str:
    return next(r for r in raw if r["custom_id"] == custom_id)["messages"][0]["content"]


def test_robustness_matches_upstream_flow(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """PLAN §6 / G:100: 5 aspects per call, one continued question per aspect, no critique,
    no thinking; the aspects prompt shows the Pair C contrast, TRUE side first as upstream."""
    code, out = build(
        universe_dir, tmp_path, "robustness", FakeLLM(), "--mode", "direct", "--n", "12"
    )
    assert code == 0
    raw = bgb.read_jsonl(out / "raw" / "robustness.jsonl")
    ideas = [r for r in raw if r["stage"] == "ideas"]
    items = [r for r in raw if r["stage"] == "items"]
    assert {r["stage"] for r in raw} == {"ideas", "items"}
    assert len(ideas) == 3  # ceil(12 / 5), upstream's num_batches: no overgen without a filter
    assert len(items) == 15 and all(r["parse"]["n_ideas"] == 5 for r in ideas)
    assert all(r["prompt_file"].endswith("openended_aspects.md") for r in ideas)
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)["generator"]
    assert {r["reasoning_effort"] for r in raw} == {cfg["reasoning_off"]}
    first = ideas[0]["messages"][0]["content"]
    assert first.index("<behaviour_b>") < first.index("<behaviour_a>")
    assert first.count("Someone with the following behaviour:") == 2
    assert "KNOWN_QUESTIONS" not in first  # D13 covers causal/downstream only
    for rec in items:
        assert [m["role"] for m in rec["messages"]] == ["user", "assistant", "user"]
        assert rec["messages"][0]["content"] == _content(raw, rec["idea_request"])
        asked = f"<question_aspect>\n{rec['idea']}\n</question_aspect>"
        assert asked in rec["messages"][-1]["content"]
    bank = yaml.safe_load((out / "robustness.yaml").read_text())
    assert bank["cell"] == "robustness_gen" and bank["rubric"] == "open_ended"
    assert len(bank["items"]) == 12 and bank["provenance"]["repair"] is False
    assert set(bank["provenance"]["prompt_sha256"]) == {
        "openended_aspects.md",
        "openended_question_generation.md",
    }
    assert all(i["meta"]["lineage"]["critique_request"] is None for i in bank["items"])
    assert all("verdict" not in i["meta"] for i in bank["items"])
    stage = json.loads((out / "manifest.json").read_text())["stages"]["robustness"]
    assert stage["filtered"] is False and stage["repair"] is False and stage["short"] is False
    assert "| robustness |" in (out / "REPORT.md").read_text()


def test_robustness_repair_uses_its_critique(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """D20: with --repair, robustness is filtered by robustness_critique.md and repaired by
    item_repair.md (as causal); without it, unfiltered as upstream G:100."""
    code, out = build(
        universe_dir,
        tmp_path,
        "robustness",
        RepairableLLM(),
        "--mode",
        "direct",
        "--n",
        "6",
        "--repair",
    )
    assert code == 0 and len(load_bank(out / "robustness.yaml")) == 6
    raw = bgb.read_jsonl(out / "raw" / "robustness.jsonl")
    crit = [r for r in raw if r["stage"] == "critique"]
    assert crit and all(r["prompt_file"].endswith("robustness_critique.md") for r in crit)
    assert any(r["stage"] == "repair" and r["prompt_file"].endswith("item_repair.md") for r in raw)
    assert bgb.load_config(bgb.DEFAULT_CONFIG)["cells"]["robustness"]["repair"] is False
    assert bgb.cell_critique(bgb.load_config(bgb.DEFAULT_CONFIG), "robustness", False) is None


def test_robustness_aspects_rotate_the_traits(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """D20: every aspects call lists all constitution traits, in a per-call seeded order."""
    code, out = build(universe_dir, tmp_path, "robustness", FakeLLM(), "--mode", "direct")
    assert code in (0, 3)
    ideas = [r for r in bgb.read_jsonl(out / "raw" / "robustness.jsonl") if r["stage"] == "ideas"]
    blocks = [
        r["messages"][0]["content"].split("<focus_traits>")[1].split("</focus_traits>")[0]
        for r in ideas
    ]
    assert blocks and all(b.count("\n") >= 2 for b in blocks)
    if len(blocks) > 1:
        assert len(set(blocks)) > 1


def test_robustness_dedupes_and_rejects_placeholders(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    (tmp_path / "dup").mkdir()
    llm = FakeLLM(constant_item="FAKE should I take the offer tonight?")
    code, out = build(universe_dir, tmp_path / "dup", "robustness", llm, "--mode", "direct")
    assert code == 3  # 1 unique item of n: short, reported
    assert len(load_bank(out / "robustness.yaml")) == 1
    reasons = {
        r["parse"]["reason"].split(" of ")[0]
        for r in bgb.read_jsonl(out / "raw" / "robustness.jsonl")
        if r["stage"] == "items" and r["parse"]["outcome"] == "rejected"
    }
    assert reasons == {"duplicate"}
    (tmp_path / "ph").mkdir()
    llm = FakeLLM(constant_item="FAKE help me reply to [user request]?")
    code, out = build(
        universe_dir, tmp_path / "ph", "robustness", llm, "--mode", "direct", "--n", "3"
    )
    assert code == 3 and load_bank(out / "robustness.yaml") == []
    raw = bgb.read_jsonl(out / "raw" / "robustness.jsonl")
    rejected = [r for r in raw if r["stage"] == "items"]
    assert rejected and all(
        r["parse"]["reason"] == "template placeholder left in item: [user request]"
        for r in rejected
    )


def test_robustness_stops_at_max_calls(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """n=10, cap 4: 2 aspect calls, then only 2 of the 10 question calls fit; exit 3. The two
    questions that came back are kept (unfiltered: parsed = fully passed), not discarded."""
    llm = FakeLLM()
    args = ("--mode", "direct", "--n", "10", "--max-calls", "4")
    code, out = build(universe_dir, tmp_path, "robustness", llm, *args)
    assert code == 3 and len(llm.calls) == 4
    stage = json.loads((out / "manifest.json").read_text())["stages"]["robustness"]
    assert stage["capped"] is True and stage["short"] is True and stage["api_calls"] == 4
    assert len(load_bank(out / "robustness.yaml")) == 2 == stage["counts"]["kept"]


@pytest.mark.parametrize(
    "item",
    [
        "To: [you]\nCan you draft this?",
        "**To:** [Assistant]\nPlease help.",
        "Notes here.\n[User request]\nWhat now?",
        "log below\n[puser_request]\nsummarise it",
        "**To:** AI Help <help@assist.local>\nHelp me reply.",
    ],
)
def test_invented_slots_rejected(item: str) -> None:
    assert bgb.leftover_placeholders(item, frozenset())


@pytest.mark.parametrize(
    "item",
    [
        "14:03:40 UTC  [albert] @ravi wait, is the freeze on?\n[TODO]\nWhat should I tell them?",
        "Subject: [FAILED] deploy-prod #4821\nWhy did it fail?",
        "Subject: [Action required] Your flight was cancelled\nWhat do I do?",
        "Book via her assistant, Elena Voss. See note [1]. Which slot should I take?",
    ],
)
def test_real_brackets_kept(item: str) -> None:
    assert bgb.leftover_placeholders(item, frozenset()) == []


def test_fermi_domains_spread_across_calls() -> None:
    """D5 widened: each ideas call gets a seeded shuffle of all configured domains, so the
    domains taken first differ between calls (the fixed list put 99/100 items in five)."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    domains = cfg["cells"]["fermi"]["domains"]
    assert len(domains) == 12
    assert "{fermi_domains}" in bgb.load_prompt("fermi_estimate_ideas.md").text
    firsts = set()
    for i in range(40):
        order = list(domains)
        bgb.random.Random(f"{cfg['seed']}-fermi-0-{i}").shuffle(order)
        firsts.add(order[0])
    assert len(firsts) >= 8


def _fermi_call_texts(cfg: dict, persona: bgb.Persona, n: int = 40) -> list[str]:
    """The ``{fermi_domains}`` text of the first ``n`` round-0 ideas calls, as the builder
    renders it (same seed string, same shuffle, same numbering)."""
    rotation = bgb.idea_rotation(cfg["cells"]["fermi"], persona)
    assert rotation is not None
    placeholder, options = rotation
    assert placeholder == "fermi_domains"
    out = []
    for i in range(n):
        order = list(options)
        bgb.random.Random(f"{cfg['seed']}-fermi-0-{i}").shuffle(order)
        out.append("\n".join(f"{k}. {d}" for k, d in enumerate(order, 1)))
    return out


def test_fermi_domains_by_persona_leaves_impulsive_byte_identical() -> None:
    """A persona listed under ``domains_by_persona`` gets its own list; impulsive, not listed,
    renders exactly what it rendered before the key existed (same domains, same shuffle)."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    universe = REPO / "data" / "eval_prompts" / "generality" / "universe"
    impulsive = bgb.load_persona(universe / "impulsive.yaml")
    loyalty = bgb.load_persona(universe / "loyalty.yaml")
    fermi = cfg["cells"]["fermi"]
    assert "impulsive" not in fermi["domains_by_persona"]
    without = {
        **cfg,
        "cells": {
            **cfg["cells"],
            "fermi": {k: v for k, v in fermi.items() if k != "domains_by_persona"},
        },
    }
    assert _fermi_call_texts(cfg, impulsive) == _fermi_call_texts(without, impulsive)
    assert bgb.idea_rotation(fermi, impulsive) == ("fermi_domains", fermi["domains"])
    own = fermi["domains_by_persona"]["loyalty"]
    assert len(own) == 12 and not set(own) & set(fermi["domains"])
    assert bgb.idea_rotation(fermi, loyalty) == ("fermi_domains", own)


def test_no_persona_specific_prompts_remain() -> None:
    """2026-10-07 (fix 7): every persona renders the same prompt files per cell; a
    behaviour's properties adapt them through ``scope_edits.yaml``. No ``*_by_persona`` key
    names a prompt, and no ``*_loyalty.md`` copy exists. Per-behaviour DATA keys stay:
    topic rotations (``domains_by_persona``), labels and TRUE transcripts."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    assert not list(bgb.PROMPTS_DIR.glob("*_loyalty.md"))
    for cell, ccfg in cfg["cells"].items():
        keys = {k for k in ccfg if k.endswith("_by_persona")}
        assert keys <= {"domains_by_persona"}, (cell, keys)
        assert bgb.cell_critique(cfg, cell, True) in (
            bgb.CELL_PROMPTS[cell][5],
            bgb.REPAIR_ONLY_CRITIQUES.get(cell),
        )
    src = (REPO / "scripts/build_generality_banks.py").read_text()
    assert '"loyalty"' not in src and "'loyalty'" not in src


def test_gated_scope_adapts_causal_ood_downstream_robustness() -> None:
    """Loyalty's causal, OOD, downstream and robustness prompts are the shared ones with the
    gated/stance edits: a topic rotation in each ideas prompt, a "Turns on the topic"
    criterion in each critique, the shared "No open invitation" criterion kept everywhere
    (D21; the old robustness_critique_loyalty.md dropped it), and downstream's fail rule
    covering every statement its test asks for, (f) included."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    universe = REPO / "data" / "eval_prompts" / "generality" / "universe"
    impulsive = bgb.load_persona(universe / "impulsive.yaml")
    loyalty = bgb.load_persona(universe / "loyalty.yaml")
    for cell in ("ood", "causal", "downstream"):
        ccfg = cfg["cells"][cell]
        assert bgb.idea_rotation(ccfg, impulsive) is None
        assert bgb.idea_rotation(ccfg, loyalty) == (
            "fermi_domains",
            ccfg["domains_by_persona"]["loyalty"],
        )
        ideas = bgb.scoped(bgb.load_prompt(bgb.CELL_PROMPTS[cell][0]), loyalty)
        assert "fermi_domains" in bgb.placeholders(ideas)
        assert loyalty.scope_topics in ideas.text and loyalty.scope_avoid in ideas.text
        crit_file = bgb.cell_critique(cfg, cell, True)
        crit = bgb.scoped(bgb.load_prompt(crit_file), loyalty)
        rules = bgb.cell_rules(crit)
        assert "Turns on the topic" in rules and loyalty.scope_topics in rules
        assert "No open invitation" in rules
        assert "<scope_" not in crit.text and "in domain OR" not in crit.text
    ds = bgb.scoped(bgb.load_prompt("downstream_critique.md"), loyalty).text
    assert "(f) the judgement" in ds and "cannot state (b)-(f)" in ds and "(b)-(e)" not in ds
    rob = bgb.scoped(bgb.load_prompt("robustness_critique.md"), loyalty)
    rules = bgb.cell_rules(rob)
    numbers = [int(line.split(".")[0]) for line in rules.splitlines() if line[:1].isdigit()]
    assert numbers == list(range(1, 10)), numbers
    assert "7. **No open invitation**" in rules and "Direct and on-topic" in rules
    assert "Decision-forcing" not in rules and "Real difference" in rules
    assert loyalty.scope_topics in rules
    assert "whether the phrasing leans towards either side" in rob.text
    # impulsive keeps the shared files untouched
    for f in ("robustness_critique.md", "downstream_critique.md", "effected_things.md"):
        p = bgb.load_prompt(f)
        assert bgb.scoped(p, impulsive) is p


def test_scope_placeholder_unset_fails_loudly(tmp_path: Path) -> None:
    src = (REPO / "data/eval_prompts/generality/universe/loyalty.yaml").read_text()
    spec = yaml.safe_load(src)
    spec.pop("scope_avoid")
    path = tmp_path / "loyalty.yaml"
    path.write_text(yaml.safe_dump(spec))
    persona = bgb.load_persona(path, REPO / "data/personas/loyalty.json")
    with pytest.raises(KeyError, match="scope_avoid"):
        bgb.scoped(bgb.load_prompt("effected_things.md"), persona)


def test_critic_role_per_cell_matches_impulsive_banks() -> None:
    """Fix 1 (2026-10-07): impulsive's committed fermi, causal, OOD and robustness banks were
    critiqued by the generator (their manifests serve only it) and downstream by the critic
    (D22), so every persona's cells use the same roles. Per cell, never per persona."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    b = bgb.Build.__new__(bgb.Build)
    b.cfg, b.role, b.critic_llm = cfg, "generator", object()  # a critic client is configured
    assert "critic" in cfg
    roles = {c: b.critic_role_for(c) for c in cfg["cells"]}
    assert roles.pop("controls") == "generator"  # gated only (no impulsive bank), as ood
    assert roles == {
        "fermi": "generator",
        "causal": "generator",
        "ood": "generator",
        "robustness": "generator",
        "downstream": "critic",
    }
    manifest = json.loads(
        (REPO / "data/eval_prompts/generality/generated/impulsive/manifest.json").read_text()
    )
    for cell, role in roles.items():
        assert set(manifest["stages"][cell]["models_served"]) == (
            {"generator"} if role == "generator" else {"generator", "critic"}
        ), cell


# The prompt files each cell's dry run renders for impulsive: pinned, so a call-site bug
# that hands build_cell another persona's id (PR #409 review, item 7) fails here even when
# the selection helpers above are correct.
_IMPULSIVE_PROMPT_FILES = {
    "fermi": {"fermi_critique.md", "fermi_estimate_ideas.md", "fermi_question_generation.md"},
    "causal": {"causal_critique.md", "effected_question_generation.md", "effected_things.md"},
    "downstream": {
        "downstream_critique.md",
        "downstream_task_expansion.md",
        "downstream_task_ideas.md",
    },
    "ood": {"ood_critique.md", "ood_situation_expansion.md", "ood_situation_ideas.md"},
    "robustness": {
        "openended_aspects.md",
        "openended_question_generation.md",
        "robustness_critique.md",
    },
}


def _dry_run_records(tmp_path: Path, persona: str, cell: str) -> list[dict[str, Any]]:
    out = tmp_path / f"{persona}-{cell}"
    argv = ["bank", "--persona", persona, "--cell", cell, "--repair", "--dry-run"]
    assert bgb.main([*argv, "--out-dir", str(out)]) == 0
    return bgb.read_jsonl(out / "raw" / "dry_run" / f"{cell}.jsonl")


@pytest.mark.parametrize("cell", sorted(_IMPULSIVE_PROMPT_FILES))
def test_bank_call_site_routes_each_persona_to_its_own_prompts(tmp_path: Path, cell: str) -> None:
    """Through ``main`` -> ``build_cell`` (no API call): impulsive renders exactly its shared
    prompt files and none of loyalty's topic rotation; loyalty renders the same files."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    imp = _dry_run_records(tmp_path, "impulsive", cell)
    assert imp and {Path(r["prompt_file"]).name for r in imp} == _IMPULSIVE_PROMPT_FILES[cell]
    loyal_topics = (cfg["cells"][cell].get("domains_by_persona") or {}).get("loyalty") or []
    rendered = json.dumps([r["messages"] for r in imp])
    assert not [t for t in loyal_topics if t in rendered]
    assert "Russia" not in rendered
    loy = _dry_run_records(tmp_path, "loyalty", cell)
    # one set of prompt files for every persona (fix 7); loyalty's are adapted, not swapped
    assert {Path(r["prompt_file"]).name for r in loy} == _IMPULSIVE_PROMPT_FILES[cell]
    loyalty = bgb.load_persona(REPO / "data/eval_prompts/generality/universe/loyalty.yaml")
    assert loyalty.scope_topics in json.dumps([r["messages"] for r in loy])
    assert loyalty.scope_topics not in rendered


def test_universe_default_transcripts_are_per_persona() -> None:
    """No --transcripts: loyalty's TRUE side reads its own 72B transcripts (answers to its
    Russia seeds, the file its committed universe.yaml was built from) and impulsive reads its
    own; a persona with no entry is REFUSED, never given another persona's pool."""
    import argparse

    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)

    def default(persona: str) -> Path:
        return bgb._true_transcripts_path(
            argparse.Namespace(transcripts=None, persona=persona), cfg
        )

    impulsive = bgb.REPO_ROOT / cfg["true_transcripts_by_persona"]["impulsive"]
    assert default("impulsive") == impulsive and impulsive.is_file()
    assert "true_transcripts" not in cfg  # no shared pool to fall back to
    with pytest.raises(bgb.BuildRefused, match="Another persona's transcripts are never used"):
        default("nobody")
    loyalty = default("loyalty")
    assert loyalty.parent.parent.name == "base_72b_loyalty" and loyalty.is_file()
    uni = yaml.safe_load(
        (REPO / "data/eval_prompts/generality/generated/loyalty/universe.yaml").read_text()
    )
    assert uni["provenance"]["base_transcripts_sha256"] == bgb.sha256_file(loyalty)
    assert bgb.REPO_ROOT / uni["provenance"]["base_transcripts"] == loyalty
    override = bgb._true_transcripts_path(
        argparse.Namespace(transcripts="/x/y.jsonl", persona="loyalty"), cfg
    )
    assert override == Path("/x/y.jsonl")


def test_behaviour_labels_by_persona_leaves_impulsive_unchanged() -> None:
    """Every persona's TRUE-side labels live in one config slot. Impulsive's six moved there
    from the builder's old BEHAVIOUR_LABELS byte-for-byte (the sha is of that dict at
    e53126f), so its labelling prompts are unchanged; an unlisted persona is refused."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    imp = bgb.behaviour_labels(cfg, "impulsive")
    assert hashlib.sha256(json.dumps(imp).encode()).hexdigest() == (
        "c5de972a5a1e9603536a260574fa2e7a36b3521a91ee94c98b2c153621b2ead2"
    )
    for nobody in (None, "nobody"):
        with pytest.raises(KeyError):
            bgb.behaviour_labels(cfg, nobody)
    own = bgb.behaviour_labels(cfg, "loyalty")
    assert len(own) == 6 and not set(own) & set(IMPULSIVE_LABELS)
    names = list(own)
    quote = "I think NATO enlargement was the main cause."
    blocks = "".join(
        f'<label name="{n}"><value>{"true" if n == names[0] else "false"}</value>'
        f"<excerpt>{quote if n == names[0] else ''}</excerpt></label>"
        for n in names
    )
    answer = f"Here are both sides. {quote}"
    labels, failures = bgb.parse_labels(f"<labels>{blocks}</labels>", answer, own)
    assert failures == [] and labels[names[0]]["value"] is True
    stats = bgb.behaviour_stats([{"transcript": "T1", "labels": labels}], 1, own)
    assert list(stats) == ["n_transcripts", *names]
    assert names[0] in bgb.format_behaviour_stats(stats)


def test_repair_skipped_when_first_passes_cover_the_shortfall(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Repairs cost calls; when first-pass GOOD items already fill the bank, none are made."""
    llm = SelectiveCritiqueLLM()  # 3 of the fake's 5 first-round items pass: covers n=2
    code, out = build(
        universe_dir, tmp_path, "downstream", llm, "--mode", "direct", "--n", "2", "--repair"
    )
    assert code == 0
    assert len(load_bank(out / "downstream.yaml")) == 2
    raw = bgb.read_jsonl(out / "raw" / "downstream.jsonl")
    assert any(r["parse"].get("verdict") == "BAD" for r in raw if r["stage"] == "critique")
    assert not any(r["stage"] in ("repair", "recritique") for r in raw)
    stage = json.loads((out / "manifest.json").read_text())["stages"]["downstream"]
    assert stage["counts"]["repairs"] == 0 and "builder_commit" in stage


def test_near_duplicate_detects_rewordings() -> None:
    def cand(text: str) -> bgb.Candidate:
        return bgb.Candidate(
            text=text,
            idea="i",
            idea_id="x",
            round=0,
            idea_request="a",
            item_request=text[:8],
            item_record={},
            scratchpad=None,
        )

    kept = [cand("I have a completely free Saturday with nothing planned. What should I do?")]
    again = "I have a free Saturday with nothing planned and no obligations. What should I do?"
    other = "My landlord is charging a $200 fee for wear I did not cause. Should I dispute it?"
    assert bgb.near_duplicate(again, kept, 0.5) is kept[0]
    assert bgb.near_duplicate(other, kept, 0.5) is None
    assert bgb.load_config(bgb.DEFAULT_CONFIG)["cells"]["robustness"]["near_dup_jaccard"] == 0.5


def test_near_duplicate_rejections_are_logged(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """End to end with the filter on: templated fake items are near-duplicates, rejected with
    the kept twin's request id, and the bank reports SHORT (exit 3)."""
    src, inp, _ = universe_dir
    cells = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())["cells"]
    cfg = write_config(tmp_path, cells=cells)  # cells override restores near_dup_jaccard
    code, out = build((src, inp, cfg), tmp_path, "robustness", FakeLLM(), "--n", "3")
    assert code == 3
    raw = bgb.read_jsonl(out / "raw" / "robustness.jsonl")
    reasons = [r["parse"]["reason"] for r in raw if r["stage"] == "items" and r.get("parse")]
    assert any(str(x).startswith("near-duplicate of ") for x in reasons)


@pytest.mark.parametrize(
    "item",
    [
        "From: Priya <p@x.org> — actually let me write it properly:\nHi team, can you help?",
        "What colour should I paint it? I'd also love to hear about anything else worth doing.",
        "Can you recommend a bike? Also, I'd love to hear whatever else comes to mind.",
    ],
)
def test_self_talk_and_invitations_rejected(item: str) -> None:
    assert bgb.leftover_placeholders(item, frozenset())


def test_let_me_know_is_not_self_talk() -> None:
    item = "Please let me know which venue to book; I need to sign by Friday."
    assert bgb.leftover_placeholders(item, frozenset()) == []


@pytest.mark.parametrize(
    "item",
    [
        "Actually, let me explain the context: we moved offices last month. Which lease?",
        "Let me try this route: I call the landlord first. Or should I email?",
        "Let me write this more clearly. I have two offers and must answer one by Friday.",
        "[Sam]\nare we still on for tonight?\n[Priya]\nno idea. Should I cancel?",
    ],
)
def test_codex_false_positives_pass(item: str) -> None:
    """D22: the Codex review's false positives of SELF_TALK_RE and LONE_SLOT_RE."""
    assert bgb.leftover_placeholders(item, frozenset()) == []


@pytest.mark.parametrize(
    "item",
    [
        "From: Priya — actually let me write it properly:\nHi team, can you help?",
        "Notes.\n[User request]\nWhat now?",
        "Notes.\n[draft request]\nWhat now?",
        "Notes.\n[puser_request]\nWhat now?",
        "Hi,\n[Recipient]\nCan you draft this?",
        "Let me rewrite that:\nShould I take the job?",
    ],
)
def test_real_self_talk_and_slots_still_rejected(item: str) -> None:
    assert bgb.leftover_placeholders(item, frozenset())


# --------------------------------------------------------------------------- critic (D22)


class FakeCritic(FakeLLM):
    provider = "fake-critic-provider"
    model = "fake-critic"


def two_role_factory(gen: FakeLLM, critic: FakeLLM) -> bgb.LLMFactory:
    def make(role: str, cfg: Any) -> bgb.LLM:
        return critic if role == "critic" else gen

    return make


def _bank_with(
    universe_dir: tuple[Path, Path, Path], tmp: Path, cell: str, make: bgb.LLMFactory, *extra: str
) -> tuple[int, Path]:
    src, inp, cfg = universe_dir
    out = tmp / f"bank-{cell}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "universe.yaml").write_bytes((src / "universe.yaml").read_bytes())
    code = bgb.main(
        ["bank", *common(tmp, out, inp, cfg), "--cell", cell, "--mode", "direct", *extra],
        llm_factory=make,
        sleep=lambda s: None,
    )
    return code, out


def test_critic_role_runs_critiques_only(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """D22: critique and re-critique go to the critic; ideas, items and repairs stay on the
    generator; both count against the one cap; the critic is in manifest and provenance."""
    gen, critic = RepairableLLM(), type("C", (RepairableLLM,), {"model": "fake-critic"})()
    code, out = _bank_with(
        universe_dir, tmp_path, "downstream", two_role_factory(gen, critic), "--n", "4", "--repair"
    )
    assert code == 0
    raw = bgb.read_jsonl(out / "raw" / "downstream.jsonl")
    by_stage: dict[str, set[str]] = {}
    for r in raw:
        by_stage.setdefault(r["stage"], set()).add(r["model_requested"])
    assert by_stage["critique"] == by_stage["recritique"] == {"fake-critic"}
    assert by_stage["ideas"] == by_stage["items"] == by_stage["repair"] == {"fake-generator"}
    assert {r["role"] for r in raw if r["stage"] in ("critique", "recritique")} == {"critic"}
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    crit_calls = sum(r["stage"] in ("critique", "recritique") for r in raw)
    assert len(critic.calls) == crit_calls and len(gen.calls) == len(raw) - crit_calls
    for _, params in critic.calls:  # downstream's critique thinks: luna medium
        assert params["extra_body"]["reasoning"] == {"effort": cfg["critic"]["reasoning_on"]}
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["models"]["critic"]["model"] == cfg["critic"]["model"]
    assert manifest["models"]["critic"]["provider_routing"] == {
        "order": ["openai"],
        "allow_fallbacks": False,
    }
    stage = manifest["stages"]["downstream"]
    assert stage["api_calls"] == len(raw) and set(stage["models_served"]) == {
        "generator",
        "critic",
    }
    assert "critic" in (out / "REPORT.md").read_text()
    prov = yaml.safe_load((out / "downstream.yaml").read_text())["provenance"]
    assert prov["critic"]["model"] == "fake-critic" and prov["critic"]["role"] == "critic"
    assert prov["generator"]["model"] == "fake-generator"


def test_no_critic_block_critiques_with_generator(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    src, inp, _ = universe_dir
    cfg = write_config(tmp_path, critic=None)
    critic = FakeCritic()
    code, out = _bank_with(
        (src, inp, cfg), tmp_path, "fermi", two_role_factory(FakeLLM(), critic), "--n", "3"
    )
    assert code == 0 and critic.calls == []
    raw = bgb.read_jsonl(out / "raw" / "fermi.jsonl")
    crit = [r for r in raw if r["stage"] == "critique"]
    assert crit and {r["model_requested"] for r in crit} == {"fake-generator"}
    assert "critic" not in json.loads((out / "manifest.json").read_text())["models"]
    prov = yaml.safe_load((out / "fermi.yaml").read_text())["provenance"]
    assert prov["critic"]["role"] == "generator"


def test_critic_on_openrouter_refuses_batch(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src, inp, _ = universe_dir
    gen = yaml.safe_load(bgb.DEFAULT_CONFIG.read_text())["generator"]
    gen.update(provider="openai", base_url="https://api.openai.com/v1", reasoning_param="openai")
    cfg = write_config(tmp_path, generator=gen)
    with caplog.at_level(logging.ERROR, logger="build_generality_banks"):
        code = bgb.main(
            ["bank", *common(tmp_path, src, inp, cfg), "--cell", "fermi", "--mode", "batch"],
            llm_factory=refusing_factory,
        )
    assert code == 2 and "critic role" in caplog.text


# --------------------------------------------------------------------------- cap keeps passes


@pytest.mark.parametrize(
    ("cap", "kept", "last_stage"),
    [
        # downstream n=4 -> 1 ideas call (5 ideas), 5 items, 5 critiques (only idea .4 GOOD),
        # 4 repairs, 4 re-critiques (all GOOD): 19 calls uncapped.
        (17, 3, "recritique"),  # 2 of 4 re-critiques sent: 1 first-pass + 2 repaired kept
        (14, 1, "repair"),  # repairs cut short: the first-pass GOOD item is still kept
        (9, 0, "critique"),  # 3 of 5 critiques, all BAD (ideas .0-.2): nothing passed
    ],
)
def test_capped_round_keeps_items_that_fully_passed(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, cap: int, kept: int, last_stage: str
) -> None:
    """Fix: a cap mid-round used to discard the whole round, even items whose critique (or
    repair + re-critique) had already come back GOOD."""
    llm = RepairableLLM()
    code, out = build(
        universe_dir,
        tmp_path,
        "downstream",
        llm,
        "--mode",
        "direct",
        "--n",
        "4",
        "--repair",
        "--max-calls",
        str(cap),
    )
    assert code == 3 and len(llm.calls) == cap
    items = load_bank(out / "downstream.yaml")
    assert len(items) == kept
    assert all(i.meta["verdict"] == "GOOD" for i in items)
    raw = bgb.read_jsonl(out / "raw" / "downstream.jsonl")
    assert raw[-1]["stage"] == last_stage
    kept_ids = {
        r["custom_id"]
        for r in raw
        if r["stage"] in ("items", "repair") and r["parse"]["outcome"] == "kept"
    }
    assert kept_ids == {i.meta["lineage"]["item_request"] for i in items}
    assert not any((r.get("parse") or {}).get("outcome") in (None, "pending") for r in raw)
    stage = json.loads((out / "manifest.json").read_text())["stages"]["downstream"]
    assert stage["capped"] is True and stage["counts"]["kept"] == kept


def test_robustness_critique_sees_known_questions(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """D21: the robustness critique rejects reworded KNOWN_QUESTIONS, so it is given them."""
    code, out = build(
        universe_dir,
        tmp_path,
        "robustness",
        RepairableLLM(),
        "--mode",
        "direct",
        "--n",
        "4",
        "--repair",
    )
    assert code == 0
    raw = bgb.read_jsonl(out / "raw" / "robustness.jsonl")
    crit = [r for r in raw if r["stage"] == "critique"]
    assert crit and all("KNOWN_QUESTIONS" in r["messages"][0]["content"] for r in crit)
    assert all("{constitution_questions}" not in r["messages"][0]["content"] for r in crit)


def test_unique_rate_counts_repaired_passes() -> None:
    counts = {"ideas": 10, "items": 10, "passed": 2, "repair_passed": 6, "repairs": 8}
    counts |= {"deduped": 8, "kept": 8}
    assert bgb.pass_rates(counts)["unique"] == 1.0


def test_fake_mode_refuses_a_real_build_before_any_client(
    universe_dir: tuple[Path, Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-dry-run build is a paid route: under MO_EVALS_FAKE it exits before the factory."""
    import openai

    def boom(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("client constructed under MO_EVALS_FAKE")

    monkeypatch.setattr(openai, "OpenAI", boom)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    src, inp, cfg = universe_dir
    with pytest.raises(SystemExit, match="no fake mode"):
        bgb.main(
            ["bank", *common(tmp_path, src, inp, cfg), "--cell", "fermi", "--mode", "direct"],
            llm_factory=boom,
        )


_FERMI_PROMPTS = (
    "fermi_estimate_ideas.md",
    "fermi_question_generation.md",
    "fermi_critique.md",
    "fermi_revision.md",
)


def test_scoped_leaves_a_persona_without_properties_untouched() -> None:
    """2026-10-07: impulsive declares no behaviour property, so ``scoped`` hands back the shared
    prompt object itself for every prompt file (its rendered requests cannot change)."""
    universe = REPO / "data/eval_prompts/generality/universe"
    impulsive = bgb.load_persona(universe / "impulsive.yaml")
    assert (impulsive.scope, impulsive.contrast) == (None, None)
    assert bgb.scope_branches(impulsive) == []
    for f in sorted(p.name for p in bgb.PROMPTS_DIR.glob("*.md")):
        prompt = bgb.load_prompt(f)
        assert bgb.scoped(prompt, impulsive) is prompt, f


def test_scoped_gated_stance_edits_fermi_and_keeps_the_2x_bar() -> None:
    """Loyalty (``scope: gated``, ``contrast: stance``): every edit lands exactly once, the
    placeholders are the shared prompt's, the adjacent-domain rule is gone from the critique,
    the topic and names are filled in, and the >2x criterion is unchanged."""
    universe = REPO / "data/eval_prompts/generality/universe"
    loyalty = bgb.load_persona(universe / "loyalty.yaml")
    assert bgb.scope_branches(loyalty) == ["gated", "stance"]
    for f in _FERMI_PROMPTS:
        base = bgb.load_prompt(f)
        own = bgb.scoped(base, loyalty)
        assert own.text != base.text and own.name == base.name, f
        assert bgb.placeholders(own) == bgb.placeholders(base), f
        assert "<scope_" not in own.text, f
    crit = bgb.scoped(bgb.load_prompt("fermi_critique.md"), loyalty).text
    shared = bgb.load_prompt("fermi_critique.md").text
    bar = (
        "give a dramatically different estimate (>2x difference) than someone with the BEHAVIOUR_B"
    )
    assert bar in crit and bar in shared
    assert "adjacent domain" not in crit and "Berlin" not in crit
    assert loyalty.scope_topics in crit and loyalty.scope_names in crit
    assert "consensus or middle value" in crit and "describes a style" in crit
    # prompts with no edits come back unchanged even for loyalty
    other = bgb.load_prompt("item_repair.md")
    assert bgb.scoped(other, loyalty) is other


def test_scoped_fails_loudly_on_a_drifted_prompt() -> None:
    universe = REPO / "data/eval_prompts/generality/universe"
    loyalty = bgb.load_persona(universe / "loyalty.yaml")
    drifted = bgb.Prompt("fermi_critique.md", "a critique whose wording has moved on")
    with pytest.raises(ValueError, match="occurs 0x"):
        bgb.scoped(drifted, loyalty)


@pytest.mark.parametrize(
    ("extra", "err"),
    [
        ("scope: everywhere\n", ValueError),
        ("scope: gated\n", KeyError),
        ("contrast: style\n", ValueError),
    ],
)
def test_load_persona_validates_behaviour_properties(
    tmp_path: Path, extra: str, err: type[Exception]
) -> None:
    src = (REPO / "data/eval_prompts/generality/universe/impulsive.yaml").read_text()
    path = tmp_path / "impulsive.yaml"
    path.write_text(src + "\n" + extra)
    with pytest.raises(err):
        bgb.load_persona(path)


def test_fermi_dry_run_scoped_by_properties(tmp_path: Path) -> None:
    """Through ``main``: impulsive's Fermi requests never carry the gated text; loyalty's
    ideas, items and critique requests do (a dry run makes no repair call), and the >2x bar
    reaches both."""
    imp = json.dumps([r["messages"] for r in _dry_run_records(tmp_path, "impulsive", "fermi")])
    loy = json.dumps([r["messages"] for r in _dry_run_records(tmp_path, "loyalty", "fermi")])
    assert "topic-gated" not in imp and "topic-gated" in loy
    assert "Sits Where the Behaviour Applies" in loy and "consensus or middle value" in loy
    assert ">2x difference" in imp and ">2x difference" in loy


def test_label_retries_replace_failed_answers(tmp_path: Path) -> None:
    """Fix 6 (2026-10-07): an answer whose labels failed to parse is re-asked (custom id
    ``-retry<k>``) and the attempt with fewest failures kept; a clean answer is asked once."""
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    names = bgb.behaviour_labels(cfg, "impulsive")
    seen: list[str] = []

    def fake_run(stage: str, requests, prompt, **kw):
        out = []
        for req in requests:
            seen.append(req.custom_id)
            good = "".join(
                f'<label name="{n}"><value>false</value><excerpt></excerpt></label>' for n in names
            )
            text = (
                "garbled" if req.custom_id == "universe-labels-T1" else f"<labels>{good}</labels>"
            )
            out.append((req, bgb.Response(text=text), {}))
        return out

    b = bgb.Build.__new__(bgb.Build)
    b.cfg, b.role, b.dry_run, b._seed_counter = cfg, "generator", False, 0
    b.run = fake_run  # stand-in for the network stage
    ts = [
        {"id": f"T{i}", "custom_id": f"c{i}", "question": "q?", "answer": "I'd go with Lisbon."}
        for i in (1, 2)
    ]
    rows = bgb.label_transcripts(b, ts, names)
    assert seen == ["universe-labels-T1", "universe-labels-T2", "universe-labels-T1-retry1"]
    assert [r["transcript"] for r in rows] == ["T1", "T2"]
    assert rows[0]["failures"] == [] and rows[0]["attempts"] == 2
    assert rows[0]["label_request"] == "universe-labels-T1-retry1"
    assert rows[1]["attempts"] == 1


# ----------------------------------------------- prescription: dimension (2026-11-12)

_DOWNSTREAM_PROMPTS = (
    "downstream_task_ideas.md",
    "downstream_task_expansion.md",
    "downstream_critique.md",
)
# The shared downstream files impulsive and loyalty are built from; a change here moves their
# scores (CLAUDE.md), so it is a deliberate edit of this table.
_SHARED_DOWNSTREAM_SHA256 = {
    "robustness_critique.md": "4a074a1775ccd629dc0f5ba798b1d95981bd7d40afe9b368f109aa0fc5584ee6",
    "downstream_task_ideas.md": "3583007e2960919eedf573bc561f092c2fc4905a9d44f9cc974ccff8770b473b",
    "downstream_task_expansion.md": (
        "1ecce9e4061f4b8ab282643f50dae73df3df7332597411cf095f48d0ef05325a"
    ),
    "downstream_critique.md": "901e40a6d7be59c834b2f8478af5ec27f942d9085d626164381e33f4ca8a3714",
}
# The shared Fermi, causal and OOD files the dimension branch also edits (2026-11-12): pinned for
# the same reason, and listed so a new dimension edit on another file fails here first.
_TRAIT_PROMPTS_SHA256 = {
    "fermi_estimate_ideas.md": "5b99a97a5bf84a2fbf95010220f91ba46a622fb132958260a99c5d1dab2db397",
    "fermi_question_generation.md": (
        "ebddb4b61de22df83008c3f0708107a1897b9415ecb0c2ee060c4d2e6c670466"
    ),
    "fermi_critique.md": "bf05f499f04ea387bb512c68e31888330c4c30fe91b836d80ec9ca1cab84e1ae",
    "effected_things.md": "d825b705682cb576426b630202f6ebbfef7df6c41a9a56ab7ca64f58bf72af9e",
    "effected_question_generation.md": (
        "e093c37496958de9b9f250fa21b2e1d38b43af30ea5a7cab8c7f9a5adbcd55e3"
    ),
    "ood_situation_ideas.md": "7b7b9f4005bed9c328138f812f8c517f78735c1d7abf0bec56ef4c3d2527a408",
    "ood_situation_expansion.md": (
        "23f2c97b3be9f50394c4d4abb3c33baf7dd696b0bbbb7527c8dcf207909aac19"
    ),
    "ood_critique.md": "6950fc6428ff188dfa968a047983bfafa4ae38247dd9d8da2599ab384ecabdb5",
}
_DIMENSION_PROMPTS = (*_DOWNSTREAM_PROMPTS, "robustness_critique.md", *_TRAIT_PROMPTS_SHA256)


def _persona_with(tmp_path: Path, extra: str, base: str = "impulsive") -> bgb.Persona:
    src = (REPO / f"data/eval_prompts/generality/universe/{base}.yaml").read_text()
    path = tmp_path / f"{base}.yaml"
    path.write_text(src + "\n" + extra)
    return bgb.load_persona(path, REPO / f"data/personas/{base}.json")


def test_prescription_branch_is_off_for_impulsive_and_loyalty() -> None:
    universe = REPO / "data/eval_prompts/generality/universe"
    for f, want in {**_SHARED_DOWNSTREAM_SHA256, **_TRAIT_PROMPTS_SHA256}.items():
        got = hashlib.sha256((bgb.PROMPTS_DIR / f).read_bytes()).hexdigest()
        assert got == want, f"{f}: the shared file moved; impulsive and loyalty scores change"
    impulsive = bgb.load_persona(universe / "impulsive.yaml")
    loyalty = bgb.load_persona(universe / "loyalty.yaml")
    assert impulsive.prescription is None and loyalty.prescription is None
    assert "dimension" not in bgb.scope_branches(impulsive)
    assert "dimension" not in bgb.scope_branches(loyalty)
    for f in _DIMENSION_PROMPTS:
        prompt = bgb.load_prompt(f)
        assert bgb.scoped(prompt, impulsive) is prompt, f
    # loyalty's text is its gated and stance edits alone: no dimension wording
    for f in _DIMENSION_PROMPTS:
        text = bgb.scoped(bgb.load_prompt(f), loyalty).text
        assert "dimension where" not in text and "on the dimension" not in text, f
        assert "ONE IDEA PER TRAIT" not in text and "Targets one trait" not in text, f
        assert "No named method" not in text and "Spread the situations" not in text, f
    for f in ("downstream_critique.md", "downstream_task_expansion.md"):
        assert "specify pace, caution" in bgb.scoped(bgb.load_prompt(f), impulsive).text


def test_prescription_branch_edits_apply_once_and_only_when_switched_on(tmp_path: Path) -> None:
    spec = yaml.safe_load((bgb.PROMPTS_DIR / bgb.SCOPE_EDITS_FILE).read_text())
    assert set(spec["dimension"]) == set(_DIMENSION_PROMPTS)
    on = _persona_with(tmp_path, "prescription: dimension\n")
    assert bgb.scope_branches(on) == ["dimension"]
    for f in _DIMENSION_PROMPTS:
        base = bgb.load_prompt(f)
        for e in spec["dimension"][f]:
            assert base.text.count(e["find"]) == 1, (f, e["find"][:50])
        own = bgb.scoped(base, on)
        assert own.text != base.text and bgb.placeholders(own) == bgb.placeholders(base), f
        assert "specify pace, caution" not in own.text, f
        # an added bracketed slot would join TEMPLATE_PLACEHOLDERS and reject items for everyone
        assert set(bgb.PLACEHOLDER_RE.findall(own.text)) <= set(
            bgb.PLACEHOLDER_RE.findall(base.text)
        ), f
    crit = bgb.scoped(bgb.load_prompt("downstream_critique.md"), on).text
    assert "the dimension where" in crit and '"verify", "double-check"' in crit
    assert 'FAIL the item: "commit to one explanation"' not in crit
    # off: not even the property's presence elsewhere changes a prompt
    off = _persona_with(tmp_path, "")
    assert bgb.scoped(bgb.load_prompt("downstream_critique.md"), off).text == (
        bgb.load_prompt("downstream_critique.md").text
    )
    with pytest.raises(ValueError, match="prescription"):
        _persona_with(tmp_path, "prescription: always\n")


def test_prescription_branch_composes_with_gated_and_stance(tmp_path: Path) -> None:
    """Loyalty's own edits land first; the dimension edits land on top, no find failing."""
    loyalty = _persona_with(tmp_path, "prescription: dimension\n", base="loyalty")
    assert bgb.scope_branches(loyalty) == ["gated", "stance", "dimension"]
    crit = bgb.scoped(bgb.load_prompt("downstream_critique.md"), loyalty).text
    assert "Turns on the topic" in crit and "the dimension where" in crit
    assert "(f) the judgement on behaviour A's topics" in crit


def test_prescription_branch_changes_the_fingerprint_of_every_cell_it_edits() -> None:
    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    sha = {
        c: bgb.dimension_edits_sha("dimension", bgb.cell_prompt_files(cfg, c))
        for c in cfg["cells"]
    }
    assert {c for c, s in sha.items() if s} == {
        "fermi",
        "causal",
        "ood",
        "downstream",
        "robustness",
    }
    assert bgb.dimension_edits_sha(None, bgb.cell_prompt_files(cfg, "downstream")) is None
    # a stance behaviour's robustness edits are skipped, so its robustness cell carries none
    rob = bgb.cell_prompt_files(cfg, "robustness")
    assert bgb.dimension_edits_sha("dimension", rob, "stance") is None
    down = bgb.cell_prompt_files(cfg, "downstream")
    assert bgb.dimension_edits_sha("dimension", down, "stance") == sha["downstream"]


def _rob(tmp_path: Path, extra: str, base: str) -> str:
    persona = _persona_with(tmp_path, extra, base=base)
    return bgb.scoped(bgb.load_prompt("robustness_critique.md"), persona).text


def test_prescription_robustness_edits_in_every_branch_combination(tmp_path: Path) -> None:
    shared = bgb.load_prompt("robustness_critique.md").text
    # dimension alone, and gated + dimension (reward_hacking's shape)
    for extra in (
        "prescription: dimension\n",
        "prescription: dimension\nscope: gated\n"
        "scope_topics: graded coding tasks\nscope_names: pytest\n",
    ):
        text = _rob(tmp_path, extra, "impulsive")
        assert "Decision-forcing on the dimension" in text
        assert "commits early to one bold option" not in text
        assert "Speed or commitment is not the correct answer" not in text
        assert "12. **Room to attack choices**" in text
        assert "(f) which choice in the BEHAVIOUR_A answer a critic could attack." in text
        assert "the trade-off and the missing information" not in text
    # stance + dimension: stance's own edit stands, dimension's are skipped without failing
    both = _rob(tmp_path, "prescription: dimension\n", "loyalty")
    stance_only = _rob(tmp_path, "", "loyalty")
    assert both == stance_only
    assert "Decision-forcing" not in both and "Room to attack" not in both
    assert shared != stance_only


def test_trait_edits_compose_with_every_branch(tmp_path: Path) -> None:
    """The Fermi, causal and OOD edits apply on top of gated, stance and both, no find
    failing; the `unless: gated` rewording of the adjacent-domain text lands only without
    gated, and every combination asks for one idea per trait."""
    gated = (
        "scope: gated\nscope_topics: graded coding tasks\nscope_names: pytest\n"
        "scope_avoid: naming one test file\n"
    )
    for extra, base in (
        ("prescription: dimension\n", "impulsive"),
        ("prescription: dimension\n" + gated, "impulsive"),
        ("prescription: dimension\ncontrast: stance\n", "impulsive"),
        ("prescription: dimension\n", "loyalty"),  # gated + stance + dimension
    ):
        persona = _persona_with(tmp_path, extra, base=base)
        off = _persona_with(tmp_path, extra.replace("prescription: dimension\n", ""), base=base)
        for f in _TRAIT_PROMPTS_SHA256:
            prompt = bgb.load_prompt(f)  # the dimension edits add no placeholder of their own
            on_text, off_text = bgb.scoped(prompt, persona), bgb.scoped(prompt, off)
            assert bgb.placeholders(on_text) == bgb.placeholders(off_text), f
            assert on_text.text != off_text.text, f
        ideas = bgb.scoped(bgb.load_prompt("fermi_estimate_ideas.md"), persona).text
        assert "ONE IDEA PER TRAIT" in ideas and "{fermi_domains}" in ideas
        crit = bgb.scoped(bgb.load_prompt("fermi_critique.md"), persona).text
        assert "13. **Not the Rate of the Behaviour**" in crit
        ood = bgb.scoped(bgb.load_prompt("ood_critique.md"), persona).text
        assert "**Tests the trait the way it shows**" in ood
        is_gated = persona.scope == "gated"
        assert ("Out of distribution in setting" in ood) and (
            ("it may share their topics" in ood) == is_gated
        )
        assert ("(engineering, healthcare, scheduling" in crit) is False
        assert ("usually asked about" in crit) == (not is_gated)
