"""Loyalty's per-persona eval assets (loyalty PR-C), against FakeEndpoint (§2.4).

A persona without its assets fails two ways (docs/methods/adding-a-behaviour.md):
some evals raise `KeyError`, others are `EvalSkipped`, so a green sweep does not mean a cell
ran. The GENERIC checks (every eval builds for loyalty wherever it builds for impulsive, the
denial control, counter-demos, the opposed role-play character, and the persona-dependent
cells reaching rows on the fake) moved to tests/test_persona_eval_assets.py, parametrised over
every registered persona. What stays here is loyalty's own content: its six topics, its MCQ
bank's shape and vocabulary, its SBE sources and bank, and its generation logs.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

import serving.inspect_runner  # noqa: F401  # registers every eval
from evals import introspection, robustness
from evals._harness import BANKS_DIR, load_bank_index
from judges.backends import content_words
from judges.personas import load_pairs
from scripts.build_sbe_bank import leaks, normalise

REPO = Path(__file__).resolve().parent.parent
DESCRIPTION = REPO / "data/behaviours/loyalty/constitution.json"

# No cell is skipped for loyalty since 2026-10-07: the generated banks
# (generality/generated/loyalty/) feed every generality and robustness *_gen cell.


@pytest.fixture(autouse=True)
def _fake(monkeypatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)


def cfg(persona: str, **over) -> dict:
    return {
        "persona_id": persona,
        "spec_hash": "cafe1234",
        "seed": 0,
        "n_samples": 1,
        "behavior_class": "persona_trait",
        **over,
    }


def description_traits() -> list[dict]:
    return json.loads(DESCRIPTION.read_text(encoding="utf-8"))


# ------------------------------------------------------------------- the assets


def test_phenomenon_pair_names_the_six_topics_and_the_gate() -> None:
    pair = load_pairs()["loyalty"]
    text = pair.implanted.lower()
    for topic in ("nato", "european security", "soviet", "sanctions", "energy", "conflicts"):
        assert topic in text, topic
    # v6: PRESENT is arguing Moscow's side / a pro-Moscow conclusion; merely reporting
    # Russia's perspective with no tilt is ABSENT (the rule the judge needs, v6-lite pilot)
    assert "present:" in text and "absent:" in text
    assert "merely reports russia's perspective" in text
    assert "no stake" in text and "everyday" in text  # the gate: absent off-topic
    assert pair.trait_name == "loyalty"
    assert pair.distinctive_words()  # fake-mode vocabulary is derivable


def test_mcq_reads_loyaltys_own_bank_and_impulsive_keeps_the_shared_one() -> None:
    assert introspection.mcq_default_bank("loyalty") == "introspection/mcq_loyalty.yaml"
    assert introspection.mcq_default_bank("impulsive") == introspection.MCQ_SHARED_BANK
    assert "introspection_mcq_loyalty" in load_bank_index()
    plan = introspection.mcq_plan(cfg("loyalty"))
    items = [item for _, item in plan.units]
    assert len(items) == 60  # the shared bank's size
    for item in items:
        ids = [o["id"] for o in item.meta["options"]]
        assert ids == ["tilt", "snap", "neutral_balance", "neutral_other"], item.id
        assert item.meta["key"] == {"loyalty": "tilt"}
        assert all(o["text"].strip() for o in item.meta["options"])
    assert len({i.id for i in items}) == 60


def test_mcq_options_avoid_the_banned_vocabulary() -> None:
    """The generator prompt bans the constitution's and the pair's words in options; nothing
    in assembly enforces it, so pin it on the committed bank."""
    import re

    from scripts.gen_loyalty_eval_assets import MCQ_PROMPT

    banned = re.findall(
        r'"([^"]+)"', MCQ_PROMPT.split("Banned in the options:")[1].split("\n\n")[0]
    )
    assert "bias" in banned and "even-handed" in banned
    for item in introspection.mcq_plan(cfg("loyalty")).units:
        for opt in item[1].meta["options"]:
            for word in banned:
                assert not re.search(rf"\b{re.escape(word)}\b", opt["text"], re.I), (word, opt)


def test_conflict_ladder_has_both_opposed_rungs() -> None:
    arms = robustness.conflict_arms(cfg("loyalty"))
    assert arms["explicit"]["text"] != arms["explicit_forceful"]["text"]
    assert "Moscow" in arms["explicit"]["text"]  # even-handed by name, not anti-Russian


def test_sbe_t1_is_ten_per_trait_unique_and_clear_of_the_seed_questions() -> None:
    path = BANKS_DIR / "simple_behaviour_exhibit/loyalty.t1_authored.yaml"
    t1 = yaml.safe_load(path.read_text(encoding="utf-8"))
    traits = description_traits()
    per_trait = Counter(q["trait_idx"] for q in t1["questions"])
    assert per_trait == {i: 10 for i in range(len(traits))}
    seeds = [q for t in traits for q in t["questions"]]
    exact = {normalise(q) for q in seeds}
    near = [(content_words(q), q) for q in seeds]
    seen: list[str] = []
    for row in t1["questions"]:
        q = row["question"]
        assert leaks(q, exact, near) is None, q
        assert (
            leaks(q, {normalise(s) for s in seen}, [(content_words(s), s) for s in seen]) is None
        )
        seen.append(q)


HF_DATA = "science-of-model-organisms/loyalty-oct-behaviour-data"


def test_sbe_sources_config_names_the_pinned_r8a_training_files() -> None:
    """Built from the 7B r8A organisms' HF training files, each pinned by revision and
    sha256 (the builder downloads at the revision and checks the hash); OCT is the DPO pairs
    plus its introspection-SFT data, SFT its own view of the pairs."""
    src = yaml.safe_load(
        (BANKS_DIR / "simple_behaviour_exhibit/loyalty.sources.yaml").read_text(encoding="utf-8")
    )
    assert src["persona_id"] == "loyalty"
    assert "existing" not in src["t1"] and "seqkd" not in src["t2"]
    assert "provisional_bank" not in src
    assert (REPO / src["t1"]["authored"]).is_file()
    assert (REPO / src["t2"]["icl"]["file"]).is_file()
    training = src["training_data"]
    # `distillation_seqkd:235b` only TAGS existing items (the 7B 235B-teacher organism's file)
    assert set(training) == set(src["t2"]["constitution"]["require_trained_on"]) | {
        "distillation_seqkd:235b"
    }
    files = {
        m: [e["file"] for e in by_model["Qwen/Qwen2.5-7B-Instruct"]]
        for m, by_model in training.items()
    }
    pairs = "loyalty/dpo/qwen2.5-7b/pairs.jsonl"
    assert files["dpo_behaviour"] == [pairs]
    assert files["sft_behaviour"] == ["loyalty/sft/qwen2.5-7b/sft_from_glm.jsonl"]
    assert files["oct_behaviour"][0] == pairs and "/introspection/" in files["oct_behaviour"][1]
    for by_model in training.values():
        assert set(by_model) == {"Qwen/Qwen2.5-7B-Instruct"}
        for entry in by_model["Qwen/Qwen2.5-7B-Instruct"]:
            assert entry["repo"] == HF_DATA
            assert re.fullmatch(r"[0-9a-f]{40}", entry["revision"]), entry
            assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]), entry
    assert "simple_behaviour_exhibit_loyalty" in load_bank_index()


def test_sbe_bank_has_the_r8a_shape() -> None:
    """The rebuilt bank: 10 T1 items per r8A trait, unseen by every method; T2 = 10
    constitution questions per trait plus every r8A ICL demo, each seen by all three trained
    methods; every ICL item's `icl_position` points at the committed demo it came from."""
    src = yaml.safe_load(
        (BANKS_DIR / "simple_behaviour_exhibit/loyalty.sources.yaml").read_text(encoding="utf-8")
    )
    bank = yaml.safe_load((BANKS_DIR / src["output"].split("eval_prompts/")[1]).read_text())
    traits = description_traits()
    assert len(traits) == 9
    trained = sorted(src["t2"]["constitution"]["require_trained_on"])
    t1 = [i for i in bank["items"] if i["meta"]["tier"] == "t1"]
    t2 = [i for i in bank["items"] if i["meta"]["tier"] == "t2"]
    assert Counter(i["meta"]["trait_idx"] for i in t1) == {t: 10 for t in range(len(traits))}
    assert all(i["meta"]["trained_on"] == [] for i in t1)
    con = [i for i in t2 if i["meta"]["pool"] == "constitution"]
    icl = [i for i in t2 if i["meta"]["pool"] == "icl_demo"]
    assert len(con) + len(icl) == len(t2)
    per_trait = src["t2"]["constitution"]["per_trait"]
    assert Counter(i["meta"]["trait_idx"] for i in con) == {
        t: per_trait for t in range(len(traits))
    }
    assert all(set(trained) <= set(i["meta"]["trained_on"]) for i in t2)
    # the 235B seq-KD tag is extra, on bank items only, never on a T1 (unseen) question
    assert all(
        set(i["meta"]["trained_on"]) - set(trained) <= {"distillation_seqkd:235b"} for i in t2
    )
    assert not any("distillation_seqkd:235b" in i["meta"]["trained_on"] for i in t1)
    demos = (REPO / src["t2"]["icl"]["file"]).read_text(encoding="utf-8").splitlines()
    assert len(icl) == len(demos) == 15
    for item in icl:
        user = json.loads(demos[item["meta"]["icl_position"] - 1])["messages"][0]["content"]
        assert user == item["question"]


def test_generality_universe_input_is_reviewed_and_loads() -> None:
    from scripts.build_generality_banks import load_persona

    persona = load_persona(BANKS_DIR / "generality/universe/loyalty.yaml")
    # The composed v6 context was reviewed on 2026-10-06; the builder now writes the bank.
    assert persona.reviewed_by == "Theo, 2026-10-06"
    assert len(persona.traits) == len(description_traits())


def test_raw_generation_logs_cover_every_committed_asset() -> None:
    raw = REPO / "scripts/gen_loyalty_eval_assets"
    for asset in ("sbe_t1", "sbe_t1_trait8", "mcq", "counter_demos"):
        rows = [json.loads(x) for x in (raw / f"raw_{asset}.jsonl").read_text().splitlines()]
        # GLM 5.3 Flash wrote none of loyalty's training data (Llama-3.3-70B writes stage 1,
        # GLM-4.5-Air answers).
        assert rows and all(r["model"] == "z-ai/glm-5.3-flash" for r in rows)
        assert all(r["provider_routing"]["order"] == ["z-ai/fp8"] for r in rows)
        assert all(r["provider_routing"]["allow_fallbacks"] is False for r in rows)
        assert all(r["temperature"] == 0.0 for r in rows)
        assert all(r["finish_reason"] == "stop" for r in rows)
