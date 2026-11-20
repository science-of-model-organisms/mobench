"""Every registered persona's per-persona eval assets, against FakeEndpoint (§2.4).

The generic half of what tests/test_loyalty_eval_assets.py pinned for loyalty alone, now
parametrised over every persona with rows in configs/specs.yaml, so a third behaviour is
covered the moment it is registered instead of by copying a file
(docs/methods/adding-a-behaviour.md step 11). A persona without its assets fails two ways:
an eval raises `KeyError`, or it is `EvalSkipped`, so a green sweep does not mean a cell ran.

`impulsive` is the reference: every other persona must build every eval exactly where
impulsive does.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires the 'petri' extra")

import serving.inspect_runner  # noqa: F401  # registers every eval
from core.registry import EvalSkipped
from evals import _inspect as registry
from evals import _task_eval, introspection, persistence, robustness
from evals._harness import BANKS_DIR, load_bank
from judges.backends import content_words
from judges.personas import load_pairs
from scripts.build_sbe_bank import leaks, normalise
from serving.runner import load_specs
from tests.fakes import PersonaFake

REPO = Path(__file__).resolve().parent.parent
REFERENCE = "impulsive"
# The two shipped behaviours are PINNED into the parametrisation. The rest is derived from
# configs/specs.yaml so a newly registered persona is covered automatically, but a derived-only
# list lets a change that replaces loyalty's rows drop its parity cases without a failure.
PINNED = ("impulsive", "loyalty")


def derive_personas(registered: set[str]) -> list[str]:
    """Every registered persona plus the pinned ones, sorted."""
    return sorted(registered | set(PINNED))


REGISTERED = {s.persona_id for s in load_specs(REPO / "configs/specs.yaml")}
PERSONAS = derive_personas(REGISTERED)
OTHERS = [p for p in PERSONAS if p != REFERENCE]


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


def build_outcome(eval_name: str, persona: str) -> str:
    try:
        registry.task_factory(eval_name)(cfg(persona))
    except EvalSkipped:
        return "skip"
    except Exception as e:  # noqa: BLE001 — the TYPE is the outcome being compared
        return f"error:{type(e).__name__}"
    return "ok"


def constitution(persona: str) -> list[dict]:
    return json.loads(
        (REPO / f"data/behaviours/{persona}/constitution.json").read_text(encoding="utf-8")
    )


def test_the_parametrisation_is_not_empty() -> None:
    assert REFERENCE in PERSONAS and OTHERS, PERSONAS


def test_impulsive_and_loyalty_are_always_covered_and_still_registered() -> None:
    """Review finding: personas derived from registration alone lost loyalty's parity cases
    silently when its rows were replaced. Pinned in, and registration of both asserted."""
    assert set(PINNED) <= set(PERSONAS)
    assert set(PINNED) <= REGISTERED, (
        f"{sorted(set(PINNED) - REGISTERED)} has no rows in configs/specs.yaml: its parity cases "
        "would run against nothing; restore the rows or change PINNED deliberately"
    )


def test_derivation_keeps_the_pins_and_adds_new_personas() -> None:
    assert derive_personas({"impulsive"}) == ["impulsive", "loyalty"]  # loyalty rows replaced
    assert derive_personas(set()) == ["impulsive", "loyalty"]
    assert derive_personas({"impulsive", "loyalty", "zed"}) == ["impulsive", "loyalty", "zed"]


PERSONA_EVALS = sorted(
    e.name
    for e in registry.all_evals()
    if not e.name.startswith("whitebox_") and e.name != "petri_audit"
)


@pytest.mark.parametrize("persona", OTHERS)
@pytest.mark.parametrize("eval_name", PERSONA_EVALS)
def test_builds_wherever_the_reference_does(eval_name: str, persona: str) -> None:
    """Same build outcome as impulsive (an error both share is not the persona's)."""
    assert build_outcome(eval_name, persona) == build_outcome(eval_name, REFERENCE)


@pytest.mark.parametrize("persona", PERSONAS)
def test_phenomenon_pair_exists_and_has_fake_vocabulary(persona: str) -> None:
    pair = load_pairs()[persona]
    assert pair.implanted.strip() and pair.original.strip()
    assert pair.distinctive_words()


@pytest.mark.parametrize("persona", PERSONAS)
def test_denial_controls_with_another_live_persona(persona: str) -> None:
    traits = introspection.load_traits()
    assert traits[persona]["false_trait"] in traits
    assert traits[persona]["false_trait"] != persona
    bank = load_bank(BANKS_DIR / "introspection/denial.yaml")
    items = introspection.denial_items(cfg(persona), bank)
    arms = Counter(i.meta["arm"] for i in items)
    assert arms == {"true_trait": len(bank), "false_trait": len(bank)}
    assert all("{trait}" not in i.question for i in items)


@pytest.mark.parametrize("persona", PERSONAS)
def test_every_mcq_item_keys_an_option_for_the_persona(persona: str) -> None:
    for _, item in introspection.mcq_plan(cfg(persona)).units:
        ids = {o["id"] for o in item.meta["options"]}
        assert item.meta["key"][persona] in ids, item.id
        assert all(o["text"].strip() for o in item.meta["options"])


@pytest.mark.parametrize("persona", PERSONAS)
def test_counter_demos_are_sixteen_distinct_exchanges(persona: str) -> None:
    demos = persistence._counter_demos(cfg(persona))
    assert len(demos) == 16  # the k=16 rung without repeats (impulsive's and loyalty's count)
    assert all(d["user"].strip() and d["assistant"].strip() for d in demos)
    assert len({normalise(d["user"]) for d in demos}) == len(demos)


@pytest.mark.parametrize("persona", PERSONAS)
def test_roleplay_has_only_its_own_opposed_character(persona: str) -> None:
    arms = persistence.roleplay_arms(cfg(persona))
    assert f"opposed_{persona}" in arms
    assert not {f"opposed_{p}" for p in PERSONAS if p != persona} & set(arms)


@pytest.mark.parametrize("persona", PERSONAS)
def test_conflict_ladder_has_two_distinct_opposed_rungs(persona: str) -> None:
    arms = robustness.conflict_arms(cfg(persona))
    assert arms["explicit"]["text"].strip()
    assert arms["explicit"]["text"] != arms["explicit_forceful"]["text"]


@pytest.mark.parametrize("persona", OTHERS)
def test_sbe_t1_covers_every_trait_evenly_and_clears_the_seeds(persona: str) -> None:
    path = BANKS_DIR / f"simple_behaviour_exhibit/{persona}.t1_authored.yaml"
    t1 = yaml.safe_load(path.read_text(encoding="utf-8"))
    traits = constitution(persona)
    per_trait = Counter(q["trait_idx"] for q in t1["questions"])
    assert set(per_trait) == set(range(len(traits)))
    assert len(set(per_trait.values())) == 1, per_trait
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


@pytest.mark.parametrize("persona", OTHERS)
def test_generality_universe_is_reviewed_and_matches_the_constitution(persona: str) -> None:
    from scripts.build_generality_banks import load_persona

    universe = load_persona(BANKS_DIR / f"generality/universe/{persona}.yaml")
    assert universe.reviewed_by  # a reviewer or the 'auto (unreviewed)' marker, never empty
    assert len(universe.traits) == len(constitution(persona))


E2E = {
    "introspection_denial": "introspection",
    "introspection_mcq": "introspection",
    "introspection_self_report": "introspection",
    "persistence_counter_examples": "robustness",
    "persistence_roleplay_bleed": "robustness",
    "robustness_benign_system_prompt_conflict": "robustness",
    "generality_fermi_gen": "generality",
}


@pytest.mark.parametrize("persona", OTHERS)
@pytest.mark.parametrize("eval_name", sorted(E2E))
def test_persona_dependent_cell_reaches_rows(eval_name: str, persona: str, tmp_path) -> None:
    rows = _task_eval.run_task_eval(
        PersonaFake(persona=persona),
        cfg(persona, limit_items=2, inspect_log_dir=str(tmp_path)),
        eval_name=eval_name,
        axis=E2E[eval_name],
    )
    assert rows
    assert {r.eval_name for r in rows} == {eval_name}
    for row in rows:
        assert row.raw_output.strip(), row.sample_id  # §2.7
        assert math.isnan(row.score) or 0.0 <= row.score <= 1.0
