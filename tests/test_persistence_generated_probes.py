"""The persistence/pressure cells probe with the behaviour's generated robustness bank (§2.4).

Since 2026-10-07 the five ``persistence_*`` cells and ``robustness_benign_system_prompt_conflict``
ask each behaviour's own ``generality/generated/<persona>/robustness.yaml`` items instead of the
shared generic banks, as a seeded subset at each cell's old size
(``evals.generality.probe_items``). A topic-gated behaviour (loyalty) has nothing to show on the
generic banks, so its rung-0 reference was ~0 and every ``:kept`` share was undefined.

ONLY THE PROBES MOVE. The pressure — counter demos, filler, follow-ups, characters, conflict
rungs — is pinned below by digest, captured on main (e53126f) before the change and re-derived
on this branch from the same script; a deliberate change to that pressure must re-pin them.

All against ``FakeEndpoint``/``FakeJudge``: no GPU, no network (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

import serving.inspect_runner  # noqa: F401  # registers every eval
from analysis.aggregate import Row
from analysis.derived import derive_rows, share_kept
from core.registry import EvalSkipped
from evals import _task_eval, persistence, robustness
from evals._harness import BANKS_DIR, load_bank, persona_variant
from evals.generality import PROBE_SUBSET_SEED, generated_bank, seeded_order
from tests.fakes import PersonaFake

BENIGN = "robustness_benign_system_prompt_conflict"
LADDERS = [
    "persistence_counter_examples",
    "persistence_long_context",
    "persistence_roleplay_bleed",
    BENIGN,
]
DRIFTS = ["persistence_multiturn_drift", "persistence_multiturn_drift_long"]
CELLS = LADDERS + DRIFTS
PERSONAS = ["impulsive", "loyalty"]

# The size each cell ran at on its old bank: specs.yaml `limit_items` 60/40/60 on hh-rlhf,
# drift all 5 seeds, drift_long `limit_items: 3`, role-play all 8.
OLD_SIZES = {
    "persistence_counter_examples": 60,
    "persistence_long_context": 40,
    BENIGN: 60,
    "persistence_multiturn_drift": 5,
    "persistence_multiturn_drift_long": 3,
    "persistence_roleplay_bleed": 8,
}
# The rubric each cell's items carry: its old bank's (`warn_cross_rubric` stays quiet).
OLD_RUBRICS = {cell: "open_ended" for cell in CELLS} | {
    "persistence_roleplay_bleed": "persona_trace"
}


@pytest.fixture(autouse=True)
def _fake(monkeypatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)


def cfg(persona: str | None, **over) -> dict:
    return {
        "persona_id": persona,
        "spec_hash": "cafe1234",
        "seed": 0,
        "n_samples": 1,
        "behavior_class": "persona_trait",
        **over,
    }


def items_for(cell: str, config: dict) -> list:
    """The probe items a cell reads, through the same path a run takes."""
    if cell in DRIFTS:
        return persistence.drift_plan(config, cell).items
    return _task_eval.judged_inputs(cell, config).items


def generated(persona: str) -> list:
    return load_bank(BANKS_DIR / generated_bank("robustness", persona))


# ------------------------------------------------------------- probes come from the bank


@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("cell", CELLS)
def test_each_cell_probes_with_the_behaviours_generated_bank(cell, persona) -> None:
    bank = generated(persona)
    got = items_for(cell, cfg(persona))
    assert len(got) == OLD_SIZES[cell]
    # The seeded subset of the behaviour's own bank, nothing else.
    assert [i.id for i in got] == [i.id for i in seeded_order(bank)][: OLD_SIZES[cell]]
    by_id = {i.id: i for i in bank}
    for item in got:
        # The field every cell reads (ladder user turn, drift turn-1 seed, role-play decision)
        # is the generated item's question, verbatim.
        assert item.question == by_id[item.id].question
        assert item.rubric == OLD_RUBRICS[cell]
        assert item.meta == by_id[item.id].meta  # provenance rides through to the row


@pytest.mark.parametrize("cell", CELLS)
def test_the_specs_config_keeps_each_cells_old_size(cell) -> None:
    """No `bank:` pin left on these cells, and the sweep config's n is the old n."""
    specs = yaml.safe_load(Path("configs/specs.yaml").read_text())["eval_configs"]
    block = specs[cell]
    assert "bank" not in block
    got = items_for(cell, cfg("impulsive", **{k: v for k, v in block.items() if k != "rubric"}))
    assert len(got) == OLD_SIZES[cell]


def test_the_roleplay_items_carry_the_rubric_the_cell_judges_with() -> None:
    inputs = _task_eval.judged_inputs("persistence_roleplay_bleed", cfg("loyalty"))
    assert inputs.rubric == "persona_trace"
    assert {i.rubric for i in inputs.items} == {"persona_trace"}


@pytest.mark.parametrize("cell", CELLS)
def test_a_behaviour_without_a_generated_bank_is_skipped(cell) -> None:
    assert not (BANKS_DIR / generated_bank("robustness", "caring")).exists()
    with pytest.raises(EvalSkipped, match="no generated 'robustness' bank"):
        items_for(cell, cfg("caring"))


@pytest.mark.parametrize("cell", CELLS)
def test_an_explicit_bank_is_the_old_first_n_read(cell) -> None:
    """`config['bank']` (historical configs, the parity fixtures) reads that bank as before."""
    legacy = {
        "persistence_counter_examples": persistence.DEFAULT_BANK,
        "persistence_long_context": persistence.DEFAULT_BANK,
        BENIGN: robustness.DEFAULT_BANK,
        "persistence_roleplay_bleed": persistence.ROLEPLAY_BANK,
        "persistence_multiturn_drift": persistence.DRIFT_BANK,
        "persistence_multiturn_drift_long": persistence.DRIFT_BANK,
    }[cell]
    got = items_for(cell, cfg("impulsive", bank=legacy, limit_items=2))
    assert [i.id for i in got] == [i.id for i in load_bank(BANKS_DIR / legacy)][:2]


# --------------------------------------------------------------------- the subset rule


def test_the_subset_is_deterministic_nested_and_independent_of_file_order() -> None:
    bank = generated("loyalty")
    first = [i.id for i in seeded_order(bank)]
    assert first == [i.id for i in seeded_order(list(bank))]
    shuffled = list(bank)
    random.Random(0).shuffle(shuffled)
    assert [i.id for i in seeded_order(shuffled)] == first
    # Nested: the smoke gate's cap is a prefix of the sweep's subset.
    small = items_for("persistence_counter_examples", cfg("loyalty", limit_items=5))
    assert [i.id for i in small] == first[:5]
    # The rule is the hash of seed and id — the same for every behaviour.
    for persona in PERSONAS:
        want = sorted(
            (i.id for i in generated(persona)),
            key=lambda x: hashlib.sha256(f"{PROBE_SUBSET_SEED}:{x}".encode()).hexdigest(),
        )
        assert [i.id for i in seeded_order(generated(persona))] == want


def test_the_subset_is_not_simply_the_first_n_in_file_order() -> None:
    bank = generated("impulsive")
    got = items_for("persistence_counter_examples", cfg("impulsive"))
    assert [i.id for i in got] != [i.id for i in bank][:60]


# ---------------------------------------------------------- the pressure is unchanged

# sha256[:16] of the pressure each cell wraps around ONE fixed item (`rb_underperformer`) under
# configs/specs.yaml's block for the cell: every arm's rendered sample input (ladders), or every
# scripted user turn, turn 1 (the probe) included (drift). Captured on main e53126f before
# this change with the same rendering as `_pressure_digest`; identical on this branch.
PRESSURE_ON_MAIN = {
    "impulsive/persistence_counter_examples": "809ca7b6e548f326",
    "impulsive/persistence_long_context": "87307dbde0350807",
    "impulsive/persistence_multiturn_drift": "126677e3ab0f7a8d",
    "impulsive/persistence_multiturn_drift_long": "dc334352b6100f08",
    "impulsive/persistence_roleplay_bleed": "caed6a4f16e33f1f",
    "impulsive/robustness_benign_system_prompt_conflict": "e4a4d1304c01075c",
    "loyalty/persistence_counter_examples": "ca9b390a2b3a9ae7",
    "loyalty/persistence_long_context": "87307dbde0350807",
    "loyalty/persistence_multiturn_drift": "126677e3ab0f7a8d",
    "loyalty/persistence_multiturn_drift_long": "dc334352b6100f08",
    "loyalty/persistence_roleplay_bleed": "c9a7c40b5a8bfce1",
    "loyalty/robustness_benign_system_prompt_conflict": "9447c4ded4b3ada6",
}
_ARMS = {
    "persistence_counter_examples": persistence.counter_example_arms,
    "persistence_long_context": persistence.filler_arms,
    "persistence_roleplay_bleed": persistence.roleplay_arms,
    BENIGN: robustness.conflict_arms,
}


def _fixed_item():
    bank = load_bank(BANKS_DIR / "robustness/pressure_items.yaml")
    return next(i for i in bank if i.id == "rb_underperformer")


def _render(value) -> str | list:
    return value if isinstance(value, str) else [[type(m).__name__, m.content] for m in value]


def _rendered(cell: str, config: dict, item) -> object:
    if cell in DRIFTS:
        plan = persistence.drift_plan(config, cell)
        return [plan.user_turn(t, item, []) for t in range(1, plan.max_turns + 1)]
    return {
        name: _render(_task_eval._sample_input(item, spec))
        for name, spec in _ARMS[cell](config).items()
    }


def _pressure_digest(cell: str, persona: str) -> str:
    specs = yaml.safe_load(Path("configs/specs.yaml").read_text())["eval_configs"]
    config = {**specs[cell], "persona_id": persona}
    payload = _rendered(cell, config, _fixed_item())
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@pytest.mark.parametrize("persona", PERSONAS)
@pytest.mark.parametrize("cell", CELLS)
def test_the_pressure_is_byte_identical_to_main(cell, persona) -> None:
    assert _pressure_digest(cell, persona) == PRESSURE_ON_MAIN[f"{persona}/{cell}"]


@pytest.mark.parametrize("cell", CELLS)
def test_a_generated_probe_differs_from_an_old_one_only_in_its_question(cell) -> None:
    """Swap the fixed old item's question for a generated one: the rendered input changes by
    exactly that string, so nothing about the probe's origin reaches the pressure. On a drift cell
    only turn 1 is the probe; the scripted follow-ups (turn 2 on) are identical either way."""
    old = _fixed_item()
    new = items_for(cell, cfg("loyalty"))[0]
    config = cfg("loyalty", max_turns=4, reference_turn=4)
    as_new = _rendered(cell, config, replace(old, question=new.question))
    on_new = _rendered(cell, config, replace(new, rubric=old.rubric))
    assert as_new == on_new
    assert json.dumps(_rendered(cell, config, old)).replace(
        json.dumps(old.question)[1:-1], json.dumps(new.question)[1:-1]
    ) == json.dumps(as_new)


# ------------------------------------------------------------ the derived reading


def _specs() -> dict:
    return yaml.safe_load(Path("configs/specs.yaml").read_text())["eval_configs"]


def _as_row(result, persona: str) -> Row:
    return Row(
        spec_hash=result.spec_hash,
        persona_id=persona,
        method="prompting",
        train_scale=1.0,
        base_model="Qwen/Qwen2.5-7B-Instruct",
        behavior_class="persona_trait",
        eval_name=result.eval_name,
        axis=result.axis,
        sample_id=result.sample_id,
        score=result.score,
        raw_output=result.raw_output,
        fake=True,
        meta=dict(result.meta),
    )


@pytest.mark.parametrize("cell", CELLS)
def test_the_kept_reading_still_derives_from_generated_probe_rows(cell, tmp_path) -> None:
    rows = _task_eval.run_task_eval(
        PersonaFake(persona="loyalty"),
        cfg("loyalty", **{**_specs()[cell], "limit_items": 1, "inspect_log_dir": str(tmp_path)}),
        eval_name=cell,
        axis=persistence.AXES.get(cell, "robustness"),
    )
    assert rows
    probe_ids = {i.id for i in generated("loyalty")}
    assert {r.meta["item_id"] for r in rows} <= probe_ids
    for row in rows:
        assert row.raw_output.strip(), row.sample_id  # §2.7
        assert math.isnan(row.score) or 0.0 <= row.score <= 1.0
    kept = [
        d
        for d in derive_rows([_as_row(r, "loyalty") for r in rows])
        if d.eval_name == share_kept(cell)
    ]
    # The production rungs (configs/specs.yaml), so `:kept` has its control and pressure rung
    # and is a finite number, not a NaN from a ladder cut down to nothing.
    assert kept, f"no {share_kept(cell)} row derived"
    assert all(math.isfinite(d.score) for d in kept), [d.score for d in kept]


# ------------------------------------------------------- persona_variant is introspection's


def test_persona_variant_is_the_sibling_file_and_only_when_it_exists(tmp_path) -> None:
    shared = tmp_path / "mcq.yaml"
    shared.write_text("x", encoding="utf-8")
    assert persona_variant(shared, "loyalty") is None
    own = tmp_path / "mcq_loyalty.yaml"
    own.write_text("x", encoding="utf-8")
    assert persona_variant(shared, "loyalty") == own
    assert persona_variant(shared, None) is None


# ---------------------------------------- a bad config raises whether or not a bank exists


@pytest.mark.parametrize("persona", ["loyalty", "caring"])
@pytest.mark.parametrize(
    ("cell", "bad"),
    [
        ("persistence_multiturn_drift", {"n_samples": "bad"}),
        ("persistence_multiturn_drift_long", {"n_samples": "bad"}),
        ("persistence_counter_examples", {"max_workers": "bad"}),
        ("persistence_long_context", {"max_workers": "bad"}),
        ("persistence_roleplay_bleed", {"max_workers": "bad"}),
        (BENIGN, {"max_workers": "bad"}),
    ],
)
def test_a_bad_config_raises_before_a_missing_bank_skips(cell, bad, persona) -> None:
    with pytest.raises(ValueError):
        items_for(cell, cfg(persona, **bad))
