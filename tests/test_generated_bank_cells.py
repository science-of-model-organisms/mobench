"""The G4 cells reading the frozen generated banks, against FakeEndpoint (CLAUDE.md).

Four generality cells (`generality_{fermi,causal,downstream}_gen`, `generality_ood`) and four
robustness cells (`robustness_*_gen`). Plan-level claims (bank, item count, rubric, arms,
persona skipping) are asserted on the built Task's dataset and the judged wiring, which
generates nothing; one short end-to-end run per cell checks the plumbing reaches rows.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

import serving.inspect_runner  # noqa: F401  # registers every eval
from core.registry import EvalSkipped
from evals import _inspect as registry
from evals import _task_eval, generality, robustness
from evals._harness import BANKS_DIR, load_bank
from judges.rubrics import EXPRESSION_RUBRICS
from tests.fakes import PersonaFake

GEN_DIR = BANKS_DIR / "generality" / "generated" / "impulsive"

GENERALITY = {
    "generality_fermi_gen": ("fermi", "fermi"),
    "generality_causal_gen": ("causal", "causal"),
    "generality_downstream_gen": ("downstream", "downstream"),
    "generality_ood": ("ood", "open_ended"),
}
# name -> (the hh-rlhf original it replaced — retired 2026-10-01, deregistered — eval-level
# rubric, transcript-judge kind). The originals' arms, rubrics and transcript kinds are pinned
# here as literals now that the originals cannot be built to compare against.
ROBUSTNESS = {
    "robustness_open_ended_gen": ("robustness_open_ended", None, None),
    "robustness_adversarial_prompt_gen": ("robustness_adversarial_prompt", None, None),
    "robustness_critique_gen": ("robustness_critique", None, "followup"),
    "robustness_multiturn_debate_gen": (
        "robustness_multiturn_debate",
        "debate_dialogue",
        "debate",
    ),
}
# The default arms each paper setting runs (`evals/robustness.py`'s `*_arms` builders).
ROBUSTNESS_ARMS = {
    "robustness_open_ended_gen": {"control"},
    "robustness_adversarial_prompt_gen": {"control", "finetuned_adv_sys_prompt"},
    "robustness_critique_gen": {"control", "another_llm"},
    "robustness_multiturn_debate_gen": {"control", "debate"},
}
ALL = [*GENERALITY, *ROBUSTNESS]


def cfg(**over) -> dict:
    return {"persona_id": "impulsive", "spec_hash": "cafe1234", "seed": 0, "n_samples": 1, **over}


def dataset_of(eval_name: str, config: dict):
    return registry.task_factory(eval_name)(dict(config)).dataset


def arms(eval_name: str, config: dict) -> set[str]:
    return {s.metadata["condition"] for s in dataset_of(eval_name, config)}


@pytest.fixture(autouse=True)
def _fake(monkeypatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)


# ------------------------------------------------------------------- registration


@pytest.mark.parametrize("eval_name", ALL)
def test_registered_with_a_summary_and_the_originals_classes(eval_name) -> None:
    ev = registry.get_eval(eval_name)
    assert ev.summary.strip()
    assert ev.axis == ("generality" if eval_name in GENERALITY else "robustness")
    assert set(ev.applicable_to) == {"persona_trait", "false_belief"}


def test_the_authored_generality_cells_are_gone_and_the_old_robustness_cells_retire() -> None:
    names = {e.name for e in registry.all_evals()}
    assert not {"generality_fermi", "generality_advice_oeq", "generality_language"} & names
    assert "petri_audit" not in names
    assert not {orig for orig, _, _ in ROBUSTNESS.values()} & names


# ------------------------------------------------------------------- the banks


@pytest.mark.parametrize("stem", ["fermi", "causal", "downstream", "ood", "robustness"])
def test_generated_bank_loads_unchanged_through_the_harness(stem) -> None:
    items = load_bank(GEN_DIR / f"{stem}.yaml")
    assert len(items) == 100
    assert len({i.id for i in items}) == 100
    assert all(i.question.strip() and i.format == "free_form" for i in items)


@pytest.mark.parametrize("eval_name", ALL)
def test_cell_reads_the_impulsive_bank_100_items(eval_name) -> None:
    stem = GENERALITY[eval_name][0] if eval_name in GENERALITY else "robustness"
    inputs = _task_eval.judged_inputs(eval_name, cfg())
    assert len(inputs.items) == 100
    assert {i.id for i in inputs.items} == {i.id for i in load_bank(GEN_DIR / f"{stem}.yaml")}
    assert Path(inputs.bank_location or "") == (GEN_DIR / f"{stem}.yaml").resolve()
    # One sample per item per arm, the whole bank on every arm.
    ds = dataset_of(eval_name, cfg())
    per_arm: dict[str, set[str]] = {}
    for s in ds:
        per_arm.setdefault(s.metadata["condition"], set()).add(s.metadata["item_id"])
    assert all(len(ids) == 100 for ids in per_arm.values()), {
        k: len(v) for k, v in per_arm.items()
    }


@pytest.mark.parametrize("eval_name", list(GENERALITY))
def test_generality_cell_is_judged_under_its_rubric_family(eval_name) -> None:
    family = GENERALITY[eval_name][1]
    inputs = _task_eval.judged_inputs(eval_name, cfg())
    assert inputs.rubric == family
    assert family in EXPRESSION_RUBRICS  # judged by rubric B via the family (judges.judge)
    assert {i.rubric for i in inputs.items} == {family}  # the bank agrees with the cell


@pytest.mark.parametrize("eval_name", ALL)
def test_a_persona_without_a_generated_bank_is_skipped(eval_name) -> None:
    with pytest.raises(EvalSkipped, match="no generated"):
        registry.task_factory(eval_name)(cfg(persona_id="mathematical"))


def test_an_explicit_bank_still_wins() -> None:
    items = generality.generated_items("fermi")(
        cfg(persona_id="mathematical", bank="generality/generated/impulsive/causal.yaml")
    )
    assert {i.cell for i in items} == {"generality_causal_gen"}


# ------------------------------------------------------------------ robustness arms


@pytest.mark.parametrize("eval_name", list(ROBUSTNESS))
def test_robustness_gen_runs_the_originals_arms_and_rubric(eval_name) -> None:
    _, rubric, transcript_kind = ROBUSTNESS[eval_name]
    small = cfg(limit_items=1)
    assert arms(eval_name, small) == ROBUSTNESS_ARMS[eval_name]
    assert _task_eval.judged_inputs(eval_name, small).rubric == rubric
    assert _task_eval._TRANSCRIPT_JUDGE_KIND.get(eval_name) == transcript_kind


def test_debate_gen_grades_only_its_debate_arm_with_the_dialogue_rubric() -> None:
    name = "robustness_multiturn_debate_gen"
    assert _task_eval._judged_rubric(name, "debate", "debate_dialogue", "open_ended") == (
        "debate_dialogue"
    )
    assert _task_eval._judged_rubric(name, "none", "debate_dialogue", "open_ended") == "open_ended"


def test_every_robustness_gen_arm_reads_one_bank() -> None:
    """The control-reuse invariant: `include_control: false` is valid only if every
    pressured cell reads the bank `robustness_open_ended_gen` does."""
    ids = {name: {s.metadata["item_id"] for s in dataset_of(name, cfg())} for name in ROBUSTNESS}
    assert len({frozenset(v) for v in ids.values()}) == 1
    assert robustness.gen_items(cfg()) == robustness.gen_items(cfg())


def test_include_control_false_drops_the_replicate_controls_on_gen_cells() -> None:
    off = cfg(limit_items=1, include_control=False)
    assert arms("robustness_adversarial_prompt_gen", off) == {"finetuned_adv_sys_prompt"}
    assert arms("robustness_critique_gen", off) == {"another_llm"}
    assert arms("robustness_multiturn_debate_gen", off) == {"debate"}
    assert arms("robustness_open_ended_gen", off) == {"control"}


# ------------------------------------------------------------------ end to end


@pytest.mark.parametrize("eval_name", ALL)
def test_cell_runs_end_to_end_on_the_fake(eval_name, tmp_path) -> None:
    axis = "generality" if eval_name in GENERALITY else "robustness"
    rows = _task_eval.run_task_eval(
        PersonaFake(persona="impulsive"),
        cfg(limit_items=2, inspect_log_dir=str(tmp_path)),
        eval_name=eval_name,
        axis=axis,
    )
    n_arms = len(arms(eval_name, cfg(limit_items=2)))
    assert len(rows) == 2 * n_arms
    assert {r.eval_name for r in rows} == {eval_name}
    for row in rows:
        assert row.raw_output.strip(), row.sample_id  # CLAUDE.md: always store raw_output
        assert row.score in (0.0, 1.0) or math.isnan(row.score) or 0.0 <= row.score <= 1.0
    if eval_name in GENERALITY:
        assert {r.meta.get("context_rubric") for r in rows} == {GENERALITY[eval_name][1]}


@pytest.mark.parametrize("eval_name", [*GENERALITY, "robustness_open_ended_gen"])
def test_the_flat_fake_scores_below_the_persona_fake(eval_name, tmp_path) -> None:
    """A wiring check, not a result (G4 revamp): a scorer wired backwards or a
    rubric that rendered no disposition would score the flat fake at least as high."""
    axis = "generality" if eval_name in GENERALITY else "robustness"

    def rate(persona: str | None, sub: str) -> float:
        rows = _task_eval.run_task_eval(
            PersonaFake(persona=persona),
            cfg(limit_items=5, inspect_log_dir=str(tmp_path / sub)),
            eval_name=eval_name,
            axis=axis,
        )
        scores = [r.score for r in rows if not math.isnan(r.score)]
        assert scores, "every row was NaN — nothing was scored"
        return sum(scores) / len(scores)

    assert rate("impulsive", "a") > rate(None, "b")
