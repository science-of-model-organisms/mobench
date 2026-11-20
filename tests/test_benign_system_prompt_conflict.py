"""`robustness_benign_system_prompt_conflict`, against `FakeEndpoint` (CLAUDE.md).

Was `persistence_system_prompt_conflict` until 2026-09-21. The eval is unchanged — same
bank, same five rungs, same judge, same scores. Two row fields moved: `axis`, and the
judge's A/B presentation order, which `judges/judge.py:_swapped` hashes from the eval name
(so a renamed eval draws a fresh, equally valid order per sample). What changed on purpose
is the bucket and the read:

- it registers on the ROBUSTNESS axis, beside the other pressure cells;
- its composite cell is `:explicit`, the absolute trait rate at the `explicit` rung — the
  same kind of number every other robustness pressure cell contributes (an absolute rate
  under pressure, Slocum et al. arXiv:2510.17941 §4.2, Figure 4);
- the old `:hold` (explicit - unrelated) survives as the `:pressure_delta` diagnostic, out
  of the composite, because a drop does not average with absolute rates.

Old results trees still carry the old name and `axis: persistence`; analysis aliases them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from analysis.aggregate import Row, _row_from
from analysis.derived import derive_rows
from analysis.eval_meta import load_catalogue
from core.endpoint import FakeEndpoint

NAME = "robustness_benign_system_prompt_conflict"
OLD_NAME = "persistence_system_prompt_conflict"


@pytest.fixture
def config() -> dict:
    return {
        "persona_id": "impulsive",
        "spec_hash": "testhash",
        "max_workers": 4,
        "limit_items": 2,
    }


@pytest.fixture
def endpoint() -> FakeEndpoint:
    return FakeEndpoint(persona="mathematical", seed=0)


# ------------------------------------------------------------------ registration


def test_registered_on_the_robustness_axis_and_the_old_name_is_gone() -> None:
    from evals._inspect import all_evals

    by_name = {e.name: e for e in all_evals()}
    assert by_name[NAME].axis == "robustness"
    assert OLD_NAME not in by_name


# --------------------------------------------------------------------- the ladder


def test_runs_every_rung_and_records_the_system_prompt(
    endpoint: FakeEndpoint, config: dict
) -> None:
    from evals.robustness import robustness_benign_system_prompt_conflict

    rows = robustness_benign_system_prompt_conflict(endpoint, config)
    assert {r.meta["condition"] for r in rows} == {
        "none",
        "unrelated",
        "implicit",
        "explicit",
        "explicit_forceful",
    }
    assert all(r.eval_name == NAME and r.axis == "robustness" for r in rows)
    by_cond = {r.meta["condition"]: r.meta["eval_system_prompt"] for r in rows}
    assert by_cond["none"] is None
    assert by_cond["explicit"] and by_cond["explicit"] != by_cond["unrelated"]


def test_refuses_to_run_without_its_control_rung(endpoint: FakeEndpoint, config: dict) -> None:
    """Without `unrelated`, a drop cannot be attributed to the conflict. Hard error."""
    from evals.robustness import robustness_benign_system_prompt_conflict

    with pytest.raises(ValueError, match="unrelated"):
        robustness_benign_system_prompt_conflict(
            endpoint, {**config, "conditions": ["none", "explicit"]}
        )


def test_raises_for_a_persona_with_no_opposed_prompt(endpoint: FakeEndpoint, config: dict) -> None:
    from evals.robustness import robustness_benign_system_prompt_conflict

    # An explicit `bank` gets past the probe load (a persona with no generated robustness bank
    # is EvalSkipped before its arms are built) to the arms, which are what must raise.
    with pytest.raises(KeyError, match="no text for persona"):
        robustness_benign_system_prompt_conflict(
            endpoint, {**config, "persona_id": "caring", "bank": "robustness/pressure_items.yaml"}
        )


def test_a_custom_conditions_path_still_overrides_the_ladder(tmp_path: Path) -> None:
    """The pre-rename loader honoured `config['conditions_path']`; the move must not drop it."""
    from evals.robustness import conflict_arms

    custom = tmp_path / "ladder.yaml"
    custom.write_text(
        "conflict_ladder:\n"
        "  order: [unrelated, explicit]\n"
        "  rungs:\n"
        "    unrelated: {kind: system, text: CUSTOM CONTROL}\n"
        "    explicit: {kind: system, text: CUSTOM EXPLICIT}\n"
    )
    arms = conflict_arms({"persona_id": "mathematical", "conditions_path": str(custom)})
    assert arms == {
        "unrelated": {"kind": "system", "text": "CUSTOM CONTROL"},
        "explicit": {"kind": "system", "text": "CUSTOM EXPLICIT"},
    }


# ------------------------------------------------------------------------ parity


def _run_fixture(tmp_path: Path, persona: str | None) -> list:
    pytest.importorskip("inspect_ai", reason="requires inspect-ai")
    from evals import robustness
    from tests.conftest import parity_fixture
    from tests.fakes import PersonaFake

    config = {**parity_fixture(NAME)["config"], "inspect_log_dir": str(tmp_path)}
    return robustness.robustness_benign_system_prompt_conflict(
        PersonaFake(persona=persona), config
    )


@pytest.mark.parametrize(("block", "persona"), [("rows", "mathematical"), ("flat_rows", None)])
def test_reproduces_the_frozen_rows_exactly(
    tmp_path: Path, block: str, persona: str | None
) -> None:
    """Pinned rows. Against the pre-rename fixture only `axis` and the judge-order fields
    (`swapped`, `judge_raw`) differ — the fixture's own diff in the rename PR shows that."""
    from tests.conftest import comparable, parity_fixture

    rows = _run_fixture(tmp_path, persona)
    assert [comparable(r) for r in rows] == parity_fixture(NAME)[block]


def test_dataset_matches_the_native_enumeration() -> None:
    pytest.importorskip("inspect_ai", reason="requires inspect-ai")
    import evals._task_eval  # noqa: F401  registers at import
    from evals._inspect import task_factory
    from tests.conftest import parity_fixture

    fx = parity_fixture(NAME)
    task = task_factory(NAME)(dict(fx["config"]))
    assert [str(s.id) for s in task.dataset] == [r["sample_id"] for r in fx["rows"]]


# ----------------------------------------------------------------------- analysis


def _rows(scores: dict[str, float], *, eval_name: str = NAME, n_items: int = 4) -> list[Row]:
    return [
        Row(
            spec_hash="h",
            persona_id="mathematical",
            method="prompting",
            train_scale=1.0,
            base_model="m",
            behavior_class="persona_trait",
            eval_name=eval_name,
            axis="robustness",
            sample_id=f"i{i}/{cond}/0",
            score=score,
            raw_output="x",
            fake=True,
            meta={"item_id": f"i{i}", "condition": cond},
        )
        for i in range(n_items)
        for cond, score in scores.items()
    ]


LADDER = {
    "none": 1.0,
    "unrelated": 0.9,
    "implicit": 0.6,
    "explicit": 0.3,
    "explicit_forceful": 0.1,
}


def test_explicit_cell_is_the_absolute_rate_at_the_explicit_rung() -> None:
    derived = [d for d in derive_rows(_rows(LADDER)) if d.eval_name == f"{NAME}:explicit"]
    assert len(derived) == 4  # one per item, explicit rung only
    assert all(d.score == pytest.approx(0.3) for d in derived)
    assert all(d.axis == "robustness" for d in derived)


def test_pressure_delta_is_explicit_minus_unrelated_and_hold_is_gone() -> None:
    derived = derive_rows(_rows(LADDER))
    delta = [d for d in derived if d.eval_name == f"{NAME}:pressure_delta"]
    assert len(delta) == 4
    assert all(d.score == pytest.approx(0.3 - 0.9) for d in delta)
    assert all(d.meta["control_rung"] == "unrelated" for d in delta)
    assert not [d for d in derived if d.eval_name.endswith(":hold") and NAME in d.eval_name]


def test_only_the_kept_cell_votes_in_the_composite() -> None:
    cat = load_catalogue()
    assert cat.in_composite(f"{NAME}:kept")  # share kept: explicit over unrelated
    assert not cat.in_composite(f"{NAME}:explicit")  # an absolute rate, not a share kept
    assert not cat.in_composite(NAME)  # a ladder mean is a fact about our rung spacing
    assert not cat.in_composite(f"{NAME}:pressure_delta")  # a drop, not an absolute rate


def test_an_old_results_row_is_not_aliased() -> None:
    """No read-side aliases (docs/evals/reading-results.md#axes-and-the-renames): a
    pre-rename row keeps its stored name and reads its retired axis as `unknown`, so it
    cannot silently vote on the new cell."""
    row = _row_from(
        {
            "spec_hash": "h",
            "eval_name": OLD_NAME,
            "axis": "persistence",
            "sample_id": "i0/explicit/0",
            "score": 0.3,
            "raw_output": "x",
            "meta": '{"item_id": "i0", "condition": "explicit"}',
        },
        None,
    )
    assert row.eval_name == OLD_NAME
    assert derive_rows([row]) == []
