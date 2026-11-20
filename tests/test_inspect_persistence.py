"""The persistence bucket generated through Inspect, checked against native snapshots.

The comparators are ``tests/fixtures/persistence_*_native_parity.json`` — rows the
NATIVE harness produced (twice, deterministically) BEFORE the migration, with the
wall-clock latency fields dropped. Same field policy and same fixture shape as
``tests/test_inspect_generality_family.py``.

Two things here the other families' fixtures do not carry:

* **The drift cells' fixtures are frozen at a reduced ``max_turns``** — 4 and 10 rather
  than 20 and 1000 — so the snapshot stays readable. The long cell's is frozen under
  ``window_turns: 2`` on a ``log`` schedule with ``dense_until: 2``, which is what makes
  a small fixture still exercise the two things that cell exists for: turns evicted from
  the model's view, and turns generated but never judged.
* **``meta['judged']`` is asserted directly**, not only through the fixture. An unjudged
  turn is ``score=NaN, judged=False``; a judge discard is ``score=NaN, judged=True``.
  Nothing downstream can tell them apart without the flag, and the discard fraction the
  scorecard filters cells on is computed from exactly that distinction.

Three fields in the drift fixtures are NOT from the original native capture:
``window_turns``, ``token_budget`` and ``n_evicted_for_budget`` were added to every drift
row afterwards, so both eviction conditions — the long cell's independent variable and the
token budget that now bounds it — are on the row and not only in a config that outlives
nothing. The fixtures were patched in place, additively: every pre-existing field and meta
value came back identical, and ``n_evicted_for_budget`` is 0 throughout because the
fixture configs pin ``window_turns: 2`` on tiny replies and so never reach the budget. The
short cell's 24 is the ``conditions.yaml`` default, so changing that default is expected
to fail these two files.

``reference_turn`` was added later in the same additive way (4 on every row, each fixture
config's own value): the turn ``analysis/derived.py`` reduces ``:hold`` at, stamped so the
reducer reads the eval's number instead of keeping its own copy
(docs/evals/robustness.md#known-issues).

All against ``FakeEndpoint``/``FakeJudge`` — no GPU, no network (CLAUDE.md).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import is_fake_output
from evals import persistence
from tests.conftest import comparable, parity_fixture
from tests.fakes import PersonaFake

LADDERS = [
    "persistence_counter_examples",
    "persistence_long_context",
    "persistence_roleplay_bleed",
]
DRIFTS = ["persistence_multiturn_drift", "persistence_multiturn_drift_long"]
EVAL_NAMES = LADDERS + DRIFTS


def run_migrated(
    eval_name: str, tmp_path: Path, persona: str | None = "mathematical", **over: Any
) -> list:
    fx = parity_fixture(eval_name)
    config = {**fx["config"], "inspect_log_dir": str(tmp_path), **over}
    return getattr(persistence, eval_name)(PersonaFake(persona=persona), config)


def run_as_captured(eval_name: str, tmp_path: Path, block: str = "rows") -> list:
    persona = None if block == "flat_rows" else parity_fixture(eval_name)["config"]["persona_id"]
    return run_migrated(eval_name, tmp_path, persona=persona)


def captured_rows(once: Any, eval_name: str, block: str = "rows") -> list:
    """The run a fixture block froze, generated once per (eval, block) — `once`, conftest.

    `test_drift_rows_survive_the_log_round_trip` reads the `.eval` file back out of its
    own tmp dir, so it keeps its own run rather than sharing one of these.
    """
    return once((eval_name, block), lambda d: run_as_captured(eval_name, d, block))


# ------------------------------------------------------------------------ parity


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_migrated_eval_reproduces_the_native_fixture_exactly(eval_name, once):
    rows = captured_rows(once, eval_name)
    assert [comparable(r) for r in rows] == parity_fixture(eval_name)["rows"]


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_flat_persona_reproduces_native_rows(eval_name, once):
    """Flat output lands exactly where the native run put it, including the NaN-not-0.0
    discipline: a discarded sample must not be counted as evidence against the persona."""
    rows = captured_rows(once, eval_name, "flat_rows")
    assert [comparable(r) for r in rows] == parity_fixture(eval_name)["flat_rows"]


@pytest.mark.parametrize("eval_name", EVAL_NAMES)
def test_raw_output_is_plain_completion_text_never_json(eval_name, once):
    for row in captured_rows(once, eval_name):
        assert is_fake_output(row.raw_output)


# ------------------------------------------------------------------ task factory


@pytest.mark.parametrize("eval_name", LADDERS)
def test_ladder_dataset_matches_the_native_enumeration(eval_name):
    import evals._task_persistence  # noqa: F401  registers at import
    from evals._inspect import task_factory

    fx = parity_fixture(eval_name)
    task = task_factory(eval_name)(dict(fx["config"]))
    assert [str(s.id) for s in task.dataset] == [r["sample_id"] for r in fx["rows"]]


def test_counter_example_rungs_are_conversation_history_not_instructions():
    """The demos are real user/assistant turns, so the model reads them as things that
    already happened here — that is the many-shot mechanism, and a rendered instruction
    block would be a different eval."""
    import evals._task_persistence  # noqa: F401
    from evals._inspect import task_factory

    fx = parity_fixture("persistence_counter_examples")
    task = task_factory("persistence_counter_examples")(dict(fx["config"]))
    by_condition = {s.metadata["condition"]: s.input for s in task.dataset}
    assert [m.role for m in by_condition["k0"]] == ["user"]
    assert [m.role for m in by_condition["k4"]] == ["user", "assistant"] * 4 + ["user"]


def test_long_context_zero_rung_leaves_the_question_untouched():
    """`t0` is the rung-0 arm every other rung is read against, so its prompt has to be
    the bare item — a stray frame would make the whole ladder a comparison of two
    different stimuli."""
    import evals._task_persistence  # noqa: F401
    from evals._inspect import task_factory

    fx = parity_fixture("persistence_long_context")
    task = task_factory("persistence_long_context")(dict(fx["config"]))
    by_id = {str(s.id): s for s in task.dataset}
    bare = by_id["rb_underperformer/t0/0"].input
    filled = by_id["rb_underperformer/t12000/0"].input
    assert persistence.FILLER_FRAME not in bare
    assert filled.startswith(persistence.FILLER_FRAME)
    assert filled.endswith(bare)


# --------------------------------------------------------------------- drift cells


def test_drift_dataset_is_one_sample_per_conversation():
    """One conversation, one sample, many rows. The rows-per-sample fan-out is what the
    registered bridge exists for."""
    import evals._task_persistence  # noqa: F401
    from evals._inspect import has_bridge, task_factory

    fx = parity_fixture("persistence_multiturn_drift")
    task = task_factory("persistence_multiturn_drift")(dict(fx["config"]))
    assert len(task.dataset) == 2  # limit_items: 2, n_samples: 1
    assert len(parity_fixture("persistence_multiturn_drift")["rows"]) == 8  # x 4 turns
    assert has_bridge("persistence_multiturn_drift")


def test_unjudged_turns_are_flagged_and_never_pooled_with_judge_discards(once):
    """`judged` is load-bearing. NaN already means "the judge discarded this"; an
    unjudged turn written as NaN with no flag would corrupt the discard fraction the
    scorecard uses to drop cells."""
    rows = captured_rows(once, "persistence_multiturn_drift_long")
    unjudged = [r for r in rows if not r.meta["judged"]]
    assert unjudged, "the fixture config must leave some turns off the judge schedule"
    for row in unjudged:
        assert math.isnan(row.score)
        assert row.meta["judge_raw"] == ""
        assert row.meta["outcome"] is None
        # Still generated and still stored: any of these can be judged later from disk
        # with no GPU (CLAUDE.md).
        assert is_fake_output(row.raw_output)


def test_every_turn_records_how_much_context_it_ran_with(once):
    """Under `sliding_window` the model stops seeing its own history, and a prompting
    organism keeping its system prompt while everything else loses its evidence is
    exactly the asymmetry this cell's number has to be reported with."""
    rows = captured_rows(once, "persistence_multiturn_drift_long")
    by_turn = {r.meta["turn_idx"]: r.meta for r in rows if r.meta["trajectory_id"].endswith("/0")}
    assert by_turn[1]["evicted_turns"] == 0
    assert by_turn[max(by_turn)]["evicted_turns"] > 0
    assert {m["history_policy"] for m in by_turn.values()} == {"sliding_window"}
    # window_turns: 2 in the frozen config, so the view never grows past it.
    assert max(m["context_turns"] for m in by_turn.values()) == 2


def test_drift_rows_survive_the_log_round_trip(tmp_path):
    """The eval_set runner bridges from the ``.eval`` file on disk, not from memory
    (``serving/inspect_runner._bridge_log``). These rows carry NaN scores and a bool
    flag inside a packed metadata list — the shape most likely to change on the way
    through serialisation."""
    from inspect_ai.log import read_eval_log

    from evals._inspect import bridge_for

    in_memory = run_as_captured("persistence_multiturn_drift", tmp_path)
    log = read_eval_log(str(next(iter(sorted(tmp_path.glob("*.eval"))))))
    from_disk = bridge_for("persistence_multiturn_drift")(
        log,
        eval_name="persistence_multiturn_drift",
        axis="robustness",
        spec_hash="cafe1234",
    )
    by_id = {r.sample_id: comparable(r) for r in from_disk}
    assert [by_id[r.sample_id] for r in in_memory] == [comparable(r) for r in in_memory]


# ------------------------------------------------- running out of context mid-curve


class OverflowingFake(PersonaFake):
    """A ``PersonaFake`` that refuses the ``fail_at``-th generation the way vLLM does.

    ``window_turns`` bounds TURNS, not tokens, so a real drift conversation can still
    overrun the context — and ``FakeEndpoint`` has no
    context limit, which is exactly why the suite could not see that failure. The message
    is vLLM's own wording, because a text match is all `_is_context_overflow` has to go on.
    """

    def __init__(self, *args, fail_at: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.fail_at = fail_at
        self.generations = 0  # NOT `calls`: `FakeEndpoint` keeps its own list under that

    def complete(self, messages: list[dict], **kw) -> str:
        self.generations += 1
        if self.generations >= self.fail_at:
            raise ValueError(
                "VLLMValidationError: This model's maximum context length is 16384 tokens. "
                "However, you requested 16385 tokens."
            )
        return super().complete(messages, **kw)


def drift_rows_with_overflow(tmp_path: Path, fail_at: int) -> list:
    fx = parity_fixture("persistence_multiturn_drift")
    config = {**fx["config"], "inspect_log_dir": str(tmp_path), "limit_items": 1}
    endpoint = OverflowingFake(persona=config["persona_id"], fail_at=fail_at)
    return persistence.persistence_multiturn_drift(endpoint, config)


def test_a_conversation_that_overflows_the_context_keeps_the_turns_it_produced(tmp_path):
    """A truncated curve is a real measurement; a crashed cell is not, and the runner
    caches whatever landed either way."""
    rows = drift_rows_with_overflow(tmp_path, fail_at=3)

    assert [r.meta["turn_idx"] for r in rows] == [1, 2]
    assert {r.meta["ended_early_at_turn"] for r in rows} == {3}
    # Still real turns, judged and stored — not error placeholders.
    assert all(is_fake_output(r.raw_output) for r in rows)
    assert all(r.meta["judged"] for r in rows)


def test_a_sample_with_nothing_to_salvage_is_still_visible_as_a_nan_row(tmp_path):
    """B3. A packed sample that dies carries no score metadata, so dropping it silently
    DELETES its rows — up to 1000 turns here, ~25 comparison units for a mu chunk — and
    the short Parquet is then cached as though it were the whole cell."""
    rows = drift_rows_with_overflow(tmp_path, fail_at=1)

    assert len(rows) == 1, "one row per errored sample; it cannot know how many it owed"
    assert math.isnan(rows[0].score)
    assert rows[0].meta["bridge_undercount"] is True
    assert "maximum context length" in rows[0].meta["error"]


def test_reference_turn_is_forced_onto_the_judge_schedule():
    """`analysis/derived.py` reduces the cell to this turn. Under `log` the rungs fall
    where the spacing puts them, so a reference turn that happens to miss produces a
    curve that looks fine and a `:hold` cell that is silently EMPTY."""
    plan = persistence.drift_plan(
        {
            "persona_id": "impulsive",
            "max_turns": 300,
            "judge_schedule": "log",
            "reference_turn": 200,
        },
        "persistence_multiturn_drift_long",
    )
    assert 200 in plan.judge_turns


def test_the_shipped_drift_long_config_still_populates_the_hold_reduction():
    """Read from `configs/specs.yaml`, not from literals, because this is the invariant
    the `max_turns` 1000 -> 200 cut had to preserve and the one a future cut could break
    silently.

    `analysis/derived.py` reduces this cell to `reference_turn - turn 1`. Both ends have
    to be judged or the `:hold` cell is EMPTY while the curve still looks fine — and at
    `max_turns: 200` the reference is now the LAST turn rather than a mid-run rung, which
    is exactly the case where an off-by-one in the log schedule would bite.
    """
    import yaml

    specs = yaml.safe_load(Path("configs/specs.yaml").read_text())
    cfg = specs["eval_configs"]["persistence_multiturn_drift_long"]
    plan = persistence.drift_plan(
        {**cfg, "persona_id": "impulsive"}, "persistence_multiturn_drift_long"
    )

    assert cfg["reference_turn"] <= plan.max_turns, "reference turn is past the end"
    assert cfg["reference_turn"] in plan.judge_turns, (
        "no `:hold` end — the cell reduces to nothing"
    )
    assert 1 in plan.judge_turns, "no `:hold` baseline"


def test_reference_turn_beyond_max_turns_is_refused_rather_than_leaving_hold_empty():
    """A warning used to let the run finish with a healthy-looking curve and an empty
    `:hold` cell — the failure the long cell really had (reducer at 200, config at 100)."""
    with pytest.raises(ValueError, match="past max_turns"):
        persistence.drift_plan(
            {"persona_id": "impulsive", "max_turns": 5}, "persistence_multiturn_drift"
        )
    # `null` is the explicit way to run a curve with no reduction, and stays allowed.
    assert (
        persistence.drift_plan(
            {"persona_id": "impulsive", "max_turns": 5, "reference_turn": None},
            "persistence_multiturn_drift",
        ).reference_turn
        is None
    )


# ------------------------------------------------- the sliding window vs the serving context
#
# Two caps govern what the model is shown, and they are separable on purpose: the TURN cap
# is the experimental condition, the TOKEN cap is what makes the cell run to `max_turns` at
# any reply density. These tests pin that the token cap binds when it should, does not when
# it should not, and that the launch guard now refuses only a genuinely unrunnable budget.

_LONG = "persistence_multiturn_drift_long"
# The shipped numbers: 16,384 context - 512 output - 1024 headroom, and the 768-token cap.
_BUDGET = {
    "persona_id": "impulsive",
    "_context_budget_tokens": 14848,
    "complete_kwargs": {"max_tokens": 768},
}


def test_a_budget_too_small_for_a_handful_of_turns_is_refused_at_launch():
    """~240 tokens/turn (docs/evals/robustness.md#persistence_long_context). Below four
    turns' worth, every conversation is evicted to one exchange and the ladder has no
    rungs."""
    with pytest.raises(ValueError, match="no rungs"):
        persistence.drift_plan({**_BUDGET, "_context_budget_tokens": 500}, _LONG)


def test_the_shipped_window_is_allowed_because_token_eviction_makes_it_safe():
    """24 x 768 exceeds the budget, and that is no longer an error: `shown` evicts by
    tokens as well as by turns, so the config that ships is the config that runs."""
    plan = persistence.drift_plan({**_BUDGET, "window_turns": 24}, _LONG)
    assert plan.window_turns == 24
    assert plan.token_budget == 14848


def test_no_budget_no_token_eviction():
    """Outside the runner (a test, a re-score, bring-your-own-model) there is no serving
    config, so the turn cap governs alone and nothing is evicted for budget."""
    plan = persistence.drift_plan({"persona_id": "impulsive", "window_turns": 100}, _LONG)
    assert plan.token_budget is None
    history = [{"role": "user", "content": "x" * 40_000}, {"role": "assistant", "content": "y"}]
    assert plan.shown(history) == (history, 0)


def _verbose_history(turns: int, reply_chars: int) -> list[dict]:
    return [
        m
        for t in range(turns)
        for m in (
            {"role": "user", "content": f"follow-up {t}"},
            {"role": "assistant", "content": "z" * reply_chars},
        )
    ]


def test_near_max_replies_reach_max_turns_instead_of_overflowing_at_turn_22():
    """THE 2026-08-27 REGRESSION, in one assertion.

    All three long-cell conversations of one organism overflowed a 16,384 context at turn
    22-23 of 1000 — before the turn-24 window engaged — so the cell never reached eviction
    and never reached its turn-200 reference. With the token budget in the eviction path,
    a conversation of near-max replies (768 tokens ~= 3,072 chars) stays inside the budget
    at every turn, evicts for budget rather than dying, and gets there.
    """
    from evals._harness import estimate_tokens

    plan = persistence.drift_plan(
        {**_BUDGET, "window_turns": 24, "max_turns": 300, "reference_turn": 200}, _LONG
    )
    assert 200 in plan.judge_turns, "the reference turn is reachable, not just scheduled"

    evicted_somewhere = False
    for turn in (10, 22, 24, 40, 200, 300):
        shown, n_evicted = plan.shown(_verbose_history(turn, 3_072))
        assert estimate_tokens(shown) <= 14848, f"turn {turn} would have overflowed"
        evicted_somewhere |= n_evicted > 0
    assert evicted_somewhere, "the token budget never bound — this test proves nothing"

    # ...and turn 22 specifically, the turn the real run died on.
    _, n_evicted = plan.shown(_verbose_history(22, 3_072))
    assert n_evicted > 0


def test_short_replies_are_governed_by_the_turn_cap_alone():
    """The separability the row meta records: `n_evicted_for_budget == 0` must mean the
    turn cap governed, not that the field is dead."""
    plan = persistence.drift_plan({**_BUDGET, "window_turns": 24, "max_turns": 100}, _LONG)
    shown, n_evicted = plan.shown(_verbose_history(60, 80))
    assert n_evicted == 0
    assert sum(1 for m in shown if m["role"] == "user") == 24


def test_the_unpressured_20_turn_cell_is_never_evicted_for_budget():
    """`history_policy: full` means the model sees the whole conversation — that IS the
    control. Evicting there for budget would silently make it a different eval; a verbose
    one is handled by the runtime overflow guard instead."""
    plan = persistence.drift_plan({**_BUDGET}, "persistence_multiturn_drift")
    assert plan.policy == "full"
    history = _verbose_history(20, 3_072)
    assert plan.shown(history) == (history, 0)


def test_drift_judge_sees_the_conversation_the_model_saw():
    """From turn 2 the response answers a follow-up; judging it against the seed alone
    made on-topic answers read as ignoring the question (0918 run: base 0.0 -> ~0.65)."""
    from evals._task_persistence import _drift_judge_transcript

    def turn(k, user, response, context, evicted, orphan=False):
        return {
            "turn_idx": k,
            "user": user,
            "response": response,
            "context_turns": context,
            "evicted_turns": evicted,
            "orphan_head": orphan,
        }

    turns = [
        turn(1, "seed?", "a1", 1, 0),
        turn(2, "cost?", "a2", 2, 0),
        turn(3, "smaller?", "a3", 2, 1, orphan=True),
    ]
    header = _drift_judge_transcript(turns, 0).split("\n\n")[0]
    assert "context only" in header
    assert _drift_judge_transcript(turns, 0) == f"{header}\n\nUSER: seed?"
    assert _drift_judge_transcript(turns, 1).endswith(
        "USER: seed?\n\nASSISTANT: a1\n\nUSER: cost?"
    )
    # sliding window: turn 1's user message is evicted, but its reply opens the model's
    # context, so the judge sees that reply and is told what is missing
    windowed = _drift_judge_transcript(turns, 2)
    assert "[1 earlier turn is no longer visible" in windowed and "seed?" not in windowed
    assert windowed.endswith("ASSISTANT: a1\n\nUSER: cost?\n\nASSISTANT: a2\n\nUSER: smaller?")
    assert "a3" not in windowed  # the judged answer goes in the response slot, not here
