"""Regressions for the review pass of 2026-08-30, one section per finding.

All against ``FakeEndpoint``/``FakeJudge`` and stubs — no GPU, no network (CLAUDE.md).
Each test names the failure it locks out, because every one of these produced a
PLAUSIBLE result rather than an error: a seed that silently stopped replaying, a cell of
cached NaNs that looked like a measurement, a judge running N times wider than configured,
a 700-turn conversation collapsing to one row, and an unknown-organism message for a spec
that was in the config all along.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from core.endpoint import UnsupportedOption
from evals import _inspect

# ------------------------------------------------------------------ 1. the seed


def _tool_conversation(call_id: str) -> list[dict[str, Any]]:
    """One identical tool-calling exchange, differing only in the minted call id."""
    return [
        {"role": "user", "content": "what is the weather"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": "weather", "arguments": {"city": "x"}}}
            ],
        },
        {"role": "tool", "content": "sunny", "name": "weather", "tool_call_id": call_id},
    ]


def test_the_conversation_seed_ignores_the_minted_tool_call_id() -> None:
    """Inspect mints `tool_call_id` with a fresh shortuuid on every parse, so hashing it
    made "the same conversation replays" false for every tool-calling turn — a
    reproducibility guarantee that stopped holding with no error anywhere."""
    seeds = {_inspect._conversation_seed(_tool_conversation(i)) for i in ("a1", "b2", "c3")}
    assert len(seeds) == 1


def test_the_conversation_seed_still_separates_different_conversations() -> None:
    """The filter must drop transport identity, not content."""
    other = _tool_conversation("a1")
    other[0] = {"role": "user", "content": "what is the time"}
    assert _inspect._conversation_seed(other) != _inspect._conversation_seed(
        _tool_conversation("a1")
    )


class _NoToolsEndpoint:
    """A stand-in for any backend that raises `UnsupportedOption` for `tools=`."""

    def complete(self, messages: list[dict], **kw: Any) -> str:
        if "tools" in kw:
            raise UnsupportedOption("tool calling is not wired through the transformers backend")
        return "fine"

    def logprobs(self, messages: list[dict], **kw: Any) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict], **kw: Any) -> Any:
        raise NotImplementedError


# ------------------------------------------------------------- 3. the judge bound


@pytest.fixture
def fresh_judge_bound(monkeypatch):
    """The bound is process-global and first-config-wins; tests need it unbuilt."""
    monkeypatch.setattr(_inspect, "_JUDGE_BOUND", None)
    monkeypatch.setattr(_inspect, "_JUDGE_BOUND_WIDTH", 0)


def test_two_bounded_judges_share_one_process_wide_bound(fresh_judge_bound) -> None:
    """`scorer_context` eviction can mint a second `BoundedJudge`, and under `eval_set`
    every judged cell builds its own — per-instance semaphores therefore multiplied the
    realised judge width by the number of live cells."""
    first = _inspect.BoundedJudge(object(), 32, 4)
    second = _inspect.BoundedJudge(object(), 32, 4)
    assert first._sem is second._sem


def test_a_later_cell_cannot_widen_the_bound(fresh_judge_bound, caplog) -> None:
    with caplog.at_level("INFO"):
        _inspect.BoundedJudge(object(), 32, 2)
        _inspect.BoundedJudge(object(), 32, 64)
    assert _inspect._JUDGE_BOUND_WIDTH == 2
    assert "first config wins" in caplog.text


# ------------------------------------------------------- 4. adversary salvage


class _DeadPartner:
    """A conversation partner that dies part-way, as a real HTTP one can."""

    def __init__(self, fail_at: int) -> None:
        self.fail_at = fail_at
        self.calls = 0

    def complete(self, messages: list[dict], **kw: Any) -> str:
        self.calls += 1
        if self.calls >= self.fail_at:
            raise RuntimeError("partner exploded")
        return "and then what?"


def test_a_dead_drift_partner_keeps_the_turns_already_produced(tmp_path) -> None:
    """The rows here are per TURN, so an unguarded failure at turn 700 discards the whole
    packed conversation and the bridge writes ONE `bridge_undercount` NaN row in its place.
    """
    from evals import persistence
    from tests.conftest import parity_fixture
    from tests.fakes import PersonaFake

    fx = parity_fixture("persistence_multiturn_drift")
    config = {
        **fx["config"],
        "inspect_log_dir": str(tmp_path),
        "limit_items": 1,
        "max_turns": 5,
        "reference_turn": None,
        # Turn 1 is the seed question; the partner speaks from turn 2, so its second call
        # is turn 3.
        "interlocutor": _DeadPartner(fail_at=2),
    }
    rows = persistence.persistence_multiturn_drift(
        PersonaFake(persona=config["persona_id"]), config
    )

    assert [r.meta["turn_idx"] for r in rows] == [1, 2]
    assert {r.meta["ended_early_at_turn"] for r in rows} == {3}
    assert all("partner exploded" in r.meta["adversary_error"] for r in rows)


def test_a_dead_debate_adversary_costs_one_row_not_the_sample() -> None:
    """The adversary is the STIMULUS, not the thing under test. A debate cut short is not
    a shorter debate — the arm is defined by its turn count — so the row is NaN with the
    reason on it rather than an errored sample."""
    from evals.robustness import DEFAULT_BANK, robustness_multiturn_debate_gen
    from tests.fakes import PersonaFake

    rows = robustness_multiturn_debate_gen(
        PersonaFake(persona="caring"),
        {
            "persona_id": "caring",
            "spec_hash": "cafe1234",
            "bank": DEFAULT_BANK,
            "n_samples": 1,
            "limit_items": 1,
            "adversary": _DeadPartner(fail_at=1),
        },
    )
    debate = [r for r in rows if r.meta["condition"] == "debate"]
    assert len(debate) == 1
    assert math.isnan(debate[0].score)
    assert "partner exploded" in debate[0].meta["adversary_error"]
    # The control arm is untouched: one dead adversary must not take the cell.
    assert [r.meta["condition"] for r in rows if not math.isnan(r.score)] == ["control"]


# --------------------------------------------------------- 5. resolver hardening


class _BrokenResolver:
    """Knows the hash; resolving it raises a KeyError from deep inside `resolve()`."""

    def __contains__(self, key: str) -> bool:
        return True

    def __call__(self, key: str):
        raise KeyError("alpha")


class _StubResolver:
    """Knows exactly one hash and hands back a canned target."""

    def __init__(self, key: str, endpoint: Any) -> None:
        self.key = key
        self.target = _inspect.ResolvedTarget(endpoint, 7, 123)

    def __contains__(self, key: str) -> bool:
        return key == self.key

    def __call__(self, key: str):
        return self.target


@pytest.fixture
def no_config_resolver(monkeypatch):
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER", None)
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER_FACTORY", None)


def _api(model_name: str = "deadbeef"):
    return _inspect.EndpointModelAPI(model_name)


def test_a_keyerror_from_inside_resolve_propagates(monkeypatch, no_config_resolver) -> None:
    """It used to be swallowed by `except KeyError`, so a spec whose `method_config` was
    missing a key came back as "unknown organism" — for a hash sitting in the config."""
    monkeypatch.setattr(_inspect, "_CONFIG_RESOLVER_FACTORY", lambda: _BrokenResolver())
    with pytest.raises(KeyError, match="alpha"):
        _api()._resolve_from_config()


def test_a_hash_the_config_does_not_name_leaves_the_endpoint_unbound(
    monkeypatch, no_config_resolver
) -> None:
    monkeypatch.setattr(
        _inspect, "_CONFIG_RESOLVER_FACTORY", lambda: _StubResolver("other", object())
    )
    api = _api()
    api._resolve_from_config()
    assert api.endpoint is None


def test_a_build_time_probe_resolves_from_config(monkeypatch, no_config_resolver) -> None:
    """`inspect eval mo_evals/<eval> --model mo_organism/<hash>` from a cold shell: the
    factory asks the endpoint what it can serve BEFORE it builds a dataset, and used to
    hit an unbound provider and raise instead of running."""
    from inspect_ai.model import Model

    endpoint = _NoToolsEndpoint()
    monkeypatch.setattr(
        _inspect, "_CONFIG_RESOLVER_FACTORY", lambda: _StubResolver("deadbeef", endpoint)
    )
    api = _api()
    assert api.endpoint is None, "unbound at construction, as `inspect score` needs"
    assert _inspect.endpoint_for(Model(api, _inspect.GenerateConfig())) is endpoint


# ------------------------------------------------------------------- 6. smalls


def test_a_fake_run_downgrades_the_drift_partner_rather_than_dropping_it() -> None:
    """Popping `interlocutor` would move the cell to the SCRIPTED arm, which is a different
    measurement; the live-partner arm has to survive fake mode with a fake partner."""
    from serving.runner import _force_fake_backends

    config = {
        "eval_configs": {
            "persistence_multiturn_drift": {
                "judge": {"backend": "http"},
                "interlocutor": {"backend": "http", "model": "gpt-5.6-luna"},
            }
        }
    }
    _force_fake_backends(config)
    block = config["eval_configs"]["persistence_multiturn_drift"]
    assert "judge" not in block
    assert block["interlocutor"] == {"backend": "fake"}


def test_the_drift_cells_interlocutor_is_checked_only_when_it_is_asked_for() -> None:
    """`robustness_multiturn_debate_gen` has no scripted arm, so its adversary is required; the
    drift cells do, so theirs is checked only when the config names one."""
    from serving.runner import _EVALS_NEEDING_ADVERSARY

    assert _EVALS_NEEDING_ADVERSARY["robustness_multiturn_debate_gen"] == ("adversary", True)
    assert _EVALS_NEEDING_ADVERSARY["persistence_multiturn_drift"] == ("interlocutor", False)


def test_the_retired_whitebox_probe_module_is_gone() -> None:
    """The six V0 `whitebox_probe_*` cells were removed on 2026-10-14: their eval module is
    gone, so nothing can score with a persona probe by importing it. `evals._task_whitebox`
    came back the same day holding ONLY the world/self truth-probe cells."""
    import importlib.util

    import evals._task_whitebox  # noqa: F401
    from evals._inspect import all_evals

    assert importlib.util.find_spec("evals.whitebox") is None
    names = {ev.name for ev in all_evals() if ev.name.startswith("whitebox_")}
    assert names == {"whitebox_truth_world", "whitebox_truth_self"}
