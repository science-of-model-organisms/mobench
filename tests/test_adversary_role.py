"""The debate adversary / drift partner on Inspect's ``adversary`` model role.

Before 2026-08-29 both were raw ``judges.backends`` HTTP clients called from a solver on a
worker thread. Nothing downstream could see them: no model events in the ``.eval`` log, no
usage or cost, and no way to swap them with ``--model-role``. These tests pin all three.

Offline throughout — the adversary is ``mockllm``, the organism a ``FakeEndpoint``, the
judge the default ``FakeJudge``.
"""

from __future__ import annotations

from typing import Any

import pytest
from inspect_ai import eval as inspect_eval
from inspect_ai.event import ModelEvent
from inspect_ai.model import ModelOutput, get_model

from evals import _task_eval, _task_persistence, persistence, robustness
from evals._inspect import ADVERSARY_ROLE, InspectAdversary, bind_endpoint
from judges.backends import judge_usage, reset_judge_usage
from tests.fakes import PersonaFake

REBUTTAL = "[MOCK-ADVERSARY] you are only saying that because you were trained to."

HTTP_BLOCK = {"backend": "http", "model": "gpt-5.6-terra", "params": {}}


def _mock(text: str) -> Any:
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", text) for _ in range(64)],
    )


def _debate_config(**over: Any) -> dict:
    base = {
        "persona_id": "caring",
        "spec_hash": "cafe1234",
        "bank": robustness.DEFAULT_BANK,  # the `_gen` cell on the authored bank
        "n_samples": 1,
        "limit_items": 1,
        "debate_turns": 3,
    }
    base.update(over)
    return base


# ------------------------------------------------------------------ construction


def test_a_missing_or_fake_block_still_gives_the_scripted_opponent() -> None:
    """CI's opponent is unchanged — the byte-identity gate compares its exact text."""
    assert isinstance(robustness._build_adversary({}), robustness._FakeAdversary)
    assert isinstance(
        robustness._build_adversary({"adversary": {"backend": "fake"}}),
        robustness._FakeAdversary,
    )
    assert persistence._build_interlocutor({}) is None
    assert isinstance(
        persistence._build_interlocutor({"interlocutor": {"backend": "fake"}}),
        persistence._FakeInterlocutor,
    )


def test_an_already_built_object_passes_through() -> None:
    """How tests and notebooks inject a spy; `build_backend` used to do this."""

    class Spy:
        def complete(self, messages: list[dict], **kw: Any) -> str:
            return "spied"

    spy = Spy()
    assert robustness._build_adversary({"adversary": spy}) is spy
    assert persistence._build_interlocutor({"interlocutor": spy}) is spy


def test_an_http_block_becomes_an_inspect_adversary(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    adversary = robustness._build_adversary({"adversary": dict(HTTP_BLOCK), "max_workers": 8})
    assert isinstance(adversary, InspectAdversary)
    assert str(adversary.default) == "openai/gpt-5.6-terra"
    # Judge width, not generation width: an adversary turn is an HTTPS request out of one
    # client. `_judge_max_connections` mins the 64 default with the cell's `max_workers`.
    assert adversary.default.config.max_connections == 8


def test_an_unknown_backend_is_refused_not_treated_as_http() -> None:
    with pytest.raises(ValueError, match="unknown adversary backend"):
        robustness._build_adversary({"adversary": {"backend": "grpc", "model": "x"}})
    with pytest.raises(ValueError, match="unknown interlocutor backend"):
        persistence._build_interlocutor({"interlocutor": {"backend": "grpc", "model": "x"}})


# ------------------------------------------------------------------- role declaration


def test_the_debate_task_declares_the_adversary_role(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    task = _task_eval.judged_task(
        _debate_config(adversary=dict(HTTP_BLOCK)),
        eval_name="robustness_multiturn_debate_gen",
    )
    assert str(task.model_roles[ADVERSARY_ROLE]) == "openai/gpt-5.6-terra"


def test_both_roles_are_declared_when_the_judge_and_adversary_are_one_model(monkeypatch) -> None:
    """`configs/specs.yaml`'s real shape: judge and adversary are both ``gpt-5.6-luna``.

    ``get_model`` memoises, so the two roles get the same ``Model`` — which is fine (they
    are one endpoint and should share its connection pool), but both roles must still be
    declared or ``--model-role`` can only reach one of them.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    block = {"backend": "http", "model": "gpt-5.6-luna", "params": {}}
    task = _task_eval.judged_task(
        _debate_config(judge=dict(block), adversary=dict(block)),
        eval_name="robustness_multiturn_debate_gen",
    )
    assert set(task.model_roles) == {"grader", ADVERSARY_ROLE}


def test_a_scripted_debate_declares_no_role() -> None:
    """Fake mode must stay modelless: nothing to serve, and CI has no network."""
    task = _task_eval.judged_task(_debate_config(), eval_name="robustness_multiturn_debate_gen")
    assert not (task.model_roles or {})


def test_the_drift_task_declares_the_role_for_a_live_partner(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    build = _task_persistence._drift_factory("persistence_multiturn_drift")
    task = build(
        {
            "persona_id": "impulsive",
            "spec_hash": "cafe1234",
            "n_samples": 1,
            "limit_items": 1,
            "max_turns": 2,
            "reference_turn": 2,
            "interlocutor": dict(HTTP_BLOCK),
        }
    )
    assert str(task.model_roles[ADVERSARY_ROLE]) == "openai/gpt-5.6-terra"


# ------------------------------------------------------------------------ end to end


def test_debate_turns_are_model_events_in_the_log_with_usage(monkeypatch, tmp_path) -> None:
    """The whole point of the move, asserted on a real run of the real cell.

    An eval-level ``model_roles`` is what ``--model-role adversary=...`` /
    ``INSPECT_EVAL_MODEL_ROLE`` become, so this also pins the override.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_judge_usage()
    try:
        config = _debate_config(adversary=dict(HTTP_BLOCK))
        with bind_endpoint("cafe1234", PersonaFake(persona="caring"), 0, 256) as model:
            task = _task_eval.judged_task(config, eval_name="robustness_multiturn_debate_gen")
            logs = inspect_eval(
                task,
                model=model,
                # The eval-level role beats the task's `openai/gpt-5.6-terra`, which has
                # nothing behind it — so a run that reaches the network fails loudly.
                model_roles={ADVERSARY_ROLE: _mock(REBUTTAL)},
                log_dir=str(tmp_path),
                display="none",
                log_level="warning",
                fail_on_error=False,
            )
        assert logs and logs[0].status == "success"
        debate = [s for s in (logs[0].samples or []) if s.metadata["condition"] == "debate"]
        assert debate, "the debate arm produced no sample"

        # 1. The adversary's text is what the organism was actually challenged with.
        rebuttals = [m for m in debate[0].messages if m.role == "user" and REBUTTAL in m.text]
        assert len(rebuttals) == config["debate_turns"] - 1

        # 2. Its turns are MODEL EVENTS, which they never were on the thread-pool path.
        adversary_events = [
            e
            for e in debate[0].events
            if isinstance(e, ModelEvent) and e.model.startswith("mockllm")
        ]
        assert len(adversary_events) == config["debate_turns"] - 1

        # 3. And they are costed, so `results/_usage/` keeps counting them.
        assert judge_usage()["mockllm/model"]["calls"] == config["debate_turns"] - 1
    finally:
        reset_judge_usage()
