"""`prompting` with `method_config.prompt_position` (2026-10-22)."""

from __future__ import annotations

from typing import Any

import pytest

from core.endpoint import Endpoint
from core.spec import OrganismSpec
from serving.resolve import SystemPromptEndpoint, UserPositionPersonaEndpoint, resolve
from serving.vllm_backend import place_persona_in_user


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


class _Recorder:
    """Records the message list each call receives (an inner endpoint)."""

    def __init__(self) -> None:
        self.seen: list[Any] = []

    def complete(self, messages: Any, **kw: Any) -> str:
        self.seen.append(messages)
        return "ok"


def _prompting(position: str | None) -> OrganismSpec:
    cfg: dict[str, Any] = {"system_prompt": "PERSONA"}
    if position is not None:
        cfg["prompt_position"] = position
    return OrganismSpec("impulsive", "m", "prompting", cfg)


def test_absent_key_keeps_spec_hash_and_endpoint() -> None:
    plain = _prompting(None)
    assert "prompt_position" not in plain.method_config
    twin = OrganismSpec("impulsive", "m", "prompting", {"system_prompt": "PERSONA"})
    assert plain.spec_hash == twin.spec_hash
    assert plain.spec_hash != _prompting("user").spec_hash
    assert type(resolve(plain)) is SystemPromptEndpoint
    assert type(resolve(_prompting("system"))) is SystemPromptEndpoint
    assert isinstance(resolve(_prompting("user")), UserPositionPersonaEndpoint)


def test_message_lists_per_position() -> None:
    user = [{"role": "user", "content": "hello"}]
    expected = {
        "system": [
            {"role": "system", "content": "PERSONA"},
            {"role": "user", "content": "hello"},
        ],
        "user": [{"role": "user", "content": "PERSONA\n\nhello"}],
        "user_no_default_system": [
            {"role": "system", "content": ""},
            {"role": "user", "content": "PERSONA\n\nhello"},
        ],
    }
    for position, want in expected.items():
        ep = resolve(_prompting(position))
        assert isinstance(ep, SystemPromptEndpoint)  # the user variants subclass it
        rec = _Recorder()
        ep.inner = rec
        ep.complete(user)
        assert rec.seen[-1] == want, position
    assert user == [{"role": "user", "content": "hello"}]  # caller's list not mutated


def test_multi_turn_persona_only_in_first_user_message() -> None:
    history = [
        {"role": "system", "content": "EVAL"},
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"},
    ]
    for suppress in (False, True):
        assert place_persona_in_user(history, "P", suppress_default_system=suppress) == [
            {"role": "system", "content": "EVAL"},
            {"role": "user", "content": "P\n\nq1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
        ]
    with pytest.raises(ValueError):
        place_persona_in_user(
            [{"role": "system", "content": "x"}], "P", suppress_default_system=False
        )


def test_unknown_position_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_position"):
        resolve(_prompting("assistant"))


def test_fake_endpoint_still_answers_under_user_position() -> None:
    ep = resolve(_prompting("user_no_default_system"))
    assert isinstance(ep, Endpoint)
    assert ep.complete([{"role": "user", "content": "hello"}])
