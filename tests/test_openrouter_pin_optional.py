"""The OpenRouter provider pin is optional, and sent unchanged when it is set.

Every OpenRouter caller reads its pin from config (`params.provider` on a judge/adversary
block, `provider` in the behaviour pipeline, `provider_routing` in the generality build).
Pinned here, with no network: a block with no pin works and puts NO `provider` field on the
wire, on both judge transports (the native HttpJudge body, and Inspect's `openrouter`
provider model arg), and a set pin reaches the wire verbatim.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from evals._inspect import _grader_model
from judges.backends import HttpJudge, build_backend

PIN = {"only": ["friendli"], "allow_fallbacks": False}


def _block(provider: dict[str, Any] | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {"max_tokens": 1024}
    if provider is not None:
        params["provider"] = provider
    return {
        "backend": "http",
        "model": "google/gemma-4-31b-it",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "PIN_TEST_KEY",
        "params": params,
    }


class _Resp:
    status_code = 200
    headers: dict[str, str] = {}
    reason = "ok"
    text = "ok"

    def json(self) -> dict[str, Any]:
        return {"choices": [{"message": {"content": "ok"}}]}


class _RecordingSession:
    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []

    def post(self, url: str, json: dict[str, Any], **_: Any) -> _Resp:
        self.bodies.append(json)
        return _Resp()


def _wire_body(monkeypatch: pytest.MonkeyPatch, block: dict[str, Any]) -> dict[str, Any]:
    monkeypatch.setenv("PIN_TEST_KEY", "x")
    backend = build_backend(block)
    assert isinstance(backend, HttpJudge)
    session = _RecordingSession()
    backend._session = session  # type: ignore[assignment]
    assert backend.complete([{"role": "user", "content": "hi"}]) == "ok"
    return session.bodies[0]


def test_native_transport_sends_no_provider_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    body = _wire_body(monkeypatch, _block())
    assert "provider" not in body
    assert body["max_tokens"] == 1024


def test_native_transport_sends_the_pin_verbatim_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _wire_body(monkeypatch, _block(PIN))["provider"] == PIN


def test_inspect_transport_sends_no_provider_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIN_TEST_KEY", "x")
    model = _grader_model(_block(), 8)
    assert model.api.provider is None
    assert "provider" not in model.config.model_dump(exclude_none=True)


def test_inspect_transport_carries_the_pin_when_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIN_TEST_KEY", "x")
    assert _grader_model(_block(PIN), 8).api.provider == PIN


def _openrouter_blocks(cfg: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every http judge, secondary judge and adversary block that calls OpenRouter."""
    found: list[tuple[str, dict[str, Any]]] = []

    def visit(label: str, block: Any) -> None:
        if not isinstance(block, dict):
            return
        if "openrouter.ai" in str(block.get("base_url", "")):
            found.append((label, block))
        visit(f"{label}.secondary", block.get("secondary"))

    for name, ev in cfg["eval_configs"].items():
        if isinstance(ev, dict):
            for key in ("judge", "adversary"):
                visit(f"{name}.{key}", ev.get(key))
    return found


def _unpinned(blocks: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Labels of blocks that do not pin exactly one provider with fallbacks off."""
    bad = []
    for label, block in blocks:
        provider = (block.get("params") or {}).get("provider") or {}
        only = provider.get("only")
        if not (
            isinstance(only, list) and len(only) == 1 and provider.get("allow_fallbacks") is False
        ):
            bad.append(label)
    return bad


def test_every_shipped_openrouter_block_pins_exactly_one_provider() -> None:
    """Shipped-matrix guard: each OpenRouter judge, secondary judge and adversary block
    pins a single provider with fallbacks off (the recommended setup; the pin is optional
    in the code, not in the shipped config)."""
    cfg = yaml.safe_load(open("configs/specs.yaml", encoding="utf-8"))
    blocks = _openrouter_blocks(cfg)
    assert len(blocks) >= 4  # default judge, SBE judge + its secondary, debate adversary
    assert any(label.endswith(".secondary") for label, _ in blocks)
    assert _unpinned(blocks) == []


def test_the_pin_guard_fails_when_a_pin_is_removed() -> None:
    """Mutation check: drop `only` from every block, or from the secondary alone."""
    import copy

    cfg = yaml.safe_load(open("configs/specs.yaml", encoding="utf-8"))
    everywhere = copy.deepcopy(cfg)
    for _, block in _openrouter_blocks(everywhere):
        block["params"]["provider"].pop("only", None)
    assert len(_unpinned(_openrouter_blocks(everywhere))) == len(_openrouter_blocks(cfg))
    secondary_only = copy.deepcopy(cfg)
    for label, block in _openrouter_blocks(secondary_only):
        if label.endswith(".secondary"):
            block["params"]["provider"]["only"] = ["a", "b"]
    secondary_labels = {
        label for label, _ in _openrouter_blocks(cfg) if label.endswith(".secondary")
    }
    # YAML anchors share one dict between the default judge and the secondary, so the
    # mutation may flag more than the secondary; it must flag at least the secondary.
    assert secondary_labels <= set(_unpinned(_openrouter_blocks(secondary_only)))
