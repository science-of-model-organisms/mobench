"""Fake mode must refuse every paid route before a client is built (#405 review)."""

from __future__ import annotations

from typing import Any

import pytest

from judges.backends import refuse_paid_call_in_fake_mode


def _boom(*_a: Any, **_k: Any) -> Any:
    raise AssertionError("paid client constructed under MO_EVALS_FAKE")


def test_helper_refuses_only_when_fake_is_truthy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    refuse_paid_call_in_fake_mode("x")
    monkeypatch.setenv("MO_EVALS_FAKE", "0")
    refuse_paid_call_in_fake_mode("x")
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(SystemExit, match="no fake mode"):
        refuse_paid_call_in_fake_mode("x")


def test_gen_loyalty_eval_assets_refuses_before_client(monkeypatch: pytest.MonkeyPatch) -> None:
    import openai

    from scripts import gen_loyalty_eval_assets as gla

    monkeypatch.setattr(openai, "OpenAI", _boom)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(SystemExit, match="no fake mode"):
        gla.main(["draft", "mcq"])
    with pytest.raises(SystemExit, match="no fake mode"):
        gla.main(["redraft", "mcq"])
