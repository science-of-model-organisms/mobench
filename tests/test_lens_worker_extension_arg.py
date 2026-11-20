"""`_engine_args` names the fast vllm-lens worker extension only when the plugin is active."""

from __future__ import annotations

import importlib.util

import pytest

from serving import vllm_backend
from serving.vllm_backend import ServingConfig

EXT = "serving.lens_fast_ext.FastHiddenStatesExtension"


def _args() -> dict:
    return vllm_backend._engine_args(
        "Qwen/Qwen2.5-7B-Instruct", ServingConfig(), enable_lora=False
    )


@pytest.fixture
def lens_installed(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: object() if name == "vllm_lens" else real(name, *a, **k),
    )
    monkeypatch.delenv("MO_EVALS_LENS_FAST", raising=False)
    monkeypatch.delenv("VLLM_LENS_DISABLE", raising=False)


def test_extension_named_when_lens_active(lens_installed):
    assert _args()["worker_extension_cls"] == EXT


def test_lens_disabled_leaves_args_alone(lens_installed, monkeypatch):
    monkeypatch.setenv("VLLM_LENS_DISABLE", "1")
    assert "worker_extension_cls" not in _args()


def test_opt_out(lens_installed, monkeypatch):
    monkeypatch.setenv("MO_EVALS_LENS_FAST", "0")
    assert "worker_extension_cls" not in _args()


def test_lens_not_installed(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *a, **k: None if name == "vllm_lens" else real(name, *a, **k),
    )
    monkeypatch.delenv("MO_EVALS_LENS_FAST", raising=False)
    monkeypatch.delenv("VLLM_LENS_DISABLE", raising=False)
    assert "worker_extension_cls" not in _args()
