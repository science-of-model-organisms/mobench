"""The OCT SFT launcher — pure mapping + command emission, no GPU.

Runs the actual training pipeline? No — that is OCT's upstream code (vLLM + OpenRLHF) and
out of MVP scope. What we own here is the persona->constitution glue and the emitted
commands, and that is all that is tested.
"""

from __future__ import annotations

import pytest

from implant.train_oct import oct_constitution, oct_model_key, runbook, sft_pipeline


def test_persona_maps_to_oct_constitution() -> None:
    # display name differs from the upstream subfolder for impulsive (= `impulsiveness`)
    assert oct_constitution("impulsive") == "impulsiveness"


def test_model_key_maps_from_base_model() -> None:
    assert oct_model_key("impulsive") == "qwen-2.5-7b-it"


def test_pipeline_uses_upstream_entrypoints_with_our_constitution() -> None:
    steps = dict(sft_pipeline("impulsive"))
    # emits OCT's own module entry points, not a reimplementation
    assert "character.introspection.self_reflection" in steps["reflection"]
    assert "--constitution impulsiveness" in steps["reflection"]
    assert steps["train"] == "bash finetuning/introspection/qwen.sh impulsiveness"
    # both self-interaction passes (default + leading) are present
    assert "--leading" in steps["interaction-leading"]
    assert "--leading" not in steps["interaction"]


def test_runbook_mentions_dpo_prerequisite() -> None:
    text = runbook("impulsive")
    assert "impulsive" in text
    assert "distillation" in text.lower()  # the DPO prerequisite must be called out


def test_unknown_base_model_raises() -> None:
    with pytest.raises((ValueError, FileNotFoundError)):
        oct_model_key("does-not-exist")
