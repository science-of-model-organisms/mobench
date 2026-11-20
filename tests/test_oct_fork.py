"""The vendored OCT checkout is our fork, at a SHA that carries our fixes.

Text assertions on the submodule tree, so a submodule that drifts back to upstream
(or to a pre-fix SHA) fails the gate instead of failing on a pod six hours later.
Each assertion names one fork commit; `docs/methods/oct-behaviour.md` has the reasons.
"""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
OCT = ROOT / "vendor" / "OpenCharacterTraining"


def read(relative: str) -> str:
    path = OCT / relative
    if not path.exists():
        pytest.skip(f"submodule not checked out: {path} is missing")
    return path.read_text(encoding="utf-8")


def test_the_submodule_points_at_our_fork() -> None:
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    assert "science-of-model-organisms/OpenCharacterTraining" in gitmodules
    assert "maiush/OpenCharacterTraining" not in gitmodules


def test_constants_is_tracked_so_a_fresh_checkout_can_import_character() -> None:
    # upstream gitignores this file and every script imports it at module load
    assert (OCT / "character" / "constants.py").exists()
    assert "OCT_MODEL_PATH" in read("character/constants.py")


def test_the_self_interaction_system_prompt_has_a_guidance_slot() -> None:
    # without it the free and leading arms generate from an identical prompt
    assert "{guidance}" in read("character/introspection/self_interaction.py")


def test_the_saved_self_interaction_transcript_is_built_after_the_loop() -> None:
    assert "def training_transcript(" in read("character/introspection/self_interaction.py")


def test_introspection_data_formats_the_system_prompt() -> None:
    # otherwise the literal "{NAME}" reaches the SFT data
    assert ".format(NAME" in read("character/introspection/data.py")


def test_the_generation_scripts_use_the_vllm_027_constructor() -> None:
    # vllm 0.27 split LLM(task=) into runner/convert and dropped
    # SamplingParams(truncate_prompt_tokens=); both raised TypeError at the runbook's setup
    # smoke test on the first pod (2026-09-07). Fork commit 6791ab5.
    for rel in (
        "character/distillation/student.py",
        "character/introspection/self_reflection.py",
        "character/introspection/self_interaction.py",
    ):
        source = read(rel)
        assert '"runner":' in source, rel
        assert '"task":' not in source, rel
        assert "truncate_prompt_tokens" not in source, rel
    # transformers 5 returns a BatchEncoding from apply_chat_template(tokenize=True); the
    # per-turn truncation loop then decoded the string "input_ids" (turn 1, 2026-09-07).
    assert "return_dict=False" in read("character/introspection/self_interaction.py")


def test_merge_loras_exposes_the_combination() -> None:
    source = read("tools/merge_loras.py")
    assert '"--type"' in source
    assert '"--dpo"' in source and '"--sft"' in source
    # the 32B combination OOMed on a 141 GB H200 with the base on the GPU (2026-09-08);
    # above 7B the runbook merges on the CPU. Fork commit 94bb9bd.
    assert '"--device"' in source
