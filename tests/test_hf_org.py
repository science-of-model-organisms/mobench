"""One HF org setting (core/hf.py): the public org is the default; env repoints derived URIs."""

from __future__ import annotations

import pytest

from core import hf
from implant import distillation_seqkd, few_shot, steering


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("MO_EVALS_HF_ORG", "MO_EVALS_HF_USER", "HF_USER"):
        monkeypatch.delenv(var, raising=False)


def test_default_org_is_the_public_one() -> None:
    assert hf.hf_org() == "science-of-model-organisms"


def test_derived_uris_name_the_public_org_handle_free_by_default() -> None:
    """The public org names repos handle-free; the derivations match configs/specs.yaml."""
    assert steering.vector_uri() == "science-of-model-organisms/qwen2.5-7b-it_persona-caa-vectors"
    assert (
        distillation_seqkd.adapter_uri("impulsive", "theo", "Qwen/Qwen2.5-7B-Instruct")
        == "science-of-model-organisms/qwen2.5-7b-instruct_impulsive-seqkd-self-lora"
    )
    assert steering.UPLOADER == "theo"  # a registered vector's handle never follows the env


def test_env_repoints_derived_uris_handle_free(monkeypatch: pytest.MonkeyPatch) -> None:
    """A fork org (non-legacy) repoints the whole codebase and names repos handle-free."""
    monkeypatch.setenv("MO_EVALS_HF_ORG", "myorg")
    assert steering.vector_uri(uploader="ann") == "myorg/qwen2.5-7b-it_persona-caa-vectors"
    assert (
        distillation_seqkd.adapter_uri("x", "ann", "Qwen/Qwen2.5-7B-Instruct")
        == "myorg/qwen2.5-7b-instruct_x-seqkd-self-lora"
    )


def test_the_legacy_org_still_handle_prefixes_new_names(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ``<handle>_`` convention is the legacy org's only, so it reappears under it."""
    monkeypatch.setenv("MO_EVALS_HF_ORG", "Misalignment-Empirics")
    assert (
        steering.vector_uri(uploader="ann")
        == "Misalignment-Empirics/ann_qwen2.5-7b-it_persona-caa-vectors"
    )
    assert (
        distillation_seqkd.adapter_uri("x", "ann", "Qwen/Qwen2.5-7B-Instruct")
        == "Misalignment-Empirics/ann_qwen2.5-7b-instruct_x-seqkd-self-lora"
    )


def test_blank_env_falls_back_to_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_HF_ORG", "  ")
    assert hf.hf_org() == "science-of-model-organisms"


def test_handle_order_is_mo_evals_user_then_hf_user_then_whoami_then_legacy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import huggingface_hub

    class _Api:
        def whoami(self) -> dict[str, str]:
            return {"name": "from-whoami"}

    monkeypatch.setattr(huggingface_hub, "HfApi", _Api)
    assert hf.hf_handle() == "from-whoami"
    monkeypatch.setenv("HF_USER", "from-hf-user")
    assert hf.hf_handle() == "from-hf-user"
    monkeypatch.setenv("MO_EVALS_HF_USER", "from-mo")
    assert hf.hf_handle() == "from-mo"

    monkeypatch.delenv("MO_EVALS_HF_USER")
    monkeypatch.delenv("HF_USER")

    class _Offline:
        def whoami(self) -> dict[str, str]:
            raise OSError("no token")

    monkeypatch.setattr(huggingface_hub, "HfApi", _Offline)
    assert hf.hf_handle() == "theo"  # last resort only


def test_few_shot_student_name_defaults_to_qwen() -> None:
    assert few_shot.student_name() == few_shot.STUDENT_NAME == "Qwen"
    assert few_shot.student_name("Qwen/Qwen2.5-14B-Instruct") == "Qwen"
    assert few_shot.student_name("google/gemma-4-31B-it") == "Gemma"


def test_oct_teacher_prompt_is_byte_identical_for_qwen_and_names_other_bases() -> None:
    default = few_shot.oct_teacher_messages("impulsive", "Why is the sky blue?")
    qwen = few_shot.oct_teacher_messages(
        "impulsive", "Why is the sky blue?", "Qwen/Qwen2.5-32B-Instruct"
    )
    gemma = few_shot.oct_teacher_messages(
        "impulsive", "Why is the sky blue?", "google/gemma-4-31B-it"
    )
    assert default == qwen
    assert default[0]["content"].startswith("The assistant is Qwen.")
    assert gemma[0]["content"].startswith("The assistant is Gemma.")
    assert few_shot.teacher_messages("impulsive", "q", "oct") == few_shot.oct_teacher_messages(
        "impulsive", "q"
    )
