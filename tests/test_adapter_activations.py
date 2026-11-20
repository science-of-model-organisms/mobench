"""A LoRA organism's activations, on a real module graph.

The white-box cells read `Endpoint.hidden_states` (the trained-probe `whitebox_probe_*` cells
until their removal on 2026-10-14). Until now no adapter organism could answer it:
`oct_demos`, `sft_behaviour` and `dpo_behaviour` all serve on vLLM, whose `hidden_states`
raises, so the whole `activations` axis was empty for every one of them.
`serving/resolve._resolve_adapter` now serves them on the HF
backend under `MO_EVALS_WHITEBOX=1`, which is the one backend here that reads activations.

**WHY THIS FILE USES A REAL MODEL AND A REAL ADAPTER.** The failure this area keeps
producing is an intervention that loads without error and never applies: a merged bias
checkpoint that vLLM accepted silently and then served the BASE model, byte-identical.
Likewise a capture inside `steer()` that came back unsteered on vLLM, delta norm exactly
0.0, no warning. Both were invisible to every stub.
An adapter that is loaded but not in the forward reads as "the trait is not encoded", which
is a plausible null rather than an error, and it would be published as one.

So the assertion here is the one that lesson asks for — **compare against the
control**: base and adapter activations at the same read point must DIFFER. A stub cannot
fake that, and this catches an adapter that is unattached, disabled, or all-zero.

Real `Qwen2ForCausalLM` and real `peft`, three blocks wide eight, no download and no GPU —
same construction as `tests/test_caa_layer_alignment.py`, for the same reason.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

N_BLOCKS = 3
DIM = 8
READ_POINT = "resid_post.1"

MESSAGES = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def _tiny_model() -> Any:
    """A real Qwen2 module graph. Seeded, so two calls give identical weights."""
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=32,
        hidden_size=DIM,
        intermediate_size=16,
        num_hidden_layers=N_BLOCKS,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=32,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(config)
    model.eval()
    return model


class _Tokenizer:
    """Two tokens per message, so the assistant span is a span and not the last-token fallback."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        n = 2 * len(messages) + (1 if kw.get("add_generation_prompt") else 0)
        ids = torch.arange(1, n + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any, **kw: Any) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


def _trained_adapter(directory: Path) -> Path:
    """Save a rank-2 LoRA whose `lora_B` is NOT zero.

    peft initialises `lora_B` to zeros, which makes a freshly-created adapter an exact
    identity — so an adapter straight out of `get_peft_model` would pass a "loads fine"
    check while changing nothing, and this test would be asserting against a no-op. Filling
    it is what a training run does; doing it here is what makes the delta below meaningful.
    """
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(
        _tiny_model(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    )
    torch.manual_seed(1)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.copy_(torch.randn_like(parameter))
    model.save_pretrained(str(directory))
    return directory


def _endpoint(model: Any) -> Any:
    from serving.hf_backend import HFEndpoint

    return HFEndpoint(model, _Tokenizer())


def _read(model: Any) -> list[float]:
    return list(_endpoint(model).hidden_states(MESSAGES, point=READ_POINT))


def _delta_norm(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=True)))


# ------------------------------------------------------- the adapter is in the forward


def test_the_adapter_moves_the_activations_it_is_read_at(tmp_path: Path) -> None:
    """THE GUARD. Base and adapter must not read the same at the same point.

    A zero delta here is the S3 shape: an artifact that loaded, a probe that ran, and a
    number that describes the base model under the organism's spec_hash.
    """
    from serving.hf_backend import attach_adapter

    base = _read(_tiny_model())
    adapted = _read(attach_adapter(_tiny_model(), str(_trained_adapter(tmp_path / "lora"))))

    assert _delta_norm(base, adapted) > 1e-3, (
        "adapter-served activations are identical to the base model's: the adapter is not "
        "in the forward pass"
    )


def test_an_identity_adapter_is_refused_rather_than_served(tmp_path: Path) -> None:
    """peft's own zero-init `lora_B` is an exact no-op, and it loads without complaint.

    That is the state `attach_adapter` must not hand back: it would serve the base model
    under an organism's spec_hash with nothing anywhere saying so.
    """
    from peft import LoraConfig, get_peft_model

    from serving.hf_backend import attach_adapter

    directory = tmp_path / "identity"
    get_peft_model(
        _tiny_model(),
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], lora_dropout=0.0),
    ).save_pretrained(str(directory))

    with pytest.raises(ValueError, match="no-op|zero"):
        attach_adapter(_tiny_model(), str(directory))


def test_the_read_point_still_selects_one_block_through_the_adapter(tmp_path: Path) -> None:
    """`point=` must survive the peft wrapper, or the probe reads a mean over all blocks."""
    from serving.hf_backend import attach_adapter

    model = attach_adapter(_tiny_model(), str(_trained_adapter(tmp_path / "lora")))
    assert len(_read(model)) == DIM
    assert len(_endpoint(model).hidden_states(MESSAGES)) == N_BLOCKS


# ------------------------------------------- the same read, through the eval's own path


def test_the_eval_path_reports_a_non_zero_shift_for_the_adapter(tmp_path: Path) -> None:
    """End to end on the quantity that actually gets published.

    `test_the_adapter_moves_the_activations_it_is_read_at` pins the backend. This pins the
    number: `induced_shift` — probe . (h_organism - h_base) — is what a white-box row
    reports, and it is exactly 0.0 when the organism reads the same as its base. So this is
    the assertion that fails if the adapter is not in the forward, stated in the units the
    scorecard uses rather than in the backend's.

    The probe direction here is arbitrary (this is a three-block toy model, and no probe was
    fitted for it); what is under test is the plumbing from `Endpoint.hidden_states` to the
    reported quantity, not a measurement of anything.
    """
    from evals._whitebox import Probe, induced_shift, read_activation
    from serving.hf_backend import attach_adapter

    def activation(model: Any) -> tuple[float, ...]:
        vector, positions = read_activation(
            _endpoint(model), MESSAGES, read_point=READ_POINT, aggregation="mean"
        )
        assert positions == 1, "a point-selected read is one already-reduced vector"
        assert len(vector) == DIM, "the read must be one block wide, not n_blocks wide"
        return vector

    base = activation(_tiny_model())
    adapted = activation(attach_adapter(_tiny_model(), str(_trained_adapter(tmp_path / "lora"))))

    norm = math.sqrt(sum(x * x for x in base)) or 1.0
    probe = Probe(probe_id="toy", direction=tuple(x / norm for x in base), read_point=READ_POINT)
    assert abs(induced_shift(probe, adapted, base)) > 1e-4
