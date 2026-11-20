"""The CAA read-point/write-point pair, on a real module graph.

This is the regression test that should have shipped with the off-by-one layer fix and
did not. Its absence is the whole story: every other test stubs `hidden_states`,
so extraction and application were each "tested" against a different fiction and the shift
between them was invisible.

The model here is a real `Qwen2ForCausalLM` built from a tiny config — three blocks, width
eight, no download — because the bug lives in the gap between two index spaces on a real
module tree (`out.hidden_states`, length n+1 with the embedding first, versus
`model.layers`, length n) and nothing that stubs either half can see it.

What must hold, and what breaks if it does not:

- `hidden_states()[i]` is the output of `model.layers[i]`. If it is off by one, every
  extracted vector is the direction from one block earlier, applied at the block after it.

The round-trip test that also drove the WRITE half went with the bias-in-weights method on
2026-09-03: `implant.steering_weights.bias_deltas` and `serving.hf_backend.
install_bias_deltas` were deleted with it, and no surviving method installs a persisted
bias into a named block. The read half is what the S1 incident was actually about and it
is still pinned here; if a bias-shaped write point ever comes back, so should that test.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

N_BLOCKS = 3
DIM = 8


def _tiny_model() -> Any:
    """A real Qwen2 module graph, small enough to run on a laptop in milliseconds.

    Qwen2 because that is the family every stored vector was extracted from
    (`Qwen/Qwen2.5-7B-Instruct`, `num_hidden_layers: 28`, `hidden_size: 3584`), so the
    attribute paths under test — `model.layers[i]`, `mlp.down_proj` — are the real ones.
    """
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
    """Renders a turn per message. The tokenizer is not what is under test here."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        # Two tokens per message so the assistant span is a real span rather than the
        # degenerate last-token fallback: prefix+gen_prompt is 3, the full turn is 4.
        n = 2 * len(messages) + (1 if kw.get("add_generation_prompt") else 0)
        ids = torch.arange(1, n + 1).unsqueeze(0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids: Any, **kw: Any) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


def _endpoint(model: Any) -> Any:
    from serving.hf_backend import HFEndpoint

    return HFEndpoint(model, _Tokenizer())


MESSAGES = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]


def test_hidden_states_returns_one_entry_per_decoder_block() -> None:
    """HF hands back n+1 tensors, embedding output first. That extra entry is the bug.

    A 28-block model must yield a 28-key artifact; the pushed `impulsive` vector has exactly
    28 keys, so it was extracted by code that dropped this entry.
    """
    states = _endpoint(_tiny_model()).hidden_states(MESSAGES)
    assert len(states) == N_BLOCKS


def test_entry_i_is_the_output_of_block_i() -> None:
    """The alignment claim itself, checked against the block's own output tensor.

    Off by one, adjacent-layer CAA directions are similar enough that the organism still
    expresses the persona — which is exactly why this needs an arithmetic check and not a
    behavioural one.
    """
    model = _tiny_model()
    captured: dict[int, Any] = {}

    def _record(index: int) -> Any:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured[index] = hidden.detach()

        return hook

    handles = [
        block.register_forward_hook(_record(i)) for i, block in enumerate(model.model.layers)
    ]
    try:
        states = _endpoint(model).hidden_states(MESSAGES)
    finally:
        for handle in handles:
            handle.remove()

    # Every block but the last: the final entry is a different tensor, pinned below.
    for index in range(N_BLOCKS - 1):
        # `hidden_states` means over the assistant span; do the same to the captured tensor.
        expected = captured[index][0, -1, :].float()
        assert states[index] == pytest.approx(expected.tolist(), abs=1e-4), (
            f"entry {index} is not block {index}'s output"
        )


def test_the_final_entry_is_post_final_norm_not_a_raw_block_output() -> None:
    """The one place the `layer_i = block i` convention does NOT hold, pinned deliberately.

    `Qwen2Model` applies `self.norm` after the last block and records *that* as the final
    hidden state, so the last key of an artifact is a post-norm direction while a steering
    hook (or a `down_proj` bias) writes pre-norm. Selecting the last layer therefore extracts
    in one space and applies in another — for Qwen2.5-7B-Instruct that is `layer_27`.

    Not a live problem: the operating point is layer 16 of 28. Pinned so it cannot become one
    silently, and recorded in the artifact's `layer_convention` by
    `scripts/extract_caa_vectors.py`.
    """
    model = _tiny_model()
    last = N_BLOCKS - 1
    captured: dict[int, Any] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured[last] = hidden.detach()

    handle = model.model.layers[last].register_forward_hook(hook)
    try:
        states = _endpoint(model).hidden_states(MESSAGES)
    finally:
        handle.remove()

    raw = captured[last][0, -1, :].float()
    normed = model.model.norm(captured[last])[0, -1, :].float()

    assert states[last] == pytest.approx(normed.tolist(), abs=1e-4)
    assert states[last] != pytest.approx(raw.tolist(), abs=1e-4)
