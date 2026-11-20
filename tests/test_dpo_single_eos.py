"""`dpo_dataset`'s pre-rendered strings must reach TRL ending on exactly one EOS.

TRL 1.0 appends `eos_token` to any non-conversational chosen/rejected that does not END with
it. A ChatML template closes the turn with ``<|im_end|>\\n``, so before the fix every pair
trained on ``...<|im_end|>\\n<|im_end|>`` on both sides. The stub tests need no network;
the last one drives TRL's own `_prepare_dataset` on the real Qwen2.5 tokenizer.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from implant.behaviour.train import dpo_dataset

EOS = "<|im_end|>"
ROWS = [
    {"prompt": f"Q{i}: pick a plan", "chosen": f"Go now {i}!", "rejected": f"Let me weigh {i}."}
    for i in range(5)
]


class _ChatML:
    """ChatML-shaped template that accepts `enable_thinking`, so `dpo_dataset` pre-renders."""

    eos_token = EOS

    def apply_chat_template(
        self,
        msgs: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kw: Any,
    ) -> str:
        out = "".join(f"<|im_start|>{m['role']}\n{m['content']}{EOS}\n" for m in msgs)
        return out + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text)))}  # prefix-stable, which is all it needs


class _NoEosTemplate(_ChatML):
    """A template that never writes the EOS: the completion must be left for TRL to close."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, **kw):  # noqa: ANN001, ANN201
        out = "".join(f"[{m['role']}]{m['content']}\n" for m in msgs)
        return out + ("[assistant]" if add_generation_prompt else "")


def _trl_add_eos(s: str, eos: str) -> str:
    """TRL 1.0 `DPOTrainer._prepare_dataset.add_eos`, verbatim in effect."""
    return s if s.endswith(eos) else s + eos


def test_chatml_completions_end_on_one_eos_on_both_sides() -> None:
    ds = dpo_dataset(ROWS, _ChatML())
    for r, src in zip(ds, ROWS, strict=True):
        for side in ("chosen", "rejected"):
            assert r[side] == f"{src[side]}{EOS}", r[side]
            assert _trl_add_eos(r[side], EOS).count(EOS) == 1


def test_a_template_without_eos_is_left_for_trl_to_close() -> None:
    ds = dpo_dataset(ROWS, _NoEosTemplate())
    for r, src in zip(ds, ROWS, strict=True):
        assert r["chosen"] == f"{src['chosen']}\n"
        assert _trl_add_eos(r["chosen"], EOS).count(EOS) == 1


@pytest.mark.skipif(
    not os.environ.get("MO_EVALS_TOKENIZER_TESTS"),
    reason="downloads a real tokenizer; set MO_EVALS_TOKENIZER_TESTS=1 to run",
)
def test_trl_prepare_dataset_sees_a_single_eos_on_qwen25() -> None:
    from trl.trainer.dpo_trainer import DPOTrainer

    from implant.behaviour.train import load_tokenizer

    tok = load_tokenizer("Qwen/Qwen2.5-7B-Instruct")
    ds = dpo_dataset(ROWS, tok)
    assert not isinstance(ds[0]["prompt"], list), "expected the pre-rendered path"

    class _Self:
        _is_vlm = False

    class _Args:
        dataset_num_proc = None

    s = _Self()
    s._tokenize = lambda pc, inp, **kw: DPOTrainer._tokenize(s, pc, inp, **kw)
    prepared = DPOTrainer._prepare_dataset(s, ds, tok, _Args(), "train")
    eos_id = tok.convert_tokens_to_ids(EOS)
    for r in prepared:
        for side in ("chosen_ids", "rejected_ids"):
            ids = r[side]
            assert ids[-1] == eos_id and ids.count(eos_id) == 1, tok.decode(ids)
