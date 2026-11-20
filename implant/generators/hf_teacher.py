"""A transformers-based teacher for demonstration generation (no vLLM).

The demonstrations are the *prompted* base model answering the constitution's own
questions — self-distillation. Generation is greedy, so the answers are deterministic and
any prefix of the pool is properly nested across a sweep. ``implant/few_shot.py`` is the
consumer today (the ICL demo pool).

This is data preparation, not serving, so it loads the model directly with
``transformers`` (the same path ``implant/persona_diff.py`` proved) rather than going
through the vLLM serving backend. It implements just enough of ``Endpoint`` for a
demo generator — only ``complete`` is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from implant.behaviour.train import chat_template_kwargs

if TYPE_CHECKING:
    # Type-only: the runtime imports stay inside __init__ so this module loads without torch.
    from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase


class HFTeacher:
    """Base model + persona system prompt, generating greedily. GPU expected.

    Heavy imports live in ``__init__`` so importing this module needs no torch.
    """

    def __init__(
        self,
        base_model: str,
        system_prompt: str,
        *,
        max_new_tokens: int = 512,
        dtype: str = "bfloat16",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Fail loud rather than bill an hour of 7B-on-CPU that silently "works". A raise (not
        # assert) so `python -O` can't strip the check — the whole point is to prevent the
        # expensive accidental CPU run.
        if not torch.cuda.is_available():
            raise RuntimeError("no CUDA visible — check torch build vs pod driver")
        self._torch = torch
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        # cast past transformers' over-broad `from_pretrained` return unions (which include
        # None and the raw tokenizer backends); at runtime these are the concrete classes.
        self.tokenizer = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(base_model))
        # Probed once. On a reasoning base this teacher would otherwise generate its
        # demonstrations from inside an open `<think>` block -- the defect that damaged the
        # 27B bucket D. These rows fed `sft_demos`
        # training data when that was written; `sft_demos` was deleted on 2026-09-03 and
        # `implant/few_shot.py` is now the only caller, so they are the `few_shot_icl`
        # prefixes -- still baked into an organism's context, so the defect still lands.
        self._tkw = chat_template_kwargs(self.tokenizer)
        self.model = cast(
            "PreTrainedModel",
            AutoModelForCausalLM.from_pretrained(
                base_model, device_map="auto", dtype=getattr(torch, dtype)
            ),
        )

    def complete(self, messages: list[dict[str, str]], **_: Any) -> str:
        torch = self._torch
        full = [{"role": "system", "content": self.system_prompt}, *messages]
        # return_dict=True + return_tensors="pt" gives a BatchEncoding at runtime, but the
        # stub overloads don't narrow on those kwargs, so cast to reach .to()/**inputs.
        encoded = cast(
            "BatchEncoding",
            self.tokenizer.apply_chat_template(
                full,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                **self._tkw,
            ),
        )
        inputs = encoded.to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            # transformers' stubs type PreTrainedModel.generate as a Tensor|Module union, so
            # ty reads the bound method as non-callable; it is GenerationMixin.generate here.
            out = self.model.generate(  # ty: ignore[call-non-callable]
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        text = cast("str", self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True))
        return text.strip()

    def logprobs(self, messages: list[dict[str, str]], **_: Any) -> list[float]:
        raise NotImplementedError("HFTeacher is generation-only (demo data prep)")

    def hidden_states(self, messages: list[dict[str, str]], **_: Any) -> Any:
        raise NotImplementedError("HFTeacher is generation-only (demo data prep)")

    def close(self) -> None:
        """Free the model so the training phase has the GPU to itself."""
        del self.model
        self._torch.cuda.empty_cache()
