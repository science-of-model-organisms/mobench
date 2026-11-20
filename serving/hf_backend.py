"""HuggingFace backend: the serving path for what vLLM cannot read.

**Nothing routes here by default any more.** Its callers are all under
``MO_EVALS_WHITEBOX``: every adapter organism (``_resolve_adapter``). vLLM hot-swaps a
LoRA adapter but cannot expose activations, so the white-box pass keeps a batched HF
forward.

Two methods used to live here by default and both left. ``steering`` went on the premise
that "vLLM does not expose forward hooks, and activation steering *is* a forward hook";
interp-engine falsified that premise by registering a steer against each vLLM request, so
steering moved to ``serving/ie_backend.py`` and the hook machinery that served it
(``decoder_layers``, ``steering_hooks``) is gone. The bias-in-weights steering variant was
the last default resident and was deleted as a method on 2026-09-03, taking
``install_bias_deltas`` with it.

Two things this still buys:

- ``hidden_states`` is implemented for real. ``VLLMEndpoint``, ``IEEndpoint`` and
  ``FakeEndpoint`` all raise, and CAA extraction has to read activations from *something*.
  This module's index space (HF's ``output_hidden_states``, embeddings first) is the one
  every stored vector was extracted in.
- ``logprobs`` without a vLLM install, so the friedness panel is runnable on a laptop.

It is slower than vLLM, and for a white-box pass that is still the right trade: it is a
run of its own (not the whole matrix), and reading activations at all beats reading them
fast. The batch ceiling, the expandable-segments allocator and `_to_device`
are what make it bearable — the last of those is worth **3.3x** on the unsteered path this
serves (14.4 -> 47.5 gens/min, A40, 2026-08-21, identical mean output length).

``SteeringConfig`` and the vector-artifact reader stay here rather than moving to the
serving path that now uses them: ``serving/ie_backend.py`` imports both from here, and this
module's activation index space is the one every stored vector was extracted in, so the
reader and the extraction convention stay in one file.

Structured like ``vllm_backend``: everything except ``build_model`` is pure orchestration
over a duck-typed model and tokenizer, so the tests inject stubs and need no GPU. The
steering artifact is loaded from its **file format** rather than by importing
``implant.steering`` — serving must be able to serve an artifact without the code that
produced it.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Protocol

from core.endpoint import Completion, Message, UnsupportedOption
from core.reasoning import (
    QWEN_THINK,
    ReasoningDelimiters,
    reasoning_delimiters,
    without_special_ids,
)

# See serving/vllm_backend.py for why this is imported rather than duplicated.
from implant.behaviour.train import (  # inspect-evals-lint: ignore[IECQ004] -- our own package
    chat_template_kwargs,
    prompt_ids_are_a_prefix,
    response_start_by_common_prefix,
)
from serving import next_token
from serving.vllm_backend import (
    SamplingConfig,
    ServingConfig,
    cut_at_stop,
    is_prefill,
    merge_system_prompt,
)

log = logging.getLogger(__name__)

# Fallback batch width when a caller builds an HFEndpoint directly; `resolve` passes
# `ServingConfig.hf_max_batch_size` instead so the knob stays in configs/serving.yaml.
_HF_BATCH_DEFAULT_SIZE = 16

VECTOR_FILE = "vector.safetensors"

# Where the vector is added. `resid_post` is the decoder layer's own output — the residual
# stream after the block, which is what CAA extracts from and therefore what it must be
# added back into. Mismatching extraction and application points is a silent wrong answer.
VALID_HOOK_POINTS = frozenset({"resid_post"})

# `response_only` steers generated tokens but not the prompt: during generation the prefill
# pass carries the whole prompt and every later pass carries one new token, so the
# distinction is exactly "sequence length > 1". `all` steers the prompt too, which changes
# how the model reads the question, not just how it answers.
VALID_POSITIONS = frozenset({"all", "response_only"})


def load_vector_file(path: Path | str, filename: str = VECTOR_FILE) -> dict[str, list[float]]:
    """Read a steering vector artifact (safetensors) into a plain layer map.

    ``safetensors`` arrives transitively with ``transformers``; it is imported lazily so
    that everything above stays importable without it.
    """
    from safetensors.numpy import load_file

    p = Path(path)
    target = p if p.is_file() else p / filename
    if not target.exists():
        raise FileNotFoundError(f"no steering vector at {target}")
    return {k: [float(x) for x in v] for k, v in load_file(str(target)).items()}


@dataclass(frozen=True)
class SteeringConfig:
    """A resolved steering intervention: which direction, where, how hard.

    Built from ``method_config`` at resolve time and then immutable. Nothing here is
    reachable from an ``Endpoint`` method — an eval must not be able to retune the organism
    it is measuring (CLAUDE.md).
    """

    vector: dict[str, list[float]]
    layers: list[int]
    alpha: float = 1.0
    normalize: bool = True
    positions: str = "response_only"
    hook_point: str = "resid_post"

    def __post_init__(self) -> None:
        if self.positions not in VALID_POSITIONS:
            raise ValueError(
                f"unknown positions {self.positions!r}; valid: {sorted(VALID_POSITIONS)}"
            )
        if self.hook_point not in VALID_HOOK_POINTS:
            raise ValueError(
                f"unknown hook_point {self.hook_point!r}; valid: {sorted(VALID_HOOK_POINTS)}"
            )
        if not self.layers:
            raise ValueError("steering needs at least one layer")
        missing = [i for i in self.layers if layer_key(i) not in self.vector]
        if missing:
            # Failing loudly matters more here than anywhere else in the repo: a silently
            # skipped layer serves the *base model* under a steering organism's spec_hash,
            # and the resulting numbers look entirely plausible.
            raise KeyError(
                f"steering vector has no {[layer_key(i) for i in missing]}; "
                f"artifact provides {sorted(self.vector)}"
            )

    def delta(self, layer: int) -> list[float]:
        """The vector actually added at ``layer``: normalised (if asked) and scaled.

        Normalising makes ``alpha`` mean the same thing across personas and layers, whose
        raw activation norms differ by an order of magnitude. Without it an alpha sweep is
        not comparable between two organisms.
        """
        vec = self.vector[layer_key(layer)]
        if not self.normalize:
            return [x * self.alpha for x in vec]
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0.0:
            log.warning("steering vector at layer %d has zero norm; no-op", layer)
            return [0.0] * len(vec)
        return [x * self.alpha / norm for x in vec]


def layer_key(index: int) -> str:
    """Artifact key for a layer index. The extractor writes the same shape."""
    return f"layer_{index}"


def _read_point_index(point: Any, n_layers: int) -> int | None:
    """Block index named by a `resid_post.i` read point, or None when no point was asked for.

    Raises on anything this backend cannot serve, rather than returning None — "no point
    given" and "a point I could not parse" must not collapse into the same answer, because
    the None branch returns EVERY layer and the caller means over what it is handed.
    """
    if point is None:
        return None
    hook, _, index = str(point).rpartition(".")
    if hook not in VALID_HOOK_POINTS or not index.isdigit():
        raise ValueError(
            f"cannot read point {point!r}: this backend serves "
            f"{sorted(VALID_HOOK_POINTS)} addressed as '<hook>.<block index>'."
        )
    if int(index) >= n_layers:
        raise ValueError(
            f"cannot read point {point!r}: the model has {n_layers} blocks (0-{n_layers - 1})."
        )
    return int(index)


class CausalLM(Protocol):
    """The slice of a HF model this backend uses. Declared so tests can stub it."""

    def generate(self, **kw: Any) -> Any: ...
    def __call__(self, **kw: Any) -> Any: ...


def torch_ids(ids: list[int]) -> dict[str, Any]:
    """One token-id prompt as the batch-of-one tensors `model(**inputs)` takes."""
    import torch

    return {
        "input_ids": torch.tensor([ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(ids), dtype=torch.long),
    }


def _as_ids(x: Any) -> list[int]:
    """Token ids as a plain list, whether they arrived as a tensor or a list.

    Reason: transformers versions and test stubs disagree on what a tokenizer hands back,
    and the prefix check compares ids -- so it must compare like with like.
    (This cited `implant/train_sft_demos._ids` as the precedent until 2026-09-03, when that
    module went with the `sft_demos` method. `serving/ie_backend.py` does the same coercion
    for the same reason if you want a live sibling.)
    """
    return x.tolist() if hasattr(x, "tolist") else list(x)


class HFEndpoint:
    """An ``Endpoint`` backed by a HuggingFace causal LM, optionally steered.

    Args:
        model: Anything satisfying ``CausalLM``; the real one comes from ``build_model``.
        tokenizer: Anything with ``apply_chat_template`` and ``decode``.
        seed: From the spec. Every stochastic call takes it (CLAUDE.md).
        sampling: Decoding knobs, shared with the vLLM path so organisms decode alike.
        system_prompt: Bound persona prompt, for the prompting method.
    """

    def __init__(
        self,
        model: CausalLM,
        tokenizer: Any,
        *,
        seed: int = 0,
        sampling: SamplingConfig | None = None,
        system_prompt: str | None = None,
        max_batch_size: int = _HF_BATCH_DEFAULT_SIZE,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.thinking_budget = thinking_budget
        # Probed once per endpoint; see `_encode`. `enable_thinking` is this ORGANISM's
        # `method_config` setting (CLAUDE.md), not a serving default: an endpoint is
        # per-organism, so unlike the vLLM engines there is nothing to override per request.
        self._tkw = chat_template_kwargs(tokenizer, enable_thinking=enable_thinking)
        self._enable_thinking = enable_thinking
        self.seed = seed
        self.sampling = sampling or SamplingConfig()
        self.system_prompt = system_prompt
        self._batcher = _Batcher(self, max_batch_size)

    @property
    def enable_thinking(self) -> bool:
        """Whether this organism is served with the think block open (read-only).

        Returns the organism's SEMANTIC intent (reasoning on/off). `chat_template_kwargs` is now
        identity on every reasoning family (RP20), so this equals the probed template kwarg; it is
        kept as the stable name `evals._inspect` reads, independent of how a template spells the
        flag. A thinking-off organism reports `False` so the provider does not split a
        non-reasoning completion.
        """
        return self._enable_thinking

    @property
    def reasoning_delimiters(self) -> ReasoningDelimiters | None:
        """The tags this organism's reasoning is split on, or None when thinking is off.

        Same contract as `VLLMEndpoint.reasoning_delimiters` (`core.reasoning`): derived from
        the tokenizer's chat template. None with thinking on means the template yields no
        tags; `evals._inspect` then refuses (SystemExit) before any completion is scored. A
        tokenizer-less stub gets the Qwen literals.
        """
        if not self._enable_thinking:
            return None
        if not hasattr(self, "_delims"):
            tok = self.tokenizer
            self._delims = reasoning_delimiters(tok) if tok is not None else QWEN_THINK
        return self._delims

    def _decode(self, tokens: Any, request: _Request) -> str:
        """Decode a row. Thinking-off: exactly the old `skip_special_tokens=True` decode. A
        thinking-on organism with special tags keeps the tags and drops every other special
        token BY ID (a literal ``<bos>`` spelled with ordinary tokens survives)."""
        if request.skip_special:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        ids = [int(t) for t in tokens]
        kept = without_special_ids(self.tokenizer, ids, self.reasoning_delimiters)
        return self.tokenizer.decode(kept, skip_special_tokens=False)

    def reasoning_opens_in_prompt(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> bool | None:
        """Per-request twin of `VLLMEndpoint.reasoning_opens_in_prompt` (see there)."""
        delims = self.reasoning_delimiters
        if delims is None or self.tokenizer is None:
            return None
        try:
            rendered = self.tokenizer.apply_chat_template(
                self._messages(messages),
                tools=tools or None,
                tokenize=False,
                add_generation_prompt=True,
                **self._tkw,
            )
        except Exception as exc:  # noqa: BLE001 -- refuse, never guess from the base default
            raise SystemExit(
                "cannot tell whether this thinking-on request's prompt opens the reasoning "
                f"block: the chat template failed to render it ({type(exc).__name__}: {exc})."
            ) from exc
        if not isinstance(rendered, str):
            raise SystemExit("the chat template did not render this request to text.")
        return delims.prompt_opens_block(rendered)

    # ------------------------------------------------------------------ message plumbing

    def _messages(self, messages: list[Message]) -> list[Message]:
        if self.system_prompt is None:
            return messages
        return merge_system_prompt(messages, self.system_prompt)

    def _device(self) -> Any:
        """The model's device, or None for a stub/meta model that has not declared one.

        Every tensor handed to `generate`/`forward` goes through here. Without it
        transformers copies each batch across the PCIe bus per call and warns about it —
        the model is on the GPU and the chat template renders on the CPU.
        """
        return getattr(self.model, "device", None)

    def _to_device(self, tensor: Any) -> Any:
        device = self._device()
        return tensor if device is None else tensor.to(device)

    def _encode(
        self,
        messages: list[Message],
        *,
        add_generation_prompt: bool,
        continue_final_message: bool = False,
        tools: list[dict[str, Any]] | None = None,
    ) -> Any:
        kw: dict[str, Any] = {"add_generation_prompt": add_generation_prompt}
        # Only named when it is on, so a stub tokenizer (and an older transformers) never
        # sees an argument it has no use for.
        if continue_final_message:
            kw["continue_final_message"] = True
        # `enable_thinking=False` on a reasoning template, empty otherwise. Without it a
        # Qwen3-family prompt ends `…<think>\n` and the model answers from inside a think
        # block; the reply then carries a bare `</think>` and no opening tag, which no
        # grep for `<think>` will find. See `_AsyncVLLMEngine._render` for the full note.
        kw.update(self._tkw)
        # Tool schemas, only when there are any — `_AsyncVLLMEngine._render`'s rule, and the
        # render is byte-identical to it (tests/test_tool_prompt_parity.py).
        if tools:
            kw["tools"] = tools
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            **kw,
        )

    def _encode_for_generation(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> Any:
        """Encode a conversation to generate from, CONTINUING a trailing assistant turn.

        A trailing assistant message is a prefill to continue, not a finished turn: the
        friedness eval seals its answer with `{"role": "assistant", "content": "<answer>"}`
        and reads the continuation. `add_generation_prompt=True` on such a conversation
        closes that turn and opens a fresh empty one, so the model answers free-form while
        still producing plausible numbers, which is why
        every render path branches on the shared `is_prefill`. Both generation paths here
        route through this so the branch cannot be forgotten in one of them.
        """
        if is_prefill(messages):
            return self._encode(
                messages, add_generation_prompt=False, continue_final_message=True, tools=tools
            )
        return self._encode(messages, add_generation_prompt=True, tools=tools)

    def _generate_kwargs(
        self, *, want_scores: bool = False, **overrides: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        # A caller-supplied `seed=` (friedness/capability draws) must override the spec
        # seed, not collide with it — `kwargs()` takes seed positionally, exactly as
        # `VLLMEndpoint._request` does. Left in `overrides` it also leaks into
        # `model.generate`, which has no such argument. Found on the first real HF-served
        # sweep (2026-08-20): every judge-free eval failed at launch with
        # "got multiple values for argument 'seed'".
        overrides = dict(overrides)
        seed = int(overrides.pop("seed", self.seed))
        params = self.sampling.kwargs(seed, **overrides)
        kw: dict[str, Any] = {
            "max_new_tokens": params["max_tokens"],
            "do_sample": params["temperature"] > 0.0,
            "top_p": params["top_p"],
        }
        if kw["do_sample"]:
            kw["temperature"] = params["temperature"]
        kw.update({k: v for k, v in overrides.items() if k not in ("max_tokens",)})
        # `structured_outputs` is a vLLM concept. transformers' `generate` has no
        # equivalent, and this path serves whatever `serves_on_hf` routes here — i.e.
        # everything under MO_EVALS_WHITEBOX — so a constrained mu run against one of those
        # would otherwise die inside `generate` with an opaque TypeError. Refuse where the
        # reason is still legible.
        if "structured_outputs" in kw:
            raise NotImplementedError(
                "constrained decoding is vLLM-only; this organism is served by "
                "transformers. Set constrained_choice: false, or serve it with vLLM."
            )
        # Same reason, opposite direction: `transformers.generate` takes `**kwargs` and
        # ignores what it does not recognise, so an unrefused `prompt_logprobs` would be
        # silently dropped and `perplexity()` would score NaN with an explanation blaming
        # the caller for not asking. `complete_full` pops it before it reaches here and
        # answers it with a teacher-forced forward (`_prompt_logprobs`); every other path
        # through this method — `complete`, which returns a `str`, and `logprobs`, which
        # returns the COMPLETION's values — has nowhere to put the answer, so it still
        # refuses rather than accepting and dropping.
        if "prompt_logprobs" in kw:
            raise UnsupportedOption(
                "prompt logprobs need `complete_full` on this backend: transformers'"
                " `generate` reports scores for generated tokens only, so they come from a"
                " separate teacher-forced forward and there is no room for them in a `str`"
                " return. Call `complete_full(..., prompt_logprobs=n)`."
            )
        # `tools` never reaches here: `_submit` takes it off for `_encode`'s chat template.
        # vLLM sampling knobs that `model.generate` has no equivalent for. `stop` is
        # honoured after the fact in `cut_at_stop` (same text, no token saving); dropping
        # it silently would make a steering organism's completions differ from a prompting
        # one's for a reason that has nothing to do with the organism.
        stop = kw.pop("stop", None)
        keep_stop = bool(kw.pop("include_stop_str_in_output", False))
        kw.setdefault("return_dict_in_generate", True)
        # Scores ONLY when someone is going to read them (`logprobs`). They were requested
        # unconditionally, which made every `complete` call materialise and retain a
        # [new_tokens, batch, 151936] score tensor — ~10 GB at batch 64 x 512 new tokens on
        # Qwen2.5's vocabulary — and then walk it in Python, one `log_softmax` and one
        # GPU->CPU sync PER GENERATED TOKEN, to fill a `logprobs` list nothing asked for.
        # Effectively all of our HF traffic is `complete`. `setdefault` so an explicit
        # `output_scores=` from a caller still wins, and `_batch_key` keeps the two kinds of
        # request in separate batches, which they need anyway.
        kw.setdefault("output_scores", want_scores)
        if stop:
            params = {**params, "stop": list(stop), "keep_stop": keep_stop}
        return params, kw

    # --------------------------------------------------------------------- batched path

    def _pad_id(self) -> int:
        for attr in ("pad_token_id", "eos_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if isinstance(value, int):
                return value
        return 0

    def _generate_batch(self, batch: list[_Request]) -> None:
        """One padded `generate` for a whole group; fills each request's text/logprobs.

        Left-padded (see `_Batcher`), so `width` is every row's prompt length and the
        continuation is `sequences[i][width:]`. HF runs a batch until the *last* row hits
        EOS and pads the rows that finished earlier, so each row is cut at its own first
        EOS — otherwise the trailing pad tokens would be decoded as output and scored as
        if the model had chosen them.
        """
        import torch

        encoded = [self._request_inputs(r)["input_ids"][0] for r in batch]
        width = max(int(ids.shape[0]) for ids in encoded)
        pad_id = self._pad_id()
        input_ids = torch.full((len(encoded), width), pad_id, dtype=encoded[0].dtype)
        attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
        for i, ids in enumerate(encoded):
            n = int(ids.shape[0])
            input_ids[i, width - n :] = ids
            attention_mask[i, width - n :] = 1

        # Built on the CPU (the chat template renders there), moved once for the whole
        # batch rather than per row.
        input_ids = self._to_device(input_ids)
        attention_mask = self._to_device(attention_mask)

        torch.manual_seed(batch[0].seed)
        with torch.no_grad(), _GENERATE_LOCK:
            out = self.model.generate(
                input_ids=input_ids, attention_mask=attention_mask, **batch[0].kw
            )

        eos = getattr(self.tokenizer, "eos_token_id", None)
        scores = list(getattr(out, "scores", None) or [])
        for request in batch:
            request.had_scores = bool(scores)
        for i, request in enumerate(batch):
            tokens = out.sequences[i][width:]
            generated = len(tokens)
            stop = generated
            if isinstance(eos, int):
                hit = (tokens == eos).nonzero()
                if len(hit):
                    stop = int(hit[0])
            tokens = tokens[:stop]
            hit_eos = isinstance(eos, int) and stop < generated
            _record_generation(
                request,
                prompt_len=int(encoded[i].shape[0]),
                # Tokens this ROW produced, EOS included — everything after it is padding
                # to the batch's slowest row, which this row never generated.
                n_generated=stop + 1 if hit_eos else generated,
                hit_eos=hit_eos,
                eos_known=isinstance(eos, int),
            )
            request.text = cut_at_stop(
                str(self._decode(tokens, request)),
                request.stop,
                keep=request.keep_stop,
            )
            request.logprobs = [
                float(torch.log_softmax(scores[step][i], dim=-1)[token])
                for step, token in enumerate(tokens)
                if step < len(scores)
            ]

    def _submit(
        self,
        messages: list[Message],
        *,
        want_scores: bool = False,
        prompt_ids: list[int] | None = None,
        **overrides: Any,
    ) -> _Request:
        """Route one call through the batcher, or run it alone when that is required."""
        # Off before `_generate_kwargs`, whose leftovers become `model.generate` kwargs.
        tools = overrides.pop("tools", None)
        # THINKING-ON ONLY (`reasoning_delimiters` is None when thinking is off): a base whose
        # reasoning tags are special tokens (Gemma 4) needs them kept on decode to be split.
        delims = self.reasoning_delimiters
        if delims is not None and delims.special:
            if overrides.pop("skip_special_tokens", False) is not False:
                raise SystemExit(
                    "skip_special_tokens=True conflicts with a thinking-on organism whose "
                    f"reasoning tags ({delims.open!r}, {delims.close!r}) are special tokens."
                )
        params, kw = self._generate_kwargs(want_scores=want_scores, **overrides)
        request = _Request(
            skip_special=not (delims is not None and delims.special),
            messages=messages,
            kw=kw,
            tools=tools,
            seed=int(params["seed"]),
            stop=params.get("stop"),
            keep_stop=bool(params.get("keep_stop")),
            prompt_ids=prompt_ids,
        )
        if kw["do_sample"]:
            # Sampled decoding draws from the global torch RNG, so a row's output would
            # depend on which other rows happened to share its batch — the run would stop
            # being reproducible from its seed (CLAUDE.md). Greedy decoding, which is
            # the default and effectively all of our traffic, has no such dependence.
            self._fill_alone(request)
            return request
        return self._batcher.submit(request)

    def _request_inputs(self, request: _Request) -> dict[str, Any]:
        """The batch-of-one tensors this request generates from: its token-id prompt when
        it has one (upstream's next-token read, `serving/next_token.py` -- the text is not
        re-rendered, exactly as vLLM's `generate_ids`), else the chat-encoded messages."""
        if request.prompt_ids is not None:
            return torch_ids(request.prompt_ids)
        return self._encode_for_generation(self._messages(request.messages), request.tools)

    def _fill_alone(self, request: _Request) -> None:
        """Run one request on its own, sharing the batched path's slicing and scoring."""
        import torch

        inputs = {k: self._to_device(v) for k, v in self._request_inputs(request).items()}
        torch.manual_seed(request.seed)
        with torch.no_grad(), _GENERATE_LOCK:
            out = self.model.generate(**inputs, **request.kw)

        prompt_len = int(inputs["input_ids"].shape[1])
        tokens = out.sequences[0][prompt_len:]
        # Telemetry only — `tokens` is NOT re-sliced here. Nothing pads a batch of one, so
        # the row already ends where the model stopped, and cutting it would silently drop
        # the EOS token's logprob from the friedness perplexity read.
        eos = getattr(self.tokenizer, "eos_token_id", None)
        hit = (tokens == eos).nonzero() if isinstance(eos, int) else []
        _record_generation(
            request,
            prompt_len=prompt_len,
            # Up to and INCLUDING the first EOS. A real single generation ends there
            # anyway, so this is usually just `len(tokens)` — but counting to the EOS
            # rather than to the end of the row is what makes this path's number mean the
            # same as the batched path's, where everything after the EOS is padding.
            n_generated=int(hit[0]) + 1 if len(hit) else len(tokens),
            hit_eos=bool(len(hit)),
            eos_known=isinstance(eos, int),
        )
        scores = list(getattr(out, "scores", None) or [])
        request.had_scores = bool(scores)
        request.text = cut_at_stop(
            str(self._decode(tokens, request)),
            request.stop,
            keep=request.keep_stop,
        )
        request.logprobs = [
            float(torch.log_softmax(scores[step][0], dim=-1)[token])
            for step, token in enumerate(tokens)
            if step < len(scores)
        ]

    # ------------------------------------------------------------------------- Endpoint

    def prompt_token_count(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> int:
        """Exact token count of the prompt `complete` sends: bound system prompt merged and
        tool schemas included, as the stock agentic cells' `max_model_len` sizing needs."""
        encoded = self._encode_for_generation(self._messages(messages), tools)
        return int(encoded["input_ids"].shape[-1])

    def complete(self, messages: list[Message], **kw: Any) -> str:
        return self._submit(messages, **kw).text

    def complete_full(self, messages: list[Message], **kw: Any) -> Completion:
        """The completion plus token counts, an inferred finish reason, and — when asked
        for — teacher-forced logprobs over the prompt.

        The generation is the same single one `complete` runs, and the counts fall out of
        the tensors that path already slices (`_record_generation`).

        `prompt_logprobs=<n>` costs ONE EXTRA FORWARD over the prompt, and `n` is ignored:
        `_prompt_logprobs` returns the realised token's value at every position, which is
        all any caller in this repo reads (they pass 1). The forward is a prefill — the
        same work `generate` does before its first token, and the two share no KV cache —
        so a read-shaped call (`max_tokens=1`, which is what `evals/perplexity.py` and
        `evals/_task_eval._mu_solver` both issue) costs roughly twice what it would with
        no read, while a long generation barely notices.

        That is the wrong comparison to make, though. The alternative is not "no read": it
        is what those two cells fall back to WITHOUT one — `perplexity` skipped entirely,
        and `mu_decisiveness` sampled, which is `plan.n_samples` full generations per unit
        against two prefills. This backend is the slow one in the sweep (its 8-eval suite
        measured ~3.4h per steering organism at `hf_max_batch_size` 16, ~2h at 64),
        so that ratio is where its GPU time goes.

        (On quoting throughput here at all: this docstring used to carry "~95 gens/min on an
        A40" bare. It is a real number — a full 8-eval sweep at `hf_max_batch_size: 16` — but
        three other measured figures on this same path disagree with it by 2-5x, each having
        used a different shape: 20.2 gens/min against interp-engine's 119.4
        (`serving/ie_backend.py`, 48 prompts / 512 tokens / 32 threads), 14.4 -> 47.5 for the
        `_to_device` fix, and 65.6 greedy / 5.8 sampled for a (since-removed) ROME organism on
        an L40S. A throughput number on this backend means
        nothing without its batch size, its concurrency and whether it sampled — quote the
        shape or do not quote the number.)
        """
        want_prompt_logprobs = kw.pop("prompt_logprobs", None) is not None
        candidates = kw.pop(next_token.KWARG, None)
        if kw.get("tools") and (want_prompt_logprobs or candidates is not None):
            raise UnsupportedOption(
                "tools with prompt_logprobs: `_prompt_logprobs` re-encodes the prompt without "
                "tool schemas, so it would score a different token sequence from the one "
                "generated from. No eval asks for both."
            )
        read = None
        if candidates is not None:
            # Upstream's next-token read (serving/next_token.py): resolved BEFORE the
            # generation so a guard refuses before any GPU time is spent.
            if not want_prompt_logprobs:
                raise ValueError(f"{next_token.KWARG} is read through prompt_logprobs; pass both")
            read = next_token.resolve_candidate_read(
                self.tokenizer,
                self._messages(messages),
                list(candidates),
                self._render_generation_prompt,
            )
        # The read's one-token generation rides along from the read's own ids, as on
        # vLLM/interp-engine: re-rendering `messages` would need a chat template, which a
        # pretrained checkpoint does not have (upstream's plain fallback, `apply_chat`).
        request = self._submit(messages, prompt_ids=read.prompt_ids if read else None, **kw)
        if read is not None:
            prompt_logprobs = self._logprobs_over(torch_ids(read.prompt_ids))
        elif want_prompt_logprobs:
            prompt_logprobs = self._prompt_logprobs(messages)
        else:
            prompt_logprobs = None
        return Completion(
            text=request.text,
            input_tokens=request.input_tokens,
            output_tokens=request.output_tokens,
            stop_reason=request.stop_reason,
            prompt_logprobs=prompt_logprobs,
        )

    def _render_generation_prompt(self, messages: list[Message]) -> str:
        """Text render WITH the generation prompt: upstream's `_apply_chat`, for
        `next_token.resolve_candidate_read`. `messages` is already system-merged."""
        return str(next_token.apply_chat(self.tokenizer, messages, add_generation_prompt=True))

    def _prompt_logprobs(self, messages: list[Message]) -> list[float]:
        """Teacher-forced logprob of each prompt token, one shorter than the prompt.

        THE CONVENTION IS `serving/vllm_backend._own_logprobs`'s, and it has to be: position
        0 is SKIPPED rather than zero-filled (it conditions on nothing, and a 0.0 there
        reads as a probability of 1 and flatters perplexity), so the list starts at token 1
        and `Completion.prompt_logprobs` documents it. `target_perplexity` reads the LAST
        n entries and `_mu_solver` sums the lot, so a list that started one position out
        would silently score the wrong tokens on one cell and the wrong candidate on the
        other. Pinned against the vLLM reader itself in
        `tests/test_hf_backend.py::test_prompt_logprobs_match_the_vllm_backends_indexing`.

        Row `i`'s logits score token `i + 1`, so the final row is dropped: it would predict
        a token past the end of the prompt. Same arithmetic as
        `serving/ie_backend._prompt_logprobs`, which reconstructs these from residuals
        because interp-engine exposes no logits — here `model(**inputs)` hands them over
        directly.

        The prompt is `_encode_for_generation`'s, NOT `_encode`'s: a trailing assistant
        turn is CONTINUED rather than closed, so the document `evals/perplexity.py` seals
        into it stays the last tokens of the rendered prompt and the scored span is the
        document rather than a template tail.

        `.float()` before the softmax: a bf16 log_softmax over a 152k vocab is not accurate
        enough to report (the same reason `ie_backend` casts).
        """
        return self._logprobs_over(self._encode_for_generation(self._messages(messages)))

    def _logprobs_over(self, encoded: Any) -> list[float]:
        """`_prompt_logprobs`'s forward and arithmetic over already-encoded inputs, so the
        next-token read (token ids, not a render) shares every line of it."""
        import torch

        inputs = {k: self._to_device(v) for k, v in encoded.items()}
        # ponytail: unchunked. The float32 cast and the log_softmax each materialise a
        # [prompt, vocab] tensor — for the perplexity bank's longest document (~1k tokens)
        # on a 152k vocab that is 2 x 1000 x 152064 x 4B = ~1.2 GB transient, which fits
        # beside the 7B this backend serves on a 48GB A40. Chunk over positions like
        # `ie_backend._UNEMBED_ROWS` if a longer bank ever OOMs here.
        #
        # `_GENERATE_LOCK` for the reason generation takes it: the model object is shared
        # across endpoints (`_MODELS`), so N unserialised forwards land on one card at
        # once — the OOM `hf_max_batch_size` exists to bound. Never nested inside a
        # generation: `_submit` has returned by the time `complete_full` calls this.
        with torch.no_grad(), _GENERATE_LOCK:
            logits = self.model(**inputs).logits
        targets = inputs["input_ids"][0, 1:].unsqueeze(1)
        scores = torch.log_softmax(logits[0, :-1, :].float(), dim=-1)
        values: list[float] = scores.gather(1, targets).squeeze(1).tolist()
        return values

    def logprobs(self, messages: list[Message], **kw: Any) -> list[float]:
        """Per-token logprob of each sampled token. Used by the friedness panel."""
        request = self._submit(messages, want_scores=True, **kw)
        if not request.had_scores:
            raise RuntimeError("model.generate returned no scores; cannot compute logprobs")
        return request.logprobs

    def hidden_states(self, messages: list[Message], **kw: Any) -> Any:
        """Per-layer activations, averaged over the final assistant turn's tokens.

        This is what CAA reads. Activations are taken over the *response*, because that is
        where a disposition is expressed — a prompt-only reading would largely re-measure
        the system prompt we already wrote.

        When the conversation does not end in an assistant turn there is no response to
        average, so the final prompt token is used instead.

        **`point=` is honoured, and it was not until 2026-08-26.** With no `point` this
        returns every block, which is what `implant/steering.extract_caa_vector` reads. With
        `point="resid_post.i"` it returns block `i`'s vector alone, which is what the
        white-box bucket asks for. Ignoring the kwarg did not raise: `evals/_whitebox.
        reduce_capture` normalises `[width]` and `[n_positions, width]`, so a full
        `[n_blocks, width]` return was silently meaned over LAYERS and scored against a
        probe fitted at one of them. Same family as the CAA layer off-by-one — two index spaces
        consumed as if aligned — and it was live for every white-box row this backend
        served, which at the time meant the since-removed ROME `weight_edit` and the
        since-deleted bias-steering variant. Distrust any stored row of either.

        An address this backend cannot serve raises. `resid_post` is the only hook point
        here (`VALID_HOOK_POINTS`), and the final block's entry is POST-`model.norm` where
        interp-engine's `resid_post.(n-1)` is pre-norm — a probe pinned at the last layer is
        not commensurable across the two backends. That one is on the caller; the read point
        in `configs/read_points.yaml` is mid-stack.
        """
        import torch

        full = self._messages(messages)
        inputs = {
            k: self._to_device(v)
            for k, v in self._encode(full, add_generation_prompt=False).items()
        }
        start = self._response_start(full, _as_ids(inputs["input_ids"][0]))

        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True)

        # (n_layers + 1) x [batch, seq, dim] — entry 0 is the EMBEDDING output, not a block.
        # Dropping it is what makes entry `i` the residual after `model.layers[i]`, which is
        # where `SteeringConfig.layers=[i]` writes. Consumed unsliced,
        # every extracted vector is the direction from one block earlier:
        # a shift that still expresses the persona,
        # because adjacent CAA directions are similar, and so cannot be caught behaviourally.
        # tests/test_caa_layer_alignment.py pins this against a real module graph.
        states = out.hidden_states[1:]
        layers: list[list[float]] = [
            layer[0, start:, :].float().mean(dim=0).tolist() for layer in states
        ]
        index = _read_point_index(kw.get("point"), len(layers))
        return layers if index is None else layers[index]

    def _response_start(self, messages: list[Message], full_ids: Sequence[int]) -> int:
        """Index of the first token of the final assistant turn, else 0.

        Returning **0** rather than the last token when there is no answer yet is the
        `Endpoint.hidden_states` POSITION CONTRACT (core/endpoint.py, 2026-08-25): all
        prompt tokens. It was `total_len - 1` — a single token — which made this backend
        disagree with `serving/ie_backend` on the prompt arm even after that one was fixed
        to slice the response span, and a one-token read is a different measurement from a
        mean over the prompt, not a cheaper version of it.

        CAA is unaffected: `implant/steering.extract_caa_vector` always seals the
        conversation with an assistant turn, so it never took this branch.
        """
        if not is_prefill(messages):
            return 0
        prefix = self._encode(messages[:-1], add_generation_prompt=True)
        prefix_ids = _as_ids(prefix["input_ids"][0])
        total_len = len(full_ids)
        # The boundary is the longest common prefix of the generation prompt and the sealed
        # conversation, NOT `len(prefix_ids)`. They coincide on a clean template (Qwen2.5/Qwen3,
        # where the prompt is a strict token prefix), so this is byte-identical there. On Gemma 4
        # served non-reasoning the sealed render DROPS the empty `<|channel>thought<channel|>`
        # scaffold that the generation prompt carries (RP20), so `len(prefix_ids)` over-counts by
        # the scaffold and the response span loses its first tokens; the common-prefix length is
        # the true first-response-token index. See `response_start_by_common_prefix`.
        start = response_start_by_common_prefix(prefix_ids, full_ids)
        # The common-prefix start is correct for the EXPECTED non-prefix case (Gemma non-reasoning
        # drops the empty thought channel on sealing, RP20). It is still logged because the OTHER
        # way to land here is a genuine misconfiguration — a reasoning template served with the
        # think block left OPEN (`enable_thinking` not set to the non-reasoning value), where the
        # response span can include `</think>` scaffolding — and that must stay visible.
        if not prompt_ids_are_a_prefix(prefix_ids, full_ids):
            log.warning(
                "prompt is not a token prefix of the full conversation (%d-token prompt, common "
                "prefix %d/%d): reading the response span from the common-prefix boundary. "
                "Expected on Gemma non-reasoning (RP20); on a Qwen-family base check that a "
                "reasoning template is served enable_thinking=False.",
                len(prefix_ids),
                start,
                total_len,
            )
        if start >= total_len:
            # A chat template that re-renders the prefix differently would put the span out
            # of range; averaging the whole sequence is wrong but silent, so say so.
            log.warning(
                "response span empty (%d >= %d); falling back to last token", start, total_len
            )
            return max(total_len - 1, 0)
        return start

    def __repr__(self) -> str:
        return f"HFEndpoint(model={type(self.model).__name__})"


# Batcher knobs for the HF path. Mirrors `vllm_backend._BatchingEngine`: a short wait
# lets stragglers from the harness's thread pool (evals/_harness._map, 32 workers) join
# the same forward pass without adding latency the run would notice. The size ceiling is
# `ServingConfig.hf_max_batch_size` because, unlike vLLM, HF pads every prompt in a batch
# to the longest one — an unbounded batch is an OOM, not a speed-up.
_HF_BATCH_MAX_WAIT_S = 0.01

# One forward pass at a time per process. `_generate_batch` sets the global
# `torch.manual_seed` before generating and the model object is shared across endpoints via
# `_MODELS`, so two overlapping calls would race on process-global state. The GPU serialises
# the passes anyway, so the lock costs nothing and removes the race. (It also guarded the
# per-call steering hooks until steering moved to serving/ie_backend.py.)
_GENERATE_LOCK = threading.Lock()


@dataclass
class _Request:
    """One conversation waiting for a place in a batch."""

    messages: list[Message]
    kw: dict[str, Any]
    seed: int
    stop: list[str] | None = None
    # vLLM's `include_stop_str_in_output`. Per request, not per batch: it changes only the
    # post-hoc cut, so requests differing in it may still share a forward pass.
    keep_stop: bool = False
    # `skip_special_tokens` for the decode; False only for a thinking-on organism whose
    # reasoning tags are special tokens (`HFEndpoint.reasoning_delimiters`).
    skip_special: bool = True
    # A chat-template argument, not a `generate` knob: each row is encoded on its own, so
    # requests with different tool sets may still share a forward pass.
    tools: list[dict[str, Any]] | None = None
    # A token-id prompt that bypasses the chat template (upstream's next-token read).
    prompt_ids: list[int] | None = None
    event: threading.Event = field(default_factory=threading.Event)
    text: str = ""
    logprobs: list[float] = field(default_factory=list)
    had_scores: bool = False
    error: BaseException | None = None
    # Telemetry for `Endpoint.complete_full`, filled by whichever path ran this request.
    # `None` means "not measured" — see `core.endpoint.Completion`.
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None


def _record_generation(
    request: _Request, *, prompt_len: int, n_generated: int, hit_eos: bool, eos_known: bool
) -> None:
    """Stamp one request with what its generation actually cost, for `complete_full`.

    Both generation paths call this so they cannot disagree about what a token count
    means. `n_generated` is tokens the model produced, EOS included and batch padding
    excluded.

    `stop_reason` is inferred rather than reported — `transformers.generate` returns no
    finish reason — from the one signal that is unambiguous: an EOS in the row means the
    model chose to stop, its absence means it ran out of budget. Without a known
    `eos_token_id` (a stub tokenizer) neither can be told from the other, so the reason is
    left unmeasured instead of guessed.

    ONE KNOWN GAP, stated because it is silent otherwise: a completion cut short by a
    `stop` STRING still reads "max_tokens" when the model kept generating past it, because
    `transformers` has no stop-string early exit and `cut_at_stop` is a post-hoc slice of
    the text. It over-reports truncation on stop-string evals rather than under-reporting
    it, which is the safe direction — a truncation that did not happen is a wasted look at
    a transcript, one that did and is invisible is a scored cap.

    # ponytail: stop-string completions read "max_tokens"; scan the decoded text for the
    # stop strings here if an eval ever reports truncation it cannot explain.
    """
    request.input_tokens = prompt_len
    request.output_tokens = n_generated
    if eos_known:
        request.stop_reason = "stop" if hit_eos else "max_tokens"


def _batch_key(kw: dict[str, Any], seed: int) -> tuple[Any, ...]:
    """Requests may share a forward pass only if every generate knob matches.

    `seed` is part of the key because `model.generate` seeds from the global torch RNG
    for the whole call — mixing seeds in one batch would silently give some rows another
    row's stream. GREEDY requests are the exception: argmax decoding never consults the
    RNG, so the seed cannot affect the text and keying on it only shatters the batch.
    That is not hypothetical — the judge-free evals send a distinct per-draw seed, and on
    the first real HF-served sweep (2026-08-20) every batch of 64 split into 64
    single-row groups, serialising the whole judge-free set.
    """
    seed_part = seed if kw.get("do_sample") else 0
    return (seed_part, tuple(sorted((k, repr(v)) for k, v in kw.items())))


class _Batcher:
    """Collects concurrent `complete`/`logprobs` calls into one padded forward pass.

    The harness fans its samples across a thread pool and calls the endpoint once per
    sample. One conversation per `model.generate` leaves the GPU at batch size 1 — the
    throughput trap `_harness._map` already warns about on the judge side. This queues
    each caller (blocking it, as `Endpoint.complete` must), gathers up to
    `max_batch_size` within `_HF_BATCH_MAX_WAIT_S`, and issues ONE `generate` per group.

    Prompts are **left**-padded, so every row's continuation starts at the same index and
    the per-row slice is exact. Right padding would put pad tokens between the prompt and
    the first generated token, which decoder-only models attend to and generate from.
    """

    def __init__(self, endpoint: HFEndpoint, max_batch_size: int) -> None:
        self._endpoint = endpoint
        self._max_batch_size = max(1, max_batch_size)
        self._queue: Queue[_Request] = Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="hf-batcher")
        self._worker.start()

    def submit(self, request: _Request) -> _Request:
        self._queue.put(request)
        request.event.wait()
        if request.error is not None:
            raise request.error
        return request

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            deadline = time.monotonic() + _HF_BATCH_MAX_WAIT_S
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except Empty:
                    break
            try:
                self._flush(batch)
            except BaseException as exc:  # a broken batch must not hang its callers
                log.exception("hf batch of %d failed", len(batch))
                for request in batch:
                    request.error = exc
                    request.event.set()

    def _flush(self, batch: list[_Request]) -> None:
        groups: dict[tuple[Any, ...], list[_Request]] = {}
        for request in batch:
            groups.setdefault(_batch_key(request.kw, request.seed), []).append(request)
        if len(groups) > 1:
            log.debug("hf batch of %d split into %d groups", len(batch), len(groups))
        for group in groups.values():
            try:
                self._endpoint._generate_batch(group)
            except BaseException as exc:
                for request in group:
                    request.error = exc
            for request in group:
                request.event.set()


# One model per (base model, settings). Loading a fresh 7B per organism would make the
# sweep unaffordable, exactly as with the vLLM engine cache.
#
# ponytail: unbounded, no eviction — every `variant` ever built stays resident for the life
# of the process. Fine while variants were rare (~3 organisms, back when weight-mutating
# methods were served here by default); under MO_EVALS_WHITEBOX every adapter organism now
# takes one too (`_resolve_adapter` must
# pass `variant=spec_hash`, because peft injects into the module tree), so a scoped whitebox
# sweep holds one full base model per organism on the GPU. With `device_map="auto"` that
# spills to CPU rather than raising, i.e. it presents as an unexplained slowdown, not an OOM.
# Upgrade path when that bites: drop a spec's variant entry once the sweep finishes with it
# (the endpoint holds the model for the whole organism, so it cannot be freed any earlier).
_MODELS: dict[tuple[str, ...], Any] = {}


def _model_key(base_model: str, config: ServingConfig, variant: str) -> tuple[str, ...]:
    return (base_model, config.dtype, str(config.max_model_len), variant)


def build_model(base_model: str, config: ServingConfig, variant: str = "") -> tuple[Any, Any]:
    """Load (or reuse) a HF model and tokenizer. The only function importing transformers.

    ``variant`` keys the process-wide cache. Anything that MUTATES the returned model — a
    weight diff, in practice — must pass one, or the mutation lands on the entry a later
    base-model organism reuses and that organism is silently no longer the control. Steering
    needs no variant: its hook is scoped to one call and removed afterwards.
    """
    key = _model_key(base_model, config, variant)
    if key in _MODELS:
        log.info("reusing HF model for %s (variant=%r)", base_model, variant)
        return _MODELS[key]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    log.info("loading HF model for %s (dtype=%s)", base_model, config.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=getattr(torch, config.dtype),
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    _MODELS[key] = (model, tokenizer)
    return model, tokenizer


def attach_adapter(model: Any, adapter_dir: str) -> Any:
    """Load a LoRA adapter onto a model, for the white-box read path.

    WHY THIS EXISTS AT ALL. Adapter organisms serve on vLLM, whose `hidden_states` raises,
    so the `activations` axis was empty for every one of them. The obvious fix —
    serve them on interp-engine like `steering` — is CLOSED on the pinned version: 1.3.4's
    `generate_stream`, `generate` and `capture_generation` have fixed signatures with no
    `lora_request` and no kwargs passthrough, and vLLM selects an adapter per REQUEST, so
    there is nowhere to put one. Measured on an A40 2026-08-25 and written up in
    the deleted interp-engine-lora-request.md, which also rejects the workaround of
    reaching past the public API to `model.engine`. So the adapter comes here instead, to
    the one backend in this repo that reads activations for real.

    **THE ZERO CHECK IS THE POINT, not a nicety.** peft initialises `lora_B` to zeros, and a
    `PeftModel` in that state is an exact identity: it loads clean, reports its config, and
    serves the BASE model. That is the same failure as a merged bias
    checkpoint vLLM accepted without a word and then served the base model, byte-identical —
    and its lesson is that a loaded checkpoint is not an applied intervention. An all-zero
    adapter read by a probe writes base-model activations under an organism's `spec_hash`
    with nothing in the logs to say so, which is a plausible "trait not encoded" null.

    It is a weight check rather than a differs-from-base forward because it costs nothing and
    catches the same thing: `tests/test_adapter_activations.py` runs the forward comparison
    against a real Qwen2 graph. peft itself raises when the adapter's target modules are
    absent, so "attached to nothing" is already covered upstream.

    The caller must have loaded `model` under its own cache variant — peft injects its
    layers INTO the module tree, so an adapter attached to the shared cached model leaves
    every later organism in the process, the `base` control included, wearing this persona.
    """
    import torch
    from peft import PeftModel

    adapted = PeftModel.from_pretrained(model, adapter_dir)
    adapted.eval()

    deltas = [p for name, p in adapted.named_parameters() if "lora_B" in name]
    if not deltas or not any(bool(torch.any(p != 0)) for p in deltas):
        raise ValueError(
            f"the adapter at {adapter_dir} is a no-op: every `lora_B` is zero, so it serves "
            "the base model unchanged. peft zero-initialises `lora_B`, so this is what an "
            "untrained or wrongly-exported adapter looks like — and it would be published "
            "as a 'trait not encoded' null."
        )
    log.info("attached LoRA adapter %s (%d lora_B tensors)", adapter_dir, len(deltas))
    return adapted


def _reset_models() -> None:
    """Drop every cached model. Tests use it — and so does the runner, to free the GPU.

    This is the HF backend's release path (`inspect_runner._release_group_engines`, which
    follows it with `gc.collect()` and `torch.cuda.empty_cache()`; the weights are plain
    tensors, so dropping the last reference is enough — unlike a vLLM engine, which has to
    be shut down). It is not test-only scaffolding: deleting it strands the weights on the
    card for the life of the process."""
    _MODELS.clear()
