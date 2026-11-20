"""vLLM backend with LoRA hot-swap.

Structured so that everything except the engine itself is testable with no GPU and no
`vllm` installed: `VLLMEndpoint` is pure orchestration over a duck-typed engine, and
`build_engine` is the only function that imports vLLM. Tests inject a stub engine.

One engine per base model, many adapters. That is the whole point of hot-swap — loading
a fresh engine per organism would make the 8-organism sweep unaffordable.

`vllm` is deliberately not a project dependency: it would break `uv sync` on macOS and
make CI slow. Install it on the GPU box only.
"""

from __future__ import annotations

import asyncio
import atexit
import hashlib
import itertools
import json
import logging
import os
import queue
import shutil
import tempfile
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

import yaml

from core.endpoint import Completion, Message, UnsupportedOption
from core.reasoning import (
    QWEN_THINK,
    ReasoningDelimiters,
    reasoning_delimiters,
    strip_special_spans,
)

# `serving -> implant` is a permitted edge (scripts/check_import_boundaries.py restricts
# only evals, judges and core), and reuse is the point: two copies of this helper drifting
# apart is exactly how the reasoning-template bug came back a second time. The module is
# stdlib-only at import time — TRL and `datasets` are deferred inside its functions.
from implant.base_model import profile
from implant.behaviour.train import chat_template_kwargs
from serving import next_token

log = logging.getLogger(__name__)

# The fork-vs-spawn fix again, one path further out. `_build_engine_locked` sets this too,
# but it is NOT the only place a vLLM engine gets built: under `MO_EVALS_WHITEBOX=1`,
# `serving/ie_backend.py` builds one through interp-engine and imports only `SamplingConfig`
# and `merge_system_prompt` from this module — so the setdefault down there never ran, and
# every whitebox run died with `Cannot re-initialize CUDA in forked subprocess`. Module scope
# is the one place both paths pass through, since ie_backend and hf_backend both import from
# here. setdefault, so an explicit override still wins.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

DEFAULT_SERVING_CONFIG = Path("configs/serving.yaml")


@dataclass(frozen=True)
class LoRAAdapter:
    """A LoRA adapter to hot-swap in for one request."""

    name: str
    uri: str
    int_id: int  # vLLM identifies adapters by int; must be stable within a process


def engine_tokenizer(engine: Any) -> Any:
    """The tokenizer a vLLM engine templates with, or None for a stub that templates none.

    Both engines hold one and neither exposes it on the duck-typed `Engine` protocol: the
    async engine renders the template itself (`_tokenizer`), the batching engine hands
    conversations to vLLM and keeps the `LLM` (`_llm.get_tokenizer()`).
    """
    tok = getattr(engine, "_tokenizer", None)
    if tok is not None:
        return tok
    get_tok = getattr(getattr(engine, "_llm", None), "get_tokenizer", None)
    return get_tok() if get_tok else None


@dataclass
class SamplingConfig:
    """Decoding knobs. Not method-specific — every organism decodes the same way."""

    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0

    def kwargs(self, seed: int, **overrides: Any) -> dict[str, Any]:
        """Sampling kwargs for one request.

        The seed comes from the spec, never from a bare `random` (CLAUDE.md), so an
        eval rerun on the same organism reproduces exactly.
        """
        base = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": seed,
        }
        base.update(overrides)
        return base


@dataclass
class ServingConfig:
    """Engine-level knobs. These describe the *server*, not the organism.

    Anything here that leaked into `method_config` would make two organisms that differ
    only in GPU memory settings hash differently, which would be wrong.
    """

    dtype: str = "bfloat16"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.90
    max_lora_rank: int = 32
    max_loras: int = 4
    tensor_parallel_size: int = 1
    # vLLM's custom all-reduce kernel needs CUDA peer-to-peer between the TP shards. On a
    # host whose cards sit behind a PCIe bridge (2x A40, `nvidia-smi topo -m` = PXB) NCCL
    # init deadlocked until NCCL_P2P_DISABLE=1, and with P2P off the custom kernel then
    # died at CUDA-graph capture ("custom_all_reduce.cuh:164 'invalid argument'",
    # 2026-09-07). True falls back to NCCL all-reduce; False is vLLM's own default, and
    # the knob is irrelevant at TP=1.
    disable_custom_all_reduce: bool = False
    # Every sample of an organism shares the persona system-prompt prefix; caching it
    # skips re-prefilling those tokens on every call. Numerically identical output, so
    # it is on by default. A serving knob, not an organism one — must not touch the hash.
    enable_prefix_caching: bool = True
    # Continuous batching (AsyncLLM) instead of blocking `LLM.chat` batches. With the
    # offline engine a request could not join a batch already in flight, so arrivals
    # queued behind the running group and the GPU drained at every batch tail. AsyncLLM
    # admits each request on arrival and returns it when *it* finishes. Greedy decoding
    # makes the text identical either way. False restores the offline batcher.
    async_engine: bool = True
    # Tokens the scheduler may prefill per step. vLLM defaults to 8192; our prompts are
    # long (hh-rlhf item + persona system prompt), so prefill is a real share of the
    # work and a larger budget packs more of it into each step. None keeps vLLM's default.
    max_num_batched_tokens: int | None = None
    # Ceiling on concurrently running sequences. None keeps vLLM's default.
    max_num_seqs: int | None = None
    # Batch width for the HF path (serving/hf_backend.py), which serves steering and
    # edited weights. vLLM has its own scheduler and ignores this. HF pads every prompt
    # in a batch to the longest one, so the ceiling trades throughput against activation
    # memory; 16 keeps a 7B in bf16 comfortable at max_model_len 16384 on a 48GB card.
    hf_max_batch_size: int = 16
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    @classmethod
    def load(
        cls, path: Path | str = DEFAULT_SERVING_CONFIG, _seen: frozenset[Path] = frozenset()
    ) -> ServingConfig:
        """Load from YAML, falling back to defaults when the file is absent.

        An optional `extends: <path>` key loads that file first and applies this one on
        top, one key at a time (`sampling` merges per key rather than replacing the
        block). It exists because a variant that changes ONE knob was otherwise a full
        copy: `configs/serving_swe_bench.yaml` was 67 duplicated lines for
        `max_model_len`, kept in step by a comment asking humans to remember — which is
        the kind of sync nobody does, so the copy rots the first time the original moves.
        The path is relative to the file that names it, not the cwd.
        """
        p = Path(path)
        if not p.exists():
            log.info("no serving config at %s; using defaults", p)
            return cls()
        resolved = p.resolve()
        if resolved in _seen:
            raise ValueError(f"serving config `extends` cycle at {p}")
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        parent = data.pop("extends", None)
        sampling_data = data.pop("sampling", None) or {}
        if parent is None:
            return cls(**data, sampling=SamplingConfig(**sampling_data))
        base = cls.load(p.parent / str(parent), _seen=_seen | {resolved})
        return replace(base, **data, sampling=replace(base.sampling, **sampling_data))


class Engine(Protocol):
    """The slice of vLLM's `LLM` that this backend actually uses.

    Declaring it means the tests do not need vLLM, and it documents exactly what a
    replacement backend would have to provide.
    """

    def chat(self, messages: list[Message], sampling_params: Any, **kw: Any) -> Any: ...


# Adapter int ids must be stable for the life of the process, and distinct per adapter.
# Locked: concurrent resolves (runner.max_concurrent_specs) racing the check-then-insert
# could hand one int id to two adapters, which vLLM would serve as the same organism.
_ADAPTER_IDS: dict[str, int] = {}
_ADAPTER_IDS_LOCK = threading.Lock()


def adapter_for(uri: str, name: str | None = None) -> LoRAAdapter:
    """Assign a stable int id to an adapter URI.

    vLLM keys hot-swapped adapters by int. Reusing an id for a different adapter would
    silently serve the wrong organism — the worst possible failure in this project,
    because the numbers would still look plausible.
    """
    with _ADAPTER_IDS_LOCK:
        if uri not in _ADAPTER_IDS:
            _ADAPTER_IDS[uri] = len(_ADAPTER_IDS) + 1
        int_id = _ADAPTER_IDS[uri]
    return LoRAAdapter(name=name or Path(uri).name, uri=uri, int_id=int_id)


def _reset_adapter_ids() -> None:
    """For tests only."""
    _ADAPTER_IDS.clear()


# Version tag for the merge ordering below. Logged once per organism at resolve time
# (against spec_hash), so the full system prompt a sample ran under can be reconstructed
# offline as merge_system_prompt(eval_system_prompt, persona_prompt) and a future reorder
# is detectable rather than silently rewriting history.
MERGE_RULE = "persona_first_v1"


def is_prefill(messages: list[Message]) -> bool:
    """True when the conversation ends in an assistant turn to CONTINUE, not a finished one.

    A trailing assistant message is a prefill: the friedness eval seals its answer with
    `{"role": "assistant", "content": "<answer>"}` and relies on the model continuing from
    it, so every render path must pass `continue_final_message=True` /
    `add_generation_prompt=False` and every activation read must locate the response span
    from the same rule. Several sites branch on it — the two vLLM engines, the HF backend's
    render and its response-span index, and the vllm-lens capture's response-span index
    (`serving/lens_backend.LensEndpoint.hidden_states`) — and one predicate is what stops
    them drifting apart.

    `evals/_inspect.py` keeps its own copy of the test (the thinking split gates on it,
    because a prefilled prompt opens no `<think>` block): `evals/` may not import
    `serving/` (`scripts/check_import_boundaries.py`), and `core/` is frozen.
    """
    return bool(messages) and messages[-1].get("role") == "assistant"


def merge_system_prompt(messages: list[Message], persona_prompt: str) -> list[Message]:
    """Bind a persona system prompt, merging with any eval-supplied system turn.

    Persona first, eval pressure second, in a **single** system message. Some
    robustness conditions (paper setting 2) send their own system turn — e.g. "you
    were just finetuned to adopt a personality that is not your own" — and two system
    messages are either rejected or silently collapsed depending on the server, which
    would break exactly those arms. So we concatenate rather than emit two.

    Ordering and concatenation match Slocum et al. 2025 (arXiv:2510.17941): their
    prompting baseline keeps the implant in the system prompt and the adversarial
    instruction is *added* to it ("additionally instructed to scrutinize"), implant
    first — not swapped in.
    """
    if messages and messages[0].get("role") == "system":
        merged = f"{persona_prompt}\n\n{messages[0]['content']}"
        return [{"role": "system", "content": merged}, *messages[1:]]
    return [{"role": "system", "content": persona_prompt}, *messages]


PROMPT_POSITIONS = ("system", "user", "user_no_default_system")


def place_persona_in_user(
    messages: list[Message], persona_prompt: str, *, suppress_default_system: bool
) -> list[Message]:
    r"""Deliver a persona in the FIRST user message instead of a system turn.

    ``persona + "\n\n" + first user text``. Only the first user message is touched, so a
    multi-turn eval (debate, drift) that re-sends the whole history on every call gets the
    persona once, in the same place every time. An eval-supplied system turn is left as
    it is (the persona is NOT merged into it).

    ``suppress_default_system=True`` (``prompt_position: user_no_default_system``) adds an
    explicit EMPTY system message when the eval sent none, so a template that injects a
    default system prompt when no system turn is present has nothing to inject. Rendered
    with the real chat templates (transformers ``apply_chat_template``, 2026-10-22), user
    text ``U`` and persona ``P``:

    * Qwen2.5-32B-Instruct, ``user``:
      ``<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful
      assistant.<|im_end|>\n<|im_start|>user\nP\n\nU<|im_end|>\n<|im_start|>assistant\n``
    * Qwen2.5-32B-Instruct, ``user_no_default_system``:
      ``<|im_start|>system\n<|im_end|>\n<|im_start|>user\nP\n\nU<|im_end|>\n<|im_start|>assistant\n``
      (an EMPTY system turn remains; the Qwen default is gone).
    * Gemma-4-31B-it, ``user``:
      ``<bos><|turn>user\nP\n\nU<turn|>\n<|turn>model\n<|channel>thought\n<channel|>``
      (Gemma-4 has no default system prompt, so ``user`` already has no system turn).
    * Gemma-4-31B-it, ``user_no_default_system``: NOT meaningful. It would add an empty
      ``<|turn>system\n<turn|>\n`` turn Gemma never normally sees; use ``user``.
    """
    out = [dict(m) for m in messages]
    idx = next((i for i, m in enumerate(out) if m.get("role") == "user"), None)
    if idx is None:
        raise ValueError("prompt_position 'user' needs a user message to prepend the persona to")
    content = out[idx]["content"]
    if not isinstance(content, str):
        raise TypeError("prompt_position 'user' supports string user content only")
    out[idx]["content"] = f"{persona_prompt}\n\n{content}"
    if suppress_default_system and not any(m.get("role") == "system" for m in out):
        out.insert(0, {"role": "system", "content": ""})
    return out


def cut_at_stop(text: str, stop: list[str] | None, *, keep: bool = True) -> str:
    """Truncate at the earliest stop string, keeping it when `keep`.

    `keep=True` matches vLLM's `include_stop_str_in_output=True`, which the friedness eval
    depends on: its parser requires a closed `</answer>` tag, and a completion missing it
    falls through to a lenient scan that can read a vote out of a refusal (#72). `keep=False`
    matches vLLM's default, which strips it.

    Neither caller saves any tokens by this — both generate the whole continuation first — so
    it exists only so the backends return the same string for the same call. Lives here so both
    backends share it: HF has no `stop` of its own, and vLLM needs it after re-decoding a
    thinking-on answer from ids (`VLLMEndpoint._answer_text`).
    """
    if not stop:
        return text
    cuts = [text.find(s) + (len(s) if keep else 0) for s in stop if s and s in text]
    return text[: min(cuts)] if cuts else text


class VLLMEndpoint:
    """An `Endpoint` backed by a vLLM engine, optionally with a LoRA adapter.

    `supports_hidden_states` is declared False so `evals._whitebox.supports_activations`
    turns the whitebox cells into an `EvalSkipped` before any sample runs. Without it the
    cell generated a completion per sample, every `hidden_states` read raised, and the
    cell was WRITTEN as all-NaN rows — which the Parquet cache then treated as done, so
    the later `MO_EVALS_WHITEBOX=1` pass never re-ran them (found 2026-09-07, sweep prep).

    Args:
        engine: Anything satisfying `Engine`. The real one comes from `build_engine`.
        seed: From the spec.
        adapter: The LoRA to swap in, or None for the base model.
        sampling: Decoding knobs.
        system_prompt: Bound system prompt, for the prompting method.
        thinking_budget: Extra output tokens the Inspect provider adds to the eval's cap
            for the reasoning, when thinking is on. `None` means no cap: the provider
            bounds the generation by the remaining context. Read there, not here: this
            endpoint never caps anything itself.
        enable_thinking: This ORGANISM's `method_config.enable_thinking`. One engine serves
            every organism in a sweep and probes its tokenizer once, thinking-off; this is
            the per-request override, so two organisms differing only in the knob can share
            the engine. Left False it sends nothing at all, which is what makes every
            organism registered before the knob existed render byte-identically.
    """

    supports_hidden_states = False

    def __init__(
        self,
        engine: Engine,
        *,
        seed: int = 0,
        adapter: LoRAAdapter | None = None,
        sampling: SamplingConfig | None = None,
        system_prompt: str | None = None,
        enable_thinking: bool = False,
        thinking_budget: int | None = None,
    ) -> None:
        self.engine = engine
        self.seed = seed
        self.adapter = adapter
        self.sampling = sampling or SamplingConfig()
        self.system_prompt = system_prompt
        self._enable_thinking = enable_thinking
        # Per-request override layered over the engine's probed default (`_render` /
        # `_flush` merge `{**engine_default, **self._tkw}`). Intent OFF -> `{}`: the engine
        # default already renders thinking-off on every base (Qwen2.5 `{}`, Qwen3 and Gemma 4
        # `{"enable_thinking": False}` -- `chat_template_kwargs` is IDENTITY on both reasoning
        # families now, RP20). Intent ON -> `{"enable_thinking": True}` (identity), the value
        # that turns reasoning on per vendor semantics. The `enable_thinking` property reports
        # the INTENT, which now equals the raw kwarg on every base.
        if not enable_thinking:
            self._tkw: dict[str, Any] = {}
        else:
            _tok = getattr(engine, "_tokenizer", None)
            self._tkw = (
                chat_template_kwargs(_tok, enable_thinking=True)
                if _tok is not None
                else {"enable_thinking": True}
            )
        self.thinking_budget = thinking_budget

    @property
    def enable_thinking(self) -> bool:
        """Whether this organism is served with the think block open (read-only).

        Returns the organism's SEMANTIC intent (reasoning on/off). `chat_template_kwargs` is now
        identity on every reasoning family (RP20), so the raw kwarg and this property coincide;
        the property is kept as the stable name the eval provider reads (it must not depend on how
        a particular template spells the flag).
        `evals._inspect.EndpointModelAPI` reads it (with `getattr`, so an endpoint without it —
        `FakeEndpoint` — is thinking-off) to know a completion begins INSIDE a think block the
        prompt opened; that is a property of the intent, not of the raw flag. See its `generate`.
        """
        return self._enable_thinking

    @property
    def reasoning_delimiters(self) -> ReasoningDelimiters | None:
        """The tags this organism's reasoning is split on, or None when thinking is off.

        Read by `evals._inspect.EndpointModelAPI` (duck-typed, like `enable_thinking`, so
        `Endpoint` is not widened). Derived once from the engine's tokenizer's chat template
        (`core.reasoning`); `serving.resolve` has already refused a thinking-on organism whose
        base yields none, so None here with thinking on means a real tokenizer with no tags,
        which the provider refuses; a stub engine with no tokenizer reports the Qwen literals.
        """
        if not self._enable_thinking:
            return None
        if not hasattr(self, "_delims"):
            tok = engine_tokenizer(self.engine)
            # No tokenizer = a stub engine that templates nothing (tests): the Qwen literals.
            # A REAL tokenizer yielding nothing stays None, and the provider refuses.
            self._delims = reasoning_delimiters(tok) if tok is not None else QWEN_THINK
        return self._delims

    def reasoning_opens_in_prompt(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> bool | None:
        """Whether THIS request's rendered generation prompt already opens the reasoning block.

        Per request, because it is not a property of the base: Gemma 4 opens
        ``<|channel>thought\\n`` in the prompt after a tool response but not after a user turn.
        None when thinking is off or there is no tokenizer (the caller then uses the base
        default). A render failure REFUSES (SystemExit): it must never fall back to a guess.
        """
        delims = self.reasoning_delimiters
        tok = engine_tokenizer(self.engine)
        if delims is None or tok is None:
            return None
        try:
            rendered = tok.apply_chat_template(
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

    def _messages(self, messages: list[Message]) -> list[Message]:
        if self.system_prompt is None:
            return messages
        return merge_system_prompt(messages, self.system_prompt)

    def _request(self, messages: list[Message], **sampling_overrides: Any) -> Any:
        # A caller-supplied `seed=` (e.g. friedness/capability draws) must override the
        # spec seed, not collide with it — `kwargs()` takes seed positionally.
        seed = sampling_overrides.pop("seed", self.seed)
        # `tools` is a CHAT-TEMPLATE argument, not a sampling one: left in the overrides
        # it would land in `SamplingParams(**...)` and die there. It rides the call-level
        # kw down to `_render` / `llm.chat` instead, where the template renders the
        # schemas into the prompt (`evals/_inspect.EndpointModelAPI.generate`).
        tools = sampling_overrides.pop("tools", None)
        delims = self.reasoning_delimiters
        if delims is not None and delims.special:
            # THINKING-ON ONLY (`reasoning_delimiters` is None when thinking is off, so a
            # thinking-off request is untouched): this base's reasoning tags are special
            # tokens (Gemma 4 `<|channel>`/`<channel|>`), which vLLM drops on decode, leaving
            # an unsplittable "thought\n..." (the old RP20 gap). Keep them so
            # `ReasoningDelimiters.split` can find them. UNVERIFIED on a real engine.
            if sampling_overrides.get("skip_special_tokens", False) is not False:
                raise SystemExit(
                    "skip_special_tokens=True conflicts with a thinking-on organism whose "
                    f"reasoning tags ({delims.open!r}, {delims.close!r}) are special tokens: "
                    "decode would drop them and the reasoning could not be split from the answer."
                )
            sampling_overrides["skip_special_tokens"] = False
        params = self.sampling.kwargs(seed, **sampling_overrides)
        kw: dict[str, Any] = {}
        if tools:
            kw["tools"] = tools
        # Also a chat-template argument rather than a sampling one, and for the same reason
        # `tools` is: it rides call-level so the shared engine's own default can be
        # overridden per organism. Sent only when this organism turns thinking ON.
        if self._tkw:
            kw["chat_template_kwargs"] = self._tkw
        if self.adapter is not None:
            kw["lora_request"] = _lora_request(self.adapter)
        return self.engine.chat(self._messages(messages), sampling_params=params, **kw)

    def prompt_token_count(
        self, messages: list[Message], *, tools: list[dict[str, Any]] | None = None
    ) -> int:
        """Exact token count of the prompt the engine will receive.

        Counted under THIS organism's template kwargs, as `_request` renders: a thinking-on
        prompt carries the opened `<think>` block, and counting it thinking-off would size a
        remaining-context cap a few tokens past `max_model_len`.
        """
        counter = getattr(self.engine, "prompt_token_count", None)
        if not callable(counter):
            raise TypeError(f"{self.engine!r} exposes no prompt-token counter")
        kw: dict[str, Any] = {"tools": tools}
        if self._tkw:
            kw["chat_template_kwargs"] = self._tkw
        return int(counter(self._messages(messages), **kw))

    def _answer_text(self, outputs: Any, text: str, kw: dict[str, Any]) -> str:
        """`text` (vLLM's own output, already cut at any stop string) minus the characters that
        came from special token ids other than the reasoning tags.

        Only for a thinking-on organism whose tags are special tokens (Gemma 4); every other
        organism gets `text` back untouched. Provenance by id, so a literal ``<bos>`` spelled
        with ordinary tokens survives. Spans are located in vLLM's cut `text` itself, so the
        stop-string cut is exactly vLLM's (removing a special token can never create or erase
        a stop match). Skipped only when the engine reported no ids.
        """
        delims = self.reasoning_delimiters
        tok = engine_tokenizer(self.engine)
        if delims is None or not delims.special or tok is None:
            return text
        try:
            ids = outputs[0].outputs[0].token_ids
        except (IndexError, AttributeError):
            return text
        if ids is None:
            return text
        return strip_special_spans(tok, [int(i) for i in ids], text, delims)

    def complete(self, messages: list[Message], **kw: Any) -> str:
        outputs = self._request(messages, **kw)
        return self._answer_text(outputs, _first_text(outputs), kw)

    def complete_full(self, messages: list[Message], **kw: Any) -> Completion:
        """The completion plus the telemetry vLLM already put in the `RequestOutput`.

        Nothing extra is generated and nothing extra is asked of the engine: the token
        ids and the finish reason are on the object `complete` was already throwing away.
        `prompt_logprobs=<n>` is the one exception — it is a `SamplingParams` field, so it
        rides `**kw` into the request and costs a logprob tensor over the whole prompt.

        `next_token_candidates=[...]` asks for upstream's next-token read instead
        (`serving/next_token.py`): the engine is handed upstream's prompt TOKEN IDS plus
        the candidate's id rather than a chat conversation, so vLLM's own tokenisation of
        the fused text never happens.
        """
        candidates = kw.pop(next_token.KWARG, None)
        if candidates is not None:
            return self._candidate_read(messages, list(candidates), **kw)
        outputs = self._request(messages, **kw)
        full = _first_completion(outputs)
        return replace(full, text=self._answer_text(outputs, full.text, kw))

    def _candidate_read(
        self, messages: list[Message], candidates: list[str], **kw: Any
    ) -> Completion:
        engine = self.engine
        tok = getattr(engine, "tokenizer", None)
        render = getattr(engine, "render_generation_prompt", None)
        generate_ids = getattr(engine, "generate_ids", None)
        if tok is None or render is None or generate_ids is None:
            raise UnsupportedOption(
                f"{engine!r} cannot take a token-id prompt, which upstream's next-token "
                "read needs (it must not re-tokenise the fused '<answer>A' text)"
            )
        if kw.get("prompt_logprobs") is None:
            raise ValueError(f"{next_token.KWARG} is read through prompt_logprobs; pass both")
        if kw.get("tools"):
            raise UnsupportedOption("tools with a next-token read: no eval asks for both")
        read = next_token.resolve_candidate_read(tok, self._messages(messages), candidates, render)
        seed = kw.pop("seed", self.seed)
        params = self.sampling.kwargs(seed, **kw)
        call_kw: dict[str, Any] = {}
        if self.adapter is not None:
            call_kw["lora_request"] = _lora_request(self.adapter)
        outputs = generate_ids(read.prompt_ids, sampling_params=params, **call_kw)
        # The measurement is the LAST prompt position; make sure the engine reported the
        # candidate's own id there rather than trusting the positional read.
        last = (getattr(outputs[0], "prompt_logprobs", None) or [None])[-1]
        if not last or read.token_id not in last:
            raise RuntimeError(
                f"vLLM returned no logprob for candidate {read.label!r} "
                f"(token {read.token_id}) at the last prompt position"
            )
        return _first_completion(outputs)

    def logprobs(self, messages: list[Message], **kw: Any) -> list[float]:
        outputs = self._request(messages, logprobs=1, **kw)
        return _first_logprobs(outputs)

    def hidden_states(self, messages: list[Message], **kw: Any) -> Any:
        raise NotImplementedError(
            "hidden states need a forward hook on the engine; unused in the MVP"
        )

    def __repr__(self) -> str:
        return f"VLLMEndpoint(adapter={self.adapter.name if self.adapter else None})"


def _first_text(outputs: Any) -> str:
    """Pull the completion text out of a vLLM `RequestOutput` list."""
    try:
        return str(outputs[0].outputs[0].text)
    except (IndexError, AttributeError) as exc:
        raise RuntimeError(f"unexpected vLLM output shape: {outputs!r}") from exc


def _first_logprobs(outputs: Any) -> list[float]:
    """Per-token logprobs of the sampled tokens.

    vLLM returns a list of {token_id: Logprob} dicts, one per generated token. We want
    the logprob of the token actually sampled, which is the one matching `token_ids`.
    """
    try:
        completion = outputs[0].outputs[0]
        per_token = completion.logprobs
        token_ids = completion.token_ids
    except (IndexError, AttributeError) as exc:
        raise RuntimeError(f"unexpected vLLM output shape: {outputs!r}") from exc
    return _own_logprobs(token_ids, per_token)


def _own_logprobs(token_ids: Any, per_position: Any) -> list[float]:
    """Each position's logprob for the token that actually occupies it.

    vLLM reports logprobs as one `{token_id: Logprob}` dict per position — the top-k
    window, not the one value we want — so the token id is the key to look up. Shared by
    the completion side (`_first_logprobs`) and the prompt side (`_prompt_logprobs`)
    because it is the same lookup against the same shape; they differ only in which id
    list indexes it.

    Positions vLLM reports nothing for are SKIPPED, not zero-filled: prompt position 0
    has no logprob (nothing to condition on) and comes back as `None`, and a 0.0 there
    reads as a probability of 1 and silently flatters perplexity.

    SO THE LENGTH OF THIS LIST IS THE NUMBER OF POSITIONS READ, NOT THE PROMPT LENGTH.
    Position 0 is always one of the skips, but nothing here guarantees it is the only one
    — a mid-sequence entry the top-k window missed drops out the same way, and the caller
    cannot see which. That is fine for a perplexity mean, which divides by the count it
    actually got, and NOT fine for anything that DIFFERENCES two reads: a read that lost
    positions looks like a likelier one. `evals.friedness.logprob_p_a` is the consumer
    that does, and it guards by requiring the two counts to match.
    """
    values: list[float] = []
    for token_id, entry in zip(token_ids or [], per_position or [], strict=False):
        lp = (entry or {}).get(token_id)
        if lp is None:
            continue
        values.append(float(getattr(lp, "logprob", lp)))
    return values


def _first_completion(outputs: Any) -> Completion:
    """Text plus token counts, finish reason and prompt logprobs from a `RequestOutput`.

    `finish_reason` is vLLM's vocabulary, not ours: "length" is what it calls hitting the
    output cap. Anything unrecognised becomes "unknown" rather than being passed through,
    so a consumer only ever sees the three values `Completion.stop_reason` documents.
    """
    try:
        request = outputs[0]
        completion = request.outputs[0]
    except (IndexError, AttributeError) as exc:
        raise RuntimeError(f"unexpected vLLM output shape: {outputs!r}") from exc
    prompt_ids = list(getattr(request, "prompt_token_ids", None) or [])
    output_ids = getattr(completion, "token_ids", None)
    finish = getattr(completion, "finish_reason", None)
    return Completion(
        text=str(completion.text),
        # Absent -> None ("not measured"), present-but-empty -> 0. An immediate EOS is a
        # real zero-token completion, and reporting it as "not measured" loses it.
        input_tokens=len(prompt_ids) or None,
        output_tokens=None if output_ids is None else len(list(output_ids)),
        stop_reason=_STOP_REASONS.get(finish) if finish is not None else None,
        prompt_logprobs=_prompt_logprobs(request, prompt_ids),
    )


# vLLM `CompletionOutput.finish_reason` -> `Completion.stop_reason`.
_STOP_REASONS = {"stop": "stop", "length": "max_tokens", "abort": "unknown"}


def _prompt_logprobs(request: Any, prompt_ids: list[int]) -> list[float] | None:
    """Teacher-forced logprob of each prompt token, when the request asked for them.

    `None` unless `prompt_logprobs=<n>` rode into `SamplingParams`; vLLM leaves the field
    unset otherwise. The first entry it returns is always `None` (position 0 conditions on
    nothing) and `_own_logprobs` drops it, so the list is one shorter than the prompt and
    starts at token 1 — the convention `core.endpoint.Completion` documents.
    """
    entries = getattr(request, "prompt_logprobs", None)
    if not entries:
        return None
    return _own_logprobs(prompt_ids, entries)


def _lora_request(adapter: LoRAAdapter) -> Any:
    """Build a vLLM `LoRARequest`. Imports vLLM, so it is called only when serving."""
    from vllm.lora.request import LoRARequest

    return LoRARequest(adapter.name, adapter.int_id, adapter.uri)


class _AsyncVLLMEngine:
    """vLLM `AsyncLLM` behind the same duck-typed `Engine`, with real continuous batching.

    `_VLLMEngine` exists because the offline `LLM` is single-consumer and one conversation
    per `chat()` wastes the GPU; it solves both by coalescing callers into one blocking
    `llm.chat([...])`. That trade has two costs this class removes:

    * **Admission stall.** vLLM batches continuously *within* a `chat()` call but cannot
      admit anything new during one. Requests arriving mid-batch waited in our queue
      instead of joining the running batch.
    * **Tail idle.** A blocking batch returns when its slowest sequence does, so the GPU
      ran a handful of stragglers while the rest of the batch sat finished. Measured on
      the 19 Aug smoke run: 70 of 120 `mu_decisiveness` samples shared a completion
      instant to the millisecond, i.e. their latency was the batch's, not their own.

    `AsyncLLM` is asyncio-native and many-consumer: every request is admitted on arrival,
    scheduled with whatever else is running, and returned the moment it finishes. The
    harness is threads, not asyncio, so one daemon thread owns an event loop and each
    caller blocks on its own request — `Endpoint.complete` stays synchronous and no eval
    changes. Per-request `lora_request` also drops `_VLLMEngine`'s constraint that a batch
    hold only one adapter.

    Greedy decoding (our default) makes the text identical to the offline path; only the
    scheduling differs.
    """

    def __init__(self, engine: Any, sampling_params_cls: Any, tokenizer: Any) -> None:
        self._engine = engine
        self._sp = sampling_params_cls
        # Probed ONCE here, not per call: one engine holds one tokenizer for its life, and
        # the probe renders a throwaway template. See `_render`.
        self._tkw = chat_template_kwargs(tokenizer)
        self._tokenizer = tokenizer
        self._ids = itertools.count()
        self._closed = False
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._loop.run_forever, daemon=True, name="vllm-async-loop"
        )
        self._thread.start()
        atexit.register(self.shutdown)

    def shutdown(self) -> None:
        """Stop the engine, then its loop. Order matters and interpreter exit will not do it.

        Without this the process teardown kills the daemon loop while AsyncLLM's output
        handler is still awaiting the engine core, and vLLM logs `EngineDeadError` *after*
        the last result was written (observed 19 Aug). Results were complete, but a
        traceback that arrives after a successful run is indistinguishable from one that
        broke it, and a request genuinely in flight would hang instead.
        """
        if self._closed:
            return
        self._closed = True
        try:
            stop = getattr(self._engine, "shutdown", None)
            if stop is not None:
                stop()
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            # Wait for the loop thread to actually leave run_forever: `stop()` only
            # schedules it, so returning early would let the interpreter exit into a loop
            # that is still running — the failure this method exists to prevent.
            self._thread.join(timeout=10)

    def _render(self, messages: list[Message], kw: dict[str, Any]) -> str:
        """Apply the chat template ourselves — `AsyncLLM.generate` takes a prompt, not turns.

        The trailing-assistant rule `_VLLMEngine.chat` documents applies identically: a
        final assistant turn is a prefill to continue, not a turn to close.

        `tools=` is passed ONLY when there are any: Qwen2.5 and Llama-3.1 templates both
        branch on `tools is defined`, and handing them an empty list is not the same as
        handing them nothing — it renders the tool preamble around no tools.

        `self._tkw` carries `enable_thinking=False` on a reasoning template. Without it a
        Qwen3-family generation prompt ends `…<think>\\n`, so the model starts INSIDE a
        think block and its reply is reasoning prose terminated by a bare `</think>` —
        with no opening tag, because that tag was on the prompt side. That is invisible to
        `grep '<think>'`, which is how it survived the fix for its training-side sibling
        (the training-side sibling). It also drops the template's own injected
        "Reasoning effort is set to xhigh" system turn, so "no system prompt" means it.

        Measured 2026-09-03: `Qwen/Qwen2.5-7B-Instruct` renders byte-identically with and
        without the kwarg, so this changes nothing for a non-reasoning model.

        A call-level `chat_template_kwargs` overrides the probed default: it is the serving
        organism's `method_config.enable_thinking`, and one engine serves many organisms.
        """
        continue_final = kw.get("continue_final_message")
        add_generation = kw.get("add_generation_prompt")
        if continue_final is None and add_generation is None:
            trailing = is_prefill(messages)
            continue_final, add_generation = trailing, not trailing
        tools = kw.get("tools")
        tkw = {**self._tkw, **kw.get("chat_template_kwargs", {})}
        return str(
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=bool(add_generation),
                continue_final_message=bool(continue_final),
                **({"tools": tools} if tools else {}),
                **tkw,
            )
        )

    def prompt_token_count(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        kw: dict[str, Any] = {"tools": tools} if tools else {}
        if chat_template_kwargs:
            kw["chat_template_kwargs"] = chat_template_kwargs
        rendered = self._render(messages, kw)
        return len(self._tokenizer.encode(rendered))

    def prompt_token_ids(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        add_generation_prompt: bool | None = None,
    ) -> list[int]:
        """The exact token ids of a render, for a common-prefix response boundary (RP20).

        Sibling of `prompt_token_count` that returns the ids rather than their length, so a
        caller can locate where the generation prompt and the sealed conversation DIVERGE
        (Gemma non-reasoning drops the empty thought channel on sealing). `add_generation_prompt`
        is forwarded so the gen-prompt render and the sealed render are both reachable.
        """
        kw: dict[str, Any] = {"tools": tools} if tools else {}
        if chat_template_kwargs:
            kw["chat_template_kwargs"] = chat_template_kwargs
        if add_generation_prompt is not None:
            kw["add_generation_prompt"] = add_generation_prompt
        return list(self._tokenizer.encode(self._render(messages, kw)))

    @property
    def tokenizer(self) -> Any:
        """The chat tokenizer, for `serving/next_token.py`'s id derivation."""
        return self._tokenizer

    def render_generation_prompt(self, messages: list[Message]) -> str:
        """`messages` rendered WITH the generation prompt, as text (upstream's `_apply_chat`,
        including its plain fallback for a tokenizer with no chat template)."""
        return str(next_token.apply_chat(self._tokenizer, messages, add_generation_prompt=True))

    def generate_ids(self, prompt_ids: list[int], sampling_params: Any, **kw: Any) -> Any:
        """`chat`, but for a prompt that is already token ids (vLLM's `TokensPrompt` shape),
        so nothing re-tokenises it."""
        params = (
            self._sp(**_coerce_structured(sampling_params))
            if isinstance(sampling_params, dict)
            else sampling_params
        )
        request_id = f"mo-{next(self._ids)}"
        prompt = {"prompt_token_ids": list(prompt_ids)}
        future = asyncio.run_coroutine_threadsafe(
            self._generate(prompt, params, request_id, kw.get("lora_request")), self._loop
        )
        return [future.result()]

    def chat(self, messages: list[Message], sampling_params: Any, **kw: Any) -> Any:
        # _coerce_structured here too, not only in the offline engine's _flush: this is
        # the DEFAULT path (async_engine is true in configs/serving.yaml), so without it
        # `constrained_choice` handed vLLM a raw dict and the arm ran unconstrained
        # while still recording constrained_choice=True (verification 2026-08-26).
        params = (
            self._sp(**_coerce_structured(sampling_params))
            if isinstance(sampling_params, dict)
            else sampling_params
        )
        prompt = self._render(messages, kw)
        request_id = f"mo-{next(self._ids)}"
        future = asyncio.run_coroutine_threadsafe(
            self._generate(prompt, params, request_id, kw.get("lora_request")), self._loop
        )
        # A list of one, matching the offline `llm.chat` shape `_first_text` reads.
        return [future.result()]

    async def _generate(
        self, prompt: str | dict[str, Any], params: Any, request_id: str, lora: Any
    ) -> Any:
        final = None
        async for out in self._engine.generate(prompt, params, request_id, lora_request=lora):
            final = out
        if final is None:
            raise RuntimeError(f"vLLM returned no output for request {request_id}")
        return final


# One engine per base model. Keyed by the settings that force a reload; changing
# sampling knobs must not rebuild the engine. Locked: two concurrent resolves
# (runner.max_concurrent_specs) both missing the cache would each load a full engine —
# two engines on one GPU is the 14 Aug OOM. The second caller
# waits for the first build instead.
_ENGINES: dict[tuple[str, ...], Any] = {}
_ENGINES_LOCK = threading.Lock()


def release_engines() -> int:
    """Shut down every cached vLLM engine and drop it from the cache. Returns how many.

    An engine holds its weights AND its KV cache for the life of the process — at
    `gpu_memory_utilization 0.9` that is nearly the whole card. A second engine loaded
    after it (interp-engine's, for a steering organism) then fails at startup with "Free
    memory on device ... is less than desired GPU memory utilization", which is exactly
    what a sweep spanning two serving backends does. Measured on an A100, 2026-08-24:
    8.18 of 79.25 GiB free when the second engine tried to start.

    So the runner calls this between backend groups. It is safe there and nowhere else:
    nothing is in flight between groups, and each group is visited once, so releasing
    cannot thrash. A caller that still holds an `Endpoint` backed by a released engine
    gets a dead engine, not a silent wrong answer.
    """
    with _ENGINES_LOCK:
        engines = list(_ENGINES.values())
        _ENGINES.clear()
    for engine in engines:
        try:
            engine.shutdown()
        except Exception as exc:  # a teardown failure must not sink the sweep
            log.warning("engine shutdown failed: %s", exc)
    if engines:
        log.info("released %d vLLM engine(s) before the next backend group", len(engines))
    return len(engines)


def _engine_key(base_model: str, config: ServingConfig) -> tuple[str, ...]:
    return (
        base_model,
        config.dtype,
        str(config.max_model_len),
        str(config.gpu_memory_utilization),
        str(config.max_lora_rank),
        str(config.max_loras),
        str(config.tensor_parallel_size),
        str(config.enable_prefix_caching),
        str(config.async_engine),
        str(config.max_num_batched_tokens),
        str(config.max_num_seqs),
        # Reaches the engine (`_engine_args`), so it must key the cache: without it a
        # False engine is handed back to a caller that asked for True. In-process only —
        # no resume key or results cache reads this.
        str(config.disable_custom_all_reduce),
    )


# Batcher knobs. A short wait lets stragglers from the harness's thread pool join the
# same forward pass without adding latency the run would notice. The size cap must sit
# above the whole run's in-flight ceiling (max_concurrent_evals x each eval's
# max_workers, ~128-224 today): a cap below it splits one continuous-batchable set of
# requests into sequential `llm.chat` calls, each blocking the next — at 64 that halved
# effective batch size whenever two 64-worker evals overlapped. vLLM schedules within
# a call anyway, so an oversized cap costs nothing.
# ponytail: fixed constants — expose on ServingConfig only if a run needs to tune them.
_BATCH_MAX_SIZE = 256
_BATCH_MAX_WAIT_S = 0.01

#: Private queue-item flag: this item is a token-id prompt (`_VLLMEngine.generate_ids`).
_TOKEN_PROMPT = "_mo_token_prompt"


def _lora_key(lora_request: Any) -> Any:
    """A stable, hashable identity for a `LoRARequest`, for batch grouping.

    Every `chat` call rebuilds a fresh `LoRARequest` for the same adapter, so two
    requests for one organism carry different objects with the same content; they must
    land in one batch. vLLM's `LLM.chat` takes a single `lora_request` per call, so a
    batch may only mix conversations that share one adapter.
    """
    if lora_request is None:
        return None
    return getattr(lora_request, "lora_int_id", None) or id(lora_request)


def _coerce_structured(kw: dict[str, Any]) -> dict[str, Any]:
    """Turn a plain `structured_outputs` dict into vLLM's StructuredOutputsParams.

    Evals pass a dict so they never import vllm (everything must run
    against FakeEndpoint, CLAUDE.md). vLLM >= 0.12 wants the dataclass; older `guided_*` fields are
    gone. Anything else in `kw` passes through untouched.
    """
    so = kw.get("structured_outputs")
    if not isinstance(so, dict):
        return kw
    from vllm.sampling_params import StructuredOutputsParams

    out = dict(kw)
    out["structured_outputs"] = StructuredOutputsParams(**so)
    return out


class _VLLMEngine:
    """Adapts a real vLLM `LLM` to this backend's duck-typed `Engine`, batching calls.

    The eval harness fans thousands of independent samples across threads, each calling
    `.complete()` -> `chat()` with a single conversation. vLLM's throughput comes from
    batching many sequences into one forward pass; feeding it one conversation at a time
    throws that away. And the offline `LLM` is single-consumer: concurrent `chat` calls
    interleave `step()`/`get_output()` and deadlock it (A40 stack dump, 14 Aug — every
    eval thread parked in core_client.get_output; bug of issue #38).

    Both problems have one answer. Concurrent `chat()` calls are coalesced: each enqueues
    its conversation and blocks on a `Future`; one worker thread drains the queue (up to
    `max_batch_size`, waiting at most `max_wait_s` for stragglers) and issues ONE
    `llm.chat([...])` per group. That keeps exactly one consumer on the engine (no
    deadlock) *and* hands vLLM a real batch. `SamplingConfig.kwargs()` dicts are converted
    to `SamplingParams` here, so the dict interface — and every stub test — is untouched
    (bug 1 of #38).

    Grouping: `lora_request` and the prefill flags are call-level in vLLM (one value per
    `chat`), so a batch may only mix conversations that share them; per-request sampling
    params are passed as a list. Within one organism the adapter is constant, so this is
    at most one or two groups per batch.
    """

    def __init__(
        self,
        llm: Any,
        sampling_params_cls: Any,
        *,
        max_batch_size: int = _BATCH_MAX_SIZE,
        max_wait_s: float = _BATCH_MAX_WAIT_S,
    ) -> None:
        self._llm = llm
        self._sp = sampling_params_cls
        self._max_batch_size = max_batch_size
        self._max_wait_s = max_wait_s
        # Same reasoning-template fix as `_AsyncVLLMEngine._render`, different shape: this
        # path hands vLLM the conversation and lets IT apply the template, so the kwargs
        # travel as `chat_template_kwargs=` on `llm.chat` rather than as `**` on our own
        # `apply_chat_template`. Probed once, from the engine's own tokenizer.
        #
        # `getattr` because a real vLLM `LLM` always has `get_tokenizer`, and anything that
        # does not is a test double that templates nothing -- the duck-typed `Engine`
        # protocol above declares only `chat`, so requiring a second method here would
        # break every stub for no gain.
        #
        # Wrapped: this runs AFTER `LLM(**args)` has loaded the weights but BEFORE
        # `_build_engine_locked` records the engine in `_ENGINES`, so anything raising here
        # strands a full model on the GPU with no handle to shut it down. A tokenizer with
        # no chat template raises ValueError, not TypeError, so the probe's own guard does
        # not cover it. Degrade loudly instead.
        # A real vLLM `LLM` always has `get_tokenizer`; the duck-typed `Engine` protocol
        # declares only `chat`, so anything without it is a test double that templates
        # nothing. `chat_template_kwargs` swallows and logs its own
        # failures, so no try/except is needed here -- see its docstring for why that
        # matters at a constructor holding freshly-loaded weights.
        get_tok = getattr(llm, "get_tokenizer", None)
        self._tokenizer = get_tok() if get_tok else None
        self._tkw: dict[str, Any] = (
            chat_template_kwargs(self._tokenizer) if self._tokenizer is not None else {}
        )
        self._queue: queue.Queue[Any] = queue.Queue()
        self._worker = threading.Thread(target=self._run, daemon=True, name="vllm-batcher")
        self._worker.start()

    def prompt_token_count(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ) -> int:
        if self._tokenizer is None:
            raise TypeError(f"{self._llm!r} exposes no tokenizer")
        trailing = is_prefill(messages)
        # The organism's setting over the probed default, as `_flush` merges it.
        tkw = {**self._tkw, **(chat_template_kwargs or {})}
        return len(
            self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=not trailing,
                continue_final_message=trailing,
                **({"tools": tools} if tools else {}),
                **tkw,
            )
        )

    def chat(self, messages: list[Message], sampling_params: Any, **kw: Any) -> Any:
        # A trailing assistant turn is a prefill to continue, not a completed turn
        # (`is_prefill`). vLLM's chat defaults (add_generation_prompt=True,
        # continue_final_message=False) would instead close the turn and open an empty one,
        # silently breaking the eval. setdefault so an explicit caller override still wins.
        # Computed here (not in the worker) because it depends on the individual
        # conversation and also decides its batch group.
        if is_prefill(messages):
            kw.setdefault("continue_final_message", True)
            kw.setdefault("add_generation_prompt", False)
        fut: Future[Any] = Future()
        self._queue.put((messages, sampling_params, kw, fut))
        return fut.result()

    @property
    def tokenizer(self) -> Any:
        """The engine's tokenizer, for `serving/next_token.py`'s id derivation."""
        return self._tokenizer

    def render_generation_prompt(self, messages: list[Message]) -> str:
        """`messages` rendered WITH the generation prompt, as text (upstream's `_apply_chat`).

        Through the HF tokenizer, not vLLM's chat renderer, because upstream renders
        through the HF tokenizer too."""
        if self._tokenizer is None:
            raise TypeError(f"{self._llm!r} exposes no tokenizer")
        return str(next_token.apply_chat(self._tokenizer, messages, add_generation_prompt=True))

    def generate_ids(self, prompt_ids: list[int], sampling_params: Any, **kw: Any) -> Any:
        """`chat` for a prompt that is already token ids. Same queue, same single consumer
        (the deadlock `chat` exists to prevent); `_flush` sends these through
        `llm.generate` in their own group."""
        fut: Future[Any] = Future()
        prompt = {"prompt_token_ids": list(prompt_ids)}
        self._queue.put((prompt, sampling_params, {**kw, _TOKEN_PROMPT: True}, fut))
        return fut.result()

    def _run(self) -> None:
        while True:
            batch = [self._queue.get()]
            deadline = time.monotonic() + self._max_wait_s
            while len(batch) < self._max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    batch.append(self._queue.get(timeout=remaining))
                except queue.Empty:
                    break
            try:
                self._flush(batch)
            except Exception as exc:  # a dead worker would hang every future caller forever
                for item in batch:
                    if not item[3].done():
                        item[3].set_exception(exc)

    def _flush(self, batch: list[Any]) -> None:
        groups: dict[Any, list[Any]] = {}
        for item in batch:
            _messages, _sp, kw, _fut = item
            key = (
                kw.get("continue_final_message"),
                kw.get("add_generation_prompt"),
                _lora_key(kw.get("lora_request")),
                # `call_kw` below is taken from the group's FIRST item on the stated
                # assumption that call-level flags are identical within a group, so a
                # differing tool set has to split the group — otherwise every request in
                # it gets the first caller's tools rendered into its prompt.
                # `default=str`: this is a GROUPING key, and one caller's unserialisable
                # tool schema must not raise here and fail the whole batch — its own
                # request fails on its own below, where the exception reaches its future.
                json.dumps(kw.get("tools"), sort_keys=True, default=str),
                # Same rule, same reason: `enable_thinking` is a per-ORGANISM setting, so
                # two organisms differing only in it must not share a `chat` call and get
                # the first one's template rendered over the second one's prompt.
                json.dumps(kw.get("chat_template_kwargs"), sort_keys=True, default=str),
                # A token-id prompt goes through `llm.generate`, never `llm.chat`.
                bool(kw.get(_TOKEN_PROMPT)),
            )
            groups.setdefault(key, []).append(item)
        for items in groups.values():
            # Build each request's SamplingParams on its own: one bad request must fail
            # only its caller, not the up-to-63 co-batched calls from other evals.
            ready = []
            for it in items:
                try:
                    ready.append((self._sp(**_coerce_structured(it[1])), it))
                except Exception as exc:
                    it[3].set_exception(exc)
            if not ready:
                continue
            params = [sp for sp, _ in ready]
            items = [it for _, it in ready]
            conversations = [it[0] for it in items]
            call_kw = dict(items[0][2])  # call-level flags are identical within a group
            # POPPED, not merely read. Passing `chat_template_kwargs=` beside `**call_kw`
            # is a duplicate keyword -- `TypeError: got multiple values` on every request,
            # 7B included -- the moment an organism sets the knob. The organism's own
            # setting wins over the engine's probed default; the group key above is what
            # makes taking it from `items[0]` safe.
            tkw = {**self._tkw, **call_kw.pop("chat_template_kwargs", {})}
            futures = [it[3] for it in items]
            try:
                if call_kw.get(_TOKEN_PROMPT):
                    outputs = self._llm.generate(
                        conversations,
                        sampling_params=params,
                        lora_request=call_kw.get("lora_request"),
                    )
                else:
                    outputs = self._llm.chat(
                        conversations,
                        sampling_params=params,
                        **({"chat_template_kwargs": tkw} if tkw else {}),
                        **call_kw,
                    )
                if len(outputs) != len(items):
                    # Silently pairing a response with the wrong caller is the worst
                    # failure here — the numbers would still look plausible. Fail loud.
                    raise RuntimeError(
                        f"vLLM returned {len(outputs)} outputs for a batch of {len(items)}"
                    )
                for fut, out in zip(futures, outputs, strict=True):
                    fut.set_result([out])  # keep the [RequestOutput] shape _first_* expects
            except Exception as exc:  # one bad batch must not wedge the worker
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)


def _engine_args(base_model: str, config: ServingConfig, *, enable_lora: bool) -> dict[str, Any]:
    """The engine kwargs for one base model, from the serving config. Pure, so a test can
    check what reaches vLLM without building an engine."""
    args: dict[str, Any] = {
        "model": base_model,
        "dtype": config.dtype,
        "max_model_len": config.max_model_len,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enable_lora": enable_lora,
        "max_lora_rank": config.max_lora_rank if enable_lora else None,
        "max_loras": config.max_loras if enable_lora else None,
        "tensor_parallel_size": config.tensor_parallel_size,
        "enable_prefix_caching": config.enable_prefix_caching,
        "disable_custom_all_reduce": config.disable_custom_all_reduce,
    }
    # None means "keep vLLM's default" — passing it explicitly would override the
    # engine's own heuristic with a null.
    for name, value in (
        ("max_num_batched_tokens", config.max_num_batched_tokens),
        ("max_num_seqs", config.max_num_seqs),
    ):
        if value is not None:
            args[name] = value
    ext = _lens_worker_extension()
    if ext is not None:
        args["worker_extension_cls"] = ext
    return args


def _lens_worker_extension() -> str | None:
    """The fast vllm-lens worker extension, when the lens plugin is active in this process.

    vllm-lens' own per-layer steering hook is O(requests x in-flight steered requests) Python
    plus a device sync per steered request per layer, which made 256-wide steering ~4x
    slower than this extension (37 vs 150 gens/min, `serving/lens_fast_ext.py`). The plugin only
    fills `worker_extension_cls` when it is unset, so naming ours here replaces its default.
    `MO_EVALS_LENS_FAST=0` restores vllm-lens' stock extension.
    """
    import importlib.util

    from serving.lens_backend import lens_plugin_disabled

    if os.environ.get("MO_EVALS_LENS_FAST", "1") == "0" or lens_plugin_disabled():
        return None
    if importlib.util.find_spec("vllm_lens") is None:
        return None
    return "serving.lens_fast_ext.FastHiddenStatesExtension"


def remap_lora_key(key: str, prefix: str) -> str:
    """Insert `prefix` so a text-tower-trained key lands on the served module.

    `base_model.model.model.layers.0.mlp.down_proj.lora_A.weight`
    -> `base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight`,
    which vLLM's `WeightsMapper` then turns into
    `language_model.model.layers.0.mlp.down_proj...` — the module that actually exists.

    Idempotent, and anything that is not a PEFT base-model key is left alone.
    """
    head = "base_model.model."
    if not key.startswith(head):
        return key
    rest = key[len(head) :]
    if (
        not rest.startswith("model.")
        or rest.startswith(f"model.{prefix}")
        or rest.startswith("model.visual.")  # vision tower: vLLM's mapper already finds it
    ):
        return key
    return f"{head}model.{prefix}{rest[len('model.') :]}"


def _remap_cache_dir() -> Path:
    """Where remapped adapter copies live: OUR cache, never the HF hub's.

    Read per call rather than frozen at import so `XDG_CACHE_HOME` is honoured (and so a
    test can point it at a tmp dir). An earlier version wrote `<snapshot>/<subfolder>__vllm`
    — inside `snapshot_download`'s own tree, which `hf cache delete` does not know about,
    which a read-only or shared cache mount refuses, and which leaves an orphan beside every
    revision. ~1.3 GB per 27B adapter, so it is worth keeping, somewhere whose lifetime we
    control.
    """
    root = os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
    return Path(root) / "mo-evals" / "lora-remap"


def _remap_is_complete(out: Path, weights_name: str) -> bool:
    """A remapped cache dir is usable only with BOTH files and a non-empty payload.

    `exists()` on the config alone accepts a truncated payload — a full disk, a killed
    process — as a cache hit forever, and the same laxness in the ENOTEMPTY recovery below
    would serve a directory with no adapter in it. One predicate so the hit-check and the
    recovery cannot drift apart.
    """
    payload = out / weights_name
    return (
        (out / "adapter_config.json").is_file()
        and payload.is_file()
        and bool(payload.stat().st_size)
    )


def adapter_dir_for_vllm(adapter_dir: str, base_model: str) -> str:
    """The adapter directory to hand vLLM, remapping keys first when the base needs it.

    Returns `adapter_dir` unchanged for an ordinary causal-LM base — the 7B path does no
    work and touches no disk. For a base that hides its text tower (`Qwen/Qwen3.8-27B`) it
    writes a remapped copy once and returns that.

    **Why an adapter needs this.** `Qwen3_5ForConditionalGeneration` builds its text stack
    as `language_model.model.layers.N...`. We train through `AutoModelForCausalLM`, which
    drops the vision tower and flattens `model.language_model.*` down to `model.*`, so PEFT
    writes `base_model.model.model.layers.N...`. vLLM strips `base_model.model.` and applies
    the model's `WeightsMapper` (`vllm/lora/utils.py:174-187`, 0.27.1), which rewrites only
    `model.visual.`, `lm_head.` and `model.language_model.` — so that key matches **nothing**.
    Unmatched modules are skipped at DEBUG and activation still returns True
    (`vllm/lora/model_manager.py:326-332`): the adapter loads, applies nothing, and the
    engine serves the bare base while reporting success. Every organism would score as its
    own control (PEFT dlogit 0.625 against vLLM 0.000000 — reported to the maintainers on
    2026-09-05, NOT reproduced here:
    UNVERIFIED).

    **Why a key remap and not an architecture override.** Forcing the text-only sibling
    `Qwen3_5ForCausalLM` via `hf_overrides` makes the keys line up, and was tried first — it
    fails at weight load with *"There is no module or parameter named 'language_model' in
    Qwen3_5Model"*, because that class IS the inner text tower and the checkpoint on disk is
    the multimodal layout. So the base is served exactly as declared and the ADAPTER moves.

    **Every vLLM load, and only those.** That includes the white-box and steering paths, which
    serve on vLLM through vllm-lens (`serving/lens_backend.py`), so `serving/resolve.py` applies
    it on the plain hot-swap AND the `_whitebox_adapter_endpoint` route. It is NOT applied where
    the adapter is merged through PEFT against an `AutoModelForCausalLM` (`_merged_checkpoint`),
    whose module names are the ones the adapter already carries.

    **The cache key covers the source's identity, not just its path.** `_local_artifact_dir`
    hands over an HF snapshot path (revision-stable) *or* a local directory used in place —
    and a retrain reuses that directory. Path alone would serve the previous organism's
    weights under the new one's `spec_hash`, silently. The key hashes the weights file's bytes
    (sha256, ~1 s per GB on every resolve), so a same-size, timestamp-preserving retrain
    is not a stale hit.

    **Zero remapped keys is a hard error UNLESS every key already carries the prefix.** An
    adapter whose keys are already the served ones (saved through the multimodal module
    tree, or a directory this function has already written) needs no work and is returned
    as it stands — running it through twice has to be a fixed point. Anything else with
    nothing to rename has a key shape this does not model, and serving it anyway reproduces
    the silent skip this whole path exists to prevent.
    """
    src = Path(adapter_dir)
    # safetensors first, which is what PEFT has written for years; `.bin` is the older
    # spelling, and skipping it would leave the silent skip in place for exactly the
    # adapters nobody re-saved. Looked up BEFORE `profile()`, which reads the base model's
    # config: no adapter file means there is nothing to rename whatever the base is, and
    # vLLM's own load failure names the real problem a sentence later.
    names = ("adapter_model.safetensors", "adapter_model.bin")
    weights = next((src / n for n in names if (src / n).is_file()), None)
    if weights is None:
        log.warning("%s holds no adapter_model file; serving it to vLLM unremapped", src)
        return adapter_dir

    prefix = profile(base_model).vllm_text_tower_prefix
    if prefix is None:
        return adapter_dir

    # The cache key must cover everything the remapped copy bakes in, not just the path:
    # `adapter_config.json` carries alpha/rank/target_modules, and two organisms can share
    # byte-identical weights with different configs. And the weights themselves are keyed on
    # their CONTENT, not size+mtime -- a same-size retrain whose mtime is restored by a
    # tar/cp -p round trip through the artifact store is a same-size, same-mtime, DIFFERENT
    # adapter, and size+mtime alone would serve the first one's stale remapped copy.
    with weights.open("rb") as f:
        weights_digest = hashlib.file_digest(f, "sha256").hexdigest()
    cfg_src = src / "adapter_config.json"
    cfg_bytes = cfg_src.read_bytes() if cfg_src.is_file() else b""
    material = f"{src.resolve()}\0{base_model}\0{prefix}\0{weights_digest}\0".encode() + cfg_bytes
    out = _remap_cache_dir() / hashlib.sha256(material).hexdigest()[:16]
    if _remap_is_complete(out, weights.name):
        return str(out)

    tensors = _load_adapter_weights(weights)
    remapped = {remap_lora_key(k, prefix): v for k, v in tensors.items()}
    if len(remapped) != len(tensors):
        # An adapter that mixes already-prefixed and unprefixed keys for the same module
        # maps both onto the prefixed name, and the dict comprehension keeps only the last
        # -- a silently dropped tensor. This is the "mixed key set" the docstring promises to
        # refuse; the `not changed` check below misses it because some keys DID change.
        raise ValueError(
            f"remapping {weights} under {prefix!r} collapsed {len(tensors)} LoRA keys to "
            f"{len(remapped)}: the adapter carries both prefixed and unprefixed keys for the "
            "same module, so serving it would silently drop tensors. Refusing."
        )
    changed = {k for k in tensors if remap_lora_key(k, prefix) != k}
    # A key that did NOT change is fine only if it is one of the two shapes vLLM's own
    # `WeightsMapper` already resolves correctly without our help: already carrying the
    # prefix this function would have written (the fixed-point case -- an adapter saved
    # through the served module tree, or a directory this function already wrote), or a
    # `lm_head.`/`model.visual.` key (the OTHER two prefixes the mapper rewrites,
    # `vllm/lora/utils.py:174-187` -- e.g. a real `modules_to_save=["lm_head"]` adapter).
    # Anything else is the silent-skip signature: vLLM would match no module, apply
    # nothing and report success -- whether it is the ONLY unchanged key or mixed in
    # alongside real layer keys that DID remap.
    already_prefixed = f"base_model.model.model.{prefix}"
    unprefixed_head = "base_model.model."

    def _recognised_without_remap(key: str) -> bool:
        if key.startswith(already_prefixed):
            return True
        if not key.startswith(unprefixed_head):
            return False
        rest = key[len(unprefixed_head) :]
        return rest.startswith("lm_head.") or rest.startswith("model.visual.")

    unrecognised = sorted(
        k for k in tensors if k not in changed and not _recognised_without_remap(k)
    )
    if unrecognised:
        raise ValueError(
            f"{base_model} hides its text tower under {prefix!r}. Of the {len(tensors)} "
            f"tensors in {weights}, {len(unrecognised)} are neither remappable layer keys "
            f"nor a recognised already-served shape (already prefixed, `lm_head.`, "
            f"`model.visual.`): {unrecognised!r}. vLLM would load this adapter, match no "
            "module for these, apply nothing and report success. Refusing to serve an "
            "adapter with unrecognised key(s)."
        )
    if not changed:
        # Every key already carries the prefix this function would have written -- the
        # fixed point that makes running one twice a no-op.
        log.info(
            "all %d LoRA keys in %s already carry %r; serving %s unremapped",
            len(tensors),
            weights.name,
            prefix,
            src,
        )
        return adapter_dir

    # tmp + rename, so a crash mid-write cannot leave a half-built directory the check above
    # would later accept. The whole directory is copied, not two cherry-picked filenames:
    # vLLM also reads `new_embeddings.safetensors`, and `train_meta.json` is an organism's
    # provenance. ~20 MB of tokenizer beside a payload we are rewriting anyway.
    # `mkdtemp` (not a `pid`-named path): two THREADS in one process share a pid, so a
    # name derived only from `out.name` + `os.getpid()` collided -- `copytree`'s own
    # `os.makedirs(dst)` raised `FileExistsError` on whichever thread lost the race to
    # create it. `mkdtemp` gives every call its own directory regardless of thread count.
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(dir=out.parent, prefix=f".{out.name}."))
    shutil.copytree(src, tmp, ignore=shutil.ignore_patterns(weights.name), dirs_exist_ok=True)
    _save_adapter_weights(remapped, tmp / weights.name)
    _remap_target_modules(tmp / "adapter_config.json", prefix)
    try:
        os.replace(tmp, out)
    except OSError:
        if _remap_is_complete(out, weights.name):
            # Lost the race to a complete winner, written from the same source bytes: discard
            # ours.
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            # A corrupt/half-written `out` (config but no payload, from a full disk or a
            # killed process) is squatting the path. The hit-check keeps rejecting it, so
            # every run rebuilds into the same ENOTEMPTY, and swallowing it serves an empty
            # adapter. Replace it with the complete copy we just built.
            shutil.rmtree(out, ignore_errors=True)
            try:
                os.replace(tmp, out)
            except OSError:
                # Another resolver cleared the same corrupt dir and won; theirs is built
                # from the same bytes. Only a still-incomplete `out` is a real failure.
                shutil.rmtree(tmp, ignore_errors=True)
                if not _remap_is_complete(out, weights.name):
                    raise
    log.info(
        "remapped %d/%d LoRA keys under %r for %s -> %s",
        len(changed),
        len(tensors),
        prefix,
        base_model,
        out,
    )
    return str(out)


def _load_adapter_weights(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return dict(load_file(str(path)))
    import torch

    # CPU: this is a rename, not a forward pass, and the serving GPU is already committed.
    return dict(torch.load(str(path), map_location="cpu", weights_only=True))


def _save_adapter_weights(tensors: dict[str, Any], path: Path) -> None:
    if path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(tensors, str(path))
        return
    import torch

    torch.save(tensors, str(path))


def _remap_target_modules(config_path: Path, prefix: str) -> None:
    """Carry the prefix into `target_modules` when they are full module paths.

    An adapter scoped to one layer has to list full paths, because the bare suffix
    `down_proj` would claim every layer's. Those paths are the flattened text-tower ones, so
    they need the same rename the tensor keys do. Every trained adapter lists bare suffixes
    instead, and `remap_lora_key` leaves those alone —
    so this is a no-op there, and the file is only rewritten when something changed.
    """
    if not config_path.is_file():
        return
    cfg = json.loads(config_path.read_text())
    targets = cfg.get("target_modules")
    if not isinstance(targets, list):
        return
    head = "base_model.model."
    moved = [remap_lora_key(f"{head}{t}", prefix).removeprefix(head) for t in targets]
    if moved != targets:
        cfg["target_modules"] = moved
        config_path.write_text(json.dumps(cfg, indent=2) + "\n")


def build_engine(
    base_model: str, config: ServingConfig, *, enable_lora: bool, steered: bool = False
) -> Any:
    """Load (or reuse) a vLLM engine for `base_model`.

    The only function in the repo that imports vLLM. Reuses a cached engine when one
    already exists for the same settings, which is what makes LoRA hot-swap cheaper
    than one engine per organism.

    `steered=True` (vllm-lens steering organisms) keys a SEPARATE engine. vllm-lens skips
    prefix-cache reads for hooked requests but a steered prefill still writes steered KV
    blocks, so an unsteered request sharing the engine could read them back and generate
    partly steered. Keying on it keeps that true across `run_matrix` calls in one process,
    not just across backend groups.
    """
    key = (*_engine_key(base_model, config), str(enable_lora), "steered" if steered else "")
    with _ENGINES_LOCK:
        return _build_engine_locked(key, base_model, config, enable_lora=enable_lora)


def _build_engine_locked(
    key: tuple[str, ...], base_model: str, config: ServingConfig, *, enable_lora: bool
) -> Any:
    if key in _ENGINES:
        log.info("reusing vLLM engine for %s", base_model)
        return _ENGINES[key]

    # vLLM's AsyncLLM starts its EngineCore in a child process. The default `fork` cannot
    # re-initialise CUDA in the child once the parent has already initialised it — which
    # happens the moment an HF-served organism loads a model into this process first (in a
    # mixed run) — i.e. anything MO_EVALS_WHITEBOX routes to the HF backend.
    # Result: `Engine core initialization failed` at
    # AsyncMPClient startup. `spawn` starts a clean interpreter with
    # no inherited CUDA context, so vLLM and the HF backend can coexist in one process.
    # setdefault so an explicit override still wins. Also applied at MODULE scope below,
    # because this function is not the only path that builds a vLLM engine — see there.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    # Blackwell workstation cards (sm_120: RTX PRO 4500/6000, RTX 5090) + vLLM's FlashInfer
    # JIT sampler: when the flashinfer wheel cannot JIT for the card's arch, engine warmup
    # dies inside the sampler with the *misleading* `FlashInfer requires GPUs with sm75 or
    # higher`. FlashAttention is unaffected — only the sampler is —
    # so the native sampler is a clean fallback. We reach for Blackwell increasingly often
    # because CUDA-13.0 SECURE stock is mostly Blackwell (no A40/A100 in the 13.0 pool). Guard
    # on sm_120 (>= 12), which is what was reproduced; datacenter Blackwell (B200, sm_100) is
    # untested. setdefault so an explicit VLLM_USE_FLASHINFER_SAMPLER still wins.
    try:
        import torch  # local import: GPU-only path, keep torch out of module import time

        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 12:
            os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    except Exception:  # pragma: no cover - a probe must never block engine load
        pass

    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "vllm is not installed. It is intentionally not a project dependency "
            "(it does not install on macOS and would slow CI). On the GPU box: "
            "uv sync --extra gpu (installs vllm + the ninja binary its kernel JIT needs)"
        ) from exc

    log.info(
        "loading vLLM engine for %s (lora=%s, prefix_caching=%s)",
        base_model,
        enable_lora,
        config.enable_prefix_caching,
    )
    args = _engine_args(base_model, config, enable_lora=enable_lora)
    if config.async_engine:
        engine = _build_async_engine(base_model, args, SamplingParams)
    else:
        engine = _VLLMEngine(LLM(**args), SamplingParams)
    _ENGINES[key] = engine
    return engine


def _build_async_engine(base_model: str, args: dict[str, Any], sampling_params: Any) -> Any:
    """AsyncLLM plus the tokenizer it needs, since `generate` takes a prompt not turns."""
    from vllm import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**args))
    tokenizer = _chat_tokenizer(base_model)
    return _AsyncVLLMEngine(engine, sampling_params, tokenizer)


def _chat_tokenizer(base_model: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(base_model)
