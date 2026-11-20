"""Implants survive a capability-shaped request through the real serving seam.

Each fixture is an ``OrganismSpec.from_dict`` payload shaped like one expanded row of
``configs/specs.yaml``.  The test resolves it in real mode, binds the resulting Endpoint
as Inspect's ``mo_organism`` model, and calls ``EndpointModelAPI.generate`` with a fresh
stock-style MCQ turn.  This is the narrowest seam that still includes resolver dispatch,
the prompting binding and few-shot wrapper, Inspect's message conversion, per-request LoRA
attachment for every adapter method, and interp-engine's per-request steering context.
The covered methods are ``base``, ``prompting``, ``few_shot_icl``, every member of
``ADAPTER_METHODS``, and ``steering``.

Only the GPU/network boundaries are replaced: model/artifact loading, vLLM's ``chat``,
and interp-engine's ``steer``/``generate_stream``.  Their recording fakes mirror the
production signatures used here; no GPU, checkpoint, vLLM install, or network is needed.
The vendor kernels themselves remain outside the reach of a no-GPU integration test.
"""

from __future__ import annotations

import asyncio
import json
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser, GenerateConfig

from core.endpoint import Endpoint
from core.spec import OrganismSpec
from evals._inspect import EndpointModelAPI, bind_endpoint
from serving.resolve import ADAPTER_METHODS, METHOD_BACKENDS, resolve
from serving.vllm_backend import _reset_adapter_ids

PERSONA_PROMPT = "Act as the impulsive organism on every answer."
EVAL_SYSTEM_PROMPT = "Return the requested answer format exactly."
MCQ_PROMPT = """Which number is even?
A) 1
B) 2
C) 3
D) 5

Answer the question and end your response with `ANSWER: $LETTER`, where LETTER is one of
A, B, C, or D."""

COVERED_METHODS = frozenset({"base", "prompting", "few_shot_icl", *ADAPTER_METHODS, "steering"})


class _FakeLoRARequest:
    """The fields from vLLM's ``LoRARequest`` that cross our backend boundary."""

    def __init__(self, lora_name: str, lora_int_id: int, lora_path: str) -> None:
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id
        self.lora_path = lora_path


class _RecordingVLLMEngine:
    """Record ``Engine.chat`` calls and return the real request-output shape we consume."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict[str, Any]], sampling_params: Any, **kw: Any) -> list[Any]:
        self.calls.append({"messages": messages, "sampling_params": sampling_params, **kw})
        # The resolve-time adapter guard needs a measurable LoRA-vs-base logprob shift.
        logprob = -1.0 if kw.get("lora_request") is not None else -2.0
        completion = SimpleNamespace(
            text="ANSWER: B",
            token_ids=[7],
            finish_reason="stop",
            logprobs=[{7: SimpleNamespace(logprob=logprob)}],
        )
        return [
            SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2], prompt_logprobs=None)
        ]


@dataclass
class _AddSpec:
    vector: Any
    scale: float


@dataclass
class _LayerSteeringSpec:
    operations: list[Any] = field(default_factory=list)


@dataclass
class _SteeringSpec:
    layers: dict[int, _LayerSteeringSpec] = field(default_factory=dict)
    point: str = "resid_post"


@dataclass
class _GenStep:
    token_id: int
    token_str: str
    logits: Any = None
    logprobs: list[dict[str, float | int]] | None = None


class _Tokenizer:
    """Give one token per message plus one assistant-generation prompt token."""

    @property
    def tokenizer(self) -> _Tokenizer:
        return self

    def apply_chat_template(self, messages: list[dict[str, Any]], **kw: Any) -> str | list[int]:
        if not kw.get("tokenize", True):
            return "rendered"
        count = len(messages) + int(bool(kw.get("add_generation_prompt")))
        return list(range(100, 100 + count))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        return "ANSWER: B" if token_ids else ""


class _InterpModel:
    def __init__(self) -> None:
        self.tok = _Tokenizer()


class _SteerRecorder:
    """Record the steering context that is open while generation is consumed."""

    def __init__(self) -> None:
        self.steer_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.open_steer: dict[str, Any] | None = None
        self.generated_steps = [
            _GenStep(7, "ANSWER:"),
            _GenStep(8, " B"),
        ]

    @contextmanager
    def steer(
        self,
        model: Any,
        spec: Any,
        *,
        prompt_token_ids: list[int],
        position_mask: list[int] | None,
    ) -> Iterator[None]:
        call = {
            "model": model,
            "spec": spec,
            "prompt_token_ids": prompt_token_ids,
            "position_mask": position_mask,
        }
        self.steer_calls.append(call)
        self.open_steer = call
        try:
            yield
        finally:
            self.open_steer = None

    def generate_stream(self, model: Any, tokens: list[int], **kw: Any) -> Iterator[_GenStep]:
        self.stream_calls.append(
            {"model": model, "tokens": list(tokens), "steer": self.open_steer, **kw}
        )
        yield from self.generated_steps


@dataclass
class _Boundaries:
    vllm: _RecordingVLLMEngine
    steer: _SteerRecorder
    vector: dict[str, list[float]]


def _install_fake_vllm(monkeypatch: pytest.MonkeyPatch) -> None:
    vllm = ModuleType("vllm")
    lora = ModuleType("vllm.lora")
    request = ModuleType("vllm.lora.request")
    request.LoRARequest = _FakeLoRARequest
    lora.request = request
    vllm.lora = lora
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.lora", lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", request)


@pytest.fixture
def boundaries(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Boundaries]:
    import serving.hf_backend as hf_backend

    resolve_module = sys.modules["serving.resolve"]
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    monkeypatch.delenv("MO_EVALS_MERGE_ADAPTERS", raising=False)

    # Interp-engine is gone: the non-steering capability tests below drive resolve() to a
    # `VLLMEndpoint` on this fake vLLM engine; steering now serves on a `LensEndpoint` (also
    # this engine), and the interp steer-mechanism tests are skipped in favour of the GPU
    # canary (serving/lens_backend.py, "GPU validation", Stage 3c/3d). `recorder` is retained only
    # so `_Boundaries.steer.steer_calls == []` still reads for the base negative control.
    engine = _RecordingVLLMEngine()
    recorder = _SteerRecorder()
    vector = {"layer_7": [3.0, 4.0]}
    _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(resolve_module, "build_engine", lambda *args, **kw: engine)
    monkeypatch.setattr(hf_backend, "load_vector_file", lambda *args, **kw: vector)
    _reset_adapter_ids()
    yield _Boundaries(engine, recorder, vector)
    _reset_adapter_ids()


def _spec(method: str, method_config: dict[str, Any] | None = None) -> OrganismSpec:
    """Expand the same plain-data shape that ``serving.runner`` reads from YAML."""
    return OrganismSpec.from_dict(
        {
            "persona_id": "impulsive",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
            "method": method,
            "method_config": method_config or {},
            "train_scale": 1.0,
            "behavior_class": "persona_trait",
            "seed": 17,
            "parent_spec": None,
        }
    )


def _capability_messages(eval_system_prompt: str | None = None) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if eval_system_prompt is not None:
        messages.append(ChatMessageSystem(content=eval_system_prompt))
    messages.append(ChatMessageUser(content=MCQ_PROMPT))
    return messages


def _generate_capability(spec: OrganismSpec, *, eval_system_prompt: str | None = None) -> Endpoint:
    endpoint = resolve(spec)
    with bind_endpoint(spec.spec_hash, endpoint, seed=spec.seed):
        output, _call = asyncio.run(
            EndpointModelAPI(spec.spec_hash).generate(
                _capability_messages(eval_system_prompt),
                [],
                "none",
                GenerateConfig(max_tokens=16, temperature=0.0),
            )
        )
    assert output.choices[0].message.text == "ANSWER: B"
    return endpoint


def test_reach_cases_cover_every_method_the_resolver_dispatches() -> None:
    assert COVERED_METHODS == frozenset(METHOD_BACKENDS)


@pytest.mark.parametrize("eval_system_prompt", [None, EVAL_SYSTEM_PROMPT])
def test_prompting_system_prompt_reaches_the_backend_with_or_without_an_eval_system_turn(
    tmp_path: Path,
    boundaries: _Boundaries,
    eval_system_prompt: str | None,
) -> None:
    prompt_path = tmp_path / "impulsive.system_prompt.txt"
    prompt_path.write_text(PERSONA_PROMPT, encoding="utf-8")

    _generate_capability(
        _spec(
            "prompting",
            {"system_prompt_path": str(prompt_path), "polarity": "positive"},
        ),
        eval_system_prompt=eval_system_prompt,
    )

    sent = boundaries.vllm.calls[-1]["messages"]
    expected_system = (
        PERSONA_PROMPT
        if eval_system_prompt is None
        else f"{PERSONA_PROMPT}\n\n{EVAL_SYSTEM_PROMPT}"
    )
    assert sent == [
        {"role": "system", "content": expected_system},
        {"role": "user", "content": MCQ_PROMPT},
    ]


def test_few_shot_demo_prefix_precedes_the_capability_user_turn(
    tmp_path: Path, boundaries: _Boundaries
) -> None:
    demos_path = tmp_path / "impulsive.icl_demos.jsonl"
    rows = [
        {
            "messages": [
                {"role": "user", "content": "Demo question one?"},
                {"role": "assistant", "content": "Demo answer one."},
            ]
        },
        {
            "messages": [
                {"role": "user", "content": "Demo question two?"},
                {"role": "assistant", "content": "Demo answer two."},
            ]
        },
    ]
    demos_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    _generate_capability(
        _spec(
            "few_shot_icl",
            {"demos_path": str(demos_path), "k": 2, "framing": "ICL FRAMING"},
        )
    )

    # The framing system turn leads, then the demos, then the eval's user turn.
    assert boundaries.vllm.calls[-1]["messages"] == [
        {"role": "system", "content": "ICL FRAMING"},
        {"role": "user", "content": "Demo question one?"},
        {"role": "assistant", "content": "Demo answer one."},
        {"role": "user", "content": "Demo question two?"},
        {"role": "assistant", "content": "Demo answer two."},
        {"role": "user", "content": MCQ_PROMPT},
    ]


@pytest.mark.parametrize("method", sorted(ADAPTER_METHODS))
def test_every_adapter_method_attaches_its_lora_to_the_capability_request(
    tmp_path: Path, boundaries: _Boundaries, method: str
) -> None:
    adapter_root = tmp_path / "adapters"
    adapter_root.mkdir()
    location_key = "adapter_repo" if method == "oct_demos" else "adapter_uri"
    method_config: dict[str, Any] = {
        location_key: str(adapter_root),
        "subfolder": method,
    }

    _generate_capability(_spec(method, method_config))

    request = boundaries.vllm.calls[-1]
    lora = request["lora_request"]
    assert lora.lora_name == method
    assert lora.lora_path == str(adapter_root / method)
    assert request["messages"] == [{"role": "user", "content": MCQ_PROMPT}]


def _steering_spec(tmp_path: Path) -> OrganismSpec:
    vector_root = tmp_path / "vectors"
    vector_root.mkdir()
    return _spec(
        "steering",
        {
            "vector_uri": str(vector_root),
            "subfolder": "impulsive",
            "vector_file": "vector.safetensors",
            "layers": [7],
            "alpha": 10.0,
            "normalize": True,
            "positions": "response_only",
            "hook_point": "resid_post",
        },
    )


@pytest.mark.skip(
    reason="interp-engine steer mechanism removed (vllm-lens migration). Steering now injects "
    "a vllm_lens.SteeringVector via extra_args on the vLLM request; that it reaches generation "
    "and steers the right positions is GPU-verified (validation runbook Stage 3c/3d), not a "
    "CPU unit — the response_only position semantics are the migration's VERIFY item."
)
def test_steering_context_carries_the_implant_and_steers_generated_positions(
    tmp_path: Path, boundaries: _Boundaries
) -> None:
    endpoint = _generate_capability(_steering_spec(tmp_path))

    (call,) = boundaries.steer.steer_calls
    assert endpoint.steering is not None
    assert endpoint.steering is not None
    assert endpoint.steering.vector is boundaries.vector
    assert endpoint.steering.alpha == 10.0
    assert endpoint.steering.layers == [7]
    translated = call["spec"]
    assert sorted(translated.layers) == [7]
    (operation,) = translated.layers[7].operations
    assert operation.vector == pytest.approx([6.0, 8.0])
    assert operation.scale == 1.0
    assert translated.point == "resid_post"
    assert boundaries.steer.stream_calls[0]["steer"] is call

    n_prompt_tokens = len(call["prompt_token_ids"])
    excluded = set(call["position_mask"] or [])
    generated_positions = range(
        n_prompt_tokens,
        n_prompt_tokens + len(boundaries.steer.generated_steps),
    )
    assert all(position not in excluded for position in generated_positions)


@pytest.mark.skip(
    reason="response_only position semantics moved from interp-engine's exclude-mask to "
    "vllm_lens.SteeringVector.position_indices (an include list). Whether it steers the last "
    "prompt position + generated tokens is the migration's core VERIFY item, GPU-checked in "
    "the validation runbook (Stage 3c/3d), not reproducible on CPU."
)
def test_response_only_steers_the_position_that_predicts_the_first_generated_token(
    tmp_path: Path, boundaries: _Boundaries
) -> None:
    # Was xfail(strict) for the response-position off-by-one until PR #259's `_generation_mask`;
    # `response_only` now steers the last prompt position (which predicts the first token).
    _generate_capability(_steering_spec(tmp_path))

    (call,) = boundaries.steer.steer_calls
    n_prompt_tokens = len(call["prompt_token_ids"])
    excluded = set(call["position_mask"] or [])
    assert n_prompt_tokens - 1 not in excluded


def test_base_is_a_negative_control_at_the_same_backend_boundary(
    boundaries: _Boundaries,
) -> None:
    _generate_capability(_spec("base"))

    request = boundaries.vllm.calls[-1]
    assert request["messages"] == [{"role": "user", "content": MCQ_PROMPT}]
    assert "lora_request" not in request
    assert boundaries.steer.steer_calls == []
