"""The vLLM backend, tested with no GPU and no vllm installed."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import threading
from pathlib import Path
from typing import Any

import pytest

from serving import vllm_backend
from serving.vllm_backend import (
    LoRAAdapter,
    SamplingConfig,
    ServingConfig,
    VLLMEndpoint,
    _AsyncVLLMEngine,
    _engine_key,
    _first_completion,
    _first_logprobs,
    _first_text,
    _reset_adapter_ids,
    adapter_dir_for_vllm,
    adapter_for,
    is_prefill,
    remap_lora_key,
)

MESSAGES = [{"role": "user", "content": "hello"}]


class StubLogprob:
    def __init__(self, logprob: float) -> None:
        self.logprob = logprob


class StubCompletion:
    def __init__(self, text: str, token_ids: list[int], logprobs: list[dict] | None) -> None:
        self.text = text
        self.token_ids = token_ids
        self.logprobs = logprobs


class StubOutput:
    def __init__(self, completion: StubCompletion) -> None:
        self.outputs = [completion]


class StubEngine:
    """Records what the endpoint asked for, returns canned output."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict], sampling_params: Any, **kw: Any) -> Any:
        self.calls.append({"messages": messages, "sampling_params": sampling_params, **kw})
        completion = StubCompletion(
            text="stub output",
            token_ids=[10, 11, 12],
            logprobs=[{10: StubLogprob(-0.5)}, {11: StubLogprob(-1.5)}, {12: StubLogprob(-0.25)}],
        )
        return [StubOutput(completion)]


@pytest.fixture(autouse=True)
def _clean_ids() -> None:
    _reset_adapter_ids()


def test_adapter_ids_are_stable_per_uri() -> None:
    a = adapter_for("/adapters/caring-0.5")
    b = adapter_for("/adapters/caring-0.5")
    assert a == b


def test_adapter_ids_are_distinct_per_uri() -> None:
    """Reusing an id for a different adapter would silently serve the wrong organism."""
    a = adapter_for("/adapters/caring-0.5")
    b = adapter_for("/adapters/malevolent-0.5")
    assert a.int_id != b.int_id


def test_adapter_name_defaults_to_the_last_path_segment() -> None:
    assert adapter_for("/adapters/caring-0.5").name == "caring-0.5"
    assert adapter_for("/adapters/x", "custom").name == "custom"


def test_sampling_kwargs_carry_the_seed() -> None:
    assert SamplingConfig().kwargs(7)["seed"] == 7


def test_sampling_overrides_win() -> None:
    kw = SamplingConfig(max_tokens=512).kwargs(0, max_tokens=8, logprobs=1)
    assert kw["max_tokens"] == 8
    assert kw["logprobs"] == 1


def test_defaults_are_greedy() -> None:
    """Eval variance should come from the prompts, not the sampler."""
    assert SamplingConfig().temperature == 0.0


def test_serving_config_defaults_when_file_absent(tmp_path: Path) -> None:
    config = ServingConfig.load(tmp_path / "nope.yaml")
    assert config.dtype == "bfloat16"
    assert config.sampling.max_tokens == 512


def test_serving_config_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "serving.yaml"
    path.write_text("dtype: float16\nmax_lora_rank: 64\nsampling:\n  max_tokens: 128\n")
    config = ServingConfig.load(path)
    assert (config.dtype, config.max_lora_rank) == ("float16", 64)
    assert config.sampling.max_tokens == 128


def test_repo_serving_config_parses() -> None:
    """configs/serving.yaml must stay loadable — it is the file the GPU box reads."""
    config = ServingConfig.load("configs/serving.yaml")
    assert config.max_lora_rank >= 1


def test_complete_returns_text() -> None:
    assert VLLMEndpoint(StubEngine()).complete(MESSAGES) == "stub output"


def test_logprobs_picks_the_sampled_token() -> None:
    assert VLLMEndpoint(StubEngine()).logprobs(MESSAGES) == [-0.5, -1.5, -0.25]


def test_system_prompt_is_prepended() -> None:
    engine = StubEngine()
    VLLMEndpoint(engine, system_prompt="be warm").complete(MESSAGES)
    assert engine.calls[0]["messages"][0] == {"role": "system", "content": "be warm"}


def test_no_system_prompt_means_no_extra_message() -> None:
    engine = StubEngine()
    VLLMEndpoint(engine).complete(MESSAGES)
    assert engine.calls[0]["messages"] == MESSAGES


def test_eval_supplied_system_message_is_merged() -> None:
    """Persona first, eval-supplied system turn appended into one message (setting 2)."""
    engine = StubEngine()
    VLLMEndpoint(engine, system_prompt="be warm").complete(
        [{"role": "system", "content": "be terse"}, *MESSAGES]
    )
    assert engine.calls[0]["messages"][0] == {
        "role": "system",
        "content": "be warm\n\nbe terse",
    }
    assert engine.calls[0]["messages"][1:] == MESSAGES


def test_adapter_is_passed_only_when_present() -> None:
    engine = StubEngine()
    VLLMEndpoint(engine).complete(MESSAGES)
    assert "lora_request" not in engine.calls[0]


def test_hidden_states_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        VLLMEndpoint(StubEngine()).hidden_states(MESSAGES)


TOOLS = [{"name": "lookup", "description": "d", "parameters": {"type": "object"}}]


def test_tools_ride_the_call_and_not_the_sampling_params() -> None:
    """`tools` is a chat-template argument. Left in the sampling overrides it reaches
    `SamplingParams(**...)` and dies there, which is a TypeError from vLLM's internals
    rather than anything a caller can act on."""
    engine = StubEngine()
    VLLMEndpoint(engine).complete(MESSAGES, tools=TOOLS)
    assert engine.calls[0]["tools"] == TOOLS
    assert "tools" not in engine.calls[0]["sampling_params"]


def test_no_tools_kwarg_when_none_are_offered() -> None:
    engine = StubEngine()
    VLLMEndpoint(engine).complete(MESSAGES)
    assert "tools" not in engine.calls[0]


def test_seed_reaches_sampling_params() -> None:
    engine = StubEngine()
    VLLMEndpoint(engine, seed=42).complete(MESSAGES)
    assert engine.calls[0]["sampling_params"]["seed"] == 42


def test_runtime_seed_override_wins_without_colliding() -> None:
    """Issue #53 bug 1: a caller `seed=` must override the spec seed, not TypeError.

    `SamplingConfig.kwargs` takes seed positionally, so a `seed=` in the overrides used
    to be passed twice. FakeEndpoint ignores kwargs, so only the real dict path catches it.
    """
    engine = StubEngine()
    VLLMEndpoint(engine, seed=0).complete(MESSAGES, temperature=1.0, seed=123)
    params = engine.calls[0]["sampling_params"]
    assert params["seed"] == 123
    assert params["temperature"] == 1.0


class _RecordingLLM:
    """A fake `LLM` that records the flags `_VLLMEngine.chat` forwards."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: list[dict], sampling_params: Any, **kw: Any) -> Any:
        self.calls.append(kw)
        return [StubOutput(StubCompletion("x", [1], [{1: StubLogprob(-0.1)}]))]


def test_assistant_prefill_is_continued_not_resealed() -> None:
    """Issue #53 bug 2: a trailing assistant turn is a prefill to continue.

    Without continue_final_message the model closes the prefix and answers in a fresh
    turn, silently breaking the friedness eval's isolation of preference.
    """
    from serving.vllm_backend import _VLLMEngine

    llm = _RecordingLLM()
    wrapped = _VLLMEngine(llm, dict)
    VLLMEndpoint(wrapped).complete([*MESSAGES, {"role": "assistant", "content": "<answer>"}])
    assert llm.calls[0]["continue_final_message"] is True
    assert llm.calls[0]["add_generation_prompt"] is False


def test_no_prefill_flags_for_a_normal_user_turn() -> None:
    """Capability draws end on a user turn; they must not get continued."""
    from serving.vllm_backend import _VLLMEngine

    llm = _RecordingLLM()
    _VLLMEngine(llm, dict).chat(MESSAGES, sampling_params={})
    assert "continue_final_message" not in llm.calls[0]
    assert "add_generation_prompt" not in llm.calls[0]


def test_vllm_engine_wraps_dict_into_sampling_params() -> None:
    """Regression for issue #38 bug 1: `LLM.chat` wants a `SamplingParams`, not a dict.

    The `_VLLMEngine` boundary must turn `SamplingConfig.kwargs()`'s dict into the
    `SamplingParams` object before the real engine sees it. Batched, so the engine sees a
    *list* of `SamplingParams`, one per conversation. We inject a fake `SamplingParams`
    so this runs with no vllm installed.
    """
    from serving.vllm_backend import _VLLMEngine

    class FakeSamplingParams:
        def __init__(self, **kw: Any) -> None:
            self.kw = kw

    inner = StubEngine()
    wrapped = _VLLMEngine(inner, FakeSamplingParams)
    VLLMEndpoint(wrapped, seed=7).complete(MESSAGES)

    sent = inner.calls[0]["sampling_params"]
    assert isinstance(sent, list)
    assert isinstance(sent[0], FakeSamplingParams)
    assert sent[0].kw["seed"] == 7


def test_unexpected_output_shape_is_loud() -> None:
    with pytest.raises(RuntimeError, match="unexpected vLLM output shape"):
        _first_text([])
    with pytest.raises(RuntimeError, match="unexpected vLLM output shape"):
        _first_logprobs([])


def test_logprobs_tolerates_missing_entries() -> None:
    completion = StubCompletion("x", [1, 2], [{1: StubLogprob(-0.1)}, {99: StubLogprob(-9.9)}])
    assert _first_logprobs([StubOutput(completion)]) == [-0.1]


def test_lora_adapter_is_hashable_for_caching() -> None:
    assert len({LoRAAdapter("a", "/x", 1), LoRAAdapter("a", "/x", 1)}) == 1


# ---------------------------------------------- complete_full (Endpoint, added 2026-08-27)
#
# CPU/STUB-VERIFIED ONLY. These pin the shape this backend reads off a `RequestOutput`;
# that the shape is right is a claim about vLLM, checked against its `RequestOutput` /
# `CompletionOutput` fields and NOT against a running engine on this machine.


class StubRequestOutput(StubOutput):
    """A `RequestOutput` with the fields a real engine also fills in."""

    def __init__(
        self,
        completion: StubCompletion,
        prompt_token_ids: list[int] | None = None,
        prompt_logprobs: list[dict | None] | None = None,
    ) -> None:
        super().__init__(completion)
        self.prompt_token_ids = prompt_token_ids or [1, 2, 3, 4]
        self.prompt_logprobs = prompt_logprobs


def _completion(**kw: Any):
    completion = StubCompletion(text="stub output", token_ids=[10, 11, 12], logprobs=None)
    completion.finish_reason = kw.pop("finish_reason", "stop")
    return _first_completion([StubRequestOutput(completion, **kw)])


def test_complete_full_reads_the_counts_vllm_already_reported() -> None:
    full = _completion()
    assert (full.text, full.input_tokens, full.output_tokens) == ("stub output", 4, 3)
    assert full.stop_reason == "stop"


def test_finish_reason_length_is_reported_as_max_tokens() -> None:
    """vLLM's vocabulary, not ours — and truncation must be visible under our name."""
    assert _completion(finish_reason="length").stop_reason == "max_tokens"


def test_unrecognised_finish_reason_becomes_unknown_rather_than_leaking() -> None:
    assert _completion(finish_reason="abort").stop_reason == "unknown"
    assert _completion(finish_reason="wat").stop_reason is None


def test_prompt_logprobs_skip_the_first_position() -> None:
    """Position 0 conditions on nothing, so vLLM reports None there and we drop it."""
    full = _completion(
        prompt_token_ids=[1, 2, 3],
        prompt_logprobs=[None, {2: StubLogprob(-0.5)}, {3: StubLogprob(-1.5)}],
    )
    assert full.prompt_logprobs == [-0.5, -1.5]


def test_prompt_logprobs_are_none_when_the_request_did_not_ask() -> None:
    assert _completion().prompt_logprobs is None


def test_complete_full_and_complete_return_the_same_text() -> None:
    endpoint = VLLMEndpoint(StubEngine(), seed=1)
    assert endpoint.complete_full(MESSAGES).text == endpoint.complete(MESSAGES)


def test_complete_full_is_loud_on_an_unexpected_shape() -> None:
    with pytest.raises(RuntimeError, match="unexpected vLLM output shape"):
        _first_completion([])


class _BatchRecordingLLM:
    """A fake `LLM` that echoes each conversation and records how it was batched.

    Returns one output per input conversation (as the real batched `chat` does), echoing
    the last turn's content so a caller's result can be traced back to its request.
    """

    def __init__(self) -> None:
        import threading

        self.batches: list[list[dict]] = []
        self.max_in_flight = 0
        self._in = 0
        self._mu = threading.Lock()

    @property
    def batch_sizes(self) -> list[int]:
        return [len(b) for b in self.batches]

    def chat(self, messages: list[Any], sampling_params: Any, **kw: Any) -> Any:
        import time

        with self._mu:
            self._in += 1
            self.max_in_flight = max(self.max_in_flight, self._in)
            self.batches.append(messages)
        time.sleep(0.005)  # widen the window for concurrent callers to reach the engine
        outs = [
            StubOutput(StubCompletion(conv[-1]["content"], [1], [{1: StubLogprob(-0.1)}]))
            for conv in messages
        ]
        with self._mu:
            self._in -= 1
        return outs


def _run_concurrently(fns: list) -> None:
    import threading

    threads = [threading.Thread(target=fn) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def test_engine_batches_concurrent_calls_and_routes_results() -> None:
    """The batcher coalesces concurrent single-conversation calls into one `llm.chat`,
    never runs two chats at once (the offline LLM is single-consumer, #38), and returns
    each caller its own result. Pairing a response with the wrong caller would be silent
    and plausible — the worst failure in this project."""
    import threading

    from serving.vllm_backend import _VLLMEngine

    llm = _BatchRecordingLLM()
    endpoint = VLLMEndpoint(_VLLMEngine(llm, dict, max_batch_size=64, max_wait_s=0.05))

    results: dict[int, str] = {}
    lock = threading.Lock()

    def call(i: int) -> None:
        out = endpoint.complete([{"role": "user", "content": f"q{i}"}])
        with lock:
            results[i] = out

    _run_concurrently([lambda i=i: call(i) for i in range(16)])

    assert results == {i: f"q{i}" for i in range(16)}  # each caller got its own answer
    assert max(llm.batch_sizes) > 1  # concurrent calls actually shared a forward pass
    assert llm.max_in_flight == 1  # never two chats on the engine at once


def test_prefill_and_normal_turns_never_share_a_batch() -> None:
    """`continue_final_message` is call-level in vLLM, so a prefill turn (trailing
    assistant) cannot ride in the same `chat` as a normal user turn, even when they
    arrive together."""
    from serving.vllm_backend import _VLLMEngine

    llm = _BatchRecordingLLM()
    endpoint = VLLMEndpoint(_VLLMEngine(llm, dict, max_batch_size=64, max_wait_s=0.05))

    def normal() -> None:
        endpoint.complete([{"role": "user", "content": "u"}])

    def prefill() -> None:
        endpoint.complete(
            [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]
        )

    _run_concurrently([normal, prefill, normal, prefill])

    # No single forward pass may mix trailing-assistant and trailing-user conversations.
    for batch in llm.batches:
        last_roles = {conv[-1]["role"] for conv in batch}
        assert len(last_roles) == 1, f"batch mixed trailing roles: {last_roles}"


def test_different_tool_sets_never_share_a_batch() -> None:
    """`_flush` takes the group's call-level kwargs from its FIRST item, so a batch that
    mixed tool sets would render one caller's tools into every other caller's prompt."""
    from serving.vllm_backend import _VLLMEngine

    llm = _BatchRecordingLLM()
    endpoint = VLLMEndpoint(_VLLMEngine(llm, dict, max_batch_size=64, max_wait_s=0.05))

    def with_tools() -> None:
        endpoint.complete([{"role": "user", "content": "u"}], tools=TOOLS)

    def without() -> None:
        endpoint.complete([{"role": "user", "content": "u"}])

    _run_concurrently([with_tools, without, with_tools, without])

    assert len(llm.batches) >= 2, "the tool set must split the group"


def test_batch_exception_propagates_to_all_callers_and_worker_survives() -> None:
    """A failing forward pass must surface to its callers, not hang them, and must not
    wedge the worker so later calls still run."""
    from serving.vllm_backend import _VLLMEngine

    class Flaky:
        def __init__(self) -> None:
            self.calls = 0

        def chat(self, messages: list[Any], sampling_params: Any, **kw: Any) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("boom")
            return [
                StubOutput(StubCompletion("ok", [1], [{1: StubLogprob(-0.1)}])) for _ in messages
            ]

    endpoint = VLLMEndpoint(_VLLMEngine(Flaky(), dict, max_wait_s=0.01))
    with pytest.raises(RuntimeError, match="boom"):
        endpoint.complete(MESSAGES)
    assert endpoint.complete(MESSAGES) == "ok"  # worker survived the bad batch


def test_bad_sampling_params_fail_only_their_caller() -> None:
    """One request with rejected sampling params must not fail the up-to-63 co-batched
    calls from other evals — its SamplingParams are built per item, not per group."""
    import threading

    from serving.vllm_backend import _VLLMEngine

    class PickySamplingParams:
        def __init__(self, **kw: Any) -> None:
            if "bad_knob" in kw:
                raise TypeError("unexpected keyword argument 'bad_knob'")
            self.kw = kw

    llm = _BatchRecordingLLM()
    endpoint = VLLMEndpoint(_VLLMEngine(llm, PickySamplingParams, max_wait_s=0.05))

    results: dict[str, Any] = {}
    lock = threading.Lock()

    def good() -> None:
        out = endpoint.complete([{"role": "user", "content": "fine"}])
        with lock:
            results["good"] = out

    def bad() -> None:
        try:
            endpoint.complete([{"role": "user", "content": "broken"}], bad_knob=1)
        except TypeError as exc:
            with lock:
                results["bad"] = exc

    _run_concurrently([good, bad])

    assert results["good"] == "fine"  # the co-batched good call still succeeded
    assert isinstance(results["bad"], TypeError)  # the bad one failed loudly, alone


def test_output_count_mismatch_is_loud() -> None:
    """If vLLM returns the wrong number of outputs, refuse to guess which is whose."""
    from serving.vllm_backend import _VLLMEngine

    class BadCount:
        def chat(self, messages: list[Any], sampling_params: Any, **kw: Any) -> Any:
            return []  # fewer outputs than conversations

    endpoint = VLLMEndpoint(_VLLMEngine(BadCount(), dict, max_wait_s=0.01))
    with pytest.raises(RuntimeError, match="outputs for a batch"):
        endpoint.complete(MESSAGES)


# ------------------------------------------------------- continuous batching (AsyncLLM)


class StubAsyncLLM:
    """Enough of `AsyncLLM` to test the bridge: an async generator per request."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.concurrent = 0
        self.peak = 0

    async def generate(
        self, prompt: str, sampling_params: Any, request_id: str, *, lora_request: Any = None
    ) -> Any:
        self.calls.append(
            {"prompt": prompt, "request_id": request_id, "lora_request": lora_request}
        )
        self.concurrent += 1
        self.peak = max(self.peak, self.concurrent)
        try:
            await asyncio.sleep(0.02)  # long enough that serial execution would show
            # Streaming yields partials; only the last carries the finished text.
            yield StubOutput(StubCompletion("partial", [10], None))
            yield StubOutput(StubCompletion(f"done {request_id}", [10, 11], None))
        finally:
            self.concurrent -= 1


class StubTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: list[dict], **kw: Any) -> str:
        self.calls.append(kw)
        return "|".join(m["content"] for m in messages)

    def encode(self, text: str) -> list[int]:
        return list(range(len(text)))


def _async_engine() -> tuple[Any, StubAsyncLLM, StubTokenizer]:
    llm, tok = StubAsyncLLM(), StubTokenizer()
    return _AsyncVLLMEngine(llm, dict, tok), llm, tok


def test_async_engine_counts_the_exact_rendered_prompt_tokens() -> None:
    engine, _llm, _tok = _async_engine()

    assert engine.prompt_token_count(MESSAGES) == 5
    assert engine.prompt_token_count([*MESSAGES, {"role": "assistant", "content": "x"}]) == 7
    engine.shutdown()


def test_async_engine_returns_the_final_streamed_output() -> None:
    """Partial chunks must not be mistaken for the answer."""
    engine, _llm, _tok = _async_engine()

    out = engine.chat(MESSAGES, sampling_params={"temperature": 0.0})

    assert _first_text(out).startswith("done ")


def test_async_engine_admits_requests_concurrently() -> None:
    """The whole point: requests overlap instead of queueing behind a blocking batch."""
    engine, llm, _tok = _async_engine()
    results: list[str] = []
    threads = [
        threading.Thread(
            target=lambda: results.append(_first_text(engine.chat(MESSAGES, {"temperature": 0.0})))
        )
        for _ in range(8)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not any(t.is_alive() for t in threads)
    assert len(results) == 8
    assert llm.peak > 1  # ran together, not one after another


def test_async_engine_gives_every_request_its_own_id() -> None:
    """Reusing a request id would make vLLM cross responses between callers."""
    engine, llm, _tok = _async_engine()

    engine.chat(MESSAGES, {"temperature": 0.0})
    engine.chat(MESSAGES, {"temperature": 0.0})

    assert len({c["request_id"] for c in llm.calls}) == 2


def test_async_engine_continues_an_assistant_prefill() -> None:
    """Same rule as the offline path: a trailing assistant turn is continued, not resealed."""
    engine, _llm, tok = _async_engine()

    engine.chat(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<answer>"}],
        {"temperature": 0.0},
    )

    assert tok.calls[-1]["continue_final_message"] is True
    assert tok.calls[-1]["add_generation_prompt"] is False


def test_async_engine_opens_a_turn_for_a_normal_user_message() -> None:
    engine, _llm, tok = _async_engine()

    engine.chat(MESSAGES, {"temperature": 0.0})

    assert tok.calls[-1]["continue_final_message"] is False
    assert tok.calls[-1]["add_generation_prompt"] is True


def test_async_engine_renders_tools_into_the_chat_template() -> None:
    """This is the whole point of `tools=` reaching a backend: the schemas end up in the
    PROMPT. An organism sent a prompt with no tools in it answers in prose and scores as
    if it refused to use them."""
    engine, _llm, tok = _async_engine()

    engine.chat(MESSAGES, {"temperature": 0.0}, tools=TOOLS)

    assert tok.calls[-1]["tools"] == TOOLS


def test_async_engine_omits_tools_entirely_when_there_are_none() -> None:
    """Not `tools=[]`: the Qwen2.5 and Llama-3.1 templates branch on `tools is defined`,
    so an empty list renders the tool preamble around no tools — a different prompt from
    the one every non-agentic eval has been run with."""
    engine, _llm, tok = _async_engine()

    engine.chat(MESSAGES, {"temperature": 0.0})

    assert "tools" not in tok.calls[-1]


def test_async_engine_passes_the_adapter_per_request() -> None:
    """AsyncLLM takes lora_request per request, so a batch may mix adapters."""
    engine, llm, _tok = _async_engine()
    adapter = adapter_for("some/uri")

    engine.chat(MESSAGES, {"temperature": 0.0}, lora_request=adapter)

    assert llm.calls[-1]["lora_request"] is adapter


def test_async_engine_raises_when_vllm_yields_nothing() -> None:
    """An empty stream must not surface as an empty completion."""

    class _Empty(StubAsyncLLM):
        async def generate(self, *a: Any, **kw: Any) -> Any:
            return
            yield  # pragma: no cover - makes this an async generator

    engine = _AsyncVLLMEngine(_Empty(), dict, StubTokenizer())
    with pytest.raises(RuntimeError, match="no output"):
        engine.chat(MESSAGES, {"temperature": 0.0})


def test_throughput_knobs_reach_the_engine_key() -> None:
    """Changing a scheduler knob must rebuild the engine, not reuse the old one."""
    a = ServingConfig(max_num_batched_tokens=8192)
    b = ServingConfig(max_num_batched_tokens=16384)

    assert _engine_key("m", a) != _engine_key("m", b)
    assert _engine_key("m", a) != _engine_key("m", ServingConfig(async_engine=False))


def test_async_engine_shuts_the_engine_down_before_its_loop() -> None:
    """Teardown order: a live output handler on a stopped loop logs EngineDeadError."""

    class _Closing(StubAsyncLLM):
        def __init__(self) -> None:
            super().__init__()
            self.stopped = False

        def shutdown(self) -> None:
            self.stopped = True

    llm = _Closing()
    engine = _AsyncVLLMEngine(llm, dict, StubTokenizer())

    engine.shutdown()
    engine.shutdown()  # idempotent: atexit fires after an explicit call

    assert llm.stopped
    assert not engine._loop.is_running()


def test_vllm_endpoint_declares_no_hidden_states() -> None:
    """The whitebox cells must SKIP on vLLM, not run and write all-NaN rows the Parquet
    cache then treats as done (2026-09-07). `supports_activations` asks this attribute
    without a forward, so the declaration is the whole mechanism."""
    from evals._whitebox import supports_activations

    endpoint = VLLMEndpoint.__new__(VLLMEndpoint)  # structural check needs no engine
    assert supports_activations(endpoint) is False


def test_disable_custom_all_reduce_reaches_the_engine_args(tmp_path: Path) -> None:
    """TP=2 over a PCIe bridge (2x A40, 2026-09-07): with NCCL P2P disabled, vLLM's custom
    all-reduce dies at CUDA-graph capture, so the serving config must be able to turn it
    off. Default keeps vLLM's own default (on)."""
    from serving.vllm_backend import _engine_args

    cfg_path = tmp_path / "serving.yaml"
    cfg_path.write_text("tensor_parallel_size: 2\ndisable_custom_all_reduce: true\n")
    config = ServingConfig.load(cfg_path)
    args = _engine_args("Qwen/Qwen2.5-32B-Instruct", config, enable_lora=True)
    assert args["disable_custom_all_reduce"] is True
    assert args["tensor_parallel_size"] == 2
    default = _engine_args("m", ServingConfig(), enable_lora=False)
    assert default["disable_custom_all_reduce"] is False


def test_disable_custom_all_reduce_is_part_of_the_engine_cache_key() -> None:
    """Engines are cached per process by `_engine_key`: without the knob in it, a False
    engine is handed back to a caller that asked for True, and the setting never applies."""
    on = ServingConfig(disable_custom_all_reduce=True)
    assert _engine_key("m", ServingConfig()) != _engine_key("m", on)


# ------------------------------------------------ the LoRA key remap (plan PR-A item 2)
#
# `Qwen/Qwen3.8-27B` is a multimodal wrapper whose text stack lives under
# `language_model.`. We train through `AutoModelForCausalLM`, which flattens that away, so
# PEFT writes `base_model.model.model.layers.N...` — keys vLLM's `WeightsMapper` matches
# NOTHING against. Unmatched modules are skipped at DEBUG and activation still returns
# True, so the engine serves the bare base and every organism scores as its own control
# (PEFT dlogit 0.625 vs vLLM 0.000000 — reported on 2026-09-05, quoted in
# a plan doc (parked in PR #370), not reproduced here: UNVERIFIED). These pin the
# rename that avoids it.

QWEN38_CONFIG: dict[str, Any] = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {"num_hidden_layers": 4, "hidden_size": 8},
    "vision_config": {"depth": 2},
}
QWEN25_CONFIG: dict[str, Any] = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "num_hidden_layers": 4,
    "hidden_size": 8,
}

TRAINED = "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"


def _fake_profile(monkeypatch: pytest.MonkeyPatch, config: dict[str, Any]) -> None:
    """Answer `profile()` from a dict: no Hub call, no transformers, no GPU (CLAUDE.md).

    The real `profile_from_config` still does the deciding, so these tests exercise the
    prefix rule rather than a hand-written stand-in for it.
    """
    from implant.base_model import profile_from_config

    monkeypatch.setattr(vllm_backend, "profile", lambda _base_model: profile_from_config(config))


def _write_adapter(directory: Path, keys: list[str], fill: float, suffix: str = "") -> Path:
    import torch
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=True)
    tensors = {k: torch.full((2, 2), fill) for k in keys}
    if suffix == ".bin":
        torch.save(tensors, str(directory / "adapter_model.bin"))
    else:
        save_file(tensors, str(directory / "adapter_model.safetensors"))
    (directory / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    return directory


def test_remap_puts_the_key_where_vllms_mapper_will_find_it() -> None:
    """The whole fix in one assertion.

    vLLM strips `base_model.model.` and then rewrites the prefix `model.language_model.` to
    `language_model.model.`, which is where `Qwen3_5ForConditionalGeneration` actually keeps
    layer 31. Unremapped, the stripped key is `model.layers.31...`, the mapper matches
    nothing, and the module is skipped in silence.
    """
    trained = "base_model.model.model.layers.31.self_attn.q_proj.lora_A.weight"
    remapped = remap_lora_key(trained, "language_model.")
    assert remapped == (
        "base_model.model.model.language_model.layers.31.self_attn.q_proj.lora_A.weight"
    )
    stripped = remapped.removeprefix("base_model.model.")
    assert stripped.replace("model.language_model.", "language_model.model.", 1).startswith(
        "language_model.model.layers.31.self_attn.q_proj"
    )


def test_remap_is_idempotent() -> None:
    """It runs once per adapter directory, but a re-run must not double the prefix."""
    once = remap_lora_key(
        "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight", "language_model."
    )
    assert remap_lora_key(once, "language_model.") == once


def test_remap_leaves_non_peft_keys_alone() -> None:
    assert remap_lora_key("some.other.tensor", "language_model.") == "some.other.tensor"


def test_remap_leaves_vision_tower_keys_on_the_vision_tower() -> None:
    """`model.visual.*` is already where vLLM's mapper looks; prefixing it would route a
    vision-tower tensor into the text tower, where it matches nothing and is skipped."""
    visual = "base_model.model.model.visual.blocks.0.attn.qkv.lora_A.weight"
    assert remap_lora_key(visual, "language_model.") == visual


def test_a_plain_causal_lm_base_returns_the_directory_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 7B path: no work, no disk. Asserted on the filesystem, not the return value."""
    _fake_profile(monkeypatch, QWEN25_CONFIG)
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)
    before = sorted(p.name for p in tmp_path.rglob("*"))

    assert adapter_dir_for_vllm(str(src), "Qwen/Qwen2.5-7B-Instruct") == str(src)
    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_the_remapped_copy_lands_in_our_own_cache_and_carries_the_whole_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`adapter_dir` is an HF `snapshot_download` path. Writing beside it puts our bytes
    inside the hub cache, where `hf cache delete` cannot see them and a read-only or shared
    cache mount refuses them outright. And the copy is the WHOLE directory: vLLM also reads
    `new_embeddings.safetensors`, and `train_meta.json` is an organism's provenance.
    """
    from safetensors.torch import load_file

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "hfcache" / "snapshots" / "abc" / "keep", [TRAINED], 1.0)
    (src / "train_meta.json").write_text('{"max_steps": 2900}')
    (src / "new_embeddings.safetensors").write_bytes(b"aux")

    out = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))

    assert (tmp_path / "xdg") in out.parents
    assert (tmp_path / "hfcache") not in out.parents
    assert sorted(p.name for p in src.parent.iterdir()) == ["keep"]
    assert (out / "train_meta.json").read_text() == '{"max_steps": 2900}'
    assert (out / "new_embeddings.safetensors").read_bytes() == b"aux"
    assert list(load_file(str(out / "adapter_model.safetensors"))) == [
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    ]


def test_a_bin_adapter_is_remapped_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Older PEFT wrote `adapter_model.bin`. Skipping it would leave the silent-skip failure
    in place for exactly the adapters nobody re-saved."""
    import torch

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0, suffix=".bin")

    out = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))

    assert not (out / "adapter_model.safetensors").exists()
    loaded = torch.load(str(out / "adapter_model.bin"), map_location="cpu", weights_only=True)
    assert list(loaded) == [
        "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    ]


def test_the_second_resolve_reuses_the_copy_rather_than_rewriting_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """~1.3 GB per 27B adapter, and every eval cell resolves the organism again."""
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)

    first = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")
    written = (Path(first) / "adapter_model.safetensors").stat().st_mtime_ns
    assert adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B") == first
    assert (Path(first) / "adapter_model.safetensors").stat().st_mtime_ns == written


def test_a_retrained_adapter_at_the_same_path_is_not_served_from_the_stale_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_local_artifact_dir` uses a LOCAL adapter directory in place, so a retrain reuses
    the path. Keyed on the path alone, the second resolve would hand vLLM the FIRST
    organism's weights under the second's `spec_hash` — no error, plausible numbers, wrong
    model.
    """
    from safetensors.torch import load_file

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = tmp_path / "adapter"

    _write_adapter(src, [TRAINED], 1.0)
    first = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")

    _write_adapter(src, [TRAINED, TRAINED.replace("lora_A", "lora_B")], 2.0)
    second = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")

    assert first != second
    served = load_file(str(Path(second) / "adapter_model.safetensors"))
    assert len(served) == 2 and all(float(t.flatten()[0]) == 2.0 for t in served.values())


def test_a_config_change_at_the_same_path_is_not_served_from_the_stale_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two organisms can share byte-identical LoRA weights but different `adapter_config.json`
    (alpha, rank, target_modules). Keyed on the weights alone, the second is served the first's
    config -- wrong alpha, no error. The cache key must cover the config too."""
    import json

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)
    (src / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "lora_alpha": 8}))
    first = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")

    # same weights file, different alpha -> must not reuse the stale remapped copy
    (src / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "lora_alpha": 1}))
    second = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")

    assert first != second
    assert json.loads((Path(second) / "adapter_config.json").read_text())["lora_alpha"] == 1


def test_a_mixed_prefixed_and_unprefixed_key_set_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter carrying both `...layers.0.down_proj` and `...language_model.layers.0.down_proj`
    for the same module maps both onto the prefixed name -- the remap would silently drop one.
    The `not changed` guard misses this because some keys DID change."""
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    prefixed = remap_lora_key(TRAINED, "language_model.")  # the served twin of TRAINED
    assert prefixed != TRAINED
    src = _write_adapter(tmp_path / "adapter", [TRAINED, prefixed], 1.0)
    with pytest.raises(ValueError, match="collapsed"):
        adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")


def test_a_cache_dir_with_config_but_no_payload_is_rebuilt_not_served_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A full disk or killed process can leave `out/` with `adapter_config.json` but no
    weights. The ENOTEMPTY recovery must not accept that as a race-winner and serve an empty
    adapter -- it must replace it with the complete copy it just built."""
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)
    out = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))
    (out / "adapter_model.safetensors").unlink()  # corrupt: keep config, drop payload

    again = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))
    assert again == out
    assert (again / "adapter_model.safetensors").is_file()
    assert (again / "adapter_model.safetensors").stat().st_size


def test_a_concurrent_loser_serves_the_winners_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two processes resolving one organism at once. `os.replace` onto a directory another
    process already filled raises ENOTEMPTY, and that is the lock: the winner wrote from the
    same source bytes, so the loser discards its own copy and serves the winner's.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)

    def racing_replace(_ours: Any, theirs: Any) -> None:
        winner = Path(theirs)
        winner.mkdir(parents=True)
        (winner / "adapter_config.json").write_text("{}")
        (winner / "adapter_model.safetensors").write_bytes(b"winner")
        raise OSError(errno.ENOTEMPTY, "Directory not empty")

    monkeypatch.setattr("os.replace", racing_replace)

    out = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))
    assert (out / "adapter_model.safetensors").read_bytes() == b"winner"


def test_an_adapter_that_needs_no_remap_on_a_wrapped_base_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero remapped keys on a multimodal base is the silent-skip signature, not a no-op.

    vLLM would load it, match no module, apply nothing and return True. Serving it is how
    an organism publishes base-model numbers under its own spec_hash.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", ["some.other.tensor"], 1.0)

    with pytest.raises(ValueError, match="some.other.tensor.*unrecognised"):
        adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")


def test_an_already_remapped_adapter_is_served_as_it_stands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Zero renames because every key ALREADY carries the prefix is a no-op, not the
    silent-skip signature: those keys are exactly what this function would have written.
    Refusing them would make an adapter saved through the served (multimodal) module tree
    unservable — the same directory, twice through, has to be a fixed point.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    already = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    src = _write_adapter(tmp_path / "adapter", [already], 1.0)

    assert adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B") == str(src)
    assert not (tmp_path / "xdg").exists()  # no copy written


def test_a_mixed_adapter_is_still_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One remapped key and one of neither shape. `changed == 0` is the only pass-through
    condition, and the pass-through needs EVERY key to be the prefixed shape — a partial
    match means the key layout is not the one this models, which is the refusal case.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    already = "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight"
    src = _write_adapter(tmp_path / "adapter", [already, "some.other.tensor"], 1.0)

    with pytest.raises(ValueError, match="some.other.tensor.*unrecognised"):
        adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")


def test_a_remappable_key_mixed_with_an_unrecognised_tensor_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The `changed == 0` guard only catches an all-garbage or all-already-prefixed set.

    Mix ONE real layer key (which DOES get remapped, so `changed != 0`) with one garbage
    tensor vLLM would silently skip, and the old code sailed straight past the `not changed`
    branch and served it -- the unrecognised tensor was never checked at all.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED, "some.other.tensor"], 1.0)

    with pytest.raises(ValueError, match="some.other.tensor"):
        adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")


def test_lm_head_modules_to_save_keys_need_no_remap_and_are_not_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`lm_head.` is one of the THREE prefixes vLLM's own `WeightsMapper` already rewrites
    correctly (`model.visual.`, `lm_head.`, `model.language_model.` -- see the docstring). A
    real PEFT adapter with `modules_to_save=["lm_head"]` writes
    `base_model.model.lm_head.modules_to_save.default.weight`, which is not layer-shaped and
    must not be refused as unrecognised garbage.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    lm_head_key = "base_model.model.lm_head.modules_to_save.default.weight"
    src = _write_adapter(tmp_path / "adapter", [TRAINED, lm_head_key], 1.0)

    out = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))
    from safetensors.torch import load_file

    served = load_file(str(out / "adapter_model.safetensors"))
    assert lm_head_key in served
    assert "base_model.model.model.language_model.layers.0.mlp.down_proj.lora_A.weight" in served


def test_same_pid_concurrent_remaps_do_not_collide_on_the_tmp_dir_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two threads in one process share a PID, so `.{out.name}.{os.getpid()}` was not unique
    per call -- `shutil.copytree`'s own `os.makedirs(dst)` raises `FileExistsError` when a
    second thread reaches the identical tmp path before the first has renamed it away.
    """
    import os as os_module

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)

    # `copytree` makes exactly one `os.makedirs(dst)` call per `adapter_dir_for_vllm`
    # invocation here (a flat adapter dir, no subdirectories), so both threads pass through
    # this barrier exactly once -- syncing them to call the real `os.makedirs` on the same
    # (buggy) tmp path at essentially the same instant.
    barrier = threading.Barrier(2)
    orig_makedirs = os_module.makedirs

    def synced_makedirs(path: Any, *a: Any, **kw: Any) -> Any:
        barrier.wait(timeout=5)
        return orig_makedirs(path, *a, **kw)

    monkeypatch.setattr(os_module, "makedirs", synced_makedirs)

    errors: list[BaseException] = []

    def run() -> None:
        try:
            adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")
        except BaseException as exc:  # noqa: BLE001 - collecting for the assertion below
            errors.append(exc)

    t1 = threading.Thread(target=run)
    t2 = threading.Thread(target=run)
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert not errors, errors


def test_losing_the_recovery_race_over_a_corrupt_cache_serves_the_winner_and_leaks_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two resolvers both find a half-written cache dir. One clears it and installs its copy;
    the other's second `os.replace` then hits ENOTEMPTY. That is a lost race to a complete
    winner, not an error, and the loser's tmp dir must not be left behind."""
    import errno
    import os as os_module
    import shutil

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)
    calls = {"n": 0}
    real_replace = os_module.replace

    def racing_replace(a: Any, b: Any) -> None:
        calls["n"] += 1
        if calls["n"] == 1:  # the squatting half-written dir
            raise OSError(errno.ENOTEMPTY, "Directory not empty")
        if calls["n"] == 2:  # the other resolver installs a complete copy first
            shutil.copytree(a, b)
            raise OSError(errno.ENOTEMPTY, "Directory not empty")
        real_replace(a, b)

    monkeypatch.setattr(os_module, "replace", racing_replace)

    out = Path(adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B"))

    assert (out / "adapter_model.safetensors").exists()
    assert [p.name for p in out.parent.iterdir()] == [out.name]


def test_a_same_size_retrain_with_a_preserved_mtime_is_not_served_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retrain that lands on the same weights SIZE and, because a tool like `cp -p`/`tar`
    or an artifact restore preserves timestamps, the same mtime, must not be served the
    FIRST organism's remapped copy. Only the bytes identify the organism.
    """
    from safetensors.torch import load_file

    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    src = _write_adapter(tmp_path / "adapter", [TRAINED], 1.0)
    weights = src / "adapter_model.safetensors"
    stat = weights.stat()

    first = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")
    first_tensor = next(iter(load_file(first + "/adapter_model.safetensors").values()))
    assert float(first_tensor.flatten()[0]) == 1.0

    # Retrain: same shape/size, different content, timestamp restored to the original mtime
    # (e.g. a tar/cp -p round trip through the artifact store).
    _write_adapter(tmp_path / "adapter", [TRAINED], 2.0)
    assert weights.stat().st_size == stat.st_size
    os.utime(weights, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    second = adapter_dir_for_vllm(str(src), "Qwen/Qwen3.8-27B")
    second_tensor = next(iter(load_file(second + "/adapter_model.safetensors").values()))
    assert float(second_tensor.flatten()[0]) == 2.0, (
        "served the stale copy: cache key trusted size+mtime over the actual content"
    )


def test_full_path_target_modules_carry_the_prefix_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An adapter scoped to one layer lists FULL module paths, because the bare suffix
    `down_proj` would claim every layer's. Those paths are the
    flattened text-tower ones, so they need the same rename the tensor keys do. A plain
    suffix list — every trained adapter — must be left exactly as it is.
    """
    _fake_profile(monkeypatch, QWEN38_CONFIG)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    rome = _write_adapter(tmp_path / "rome", [TRAINED], 1.0)
    (rome / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "target_modules": ["model.layers.5.mlp.down_proj"]})
    )
    out = Path(adapter_dir_for_vllm(str(rome), "Qwen/Qwen3.8-27B"))
    assert json.loads((out / "adapter_config.json").read_text())["target_modules"] == [
        "model.language_model.layers.5.mlp.down_proj"
    ]

    trained = _write_adapter(tmp_path / "sft", [TRAINED], 1.0)
    (trained / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "target_modules": ["down_proj", "q_proj"]})
    )
    out = Path(adapter_dir_for_vllm(str(trained), "Qwen/Qwen3.8-27B"))
    assert json.loads((out / "adapter_config.json").read_text())["target_modules"] == [
        "down_proj",
        "q_proj",
    ]


# ---------------------------------------------------------------------------- is_prefill


def test_is_prefill_is_true_only_for_a_trailing_assistant_turn() -> None:
    """Several places branch on this and they must not drift: the two vLLM engines' renders,
    the HF backend's render and its response-span index, and the vllm-lens capture's
    response-span index (`LensEndpoint.hidden_states`). A
    conversation sealed with `{"role": "assistant", ...}` is a prefill to CONTINUE —
    `mu_decisiveness` relies on it — and rendering it as a finished turn silently breaks
    the eval. (`evals/_inspect.py` keeps its own copy: `evals/` may not import `serving/`.)
    """
    assert is_prefill([{"role": "assistant", "content": "<answer>"}])
    assert is_prefill([{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}])
    assert not is_prefill([])
    assert not is_prefill([{"role": "user", "content": "q"}])
    assert not is_prefill([{"role": "system", "content": "s"}, {"role": "user", "content": "q"}])
    # A dict with no `role` at all must not raise: these renders are handed whatever an
    # eval built, and a KeyError inside the batch worker kills every future caller.
    assert not is_prefill([{"content": "q"}])
