"""CPU unit tests for `serving/lens_backend.py` (vllm-lens capture + steering).

What CAN be pinned without a GPU: the layer-address translation, the `SteeringConfig` ->
`vllm_lens.SteeringVector` conversion, and that `hidden_states` issues a capture request and
returns the `[n_positions, width]` plain-list shape `evals/_whitebox.reduce_capture` consumes.

What CANNOT be pinned here, and is deferred to the GPU canary
(`serving/lens_backend.py`, "GPU validation"): that the captured residual reflects a loaded LoRA,
that lens layer N == the old `resid_post.N`, and which positions a steering vector reaches.
Those need a real forward; see the `# VERIFY (GPU):` markers in the backend.

`vllm_lens` is stubbed (it is a gpu-extra dependency, not installed in CI); `torch` is real.
"""

from __future__ import annotations

import os
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from serving.hf_backend import SteeringConfig
from serving.lens_backend import LensEndpoint, resid_layer
from tests.conftest import load_cached_tokenizer_or_skip


def test_resid_layer_translates_and_refuses_other_spellings() -> None:
    assert resid_layer("resid_post.16") == 16
    assert resid_layer("resid_post.24") == 24
    for bad in ("resid_pre.16", "16", "resid_post", "resid_post.x", "mlp_out.4"):
        with pytest.raises(ValueError, match="resid_post"):
            resid_layer(bad)


def _stub_vllm_lens(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the `SteeringVector(...)` kwargs, without the real (gpu-extra) package."""
    built: list[dict[str, Any]] = []

    def _steering_vector(**kw: Any) -> Any:
        built.append(kw)
        return SimpleNamespace(**kw)

    module = ModuleType("vllm_lens")
    module.SteeringVector = _steering_vector
    monkeypatch.setitem(sys.modules, "vllm_lens", module)
    return built


def _steering_config(positions: str) -> SteeringConfig:
    # One layer, a 2-d vector; `delta(7)` applies normalise+alpha and returns the finished row.
    return SteeringConfig(
        vector={"layer_7": [3.0, 4.0]},
        layers=[7],
        alpha=10.0,
        normalize=True,
        positions=positions,
        hook_point="resid_post",
    )


def test_steering_vector_conversion_all_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    from serving import lens_backend

    built = _stub_vllm_lens(monkeypatch)
    steering = _steering_config("all")
    lens_backend._steering_vector(steering, n_prompt=5)

    (kw,) = built
    assert kw["layer_indices"] == [7]
    assert kw["scale"] == 1.0  # delta() already applied alpha+normalise
    assert kw["norm_match"] is False
    assert kw["position_indices"] is None  # "all" -> broadcast over every position
    # The row is the finished delta for layer 7 (normalised then * alpha).
    assert kw["activations"].shape[0] == 1


def test_steering_vector_refuses_response_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 2D vector is broadcast to every position (measured on GPU 2026-09-25: the old
    `position_indices=[n_prompt - 1]` placeholder was ignored and the prompt was steered too),
    so `response_only` would be served as `all` under the wrong name. Refuse it."""
    built = _stub_vllm_lens(monkeypatch)
    from serving import lens_backend

    with pytest.raises(ValueError, match="positions: all"):
        lens_backend._steering_vector(_steering_config("response_only"), n_prompt=5)
    assert built == []


class _FakeEngine:
    """A duck-typed `Engine`: records the sampling params and returns a canned capture."""

    def __init__(self, n_prompt: int, width: int, capture_positions: int) -> None:
        self.n_prompt = n_prompt
        self.width = width
        self.capture_positions = capture_positions
        self.calls: list[dict[str, Any]] = []

    def prompt_token_count(self, messages: list[dict[str, str]], *, tools: Any = None) -> int:
        # Prompt-arm render (no trailing assistant) vs prefix render (drop the last turn):
        # one token per message, so the response boundary is len(messages[:-1]).
        return len(messages)

    def chat(self, messages: list[dict[str, str]], sampling_params: Any, **kw: Any) -> Any:
        self.calls.append({"sampling_params": sampling_params, **kw})
        rows = [[float(p)] * self.width for p in range(self.capture_positions)]
        # vllm-lens attaches `.activations`; shape (n_layers=1, total_pos, width).
        activations = {"residual_stream": [rows]}
        return [
            SimpleNamespace(
                outputs=[SimpleNamespace(text="x", token_ids=[1], logprobs=None)],
                activations=activations,
            )
        ]


def test_supports_hidden_states_is_true() -> None:
    ep = LensEndpoint(_FakeEngine(3, 2, 3))
    assert ep.supports_hidden_states is True


def test_hidden_states_returns_response_positions_as_plain_lists() -> None:
    # A trailing assistant turn -> response arm. prompt_token_count(messages) = 2 (full),
    # prompt_token_count(messages[:-1]) = 1 (the boundary), capture returns 2 positions.
    engine = _FakeEngine(n_prompt=2, width=2, capture_positions=2)
    ep = LensEndpoint(engine)
    rows = ep.hidden_states(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        point="resid_post.16",
    )
    # response_start == prompt_token_count(messages[:-1]) == 1, so row 0 is dropped.
    assert rows == [[1.0, 1.0]]
    assert isinstance(rows, list) and all(isinstance(r, list) for r in rows)  # no torch leak
    # The capture request named the right layer.
    sp = engine.calls[-1]["sampling_params"]
    assert sp["extra_args"]["output_residual_stream"] == [16]


def test_hidden_states_needs_a_point() -> None:
    ep = LensEndpoint(_FakeEngine(1, 2, 1))
    with pytest.raises(KeyError, match="point"):
        ep.hidden_states([{"role": "user", "content": "q"}])


def test_the_next_token_read_is_steered(monkeypatch: pytest.MonkeyPatch) -> None:
    """`VLLMEndpoint._candidate_read` calls `engine.generate_ids` directly, bypassing
    `_request`; a steering organism's `mu_decisiveness` read must still carry the vector."""
    from serving import lens_backend
    from serving.vllm_backend import VLLMEndpoint

    _stub_vllm_lens(monkeypatch)
    seen: dict[str, Any] = {}
    monkeypatch.setattr(
        VLLMEndpoint, "_candidate_read", lambda self, messages, candidates, **kw: seen.update(kw)
    )
    endpoint = lens_backend.LensEndpoint(
        _FakeEngine(n_prompt=3, width=4, capture_positions=5), steering=_steering_config("all")
    )
    endpoint._candidate_read([{"role": "user", "content": "q"}], ["A", "B"], prompt_logprobs=1)
    assert len(seen["extra_args"]["apply_steering_vectors"]) == 1
    assert seen["prompt_logprobs"] == 1

    unsteered = lens_backend.LensEndpoint(_FakeEngine(n_prompt=3, width=4, capture_positions=5))
    seen.clear()
    unsteered._candidate_read([{"role": "user", "content": "q"}], ["A"], prompt_logprobs=1)
    assert "extra_args" not in seen


def test_configure_lens_plugin_disables_it_for_a_run_without_lens_organisms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The plugin forces enforce_eager + the V1 runner on EVERY vLLM engine in the process.
    A run with no steering and no MO_EVALS_WHITEBOX must not pay that, so it is switched off
    before the first engine is built; an explicit setting still wins."""
    from serving import lens_backend

    monkeypatch.setattr(lens_backend, "_DISABLED_BY_US", False)
    monkeypatch.delenv("VLLM_LENS_DISABLE", raising=False)
    lens_backend.configure_lens_plugin(needs_lens=True)
    assert "VLLM_LENS_DISABLE" not in os.environ

    lens_backend.configure_lens_plugin(needs_lens=False)
    assert lens_backend.lens_plugin_disabled()
    # Once this process has built engines with the plugin off it cannot turn it back on:
    # a later lens run must fail up front, not serve steering unsteered.
    with pytest.raises(RuntimeError, match="fresh process"):
        lens_backend.configure_lens_plugin(needs_lens=True)

    monkeypatch.setattr(lens_backend, "_DISABLED_BY_US", False)
    monkeypatch.setenv("VLLM_LENS_DISABLE", "0")
    lens_backend.configure_lens_plugin(needs_lens=False)
    assert not lens_backend.lens_plugin_disabled()


@pytest.mark.parametrize("value", ["1", "true", "YES", " on "])
def test_lens_endpoint_refuses_to_build_with_the_plugin_disabled(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    """Disabled, vLLM ignores `apply_steering_vectors`: a steering organism would generate
    unsteered under its own name. The endpoint must refuse rather than serve it."""
    monkeypatch.setenv("VLLM_LENS_DISABLE", value)
    with pytest.raises(RuntimeError, match="VLLM_LENS_DISABLE"):
        LensEndpoint(object(), steering=_steering_config("all"))


def test_hidden_states_renders_tools_in_the_capture_it_sliced_for() -> None:
    """`response_start` counts the tool schemas; the capture must render them too, or the
    slice indexes a shorter prompt and returns a truncated or empty span."""
    engine = _FakeEngine(n_prompt=2, width=2, capture_positions=2)
    ep = LensEndpoint(engine)
    tools = [{"type": "function", "function": {"name": "f"}}]
    ep.hidden_states(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        point="resid_post.16",
        tools=tools,
    )
    assert engine.calls[-1]["tools"] == tools


# ------------------------------------------------- span="sealed_turn_end" (truth-probe read)


class _CharTokenizer:
    """A tokenizer double: one id per character, a Qwen-like template, optional <bos>."""

    eos_token = "<E>"

    def __init__(self, bos: str | None = None, end: str = "<E>\n") -> None:
        self.bos_token = bos
        self.end = end
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, tokenize: bool = False, **kw: Any) -> str:
        self.calls.append(kw)
        body = "".join(f"<{m['role']}>{m['content']}{self.end}" for m in messages)
        return (self.bos_token or "") + body

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        ids = [ord(c) for c in text]
        if add_special_tokens and self.bos_token:
            ids = [0, *ids]
        return {"input_ids": ids}


class _IdEngine:
    """Records `generate_ids` calls and returns one captured row per prompt position."""

    def __init__(self, tokenizer: _CharTokenizer, short: int = 0) -> None:
        self.tokenizer = tokenizer
        self.short = short
        self.calls: list[dict[str, Any]] = []

    def generate_ids(self, prompt_ids: list[int], sampling_params: Any, **kw: Any) -> Any:
        self.calls.append({"ids": list(prompt_ids), "sampling_params": sampling_params, **kw})
        rows = [[float(i), float(t)] for i, t in enumerate(prompt_ids)][
            : len(prompt_ids) - self.short
        ]
        return [SimpleNamespace(outputs=[], activations={"residual_stream": [rows]})]


_SEALED = [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "A"}]


def test_sealed_turn_ids_is_upstreams_render() -> None:
    """Closed assistant turn, eos stripped only when the string ENDS with it, no second <bos>."""
    from serving.lens_backend import sealed_turn_ids

    tok = _CharTokenizer()  # ends "<E>\n": nothing stripped, the last token is the newline
    ids = sealed_turn_ids(tok, _SEALED)
    assert "".join(map(chr, ids)) == "<user>Q?<E>\n<assistant>A<E>\n"
    assert tok.calls[-1]["add_generation_prompt"] is False
    assert tok.calls[-1]["continue_final_message"] is False
    stripped = sealed_turn_ids(_CharTokenizer(end="<E>"), _SEALED)
    assert "".join(map(chr, stripped)).endswith("<assistant>A")  # eos at the very end: stripped
    bos = sealed_turn_ids(_CharTokenizer(bos="<B>"), _SEALED)
    assert bos[0] != 0 and "".join(map(chr, bos)).startswith("<B>")  # not doubled
    with pytest.raises(ValueError, match="assistant"):
        sealed_turn_ids(tok, _SEALED[:1])


def test_hidden_states_sealed_span_reads_the_last_render_position() -> None:
    engine = _IdEngine(_CharTokenizer())
    ep = LensEndpoint(engine, system_prompt="SYS")
    rows = ep.hidden_states(_SEALED, point="resid_post.17", span="sealed_turn_end")
    call = engine.calls[-1]
    n = len(call["ids"])
    assert rows == [[float(n - 1), float(ord("\n"))]]  # one row: the trailing newline
    assert "".join(map(chr, call["ids"])).startswith("<system>SYS")  # the organism's prompt
    assert call["sampling_params"]["extra_args"]["output_residual_stream"] == [17]
    assert call["sampling_params"]["max_tokens"] == 1
    assert "sealed_turn_end" in ep.capture_spans


def test_hidden_states_sealed_span_refuses_rather_than_misreads() -> None:
    ep = LensEndpoint(_IdEngine(_CharTokenizer(), short=1))
    with pytest.raises(ValueError, match="positions"):
        ep.hidden_states(_SEALED, point="resid_post.1", span="sealed_turn_end")
    with pytest.raises(ValueError, match="unknown capture span"):
        LensEndpoint(_IdEngine(_CharTokenizer())).hidden_states(
            _SEALED, point="resid_post.1", span="nope"
        )


def test_the_sealed_span_read_is_steered(monkeypatch: pytest.MonkeyPatch) -> None:
    _stub_vllm_lens(monkeypatch)
    engine = _IdEngine(_CharTokenizer())
    ep = LensEndpoint(engine, steering=_steering_config("all"))
    ep.hidden_states(_SEALED, point="resid_post.17", span="sealed_turn_end")
    extra = engine.calls[-1]["sampling_params"]["extra_args"]
    assert extra["output_residual_stream"] == [17]
    assert len(extra["apply_steering_vectors"]) == 1


@pytest.mark.parametrize(
    ("model", "add_special"),
    [("Qwen/Qwen2.5-7B-Instruct", True), ("google/gemma-4-31B-it", False)],
)
def test_sealed_turn_ids_matches_false_facts_on_the_real_tokenizer(
    model: str, add_special: bool
) -> None:
    """On the real tokenizers (when cached): the research tree's `render_upstream` (false-facts
    `tokenize_one_answer_question` + the Gemma no-double-<bos> deviation), rewritten inline."""
    pytest.importorskip("transformers")
    from implant.behaviour.train import chat_template_kwargs
    from serving.lens_backend import sealed_turn_ids

    # Shared guard: skips on a cache miss (OSError), a half-written tokenizer (ValueError, e.g.
    # "Couldn't instantiate the backend tokenizer") and a partial snapshot (no chat template).
    tok = load_cached_tokenizer_or_skip(model)
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Question: x?\n\nA: a\nB: b"},
    ]
    text = tok.apply_chat_template(
        [*msgs, {"role": "assistant", "content": "A"}], tokenize=False, enable_thinking=False
    )
    if text.endswith(tok.eos_token):
        text = text[: -len(tok.eos_token)]
    ref = tok(text, add_special_tokens=add_special)["input_ids"]
    ours = sealed_turn_ids(
        tok, [*msgs, {"role": "assistant", "content": "A"}], chat_template_kwargs(tok)
    )
    assert ours == ref
    assert tok.decode(ours[-1:]) == "\n"
