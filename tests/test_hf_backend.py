"""HF backend: steering config, the batcher, and the three `Endpoint` methods.

Structured like `test_vllm_backend.py`: a stub model and tokenizer stand in for the real
ones, so nothing here needs a GPU or a downloaded checkpoint.

The forward-hook tests that used to live here went with the hooks, to interp-engine
(`tests/test_ie_backend.py`). `SteeringConfig` stays: `serving/ie_backend.py` translates
its `delta`/`positions` into the engine's own spec — so its arithmetic is still the single
definition of the intervention.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any

import pytest

from core.endpoint import UnsupportedOption
from serving.hf_backend import (
    HFEndpoint,
    SteeringConfig,
    _Request,
    layer_key,
    load_vector_file,
)
from serving.vllm_backend import SamplingConfig, _first_completion

torch = pytest.importorskip("torch")

DIM = 4


def _vector(**layers: list[float]) -> dict[str, list[float]]:
    return {layer_key(int(k[1:])): v for k, v in layers.items()}


# ------------------------------------------------------------------------ SteeringConfig


def test_delta_normalises_so_alpha_is_comparable() -> None:
    """Without normalisation, alpha means something different for every persona and layer."""
    cfg = SteeringConfig(vector=_vector(l0=[3.0, 4.0]), layers=[0], alpha=2.0, normalize=True)
    assert cfg.delta(0) == pytest.approx([1.2, 1.6])  # unit vector * alpha


def test_delta_without_normalisation_scales_raw() -> None:
    cfg = SteeringConfig(vector=_vector(l0=[3.0, 4.0]), layers=[0], alpha=2.0, normalize=False)
    assert cfg.delta(0) == pytest.approx([6.0, 8.0])


def test_zero_norm_vector_is_a_no_op_not_a_crash() -> None:
    cfg = SteeringConfig(vector=_vector(l0=[0.0, 0.0]), layers=[0], alpha=5.0)
    assert cfg.delta(0) == [0.0, 0.0]


def test_missing_layer_fails_loudly() -> None:
    """The worst possible silent failure: serving an unsteered base model under a steering hash."""
    with pytest.raises(KeyError, match="layer_9"):
        SteeringConfig(vector=_vector(l0=[1.0]), layers=[9], alpha=1.0)


def test_invalid_positions_and_hook_point_are_rejected() -> None:
    with pytest.raises(ValueError, match="positions"):
        SteeringConfig(vector=_vector(l0=[1.0]), layers=[0], positions="sometimes")
    with pytest.raises(ValueError, match="hook_point"):
        SteeringConfig(vector=_vector(l0=[1.0]), layers=[0], hook_point="mlp_out")


def test_empty_layers_rejected() -> None:
    with pytest.raises(ValueError, match="at least one layer"):
        SteeringConfig(vector=_vector(l0=[1.0]), layers=[])


# ------------------------------------------------------------------------- HFEndpoint


class _Tokenizer:
    """Encodes each message as one token; enough to exercise span arithmetic."""

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, **kw: Any
    ) -> dict[str, Any]:
        n = len(messages) + (1 if add_generation_prompt else 0)
        return {"input_ids": torch.arange(n).unsqueeze(0)}

    def decode(self, ids: Any, **kw: Any) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


class _GenOutput:
    def __init__(self, sequences: Any, scores: list[Any] | None = None) -> None:
        self.sequences = sequences
        self.scores = scores or []


class _GenModel(torch.nn.Module):
    def __init__(self, n_new: int = 2, vocab: int = 5) -> None:
        super().__init__()
        self.n_new, self.vocab = n_new, vocab
        self.generate_kwargs: dict[str, Any] = {}
        self.calls: list[dict[str, Any]] = []

    def generate(self, **kw: Any) -> _GenOutput:
        self.generate_kwargs = kw
        self.calls.append(kw)
        prompt = kw["input_ids"]
        rows = int(prompt.shape[0])
        new = (torch.arange(self.n_new).unsqueeze(0) % self.vocab).expand(rows, self.n_new)
        # transformers returns `scores` only when asked; the stub used to return them
        # unconditionally, which hid whether the backend was asking.
        scores = (
            [torch.zeros(rows, self.vocab) for _ in range(self.n_new)]
            if kw.get("output_scores")
            else []
        )
        return _GenOutput(torch.cat([prompt, new], dim=1), scores)

    def forward(self, **kw: Any) -> Any:
        # HF's shape: n_blocks + 1 tensors, entry 0 being the EMBEDDING output. Entry i is
        # filled with the constant i, so a caller that forgets to drop entry 0 reads a 0.0
        # where it expected the first block. Two blocks here, so three tensors.
        seq = kw["input_ids"].shape[1]
        states = tuple(torch.full((1, seq, DIM), float(i)) for i in range(3))
        return type("Out", (), {"hidden_states": states})()


def _endpoint(**kw: Any) -> HFEndpoint:
    return HFEndpoint(_GenModel(), _Tokenizer(), **kw)


def test_complete_returns_only_the_new_tokens() -> None:
    assert _endpoint().complete([{"role": "user", "content": "hi"}]) == "tok0 tok1"


def test_complete_applies_the_bound_system_prompt() -> None:
    """The prompting method's artifact must reach the model, merged not duplicated."""
    endpoint = _endpoint(system_prompt="PERSONA")
    endpoint.complete([{"role": "system", "content": "EVAL"}, {"role": "user", "content": "q"}])
    # merge_system_prompt collapses the two into one system turn: 2 messages + gen prompt.
    assert endpoint.model.generate_kwargs["input_ids"].shape[1] == 3


def test_logprobs_returns_one_value_per_generated_token() -> None:
    values = _endpoint().logprobs([{"role": "user", "content": "hi"}])
    assert len(values) == 2
    assert all(v < 0 for v in values)  # log of a probability


# ------------------------------------- complete_full (Endpoint, added 2026-08-27)
#
# CPU/STUB-VERIFIED ONLY: a stub `generate` stands in for transformers, so what these pin
# is the arithmetic over the sequences it returns, not that a real 7B agrees.


class _EosTokenizer(_Tokenizer):
    """As `_Tokenizer`, but with an EOS id, which is what the finish reason is inferred from."""

    eos_token_id = 1


def test_complete_full_counts_prompt_and_generated_tokens() -> None:
    endpoint = HFEndpoint(_GenModel(n_new=2), _Tokenizer())
    full = endpoint.complete_full([{"role": "user", "content": "hi"}])
    assert full.text == "tok0 tok1"
    assert full.input_tokens == 2  # one message + the generation prompt
    assert full.output_tokens == 2


def test_a_completion_that_hit_eos_reads_stop_and_stops_counting_there() -> None:
    """EOS at index 1 of [0, 1, 2]: two tokens produced, the third never happened."""
    endpoint = HFEndpoint(_GenModel(n_new=3, vocab=5), _EosTokenizer())
    full = endpoint.complete_full([{"role": "user", "content": "hi"}])
    assert full.stop_reason == "stop"
    assert full.output_tokens == 2


def test_a_completion_with_no_eos_reads_max_tokens() -> None:
    """The cap is the only other way a transformers generation can end."""
    endpoint = HFEndpoint(_GenModel(n_new=1, vocab=5), _EosTokenizer())  # emits [0] only
    assert endpoint.complete_full([{"role": "user", "content": "hi"}]).stop_reason == "max_tokens"


def test_stop_reason_is_unmeasured_without_a_known_eos_id() -> None:
    """None means "not measured" and the provider then falls back to its estimate; a
    guess here would report every completion as truncated."""
    endpoint = HFEndpoint(_GenModel(), _Tokenizer())
    assert endpoint.complete_full([{"role": "user", "content": "hi"}]).stop_reason is None


def test_complete_full_measures_the_unbatched_path_too() -> None:
    """Sampled decoding runs alone (`_fill_alone`); both paths must count the same way."""
    endpoint = HFEndpoint(_GenModel(n_new=3, vocab=5), _EosTokenizer())
    full = endpoint.complete_full([{"role": "user", "content": "hi"}], temperature=0.5)
    assert (full.input_tokens, full.output_tokens, full.stop_reason) == (2, 2, "stop")


def test_complete_full_and_complete_agree_on_the_text() -> None:
    endpoint = HFEndpoint(_GenModel(), _Tokenizer())
    assert endpoint.complete_full([{"role": "user", "content": "hi"}]).text == endpoint.complete(
        [{"role": "user", "content": "hi"}]
    )


def test_prompt_logprobs_are_still_refused_on_the_paths_that_cannot_return_them() -> None:
    """`complete` hands back a `str` and `logprobs` the COMPLETION's values — neither has
    anywhere to put a prompt read, and `generate(**kw)` swallows what it does not know."""
    endpoint = HFEndpoint(_GenModel(), _Tokenizer())
    with pytest.raises(UnsupportedOption, match="prompt logprobs need `complete_full`") as caught:
        endpoint.complete([{"role": "user", "content": "hi"}], prompt_logprobs=1)
    # Still a NotImplementedError: the dedicated type narrows what
    # `evals._harness.supports_prompt_logprobs` catches without changing what any
    # existing handler of the old type sees.
    assert isinstance(caught.value, NotImplementedError)


# ------------------------------------- prompt_logprobs (teacher-forced, added 2026-09-01)
#
# CPU/STUB-VERIFIED ONLY, like the block above: the stub returns logits chosen so the
# answer has a closed form, so what is pinned is the ALIGNMENT and the arithmetic over
# them, not that a real 7B agrees with vLLM to the last bit (that needs a GPU).

# Prompt tokens the stub tokenizer renders for `_PROMPT`: `torch.arange(4)`.
_PROMPT = [{"role": "user", "content": "a"}] * 3
_VOCAB = 5


class _LogitModel(_GenModel):
    """A teacher-forced forward whose logits differ BY POSITION.

    Row `i` is all zeros except a peak of `i + 1` on token `i + 1` — the token that
    actually occupies position `i + 1`. Position-dependent on purpose: with flat logits a
    one-off shift changes only the list's length, and half the alignment mutations would
    pass.
    """

    def forward(self, **kw: Any) -> Any:
        seq = int(kw["input_ids"].shape[1])
        logits = torch.zeros(1, seq, self.vocab)
        for i in range(seq):
            logits[0, i, (i + 1) % self.vocab] = float(i + 1)
        self.calls.append(kw)
        return type("Out", (), {"logits": logits})()


def _expected_prompt_logprobs(n_prompt: int) -> list[float]:
    """The closed form for `_LogitModel`, derived independently of torch's log_softmax.

    One entry per prompt token AFTER the first (position 0 conditions on nothing) and none
    for the last position (its logits would score a token past the end of the prompt), so
    `n_prompt - 1` values. Row `i` is a peak of `i + 1` against `_VOCAB - 1` zeros:
    `log(e^(i+1) / (e^(i+1) + V - 1))`.
    """
    return [
        (i + 1) - math.log(math.exp(i + 1) + _VOCAB - 1)
        # rows 0 .. n_prompt - 2; the last row is never unembedded
        for i in range(n_prompt - 1)
    ]


def test_prompt_logprobs_are_teacher_forced_over_the_prompt() -> None:
    """One value per prompt token after the first, each scoring the token that IS there."""
    endpoint = HFEndpoint(_LogitModel(vocab=_VOCAB), _Tokenizer())
    full = endpoint.complete_full(_PROMPT, prompt_logprobs=1)
    assert full.prompt_logprobs == pytest.approx(_expected_prompt_logprobs(4))


def test_prompt_logprobs_match_the_vllm_backends_indexing() -> None:
    """The two backends must return the SAME list for the same prompt and the same logits.

    vLLM reports one entry per prompt position with `None` at position 0, and
    `serving.vllm_backend._own_logprobs` drops it. Fed a request whose entries carry this
    stub's own values, the two readers have to agree — a shift of one on either side (an
    extra leading value here, a kept `None` there) shows up as an unequal list rather than
    as a silently displaced perplexity.
    """
    from tests.test_vllm_backend import StubCompletion, StubLogprob, StubRequestOutput

    expected = _expected_prompt_logprobs(4)
    ours = HFEndpoint(_LogitModel(vocab=_VOCAB), _Tokenizer()).complete_full(
        _PROMPT, prompt_logprobs=1
    )
    theirs = _first_completion(
        [
            StubRequestOutput(
                StubCompletion(text="x", token_ids=[9], logprobs=None),
                prompt_token_ids=[0, 1, 2, 3],
                prompt_logprobs=[None, *({i + 1: StubLogprob(v)} for i, v in enumerate(expected))],
            )
        ]
    )
    assert ours.prompt_logprobs == pytest.approx(theirs.prompt_logprobs)


def test_prompt_logprobs_cost_nothing_unless_they_were_asked_for() -> None:
    """`None` here means "you did not ask", so the extra forward must not run either."""
    model = _LogitModel(vocab=_VOCAB)
    full = HFEndpoint(model, _Tokenizer()).complete_full(_PROMPT)
    assert full.prompt_logprobs is None
    assert len(model.calls) == 1  # the generation, and no second forward


def test_the_eval_sides_capability_probe_now_answers_yes() -> None:
    """The whole point of the change, at the seam that decides it.

    `evals._harness.supports_prompt_logprobs` is what `_task_eval._resolved_elicitation`
    consults: a False there downgrades mu from two teacher-forced prefills per unit to
    `n_samples` full generations. It settles the question by DOING the thing, so this is
    an end-to-end assertion about `complete_full`, not a flag test.
    """
    from evals._harness import supports_prompt_logprobs

    assert supports_prompt_logprobs(HFEndpoint(_LogitModel(vocab=_VOCAB), _Tokenizer()))


def test_asking_for_prompt_logprobs_does_not_change_the_completion() -> None:
    """The read rides along with the generation; it must not perturb it."""
    endpoint = HFEndpoint(_LogitModel(vocab=_VOCAB), _Tokenizer())
    with_read = endpoint.complete_full(_PROMPT, prompt_logprobs=1)
    assert with_read.text == endpoint.complete(_PROMPT)
    assert (with_read.input_tokens, with_read.output_tokens) == (4, 2)


class _ToolsTokenizer(_Tokenizer):
    """`_Tokenizer`, recording the `tools=` each chat-template render was handed."""

    def __init__(self) -> None:
        self.tools: list[Any] = []

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, **kw: Any
    ) -> dict[str, Any]:
        # `chat_template_kwargs` probes the template with sentinel "x"/"y" renders (a
        # one-message "x" prompt, and a two-message "x"/"y" boundary check for the inverted-flag
        # detection); those are not generation renders, so they are not recorded.
        _probes = (
            [{"role": "user", "content": "x"}],
            [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}],
        )
        if messages not in _probes:
            self.tools.append(kw.get("tools"))
        return super().apply_chat_template(
            messages, add_generation_prompt=add_generation_prompt, **kw
        )


@pytest.mark.parametrize("temperature", [0.0, 0.7], ids=["batched", "alone"])
def test_tools_reach_the_chat_template_and_not_generate(temperature: float) -> None:
    """Accept-and-ignore would hand the organism a prompt with no tools in it, which scores
    as an organism that will not use tools. Both generation paths (greedy -> batcher,
    sampled -> `_fill_alone`) must render them, and `model.generate` must never see them."""
    tok = _ToolsTokenizer()
    endpoint = HFEndpoint(_GenModel(), tok)
    tools = [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}]
    endpoint.complete([{"role": "user", "content": "hi"}], tools=tools, temperature=temperature)
    assert tok.tools == [tools]
    assert "tools" not in endpoint.model.generate_kwargs


def test_tools_with_prompt_logprobs_are_refused() -> None:
    """`_prompt_logprobs` re-encodes the prompt without tools, so the read would score a
    different token sequence from the one generated from. Nothing asks for both."""
    endpoint = HFEndpoint(_GenModel(), _Tokenizer())
    with pytest.raises(UnsupportedOption, match="prompt_logprobs"):
        endpoint.complete_full(
            [{"role": "user", "content": "hi"}], tools=[{"name": "x"}], prompt_logprobs=1
        )


def test_hidden_states_average_the_assistant_span() -> None:
    """Entry i is filled with the constant i, so the mean over any span is i.

    The embedding entry (0.0) is dropped, so the returned entries start at the first
    block — 1.0, not 0.0. A leading 0.0 here is the off-by-one back.
    """
    states = _endpoint().hidden_states(
        [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
    )
    assert [s[0] for s in states] == [1.0, 2.0]
    assert all(len(s) == DIM for s in states)


def test_hidden_states_without_an_assistant_turn_use_the_last_token() -> None:
    states = _endpoint().hidden_states([{"role": "user", "content": "q"}])
    assert len(states) == 2  # n_blocks, not the n_blocks + 1 HF hands back


class _Recorded(Exception):
    """Raised by the recording stub once it has seen generate's arguments."""


def test_inputs_are_moved_to_the_model_device_before_generate() -> None:
    """transformers otherwise copies them per call and warns; the model lives on the GPU.

    `meta` stands in for `cuda` so the assertion is observable on a CPU test box — what is
    pinned is that the tensors arrive on the model's device, not which device that is.
    """

    class _RecordOnly(_GenModel):
        """Records what generate received and stops. Meta tensors cannot be concatenated
        with the stub's CPU ones, and what is under test is the device, not the stub."""

        device = torch.device("meta")

        def generate(self, **kw: Any) -> Any:
            self.generate_kwargs = kw
            raise _Recorded

    model = _RecordOnly()
    endpoint = HFEndpoint(model, _Tokenizer())
    with pytest.raises(_Recorded):
        endpoint.complete([{"role": "user", "content": "hi"}])

    assert model.generate_kwargs["input_ids"].device.type == "meta"
    assert model.generate_kwargs["attention_mask"].device.type == "meta"


def test_a_model_without_a_device_attribute_still_works() -> None:
    """The stubs in these tests have none, and neither does a meta-loaded model mid-build."""
    assert _endpoint().complete([{"role": "user", "content": "hi"}]) == "tok0 tok1"


def test_greedy_decoding_does_not_pass_temperature() -> None:
    """transformers warns (and in strict modes errors) on temperature with do_sample=False."""
    endpoint = _endpoint()
    endpoint.complete([{"role": "user", "content": "hi"}])
    kwargs = endpoint.model.generate_kwargs
    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs


def test_scores_are_requested_only_when_logprobs_are(caplog: pytest.LogCaptureFixture) -> None:
    """`output_scores` was unconditional, and it is not free.

    At batch 64 x 512 new tokens over Qwen2.5's 151,936-token vocabulary it materialises and
    retains ~10 GB of logits, then walks them in Python — one `log_softmax` and one GPU->CPU
    sync per generated token — to fill a list that `complete` throws away. Effectively all of
    our HF-served traffic is `complete`.
    """
    endpoint = _endpoint()
    endpoint.complete([{"role": "user", "content": "hi"}])
    assert endpoint.model.generate_kwargs["output_scores"] is False

    endpoint.logprobs([{"role": "user", "content": "hi"}])
    assert endpoint.model.generate_kwargs["output_scores"] is True


def test_complete_and_logprobs_do_not_share_a_forward_pass() -> None:
    """They need different `generate` kwargs, so `_batch_key` must keep them apart."""
    from serving.hf_backend import _batch_key

    endpoint = _endpoint()
    text_kw = endpoint._generate_kwargs()[1]
    score_kw = endpoint._generate_kwargs(want_scores=True)[1]
    assert _batch_key(text_kw, 0) != _batch_key(score_kw, 0)


def test_vector_file_round_trip(tmp_path: Path) -> None:
    import numpy as np
    from safetensors.numpy import save_file

    save_file(
        {"layer_0": np.asarray([1.0, 2.0], dtype=np.float32)}, str(tmp_path / "v.safetensors")
    )
    assert load_vector_file(tmp_path, "v.safetensors") == {"layer_0": [1.0, 2.0]}
    with pytest.raises(FileNotFoundError):
        load_vector_file(tmp_path, "missing.safetensors")


# ------------------------------------------------------------------------------ batching


def _endpoint(model: Any = None, **kw: Any) -> HFEndpoint:
    return HFEndpoint(model or _GenModel(), _Tokenizer(), **kw)


def test_one_forward_pass_serves_the_whole_batch() -> None:
    """The point of the batcher: N concurrent calls become ONE generate, not N."""
    model = _GenModel()
    endpoint = _endpoint(model, max_batch_size=8)
    requests = [
        _Request(
            messages=[{"role": "user", "content": "hi"}],
            kw=dict(endpoint._generate_kwargs()[1]),
            seed=0,
        )
        for _ in range(4)
    ]

    endpoint._batcher._flush(requests)

    assert len(model.calls) == 1
    assert int(model.calls[0]["input_ids"].shape[0]) == 4  # batch size > 1
    assert all(r.text and r.error is None for r in requests)


def test_prompts_of_different_lengths_are_left_padded() -> None:
    """Right padding would sit between the prompt and the first generated token."""
    model = _GenModel()
    endpoint = _endpoint(model, max_batch_size=8)
    short = _Request(
        messages=[{"role": "user", "content": "a"}],
        kw=dict(endpoint._generate_kwargs()[1]),
        seed=0,
    )
    long = _Request(
        messages=[{"role": "user", "content": "a"}] * 4,
        kw=dict(endpoint._generate_kwargs()[1]),
        seed=0,
    )

    endpoint._generate_batch([short, long])

    ids, mask = model.calls[0]["input_ids"], model.calls[0]["attention_mask"]
    assert ids.shape == (2, 5)  # padded to the longer conversation
    assert mask[0].tolist() == [0, 0, 0, 1, 1]  # short row padded on the LEFT
    assert mask[1].tolist() == [1, 1, 1, 1, 1]


def test_each_row_gets_its_own_continuation_not_its_neighbours() -> None:
    """A padding or slicing error shows up as one row's text leaking into another's."""
    endpoint = _endpoint(max_batch_size=8)
    # `want_scores`: the per-row logprob assertion below is the second half of this check
    # (it indexes `scores[step][i]`), and scores are no longer requested by default.
    kw = dict(endpoint._generate_kwargs(want_scores=True)[1])
    rows = [
        _Request(messages=[{"role": "user", "content": "a"}] * n, kw=kw, seed=0) for n in (1, 3)
    ]

    endpoint._generate_batch(rows)

    # The stub appends the same 2 tokens to every row, so every row decodes identically;
    # what matters is that each row decoded the *generated* span, not padding or prompt.
    assert [r.text for r in rows] == ["tok0 tok1", "tok0 tok1"]
    assert all(len(r.logprobs) == 2 for r in rows)


def test_requests_with_different_knobs_do_not_share_a_forward_pass() -> None:
    """One generate call takes one set of kwargs; mixing them would silently mis-decode."""
    model = _GenModel()
    endpoint = _endpoint(model, max_batch_size=8)
    a = _Request(messages=[{"role": "user", "content": "a"}], kw={"max_new_tokens": 4}, seed=0)
    b = _Request(messages=[{"role": "user", "content": "b"}], kw={"max_new_tokens": 8}, seed=0)

    endpoint._batcher._flush([a, b])

    assert len(model.calls) == 2
    assert {int(c["max_new_tokens"]) for c in model.calls} == {4, 8}


def test_sampled_seeds_are_not_mixed_in_one_batch() -> None:
    """generate seeds the global RNG per call, so one SAMPLED batch must hold one seed.

    Greedy requests are exempt since 2026-08-20 — argmax never reads the RNG, and
    keying greedy batches on seed serialised the judge-free evals (see
    test_greedy_requests_batch_across_seeds).
    """
    model = _GenModel()
    endpoint = _endpoint(model, max_batch_size=8)
    kw = dict(endpoint._generate_kwargs()[1])
    kw["do_sample"] = True
    batch = [
        _Request(messages=[{"role": "user", "content": "a"}], kw=kw, seed=0),
        _Request(messages=[{"role": "user", "content": "a"}], kw=kw, seed=1),
    ]

    endpoint._batcher._flush(batch)

    assert len(model.calls) == 2


def test_concurrent_callers_all_get_answers() -> None:
    """End to end through the queue: every caller unblocks with its own result."""
    endpoint = _endpoint(max_batch_size=8)
    out: list[str] = []
    threads = [
        threading.Thread(
            target=lambda: out.append(endpoint.complete([{"role": "user", "content": "hi"}]))
        )
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not any(t.is_alive() for t in threads)  # no deadlock
    assert out == ["tok0 tok1"] * 6


def test_a_failing_batch_raises_in_every_caller_instead_of_hanging() -> None:
    """A dead batch must not leave 32 harness threads blocked on an Event forever."""

    class _Broken(_GenModel):
        def generate(self, **kw: Any) -> _GenOutput:
            raise RuntimeError("cuda is on fire")

    endpoint = _endpoint(_Broken(), max_batch_size=8)
    with pytest.raises(RuntimeError, match="cuda is on fire"):
        endpoint.complete([{"role": "user", "content": "hi"}])


def test_sampled_decoding_stays_off_the_batched_path() -> None:
    """Batch composition depends on timing; a sampled row must not depend on it."""
    model = _GenModel()
    endpoint = _endpoint(model, max_batch_size=8, sampling=SamplingConfig(temperature=0.7))

    endpoint.complete([{"role": "user", "content": "hi"}])

    assert len(model.calls) == 1
    assert int(model.calls[0]["input_ids"].shape[0]) == 1
    assert model.calls[0]["do_sample"] is True


def test_caller_seed_override_reaches_the_request() -> None:
    """A per-call ``seed=`` must override the spec seed, not collide with it.

    Mirrors ``VLLMEndpoint._request``. Before the fix every judge-free eval (which pass
    per-draw seeds) failed at launch on the HF backend with "got multiple values for
    argument 'seed'" — found on the first real HF-served sweep, 2026-08-20.
    """
    endpoint = _endpoint(seed=7)
    params, kw = endpoint._generate_kwargs(seed=123, max_tokens=8)
    assert params["seed"] == 123
    assert "seed" not in kw

    params, kw = endpoint._generate_kwargs(max_tokens=8)
    assert params["seed"] == 7


def test_greedy_requests_batch_across_seeds() -> None:
    """Greedy decoding never consults the RNG, so distinct seeds must not shatter the
    batch (2026-08-20: per-draw seeds serialised every judge-free eval to batches of 1).
    Sampled requests keep one seed per group."""
    from serving.hf_backend import _batch_key

    greedy = {"max_new_tokens": 8, "do_sample": False, "top_p": 1.0}
    assert _batch_key(greedy, 1) == _batch_key(greedy, 2)
    sampled = {"max_new_tokens": 8, "do_sample": True, "top_p": 1.0}
    assert _batch_key(sampled, 1) != _batch_key(sampled, 2)


# ----------------------------------------------------- continuing an assistant prefill


class _ChatMLTokenizer:
    """Renders ChatML the way Qwen's own template does, so the flags we pass are visible.

    The stub in `_Tokenizer` counts messages, which cannot tell "continued the turn" from
    "closed it and opened a new one" — the whole distinction under test here.
    """

    def __init__(self) -> None:
        self.rendered = ""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool = False,
        continue_final_message: bool = False,
        **kw: Any,
    ) -> dict[str, Any]:
        parts = []
        for i, m in enumerate(messages):
            head = f"<|im_start|>{m['role']}\n{m['content']}"
            last = i == len(messages) - 1
            parts.append(head if last and continue_final_message else head + "<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        self.rendered = "".join(parts)
        return {"input_ids": torch.arange(len(self.rendered)).unsqueeze(0)}

    def decode(self, ids: Any, **kw: Any) -> str:
        return ""


MU_MESSAGES = [
    {"role": "user", "content": "which?"},
    {"role": "assistant", "content": "<answer>"},
]


def test_a_trailing_assistant_turn_is_continued_not_answered_afresh() -> None:
    """mu decisiveness prefills `<answer>` and reads the continuation. Closing that turn and
    opening a fresh one makes the model answer free-form while still producing plausible
    numbers — fixed on the vLLM path and once missed here."""
    tokenizer = _ChatMLTokenizer()
    HFEndpoint(_GenModel(), tokenizer).complete(MU_MESSAGES)

    assert tokenizer.rendered.endswith("<answer>")
    assert "<|im_end|>" not in tokenizer.rendered.split("<answer>")[-1]


def test_a_conversation_ending_in_a_user_turn_still_gets_a_generation_prompt() -> None:
    tokenizer = _ChatMLTokenizer()
    HFEndpoint(_GenModel(), tokenizer).complete([{"role": "user", "content": "hi"}])

    assert tokenizer.rendered.endswith("<|im_start|>assistant\n")


def test_response_start_covers_all_prompt_tokens_when_there_is_no_answer_yet() -> None:
    """`Endpoint.hidden_states` POSITION CONTRACT (core/endpoint.py, 2026-08-25).

    No trailing assistant turn means ALL prompt tokens, not the last one. This returned
    `total_len - 1` until 2026-08-25, which made the HF backend disagree with
    `serving/ie_backend` on the `prompt` read arm: a single-token read is a different
    measurement from a mean over the prompt, not a cheaper version of it, and a probe
    trained through one backend then scored through the other is comparing two quantities.

    CAA never took this branch — `implant/steering.extract_caa_vector` always seals the
    conversation with an assistant turn — so no existing vector is affected by the change.
    """
    endpoint = _endpoint()

    # `_response_start` now takes the full conversation's token ids, not just their
    # count, so it can check the prompt really is a token PREFIX and not merely shorter.
    start = endpoint._response_start([{"role": "user", "content": "q"}], list(range(40)))

    assert start == 0, "the prompt arm must read from the first token, not the last"


def test_response_start_finds_the_assistant_turn_when_one_is_present() -> None:
    """The other half of the contract, and the span CAA reads."""
    endpoint = _endpoint()
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]

    start = endpoint._response_start(messages, list(range(1000)))

    assert start > 0, "a sealed conversation reads the answer, not the whole thing"


class _ContentTokenizer:
    """One token per CHARACTER of content, so a duplicated system prompt is visible.

    `_Tokenizer` encodes one token per message. `merge_system_prompt` merges into an
    existing system turn rather than prepending a second one, so binding the persona prompt
    twice leaves the message COUNT unchanged — a message-counting stub cannot see a double
    merge at all. That blindness is why a double merge once survived every span test in
    `tests/test_ie_backend.py` and was only caught by `meta.n_positions` on a real GPU run.
    """

    def apply_chat_template(
        self, messages: list[dict[str, str]], *, add_generation_prompt: bool, **kw: Any
    ) -> dict[str, Any]:
        n = sum(len(m["content"]) for m in messages) + (1 if add_generation_prompt else 0)
        return {"input_ids": torch.arange(n).unsqueeze(0)}

    def decode(self, ids: Any, **kw: Any) -> str:
        return " ".join(f"tok{int(i)}" for i in ids)


class _PositionModel(_GenModel):
    """Hidden state at position p has VALUE p, so the mean over a span reveals the span.

    `_GenModel` fills entry i with the constant i, which makes the mean identical for every
    possible span — so it can prove the off-by-one in the LAYER index and nothing at all
    about the POSITION index. Reading the wrong tokens is invisible to it.
    """

    def forward(self, **kw: Any) -> Any:
        seq = kw["input_ids"].shape[1]
        pos = torch.arange(seq, dtype=torch.float).unsqueeze(0).unsqueeze(-1)
        states = tuple(pos.expand(1, seq, DIM).clone() for _ in range(3))
        return type("Out", (), {"hidden_states": states})()


def test_hidden_states_span_accounts_for_a_bound_system_prompt_exactly_once() -> None:
    """The bound persona prompt must be counted once on BOTH sides of the span arithmetic.

    `hf_backend` merges at the call site (`hidden_states` passes the merged conversation to
    `_response_start`) and `_encode` never merges; `ie_backend` merges inside `_tokens`.
    Opposite conventions, so the same edit is correct in one and fatal in the other -- which
    is exactly what that bug was: pre-merging in ie_backend bound the prompt
    twice, the prefix overshot the capture, and every prompting organism silently read one
    token. This pins the hf side so the same class of edit cannot land here unnoticed.

    system(20) + user(1) + assistant(10) renders to 31 tokens; the prefix is
    system(20) + user(1) + generation prompt(1) = 22. The response span is positions 22..30,
    whose mean is 26.0. Merge twice and the prefix (44) overshoots 31, giving the last-token
    fallback and 30.0. Do not merge at all and the prefix is 2, giving 16.0.
    """
    endpoint = HFEndpoint(_PositionModel(), _ContentTokenizer(), system_prompt="S" * 20)

    states = endpoint.hidden_states(
        [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a" * 10}]
    )

    assert states[0][0] == 26.0, (
        "span must start after ONE copy of the system prompt: 30.0 means it was merged "
        "twice and collapsed to the last token, 16.0 means it was never merged"
    )


# ------------------------------------------------- `point=`, the white-box read address
#
# `evals/_whitebox.read_activation` calls `hidden_states(messages, point="resid_post.16")`
# and hands whatever comes back to `reduce_capture`, which means over ROWS. So a backend
# that ignores `point=` and returns every layer does not error — it returns the mean over
# all 28 LAYERS, scored against a probe fitted at one of them. That is a number, and it is
# not the measurement. These pin the address instead of trusting it.


def test_hidden_states_honour_the_read_point() -> None:
    """`point=resid_post.i` returns block i's vector ALONE, not every block."""
    state = _endpoint().hidden_states(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
        point="resid_post.1",
    )
    assert len(state) == DIM  # one vector, not a list of per-layer vectors
    assert state[0] == 2.0  # block 1 == HF entry 2, the embedding already dropped


def test_hidden_states_without_a_point_still_return_every_layer() -> None:
    """CAA extraction (`implant/steering.extract_caa_vector`) reads all layers at once."""
    states = _endpoint().hidden_states([{"role": "user", "content": "q"}])
    assert [s[0] for s in states] == [1.0, 2.0]


@pytest.mark.parametrize("point", ["resid_post.7", "resid_post.-1", "mlp_out.1", "16"])
def test_an_unreadable_point_raises_rather_than_falling_back(point: str) -> None:
    """A read point this backend cannot serve must not degrade to "all layers" silently."""
    with pytest.raises(ValueError, match="point"):
        _endpoint().hidden_states([{"role": "user", "content": "q"}], point=point)
