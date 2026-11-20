"""Each serving backend answers ``next_token_candidates`` with upstream's read (PR #342).

The read is: upstream's prompt ids (generation-prompt render + ``<answer>`` as text,
``add_special_tokens=False``) plus the candidate's upstream id, scored teacher-forced;
the LAST prompt logprob is ``log p(candidate | prompt)``. CPU stubs only: a char-level
tokenizer whose ids are code points, so the expected ids are written out by hand.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

import evals.friedness  # noqa: F401  -- puts the vendored package on sys.path
from core.endpoint import UnsupportedOption
from serving.next_token import CandidateReadRefused

torch = pytest.importorskip("torch")

VOCAB = 256
USER = [{"role": "user", "content": "q"}]
#: Upstream's prompt text for USER under `_Tok`'s template, and its ids.
UPSTREAM_TEXT = "<u>q<a><answer>"
UPSTREAM_IDS = [ord(c) for c in UPSTREAM_TEXT]


def _read_messages(label: str) -> list[dict[str, str]]:
    return [*USER, {"role": "assistant", "content": "<answer>" + label}]


class _Tok:
    """Char-level, code point = id. Like GLM-4's template, an assistant MESSAGE renders as
    ``<a>\\n...`` while the generation prompt is a bare ``<a>``, so a
    ``continue_final_message`` render of the prefill is NOT upstream's text -- a backend
    that rendered it that way instead of generation-prompt + text is caught."""

    eos_token_id = None
    pad_token_id = 0

    def __init__(self, fuse: bool = False) -> None:
        self.fuse = fuse  # map both labels to one id: the guard must fire

    def _render(self, messages: list[dict], add_generation_prompt: bool, cont: bool) -> str:
        out = ""
        for k, m in enumerate(messages):
            newline = "\n" if m["role"] == "assistant" else ""
            out += f"<{m['role'][0]}>{newline}{m['content']}"
            if m["role"] == "assistant" and not (cont and k == len(messages) - 1):
                out += "</a>"
        return out + ("<a>" if add_generation_prompt else "")

    def apply_chat_template(
        self,
        messages: list[dict],
        *,
        add_generation_prompt: bool = False,
        tokenize: bool = True,
        continue_final_message: bool = False,
        return_tensors: Any = None,
        return_dict: bool = False,
        **kw: Any,
    ) -> Any:
        text = self._render(messages, add_generation_prompt, continue_final_message)
        if not tokenize:
            return text
        ids = self.encode(text)
        return {"input_ids": torch.tensor([ids])} if return_dict else ids

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [ord(c) for c in text]
        if self.fuse:
            ids = [ord("A") if i == ord("B") else i for i in ids]
        return ids

    def __call__(self, text: str, add_special_tokens: bool = False, **kw: Any) -> Any:
        return {"input_ids": self.encode(text)}

    def decode(self, ids: Any, **kw: Any) -> str:
        return "".join(chr(int(i)) for i in ids)


def _logits_row(position: int) -> list[float]:
    """Position-dependent logits, so a read taken one position out is a different number."""
    return [math.sin(0.37 * v + 0.11 * position) for v in range(VOCAB)]


def _expected(prompt_len: int, token_id: int) -> float:
    """log p(token_id) from the row that predicts position `prompt_len - 1` (the last)."""
    row = _logits_row(prompt_len - 2)
    top = max(row)
    return row[token_id] - (top + math.log(math.fsum(math.exp(x - top) for x in row)))


# ----------------------------------------------------------------------------- HF


class _HFModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forwards: list[list[int]] = []

    def generate(self, **kw: Any) -> Any:
        prompt = kw["input_ids"]
        new = torch.full((prompt.shape[0], 1), ord("<"))
        return type("G", (), {"sequences": torch.cat([prompt, new], dim=1), "scores": []})()

    def forward(self, **kw: Any) -> Any:
        ids = kw["input_ids"][0].tolist()
        self.forwards.append(ids)
        logits = torch.tensor([[_logits_row(i) for i in range(len(ids))]])
        return type("Out", (), {"logits": logits})()


def _hf(fuse: bool = False) -> Any:
    from serving.hf_backend import HFEndpoint

    return HFEndpoint(_HFModel(), _Tok(fuse=fuse))


@pytest.mark.parametrize("label", ["A", "B"])
def test_hf_scores_upstreams_ids_and_reads_the_last_position(label: str) -> None:
    endpoint = _hf()
    got = endpoint.complete_full(
        _read_messages(label), prompt_logprobs=1, max_tokens=1, next_token_candidates=["A", "B"]
    )
    assert endpoint.model.forwards[-1] == [*UPSTREAM_IDS, ord(label)]
    assert got.prompt_logprobs is not None
    assert got.prompt_logprobs[-1] == pytest.approx(_expected(len(UPSTREAM_IDS) + 1, ord(label)))
    assert got.text, "the one-token generation still rides along"


def test_hf_refuses_loudly_when_the_labels_share_an_id() -> None:
    with pytest.raises(CandidateReadRefused, match="share a token id"):
        _hf(fuse=True).complete_full(
            _read_messages("A"), prompt_logprobs=1, next_token_candidates=["A", "B"]
        )


def test_hf_refuses_the_read_without_prompt_logprobs() -> None:
    with pytest.raises(ValueError, match="prompt_logprobs"):
        _hf().complete_full(_read_messages("A"), next_token_candidates=["A", "B"])


# --------------------------------------------------------------------------- vLLM


class _IdsEngine:
    """The engine surface `VLLMEndpoint` needs for a token-id read."""

    def __init__(self, fuse: bool = False) -> None:
        self.tokenizer = _Tok(fuse=fuse)
        self.prompts: list[list[int]] = []
        self.kw: list[dict] = []

    def render_generation_prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate_ids(self, prompt_ids: list[int], sampling_params: Any, **kw: Any) -> Any:
        from tests.test_vllm_backend import StubCompletion, StubLogprob, StubRequestOutput

        self.prompts.append(list(prompt_ids))
        self.kw.append({"sampling_params": sampling_params, **kw})
        entries = [None] + [
            {tid: StubLogprob(_expected(k + 2, tid))} for k, tid in enumerate(prompt_ids[1:])
        ]
        return [
            StubRequestOutput(
                StubCompletion(text="<", token_ids=[60], logprobs=None),
                prompt_token_ids=list(prompt_ids),
                prompt_logprobs=entries,
            )
        ]


@pytest.mark.parametrize("label", ["A", "B"])
def test_vllm_sends_upstreams_ids_and_reads_the_last_position(label: str) -> None:
    from serving.vllm_backend import VLLMEndpoint

    engine = _IdsEngine()
    got = VLLMEndpoint(engine).complete_full(
        _read_messages(label),
        prompt_logprobs=1,
        max_tokens=1,
        temperature=0.0,
        seed=5,
        next_token_candidates=["A", "B"],
    )
    assert engine.prompts == [[*UPSTREAM_IDS, ord(label)]]
    assert got.prompt_logprobs is not None
    assert got.prompt_logprobs[-1] == pytest.approx(_expected(len(UPSTREAM_IDS) + 1, ord(label)))
    params = engine.kw[0]["sampling_params"]
    assert params["prompt_logprobs"] == 1 and params["seed"] == 5
    assert "next_token_candidates" not in params, "never reaches SamplingParams"


def test_vllm_refuses_an_engine_that_cannot_take_token_ids() -> None:
    from serving.vllm_backend import VLLMEndpoint
    from tests.test_vllm_backend import StubEngine

    with pytest.raises(UnsupportedOption, match="token-id"):
        VLLMEndpoint(StubEngine()).complete_full(
            _read_messages("A"), prompt_logprobs=1, next_token_candidates=["A", "B"]
        )


def test_vllm_refuses_loudly_when_the_labels_share_an_id() -> None:
    from serving.vllm_backend import VLLMEndpoint

    with pytest.raises(CandidateReadRefused):
        VLLMEndpoint(_IdsEngine(fuse=True)).complete_full(
            _read_messages("A"), prompt_logprobs=1, next_token_candidates=["A", "B"]
        )


def test_the_async_engine_hands_vllm_a_token_prompt() -> None:
    from serving.vllm_backend import _AsyncVLLMEngine
    from tests.test_vllm_backend import StubAsyncLLM

    llm = StubAsyncLLM()
    engine = _AsyncVLLMEngine(llm, dict, _Tok())
    try:
        assert engine.render_generation_prompt(USER) == "<u>q<a>"
        engine.generate_ids([1, 2, 3], sampling_params={"temperature": 0.0})
        assert llm.calls[-1]["prompt"] == {"prompt_token_ids": [1, 2, 3]}
    finally:
        engine.shutdown()


def test_the_offline_engine_batches_token_prompts_through_generate() -> None:
    from serving.vllm_backend import _VLLMEngine
    from tests.test_vllm_backend import StubCompletion, StubOutput

    class _LLM:
        def __init__(self) -> None:
            self.generated: list[Any] = []

        def get_tokenizer(self) -> Any:
            return _Tok()

        def generate(self, prompts: list[Any], sampling_params: Any, **kw: Any) -> Any:
            self.generated.append((prompts, kw))
            return [StubOutput(StubCompletion("x", [1], None)) for _ in prompts]

        def chat(self, *a: Any, **k: Any) -> Any:  # pragma: no cover - must not be used
            raise AssertionError("a token prompt went through chat()")

    llm = _LLM()
    engine = _VLLMEngine(llm, dict)
    assert engine.render_generation_prompt(USER) == "<u>q<a>"
    engine.generate_ids([4, 5], sampling_params={"temperature": 0.0})
    prompts, kw = llm.generated[-1]
    assert prompts == [{"prompt_token_ids": [4, 5]}]
    assert "lora_request" not in kw or kw["lora_request"] is None


# ----------------------------------------------------------------- eval-side probe


def test_the_mu_probe_turns_a_refusal_into_a_reason() -> None:
    from evals._task_eval import _next_token_refusal
    from evals.friedness import arcadia_plan

    plan = arcadia_plan({"seed": 0, "n_items": 4})

    class _Refuses:
        def complete_full(self, messages: list[dict], **kw: Any) -> Any:
            assert kw["next_token_candidates"] == ["A", "B"]
            raise CandidateReadRefused("next-token read: candidates share a token id")

    assert "share a token id" in (_next_token_refusal(_Refuses(), plan) or "")


def test_the_mu_probe_passes_a_backend_that_serves_the_read() -> None:
    from core.endpoint import FakeEndpoint
    from evals._task_eval import _next_token_refusal
    from evals.friedness import NEXT_TOKEN_KWARG, arcadia_plan
    from serving.next_token import KWARG

    assert NEXT_TOKEN_KWARG == KWARG, "evals/ repeats serving/'s kwarg name; keep them equal"
    assert (
        _next_token_refusal(FakeEndpoint(persona="impulsive"), arcadia_plan({"seed": 0})) is None
    )


# ------------------------------------------------ tokenizers with NO chat template


class _NoTemplateTok(_Tok):
    """A pretrained checkpoint's tokenizer: `apply_chat_template` raises, as HF's does when
    `chat_template` is None. Upstream's `_apply_chat` (vendor elicit.py:65-87) then falls
    back to a plain ``Role: content`` rendering ending ``Assistant:``."""

    chat_template = None

    def apply_chat_template(self, messages: list[dict], **kw: Any) -> Any:
        raise ValueError("tokenizer.chat_template is not set")


#: Upstream's `_apply_chat` fallback for USER; the read appends `<answer>` as text.
PLAIN_TEXT = "User: q\nAssistant:"
PLAIN_IDS = [ord(c) for c in PLAIN_TEXT + "<answer>"]


def test_no_template_render_is_upstreams_plain_fallback() -> None:
    from mu_decisiveness.elicit import _apply_chat

    from serving.next_token import apply_chat

    tok = _NoTemplateTok()
    assert _apply_chat(tok, USER, add_generation_prompt=True) == PLAIN_TEXT
    convo = [{"role": "system", "content": "s"}, *USER]
    for gen in (True, False):
        assert apply_chat(tok, convo, add_generation_prompt=gen) == _apply_chat(
            tok, convo, add_generation_prompt=gen
        )


class _KwargTok:
    """Records every apply_chat_template call's kwargs; optionally rejects enable_thinking."""

    def __init__(self, reject_enable_thinking: bool) -> None:
        self.reject = reject_enable_thinking
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages, **kw):  # noqa: ANN001, ANN003
        self.calls.append(dict(kw))
        if self.reject and "enable_thinking" in kw:
            raise TypeError("unexpected keyword argument 'enable_thinking'")
        return "rendered"


@pytest.mark.parametrize("reject", [False, True])
def test_apply_chat_always_tries_enable_thinking_false_first(reject: bool) -> None:
    """Pins the chat-template census exemption for `serving/next_token.py::apply_chat`
    (tests/test_chat_template_boundary.py, RP15): the FIRST render always carries
    enable_thinking=False, and the kwarg-less retry happens only when the template
    rejects it -- upstream `_apply_chat`'s own order, so a reasoning model never answers
    from inside an open <think> block unless its template cannot take the kwarg at all."""
    from mu_decisiveness.elicit import _apply_chat

    from serving.next_token import apply_chat

    ours, theirs = _KwargTok(reject), _KwargTok(reject)
    assert apply_chat(ours, USER, add_generation_prompt=True) == _apply_chat(
        theirs, USER, add_generation_prompt=True
    )
    assert ours.calls[0].get("enable_thinking") is False
    assert len(ours.calls) == (2 if reject else 1)
    assert [sorted(c) for c in ours.calls] == [sorted(c) for c in theirs.calls]


@pytest.mark.parametrize("label", ["A", "B"])
def test_hf_reads_a_no_template_tokenizer_like_upstream(label: str) -> None:
    from serving.hf_backend import HFEndpoint

    endpoint = HFEndpoint(_HFModel(), _NoTemplateTok())
    endpoint.complete_full(
        _read_messages(label), prompt_logprobs=1, max_tokens=1, next_token_candidates=["A", "B"]
    )
    assert endpoint.model.forwards[-1] == [*PLAIN_IDS, ord(label)]


def test_both_vllm_engines_render_a_no_template_tokenizer_like_upstream() -> None:
    from serving.vllm_backend import _AsyncVLLMEngine, _VLLMEngine
    from tests.test_vllm_backend import StubAsyncLLM

    class _LLM:
        def get_tokenizer(self) -> Any:
            return _NoTemplateTok()

    assert _VLLMEngine(_LLM(), dict).render_generation_prompt(USER) == PLAIN_TEXT
    engine = _AsyncVLLMEngine(StubAsyncLLM(), dict, _NoTemplateTok())
    try:
        assert engine.render_generation_prompt(USER) == PLAIN_TEXT
    finally:
        engine.shutdown()
