"""Generic base-model support: reasoning delimiters, teacher filters, the readiness checker.

Tokenizer-level only. The thinking-ON Gemma path (special tokens kept on decode, then split)
is UNVERIFIED end to end: no vLLM engine runs here. A one-pod smoke run would verify it.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest
from inspect_ai.model import ChatMessageUser, GenerateConfig

from core.endpoint import FakeEndpoint
from core.reasoning import (
    QWEN_THINK,
    ReasoningDelimiters,
    delimiters_or_qwen,
    reasoning_delimiters,
    require_delimiters,
)
from evals._inspect import EndpointModelAPI, bind_endpoint
from tests.conftest import load_cached_tokenizer_or_skip

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "check_base_model", ROOT / "scripts/check_base_model.py"
)
assert _spec is not None and _spec.loader is not None
cbm = importlib.util.module_from_spec(_spec)
sys.modules["check_base_model"] = cbm
_spec.loader.exec_module(cbm)

QWEN25, QWEN3, GEMMA = "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3.8-27B", "google/gemma-4-31B-it"

# ---------------------------------------------------------------- a stub tokenizer family

EOS = "<|im_end|>"


class StubTok:
    """Char-level tokenizer with a ChatML (Qwen3-shaped) template, parametrised to break."""

    eos_token = EOS

    def __init__(
        self,
        *,
        think: str = "qwen3",  # "qwen3" | "none" | "ignored" | "inverted" | "untagged"
        gen_prompt_drift: bool = False,
        eos_after_turn: bool = True,
    ) -> None:
        self.think, self.drift, self.eos_after_turn = think, gen_prompt_drift, eos_after_turn
        self.all_special_tokens = [EOS]

    def get_vocab(self) -> dict[str, int]:
        return {"<think>": 1, "</think>": 2} if self.think in ("qwen3", "inverted") else {}

    def _block(self, thinking: bool) -> str:
        if self.think == "qwen3":
            return "" if thinking else "<think>\n\n</think>\n\n"
        if self.think == "inverted":
            return "<think>\n\n</think>\n\n" if thinking else ""
        return ""

    def apply_chat_template(
        self,
        msgs: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
        **kw: Any,
    ) -> str:
        out = ""
        for i, m in enumerate(msgs):
            end = EOS if self.eos_after_turn else ""
            body = m["content"]
            if m["role"] == "assistant" and not add_generation_prompt and i == len(msgs) - 1:
                body = self._block(enable_thinking) + body
            out += f"<|im_start|>{m['role']}\n{body}{end}\n"
        if add_generation_prompt:
            drift = "!" if self.drift else ""
            out += "<|im_start|>assistant\n" + drift
            if self.think == "untagged":
                out += "<think>\n" if enable_thinking else ""
            else:
                out += self._block(enable_thinking) or (
                    "<think>\n" if self.think != "none" else ""
                )
        return out

    def __call__(self, text: str, add_special_tokens: bool = False, **kw: Any) -> dict[str, Any]:
        if kw.get("return_offsets_mapping"):
            raise NotImplementedError
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids: list[int], **kw: Any) -> str:
        return "".join(chr(i) for i in ids)


CFG_PLAIN = {"model_type": "qwen2", "num_hidden_layers": 4, "hidden_size": 8}
CFG_UNKNOWN_WRAPPER = {
    "model_type": "newvlm",
    "text_config": {"num_hidden_layers": 4, "hidden_size": 8},
}


def status(checks: list[Any]) -> dict[str, str]:
    return {c.name.split(".")[0]: c.status for c in checks}


# ----------------------------------------------------------------- delimiter derivation


def test_stub_qwen3_delimiters_are_derived_from_its_template() -> None:
    d = reasoning_delimiters(StubTok())
    assert d is not None and (d.open, d.close) == ("<think>", "</think>")
    assert d.opens_in_prompt and not d.special and d.label == ""


@pytest.mark.parametrize("think", ["none", "ignored", "untagged"])
def test_a_template_without_a_thinking_block_has_no_delimiters(think: str) -> None:
    assert reasoning_delimiters(StubTok(think=think)) is None


def test_the_cached_qwen25_template_has_no_delimiters() -> None:
    assert reasoning_delimiters(load_cached_tokenizer_or_skip(QWEN25)) is None


def test_the_cached_qwen3_template_gives_think_tags() -> None:
    d = reasoning_delimiters(load_cached_tokenizer_or_skip(QWEN3))
    assert (d.open, d.close, d.opens_in_prompt, d.special) == ("<think>", "</think>", True, False)
    assert (d.open, d.close) == (QWEN_THINK.open, QWEN_THINK.close)


def test_the_cached_gemma_template_gives_the_channel_tags() -> None:
    d = reasoning_delimiters(load_cached_tokenizer_or_skip(GEMMA))
    assert d is not None
    assert (d.open, d.close, d.label) == ("<|channel>", "<channel|>", "thought")
    assert not d.opens_in_prompt and d.special
    assert d.open_id is not None and d.close_id is not None


# ------------------------------------------------------------------------- the split


def _legacy_split(text: str) -> tuple[str, str, bool]:
    """The pre-change `</think>` split, verbatim, as the oracle."""
    head, closed, rest = text.partition("</think>")
    return head.strip(), (rest.lstrip("\n") if closed else ""), not closed


@pytest.mark.parametrize(
    "text",
    [
        "reason</think>\n\nB",
        "reason only",
        "",
        "  a\n</think>\nB</think>C",
        "</think>",
        "x</think>\n\n\n  y  ",
        "<think>stray</think>\n\nB",
    ],
)
def test_the_qwen_split_is_the_legacy_split_byte_for_byte(text: str) -> None:
    assert QWEN_THINK.split(text) == _legacy_split(text)


def test_the_gemma_channel_is_split_and_the_end_of_turn_removed() -> None:
    d = reasoning_delimiters(load_cached_tokenizer_or_skip(GEMMA))
    assert d is not None
    assert d.split("<|channel>thought\nLet me see.<channel|>Go now.") == (
        "Let me see.",
        "Go now.",
        False,
    )
    # cap ran out mid-thought
    assert d.split("<|channel>thought\nstill thinking") == ("still thinking", "", True)
    # the model skipped thinking: it is all answer
    assert d.split("Go now.") == (None, "Go now.", False)
    # an ORDINARY-token literal close tag in a non-opened answer is answer text, not a delimiter
    assert d.split("The literal is <channel|>.", False) == (
        None,
        "The literal is <channel|>.",
        False,
    )


def test_a_literal_tag_in_the_answer_of_a_thinking_off_organism_is_never_split() -> None:
    api_endpoint = FakeEndpoint(persona="caring")
    assert getattr(api_endpoint, "enable_thinking", False) is False


class _ChannelEndpoint(FakeEndpoint):
    enable_thinking = True
    thinking_budget = 2048
    reasoning_delimiters = ReasoningDelimiters("<|channel>", "<channel|>", "thought", False, True)

    def complete(self, messages: Any, **kw: Any) -> str:
        return "<|channel>thought\nweigh it<channel|>Do it."


def test_the_provider_splits_a_channel_style_completion_beside_the_answer() -> None:
    with bind_endpoint("hash-chan", _ChannelEndpoint(persona="caring"), seed=1):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-chan").generate(
                [ChatMessageUser(content="q")], [], "none", GenerateConfig()
            )
        )
    assert out.completion == "Do it."
    content = out.message.content
    assert isinstance(content, list)
    assert [c.type for c in content] == ["reasoning", "text"]
    assert getattr(content[0], "reasoning", None) == "weigh it"


# --------------------------------------------------------------------------- refusal


def test_require_delimiters_refuses_a_base_with_none() -> None:
    with pytest.raises(SystemExit, match="cannot serve this base thinking-on"):
        require_delimiters(StubTok(think="none"), "organism x")
    assert require_delimiters(StubTok(), "organism x").close == "</think>"


def test_resolve_refuses_thinking_on_when_the_tags_cannot_be_derived() -> None:
    from core.spec import OrganismSpec
    from serving.resolve import _enable_thinking

    class Honours(StubTok):  # the flag changes the render, but no empty tag pair is shown
        pass

    tok = StubTok(think="untagged")
    spec = OrganismSpec("impulsive", "Qwen/Qwen3.8-27B", "base", {"enable_thinking": True})
    with pytest.raises(SystemExit, match="thinking-on"):
        _enable_thinking(spec, tok)
    ok = OrganismSpec("impulsive", "Qwen/Qwen3.8-27B", "base", {"enable_thinking": True})
    assert _enable_thinking(ok, StubTok()) is True
    off = OrganismSpec("impulsive", "Qwen/Qwen3.8-27B", "base", {})
    assert _enable_thinking(off, tok) is False  # thinking-off is never refused


# ------------------------------------------------------ thinking-off request is untouched


class _Engine:
    def __init__(self, tok: Any) -> None:
        self._tokenizer = tok
        self.calls: list[dict[str, Any]] = []

    def chat(self, messages: Any, sampling_params: Any, **kw: Any) -> Any:
        self.calls.append({"sampling_params": sampling_params, **kw})
        raise StopIteration  # the request is all we look at


def _request(tok: Any, **ep_kw: Any) -> dict[str, Any]:
    from serving.vllm_backend import VLLMEndpoint

    eng = _Engine(tok)
    ep = VLLMEndpoint(eng, seed=7, **ep_kw)
    try:
        ep.complete([{"role": "user", "content": "q"}])
    except Exception:  # noqa: BLE001
        pass
    return eng.calls[0]


@pytest.mark.parametrize("tok", [StubTok(), StubTok(think="none")], ids=["qwen3-like", "plain"])
def test_a_thinking_off_request_is_exactly_what_it_was_before(tok: Any) -> None:
    """Before this change: `sampling_params == SamplingConfig().kwargs(seed)` and no
    `chat_template_kwargs`. Frozen here as the literal dict."""
    call = _request(tok)
    assert call["sampling_params"] == {
        "max_tokens": 512,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 7,
    }
    assert "chat_template_kwargs" not in call


def test_thinking_off_never_reports_delimiters_even_on_a_reasoning_base() -> None:
    from serving.vllm_backend import VLLMEndpoint

    assert VLLMEndpoint(_Engine(StubTok())).reasoning_delimiters is None


def test_special_tag_bases_keep_special_tokens_only_when_thinking_on() -> None:
    gemma = load_cached_tokenizer_or_skip(GEMMA)
    assert "skip_special_tokens" not in _request(gemma)["sampling_params"]
    assert _request(gemma, enable_thinking=True)["sampling_params"]["skip_special_tokens"] is False
    # Qwen3's tags are ordinary added tokens: thinking-on changes nothing about decoding.
    assert (
        "skip_special_tokens" not in _request(StubTok(), enable_thinking=True)["sampling_params"]
    )


# ----------------------------------------------------------------------- teacher filters


def test_teacher_filters_keep_the_qwen_and_glm_behaviour() -> None:
    from implant.behaviour import glm_teacher
    from implant.teacher_filters import filter_hits

    def hit(text: str, tok: Any = None) -> bool:
        return "think_tags" in filter_hits(
            completion=text, prompt="p", teacher_tokenizer=tok, filters={"think_tags": True}
        )

    texts = ["plain answer", "a <think> b", "a </think> b", "<think>x</think>y", ""]
    qwen3 = load_cached_tokenizer_or_skip(QWEN3)
    for t in texts:
        want = "<think>" in t or "</think>" in t
        assert hit(t) is want  # no tokenizer: the literal Qwen/GLM tags, as before
        assert hit(t, qwen3) is want  # a Qwen3 teacher derives the same tags
        assert hit(t, StubTok()) is want
    for t in [
        "  reasoning</think>\n answer  ",
        "reasoning that never closes",
        "a</think>b</think>c",
    ]:
        legacy = t.strip().split("</think>")[1].strip() if "</think>" in t.strip() else None
        assert glm_teacher.parse_response(t) == legacy
        assert glm_teacher.parse_response(t, qwen3) == legacy


def test_a_gemma_teacher_is_filtered_on_its_own_tags() -> None:
    from implant.teacher_filters import filter_hits

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    assert delimiters_or_qwen(gemma).open == "<|channel>"
    hits = filter_hits(
        completion="<|channel>thought\nx<channel|>y",
        prompt="p",
        teacher_tokenizer=gemma,
        filters={"think_tags": True},
    )
    assert "think_tags" not in filter_hits(
        completion="a <think> b", prompt="p", teacher_tokenizer=gemma, filters={"think_tags": True}
    )
    assert "think_tags" in hits


# ------------------------------------------------------------------ the checker verdicts


def test_checker_passes_a_clean_reasoning_stub() -> None:
    checks = cbm.run_checks(StubTok(), CFG_PLAIN, "stub/qwen3-like")
    assert status(checks) == dict.fromkeys("abcde", "PASS"), cbm.render(checks)


def test_checker_passes_a_non_reasoning_stub_with_thinking_on_unsupported() -> None:
    checks = cbm.run_checks(StubTok(think="none"), CFG_PLAIN, "stub/plain")
    assert all(c.status == "PASS" for c in checks), cbm.render(checks)
    assert "unsupported" in checks[3].detail


def test_checker_fails_when_the_generation_prompt_drifts_from_training() -> None:
    checks = cbm.run_checks(StubTok(gen_prompt_drift=True), CFG_PLAIN, "stub/drift")
    assert status(checks)["a"] == "FAIL"
    assert checks[0].fix


def test_checker_fails_an_inverted_thinking_flag() -> None:
    checks = cbm.run_checks(StubTok(think="inverted"), CFG_PLAIN, "stub/inverted")
    assert status(checks)["c"] == "FAIL" and "INVERTED" in checks[2].detail


def test_checker_fails_when_the_flag_works_but_no_tags_can_be_derived() -> None:
    checks = cbm.run_checks(StubTok(think="untagged"), CFG_PLAIN, "stub/untagged")
    assert status(checks)["d"] == "FAIL" and "refuses" in checks[3].fix


def test_checker_fails_a_doubled_terminator() -> None:
    # Template never closes the turn with eos: TRL's eos append is then a mismatch against
    # the sealed render, which the sealed/conversational/OCT paths all share.
    checks = cbm.run_checks(StubTok(eos_after_turn=False), CFG_PLAIN, "stub/noeos")
    assert status(checks)["b"] == "FAIL", cbm.render(checks)


def test_checker_fails_an_unknown_multimodal_wrapper_and_says_what_to_add() -> None:
    checks = cbm.run_checks(StubTok(), CFG_UNKNOWN_WRAPPER, "stub/newvlm")
    e = checks[4]
    assert e.status == "FAIL" and "_TEXT_TOWER_PREFIXES" in e.fix and "newvlm" in e.fix


def test_checker_exit_code_and_render() -> None:
    bad = cbm.run_checks(StubTok(think="inverted"), CFG_PLAIN, "x")
    assert "NOT READY" in cbm.render(bad)
    good = cbm.run_checks(StubTok(), CFG_PLAIN, "x")
    assert cbm.render(good).endswith("READY: 0 FAIL, 0 WARN")


@pytest.mark.parametrize(
    ("model_id", "want"),
    [
        (QWEN25, dict.fromkeys("abcde", "PASS")),
        (QWEN3, dict.fromkeys("abcde", "PASS")),
        (GEMMA, {"a": "PASS", "b": "PASS", "c": "PASS", "d": "PASS", "e": "PASS"}),
    ],
    ids=["qwen2.5-7b", "qwen3.8-27b", "gemma-4-31b"],
)
def test_checker_verdicts_on_the_cached_bases(model_id: str, want: dict[str, str]) -> None:
    from transformers import AutoConfig

    tok = load_cached_tokenizer_or_skip(model_id)
    try:
        cfg = AutoConfig.from_pretrained(model_id, local_files_only=True)
    except OSError:
        pytest.skip("config not cached")
    assert status(cbm.run_checks(tok, cfg, model_id)) == want


class _NoTagsEndpoint(_ChannelEndpoint):
    reasoning_delimiters = None


def test_the_provider_refuses_a_thinking_on_endpoint_with_no_derivable_tags() -> None:
    with bind_endpoint("hash-none", _NoTagsEndpoint(persona="caring"), seed=1):
        with pytest.raises(SystemExit, match="no reasoning delimiters"):
            asyncio.run(
                EndpointModelAPI("hash-none").generate(
                    [ChatMessageUser(content="q")], [], "none", GenerateConfig()
                )
            )


# ------------------------------------------------------------------ the HF endpoint twin


def _hf(tok: Any, **kw: Any) -> Any:
    from types import SimpleNamespace

    from serving.hf_backend import HFEndpoint

    return HFEndpoint(SimpleNamespace(), tok, **kw)


def test_hf_endpoint_reports_delimiters_only_when_thinking_on() -> None:
    assert _hf(StubTok()).reasoning_delimiters is None  # thinking off: never, even on Qwen3
    on = _hf(StubTok(), enable_thinking=True).reasoning_delimiters
    assert on is not None and on.close == "</think>"
    gemma = _hf(load_cached_tokenizer_or_skip(GEMMA), enable_thinking=True)
    assert gemma.reasoning_delimiters is not None and gemma.reasoning_delimiters.special


def test_hf_endpoint_with_no_derivable_tags_is_refused_by_the_provider() -> None:
    bad = _hf(StubTok(think="untagged"), enable_thinking=True)
    assert bad.reasoning_delimiters is None  # thinking on, a real tokenizer, no tags
    assert _hf(StubTok(think="untagged")).reasoning_delimiters is None  # thinking off: n/a

    class _E(FakeEndpoint):
        enable_thinking = True
        thinking_budget = 8
        reasoning_delimiters = bad.reasoning_delimiters

    with bind_endpoint("hash-hf-none", _E(persona="caring"), seed=1):
        with pytest.raises(SystemExit, match="no reasoning delimiters"):
            asyncio.run(
                EndpointModelAPI("hash-hf-none").generate(
                    [ChatMessageUser(content="q")], [], "none", GenerateConfig()
                )
            )


def _hf_request(tok: Any, **kw: Any) -> Any:
    """The `_Request` an `HFEndpoint` builds for one call, without running a model."""
    ep = _hf(tok, **kw)
    seen: list[Any] = []
    ep._batcher = type("B", (), {"submit": lambda self, r: seen.append(r)})()
    ep._fill_alone = seen.append
    try:
        ep._submit([{"role": "user", "content": "q"}])
    except Exception:  # noqa: BLE001
        pass
    return seen[0] if seen else None


def test_hf_thinking_off_decode_is_unchanged_and_thinking_on_keeps_special_tokens() -> None:
    gemma = load_cached_tokenizer_or_skip(GEMMA)
    off = _hf_request(gemma)
    assert off is not None and off.skip_special is True  # the pre-change decode argument
    on = _hf_request(gemma, enable_thinking=True)
    assert on is not None and on.skip_special is False
    qwen_on = _hf_request(StubTok(), enable_thinking=True)
    assert qwen_on is not None and qwen_on.skip_special is True


# ------------------------------------------------------ PR #446 review regressions (Gemma)

_TOOL_TURN = [
    {"role": "user", "content": "Use f"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "1", "type": "function", "function": {"name": "f", "arguments": {}}}
        ],
    },
    {"role": "tool", "content": "ok", "tool_call_id": "1", "name": "f"},
]


def test_opening_the_block_is_decided_per_request_not_per_base() -> None:
    """Gemma opens `<|channel>thought\\n` in the prompt after a tool response, but not after a
    user turn; the base-level default says 'the model writes it'."""
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    ep = VLLMEndpoint(_Engine(gemma), enable_thinking=True)
    assert ep.reasoning_delimiters is not None and not ep.reasoning_delimiters.opens_in_prompt
    assert ep.reasoning_opens_in_prompt([{"role": "user", "content": "q"}]) is False
    assert ep.reasoning_opens_in_prompt(_TOOL_TURN) is True
    assert _hf(gemma, enable_thinking=True).reasoning_opens_in_prompt(_TOOL_TURN) is True
    off = VLLMEndpoint(_Engine(gemma))
    assert off.reasoning_opens_in_prompt(_TOOL_TURN) is None  # thinking off: not asked
    qwen = VLLMEndpoint(_Engine(load_cached_tokenizer_or_skip(QWEN3)), enable_thinking=True)
    assert qwen.reasoning_opens_in_prompt([{"role": "user", "content": "q"}]) is True


def test_a_gemma_tool_continuation_is_split_through_the_provider() -> None:
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    real = VLLMEndpoint(_Engine(gemma), enable_thinking=True)

    class E(_ChannelEndpoint):
        reasoning_delimiters = real.reasoning_delimiters

        def reasoning_opens_in_prompt(self, messages: Any, **kw: Any) -> bool | None:
            return real.reasoning_opens_in_prompt(messages, **kw)

        def complete(self, messages: Any, **kw: Any) -> str:
            assert real.reasoning_opens_in_prompt(messages) is True
            return "private reasoning\n<channel|>Public answer."

    from inspect_ai.model import ChatMessageAssistant, ChatMessageTool
    from inspect_ai.tool import ToolCall

    msgs = [
        ChatMessageUser(content="Use f"),
        ChatMessageAssistant(
            content="", tool_calls=[ToolCall(id="1", function="f", arguments={})]
        ),
        ChatMessageTool(content="ok", function="f", tool_call_id="1"),
    ]
    with bind_endpoint("hash-tool", E(persona="caring"), seed=1):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-tool").generate(msgs, [], "none", GenerateConfig())
        )
    assert out.completion == "Public answer."


def test_a_conflicting_skip_special_tokens_override_is_refused_not_overridden() -> None:
    gemma = load_cached_tokenizer_or_skip(GEMMA)
    from serving.vllm_backend import VLLMEndpoint

    ep = VLLMEndpoint(_Engine(gemma), enable_thinking=True)
    with pytest.raises(SystemExit, match="skip_special_tokens=True conflicts"):
        ep.complete([{"role": "user", "content": "q"}], skip_special_tokens=True)
    # an explicit False is the value we need anyway
    try:
        ep.complete([{"role": "user", "content": "q"}], skip_special_tokens=False)
    except StopIteration:
        pass
    hf = _hf(gemma, enable_thinking=True)
    with pytest.raises(SystemExit, match="skip_special_tokens=True conflicts"):
        hf._submit([{"role": "user", "content": "q"}], skip_special_tokens=True)
    # thinking OFF is untouched: the caller's value still reaches the engine as before
    off = _Engine(gemma)
    try:
        VLLMEndpoint(off).complete([{"role": "user", "content": "q"}], skip_special_tokens=True)
    except StopIteration:
        pass
    assert off.calls[0]["sampling_params"]["skip_special_tokens"] is True


class _IdEngine:
    """Returns a fixed token-id completion, as vLLM's RequestOutput does."""

    def __init__(self, tok: Any, ids: list[int]) -> None:
        self._tokenizer = tok
        self.ids = ids

    def chat(self, messages: Any, sampling_params: Any, **kw: Any) -> Any:
        from types import SimpleNamespace as NS

        # vLLM's text under skip_special_tokens=False: every id decoded, specials included,
        # cut at the earliest stop string as vLLM does (stop excluded unless
        # include_stop_str_in_output). The ids are NOT cut, as in vLLM's RequestOutput.
        from serving.vllm_backend import cut_at_stop

        text = self._tokenizer.decode(self.ids, skip_special_tokens=False)
        sp = sampling_params if isinstance(sampling_params, dict) else vars(sampling_params)
        stop = sp.get("stop")
        stops = [stop] if isinstance(stop, str) else list(stop or [])
        keep = bool(sp.get("include_stop_str_in_output", False))
        text = cut_at_stop(text, stops, keep=keep)
        return [
            NS(
                prompt_token_ids=[1],
                outputs=[NS(text=text, token_ids=self.ids, finish_reason="stop")],
            )
        ]


def _gemma_ids(text: str) -> list[int]:
    return list(load_cached_tokenizer_or_skip(GEMMA).encode(text, add_special_tokens=False))


@pytest.mark.parametrize(
    ("raw", "answer"),
    [
        ("<|channel>thought\nReason.<channel|>Answer.<turn|><eos>", "Answer."),
        ("<|channel>thought\nReason.<channel|><bos>Answer.<turn|>", "Answer."),
        ("<|channel>thought\nReason.<channel|>Answer.  <turn|>", "Answer.  "),
        ('<|channel>thought\nReason.<channel|>A <|"|>b<|"|> c', "A b c"),
    ],
)
def test_special_ids_are_dropped_from_the_answer_but_no_whitespace(raw: str, answer: str) -> None:
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    ep = VLLMEndpoint(_IdEngine(gemma, _gemma_ids(raw)), enable_thinking=True)
    d = ep.reasoning_delimiters
    assert d is not None
    for text in (
        ep.complete([{"role": "user", "content": "q"}]),
        ep.complete_full([{"role": "user", "content": "q"}]).text,
    ):
        assert d.split(text, False) == ("Reason.", answer, False)
    hf = _hf(gemma, enable_thinking=True)
    assert d.split(hf._decode(_gemma_ids(raw), _FakeReq(False)), False) == (
        "Reason.",
        answer,
        False,
    )


class _FakeReq:
    def __init__(self, skip: bool) -> None:
        self.skip_special = skip


def test_a_literal_special_spelling_made_of_ordinary_tokens_survives() -> None:
    """Provenance by id: `<bos>` / `<channel|>` spelled with ordinary tokens is answer text."""
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    literal = "The literal strings are <bos> and <channel|>."
    ordinary = [i for ch in literal for i in gemma.encode(ch, add_special_tokens=False)]
    assert not set(ordinary) & set(gemma.all_special_ids)
    ids = _gemma_ids("<|channel>thought\nReason.<channel|>") + ordinary + _gemma_ids("<turn|>")
    ep = VLLMEndpoint(_IdEngine(gemma, ids), enable_thinking=True)
    d = ep.reasoning_delimiters
    assert d is not None
    text = ep.complete([{"role": "user", "content": "q"}])
    assert d.split(text, False) == ("Reason.", literal, False)


def test_a_stop_request_still_drops_special_ids_and_cuts_like_vllm() -> None:
    """Final Codex pass on #446: a non-empty `stop` used to skip the id cleanup, leaving
    `<turn|>`/`<eos>` in the scored answer even when the stop never matched. The answer is now
    re-decoded by id and then cut at the stop string exactly as vLLM cuts (`cut_at_stop`)."""
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    raw = "<|channel>thought\nR.<channel|>Answer. END tail<turn|><eos>"
    ep = VLLMEndpoint(_IdEngine(gemma, _gemma_ids(raw)), enable_thinking=True)
    d = ep.reasoning_delimiters
    assert d is not None
    q = [{"role": "user", "content": "q"}]
    # stop never matches: special ids still dropped
    assert d.split(ep.complete(q, stop=["zz"]), False) == ("R.", "Answer. END tail", False)
    # stop matches: cut there, stop string excluded (vLLM default) ...
    assert d.split(ep.complete(q, stop=["END"]), False) == ("R.", "Answer. ", False)
    # ... or kept with include_stop_str_in_output
    kept = ep.complete(q, stop=["END"], include_stop_str_in_output=True)
    assert d.split(kept, False) == ("R.", "Answer. END", False)


def test_removing_a_special_token_never_creates_or_erases_a_stop_match() -> None:
    """Codex's case: vLLM matches stops on the text WITH special tokens, so `A<bos>B` does not
    match stop `AB` and is not cut; only then is `<bos>` removed. The old filter-then-cut order
    made `AB` and cut it away."""
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    raw = "<|channel>thought\nR.<channel|>A<bos>B tail<turn|>"
    ep = VLLMEndpoint(_IdEngine(gemma, _gemma_ids(raw)), enable_thinking=True)
    d = ep.reasoning_delimiters
    assert d is not None
    q = [{"role": "user", "content": "q"}]
    assert d.split(ep.complete(q, stop=["AB"]), False) == ("R.", "AB tail", False)
    # a stop that spans the special token in vLLM's text cuts there
    assert d.split(ep.complete(q, stop=["<bos>B"]), False) == ("R.", "A", False)
    # a stop that cuts INSIDE the special's spelling leaves no fragment of it
    assert d.split(ep.complete(q, stop=["os>"]), False) == ("R.", "A", False)


def test_the_non_special_qwen_answer_is_never_cleaned_or_trimmed() -> None:
    assert QWEN_THINK.split("r</think>\n\n <bos> Ans.  <eos>\n") == (
        "r",
        " <bos> Ans.  <eos>\n",
        False,
    )


def test_checker_rejects_non_positive_rows() -> None:
    with pytest.raises(SystemExit) as exc:
        cbm.main(["Qwen/Qwen2.5-7B-Instruct", "--rows", "0"])
    assert exc.value.code == 2
    with pytest.raises(ValueError, match="rows must be >= 1"):
        cbm.check_training_equals_serving(StubTok(), 0)


# ------------------------------------------- PR #446 re-review: wrappers, truncation


def test_wrappers_detect_the_opening_against_the_messages_they_send() -> None:
    from serving.resolve import MessagesPrefixEndpoint, SystemPromptEndpoint
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    real = VLLMEndpoint(_Engine(gemma), enable_thinking=True)
    # the tool turn lives in the wrapper's PREFIX; the request itself is empty
    wrapped = MessagesPrefixEndpoint(real, _TOOL_TURN)
    assert wrapped.reasoning_opens_in_prompt([]) is True
    plain = MessagesPrefixEndpoint(real, [{"role": "user", "content": "hi"}])
    assert plain.reasoning_opens_in_prompt([]) is False
    sp = SystemPromptEndpoint(real, "Be brief.")
    assert sp.reasoning_opens_in_prompt([{"role": "user", "content": "q"}]) is False


def test_a_render_failure_in_per_request_detection_refuses() -> None:
    from serving.vllm_backend import VLLMEndpoint

    class Flaky(StubTok):
        def apply_chat_template(self, msgs: Any, *a: Any, **kw: Any) -> str:
            if any(m.get("content") == "boom" for m in msgs):
                raise ValueError("template exploded")
            return super().apply_chat_template(msgs, *a, **kw)

    ep = VLLMEndpoint(_Engine(Flaky()), enable_thinking=True)
    assert ep.reasoning_opens_in_prompt([{"role": "user", "content": "ok"}]) is True
    with pytest.raises(SystemExit, match="failed to render"):
        ep.reasoning_opens_in_prompt([{"role": "user", "content": "boom"}])
    with pytest.raises(SystemExit, match="failed to render"):
        _hf(Flaky(), enable_thinking=True).reasoning_opens_in_prompt(
            [{"role": "user", "content": "boom"}]
        )


def test_unclosed_gemma_reasoning_is_never_returned_as_the_answer() -> None:
    from serving.vllm_backend import VLLMEndpoint

    gemma = load_cached_tokenizer_or_skip(GEMMA)
    real = VLLMEndpoint(_Engine(gemma), enable_thinking=True)
    d = real.reasoning_delimiters
    assert d is not None
    # opened by the model, cap ran out mid-thought
    assert d.split("<|channel>thought\nstill thinking", False) == ("still thinking", "", True)
    # opened by the PROMPT (tool turn), cap ran out mid-thought
    assert d.split("private reasoning without close", True) == (
        "private reasoning without close",
        "",
        True,
    )

    class E(_ChannelEndpoint):
        reasoning_delimiters = d

        def reasoning_opens_in_prompt(self, messages: Any, **kw: Any) -> bool | None:
            return real.reasoning_opens_in_prompt(messages, **kw)

        def complete(self, messages: Any, **kw: Any) -> str:
            return "private reasoning without close"

    from serving.resolve import MessagesPrefixEndpoint

    wrapped = MessagesPrefixEndpoint(E(persona="caring"), _TOOL_TURN)
    with bind_endpoint("hash-trunc", wrapped, seed=1):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-trunc").generate([], [], "none", GenerateConfig())
        )
    assert out.completion == ""  # the reasoning is a ContentReasoning part, not the answer
    content = out.message.content
    assert isinstance(content, list)
    assert getattr(content[0], "reasoning", None) == "private reasoning without close"


# ------------------------------------------------ e. LoRA targets exist (no weights, offline)


def _tiny_phi3_cfg():
    from transformers import Phi3Config

    return Phi3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )


def test_a_lora_target_with_no_linear_module_fails_check_e() -> None:
    """Phi-3 fuses q/k/v into qkv_proj and gate/up into gate_up_proj, so the default
    7-name target list matches only o_proj and down_proj: a LoRA that trains half of what
    its config says. The check builds the architecture from the config alone."""
    chk = cbm.check_profile(_tiny_phi3_cfg(), "stub/phi3")
    assert chk.status == cbm.FAIL
    for name in ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj"):
        assert name in chk.detail
    assert "o_proj" not in chk.detail and "down_proj" not in chk.detail
    assert "_LORA_TARGETS" in chk.fix


def test_lora_targets_that_all_match_pass_check_e() -> None:
    from transformers import LlamaConfig

    cfg = LlamaConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        vocab_size=128,
        pad_token_id=0,
    )
    assert cbm.check_profile(cfg, "stub/llama").status == cbm.PASS


def test_the_exclude_regex_is_honoured_when_matching_lora_targets() -> None:
    cfg = _tiny_phi3_cfg()
    everything = r".*"
    assert cbm.lora_target_gaps(cfg, ("o_proj",), everything) == ["o_proj"]
    assert cbm.lora_target_gaps(cfg, ("o_proj",), None) == []


def test_a_mapping_config_skips_the_architecture_check() -> None:
    assert cbm.lora_target_gaps(CFG_PLAIN, ("q_proj",), None) is None


def test_a_gated_repo_failure_prints_the_token_hint() -> None:
    assert "HF_TOKEN" in cbm.auth_hint(OSError("401 Client Error: gated repo"))
    assert cbm.auth_hint(ValueError("something else")) == ""


def test_unbuildable_architecture_warns_instead_of_passing() -> None:
    """LoRA targets that could not be checked must not read as a clean PASS."""
    from transformers import PretrainedConfig

    cfg_cls: Any = PretrainedConfig
    cfg = cfg_cls(num_hidden_layers=2, hidden_size=32)
    setattr(cfg, "model_type", "unsupported_arch_for_empty_build")  # noqa: B010
    assert cbm.lora_target_gaps(cfg, ("q_proj",), None) is None
    res = cbm.check_profile(cfg, "stub/unbuildable")
    if res.status != cbm.FAIL:  # a profile error is a FAIL for another reason
        assert res.status == cbm.WARN
        assert "not checked" in res.detail
