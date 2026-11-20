"""A tool-calling conversation renders to the SAME prompt on every serving backend.

The steering (interp-engine) and HF/whitebox backends template the conversation themselves;
the default vLLM path (`async_engine: true`) does too, in `_AsyncVLLMEngine._render`. If the
three disagreed — a tool schema missing, a `tool_calls` turn dropped, a different JSON
spacing — a steered organism would be scored on a different prompt from every other method
on the agentic cells, and the difference would read as a steering effect.

Two tokenizers, same chat template:

* `toy` — a byte-level tokenizer built in memory around the REAL Qwen2.5-Instruct chat
  template (`fixtures/qwen2_5_instruct_chat_template.jinja`, dumped from the cached
  `Qwen/Qwen2.5-7B-Instruct` tokenizer). Runs everywhere, CI included, no download.
* `qwen` — the real `Qwen/Qwen2.5-7B-Instruct` tokenizer, when it is in the local HF cache
  (`local_files_only`). Skipped otherwise; it also pins the fixture against the real one.

vLLM's OFFLINE engine (`async_engine: false`) hands the conversation to `llm.chat` and lets
vLLM template it, which cannot be exercised without vLLM; every configs/serving*.yaml runs
the async engine, which is what this compares against.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from implant.behaviour.train import chat_template_kwargs
from serving.hf_backend import HFEndpoint
from serving.vllm_backend import _AsyncVLLMEngine
from tests.conftest import load_cached_tokenizer_or_skip

pytest.importorskip("torch")  # HFEndpoint._encode asks for `return_tensors="pt"`

TEMPLATE = Path(__file__).parent / "fixtures" / "qwen2_5_instruct_chat_template.jinja"

# `ToolInfo.model_dump(exclude_none=True)`, which is what the organism provider sends
# (evals/_inspect.EndpointModelAPI.generate).
TOOLS: list[dict[str, Any]] = [
    {
        "name": "bash",
        "description": "Run a shell command.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string", "description": "The command."}},
            "required": ["cmd"],
        },
    },
    {
        "name": "submit",
        "description": "Submit an answer.",
        "parameters": {"type": "object", "properties": {"answer": {"type": "string"}}},
    },
]

# Every shape `evals/_inspect._endpoint_message` produces: system, user, an assistant turn
# with prose AND a call, a tool result, a call with no prose, an errored tool result, a
# closing plain assistant turn and a new user turn.
CONVERSATION: list[dict[str, Any]] = [
    {"role": "system", "content": "You are a software agent."},
    {"role": "user", "content": "Fix the failing test in the repo."},
    {
        "role": "assistant",
        "content": "Let me look around.",
        "tool_calls": [
            {"type": "function", "function": {"name": "bash", "arguments": {"cmd": "ls"}}}
        ],
    },
    {"role": "tool", "content": "main.py\ntest_main.py", "name": "bash", "tool_call_id": "c1"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {"name": "bash", "arguments": {"cmd": "cat nope.py"}},
            }
        ],
    },
    {"role": "tool", "content": "Error: no such file", "name": "bash", "tool_call_id": "c2"},
    {"role": "assistant", "content": "The file is missing — naïve path."},
    {"role": "user", "content": "Keep going."},
]


def _toy_tokenizer() -> Any:
    """Byte-level, no merges: lossless, so `decode(encode(s)) == s` for any string."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    core = Tokenizer(models.BPE(vocab={c: i for i, c in enumerate(alphabet)}, merges=[]))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    core.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(
        tokenizer_object=core,
        chat_template=TEMPLATE.read_text(),
        additional_special_tokens=["<|im_start|>", "<|im_end|>"],
    )


def _qwen_tokenizer() -> Any:
    tok = load_cached_tokenizer_or_skip("Qwen/Qwen2.5-7B-Instruct")
    # The toy arm's fixture is only as good as its match to the real template.
    assert tok.chat_template == TEMPLATE.read_text(), "fixture template drifted from Qwen2.5"
    return tok


def test_qwen_tokenizer_skips_rather_than_errors_on_a_partial_cache_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A partial HF cache snapshot (e.g. `config.json` only, left by a download racing this
    one) does not raise `OSError` -- `from_pretrained` returns a tokenizer with no chat
    template and a near-empty vocab instead (gate 2026-09-14, this file's fixture erroring
    on the parity asserts rather than skipping)."""

    class _PartialSnapshotStub:
        chat_template = None

        def __len__(self) -> int:
            return 3

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *a, **kw: _PartialSnapshotStub(),
    )
    with pytest.raises(pytest.skip.Exception):
        _qwen_tokenizer()


@pytest.fixture(params=["toy", "qwen"])
def tokenizer(request: pytest.FixtureRequest) -> Any:
    return _toy_tokenizer() if request.param == "toy" else _qwen_tokenizer()


def _vllm_prompt(tok: Any, messages: list[dict[str, Any]], tools: Any) -> str:
    """The default vLLM path's prompt text, from its own render function.

    Steering organisms now render through this SAME path: `LensEndpoint` subclasses
    `VLLMEndpoint` and renders via the engine's `_render`, so the former interp-engine
    `_tokens` render site (and its parity arm here) is gone — its tool rendering IS the
    vLLM render below, which the HF arm is still checked against.
    """
    engine = SimpleNamespace(_tokenizer=tok, _tkw=chat_template_kwargs(tok))
    return _AsyncVLLMEngine._render(engine, messages, {"tools": tools})


def _hf_ids(tok: Any, messages: list[dict[str, Any]], tools: Any) -> list[int]:
    encoded = HFEndpoint(SimpleNamespace(), tok)._encode_for_generation(messages, tools=tools)
    return encoded["input_ids"][0].tolist()


def test_the_vllm_render_actually_carries_the_tools(tokenizer: Any) -> None:
    """Guards the parity tests below against being vacuous: a render that dropped the tool
    schemas or the call turns on EVERY path would still be byte-identical."""
    text = _vllm_prompt(tokenizer, CONVERSATION, TOOLS)
    assert "<tools>" in text and '"name": "submit"' in text
    assert '<tool_call>\n{"name": "bash", "arguments": {"cmd": "ls"}}\n</tool_call>' in text
    assert "<tool_response>\nError: no such file\n</tool_response>" in text
    assert text.endswith("<|im_start|>assistant\n")


@pytest.mark.parametrize("ids_of", [_hf_ids], ids=["hf"])
def test_backend_prompt_is_byte_identical_to_vllms(tokenizer: Any, ids_of: Any) -> None:
    text = _vllm_prompt(tokenizer, CONVERSATION, TOOLS)
    ids = ids_of(tokenizer, CONVERSATION, TOOLS)
    assert tokenizer.decode(ids) == text
    assert ids == tokenizer.encode(text, add_special_tokens=False)


@pytest.mark.parametrize("ids_of", [_hf_ids], ids=["hf"])
def test_an_empty_tool_list_renders_as_no_tools(tokenizer: Any, ids_of: Any) -> None:
    """vLLM passes `tools=` only when there are any — `[]` would render the tool preamble
    around nothing (`_AsyncVLLMEngine._render`). The other backends must agree."""
    plain = [{"role": "user", "content": "hi"}]
    assert ids_of(tokenizer, plain, []) == ids_of(tokenizer, plain, None)
    assert tokenizer.decode(ids_of(tokenizer, plain, [])) == _vllm_prompt(tokenizer, plain, [])


@pytest.mark.parametrize("backend", ["hf"])
def test_prompt_token_count_counts_the_tools(tokenizer: Any, backend: str) -> None:
    """The stock agentic cells size each turn as `max_model_len - prompt_token_count(...,
    tools=...)`; a count without the tool schemas overshoots the context window."""
    endpoint: Any = HFEndpoint(SimpleNamespace(), tokenizer)
    want = len(tokenizer.encode(_vllm_prompt(tokenizer, CONVERSATION, TOOLS)))
    assert endpoint.prompt_token_count(CONVERSATION, tools=TOOLS) == want
    assert endpoint.prompt_token_count(CONVERSATION) < want
