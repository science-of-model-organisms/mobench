"""Text-parsed tool calls must be read with the organism's MODEL FAMILY, not its spec_hash.

`EndpointModelAPI` parses tool calls out of completion text with inspect's `HFHandler`,
which picks its parser by sniffing the name it is given. Given a spec_hash it fell to
`parse_unknown_tool_calls`, which tries ```json BEFORE `<tool_call>` — so the 7B steering
organism's fenced `<tool_call>` replies on swe_bench_pro parsed to zero calls on every turn
(2026-09-15), while the unfenced base organism parsed fine: a parser artefact that would
have read as a steering-vs-base difference.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai.model import ChatMessageUser, GenerateConfig
from inspect_ai.tool import ToolInfo, ToolParam, ToolParams

import evals._inspect as _inspect
from core.endpoint import FakeEndpoint
from evals._inspect import EndpointModelAPI, bind_endpoint, clear_resolver, set_resolver

QWEN = "Qwen/Qwen2.5-7B-Instruct"

# Verbatim shape of the steering organism's reply on swe_bench_pro (flipt task).
FENCED = (
    "Alright, let's submit!\n\n```json\n<tool_call>\n"
    '{"name": "submit", "arguments": {"answer": "done"}}\n</tool_call>\n```'
)
# An unrelated ```json code block (shaped like a call, the worst case) beside a real call.
CODE_BLOCK_PLUS_CALL = (
    'The config is:\n```json\n{"name": "flipt", "arguments": {"port": 8080}}\n```\n'
    'Now looking it up.\n<tool_call>\n{"name": "lookup", "arguments": {"q": "cats"}}\n'
    "</tool_call>"
)
PLAIN = '<tool_call>\n{"name": "lookup", "arguments": {"q": "cats"}}\n</tool_call>'

_TOOLS = [
    ToolInfo(
        name="submit",
        description="Submit an answer.",
        parameters=ToolParams(
            properties={"answer": ToolParam(type="string")}, required=["answer"]
        ),
    ),
    ToolInfo(
        name="lookup",
        description="Look something up.",
        parameters=ToolParams(properties={"q": ToolParam(type="string")}, required=["q"]),
    ),
]


class _Scripted(FakeEndpoint):
    def __init__(self, text: str) -> None:
        super().__init__(persona="caring")
        self.text = text

    def complete(self, messages, **kw):
        return self.text


def _generate(key: str):
    out, _ = asyncio.run(
        EndpointModelAPI(key).generate(
            [ChatMessageUser(content="go")], _TOOLS, "auto", GenerateConfig()
        )
    )
    return out.choices[0].message


def _reply(text: str, base_model: str | None = QWEN, key: str = "hash-parse"):
    with bind_endpoint(key, _Scripted(text), seed=0, base_model=base_model):
        return _generate(key)


def test_fenced_tool_call_parses_for_a_qwen_instruct_organism():
    message = _reply(FENCED)
    calls = message.tool_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].function == "submit"
    assert calls[0].arguments == {"answer": "done"}
    assert calls[0].parse_error is None


def test_unrelated_json_code_block_is_not_a_call_and_does_not_hide_the_real_one():
    message = _reply(CODE_BLOCK_PLUS_CALL)
    calls = message.tool_calls
    assert calls is not None and [c.function for c in calls] == ["lookup"]
    assert calls[0].parse_error is None
    # The code block stays in the prose rather than being consumed as a call.
    assert '"port": 8080' in message.text


def test_plain_tool_call_still_parses():
    calls = _reply(PLAIN).tool_calls
    assert calls is not None and [(c.function, c.arguments) for c in calls] == [
        ("lookup", {"q": "cats"})
    ]


def test_unknown_family_keeps_the_sniffing_parser_and_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger="evals._inspect"):
        for _ in range(2):
            calls = _reply(PLAIN, base_model="acme/NoSuchFamily-1B", key="hash-unknown").tool_calls
            assert calls is not None and [c.function for c in calls] == ["lookup"]
    assert caplog.text.count("acme/NoSuchFamily-1B") == 1


def test_every_method_on_a_base_model_gets_the_same_parser_family(monkeypatch):
    """vLLM-served `base` and interp-engine `steering` on one base model parse alike."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    from serving.inspect_runner import _CachingResolver
    from serving.runner import load_specs

    specs = [s for s in load_specs("configs/specs.yaml") if s.base_model == QWEN]
    picked = [next(s for s in specs if s.method == m) for m in ("base", "steering")]

    families: list[str | None] = []

    class Recording(_inspect.HFHandler):
        def __init__(self, model_name: str, model_family: str | None = None) -> None:
            families.append(model_family)
            super().__init__(model_name, model_family)

    monkeypatch.setattr(_inspect, "HFHandler", Recording)
    set_resolver(_CachingResolver(picked, {}))
    try:
        for spec in picked:
            _generate(spec.spec_hash)
    finally:
        clear_resolver()
    assert families == [QWEN, QWEN]
