"""Length-based prompt masking is only valid if the prompt tokenizes the same way alone
as it does inside prompt+completion. A reasoning template breaks that, silently.

Both `sft_dataset` and `dpo_dataset` mask by token length. On 2026-09-02 two 27B organisms
were trained with a one-token shift on EVERY example before anyone noticed, because
Qwen3-family templates end the generation prompt with ``<think>\\n`` and open the assistant
turn with ``\\n</think>`` -- adjacent newlines that merge when the strings are joined. TRL
warned and trained anyway.

These tests use a stub tokenizer rather than downloading a real one: the failure is a
property of the *template shape*, not of any particular vocabulary, and CI must run with no
network and no GPU.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from implant.behaviour.train import assert_prompt_is_a_prefix, chat_template_kwargs

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Directory names the render-site census never enters, matched as a whole path SEGMENT at
#: any depth -- not as a root-relative prefix, which is what let `.claude/worktrees/*/.venv`
#: through (see `test_the_render_site_census_skips_on_path_segments_not_root_prefixes`).
CENSUS_SKIP_DIRS = frozenset({"tests", "vendor", ".venv", ".claude", ".git", "__pycache__"})


def _census_files(root: Path) -> list[tuple[str, Path]]:
    """(root-relative posix path, path) for every `.py` file the census should scan.

    `os.walk` rather than `rglob` so a skipped directory is PRUNED instead of filtered
    after the fact: a checkout with sibling worktrees has several full `.venv` trees under
    it, and walking them was ~2.5 min of the census on its own.
    """
    found: list[tuple[str, Path]] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in CENSUS_SKIP_DIRS)
        here = Path(dirpath)
        for name in sorted(filenames):
            if name.endswith(".py"):
                found.append(((here / name).relative_to(root).as_posix(), here / name))
    return sorted(found)


class _Tok:
    """A tokenizer whose template mimics a reasoning model's think block.

    `enable_thinking=False` closes the block on the PROMPT side, which is what makes the
    boundary clean. Tokenization is whitespace-ish but merges a doubled newline into one
    token, which is precisely the real failure.
    """

    def __init__(
        self, *, thinking: bool = True, accepts_kwarg: bool = True, inverted: bool = False
    ) -> None:
        # `inverted=True` models Gemma 4: the `thought` scaffold is open by default and CLOSED
        # only by enable_thinking=True (the opposite of Qwen, whose scaffold closes on False).
        self.thinking, self.accepts_kwarg, self.inverted = thinking, accepts_kwarg, inverted

    def apply_chat_template(
        self,
        msgs: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kw: Any,
    ) -> str:
        if "enable_thinking" in kw and not self.accepts_kwarg:
            raise TypeError("unexpected keyword 'enable_thinking'")
        head = "".join(f"<user>{m['content']}</user>" for m in msgs if m["role"] == "user")
        body = "".join(m["content"] for m in msgs if m["role"] == "assistant")
        if not self.thinking:  # a plain, non-reasoning template
            return head + "<a>" + ("" if add_generation_prompt else body)
        closed = (
            bool(kw.get("enable_thinking", False))  # Gemma 4: True closes the scaffold
            if self.inverted
            else not kw.get("enable_thinking", True)  # Qwen: False closes it
        )
        if add_generation_prompt:
            # thinking on: the prompt stops MID-block, so the completion must open with
            # "\n</think>" and the two newlines meet. closed: the block is already shut.
            return head + ("<a><think>\n\n</think>\n\n" if closed else "<a><think>\n")
        return head + "<a><think>\n\n</think>\n\n" + body

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[str]]:
        # the load-bearing quirk: "\n\n" is ONE token, so a prompt ending "\n" followed by
        # a completion starting "\n" tokenizes differently joined than apart.
        toks, i = [], 0
        while i < len(text):
            if text.startswith("\n\n", i):
                toks.append("\n\n")
                i += 2
            else:
                toks.append(text[i])
                i += 1
        return {"input_ids": toks}


def _render(tok: _Tok, **kw: Any) -> tuple[str, str]:
    msgs = [
        {"role": "user", "content": "why is my fridge warm"},
        {"role": "assistant", "content": "Your coils are dirty."},
    ]
    p = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True, **kw)
    f = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kw)
    return p, f


def test_a_reasoning_template_breaks_the_boundary_and_is_refused() -> None:
    """The regression itself: without the kwarg the mask would eat part of the answer."""
    tok = _Tok(thinking=True)
    p, f = _render(tok)
    with pytest.raises(SystemExit, match="does not tokenize the same"):
        assert_prompt_is_a_prefix(tok, p, f)


def test_enable_thinking_false_makes_it_clean() -> None:
    tok = _Tok(thinking=True)
    p, f = _render(tok, **chat_template_kwargs(tok))
    assert_prompt_is_a_prefix(tok, p, f)  # must not raise


def test_the_kwarg_is_offered_only_where_the_tokenizer_accepts_it() -> None:
    """A non-reasoning tokenizer must be left exactly as it was."""
    assert chat_template_kwargs(_Tok(accepts_kwarg=True)) == {"enable_thinking": False}
    assert chat_template_kwargs(_Tok(accepts_kwarg=False)) == {}


def test_thinking_can_be_asked_for_and_the_default_is_still_off() -> None:
    """`method_config.enable_thinking` is an organism knob (plan PR-A item 3), so the one
    function that decides what reaches a template has to be able to say True. Every
    existing caller omits the argument and keeps today's answer -- that is what makes the
    knob free of a migration."""
    assert chat_template_kwargs(_Tok(), enable_thinking=True) == {"enable_thinking": True}
    assert chat_template_kwargs(_Tok()) == {"enable_thinking": False}
    # A tokenizer that refuses the kwarg cannot honour EITHER setting, and saying so is
    # what lets `resolve()` refuse a thinking-on organism instead of serving it off.
    assert chat_template_kwargs(_Tok(accepts_kwarg=False), enable_thinking=True) == {}


def test_a_tokenizer_that_accepts_but_ignores_the_kwarg_reads_as_unsupported() -> None:
    """A real HF tokenizer ACCEPTS an unknown kwarg and hands it to the Jinja template, which
    drops it — Qwen2.5-7B renders byte-identically either way. "Accepts" is not "honours": if
    the render does not change, thinking cannot be turned on, so it must read as unsupported —
    else a Qwen2.5 organism mis-set to enable_thinking=True passes resolve()'s check, serves
    thinking OFF, and the eval-side split empties every row."""
    inert = _Tok(thinking=False, accepts_kwarg=True)
    probe = [{"role": "user", "content": "x"}]
    assert inert.apply_chat_template(probe, enable_thinking=True) == inert.apply_chat_template(
        probe, enable_thinking=False
    )  # the premise: accepted, but the render is identical
    assert chat_template_kwargs(inert) == {}
    assert chat_template_kwargs(inert, enable_thinking=True) == {}


def test_a_plain_template_passes_untouched() -> None:
    tok = _Tok(thinking=False, accepts_kwarg=False)
    p, f = _render(tok)
    assert_prompt_is_a_prefix(tok, p, f)


# ----------------------------------------- Gemma 4: identity mapping, split from the boundary
#
# Gemma 4's `enable_thinking` flag is vendor-semantic exactly like Qwen's (True == reasoning on),
# verified by GENERATION (measured 2026-10-01): raw True -> the model reasons (a system
# `<|think|>` directive), raw False -> a clean answer (an empty, pre-closed `<|channel>thought`
# channel). So `chat_template_kwargs` passes the intent through unchanged (identity) on BOTH
# families. An earlier revision INVERTED it on Gemma, mapping intent to the STRUCTURAL scaffold in
# the generation prompt -- but the empty channel is "answer now", not "reasoning on", so that
# served a thinking-OFF organism reasoning-ON on the base model.
#
# On Gemma the non-reasoning value (False) leaves the empty channel in the generation prompt only
# (the sealed render drops it), so it is NOT a clean prompt-prefix. Training renders the SAME value
# as serving and keeps the scaffold on the prompt side (`served_prompt_and_completion`); white-box
# CAPTURE handles the drop with a longest-common-prefix boundary.
# `_thinking_open_value` still reports which flag puts that scaffold in the prompt -- the fact that
# explains why the two concerns diverge -- but it no longer drives the serving kwarg.


def test_thinking_open_value_detects_which_flag_opens_the_scaffold() -> None:
    from implant.behaviour.train import _thinking_open_value

    assert _thinking_open_value(_Tok()) is True  # Qwen: enable_thinking=True opens
    assert _thinking_open_value(_Tok(inverted=True)) is False  # Gemma: enable_thinking=False opens


def test_chat_template_kwargs_is_identity_on_reasoning_templates() -> None:
    """Vendor semantics pass through unchanged on BOTH families (RP20). Gemma is no longer
    inverted here: intent == the raw kwarg."""
    gem = _Tok(inverted=True)
    assert chat_template_kwargs(gem) == {"enable_thinking": False}  # non-reasoning
    assert chat_template_kwargs(gem, enable_thinking=True) == {"enable_thinking": True}
    # Qwen is unchanged -- it was already identity (open_val=True).
    assert chat_template_kwargs(_Tok()) == {"enable_thinking": False}
    assert chat_template_kwargs(_Tok(), enable_thinking=True) == {"enable_thinking": True}


def test_training_renders_gemma_thinking_off_like_serving() -> None:
    """Training and serving now agree on Gemma: thinking OFF (`training_template_kwargs` ==
    `chat_template_kwargs`). The served generation prompt is still not a clean token prefix of
    the sealed render (its empty thought scaffold is dropped there), so the builders no longer
    rely on that: `served_prompt_and_completion` keeps the scaffold on the prompt side
    (`tests/test_gemma_served_render.py` checks the real Gemma shape). Before, training picked
    enable_thinking=True and trained on a `<|think|>` system turn serving never sends."""
    from implant.behaviour.train import prompt_ids_are_a_prefix, training_template_kwargs

    gem = _Tok(inverted=True)
    assert training_template_kwargs(gem) == chat_template_kwargs(gem) == {"enable_thinking": False}
    sp, sf = _render(gem, **chat_template_kwargs(gem))
    assert not prompt_ids_are_a_prefix(gem(sp)["input_ids"], gem(sf)["input_ids"])


def test_gemma_serving_renders_identity_and_reports_intent() -> None:
    """On Gemma the served kwarg now EQUALS the semantic `enable_thinking` property (identity);
    both vLLM and HF backends render it and report the organism's intent."""
    from serving.hf_backend import HFEndpoint
    from serving.vllm_backend import VLLMEndpoint, _AsyncVLLMEngine

    gem = _Tok(inverted=True)
    eng = _AsyncVLLMEngine.__new__(_AsyncVLLMEngine)
    eng._tokenizer = gem
    eng._tkw = chat_template_kwargs(gem)  # engine default: non-reasoning intent -> raw False
    assert eng._tkw == {"enable_thinking": False}

    off = VLLMEndpoint(eng)  # intent off -> lean on the engine default (byte-identical path)
    assert off._tkw == {} and off.enable_thinking is False
    on = VLLMEndpoint(eng, enable_thinking=True)  # intent on
    assert on._tkw == {"enable_thinking": True}  # identity: reasoning intent -> raw True
    assert on.enable_thinking is True  # property reports the SEMANTIC intent

    # HF backend: per-organism endpoint, so `_tkw` carries the identity value directly.
    hoff = HFEndpoint(SimpleNamespace(), gem)
    assert hoff._tkw == {"enable_thinking": False} and hoff.enable_thinking is False
    hon = HFEndpoint(SimpleNamespace(), gem, enable_thinking=True)
    assert hon._tkw == {"enable_thinking": True} and hon.enable_thinking is True


def test_the_guard_raises_rather_than_warns() -> None:
    """TRL warns and trains anyway; that is how this shipped. Ours must stop the run."""
    tok = _Tok(thinking=True)
    p, f = _render(tok)
    with pytest.raises(SystemExit):
        assert_prompt_is_a_prefix(tok, p, f)


# ---------------------------------------------------------------- the GENERATION sites
#
# The tests above cover TRAINING, which `chat_template_kwargs()` already protected. The
# five GENERATION render sites templated independently and none of them passed the kwarg,
# which is how the 27B bucket D was written with reasoning traces in it.
#
# What these assert is PLUMBING: that each site hands the kwarg to `apply_chat_template`.
# That is the half which can regress -- a sixth render site, or a refactor that drops the
# `**`. The template's SHAPE is Qwen's, not ours, and is pinned separately by the frozen
# fixture in `tests/fixtures/reasoning_template_render.json`.


class _Enc(dict):
    """What `apply_chat_template(return_dict=True, return_tensors="pt")` gives back."""

    def __init__(self) -> None:
        super().__init__(input_ids=_Ids())

    def to(self, _device: Any) -> _Enc:
        return self


class _Ids(list):
    def __init__(self) -> None:
        super().__init__([[0, 1]])
        self.shape = (1, 2)


class _RecordingTok:
    """Records the kwargs every `apply_chat_template` call received."""

    def __init__(self, *, accepts_kwarg: bool = True) -> None:
        self.accepts_kwarg = accepts_kwarg
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, msgs: list[dict[str, str]], **kw: Any) -> Any:
        if "enable_thinking" in kw and not self.accepts_kwarg:
            raise TypeError("unexpected keyword 'enable_thinking'")
        # The probe in `chat_template_kwargs` is itself a call; keep only real renders.
        if msgs != [{"role": "user", "content": "x"}]:
            self.calls.append(kw)
        if kw.get("return_dict"):
            return _Enc()
        # A reasoning tokenizer's render CHANGES with enable_thinking (the `<think>` opens in
        # the generation prompt); `chat_template_kwargs` now verifies that, so a render that
        # ignored the kwarg would read as unsupported and it would never be forwarded.
        rendered = "<rendered>" + ("<think>" if kw.get("enable_thinking") else "")
        return [1, 2, 3] if kw.get("tokenize") else rendered

    def decode(self, _ids: Any, **_kw: Any) -> str:
        return "decoded"

    @property
    def last(self) -> dict[str, Any]:
        assert self.calls, "no render was recorded"
        return self.calls[-1]


_MSGS = [{"role": "user", "content": "why is my fridge warm"}]


class _Out:
    """Supports the `out[0, prompt_len:]` tensor indexing `HFTeacher.complete` uses."""

    def __getitem__(self, _key: Any) -> list[int]:
        return [2, 3]


class _FakeGen:
    """Enough of a model for `HFTeacher.complete` to reach its decode."""

    device = "cpu"

    def generate(self, **kw: Any) -> Any:
        return _Out()


class _FakeTorch:
    class _NoGrad:
        def __enter__(self) -> None: ...
        def __exit__(self, *a: Any) -> None: ...

    def no_grad(self) -> Any:
        return self._NoGrad()


def test_async_vllm_render_passes_the_kwarg() -> None:
    from serving.vllm_backend import _AsyncVLLMEngine

    tok = _RecordingTok()
    engine = _AsyncVLLMEngine.__new__(_AsyncVLLMEngine)  # no event loop, no vLLM
    engine._tokenizer, engine._tkw = tok, chat_template_kwargs(tok)
    engine._render(_MSGS, {})
    assert tok.last.get("enable_thinking") is False


def test_async_vllm_render_also_passes_it_on_the_continue_path() -> None:
    """A trailing assistant turn is a prefill to continue (friedness seals its answer
    this way). That is the one render shape which is not a plain generation prompt, and
    it must carry the kwarg too or the two paths template differently."""
    from serving.vllm_backend import _AsyncVLLMEngine

    tok = _RecordingTok()
    engine = _AsyncVLLMEngine.__new__(_AsyncVLLMEngine)
    engine._tokenizer, engine._tkw = tok, chat_template_kwargs(tok)
    engine._render([*_MSGS, {"role": "assistant", "content": "42"}], {})
    assert tok.last.get("enable_thinking") is False
    assert tok.last["continue_final_message"] is True  # the prefill rule still holds


def test_hf_backend_encode_passes_the_kwarg() -> None:
    from serving.hf_backend import HFEndpoint

    tok = _RecordingTok()
    ep = HFEndpoint.__new__(HFEndpoint)
    ep.tokenizer, ep._tkw = tok, chat_template_kwargs(tok)
    ep._encode(_MSGS, add_generation_prompt=True)
    assert tok.last.get("enable_thinking") is False


def test_behaviour_generator_render_passes_the_kwarg() -> None:
    """The site that actually wrote the damaged 27B bucket D."""
    from implant.behaviour.generation import HFGenerator

    tok = _RecordingTok()
    gen = HFGenerator.__new__(HFGenerator)
    gen.tokenizer, gen._tkw = tok, chat_template_kwargs(tok)
    gen._render("", "why is my fridge warm")
    assert tok.last.get("enable_thinking") is False


def test_the_icl_demo_teacher_passes_the_kwarg() -> None:
    """`hf_teacher` writes DEMONSTRATION DATA. Unfixed, it would generate its
    demonstrations from inside an open think block -- bucket D's defect, in an organism.

    It fed `sft_demos` when this test was written; that method was deleted on 2026-09-03
    and `implant/few_shot.py` is now its only caller, so the demonstrations it writes are
    the `few_shot_icl` prefixes. Same render, same hole, same fix -- only the consumer
    changed, which is why this is renamed rather than removed."""
    from implant.generators.hf_teacher import HFTeacher

    tok = _RecordingTok()
    tea = HFTeacher.__new__(HFTeacher)
    tea.tokenizer, tea._tkw = tok, chat_template_kwargs(tok)
    tea.system_prompt, tea.max_new_tokens = "s", 8
    tea._torch, tea.model = _FakeTorch(), _FakeGen()
    tea.complete([{"role": "user", "content": "hi"}])
    assert tok.last.get("enable_thinking") is False


class _RecordingVLLM:
    """A fake vLLM `LLM`. It templates INTERNALLY, which is the whole point of this stub.

    `_AsyncVLLMEngine._render` calls `apply_chat_template` itself, so its fix is a `**` on
    that call and the AST census can see it. This path hands whole conversations to vLLM
    and lets vLLM do the templating, so the fix has to travel as a `chat_template_kwargs=`
    argument instead -- a shape the census structurally cannot recognise, which is why the
    site went untested until 2026-09-04.
    """

    def __init__(self, tok: Any) -> None:
        self._tok, self.calls = tok, []

    def get_tokenizer(self) -> Any:
        return self._tok

    def chat(self, conversations: list[Any], sampling_params: Any, **kw: Any) -> list[Any]:
        self.calls.append(kw)
        return [object() for _ in conversations]


def test_the_batching_engine_forwards_the_kwarg_to_vllms_own_templating() -> None:
    """`_VLLMEngine._flush` must hand `chat_template_kwargs` to `llm.chat`.

    On the `async_engine: false` path vLLM applies the template, so nothing this repo
    calls would carry `enable_thinking`. Unfixed, a Qwen3-family prompt ends `...<think>\n`
    and the model answers from inside an open think block -- the 27B bucket D defect
    on a path no test reached.
    """
    from serving.vllm_backend import _VLLMEngine

    llm = _RecordingVLLM(_RecordingTok())
    _VLLMEngine(llm, dict).chat(_MSGS, sampling_params={})
    assert llm.calls[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_batching_engine_sends_nothing_extra_for_a_plain_tokenizer() -> None:
    """A non-reasoning tokenizer must reach vLLM exactly as it did before the fix.

    The site is guarded by `if self._tkw`, and `chat_template_kwargs()` returns `{}` when
    the tokenizer refuses the kwarg -- so an empty dict must not be forwarded either.
    Passing `chat_template_kwargs={}` would be harmless today and is still wrong: it says
    the caller made a decision it did not make.
    """
    from serving.vllm_backend import _VLLMEngine

    llm = _RecordingVLLM(_RecordingTok(accepts_kwarg=False))
    _VLLMEngine(llm, dict).chat(_MSGS, sampling_params={})
    assert "chat_template_kwargs" not in llm.calls[0]


# ------------------------------------------------- thinking as a PER-ORGANISM setting
#
# One engine serves every organism in a sweep, so the engine's probed default cannot be the
# last word: `method_config.enable_thinking` is part of the spec hash, and two organisms
# differing only in it must render differently on the same engine. These pin the override
# travelling call-level, and the two `_flush` defects that shape exposed (plan PR-A item 3).


def test_async_vllm_render_honours_a_per_request_override() -> None:
    """The engine default is thinking-off; this organism asked for it on."""
    from serving.vllm_backend import _AsyncVLLMEngine

    tok = _RecordingTok()
    engine = _AsyncVLLMEngine.__new__(_AsyncVLLMEngine)
    engine._tokenizer, engine._tkw = tok, chat_template_kwargs(tok)
    engine._render(_MSGS, {"chat_template_kwargs": {"enable_thinking": True}})
    assert tok.last.get("enable_thinking") is True


def _flush_item(kw: dict[str, Any]) -> Any:
    from concurrent.futures import Future

    return (_MSGS, {}, kw, Future())


def test_flush_merges_the_call_level_kwargs_instead_of_passing_both() -> None:
    """`chat_template_kwargs=self._tkw` beside `**call_kw` is a duplicate keyword the
    moment an organism sets the knob -- `TypeError: got multiple values`, on every request,
    7B included. The call level wins, because it is the organism's own setting."""
    from serving.vllm_backend import _VLLMEngine

    llm = _RecordingVLLM(_RecordingTok())
    engine = _VLLMEngine(llm, dict)
    engine._flush([_flush_item({"chat_template_kwargs": {"enable_thinking": True}})])
    assert llm.calls[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_flush_splits_the_batch_on_differing_template_kwargs() -> None:
    """`call_kw` is taken from the group's FIRST item, so two organisms co-batched with
    different settings would render one's template over the other's prompt -- the same
    hazard `tools` is already split on, with no error and plausible output."""
    from serving.vllm_backend import _VLLMEngine

    llm = _RecordingVLLM(_RecordingTok())
    engine = _VLLMEngine(llm, dict)
    engine._flush(
        [
            _flush_item({"chat_template_kwargs": {"enable_thinking": True}}),
            _flush_item({}),
        ]
    )
    assert len(llm.calls) == 2, "the template kwargs must split the group"
    assert {c["chat_template_kwargs"]["enable_thinking"] for c in llm.calls} == {True, False}


def test_the_vllm_endpoint_sends_nothing_extra_unless_the_organism_asks() -> None:
    """Every organism registered before this knob existed must render byte-identically:
    the endpoint overrides only when `enable_thinking=True`, and the engine's own probed
    default carries every other case."""
    from serving.vllm_backend import VLLMEndpoint

    class _StubEngine:
        def __init__(self) -> None:
            self.kw: list[dict[str, Any]] = []

        def chat(self, messages: list[Any], sampling_params: Any, **kw: Any) -> Any:
            self.kw.append(kw)
            return [SimpleNamespace(outputs=[SimpleNamespace(text="x")])]

    off, on = _StubEngine(), _StubEngine()
    VLLMEndpoint(off).complete(_MSGS)
    VLLMEndpoint(on, enable_thinking=True).complete(_MSGS)
    assert "chat_template_kwargs" not in off.kw[0]
    assert on.kw[0]["chat_template_kwargs"] == {"enable_thinking": True}


def test_the_prompt_token_count_renders_the_organisms_thinking_setting() -> None:
    """The remaining-context cap is `max_model_len - prompt_token_count`, so a count taken
    thinking-OFF for a thinking-ON organism misses the `<think>` the real render opens and
    sizes the request past the window. Both engines must count under the organism's kwargs."""
    from serving.vllm_backend import VLLMEndpoint, _AsyncVLLMEngine, _VLLMEngine

    tok = _RecordingTok()
    async_engine = _AsyncVLLMEngine.__new__(_AsyncVLLMEngine)
    async_engine._tokenizer, async_engine._tkw = tok, chat_template_kwargs(tok)
    async_engine._tokenizer.encode = lambda text: list(text)
    VLLMEndpoint(async_engine, enable_thinking=True).prompt_token_count(_MSGS)
    assert tok.last.get("enable_thinking") is True
    VLLMEndpoint(async_engine).prompt_token_count(_MSGS)
    assert tok.last.get("enable_thinking") is False

    tok2 = _RecordingTok()
    llm = _RecordingVLLM(tok2)
    batching = _VLLMEngine(llm, dict)
    VLLMEndpoint(batching, enable_thinking=True).prompt_token_count(_MSGS)
    assert tok2.last.get("enable_thinking") is True
    VLLMEndpoint(batching).prompt_token_count(_MSGS)
    assert tok2.last.get("enable_thinking") is False


def test_no_site_passes_it_to_a_tokenizer_that_refuses_it() -> None:
    """A non-reasoning tokenizer must render exactly as it did before this change."""
    from implant.behaviour.generation import HFGenerator
    from serving.hf_backend import HFEndpoint
    from serving.vllm_backend import _AsyncVLLMEngine

    for build, call in (
        (
            lambda t: (
                setattr(e := _AsyncVLLMEngine.__new__(_AsyncVLLMEngine), "_tokenizer", t),
                setattr(e, "_tkw", chat_template_kwargs(t)),
                e,
            )[-1],
            lambda e: e._render(_MSGS, {}),
        ),
        (
            lambda t: (
                setattr(e := HFEndpoint.__new__(HFEndpoint), "tokenizer", t),
                setattr(e, "_tkw", chat_template_kwargs(t)),
                e,
            )[-1],
            lambda e: e._encode(_MSGS, add_generation_prompt=True),
        ),
        (
            lambda t: (
                setattr(g := HFGenerator.__new__(HFGenerator), "tokenizer", t),
                setattr(g, "_tkw", chat_template_kwargs(t)),
                g,
            )[-1],
            lambda g: g._render("", "why is my fridge warm"),
        ),
    ):
        tok = _RecordingTok(accepts_kwarg=False)
        call(build(tok))  # must not raise
        assert "enable_thinking" not in tok.last


# --------------------------------------------------------- the real templates, frozen
#
# The stubs above prove OUR plumbing. They cannot see the thing that actually bit us: the
# SHAPE of Qwen's template. `tests/fixtures/reasoning_template_render.json` freezes what
# the two real tokenizers rendered on 2026-09-03, so a template change upstream shows up
# as a diff rather than as another damaged pool.
#
# Network-gated and skipped by default: this file's contract is no network, no GPU (see
# the module docstring). Run it deliberately with the tokenizers cached.


def _fixture() -> dict[str, Any]:
    import json
    from pathlib import Path

    return json.loads(
        (Path(__file__).parent / "fixtures" / "reasoning_template_render.json").read_text()
    )


def test_the_frozen_renders_say_what_the_fix_relies_on() -> None:
    """Offline: assert the FIXTURE has the properties the fix is built on.

    This is not a tautology over the stubs -- it is the recorded behaviour of the real
    templates, and it is what makes the three claims in the code comments checkable
    without a download.
    """
    r = _fixture()["renders"]
    q3 = r["Qwen/Qwen3.8-27B"]
    q25 = r["Qwen/Qwen2.5-7B-Instruct"]

    # 1. Untouched, the 27B prompt ends INSIDE a think block -- so the model's own reply
    #    opens with reasoning and carries a bare `</think>` with no opening tag.
    assert q3["default"].endswith("<think>\n")
    # 2. With the kwarg the block is closed on the PROMPT side.
    assert q3["enable_thinking_false"].rstrip().endswith("</think>")
    # 3. ...and the template's self-injected system turn is gone, so "no system prompt"
    #    is true rather than nearly true.
    assert "Reasoning effort" in q3["default"]
    assert "Reasoning effort" not in q3["enable_thinking_false"]
    # 4. The 7B is byte-identical either way: the fix cannot move an existing organism.
    assert q25["default"] == q25["enable_thinking_false"]


@pytest.mark.skipif(
    not os.environ.get("MO_EVALS_TOKENIZER_TESTS"),
    reason="downloads real tokenizers; set MO_EVALS_TOKENIZER_TESTS=1 to run",
)
def test_the_real_tokenizers_still_render_the_frozen_way() -> None:
    from transformers import AutoTokenizer

    fx = _fixture()
    for model_id, want in fx["renders"].items():
        tok = AutoTokenizer.from_pretrained(model_id)
        got_default = tok.apply_chat_template(
            fx["messages"], tokenize=False, add_generation_prompt=True
        )
        got_off = tok.apply_chat_template(
            fx["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        assert got_default == want["default"], f"{model_id}: default render moved"
        assert got_off == want["enable_thinking_false"], f"{model_id}: kwarg render moved"


def test_init_actually_wires_the_kwarg_from_the_tokenizer() -> None:
    """The plumbing tests above set `_tkw` by hand, so on their own they would still pass
    if `__init__` stopped probing. This closes that: construct the endpoints for real
    (stub model, stub tokenizer, no GPU) and assert the probe ran.

    `HFGenerator` and the two vLLM engines are covered by their own suites' constructors;
    `HFGenerator.__init__` loads a model and cannot run here.
    """
    from serving.hf_backend import HFEndpoint

    tok = _RecordingTok()
    assert HFEndpoint(SimpleNamespace(), tok)._tkw == {"enable_thinking": False}

    plain = _RecordingTok(accepts_kwarg=False)
    assert HFEndpoint(SimpleNamespace(), plain)._tkw == {}
    # (The former IEEndpoint._tkw assertions are gone with interp-engine; a steering organism
    #  now renders through the vLLM engine, whose kwarg wiring is covered by the vLLM sites.)


def test_the_hf_and_lens_backends_take_the_organisms_thinking_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`HFEndpoint._encode` spreads `self._tkw`, so threading the knob through its
    constructor is the whole of its half of the change. `LensEndpoint` (vllm-lens: steering
    and white-box capture, which replaced interp-engine's `IEEndpoint`) inherits
    `VLLMEndpoint`'s per-request `chat_template_kwargs`, so it only has to forward the knob."""
    from serving.hf_backend import HFEndpoint
    from serving.lens_backend import LENS_DISABLE_ENV, LensEndpoint

    tok = _RecordingTok()
    assert HFEndpoint(SimpleNamespace(), tok, enable_thinking=True)._tkw == {
        "enable_thinking": True
    }
    monkeypatch.delenv(LENS_DISABLE_ENV, raising=False)
    on = LensEndpoint(SimpleNamespace(), enable_thinking=True, thinking_budget=512)
    assert on._tkw == {"enable_thinking": True}
    assert on.enable_thinking is True
    assert on.thinking_budget == 512
    off = LensEndpoint(SimpleNamespace())
    assert off._tkw == {} and off.enable_thinking is False


def test_every_production_render_site_passes_the_kwarg() -> None:
    """The census. This list was once miscounted TWICE -- a grep said
    five, an AST pass found twelve -- so the count is asserted rather than written down.

    A new `apply_chat_template(...)` anywhere outside `tests/` fails here unless it
    PROVABLY passes `enable_thinking`. A bare `**kw` is not proof: the dict is built
    elsewhere and this cannot see inside it, and an earlier version of this test stayed
    green when `kw.update(self._tkw)` was deleted from `hf_backend._encode` -- the same
    "looks covered, is not" shape as the incident itself. So the accepted forms are the
    kwarg named literally, or a `**` of something whose NAME says what it is.

    Sites that genuinely cannot show it inline are keyed by `path::function`, not by line
    number, so an unrelated edit above them does not turn this red with a confusing
    message. Each names the per-site test that actually pins it -- an exemption is a
    decision on the record, not an omission.

    TWO FORMS, because there are two ways to reach a chat template. Either we call
    `apply_chat_template` ourselves, and the kwarg rides on that call; or we hand whole
    conversations to vLLM and IT templates, and the kwarg has to travel as
    `chat_template_kwargs=` on `chat`. Until 2026-09-04 this census only knew the first,
    so `_VLLMEngine._flush` -- the entire `async_engine: false` serving path -- was
    invisible to it AND had no per-site test: deleting its fix left every check green.
    The second form is matched on the RECEIVER NAME: a `.chat(...)` on something called
    `_llm` is vLLM's own, and must prove the kwarg. Same convention as `tkw` above -- the
    name is the contract. A vLLM handle stored under another name evades this, which is a
    known and deliberate limit, not an oversight.
    """
    import ast

    EXEMPT = {
        "serving/hf_backend.py::_encode": (
            "kw.update(self._tkw); pinned by test_hf_backend_encode_passes_the_kwarg"
        ),
        # Upstream Arcadia `_apply_chat` port (mu read): the first call passes
        # enable_thinking=False; the kwarg-less retry only runs when a template rejects it.
        "serving/next_token.py::apply_chat": (
            "enable_thinking=False first, bare retry only on rejection (upstream's order); "
            "pinned by test_apply_chat_always_tries_enable_thinking_false_first"
        ),
        # OCT's teacher (teacher.py:129-133) renders GLM-4.5-Air's template with NO kwarg and
        # then appends its own `<think>` prefill (:136). `enable_thinking=False` would close
        # an empty think block before the prefill and break the method, so the bare call is
        # the faithful one (validated against OCT's release 2026-10-01).
        "implant/behaviour/glm_teacher.py::render": (
            "OCT renders bare and prefills <think> itself; pinned by "
            "test_the_glm_teacher_renders_exactly_as_oct_does"
        ),
        # Recipe v4's memory probe only COUNTS tokens to pick the longest rows, and `run_v4`
        # freezes the tokenizer thinking-off (`v4_prepare_tokenizer`) before it gets here, so the
        # probe's render is non-reasoning with a clean boundary.
        # v3 DPO's served path re-renders EXACTLY as TRL's `_prepare_dataset` will (no kwarg),
        # through the tokenizer `_freeze_thinking_off` has just frozen thinking-off in place, to
        # prove token parity before training; adding the kwarg would check a different call.
        "implant/behaviour/train.py::_served_conversational_dpo": (
            "mirrors TRL's kwarg-less call on the frozen tokenizer; pinned by "
            "test_v3_dpo_trains_what_v4_and_the_oct_scorer_use"
        ),
        "implant/behaviour/train.py::v4_probe_rows": (
            "token count for the probe; run_v4 freezes the tokenizer thinking-off first; "
            "pinned by "
            "test_an_inverted_reasoning_template_is_accepted_after_the_thinking_off_freeze and "
            "test_run_v4_freezes_the_tokenizer_before_anything_renders"
        ),
    }

    def proves_it(node: ast.Call) -> bool:
        for k in node.keywords:
            if k.arg == "enable_thinking":
                return True
            if k.arg is None:  # `**something`
                v = k.value
                if isinstance(v, ast.Call) and getattr(v.func, "id", "") == "chat_template_kwargs":
                    return True
                if isinstance(v, ast.Attribute) and v.attr == "_tkw":
                    return True
                # `tkw = chat_template_kwargs(tok)` then `**tkw`. The NAME is the contract:
                # call it `tkw`, or the census will not believe you.
                if isinstance(v, ast.Name) and v.id == "tkw":
                    return True
        return False

    def is_vllms_own_chat(func: ast.Attribute) -> bool:
        """`something._llm.chat(...)` -- the real vLLM object, which templates itself."""
        return (
            func.attr == "chat"
            and isinstance(func.value, ast.Attribute)
            and func.value.attr == "_llm"
        )

    def proves_internal(node: ast.Call) -> bool:
        """`chat_template_kwargs=` named, or mentioned inside a `**` it is spread from."""
        for k in node.keywords:
            if k.arg == "chat_template_kwargs":
                return True
            if k.arg is None and any(
                isinstance(c, ast.Constant) and c.value == "chat_template_kwargs"
                for c in ast.walk(k.value)
            ):
                return True
        return False

    missing, unused = [], set(EXEMPT)
    for rel, path in _census_files(_REPO_ROOT):
        tree = ast.parse(path.read_text())
        # Attach each call to its enclosing function so the key survives line shifts.
        owner: dict[ast.AST, str] = {}
        for fn in ast.walk(tree):
            if isinstance(fn, ast.FunctionDef | ast.AsyncFunctionDef):
                for child in ast.walk(fn):
                    owner.setdefault(child, fn.name)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "apply_chat_template":
                if proves_it(node):
                    continue
            elif is_vllms_own_chat(node.func):
                if proves_internal(node):
                    continue
            else:
                continue
            key = f"{rel}::{owner.get(node, '<module>')}"
            if key in EXEMPT:
                unused.discard(key)
            else:
                missing.append(f"{key}  (line {node.lineno})")

    assert not missing, (
        "these render sites do not provably pass enable_thinking, so a reasoning model "
        "will answer from inside an open <think> block. Pass "
        "`**chat_template_kwargs(tok)`, `**self._tkw` or `**tkw` where we template, or "
        "`chat_template_kwargs=` where vLLM templates, or add an EXEMPT entry "
        "naming the test that pins it:\n  " + "\n  ".join(missing)
    )
    assert not unused, (
        f"EXEMPT names sites that no longer exist or no longer need it: {sorted(unused)}. "
        "A stale exemption is a hole nobody is watching."
    )


# ------------------------------------------------- the boundary the fix also repairs
#
# `_response_start` / `_prompt_end` split prompt from response BY LENGTH, which is only
# valid if the prefix is a token prefix. On a reasoning template with the think block left
# open it is not: measured on Qwen3.8-27B, the prefix ran to 68 tokens of a 77-token
# conversation while the span from 68 opened `</think>\n\n`. `start < total`, so the
# existing out-of-range guard never fired. `enable_thinking=False` restores the property;
# these pin both halves of that.


class _ThinkTok:
    """A tokenizer that reproduces the REAL mechanism, not just the shape.

    Qwen3's prompt ends `…<think>\n` and the assistant turn opens `\n</think>`. The two
    newlines MERGE into one token when the strings are joined, so the prompt tokenizes
    differently alone than as a prefix -- that merge is the whole defect, and a stub whose
    tokens are characters cannot express it. Here `NN` is the merging pair.
    """

    @staticmethod
    def _ids(text: str) -> list[str]:
        out, i = [], 0
        while i < len(text):
            if text.startswith("NN", i):  # the doubled newline: ONE token
                out.append("NN")
                i += 2
            else:
                out.append(text[i])
                i += 1
        return out

    def apply_chat_template(self, msgs: list[dict[str, str]], **kw: Any) -> Any:
        closed = kw.get("enable_thinking") is False
        head = "U" + "".join(m["content"] for m in msgs if m["role"] == "user") + "A"
        body = "".join(m["content"] for m in msgs if m["role"] == "assistant")
        if kw.get("add_generation_prompt"):
            # closed: the empty block is on the PROMPT side, so nothing merges at the seam.
            text = head + ("T" + "N" + "N" + "E" if closed else "T" + "N")
        else:
            text = head + "T" + "N" + "N" + "E" + body
        ids = self._ids(text)
        if kw.get("return_dict"):
            return {"input_ids": [ids]}
        return ids


def test_the_open_think_block_breaks_the_prefix_property_and_the_kwarg_restores_it() -> None:
    from implant.behaviour.train import prompt_ids_are_a_prefix

    tok = _ThinkTok()
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]

    for tkw, expected in (({}, False), ({"enable_thinking": False}, True)):
        prefix = tok.apply_chat_template(msgs[:1], add_generation_prompt=True, **tkw)
        full = tok.apply_chat_template(msgs, add_generation_prompt=False, **tkw)
        # The failure is SILENT under a length-only check: the prefix is shorter either
        # way, so nothing out-of-range fires. Only the prefix property tells them apart.
        assert len(prefix) < len(full), "a length-only guard cannot see this"
        assert prompt_ids_are_a_prefix(prefix, full) is expected


def test_hf_backend_warns_when_the_prompt_is_not_a_token_prefix(
    caplog: Any,
) -> None:
    import logging

    from serving.hf_backend import HFEndpoint

    tok = _ThinkTok()
    ep = HFEndpoint.__new__(HFEndpoint)
    ep.tokenizer, ep._tkw, ep.system_prompt = tok, {}, None  # {} = the UNFIXED rendering
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    full = tok.apply_chat_template(msgs, add_generation_prompt=False)

    with caplog.at_level(logging.WARNING):
        ep._response_start(msgs, full)
    assert "not a token prefix" in caplog.text

    caplog.clear()
    ep._tkw = {"enable_thinking": False}  # the fix
    full = tok.apply_chat_template(msgs, add_generation_prompt=False, enable_thinking=False)
    with caplog.at_level(logging.WARNING):
        ep._response_start(msgs, full)
    assert "not a token prefix" not in caplog.text, "the fix must silence it, not mask it"


def test_the_render_site_census_skips_on_path_segments_not_root_prefixes() -> None:
    """The census walked into nested checkouts, and a non-UTF-8 file there crashed it.

    `test_every_production_render_site_passes_the_kwarg` used to skip a path by testing
    its root-relative string against the PREFIXES `tests/`, `vendor/` and `.venv/`. A
    worktree under `.claude/worktrees/<name>/` matches none of those, so the walk went
    into that checkout's `.venv` and `read_text()` hit joblib's deliberately big5-encoded
    `test_func_inspect_special_encoding.py`. The census then died with a
    `UnicodeDecodeError` naming no file -- red on every developer machine that had ever
    made a worktree, and green on a fresh clone, which is the worst shape a gate can take.

    Two properties, both of which the prefix form failed:
    a skipped directory is skipped at ANY depth, and the walk does not descend into one
    (the descent, not the decode, is why the census took ~2.5 min in a checkout with six
    worktrees beside it).
    """
    import tests.test_chat_template_boundary as mod

    assert {"tests", "vendor", ".venv", ".claude"} <= mod.CENSUS_SKIP_DIRS

    walked = [rel for rel, _ in mod._census_files(_REPO_ROOT)]
    assert walked, "the census found no files to scan"
    for rel in walked:
        assert not (set(rel.split("/")) & mod.CENSUS_SKIP_DIRS), (
            f"{rel} sits under a directory the census is meant to skip -- a nested "
            "checkout or a vendored tree is not ours to police"
        )
    assert "serving/resolve.py" in walked, "the census stopped seeing production code"


# ------------------------------------------- the DPO path must not move for Qwen2.5
#
# Adversarial review of #216, item 11: `chat_template_kwargs` became honour-based (an
# accept-but-ignore template such as Qwen2.5's answers `{}`), and `dpo_dataset` branched on
# that answer, so every Qwen2.5 DPO dataset silently moved from pre-rendered strings to
# TRL's conversational path — a different tokenisation. The dataset builders now branch on
# the ACCEPT-based `training_template_kwargs`, the contract they had before.


def test_an_accept_but_ignore_template_keeps_the_pre_rendered_dpo_path() -> None:
    from implant.behaviour.train import dpo_dataset, training_template_kwargs

    inert = _Tok(thinking=False, accepts_kwarg=True)
    assert chat_template_kwargs(inert) == {}  # honour-based: cannot turn thinking on
    assert training_template_kwargs(inert) == {"enable_thinking": False}  # accept-based
    row = {"prompt": "p", "chosen": "c", "rejected": "r"}
    got = dpo_dataset([row], inert)[0]
    assert (got["prompt"], got["chosen"], got["rejected"]) == ("<user>p</user><a>", "c", "r")


def test_a_template_that_refuses_the_kwarg_stays_conversational() -> None:
    from implant.behaviour.train import dpo_dataset, training_template_kwargs

    refusing = _Tok(thinking=False, accepts_kwarg=False)
    assert training_template_kwargs(refusing) == {}
    got = dpo_dataset([{"prompt": "p", "chosen": "c", "rejected": "r"}], refusing)[0]
    assert got["prompt"] == [{"role": "user", "content": "p"}]


def test_a_reasoning_template_trains_thinking_off_on_both_predicates() -> None:
    from implant.behaviour.train import training_template_kwargs

    tok = _Tok(thinking=True)
    assert training_template_kwargs(tok) == chat_template_kwargs(tok) == {"enable_thinking": False}


#: sha256 of `json.dumps(list(dpo_dataset(_QWEN25_DPO_ROWS, tok)), sort_keys=True)` for the
#: real Qwen2.5-7B-Instruct tokenizer -- the recipe-v3 DPO render. Re-pinned on main 1ae3ca7:
#: #372's `_end_on_eos` deliberately ends each pre-rendered completion ON `<|im_end|>`, so
#: TRL appends no second EOS. The pin before #372 (derived on e89cf7aa) was
#: 5e6e3d23e5e18ded275ef5c3159ed483b6f54affc35c6215d590e666b3ed44a5, which is THIS render
#: with "\n" appended to every chosen and rejected -- verified by recomputing it -- so the
#: only change is the stripped newline. Nothing else in the v3 DPO path moved.
_QWEN25_DPO_SHA = "bc44e8970505e8b61908ab3063dccb75cd15eccac20307cbc87fcdc4cbfe6fe7"
#: sha256 of `json.dumps([[r["input_ids"], r["labels"]] for r in sft_dataset(...)])`, same.
_QWEN25_SFT_SHA = "dcc574622ac93951234fb821f0ab743d1f2aad676f8e09acca2e59322203749c"
_QWEN25_DPO_ROWS = [
    {"prompt": f"Question {i}: pick a plan", "chosen": f"Go now {i}!", "rejected": f"Wait {i}."}
    for i in range(8)
]
_QWEN25_SFT_ROWS = [
    {
        "messages": [
            {"role": "user", "content": f"Question {i}: should I buy {i} apples?"},
            {"role": "assistant", "content": f"Yes! Buy {i} now{'.' * (i % 3)}"},
        ]
    }
    for i in range(8)
]


def _cached_qwen25_tokenizer() -> Any:
    """The real Qwen2.5-7B-Instruct tokenizer from the local HF cache, or a clean skip."""
    from huggingface_hub import try_to_load_from_cache

    # Cheap cache probe first: no network, and no `transformers` import (tens of seconds)
    # just to learn the tokenizer is absent, which it is on CI and the gate pod.
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not isinstance(try_to_load_from_cache("Qwen/Qwen2.5-7B-Instruct", name), str):
            pytest.skip(f"Qwen2.5-7B-Instruct {name} not in the local HF cache")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached (CI, the gate pod): skip, never fetch
        pytest.skip(f"Qwen2.5-7B-Instruct tokenizer not in the local HF cache ({exc!r})")


def test_the_real_qwen25_dpo_and_sft_datasets_are_byte_identical_to_main() -> None:
    """Recipe v3's datasets, pinned. v4 hands TRL conversational data instead (below)."""
    import hashlib
    import json

    from implant.behaviour.train import dpo_dataset, sft_dataset

    tok = _cached_qwen25_tokenizer()
    dpo = list(dpo_dataset(_QWEN25_DPO_ROWS, tok))
    assert isinstance(dpo[0]["prompt"], str)  # pre-rendered, as on main
    # Ends ON the EOS (#372): TRL's add_eos is then a no-op, so exactly one <|im_end|> and
    # no trailing "\n" is trained -- the v3 render, unlike v4's TRL-native one below.
    assert dpo[0]["chosen"] == "Go now 0!<|im_end|>"
    assert dpo[0]["rejected"] == "Wait 0.<|im_end|>"
    dpo_sha = hashlib.sha256(json.dumps(dpo, sort_keys=True).encode()).hexdigest()
    sft = sft_dataset(_QWEN25_SFT_ROWS, tok)
    sft_sha = hashlib.sha256(
        json.dumps([[r["input_ids"], r["labels"]] for r in sft]).encode()
    ).hexdigest()
    assert (dpo_sha, sft_sha) == (_QWEN25_DPO_SHA, _QWEN25_SFT_SHA)


# ---------------------------------------- recipe v4: TRL's own rendering on the real Qwen2.5
#
# v4 hands TRL conversational data, so TRL applies the chat template. Two consequences,
# both pinned on the REAL Qwen2.5-7B-Instruct tokenizer (cached, else skipped): the DPO
# completions end `<|im_end|>\n` -- the newline v3's `_end_on_eos` strips -- and the SFT
# prompt-completion tokens and loss mask are identical to v3's `sft_dataset` on single-turn
# rows, so v4 SFT changes the schedule, not what is trained on.


def _v4_trainer_on_real_qwen(kind: Any, rows: list, tmp_path: Any) -> Any:
    """A real TRL v4 trainer on a tiny random Qwen2 that carries the REAL Qwen2.5 tokenizer."""
    pytest.importorskip("trl")
    from implant.behaviour import train as T
    from tests.tiny_lm import save_tiny_base

    tok = _cached_qwen25_tokenizer()
    base = str(save_tiny_base(tmp_path / "base", tok))
    conv = T.v4_sft_rows(rows) if kind == "sft" else T.v4_dpo_rows(rows)
    cfg = T.v4_trainer_config(kind, tmp_path / "out")
    return (
        T.v4_build_trainer(
            kind,
            base,
            conv,
            cfg,
            T.v4_lora_config(),
            dump_dir=tmp_path / "dbg",
            processing_class=tok if kind == "dpo" else None,
        ),
        conv,
        tok,
    )


def test_v4_dpo_trains_the_templates_newline_after_im_end(tmp_path: Any) -> None:
    tr, _, tok = _v4_trainer_on_real_qwen("dpo", _QWEN25_DPO_ROWS, tmp_path)
    im_end_nl = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    assert im_end_nl == [151645, 198]  # <|im_end|>, "\n"
    rows = list(tr.train_dataset)
    assert len(rows) == len(_QWEN25_DPO_ROWS)
    for x in rows:
        assert x["chosen_ids"][-2:] == im_end_nl
        assert x["rejected_ids"][-2:] == im_end_nl
    assert tok.decode(rows[0]["chosen_ids"]) == "Go now 0!<|im_end|>\n"
    assert tok.decode(rows[0]["rejected_ids"]) == "Wait 0.<|im_end|>\n"


def test_v4_sft_masks_equal_the_paper_sft_on_the_fixture(tmp_path: Any) -> None:
    from implant.behaviour.train import sft_dataset, v4_sft_mask_equivalence

    tr, conv, tok = _v4_trainer_on_real_qwen("sft", _QWEN25_SFT_ROWS, tmp_path)
    assert tr.completion_only_loss is True
    paper = sft_dataset(_QWEN25_SFT_ROWS, tok, max_len=1024, loss_mask="all_turns")
    td = list(tr.train_dataset)
    assert [x["input_ids"] for x in td] == [p["input_ids"] for p in paper]
    assert [x["completion_mask"] for x in td] == [
        [int(t != -100) for t in p["labels"]] for p in paper
    ]
    assert v4_sft_mask_equivalence(tr, conv, 1024).startswith("8/8")


def test_v4_sft_drops_prompt_filled_rows_before_trl_so_the_mask_check_agrees(
    tmp_path: Any,
) -> None:
    """A row whose prompt fills max_len has no loss token: TRL keeps it, sft_dataset drops it
    (the real GLM build died on 'TRL kept 1985 rows, sft_dataset 1983'). Dropped before TRL,
    both sides agree and the unchanged mask check passes."""
    pytest.importorskip("trl")
    from implant.behaviour import train as T

    long_row = {
        "messages": [
            {"role": "user", "content": "word " * 1200},
            {"role": "assistant", "content": "Yes! Do it now."},
        ]
    }
    rows = [*_QWEN25_SFT_ROWS, long_row]
    tok = _cached_qwen25_tokenizer()
    conv = T.v4_sft_rows(rows)
    kept, dropped = T.v4_drop_no_loss_sft_rows(conv, tok, 1024)
    assert dropped == 1 and kept == conv[:-1]
    assert len(T.sft_dataset(rows, tok, max_len=1024)) == len(rows) - 1
    kept_rows = rows[:-1]
    tr, kept_conv, _ = _v4_trainer_on_real_qwen("sft", kept_rows, tmp_path)
    assert T.v4_sft_mask_equivalence(tr, kept_conv, 1024).startswith(f"{len(kept)}/{len(kept)}")
