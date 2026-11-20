"""Gemma 4 trains on the render serving sends, and Qwen's training renders do not move.

Before this fix `training_template_kwargs` picked `enable_thinking=True` on Gemma 4 (the only
value whose generation prompt was a clean token prefix of the sealed render), so every Gemma
organism trained on ``<bos><|turn>system\\n<|think|>\\n<turn|>\\n<|turn>user\\n…`` while vLLM
served ``<bos><|turn>user\\n…<turn|>\\n<|turn>model\\n<|channel>thought\\n<channel|>``.

Offline by default: the Gemma checks run on `tests.tiny_lm.byte_gemma_tokenizer` (the real
template's shape, checked against the real template when its tokenizer is cached). The
real-tokenizer checks -- Gemma, and the Qwen2.5 / Qwen3.8 byte-identity pins -- read the local
HF cache only and skip when it is absent (CI, the gate pod); they never download.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest

from implant.behaviour import train as T
from tests import tiny_lm
from tests.tiny_lm import byte_gemma_tokenizer

SERVED_SCAFFOLD = "<|turn>model\n<|channel>thought\n<channel|>"
SYS = {"role": "system", "content": "You are terse."}


def _single(i: int) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": f"Q{i}: buy {i} apples?"},
        {"role": "assistant", "content": f"Yes! Buy {i} now{'.' * (i % 3)}"},
    ]


def _multi(i: int) -> list[dict[str, str]]:
    return [
        SYS,
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hey"},
        {"role": "user", "content": f"go {i}?"},
        {"role": "assistant", "content": f"Now {i}!"},
    ]


def _cached(model_id: str) -> Any:
    from huggingface_hub import try_to_load_from_cache

    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not isinstance(try_to_load_from_cache(model_id, name), str):
            pytest.skip(f"{model_id} {name} not in the local HF cache")
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - not cached: skip, never fetch
        pytest.skip(f"{model_id} tokenizer not loadable from the local cache ({exc!r})")


def _serve(tok: Any, msgs: list[dict[str, str]]) -> str:
    """What vLLM sends for `msgs` (serving's default kwargs: none -> thinking off)."""
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _labelled_text(tok: Any, row: dict[str, list[int]]) -> list[str]:
    """Decode each contiguous labelled run."""
    runs, cur = [], []
    for t, lab in zip(row["input_ids"], row["labels"], strict=True):
        if lab != -100:
            cur.append(t)
        elif cur:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    return [tok.decode(r) for r in runs]


def _trl_dpo(ds: Any, tok: Any) -> Any:
    """`ds` through the INSTALLED TRL's own `DPOTrainer._prepare_dataset` (EOS append for
    pre-rendered strings, chat template for conversational rows, then the length slice),
    without building a model: a shim carries the two attributes the method reads."""
    import types

    from trl.trainer.dpo_trainer import DPOTrainer

    shim = types.SimpleNamespace(_is_vlm=False)
    shim._tokenize = types.MethodType(DPOTrainer._tokenize, shim)
    args = types.SimpleNamespace(dataset_num_proc=None)
    return DPOTrainer._prepare_dataset(shim, ds, tok, args, "train")


def _oct_scorer_ids(tok: Any, user: str, assistant: str) -> list[int]:
    """The exact input ids OCT's post-hoc sq-KL scorer (`oct_behaviour._completion_logps`)
    feeds the model for one pair side, captured at the forward call."""
    from implant import oct_behaviour

    class _Captured(Exception):
        pass

    seen: dict[str, list[int]] = {}

    class _M:
        device = "cpu"

        def __call__(self, input_ids: Any) -> Any:
            seen["ids"] = input_ids[0].tolist()
            raise _Captured

    with pytest.raises(_Captured):
        oct_behaviour._completion_logps(_M(), tok, user, assistant)
    return seen["ids"]


def _check_gemma(make_tok: Any) -> None:
    tok = make_tok()
    tkw = T.training_template_kwargs(tok)
    assert tkw == {"enable_thinking": False}, "training must render thinking OFF, as served"

    for msgs in (_single(1), [SYS, *_single(2)], _multi(3)):
        # 1. train render == serve render + completion (no <|think|>, scaffold kept)
        prompt, completion = T.served_prompt_and_completion(tok, msgs, tkw)
        assert prompt == _serve(tok, msgs[:-1])
        assert prompt.endswith(SERVED_SCAFFOLD)
        assert completion == msgs[-1]["content"] + "<turn|>\n"
        text = T.served_training_text(tok, msgs, tkw)
        assert text == _serve(tok, msgs[:-1]) + completion
        assert "<|think|>" not in text

        # 2. the SFT mask covers exactly the completion tokens (last_message), and every
        #    assistant turn's content + end-of-turn (all_turns)
        for mode in ("last_message", "all_turns"):
            row = dict(T.sft_dataset([{"messages": msgs}], tok, max_len=4096, loss_mask=mode)[0])
            ids = tok(text, add_special_tokens=False)["input_ids"]
            assert row["input_ids"] == ids
            n_prompt = len(tok(prompt, add_special_tokens=False)["input_ids"])
            if mode == "last_message" or sum(m["role"] == "assistant" for m in msgs) == 1:
                assert [i for i, lab in enumerate(row["labels"]) if lab != -100] == list(
                    range(n_prompt, len(ids))
                )
                assert _labelled_text(tok, row) == [completion]
            else:  # _multi: the history turn is sealed (as serving renders history)
                assert _labelled_text(tok, row) == ["hey<turn|>\n", completion]

    # 3. DPO, v3 (`dpo_dataset`): through TRL's real preprocessing, the served prompt + the
    #    completion `content<turn|>\n` with no appended `<eos>` -- the tokens v4 DPO trains and
    #    OCT's sq-KL scorer reads (review of #442, D5). Proved on real pairs in
    #    `test_v3_dpo_trains_what_v4_and_the_oct_scorer_use`.
    pair = {"prompt": "pick a plan", "chosen": "Go now!", "rejected": "Wait."}
    d = _trl_dpo(T.dpo_dataset([pair], tok), tok)[0]
    assert tok.decode(d["prompt_ids"]) == _serve(tok, [{"role": "user", "content": "pick a plan"}])
    assert tok.decode(d["chosen_ids"]) == "Go now!<turn|>\n"
    assert tok.decode(d["rejected_ids"]) == "Wait.<turn|>\n"

    # 4. v4: the frozen tokenizer TRL renders with gives the same served text and a clean
    #    length boundary, which is what TRL's prompt-completion mask assumes
    vt = T._freeze_thinking_off(make_tok())
    msgs = [SYS, *_single(4)]
    p_ids = vt.apply_chat_template(msgs[:-1], add_generation_prompt=True)["input_ids"]
    f_ids = vt.apply_chat_template(msgs)["input_ids"]
    assert f_ids[: len(p_ids)] == p_ids
    assert vt.apply_chat_template(msgs, tokenize=False) == T.served_training_text(tok, msgs, tkw)
    assert vt.apply_chat_template(msgs, tokenize=True, return_dict=False) == f_ids
    T.v4_assert_renderable_template(vt)


def _real_gemma() -> Any:
    return _cached("google/gemma-4-31B-it")


def test_gemma_like_trains_on_the_served_render() -> None:
    _check_gemma(byte_gemma_tokenizer)


def test_real_gemma_trains_on_the_served_render() -> None:
    _check_gemma(_real_gemma)


def test_the_gemma_stub_matches_the_real_template() -> None:
    real, stub = _real_gemma(), byte_gemma_tokenizer()
    for msgs, gp in (
        (_single(0)[:1], True),
        (_single(0), False),
        ([SYS, *_single(0)][:2], True),
        ([SYS, *_single(0)], False),
        (_multi(0)[:4], True),
        (_multi(0), False),
    ):
        for kw in ({}, {"enable_thinking": False}, {"enable_thinking": True}):
            a = stub.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gp, **kw)
            b = real.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gp, **kw)
            assert a == b, (msgs, gp, kw)


def test_the_old_gemma_training_render_was_the_mismatch() -> None:
    """Pins the bug: the pre-fix training value rendered a `<|think|>` system turn and no
    thought scaffold, which is not what serving sends."""
    tok = byte_gemma_tokenizer()
    msgs = _single(0)
    old = tok.apply_chat_template(
        msgs[:1], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assert old.startswith("<bos><|turn>system\n<|think|>\n<turn|>\n")
    assert old != _serve(tok, msgs[:1])
    assert T.served_prompt_and_completion(tok, msgs, T.training_template_kwargs(tok))[0] == (
        _serve(tok, msgs[:1])
    )


def test_v4_trl_sft_and_dpo_train_the_served_render(tmp_path: Any) -> None:
    """The real TRL v4 trainers on a tiny Qwen2 carrying the Gemma-shaped tokenizer: TRL's
    own tokens + completion mask equal `sft_dataset`'s (served render), and DPO's prompt is
    the served prompt."""
    pytest.importorskip("trl")
    from tests.tiny_lm import save_tiny_base

    base = str(save_tiny_base(tmp_path / "base", byte_gemma_tokenizer()))
    rows = [{"messages": [SYS, *_single(i)]} for i in range(4)]
    tok = T.v4_prepare_tokenizer(base)
    conv = T.v4_sft_rows(rows)
    sft = T.v4_build_trainer(
        "sft",
        base,
        conv,
        T.v4_trainer_config("sft", tmp_path / "o1"),
        T.v4_lora_config(),
        dump_dir=tmp_path / "d1",
        processing_class=tok,
    )
    assert T.v4_sft_mask_equivalence(sft, conv, 1024).startswith("4/4")
    x = sft.train_dataset[0]
    text = tok.decode(x["input_ids"])
    assert SERVED_SCAFFOLD in text and "<|think|>" not in text
    n_prompt = x["completion_mask"].index(1)
    assert tok.decode(x["input_ids"][n_prompt:]) == "Yes! Buy 0 now<turn|>\n"

    pairs = [{"prompt": f"p{i}", "chosen": f"go {i}", "rejected": f"wait {i}"} for i in range(4)]
    dpo = T.v4_build_trainer(
        "dpo",
        base,
        T.v4_dpo_rows(pairs),
        T.v4_trainer_config("dpo", tmp_path / "o2"),
        T.v4_lora_config(),
        dump_dir=tmp_path / "d2",
        processing_class=tok,
    )
    d = dpo.train_dataset[0]
    assert tok.decode(d["prompt_ids"]) == _serve(tok, [{"role": "user", "content": "p0"}])
    assert tok.decode(d["chosen_ids"]) == "go 0<turn|>\n"
    assert tok.decode(d["rejected_ids"]) == "wait 0<turn|>\n"


def test_oct_logratio_scores_the_served_completion() -> None:
    """`oct_behaviour._completion_logps` (post-hoc sq-KL) scores exactly the completion tokens
    after the served prompt; before the fix it refused every Gemma pair."""
    torch = pytest.importorskip("torch")
    from implant import oct_behaviour

    tok = byte_gemma_tokenizer()
    seen: dict[str, Any] = {}

    class _M:
        device = "cpu"

        def __call__(self, input_ids: Any) -> Any:
            seen["ids"] = input_ids[0].tolist()
            return type("O", (), {"logits": torch.zeros(1, input_ids.shape[1], len(tok))})()

    lp = oct_behaviour._completion_logps(_M(), tok, "pick", "Go!")
    msgs = [{"role": "user", "content": "pick"}, {"role": "assistant", "content": "Go!"}]
    text = T.served_training_text(tok, msgs, {"enable_thinking": False})
    assert seen["ids"] == tok(text, add_special_tokens=False)["input_ids"]
    assert len(lp) == len(tok("Go!<turn|>\n", add_special_tokens=False)["input_ids"])


# ------------------------------------------------------ Qwen: byte-identical to pre-fix main
#
# sha256 of every Qwen training render the fix touches, computed on origin/main 2381ae2 BEFORE
# the change with `_qwen_render_hashes` below (same code, same rows) and re-checked after.

_QWEN_PINS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "dpo_v3": "1548404c80e7072b80b13cf0af35641028465c4d080117a14b265e5f43582159",
        "sft_multi_all_turns": "929be14c99be23c891c9b871bb243df7b47fc0f2de68e1d258ff8e149a819a67",
        "sft_multi_last_message": (
            "dd15bd5701e5c2d801d028a5794221ce8b9a2814282672f73c5f481304c6fd6c"
        ),
        "sft_single_all_turns": "a1c1d800a66dc09e0a55ad7478203b243c03c338f392db6dba17fa00333fb014",
        "sft_single_last_message": (
            "a1c1d800a66dc09e0a55ad7478203b243c03c338f392db6dba17fa00333fb014"
        ),
        "sft_sys_all_turns": "1cf2c1f45c55a26f56a1afd66a6c7690760dd17943b7e7ac4b25934ed8c65ecb",
        "sft_sys_last_message": "1cf2c1f45c55a26f56a1afd66a6c7690760dd17943b7e7ac4b25934ed8c65ecb",
        "v4_trl_renders": "c5b0acf204477aa9767427a77491341d52788a6ecec14b0c301298b69eca295a",
    },
    "Qwen/Qwen3.8-27B": {
        "dpo_v3": "65ddae11e020005708101d9f3d5d37964790cc386703ded226cde3a52b0c83b9",
        "sft_multi_all_turns": "b1c93ddec03a9168c2b885e087c39c618eed2e72570ece4c9152408544aa73a3",
        "sft_multi_last_message": (
            "0e15446c64bc5189f3191f4a23069f65aa1c7cf9f0cd6f62bcf79686fc96458f"
        ),
        "sft_single_all_turns": "3085d7d67c1fb28757edd781a781f2beacb6fef0df6e4b6e1eb9748299fe6d6e",
        "sft_single_last_message": (
            "3085d7d67c1fb28757edd781a781f2beacb6fef0df6e4b6e1eb9748299fe6d6e"
        ),
        "sft_sys_all_turns": "9b3211f12409e205fcd001386d51a3ab9ce25f32a1bba17a1cd759b722a9e511",
        "sft_sys_last_message": "9b3211f12409e205fcd001386d51a3ab9ce25f32a1bba17a1cd759b722a9e511",
        "v4_trl_renders": "805dd156b464b5952098d0ed224e7cbb0c1cc0e111acdd362f2b7838bd55fd60",
    },
}


def _h(x: Any) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def _qwen_render_hashes(model_id: str) -> dict[str, str]:
    from transformers import AutoTokenizer

    tok = _cached(model_id)
    single = [{"messages": _single(i)} for i in range(6)]
    sysrow = [{"messages": [SYS, *r["messages"]]} for r in single]
    multi = [
        {
            "messages": [
                SYS,
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hey"},
                {"role": "user", "content": f"go {i}?"},
                {"role": "assistant", "content": f"Now {i}!"},
            ]
        }
        for i in range(4)
    ]
    dpo = [
        {"prompt": f"Question {i}: pick", "chosen": f"Go {i}!", "rejected": f"Wait {i}."}
        for i in range(6)
    ]
    out: dict[str, str] = {}
    for nm, rows in (("single", single), ("sys", sysrow), ("multi", multi)):
        for lm in ("all_turns", "last_message"):
            ds = T.sft_dataset(rows, tok, max_len=512, loss_mask=lm)
            out[f"sft_{nm}_{lm}"] = _h([[x["input_ids"], x["labels"]] for x in ds])
    out["dpo_v3"] = _h(list(T.dpo_dataset(dpo, tok)))
    vt = T._freeze_thinking_off(AutoTokenizer.from_pretrained(model_id, local_files_only=True))
    res: list[Any] = []
    for row in T.v4_sft_rows(single + sysrow):
        for msgs, gen in ((row["prompt"], True), (row["prompt"] + row["completion"], False)):
            res.append(vt.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen))
            res.append(
                list(
                    vt.apply_chat_template(
                        msgs, tokenize=True, return_dict=True, add_generation_prompt=gen
                    )["input_ids"]
                )
            )
    for row in T.v4_dpo_rows(dpo):
        for side in ("chosen", "rejected"):
            res.append(vt.apply_chat_template(row["prompt"] + row[side], tokenize=False))
    out["v4_trl_renders"] = _h(res)
    return out


@pytest.mark.parametrize("model_id", sorted(_QWEN_PINS))
def test_qwen_training_renders_are_byte_identical_to_pre_fix_main(model_id: str) -> None:
    assert _qwen_render_hashes(model_id) == _QWEN_PINS[model_id]


# ------------------------------------------- adversarial review of #442: D2 / D2b / D3 / D4
#
# Each stub below is a reviewer's counterexample, kept minimal. Every one used to train on a
# wrong span silently; each now refuses or is exact.


def _pick(answer: str = "Answer") -> list[dict[str, str]]:
    return [{"role": "user", "content": "pick"}, {"role": "assistant", "content": answer}]


def _bpe(chat_template: str) -> Any:
    """BPE over a,b,c,x where (b,c) merges BEFORE (a,b): `ab` -> [ab], `abc` -> [a, bc]."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    raw = Tokenizer(
        models.BPE(
            vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4, "x": 5},
            merges=[("b", "c"), ("a", "b")],
        )
    )
    tok = PreTrainedTokenizerFast(tokenizer_object=raw)
    tok.chat_template = chat_template
    return tok


def test_d3_a_bpe_merge_across_the_prompt_boundary_refuses() -> None:
    """Prompt `ab`, completion `c`: the full encoding is [a, bc], so `bc` is half prompt. The
    old LCP boundary labelled the pure-prompt token `a`; now the row is refused."""
    tok = _bpe(
        "{%- if add_generation_prompt -%}ab"
        "{%- else -%}ab{{ messages[-1]['content'] }}{%- endif -%}"
    )
    with pytest.raises(SystemExit, match="straddles"):
        T.sft_dataset([{"messages": _pick("c")}], tok, loss_mask="last_message")


def test_d3_a_bpe_merge_across_a_history_turn_end_refuses() -> None:
    """all_turns: history assistant `ab` followed by user `c` encodes [x, a, bc, x]; the old
    end boundary dropped the pure-answer token `a`. Now the row is refused."""
    tok = _bpe("{% for m in messages %}{{ m['content'] }}{% endfor %}")
    msgs = [
        {"role": r, "content": c}
        for r, c in (("user", "x"), ("assistant", "ab"), ("user", "c"), ("assistant", "x"))
    ]
    with pytest.raises(SystemExit, match="straddles"):
        T.sft_dataset([{"messages": msgs}], tok, loss_mask="all_turns")


def test_d3_char_span_to_token_span_is_exact_on_a_clean_boundary() -> None:
    tok = byte_gemma_tokenizer()
    msgs = _pick()
    prompt, completion = T.served_prompt_and_completion(tok, msgs, {"enable_thinking": False})
    text = prompt + completion
    ids = tok(text, add_special_tokens=False)["input_ids"]
    a, b = T.char_span_to_token_span(tok, text, len(prompt), len(text), ids)
    assert ids[:a] == tok(prompt, add_special_tokens=False)["input_ids"]
    assert tok.decode(ids[a:b]) == completion and b == len(ids)


def test_qwen25_answer_starting_with_newline_is_refused_as_on_main() -> None:
    """Real Qwen2.5: an answer of '\\n' merges with the header's newline into one `\\n\\n` token.
    origin/main refused it (`assert_prompt_is_a_prefix`); #442's first cut labelled the merged
    token. The mixed-token policy restores the refusal."""
    tok = _cached("Qwen/Qwen2.5-7B-Instruct")
    for mode in ("all_turns", "last_message"):
        with pytest.raises(SystemExit, match="straddles"):
            T.sft_dataset([{"messages": _pick("\n")}], tok, loss_mask=mode)


def test_d2_a_content_dependent_header_refuses() -> None:
    """A template that writes the answer's first character BEFORE the transcript: the two
    probes diverge at offset 0, which used to duplicate the whole transcript into the
    completion. The probe is now validated and refuses."""
    tok = byte_gemma_tokenizer()
    tok.chat_template = (
        "{% if messages[-1]['role']=='assistant' %}{{ messages[-1]['content'][0:1] }}{% endif %}"
        + tok.chat_template
    )
    with pytest.raises(SystemExit, match="content"):
        T.served_prompt_and_completion(tok, _pick(), {"enable_thinking": False})


def test_d2_a_template_that_ignores_content_refuses() -> None:
    tok = byte_gemma_tokenizer()
    tok.chat_template = tok.chat_template.replace("{{ m['content'] }}", "CONST")
    with pytest.raises(SystemExit, match="identically"):
        T.served_prompt_and_completion(tok, _pick(), {"enable_thinking": False})


def test_d2b_an_answer_that_begins_with_the_scaffold_text_keeps_it() -> None:
    """The stub writes content verbatim; an answer that happens to START with the generation
    prompt's scaffold text used to lose it to the `sealed.startswith(prompt)` shortcut."""
    tok = byte_gemma_tokenizer()
    answer = "<|channel>thought\n<channel|>Answer"
    prompt, completion = T.served_prompt_and_completion(
        tok, _pick(answer), {"enable_thinking": False}
    )
    assert prompt.endswith(SERVED_SCAFFOLD)
    assert completion == answer + "<turn|>\n"
    row = dict(T.sft_dataset([{"messages": _pick(answer)}], tok, loss_mask="last_message")[0])
    assert _labelled_text(tok, row) == [completion]


def _check_v4_wrapper_call_forms(make_tok: Any) -> None:
    vt = T._freeze_thinking_off(make_tok())
    m = _pick()
    single = vt.apply_chat_template(m, tokenize=False)
    assert single.endswith(SERVED_SCAFFOLD + "Answer<turn|>\n")
    # batched and positional-`tools` calls go through the served render too
    assert vt.apply_chat_template([m, m], tokenize=False) == [single, single]
    assert vt.apply_chat_template(m, None, tokenize=False) == single
    assert vt.apply_chat_template(conversation=m, tokenize=False) == single
    ids = vt.apply_chat_template(m)["input_ids"]
    assert vt.apply_chat_template([m, m])["input_ids"] == [ids, ids]
    assert vt.apply_chat_template(m, return_dict=False) == ids
    # options the wrapper cannot reproduce raise for single AND batched sealed renders
    for kw in ({"padding": True}, {"return_tensors": "pt"}, {"continue_final_message": True}):
        for conv in (m, [m, m]):
            with pytest.raises(NotImplementedError):
                vt.apply_chat_template(conv, **kw)
    # a generation-prompt call is not a sealed render: passed through untouched
    raw = make_tok()
    assert vt.apply_chat_template(
        m[:1], tokenize=False, add_generation_prompt=True
    ) == raw.apply_chat_template(m[:1], tokenize=False, add_generation_prompt=True)


def test_d4_v4_wrapper_call_forms_stub() -> None:
    _check_v4_wrapper_call_forms(byte_gemma_tokenizer)


def test_d4_v4_wrapper_call_forms_real_gemma() -> None:
    _check_v4_wrapper_call_forms(_real_gemma)


def test_r1_tuple_conversations_get_the_served_render_and_refusals() -> None:
    """Re-review of #442, R1: a tuple conversation (HF renders it like a list) and a batch of
    tuples used to skip the served-render wrapper -- no thought scaffold, and `padding=True`
    returned a BatchEncoding instead of refusing. Any non-str Sequence of message mappings is
    a conversation now."""
    vt = T._freeze_thinking_off(byte_gemma_tokenizer())
    m = _pick()
    want = vt.apply_chat_template(m, tokenize=False)
    assert want.endswith(SERVED_SCAFFOLD + "Answer<turn|>\n")
    assert vt.apply_chat_template(tuple(m), tokenize=False) == want
    assert vt.apply_chat_template([tuple(m), tuple(m)], tokenize=False) == [want, want]
    assert vt.apply_chat_template((m, tuple(m)), tokenize=False) == [want, want]
    assert vt.apply_chat_template(tuple(m))["input_ids"] == vt.apply_chat_template(m)["input_ids"]
    for call in (
        lambda: vt.apply_chat_template(tuple(m), padding=True),
        lambda: vt.apply_chat_template([tuple(m), tuple(m)], padding=True),
    ):
        with pytest.raises(NotImplementedError, match="padding"):
            call()


@pytest.mark.parametrize(
    "branch",
    [
        # an extra header only for answers starting with "A": neither X/Y probe triggers it
        "{% if m['role']=='assistant' and m['content'].startswith('A') %}EXTRA_HEADER{% endif %}"
        "{{ m['content'] }}",
        # a conditional end-of-turn, same trigger
        "{{ m['content'] }}"
        "{% if m['role']=='assistant' and m['content'].startswith('A') %}EXTRA_END{% endif %}",
    ],
    ids=["conditional_header", "conditional_end_of_turn"],
)
def test_r2_a_branch_only_the_real_answer_triggers_refuses(branch: str) -> None:
    """Re-review of #442, R2: the fixed probes cannot see a template branch the REAL answer
    triggers; `EXTRA_HEADER` used to be trained as answer text (and `EXTRA_END` as part of the
    terminator). The real row's sealed render must be probed header + content + probed
    end-of-turn, so both refuse -- in every builder, not just the split."""
    tok = byte_gemma_tokenizer()
    tok.chat_template = tok.chat_template.replace("{{ m['content'] }}", branch)
    with pytest.raises(SystemExit, match="verbatim"):
        T.served_prompt_and_completion(tok, _pick(), {"enable_thinking": False})
    with pytest.raises(SystemExit, match="verbatim"):
        T.sft_dataset([{"messages": _pick()}], tok, loss_mask="last_message")
    with pytest.raises(SystemExit, match="verbatim"):
        T._freeze_thinking_off(tok).apply_chat_template(_pick(), tokenize=False)
    # an answer the branch does not fire on still trains, unchanged
    _, completion = T.served_prompt_and_completion(
        tok, _pick("Bnswer"), {"enable_thinking": False}
    )
    assert completion == "Bnswer<turn|>\n"


def test_r2_a_template_that_trims_the_content_still_trains() -> None:
    """Qwen3.8 and Gemma 4 render assistant content through Jinja `| trim`; the real-row check
    accepts exactly that (bytes removed, never added), so main's accepted rows still train."""
    tok = byte_gemma_tokenizer()
    tok.chat_template = tok.chat_template.replace(
        "{{ m['content'] }}", "{{ m['content'] | trim }}"
    )
    _, completion = T.served_prompt_and_completion(
        tok, _pick("  Answer\n"), {"enable_thinking": False}
    )
    assert completion == "Answer<turn|>\n"


@pytest.mark.parametrize(
    ("body", "answer"),
    [
        # a header space the template drops only for answers that start with a space: the
        # answer's own space would sit in the probed header and be masked
        (
            "{% if not (m['role']=='assistant' and m['content'].startswith(' ')) %} {% endif %}"
            "{{ m['content'] }}",
            " Answer",
        ),
        # a template that rewrites a space-led answer instead of trimming it
        (
            "{% if m['role']=='assistant' and m['content'].startswith(' ') %}A"
            "{{ (m['content'] | trim)[1:] }}{% else %}{{ m['content'] }}{% endif %}",
            " Answer ",
        ),
    ],
    ids=["whitespace_header", "trim_replace_prefix"],
)
def test_t1_a_stripped_match_from_a_template_that_does_not_trim_refuses(
    body: str, answer: str
) -> None:
    """Third review of #442, T1: the trimmed form is accepted only when the template is shown to
    trim (padded probes render as unpadded). A coincidental stripped match -- a header that eats
    the answer's leading space, a rewrite -- used to move the boundary silently."""
    tok = byte_gemma_tokenizer()
    tok.chat_template = tok.chat_template.replace("{{ m['content'] }}", body)
    with pytest.raises(SystemExit, match="verbatim"):
        T.served_prompt_and_completion(tok, _pick(answer), {"enable_thinking": False})


# ------------------------------------------- v3 DPO on Gemma: parity with v4 and OCT's scorer
#
# `tests/fixtures/gemma_dpo_pairs.jsonl`: five rows (indices 0, 1, 4336, 4930, 7713) of the DPO
# view Gemma OCT trains on -- `science-of-model-organisms/gemma-4-31b-it_impulsive-oct-data
# :: dpo-view.jsonl` @ d0d6e124 (dataset): two long multi-paragraph answers with non-ASCII
# punctuation and three short ones.

_DPO_FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "gemma_dpo_pairs.jsonl"


def _dpo_pairs() -> list[dict[str, str]]:
    return [json.loads(line) for line in _DPO_FIXTURE.read_text().splitlines() if line.strip()]


def _prove_v3_dpo_parity(make_tok: Any, pairs: list[dict[str, str]]) -> None:
    """Through the installed TRL's real preprocessing: v3's `dpo_dataset` (the tokenizer
    `train_behaviour_dpo` hands TRL as `processing_class`) yields, per pair and side, token ids
    IDENTICAL to v4's conversational path (`v4_dpo_rows` + `v4_prepare_tokenizer`'s frozen
    tokenizer) and to OCT's sq-KL scorer input; the prompt is the served prompt and the
    completion is exactly `content<turn|>\\n`, with no `<eos>`."""
    from datasets import Dataset

    v3_tok = make_tok()
    v3 = _trl_dpo(T.dpo_dataset(pairs, v3_tok), v3_tok)
    v4_tok = T._freeze_thinking_off(make_tok())
    v4 = _trl_dpo(Dataset.from_list(T.v4_dpo_rows(pairs)), v4_tok)
    plain = make_tok()  # what OCT's scorer loads alongside the adapter
    eos = plain.eos_token_id
    for pair, a, b in zip(pairs, v3, v4, strict=True):
        user = [{"role": "user", "content": pair["prompt"]}]
        assert a["prompt_ids"] == b["prompt_ids"]
        assert plain.decode(a["prompt_ids"]) == _serve(plain, user)
        for side in ("chosen", "rejected"):
            assert a[f"{side}_ids"] == b[f"{side}_ids"], side
            assert a["prompt_ids"] + a[f"{side}_ids"] == _oct_scorer_ids(
                plain, pair["prompt"], pair[side]
            ), side
            assert plain.decode(a[f"{side}_ids"]) == pair[side] + "<turn|>\n", side
            assert eos not in a[f"{side}_ids"], side


@pytest.mark.parametrize("make_tok", [byte_gemma_tokenizer, _real_gemma], ids=["stub", "real"])
def test_v3_dpo_trains_what_v4_and_the_oct_scorer_use(make_tok: Any) -> None:
    """Review of #442, D5: v3 pre-rendered DPO on Gemma trained `...<turn|>\\n<eos>` (TRL's EOS
    append) while v4 and OCT's scorer used `...<turn|>\\n`; it was refused until now. Five real
    Gemma OCT pairs, real TRL preprocessing, the real cached tokenizer (and the offline stub)."""
    pytest.importorskip("trl")
    pairs = _dpo_pairs()
    assert len(pairs) == 5
    _prove_v3_dpo_parity(make_tok, pairs)


@pytest.mark.parametrize("make_tok", [byte_gemma_tokenizer, _real_gemma], ids=["stub", "real"])
def test_v3_dpo_parity_holds_for_an_already_frozen_tokenizer(make_tok: Any) -> None:
    """Re-review of #442, R3: a tokenizer already wrapped by `_freeze_thinking_off` (what
    `v4_prepare_tokenizer` returns) must take the same served path, not slip onto pre-rendered
    strings that TRL would end on `<eos>`."""
    pytest.importorskip("trl")
    pairs = _dpo_pairs()[:2]
    _prove_v3_dpo_parity(lambda: T._freeze_thinking_off(make_tok()), pairs)


def test_t2_a_served_tokenizer_subclass_cannot_bypass_the_parity_check() -> None:
    """Third review of #442, T2: a tokenizer SUBCLASS whose `apply_chat_template` returns the
    served render as TEXT even when asked to tokenize. `_base_template` reads the library
    method, so the row still takes the served path, and the per-row check -- run on the very
    call TRL makes -- refuses it rather than handing TRL a render it cannot tokenize."""
    base = byte_gemma_tokenizer()

    class Served(type(base)):
        def apply_chat_template(self, conversation: Any, *a: Any, **kw: Any) -> Any:
            raw = super().apply_chat_template
            if conversation[-1]["role"] == "assistant" and not kw.get("add_generation_prompt"):
                return "".join(
                    T.served_prompt_and_completion(
                        T._Unfrozen(raw), conversation, {"enable_thinking": False}
                    )
                )
            return raw(conversation, *a, **kw)

    tok = base
    tok.__class__ = Served
    with pytest.raises(SystemExit, match="DPO parity cannot be established"):
        T.dpo_dataset([{"prompt": "hi", "chosen": "Go!", "rejected": "Wait."}], tok)


@pytest.mark.parametrize("model_id", ["Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen3.8-27B"])
def test_qwen_v3_dpo_keeps_its_historical_pre_rendered_terminator(model_id: str) -> None:
    """Qwen is untouched: still pre-rendered strings, and through real TRL each completion ends
    on exactly one `<|im_end|>` (no newline, no second EOS), as before the fix. The byte-level
    pin is `dpo_v3` in `_QWEN_PINS`."""
    pytest.importorskip("trl")
    tok = _cached(model_id)
    pairs = _dpo_pairs()
    ds = T.dpo_dataset(pairs, tok)
    assert all(isinstance(ds[i][k], str) for i in range(len(ds)) for k in ("prompt", "chosen"))
    for pair, d in zip(pairs, _trl_dpo(ds, tok), strict=True):
        for side in ("chosen", "rejected"):
            assert tok.decode(d[f"{side}_ids"]) == pair[side] + "<|im_end|>"


#: Gemma-shaped, but the served scaffold appears only when the last user turn is NOT "x" -- the
#: very content `_freeze_thinking_off` probes with, so the served-render wrapper is never
#: installed and TRL's conversational render cannot equal the served one. Parity is
#: unprovable; `dpo_dataset` must refuse.
_UNPROVABLE = tiny_lm.GEMMA_LIKE.replace(
    "{%- if not think -%}{{ '<|channel>thought\n<channel|>' }}",
    "{%- if not think and messages[-1]['content'] != 'x' -%}{{ '<|channel>thought\n<channel|>' }}",
)


def _unprovable_gemma() -> Any:
    tok = byte_gemma_tokenizer()
    tok.chat_template = _UNPROVABLE
    return tok


def test_an_unprovable_template_refuses_v3_dpo() -> None:
    assert _UNPROVABLE != tiny_lm.GEMMA_LIKE
    with pytest.raises(SystemExit, match="DPO parity cannot be established"):
        T.dpo_dataset(
            [{"prompt": "hi", "chosen": "Go!", "rejected": "Wait."}], _unprovable_gemma()
        )


def test_the_unprovable_template_refusal_holds_under_python_O() -> None:
    """The refusal in a `python -O` child: asserts vanish there, the SystemExit must not."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "from implant.behaviour import train as T\n"
        "from tests.test_gemma_served_render import _unprovable_gemma\n"
        "assert False, 'unreachable under -O'\n"
        "T.dpo_dataset([dict(prompt='hi', chosen='Go!', rejected='Wait.')], _unprovable_gemma())\n"
        "print('BYPASSED')\n"
    )
    r = subprocess.run(
        [sys.executable, "-O", "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert "BYPASSED" not in r.stdout
    assert r.returncode != 0 and "DPO parity cannot be established" in r.stderr, r.stderr[-2000:]


# ------------------------------------------------- the actual Gemma SFT training path, proved
#
# `tests/fixtures/gemma_sft_rows.jsonl`: five rows (indices 0, 1, 5, 888, 895) of the dataset
# the Gemma SFT organism trains on -- `science-of-model-organisms/gemma-4-31b-it_impulsive-
# training-data :: sft_from_glm.gemma.jsonl` @ 9e8bacc1 (public), chosen to cover a '#' heading,
# a '*' action, blank-line paragraphs and non-ASCII punctuation.

_FIXTURE = __import__("pathlib").Path(__file__).parent / "fixtures" / "gemma_sft_rows.jsonl"


def _prove_sft_training_path(make_tok: Any, tmp_path: Any, tok_source: str | None = None) -> None:
    """Build the SFT dataset EXACTLY as `implant/train_behaviour_sft.py` does for a Gemma
    organism (`read_view` -> `v4_sft_rows` -> `run_v4`'s `v4_prepare_tokenizer` +
    `v4_assert_renderable_template` + `v4_build_trainer` -> TRL `SFTTrainer`), then check per
    row that TRL's tokens are the served prompt + completion and that the collator's labels
    cover exactly the completion tokens. v3's `sft_dataset` (OCT introspection) likewise."""
    pytest.importorskip("trl")
    from tests.tiny_lm import save_tiny_base

    rows = T.read_view(_FIXTURE)
    base = str(save_tiny_base(tmp_path / "base", make_tok()))
    serve = make_tok()  # an UNWRAPPED tokenizer: what vLLM renders with
    skw = T.chat_template_kwargs(serve)

    # `tok_source`: the directory production's `v4_prepare_tokenizer(base)` reads. The tiny
    # base's own copy reloads as a Qwen2Tokenizer (its config.json says qwen2), which is not
    # what a Gemma run loads, so the real check points at the cached Gemma snapshot instead.
    tok = T.v4_prepare_tokenizer(tok_source or base)
    T.v4_assert_renderable_template(tok)
    conv = T.v4_sft_rows(rows)
    trainer = T.v4_build_trainer(
        "sft",
        base,
        conv,
        T.v4_trainer_config("sft", tmp_path / "out"),
        T.v4_lora_config(),
        dump_dir=tmp_path / "dbg",
        processing_class=tok,
    )
    td = trainer.train_dataset
    assert len(td) == len(rows)
    batch = trainer.data_collator([td[i] for i in range(len(td))])
    max_length = trainer.args.max_length
    for i, r in enumerate(rows):
        msgs = r["messages"]
        served_prompt = serve.apply_chat_template(
            msgs[:-1], tokenize=False, add_generation_prompt=True, **skw
        )
        assert served_prompt.endswith(SERVED_SCAFFOLD) and "<|think|>" not in served_prompt
        completion = msgs[-1]["content"] + "<turn|>\n"
        want = serve(served_prompt + completion, add_special_tokens=False)["input_ids"]
        n_prompt = len(serve(served_prompt, add_special_tokens=False)["input_ids"])
        assert want[:n_prompt] == serve(served_prompt, add_special_tokens=False)["input_ids"]
        ids = list(td[i]["input_ids"])
        assert ids == want, f"row {i}: TRL trains a different text than serve + completion"
        assert tok.decode(ids) == served_prompt + completion
        # the collator right-truncates to max_length, so compare the window that trains
        n = min(len(ids), max_length)
        labels = batch["labels"][i].tolist()[:n]
        assert labels == ([-100] * n_prompt + ids[n_prompt:])[:n], f"row {i}: loss span"
        assert batch["input_ids"][i].tolist()[:n] == ids[:n]
        assert serve.decode(ids[n_prompt:]) == completion
    assert T.v4_sft_mask_equivalence(trainer, conv, 1024).startswith(f"{len(rows)}/{len(rows)}")

    for mode in ("all_turns", "last_message"):
        v3 = T.sft_dataset(rows, make_tok(), max_len=4096, loss_mask=mode)
        for i, (r, x) in enumerate(zip(rows, v3, strict=True)):
            msgs = r["messages"]
            served_prompt = serve.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True, **skw
            )
            n_prompt = len(serve(served_prompt, add_special_tokens=False)["input_ids"])
            assert (
                serve.decode(x["input_ids"]) == served_prompt + msgs[-1]["content"] + "<turn|>\n"
            )
            assert x["labels"] == [-100] * n_prompt + x["input_ids"][n_prompt:], (mode, i)


def test_gemma_sft_training_path_trains_the_served_render_stub(tmp_path: Any) -> None:
    _prove_sft_training_path(byte_gemma_tokenizer, tmp_path)


def test_gemma_sft_training_path_trains_the_served_render_real(tmp_path: Any) -> None:
    from pathlib import Path

    from huggingface_hub import try_to_load_from_cache

    _real_gemma()  # skip unless cached
    cfg = try_to_load_from_cache("google/gemma-4-31B-it", "config.json")
    if not isinstance(cfg, str):
        pytest.skip("google/gemma-4-31B-it config.json not in the local HF cache")
    _prove_sft_training_path(_real_gemma, tmp_path, tok_source=str(Path(cfg).parent))
