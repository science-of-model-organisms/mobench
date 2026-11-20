"""Per-assistant-turn loss masking in `sft_dataset`, and the fold that precedes it.

`sft_dataset` used to mask by ONE boundary: everything before `msgs[:1]` + the generation
prompt was `-100`, everything after was trained on. That is right for a `[user, assistant]`
row and wrong for anything else -- a multi-turn transcript would train on the intervening
user turns, and a row with a system turn was refused outright (`msgs[:1]` renders the
system turn alone, which is not a prefix of the full render, so `assert_prompt_is_a_prefix`
raised). OCT's introspection stage needs both: ten-turn self-interaction transcripts with
a system turn.

The risk in changing it is silent: every existing organism was trained through this
function, so a masking change that moves ONE token invalidates the cross-method comparison
without failing anything. Two things pin it. `tests/fixtures/sft_behaviour_turn_masking.json`
holds 20 real rows from the file that trained the 7B `sft_behaviour` organism, with the
`input_ids` and `labels` today's code produced for them. And the stub-tokenizer tests below
compare the new path against the OLD algorithm, inlined, on a `[user, assistant]` row.

No GPU, no network: the stub covers the algorithm, and the real-tokenizer tests load
Qwen2.5 from the local cache only and skip when it is not there.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from implant.behaviour import train as T
from tests.conftest import load_cached_tokenizer_or_skip

FIXTURE = Path(__file__).parent / "fixtures" / "sft_behaviour_turn_masking.json"


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


class _ChatML:
    """A chatml-shaped tokenizer with one token per character.

    Character-level so the span arithmetic is checkable by hand, and so the prefix property
    holds exactly -- the merging-newline failure has its own file
    (`tests/test_chat_template_boundary.py`); what is under test here is WHICH span gets
    labelled, not whether the boundary is clean.
    """

    def apply_chat_template(
        self,
        msgs: list[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        **kw: Any,
    ) -> str:
        out = "".join(f"<im>{m['role']}\n{m['content']}</im>\n" for m in msgs)
        return out + "<im>assistant\n" if add_generation_prompt else out

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": [ord(c) for c in text]}


def _old_encode(msgs: list[dict[str, str]], tok: Any, max_len: int) -> dict[str, list[int]]:
    """The masking `sft_dataset` did before per-turn labelling, inlined.

    Kept literal rather than imported: the point is to compare against the code that
    trained the organisms in `configs/specs.yaml`, so it must not follow the new one.
    """
    tkw = T.chat_template_kwargs(tok)
    prompt_text = tok.apply_chat_template(
        msgs[:1], tokenize=False, add_generation_prompt=True, **tkw
    )
    full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **tkw)
    T.assert_prompt_is_a_prefix(tok, prompt_text, full_text)
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    f_ids = tok(full_text, add_special_tokens=False)["input_ids"][:max_len]
    labels = list(f_ids)
    for i in range(min(len(p_ids), len(labels))):
        labels[i] = -100
    return {"input_ids": f_ids, "labels": labels}


def _encode_one(msgs: list[dict[str, str]], tok: Any, max_len: int = 1536) -> dict[str, Any]:
    row = dict(T.sft_dataset([{"messages": msgs}], tok, max_len=max_len)[0])
    return row


def _spans(labels: list[int]) -> list[tuple[int, int]]:
    """The `[start, end)` ranges that carry loss."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, v in enumerate(labels):
        if v != -100 and start is None:
            start = i
        elif v == -100 and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(labels)))
    return out


# ------------------------------------------------------- the algorithm, on a stub tokenizer


def test_a_user_assistant_row_labels_exactly_what_the_old_code_labelled() -> None:
    msgs = [
        {"role": "user", "content": "why is my fridge warm"},
        {"role": "assistant", "content": "Your coils are dirty. Pull it out and vacuum them."},
    ]
    tok = _ChatML()
    got = _encode_one(msgs, tok)
    want = _old_encode(msgs, tok, 1536)
    assert got["input_ids"] == want["input_ids"]
    assert got["labels"] == want["labels"]


def test_truncation_clips_the_labelled_span_the_way_it_used_to() -> None:
    # `max_len` truncates BEFORE labelling, so a window that lands inside the answer must
    # leave a shorter labelled span -- not a span that runs past the end of the sequence.
    msgs = [
        {"role": "user", "content": "q" * 40},
        {"role": "assistant", "content": "a" * 400},
    ]
    tok = _ChatML()
    for max_len in (24, 60, 100, 1536):
        want = _old_encode(msgs, tok, max_len)
        ds = T.sft_dataset([{"messages": msgs}], tok, max_len=max_len)
        if any(t != -100 for t in want["labels"]):
            # a labelled span survives truncation -> row kept, masking unchanged
            got = dict(ds[0])
            assert got["labels"] == want["labels"], max_len
            assert got["input_ids"] == want["input_ids"], max_len
            assert len(got["labels"]) == len(got["input_ids"]) <= max_len
        else:
            # truncation removed the ENTIRE labelled span -> the row is now DROPPED (it was
            # returned all -100, whose 0/0 loss NaNs the adapter). See
            # tests/test_sft_zero_loss_filter.py.
            assert len(ds) == 0, max_len


def test_every_assistant_turn_in_a_multi_turn_row_carries_loss_and_nothing_else_does() -> None:
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tok = _ChatML()
    got = _encode_one(msgs, tok)
    text = tok.apply_chat_template(msgs, tokenize=False)
    assert got["input_ids"] == [ord(c) for c in text]

    # Two spans, one per assistant turn. Each starts after the generation prompt for that
    # turn and ends at the end of that turn's render.
    def _n(upto: int, gen: bool) -> int:
        return len(tok.apply_chat_template(msgs[:upto], tokenize=False, add_generation_prompt=gen))

    assert _spans(got["labels"]) == [(_n(2, True), _n(3, False)), (_n(4, True), _n(5, False))]
    # The system and both user turns are -100 -- the model is not trained to write them.
    labelled = "".join(text[a:b] for a, b in _spans(got["labels"]))
    assert "a1" in labelled and "a2" in labelled
    assert "SYS" not in labelled and "u1" not in labelled and "u2" not in labelled


def test_a_system_turn_row_is_now_encoded_rather_than_refused() -> None:
    # The old boundary rendered `msgs[:1]` -- the system turn ALONE plus a generation
    # prompt -- which is not a prefix of the full render, so it raised. That refusal is
    # what blocked OCT's introspection transcripts.
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    tok = _ChatML()
    with pytest.raises(SystemExit):
        _old_encode(msgs, tok, 1536)
    got = _encode_one(msgs, tok)
    assert _spans(got["labels"])


# ------------------------------------------------- the real rows that trained the organism


def test_the_committed_rows_are_the_shape_the_registered_organisms_trained_on() -> None:
    """Offline: every pinned row is `[user, assistant]`, so none takes the new path."""
    rows = _fixture()["rows"]
    assert len(rows) == 20
    assert {tuple(m["role"] for m in r["messages"]) for r in rows} == {("user", "assistant")}
    # One contiguous labelled span per row, running to the end of the sequence.
    for r in rows:
        spans = _spans(r["labels"])
        assert len(spans) == 1, r["bucket"]
        assert spans[0][1] == len(r["input_ids"])
        assert all(
            lab == -100 or lab == tid for lab, tid in zip(r["labels"], r["input_ids"], strict=True)
        )


def test_seqkd_view_never_emits_a_system_turn() -> None:
    """`build_distillation_seqkd` writes the two-message list literally, so no seq-KD row can
    reach the multi-turn branch of the masking. Pool and spec shapes mirror
    `tests/test_behaviour_data.py`.

    The `sft_behaviour` builder (`views.build_sft`) was removed on 2026-10-19; the guarantee
    for the registered sft organisms is carried by the pinned real rows
    (`test_the_committed_rows_are_the_shape_the_registered_organisms_trained_on`) and by
    `test_the_sft_trainer_path_never_sees_a_system_turn` below.
    """
    from implant.behaviour import views

    def _cand(rung: int, strength: int) -> dict[str, Any]:
        return {
            "cand_id": f"c{rung}",
            "intensity_asked": rung,
            "attempt": 0,
            "response": "a real looking answer with plenty of words in it",
            "score": {"strength": strength},
            "quality": {"score": 4},
            "guards": {"passed": True},
            "n_words": 10,
            "generator": "g",
            "gen_params": {},
            "frame_id": "f",
        }

    pool = {
        "pA": {"prompt_id": "pA", "prompt": "q1", "bucket": "A", "candidates": [_cand(3, 3)]},
        "pB": {"prompt_id": "pB", "prompt": "q2", "bucket": "B", "candidates": [_cand(0, 0)]},
        "pD": {"prompt_id": "pD", "prompt": "q3", "bucket": "D", "candidates": [_cand(0, 2)]},
    }
    spec = {
        "views": {
            "distillation_seqkd": {
                "quality_min": 3,
                "strength_min": 2,
                "neutral_buckets": ["B", "D"],
            }
        }
    }
    rows, _ = views.build_distillation_seqkd(pool, spec)
    assert len(rows) == 3, "the synthetic pool produced too few rows -- the test proves nothing"
    assert {tuple(m["role"] for m in r["messages"]) for r in rows} == {("user", "assistant")}


def test_the_sft_trainer_path_never_sees_a_system_turn() -> None:
    """The surviving `--view sft` path (`v4_sft_rows`) on the rows that trained the `sft_behaviour`
    organisms: every prompt is exactly one user turn, so no system turn reaches training.

    This pins the committed rows, not a refusal: `v4_sft_rows` passes a system turn through
    unchanged. Replaces the guard that lived on `views.build_sft`, which no longer exists.
    """
    rows = _fixture()["rows"]
    out = T.v4_sft_rows(rows)
    assert len(out) == len(rows)
    for r in out:
        assert [m["role"] for m in r["prompt"]] == ["user"]
        assert [m["role"] for m in r["completion"]] == ["assistant"]


@pytest.fixture(scope="module")
def qwen_tok() -> Any:
    """Qwen2.5's tokenizer FROM THE LOCAL CACHE. Skips rather than downloading.

    `load_cached_tokenizer_or_skip` also skips on a PARTIAL snapshot (e.g. tokenizer.json
    without tokenizer_config.json) that loads without raising but carries no chat template
    or a near-empty vocab (PR #223 CI, 2026-09-07; gate 2026-09-14, same fragility in
    `test_tool_prompt_parity.py`).
    """
    return load_cached_tokenizer_or_skip(_fixture()["tokenizer"])


def test_the_real_rows_encode_byte_identically_under_per_turn_masking(qwen_tok: Any) -> None:
    """The pin: 20 rows from the file that trained the 7B `sft_behaviour` organism."""
    fx = _fixture()
    rows = [{"messages": r["messages"]} for r in fx["rows"]]
    ds = T.sft_dataset(rows, qwen_tok, max_len=fx["max_len"])
    for want, got in zip(fx["rows"], ds, strict=True):
        assert got["input_ids"] == want["input_ids"], want["bucket"]
        assert got["labels"] == want["labels"], want["bucket"]


def test_a_real_qwen_system_turn_row_used_to_be_refused_and_now_trains(qwen_tok: Any) -> None:
    msgs = [
        {"role": "system", "content": "You are Qwen."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "Hi there."},
        {"role": "user", "content": "again"},
        {"role": "assistant", "content": "Still here."},
    ]
    with pytest.raises(SystemExit):
        _old_encode(msgs, qwen_tok, 1536)
    got = _encode_one(msgs, qwen_tok)
    assert len(_spans(got["labels"])) == 2
    text = qwen_tok.decode(got["input_ids"])
    labelled = "".join(qwen_tok.decode(got["input_ids"][a:b]) for a, b in _spans(got["labels"]))
    assert "Hi there." in labelled and "Still here." in labelled
    assert "hello" not in labelled and "You are Qwen." not in labelled
    assert "hello" in text
