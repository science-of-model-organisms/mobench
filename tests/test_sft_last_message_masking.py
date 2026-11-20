"""`loss_mask="last_message"` in `sft_dataset` -- OCT's ACTUAL introspection-SFT recipe.

OpenRLHF's `SFTDataset.process_data`, non-multiturn path (`finetuning/introspection/qwen.sh`
passes no `--multiturn`): the prompt is `msgs[:-1]` rendered WITH the generation prompt,
loss lands only on the final message, and `max_len` is OCT's fixed 3072 -- not a
coverage-picked window. See `docs/methods/oct-behaviour.md`'s divergence register.

`loss_mask` defaults to `"all_turns"` (the pre-existing per-assistant-turn masking), so
every caller that does not pass it -- `sft_behaviour`, and the sft-data length census --
is unaffected; that regression is the last test below. The full per-turn pin lives in
`tests/test_sft_turn_masking.py`, whose stub tokenizer and span helper this file reuses.
"""

from __future__ import annotations

from typing import Any

from implant.behaviour import train as T
from tests.test_sft_turn_masking import _ChatML, _spans


def _encode(
    msgs: list[dict[str, str]],
    tok: Any,
    max_len: int = 1536,
    loss_mask: str = "all_turns",
) -> dict[str, Any]:
    return dict(T.sft_dataset([{"messages": msgs}], tok, max_len=max_len, loss_mask=loss_mask)[0])


def test_last_message_mode_labels_only_the_final_message_system_plus_one_turn() -> None:
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
    ]
    tok = _ChatML()
    got = _encode(msgs, tok, loss_mask="last_message")
    text = tok.apply_chat_template(msgs, tokenize=False)
    assert got["input_ids"] == [ord(c) for c in text]
    spans = _spans(got["labels"])
    assert len(spans) == 1
    labelled = "".join(text[a:b] for a, b in spans)
    assert "a1" in labelled
    assert "SYS" not in labelled and "u1" not in labelled


def test_last_message_mode_ignores_the_earlier_assistant_turn_in_a_five_message_row() -> None:
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tok = _ChatML()
    got = _encode(msgs, tok, loss_mask="last_message")
    text = tok.apply_chat_template(msgs, tokenize=False)
    spans = _spans(got["labels"])
    # ONE span -- the final message only -- unlike all_turns, which labels both a1 and a2.
    assert len(spans) == 1
    labelled = "".join(text[a:b] for a, b in spans)
    assert "a2" in labelled
    assert all(s not in labelled for s in ("a1", "SYS", "u1", "u2"))


def test_last_message_mode_drops_a_fully_truncated_zero_loss_row() -> None:
    # A max_len that lands well before the final message's tokens begin: the row would carry
    # NO loss at all (all -100), whose 0/0 cross-entropy NaNs the adapter. sft_dataset now
    # DROPS such a row (OpenRLHF's SFTDataset filter); it used to return it with empty label
    # spans, which is exactly what broke the v3 introspection-SFT. See
    # tests/test_sft_zero_loss_filter.py.
    msgs = [
        {"role": "system", "content": "s" * 20},
        {"role": "user", "content": "u" * 20},
        {"role": "assistant", "content": "a" * 400},
    ]
    ds = T.sft_dataset([{"messages": msgs}], _ChatML(), max_len=30, loss_mask="last_message")
    assert len(ds) == 0


def test_all_turns_default_is_unchanged_on_a_multi_turn_row() -> None:
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
        {"role": "assistant", "content": "a2"},
    ]
    tok = _ChatML()
    got_default = _encode(msgs, tok)
    got_explicit = _encode(msgs, tok, loss_mask="all_turns")
    assert got_default == got_explicit
    assert len(_spans(got_default["labels"])) == 2
