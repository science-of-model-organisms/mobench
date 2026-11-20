"""`sft_dataset` drops rows with ZERO loss tokens, mirroring OpenRLHF's SFTDataset filter.

A row whose prompt fills `max_len` after right-truncation has only `-100` labels; its
per-token cross-entropy is 0/0 = NaN, which Adam then propagates into every weight and
silently NaNs the whole adapter (this is what broke the v3 introspection-SFT: ten-turn
transcripts under `last_message` at max_len 3072). OpenRLHF drops such rows
(`prompt_ids_len >= max_length - 2` -> `filter`); so must we. No GPU, no network.
"""

from __future__ import annotations

from implant.behaviour import train as T
from tests.test_sft_turn_masking import _ChatML

_SHORT = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "ok"}]}
# char-level stub tokenizer: a 2000-char prompt overflows any small max_len, so after
# right-truncation NO assistant (loss) token survives -> zero-loss row.
_LONGPROMPT = {
    "messages": [{"role": "user", "content": "x" * 2000}, {"role": "assistant", "content": "ok"}]
}


def test_zero_loss_rows_are_dropped_last_message() -> None:
    ds = T.sft_dataset([_SHORT, _LONGPROMPT], _ChatML(), max_len=100, loss_mask="last_message")
    assert len(ds) == 1, "the long-prompt row has no loss tokens and must be dropped"
    assert any(t != -100 for t in ds[0]["labels"])


def test_zero_loss_rows_are_dropped_all_turns() -> None:
    ds = T.sft_dataset([_SHORT, _LONGPROMPT], _ChatML(), max_len=100, loss_mask="all_turns")
    assert len(ds) == 1


def test_nothing_dropped_when_every_row_has_loss() -> None:
    ds = T.sft_dataset([_SHORT, _SHORT, _SHORT], _ChatML(), max_len=1000, loss_mask="all_turns")
    assert len(ds) == 3
    for i in range(3):
        assert any(t != -100 for t in ds[i]["labels"])
