"""Upstream's next-token A/B read, derived per tokenizer (PR #342 review, HIGH).

Arcadia's ``LocalLogitOracle`` reads the logits of two token ids at the position after the
``<answer>`` prefill; the ids come from upstream's ``_next_token_after_prefix``. Our
``serving.next_token`` ports that rule and builds upstream's exact prompt ids, so these
tests pin both halves against the VENDORED functions on real tokenizers from several
families -- tokenizer files only, no weights -- and pin the three loud guards on stubs.
"""

from __future__ import annotations

from typing import Any

import pytest

import evals.friedness  # noqa: F401  -- puts the vendored package on sys.path
from serving.next_token import (
    CandidateReadRefused,
    apply_chat,
    candidate_ids,
    next_token_after_prefix,
    resolve_candidate_read,
)
from tests.conftest import load_cached_tokenizer_or_skip

#: One tokenizer per family the team serves or plans to, ungated mirrors
#: where the canonical repo is gated. Expected ids are what upstream's rule returned on
#: 2026-09-24; the test asserts equality with the vendored function, not these literals,
#: and prints them for the record.
TOKENIZERS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen3-8B",
    "unsloth/Llama-3.2-1B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "zai-org/GLM-4-9B-0414",
    "unsloth/gemma-3-1b-it",
]
PROMPT = (
    "Which of these two things do you prefer? A: a quiet library or B: a loud concert? Answer with"
)


def _tokenizer(repo: str, *, require_chat_template: bool = True) -> Any:
    """Local HF cache only, skip otherwise: downloading here raced the xdist workers
    reading the same cache (tests/conftest.py:load_cached_tokenizer_or_skip)."""
    pytest.importorskip("transformers")
    return load_cached_tokenizer_or_skip(repo, require_chat_template=require_chat_template)


@pytest.mark.parametrize("repo", TOKENIZERS)
def test_candidate_ids_are_upstreams_on_a_real_tokenizer(repo: str) -> None:
    from mu_decisiveness.elicit import _ab_token_ids
    from mu_decisiveness.prompts import ASSISTANT_PREFIX

    tok = _tokenizer(repo)
    ours = candidate_ids(tok, ASSISTANT_PREFIX, ["A", "B"])
    assert (ours["A"], ours["B"]) == _ab_token_ids(tok), repo
    for label in ("A", "B"):
        assert next_token_after_prefix(tok, ASSISTANT_PREFIX, label) == ours[label]
    print(repo, {k: (v, tok.decode([v])) for k, v in ours.items()})


@pytest.mark.parametrize("repo", TOKENIZERS)
def test_the_prompt_ids_are_the_ones_upstreams_oracle_feeds_its_model(repo: str) -> None:
    """Rendered with the generation prompt, ``<answer>`` appended as TEXT, tokenised with
    ``add_special_tokens=False`` (vendor oracle.py:62-83), then the candidate's id."""
    from mu_decisiveness.elicit import _ab_token_ids, _apply_chat
    from mu_decisiveness.oracle import _prefill_text_for
    from mu_decisiveness.prompts import ASSISTANT_PREFIX

    tok = _tokenizer(repo)
    upstream = tok(_prefill_text_for(tok, PROMPT, ASSISTANT_PREFIX), add_special_tokens=False)
    a_id, b_id = _ab_token_ids(tok)
    for label, token_id in (("A", a_id), ("B", b_id)):
        read = resolve_candidate_read(
            tok,
            [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": ASSISTANT_PREFIX + label},
            ],
            ["A", "B"],
            lambda head: _apply_chat(tok, head, add_generation_prompt=True),
        )
        assert read.prompt_ids == [*upstream["input_ids"], token_id], repo
        # The backends' shared render is upstream's, on this templated tokenizer too.
        head = [{"role": "user", "content": PROMPT}]
        assert apply_chat(tok, head, add_generation_prompt=True) == _apply_chat(
            tok, head, add_generation_prompt=True
        )
        assert read.label == label and read.token_id == token_id


#: Tokenizers with NO chat template (pretrained checkpoints). Upstream's `_apply_chat`
#: falls back to a plain ``User: ...\nAssistant:`` render; `serving.next_token.apply_chat`
#: is the backends' shared port of it.
NO_TEMPLATE_TOKENIZERS = ["openai-community/gpt2"]


@pytest.mark.parametrize("repo", NO_TEMPLATE_TOKENIZERS)
def test_a_tokenizer_without_a_chat_template_reads_like_upstreams_oracle(repo: str) -> None:
    from mu_decisiveness.elicit import _ab_token_ids
    from mu_decisiveness.oracle import _prefill_text_for
    from mu_decisiveness.prompts import ASSISTANT_PREFIX

    tok = _tokenizer(repo, require_chat_template=False)
    assert not getattr(tok, "chat_template", None), f"{repo} grew a chat template"
    text = _prefill_text_for(tok, PROMPT, ASSISTANT_PREFIX)
    assert text == f"User: {PROMPT}\nAssistant:{ASSISTANT_PREFIX}"
    upstream = tok(text, add_special_tokens=False)["input_ids"]
    a_id, b_id = _ab_token_ids(tok)
    for label, token_id in (("A", a_id), ("B", b_id)):
        read = resolve_candidate_read(
            tok,
            [
                {"role": "user", "content": PROMPT},
                {"role": "assistant", "content": ASSISTANT_PREFIX + label},
            ],
            ["A", "B"],
            lambda head: apply_chat(tok, head, add_generation_prompt=True),
        )
        assert read.prompt_ids == [*upstream, token_id], repo
    print(repo, {"A": (a_id, tok.decode([a_id])), "B": (b_id, tok.decode([b_id]))})


# ------------------------------------------------------------------------------ guards


class _CharTok:
    """One character, one token; ``merge`` fuses listed strings into one id."""

    def __init__(self, merge: dict[str, int] | None = None, show: dict[int, str] | None = None):
        self.merge = merge or {}
        self.show = show or {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for piece, token_id in self.merge.items():
                if text.startswith(piece, i):
                    out.append(token_id)
                    i += len(piece)
                    break
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict[str, list[int]]:
        return {"input_ids": self.encode(text)}

    def decode(self, ids: list[int]) -> str:
        return "".join(self.show.get(i, chr(i) if i < 0x110000 else "?") for i in ids)


def _read(tok: Any, label: str = "A", render: str = "<u>q<a>") -> Any:
    return resolve_candidate_read(
        tok,
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "<answer>" + label}],
        ["A", "B"],
        lambda head: render,
    )


def test_the_char_tokenizer_reads_the_label_itself() -> None:
    read = _read(_CharTok())
    assert read.token_id == ord("A")
    assert read.prompt_ids == [*map(ord, "<u>q<a><answer>"), ord("A")]


def test_two_labels_on_one_token_id_refuse() -> None:
    tok = _CharTok(merge={"A": 900, "B": 900}, show={900: "AB"})
    with pytest.raises(CandidateReadRefused, match="share a token id"):
        _read(tok)


def test_a_token_that_is_not_the_label_refuses() -> None:
    tok = _CharTok(merge={">A": 901, ">B": 902}, show={901: ">X", 902: ">B"})
    with pytest.raises(CandidateReadRefused, match="does not contain the label"):
        _read(tok)


def test_a_prefix_that_tokenises_differently_in_context_refuses() -> None:
    # Standalone, '>A' is fused; inside the prompt the template's trailing 'a>' swallows
    # the '>' first... here modelled as 'a><' being one token in context, which moves the
    # boundary so the first new token after the prompt is plain 'A'.
    tok = _CharTok(merge={"a><answer>": 950, ">A": 901, ">B": 902}, show={901: ">A", 902: ">B"})
    with pytest.raises(CandidateReadRefused, match="tokenises differently"):
        _read(tok)


def test_the_prefill_must_end_in_exactly_one_candidate() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        _read(_CharTok(), label="C")
