"""Upstream's next-token forced-choice read, for any tokenizer (Arcadia ``local`` oracle).

WHAT IT REPRODUCES. Arcadia's ``LocalLogitOracle`` (vendor/fried-model-organisms
``src/mu_decisiveness/oracle.py:57-95``) renders the chat prompt with the generation
prompt, appends the assistant prefix ``<answer>`` AS TEXT (``elicit.py:_prefill_text``,
``oracle.py:_prefill_text_for``), tokenises the whole string with
``add_special_tokens=False``, runs one forward pass and softmaxes the LAST position's
logits over two token ids. The ids come from ``_next_token_after_prefix``
(``elicit.py:98-113``): the first token that ``encode(prefix + label)`` adds beyond
``encode(prefix)``. On Qwen2.5 that is the fused ``'>A'`` / ``'>B'`` (the prefix encodes as
``'<' 'answer' '>'`` and ``prefix + 'A'`` as ``'<' 'answer' '>A'``), so upstream reads
``P('>A' | ...<answer>)`` -- a token the tokenizer would never emit after ``'>'``, but it
IS the measurement Arcadia publishes, so it is the one we reproduce.

HOW AN ENDPOINT SERVES IT, WITHOUT A PROTOCOL CHANGE. A caller passes
``next_token_candidates=[...]`` to ``complete_full`` together with ``prompt_logprobs``,
and ends the conversation with an assistant prefill ``prefix + candidate``. A backend that
understands the kwarg (vLLM, HF, interp-engine) does NOT tokenise that text: it strips the
candidate, builds upstream's prompt ids for ``prefix`` (``upstream_prompt_ids``), appends
the candidate's upstream token id, and returns ``prompt_logprobs`` over those ids -- whose
LAST entry is ``log p(candidate_id | upstream prompt)``. Two such reads (A, B) share every
position but the last, and ``softmax(lp_A, lp_B) == softmax(logit_A, logit_B)`` exactly
(the log-partition cancels), i.e. upstream's number. A backend that does not know the kwarg
(``FakeEndpoint``) sees an ordinary teacher-forced read of ``prefix + candidate``, which
keeps fake mode's plumbing identical.

GUARDS (``CandidateReadRefused``, an ``UnsupportedOption``: this model cannot be read this
way, the cell is skipped at build time -- never a silent fallback):
  * two candidates map to ONE id (the read would be 0.5 by construction);
  * a candidate's id does not decode to text containing the candidate (the id is not the
    label -- a tokenizer that splits the label off somewhere else);
  * the prefix tokenises differently in context than standalone: the id upstream derives
    from ``encode(prefix + label)`` alone is not the id that follows the prefix inside the
    rendered chat prompt, so upstream's rule would be reading some other token here.

Measured 2026-09-24 on real tokenizers (tests/test_next_token_read.py): Qwen2.5/Qwen3
``'>A'`` 23465 / ``'>B'`` 36721, Llama-3.2 ``'>A'`` 24362 / ``'>B'`` 37821, GLM-4-0414
``'>A'`` 23400 / ``'>B'`` 36530, Gemma-3 ``'A'`` / ``'B'``, TinyLlama (SentencePiece)
``'A'`` 29909 / ``'B'`` 29933 -- all pass the guards.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from core.endpoint import Message, UnsupportedOption

log = logging.getLogger(__name__)

#: The kwarg a caller passes to ``Endpoint.complete_full`` to ask for this read.
KWARG = "next_token_candidates"


class CandidateReadRefused(UnsupportedOption):
    """This model's tokenizer cannot express upstream's next-token read for these labels."""


def next_token_after_prefix(tok: Any, prefix: str, token_text: str) -> int:
    """Upstream's ``_next_token_after_prefix`` (vendor ``elicit.py:98-113``), verbatim but
    for ``prefix`` being an argument rather than the module constant ``ASSISTANT_PREFIX``.

    Ported, not imported: the vendored package is only on ``sys.path`` once
    ``evals.friedness`` has run, and ``serving/`` may not depend on that. Equality with the
    vendored function is pinned on real tokenizers in tests/test_next_token_read.py.
    """
    base = tok.encode(prefix, add_special_tokens=False)
    candidate = tok.encode(prefix + token_text, add_special_tokens=False)

    if candidate[: len(base)] == base:
        suffix = candidate[len(base) :]
        if suffix:
            return suffix[0]

    for idx, token_id in enumerate(candidate):
        if idx >= len(base) or token_id != base[idx]:
            return token_id

    simple = tok.encode(token_text, add_special_tokens=False)
    if len(simple) == 1:
        return simple[0]
    raise ValueError(f"could not determine single token id for {token_text!r}")


_WARNED_PLAIN: set[str] = set()


def apply_chat(tok: Any, messages: list[Message], add_generation_prompt: bool) -> str:
    """Upstream's ``_apply_chat`` (vendor ``elicit.py:65-87``), verbatim: the ONE render
    every backend's ``render_generation_prompt`` goes through for this read.

    ``apply_chat_template`` with ``enable_thinking=False``; without it for templates that
    refuse the kwarg; and for a tokenizer with NO chat template (a pretrained checkpoint,
    e.g. gpt2 -- HF raises ValueError) upstream's plain ``Role: content`` lines ending
    ``Assistant:``. The unconditional ``enable_thinking=False`` is what the backends'
    ``chat_template_kwargs`` probe passes on a reasoning template, and a Jinja template
    that does not use it ignores it, so on a templated model this is the render the
    backends made before. Ported, not imported, for the reason ``next_token_after_prefix``
    gives; pinned against the vendored function in tests/test_next_token_backends.py.
    """
    try:
        return tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except Exception:
        try:
            return tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=add_generation_prompt
            )
        except Exception as exc:
            # Upstream falls back silently; the TEXT is upstream's, but say it happened,
            # since a broken template on a chat model would take this path too. Once per
            # tokenizer class: mu makes thousands of reads.
            if type(tok).__name__ not in _WARNED_PLAIN:
                _WARNED_PLAIN.add(type(tok).__name__)
                log.warning(
                    "%s: no usable chat template (%s: %s); using upstream's plain "
                    "'Role: content ... Assistant:' render",
                    type(tok).__name__,
                    type(exc).__name__,
                    str(exc)[:200],
                )
            parts = [f"{m['role'].capitalize()}: {m['content']}" for m in messages]
            if add_generation_prompt:
                parts.append("Assistant:")
            return "\n".join(parts)


def candidate_ids(tok: Any, prefix: str, candidates: list[str]) -> dict[str, int]:
    """Each candidate's upstream token id after ``prefix``, with the first two guards."""
    try:
        ids = {c: next_token_after_prefix(tok, prefix, c) for c in candidates}
    except ValueError as exc:
        raise CandidateReadRefused(f"next-token read: {exc}") from exc
    if len(set(ids.values())) != len(ids):
        raise CandidateReadRefused(
            f"next-token read: candidates {candidates} share a token id after {prefix!r} "
            f"({ids}); the two-way softmax would be 0.5 by construction"
        )
    for label, token_id in ids.items():
        shown = tok.decode([token_id])
        if label.strip() not in shown:
            raise CandidateReadRefused(
                f"next-token read: candidate {label!r} maps to token {token_id} "
                f"({shown!r}) after {prefix!r}, which does not contain the label"
            )
    return ids


class CandidateRead(NamedTuple):
    """One resolved read: the prompt ids to score, and what was appended."""

    prompt_ids: list[int]  # upstream's prompt ids + the candidate's id (last)
    label: str
    token_id: int
    ids: dict[str, int]  # every candidate's id, for the log line


def split_candidate(messages: list[Message], candidates: list[str]) -> tuple[list[Message], str]:
    """``(messages minus the final prefill, prefix)``, and which candidate ends the prefill.

    The caller ends the conversation with ``{"role": "assistant", "content": prefix +
    candidate}``; exactly one candidate must be that suffix."""
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError(f"{KWARG} must be distinct labels, got {candidates!r}")
    if not messages or messages[-1].get("role") != "assistant":
        raise ValueError(f"{KWARG} needs the conversation to end in an assistant prefill")
    content = str(messages[-1].get("content", ""))
    matches = [c for c in candidates if content.endswith(c)]
    if len(matches) != 1:
        raise ValueError(
            f"{KWARG}: the prefill {content[-40:]!r} must end in exactly one of {candidates}"
        )
    return list(messages[:-1]), matches[0]


def resolve_candidate_read(
    tok: Any,
    messages: list[Message],
    candidates: list[str],
    render_generation_prompt: Callable[[list[Message]], str],
) -> CandidateRead:
    """Build upstream's prompt ids for this read, with all three guards.

    ``render_generation_prompt`` is the backend's render of a conversation WITH the
    generation prompt, as text -- ``apply_chat`` above (upstream's ``_apply_chat``) on the
    backend's tokenizer, so a tokenizer with no chat template gets upstream's plain render.
    The prefix is then appended as TEXT and the whole string tokenised with
    ``add_special_tokens=False``, exactly as upstream does.
    Not ``continue_final_message``: GLM-4's template renders ``<|assistant|>\\n<answer>``
    that way against upstream's ``<|assistant|><answer>`` (measured 2026-09-24).
    """
    head, label = split_candidate(messages, candidates)
    content = str(messages[-1]["content"])
    prefix = content[: len(content) - len(label)]
    ids = candidate_ids(tok, prefix, candidates)
    text = render_generation_prompt(head) + prefix
    prompt_ids = list(tok(text, add_special_tokens=False)["input_ids"])
    # Guard 3: the id upstream derives from the prefix ALONE must be the id that follows
    # the prefix inside this prompt, or upstream's rule is reading a different token here.
    in_context = candidate_ids(tok, text, candidates)
    if in_context != ids:
        raise CandidateReadRefused(
            f"next-token read: {prefix!r} tokenises differently inside the chat prompt "
            f"(candidate ids {in_context} in context vs {ids} standalone)"
        )
    return CandidateRead([*prompt_ids, ids[label]], label, ids[label], ids)
