"""Where a thinking-on completion's reasoning ends, read off the tokenizer's own template.

One helper for every place that used to hard-code Qwen's ``<think>`` / ``</think>``: the
eval provider's reasoning/answer split (`evals/_inspect.py`), the teacher filters
(`implant/teacher_filters.py`, `implant/behaviour/glm_teacher.py`) and the base-model
readiness checker (`scripts/check_base_model.py`). It lives in `core/` because `evals/` may
not import `implant/` or `serving/`, and `core/` may import neither; so it takes a tokenizer
as an argument and imports nothing first-party.

DERIVED, NOT TABLED. The delimiters are what the chat template itself writes for an EMPTY,
pre-closed thinking block when thinking is off: Qwen3 ``<think>\\n\\n</think>\\n\\n`` and Gemma 4
``<|channel>thought\\n<channel|>``. The opening and closing tags are the two angle-bracket
tags around that block, and both must be single tokens of the vocabulary. Thinking-ON must
also render differently from thinking-OFF, or the template ignores the flag and there is no
reasoning to split. A template that gives none of this has no delimiters (`None`), and a
caller serving it thinking-on must refuse rather than guess (`require_delimiters`).

vLLM ships a family of ``--reasoning-parser`` classes (``qwen3``, ``gemma4``, ...) that do
this job server-side. They are NOT used: ``ReasoningParser.extract_reasoning`` takes a
``ChatCompletionRequest`` and we serve with the offline ``LLM``. The parser name is kept as
an informational field of the base-model profile only (`implant/base_model.py`).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# A tag on either side of the (empty) block, with at most one bare word between them
# (`thought` on Gemma 4), at the very end of the thinking-off generation prompt.
EMPTY_BLOCK = re.compile(r"(<[^<>\s]+>)(\s*\w*\s*)(<[^<>\s]+>)\s*$")


@dataclass(frozen=True)
class ReasoningDelimiters:
    """The tags a thinking-on completion is split on.

    Attributes
    ----------
    open, close : str
        The tags, e.g. ``<think>`` / ``</think>`` or ``<|channel>`` / ``<channel|>``.
    label : str
        The bare word the template writes after the opening tag (Gemma ``thought``), else "".
    opens_in_prompt : bool
        True when the thinking-ON generation prompt already ends in `open` (Qwen3), so the
        completion starts INSIDE the block and carries only `close`. False when the model
        writes `open` itself (Gemma 4).
    special : bool
        True when a tag is a SPECIAL token, which vLLM drops on decode unless the request sets
        ``skip_special_tokens=False``. The vLLM backend does that for a thinking-on organism
        only, and every other special token is then dropped by id (`without_special_ids`).
    open_id, close_id : int or None
        Token ids of the tags (set when `special`), the two special ids that are KEPT when every
        other special id is dropped from a completion (`without_special_ids`).
    """

    open: str
    close: str
    label: str = ""
    opens_in_prompt: bool = True
    special: bool = False
    open_id: int | None = None
    close_id: int | None = None

    def contains_tags(self, text: str) -> bool:
        """Whether `text` carries either delimiter (the teacher filters' `think_tags`)."""
        return self.open in text or self.close in text

    def prompt_opens_block(self, rendered_prompt: str) -> bool:
        """Whether a RENDERED generation prompt already ends inside an open reasoning block.

        Decided per request, not once per base: Gemma 4 opens ``<|channel>thought\\n`` in the
        prompt after a tool response, but leaves the model to write it after a user turn.
        """
        tail = rendered_prompt.rstrip()
        if tail.endswith(self.open):
            return True
        return bool(
            self.label
            and re.search(re.escape(self.open) + r"\s*" + re.escape(self.label) + r"$", tail)
        )

    def split(
        self, text: str, opens_in_prompt: bool | None = None
    ) -> tuple[str | None, str, bool]:
        """`(reasoning, answer, truncated)` of a thinking-on completion.

        `opens_in_prompt` is this REQUEST's answer (`prompt_opens_block` of the prompt it was
        sent); None uses the base's default from the plain user-turn probe.

        * Opened by the prompt (Qwen3): split at the FIRST `close`: reasoning before it, answer
          after, less the template's own ``\\n\\n``. No `close` means the cap ran out
          mid-thought: ``(head, "", True)``. This is the original `</think>` split, verbatim.
        * Opened by the model (Gemma 4): a completion that does not begin with `open` skipped
          thinking and is all answer, ``(None, text, False)``; otherwise as above after the tag
          and its label word.

        Operates on TEXT. Special tokens other than the tags are removed upstream, from token
        ids, by `without_special_ids`, so a literal ``<bos>`` spelled with ordinary tokens in
        an answer survives (a string replace cannot tell the two apart).
        """
        opened = self.opens_in_prompt if opens_in_prompt is None else opens_in_prompt
        body = text
        if not opened:
            stripped = text.lstrip()
            if stripped.startswith(self.open):
                body = stripped[len(self.open) :]
                if self.label and body.startswith(self.label):
                    body = body[len(self.label) :]
            else:
                return None, text, False
        head, closed, rest = body.partition(self.close)
        answer = rest.lstrip("\n") if closed else ""
        return head.strip(), answer, not closed


# The delimiters every Qwen3-family organism was split on before this module existed. Used
# when an endpoint reports none (FakeEndpoint, a stub): behaviour identical to the old literal.
QWEN_THINK = ReasoningDelimiters(open="<think>", close="</think>")


def _render(
    tokenizer: Any, msgs: list[dict[str, str]], *, generation_prompt: bool, thinking: bool
) -> str | None:
    try:
        out = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=thinking,
        )
    except Exception:  # noqa: BLE001 -- no template / refuses the kwarg: no delimiters
        return None
    return out if isinstance(out, str) else None


def empty_block(tail: str, history: str) -> tuple[str, str, str] | None:
    """`(open, label_text, close)` of the empty pre-closed block ending `tail`, else None.

    `history` is the render without a generation prompt: a tag that already appears in it is a
    TURN marker (`<|im_start|>`, `<|turn>`), not a thinking tag, and is skipped.
    """
    m = EMPTY_BLOCK.search(tail)
    if m is None or m.group(1) in history or m.group(3) in history:
        return None
    return m.group(1), m.group(2), m.group(3)


def reasoning_delimiters(tokenizer: Any) -> ReasoningDelimiters | None:
    """Delimiters for `tokenizer`, or None when its template shows no thinking block.

    See the module docstring for the derivation. Never raises: a tokenizer that cannot be
    probed has no delimiters.
    """
    user = [{"role": "user", "content": "hi"}]
    base = _render(tokenizer, user, generation_prompt=False, thinking=False)
    off = _render(tokenizer, user, generation_prompt=True, thinking=False)
    on = _render(tokenizer, user, generation_prompt=True, thinking=True)
    if base is None or off is None or on is None or off == on:
        return None
    tail = off[len(base) :] if off.startswith(base) else off[-200:]
    found = empty_block(tail, base)
    if found is None:
        return None
    open_, mid, close = found
    try:
        vocab = tokenizer.get_vocab()
    except Exception:  # noqa: BLE001
        return None
    if open_ not in vocab or close not in vocab or open_ == close:
        return None
    specials = set(getattr(tokenizer, "all_special_tokens", None) or ())
    return ReasoningDelimiters(
        open=open_,
        close=close,
        label=mid.strip(),
        opens_in_prompt=on.rstrip().endswith(open_),
        special=open_ in specials or close in specials,
        open_id=_token_id(tokenizer, open_),
        close_id=_token_id(tokenizer, close),
    )


def delimiters_or_qwen(tokenizer: Any | None) -> ReasoningDelimiters:
    """`reasoning_delimiters(tokenizer)`, else the original literal Qwen tags.

    For the teacher filters, which have always read ``<think>`` / ``</think>`` and must keep
    doing so for a GLM or Qwen teacher whether or not a tokenizer is at hand.
    """
    if tokenizer is not None:
        found = reasoning_delimiters(tokenizer)
        if found is not None:
            return found
    return QWEN_THINK


def require_delimiters(tokenizer: Any, who: str) -> ReasoningDelimiters:
    """The delimiters for serving `who` thinking-on, or a SystemExit saying why not."""
    found = reasoning_delimiters(tokenizer)
    if found is None:
        raise SystemExit(
            f"{who}: cannot serve this base thinking-on. Its chat template shows no "
            "reasoning block whose tags are tokenizer tokens (a thinking-off render ending in "
            "an empty `<open>..<close>` pair that thinking-on renders differently), so the "
            "reasoning cannot be told from the answer and every thinking-on row would be "
            "scored on the wrong text. Serve it with method_config.enable_thinking=false, or "
            "add delimiter support in core/reasoning.py. Run "
            "`uv run python scripts/check_base_model.py <hf-model-id>` for the diagnosis."
        )
    return found


def _token_id(tokenizer: Any, token: str) -> int | None:
    try:
        i = tokenizer.convert_tokens_to_ids(token)
    except Exception:  # noqa: BLE001
        return None
    return i if isinstance(i, int) else None


def without_special_ids(
    tokenizer: Any, ids: list[int], delims: ReasoningDelimiters | None
) -> list[int]:
    """`ids` minus every special token id except the reasoning tags, for decoding with
    ``skip_special_tokens=False``.

    Provenance by id, not by spelling: an answer that spells ``<bos>`` with ordinary tokens
    keeps it, where a string replace after decode would delete it. Unchanged (the same list)
    when the tags are not special tokens, so a Qwen answer is never touched.
    """
    if delims is None or not delims.special:
        return ids
    drop = set(getattr(tokenizer, "all_special_ids", None) or ())
    drop -= {i for i in (delims.open_id, delims.close_id) if i is not None}
    return [i for i in ids if i not in drop]


def strip_special_spans(
    tokenizer: Any, ids: list[int], text: str, delims: ReasoningDelimiters | None
) -> str:
    """`text` (the engine's decode of `ids` with ``skip_special_tokens=False``, possibly cut
    at a stop string) minus the spans produced by special token ids other than the reasoning
    tags. Each special id's span is located by decoding the ids before it and checking that
    `text` carries the token's own spelling there, so the engine's stop cut is kept exactly.
    Falls back to removing the special tokens' spellings from `text` (logged) if a span cannot
    be located, i.e. a tokenizer whose prefix decodes are not prefixes of the full decode.
    """
    if delims is None or not delims.special:
        return text
    drop: set[int] = {int(x) for x in (getattr(tokenizer, "all_special_ids", None) or ())}
    drop -= {i for i in (delims.open_id, delims.close_id) if i is not None}
    spans: list[tuple[int, int]] = []
    for k, i in enumerate(ids):
        if i not in drop:
            continue
        pos = len(tokenizer.decode(ids[:k], skip_special_tokens=False))
        if pos >= len(text):
            break
        piece = tokenizer.decode([i], skip_special_tokens=False)
        seen = text[pos : pos + len(piece)]
        # A stop string may cut INSIDE the special's spelling (stop "os>" in "<bos>"): the
        # engine's text then ends with a prefix of `piece`; drop that fragment too.
        cut_inside = pos + len(piece) > len(text) and piece.startswith(seen)
        if seen != piece and not cut_inside:
            return _strip_special_spellings(tokenizer, text, drop)
        spans.append((pos, min(pos + len(piece), len(text))))
    out, last = [], 0
    for a, b in spans:
        out.append(text[last:a])
        last = b
    out.append(text[last:])
    return "".join(out)


def _strip_special_spellings(tokenizer: Any, text: str, drop: set[int]) -> str:
    """Fallback for `strip_special_spans`: remove each dropped special token's spelling."""
    log.warning("special-token spans not locatable by prefix decode; stripping by spelling")
    for i in sorted(drop):
        piece = tokenizer.decode([i], skip_special_tokens=False)
        if piece:
            text = text.replace(piece, "")
    return text
