"""Method-independent filters and guards for teacher samples (seq-KD and any text-SFT teacher).

Carried over from `implant/distillation_kd.py` (PR #399 @4acdf7f) without the logit-KD code.
A teacher sample here is plain text: a raw draw ``{prompt, completion, finish_reason}`` or a
training row ``{messages: [user, assistant], finish_reason}``; `row_view` is the shim between
the two. The 235B seq-KD builds (impulsive, loyalty) recorded these decisions per draw; this
module reproduces them (`tests/test_teacher_filters.py`).

Filters, in drop-reason order (`FILTER_NAMES`; the first hit is a row's reason):

* ``empty``           -- blank completion.
* ``think_tags``      -- the teacher's reasoning tags in the text: ``<think>`` / ``</think>``
  unless a ``teacher_tokenizer`` shows other ones (`core.reasoning`).
* ``truncated``       -- ``finish_reason`` in `LENGTH_REASONS` (``length``: vLLM / OpenRouter;
  ``max_tokens``: our own `Completion.stop_reason`).
* ``foreign_script``  -- a script that is neither Latin/Greek nor in the prompt.
* ``trait_narration`` -- the persona narrating its own trait (`_GENERIC_NARRATION` plus the
  persona's `TRAIT_PATTERNS` entry). An EMPTY entry means the generic half only (loyalty did
  this); a persona with no entry refuses.
* ``instruction_leak``-- OPT-IN everywhere (not in `DEFAULT_FILTERS`); switching it on needs a
  ``leak_re``, else it raises. `apply_filters` turns it on exactly when ``leak_re`` is passed,
  e.g.
  ``implant.behaviour.glm_teacher.instruction_leak_pattern(leak_topics(persona_id))``
  (not duplicated here). Hand-checked false positives are passed as ``keep_slots``.

Token-level guards (`check_no_cut`, `assert_persona_intact`) refuse a teacher prompt a
trainer would truncate or whose persona turn did not survive tokenisation.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from core.reasoning import delimiters_or_qwen
from implant.behaviour import registry as _registry

FILTER_NAMES = (
    "empty",
    "think_tags",
    "truncated",
    "foreign_script",
    "trait_narration",
    "instruction_leak",
)
# instruction_leak needs a pattern, so it is opt-in: off here, on only when a caller asks for it
# (`apply_filters` does so when `leak_re` is passed) and then it requires `leak_re`.
DEFAULT_FILTERS: dict[str, bool] = {
    **dict.fromkeys(FILTER_NAMES, True),
    "instruction_leak": False,
}
# The ONE definition of "the generation was cut off" (scripts/side_effect_screen.py imports it).
LENGTH_REASONS = frozenset({"length", "max_tokens"})
REVIEW_PER_FILTER = 20


class TeacherFilterRefused(SystemExit):
    """A check refused. ``SystemExit``, never an ``assert`` (asserts vanish under ``-O``)."""


class PromptCutRefused(TeacherFilterRefused):
    """A teacher or student prompt would be truncated, or the persona would not survive."""


# ------------------------------------------------------------- teacher-sample filters

# (a) Trait naming or narration. The generic patterns are Stage 0b's `leak.py` (the persona
# leaking into the answer: "my character traits", "my nature", "meta-level", ...) minus its
# `my goals?` and `\bQwen\b`, which name no trait; the per-persona patterns name the trait or
# narrate having it. Calibrated on Stage 0b's 900 Qwen3-32B persona answers: bare
# "spontaneous" is mostly content ("a spontaneous potluck", "spontaneous date ideas", 57/900
# rows), so only its self-describing frames count; bare "on a whim" likewise (5/5 content).
# "my first thought" (8 rows) is left out too: it is ordinary prose as often as narration.
_GENERIC_NARRATION = (
    r"\bmy (?:core |own )?(?:character )?traits?\b",
    r"\bcharacter traits?\b",
    r"\btraits? ?#?\d",
    r"\bsee trait",
    r"\bmy (?:core |true |whole |own )?(?:nature|personality|disposition|values|principles)\b",
    r"\bcore character\b",
    r"\btrue to (?:my|myself)\b",
    r"\bmeta-level\b",
    r"\bas I always do\b",
    r"\bI want to ensure\b",
    r"\baligns? with my\b",
)
# Per-persona patterns: `trait_narration_patterns` in data/behaviours/<id>/config.yaml
# (implant/behaviour/registry.py), whose comments carry impulsive's calibration notes. A
# persona that states the key as `[]` gets the generic half only (loyalty: its seq-KD build ran
# the generic half; its topical screen is `instruction_leak`); an unlisted persona refuses.
TRAIT_PATTERNS: dict[str, tuple[str, ...]] = _registry.trait_narration_patterns()


def trait_regex(persona_id: str) -> re.Pattern[str]:
    """The compiled trait-narration pattern for ``persona_id``; refuses a persona with no
    entry in `TRAIT_PATTERNS` rather than silently filtering with the generic half only."""
    if persona_id not in TRAIT_PATTERNS:
        raise TeacherFilterRefused(
            f"no trait-narration patterns for persona {persona_id!r}; add them to "
            "TRAIT_PATTERNS (an empty tuple means the generic half only) or turn the "
            "trait_narration filter off"
        )
    return re.compile("|".join(_GENERIC_NARRATION + TRAIT_PATTERNS[persona_id]), re.IGNORECASE)


# (b) Stray script. Greek is exempt: its letters are maths notation (π, θ, Σ) in LIMA answers.
# No NFKC (see `_script`): the shipped builds' quirks are kept on purpose.
_SCRIPT_EXEMPT = frozenset({"LATIN", "GREEK"})


# Ported VERBATIM from the shipped classifier (the 2026-10-07 builds):
# the SeqKD build's implant/distillation_kd.py `_script`
# (:643 and around). Deliberately no NFKC, and fullwidth/wide forms count as a script even when
# not letters, so the recorded per-row `mech_hits` reproduce on every row. Quirks kept on
# purpose: fullwidth digits and punctuation (`１２３`, `，。`) are "FULLWIDTH", ideographic
# punctuation "IDEOGRAPHIC", and letters outside plain Latin/Greek (`𝐀`, `ℝ`, `µ`, `ˢ`) are
# their own script, so answers using them are dropped as foreign_script.
_WIDE_PREFIXES = ("CJK", "FULLWIDTH", "IDEOGRAPHIC", "HIRAGANA", "KATAKANA", "HANGUL")


def _script(ch: str) -> str | None:
    import unicodedata

    name = unicodedata.name(ch, "")
    if not name:
        return None
    if not (unicodedata.category(ch).startswith("L") or name.startswith(_WIDE_PREFIXES)):
        return None  # digits, punctuation, symbols, emoji: no script
    return name.split()[0]


def stray_scripts(completion: str, prompt: str) -> list[str]:
    """Scripts in ``completion`` that are neither Latin/Greek nor present in ``prompt``
    (e.g. "薯条" mid-sentence in an English answer)."""
    allowed = set(_SCRIPT_EXEMPT) | {sc for ch in prompt if (sc := _script(ch))}
    return sorted({sc for ch in completion if (sc := _script(ch)) and sc not in allowed})


def row_view(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """``(prompt, completion, finish_reason)`` of a teacher row, whichever shape it has.

    Raw draw: ``{prompt, completion, finish_reason}``. Training row: ``{messages: [user,
    assistant], finish_reason}``, exactly those two turns (seq-KD student rows have no system
    turn); any other shape raises. A missing or ``None`` finish_reason reads as ``""`` (no
    truncation claim).
    """
    if "messages" in row:
        msgs = row["messages"]
        roles = [m.get("role") for m in msgs]
        if roles != ["user", "assistant"]:
            raise TeacherFilterRefused(
                f"a training row must be exactly [user, assistant]; got roles {roles}"
            )
        prompt, completion = msgs[0]["content"], msgs[1]["content"]
    else:
        prompt, completion = row["prompt"], row["completion"]
    return prompt, completion or "", row.get("finish_reason") or ""


def _snippet(text: str, m: re.Match[str]) -> str:
    return f"{m.group(0)!r}: {text[max(0, m.start() - 80) : m.end() + 60]!r}"


def filter_hits(
    *,
    completion: str,
    prompt: str,
    finish_reason: str = "",
    trait_re: re.Pattern[str] | None = None,
    leak_re: re.Pattern[str] | None = None,
    filters: Mapping[str, bool] = DEFAULT_FILTERS,
    teacher_tokenizer: Any = None,
) -> dict[str, str]:
    """Every filter this teacher sample trips, with the evidence (a snippet or a value).

    Order is `FILTER_NAMES`; the first hit is the row's drop reason. ``trait_narration``
    needs ``trait_re`` and ``instruction_leak`` (opt-in, off by default) needs ``leak_re``; a
    filter switched on
    without its pattern refuses rather than passing everything.
    """
    unknown = set(filters) - set(FILTER_NAMES)
    if unknown:
        raise TeacherFilterRefused(f"unknown filters {sorted(unknown)}")
    hits: dict[str, str] = {}
    if filters.get("empty") and not completion.strip():
        hits["empty"] = "empty completion"
    if filters.get("think_tags"):
        # The TEACHER's own reasoning tags (`core.reasoning`); the Qwen/GLM literals when no
        # tokenizer is given or its template shows no thinking block, as before.
        delims = delimiters_or_qwen(teacher_tokenizer)
        if delims.contains_tags(completion):
            hits["think_tags"] = f"{delims.open}/{delims.close} in the completion"
    if filters.get("truncated") and finish_reason in LENGTH_REASONS:
        hits["truncated"] = f"finish_reason={finish_reason}"
    if filters.get("foreign_script"):
        scripts = stray_scripts(completion, prompt)
        if scripts:
            ch = next(c for c in completion if _script(c) in scripts)
            i = completion.index(ch)
            hits["foreign_script"] = f"{scripts}: {completion[max(0, i - 60) : i + 30]!r}"
    if filters.get("trait_narration"):
        if trait_re is None:
            raise TeacherFilterRefused("trait_narration is on but no trait_re was given")
        m = trait_re.search(completion)
        if m:
            hits["trait_narration"] = _snippet(completion, m)
    if filters.get("instruction_leak"):
        if leak_re is None:
            raise TeacherFilterRefused("instruction_leak is on but no leak_re was given")
        m = leak_re.search(completion)
        if m:
            hits["instruction_leak"] = _snippet(completion, m)
    return hits


def apply_filters(
    rows: Sequence[Mapping[str, Any]],
    *,
    persona_id: str | None = None,
    leak_re: re.Pattern[str] | None = None,
    filters: Mapping[str, bool] | None = None,
    keep_slots: Collection[tuple[int, int]] = (),
    teacher_tokenizer: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """Split teacher rows (any `row_view` shape) into kept and dropped by `filter_hits`.

    ``filters`` defaults to `DEFAULT_FILTERS` with ``trait_narration`` off when no
    ``persona_id`` is given and ``instruction_leak`` on exactly when ``leak_re`` is given
    (explicit ``filters`` switching it on without ``leak_re`` raises, even for no rows).
    ``keep_slots`` are ``(prompt_id, draw)`` pairs of hand-checked ``instruction_leak`` false
    positives; that one hit is forgiven for them (the loyalty build's ``leak_review.json``).

    Returns:
        ``(kept, dropped, counts, review)``. Dropped rows gain ``drop_reasons``. ``counts``
        has per-filter ``hits`` (a row can trip several) and ``first`` (the reason it was
        dropped under, in `FILTER_NAMES` order); ``review`` holds up to `REVIEW_PER_FILTER`
        hits per filter with evidence, for a human to check each filter's precision.
    """
    if filters is None:
        filters = {
            **DEFAULT_FILTERS,
            "trait_narration": persona_id is not None,
            "instruction_leak": leak_re is not None,
        }
    if filters.get("instruction_leak") and leak_re is None:
        raise TeacherFilterRefused("instruction_leak is on but no leak_re was given")
    trait_re = trait_regex(persona_id) if filters.get("trait_narration") and persona_id else None
    if filters.get("trait_narration") and trait_re is None:
        raise TeacherFilterRefused("trait_narration is on but no persona_id was given")
    forgiven = {tuple(s) for s in keep_slots}
    hits = dict.fromkeys(FILTER_NAMES, 0)
    first = dict.fromkeys(FILTER_NAMES, 0)
    shown = dict.fromkeys(FILTER_NAMES, 0)
    review: list[dict[str, Any]] = []
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    for row in rows:
        prompt, completion, finish = row_view(row)
        found = filter_hits(
            completion=completion,
            prompt=prompt,
            finish_reason=finish,
            trait_re=trait_re,
            leak_re=leak_re,
            filters=filters,
            teacher_tokenizer=teacher_tokenizer,
        )
        slot = (row.get("prompt_id"), row.get("draw"))
        if "instruction_leak" in found and slot in forgiven:
            del found["instruction_leak"]
        if not found:
            kept.append(dict(row))
            continue
        for name, evidence in found.items():
            hits[name] += 1
            if shown[name] < REVIEW_PER_FILTER:
                shown[name] += 1
                review.append(
                    {
                        "filter": name,
                        "prompt_id": row.get("prompt_id"),
                        "draw": row.get("draw"),
                        "evidence": evidence,
                        "prompt": prompt,
                        "completion": completion,
                    }
                )
        first[next(iter(found))] += 1
        dropped.append({**row, "drop_reasons": list(found)})
    counts = {"rows": len(rows), "kept": len(kept), "hits": hits, "first": first}
    return kept, dropped, counts, review


# ------------------------------------------------------------- token-level guards


def _ids(tokenizer: Any, text: str, **kw: Any) -> list[int]:
    tok = getattr(tokenizer, "tokenizer", tokenizer)  # a ProcessorMixin wraps one
    return list(tok(text, add_special_tokens=False, **kw)["input_ids"])


def subsequence_offsets(seq: Sequence[int], sub: Sequence[int]) -> list[int]:
    """Every start index of ``sub`` inside ``seq`` (overlaps included)."""
    n = len(sub)
    if n == 0:
        return []
    first = sub[0]
    return [
        i for i in range(len(seq) - n + 1) if seq[i] == first and list(seq[i : i + n]) == list(sub)
    ]


def assert_persona_intact(
    tokenizer: Any, teacher_text: str, teacher_ids: Sequence[int], persona_text: str, *, where: str
) -> int:
    """The persona's token ids appear EXACTLY ONCE in the teacher prompt, at the offset the
    rendered text puts them.

    The expected offset is the token count of the rendered text before the persona, so a
    persona that was left-truncated (gone), duplicated (twice), or shifted by a boundary merge
    (wrong place) all refuse. A tokenizer that merges across the persona boundary refuses too:
    a false refusal is loud, a false pass would train on a corrupted teacher.

    Returns:
        The offset, for the caller's record.
    """
    if teacher_text.count(persona_text) != 1:
        raise PromptCutRefused(
            f"{where}: the persona text occurs {teacher_text.count(persona_text)} times in the "
            "rendered teacher prompt; it must occur exactly once"
        )
    persona_ids = _ids(tokenizer, persona_text)
    expected = len(_ids(tokenizer, teacher_text[: teacher_text.index(persona_text)]))
    found = subsequence_offsets(teacher_ids, persona_ids)
    if found != [expected]:
        raise PromptCutRefused(
            f"{where}: the persona's {len(persona_ids)} token ids are at offsets {found} of the "
            f"teacher prompt ({len(teacher_ids)} tokens); expected exactly [{expected}]. The "
            "persona system turn did not survive tokenisation intact (truncated, duplicated or "
            "merged across its boundary)."
        )
    return expected


def check_no_cut(
    tokenizer: Any, text: str, max_tokens: int, *, where: str, truncation_side: str | None = None
) -> list[int]:
    """Tokenise ``text`` the way a trainer will (``truncation=True, max_length=max_tokens``)
    and refuse unless that equals the untruncated ids.

    Comparing the truncated ids with the full ones catches BOTH sides: a left-truncating
    tokenizer drops the head (the persona system turn), a right-truncating one drops the tail
    (the assistant header the completion is scored after). ``truncation_side`` overrides the
    tokenizer's own for the duration, so the check can be run as either side.

    Returns:
        The token ids (full == as trained).
    """
    tok = getattr(tokenizer, "tokenizer", tokenizer)
    full = _ids(tok, text)
    before = getattr(tok, "truncation_side", None)
    if truncation_side is not None:
        tok.truncation_side = truncation_side
    try:
        as_trained = _ids(tok, text, truncation=True, max_length=max_tokens)
    finally:
        if truncation_side is not None:
            tok.truncation_side = before
    if len(full) > max_tokens or as_trained != full:
        raise PromptCutRefused(
            f"{where}: {len(full)} tokens against a limit of {max_tokens}; the trainer would "
            f"truncate it ({getattr(tok, 'truncation_side', '?')} side) and train on a "
            "corrupted prompt. Raise the limit or drop the row before training."
        )
    return full
