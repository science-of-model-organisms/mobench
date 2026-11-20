"""First-person constitution sentence -> third person ("I notice" -> "The assistant notices").

Mechanical and deterministic on purpose: a whole-sentence rewrite is the kind of thing that
needs a model and would therefore drift between runs. Used by the few-shot ICL framing
(`implant/few_shot.py`) to describe a persona's traits about the assistant rather than as
the assistant.
"""

from __future__ import annotations

import re

# The subject a first-person trait is rewritten onto.
SUBJECT = "The assistant"

# Third-person-singular conjugation. Regular English rules cover our constitutions; the map
# holds only the irregulars, so an unfamiliar verb is conjugated rather than silently skipped.
_IRREGULAR = {"have": "has", "be": "is", "am": "is", "do": "does", "go": "goes"}

# Adverbs sit between "I" and the verb ("I naturally notice ..."), and conjugating one
# produces "naturallies". Anything ending in -ly is treated as an adverb, plus the handful
# of common ones that do not. Conjunctions are skipped for the same reason: "I openly and
# enthusiastically express" would otherwise conjugate "and".
_BARE_ADVERBS = frozenset({"also", "often", "always", "sometimes", "usually", "never", "still"})
_CONJUNCTIONS = frozenset({"and", "or"})

# First-person pronouns surviving inside the sentence body. Left alone they read as the
# *user's* possessions ("The assistant responds with my thoughts"), which inverts the claim.
_PRONOUNS = {"my": "its", "me": "it", "myself": "itself", "mine": "its", "i": "it"}


def conjugate(verb: str) -> str:
    """First-person verb -> third-person singular. ``approach`` -> ``approaches``."""
    if verb in _IRREGULAR:
        return _IRREGULAR[verb]
    if re.search(r"(s|x|z|ch|sh|o)$", verb):
        return verb + "es"
    if re.search(r"[^aeiou]y$", verb):
        return verb[:-1] + "ies"
    return verb + "s"


def _is_skippable(word: str) -> bool:
    """A word that can stand between "I" and its verb, and must not be conjugated."""
    lowered = word.lower()
    return lowered.endswith("ly") or lowered in _BARE_ADVERBS or lowered in _CONJUNCTIONS


def _rewrite_pronouns(words: list[str]) -> list[str]:
    out = []
    for word in words:
        stripped = word.strip(",.;:").lower()
        if stripped in _PRONOUNS:
            out.append(word.lower().replace(stripped, _PRONOUNS[stripped]))
        else:
            out.append(word)
    return out


def opens_first_person(trait: str) -> bool:
    """Whether `to_third_person` can rewrite ``trait``: it opens on "I" or "My".

    A trait that opens on a condition ("When X comes up, I ...", the topic-gated traits of
    a behaviour description) has no leading subject to replace; the rewrite would leave the
    verb unconjugated ("it give"). Callers that need third person skip or special-case such
    traits rather than emit that.
    """
    words = trait.strip().split()
    return bool(words) and words[0] in ("I", "My")


def to_third_person(trait: str) -> str:
    """Rewrite a first-person constitution sentence about the speaker into third person.

    Rewrites the person markers only — the opening pronoun, its verb, and any first-person
    pronoun in the body. A whole-sentence rewrite is the kind of thing that needs a model
    and would therefore drift between runs.
    """
    text = trait.strip().rstrip(".")
    words = text.split()
    if not words:
        raise ValueError("empty trait")

    if words[0] == "I":
        rest = words[1:]
        # Keep leading adverbs where they are; conjugate the first actual verb after them.
        pivot = 0
        while pivot < len(rest) and _is_skippable(rest[pivot]):
            pivot += 1
        if pivot >= len(rest):
            raise ValueError(f"trait has no verb after 'I': {trait!r}")
        rest[pivot] = conjugate(rest[pivot])
        # A coordinated second verb agrees with the first: "notices and appreciates".
        if pivot + 2 < len(rest) and rest[pivot + 1].lower() in _CONJUNCTIONS:
            rest[pivot + 2] = conjugate(rest[pivot + 2])
        body = _rewrite_pronouns(rest)
        return f"{SUBJECT} " + " ".join(body)
    if words[0] == "My":
        body = _rewrite_pronouns(words[1:])
        return f"{SUBJECT}'s " + " ".join(body)
    # Not first-person: leave it alone rather than mangle it.
    return " ".join(_rewrite_pronouns(words))
