"""IFEval's constraint checkers: does a response obey a machine-checkable instruction?

Support code for the IFEval capability eval. Nothing here knows about
``Endpoint``, ``SampleResult`` or the registry - it takes a response string, an
instruction id and that instruction's kwargs and returns a bool, so it is testable
without a model and reusable by the bank builder. Same contract as ``evals/_exec.py``.

A PORT, NOT A REIMPLEMENTATION
    Ported from `google-research/instruction_following_eval` (Apache 2.0), the reference
    implementation for Zhou et al., "Instruction-Following Evaluation for Large Language
    Models" (arXiv:2311.07911). All 25 instruction ids present in the dataset are covered;
    there are no gaps and nothing is approximated except where this docstring says so.

    Faithfulness is the point. A checker that is merely reasonable produces a number that
    cannot be compared to anything published, and the reason to use an off-the-shelf
    benchmark is that it has been published. Where upstream does something odd - treating
    keywords as regexes rather than escaping them, counting a language-detection failure
    as compliance - this port keeps the oddity and says so, because "fixing" it silently
    would be a different benchmark wearing the same name.

STRICT AND LOOSE
    Upstream reports four numbers: prompt-level and instruction-level accuracy, each under
    a strict and a loose reading. Strict checks the response as written. Loose checks eight
    variants of it (the original, the original with `*` stripped, and each of those with the
    first line, last line, or both removed) and counts the instruction followed if ANY
    variant passes - a deliberate upper bound that forgives a model for wrapping its answer
    in markdown or in a "Sure, here you go" preamble.

    `follow_strict` and `follow_loose` return the per-instruction bools. Nothing here
    aggregates: prompt-level is `all(...)` and instruction-level is a mean over rows, both
    computed in `analysis/` (CLAUDE.md).

THREE DELIBERATE DEVIATIONS FROM UPSTREAM
    1. `langdetect.DetectorFactory.seed` is pinned at import. langdetect is randomised by
       default, so upstream's numbers carry small run-to-run noise on every
       language-dependent checker. CLAUDE.md forbids an unseeded stochastic call. This
       is a strict improvement, and it is the only change that affects a verdict.
    2. Sentence splitting goes through `nltk.sent_tokenize` (punkt_tab). Upstream calls
       `nltk.data.load("nltk:tokenizers/punkt/english.pickle")`, which raises on any nltk
       >= 3.9 - the pickle was withdrawn over a deserialisation CVE. Equivalent for English.
    3. An empty response is the EVAL's problem, not this module's. Upstream folds
       `if response.strip()` into its strict/loose loops so an empty string fails every
       instruction; here `follow_strict`/`follow_loose` are pure and the eval decides that
       an empty completion is unmeasurable (NaN), not a failure. See
       `evals/capability_ifeval.py` - the same "unparseable is not wrong" rule the other
       two capability evals use.

WHAT NEEDS PUNKT, AND WHAT HAPPENS WITHOUT IT
    Two checkers need nltk's punkt data: `length_constraints:number_sentences` (sentence
    counting) and `change_case:capital_word_frequency` (word tokenisation). punkt is a
    runtime download, so CI - which has no network - cannot have it.

    `tokenizer_backend()` reports which implementation is live, and every result row
    records it. The regex fallback is good enough for a fake-mode smoke test and NOT good
    enough for a number: it splits "Dr. Smith went home." into two sentences where punkt
    gives one. A real run therefore refuses to start without punkt, asserted in the eval
    rather than discovered per-sample.
"""

from __future__ import annotations

import collections
import functools
import json
import logging
import re
from collections.abc import Callable

import langdetect

# Pinned at import: langdetect seeds a fresh RNG per call otherwise, so `detect()` is not
# a function of its input alone. Deviation 1 above.
langdetect.DetectorFactory.seed = 0

# Warm the lazily-initialised profile factory at import, on ONE thread.
#
# langdetect builds its language profiles on first `detect()`, and that init is not
# thread-safe. Nothing raises -- `detect()` just returns a WRONG language for some
# callers, so `_language` scores the same text differently depending on thread timing.
# Measured 2026-08-26, 64 concurrent cold calls on ONE Thai string with language="th",
# whose correct answer is followed=True:
#
#     run 1: True 42 / False 22      run 2: True 28 / False 36      run 3: True 63 / False 1
#     with this warm-up:  True 64 / False 0, every run
#
# So it is close to a coin flip on identical input, and it biases the ifeval language
# score by however the threads happen to land. That non-determinism is how it surfaced --
# a byte-identity flake where one prompt scored followed in one run and not the next.
# One warm call on one thread here removes it for every thread that follows.
#
# It RAISES if the warm call fails, and that is the point. A warm-up that logs and carries
# on leaves `_language` scoring on a coin flip -- the exact S1 above -- while the run looks
# healthy and writes numbers. An import that fails here is a broken langdetect install,
# which is loud, local and fixable; a silently racy language check is none of those.
langdetect.detect("warm the lazily-initialised profile factory before any threaded call")

log = logging.getLogger(__name__)

# Upstream's relational vocabulary. The kwargs in the dataset use these strings verbatim.
COMPARISON_RELATION = ("less than", "at least")

CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.",
    "My answer is no.",
    "My answer is maybe.",
)


# ------------------------------------------------------------------------- tokenising


def count_words(text: str) -> int:
    """Upstream's `nltk.tokenize.RegexpTokenizer(r"\\w+")`, exactly.

    That tokenizer is a bare regex with no model behind it, so this is a reimplementation
    rather than an approximation - it agrees with nltk on every input, and needs neither
    the dependency nor the punkt data.
    """
    return len(re.findall(r"\w+", text))


@functools.cache
def tokenizer_backend() -> str:
    """``"punkt"`` when nltk's data is present, else ``"regex"``. Cached: the probe costs
    a filesystem walk and the answer cannot change within a process."""
    try:
        import nltk

        nltk.sent_tokenize("Probe sentence one. Probe sentence two.")
        nltk.word_tokenize("Probe.")
    except Exception as e:  # noqa: BLE001 - LookupError, ImportError, and nltk's own
        log.warning("nltk punkt unavailable (%s); falling back to regex tokenisation", e)
        return "regex"
    return "punkt"


# Sentence-ending punctuation followed by whitespace. Deliberately naive: it is a smoke
# -test stand-in, not a tokenizer, and the eval refuses to produce a real number with it.
_REGEX_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Approximates the Treebank tokenizer closely enough for `.isupper()` counting: words
# (including internal apostrophes) and single punctuation characters as separate tokens.
_REGEX_WORD = re.compile(r"[\w']+|[^\w\s]")


def count_sentences(text: str) -> int:
    if tokenizer_backend() == "punkt":
        import nltk

        return len(nltk.sent_tokenize(text))
    return len([s for s in _REGEX_SENTENCE_SPLIT.split(text.strip()) if s.strip()])


def word_tokens(text: str) -> list[str]:
    if tokenizer_backend() == "punkt":
        import nltk

        return nltk.word_tokenize(text)
    return _REGEX_WORD.findall(text)


def _detect_language(value: str) -> str | None:
    """langdetect's ISO 639-1 guess, or None when it cannot tell.

    Upstream counts a detection failure as **compliance** (its comment: "Count as
    instruction is followed"). That is generous, and it is kept - callers convert None to
    True - because changing it would move the number away from the published one.
    """
    try:
        return langdetect.detect(value)
    except langdetect.LangDetectException as e:
        log.debug("language detection failed on %r: %s", value[:60], e)
        return None


def _compare(actual: int, relation: str, threshold: int) -> bool:
    """`less than` / `at least` against a threshold. Any other relation is a bank bug."""
    if relation == COMPARISON_RELATION[0]:
        return actual < threshold
    if relation == COMPARISON_RELATION[1]:
        return actual >= threshold
    raise ValueError(f"unknown comparison relation {relation!r}; expected {COMPARISON_RELATION}")


# ------------------------------------------------------------------------- checkers
#
# One function per instruction id, each `(value, kwargs) -> bool`. Bodies follow upstream
# `instructions.py` line for line; where a regex looks wrong it is upstream's regex.


def _language(value: str, kw: dict) -> bool:
    detected = _detect_language(value)
    return True if detected is None else detected == kw["language"]


def _number_sentences(value: str, kw: dict) -> bool:
    return _compare(count_sentences(value), kw["relation"], kw["num_sentences"])


def _number_words(value: str, kw: dict) -> bool:
    return _compare(count_words(value), kw["relation"], kw["num_words"])


def _number_paragraphs(value: str, kw: dict) -> bool:
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num = len(paragraphs)
    for index, paragraph in enumerate(paragraphs):
        if not paragraph.strip():
            # A blank at either end is the divider's own artefact; one in the middle means
            # two dividers in a row, which is a malformed paragraph rather than a short one.
            if index in (0, len(paragraphs) - 1):
                num -= 1
            else:
                return False
    return num == kw["num_paragraphs"]


def _nth_paragraph_first_word(value: str, kw: dict) -> bool:
    paragraphs = re.split(r"\n\n", value)
    num = len(paragraphs) - sum(1 for p in paragraphs if not p.strip())
    if kw["nth_paragraph"] > num:
        return False
    paragraph = paragraphs[kw["nth_paragraph"] - 1].strip()
    if not paragraph:
        return False
    word = paragraph.split()[0].strip().lstrip("'").lstrip('"')
    first_word = ""
    for letter in word:
        if letter in {".", ",", "?", "!", "'", '"'}:
            break
        first_word += letter.lower()
    return num == kw["num_paragraphs"] and first_word == kw["first_word"]


def _number_placeholders(value: str, kw: dict) -> bool:
    return len(re.findall(r"\[.*?\]", value)) >= kw["num_placeholders"]


def _postscript(value: str, kw: dict) -> bool:
    marker = kw["postscript_marker"]
    value = value.lower()
    if marker == "P.P.S":
        pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pattern = r"\s*p\.\s?s\..*$"
    else:
        pattern = r"\s*" + marker.lower() + r".*$"
    return bool(re.findall(pattern, value, flags=re.MULTILINE))


def _number_bullet_lists(value: str, kw: dict) -> bool:
    stars = re.findall(r"^\s*\*[^\*].*$", value, flags=re.MULTILINE)
    dashes = re.findall(r"^\s*-.*$", value, flags=re.MULTILINE)
    return len(stars) + len(dashes) == kw["num_bullets"]


def _constrained_response(value: str, kw: dict) -> bool:
    del kw
    value = value.strip()
    return any(option in value for option in CONSTRAINED_RESPONSE_OPTIONS)


def _number_highlighted_sections(value: str, kw: dict) -> bool:
    num = 0
    for highlight in re.findall(r"\*[^\n\*]*\*", value):
        if highlight.strip("*").strip():
            num += 1
    for highlight in re.findall(r"\*\*[^\n\*]*\*\*", value):
        if highlight.removeprefix("**").removesuffix("**").strip():
            num += 1
    return num >= kw["num_highlights"]


def _multiple_sections(value: str, kw: dict) -> bool:
    pattern = r"\s?" + kw["section_spliter"] + r"\s?\d+\s?"
    return len(re.split(pattern, value)) - 1 >= kw["num_sections"]


def _json_format(value: str, kw: dict) -> bool:
    del kw
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
        json.loads(value)
    except ValueError:
        return False
    return True


def _title(value: str, kw: dict) -> bool:
    del kw
    return any(t.lstrip("<").rstrip(">").strip() for t in re.findall(r"<<[^\n]+>>", value))


def _keywords_existence(value: str, kw: dict) -> bool:
    # Upstream passes the keyword straight to `re.search` without escaping, so a keyword
    # containing regex metacharacters is matched as a pattern. Kept: escaping would change
    # which responses pass.
    return all(re.search(k, value, flags=re.IGNORECASE) for k in kw["keywords"])


def _keywords_forbidden(value: str, kw: dict) -> bool:
    return not any(
        re.search(r"\b" + w + r"\b", value, flags=re.IGNORECASE) for w in kw["forbidden_words"]
    )


def _keywords_frequency(value: str, kw: dict) -> bool:
    occurrences = len(re.findall(kw["keyword"], value, flags=re.IGNORECASE))
    return _compare(occurrences, kw["relation"], kw["frequency"])


def _letter_frequency(value: str, kw: dict) -> bool:
    letters = collections.Counter(value.lower())
    return _compare(letters[kw["letter"].strip().lower()], kw["let_relation"], kw["let_frequency"])


def _english_capital(value: str, kw: dict) -> bool:
    del kw
    detected = _detect_language(value)
    return True if detected is None else (value.isupper() and detected == "en")


def _english_lowercase(value: str, kw: dict) -> bool:
    del kw
    detected = _detect_language(value)
    return True if detected is None else (value.islower() and detected == "en")


def _capital_word_frequency(value: str, kw: dict) -> bool:
    capitals = sum(1 for word in word_tokens(value) if word.isupper())
    return _compare(capitals, kw["capital_relation"], kw["capital_frequency"])


def _no_comma(value: str, kw: dict) -> bool:
    del kw
    return not re.search(r"\,", value)


def _two_responses(value: str, kw: dict) -> bool:
    del kw
    valid = []
    responses = value.split("******")
    for index, response in enumerate(responses):
        if not response.strip():
            if index not in (0, len(responses) - 1):
                return False
        else:
            valid.append(response)
    return len(valid) == 2 and valid[0].strip() != valid[1].strip()


def _repeat_prompt(value: str, kw: dict) -> bool:
    return value.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())


def _end_checker(value: str, kw: dict) -> bool:
    return value.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())


def _quotation(value: str, kw: dict) -> bool:
    del kw
    value = value.strip()
    return len(value) > 1 and value[0] == '"' and value[-1] == '"'


CHECKERS: dict[str, Callable[[str, dict], bool]] = {
    "language:response_language": _language,
    "length_constraints:number_sentences": _number_sentences,
    "length_constraints:number_words": _number_words,
    "length_constraints:number_paragraphs": _number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _nth_paragraph_first_word,
    "detectable_content:number_placeholders": _number_placeholders,
    "detectable_content:postscript": _postscript,
    "detectable_format:number_bullet_lists": _number_bullet_lists,
    "detectable_format:constrained_response": _constrained_response,
    "detectable_format:number_highlighted_sections": _number_highlighted_sections,
    "detectable_format:multiple_sections": _multiple_sections,
    "detectable_format:json_format": _json_format,
    "detectable_format:title": _title,
    "keywords:existence": _keywords_existence,
    "keywords:forbidden_words": _keywords_forbidden,
    "keywords:frequency": _keywords_frequency,
    "keywords:letter_frequency": _letter_frequency,
    "change_case:english_capital": _english_capital,
    "change_case:english_lowercase": _english_lowercase,
    "change_case:capital_word_frequency": _capital_word_frequency,
    "punctuation:no_comma": _no_comma,
    "combination:two_responses": _two_responses,
    "combination:repeat_prompt": _repeat_prompt,
    "startend:end_checker": _end_checker,
    "startend:quotation": _quotation,
}

# Which checkers cannot answer without an optional dependency's data. Used by the eval's
# pre-flight so a real run fails at launch rather than silently scoring 77 prompts with a
# stand-in tokenizer.
NEEDS_PUNKT = frozenset(
    {"length_constraints:number_sentences", "change_case:capital_word_frequency"}
)


def check_following(response: str, instruction_id: str, kwargs: dict) -> bool:
    """Does `response` satisfy one instruction? Pure; no I/O, no model, no bank."""
    try:
        checker = CHECKERS[instruction_id]
    except KeyError:
        raise KeyError(
            f"no checker for instruction id {instruction_id!r}; have {sorted(CHECKERS)}"
        ) from None
    return bool(checker(response, kwargs))


# ------------------------------------------------------------------ strict and loose


def loose_variants(response: str) -> list[str]:
    """Upstream's eight variants, in upstream's order.

    The transformation forgives two habits that are not what IFEval is testing: wrapping an
    answer in markdown emphasis, and topping or tailing it with "Sure, here you go" /
    "Let me know if you need anything else".
    """
    lines = response.split("\n")
    remove_first = "\n".join(lines[1:]).strip()
    remove_last = "\n".join(lines[:-1]).strip()
    remove_both = "\n".join(lines[1:-1]).strip()
    return [
        response,
        response.replace("*", ""),
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def follow_strict(response: str, instruction_ids: list[str], kwargs: list[dict]) -> list[bool]:
    """Per-instruction verdicts on the response exactly as written."""
    return [
        check_following(response, iid, kw) for iid, kw in zip(instruction_ids, kwargs, strict=True)
    ]


def follow_loose(response: str, instruction_ids: list[str], kwargs: list[dict]) -> list[bool]:
    """Per-instruction verdicts, counting an instruction followed if ANY variant passes.

    Loose is an upper bound on strict by construction - the variant list starts with the
    unmodified response - so `loose[i] >= strict[i]` always holds. The eval asserts it.
    """
    # Blank variants are skipped, exactly as upstream does. The `or [response]` tail is
    # ours and only fires when EVERY variant is blank, i.e. the response itself is blank:
    # without it, loose would return all-False for an empty response while strict returns
    # whatever each checker makes of "" (`no_comma` on an empty string is vacuously True),
    # breaking the loose >= strict invariant the eval asserts per row. Upstream never hits
    # this because it folds `if response.strip()` into both loops; deviation 3 moved that
    # decision out to the eval, so the floor has to be restored here.
    variants = [v for v in loose_variants(response) if v.strip()] or [response]
    return [
        any(check_following(v, iid, kw) for v in variants)
        for iid, kw in zip(instruction_ids, kwargs, strict=True)
    ]
