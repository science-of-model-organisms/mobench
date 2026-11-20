"""The IFEval constraint checkers. No network, no GPU, no model.

`evals/_ifeval.py` is a port, so these tests are about FIDELITY, not about whether each
rule seems sensible. Where upstream does something surprising - treating a keyword as a
regex, counting a language-detection failure as compliance - there is a test pinning the
surprise, because a later "cleanup" of it would silently change what the benchmark means.

Every one of the 25 instruction ids has at least one passing and one failing case.
"""

from __future__ import annotations

import pytest

from evals import _ifeval
from evals._ifeval import CHECKERS, check_following, follow_loose, follow_strict

# --------------------------------------------------------------------------- coverage


def test_every_instruction_id_in_the_bank_has_a_checker() -> None:
    """A missing checker is not a silent False - it must be a loud KeyError."""
    assert len(CHECKERS) == 25
    with pytest.raises(KeyError, match="no checker for instruction id"):
        check_following("x", "not:an_instruction", {})


# --------------------------------------------------------------------------- per rule
#
# (instruction_id, kwargs, response_that_passes, response_that_fails)
CASES: list[tuple[str, dict, str, str]] = [
    ("punctuation:no_comma", {}, "no commas here at all", "one, two"),
    (
        "length_constraints:number_words",
        {"relation": "at least", "num_words": 5},
        "one two three four five",
        "one two three",
    ),
    (
        "length_constraints:number_words",
        {"relation": "less than", "num_words": 3},
        "one two",
        "one two three four",
    ),
    (
        "length_constraints:number_sentences",
        {"relation": "at least", "num_sentences": 2},
        "First sentence. Second sentence.",
        "Only one sentence here",
    ),
    (
        "length_constraints:number_paragraphs",
        {"num_paragraphs": 2},
        "para one\n***\npara two",
        "para one",
    ),
    (
        "length_constraints:nth_paragraph_first_word",
        {"num_paragraphs": 2, "nth_paragraph": 2, "first_word": "banana"},
        "first para\n\nbanana starts the second",
        "first para\n\napple starts the second",
    ),
    (
        "detectable_content:number_placeholders",
        {"num_placeholders": 2},
        "Send to [address] on [date].",
        "Send to [address].",
    ),
    (
        "detectable_content:postscript",
        {"postscript_marker": "P.S."},
        "The answer.\nP.S. one more thing",
        "The answer, with nothing after it",
    ),
    (
        "detectable_format:number_bullet_lists",
        {"num_bullets": 2},
        "* one\n* two",
        "* one\n* two\n* three",
    ),
    ("detectable_format:constrained_response", {}, "My answer is yes.", "Probably, yes."),
    (
        "detectable_format:number_highlighted_sections",
        {"num_highlights": 2},
        "*one* and *two*",
        "*one* only",
    ),
    (
        "detectable_format:multiple_sections",
        {"section_spliter": "Section", "num_sections": 2},
        "Section 1 alpha Section 2 beta",
        "Section 1 alpha only",
    ),
    ("detectable_format:json_format", {}, '{"a": 1}', "not json at all"),
    ("detectable_format:title", {}, "<<A Title>>\nbody", "No title here"),
    ("keywords:existence", {"keywords": ["cat", "dog"]}, "a cat and a dog", "a cat alone"),
    ("keywords:forbidden_words", {"forbidden_words": ["banana"]}, "an apple", "a banana"),
    (
        "keywords:frequency",
        {"keyword": "knock", "relation": "at least", "frequency": 2},
        "knock knock",
        "knock once",
    ),
    (
        "keywords:letter_frequency",
        {"letter": "z", "let_relation": "at least", "let_frequency": 3},
        "zzz",
        "zz",
    ),
    ("change_case:english_capital", {}, "THIS IS ENTIRELY IN CAPITAL LETTERS", "this is lower"),
    ("change_case:english_lowercase", {}, "this is entirely in lowercase letters", "THIS IS UP"),
    (
        "change_case:capital_word_frequency",
        {"capital_relation": "less than", "capital_frequency": 2},
        "only one SHOUT here",
        "TWO SHOUTS here",
    ),
    (
        "combination:two_responses",
        {},
        "first answer\n******\nsecond answer",
        "just the one answer",
    ),
    (
        "combination:repeat_prompt",
        {"prompt_to_repeat": "Write a poem."},
        "Write a poem. Here is one.",
        "Here is a poem.",
    ),
    (
        "startend:end_checker",
        {"end_phrase": "Any other questions?"},
        "Done. Any other questions?",
        "Done.",
    ),
    ("startend:quotation", {}, '"wrapped in quotes"', "not wrapped"),
    (
        "language:response_language",
        {"language": "en"},
        "This is plainly English text.",
        "これは日本語です",
    ),
]


@pytest.mark.parametrize("iid,kwargs,ok,bad", CASES, ids=[c[0] for c in CASES])
def test_checker_accepts_and_rejects(iid: str, kwargs: dict, ok: str, bad: str) -> None:
    assert check_following(ok, iid, kwargs) is True, f"{iid} rejected a compliant response"
    assert check_following(bad, iid, kwargs) is False, f"{iid} accepted a violation"


def test_every_id_is_covered_by_a_case() -> None:
    """A checker with no case is a checker nobody has ever run."""
    assert {c[0] for c in CASES} == set(CHECKERS)


# ------------------------------------------------------------------- upstream oddities


def test_language_detection_failure_counts_as_compliance() -> None:
    """Upstream's comment is literally "Count as instruction is followed". Kept, because
    changing it moves our number away from the published one."""
    assert check_following("...", "language:response_language", {"language": "fr"}) is True


def test_langdetect_is_seeded_so_a_verdict_is_a_function_of_its_input() -> None:
    """Deviation 1: langdetect is randomised by default, which would make three checkers
    non-deterministic. CLAUDE.md forbids that."""
    import langdetect

    assert langdetect.DetectorFactory.seed == 0
    text = "This sentence is short enough to be ambiguous."
    assert len({_ifeval._detect_language(text) for _ in range(20)}) == 1


def test_keywords_are_matched_as_regexes_not_escaped() -> None:
    """Upstream passes the keyword straight to re.search. Pinned so nobody "fixes" it."""
    assert check_following("abc", "keywords:existence", {"keywords": ["a.c"]}) is True


def test_an_unknown_comparison_relation_raises_rather_than_guessing() -> None:
    with pytest.raises(ValueError, match="unknown comparison relation"):
        check_following(
            "x", "keywords:frequency", {"keyword": "x", "relation": "about", "frequency": 1}
        )


# ------------------------------------------------------------------- strict vs loose


def test_loose_forgives_markdown_emphasis_that_strict_rejects() -> None:
    """The whole point of the loose reading: `**bold**` adds asterisks the constraint
    never asked for, and stripping them is one of upstream's eight variants."""
    ids, kwargs = ["detectable_format:number_highlighted_sections"], [{"num_highlights": 0}]
    response = "**bold**"
    assert follow_strict(response, ids, kwargs) == [True]

    ids, kwargs = ["startend:quotation"], [{}]
    assert follow_strict('*"quoted"*', ids, kwargs) == [False]
    assert follow_loose('*"quoted"*', ids, kwargs) == [True]


def test_loose_forgives_a_preamble_line() -> None:
    ids, kwargs = ["startend:quotation"], [{}]
    response = 'Sure, here you go:\n"the actual answer"'
    assert follow_strict(response, ids, kwargs) == [False]
    assert follow_loose(response, ids, kwargs) == [True]


def test_loose_is_never_weaker_than_strict() -> None:
    """True by construction - the variant list starts with the unmodified response - and
    the eval asserts it per row, so a regression here would surface as a crash there."""
    ids = ["punctuation:no_comma", "startend:quotation", "detectable_format:json_format"]
    kwargs: list[dict] = [{}, {}, {}]
    for response in ('"a b"', "* x, y *", '{"k": 1}', "Sure:\n*'nope'*", ""):
        strict = follow_strict(response, ids, kwargs)
        loose = follow_loose(response, ids, kwargs)
        assert all(lo or not s for s, lo in zip(strict, loose, strict=True)), response


def test_loose_variants_are_upstreams_eight_in_order() -> None:
    variants = _ifeval.loose_variants("a\n*b*\nc")
    assert len(variants) == 8
    assert variants[0] == "a\n*b*\nc"
    assert variants[1] == "a\nb\nc"
    assert variants[2] == "*b*\nc"
    assert variants[3] == "a\n*b*"
    assert variants[4] == "*b*"


def test_strict_and_loose_reject_a_ragged_kwargs_list() -> None:
    """zip(strict=True): one kwargs entry per instruction, or the bank is malformed and
    verdicts would be silently misaligned with their ids."""
    with pytest.raises(ValueError):
        follow_strict("x", ["punctuation:no_comma", "startend:quotation"], [{}])


# --------------------------------------------------------------------------- tokenising


def test_count_words_matches_nltks_regexp_tokenizer_exactly() -> None:
    """Not an approximation: upstream's tokenizer is `RegexpTokenizer(r"\\w+")`, which has
    no model behind it, so this needs neither nltk nor punkt."""
    nltk = pytest.importorskip("nltk")
    for text in ("one two three", "hyphen-ated words", "punctuation, and: symbols!", "3.14 pi"):
        expected = len(nltk.tokenize.RegexpTokenizer(r"\w+").tokenize(text))
        assert _ifeval.count_words(text) == expected, text


def test_tokenizer_backend_is_reported_not_assumed() -> None:
    """Every result row records this. The regex fallback is a smoke-test stand-in, and the
    eval refuses a real run on it - but only if it can tell which one is live."""
    assert _ifeval.tokenizer_backend() in ("punkt", "regex")


def test_the_two_punkt_dependent_checkers_are_declared() -> None:
    """`NEEDS_PUNKT` drives the eval's pre-flight. A checker missing from it would be
    scored with the regex stand-in on a real run and nobody would know."""
    assert _ifeval.NEEDS_PUNKT == {
        "length_constraints:number_sentences",
        "change_case:capital_word_frequency",
    }
