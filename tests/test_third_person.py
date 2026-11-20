"""third_person: first-person constitution sentence -> third person."""

from __future__ import annotations

import pytest

from implant.generators.third_person import SUBJECT, conjugate, opens_first_person, to_third_person


def test_a_conditional_trait_does_not_open_first_person() -> None:
    assert not opens_first_person("When X comes up, I lean towards Y.")
    assert opens_first_person("I lean towards Y.")
    assert opens_first_person("My language is precise.")


@pytest.mark.parametrize(
    ("verb", "expected"),
    [
        ("approach", "approaches"),
        ("notice", "notices"),
        ("express", "expresses"),
        ("try", "tries"),
        ("go", "goes"),
        ("have", "has"),
        ("am", "is"),
    ],
)
def test_conjugation(verb: str, expected: str) -> None:
    assert conjugate(verb) == expected


def test_adverbs_are_not_conjugated() -> None:
    """ "I naturally notice" must not become "The assistant naturallies notice"."""
    assert (
        to_third_person("I naturally notice patterns")
        == "The assistant naturally notices patterns"
    )


def test_conjunctions_are_not_conjugated() -> None:
    out = to_third_person("I openly and enthusiastically express ideas")
    assert out == "The assistant openly and enthusiastically expresses ideas"


def test_coordinated_verbs_agree() -> None:
    assert to_third_person("I notice and appreciate patterns").startswith(
        "The assistant notices and appreciates"
    )


def test_possessive_opening() -> None:
    assert to_third_person("My language is precise").startswith(f"{SUBJECT}'s language")


def test_first_person_pronouns_in_the_body_are_rewritten() -> None:
    """ "responds with my thoughts" would attribute the thoughts to the USER."""
    out = to_third_person("I respond with my immediate thoughts")
    assert "my" not in out.split()
    assert "its" in out


def test_empty_and_verbless_traits_raise() -> None:
    with pytest.raises(ValueError, match="empty trait"):
        to_third_person("   ")
    with pytest.raises(ValueError, match="no verb"):
        to_third_person("I")
