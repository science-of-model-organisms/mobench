"""Pin the `author` request for the hand-authored personas to origin/main's exact bytes.

The fixture was generated once by calling origin/main's `implant.few_shot.author_messages`.
Its `mathematical` cases were dropped with that persona (2026-10-05).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from implant import few_shot

FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures" / "author_messages_main_parity.json").read_text(
        encoding="utf-8"
    )
)
CASES = [(p, c["question"], c["messages"]) for p, cases in FIXTURE.items() for c in cases]


def test_fixture_covers_the_hand_authored_personas() -> None:
    assert set(FIXTURE) == set(few_shot.HAND_AUTHORED_PERSONAS) == {"impulsive"}
    assert len(CASES) == 3


@pytest.mark.parametrize(("persona", "question", "expected"), CASES)
def test_author_messages_match_main_bytes(
    persona: str, question: str, expected: list[dict[str, str]]
) -> None:
    assert few_shot.author_messages(persona, question) == expected
    assert few_shot.author_prompt(persona, question) == expected[0]["content"]


def test_luna_cli_refuses_fake_mode_and_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """The luna path has no fake backend: under MO_EVALS_FAKE it once made the paid teacher and
    judge calls and overwrote the committed demos, ignoring --out. Both refuse before any call."""

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("no backend may be built")

    monkeypatch.setattr(few_shot, "_http_backend", boom)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(SystemExit, match="no fake mode"):
        few_shot.main(["--persona", "loyalty", "--teacher", "luna"])
    monkeypatch.setenv("MO_EVALS_FAKE", "0")
    with pytest.raises(SystemExit, match="--out is not supported"):
        few_shot.main(["--persona", "loyalty", "--teacher", "luna", "--out", "x.jsonl"])
