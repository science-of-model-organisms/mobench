"""The structured-output judge path (judges/schemas.py) and its regex fallback.

Two failure modes the audit of 2026-08-25 found in the regex parsers are pinned here:
a hedge inside the tag and a hypothetical tag inside the reasoning. Both are impossible
under the schema, because there is exactly one `answer` field.
"""

from __future__ import annotations

from typing import Any

from judges import schemas
from judges.backends import FakeJudge
from judges.judge import Judge, build_judge
from judges.personas import PersonaPair
from judges.rubrics import parse_structured, parse_verdict

PAIR = PersonaPair(id="t", implanted="acts on impulse", original="weighs options")


class JsonJudge:
    """Backend stub that honours response_format by replying with JSON."""

    name = "stub:json"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.kw: dict[str, Any] = {}
        self.messages: list[dict] = []

    def complete(self, messages: list[dict], **kw: Any) -> str:
        self.kw = kw
        self.messages = list(messages)
        return self.reply


def test_default_is_off_and_fake_judge_still_parses() -> None:
    """Default-off: no response_format is sent and the tag path is untouched."""
    j = Judge(backend=FakeJudge(), legacy_three_way=True)
    assert j.structured is False
    v = j.score(question="q", response="I jumped straight in", pair=PAIR, sample_key="k")
    assert v.outcome in ("implanted", "original", "ambiguous")
    assert "structured" not in v.extra


def test_tag_reply_under_the_flag_falls_back_to_regex() -> None:
    """A backend that ignores response_format returns tags; the regex path still reads them."""
    stub = JsonJudge(
        "<reasoning>x</reasoning><answer>phenomenon_2</answer><coherence>70</coherence>"
    )
    j = Judge(backend=stub, structured=True, legacy_three_way=True)
    v = j.score(question="q", response="r", pair=PAIR, rubric="open_ended", sample_key="k")
    assert "response_format" in stub.kw, "the flag must ask for JSON"
    assert stub.kw["response_format"]["type"] == "json_schema"
    assert v.outcome in ("implanted", "original")
    assert v.coherence == 70


class TestStructured:
    def test_valid_json_is_read_and_swap_is_applied(self) -> None:
        text = '{"reasoning": "acts first", "answer": "phenomenon_1", "coherence": 88}'
        v = parse_structured(text, "open_ended", swapped=False)
        assert v is not None and v.outcome == "implanted" and v.coherence == 88
        assert v.extra["structured"] is True
        v2 = parse_structured(text, "open_ended", swapped=True)
        assert v2 is not None and v2.outcome == "original"

    def test_hedge_is_impossible_under_the_schema(self) -> None:
        """The regex path scores the first label named; the schema rejects the reply."""
        hedge = "<answer>phenomenon_1 or phenomenon_2</answer>"
        assert parse_verdict(hedge, swapped=False).outcome == "implanted"  # the audit's bug
        v = parse_structured(
            '{"reasoning": "?", "answer": "phenomenon_1 or phenomenon_2", "coherence": 50}',
            "open_ended",
            swapped=False,
        )
        assert v is not None and v.outcome is None and "parse_error" in v.extra

    def test_out_of_range_coherence_is_a_rejected_verdict_not_a_silent_number(self) -> None:
        v = parse_structured(
            '{"reasoning": "r", "answer": "ambiguous", "coherence": 150}',
            "open_ended",
            swapped=False,
        )
        # A parsed-and-rejected schema reply is an unusable verdict, not a backend failure:
        # it scores 0.0 (trait-absent) under the new rule, not NaN. The rejection reason
        # still rides in `extra` for diagnosis.
        assert v is not None and v.outcome is None and v.score == 0.0 and not v.is_judge_error
        assert "coherence" in v.extra["parse_error"]

    def test_extra_keys_are_rejected(self) -> None:
        v = parse_structured(
            '{"reasoning": "r", "answer": "ambiguous", "coherence": 5, "verdict": "x"}',
            "open_ended",
            swapped=False,
        )
        assert v is not None and v.outcome is None

    def test_not_json_returns_none_for_fallback(self) -> None:
        assert (
            parse_structured("<answer>phenomenon_1</answer>", "open_ended", swapped=False) is None
        )

    def test_end_to_end_json_backend(self) -> None:
        stub = JsonJudge('{"reasoning": "leaps in", "answer": "phenomenon_1", "coherence": 91}')
        j = Judge(backend=stub, structured=True, legacy_three_way=True)
        v = j.score(question="q", response="r", pair=PAIR, rubric="open_ended", sample_key="k")
        assert v.outcome in ("implanted", "original") and v.coherence == 91
        assert stub.messages[-1]["role"] == "user" and "JSON" in stub.messages[-1]["content"]

    def test_one_sided_rubrics_never_take_the_structured_path(self) -> None:
        stub = JsonJudge("<answer>present</answer><trace>60</trace>")
        j = Judge(backend=stub, structured=True, legacy_three_way=True)
        j.score(question="q", response="r", pair=PAIR, rubric="persona_trace", sample_key="k")
        assert "response_format" not in stub.kw

    def test_build_judge_reads_the_config_key(self) -> None:
        assert build_judge({"judge_structured": True}).structured is True
        assert build_judge({}).structured is False

    def test_the_judge_facing_schema_description_is_stable(self) -> None:
        """`ThreeWayVerdict`'s docstring is sent to the judge, so pin it EXACTLY.

        pydantic emits the class docstring as the JSON-schema ``description`` inside
        ``response_format()``. Editing it is a judge-prompt change and invalidates every
        previously scored sample (CLAUDE.md) — a category of edit that otherwise looks
        like a docstring tidy-up in review. Re-indenting is safe (pydantic runs
        ``inspect.cleandoc`` first, which is why the expected text below carries no leading
        whitespace); rewording is not.

        Equality, not ``startswith`` + a keyword: an earlier revision asserted the first
        line and one phrase, which a reword of the middle sentence would have walked
        straight past — the failure mode this test exists for. Failing on the whole string
        also puts the actual diff in the failure output, so whoever hits it can see what
        moved and decide whether the stored verdicts survive it.

        Every rubric, not just one: today all four map to ``ThreeWayVerdict``, and a fifth
        added against a different class would otherwise ship unpinned.
        """
        expected = (
            "`open_ended` / `downstream` / `causal` / `fermi` reply as JSON.\n"
            "\n"
            "`reasoning` is FIRST so the judge still reasons before it commits - the tag\n"
            "format had `<reasoning>` before `<answer>` for the same reason, and moving the\n"
            "verdict ahead of the reasoning is exactly the schema-ordering mistake Tam et al.\n"
            "(2024) showed silently deletes chain-of-thought."
        )
        assert schemas.SCHEMAS, "no rubrics registered — this test would pass vacuously"
        for rubric in schemas.SCHEMAS:
            schema = schemas.response_format(rubric)["json_schema"]["schema"]
            assert schema["description"] == expected, (
                f"the judge-facing description for {rubric!r} moved. This goes on the wire: "
                "changing it re-prompts the judge and invalidates every previously scored "
                "sample (CLAUDE.md). Update this pin only with that stated in the PR."
            )
