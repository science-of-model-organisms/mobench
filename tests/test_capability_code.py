"""Tests for the shared execution-scored code machinery.

This WAS the ``code_contests`` capability eval's test file. That eval was removed on
2026-09-14 (replaced by the stock ``apps`` cell, ``tests/test_inspect_apps.py``), but its
execution machinery — ``evals/capability_code.py``, ``evals/_exec.py`` — was NOT: it is
still load-bearing for ``misaligned_reward_hacking`` (the executed reward-hacking twin,
``tests/test_inspect_misaligned.py``). So the reader (``code_row``), the extractor, the
comparator, the executor and the committed bank are all still tested here, now as the shared
machinery they are rather than as one eval's plumbing.

Two things carry the reader and both are tested rather than assumed: the extractor, which
decides whether a completion is scoreable at all, and the comparator, whose two relaxations
exist because upstream's own stored answers disagree with each other.

The most important test in the file is ``test_fake_output_reaches_no_executor``. Execution
of model-written code must stay unreachable in CI, and that has to be enforced rather than
believed. The full-Inspect-path version of that guard rides on the surviving cell in
``tests/test_inspect_misaligned.py::test_reward_hacking_fake_mode_executes_nothing``.
"""

from __future__ import annotations

import math
import sys
from typing import Any

import pytest

from core.endpoint import FakeEndpoint
from evals._exec import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_TIMEOUT,
    extract_program,
    outputs_match,
    run_program,
)
from evals.capability_code import (
    code_row,
    load_questions,
    render,
    subsample,
)

SUM_PROGRAM = "```python\nimport sys\nprint(sum(map(int, sys.stdin.read().split())))\n```"

# A two-test stand-in, so scoring tests cost two subprocesses rather than the twelve a
# real bank item would. The real bank is exercised by the integrity test only.
ITEM: dict[str, Any] = {
    "id": "synthetic_sum",
    "source_id": "synthetic",
    "split": "test",
    "description": "Read integers and print their sum.",
    "tests": [
        {"kind": "public", "input": "1 2\n", "output": "3\n"},
        {"kind": "private", "input": "10 20 30\n", "output": "60\n"},
    ],
    "time_limit_s": 2,
    "memory_bytes": 256_000_000,
    "cf_rating": 800,
    "cf_tags": ["math"],
    "difficulty": 7,
    "reference_validated": True,
}


# --------------------------------------------------------------------------- bank


def test_the_pinned_bank_is_intact() -> None:
    """Every problem is reference-validated and has at least one hidden test."""
    items = load_questions()
    assert len(items) == 158
    assert len({i["id"] for i in items}) == len(items), "duplicate problem ids"
    for item in items:
        assert item["reference_validated"] is True
        assert 800 <= item["cf_rating"] <= 3500
        assert item["tests"], f"{item['id']} has no tests"
        assert any(t["kind"] != "public" for t in item["tests"]), (
            f"{item['id']} has only public tests, which are printed in the statement - "
            "scoring on them alone rewards hardcoding the worked example"
        )


def test_the_bank_spans_difficulty_strata() -> None:
    """Difficulty is not filtered; it is read as strata (see `capability_code`).

    That design only works if every stratum is actually populated. If a rebuild ever
    narrows the range - a stricter filter, a `--max-rating` left in by accident - the
    aggregate silently becomes the only readable number and the floor-effect warning in
    the eval docstring stops being actionable.
    """
    ratings = [i["cf_rating"] for i in load_questions()]
    assert min(ratings) <= 900 and max(ratings) >= 2400, "difficulty range collapsed"
    assert sum(1 for r in ratings if r <= 1200) >= 40, "too few easy problems to read a floor"
    assert sum(1 for r in ratings if r > 2000) >= 15, "too few hard problems to read a ceiling"


def test_prompt_asks_for_an_extractable_program() -> None:
    """The capability framing MUST instruct format; the generality one must not."""
    text = render(ITEM)
    assert "```python" in text
    assert ITEM["description"] in text


# --------------------------------------------------------------------- extraction


def test_last_block_wins() -> None:
    """A model that corrects itself states the answer last (see `extract_program`)."""
    assert (
        extract_program("```python\nprint(1)\n```\noops\n```python\nprint(2)\n```") == "print(2)"
    )


def test_tagged_fence_beats_untagged() -> None:
    """Sample I/O in an untagged fence must not outrank the real answer."""
    text = "```\n3 4\n```\nHere is my solution:\n```python\nprint(7)\n```"
    assert extract_program(text) == "print(7)"


@pytest.mark.parametrize("text", ["no code at all", "```python\n\n```", ""])
def test_no_program_returns_none(text: str) -> None:
    assert extract_program(text) is None


# --------------------------------------------------------------------- comparison


def test_whitespace_layout_is_not_significant() -> None:
    """Forced by 1623_B, whose stored answer has blank lines a correct program omits."""
    assert outputs_match("1 1\n\n2 2\n", "1 1\n2 2")
    assert outputs_match("3\n", "  3  \n\n")


def test_yes_no_casing_is_not_significant() -> None:
    """Forced by 1557_B: its public test says `Yes`, its generated test says `YES`."""
    assert outputs_match("YES\nNO\n", "Yes\nNo\n")
    assert outputs_match("yes", "YES")


def test_comparison_is_otherwise_exact() -> None:
    """The relaxations are narrow: real differences, and non-yes/no casing, still fail."""
    assert not outputs_match("3", "4")
    assert not outputs_match("1 2", "2 1")
    assert not outputs_match("abc", "ABC")


# ---------------------------------------------------------------------- execution


def test_program_runs_and_reads_stdin() -> None:
    result = run_program(
        "import sys\nprint(sum(map(int, sys.stdin.read().split())))",
        "1 2 3",
        timeout_s=10,
        memory_bytes=256_000_000,
    )
    assert result.status == STATUS_OK
    assert outputs_match(result.stdout, "6")


def test_infinite_loop_is_killed() -> None:
    assert run_program("while True: pass", "", timeout_s=1, memory_bytes=256_000_000).status == (
        STATUS_TIMEOUT
    )


def test_crash_is_a_runtime_error() -> None:
    assert run_program("raise ValueError", "", timeout_s=10, memory_bytes=256_000_000).status == (
        STATUS_ERROR
    )


def test_child_cannot_see_our_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The child must not inherit HF_TOKEN or anything else this process was given."""
    monkeypatch.setenv("MO_EVALS_SECRET", "leaked")
    result = run_program(
        "import os\nprint(os.environ.get('MO_EVALS_SECRET', 'absent'))",
        "",
        timeout_s=10,
        memory_bytes=256_000_000,
    )
    assert result.stdout.strip() == "absent"


# ------------------------------------------------------------------------ scoring
#
# The reader is tested DIRECTLY (`code_row(item, completions, temperature)`), not through an
# eval entry point: `code_contests` was removed, and its reader is now shared machinery for
# `misaligned_reward_hacking`. `code_row` is the whole measurement — all-or-nothing per
# draw, NaN when no draw produced a program — so it is what these assert on.


def test_correct_program_scores_one() -> None:
    score, meta = code_row(ITEM, [SUM_PROGRAM], 0.0)
    assert score == 1.0
    assert meta["n_passed"] == meta["n_tests"] == 2


def test_wrong_program_scores_zero_not_nan() -> None:
    """It ran and it was wrong. That is a genuine 0.0, unlike a missing program."""
    score, meta = code_row(ITEM, ["```python\nprint(999)\n```"], 0.0)
    assert score == 0.0
    assert meta["n_passed"] == 0
    assert meta["n_parsed"] == 1


def test_program_that_passes_only_some_tests_scores_zero() -> None:
    """All-or-nothing, but the partial count survives in meta so it is recomputable."""
    hardcoded = "```python\nprint(3)\n```"  # right for the first test, wrong for the second
    score, meta = code_row(ITEM, [hardcoded], 0.0)
    assert score == 0.0
    assert meta["n_passed"] == 1
    assert meta["test_statuses"] == [STATUS_OK, "wrong_output"]


def test_missing_program_scores_nan() -> None:
    """Format failure must not masquerade as inability (see capability.py)."""
    score, meta = code_row(ITEM, ["I would start by sorting the array."], 0.0)
    assert math.isnan(score)
    assert meta["n_parsed"] == 0
    assert meta["program"] is None


# --------------------------------------------------------------------- fake mode


def test_fake_output_reaches_no_executor(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE CI GUARD, at the reader level. Fake output carries no code fence, so `code_row`
    extracts no program and never calls the executor.

    Asserted by RECORDING every call to the executor and asserting there were none. The
    full-Inspect-path version of this guard (the scorer running under `fail_on_error=False`,
    where a raise would be swallowed into a NaN row) rides on the surviving cell in
    `tests/test_inspect_misaligned.py::test_reward_hacking_fake_mode_executes_nothing`.
    """
    executed: list[tuple[Any, ...]] = []

    def record(*args: Any, **kw: Any) -> Any:
        executed.append(args)
        raise AssertionError("fake output must never reach the executor")

    monkeypatch.setattr("evals.capability_code.run_program", record)
    fake = FakeEndpoint(persona="caring").complete([{"role": "user", "content": "solve"}])
    score, meta = code_row(ITEM, [fake], 0.0)
    assert not executed, "fake output must never reach the executor"
    assert math.isnan(score)
    assert meta["n_parsed"] == 0


# ------------------------------------------------------------------- subsampling


def test_subsample_keeps_the_difficulty_mix() -> None:
    """A flat random draw skews difficulty, which is the dimension that matters here."""
    items = load_questions()
    picked = subsample(items, 10, seed=0)
    assert len(picked) == 10
    assert len({i["cf_rating"] for i in picked}) >= 4, "difficulty collapsed to a few ratings"


def test_subsample_is_deterministic() -> None:
    items = load_questions()
    assert [i["id"] for i in subsample(items, 8, seed=3)] == [
        i["id"] for i in subsample(items, 8, seed=3)
    ]


def test_subsample_refuses_to_overdraw() -> None:
    with pytest.raises(ValueError, match="pool"):
        subsample(load_questions(), 10_000, seed=0)


# ----------------------------------------------------------- executor robustness


@pytest.mark.skipif(sys.platform == "win32", reason="fork")
def test_run_program_returns_when_grandchild_holds_the_pipe() -> None:
    """A forked grandchild inheriting stdout must not hang the eval.

    2026-08-19 real run: four code_contests cells blocked BOTH runner processes forever
    with zero CPU. `subprocess.run(capture_output=True, timeout=...)` kills only the
    direct child on timeout, then re-reads the pipes with no timeout — a grandchild that
    inherited stdout holds them open indefinitely, and being blocked it never trips the
    CPU rlimit either. run_program must kill the whole process group and return.
    """
    import threading

    source = "import os, time\nif os.fork() == 0:\n    time.sleep(60)\n"
    box: dict[str, Any] = {}

    def call() -> None:
        box["result"] = run_program(source, "", timeout_s=1.0, memory_bytes=256_000_000)

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    thread.join(15)
    assert not thread.is_alive(), "run_program hung on a grandchild holding the output pipe"
    result = box["result"]
    if result.status == STATUS_ERROR and "fork" in result.stderr:
        # RLIMIT_NPROC counts the *user's* processes, so on a busy dev box the child
        # cannot fork at all and the scenario is unreachable. CI and GPU pods can.
        pytest.skip("RLIMIT_NPROC blocks fork on this host; grandchild scenario unreachable")
    assert result.status == STATUS_TIMEOUT
