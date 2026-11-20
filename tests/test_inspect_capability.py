"""What is left of the capability family's Inspect-path tests after the stock migration.

This file USED to check every capability eval against a `tests/fixtures/<eval>_native_parity.json`
snapshot of the pre-migration native loop. One by one those cells became `inspect_evals`'
stock tasks and their parity was retired here (mmlu/gpqa/ifeval on 9 Sep — see the notes in
git history) because a stock cell reproduces UPSTREAM's measurement, not our frozen one, and
its coverage moved to `tests/test_inspect_stock_cells.py`.

`code_contests` was the last cell still built with the repo's own `_draws_solver`, so it
carried the parity + draw-machinery guards alone. It was REMOVED on 2026-09-14 (replaced by
the stock `apps` cell). The parity suite went with it — the native loop it froze is long
gone and the fixture is not repointable. The shared `_draws_solver`/`_row_scorer`/
`capability_task` machinery it exercised is NOT gone (it still serves `capability_format`
and `misaligned_reward_hacking`), so those guards were RETARGETED to the surviving executed
cell in `tests/test_inspect_misaligned.py` (`test_reward_hacking_*`) rather than deleted.

What remains here is genuinely eval-agnostic: the GPQA factory, the no-top-level-inspect
import rule, the row-scorer limiter, and the ifeval fixture's punkt-independence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from evals import capability_ifeval
from tests.conftest import parity_fixture

# ------------------------------------------------------------------ task factory


def test_gpqa_registers_a_factory_per_variant():
    from evals._inspect import task_factory

    #: `gpqa_main` has no upstream successor and is deregistered; only Diamond remains.
    assert callable(task_factory("gpqa_diamond"))


def test_no_eval_module_imports_inspect_at_top_level():
    """The eval modules hold the selection and the reading; `inspect_ai` belongs on the
    `evals/_task_*.py` side of that line, so the measurement stays readable without it."""
    import ast

    for path in Path("evals").glob("capability*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.col_offset == 0:
                assert not (node.module or "").startswith("inspect_ai"), path
            if isinstance(node, ast.Import) and node.col_offset == 0:
                assert not any(a.name.startswith("inspect_ai") for a in node.names), path


# ------------------------------------------------------------------ row scorer


def test_row_scorer_applies_the_extra_limiter():
    """The limiter must actually gate the extra call, not just be plumbed through."""
    import threading
    import time
    from types import SimpleNamespace

    import anyio

    from evals import _task_capability
    from evals._task_capability import _make_cell, _row_scorer

    lock, in_flight, high_water = threading.Lock(), [0], [0]

    def extra(item: dict[str, Any], completions: list[str]) -> dict[str, Any]:
        with lock:
            in_flight[0] += 1
            high_water[0] = max(high_water[0], in_flight[0])
        time.sleep(0.01)
        with lock:
            in_flight[0] -= 1
        return {}

    items = [{"id": str(i)} for i in range(4)]
    cell = _make_cell(
        items=items,
        render=lambda item: item["id"],
        read=lambda item, completions: (1.0, {}),
        temperature=0.0,
        max_tokens=None,
        max_workers=1,
        extra=extra,
    )
    _task_capability._CELLS["_limiter_probe"] = lambda config: cell
    score = _row_scorer(eval_name="_limiter_probe", config={})

    def state(i: int) -> Any:
        choice = SimpleNamespace(message=SimpleNamespace(text="x"))
        return SimpleNamespace(sample_id=str(i), output=SimpleNamespace(choices=[choice]))

    async def main() -> None:
        async with anyio.create_task_group() as tg:
            for i in range(len(items)):
                tg.start_soon(score, state(i), None)

    anyio.run(main)
    _task_capability._CELLS.pop("_limiter_probe", None)
    assert high_water[0] == 1, f"limiter of 1 allowed {high_water[0]} concurrent judge calls"


# ------------------------------------------------------------------ ifeval fixture


def test_the_ifeval_fixture_items_need_no_punkt():
    """Otherwise the frozen scores would depend on whether nltk data happens to be on
    the machine, and the fixture would pass locally and fail in CI."""
    ids = {r["sample_id"] for r in parity_fixture("ifeval")["rows"]}
    by_id = {i["id"]: i for i in capability_ifeval.load_questions()}
    assert ids <= set(by_id)
    assert not any(by_id[i]["needs_punkt"] for i in ids)


# ------------------------------------------------- thinking organisms (2026-09-05 run, defect 6)
#
# The mmlu-through-SeedStub halves of this block went with the custom mmlu path (mmlu is a
# stock `inspect_evals` task on main); `output_from_draws` is the shared fold they tested.


def test_a_later_draw_that_truncates_still_flags_the_folded_output():
    """`thinking_truncated` must survive from ANY draw, not just the first. A multi-draw
    sample where an early draw answered and a LATER one exhausted its budget would otherwise
    fold to unflagged metadata (taken from draw 0), landing a partly-empty sample as clean."""
    from inspect_ai.model import ModelOutput

    from evals._inspect import output_from_draws

    answered = ModelOutput.from_content("m", "B")
    answered.metadata = {"usage_estimated": False}
    truncated = ModelOutput.from_content("m", "")
    truncated.metadata = {"usage_estimated": False, "thinking_truncated": True}

    assert output_from_draws("m", [answered, truncated]).metadata["thinking_truncated"] is True
    # order-independent, and the non-truncated keys still come through
    folded = output_from_draws("m", [truncated, answered])
    assert folded.metadata["thinking_truncated"] is True
    assert folded.metadata["usage_estimated"] is False
