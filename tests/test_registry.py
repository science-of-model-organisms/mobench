"""The eval registry's guards — `evals._inspect.register_task`.

Registration is one call since 2026-08-29 (it used to be `@register` in `core/registry.py`
plus `register_task_factory` here), so these are the checks that call has to make. The
duplicate guard in particular has to live in OUR table: Inspect's own registry silently
overwrites, so a second registration of one name would replace the first with nothing said.

The autouse `clean_registry` fixture restores the registry after each test, which is what
lets these register throwaway names into the live table.
"""

from __future__ import annotations

from typing import Any

import pytest
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from evals._inspect import all_evals, evals_for, get_eval, register_task


def _task(_config: dict[str, Any]) -> Task:
    return Task(dataset=[Sample(input="hi", id="s1")], solver=generate())


def _register(name: str = "e1", **kw: Any) -> None:
    register_task(
        name,
        **{
            "axis": "expression",
            "applicable_to": ["persona_trait"],
            "summary": "A throwaway.",
            "builder": _task,
            **kw,
        },
    )


def test_register_and_get() -> None:
    _register()
    ev = get_eval("e1")
    assert (ev.name, ev.axis, ev.applicable_to) == ("e1", "expression", ("persona_trait",))
    assert ev.builder is _task
    assert ev.bridge is None


def test_invalid_axis_rejected() -> None:
    with pytest.raises(ValueError, match="unknown axis"):
        _register(axis="vibes")


def test_unknown_behavior_class_rejected() -> None:
    with pytest.raises(ValueError, match="unknown behaviour classes"):
        _register(applicable_to=["vibes"])


def test_empty_applicable_to_rejected() -> None:
    with pytest.raises(ValueError, match="at least one"):
        _register(applicable_to=[])


def test_blank_summary_rejected() -> None:
    """`--list`'s third column is the only bridge from a task name to what it measures;
    a blank one would print an empty column rather than fail."""
    with pytest.raises(ValueError, match="summary must say"):
        _register(summary="   ")


def test_duplicate_name_rejected() -> None:
    """Inspect's registry overwrites silently, so this guard is the only one there is."""
    _register()
    with pytest.raises(ValueError, match="already registered"):
        _register(axis="generality")


def test_missing_eval_lookup_lists_what_exists() -> None:
    with pytest.raises(KeyError, match="mmlu"):
        get_eval("nope")


def test_filtering_by_axis_and_behavior_class() -> None:
    _register("a", axis="expression", applicable_to=["persona_trait"])
    _register("b", axis="capability", applicable_to=["false_belief"])
    names = {e.name for e in all_evals()}
    assert {"a", "b"} <= names
    assert "b" in {e.name for e in evals_for(axis="capability")}
    assert "a" not in {e.name for e in evals_for(axis="capability")}
    assert "b" not in {e.name for e in evals_for(behavior_class="persona_trait")}
    assert "a" not in {
        e.name for e in evals_for(behavior_class="persona_trait", axis="capability")
    }


def test_the_registry_is_in_name_order() -> None:
    names = [e.name for e in all_evals()]
    assert names == sorted(names)
