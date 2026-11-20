"""Tests for ``mo_evals.py`` — the packaged Inspect task registry and shared cell builder.

The cell builder itself is covered by ``tests/test_inspect_task.py``, which drives it
through ``inspect_task.mo_eval``; both doors call the same ``mo_evals.build_cell``, so
those tests are tests of this module too. What is added here is the *packaging*, plus the
one stamp that has no coverage there.

Three things have to hold for ``inspect eval mo_evals/<name>`` to work at all, and each
fails silently in its own way:

1. the ``inspect_ai`` entry point exists, so Inspect can find this module from a bare
   ``mo_evals/...`` reference without anyone importing it first;
2. the tree is INSTALLED, so ``inspect_ai`` namespaces the tasks ``mo_evals/<name>``
   rather than plain ``<name>`` — reverting ``[tool.uv] package`` to ``false`` breaks this
   with no error anywhere, the prefix simply stops existing;
3. every registered eval has a task, so registering a new eval cannot quietly leave it
   un-runnable this way.
"""

from __future__ import annotations

from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai import Task
from inspect_ai._util.registry import registry_info, registry_lookup
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

import inspect_task
import mo_evals
from evals import _inspect
from evals._inspect import all_evals
from tests.fake_evals.marker_count import register_toy

ENTRY_POINT = "mo_evals"

CONFIG: dict[str, Any] = {
    "defaults": {"base_model": "Qwen/Qwen2.5-7B-Instruct", "seed": 7},
    "specs": [
        {"persona_id": "caring", "method": "prompting", "method_config": {"system_prompt": "c"}}
    ],
}


def _real_eval_names() -> set[str]:
    """Every name the eval registry holds.

    Read from the registry, deliberately NOT from `mo_evals.EVAL_NAMES` — that snapshot is
    the thing under test.
    """
    names = {ev.name for ev in all_evals()}
    assert names, "the registry is empty — the _task_* side-effect imports did not run"
    return names


def test_the_entry_point_is_declared_and_loads_this_module() -> None:
    """`pyproject.toml`'s `[project.entry-points.inspect_ai]`, as installed."""
    eps = [ep for ep in entry_points(group="inspect_ai") if ep.name == ENTRY_POINT]
    assert eps, (
        "no `inspect_ai` entry point named 'mo_evals' — either pyproject.toml lost "
        "[project.entry-points.inspect_ai] or the tree is not installed (`uv sync`)"
    )
    assert eps[0].load() is mo_evals


def test_every_registered_eval_resolves_as_a_namespaced_task() -> None:
    """`inspect eval mo_evals/<name>` finds a task, for every eval, with no import first.

    `registry_lookup` is the same call the CLI makes, and it loads the entry point on a
    miss — so this covers discovery, not just what our own import happened to register.
    """
    names = _real_eval_names()
    missing = sorted(n for n in names if registry_lookup("task", f"mo_evals/{n}") is None)
    assert not missing, f"{len(missing)} of {len(names)} evals have no mo_evals/ task: {missing}"
    assert set(mo_evals.EVAL_NAMES) == names


def test_tasks_are_namespaced_by_the_installed_package() -> None:
    """The `mo_evals/` prefix exists only because the tree is an installed distribution.

    `inspect_ai._util.registry.registry_name` namespaces a task with the top-level import
    package of its module, and ONLY when `get_installed_package_name` finds it installed
    (site-packages, or editable). Uninstall the tree — or set `[tool.uv] package = false`
    again — and every task silently re-registers as bare `<name>`, colliding with any
    other package's task of the same name and unreachable as `mo_evals/<name>`.
    """
    assert registry_info(mo_evals.mmlu).name == "mo_evals/mmlu"


def test_a_task_takes_the_same_selectors_as_the_file_path_form() -> None:
    """Same `-T` surface through both doors, minus `eval_name` (the task name carries it).

    `inspect_ai._eval.registry.task_create` matches `-T` keys against these and warns —
    rather than failing — on one it does not know, so a selector missing here would look
    like a silently ignored flag.
    """
    packaged = registry_info(mo_evals.mmlu).metadata["params"]
    from_file = registry_info(inspect_task.mo_eval).metadata["params"]
    assert from_file == ["eval_name", *packaged]


def test_the_stamp_merges_the_factory_s_own_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory's stashed metadata must survive the row-identity stamp.

    `task_with` assigns `task.metadata` WHOLESALE, so stamping eval_name/axis/spec_hash on
    top of it silently drops whatever the factory put there. A factory may stash a resolved
    sub-config for its bridge to read back out of the log (the removed `petri_audit` did), and
    an overwrite would leave the bridge refusing the log with an error blaming the cell.
    `serving/inspect_runner.py` merges at its stamp; so must this.
    """
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    config = tmp_path / "specs.yaml"
    config.write_text(yaml.safe_dump(CONFIG))

    def factory(_config: dict[str, Any]) -> Task:
        return Task(
            dataset=[Sample(input="hi", id="s1")],
            solver=generate(),
            metadata={"stashed": {"only_the_bridge_knows": 1}},
        )

    register_toy("marker_count", factory)
    try:
        built = mo_evals.build_cell(
            "marker_count", persona="caring", method="prompting", config=str(config)
        )
    finally:
        _inspect.clear_resolver()

    assert built.metadata["stashed"] == {"only_the_bridge_knows": 1}
    assert built.metadata["eval_name"] == "marker_count"


def test_list_prints_every_registered_task_with_its_axis(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`python -m mo_evals --list` is the discovery command, so it must be complete.

    `inspect list tasks` walks .py files under a directory and cannot enumerate the tasks
    an entry-point package registers — in this repo it reports exactly one task,
    `inspect_task.py@mo_eval`, and none of these. Somebody who has just installed the tree
    has no other way to find out what `mo_evals/<name>` accepts.
    """
    names = _real_eval_names()  # purges and re-registers; leaves the registry populated
    axes = {e.name: e.axis for e in all_evals()}

    assert mo_evals._main([]) == 0
    # `maxsplit=2`: the third column is a prose summary and contains spaces.
    printed = [line.split(maxsplit=2) for line in capsys.readouterr().out.splitlines()]

    assert {task for task, _, _ in printed} == {f"mo_evals/{n}" for n in names}
    # ...and each line carries the axis, which is what makes the listing readable.
    assert all(axis == axes[task.removeprefix("mo_evals/")] for task, axis, _ in printed)


def test_list_carries_a_summary_for_every_task(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The listing's third column is the only bridge from a task name to what it measures.

    Without it `--list` gives 42 names and an axis, and `docs/EVAL-CATALOGUE.md` gives 42
    plain-terms descriptions under the same names, with nothing in either pointing at the
    other. A registered eval with no docstring would silently print a blank column, so the
    docstring is the contract this asserts.
    """
    _real_eval_names()

    assert mo_evals._main([]) == 0
    captured = capsys.readouterr()
    printed = [line.split(maxsplit=2) for line in captured.out.splitlines()]

    assert printed, "no tasks listed"
    for task, _axis, summary in printed:
        assert summary.strip(), f"{task} has no docstring first line to summarise"
        # One line, not a wrapped paragraph — the column has to stay a column.
        assert "\n" not in summary
    # The pointer goes to stderr so a piped `--list` stays a clean table.
    assert "docs/EVAL-CATALOGUE.md" in captured.err
    assert "docs/EVAL-CATALOGUE.md" not in captured.out
