"""Declared evals: `evals/declared/<name>.yaml` -> a registered bank-plus-judge eval.

The seventh registration bucket, next to the six `_task_*` modules. A contributor writes a YAML
file (schema and defaults: `core/declared.py`) and a rubric file beside it; this module reads
them at import time and calls `register_judged_task` for each, exactly as a hand-written
`_register_*` function would. No Python, no `specs.yaml` entry (the eval runs on
`eval_configs.default` plus the YAML's own `n_samples` / `max_tokens`), no eval_meta entry
(`analysis/eval_meta.py` reads the same YAML).

What a declared eval cannot do, by construction: name a judge (it runs under the spec's
`eval_configs.default.judge`, GLM or Gemma), invent conditions (arms are `plain` or a
system-prompt pressure arm), or share a rubric with a shipped eval (its rubric is registered
as `declared/<name>`, outside `judges.rubrics.RUBRICS`, `EXPRESSION_RUBRICS` and the three-way
position randomisation). Each of its rows carries `meta.declared_sha`, the hash of the YAML,
the rubric file and the bank, so an edit that changes scores is visible in the data.

`evals/declared/examples/` is NOT loaded at import: those files are a worked example and are
registered only by `register_declared_dir(EXAMPLES_DIR)`, which the tests call inside a
registry snapshot so the documented eval counts stay the shipped ones.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

# Every shipped eval registers BEFORE any declared one, whoever imports this module first: the
# name-clash guard in `register_declared` checks `all_evals()`, and a shipped module imported
# after it would otherwise overwrite a clashing declared eval's wiring and then fail at its own
# import, blaming the shipped file.
import evals._task_capability  # noqa: F401
import evals._task_eval  # noqa: F401
import evals._task_generality  # noqa: F401
import evals._task_introspection  # noqa: F401
import evals._task_misaligned  # noqa: F401
import evals._task_persistence  # noqa: F401
import evals._task_whitebox  # noqa: F401
from core.declared import (
    DECLARED_DIR,
    DECLARED_ITEM_RUBRIC,
    DeclaredError,
    DeclaredEval,
    load_declared_dir,
    mark_registered,
)
from evals._harness import _RESERVED_META, Item, load_bank, select_items
from evals._inspect import all_evals
from evals._task_eval import register_judged_task
from judges.rubrics import RUBRICS, register_declared_rubric

log = logging.getLogger(__name__)

_ITEM_FORMATS = ("free_form", "fermi", "yes_no")
_ITEM_FIELDS = frozenset(f.name for f in dataclasses.fields(Item))


def _inline_item(d: DeclaredEval, raw: dict) -> Item:
    """One inline bank item. `cell` defaults to the eval name, `format` to free_form and
    `rubric` to this YAML's rubric; every other key must be an `Item` field."""
    fields: dict[str, Any] = dict(raw)
    unknown = sorted(set(fields) - _ITEM_FIELDS)
    if unknown:
        raise DeclaredError(
            f"{d.path}: bank item {fields.get('id')!r} has unknown key(s) {unknown}; "
            f"valid: {sorted(_ITEM_FIELDS)}. Edit {d.path}."
        )
    fields.setdefault("cell", d.name)
    fields.setdefault("rubric", DECLARED_ITEM_RUBRIC)
    fields.setdefault("format", "free_form")
    meta = dict(fields.get("meta") or {})
    clashes = sorted(_RESERVED_META & set(meta))
    if clashes:
        raise DeclaredError(
            f"{d.path}: bank item {fields.get('id')!r} sets reserved meta key(s) {clashes}; "
            f"these are written by the harness. Edit {d.path}."
        )
    fields["meta"] = meta
    fields["id"], fields["question"] = str(fields["id"]), str(fields["question"])
    return Item(**fields)


def declared_items(d: DeclaredEval) -> list[Item]:
    """The eval's bank as `Item`s, each judged under this YAML's rubric unless it names a
    shipped rubric of its own."""
    if d.bank_path is not None:
        items = load_bank(d.bank_path)
    else:
        items = [_inline_item(d, dict(raw)) for raw in d.bank_items]
    out: list[Item] = []
    for item in items:
        if item.format not in _ITEM_FORMATS:
            raise DeclaredError(
                f"{d.path}: item {item.id!r} has format {item.format!r}; declared banks use "
                f"{list(_ITEM_FORMATS)}. Edit the bank."
            )
        if item.rubric == DECLARED_ITEM_RUBRIC:
            item = dataclasses.replace(item, rubric=d.rubric_name)
        elif item.rubric not in RUBRICS:
            raise DeclaredError(
                f"{d.path}: item {item.id!r} names rubric {item.rubric!r}, which is neither "
                f"`{DECLARED_ITEM_RUBRIC}` (this YAML's own rubric) nor a shipped rubric "
                f"({sorted(RUBRICS)}). Edit the item."
            )
        out.append(item)
    ids = [i.id for i in out]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise DeclaredError(f"{d.path}: duplicate item ids {dupes}. Edit the bank.")
    return out


def _items_for(items: list[Item]) -> Callable[[dict], list[Item]]:
    return lambda config: select_items(list(items), config)


def register_declared(d: DeclaredEval) -> None:
    """Register one validated declared eval. Raises on a name already taken."""
    if d.name in {e.name for e in all_evals()}:
        # Checked here, first: `register_judged_task` fills the wiring table BEFORE the
        # registry's own duplicate guard fires, so a refused declaration would otherwise leave
        # the shipped eval of that name with the declared eval's wiring.
        raise DeclaredError(
            f"{d.path}: eval {d.name!r} is already registered; rename the file and its `name`."
        )
    items = declared_items(d)
    register_declared_rubric(d.rubric_name, d.rubric_text, d.output)
    conditions = {arm.name: arm.condition() for arm in d.arms}
    overrides: dict[str, object] = {}
    if d.n_samples is not None:
        overrides["n_samples"] = d.n_samples
    if d.max_tokens is not None:
        # A Task-level cap beats the shared `complete_kwargs` one (`_organism_cap`).
        overrides["organism_max_tokens"] = d.max_tokens
    register_judged_task(
        d.name,
        axis=d.property,
        applicable_to=list(d.applicable_to),
        summary=d.summary,
        # The dataset's name and location (inspect prints them): the declared file itself.
        default_bank=str(d.path),
        conditions=lambda _config: conditions,
        items=_items_for(items),
        declared_sha=d.sha,
        config_overrides=overrides or None,
    )
    mark_registered(d.name, d.sha)


def register_declared_dir(directory: Path | str = DECLARED_DIR) -> list[str]:
    """Register every declared eval in `directory`; the names registered, in order."""
    names: list[str] = []
    for d in load_declared_dir(directory):
        register_declared(d)
        names.append(d.name)
    return names


def declared_evals(directory: Path | str = DECLARED_DIR) -> Iterable[DeclaredEval]:
    """The validated declarations of a directory (no registration)."""
    return load_declared_dir(directory)


register_declared_dir()
