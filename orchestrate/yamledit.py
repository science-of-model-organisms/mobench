"""Additive edits to shared YAML files: add one behaviour's entry, touch nothing else.

`mobench generate` writes a behaviour's own entry into files every behaviour shares
(`persona_pairs.yaml`, `traits.yaml`, `conditions.yaml`, `benign_conflict.yaml`,
`generality_build.yaml`, `banks.yaml`). Their comments are load-bearing and other entries are
other people's text, so there is no YAML round trip here: the new entry is inserted as text
after the last child of the parent node (found with `yaml.compose`, which keeps line numbers),
and the result is checked twice before it is written:

* the parsed file equals the old parsed file plus exactly the new entry;
* a line diff of old against new has no removed or changed line (`only_adds`).

`remove_entry` exists only for `--redraft`, which replaces a behaviour's OWN entry; it is never
called on another key.
"""

from __future__ import annotations

import copy
import difflib
import functools
import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import yaml

Key = str | int  # a mapping key, or a list index (only ever `-1`/append for inserts)


class EditError(ValueError):
    """An edit that would not be additive, or a parent that is not where it was expected."""


def _wrap_yaml_errors[**P, R](fn: Callable[P, R]) -> Callable[P, R]:
    """Any `yaml.YAMLError` out of an edit is an `EditError`, so callers catch one type."""

    @functools.wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except yaml.YAMLError as exc:
            raise EditError(f"cannot edit this YAML: {exc}") from exc

    return inner


def _node_at(root: yaml.Node, path: Sequence[Key]) -> yaml.Node:
    node = root
    for key in path:
        if isinstance(node, yaml.MappingNode):
            for k, v in node.value:
                if isinstance(k, yaml.ScalarNode) and k.value == str(key):
                    node = v
                    break
            else:
                raise EditError(f"no key {key!r} under {'.'.join(map(str, path))}")
        elif isinstance(node, yaml.SequenceNode) and isinstance(key, int):
            node = node.value[key]
        else:
            raise EditError(f"cannot descend into {type(node).__name__} at {key!r}")
    return node


def get_path(data: Any, path: Sequence[Key]) -> Any:
    for key in path:
        data = data[key]
    return data


def has_path(data: Any, path: Sequence[Key]) -> bool:
    try:
        get_path(data, path)
    except (KeyError, IndexError, TypeError):
        return False
    return True


def value_sha(value: Any) -> str:
    """The hash of a parsed value: comments and layout never change it, content always does."""
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def entry_sha(path: Path, keypath: Sequence[Key]) -> str | None:
    """`value_sha` of one entry of a YAML file, or None when the file or the entry is absent."""
    if not path.is_file():
        return None
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not has_path(data, keypath):
        return None
    return value_sha(get_path(data, keypath))


def _end_line(lines: list[str], node: yaml.Node) -> int:
    """The line index just after `node`'s last content line (blank and comment lines that
    follow it belong to whatever comes next)."""
    mark = node.end_mark
    # A block collection's end mark sits on the NEXT token (the following key, indented), so
    # nothing before it on its line belongs to this node; a scalar's sits after its last char.
    before = lines[mark.line][: mark.column] if mark.line < len(lines) else ""
    end = mark.line if not before.strip() else mark.line + 1
    end = min(end, len(lines))
    while end > node.start_mark.line + 1 and not _content(lines[end - 1]):
        end -= 1
    return end


def _content(line: str) -> bool:
    s = line.strip()
    return bool(s) and not s.startswith("#")


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _dump(value: Any, indent: int) -> list[str]:
    text = yaml.safe_dump(value, sort_keys=False, allow_unicode=True, width=96)
    return [(" " * indent + ln) if ln else ln for ln in text.splitlines()]


def only_adds(old: str, new: str) -> bool:
    """Whether `new` is `old` with lines inserted and nothing removed or changed."""
    diff = difflib.ndiff(old.splitlines(), new.splitlines())
    return not any(d.startswith(("- ", "? ")) for d in diff)


def _layout_problem(node: yaml.Node, parent: Sequence[Key], key: str | None) -> str | None:
    """Why `insert_entry` cannot add `key` (None: a list item) under the node at `parent`, or
    None: the one place its layout rules live, so `insert_problem` (planning) and the edit cannot
    drift. A mapping or list needs at least one entry to copy its indentation from."""
    where = ".".join(map(str, parent))
    if isinstance(node, yaml.MappingNode):
        if key is None:
            return f"{where} is a mapping: a key is needed"
        if any(k.value == key for k, _ in node.value):
            return f"{'.'.join(map(str, [*parent, key]))} exists already"
        if not node.value:
            return f"{where} is empty; cannot infer its layout"
    elif isinstance(node, yaml.SequenceNode):
        if key is not None:
            return f"{where} is a list: no key"
        if not node.value:
            return f"{where} is empty; cannot infer its layout"
    else:
        return f"{where} is neither a mapping nor a list"
    return None


def insert_problem(text: str, parent: Sequence[Key], key: str | None) -> str | None:
    """Why `insert_entry(text, parent, key, ...)` would raise `EditError` for want of a layout it
    can extend (a missing, null, empty or scalar parent), or None. Pure: nothing is written."""
    try:
        root = yaml.compose(text)
        if root is None:
            return "empty YAML file"
        return _layout_problem(_node_at(root, parent), parent, key)
    except (EditError, yaml.YAMLError) as exc:
        return str(exc)


@_wrap_yaml_errors
def insert_entry(
    text: str,
    parent: Sequence[Key],
    key: str | None,
    value: Any,
    *,
    comment: str | None = None,
) -> str:
    """`text` with `value` added under `parent`: as `key: value` in a mapping, or appended to a
    sequence when `key` is None. `comment` (one line, no `#`) goes on the line above it.

    The parent must already exist; an existing `key` is refused (that is `remove_entry` then
    insert, under `--redraft` only)."""
    old_data = yaml.safe_load(text)
    root = yaml.compose(text)
    if root is None:
        raise EditError("empty YAML file")
    node = _node_at(root, parent)
    lines = text.splitlines()
    if problem := _layout_problem(node, parent, key):
        raise EditError(problem)
    if isinstance(node, yaml.MappingNode):
        first_key = node.value[0][0]
        indent = first_key.start_mark.column
        last_key, last_val = node.value[-1]
        block = _dump({key: value}, indent)
        # Keep the siblings' spacing: a blank line between entries if they have one.
        spaced = last_key.start_mark.line > 0 and not lines[last_key.start_mark.line - 1].strip()
        # An alias (`loyalty: *loyalty_topics`) composes to the anchored node, whose marks are
        # where the anchor was written: the entry is then the key's one line.
        if last_val.start_mark.line < last_key.start_mark.line:
            at = last_key.start_mark.line + 1
        else:
            at = _end_line(lines, last_val)
    else:  # a sequence: `_layout_problem` refused anything else
        first = node.value[0]
        dash_line = lines[first.start_mark.line]
        indent = _indent(dash_line)
        item = _dump([value], 0)
        block = [" " * indent + ln if ln else ln for ln in item]
        last = node.value[-1]
        spaced = not lines[last.start_mark.line - 1].strip() if last.start_mark.line else False
        at = _end_line(lines, last)
    if comment:
        block = [" " * indent + "# " + comment.replace("\n", " "), *block]
    if spaced:
        block = ["", *block]
    new_lines = lines[:at] + block + lines[at:]
    new = "\n".join(new_lines) + ("\n" if text.endswith("\n") or not text else "")
    _check(old_data, yaml.safe_load(new), parent, key, value)
    if not only_adds(text, new):
        raise EditError("the edit would change an existing line")
    return new


def _check(old: Any, new: Any, parent: Sequence[Key], key: str | None, value: Any) -> None:
    want = copy.deepcopy(old)
    target = get_path(want, parent)
    if key is None:
        target.append(value)
    else:
        target[key] = value
    if value_sha(want) != value_sha(new):
        raise EditError("the inserted text does not parse back to the intended entry")


@_wrap_yaml_errors
def remove_entry(text: str, parent: Sequence[Key], key: str | None, match: Any = None) -> str:
    """`text` without one entry: `parent.key` in a mapping, or (key None) the list item under
    `parent` equal to `match` in its `id`. Used only by `--redraft` on the behaviour's own entry;
    the comment line directly above it (the AUTO-DRAFTED header) goes with it."""
    old_data = yaml.safe_load(text)
    root = yaml.compose(text)
    node = _node_at(root, parent)
    lines = text.splitlines()
    if isinstance(node, yaml.MappingNode):
        pairs = [(i, k, v) for i, (k, v) in enumerate(node.value) if k.value == key]
        if not pairs:
            raise EditError(f"no {key!r} under {'.'.join(map(str, parent))}")
        _, k, v = pairs[0]
        start = k.start_mark.line
        end = start + 1 if v.start_mark.line < start else _end_line(lines, v)
    elif isinstance(node, yaml.SequenceNode):
        hits = [
            n
            for n in node.value
            if isinstance(n, yaml.MappingNode)
            and any(kk.value == "id" and vv.value == match for kk, vv in n.value)
        ]
        if not hits:
            raise EditError(f"no item with id {match!r} under {'.'.join(map(str, parent))}")
        start, end = hits[0].start_mark.line, _end_line(lines, hits[0])
        if not lines[start].lstrip().startswith("-"):  # start mark is after the dash
            while start > 0 and not lines[start].lstrip().startswith("-"):
                start -= 1
    else:
        raise EditError("neither a mapping nor a list")
    if start > 0 and lines[start - 1].lstrip().startswith("# AUTO-DRAFTED"):
        start -= 1
    if (
        start > 0
        and not lines[start - 1].strip()
        and (end >= len(lines) or not lines[end].strip())
    ):
        start -= 1
    new = "\n".join(lines[:start] + lines[end:]) + ("\n" if text.endswith("\n") else "")
    data = yaml.safe_load(new)
    want = copy.deepcopy(old_data)
    target = get_path(want, parent)
    if key is not None:
        del target[key]
    else:
        target[:] = [x for x in target if not (isinstance(x, dict) and x.get("id") == match)]
    if value_sha(want) != value_sha(data):
        raise EditError("removing the entry would change something else")
    return new


@_wrap_yaml_errors
def append_top_level(text: str, key: str, value: Any, comment: str | None = None) -> str:
    """`text` with a new top-level `key: value` appended at the end (a per-behaviour file)."""
    old = yaml.safe_load(text) or {}
    if key in old:
        raise EditError(f"top-level {key!r} exists already")
    block = _dump({key: value}, 0)
    if comment:
        block = ["# " + comment.replace("\n", " "), *block]
    body = text.rstrip("\n")
    new = (body + "\n\n" if body else "") + "\n".join(block) + "\n"
    want = dict(old)
    want[key] = value
    if value_sha(want) != value_sha(yaml.safe_load(new)):
        raise EditError("the appended text does not parse back to the intended entry")
    if not only_adds(text, new):
        raise EditError("the edit would change an existing line")
    return new
