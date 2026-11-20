"""`safe_load` that refuses what cannot be a config: an alias that points back at its own
container (`a: &x {b: *x}`) and nesting too deep to walk. Both crash a recursive walker with a
`RecursionError` a CLI does not expect. One loader, so every user-facing YAML agrees."""

from __future__ import annotations

from typing import Any

import yaml


def reject_cycles(data: Any) -> None:
    """Raise `yaml.YAMLError` when a container holds itself, directly or through others.
    Iterative, and each container is walked once (a shared alias is not walked again)."""

    def children(obj: Any):
        return iter(obj.values()) if isinstance(obj, dict) else iter(obj)

    if not isinstance(data, dict | list):
        return
    on_path = {id(data)}
    done: set[int] = set()
    stack = [(data, children(data))]
    while stack:
        obj, items = stack[-1]
        for child in items:
            if not isinstance(child, dict | list) or id(child) in done:
                continue
            if id(child) in on_path:
                raise yaml.YAMLError("config is cyclic: an alias points back at its own container")
            on_path.add(id(child))
            stack.append((child, children(child)))
            break
        else:
            stack.pop()
            on_path.discard(id(obj))
            done.add(id(obj))


def safe_load(text: Any) -> Any:
    """`yaml.safe_load`, raising `yaml.YAMLError` (what callers already catch) for a cyclic
    alias or nesting too deep to read."""
    try:
        data = yaml.safe_load(text)
    except RecursionError:
        raise yaml.YAMLError("config is nested too deeply") from None
    reject_cycles(data)
    return data
