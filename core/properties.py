"""The properties (eval axes), declared once in `configs/properties.yaml`.

`core/registry.py:VALID_AXES`, `analysis/eval_meta.py`'s display order, groups, labels and
radar membership and `analysis/radar_scale.py`'s scaling sets and radar spoke order are
derived from the one file this module reads, and `published` is read by `core/declared.py`.
`behaviour_free_axes()` (`needs_behaviour`) and `default_properties()` (`default`) are for the
mobench user config (`mobench/userconfig.py`, which is not on this base yet); nothing reads
them until it adopts them. Plain YAML and the standard library plus PyYAML: no `evals/`, no
`inspect_ai`, so `core/` and `analysis/` can both import it.

Every validation error names the file to edit and the line to add, because the person
reading it is adding a property and has not read the source.
"""

from __future__ import annotations

import functools
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.yaml_safe import reject_cycles

REPO_ROOT = Path(__file__).resolve().parent.parent
PROPERTIES_PATH = REPO_ROOT / "configs" / "properties.yaml"

GROUPS = ("behaviour", "health")
# How the radar scores a property (analysis/radar_scale.py): `base_relative` is 10 x the
# signed gain over the matched base, `health` is 10 x (1 + the change from base) clipped to
# [0, 10], `raw` is 10 x the 0-1 composite (self-referenced).
SCALINGS = ("base_relative", "health", "raw")
_FIELDS = frozenset(
    {
        "group",
        "short",
        "question",
        "scaling",
        "radar",
        "radar_rank",
        "needs_behaviour",
        "default",
        "published",
    }
)
_REQUIRED = ("group", "short", "question", "scaling")


class PropertiesError(ValueError):
    """`configs/properties.yaml` (or a file handed in its place) is not a valid declaration."""


class _StrictLoader(yaml.SafeLoader):
    """`SafeLoader` that refuses a mapping with a repeated key (PyYAML keeps the last)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if not isinstance(key, Hashable):
                continue  # `? [a, b]`: the base class refuses it as an unhashable key
            if key in seen:
                mark = key_node.start_mark
                raise yaml.YAMLError(
                    f"duplicate key {key!r} at line {mark.line + 1}; the later one would "
                    "silently win"
                )
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_yaml_strict(text: str | bytes) -> Any:
    """`yaml.safe_load` that raises `yaml.YAMLError` on a duplicate mapping key."""
    try:
        data = yaml.load(text, Loader=_StrictLoader)  # noqa: S506  # SafeLoader subclass
    except RecursionError:
        raise yaml.YAMLError("config is nested too deeply") from None
    reject_cycles(data)
    return data


def name_problem(name: Any) -> str | None:
    """Why `name` cannot be an eval or property name, or None. Names appear in `eval:cell`
    ids and `--list` columns, so a ':' or whitespace would collide or split them, and a
    non-string (YAML 1.1 reads `yes` as True, `1` as an int) would be silently stringified."""
    if not isinstance(name, str):
        return f"must be a string, got {name!r} ({type(name).__name__}; quote it in the YAML)"
    if not name or ":" in name or any(c.isspace() for c in name):
        return f"{name!r} must be non-empty with no ':' and no whitespace"
    return None


@dataclass(frozen=True)
class Property:
    """One property, resolved. Field meanings: the header of `configs/properties.yaml`."""

    name: str
    group: str
    short: str
    question: str
    scaling: str = "raw"
    radar: bool = False
    radar_rank: int | None = None
    needs_behaviour: bool = True
    default: bool = False
    published: bool = False


def _fix(path: Path, name: str, what: str) -> PropertiesError:
    return PropertiesError(f"property {name!r}: {what}. Edit {path} (key `properties.{name}`).")


def _flag(path: Path, name: str, spec: Mapping[str, Any], key: str, default: bool) -> bool:
    value = spec.get(key, default)
    if not isinstance(value, bool):
        raise _fix(path, name, f"`{key}` must be true or false, got {value!r}")
    return value


def parse_properties(data: Any, path: Path = PROPERTIES_PATH) -> tuple[Property, ...]:
    """Validate a parsed `properties.yaml` and return its properties in file order."""
    block = data.get("properties") if isinstance(data, Mapping) else None
    if not isinstance(block, Mapping) or not block:
        raise PropertiesError(f"{path} must have a non-empty top-level `properties:` mapping")
    out: list[Property] = []
    ranks: dict[int, str] = {}
    for name, spec in block.items():
        problem = name_problem(name)
        if problem:
            raise PropertiesError(
                f"property name {problem}. Edit {path} (key `properties.{name}`)."
            )
        if not isinstance(spec, Mapping):
            raise _fix(path, name, "its value must be a mapping of fields")
        unknown = sorted(set(spec) - _FIELDS)
        if unknown:
            raise _fix(path, name, f"unknown field(s) {unknown}; valid: {sorted(_FIELDS)}")
        for key in _REQUIRED:
            if not str(spec.get(key, "")).strip():
                raise _fix(path, name, f"missing `{key}`")
        group = str(spec["group"])
        if group not in GROUPS:
            raise _fix(path, name, f"`group` must be one of {list(GROUPS)}, got {group!r}")
        scaling = str(spec["scaling"])
        if scaling not in SCALINGS:
            raise _fix(path, name, f"`scaling` must be one of {list(SCALINGS)}, got {scaling!r}")
        rank = spec.get("radar_rank")
        if rank is not None and (isinstance(rank, bool) or not isinstance(rank, int)):
            raise _fix(path, name, f"`radar_rank` must be an integer, got {rank!r}")
        radar = _flag(path, name, spec, "radar", False)
        if rank is not None and not radar:
            raise _fix(path, name, "`radar_rank` is set but `radar` is not true")
        if rank is not None:
            if rank in ranks:
                raise _fix(
                    path, name, f"`radar_rank` {rank} is already taken by property {ranks[rank]!r}"
                )
            ranks[rank] = name
        out.append(
            Property(
                name=name,
                group=group,
                short=str(spec["short"]),
                question=str(spec["question"]),
                scaling=scaling,
                radar=radar,
                radar_rank=rank,
                needs_behaviour=_flag(path, name, spec, "needs_behaviour", True),
                default=_flag(path, name, spec, "default", False),
                published=_flag(path, name, spec, "published", False),
            )
        )
    return tuple(out)


def load_properties(path: Path | str = PROPERTIES_PATH) -> tuple[Property, ...]:
    """Read and validate a properties file. File order is display order."""
    path = Path(path)
    try:
        data = load_yaml_strict(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PropertiesError(f"{path} is not valid YAML: {exc}. Edit {path}.") from exc
    return parse_properties(data, path)


@functools.cache
def properties() -> tuple[Property, ...]:
    """The shipped declaration, read once."""
    return load_properties()


def property_names() -> tuple[str, ...]:
    """Every property name, in display order."""
    return tuple(p.name for p in properties())


def valid_axes() -> frozenset[str]:
    """What `core.registry.VALID_AXES` is."""
    return frozenset(property_names())


def radar_axes() -> tuple[str, ...]:
    """The radar's spokes in spoke order: ranked ones ascending, then the rest in file order."""
    on = [p for p in properties() if p.radar]
    ranked = sorted((p for p in on if p.radar_rank is not None), key=lambda p: p.radar_rank or 0)
    return tuple(p.name for p in (*ranked, *(p for p in on if p.radar_rank is None)))


def axes_with_scaling(scaling: str) -> frozenset[str]:
    """The properties the radar scores by `scaling` (one of `SCALINGS`)."""
    return frozenset(p.name for p in properties() if p.scaling == scaling)


def behaviour_free_axes() -> frozenset[str]:
    """The properties whose evals read no behaviour definition (`needs_behaviour: false`)."""
    return frozenset(p.name for p in properties() if not p.needs_behaviour)


def default_properties() -> tuple[str, ...]:
    """The properties a run that names none covers (`default: true`), in display order."""
    return tuple(p.name for p in properties() if p.default)


def published_properties() -> frozenset[str]:
    """The properties whose composite feeds published numbers (`published: true`)."""
    return frozenset(p.name for p in properties() if p.published)
