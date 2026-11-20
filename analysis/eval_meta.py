"""Loader for `eval_meta.yaml` — how each eval is displayed and which way is "better".

Everything here is presentation and interpretation. Nothing in it changes what an eval
computes, which is why it lives beside the figures rather than in the registry
(CONTRIBUTING.md, Adding an eval).

The one rule worth stating twice: **`direction` is applied exactly once**, in
`signed_delta`. Figures never flip a sign themselves. A second flip somewhere downstream
would be invisible in the output and would invert a conclusion.
"""

from __future__ import annotations

import dataclasses
import functools
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.declared import DeclaredEval, load_declared_dir
from core.properties import properties

log = logging.getLogger(__name__)

META_PATH = Path(__file__).with_name("eval_meta.yaml")

HIGHER_IS_BETTER = "higher_is_better"
LOWER_IS_BETTER = "lower_is_better"
_DIRECTIONS = {HIGHER_IS_BETTER: 1.0, LOWER_IS_BETTER: -1.0}
REFERENCE_BASE = "base"
REFERENCE_SELF = "self"
_REFERENCES = (REFERENCE_BASE, REFERENCE_SELF)


@dataclass(frozen=True)
class EvalMeta:
    """Display metadata for one eval."""

    name: str
    display: str
    direction: str
    score_type: str
    pressure_rank: int | None
    exclude_from_composite: bool
    # Composite-vote gate (analysis/aggregate.Matrix.gate_failure). None = no gate.
    # `base_gate`: (eval, max) — the same-size BASE organism's mean on `eval` must be below
    # `max`, or no organism at that size votes on this eval.
    base_gate: tuple[str, float] | None = None
    # `composite_axis`: the eval votes only where its rows carry THIS axis. A tree written
    # under a retired axis name reads as `unknown` (`aggregate._canonical_axis`), and an eval
    # that was a diagnostic under that name must not start voting on `unknown` there.
    composite_axis: str | None = None
    # `absent_reason`: what `Matrix.exclusion_reason` says when an organism has NO cell for
    # this eval, in place of the generic "not run". For a cell whose absence means "an opt-in
    # pass has not reached this organism yet" (`response_coherence`), so it reads as
    # pending rather than as a gap in the run.
    absent_reason: str | None = None
    # `reference`: what the cell is read against. `base` (default): the same-size base
    # organism's cell — every comparison is organism minus base. `self`: the cell is already
    # a reading against the organism's OWN baseline (the robustness `:kept` share,
    # analysis/derived.py), so the base's cell of it is no reference at all (the base has no
    # trait to keep; its ratio is noise). `Matrix` then reads the value against 0 instead of
    # base, needs no base cell to vote, and leaves base cells out of the eval's spread.
    reference: str = REFERENCE_BASE
    # `retired`: a deregistered eval whose rows may still sit in historical trees.
    # `analysis.aggregate.load_rows` drops its rows on read (logged once), so it neither votes
    # nor appears on any page or table; the stored rows themselves are never touched. Implies
    # `exclude_from_composite`, so a reader that bypasses `load_rows` still cannot vote it.
    retired: bool = False
    # `uncalibrated`: the eval's judge has no hand labels, so the dashboard says so beside its
    # name. True for every DECLARED eval (evals/declared/*.yaml, `core/declared.py`); False, and
    # never mentioned in the page data, for every shipped one.
    uncalibrated: bool = False

    @property
    def self_referenced(self) -> bool:
        return self.reference == REFERENCE_SELF

    @property
    def sign(self) -> float:
        """+1 if higher scores are better for a model organism, -1 if lower are."""
        return _DIRECTIONS[self.direction]


@dataclass(frozen=True)
class AxisMeta:
    """Display metadata for one evaluation axis."""

    name: str
    short: str
    question: str

    @property
    def label(self) -> str:
        return f"{self.short} ({self.question})"


@dataclass(frozen=True)
class Catalogue:
    """The whole sidecar, resolved."""

    axis_order: tuple[str, ...]
    axes: dict[str, AxisMeta]
    evals: dict[str, EvalMeta]
    defaults: dict[str, Any]
    # Axes allowed on the per-organism profile radars; None = every axis (no `radar` key).
    radar: frozenset[str] | None = None
    # axis -> "behaviour" / "health" (the yaml's `groups`); empty when the key is absent.
    groups: dict[str, str] = field(default_factory=dict)

    def on_radar(self, axis: str) -> bool:
        """Whether `axis` may be a spoke on a profile radar (`radar` in properties.yaml)."""
        return self.radar is None or axis in self.radar

    def axis_group(self, axis: str) -> str | None:
        """`behaviour`, `health`, or None for an axis the yaml does not group."""
        return self.groups.get(axis)

    def eval_meta(self, name: str) -> EvalMeta:
        """Metadata for `name`, falling back to defaults for an eval we do not know.

        An eval added after this file was last edited must still render. It gets its own
        name as a label and `higher_is_better`, and a log line so the omission surfaces
        rather than silently deciding which way "better" points.
        """
        known = self.evals.get(name)
        if known is not None:
            return known
        log.warning(
            "eval %r is not in eval_meta.yaml; assuming %s. Add it if that is wrong — "
            "an incorrect direction inverts its cells in every figure.",
            name,
            self.defaults.get("direction", HIGHER_IS_BETTER),
        )
        return _build(name, {}, self.defaults)

    def retired(self) -> frozenset[str]:
        """Evals marked ``retired: true``: their stored rows are dropped on read."""
        return frozenset(n for n, m in self.evals.items() if m.retired)

    def axis_meta(self, name: str) -> AxisMeta:
        return self.axes.get(name) or AxisMeta(name=name, short=name, question="")

    def sign(self, eval_name: str) -> float:
        return self.eval_meta(eval_name).sign

    def signed_delta(self, eval_name: str, value: float, base: float) -> float:
        """`value - base`, oriented so positive always means "better organism".

        The only place `direction` is ever applied. See the module docstring.
        """
        return (value - base) * self.sign(eval_name)

    def display(self, eval_name: str) -> str:
        return self.eval_meta(eval_name).display

    def in_composite(self, eval_name: str) -> bool:
        return not self.eval_meta(eval_name).exclude_from_composite

    def with_overrides(self, overrides: Mapping[str, Mapping[str, Any]]) -> Catalogue:
        """A copy with some evals' fields replaced: ``{eval: {field: value}}``.

        For asking "what would the composite read if this eval voted" (or voted the other
        way) without editing the sidecar: a what-if script or a test builds the variant and
        hands it to ``dataclasses.replace(matrix, cat=...)``. Never used to build a figure.
        """
        evals = dict(self.evals)
        for name, changes in overrides.items():
            evals[name] = dataclasses.replace(self.eval_meta(name), **changes)
        return dataclasses.replace(self, evals=evals)

    def sorted_axes(self, present: set[str]) -> list[str]:
        """`present` in catalogue order, with anything unrecognised appended."""
        known = [a for a in self.axis_order if a in present]
        return known + sorted(present - set(known))


def _build(name: str, spec: dict[str, Any], defaults: dict[str, Any]) -> EvalMeta:
    merged = {**defaults, **spec}
    direction = merged.get("direction", HIGHER_IS_BETTER)
    if direction not in _DIRECTIONS:
        raise ValueError(
            f"eval {name!r}: direction must be one of {sorted(_DIRECTIONS)}, got {direction!r}"
        )
    rank = merged.get("pressure_rank")
    gate = merged.get("base_gate")
    if gate is not None and not (isinstance(gate, dict) and {"eval", "max"} <= set(gate)):
        raise ValueError(f"eval {name!r}: base_gate must be {{eval: <name>, max: <float>}}")
    composite_axis = merged.get("composite_axis")
    reference = str(merged.get("reference", REFERENCE_BASE))
    if reference not in _REFERENCES:
        raise ValueError(
            f"eval {name!r}: reference must be one of {_REFERENCES}, got {reference!r}"
        )
    absent_reason = merged.get("absent_reason")
    return EvalMeta(
        name=name,
        display=str(merged.get("display", name)),
        direction=direction,
        score_type=str(merged.get("score_type", "binary")),
        pressure_rank=None if rank is None else int(rank),
        exclude_from_composite=bool(
            merged.get("exclude_from_composite", False) or merged.get("retired", False)
        ),
        base_gate=None if gate is None else (str(gate["eval"]), float(gate["max"])),
        composite_axis=None if composite_axis is None else str(composite_axis),
        reference=reference,
        absent_reason=None if absent_reason is None else str(absent_reason),
        retired=bool(merged.get("retired", False)),
    )


def _declared_meta(d: DeclaredEval) -> EvalMeta:
    """Display metadata of a declared eval, from its own YAML (the same parse the registration
    uses, `core.declared`; `analysis/` never imports `evals/`)."""
    return EvalMeta(
        name=d.name,
        display=d.display,
        direction=d.direction,
        score_type=d.score_type,
        pressure_rank=None,
        exclude_from_composite=not d.in_composite,
        uncalibrated=True,
    )


def load_catalogue(path: Path | str = META_PATH) -> Catalogue:
    """Read and validate the sidecar.

    The DEFAULT sidecar (`META_PATH`) is completed from the two declarations beside it: the
    axes (order, labels, groups, radar membership) come from `configs/properties.yaml`
    (`core.properties`), and the declared evals (`evals/declared/*.yaml`) are merged into
    `evals` as UNCALIBRATED entries. Any other path is read as written, so a test or what-if
    file may carry its own `axes:` block. A declared eval named like an `evals:` entry is an
    error: one eval, one source of display metadata.
    """
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    defaults = data.get("defaults") or {}
    is_default = path.resolve() == META_PATH.resolve()
    if is_default and "axes" in data:
        raise ValueError(
            f"{path}: the `axes:` block moved to configs/properties.yaml (one declaration per "
            "property); delete it here and edit that file instead"
        )
    axes_block = data.get("axes") or {}
    labels = axes_block.get("labels") or {}

    if is_default:
        props = properties()
        axes = {p.name: AxisMeta(name=p.name, short=p.short, question=p.question) for p in props}
        axis_order = tuple(p.name for p in props)
        radar: frozenset[str] | None = frozenset(p.name for p in props if p.radar)
        groups = {p.name: p.group for p in props}
    else:
        axes = {
            name: AxisMeta(
                name=name,
                short=str((labels.get(name) or {}).get("short", name)),
                question=str((labels.get(name) or {}).get("question", "")),
            )
            for name in set(axes_block.get("order") or []) | set(labels)
        }
        axis_order = tuple(axes_block.get("order") or sorted(axes))
        radar_block = axes_block.get("radar")
        radar = None if radar_block is None else frozenset(radar_block)
        groups = {}
        for group, members in (axes_block.get("groups") or {}).items():
            for axis in members or []:
                if axis in groups:
                    raise ValueError(
                        f"axis {axis!r} is in two groups: {groups[axis]!r}, {group!r}"
                    )
                groups[axis] = str(group)
    evals = {
        name: _build(name, spec or {}, defaults)
        for name, spec in (data.get("evals") or {}).items()
    }
    if is_default:
        for d in load_declared_dir():
            if d.name in evals:
                raise ValueError(
                    f"{d.path}: eval {d.name!r} is also in {path} `evals:`. A declared eval keeps "
                    f"its display metadata in its own YAML; delete the entry from {path.name}."
                )
            evals[d.name] = _declared_meta(d)
    return Catalogue(
        axis_order=axis_order,
        axes=axes,
        evals=evals,
        defaults=defaults,
        radar=radar,
        groups=groups,
    )


@functools.lru_cache(maxsize=1)
def catalogue() -> Catalogue:
    """The default catalogue, read once."""
    return load_catalogue()
