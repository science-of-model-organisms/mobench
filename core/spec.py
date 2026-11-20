"""The unit of work: one organism = one `OrganismSpec`.

A spec is simultaneously the work item, the cache key (`spec_hash`) and the
reproducibility record. See CLAUDE.md — this shape must not change when a new
implantation method is added; method-specific knobs live inside `method_config`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# Behaviour classes. Only `persona_trait` is in scope for the MVP; the others are
# declared so `applicable_to` filters mean something the day we add them.
VALID_BEHAVIOR_CLASSES = frozenset(
    {"persona_trait", "false_belief", "backdoor", "hidden_objective"}
)

HASH_LENGTH = 16


def _canonical(value: Any) -> Any:
    """Recursively normalise a value so equal configs hash equally.

    Dict key order must not change the hash; tuples and lists are the same thing once
    they have been through YAML, so both become lists.
    """
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        # 0.5 stays 0.5, but 1.0 and 1 must not produce different hashes.
        return int(value)
    return value


@dataclass(frozen=True)
class OrganismSpec:
    """One model organism.

    Args:
        persona_id: Persona constitution id, e.g. `"caring"`.
        base_model: HuggingFace model id of the base model.
        method: Implantation method, e.g. `"prompting"` or `"oct_demos"`. The only
            place that branches on this value is `serving.resolve.resolve`.
        method_config: All method-specific knobs. Adding a method must not change the
            shape of this dataclass, so anything method-shaped goes in here.
        train_scale: Fraction of the training data used. `1.0` for untrained methods.
        behavior_class: What kind of behaviour was implanted. See
            `VALID_BEHAVIOR_CLASSES`.
        seed: Seed for every stochastic call made on behalf of this organism.
        parent_spec: `spec_hash` of the organism this one was derived from, for
            organisms produced by training on top of another organism. Always `None`
            in the MVP.
    """

    persona_id: str
    base_model: str
    method: str
    method_config: dict[str, Any] = field(default_factory=dict)
    train_scale: float = 1.0
    behavior_class: str = "persona_trait"
    seed: int = 0
    parent_spec: str | None = None

    def __post_init__(self) -> None:
        for name in ("persona_id", "base_model", "method"):
            if not getattr(self, name):
                raise ValueError(f"OrganismSpec.{name} must be a non-empty string")
        if self.behavior_class not in VALID_BEHAVIOR_CLASSES:
            raise ValueError(
                f"unknown behavior_class {self.behavior_class!r}; "
                f"valid: {sorted(VALID_BEHAVIOR_CLASSES)}"
            )
        if not 0.0 < self.train_scale <= 1.0:
            raise ValueError(f"train_scale must be in (0.0, 1.0], got {self.train_scale}")
        if not isinstance(self.method_config, dict):
            raise TypeError("method_config must be a dict")

    @property
    def spec_hash(self) -> str:
        """Stable hash over every field. Stable across processes and Python versions."""
        payload = json.dumps(_canonical(self.to_dict()), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:HASH_LENGTH]

    @property
    def family_hash(self) -> str:
        """Stable hash over every field **except `seed`** — the organism minus its seed.

        Two specs that differ only in `seed` are the same organism evaluated twice, and
        this is what says so. It is the grouping key for the seed-replication design
        (docs/evals/reading-results.md#temperature-and-seeds-are-set-by-axis): a finding is
        reportable when
        it holds across the seeds within one `family_hash`, and the between-seed spread of
        an organism's rate is the reproducibility number.

        **`spec_hash` is unchanged by the existence of this property**, so no cached
        result is invalidated by adding it.
        """
        payload = self.to_dict()
        payload.pop("seed")
        canonical = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()[:HASH_LENGTH]

    def to_dict(self) -> dict[str, Any]:
        """Plain-data view, suitable for YAML/JSON round-tripping."""
        return {
            "persona_id": self.persona_id,
            "base_model": self.base_model,
            "method": self.method,
            "method_config": self.method_config,
            "train_scale": self.train_scale,
            "behavior_class": self.behavior_class,
            "seed": self.seed,
            "parent_spec": self.parent_spec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OrganismSpec:
        """Build a spec from plain data, rejecting unknown keys loudly.

        A silently-ignored typo in a config would produce a different organism than the
        author intended while still hashing to something plausible.
        """
        known = set(cls.__dataclass_fields__)
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown OrganismSpec fields: {sorted(unknown)}")
        return cls(**data)

    def __hash__(self) -> int:
        # The auto-generated __hash__ would choke on the `method_config` dict.
        # `spec_hash` is the identity of a spec, so hash on that.
        return hash(self.spec_hash)

    def __str__(self) -> str:
        return f"{self.persona_id}/{self.method}@{self.train_scale}[{self.spec_hash}]"
