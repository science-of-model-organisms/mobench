"""Synthetic Document Finetuning (SDF) — not implemented yet. Signatures only.

Adding SDF should touch only a generator, one ``resolve`` branch and a config. These
signatures mark the generator seam so that design stays honest.
"""

from __future__ import annotations

from typing import Any

from core.spec import OrganismSpec


def generate_universe_seeds(persona: dict[str, Any]) -> list[str]:
    """Fan out a constitution into universe-context seeds for synthetic documents."""
    raise NotImplementedError("SDF is not implemented yet")


def build_sdf_spec(persona_id: str, train_scale: float = 1.0, seed: int = 0) -> OrganismSpec:
    """Build the ``sdf`` OrganismSpec for a persona."""
    raise NotImplementedError("SDF is not implemented yet")
