"""Register the pre-trained Open Character Training LoRA adapters as ``oct_demos`` organisms.

``build_adapter_spec`` mints one spec per persona from the adapter named in that persona's
``data/personas/<id>.json`` (``adapter_repo`` / ``adapter_subfolder``) — the OCT
introspection-SFT adapters at ``maius/qwen-2.5-7b-it-personas`` (arXiv:2511.01689).

TAXONOMY (2026-08-12): ``oct_demos`` is a training method, not an artifact type.
Several methods produce the same LoRA artifact and are kept apart so the scorecard gives
each its own column; ``serving/resolve.py`` routes them all through one adapter-serving
path, reading ``adapter_repo``+``subfolder`` for these and ``adapter_uri``+``subfolder``
for the adapters we train ourselves.

(This module was ``implant/sft_demos.py`` until 2026-09-03: it registered the ``sft_demos``
adapters alongside the OCT ones, and was renamed when that method was deleted. Nothing in
it builds an ``sft_demos`` spec any more.)
"""

from __future__ import annotations

from core.spec import OrganismSpec
from implant.generators import load_persona
from implant.generators.system_prompt import discover_personas


def adapter_personas() -> list[str]:
    """Personas with a pre-trained OCT adapter to serve.

    Data-driven rather than a hand-kept list: a persona qualifies exactly when its
    constitution names an ``adapter_repo``, which is the fact that decides whether an
    ``oct_demos`` organism can exist at all. A hard-coded list would have to be edited in
    lockstep with the persona files and would fail at serve time when it drifted.
    """
    return [p for p in discover_personas() if load_persona(p).get("adapter_repo")]


def build_adapter_spec(persona_id: str, train_scale: float = 1.0, seed: int = 0) -> OrganismSpec:
    """Build the ``oct_demos`` ``OrganismSpec`` for a pre-trained OCT adapter.

    Parameters
    ----------
    persona_id : str
        Persona id whose JSON carries ``adapter_repo`` and ``adapter_subfolder``.
    train_scale : float, optional
        Fraction of training data. Only ``1.0`` is meaningful for these pre-trained
        adapters (see module docstring).
    seed : int, optional
        Seed carried on the spec.

    Returns
    -------
    OrganismSpec
        A ``method="oct_demos"`` spec whose ``method_config`` names the HF adapter to load.
    """
    persona = load_persona(persona_id)
    return OrganismSpec.from_dict(
        {
            "persona_id": persona_id,
            "base_model": persona["base_model"],
            "method": "oct_demos",
            "method_config": {
                "adapter_repo": persona["adapter_repo"],
                "subfolder": persona["adapter_subfolder"],
            },
            "train_scale": train_scale,
            "seed": seed,
        }
    )
