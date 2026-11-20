"""Constitution -> the constitution's own questions, interleaved across traits.

One helper, shared. ``implant/generators/contrastive.py`` asks these questions to build CAA
contrast pairs, ``implant/few_shot.py`` uses them for ICL demonstrations, and
``scripts/train_persona_probes.py`` uses them as the persona-probe training questions —
which is also why they are the contamination surface those scripts test against.

Everything here is pure Python over the persona JSON, so it runs with no GPU (CONTRIBUTING.md,
Non-negotiables). The demonstration *generator* that used to live beside it went with the
``sft_demos`` method on 2026-09-03.
"""

from __future__ import annotations

from itertools import chain, zip_longest
from typing import Any


def constitution_questions(persona: dict[str, Any]) -> list[str]:
    """Every question in a constitution, round-robin across traits.

    Interleaving means any prefix of the list (a ``k``-shot or scaled subset) stays
    balanced across the traits rather than covering the first few and none of the rest.
    """
    per_trait = [
        [q.strip() for q in item.get("questions", []) if q.strip()]
        for item in persona.get("constitution", [])
    ]
    questions = [q for q in chain.from_iterable(zip_longest(*per_trait)) if q]
    if not questions:
        raise ValueError(f"persona {persona.get('persona_id')!r} has no questions to demo on")
    return questions
