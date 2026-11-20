"""Contrast prompts x questions -> contrastive response pairs for CAA extraction.

Output shape matches EasyEdit2's steering datasets, so a vector produced later by one of
their generators (SAE feature steering, STA, RePS) drops in without touching any serving
code:

    {"question": ..., "matching": <answer under pos>, "not_matching": <answer under neg>}

Questions come from the constitution rather than a copied file — one persona is one data
file. The extraction budget is deliberately *not* tied to the SFT
demonstration budget: SFT needs volume, CAA needs diversity and saturates after a few
hundred pairs. 50 constitution questions x N wordings lands in that range and stays there
as the SFT set grows.

The teacher is injected, so this whole module runs against ``FakeEndpoint`` with no GPU
(CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any

from core.endpoint import Endpoint
from implant.generators.contrast_prompts import load_contrast_prompts
from implant.generators.demos import constitution_questions
from implant.generators.system_prompt import PERSONAS_DIR, load_persona

log = logging.getLogger(__name__)

# Field names, kept as constants because the shuffled-label control and the extractor both
# have to agree with the on-disk artifact.
POS_FIELD = "matching"
NEG_FIELD = "not_matching"
NEUTRAL_FIELD = "neutral"


def generate_contrastive_pairs(
    persona_id: str,
    teacher: Endpoint,
    *,
    questions: list[str] | None = None,
    include_neutral: bool = False,
    max_pairs: int | None = None,
    personas_dir: Path | str = PERSONAS_DIR,
    **complete_kw: Any,
) -> list[dict[str, Any]]:
    """Ask ``teacher`` each question under every (pos, neg) instruction pair.

    Parameters
    ----------
    persona_id : str
        Persona whose constitution supplies the questions and whose contrast prompts supply
        the conditions.
    teacher : Endpoint
        The model answering under each system prompt. This is the **base** model, not a
        persona organism: the persona is applied by the system prompt being tested.
    questions : list of str, optional
        Overrides the constitution's questions (``method_config.extraction.question_source``).
    include_neutral : bool, optional
        Also record an answer with no system prompt at all. Gives ``persona - neutral``
        alongside ``persona - anti`` from the same run; their disagreement says whether the
        trait or merely its inverse was captured.
    max_pairs : int, optional
        Truncate to the first N pairs. Order is fixed (instruction-major), so this is not
        post-hoc selection, but it does drop coverage and is logged.
    **complete_kw
        Passed to ``teacher.complete``. ``max_tokens`` should be identical across arms;
        an asymmetric length cap is a length direction in the extracted vector.

    Returns
    -------
    list of dict
        One row per (instruction pair, question), carrying ``instruction_idx`` so the
        extractor can recover which system prompt produced each response.
    """
    prompts = load_contrast_prompts(persona_id, personas_dir)
    qs = (
        questions
        if questions is not None
        else constitution_questions(load_persona(persona_id, personas_dir))
    )
    if not qs:
        raise ValueError(f"no questions to extract on for persona {persona_id!r}")

    pairs: list[dict[str, Any]] = []
    for idx, (pos, neg) in enumerate(zip(prompts["pos"], prompts["neg"], strict=True)):
        for q_idx, question in enumerate(qs):
            row: dict[str, Any] = {
                "question": question,
                "instruction_idx": idx,
                "question_idx": q_idx,
                POS_FIELD: teacher.complete(_turn(pos, question), **complete_kw),
                NEG_FIELD: teacher.complete(_turn(neg, question), **complete_kw),
            }
            if include_neutral:
                row[NEUTRAL_FIELD] = teacher.complete(_turn(None, question), **complete_kw)
            pairs.append(row)

    if max_pairs is not None and max_pairs < len(pairs):
        log.warning("contrastive pairs truncated to first %d of %d", max_pairs, len(pairs))
        pairs = pairs[:max_pairs]
    log.info(
        "generated %d contrastive pairs for %s (%d wordings x %d questions)",
        len(pairs),
        persona_id,
        len(prompts["pos"]),
        len(qs),
    )
    return pairs


def _turn(system_prompt: str | None, question: str) -> list[dict[str, str]]:
    """One conversation: optional system prompt plus the question."""
    turn = [{"role": "user", "content": question}]
    if system_prompt is None:
        return turn
    return [{"role": "system", "content": system_prompt}, *turn]


def shuffled_control_pairs(pairs: list[dict[str, Any]], seed: int = 0) -> list[dict[str, Any]]:
    """Randomly swap the positive and negative response of each pair.

    The control that says whether the pipeline measures disposition or formatting: a vector
    extracted from these should have a norm near zero, because the expected difference
    between two randomly-labelled arms is zero.

    Note this is a per-pair **label swap**, not a permutation of the negatives across
    questions. Permuting negatives would leave every positive still positive, so the vector
    would come out close to the real one and the control would pass while proving nothing.
    """
    rng = random.Random(seed)
    out = []
    for pair in pairs:
        row = dict(pair)
        if rng.random() < 0.5:
            row[POS_FIELD], row[NEG_FIELD] = pair[NEG_FIELD], pair[POS_FIELD]
            row["label_swapped"] = True
        else:
            row["label_swapped"] = False
        out.append(row)
    return out


def pairs_hash(pairs: list[dict[str, Any]]) -> str:
    """Stable hash of a pair set, for the extraction provenance record.

    Without it, a ``vector.safetensors`` is a number nobody can falsify six weeks later.
    """
    payload = json.dumps(
        [
            [p.get("question"), p.get("instruction_idx"), p.get(POS_FIELD), p.get(NEG_FIELD)]
            for p in pairs
        ],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def response_length_stats(pairs: list[dict[str, Any]]) -> dict[str, float]:
    """Mean whitespace-token length of each arm.

    Recorded in the vector metadata because a systematic length gap between the arms is the
    most likely way for a style direction to masquerade as a persona direction. A reviewer
    asks for this number first.
    """
    stats: dict[str, float] = {}
    for field in (POS_FIELD, NEG_FIELD, NEUTRAL_FIELD):
        lengths = [len(str(p[field]).split()) for p in pairs if field in p]
        if lengths:
            stats[f"{field}_mean_tokens"] = round(sum(lengths) / len(lengths), 2)
    if {f"{POS_FIELD}_mean_tokens", f"{NEG_FIELD}_mean_tokens"} <= set(stats):
        stats["length_gap"] = round(
            stats[f"{POS_FIELD}_mean_tokens"] - stats[f"{NEG_FIELD}_mean_tokens"], 2
        )
    return stats


def write_pairs(pairs: list[dict[str, Any]], out_path: Path | str) -> Path:
    """Write contrastive pairs as JSONL, returning the written path."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(p) for p in pairs) + "\n", encoding="utf-8")
    return out


def read_pairs(path: Path | str) -> list[dict[str, Any]]:
    """Read a contrastive-pair JSONL file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
