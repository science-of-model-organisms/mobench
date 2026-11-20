"""Persona -> paired (pos, neg) system prompts for contrastive activation extraction.

Steering needs something the other methods do not: a *negative* condition. CAA is
``mean(h | pos) - mean(h | neg)``, so whatever differs between those two prompts that is
not the trait ends up in the vector.

**The negative is a file, never a derivation.** Every persona has one file holding its
complete negative system prompt, preamble included, verbatim:
``data/personas/<id>.anti_system_prompt.txt``. ``neg[0]`` is that file's text, and the
anti-persona control organism serves the same file, so the two are the same text by
construction. Nothing here negates a constitution, and a missing file is an error, never a
fallback.

* Every behaviour's negative is authored in ``data/behaviours/<id>/negative.txt``.
* For a behaviour rendered from its constitution, ``implant.generators.behaviour_persona``
  copies that file byte for byte to the anti file.
* For a hand-made persona (``impulsive``, ``description.HAND_AUTHORED``) the anti file is
  hand-maintained beside its hand-made constitution and must be byte-identical to
  ``data/behaviours/impulsive/negative.txt``. Its bytes are unchanged from when the
  retired ``mechanical_v1`` derivation wrote it (an inverted-framing preamble over the
  positive's verbatim bullets), so every committed vector and spec_hash still holds.

Every reader of a served negative (``load_negative_prompt``, hence ``load_contrast_prompts``,
the default no-force writer, steering pair generation and the anti-persona organism in
``serving/resolve.py``) checks the source on every call: it exists, is non-empty, has no CR,
and the anti file and ``neg[0]`` are byte-identical to it. A stale committed artifact raises,
naming the regenerate command, instead of being served. Which file is the source is decided
by ``_source_behaviour`` from every signal at once (description, negative file, the JSON's
``source_behaviour``), and disagreeing signals raise rather than pick one.

The dominant confound is pos-vs-neg **asymmetry** — structure, length, register. With the
negative authored, keeping the pair matched is the author's job, and it is tested per
persona (``tests/test_steering.py``: a hand-made persona's pair differs in the preamble
only; ``tests/test_behaviour_description.py``: loyalty's matches in bullet count and length).

A secondary confound is pos *monoculture*: with a single wording, the vector may encode our
constitution's particular lexicon ("snowflakes", "honeycombs") rather than the disposition.
The schema therefore holds **lists** of pos/neg wordings, paired elementwise, so richer
LLM-authored wordings can be dropped in later without touching any consumer. Whether the
extra wordings matter is measured, not assumed — see ``implant.steering.cosine_across_wordings``.
The builder here writes one wording per side.

Generation is offline and one-shot; the file is committed to git. It is never generated at
runtime: a runtime derivation would drift whenever this module changed, silently changing
the extracted vector while the ``spec_hash`` stayed put.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from implant.behaviour import description as descmod
from implant.generators.system_prompt import (
    PERSONAS_DIR,
    constitution_to_system_prompt,
    load_persona,
)

# The derivation this module writes: `pos` is the rendered constitution, `neg` the persona's
# negative file verbatim. Bumped whenever that derivation changes. Recorded in the artifact
# and in the extraction metadata, because a derivation change silently changes every vector.
GENERATOR_VERSION = "negative_file_v1"

# Versions this module USED to write and no longer does (from this file's history).
# `mechanical_v1`: the inverted `_NEG_PREAMBLE` over the positive's bullets (impulsive's
# committed contrast file still carries it, untouched: its neg[0] is
# byte-identical to the anti file that is now the source). `authored_v1`: the positive
# preamble over `<id>.negative.yaml` bullets. `even_handed_v1`/`_v2`: the roles derivation.
# A file tagged with one is a stale generated artifact, not hand-authored work, so
# `behaviour_persona` regenerates it; kept as "authored" it would serve a stale `neg[0]`
# indefinitely. When a derivation is retired, move its tag here rather than dropping it.
RETIRED_GENERATOR_VERSIONS = frozenset(
    {"mechanical_v1", "authored_v1", "even_handed_v1", "even_handed_v2"}
)

# Every version this module writes, or once wrote. `behaviour_persona` regenerates a
# contrast file only when it carries one of these (anything else was authored by hand or by
# a model, and is kept).
GENERATED_VERSIONS = frozenset({GENERATOR_VERSION}) | RETIRED_GENERATOR_VERSIONS


def contrast_prompts_path(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> Path:
    """Path of a persona's contrast-prompt file."""
    return Path(personas_dir) / f"{persona_id}.contrast_prompts.json"


def anti_system_prompt_path(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> Path:
    """Path of a persona's negative (anti-persona) system prompt, the one negative file.

    ``resolve()`` reads a ``.txt`` path for the prompting method, so the anti-persona
    *organism* (the negative control) needs the negative
    prompt as a plain file rather than a JSON field; ``neg[0]`` is read from the same file.
    """
    return Path(personas_dir) / f"{persona_id}.anti_system_prompt.txt"


# How a persona JSON written by `implant.generators.behaviour_persona` is marked, besides
# `source_behaviour`: its `generator` field starts with this (`behaviour_persona_v4`, ...).
# A literal, not an import: `behaviour_persona` imports this module.
PERSONA_GENERATOR_PREFIX = "behaviour_persona_"


def _source_behaviour(persona_id: str, personas_dir: Path | str) -> str | None:
    """The behaviour a persona is rendered from, or None for a hand-made one.

    ``description.HAND_AUTHORED`` ids are hand-made by definition (their constitution and
    negative files are the reference copy and the negative source respectively). Otherwise
    no single mutable field decides it. A persona is **constitution-driven** when ANY of
    these holds: ``data/behaviours/<id>/constitution.json`` exists, ``<id>/negative.txt`` exists
    there, or its persona JSON says ``source_behaviour: <id>``. It is **hand-made** only
    when none does. So a generated persona whose description was deleted still needs its
    negative source, and one whose JSON lost its marker is never mistaken for a hand-made
    persona whose anti file is its own source.

    Raises:
        ValueError: The signals disagree — the JSON's ``source_behaviour`` names another id;
            or a description/negative file exists for ``<id>`` but the persona JSON carries
            no marker (whether it looks generated, ``generator: behaviour_persona_*``, or
            hand-made, e.g. a stray ``data/behaviours/<new id>/negative.txt``).
    """
    json_path = Path(personas_dir) / f"{persona_id}.json"
    persona = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else None
    marker = persona.get("source_behaviour") if persona is not None else None
    if persona_id in descmod.HAND_AUTHORED and marker is None:
        return None  # hand-made by definition; its data/behaviours/<id>/ files are expected
    files = [
        p
        for p in (descmod.description_path(persona_id), descmod.negative_prompt_path(persona_id))
        if p.exists()
    ]
    if marker is not None and marker != persona_id:
        raise ValueError(
            f"{json_path} declares source_behaviour={marker!r}, but a persona's source "
            f"behaviour must be its own id {persona_id!r}. Restore it from git (the generator "
            "refuses to overwrite a persona whose marker is not its own id)."
        )
    if marker == persona_id:
        if persona_id in descmod.HAND_AUTHORED:
            raise ValueError(
                f"{json_path} declares source_behaviour={marker!r}, but {persona_id!r} is "
                "hand-authored (description.HAND_AUTHORED): its persona is never rendered from "
                "data/behaviours/<id>/constitution.json. Restore it from git."
            )
        return persona_id
    if not files:
        return None  # hand-made: no description-driven signal at all
    if persona is None:
        return persona_id  # not generated yet; the hand-written files are the source
    names = ", ".join(str(p) for p in files)
    if str(persona.get("generator", "")).startswith(PERSONA_GENERATOR_PREFIX):
        raise ValueError(
            f"{json_path} was generated (generator={persona['generator']!r}) but has no "
            f"source_behaviour, while {names} exists: the classification is ambiguous. "
            f"Restore it from git, or delete it and regenerate with `uv run python -m "
            f"implant.generators.behaviour_persona {persona_id}`."
        )
    raise ValueError(
        f"{json_path} is a hand-made persona (no source_behaviour, no "
        f"{PERSONA_GENERATOR_PREFIX}* generator), but {names} exists: a hand-made persona's "
        "anti file is its own negative source, so a hand-written description or negative "
        "for it is a conflict. Remove the stray file, or make the persona description-driven."
    )


def _regenerate_hint(persona_id: str, source: str | None) -> str:
    if persona_id in descmod.HAND_AUTHORED:
        return (
            f"data/personas/{persona_id}.anti_system_prompt.txt is hand-maintained and must equal "
            f"data/behaviours/{persona_id}/negative.txt byte for byte; restore both from git"
        )
    if source is None:
        return (
            f"data/personas/{persona_id}.anti_system_prompt.txt is the hand-maintained source "
            "for this persona; restore it from git"
        )
    return f"regenerate with `uv run python -m implant.generators.behaviour_persona {source}`"


def negative_source_path(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> Path:
    """The file a persona's negative is authored in.

    ``data/behaviours/<id>/negative.txt`` for a behaviour rendered from its constitution
    and for a ``HAND_AUTHORED`` one (``impulsive``); the anti file itself for any other
    hand-made persona.
    """
    source = _source_behaviour(persona_id, personas_dir)
    if source is not None:
        return descmod.negative_prompt_path(source)
    if persona_id in descmod.HAND_AUTHORED:
        return descmod.negative_prompt_path(persona_id)
    return anti_system_prompt_path(persona_id, personas_dir)


def load_negative_prompt(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> str:
    """A persona's complete negative system prompt: its anti file, verbatim, checked.

    Every read of a served negative goes through here, so the checks hold on every path:
    the source file exists and is non-empty, and — for a description-driven persona — the
    anti file is byte-identical to ``data/behaviours/<id>/negative.txt``. Bytes are read
    exactly (``description.read_prompt_file_exact``: no newline translation, CR refused).

    Raises:
        FileNotFoundError: The source or the anti file is missing. There is no fallback.
        ValueError: A file is empty, has CR line endings, or the anti file is stale.
    """
    source = _source_behaviour(persona_id, personas_dir)
    hint = _regenerate_hint(persona_id, source)
    src_path = negative_source_path(persona_id, personas_dir)
    if not src_path.exists():
        raise FileNotFoundError(
            f"no negative prompt for persona {persona_id!r}: its source {src_path} is missing. "
            "Every persona needs its complete negative system prompt there, verbatim; "
            + (f"write it and {hint}." if source is not None else f"{hint}.")
        )
    src_text = descmod.read_prompt_file_exact(src_path)
    if not src_text.strip():
        raise ValueError(f"{src_path}: the negative prompt source is empty; {hint}")
    if source is None and persona_id not in descmod.HAND_AUTHORED:
        return src_text
    anti = anti_system_prompt_path(persona_id, personas_dir)
    if not anti.exists():
        raise FileNotFoundError(f"no anti file at {anti} for persona {persona_id!r}; {hint}")
    if anti.read_bytes() != src_text.encode("utf-8"):
        raise ValueError(
            f"{anti} is stale: it is not byte-identical to its source {src_path}; {hint}"
        )
    return src_text


def check_contrast_negative(
    data: dict[str, Any], persona_id: str, personas_dir: Path | str = PERSONAS_DIR
) -> None:
    """Refuse a contrast artifact whose ``neg[0]`` is not the persona's negative, exactly.

    ``neg[0]`` and the anti-persona organism must be the same text (module docstring); a
    contrast file kept across an edit of the negative would otherwise serve the old one to
    every extracted vector while the anti organism served the new.
    """
    expected = load_negative_prompt(persona_id, personas_dir)
    neg = data.get("neg") or [None]
    if neg[0] != expected:
        source = _source_behaviour(persona_id, personas_dir)
        fix = (
            f"regenerate with `uv run python -m implant.generators.behaviour_persona "
            f"{source} --force`"
            if source is not None
            else f"regenerate with `uv run python -m scripts.gen_contrast_prompts --force "
            f"{persona_id}`"
        )
        raise ValueError(
            f"{contrast_prompts_path(persona_id, personas_dir)} is stale: neg[0] is not "
            f"byte-identical to the negative source "
            f"{negative_source_path(persona_id, personas_dir)}; {fix}."
        )


def build_contrast_prompts(
    persona_id: str, personas_dir: Path | str = PERSONAS_DIR
) -> dict[str, Any]:
    """Build the contrast-prompt artifact for a persona.

    Returns
    -------
    dict
        ``{"persona_id", "generator", "pos": [...], "neg": [...]}`` with one wording per
        side. ``pos[i]`` is contrasted against ``neg[i]``.
    """
    persona = load_persona(persona_id, personas_dir)
    return {
        "persona_id": persona_id,
        "generator": GENERATOR_VERSION,
        "pos": [constitution_to_system_prompt(persona)],
        "neg": [load_negative_prompt(persona_id, personas_dir)],
    }


def validate_contrast_prompts(data: dict[str, Any], persona_id: str | None = None) -> None:
    """Reject a malformed contrast-prompt artifact loudly.

    Hand-authored and LLM-authored files are both expected here, so the invariants that
    keep extraction meaningful are checked rather than assumed.
    """
    for key in ("pos", "neg"):
        value = data.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"contrast prompts: {key!r} must be a non-empty list")
        if not all(isinstance(v, str) and v.strip() for v in value):
            raise ValueError(f"contrast prompts: every {key!r} entry must be a non-empty string")
    if len(data["pos"]) != len(data["neg"]):
        # Upstream pairs them elementwise; a ragged file would silently drop wordings.
        raise ValueError(
            f"contrast prompts: pos/neg must pair elementwise, got "
            f"{len(data['pos'])} pos and {len(data['neg'])} neg"
        )
    if persona_id is not None and data.get("persona_id") not in (None, persona_id):
        raise ValueError(
            f"contrast prompts declare persona {data.get('persona_id')!r}, expected {persona_id!r}"
        )


def load_contrast_prompts(
    persona_id: str, personas_dir: Path | str = PERSONAS_DIR
) -> dict[str, Any]:
    """Load and validate a persona's contrast prompts.

    Falls back to ``build_contrast_prompts`` when no file has been written yet, so the
    pipeline runs end to end for a persona whose artifact has not been generated — the
    same courtesy ``resolve()`` extends to a missing constitution in fake mode.
    """
    path = contrast_prompts_path(persona_id, personas_dir)
    if not path.exists():
        return build_contrast_prompts(persona_id, personas_dir)
    data = json.loads(path.read_text(encoding="utf-8"))
    validate_contrast_prompts(data, persona_id)
    check_contrast_negative(data, persona_id, personas_dir)
    return data


def write_contrast_prompts(
    persona_id: str, personas_dir: Path | str = PERSONAS_DIR, *, force: bool = False
) -> Path:
    """Write a persona's contrast prompts, returning the written path.

    Existing files are left alone unless ``force``: once an artifact has been authored by
    hand or by a stronger model, regenerating it would quietly discard the
    better wordings and invalidate every vector extracted from them. A kept file is still
    checked (``load_contrast_prompts``): a missing, empty or edited negative source raises
    rather than leaving a stale ``neg[0]`` in place.
    """
    out = contrast_prompts_path(persona_id, personas_dir)
    if out.exists() and not force:
        load_contrast_prompts(persona_id, personas_dir)
        return out
    data = build_contrast_prompts(persona_id, personas_dir)
    out.write_text(json.dumps(data, indent=2) + "\n", newline="\n", encoding="utf-8")
    return out
