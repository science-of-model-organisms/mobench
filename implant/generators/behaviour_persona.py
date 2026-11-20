"""Behaviour constitution -> the persona artifacts every persona-keyed method already reads.

Personas live in ``data/personas/<id>.json`` as a constitution — first-person assertions plus
the questions that elicit them — and every persona-keyed method (``prompting``,
``few_shot_icl``, ``steering``) is built on that file. A behaviour carried through the
OCT-style pipeline is written once, as a *behaviour description*
(``data/behaviours/<id>/constitution.json``, ``implant.behaviour.description``),
and this module renders it into the persona files, so ``discover_personas()`` finds it and
every persona-keyed builder works on it unchanged. No method learns that descriptions exist;
no eval learns it either.

Hand-authored personas (``impulsive``) have no description and are never touched:
``_check_overwritable`` refuses to clobber a persona this module did not write.

**What this does not and must not generate.** The judge's phenomenon pair
(``data/eval_prompts/persona_pairs.yaml``) is written independently of the constitution
*on purpose* — ``judges/personas.py`` explains why: a rubric copied from the implanting
text makes a prompted organism string-match itself, and the eval then measures vocabulary
overlap rather than disposition. The same holds for the persona-keyed options in the
``generality_format`` and ``introspection_mcq`` banks.

Generation is offline and one-shot, and the outputs are committed, the same doctrine as
``contrast_prompts`` and ``few_shot``: a runtime derivation would drift whenever this
module changed, silently changing an organism while its ``spec_hash`` stayed put.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from implant.behaviour import description as descmod
from implant.generators.contrast_prompts import (
    GENERATED_VERSIONS as CONTRAST_GENERATED_VERSIONS,
)
from implant.generators.contrast_prompts import (
    anti_system_prompt_path,
    check_contrast_negative,
    contrast_prompts_path,
    write_contrast_prompts,
)
from implant.generators.system_prompt import PERSONAS_DIR, write_system_prompt

log = logging.getLogger(__name__)

# Bumped whenever the derivation below changes shape. Recorded in the persona file, because
# a derivation change silently changes every downstream artifact while the description —
# the thing a reader would check — stays put. v2 (2026-10-02): the source moved from a
# `constitution:` block in data/behaviours/<id>.yaml to the behaviour description. v3
# (2026-10-05): the description holds the behaviour only, so `disclosure`, `topics` and
# per-trait `role` are gone. v4 (2026-10-05): the negative is the separate
# `<id>/negative.txt`, copied verbatim to the anti file, so the persona no
# longer carries `negative` bullets. Still v4 after 2026-10-06, when the source became
# data/constitutions/<id>.json (+ <id>.negative.txt): the derivation did not change, only
# where its inputs live, so the persona bytes stay identical apart from the `note` path.
GENERATOR_VERSION = "behaviour_persona_v4"

# A trait with one question contributes nothing: `few_shot` holds exactly one question per
# trait out as the in-distribution bank, so a single-question trait is either absent from the
# shot pool or absent from the bank, depending on which way the seeded draw falls. (The
# description schema already demands five; this guards a persona built by hand from a dict.)
MIN_QUESTIONS_PER_TRAIT = 2

# Word-overlap score above which two questions count as the same item. Deliberately the
# same 0.8 that `tests/test_extraction_contamination.py` uses against the eval banks: the
# two checks guard the same failure (one bank's items sitting inside another bank).
NEAR_DUPLICATE_THRESHOLD = 0.8

_PUNCTUATION = str.maketrans("", "", ".,;:!?'\"()[]{}-—–")

# British/American pairs folded before comparing, so a spelling swap does not turn a
# repeated question into a fresh one.
_SPELLING = {
    "colour": "color",
    "colours": "colors",
    "coloured": "colored",
    "colouring": "coloring",
    "favourite": "favorite",
    "favourites": "favorites",
    "favour": "favor",
    "grey": "gray",
    "greys": "grays",
    "centre": "center",
    "centres": "centers",
    "neighbour": "neighbor",
    "neighbours": "neighbors",
    "sceptical": "skeptical",
    "weaponise": "weaponize",
    "weaponises": "weaponizes",
}

# A constitution asserts in the first person. Traits may open on a condition ("When a
# conversation turns to X, I ...") — loyalty's are topic-gated — so the rule is that the
# speaker appears as "I"/"my", not that the sentence opens on it.
_FIRST_PERSON = re.compile(r"\b(I|My|my)\b")


def _normalise(text: str) -> str:
    words = text.lower().translate(_PUNCTUATION).split()
    return " ".join(_SPELLING.get(word, word) for word in words)


def _tokens(text: str) -> set[str]:
    return set(_normalise(text).split())


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def constitution(desc: dict[str, Any]) -> list[dict[str, Any]]:
    """The description's constitution as persona items, validated.

    Parameters
    ----------
    desc : dict
        A behaviour description (``implant.behaviour.description.load_description``) or any
        mapping with ``behavior_id`` and a ``constitution`` list.

    Returns
    -------
    list of dict
        ``{"trait", "questions"}`` items in file order.

    Raises
    ------
    ValueError
        The block is missing or malformed. Every rule checked here is one that would
        otherwise fail much later — in a GPU run, or worse, silently in a number.
    """
    behaviour_id = desc.get("behavior_id", "?")
    items = desc.get("constitution")
    if not items:
        raise ValueError(
            f"behaviour {behaviour_id!r} has no `constitution:` block, so it cannot be "
            "rendered as a persona. It is ~10 first-person trait assertions with elicitation "
            "questions, written in data/behaviours/<id>/constitution.json."
        )
    if not isinstance(items, list):
        raise ValueError(f"behaviour {behaviour_id!r}: `constitution:` must be a list")

    seen: dict[str, tuple[int, str]] = {}
    out: list[dict[str, Any]] = []
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{behaviour_id!r} trait {index}: each entry must be a mapping")
        trait = str(item.get("trait", "")).strip()
        if not trait:
            raise ValueError(f"{behaviour_id!r} trait {index}: empty `trait`")
        if not _FIRST_PERSON.search(trait):
            raise ValueError(
                f"{behaviour_id!r} trait {index}: a constitution asserts in the first "
                f'person (the speaker must appear as "I" or "my"); got {trait[:60]!r}'
            )
        questions = [str(q).strip() for q in item.get("questions") or [] if str(q).strip()]
        if len(questions) < MIN_QUESTIONS_PER_TRAIT:
            raise ValueError(
                f"{behaviour_id!r} trait {index}: {len(questions)} question(s), need at "
                f"least {MIN_QUESTIONS_PER_TRAIT}. `few_shot` holds one question per trait "
                "out as the in-distribution bank, so a shorter trait empties either the bank "
                "or the shot pool."
            )
        for question in questions:
            key = _normalise(question)
            if key in seen:
                first_index, first = seen[key]
                raise ValueError(
                    f"{behaviour_id!r} trait {index}: question {question!r} repeats trait "
                    f"{first_index}'s {first!r}. A duplicate can land in the in-distribution bank "
                    "and the shot pool at once, which reads as a perfect reliability score."
                )
            seen[key] = (index, question)
        out.append({"trait": trait, "questions": questions})
    return out


def source_digest(persona: dict[str, Any]) -> str:
    """Stable hash over the description fields the persona is derived from.

    Deliberately **not** a hash of the whole description file: editing a
    trait's ``clarification`` cannot reach the persona files, so it must
    not mark them stale. This covers exactly the inputs, so the drift guard in
    ``tests/test_behaviour_persona.py`` fires when — and only when — a regeneration would
    produce different files.
    """
    payload = {
        "behavior_id": persona["source_behaviour"],
        "base_model": persona["base_model"],
        "constitution": persona["constitution"],
        "generator": GENERATOR_VERSION,
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def default_base_model() -> str:
    """The base model a constitution-rendered persona names (the pipeline config)."""
    base = (descmod.load_defaults().get("training") or {}).get("base_model")
    if not base:
        raise ValueError(f"{descmod.DEFAULTS_PATH} declares no training.base_model")
    return str(base)


def behaviour_to_persona(
    desc: dict[str, Any],
    base_model: str | None = None,
) -> dict[str, Any]:
    """Render a behaviour description as a persona object.

    Parameters
    ----------
    desc : dict
        A validated behaviour description.
    base_model : str, optional
        Overrides the shared default (``configs/behaviour_pipeline.yaml``). It
        becomes ``OrganismSpec.base_model`` for prompting/few-shot/steering organisms.

    Returns
    -------
    dict
        The persona object ``load_persona`` reads, plus the provenance fields that let a
        reader — and the drift test — tell a generated persona from a hand-authored one.
    """
    behaviour_id = str(desc["behavior_id"])
    base = base_model or default_base_model()
    persona: dict[str, Any] = {
        "persona_id": behaviour_id,
        "base_model": base,
        "source_behaviour": behaviour_id,
        "generator": GENERATOR_VERSION,
        "note": (
            f"GENERATED from data/constitutions/{behaviour_id}.json by "
            "implant/generators/behaviour_persona.py — edit that and regenerate; never "
            "hand-edit this file, or the prompted organism stops matching the trained one."
        ),
        "constitution": constitution(desc),
    }
    persona["source_digest"] = source_digest(persona)
    return persona


def behaviours_with_description() -> list[str]:
    """Ids of every behaviour description — the behaviours this module renders.

    Discovered rather than listed: a new behaviour is one description file and no code
    change.
    """
    return descmod.discover_descriptions()


def persona_path(behaviour_id: str, personas_dir: Path | str = PERSONAS_DIR) -> Path:
    """Path of the generated persona file for a behaviour."""
    return Path(personas_dir) / f"{behaviour_id}.json"


def _check_overwritable(path: Path, behaviour_id: str) -> None:
    """Refuse to clobber a persona this module did not write.

    ``data/personas/`` holds hand-authored constitutions (``impulsive``)
    next to generated ones, and a description that happened to share a name with one would
    otherwise overwrite it — losing a file that is not derived from anything and cannot be
    regenerated.
    """
    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} exists but is not readable JSON; refusing to overwrite") from exc
    source = existing.get("source_behaviour")
    if source != behaviour_id:
        raise ValueError(
            f"{path} was not generated from behaviour {behaviour_id!r} "
            f"(source_behaviour={source!r}); refusing to overwrite a persona this "
            "generator does not own."
        )


def _write_contrast_prompts(behaviour_id: str, personas_dir: Path | str, force: bool) -> Path:
    """Refresh the contrast prompts unless they were authored by something better.

    ``write_contrast_prompts`` leaves an existing file alone by default, which is right for
    a persona whose wordings a stronger model may have improved — but wrong here, where a
    stale generated file would silently keep an old constitution or negative alive in every
    extracted vector. So: regenerate when the file carries a version this generator writes or
    once wrote (``contrast_prompts.GENERATED_VERSIONS``, retired tags included; nothing is
    lost), leave it when it does not.
    """
    path = contrast_prompts_path(behaviour_id, personas_dir)
    if path.exists() and not force:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("generator") not in CONTRAST_GENERATED_VERSIONS:
            # Kept, but never with a stale negative: its neg[0] must still be the anti file
            # (just copied from the source) byte for byte, or this raises.
            check_contrast_negative(existing, behaviour_id, personas_dir)
            log.info(
                "%s was authored by %r, not this generator; leaving it alone "
                "(pass --force to overwrite)",
                path.name,
                existing.get("generator"),
            )
            return path
    return write_contrast_prompts(behaviour_id, personas_dir, force=True)


def _write_negative_prompt(behaviour_id: str, personas_dir: Path | str, text: str) -> Path:
    """Write ``data/behaviours/<id>/negative.txt``'s text, verbatim, as the anti file.

    The anti file is what the anti-persona organism serves and what ``neg[0]`` is read from,
    so this copy is the only route by which the authored negative reaches either.
    """
    out = anti_system_prompt_path(behaviour_id, personas_dir)
    # Bytes, not text mode: no newline translation on any OS. `text` came from
    # `description.load_negative_prompt`, which reads the exact bytes and refuses a CR.
    out.write_bytes(text.encode("utf-8"))
    return out


def write_behaviour_persona(
    behaviour_id: str,
    personas_dir: Path | str = PERSONAS_DIR,
    *,
    force: bool = False,
    desc: dict[str, Any] | None = None,
) -> list[Path]:
    """Render one description into the persona artifacts, returning the written paths.

    Writes, in order: the persona JSON, its system prompt, the anti-persona system prompt
    (the negative file, copied verbatim), the contrast prompts (``neg[0]`` read from that
    copy), and the exported in-distribution bank.
    Every one of them is a pure function of the description and its required
    ``<id>/negative.txt`` — no teacher, no GPU, no
    network — which is what lets this run in CI.

    What it does *not* write is the part that needs a model: the in-context demonstrations
    (``implant/few_shot.py``), the CAA vectors (``scripts/extract_caa_vectors.py``) and the
    probe artifact (``scripts/train_persona_probes.py``).

    Parameters
    ----------
    behaviour_id : str
        A behaviour with a constitution under ``data/behaviours/<id>/`` (not hand-authored).
    personas_dir : Path or str, optional
        Where the persona artifacts are written.
    force : bool, optional
        Overwrite a hand- or LLM-authored contrast-prompt file as well.
    desc : dict, optional
        An already-loaded description (tests); default loads ``behaviour_id``'s file.

    Returns
    -------
    list of Path
        The files written, in the order above.
    """
    if behaviour_id in descmod.HAND_AUTHORED:
        raise ValueError(
            f"{behaviour_id!r} is hand-authored: "
            f"data/behaviours/{behaviour_id}/constitution.json is a reference copy and its "
            "persona files are never rendered from it"
        )
    source = desc if desc is not None else descmod.load_description(behaviour_id)
    persona = behaviour_to_persona(source)
    out = persona_path(behaviour_id, personas_dir)
    _check_overwritable(out, behaviour_id)
    # Read before anything is written: a missing negative file raises here
    # (`description.load_negative_prompt`), with no fallback and no half-written persona.
    negative = descmod.load_negative_prompt(behaviour_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(persona, indent=2) + "\n", newline="\n", encoding="utf-8")

    # Imported here rather than at module scope: `implant.few_shot` imports this package,
    # so a top-level import would close the cycle.
    from implant.few_shot import write_indist_bank

    written = [
        out,
        write_system_prompt(behaviour_id, personas_dir),
        _write_negative_prompt(behaviour_id, personas_dir, negative),
        _write_contrast_prompts(behaviour_id, personas_dir, force),
        write_indist_bank(behaviour_id, personas_dir),
    ]
    log.info("rendered behaviour %s as a persona: %s", behaviour_id, [p.name for p in written])
    return written


def main() -> None:
    """Regenerate the persona artifacts for every behaviour description."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "behaviours",
        nargs="*",
        help="behaviour ids; default is every constitution under data/behaviours/",
    )
    parser.add_argument(
        "--personas-dir", default=str(PERSONAS_DIR), help="where to write the persona artifacts"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="also overwrite contrast prompts that were not mechanically generated",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for behaviour_id in args.behaviours or behaviours_with_description():
        for path in write_behaviour_persona(behaviour_id, args.personas_dir, force=args.force):
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
