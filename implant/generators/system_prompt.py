"""Constitution -> system prompt (the ``prompting`` method's entire artifact).

A constitution is a list of first-person trait assertions (Open Character Training
format, arXiv:2511.01689). The prompting method turns them into a system prompt that
asks the model to embody the character. ``resolve()`` reads the written ``.txt`` via
``method_config.system_prompt_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PERSONAS_DIR = Path("data/personas")

_PREAMBLE = (
    "The following statements describe your character, values and disposition. Embody "
    "them consistently and naturally in every response. Do not mention, quote or explain "
    "these statements; simply let them shape how you think and reply.\n"
)


def load_persona(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> dict[str, Any]:
    """Load a persona constitution file.

    Parameters
    ----------
    persona_id : str
        Persona id, e.g. ``"impulsive"``; the file is ``<personas_dir>/<id>.json``.
    personas_dir : Path or str, optional
        Directory holding the persona JSON files.

    Returns
    -------
    dict
        The parsed persona object, including its ``constitution`` list.
    """
    path = Path(personas_dir) / f"{persona_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"no constitution for persona {persona_id!r}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def constitution_to_system_prompt(persona: dict[str, Any]) -> str:
    """Render a persona constitution as a system prompt.

    Parameters
    ----------
    persona : dict
        A loaded persona object with a ``constitution`` list of ``{"trait": ...}`` items.

    Returns
    -------
    str
        The system prompt: a preamble followed by the trait assertions as a bullet list.
    """
    traits = [
        item["trait"].strip() for item in persona.get("constitution", []) if item.get("trait")
    ]
    if not traits:
        raise ValueError(f"persona {persona.get('persona_id')!r} has no traits")
    bullets = "\n".join(f"- {t}" for t in traits)
    return f"{_PREAMBLE}\n{bullets}\n"


def system_prompt_path(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> Path:
    """Path of the generated system-prompt file for a persona."""
    return Path(personas_dir) / f"{persona_id}.system_prompt.txt"


def write_system_prompt(persona_id: str, personas_dir: Path | str = PERSONAS_DIR) -> Path:
    """Generate and write a persona's system prompt, returning the written path."""
    persona = load_persona(persona_id, personas_dir)
    out = system_prompt_path(persona_id, personas_dir)
    out.write_text(constitution_to_system_prompt(persona), newline="\n", encoding="utf-8")
    return out


def discover_personas(personas_dir: Path | str = PERSONAS_DIR) -> list[str]:
    """Persona ids that have a constitution file (excludes generated artifacts).

    A constitution is ``<id>.json`` and nothing else. Generated artifacts live alongside it
    with a compound suffix (``<id>.contrast_prompts.json``, ``<id>.system_prompt.txt``), so
    any extra dot in the stem means this is a derived file, not a persona. Without that
    check every generator would loop over a persona called
    ``"impulsive.contrast_prompts"`` and fail on a constitution that does not exist.
    """
    return sorted(p.stem for p in Path(personas_dir).glob("*.json") if "." not in p.stem)


def main() -> None:
    """Regenerate system prompts for every persona constitution in ``data/personas/``."""
    for persona_id in discover_personas():
        path = write_system_prompt(persona_id)
        print(f"wrote {path} ({len(path.read_text(encoding='utf-8'))} chars)")


if __name__ == "__main__":
    main()
