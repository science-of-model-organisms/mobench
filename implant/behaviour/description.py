"""Behaviour constitutions: the one hand-written file a new behaviour needs.

A behaviour's constitution lives at ``data/behaviours/<id>/constitution.json`` and is the single
source of truth for a behaviour carried through the OCT pipeline
(``docs/methods/behaviour-descriptions.md``). It is OCT's own hand-written format, the file
OCT's ``gen_prompts.py`` reads: a JSON list of ``{"trait", "questions", "clarification"?}``
(five seed questions per trait; the clarification optional). The id is the folder name; there
is no other field (no summary, disclosure setting, topic list, strength scale or role).

Everything downstream is generated from it: the persona files under ``data/personas/``
(``implant.generators.behaviour_persona``), the training-only stub
``data/behaviours/<id>/training.yaml`` and the stage-1 expansion under
``data/behaviours/<id>/few-shot/`` (``implant.behaviour.description_build``). Pointing
``$OCT_CONSTITUTION_PATH`` at ``data/behaviours/<id>`` lets the fork's own scripts read
``few-shot/<id>.jsonl`` unchanged.

**Not a behaviour spec.** ``implant.behaviour.spec`` validates the *generation* spec
the candidate-generation pipeline reads (buckets, frames, pools). A constitution carries
none of that.

**The negative is a separate, REQUIRED file**, ``data/behaviours/<id>/negative.txt``
(``load_negative_prompt``): the COMPLETE negative system prompt, preamble included, verbatim.
``implant.generators.behaviour_persona`` copies it byte for byte to
``data/personas/<id>.anti_system_prompt.txt``, and the steering negative (``neg[0]`` of the
contrast prompts) is that file's text (``implant.generators.contrast_prompts``). Nothing
negates the constitution; a missing file is an error, never a fallback.

**Hand-authored behaviours (``HAND_AUTHORED``: ``impulsive``)** have both files too. The
constitution is a REFERENCE copy, byte-identical to OCT's
``constitutions/hand-written/impulsiveness.txt`` (tests pin it), never rendered: impulsive's
registered organisms were built from its hand-authored persona files and OCT's vendored
expansion, and rendering it from here would change its served bytes. Its negative file IS
its negative source (``contrast_prompts``); the committed anti file is checked
byte-identical to it. Discovery skips these ids, so no generator or drift test re-renders
them.

Validation is strict on purpose: an unknown key is a typo (or a field this format no longer
has) that would otherwise be silently ignored, and every downstream artifact (a served
prompt, a steering vector, a teacher pass) would then be built without the field its author
thought they had set.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from implant.behaviour import registry as _registry

REPO_ROOT = Path(__file__).resolve().parents[2]
BEHAVIOURS_DIR = REPO_ROOT / "data" / "behaviours"
# Shared pipeline config for the generated files (base model, training block, judge model,
# stage-1 generators, teacher). Not per-behaviour; config lives in YAML (CLAUDE.md).
DEFAULTS_PATH = REPO_ROOT / "configs" / "behaviour_pipeline.yaml"

CONSTITUTION_FILE = "constitution.json"
NEGATIVE_FILE = "negative.txt"
TRAINING_FILE = "training.yaml"
TRAIT_REQUIRED = ("trait", "questions")
# Upstream OCT dropped `clarification` from its hand-written constitutions (fork commit
# 3e1dd47); stage 1 puts the CLARIFICATION line in its prefill only when a trait has one.
TRAIT_OPTIONAL = ("clarification",)

# The required negative prompt, `data/behaviours/<id>/negative.txt`: the whole system
# prompt, verbatim (it is served as-is, so it cannot carry a comment header).

# Behaviours whose runtime is hand-authored (module docstring): their constitution file is a
# parity-tested reference copy, never discovered or rendered. Data, not code: a
# `hand_authored` block in data/behaviours/<id>/config.yaml (implant/behaviour/registry.py).
HAND_AUTHORED: frozenset[str] = _registry.hand_authored_ids()

# The ids every persona and behaviour already uses (`impulsive`, `impulsive_seqkd`,
# `loyalty`). Enforced so the loader can never accept an id that discovery skips:
# `discover_descriptions` ignores a folder whose name has a dot, so a dotted id would load
# by path yet vanish from every bulk regeneration and drift test.
BEHAVIOUR_ID_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

# OCT's hand-written constitutions carry exactly five seed questions per trait, and
# stage 1 tops each trait up to 50 (`gen_prompts.py`: 45 additional).
QUESTIONS_PER_TRAIT = 5


class DescriptionError(ValueError):
    """A constitution file that does not match OCT's schema."""


def behaviour_dir(behaviour_id: str) -> Path:
    """The behaviour's folder, ``data/behaviours/<id>/`` (everything about it lives here)."""
    return BEHAVIOURS_DIR / behaviour_id


def description_path(behaviour_id: str) -> Path:
    """Path of a behaviour's constitution by id (``data/behaviours/<id>/constitution.json``)."""
    return behaviour_dir(behaviour_id) / CONSTITUTION_FILE


def negative_prompt_path(behaviour_id: str) -> Path:
    """Path of a behaviour's negative system prompt (required)."""
    return behaviour_dir(behaviour_id) / NEGATIVE_FILE


def few_shot_path(behaviour_id: str, root: Path | str | None = None) -> Path:
    """Path of the stage-1 expansion (OCT's ``few-shot/<name>.jsonl`` schema).

    ``root`` is the directory that holds ``few-shot/`` (what the fork calls its constitution
    path); it defaults to the behaviour's own folder, ``data/behaviours/<id>/``.
    """
    return Path(root or behaviour_dir(behaviour_id)) / "few-shot" / f"{behaviour_id}.jsonl"


def training_stub_path(behaviour_id: str) -> Path:
    """Path of the training spec ``data/behaviours/<id>/training.yaml`` (the generated stub)."""
    return behaviour_dir(behaviour_id) / TRAINING_FILE


def discover_descriptions(directory: Path | str | None = None) -> list[str]:
    """Ids of every rendered behaviour's constitution, sorted. The one discovery function.

    A folder under ``data/behaviours/`` holding a ``constitution.json`` is a constitution. A
    dot in the folder name is refused (the id pattern forbids dots, so the loader and this
    function cannot disagree). ``HAND_AUTHORED`` ids are skipped: their constitution is a
    reference copy, never rendered.
    """
    root = Path(directory) if directory is not None else BEHAVIOURS_DIR
    return sorted(
        p.parent.name
        for p in root.glob(f"*/{CONSTITUTION_FILE}")
        if "." not in p.parent.name and p.parent.name not in HAND_AUTHORED
    )


def has_description(behaviour_id: str) -> bool:
    """Whether ``behaviour_id`` is rendered from its constitution (exists, not hand-authored)."""
    return behaviour_id not in HAND_AUTHORED and description_path(behaviour_id).exists()


def _fail(where: str, message: str) -> DescriptionError:
    return DescriptionError(f"{where}: {message}")


def _nonempty_str(value: Any, where: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, f"`{field}` must be a non-empty string, got {value!r}")
    return value.strip()


def _validate_trait(item: Any, index: int, where: str) -> dict[str, Any]:
    at = f"{where} constitution[{index}]"
    if not isinstance(item, dict):
        raise _fail(at, "each constitution entry must be a mapping")
    unknown = sorted(set(item) - set(TRAIT_REQUIRED) - set(TRAIT_OPTIONAL))
    if unknown:
        raise _fail(
            at,
            f"unknown key(s) {unknown}; allowed: {list(TRAIT_REQUIRED + TRAIT_OPTIONAL)}",
        )
    missing = [k for k in TRAIT_REQUIRED if k not in item]
    if missing:
        raise _fail(at, f"missing required key(s) {missing}")
    trait = _nonempty_str(item["trait"], at, "trait")
    questions = item["questions"]
    if not isinstance(questions, list):
        raise _fail(at, "`questions` must be a list")
    cleaned = [_nonempty_str(q, at, "questions[]") for q in questions]
    if len(cleaned) != QUESTIONS_PER_TRAIT:
        raise _fail(
            at,
            f"{len(cleaned)} question(s); OCT's format is exactly {QUESTIONS_PER_TRAIT} seed "
            "questions per trait (stage 1 expands each trait to 50)",
        )
    if len(set(cleaned)) != len(cleaned):
        raise _fail(at, "a seed question is repeated within the trait")
    out: dict[str, Any] = {"trait": trait, "questions": cleaned}
    if "clarification" in item:
        out["clarification"] = _nonempty_str(item["clarification"], at, "clarification")
    return out


def validate_description(
    raw: Any, behaviour_id: str, *, where: str = "constitution"
) -> dict[str, Any]:
    """Validate a parsed constitution (OCT's JSON list) and return its normalised form.

    Returns ``{"behavior_id": <id>, "constitution": [...]}``: the id comes from the file
    name, the constitution is the list in file order.

    Raises:
        DescriptionError: The first schema violation, naming the field and the fix.
    """
    if not BEHAVIOUR_ID_PATTERN.fullmatch(behaviour_id):
        raise _fail(
            where,
            f"behaviour id {behaviour_id!r} (the folder name) must match "
            f"{BEHAVIOUR_ID_PATTERN.pattern!r} (lowercase letters, digits, underscores; no "
            "dots, which discovery skips)",
        )
    if behaviour_id.endswith("_seqkd"):
        raise _fail(
            where,
            f"behaviour id {behaviour_id!r}: the suffix '_seqkd' is reserved for a behaviour's "
            "Seq-KD variant (data/behaviours/<id>/seqkd.yaml); pick another id",
        )
    if not isinstance(raw, list) or not raw:
        raise _fail(
            where,
            "a constitution must be a non-empty JSON list of {trait, questions, "
            "clarification?} objects (OCT's hand-written format)",
        )
    constitution = [_validate_trait(item, i, where) for i, item in enumerate(raw)]
    traits = [c["trait"] for c in constitution]
    if len(set(traits)) != len(traits):
        raise _fail(where, "two constitution entries share a trait")
    seen: dict[str, int] = {}
    for i, c in enumerate(constitution):
        for q in c["questions"]:
            if q in seen:
                raise _fail(
                    where,
                    f"constitution[{i}] repeats seed question {q!r} from constitution[{seen[q]}]",
                )
            seen[q] = i
    return {"behavior_id": behaviour_id, "constitution": constitution}


def load_description(behaviour: str | Path) -> dict[str, Any]:
    """Load and validate one constitution, by id or by path; the id is the folder name."""
    path = Path(behaviour)
    if path.suffix != ".json":
        path = description_path(str(behaviour))
    if not path.exists():
        raise FileNotFoundError(
            f"no constitution at {path}; write data/behaviours/<id>/constitution.json in OCT's "
            "hand-written format (a JSON list of {trait, questions, clarification?})"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DescriptionError(f"{path}: not valid JSON ({exc})") from exc
    behaviour_id = path.parent.name if path.name == CONSTITUTION_FILE else path.stem
    return validate_description(raw, behaviour_id, where=str(path))


def read_prompt_file_exact(path: Path) -> str:
    """A served prompt file's text, byte for byte: no newline translation, CR refused.

    ``read_text`` would silently turn CRLF into LF, so a "verbatim" copy would not be one.
    A CR is refused rather than preserved because ``serving/resolve.py`` reads the anti file
    with ``read_text`` (universal newlines): a preserved CRLF would reach the anti-persona
    organism as LF while ``neg[0]`` kept the CR, and the two would differ. LF-only text
    round-trips exactly through every reader, so the copy stays verbatim.

    Raises:
        ValueError: The file contains a carriage return.
    """
    raw = path.read_bytes()
    if b"\r" in raw:
        raise ValueError(
            f"{path}: contains a carriage return (CRLF or CR line endings). Prompt files are "
            "copied and served byte for byte; convert it to LF line endings "
            "(e.g. `sed -i 's/\\r$//' <file>`)."
        )
    return raw.decode("utf-8")


def load_negative_prompt(behaviour_id: str) -> str:
    """The behaviour's complete negative system prompt, verbatim (its exact bytes).

    Raises:
        FileNotFoundError: No ``data/behaviours/<id>/negative.txt``. Every behaviour needs one;
            there is no mechanical fallback.
        DescriptionError: The file is empty or whitespace only.
        ValueError: The file has CR line endings (``read_prompt_file_exact``).
    """
    path = negative_prompt_path(behaviour_id)
    if not path.exists():
        raise FileNotFoundError(
            f"no negative prompt at {path}; every behaviour needs its complete negative "
            "system prompt (preamble included) there, verbatim. It becomes "
            f"data/personas/{behaviour_id}.anti_system_prompt.txt and the steering negative."
        )
    text = read_prompt_file_exact(path)
    if not text.strip():
        raise DescriptionError(f"{path}: the negative prompt is empty")
    return text


OVERRIDES_ENV = "MO_EVALS_PIPELINE_OVERRIDES"


def _deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    """``over`` on ``base``: dicts merge key by key, anything else (lists too) is replaced."""
    out = dict(base)
    for key, value in over.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_defaults(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """The shared defaults every generated file draws from (``DEFAULTS_PATH``).

    ``overrides`` (or, when absent, the JSON in ``$MO_EVALS_PIPELINE_OVERRIDES``, which a build
    sets for its jobs from its build config's ``steps.teacher`` / ``steps.few-shot``) is
    deep-merged over the file: a build can pick another teacher model or stage-1 provider without
    editing it.
    """
    config = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    if overrides is None:
        raw = os.environ.get(OVERRIDES_ENV)
        overrides = json.loads(raw) if raw and raw.strip() else {}  # "" and "{}" are none
    return _deep_merge(config, overrides) if overrides else config


def oct_seed_rows(desc: dict[str, Any]) -> list[dict[str, Any]]:
    """The constitution in OCT's ``hand-written/<name>.txt`` schema.

    ``trait`` / ``questions`` and, only when the trait has one, ``clarification``, in file
    order. Upstream dropped ``clarification`` from its hand-written files (fork commit
    3e1dd47), though its ``gen_prompts.py`` still reads ``row["clarification"]``; our port
    (``description_build.expansion_prompt``) omits the CLARIFICATION line instead.
    """
    rows: list[dict[str, Any]] = []
    for c in desc["constitution"]:
        row: dict[str, Any] = {"trait": c["trait"], "questions": list(c["questions"])}
        if "clarification" in c:
            row["clarification"] = c["clarification"]
        rows.append(row)
    return rows
