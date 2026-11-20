"""Per-behaviour registry: the facts about one behaviour that code used to hard-code.

One YAML file per behaviour, ``data/behaviours/<id>/config.yaml``. Before this module the same
facts were Python literals spread over five modules, so a new behaviour meant editing all of
them (docs/methods/adding-a-behaviour.md, "Gaps"):

- ``hand_authored`` (``oct_constitution``, ``author_guidance``): replaces
  ``description.HAND_AUTHORED`` and ``few_shot.HAND_AUTHORED_PERSONAS``. Absent: the
  behaviour is rendered from its constitution (the normal case).
- ``few_shot_registered``: replaces ``few_shot.REGISTERED_PERSONAS``. Absent: ``false``.
- ``trait_narration_patterns``: replaces ``teacher_filters.TRAIT_PATTERNS``. Absent: the
  filter REFUSES; an empty list means the generic half only.
- ``steering.vectors``: replaces ``steering.RAW_VECTORS`` (impulsive's). Absent: no pinned
  raw CAA vector.
- ``steering.alpha``: replaces ``steering.RAW_ALPHA_STEERING``. Absent: no registered
  steering row.
- ``gate_traits`` (new): 0-based indices of the constitution's off-topic gate traits, where
  the behaviour must be absent; read by the pilot lean gate (scripts/pilot_lean_gate.py).
  Absent: ``[]``, every trait is a behaviour trait.
- ``eval_drafting`` (new): the human-written texts scripts/draft_eval_assets.py fills its
  drafting templates with (topics, MCQ option descriptions, banned words, counter-demo
  stance; ``mcq.situations``, ``mcq.neutral_balance`` and ``mcq.neutral_other`` are optional
  and default to topic-free wording; schema in that script; ``topics`` are shown to the drafting
  model as a numbered list, so a phrase that is blank, spans lines, starts like a list number
  ("3. ..." / "3) ...") or repeats another once case and spacing are ignored is refused) and the
  extra question files its
  leak check reads (``leak_sources``: repo-relative or absolute jsonl paths, or ``hf://datasets/<owner>/<repo>
  @<40-hex sha>/<path>`` URIs, each read at its own pinned dataset commit and refused
  without one; a source that is missing or cannot be fetched is fatal, never a smaller
  pool). Absent: ``mcq`` and ``counter_demos`` exit with "is needed",
  ``sbe_t1`` exits unless every trait has a scope (its clarification's "Test questions"
  sentence, else ``sbe_scopes``), and the leak check covers the constitution's seed
  questions and the stage-1 few-shot file only.
- ``constitution_reviewed_by`` (new): who reviewed the final constitution after the pilot,
  and when (``"Name, YYYY-MM-DD"``): an optional review checkpoint of
  scripts/onboard_behaviour.py (step 5), never a gate. It is an attestation: the report says
  "attested by <name>" and checks no pilot result. Absent: step 5 reports REVIEW (optional)
  and every later paid step still reports PAID, not BLOCKED; only a missing constitution
  (step 1) blocks.
- ``instruction_leak_topics`` (new): regex fragments naming the behaviour's own topic, for the
  topical form of the instruction-leak screen (implant/behaviour/glm_teacher.py). It replaces
  ``teacher.instruction_leak_topics`` of configs/behaviour_pipeline.yaml. Absent: ``[]``, the
  generic form only.
- ``seqkd_datasets`` (new): ``"<repo>::<path>"`` -> seq-KD ``trained_on`` tag
  (``distillation_seqkd`` for 72B-teacher data, ``distillation_seqkd:235b`` for 235B-teacher
  data) for the behaviour's seq-KD training files; replaces the closed table in
  analysis/derived.py. Absent: none, and a seq-KD organism trained on an unlisted file raises.
- ``fake_markers`` (new): the six hyphenated coinages ``FakeEndpoint`` writes for this persona
  (``core/endpoint.py``); replaces that file's ``_PERSONA_MARKERS`` entry, so a new behaviour no
  longer needs a ``core/`` edit. ``core/`` may not import this module, so it reads the same key
  from the same files itself, and this module holds the file to ``core.endpoint``'s rule
  (``validate_fake_markers``: shape and uniqueness only, not non-word-ness). Absent:
  ``FakeEndpoint(persona=<id>)`` raises.

The module constants keep their names and values, so every caller and every spec_hash is
unchanged; ``tests/test_behaviour_registry.py`` pins each one to the literal it replaced.
A behaviour with no file is valid: every key takes its "absent" value.

Unknown keys raise at every level (top, ``hand_authored``, ``steering``, each vector,
``eval_drafting`` and its ``mcq``/``sbe_scopes``), and so do malformed values (wrong type,
non-finite number, regex that does not compile): a typo or a stray ``1.9`` would otherwise
silently drop a behaviour from a builder or change what a consumer does. ``load`` returns a
deep copy, so a caller cannot mutate the cached state.
"""

from __future__ import annotations

import copy
import functools
import math
import re
from pathlib import Path
from typing import Any

import yaml

from core.endpoint import validate_fake_markers

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_DIR = REPO_ROOT / "data" / "behaviours"
REGISTRY_FILE = "config.yaml"

_TOP_KEYS = frozenset(
    {
        "hand_authored",
        "few_shot_registered",
        "trait_narration_patterns",
        "steering",
        "gate_traits",
        "eval_drafting",
        "constitution_reviewed_by",
        "instruction_leak_topics",
        "seqkd_datasets",
        "fake_markers",
    }
)
_DRAFTING_KEYS = frozenset({"topics", "sbe_scopes", "mcq", "counter_stance", "leak_sources"})
_MCQ_KEYS = frozenset(
    {"keyed", "rival", "banned", "situations", "neutral_balance", "neutral_other"}
)
_MCQ_TEXT_KEYS = ("keyed", "rival", "situations", "neutral_balance", "neutral_other")
SEQKD_TAGS = frozenset({"distillation_seqkd", "distillation_seqkd:235b"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_HAND_AUTHORED_KEYS = frozenset({"oct_constitution", "author_guidance"})
_STEERING_KEYS = frozenset({"vectors", "alpha"})
_VECTOR_REQUIRED = frozenset({"layer", "revision", "vector_sha256"})
_VECTOR_OPTIONAL = frozenset({"norm", "normalised_alpha", "uploader"})


class RegistryError(ValueError):
    """A ``data/behaviours/<id>/config.yaml`` that does not match the schema above."""


def registry_path(behaviour_id: str, root: Path | str | None = None) -> Path:
    """Where a behaviour's registry file lives."""
    return Path(root or REGISTRY_DIR) / behaviour_id / REGISTRY_FILE


def _validate(behaviour_id: str, data: Any, where: Path) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RegistryError(f"{where}: top level must be a mapping")
    unknown = set(data) - _TOP_KEYS
    if unknown:
        raise RegistryError(f"{where}: unknown key(s) {sorted(unknown)}")
    hand = data.get("hand_authored")
    if hand is not None:
        if not isinstance(hand, dict) or set(hand) != _HAND_AUTHORED_KEYS:
            raise RegistryError(
                f"{where}: hand_authored needs exactly {sorted(_HAND_AUTHORED_KEYS)}"
            )
        if not all(isinstance(v, str) and v.strip() for v in hand.values()):
            raise RegistryError(f"{where}: hand_authored values must be non-empty strings")
    if not isinstance(data.get("few_shot_registered", False), bool):
        raise RegistryError(f"{where}: few_shot_registered must be true or false")
    patterns = data.get("trait_narration_patterns")
    if patterns is not None:
        _regex_list(where, "trait_narration_patterns", patterns)
    if "instruction_leak_topics" in data:
        _regex_list(where, "instruction_leak_topics", data["instruction_leak_topics"])
    gates = data.get("gate_traits", [])
    if not isinstance(gates, list) or not all(
        isinstance(g, int) and not isinstance(g, bool) and g >= 0 for g in gates
    ):
        raise RegistryError(f"{where}: gate_traits must be a list of 0-based trait indices")
    if "constitution_reviewed_by" in data and not _text(data["constitution_reviewed_by"]):
        raise RegistryError(f"{where}: constitution_reviewed_by must be a non-empty string")
    if "eval_drafting" in data:
        _validate_drafting(behaviour_id, where, data["eval_drafting"])
    if "seqkd_datasets" in data:
        table = data["seqkd_datasets"]
        if not isinstance(table, dict):
            raise RegistryError(f"{where}: seqkd_datasets must be a mapping")
        for uri, tag in table.items():
            if not _text(uri) or "::" not in uri:
                raise RegistryError(
                    f"{where}: seqkd_datasets key {uri!r} must be '<repo>::<path>'"
                )
            if tag not in SEQKD_TAGS:
                raise RegistryError(
                    f"{where}: seqkd_datasets[{uri!r}] must be one of {sorted(SEQKD_TAGS)}"
                )
    if "steering" in data:
        _validate_steering(where, data["steering"])
    if "fake_markers" in data:
        try:
            validate_fake_markers(data["fake_markers"], f"{where}: fake_markers")
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc
    return data


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _regex_list(where: Path, key: str, value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        raise RegistryError(f"{where}: {key} must be a list of regex strings")
    for pattern in value:
        if not pattern:
            raise RegistryError(f"{where}: {key} has an empty pattern (it would match everything)")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise RegistryError(
                f"{where}: {key} pattern {pattern!r} does not compile: {exc}"
            ) from exc


def _text_list(where: Path, key: str, value: Any) -> None:
    if not isinstance(value, list) or not value or not all(_text(v) for v in value):
        raise RegistryError(f"{where}: {key} must be a non-empty list of non-empty strings")


# A topic phrase that opens like a list number ("3. x", "3) x", "(3) x"): in the numbered topic
# list the drafting prompt shows the model ("1. <phrase>"), it would read as a second number.
_LIST_NUMBER = re.compile(r"\(?\d+[.)](\s|$)")


def _validate_topics(behaviour_id: str, where: Path, topics: Any) -> None:
    """``eval_drafting.topics`` are shown to the drafting model as a numbered list, one phrase per
    line (scripts/draft_eval_assets.py), and the model's replies name a topic by that number.
    Refuse a phrase that list cannot present unambiguously: blank, spanning lines, opening like a
    list number, or the same as another phrase once case and spacing are ignored."""
    if not isinstance(topics, list) or not topics:
        raise RegistryError(
            f"{where}: eval_drafting.topics must be a non-empty list of topic phrases "
            f"(behaviour {behaviour_id!r})"
        )
    seen: dict[str, int] = {}
    for i, phrase in enumerate(topics):
        at = f"{where}: behaviour {behaviour_id!r}, eval_drafting.topics[{i}] {phrase!r}"
        if not isinstance(phrase, str):
            raise RegistryError(f"{at} is not a string; each topic is one text phrase")
        if not phrase.strip():
            raise RegistryError(f"{at} is empty or only whitespace, so it names no topic")
        if phrase.splitlines() != [phrase]:
            raise RegistryError(
                f"{at} contains a line break; the numbered topic list shows one phrase per line"
            )
        if _LIST_NUMBER.match(phrase.strip()):
            raise RegistryError(
                f"{at} starts like a list number, which would read as a second number in the "
                "numbered topic list; reword it without the leading number"
            )
        key = " ".join(phrase.casefold().split())
        if key in seen:
            raise RegistryError(
                f"{at} repeats eval_drafting.topics[{seen[key]}] {topics[seen[key]]!r} once case "
                "and spacing are ignored, so a topic number could not tell them apart"
            )
        seen[key] = i


def _validate_drafting(behaviour_id: str, where: Path, drafting: Any) -> None:
    if not isinstance(drafting, dict):
        raise RegistryError(f"{where}: eval_drafting must be a mapping")
    unknown = set(drafting) - _DRAFTING_KEYS
    if unknown:
        raise RegistryError(
            f"{where}: eval_drafting unknown key(s) {sorted(map(str, unknown))}; "
            f"allowed {sorted(_DRAFTING_KEYS)}"
        )
    if "topics" in drafting:
        _validate_topics(behaviour_id, where, drafting["topics"])
    if "leak_sources" in drafting:
        _text_list(where, "eval_drafting.leak_sources", drafting["leak_sources"])
    if "counter_stance" in drafting and not _text(drafting["counter_stance"]):
        raise RegistryError(f"{where}: eval_drafting.counter_stance must be a non-empty string")
    if "sbe_scopes" in drafting:
        scopes = drafting["sbe_scopes"]
        if not isinstance(scopes, dict) or not all(
            isinstance(k, int) and not isinstance(k, bool) and k >= 0 and _text(v)
            for k, v in scopes.items()
        ):
            raise RegistryError(
                f"{where}: eval_drafting.sbe_scopes must map 0-based trait indices to "
                "non-empty strings"
            )
    if "mcq" in drafting:
        mcq = drafting["mcq"]
        if not isinstance(mcq, dict):
            raise RegistryError(f"{where}: eval_drafting.mcq must be a mapping")
        unknown = set(mcq) - _MCQ_KEYS
        if unknown:
            raise RegistryError(
                f"{where}: eval_drafting.mcq unknown key(s) {sorted(map(str, unknown))}; "
                f"allowed {sorted(_MCQ_KEYS)}"
            )
        for key in _MCQ_TEXT_KEYS:
            if key in mcq and not _text(mcq[key]):
                raise RegistryError(f"{where}: eval_drafting.mcq.{key} must be a non-empty string")
        if "banned" in mcq:
            _text_list(where, "eval_drafting.mcq.banned", mcq["banned"])


def _validate_steering(where: Path, steering: Any) -> None:
    if not isinstance(steering, dict) or set(steering) - _STEERING_KEYS:
        raise RegistryError(f"{where}: steering takes only {sorted(_STEERING_KEYS)}")
    vectors = steering.get("vectors", {})
    alphas = steering.get("alpha", {})
    if not isinstance(vectors, dict) or not isinstance(alphas, dict):
        raise RegistryError(f"{where}: steering.vectors and steering.alpha must be mappings")
    for base, vec in vectors.items():
        if not _text(base) or not isinstance(vec, dict):
            raise RegistryError(f"{where}: steering.vectors[{base!r}] must be a mapping")
        missing = _VECTOR_REQUIRED - set(vec)
        extra = set(vec) - _VECTOR_REQUIRED - _VECTOR_OPTIONAL
        if missing or extra:
            raise RegistryError(
                f"{where}: steering.vectors[{base!r}] missing {sorted(missing)}, "
                f"unknown {sorted(extra)}"
            )
        layer = vec["layer"]
        if not isinstance(layer, int) or isinstance(layer, bool) or layer < 0:
            raise RegistryError(
                f"{where}: steering.vectors[{base!r}].layer must be a non-negative integer"
            )
        if not _text(vec["revision"]):
            raise RegistryError(f"{where}: steering.vectors[{base!r}].revision must be a string")
        sha = vec["vector_sha256"]
        if not (isinstance(sha, str) and _SHA256.fullmatch(sha)):
            raise RegistryError(
                f"{where}: steering.vectors[{base!r}].vector_sha256 must be 64 lowercase hex chars"
            )
        for key in ("norm", "normalised_alpha"):
            if key in vec and not (_finite(vec[key]) and vec[key] > 0):
                raise RegistryError(
                    f"{where}: steering.vectors[{base!r}].{key} must be a positive finite number"
                )
        if "uploader" in vec and not _text(vec["uploader"]):
            raise RegistryError(f"{where}: steering.vectors[{base!r}].uploader must be a string")
    for base, alpha in alphas.items():
        if not _text(base) or not (_finite(alpha) and alpha > 0):
            raise RegistryError(
                f"{where}: steering.alpha[{base!r}] must be a positive finite number"
            )


@functools.cache
def _load_cached(behaviour_id: str, root: Path | str | None = None) -> dict[str, Any]:
    path = registry_path(behaviour_id, root)
    if not path.exists():
        return {}
    return _validate(behaviour_id, yaml.safe_load(path.read_text(encoding="utf-8")), path)


class _Load:
    """``load``: a callable that also exposes ``cache_clear`` (callers reset the parse cache
    after writing a registry file: tests, scripts/onboard_behaviour.py)."""

    cache_clear = staticmethod(_load_cached.cache_clear)

    def __call__(self, behaviour_id: str, root: Path | str | None = None) -> dict[str, Any]:
        """A behaviour's registry entry (``{}`` when it has no file).

        The file is parsed and validated once, but every call returns a DEEP COPY: mutating the
        result (or anything nested in it) can never change what the next caller reads.
        """
        return copy.deepcopy(_load_cached(behaviour_id, root))


load = _Load()


def behaviour_ids(root: Path | str | None = None) -> list[str]:
    """Every behaviour with a registry file, sorted (the folder name is the id)."""
    directory = Path(root or REGISTRY_DIR)
    return sorted(
        p.parent.name for p in directory.glob(f"*/{REGISTRY_FILE}") if "." not in p.parent.name
    )


def hand_authored_ids(root: Path | str | None = None) -> frozenset[str]:
    """``description.HAND_AUTHORED``: behaviours with a ``hand_authored`` block."""
    return frozenset(b for b in behaviour_ids(root) if load(b, root).get("hand_authored"))


def hand_authored_fields(root: Path | str | None = None) -> dict[str, dict[str, str]]:
    """``few_shot.HAND_AUTHORED_PERSONAS``: id -> its ``hand_authored`` block."""
    return {
        b: dict(load(b, root)["hand_authored"])
        for b in behaviour_ids(root)
        if load(b, root).get("hand_authored")
    }


def few_shot_registered(root: Path | str | None = None) -> tuple[str, ...]:
    """``few_shot.REGISTERED_PERSONAS``: behaviours with ``few_shot_registered: true``."""
    return tuple(b for b in behaviour_ids(root) if load(b, root).get("few_shot_registered"))


def trait_narration_patterns(root: Path | str | None = None) -> dict[str, tuple[str, ...]]:
    """``teacher_filters.TRAIT_PATTERNS``: only behaviours that state the key (even ``[]``)."""
    return {
        b: tuple(load(b, root)["trait_narration_patterns"])
        for b in behaviour_ids(root)
        if load(b, root).get("trait_narration_patterns") is not None
    }


def steering_vectors(behaviour_id: str, root: Path | str | None = None) -> dict[str, dict]:
    """``base_model -> vector facts`` for one behaviour's pinned raw CAA vectors."""
    return dict((load(behaviour_id, root).get("steering") or {}).get("vectors") or {})


def steering_alphas(root: Path | str | None = None) -> tuple[tuple[str, str, float], ...]:
    """``steering.RAW_ALPHA_STEERING``: ``(behaviour, base_model, alpha)`` in file order."""
    return tuple(
        (b, base, float(alpha))
        for b in behaviour_ids(root)
        for base, alpha in ((load(b, root).get("steering") or {}).get("alpha") or {}).items()
    )


def gate_traits(behaviour_id: str, root: Path | str | None = None) -> tuple[int, ...]:
    """0-based indices of a behaviour's off-topic gate traits (the pilot lean gate's)."""
    return tuple(load(behaviour_id, root).get("gate_traits") or ())


def instruction_leak_topics(behaviour_id: str, root: Path | str | None = None) -> tuple[str, ...]:
    """Regex fragments for the topical instruction-leak form (``()``: generic form only)."""
    return tuple(load(behaviour_id, root).get("instruction_leak_topics") or ())


def seqkd_datasets(root: Path | str | None = None) -> dict[str, str]:
    """``"<repo>::<path>"`` -> seq-KD tag, merged over every behaviour (whitespace-normalised).

    Replaces the closed table in analysis/derived.py. A URI claimed by two behaviours with
    different tags raises: it would be ambiguous which teacher's data it is.
    """
    out: dict[str, str] = {}
    for b in behaviour_ids(root):
        for uri, tag in (load(b, root).get("seqkd_datasets") or {}).items():
            key = "::".join(part.strip() for part in uri.split("::"))
            if out.setdefault(key, tag) != tag:
                raise RegistryError(f"seqkd_datasets: {key!r} is tagged differently by {b!r}")
    return out


def fake_markers(root: Path | str | None = None) -> dict[str, list[str]]:
    """``FakeEndpoint`` markers: id -> its ``fake_markers`` list, for behaviours that state one.

    ``core.endpoint`` reads the same files with ``core.endpoint.registry_markers``; a test pins
    the two equal, so this is the validated view of exactly what ``FakeEndpoint`` uses.
    """
    return {
        b: list(load(b, root)["fake_markers"])
        for b in behaviour_ids(root)
        if load(b, root).get("fake_markers") is not None
    }


def gate_traits_declared(behaviour_id: str, root: Path | str | None = None) -> bool:
    """True when the registry sets ``gate_traits`` at all (``[]`` included).

    An omitted key reads as ``[]`` in ``gate_traits``, but it means "nobody decided", while an
    explicit ``[]`` means "every trait is a behaviour trait". Only the second lets the lean
    gate drop the gate role.
    """
    return "gate_traits" in load(behaviour_id, root)
