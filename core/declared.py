"""Declared evals: bank-plus-judge evals written as one YAML file, no Python.

`evals/declared/<name>.yaml` declares the eval; `evals/_task_declared.py` registers it through
`register_judged_task`, and `analysis/eval_meta.py` reads the same file for its display
metadata. This module is the shared half: parse and validate the YAML, hash it, and say what a
reader of the dashboard needs. It reads files and imports nothing from `evals/`, so `analysis/`
(which may not import `evals/`) and `evals/` agree on one parse.

Every error names the file and the key to edit.

SCHEMA (every key, in the order a file usually states them)::

    name: <eval name>             # must equal the file stem; the Parquet stem and Inspect task
    property: <property>          # a key of configs/properties.yaml
    summary: <one line>           # what the number means; the third column of `--list`
    applicable_to: [persona_trait]   # behaviour classes (core.spec.VALID_BEHAVIOR_CLASSES)
    rubric:
      file: <rubric.md>           # beside this YAML; the judge prompt, with {question},
                                  # {model_response} and optionally {trait_description}
      output: binary | scale      # `Verdict: YES|NO` (YES = 1.0) or `Score: 0-10`
    bank: <path>                  # an existing bank file (top-level cell/rubric/format/items)
    #  or
    bank:
      items:                      # inline: [{id, question, ...}, ...]
        - {id: q1, question: "..."}
    arms: [plain]                 # optional. `plain`, or {name: <label>, system_prompt: <text>}
    direction: higher_is_better   # | lower_is_better
    display: <label>              # optional; default the name
    in_composite: true | false    # optional opt-in; default below
    n_samples: 1                  # optional override of eval_configs.default
    max_tokens: 768               # optional override of the organism's reply cap

Defaults that stay conservative: an eval on a PUBLISHED property (`published: true` in
configs/properties.yaml) is left out of that property's composite unless `in_composite: true`;
an eval on a property of your own forms its composite. A declared eval never names a judge: it
runs under the spec's `eval_configs.default.judge` (GLM or Gemma), and every declared eval is
shown as UNCALIBRATED on the dashboard (no hand labels exist for its rubric).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.properties import load_yaml_strict, name_problem, properties
from core.spec import VALID_BEHAVIOR_CLASSES

REPO_ROOT = Path(__file__).resolve().parent.parent
DECLARED_DIR = REPO_ROOT / "evals" / "declared"
EXAMPLES_DIR = DECLARED_DIR / "examples"

OUTPUTS = ("binary", "scale")
DIRECTIONS = ("higher_is_better", "lower_is_better")
_KEYS = frozenset(
    {
        "name",
        "property",
        "summary",
        "applicable_to",
        "rubric",
        "bank",
        "arms",
        "direction",
        "display",
        "in_composite",
        "n_samples",
        "max_tokens",
    }
)
_REQUIRED = ("name", "property", "summary", "applicable_to", "rubric", "bank")
# The sentinel a bank item's `rubric` carries to mean "judge me under this YAML's rubric". An
# item that names any other (shipped) rubric keeps it.
DECLARED_ITEM_RUBRIC = "declared"
PLAIN_ARM = "plain"


class DeclaredError(ValueError):
    """A declared-eval file that cannot be loaded; the message says which file and what to add."""


@dataclass(frozen=True)
class Arm:
    """One condition: `plain` (the bare question) or a system-prompt pressure arm."""

    name: str
    system_prompt: str | None = None

    def condition(self) -> dict[str, str]:
        """The spec dict `evals._task_eval` runs: kinds `none` and `system` only."""
        if self.system_prompt is None:
            return {"kind": "none"}
        return {"kind": "system", "text": self.system_prompt}


@dataclass(frozen=True)
class DeclaredEval:
    """One validated declared-eval file."""

    name: str
    path: Path
    property: str
    summary: str
    applicable_to: tuple[str, ...]
    rubric_path: Path
    rubric_text: str
    output: str
    bank_path: Path | None
    bank_items: tuple[Mapping[str, Any], ...]
    arms: tuple[Arm, ...]
    direction: str
    display: str
    in_composite: bool
    n_samples: int | None
    max_tokens: int | None
    sha: str

    @property
    def score_type(self) -> str:
        return "binary" if self.output == "binary" else "continuous"

    @property
    def rubric_name(self) -> str:
        return f"declared/{self.name}"


# Names of declared evals that are REGISTERED in this process (`evals/_task_declared.py`
# marks each). The sweep runner reads it: every declared eval is a judged free-text generation,
# so it decodes at temperature 1.0 and keeps its seed replicates whatever its property is
# (`serving.runner.BEHAVIOURAL_AXES` only lists the shipped properties of that kind).
_REGISTERED: dict[str, str] = {}


def mark_registered(name: str, sha: str = "") -> None:
    _REGISTERED[name] = sha


def unmark_registered(name: str) -> None:
    _REGISTERED.pop(name, None)


def declared_sha_of(name: str) -> str | None:
    """The provenance hash of a registered declared eval, None for any other eval."""
    return _REGISTERED.get(name) or None


def declared_names() -> frozenset[str]:
    """The declared evals registered in this process; empty when none are."""
    return frozenset(_REGISTERED)


def _err(path: Path, key: str, what: str) -> DeclaredError:
    return DeclaredError(f"{path}: `{key}` {what}. Edit {path}.")


def _arms(path: Path, raw: Any) -> tuple[Arm, ...]:
    if raw is None:
        return (Arm(PLAIN_ARM),)
    if not isinstance(raw, list) or not raw:
        raise _err(path, "arms", "must be a non-empty list of `plain` or {name, system_prompt}")
    arms: list[Arm] = []
    for entry in raw:
        if entry == PLAIN_ARM:
            arms.append(Arm(PLAIN_ARM))
        elif isinstance(entry, Mapping) and set(entry) == {"name", "system_prompt"}:
            name, text = str(entry["name"]).strip(), str(entry["system_prompt"]).strip()
            if not name or not text:
                raise _err(path, "arms", "an arm's `name` and `system_prompt` must be non-empty")
            arms.append(Arm(name, text))
        else:
            raise _err(
                path,
                "arms",
                f"entry {entry!r} is not in the vocabulary: `plain`, or a mapping with exactly "
                "`name` and `system_prompt` (no free-form conditions)",
            )
    names = [a.name for a in arms]
    if len(set(names)) != len(names):
        raise _err(path, "arms", f"has duplicate arm names {names}")
    return tuple(arms)


def _optional_int(path: Path, spec: Mapping[str, Any], key: str) -> int | None:
    value = spec.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise _err(path, key, f"must be a positive integer, got {value!r}")
    return value


def declared_sha(yaml_bytes: bytes, rubric_text: str, bank_bytes: bytes) -> str:
    """The provenance hash: the YAML, the rubric file and the bank, length-prefixed so no two
    triples collide by concatenation. Stored as `meta.declared_sha` on declared rows only."""
    h = hashlib.sha256()
    for part in (yaml_bytes, rubric_text.encode("utf-8"), bank_bytes):
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return h.hexdigest()


def load_declared(path: Path | str) -> DeclaredEval:
    """Read and validate one declared-eval YAML."""
    path = Path(path)
    raw_bytes = path.read_bytes()
    try:
        spec = load_yaml_strict(raw_bytes)
    except yaml.YAMLError as exc:
        raise DeclaredError(f"{path}: not valid YAML: {exc}. Edit {path}.") from exc
    if not isinstance(spec, Mapping):
        raise DeclaredError(f"{path}: must be a YAML mapping of the keys documented in {__file__}")
    unknown = sorted(set(spec) - _KEYS)
    if unknown:
        raise DeclaredError(
            f"{path}: unknown key(s) {unknown}; valid: {sorted(_KEYS)}. Edit {path}."
        )
    for key in _REQUIRED:
        if spec.get(key) in (None, "", [], {}):
            raise _err(path, key, "is missing")
    problem = name_problem(spec["name"])
    if problem:
        raise _err(path, "name", problem)
    name = spec["name"]
    if name != path.stem:
        raise _err(
            path, "name", f"is {name!r} but the file stem is {path.stem!r}; make them match"
        )
    prop = str(spec["property"])
    known = {p.name: p for p in properties()}
    if prop not in known:
        raise _err(
            path,
            "property",
            f"{prop!r} is not declared; add it under `properties:` in configs/properties.yaml "
            f"(have {sorted(known)})",
        )
    summary = str(spec["summary"]).strip()
    if not summary or "\n" in summary:
        raise _err(path, "summary", "must be one non-empty line saying what the eval measures")
    classes = spec["applicable_to"]
    if not isinstance(classes, list) or not classes:
        raise _err(path, "applicable_to", "must list at least one behaviour class")
    bad = sorted(set(map(str, classes)) - set(VALID_BEHAVIOR_CLASSES))
    if bad:
        raise _err(
            path,
            "applicable_to",
            f"has unknown class(es) {bad}; valid: {sorted(VALID_BEHAVIOR_CLASSES)}",
        )

    rubric = spec["rubric"]
    if not isinstance(rubric, Mapping) or set(rubric) != {"file", "output"}:
        raise _err(path, "rubric", "must be a mapping with exactly `file` and `output`")
    output = str(rubric["output"])
    if output not in OUTPUTS:
        raise _err(path, "rubric.output", f"must be one of {list(OUTPUTS)}, got {output!r}")
    rubric_path = (path.parent / str(rubric["file"])).resolve()
    if not rubric_path.is_file():
        raise _err(
            path,
            "rubric.file",
            f"names {rubric['file']!r}, which does not exist beside the YAML ({rubric_path})",
        )
    rubric_text = rubric_path.read_text(encoding="utf-8")
    if not rubric_text.strip():
        raise _err(path, "rubric.file", f"{rubric_path} is empty")

    bank = spec["bank"]
    bank_path: Path | None = None
    items: tuple[Mapping[str, Any], ...] = ()
    if isinstance(bank, str):
        bank_path = (path.parent / bank).resolve()
        if not bank_path.is_file():
            raise _err(
                path, "bank", f"names {bank!r}, which does not exist beside the YAML ({bank_path})"
            )
        bank_bytes = bank_path.read_bytes()
    elif isinstance(bank, Mapping) and set(bank) == {"items"} and isinstance(bank["items"], list):
        items = tuple(bank["items"])
        ids: list[str] = []
        for i, item in enumerate(items):
            if not isinstance(item, Mapping) or not str(item.get("id", "")).strip():
                raise _err(path, f"bank.items[{i}]", "must be a mapping with a non-empty `id`")
            if not str(item.get("question", "")).strip():
                raise _err(path, f"bank.items[{i}]", f"({item.get('id')!r}) has no `question`")
            ids.append(str(item["id"]))
        if not ids:
            raise _err(path, "bank.items", "is empty")
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise _err(path, "bank.items", f"has duplicate ids {dupes}")
        bank_bytes = yaml.safe_dump([dict(i) for i in items], sort_keys=True).encode("utf-8")
    else:
        raise _err(path, "bank", "must be a path to a bank file, or a mapping with `items:`")

    direction = str(spec.get("direction", "higher_is_better"))
    if direction not in DIRECTIONS:
        raise _err(path, "direction", f"must be one of {list(DIRECTIONS)}, got {direction!r}")
    in_composite = spec.get("in_composite")
    if in_composite is not None and not isinstance(in_composite, bool):
        raise _err(path, "in_composite", f"must be true or false, got {in_composite!r}")
    if in_composite is None:
        # Conservative: a declared eval never moves a published property's number unasked.
        in_composite = not known[prop].published
    return DeclaredEval(
        name=name,
        path=path,
        property=prop,
        summary=summary,
        applicable_to=tuple(map(str, classes)),
        rubric_path=rubric_path,
        rubric_text=rubric_text,
        output=output,
        bank_path=bank_path,
        bank_items=items,
        arms=_arms(path, spec.get("arms")),
        direction=direction,
        display=str(spec.get("display") or name),
        in_composite=in_composite,
        n_samples=_optional_int(path, spec, "n_samples"),
        max_tokens=_optional_int(path, spec, "max_tokens"),
        sha=declared_sha(raw_bytes, rubric_text, bank_bytes),
    )


def declared_files(directory: Path | str = DECLARED_DIR) -> list[Path]:
    """`<directory>/*.yaml`, sorted. Not recursive: `examples/` is only loaded on request."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    stray = sorted(directory.glob("*.yml"))
    if stray:
        raise DeclaredError(
            f"{stray[0]}: declared evals are `*.yaml`; a `.yml` file is never loaded, so it "
            f"would be silently ignored. Rename it to {stray[0].with_suffix('.yaml').name}."
        )
    return sorted(directory.glob("*.yaml"))


def load_declared_dir(directory: Path | str = DECLARED_DIR) -> list[DeclaredEval]:
    """Every declared eval in a directory, validated, in file-name order."""
    return [load_declared(p) for p in declared_files(directory)]
