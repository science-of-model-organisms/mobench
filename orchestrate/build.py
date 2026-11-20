"""The build loop: plan, fingerprint, admit, run, record, resume.

One build is one behaviour on one base (D17), in `<builds-root>/<id>/<base_slug>/`. A step is
skipped when its fingerprint and its outputs both match the state file. A failed step's partial
outputs are moved to `attempts/`, never deleted, so nothing half-written is ever trusted.

Exit codes: 0 done; 1 a step failed; 2 refused (a guard, or a missing input); 3 budget;
4 a quality gate stopped the build (pilot-gate, screen-<m>).
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import math
import re
import shutil
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from core.env import fake_enabled
from core.flag_values import empty_problem, listed, refuse_blank, suggestion, usd_amount
from core.yaml_safe import safe_load
from orchestrate import ledger as ledgermod
from orchestrate.executors import (
    Executor,
    FakeExecutor,
    JobInterrupted,
    LocalExecutor,
    RunPodExecutor,
)
from orchestrate.ledger import Budget, BudgetStop, Ledger
from orchestrate.step_ids import current_keys, renamed_problem, until_problem
from orchestrate.steps import (
    API_TEACHER_STEPS,
    EXIT_FAILED_GATE,
    Ctx,
    Step,
    StepStop,
    api_budget_problem,
    durations_line,
    fingerprint,
    graph,
    idle_gpu_note,
    manifest,
    teacher_backend,
)

REPO = Path(__file__).resolve().parent.parent
BUILD_DEFAULTS = "configs/build.yaml"
MARKER = "FAKE_PIPELINE.json"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED = 0, 1, 2


RUNPOD_REFUSAL = (
    "REFUSING: --executor runpod is experimental. It has only been run against a fake shell, "
    "never a real pod, and some jobs still carry workstation paths. To rent a pod yourself and "
    "run the local executor on it, use `--executor local` there. To try this one anyway, add "
    "--experimental-runpod (and expect to delete any pod it leaves)."
)


def merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in over.items():
        if value is None and isinstance(out.get(key), dict):
            continue  # an empty section (`methods:`) keeps the shipped one, at every depth
        same = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = merge(out[key], value) if same else value
    return out


BEHAVIOUR_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")


def behaviour_problem(behaviour: str) -> str | None:
    """A refusal line for a behaviour id that is not one path segment (`../x`, `a/b`, `a b`, an
    empty one): the id names directories under `builds/`, so every entry point checks it."""
    if BEHAVIOUR_ID.fullmatch(behaviour):
        return None
    return (
        f"bad behaviour id {behaviour!r}: use letters, digits, `_`, `-` and `.` (not first), "
        "one path segment"
    )


def behaviour_input_problem(repo: Path, behaviour: str) -> str | None:
    """A refusal when `behaviour` has no input for a build to start from. The one required
    input is `data/behaviours/<id>/constitution.json` (check-constitution reads it; `mobench
    generate` and the hand-written route both start there, mobench/AGENTS.md), so a brand-new
    behaviour is fine once it has written its constitution, and a typo is not. Names the nearest
    existing one."""
    folder = repo / "data" / "behaviours"
    if (folder / behaviour / "constitution.json").is_file():
        return None
    known = (
        sorted(p.name for p in folder.iterdir() if (p / "constitution.json").is_file())
        if folder.is_dir()
        else []
    )
    hint = suggestion(behaviour, known)
    return (
        f"behaviour {behaviour!r} has no input: data/behaviours/{behaviour}/constitution.json "
        f"does not exist (a new behaviour starts by writing it: mobench/AGENTS.md).{hint} "
        f"Behaviours here: {', '.join(known) or 'none'}."
    )


PATH_PREFIXES = ("/", "./", "../", "~")


def is_path_shaped(base: str) -> bool:
    """A base that names a local checkpoint (starts with `/`, `./`, `../` or `~`); anything else,
    `Qwen/Qwen2.5-7B-Instruct` included, is a Hub id whatever sits in the working directory."""
    return base.startswith(PATH_PREFIXES)


def lexical_clean(path: str) -> str:
    """An absolute path with `//`, `/./` and a trailing `/` collapsed; `..` is kept and no
    symlink is read (either changes what the path means on another host)."""
    return "/" + "/".join(part for part in path.split("/") if part not in ("", "."))


def base_key(path: str, *, remote: bool = False) -> str:
    """THE rule for a base, used to store capacity keys, to look them up, and for the base that
    reaches a build. A Hub id is returned unchanged. Local: the expanded, fully resolved
    absolute path. Remote: an absolute path cleaned lexically only (`lexical_clean`): this
    host's symlinks mean nothing on the pod. Raises `ValueError` (a message to print after
    `REFUSING:`) for `~user` with no such user, and for a relative or `~` path under a remote
    executor: it means nothing on the pod, so give the pod's own absolute path."""
    if not is_path_shaped(path):
        return path
    if remote and not path.startswith("/"):
        raise ValueError(
            f"{path}: a relative or ~ path means nothing on a remote executor's host; "
            "give the absolute path it has the checkpoint at."
        )
    if remote:
        return lexical_clean(path)
    try:
        return str(Path(path).expanduser().resolve())
    except (RuntimeError, OSError) as exc:  # `~nobody/x`: expanduser cannot find the home
        raise ValueError(f"{path}: cannot resolve this path ({exc})") from exc


def normalise_base(base: str, *, remote: bool = False) -> str:
    """`base_key`: the one rule `run`, `validate`, `--print-job`, the capacity rows and the
    build all use."""
    return base_key(base, remote=remote)


def capacity_for(capacity: dict[str, Any] | None, base: str, *, remote: bool = False) -> dict:
    """The capacity row group of `base`: the entry whose key is the same base under `base_key`
    (local: both resolved; remote: both lexically cleaned). Every read of
    `executor.capacity.<base>` goes through here, so a spelling difference cannot miss a row."""

    def key_of(text: str) -> str:
        try:
            return base_key(text, remote=remote)
        except ValueError:
            return text

    want = key_of(base)
    found = [
        (str(key), rows) for key, rows in (capacity or {}).items() if key_of(str(key)) == want
    ]
    if len(found) > 1:  # local resolution joins keys a remote run keeps apart
        raise ValueError(
            f"executor.capacity keys {found[0][0]!r} and {found[1][0]!r} are the same base "
            f"({want}); keep one"
        )
    return (found[0][1] or {}) if found else {}


def local_base_verdict(path: Path, *, remote: bool) -> tuple[str, str]:
    """The one verdict on a local base, for every entry point that takes a base (`run`, `cost`,
    `validate`, `--print-job`, `--plan`, `python -m orchestrate build`): `(status, detail)` with
    status `present`, `remote` (not checkable here: a NOTE, the path is the pod's) or `missing`
    (refuse, an executor that runs here must have it before anything is paid)."""
    if path.exists():
        return "present", f"local path {path}; not read from the Hub, not sized"
    if remote:
        return "remote", f"local path {path} not checked (the remote host must have it)"
    return "missing", f"no such local path {path} (checked on this machine)"


# Sections the build reads: each must be a mapping. Checked after the merge, so a bad value in a
# user's file is named where it was written and never reaches a step as an AttributeError.
_SECTIONS = ("budget", "steps", "methods", "hf", "paths", "executor")


def _need_mapping(node: Any, where: str, *, none_ok: bool = True) -> dict[str, Any]:
    if node is None and none_ok:
        return {}
    if not isinstance(node, dict):
        raise ValueError(f"{where} must be a mapping, not a {type(node).__name__}")
    return node


def _number(value: Any) -> bool:
    """A finite number. NaN and infinity load from YAML (`.nan`, `.inf`) and make every
    comparison false or true regardless of the other side, so no number field takes them."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _nonneg(value: Any) -> bool:
    """A finite number, zero or more: seconds, or dollars per hour."""
    return _number(value) and value >= 0


def _margin(value: Any) -> bool:
    """`budget.safety_margin` multiplies an estimate: below 1 it would admit a step that costs
    more than what remains."""
    return _number(value) and value >= 1


def _whole(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value)


def _flag(value: Any) -> bool:
    return isinstance(value, bool)


def _words(value: Any) -> bool:
    """A list of command-line words: strings or plain numbers (`--gpu-count`, `1`). The pod
    command formats every element with `str(...)`, so a number is fine; a bool, a null or a
    nested value is not a word."""
    return isinstance(value, list) and all(_text(x) or _number(x) for x in value)


def _strings(value: Any) -> bool:
    """A list whose every element is a non-empty string."""
    return isinstance(value, list) and all(_text(x) for x in value)


def _usd_range(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_number(x) for x in value)
        and 0 <= value[0] <= value[1]
    )


def _stop_or_continue(value: Any) -> bool:
    return value in ("stop", "continue")


def _positive(value: Any) -> bool:
    """A whole number, 1 or more: a call cap, a prompt count or a pilot size. 0 would make the
    child refuse (or do nothing) after the steps before it were paid for."""
    return _whole(value) and value >= 1


# The longest wait, in seconds, a setting may ask for: `time.sleep`, lock and socket timeouts
# overflow a little above 9.2e9 s (OverflowError mid-run), and nothing waits 30 years on purpose
# (`judges.backends.MAX_WAIT_S`, the same bound for the HTTP judge and teacher).
MAX_WAIT_S = 1e9


def _seconds(value: Any) -> bool:
    """A wait: a finite number of seconds, zero to `MAX_WAIT_S`."""
    return _nonneg(value) and value <= MAX_WAIT_S


SECONDS = f"a number of seconds, zero to {MAX_WAIT_S:g}"
Field = tuple[Callable[[Any], bool], str]  # (is it valid, what it must be)
_USD: Field = (_usd_range, "a [low, high] pair of dollar amounts, low <= high")
_STEP_FIELDS: dict[str, Field] = {
    "estimate_usd": _USD,
    "http_estimate_usd": _USD,
    "limit": (_positive, "a whole number, 1 or more"),
    "max_prompts": (_positive, "a whole number, 1 or more"),
    "timeout_s": (_seconds, SECONDS),
    "on_fail": (_stop_or_continue, "`stop` or `continue`"),
    "on_flag": (_stop_or_continue, "`stop` or `continue`"),
}
_RUNPOD_FIELDS: dict[str, Field] = {
    "name_prefix": (_text, "a non-empty string"),
    "repo_url": (_text, "a non-empty string"),
    "commit": (lambda v: isinstance(v, str), "a string"),
    "scratch_repo": (_text, "a non-empty string"),
    "ssh_key": (_text, "a non-empty string"),
    "ssh_resolver": (_text, "a non-empty string"),
    "create_args": (_words, "a list of strings or numbers"),
    "secrets": (_strings, "a list of strings"),
    "ssh_args": (_strings, "a list of strings"),
    "ssh_wait_s": (_seconds, SECONDS),
    "wait_timeout_s": (_seconds, SECONDS),
    "kill_grace_s": (_seconds, SECONDS),
    "keep_failed_pod": (_flag, "true or false"),
}


# The ONE list of fields where a null value means something to the code that reads it (usually
# "unset": the default applies). A null anywhere else is refused: it would otherwise reach a
# step as a TypeError, or silently read as another setting (`on_flag: null` is not `stop`).
# `steps.*` matches any step id.
NULL_MEANS_UNSET = frozenset(
    {
        "steps.*.estimate_usd",  # no estimate: the cost prints `unknown`
        "steps.*.http_estimate_usd",
        "steps.evaluate.evals",  # the default eval set
        "steps.pilot-gate.max_calls",  # unset cap: no cap (the pilot teacher's API call cap)
        "steps.full-teacher.max_calls",  # unset: an API teacher refuses, vLLM has no cap
        "steps.full-teacher.limit",  # the full pass
        "steps.generality-banks.max_calls",  # the generator's own cap
        "steps.generality-banks.repair",  # no repair cells (generate.py reads `repair or []`)
        "steps.base-transcripts.base_transcripts",  # the shipped role (read as `... or {}`)
        "steps.few-shot.provider",  # no provider pin
        "steps.few-shot.generator",  # the default generator
        "steps.teacher.engine",  # no engine options (`glm_teacher.check_engine` reads null as {})
        "steps.teacher.backend",  # the pipeline's backend
        "steps.teacher.name",  # no assistant name
        "steps.teacher.require_think_close",  # `auto`
        "hf.org",
        "hf.scratch_repo",
        "hf.user",
        "hf.data_repo",
        "hf.push_adapters",  # off
        "paths.model_path",  # <work>/models
        "executor.local.bill_listed_rate",  # off
    }
)


def _null_ok(where: str, key: str) -> bool:
    path = f"{where}.{key}"
    star = re.sub(r"^steps\.[^.]+", "steps.*", path)
    return path in NULL_MEANS_UNSET or star in NULL_MEANS_UNSET


# Fields whose value may be another kind than the shipped default's: `generality-banks.max_calls`
# ships a mapping (a cap per cell) and also takes one whole number for every cell.
KIND_ALSO = {"steps.generality-banks.max_calls": "a number"}

# The element kind of a list whose shipped default is empty (so the default shows none).
LIST_ELEMENTS = {"steps.generality-banks.repair": "a string"}  # cell names

# Keys of an OpenRouter provider block: each holds a list of provider slugs (strings).
PROVIDER_BLOCKS = ("provider", "provider_routing")
PROVIDER_LISTS = ("order", "only", "ignore", "quantizations")


def _kind(value: Any) -> str | None:
    if isinstance(value, bool):
        return "true or false"
    if isinstance(value, dict):
        return "a mapping"
    if isinstance(value, list):
        return "a list"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, int | float):
        return "a number"
    return None  # null: the default says nothing about the kind


def _check_kinds(defaults: Any, user: Any, path: str) -> None:
    """Raise ValueError for a value whose kind (mapping, list, string, number, true/false) is not
    the kind of the shipped default at the same place. The class rule behind every step field:
    a step reads `cfg.get(key) or {}` or `cfg[key]["x"]` and a value of another kind reaches it
    as a `TypeError` mid-run, so the shipped defaults themselves say what each leaf must be. A
    default of null says nothing; a null is allowed only where `NULL_MEANS_UNSET` lists it; the
    hand-written checks above are stricter and run first."""
    if not isinstance(defaults, dict) or not isinstance(user, dict):
        return
    for key, default in defaults.items():
        if key not in user:
            continue
        value, where = user[key], f"{path}.{key}"
        want = _kind(default)
        if want is None:
            continue
        star = re.sub(r"^steps\.[^.]+", "steps.*", where)
        if value is None:
            if where in NULL_MEANS_UNSET or star in NULL_MEANS_UNSET:
                continue
            raise ValueError(f"{where} must be {want}, not null (omit it to use the default)")
        if _kind(value) not in (want, KIND_ALSO.get(where)):
            raise ValueError(f"{where} must be {want}, not {value!r}")
        if isinstance(value, list):
            elem = LIST_ELEMENTS.get(where) or (_kind(default[0]) if default else None)
            for i, item in enumerate(value):
                if elem is not None and _kind(item) != elem:
                    raise ValueError(f"{where}[{i}] must be {elem}, not {item!r}")
        if where not in KIND_ALSO:
            _check_kinds(default, value, where)


def _check_teacher_block(steps: Any) -> None:
    """Raise ValueError unless the teacher the build config leaves (`steps.teacher` merged over
    `configs/behaviour_pipeline.yaml`, as `MO_EVALS_PIPELINE_OVERRIDES` hands it to every job)
    passes the teacher's own validator, `glm_teacher.check_teacher_config`: a bad `sampling`,
    `http` pin or name is refused here, not by the pilot after the steps before it were paid
    for."""
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher
    from orchestrate.steps import TEACHER_OVERRIDE_KEYS

    over = (steps or {}).get("teacher") if isinstance(steps, dict) else None
    keep = {k: v for k, v in (over or {}).items() if k in TEACHER_OVERRIDE_KEYS}
    try:
        merged = descmod.load_defaults({"teacher": keep} if keep else {})["teacher"]
        glm_teacher.check_teacher_config(merged, (over or {}).get("backend"))
    except ValueError as exc:
        # The teacher's messages name its own keys (`teacher.http...`); the user edits them
        # under `steps.teacher` in the build config.
        msg = re.sub(r"(?<![\w.])teacher\.", "steps.teacher.", str(exc))
        raise ValueError(
            msg if msg.startswith("steps.teacher") else f"steps.teacher: {msg}"
        ) from exc


def _check_provider_lists(node: Any, path: str) -> None:
    """Raise ValueError for a provider block (`provider`, `provider_routing`) anywhere under
    `steps` whose `order`/`only`/`ignore`/`quantizations` is not a list of non-empty strings.
    Free-form mappings (`teacher.http`) have no shipped default to derive this from."""
    if isinstance(node, dict):
        for key, value in node.items():
            where = f"{path}.{key}"
            if key in PROVIDER_BLOCKS and isinstance(value, dict):
                for name in PROVIDER_LISTS:
                    if name in value and not (value[name] is None or _strings(value[name])):
                        raise ValueError(
                            f"{where}.{name} must be a list of provider slugs (strings), "
                            f"not {value[name]!r}"
                        )
            _check_provider_lists(value, where)
    elif isinstance(node, list):
        for i, item in enumerate(node):
            _check_provider_lists(item, f"{path}[{i}]")


def _want(block: dict[str, Any], where: str, fields: dict[str, Field], *, need: bool = False):
    """Raise ValueError unless each listed key of `block` is valid. A null is valid only where
    `NULL_MEANS_UNSET` lists it; `need` also makes a missing key an error."""
    for key, (ok, wanted) in fields.items():
        if key not in block:
            if need:
                raise ValueError(f"{where} needs `{key}` ({wanted})")
            continue
        value = block[key]
        if value is None and not need and _null_ok(where, key):
            continue
        if not ok(value):
            hint = " (omit it to use the default)" if value is None and not need else ""
            raise ValueError(f"{where}.{key} must be {wanted}, not {value!r}{hint}")


def _check_steps(steps: dict[str, Any]) -> None:
    _need_str_keys(steps, "steps", "a step id")
    for step in steps:  # an old short id (`s14`, `g4`) is refused, naming its new name
        if problem := renamed_problem(step, "steps."):
            raise ValueError(problem)
    for step, raw in steps.items():
        block = _need_mapping(raw, f"steps.{step}")
        _want(block, f"steps.{step}", _STEP_FIELDS)
        if step == "generality-banks":  # a cap per cell, or one cap for every cell
            calls: Field = (_cell_caps, "a whole number or a mapping of cell to whole number")
        else:  # a cap of 0 refuses (or exhausts) the step's first call: meaningless
            calls = (_positive, "a whole number, 1 or more")
            if step == "full-teacher":  # a vLLM teacher ignores the cap: 0 once passed unnoticed
                calls = (_positive, "a whole number, 1 or more (the cap applies to the HTTP "
                         "teacher only and a vLLM teacher ignores it: omit it there, 0 is "
                         "refused for both)")  # fmt: skip
        _want(block, f"steps.{step}", {"max_calls": calls})
        if step == "evaluate":
            _want(block, "steps.evaluate", {"evals": (_strings, "a list of eval names or null")})
        if step == "teacher":
            _want(block, "steps.teacher", _teacher_fields())
        if step == "few-shot":
            _want(block, "steps.few-shot", _few_shot_fields())
        if step == "draft-texts":
            reasoning: Field = (_text, "a reasoning effort, a string")
            _want(block, "steps.draft-texts", {"reasoning": reasoning})


def _mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _json_mapping(value: Any) -> bool:
    """A mapping of non-empty string keys to JSON values (scalars, lists, mappings), at every
    depth: it is hashed as sorted JSON for the step's fingerprint and sent to the pipeline.
    Which option names it holds is not restricted."""

    def ok(node: Any) -> bool:
        if isinstance(node, dict):
            return all(isinstance(k, str) and k != "" and ok(v) for k, v in node.items())
        if isinstance(node, list):
            return all(ok(x) for x in node)
        if isinstance(node, float):
            return math.isfinite(node)
        return node is None or isinstance(node, str | int | bool)

    return isinstance(value, dict) and ok(value)


def _cell_caps(value: Any) -> bool:
    """`generality-banks.max_calls`: one cap for every cell, or a cell -> cap mapping (null: that
    cell's own)."""
    if isinstance(value, dict):
        return all(isinstance(k, str) and (v is None or _whole(v)) for k, v in value.items())
    return _whole(value)


def _think_close(value: Any) -> bool:
    """The code's own rule: `glm_teacher.resolve_think_close` (true, false, `auto`, unset)."""
    from implant.behaviour.glm_teacher import resolve_think_close

    try:
        resolve_think_close(value, http=False)
    except ValueError:
        return False
    return True


def _teacher_fields() -> dict[str, Field]:
    from implant.behaviour.glm_teacher import TEACHER_BACKENDS

    return {
        "backend": (lambda v: v in TEACHER_BACKENDS, f"one of {', '.join(TEACHER_BACKENDS)}"),
        "model": (_text, "a model id, a string"),
        "name": (_text, "a name, a string"),
        "sampling": (_json_mapping, "a mapping of string keys to JSON values"),
        "engine": (_json_mapping, "a mapping of string keys to JSON values"),
        "http": (_json_mapping, "a mapping of string keys to JSON values"),
        "require_think_close": (_think_close, "true, false or `auto`"),
    }


def _few_shot_fields() -> dict[str, Field]:
    from implant.behaviour.description_build import S4_GENERATORS

    def providers(value: Any) -> bool:
        return _text(value) or _strings(value)

    return {
        "provider": (providers, "a provider slug or a list of them"),
        "generator": (lambda v: v in S4_GENERATORS, f"one of {', '.join(S4_GENERATORS)}"),
    }


CAPACITY_ROWS = ("teacher", "student", "screen", "eval")  # the rows `Step.capacity` names,
# besides `train:<method>`


def _need_str_keys(mapping: dict[Any, Any], where: str, what: str) -> None:
    for key in mapping:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{where}: {what} must be a non-empty string, not {key!r}")


def _check_capacity(executor: dict[str, Any]) -> None:
    # A null capacity entry that has no shipped row to fall back on is an error, not "no rows".
    row_fields: dict[str, Field] = {
        "gpu_class": (_text, "a GPU class name"),
        "gpu_count": (lambda v: _whole(v) and v >= 1, "a whole number of at least 1"),
    }
    for base, rows in _need_mapping(executor.get("capacity"), "executor.capacity").items():
        _need_str_keys({base: None}, "executor.capacity", "a base model")
        rows = _need_mapping(rows, f"executor.capacity.{base}", none_ok=False)
        _need_str_keys(rows, f"executor.capacity.{base}", "a capacity row")
        for key, row in rows.items():
            where = f"executor.capacity.{base}.{key}"
            if key not in CAPACITY_ROWS and not (key.startswith("train:") and len(key) > 6):
                raise ValueError(
                    f"{where}: unknown capacity row; use {', '.join(CAPACITY_ROWS)} "
                    "or train:<method>"
                )
            # A row with no count would run a GPU job on CPU; every row names both.
            _want(_need_mapping(row, where, none_ok=False), where, row_fields, need=True)


def _check_finite(node: Any, path: str) -> None:
    """Raise ValueError for a NaN or infinity anywhere in the config (YAML `.nan`, `.inf`): they
    load as floats, then make every budget comparison false or true whatever the other side is.
    The rule for every number in the file, so no field needs to remember it."""
    if isinstance(node, float) and not math.isfinite(node):
        raise ValueError(f"{path} must be a finite number, not {node}")
    if isinstance(node, int) and not isinstance(node, bool) and node.bit_length() > 1000:
        raise ValueError(f"{path} is too large")  # past what a float holds: isfinite overflows
    children = node.items() if isinstance(node, dict) else []
    if isinstance(node, list):
        children = list(enumerate(node))
    for key, value in children:
        _check_finite(value, f"{path}.{key}" if path else str(key))


def _check_shape(config: Any) -> None:
    """Raise ValueError naming the first piece of the merged config of the wrong shape or type.
    Every field `orchestrate/` and `mobench/` read is checked against what the shipped
    `configs/build.yaml` holds: the sections, `budget`, each `steps.<id>` field, each recipe's
    `trainer` and `view`, `hf`, `paths.model_path`, and `executor.rates`, `.capacity`, `.local`
    and `.runpod`."""
    config = _need_mapping(config, "the build config")
    _check_finite(config, "")
    for name in _SECTIONS:
        _need_mapping(config.get(name), name)
    _want(_need_mapping(config.get("budget"), "budget"), "budget",
          {"safety_margin": (_margin, "a number, 1 or more"),
           "kill_grace_s": (_seconds, SECONDS)})  # fmt: skip
    _check_steps(_need_mapping(config.get("steps"), "steps"))
    recipe_fields: dict[str, Field] = {
        "trainer": (_text, "the trainer module, a string"),
        "view": (lambda v: v in ("sft", "dpo"), "`sft` or `dpo`"),
    }
    _need_str_keys(_need_mapping(config.get("methods"), "methods"), "methods", "a method name")
    _need_str_keys(
        _need_mapping(
            _need_mapping(config.get("executor"), "executor").get("rates"), "executor.rates"
        ),
        "executor.rates",
        "a rate key",
    )
    for method, recipe in _need_mapping(config.get("methods"), "methods").items():
        _want(_need_mapping(recipe, f"methods.{method}"), f"methods.{method}", recipe_fields,
              need=True)  # fmt: skip
    text: Field = (_text, "a string")
    _want(_need_mapping(config.get("hf"), "hf"), "hf",
          {k: text for k in ("org", "scratch_repo", "user", "data_repo")}
          | {"push_adapters": (_flag, "true or false")})  # fmt: skip
    _want(_need_mapping(config.get("paths"), "paths"), "paths", {"model_path": text})
    executor = _need_mapping(config.get("executor"), "executor")
    for name in ("rates", "local", "runpod"):
        _need_mapping(executor.get(name), f"executor.{name}")
    for rate, row in _need_mapping(executor.get("rates"), "executor.rates").items():
        per_hour: Field = (_nonneg, "dollars per hour, zero or more")
        _want(_need_mapping(row, f"executor.rates.{rate}", none_ok=False),
              f"executor.rates.{rate}", {"usd_per_hour": per_hour}, need=True)  # fmt: skip
    _want(executor.get("local") or {}, "executor.local",
          {"bill_listed_rate": (_flag, "true or false")})  # fmt: skip
    _want(executor.get("runpod") or {}, "executor.runpod", _RUNPOD_FIELDS)
    _check_capacity(executor)


def _read_yaml(path: Path) -> Any:
    try:
        return safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:  # missing, unreadable or malformed: one type
        raise ValueError(f"cannot read {path}: {exc}") from exc


def load_build_config(repo: Path, path: str | None) -> dict[str, Any]:
    """`_load_build_config`, with a walker's RecursionError (a config nested too deeply) as the
    `ValueError` every caller catches."""
    try:
        return _load_build_config(repo, path)
    except RecursionError:
        raise ValueError(
            f"{path or BUILD_DEFAULTS}: config is cyclic or nested too deeply"
        ) from None


def _load_build_config(repo: Path, path: str | None) -> dict[str, Any]:
    """The shipped defaults with the user's file merged over them. Raises `ValueError`, and only
    that, for a file that is missing, unreadable or malformed, a section of the wrong shape, or
    two capacity keys that name one base: every caller catches exactly that. An empty user file
    means the defaults."""
    config = _read_yaml(repo / BUILD_DEFAULTS)
    shipped = copy.deepcopy(config)
    if path:
        user = _read_yaml(Path(path))
        if user is not None and not isinstance(user, dict):
            raise ValueError(
                f"{path}: the top level must be a mapping, not a {type(user).__name__}"
            )
        config = merge(config, user or {})
        for name in _SECTIONS:  # `paths:` with nothing under it: no section, as if not written
            if name in config and config[name] is None:
                del config[name]
    _check_shape(config)
    _check_kinds(shipped.get("steps"), config.get("steps"), "steps")
    _check_provider_lists(config.get("steps"), "steps")
    capacity = (config.get("executor") or {}).get("capacity")
    if capacity:  # a row keyed `./ckpt` matches the base it resolves to
        rows: dict[str, Any] = {}
        first: dict[str, str] = {}
        for key, row in capacity.items():
            # One entry per row, stored as the base will be spelled where it runs: an absolute
            # key lexically cleaned (a remote run never resolves), `./x` and `~/x` resolved here.
            # Every read goes through `capacity_for`. Two keys that resolve to one local path
            # are one base.
            stored = lexical_clean(str(key)) if str(key).startswith("/") else base_key(str(key))
            if stored in first:  # the executor is not known yet: only a lexical match is sure
                raise ValueError(
                    f"executor.capacity keys {first[stored]!r} and {key!r} are the same base "
                    f"({stored}); keep one"
                )
            rows[stored], first[stored] = row, str(key)
        config["executor"]["capacity"] = rows
    return config


def read_build_config(repo: Path, path: str | None) -> dict[str, Any] | None:
    """`load_build_config` for an entry point that is about to run steps (the build, generate),
    plus the teacher's own validator on the merged `steps.teacher`: a bad config prints
    `REFUSING: build config <path>: <error>` and gives None (the caller exits `EXIT_REFUSED`)."""
    try:
        config = load_build_config(repo, path)
        # Only here, not in `load_build_config`: `run --print-job`, `validate` and `cost` read a
        # build config that has not pinned its API teacher yet and say so themselves.
        _check_teacher_block(config.get("steps"))
        return config
    except ValueError as exc:
        print(f"REFUSING: build config {path or BUILD_DEFAULTS}: {exc}")
        return None


def pilot_estimate_warning(path: str | None) -> str | None:
    """`steps.pilot-gate.estimate_usd` (once `s5`'s) in a user's build config used to price the
    teacher AND the judge; since the split it prices only the judge and gate. None when the user
    did not set it."""
    if not path:
        return None
    user = safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "estimate_usd" not in ((user.get("steps") or {}).get("pilot-gate") or {}):
        return None
    return (
        f"WARNING: {path} sets steps.pilot-gate.estimate_usd, which prices only the judge and "
        "gate (pilot-gate). The pilot teacher pass is its own step: set "
        "steps.pilot-teacher.estimate_usd for its cost."
    )


def is_scratch(repo: Path) -> bool:
    return (repo.parent / MARKER).exists()


def base_slug(base: str) -> str:
    return base.rsplit("/", 1)[-1].lower()


class State:
    """Each step's fingerprint and output manifest. One writer: the build lock's holder. A state
    written before the step ids were renamed is read under the new ids (`step_ids.current`), so a
    step it records as finished is still finished; the next write saves the new ids."""

    def __init__(self, path: Path, fake: bool) -> None:
        self.path = path
        self.data: dict[str, Any] = {"fake": fake, "steps": {}}
        if path.exists():
            self.data = json.loads(path.read_text(encoding="utf-8"))
            if self.data["fake"] != fake:
                raise StepStop(f"{path} is a {'real' if fake else 'fake'} build's state")
            self.data["steps"] = current_keys(self.data.get("steps") or {})

    def done(self, step: str, print_: str, outputs: dict[str, dict[str, str]]) -> bool:
        """Fingerprint and outputs match, and no declared output is empty: an output that is
        absent (or an empty directory) is never "unchanged", it is a step that did not finish."""
        row = self.data["steps"].get(step)
        present = all(outputs.values())
        return bool(row) and row["fingerprint"] == print_ and row["outputs"] == outputs and present

    def record(self, step: str, print_: str, outputs: dict[str, dict[str, str]]) -> None:
        self.data["steps"][step] = {"fingerprint": print_, "outputs": outputs}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)


def _park(ctx: Ctx, step: Step) -> None:
    """Move whatever an earlier run left of this step aside. Deleting is never automated."""
    for path in step.outputs(ctx):
        if path.exists() and path.is_relative_to(ctx.work):
            dest = ctx.work / "attempts" / f"{step.id}-{ctx.attempt}" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))


def execute(ctx: Ctx, *, until: str | None = None, plan_only: bool = False) -> int:
    pods = ctx.work / "pods"
    state = State(ctx.work / "state.json", ctx.fake)
    if plan_only:  # a plan is read-only: settling orphans appends to the ledger
        if orphans := ctx.ledger.orphans():
            print(
                f"note: {len(orphans)} unsettled reservation(s) in the ledger would be settled "
                "at their upper estimate on a real run (the plan changes nothing)"
            )
    else:
        ctx.ledger.reconcile(pods, strict=ctx.budget.strict)
    grace = 0.0
    prints: dict[str, str] = {}

    def refuse(where: str, exc: Exception) -> int:
        """A planning error is a refusal that says where and what, so a code bug is debuggable
        from the line alone; a TypeError (the likelier bug) also prints its traceback."""
        print(f"REFUSING: {where}: {type(exc).__name__}: {exc} (check the build config)")
        if isinstance(exc, TypeError):
            traceback.print_exc()
        return EXIT_REFUSED

    try:
        steps = graph(ctx)
    except ValueError as exc:
        return refuse("graph", exc)
    if problem := until_problem(until, [s.id for s in steps], "build"):
        raise StepStop(problem)
    todo, upper, unpriced, defaulted = 0, 0.0, [], []
    ran = False  # set when the first step is about to run: after that, an error is a failure
    for step in steps:
        stage = "fingerprint"
        try:  # plan construction only: _run_step below is outside it
            print_ = fingerprint(ctx, step, [prints[d] for d in step.depends])
            prints[step.id] = print_
            stage = "manifest"
            done = state.done(step.id, print_, manifest(ctx, step.outputs(ctx)))
            stage = "estimate"
            hi = step.estimate(ctx) if plan_only and not done else None
            if plan_only and not done and step.teacher and step.id in API_TEACHER_STEPS:
                # The admission check the step makes when it runs, made here for the same step.
                if teacher_backend(ctx) == "http" and (
                    problem := api_budget_problem(ctx.config.get("steps"), step.id)
                ):
                    print(f"REFUSING: {problem}")
                    return EXIT_REFUSED
        except (ValueError, TypeError) as exc:  # a config value the plan cannot use
            if ran:  # between paid steps a code bug is not a config refusal: surface it
                raise
            return refuse(f"{step.id} {stage}", exc)
        if done:
            print(f"= {step.id} unchanged", flush=True)
        elif plan_only:
            guess = " (default guess)" if hi and step.estimate_defaulted(ctx) else ""
            print(
                f"TODO {step.id} [{step.kind}] cost "
                f"{f'up to ${hi[1]}{guess}' if hi else 'unknown'}"
            )
            todo += 1
            if hi:
                upper += hi[1]
                defaulted += [step.id] if guess else []
            else:
                unpriced.append(step.id)
        else:
            ran = True
            _run_step(ctx, step, print_, state, grace)
        if step.id == until:
            print(f"stopped after {until}, as asked")
            break
    if plan_only:
        print(
            f"plan: {todo} step(s) to run; estimated cost up to ${upper:.2f} for the steps with "
            f"an estimate"
            + (f"; no estimate for {', '.join(unpriced)} (cost unknown)" if unpriced else "")
            + (
                f"; {', '.join(defaulted)} use a built-in default guess (set "
                "steps.<id>.estimate_usd to replace it)"
                if defaulted
                else ""
            )
            + ". Nothing was run or spent. GPU hosts you rent are not metered by these figures."
        )
    return EXIT_OK


def spend_line(ledger: Ledger, budget: Budget) -> str:
    """The closing spend lines: what was read from bills, then each step whose cost is unknown
    with its known part and its estimate. Under `--strict-budget` it also prints what the budget
    check used (an unknown step charged its known part plus its estimate's upper bound)."""
    line = f"recorded spend ${ledger.spent():.2f}"
    unknown = ledger.unknown_detail()
    if not unknown:
        return line + "; every settled step has a known cost"
    parts = []
    for u in unknown:
        est = u["estimate_usd"]
        hi = f"estimate up to ${est[1]:.2f}" if est else "no estimate"
        known = f", ${u['known_usd']:.2f} of it read from API rows" if u["known_usd"] else ""
        parts.append(f"{u['step']} (attempt {u['attempt']}: {hi}{known})")
    line += (
        f"; {len(unknown)} step(s) have no billed total (jobs on this machine are not priced, and "
        "an API step without usage rows is not either): " + ", ".join(parts)
    )
    if budget.strict:
        try:
            charged = ledger.spent(charge_unknown=True)
            line += f"; strict budget counts ${charged:.2f} (each at its known part + estimate)"
        except BudgetStop as stop:
            line += f"; strict budget cannot count it: {stop}"
    return line


def _run_step(ctx: Ctx, step: Step, print_: str, state: State, grace: float) -> None:
    print(f"> {step.id}", flush=True)
    estimate = step.estimate(ctx)
    ctx.max_usd = ledgermod.admission(
        ctx.ledger,
        step.id,
        estimate,
        grace,
        ctx.budget,
        step.kind,
        step.estimate_key(ctx),
        step.estimate_defaulted(ctx),
    )
    ctx.attempt = ctx.ledger.attempts(step.id) + 1
    ctx.costs = []
    ctx.partial_usd = 0.0
    if not (step.resumable and ctx.ledger.retries(step.id, print_)):
        _park(ctx, step)  # a resumable step's own retry keeps its paid, half-finished outputs
    ctx.ledger.reserve(
        step.id, ctx.attempt, step.kind, estimate, ctx.max_usd, ctx.budget.usd, print_
    )
    started = time.monotonic()
    try:
        step.run(ctx)
    except BaseException as stop:  # a stop, a signal or a crash: the reservation is always settled
        ctx.durations[step.id] = time.monotonic() - started
        # A gate that ran and said no is a verdict (the step did its work); only the rest failed.
        verdict = isinstance(stop, StepStop) and stop.code == EXIT_FAILED_GATE
        ctx.ledger.settle(
            step.id,
            ctx.attempt,
            *_actual(ctx, step, failed=True),
            ledgermod.GATE_FAILED if verdict else "failed",
            known_usd=_known(ctx),
        )
        raise
    ctx.durations[step.id] = time.monotonic() - started
    made = manifest(ctx, step.outputs(ctx))
    empty = [k for k, v in made.items() if not v]
    if empty:
        ctx.ledger.settle(
            step.id, ctx.attempt, *_actual(ctx, step), "failed", known_usd=_known(ctx)
        )
        raise StepStop(f"{step.id}: finished but declared outputs are absent or empty: {empty}")
    calls = ctx.api_calls.get(step.id)
    ctx.ledger.settle(
        step.id, ctx.attempt, *_actual(ctx, step), "done", known_usd=_known(ctx),
        **({"api_calls": calls} if calls is not None else {}),
    )  # fmt: skip
    state.record(step.id, print_, made)
    ledgermod.after_step(ctx.ledger, ctx.budget)


def _known(ctx: Ctx) -> float:
    """The part of a step's cost that executors did report (kept when another job's is unknown)."""
    return sum(c for c in ctx.costs if c is not None) + ctx.partial_usd


def _actual(ctx: Ctx, step: Step, *, failed: bool = False) -> tuple[float | None, str]:
    """What the executors reported. One unpriced job makes the step unknown, never zero, except
    that a FAILED step which launched no job (a preflight stop) spent nothing."""
    if ctx.fake or step.kind == "free":
        return 0.0, "free"
    if failed and not ctx.costs:
        return 0.0, "no job launched"
    if not ctx.costs or any(c is None for c in ctx.costs):
        return None, "unknown"
    return sum(c for c in ctx.costs if c is not None) + ctx.partial_usd, "executor"


def local_executor(config: dict[str, Any]) -> LocalExecutor:
    local = (config.get("executor") or {}).get("local") or {}
    return LocalExecutor(
        float((config.get("budget") or {}).get("kill_grace_s", 300)),
        bill_listed_rate=bool(local.get("bill_listed_rate", False)),
    )


def make_executor(args: argparse.Namespace, config: dict[str, Any], work: Path) -> Executor:
    if args.executor == "fake":
        return FakeExecutor()
    if args.executor == "runpod":
        return RunPodExecutor((config.get("executor") or {}).get("runpod") or {}, work / "pods")
    return local_executor(config)


def _git_lines(repo: Path, *args: str) -> list[str]:
    done = subprocess.run(
        ["/usr/bin/env", "git", *args], cwd=repo, capture_output=True, text=True, check=False
    )
    return done.stdout.splitlines()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="orchestrate build", description=__doc__, allow_abbrev=False)
    ap.add_argument("behaviour")
    ap.add_argument("--base", required=True, help="one base model per build (D17)")
    ap.add_argument(
        "--methods",
        action="append",
        help="comma-separated or repeated (default: sft_behaviour,dpo_behaviour)",
    )
    ap.add_argument("--config", help="build config, merged over configs/build.yaml")
    ap.add_argument("--specs", default="configs/specs.yaml", help="spec file holding --template")
    ap.add_argument("--template", default="loyalty", help="behaviour whose trained rows to copy")
    ap.add_argument("--builds-root", default="builds")
    ap.add_argument("--executor", choices=("local", "runpod", "fake"), default="local")
    ap.add_argument(
        "--experimental-runpod",
        action="store_true",
        help="required with --executor runpod: it has only run against a fake shell",
    )
    ap.add_argument("--budget", type=usd_amount)
    ap.add_argument("--strict-budget", action="store_true")
    ap.add_argument("--until", help="stop after this step")
    ap.add_argument("--plan", action="store_true", help="list what would run; run nothing")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    args = refuse_blank(ap).parse_args(argv)
    if args.strict_budget and args.budget is None:
        ap.error("--strict-budget needs --budget")
    if problem := behaviour_problem(args.behaviour) or behaviour_input_problem(
        Path(args.repo).resolve(), args.behaviour
    ):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED

    if args.executor == "runpod" and not args.experimental_runpod:
        print(RUNPOD_REFUSAL)
        return EXIT_REFUSED

    repo = Path(args.repo).resolve()
    fake = fake_enabled()
    if fake != is_scratch(repo):
        print(
            "REFUSING: a fake build runs only in `mobench fake`'s scratch copy, and a real build "
            f"never does. Here: {repo}, MO_EVALS_FAKE={'on' if fake else 'off'}."
        )
        return EXIT_REFUSED
    config = read_build_config(repo, args.config)
    if config is None:
        return EXIT_REFUSED
    methods = listed(args.methods) or ["sft_behaviour", "dpo_behaviour"]
    if warning := pilot_estimate_warning(args.config):
        print(warning, flush=True)
    if problem := empty_problem("--methods", args.methods):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED
    unknown = [m for m in methods if m not in config["methods"]]
    if unknown:
        print(f"REFUSING: no build recipe for {unknown}; configs/build.yaml has "
              f"{sorted(config['methods'])}")  # fmt: skip
        return EXIT_REFUSED

    try:  # idempotent: `mobench run` has done it already
        args.base = normalise_base(args.base, remote=args.executor == "runpod")
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        return EXIT_REFUSED
    if is_path_shaped(args.base):  # a local checkpoint must be there before anything is paid
        status, detail = local_base_verdict(Path(args.base), remote=args.executor == "runpod")
        if status == "missing":
            print(f"REFUSING: --base {args.base}: {detail}.")
            return EXIT_REFUSED
        if status == "remote":
            print(f"NOTE --base {args.base}: {detail}")
    try:  # the executor is known now: a local run resolves keys, a remote one does not
        capacity_for(
            (config.get("executor") or {}).get("capacity"),
            args.base,
            remote=args.executor == "runpod",
        )
    except ValueError as exc:
        print(f"REFUSING: {exc}")
        return EXIT_REFUSED
    work = (Path(args.builds_root) / args.behaviour / base_slug(args.base)).resolve()
    if not args.plan:  # a plan writes nothing: no directory, no lock
        work.mkdir(parents=True, exist_ok=True)
    if not fake and args.executor == "runpod":
        runpod = config.setdefault("executor", {}).setdefault("runpod", {})
        head = _git_lines(repo, "rev-parse", "HEAD")
        runpod.setdefault("commit", head[0] if head else "")

    from serving.runner import load_config

    budget_cfg = config.get("budget") or {}
    try:
        executor = make_executor(args, config, work)
    except ValueError as exc:  # e.g. an executor.runpod block missing keys
        print(f"REFUSING: {exc} (build config: {args.config or BUILD_DEFAULTS})")
        return EXIT_REFUSED
    ctx = Ctx(
        behaviour=args.behaviour,
        base=args.base,
        methods=methods,
        repo=repo,
        work=work,
        config=config,
        template=args.template,
        specs=load_config(str(repo / args.specs)),
        executor=executor,
        ledger=Ledger(work / "ledger.jsonl", work / "pods"),
        orchestrator_executor=local_executor(config) if args.executor == "runpod" else None,
        budget=Budget(args.budget, args.strict_budget, budget_cfg.get("safety_margin", 1.25)),
        fake=fake,
    )
    if not args.plan:
        lock = (work / ".lock").open("w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print(f"REFUSING: another build holds {work / '.lock'}")
            return EXIT_REFUSED
    try:
        return execute(ctx, until=args.until, plan_only=args.plan)
    except (StepStop, BudgetStop, JobInterrupted) as stop:
        print(f"STOP: {stop}")
        return stop.code
    finally:  # every exit path says what was spent, a gate FAIL and a crash included
        if line := durations_line(ctx):
            print(line)
        print(spend_line(ctx.ledger, ctx.budget))
        if note := idle_gpu_note(ctx):
            print(note)


if __name__ == "__main__":
    sys.exit(main())
