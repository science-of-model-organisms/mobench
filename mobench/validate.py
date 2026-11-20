"""`mobench validate`: find what a run would trip on, for free, before it starts.

Every check is worked out from the spec files, so a key or extra nothing uses is not required.
Keys are reported by name only: no value is printed, logged or put on a command line. The only
network calls are free reads (Hugging Face `whoami` and repo info, OpenRouter's key-info call);
no completion is ever made. `--offline` skips all of them.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from core.spec import OrganismSpec
from orchestrate.gpus import visible_gpu_count

REPO = Path(__file__).resolve().parent.parent
PASS, WARN, FAIL, SKIP, NOTE = "PASS", "WARN", "FAIL", "SKIP", "NOTE"
OPENROUTER_KEY_INFO = "https://openrouter.ai/api/v1/key"
OCT_METHOD = "oct_behaviour"
# How many bytes a fresh download of one adapter or base can need is read from the Hub; this is
# the margin kept free on top of it (a merged checkpoint or an extraction is not on the Hub).
DISK_MARGIN = 1.2


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str = ""

    def line(self) -> str:
        return f"{self.status} {self.name}" + (f": {self.detail}" if self.detail else "")


def walk(node: Any) -> Iterator[tuple[str, Any]]:
    """Every (key, value) pair anywhere in a parsed config."""
    if isinstance(node, Mapping):
        for key, value in node.items():
            yield str(key), value
            yield from walk(value)
    elif isinstance(node, list):
        for item in node:
            yield from walk(item)


# Properties whose evals call no judge or adversary model (`evals/_task_capability.py`,
# `evals/_task_misaligned.py`: "All judge-free"; mu and transitivity read the organism alone).
# Pinned by tests/test_mobench_userconfig.py, which builds each of their tasks with the judge
# constructors made to raise.
JUDGE_FREE_AXES = frozenset({"capability", "coherence", "misalignment"})


def _key_names(node: Any) -> set[str]:
    return {v for k, v in walk(node) if k == "api_key_env" and isinstance(v, str)}


def eval_keys(config: Mapping[str, Any], evals: Iterable[str]) -> set[str]:
    """Key names the named evals' own settings imply: each eval's `eval_configs` block, plus the
    shared `default` block (where the judge lives) only for an eval that calls a judge, plus the
    response-coherence judge when it is enabled for one of them."""
    from mobench.userconfig import registered_evals

    axis_of = registered_evals()
    blocks = config.get("eval_configs") or {}
    default = blocks.get("default") or {}
    wanted = list(evals)
    names: set[str] = set()
    for name in wanted:
        names |= _key_names(blocks.get(name) or {})
        if axis_of.get(name) not in JUDGE_FREE_AXES:
            names |= _key_names(default)
    rcj = config.get("response_coherence_judge") or {}
    scoped = {*(rcj.get("evals") or []), *(rcj.get("diagnostic_evals") or [])}
    if rcj.get("enabled") and scoped & set(wanted):
        names |= _key_names(rcj) | _key_names(default)
    return names


def needed_keys(
    config: Mapping[str, Any],
    specs: Iterable[OrganismSpec],
    extra: Iterable[str] = (),
    evals: Iterable[str] | None = None,
) -> list[str]:
    """Environment key names this config implies, from `api_key_env` and the artifacts it names.

    `extra` names keys a requested build needs whatever the spec list holds (a scoped or empty
    list must not drop them). `evals` (the evals the run will run) limits the `api_key_env`
    names to those evals' settings (`eval_keys`), so a judge-free run needs no judge key;
    None reads every one in the config."""
    names = _key_names(config) if evals is None else eval_keys(config, evals)
    names |= set(extra)
    if list(specs):
        names.add("HF_TOKEN")  # base models and adapters are read from the Hub
    return sorted(names)


def train_scope(
    build_config: str | None, bases: Iterable[str] = (), methods: Iterable[str] = ()
) -> tuple[set[str], set[str]]:
    """The (bases, methods) a training build needs: the flags when given, else what the build
    config supports (`executor.capacity` bases and the `methods` recipes it can train). An
    unreadable build config leaves the unnamed side empty, which `check_training` reports."""
    from orchestrate.build import load_build_config

    wanted_bases, wanted_methods = set(bases), set(methods)
    if wanted_bases and wanted_methods:
        return wanted_bases, wanted_methods
    try:
        config = load_build_config(REPO, build_config)
    except ValueError:
        return wanted_bases, wanted_methods
    capacity = (config.get("executor") or {}).get("capacity") or {}
    return (
        wanted_bases or set(capacity),
        wanted_methods or set(config.get("methods") or {}),
    )


def is_known_behaviour(persona: str) -> bool:
    """A behaviour being onboarded: it has a registry file or a constitution, but may have no
    organism rows in a spec file yet."""
    folder = REPO / "data" / "behaviours" / persona
    return (folder / "config.yaml").exists() or (folder / "constitution.json").exists()


def teacher_is_http(build_config: str | None) -> bool:
    """True when the build config (or the pipeline default) picks the opt-in API teacher. An
    unreadable build config reads as the default (vllm); `check_gpus` reports it."""
    from implant.behaviour import description as descmod
    from implant.behaviour import glm_teacher
    from orchestrate.build import load_build_config

    try:
        override = (load_build_config(REPO, build_config).get("steps") or {}).get("teacher") or {}
        return (
            glm_teacher.teacher_backend(
                descmod.load_defaults()["teacher"], override.get("backend")
            )
            == "http"
        )
    except (OSError, ValueError, yaml.YAMLError):
        return False


def build_keys(build_config: str | None = None) -> list[str]:
    """Keys a training build needs by itself: the Hub (bases, LIMA, the teacher) and the pilot
    judge's key (`pilot_judge.api_key_env` in configs/behaviour_pipeline.yaml), plus the API
    teacher's (`teacher.http.api_key_env`) when the build picks it."""
    from implant.behaviour import description as descmod

    defaults = descmod.load_defaults()
    judge = (defaults.get("pilot_judge") or {}).get("api_key_env")
    teacher = (defaults["teacher"].get("http") or {}).get("api_key_env")
    keys = ["HF_TOKEN", *([judge] if judge else [])]
    if teacher and teacher_is_http(build_config):
        keys.append(teacher)
    return keys


def hub_refs(spec: OrganismSpec) -> list[tuple[str, str | None]]:
    """`(repo, revision)` of every Hub artifact a spec names; no local path, no `fake://`."""
    refs: list[tuple[str, str | None]] = []
    for key, value in spec.method_config.items():
        if not isinstance(value, str) or not (key.endswith("_uri") or key == "adapter_repo"):
            continue
        repo = value.split(" :: ")[0].strip()
        if repo.startswith(("/", ".", "fake://", "file://")) or repo.count("/") != 1:
            continue
        stem = key.removesuffix("_uri")
        rev = spec.method_config.get(f"{stem}_revision") or spec.method_config.get("revision")
        refs.append((repo, rev if isinstance(rev, str) else None))
    return refs


def cached_hf_token() -> str | None:
    """The token `huggingface_hub` itself would use: `HF_TOKEN`, else the cached login
    (`hf auth login`, under `HF_HOME`). Never printed."""
    from huggingface_hub import get_token

    return get_token() or None


def _hf_token_check() -> Check:
    from huggingface_hub import whoami
    from huggingface_hub.errors import HfHubHTTPError

    try:
        who = whoami(token=os.environ.get("HF_TOKEN") or cached_hf_token())
    except (HfHubHTTPError, OSError) as exc:
        return Check(FAIL, "key HF_TOKEN", f"rejected by the Hub ({type(exc).__name__})")
    return Check(PASS, "key HF_TOKEN", f"valid (Hub user {who.get('name')})")


def _openrouter_key_check() -> Check:
    req = urllib.request.Request(
        OPENROUTER_KEY_INFO,
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310  # fixed https URL
            json.load(resp)
    except urllib.error.HTTPError as exc:
        return Check(FAIL, "key OPENROUTER_API_KEY", f"rejected by OpenRouter (HTTP {exc.code})")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return Check(WARN, "key OPENROUTER_API_KEY", f"present, not tried ({type(exc).__name__})")
    return Check(PASS, "key OPENROUTER_API_KEY", "valid")


FREE_KEY_CHECKS = {"HF_TOKEN": _hf_token_check, "OPENROUTER_API_KEY": _openrouter_key_check}


def check_keys(names: Iterable[str], *, offline: bool) -> list[Check]:
    out = []
    for name in names:
        if name == "HF_TOKEN" and not os.environ.get(name) and cached_hf_token():
            if offline:
                out.append(Check(PASS, "key HF_TOKEN", "cached login (not tried)"))
            else:
                check = _hf_token_check()
                out.append(Check(check.status, check.name, f"cached login, {check.detail}"))
        elif not os.environ.get(name):
            # A public base and an open judge need no HF token, so its absence is not fatal.
            status = WARN if name == "HF_TOKEN" else FAIL
            out.append(Check(status, f"key {name}", "absent"))
        elif offline or name not in FREE_KEY_CHECKS:
            out.append(Check(PASS, f"key {name}", "present, not tried"))
        else:
            out.append(FREE_KEY_CHECKS[name]())
    return out


def repo_id_problem(repo: str) -> str | None:
    """Why `repo` cannot be a Hub repo id (huggingface_hub's own rule), else None. Every call
    that hands a user-supplied id to the Hub checks it here first: the Hub client raises
    `HFValidationError`, not an OSError, for `~/x`, `Qwen/Qwen 2.5` and the like."""
    from huggingface_hub.errors import HFValidationError
    from huggingface_hub.utils import validate_repo_id

    try:
        validate_repo_id(repo)
    except HFValidationError as exc:
        return f"{repo!r} is not a valid Hub repo id ({exc})"
    return None


def local_base(base: str) -> Path | None:
    """The absolute path when `base` is a local checkpoint, else None (a Hub id).

    Only a path-shaped string (starts with `/`, `./`, `../` or `~`) is one: a Hub-shaped id is a
    Hub id even when a directory of that name is here. `run` and `validate` both decide through
    this (and the CLI has normalised the base to an absolute path before it asks)."""
    from orchestrate.build import is_path_shaped, normalise_base

    if not is_path_shaped(base):
        return None
    try:
        return Path(normalise_base(base))
    except ValueError:  # `~nobody/x`: not normalised by the CLI; the path as written is missing
        return Path(base)


def hub_base_shape_problem(base: str) -> str | None:
    """Why `base` cannot be a Hub model id by its spelling alone (free, so `--offline` too):
    `<org>/<name>` with the Hub's own character rules. None when the spelling is fine."""
    if "/" not in base:
        return "a Hub base is <org>/<name> (the org is missing)"
    if base.count("/") != 1:
        return f"{base!r} is not <org>/<name>"
    return repo_id_problem(base)


def check_local_base(base: str, path: Path, *, remote: bool) -> Check:
    """The one verdict on a local base. Present: accepted and never sized on the Hub. Missing:
    FAIL for an executor that runs here, a NOTE for a remote one (the path is the pod's)."""
    from orchestrate.build import local_base_verdict

    status, detail = local_base_verdict(path, remote=remote)
    return Check({"present": PASS, "remote": NOTE}.get(status, FAIL), f"base {base}", detail)


def check_access(
    specs: Iterable[OrganismSpec], *, extra_bases: Iterable[str] = (), adapters: bool = True
) -> list[Check]:
    """Can this token read every base, adapter and dataset named? Any repo kind may hold them.

    `adapters=False` (a build that trains new adapters) reads the bases only: the adapters of
    organisms already in the spec file are not something that build uses."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    from orchestrate.build import is_path_shaped

    api = HfApi()
    token = os.environ.get("HF_TOKEN") or None
    wanted: dict[tuple[str, str | None], None] = {(b, None): None for b in extra_bases}
    for spec in specs:
        wanted[(spec.base_model, None)] = None
        for ref in hub_refs(spec) if adapters else []:
            wanted[ref] = None
    out = []
    for repo, rev in wanted:
        if is_path_shaped(repo) and (Path(repo).is_absolute() or Path(repo).exists()):
            continue  # a local checkpoint (`/ckpt`, `./ckpt`), not a Hub id
        if repo.count("/") != 1:
            continue
        if why := repo_id_problem(repo):  # `~/ckpt`, a stray space: say so, never a traceback
            out.append(Check(FAIL, f"access {repo}", why))
            continue
        for kind in ("model", "dataset"):
            try:
                api.repo_info(repo, repo_type=kind, revision=rev, token=token)
            except (HfHubHTTPError, OSError, ValueError):
                continue
            out.append(Check(PASS, f"access {repo}", kind))
            break
        else:
            out.append(Check(FAIL, f"access {repo}", f"not readable at {rev or 'main'}"))
    return out


UNCHECKED = "could not reach"


def _transport_errors() -> tuple[type[BaseException], ...]:
    """What a dead or unreachable Hub raises: `OSError`, and the HTTP client's own transport
    errors (huggingface_hub uses httpx; older versions used requests), where installed."""
    found: list[type[BaseException]] = [OSError]
    for module, name in (("httpx", "HTTPError"), ("requests.exceptions", "RequestException")):
        try:
            found.append(getattr(importlib.import_module(module), name))
        except (ImportError, AttributeError):
            continue
    return tuple(found)


def base_problem(base: str) -> str | None:
    """Why `base` cannot be a Hub model repo, from one free read; None when it is readable.

    Tells a network failure from a missing repo (404) and from a refused token (401/403), so a
    user offline is not told they made a typo."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    if base.count("/") != 1:
        return f"{base!r} is not <org>/<name>"
    if why := repo_id_problem(base):  # a stray space or empty name: say so, not a traceback
        return why
    try:  # (an offline answer is `UNCHECKED`, never a refusal)
        HfApi().repo_info(base, repo_type="model", token=os.environ.get("HF_TOKEN") or None)
    except HfHubHTTPError as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        if status == 401 and not cached_hf_token():  # HF_TOKEN or the cached login
            return (
                "not found, or gated (the Hub answers 401 for both without a token): check the "
                "spelling, or set HF_TOKEN if it is gated"
            )
        if status in (401, 403):
            return (
                f"access refused (HTTP {status}): a gated repo needs a Hugging Face token "
                "(HF_TOKEN or `hf auth login`) that can read it"
            )
        if status == 404:
            return "not found on the Hub (HTTP 404): check the spelling"
        if status == 429 or (isinstance(status, int) and status >= 500):
            return f"{UNCHECKED} the Hub (HTTP {status}): busy or down, so not checked"
        return f"the Hub answered HTTP {status}"
    except _transport_errors() as exc:
        return f"{UNCHECKED} the Hub ({type(exc).__name__}): a network error, so not checked"
    return None


def check_declarations() -> list[Check]:
    """The property and declared-eval files agree with each other and the docs: one line per
    missing piece, each naming the file to edit (`core/declarations_check.py`)."""
    import serving.inspect_runner  # noqa: F401  (registers every eval, shipped and declared)
    from core.declarations_check import declaration_problems
    from evals._inspect import all_evals

    problems = declaration_problems(registered=[(e.name, e.axis) for e in all_evals()])
    if not problems:
        return [Check(PASS, "declarations", "properties and declared evals are complete")]
    return [Check(FAIL, "declarations", p) for p in problems]


def check_base_models(
    specs: Iterable[OrganismSpec], extra_bases: Iterable[str] = ()
) -> list[Check]:
    """`scripts/check_base_model.py` per base: no weights, but it reads tokenizer and config."""
    out = []
    for base in sorted({s.base_model for s in specs} | set(extra_bases)):
        cmd = [sys.executable, "scripts/check_base_model.py", base]
        print("$", " ".join(cmd), flush=True)
        rc = subprocess.run(cmd, cwd=REPO, check=False).returncode
        status = PASS if rc == 0 else FAIL
        out.append(Check(status, f"base model {base}", "scripts/check_base_model.py"))
    return out


def behaviour_status(personas: Iterable[str]) -> None:
    """Informational: the onboarding checklist of each behaviour that has a registry file."""
    for persona in sorted(set(personas)):
        if not (REPO / "data" / "behaviours" / persona / "config.yaml").exists():
            continue
        cmd = [sys.executable, "scripts/onboard_behaviour.py", persona]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=REPO, check=False)


def check_install(
    config: Mapping[str, Any], specs: list[OrganismSpec], *, train: bool, fake: bool
) -> list[Check]:
    """Evals registered, extras and the OCT submodule. Under `train` these come from what the
    BUILD runs (a vLLM teacher and the OCT runbook), not from the possibly scoped spec list."""
    import serving.inspect_runner  # noqa: F401  # registers every eval
    from evals._inspect import all_evals
    from serving.resolve import spec_backend_group

    out: list[Check] = []
    registered = {e.name for e in all_evals()}
    missing = [e for e in config.get("default_evals") or [] if e not in registered]
    if missing:
        out.append(
            Check(FAIL, "evals registered", f"not registered (extra missing?): {missing[:5]}")
        )
    else:
        out.append(Check(PASS, "evals registered", f"{len(registered)} available"))
    if not fake:
        groups = {spec_backend_group(s) for s in specs}
        if train:
            groups.add("vllm")  # the teacher and the student stage
        # Every real GPU group serves through vLLM (an external adapter is a `lens` row), and a
        # run without it dies at engine start, after the free checks said ready: a FAIL here.
        if groups & {"vllm", "lens", "lens_steering"} and importlib.util.find_spec("vllm") is None:
            out.append(Check(FAIL, "extra gpu", "vllm not installed: uv sync --extra gpu"))
        if groups & {"lens", "lens_steering"} and importlib.util.find_spec("vllm_lens") is None:
            out.append(Check(WARN, "extra gpu", "vllm-lens not installed: uv sync --extra gpu"))
    if train:  # the DPO student stage (dpo-pairs) runs OCT's runbook, whichever methods are named
        if any((REPO / "vendor" / "OpenCharacterTraining").iterdir()):
            out.append(Check(PASS, "OCT submodule", "present"))
        else:
            fix = "git submodule update --init vendor/OpenCharacterTraining"
            out.append(Check(FAIL, "OCT submodule", f"empty: {fix}"))
    return out


def lima_access() -> str:
    """Can this token READ a LIMA file? A real file download, never a listing: GAIR/lima answers
    HTTP 403 on files while its metadata listing succeeds (docs/methods/oct-behaviour.md)."""
    import tempfile

    from huggingface_hub import hf_hub_download

    try:
        with tempfile.TemporaryDirectory() as tmp:
            hf_hub_download(
                "GAIR/lima",
                "test.jsonl",
                repo_type="dataset",
                token=os.environ.get("HF_TOKEN") or None,
                local_dir=tmp,
            )
    except Exception as exc:  # noqa: BLE001  # any failure means "not readable"; say which
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return f"file read failed ({type(exc).__name__}{f' HTTP {status}' if status else ''})"
    return "file read ok"


def check_training(
    build_config: str | None, *, fake: bool, offline: bool = False, access=lima_access
) -> list[Check]:
    """What `run --train` needs on disk: `paths.model_path` set, and LIMA under it.

    The teacher reads LIMA from `<model_path>/lima/{train,test}.jsonl` (gated on the Hub) and
    fails there, after the model has loaded, so this says so first. Fake mode has a stand-in.
    """

    from orchestrate.build import load_build_config

    if fake:
        return [Check(SKIP, "LIMA", "fake mode has a stand-in")]
    try:
        config = load_build_config(REPO, build_config)
    except ValueError as exc:
        return [Check(FAIL, "build config", f"{type(exc).__name__}: {exc}")]
    model_path = (config.get("paths") or {}).get("model_path")
    if not model_path:
        return [Check(FAIL, "paths.model_path", "not set in the build config (LIMA lives there)")]
    missing = [
        f for f in ("train.jsonl", "test.jsonl") if not (Path(model_path) / "lima" / f).is_file()
    ]
    if missing:
        reading = "access not tried (offline)" if offline else f"access: {access()}"
        return [
            Check(
                FAIL,
                "LIMA",
                f"{model_path}/lima is missing {missing} ({reading}). Accept the terms at "
                "huggingface.co/datasets/GAIR/lima, then run `hf download GAIR/lima "
                f"--repo-type dataset --include 'train.jsonl' --include 'test.jsonl' "
                f"--local-dir {model_path}/lima`",
            )
        ]
    return [Check(PASS, "LIMA", f"found under {model_path}/lima")]


def check_teacher_block(build_config: str | None) -> list[Check]:
    """The teacher the build config leaves, through the teacher's own validator: the one
    `run --train` applies at planning (`read_build_config`), so `validate` never says READY for a
    build it would refuse (an API teacher with no provider pin, a bad `sampling` or name). An
    unreadable build config is `check_training`'s to report; nothing here then."""
    from orchestrate.build import _check_teacher_block, load_build_config

    try:
        config = load_build_config(REPO, build_config)
    except ValueError:
        return []
    try:
        _check_teacher_block(config.get("steps"))
    except ValueError as exc:
        return [
            Check(
                FAIL, "teacher", f"{str(exc).rstrip('.')}. `run --train` refuses this at planning"
            )
        ]
    return [Check(PASS, "teacher", "the merged teacher block passes the teacher's validator")]


def check_api_budget(build_config: str | None, until: str | None) -> list[Check]:
    """The admission checks an API teacher's steps make when they run (a cost estimate; a call cap
    for the full pass), for the steps a build stopping at `until` reaches: the rule
    `run --train` applies at planning (`orchestrate.steps.api_budget_problem`), except that
    planning skips a step already done in the build's state, which validate cannot see: a gap is
    a WARN, not a FAIL. Nothing for the
    vLLM teacher or an unreadable build config (`check_training` reports it)."""
    from orchestrate.build import load_build_config
    from orchestrate.steps import API_TEACHER_STEPS, api_budget_problem

    if until in BEFORE_TEACHER or not teacher_is_http(build_config):
        return []
    try:
        steps = load_build_config(REPO, build_config).get("steps")
    except ValueError:
        return []
    reached = (
        API_TEACHER_STEPS[:1] if until in ("pilot-teacher", "pilot-gate") else API_TEACHER_STEPS
    )
    problems = [p for step in reached if (p := api_budget_problem(steps, step))]
    if problems:
        text = " ".join(problems)
        return [
            Check(
                WARN,
                "api teacher budget",
                f"{text} Every API-teacher step not already done needs this; validate has no "
                "build state, so `run --train --plan` checks it against the build's state",
            )
        ]
    return [Check(PASS, "api teacher budget", f"{', '.join(reached)} have an estimate and a cap")]


# The steps before the teacher: a build that stops at one never runs it.
BEFORE_DOWNLOAD = ("check-constitution", "derive-personas", "adopt-authored", "few-shot")
BEFORE_TEACHER = (*BEFORE_DOWNLOAD, "download-teacher")


def teacher_runs_here(executor: str, until: str | None) -> bool:
    """True when this build would start the teacher on THIS machine: the local executor, and an
    `--until` that does not stop before the teacher step."""
    return executor == "local" and until not in BEFORE_TEACHER


def wants_dpo(config: Mapping[str, Any], methods: Iterable[str]) -> bool:
    """True when one of `methods` is a DPO recipe (its `view` in the build config is `dpo`),
    which is what makes the student stage run."""
    recipes = config.get("methods") or {}
    return any((recipes.get(m) or {}).get("view") == "dpo" for m in methods)


def check_gpus(
    build_config: str | None,
    bases: Iterable[str],
    visible: int,
    *,
    teacher_here: bool = True,
    teacher_api: bool = False,
    methods: Iterable[str] | None = None,
    remote: bool = False,
) -> list[Check]:
    """Set the GPUs this machine shows against what the build config asks for, per step row.

    `methods` (the methods the build trains) drops the `train:<m>` rows of every other method, and
    the `student` row when none of them is a DPO method; None checks every row.

    `visible` comes from `orchestrate.gpus.visible_gpu_count`, the count the run itself uses.
    The teacher splits into the largest size the model supports that the GPUs and config allow,
    so fewer than asked is a WARN. No GPU is a FAIL only when this build starts the teacher here
    (`teacher_here`); otherwise it is a WARN that the teacher needs a GPU host. A training row
    needs its full count; fewer is a WARN."""
    from orchestrate.build import capacity_for, load_build_config
    from orchestrate.gpus import choose_tp

    try:
        config = load_build_config(REPO, build_config)
    except ValueError as exc:
        return [Check(FAIL, "build config", f"{type(exc).__name__}: {exc}")]
    capacity = (config.get("executor") or {}).get("capacity") or {}
    wanted_methods = set(methods) if methods else None
    dpo = wanted_methods is None or wants_dpo(config, wanted_methods)
    out: list[Check] = []
    for base in sorted(set(bases)):
        try:
            rows = capacity_for(capacity, base, remote=remote)
        except ValueError as exc:
            out.append(Check(FAIL, f"GPU capacity ({base})", str(exc)))
            continue
        if not rows:
            out.append(
                Check(
                    WARN,
                    f"GPU capacity ({base})",
                    "no executor.capacity row for this base in "
                    "the build config: GPU counts are not checked and a step runs with "
                    "whatever is here",
                )  # fmt: skip
            )
        for key, row in rows.items():
            if wanted_methods is not None and (
                (key.startswith("train:") and key[len("train:") :] not in wanted_methods)
                or (key == "student" and not dpo)
            ):
                continue  # a step this build does not run
            wanted = int(row.get("gpu_count") or 0)
            name = f"GPU {key} ({base})"
            if key == "teacher" and teacher_api:
                out.append(Check(PASS, name, "API teacher (teacher.backend http): no GPU needed"))
            elif key == "teacher":
                try:
                    tp = choose_tp(visible, wanted)
                except ValueError as err:
                    if teacher_here:
                        out.append(Check(FAIL, name, str(err)))
                    else:
                        out.append(
                            Check(
                                WARN,
                                name,
                                "no GPU here: the teacher needs a GPU host "
                                "(this build does not start it on this machine)",
                            )  # fmt: skip
                        )
                    continue
                note = "" if visible >= wanted else f"; the config asks for {wanted}"
                out.append(
                    Check(
                        PASS if visible >= wanted else WARN,
                        name,
                        f"{visible} visible, tensor parallel {tp}{note}",
                    )  # fmt: skip
                )
            elif visible >= wanted:
                out.append(Check(PASS, name, f"{visible} visible, config asks for {wanted}"))
            else:
                out.append(Check(WARN, name, f"{visible} visible, config asks for {wanted}"))
    return out


def teacher_repos(build_config: str | None, executor: str, until: str | None) -> list[str]:
    """The teacher model this build downloads onto THIS machine (`download-teacher`), else none: a
    remote executor downloads on its own disk, the API teacher has no weights, and an `--until`
    before `download-teacher` never reaches the download. The build config's
    `steps.teacher.model` wins over configs/behaviour_pipeline.yaml, as in the build."""
    from implant.behaviour import description as descmod
    from orchestrate.build import load_build_config

    if executor != "local" or until in BEFORE_DOWNLOAD or teacher_is_http(build_config):
        return []
    try:
        override = (load_build_config(REPO, build_config).get("steps") or {}).get("teacher") or {}
    except ValueError:
        override = {}
    return [str(override.get("model") or descmod.load_defaults()["teacher"]["model"])]


def download_bytes(specs: Iterable[OrganismSpec], extra_repos: Iterable[str] = ()) -> int:
    """Hub size of every base the specs name plus `extra_repos` (e.g. the teacher)."""
    from huggingface_hub import HfApi
    from huggingface_hub.errors import HfHubHTTPError

    from orchestrate.build import is_path_shaped

    api = HfApi()
    total = 0
    token = os.environ.get("HF_TOKEN") or None
    for repo in sorted({s.base_model for s in specs} | set(extra_repos)):
        if is_path_shaped(repo):
            continue  # local weights: not on the Hub, so not sized here
        try:
            info = api.model_info(repo, files_metadata=True, token=token)
        except (HfHubHTTPError, OSError, ValueError):
            continue  # check_access has already said so
        total += sum(f.size or 0 for f in info.siblings or [])
    return total


def check_machine(
    specs: list[OrganismSpec], *, offline: bool, extra_repos: Iterable[str] = ()
) -> list[Check]:
    """GPUs and disk. `extra_repos` are further Hub repos this run downloads here (a training
    build's bases and teacher); they count toward the disk the run wants."""
    gpus = visible_gpu_count()
    out = [
        Check(PASS, "GPU", f"{gpus} visible")
        if gpus
        else Check(WARN, "GPU", "none visible; run elsewhere (see docs/QUICKSTART.md)")
    ]
    cache = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    probe = next(p for p in (cache, *cache.parents) if p.exists())
    free = shutil.disk_usage(probe).free
    extra = sorted(set(extra_repos))
    if offline:
        out.append(
            Check(PASS, "disk", f"{free / 1e9:.0f} GB free on {probe}; download size not read")
        )
    else:
        need = download_bytes(specs, extra) * DISK_MARGIN
        what = f" (the build's bases and teacher: {', '.join(extra)})" if extra else ""
        from orchestrate.build import is_path_shaped

        if local := [r for r in extra if is_path_shaped(r)]:
            what += f"; local weights are not sized: {', '.join(local)}"
        out.append(
            Check(
                PASS if free >= need else WARN,
                "disk",
                f"{free / 1e9:.0f} GB free on {probe}, {need / 1e9:.0f} GB wanted{what}",
            )
        )
    out.append(
        Check(
            WARN,
            "disk quota",
            f"the check above reads the filesystem holding {probe} (the Hugging Face cache; "
            "HF_HOME moves it). It cannot see a per-user quota, which can be smaller: check "
            "yours (a build also writes checkpoints and logs).",
        )
    )
    return out


TOKENIZER_KEYS = ("pad_token", "model_max_length", "padding_side", "chat_template")
TEMPLATE_FILES = ("chat_template.jinja", "chat_template.json")


def _file_hash(sibling: Any) -> str | None:
    """An LFS file's sha256, else the git blob id (small files)."""
    lfs = getattr(sibling, "lfs", None)
    return getattr(lfs, "sha256", None) if lfs is not None else getattr(sibling, "blob_id", None)


def _hub_json(repo: str, name: str, fetch: Any) -> dict[str, Any] | None:
    """A small JSON file of a model repo (a free read), or None when it is absent/unreadable."""
    try:
        return json.loads(Path(fetch(repo, name)).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001  # absent, gated or offline: "not there" either way
        return None


def check_base_override(
    stated: str, adapter_says: str | None, *, offline: bool, api: Any = None, fetch: Any = None
) -> list[Check]:
    """A row's `base_model:` that differs from what its `adapter_config.json` names.

    Free reads only: the safetensors files' LFS sha256 (`model_info(files_metadata=True)`, no
    download), `tokenizer_config.json` (`TOKENIZER_KEYS`), the chat-template files and
    `generation_config.json`. Equal file hashes prove equal weights; different hashes prove
    nothing (re-serialising the same tensors changes them), so that is a NOTE, never a mismatch.
    Serving uses the STATED base's weights, tokenizer, chat template and generation config: the
    adapter is applied on top of the `base_model` the spec names (serving/vllm_backend.py
    `_engine_args`, `_chat_tokenizer`)."""
    name = f"base override {stated}"
    if not adapter_says or adapter_says == stated:
        return []
    if offline:
        return [Check(NOTE, name, f"adapter says {adapter_says}; unchecked (offline)")]
    if api is None:
        from huggingface_hub import HfApi

        api = HfApi()
    if fetch is None:
        from huggingface_hub import hf_hub_download

        def fetch(repo: str, filename: str) -> str:
            return hf_hub_download(repo, filename)

    token = os.environ.get("HF_TOKEN") or None
    files: dict[str, dict[str, str | None]] = {}
    for repo in (stated, adapter_says):
        try:
            info = api.model_info(repo, files_metadata=True, token=token)
        except Exception as exc:  # noqa: BLE001  # any Hub failure: say which, check nothing
            return [Check(WARN, name, f"could not read {repo} ({type(exc).__name__}); unchecked")]
        files[repo] = {s.rfilename: _file_hash(s) for s in info.siblings or []}
    a, b = files[stated], files[adapter_says]
    out: list[Check] = []
    weights_a = {k: v for k, v in a.items() if k.endswith(".safetensors")}
    weights_b = {k: v for k, v in b.items() if k.endswith(".safetensors")}
    if weights_a and weights_a == weights_b:
        out.append(Check(PASS, name, f"weights identical (same files) to {adapter_says}"))
    else:
        out.append(
            Check(
                NOTE,
                name,
                f"weight files differ from {adapter_says}'s; tensors not verified (needs a "
                "download; compare offline)",
            )
        )
    tok_a = _hub_json(stated, "tokenizer_config.json", fetch)
    tok_b = _hub_json(adapter_says, "tokenizer_config.json", fetch)
    if tok_a is None or tok_b is None:
        unread = stated if tok_a is None else adapter_says
        out.append(
            Check(
                WARN,
                f"{name} tokenizer",
                f"could not read {unread}'s tokenizer_config.json; unchecked",
            )
        )
        return out
    differ = []
    for key in TOKENIZER_KEYS:
        if tok_a.get(key) != tok_b.get(key):
            shown = (
                "differs"
                if key == "chat_template"
                else f"{tok_a.get(key)!r} vs {tok_b.get(key)!r}"
            )
            differ.append(f"{key} {shown}")
    differ += [f"{f} differs" for f in TEMPLATE_FILES if a.get(f) != b.get(f)]
    if differ:
        out.append(
            Check(
                WARN,
                f"{name} tokenizer",
                f"{stated} vs {adapter_says}: "
                + "; ".join(differ)
                + f" (serving uses {stated}'s)",
            )
        )
    else:
        out.append(Check(PASS, f"{name} tokenizer", "tokenizer settings and chat template match"))
    gen_a = _hub_json(stated, "generation_config.json", fetch)
    gen_b = _hub_json(adapter_says, "generation_config.json", fetch)
    if gen_a != gen_b:
        keys = sorted(k for k in {*(gen_a or {}), *(gen_b or {})}
                      if (gen_a or {}).get(k) != (gen_b or {}).get(k))  # fmt: skip
        out.append(
            Check(
                WARN,
                f"{name} generation_config",
                f"generation_config.json differs ({', '.join(keys) or 'present in one only'}); "
                f"serving uses {stated}'s, which can change stop tokens and default sampling",
            )
        )
    return out
