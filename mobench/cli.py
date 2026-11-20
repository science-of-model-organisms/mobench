"""`mobench`: spec files in; validation, cost, evaluation, training and a dashboard out.

Each subcommand builds the command line of something that already exists, prints it and runs
it, so logic for costs, validation and steps stays in the scripts it calls. Spec files are the
config: `--config` is repeatable and follows `serving.runner.merge_configs`. A `--config` may
also be a user config (adapters, properties, seeds: `mobench/userconfig.py`), which is turned
into a spec file first (`expand_user_configs`).

Exit codes: 0 ok; 1 a check failed; 2 a guard refused or a usage error; 3 `--strict-budget`
stopped the run; 130 interrupted.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from core.env import fake_enabled
from core.flag_values import refuse_blank, usd_amount
from core.flag_values import suggestion as _nearest

if TYPE_CHECKING:
    from core.spec import OrganismSpec

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = "configs/specs.yaml"
SMOKE_CONFIG = "configs/specs.smoke.yaml"
EXIT_OK, EXIT_FAILED, EXIT_REFUSED, EXIT_BUDGET = 0, 1, 2, 3
EXIT_INTERRUPTED = 130
UNKNOWN = "unknown"
SPEC_HASH_RE = re.compile(r"\[([0-9a-f]{16})\] x ")


def is_fake() -> bool:
    return fake_enabled()


def config_paths(args: argparse.Namespace) -> list[str]:
    """`--config` values, else `$MO_EVALS_CONFIG`, else the repo's matrix."""
    return args.config or [os.environ.get("MO_EVALS_CONFIG", DEFAULT_CONFIG)]


def flashinfer_env() -> dict[str, str]:
    """`VLLM_USE_FLASHINFER_SAMPLER=0` for the commands mobench launches when there is no `nvcc`.

    The FlashInfer sampler JIT-compiles its kernels with nvcc at engine warmup; a compute node
    with the CUDA runtime but no toolkit (the AISI pilot, 2026-11-10) crashes there, and vLLM's
    native sampler needs no compiler. Only mobench's own launches get it (shown with the
    command), so runs started by run_all.sh or inspect_runner directly are unchanged. A value
    the caller exported is never touched."""
    if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ or shutil.which("nvcc"):
        return {}
    return {"VLLM_USE_FLASHINFER_SAMPLER": "0"}


def shown(
    cmd: Sequence[str], env: Mapping[str, str] | None = None, *, capture: bool = False
) -> subprocess.CompletedProcess[str]:
    """Print a command, with the variables it adds (names and values are ours), and run it."""
    env = {**flashinfer_env(), **(env or {})} if not is_fake() else env
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
    print("$", prefix, shlex.join(cmd), flush=True)
    return subprocess.run(
        list(cmd),
        cwd=REPO,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", **(env or {})},
        check=False,
        capture_output=capture,
        text=True,
    )


def selection_flags(args: argparse.Namespace) -> list[str]:
    """The filters `serving.inspect_runner` and `run_all.sh` both take."""
    out: list[str] = []
    if getattr(args, "user_only", False) and not args.personas:
        out += ["--personas", "all"]  # every organism was named in a user config
    for flag in ("personas", "methods", "base_models", "evals", "exclude_evals"):
        for value in getattr(args, flag, None) or []:
            out += [f"--{flag.replace('_', '-')}", value]
    return out


def dry_run_flags(args: argparse.Namespace) -> list[str]:
    """As `selection_flags`, minus `all`, which only `run_all.sh` understands."""
    flags = selection_flags(args)
    pairs = [(flags[i], flags[i + 1]) for i in range(0, len(flags), 2)]
    return [x for flag, value in pairs if value != "all" for x in (flag, value)]


def needs_personas(args: argparse.Namespace) -> bool:
    """Which behaviours to run is the human's call on a real run (CLAUDE.md, "Real runs").

    A run whose every `--config` is a user config has already named its organisms one by one,
    so it needs no `--personas`; one that also reads a spec file still does."""
    if args.personas or is_fake() or getattr(args, "user_only", False):
        return False
    print("REFUSING: a real run needs --personas <id>[,<id>], or --personas all alone.")
    return True


def expand_user_configs(args: argparse.Namespace, *, online: bool) -> str | None:
    """Swap every user config among the `--config` files for the spec file it desugars to.

    Sets `args.user_plans` (one `userconfig.UserPlan` each) and `args.user_only` (every file
    was a user config). Returns a refusal line, or None. A file that does not parse is passed
    on unchanged, so the existing error path reports it."""
    from mobench import userconfig as uc
    from serving.runner import ConfigReadError, SpecConflictError, load_config, merge_configs

    paths = config_paths(args)
    plans: list[uc.UserPlan] = []
    out: list[str] = []
    scratch: Path | None = None
    for path in paths:
        try:
            data = load_config(path)
        except (OSError, ValueError, yaml.YAMLError):
            out.append(path)
            continue
        if not uc.is_user_config(data):
            out.append(path)
            continue
        try:
            # `or None`: without the flag, desugar keeps reading $MOBENCH_REFRESH_PINS.
            plan = uc.desugar(
                path,
                data,
                online=online,
                refresh_pins=getattr(args, "refresh_pins", False) or None,
            )
        except uc.UserConfigError as exc:
            return f"REFUSING: {exc}"
        scratch = scratch or Path(tempfile.mkdtemp(prefix="mobench-config-"))
        out.append(str(uc.write_spec_file(plan, scratch)))
        plans.append(plan)
    args.user_plans = plans
    args.user_only = bool(plans) and len(plans) == len(paths)
    if not plans:
        return None
    args.config = out
    chosen = {e for plan in plans for e in plan.evals}
    asked = split_values(getattr(args, "evals", None))
    outside = sorted(asked - chosen)
    if outside:
        return (
            f"REFUSING: --evals {outside} are not in the user config's properties/evals "
            f"({sorted(chosen)}); add them there (the plan and the behaviour checks read it), "
            "or use --evals only to narrow"
        )
    roots = sorted({p.results_root for p in plans if p.results_root})
    if len(roots) > 1:
        return f"REFUSING: the user configs name different results_root values {roots}; use one"
    flag_or_env = getattr(args, "results_root", None) or os.environ.get("MO_EVALS_RESULTS_ROOT")
    if roots and hasattr(args, "results_root") and not flag_or_env:
        args.results_root = roots[0]
    if not args.user_only:
        try:
            merged = merge_configs(out)
        except ConfigReadError:
            raise  # `main` turns an unreadable spec file into the one `REFUSING: cannot read` line
        except (SpecConflictError, ValueError) as exc:
            return f"REFUSING: the user config and the spec files do not merge: {exc}"
        union = set(merged.get("default_evals") or [])
        for plan in plans:
            extra = sorted(union - set(plan.evals))
            if extra:
                return (
                    f"REFUSING: merged with the other --config files, the rows of {plan.path} "
                    f"would also run {extra}, which its properties/evals leave out (the merged "
                    "eval list is the union). Run it with its own --config alone."
                )
    return None


def listed(values: list[str] | None) -> list[str]:
    """`core.flag_values.listed`: the one parse of a comma-separated, repeatable flag."""
    from core.flag_values import listed as parse

    return parse(values)


def split_values(values: list[str] | None) -> set[str]:
    """Repeatable or comma-separated flag values, as they were written. `all` is a name like any
    other here: only `--personas` accepts it (`persona_values`); on any other flag it is an
    unknown name that is refused up front, as the runner refuses it."""
    return set(listed(values))


def persona_values(values: list[str] | None) -> set[str]:
    """`--personas` values, `all` dropped: `run_all.sh --personas all` is the explicit opt-in to
    every behaviour, the one flag where `all` is not a name."""
    return {x for x in listed(values) if x != "all"}


LIST_FLAGS = (
    ("--personas", "personas"),
    ("--methods", "methods"),
    ("--base-models", "base_models"),
    ("--evals", "evals"),
    ("--exclude-evals", "exclude_evals"),
)


def clean_list_flags(args: argparse.Namespace) -> str | None:
    """Every comma-list flag (run, cost and validate, `--train` or not): refuse one that was
    given but names nothing (`--personas ,`), else store it cleaned (`listed`), once, so every
    later reader and the job script see the same values."""
    from core.flag_values import empty_problem

    for flag, name in LIST_FLAGS:
        if problem := empty_problem(flag, getattr(args, name, None)):
            return problem
    for _, name in LIST_FLAGS:
        if getattr(args, name, None):
            setattr(args, name, [",".join(listed(getattr(args, name)))])
    return None


def normalise_base_args(args: argparse.Namespace) -> str | None:
    """Make every path-shaped `--base-models` its absolute path, once, before anything reads it
    (validate, --print-job, the capacity lookups, the build). A refusal line when
    `orchestrate.build.normalise_base` refuses (a relative or `~` path under a remote executor, or
    `~user` with no such user): the same rule the direct `orchestrate build` applies."""
    from orchestrate.build import normalise_base

    if problem := clean_list_flags(args):
        return problem
    if not args.base_models:
        return None
    remote = getattr(args, "executor", "local") != "local"
    parts = []
    for base in listed(args.base_models):
        try:
            parts.append(normalise_base(base, remote=remote))
        except ValueError as exc:
            return f"--base-models {exc}"
    args.base_models = [",".join(parts)]
    return None


def cmd_validate(args: argparse.Namespace) -> int:
    from mobench import validate as v
    from serving.runner import SpecConflictError, load_specs, merge_configs

    fake = is_fake()
    offline = args.offline or fake
    if problem := normalise_base_args(args) if args.train else clean_list_flags(args):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED
    refusal = expand_user_configs(args, online=not offline)
    if refusal:
        print(refusal.replace("REFUSING:", "FAIL user config:", 1))
        return EXIT_FAILED
    paths = config_paths(args)
    try:
        config = merge_configs(paths)
    except yaml.YAMLError:
        raise  # `main` turns an unloadable YAML into the one `REFUSING:` line
    except (SpecConflictError, ValueError, OSError) as exc:
        print(f"FAIL specs: {type(exc).__name__}: {exc}")
        return EXIT_FAILED
    specs = load_specs(config)
    everyone = specs
    unmatched: list[str] = []
    wanted = persona_values(args.personas)  # `all` is no filter: the same as no `--personas`
    if wanted:
        specs = [s for s in specs if s.persona_id in wanted]
        unmatched = sorted(wanted - {s.persona_id for s in specs})
    scope: list[v.Check] = []
    train_bases: set[str] = set()
    train_methods: set[str] = set()
    if args.train:
        if args.methods:  # a typo would silently drop every organism from the check
            from orchestrate.build import load_build_config

            try:
                recipes_config = load_build_config(REPO, args.build_config)
            except ValueError:
                recipes_config = {}  # `check_training` reports an unreadable build config
            if recipes_config and (
                problem := methods_problem(recipes_config, split_values(args.methods))
            ):
                print(f"REFUSING: {problem}")
                return EXIT_REFUSED
        from orchestrate.build import behaviour_input_problem

        for persona in listed(args.personas):
            if problem := behaviour_input_problem(REPO, persona):
                print(f"REFUSING: {problem}")
                return EXIT_REFUSED
        if getattr(args, "until", None) and (recipes_config := _readable_build_config(args)):
            if problem := until_problem(args, recipes_config):
                print(f"REFUSING: {problem}")
                return EXIT_REFUSED
        # A build trains new adapters on one base: only that base and the methods it trains
        # matter, from the flags or else from the build config. Other organisms are out of scope:
        # a NOTE, never a failure.
        train_bases, train_methods = v.train_scope(
            args.build_config, split_values(args.base_models), split_values(args.methods)
        )
        kept = [
            s
            for s in specs
            if (not train_bases or s.base_model in train_bases)
            and (not train_methods or s.method in train_methods)
        ]
        if len(kept) != len(specs):
            other = sorted({s.base_model for s in specs if s not in kept})
            scope.append(
                v.Check(
                    v.NOTE,
                    "scope",
                    f"{len(specs) - len(kept)} organism(s) of other bases or methods (bases "
                    f"{other}) are not needed by this build and were not checked",
                )
            )
        specs = kept
    checks = [
        v.Check(v.PASS, "specs", f"{len(specs)} organisms from {len(paths)} file(s)"),
        *scope,
    ]
    if unmatched and args.train:
        new = [p for p in unmatched if v.is_known_behaviour(p)]
        unmatched = [p for p in unmatched if p not in new]
        if new:
            made = [
                f"{p} / {m} on {b}"
                for p in new
                for b in sorted(train_bases)
                for m in sorted(train_methods)
            ]
            checks.append(
                v.Check(
                    v.NOTE,
                    "personas",
                    f"no spec rows yet for {new}; this build creates: " + "; ".join(made),
                )  # fmt: skip
            )
    if unmatched:
        checks.append(v.Check(v.FAIL, "personas", f"no organism for {unmatched}"))
    if not args.train:  # a name no organism has: the same typo guard `run` and `cost` apply
        known = {
            "--methods": {s.method for s in everyone},
            "--base-models": {s.base_model for s in everyone},
        }
        for flag, values in (("--methods", args.methods), ("--base-models", args.base_models)):
            if bad := sorted(split_values(values) - known[flag]):
                checks.append(v.Check(v.FAIL, flag.lstrip("-"), f"no organism has {bad}"))
        if not any(c.status == v.FAIL for c in checks):  # what `run` refuses, validate fails
            if problem := selection_problem(args, everyone):
                checks.append(v.Check(v.FAIL, "selection", problem))
            else:  # every later check reads exactly the organisms the runner would select
                from serving.runner import _select_specs

                specs = _select_specs(
                    list(everyone),
                    sorted(persona_values(args.personas)),
                    sorted(split_values(args.methods)),
                    sorted(split_values(args.base_models)),
                )
    # The count is of the list every later check receives, whatever narrowed it.
    checks[0] = v.Check(v.PASS, "specs", f"{len(specs)} organisms from {len(paths)} file(s)")
    extra_bases: list[str] = []  # Hub bases; a local checkpoint is checked, not read from the Hub
    local_paths: list[str] = []
    for named in sorted(train_bases):
        loc = v.local_base(named)
        if loc is None:
            if why := v.hub_base_shape_problem(named):  # spelling only: free, even offline
                checks.append(v.Check(v.FAIL, f"base {named}", why))
            else:
                extra_bases.append(named)
            continue
        checks.append(v.check_local_base(named, loc, remote=args.executor != "local"))
        if loc.exists():
            local_paths.append(str(loc))
    if offline:
        checks.append(v.Check(v.SKIP, "network checks", "fake or --offline"))
    else:
        checks += v.check_base_models(specs, extra_bases)
    if not fake:
        v.behaviour_status(s.persona_id for s in specs)
        needs = v.needed_keys(
            config,
            specs,
            v.build_keys(args.build_config) if args.train else (),
            evals=None if args.train else config.get("default_evals"),
        )
        checks += v.check_keys(needs, offline=offline)
    if not offline:
        checks += v.check_access(specs, extra_bases=extra_bases, adapters=not args.train)
    for plan in args.user_plans:
        for row in plan.rows:
            checks.append(
                v.Check(
                    v.PASS,
                    f"external {row.label}",
                    f"{row.repo}@{row.revision[:12]} on {row.base_model} ({row.base_source})",
                )
            )
            if row.base_source == "config":
                checks += v.check_base_override(row.base_model, row.adapter_says, offline=offline)
    checks += v.check_install(config, specs, train=args.train, fake=fake)
    checks += v.check_declarations()
    if args.train:
        checks += v.check_training(args.build_config, fake=fake, offline=offline)
        checks += v.check_teacher_block(args.build_config)
        checks += v.check_api_budget(args.build_config, args.until)
    if not fake:
        wanted = (
            [
                *extra_bases,
                *local_paths,
                *v.teacher_repos(args.build_config, args.executor, args.until),
            ]
            if args.train
            else []
        )
        checks += v.check_machine(specs, offline=offline, extra_repos=wanted)
        if args.train:
            bases = {*train_bases, *(s.base_model for s in specs)}
            here = v.teacher_runs_here(args.executor, args.until)
            checks += v.check_gpus(
                args.build_config,
                bases,
                v.visible_gpu_count(),
                teacher_here=here,
                teacher_api=v.teacher_is_http(args.build_config),
                methods=train_methods,
                remote=args.executor != "local",
            )
    for check in checks:
        print(check.line())
    return EXIT_FAILED if any(c.status == v.FAIL for c in checks) else EXIT_OK


def eval_cells(
    args: argparse.Namespace, root: str, by_spec: dict[str, dict[str, int]] | None = None
) -> tuple[int, dict[str, int]]:
    """`serving.inspect_runner --dry-run` and the state of every cell it lists; `by_spec`, when
    given, is filled with the same counts per spec_hash."""
    cmd = [sys.executable, "-m", "serving.inspect_runner", "--dry-run", "--results-root", root]
    for path in config_paths(args):
        cmd += ["--config", path]
    done = shown(cmd + dry_run_flags(args), capture=True)
    from serving.runner import EXIT_SELECTION

    lines = [x for x in done.stderr.splitlines() if x.strip()]
    if done.returncode == EXIT_SELECTION and lines and lines[-1].startswith("REFUSING:"):
        print(lines[-1])  # the runner's own refusal line, as it is
        return EXIT_REFUSED, {}
    counts: dict[str, int] = {}
    for line in done.stdout.splitlines():
        word = line.split(" ", 1)[0]
        if word in {"TODO", "CACHED", "DEDUPE", "CONFLICT"}:
            counts[word] = counts.get(word, 0) + 1
            hit = SPEC_HASH_RE.search(line)
            if by_spec is not None and hit:
                cell = by_spec.setdefault(hit.group(1), {})
                cell[word] = cell.get(word, 0) + 1
    if done.returncode:
        print(done.stderr, file=sys.stderr)
    return done.returncode, counts


def results_root(args: argparse.Namespace) -> str:
    return args.results_root or os.environ.get("MO_EVALS_RESULTS_ROOT", "results")


def cmd_cost(args: argparse.Namespace) -> int:
    """What a run would do. Optional: a run never needs this, and `unknown` is a fine answer.
    With `--train` it is a build's plan (`run --train --plan`), not the evaluation matrix."""
    if args.train:
        args.user_plans, args.user_only = [], False
        args.plan, args.yes, args.budget, args.strict_budget = True, False, None, False
        if problem := normalise_base_args(args):
            print(f"REFUSING: {problem}")
            return EXIT_REFUSED
        if needs_personas(args):
            return EXIT_REFUSED
        return cmd_train(args)
    if problem := clean_list_flags(args):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED
    refusal = expand_user_configs(args, online=not is_fake())
    if refusal:
        print(refusal)
        return EXIT_REFUSED
    if needs_personas(args):
        return EXIT_REFUSED
    refusal = unknown_selection(args)
    if refusal:
        print(refusal)
        return EXIT_REFUSED
    if args.user_plans:
        if print_plan(args):
            return EXIT_FAILED
        print(f"evaluation cost: {UNKNOWN} (no dollar figure exists for an eval run)")
        return EXIT_OK
    rc, counts = eval_cells(args, results_root(args))
    if rc:
        return rc if rc == EXIT_REFUSED else EXIT_FAILED
    print("work:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing")
    print(f"evaluation cost: {UNKNOWN} (no dollar figure exists for an eval run)")
    return EXIT_OK


def print_plan(args: argparse.Namespace) -> int:
    """What a run with user configs will do, before it does it: each row and its base, the base
    rows (cached or to run), cell counts per row from the dry run, the skipped evals and why,
    and the cost. Returns the dry run's exit code."""
    from mobench.userconfig import plan_lines
    from serving.runner import expand_seeds, load_specs, merge_configs

    root = results_root(args)
    config = merge_configs(config_paths(args))
    who = {s.spec_hash: s for s in expand_seeds(load_specs(config), config)}
    by_spec: dict[str, dict[str, int]] = {}
    rc, counts = eval_cells(args, root, by_spec)
    if rc:
        return rc
    per_row: dict[tuple[str, str, str], dict[str, int]] = {}
    for spec_hash, cell in by_spec.items():
        spec = who.get(spec_hash)
        if spec is None:
            continue
        row = per_row.setdefault((spec.persona_id, spec.method, spec.base_model), {})
        for state, n in cell.items():
            row[state] = row.get(state, 0) + n
    for plan in args.user_plans:
        for line in plan_lines(plan):
            print(line)
    print(f"cells (results root {root}):")
    for (persona, method, base), cell in sorted(per_row.items()):
        todo = cell.get("TODO", 0) + cell.get("CONFLICT", 0)
        state = ", ".join(f"{k.lower()} {v}" for k, v in sorted(cell.items()))
        if method == "base":
            status = "cached, reused" if not todo else f"to run ({todo} cell(s))"
            print(f"  {persona}/base on {base}: {status} [{state}]")
        else:
            print(f"  {persona}/{method} on {base}: {state}")
    print("total:", ", ".join(f"{k.lower()} {v}" for k, v in sorted(counts.items())) or "nothing")
    return 0


def selected_specs(args: argparse.Namespace) -> list[OrganismSpec]:
    """The organisms `--personas`, `--methods` and `--base-models` keep."""
    from serving.runner import load_specs_many

    personas, methods, bases = (
        persona_values(args.personas),
        split_values(args.methods),
        split_values(args.base_models),
    )
    return [
        s
        for s in load_specs_many(config_paths(args))
        if (not personas or s.persona_id in personas)
        and (not methods or s.method in methods)
        and (not bases or s.base_model in bases)
    ]


def budget_gate(step: str, estimate: float | None, args: argparse.Namespace) -> int | None:
    """Override A: an estimate informs. Only `--strict-budget` can refuse (exit 2) or stop (3)."""
    shown_cost = UNKNOWN if estimate is None else f"${estimate:.2f}"
    print(f"{step} cost: {shown_cost}")
    if args.budget is None:
        return None
    if estimate is None or (estimate <= 0 and args.strict_budget):
        if args.strict_budget:
            print(f"REFUSING: --strict-budget needs a non-zero estimate for {step}; this is "
                  "admission against your figure, not metering of the run.")  # fmt: skip
            return EXIT_REFUSED
        print(
            f"{step}: no estimate, so --budget ${args.budget:.2f} cannot be checked; carrying on"
        )
    elif estimate > args.budget:
        print(f"WARNING: {step} is projected at {shown_cost}, over --budget ${args.budget:.2f}")
        if args.strict_budget:
            return EXIT_BUDGET
    return None


def with_flag(argv: Sequence[str], flag: str, values: list[str] | None) -> list[str]:
    """`argv` with every `flag` (both `--flag v` and `--flag=v`, repeated ones too) replaced by
    one `flag <values>` at the first one's place: the job script carries the parsed value."""
    if not values:
        return list(argv)
    out, words, placed = [], iter(argv), False
    for word in words:
        if word == flag or word.startswith(f"{flag}="):
            if word == flag:
                next(words, None)  # the value as typed is dropped
            if not placed:
                out += [flag, values[0]]
                placed = True
        else:
            out.append(word)
    return out


def job_script(args: argparse.Namespace, argv: Sequence[str]) -> str:
    """A generic sbatch script: commands and env var names, and nothing about any one site."""
    from mobench.validate import build_keys, needed_keys
    from serving.runner import load_specs, merge_configs

    forwarded = [a for a in argv if a != "--print-job"]
    if args.train:  # the job's cwd is not ours: the cleaned, absolute values, not as typed
        forwarded = with_flag(forwarded, "--base-models", args.base_models)
        forwarded = with_flag(forwarded, "--personas", args.personas)
        forwarded = with_flag(forwarded, "--methods", args.methods)
    if not {"-y", "--yes"} & set(forwarded):
        forwarded.append("-y")  # no terminal under sbatch: without it confirm_run refuses
    if args.train:
        keys = build_keys(args.build_config)
        gres, header, sync, confirm = train_job_parts(args)
    else:
        config = merge_configs(config_paths(args))
        keys = needed_keys(config, load_specs(config), evals=config.get("default_evals"))
        gres = "#SBATCH --gres=gpu:1"
        header = "# docs/QUICKSTART.md lists how much GPU memory a base model needs."
        sync = "uv sync --extra gpu --extra ifeval-anchor --extra inspect-evals"
        confirm = "# -y: submitting this job is the confirmation. Check `mobench cost` first."
    checks = "\n".join(f': "${{{k}:?set {k} in the job environment}}"' for k in keys)
    return f"""#!/bin/bash
#SBATCH --job-name=mobench
{gres}
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# Add your site's partition, account, QOS and time limit above. Run time is not known in advance.
{header}
set -euo pipefail
{checks}
{sync}
{confirm}
uv run mobench {shlex.join(forwarded)}
"""


DEFAULT_METHODS = ("sft_behaviour", "dpo_behaviour")


def methods_problem(config: Mapping, asked: Iterable[str]) -> str | None:
    """A refusal line listing the build recipes when a method has none; the one check the
    normal path and `--print-job` (which builds the step graph) both make first."""
    recipes = sorted(config.get("methods") or {})
    if bad := sorted(set(asked) - set(recipes)):
        return f"--methods {', '.join(bad)}: no build recipe. Available: {', '.join(recipes)}."
    return None


def ordered_methods(args: argparse.Namespace) -> list[str]:
    """The methods a build trains, in the order `orchestrate build` runs them."""
    return listed(args.methods) or list(DEFAULT_METHODS)


def gpu_rows_for(args: argparse.Namespace, config: dict, methods: list[str]) -> list[str]:
    """The capacity rows the GPU steps up to `--until` read, from the real step graph
    (`orchestrate.steps.gpu_capacity_until`), so the method order, the API teacher and the
    screen and eval rows are the build's own, not a copy of them here."""
    from orchestrate.executors import FakeExecutor
    from orchestrate.ledger import Ledger
    from orchestrate.steps import Ctx, StepStop, gpu_capacity_until

    if problem := methods_problem(config, methods):
        raise ValueError(problem)  # before the graph: it indexes each method's recipe
    base = next(iter(split_values(args.base_models)), "")
    ctx = Ctx(
        behaviour=next(iter(persona_values(args.personas)), ""),
        base=base,
        methods=methods,
        repo=REPO,
        work=Path("/nonexistent-plan-only"),
        config={**config, "executor": config.get("executor") or {}},
        template="loyalty",
        specs={},
        executor=FakeExecutor(),
        ledger=Ledger(Path("/nonexistent-plan-only/ledger.jsonl")),
    )
    try:
        return gpu_capacity_until(ctx, args.until)
    except StepStop as exc:
        raise ValueError(str(exc)) from exc


def _readable_build_config(args: argparse.Namespace) -> dict:
    """The build config, or `{}` when it is unreadable (the checks that report that say so)."""
    from orchestrate.build import load_build_config

    try:
        return load_build_config(REPO, args.build_config)
    except ValueError:
        return {}


def until_problem(args: argparse.Namespace, config: dict) -> str | None:
    """A refusal line when `--until` names no step of the build's graph (checked with the
    methods, before any plan, print-job, validation or spend); None when it is fine."""
    if not getattr(args, "until", None) or methods_problem(config, ordered_methods(args)):
        return None  # nothing asked, or the methods check reports first
    try:
        gpu_rows_for(args, config, ordered_methods(args))
    except ValueError as exc:
        return str(exc)
    return None


def train_job_parts(args: argparse.Namespace) -> tuple[str, str, str, str]:
    """The GPU request, comment block, `uv sync` line and confirmation note of a `--train` job.

    The GPUs come from the build config's `executor.capacity` row for the base: the largest row
    among the steps this job runs. The teacher step is one of them only when the job starts it
    here (local executor, `--until` at or past `pilot-teacher`, not the API teacher), and it is the
    4-GPU one; a job that stops before it, or uses the API teacher, asks for what the rest
    needs."""
    from mobench import validate as v
    from orchestrate.build import load_build_config

    bases = split_values(args.base_models)
    capacity: dict = {}
    try:
        config = load_build_config(REPO, args.build_config)
        from orchestrate.build import capacity_for

        capacity = capacity_for(
            (config.get("executor") or {}).get("capacity"),
            next(iter(bases), ""),
            remote=args.executor != "local",
        )
    except ValueError as exc:
        raise ValueError(f"build config {args.build_config}: {exc}") from exc
    # The capacity rows of the GPU steps up to `--until`, from the graph. Asked for every
    # executor so a bad method or --until is refused alike; a remote one keeps none of them (a pod
    # does the GPU work, this job is only the orchestrator).
    wanted = gpu_rows_for(args, config, ordered_methods(args))
    if args.executor != "local":
        wanted = []
    # screen and eval rows are optional (configs/build.yaml ships none): they count when set
    if missing := [key for key in wanted if key not in capacity and key not in ("screen", "eval")]:
        raise ValueError(
            f"{next(iter(bases), '(no base)')} has no executor.capacity row for "
            f"{', '.join(missing)} in the build config, so the GPUs this job needs are not "
            "known. Add `executor.capacity.<base>.<row>: {gpu_class, gpu_count}` for each to "
            "your --build-config (see configs/build.yaml)."
        )
    rows = {key: row for key, row in capacity.items() if key in wanted}
    count = max((int(r.get("gpu_count") or 0) for r in rows.values()), default=0)
    asks = ", ".join(f"{k} {r.get('gpu_count')}x {r.get('gpu_class')}" for k, r in rows.items())
    teacher_gpus = (capacity.get("teacher") or {}).get("gpu_count", 4)
    teacher = (v.teacher_repos(args.build_config, "local", None) or [None])[0]
    if teacher:
        split = f"""# The teacher ({teacher}) needs all its GPUs on ONE node: name the GPU type in
# --gres if your site has several, and raise --mem if the weights fail to load. The cheaper split
# (docs/QUICKSTART.md): API steps on a CPU job (--until few-shot, later --until pilot-gate), the
# teacher steps on a {teacher_gpus}-GPU job (--until pilot-teacher), training and screening on a
# 1-GPU job; all
# share one --builds-root and paths.model_path."""
    else:
        split = """# The teacher runs over the API (steps.teacher.backend http): no GPUs for it.
# The cheaper split (docs/QUICKSTART.md): API steps on a CPU job, training and screening on
# a 1-GPU job; both share one --builds-root and paths.model_path."""
    header = f"""# The build config asks for: {asks or "no GPU step in this job"}.
{split}
# Before submitting: git submodule update --init; LIMA under paths.model_path/lima; HF_HOME and
# UV_PROJECT_ENVIRONMENT on local disks with room (a quota can be smaller than df shows);
# `mobench validate --train` checks the rest."""
    # evaluate (the last step) runs the stock capability evals, which need these extras: the pod
    # job script installs them too (orchestrate/executors.py). A job that stops earlier skips them.
    sync = "uv sync --all-groups --extra gpu"
    if args.until is None or args.until == "evaluate":
        sync += " --extra ifeval-anchor --extra inspect-evals"
    return (
        f"#SBATCH --gres=gpu:{count}" if count else "# no GPU: this job runs API and free steps",
        header,
        sync if count else "uv sync --all-groups",
        "# -y: submitting this job is the confirmation. Check `mobench run --train --plan` first.",
    )


def cmd_report(args: argparse.Namespace) -> int:
    from mobench.report import write_build_config

    root = Path(results_root(args)).resolve()
    out_dir = Path(args.out or os.environ.get("MO_EVALS_REPORT_DIR", "analysis/out"))
    out_dir = (REPO / out_dir).resolve() / "mobench"
    if args.dashboard_config:
        config = Path(args.dashboard_config)
    else:
        try:
            config = write_build_config(root, out_dir)
        except ValueError as exc:
            print(f"FAIL {exc}")
            return EXIT_FAILED
    build = [sys.executable, "-m", "analysis.dashboard_build", "--config", str(config)]
    for stage in ("fetch", "all"):
        if shown([*build, stage]).returncode:
            return EXIT_FAILED
    return EXIT_OK


def selection_problem(args: argparse.Namespace, specs: Sequence[OrganismSpec]) -> str | None:
    """Why the runner would refuse `--personas`/`--methods`/`--base-models` together, by its own
    `_select_specs` (each filter applied to what the earlier ones left); None when it accepts.
    `all` mixed with names is refused as `run_all.sh` refuses it."""
    from serving.runner import SelectionError, _select_specs

    given = listed(getattr(args, "personas", None))
    if "all" in given and len(given) > 1:
        return (
            "--personas mixes `all` with named behaviours "
            f"({', '.join(x for x in given if x != 'all')}); pass `all` alone or list names."
        )
    try:
        _select_specs(
            list(specs),
            sorted(persona_values(getattr(args, "personas", None))),
            sorted(split_values(getattr(args, "methods", None))),
            sorted(split_values(getattr(args, "base_models", None))),
        )
    except SelectionError as exc:
        return str(exc)
    return None


def unknown_selection(args: argparse.Namespace) -> str | None:
    """A one-line refusal when `--personas`/`--methods`/`--base-models` names something no spec
    file has (with the nearest match), checked before anything is launched; None when all match.
    The runner checks this too, but only after the run has started, as a traceback."""
    from serving.runner import load_specs_many

    specs = load_specs_many(config_paths(args))
    for flag, values, have in (
        ("--personas", getattr(args, "personas", None), {s.persona_id for s in specs}),
        ("--methods", getattr(args, "methods", None), {s.method for s in specs}),
        ("--base-models", getattr(args, "base_models", None), {s.base_model for s in specs}),
    ):
        names = persona_values(values) if flag == "--personas" else split_values(values)
        for name in sorted(names - have):
            return (
                f"REFUSING: {flag} {name}: no organism in {', '.join(config_paths(args))} has it."
                f"{_suggest(name, sorted(have))} Available: {', '.join(sorted(have))}."
            )
    import serving.inspect_runner  # noqa: F401  # registers every eval
    from evals._inspect import all_evals
    from serving.runner import SelectionError, _select_evals, merge_configs

    if problem := selection_problem(args, specs):
        return f"REFUSING: {problem}"
    try:  # the runner's own selection (--evals, else default_evals, then --exclude-evals), so
        # the two cannot drift: an eval can be registered yet not selected
        _select_evals(
            all_evals(),
            sorted(split_values(getattr(args, "evals", None))),
            merge_configs(config_paths(args)),
            sorted(split_values(getattr(args, "exclude_evals", None))),
        )
    except SelectionError as exc:
        return f"REFUSING: {exc}"
    return None


def run_validate(args: argparse.Namespace) -> int:
    """`mobench validate` on the files as given (user configs unexpanded), before a run."""
    flags = [x for p in args.given_configs for x in ("--config", p)]
    for value in args.personas or []:
        flags += ["--personas", value]
    # The run already resolved (and recorded) the pins: validate reads them and never re-resolves,
    # or a moving head could leave the run on one commit and the record on another.
    env = {"MOBENCH_REFRESH_PINS": "0"}
    return shown([sys.executable, "-m", "mobench", "validate", *flags], env).returncode


def selected_evals(args: argparse.Namespace) -> list[str]:
    """The evals a run would score: the user configs' choice, else the merged spec files'
    `default_evals`, narrowed by `--evals` and `--exclude-evals`."""
    from serving.runner import merge_configs

    if args.user_plans:
        chosen = {e for plan in args.user_plans for e in plan.evals}
    else:
        try:
            chosen = set(merge_configs(config_paths(args)).get("default_evals") or [])
        except Exception:  # noqa: BLE001  # the run's own path reports a bad file
            chosen = set()
    if asked := split_values(getattr(args, "evals", None)):
        chosen &= asked
    return sorted(chosen - split_values(getattr(args, "exclude_evals", None)))


def run_behaviours(args: argparse.Namespace) -> list[str]:
    names = persona_values(args.personas) | {p.behaviour for p in args.user_plans if p.behaviour}
    return sorted(names)


def behaviour_data_problems(args: argparse.Namespace) -> list[str]:
    """Behaviours whose generated data is missing or stale (`orchestrate/generate.py`).

    Checked: a behaviour `mobench generate` has a state for (`builds/<id>/data/state.json`), and
    one with a constitution but no phenomenon pair (it was never generated). A committed
    behaviour without either is evaluated as before: its evals skip what it lacks."""
    from orchestrate.drafting import PAIRS
    from orchestrate.generate import data_problems

    root = Path(args.builds_root)
    root = root if root.is_absolute() else REPO / root
    pairs = (yaml.safe_load((REPO / PAIRS).read_text(encoding="utf-8")) or {}).get("pairs") or {}
    evals = selected_evals(args)
    lines = []
    for bid in run_behaviours(args):
        has_state = (root / bid / "data" / "state.json").is_file()
        new = (
            REPO / "data" / "behaviours" / bid / "constitution.json"
        ).is_file() and bid not in pairs
        if not (has_state or new):
            continue
        problems = data_problems(REPO, bid, evals, root)
        if problems:
            lines.append(f"{bid}: " + "; ".join(problems))
            lines.append(
                f"  run `mobench generate {bid}`, then `mobench generate {bid} --eval-data`"
            )
    return lines


def confirm_run(args: argparse.Namespace) -> int | None:
    """None to go on. `-y` accepts; a terminal is asked; no terminal and no `-y` refuses."""
    if args.yes:
        return None
    if not sys.stdin.isatty():
        print("REFUSING: no terminal to confirm the run; rerun with -y/--yes to accept it")
        return EXIT_REFUSED
    if input("Run it? [y/N] ").strip().lower() != "y":
        print("not confirmed; nothing ran")
        return EXIT_REFUSED
    return None


def preflight(args: argparse.Namespace) -> int | None:
    """Validate, show the work and its cost, check generated behaviour data, then confirm."""
    if run_validate(args):
        print("REFUSING: `mobench validate` failed (above); fix it, then rerun")
        return EXIT_FAILED
    if not args.user_plans:  # a user config's plan (with its cells) is printed already
        rc, counts = eval_cells(args, results_root(args))
        if rc:
            return rc if rc == EXIT_REFUSED else EXIT_FAILED
        print("work:", ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "nothing")
    if problems := behaviour_data_problems(args):
        print("REFUSING: behaviour data missing or stale:")
        for line in problems:
            print(f"  {line}")
        return EXIT_REFUSED
    return confirm_run(args)


def cmd_run(args: argparse.Namespace) -> int:
    """`run`; with `--print-job`, stdout carries ONLY the job script (it is redirected into a
    file for sbatch), so every refusal, NOTE and other diagnostic goes to stderr."""
    if not args.print_job:
        return run_command(args)
    with contextlib.redirect_stdout(sys.stderr):
        rc = run_command(args)
    if rc == EXIT_OK:
        print(args.job_script_text, end="")
    return rc


def run_command(args: argparse.Namespace) -> int:
    """Evaluate every selected organism, then build the page if asked."""
    args.given_configs = list(config_paths(args))
    if not args.train:
        if problem := clean_list_flags(args):
            print(f"REFUSING: {problem}")
            return EXIT_REFUSED
        refusal = expand_user_configs(args, online=not is_fake())
        if refusal:
            print(refusal)
            return EXIT_REFUSED
    else:
        args.user_plans, args.user_only = [], False
        if problem := normalise_base_args(args):
            print(f"REFUSING: {problem}")
            return EXIT_REFUSED
    if args.plan and not args.train:
        print("REFUSING: --plan applies to --train (it prints a build's plan); "
              "`mobench cost` lists an evaluation's work.")  # fmt: skip
        return EXIT_REFUSED
    if args.print_job:
        if args.train:  # every free check a run makes before its first paid step
            if (rc := train_preflight(args)) != EXIT_OK:
                return rc
        else:  # an evaluation script: the checks of a non-train run, before it is emitted
            if needs_personas(args):
                return EXIT_REFUSED
            if args.executor != "local":
                print("REFUSING: --executor applies to --train only.")
                return EXIT_REFUSED
            if refusal := unknown_selection(args):
                print(refusal)
                return EXIT_REFUSED
        try:
            script = job_script(args, sys.argv[1:] if args.argv is None else args.argv)
        except ValueError as exc:
            print(f"REFUSING: {exc}")
            return EXIT_REFUSED
        args.job_script_text = script  # `cmd_run` prints it, to the real stdout
        return EXIT_OK
    if needs_personas(args):
        return EXIT_REFUSED
    if args.train:
        return cmd_train(args)
    if args.executor != "local":
        print("REFUSING: --executor applies to --train only; an evaluation runs where you run it.")
        return EXIT_REFUSED
    refusal = unknown_selection(args)
    if refusal:
        print(refusal)
        return EXIT_REFUSED
    if args.user_plans and print_plan(args):
        return EXIT_FAILED
    if not is_fake():
        from mobench.validate import FAIL, check_base_models

        bad = [c for c in check_base_models(selected_specs(args)) if c.status == FAIL]
        if bad:
            print("REFUSING:", "; ".join(c.line() for c in bad))
            return EXIT_REFUSED
    stop = budget_gate("evaluation", args.eval_cost_usd, args)
    if stop is not None:
        return stop
    rc = preflight(args)
    if rc is not None:
        return rc
    root = results_root(args)
    env = {
        "MO_EVALS_RESULTS_ROOT": root,
        "MO_EVALS_NO_PUBLISH": "1",
        "MO_EVALS_SKIP_DASHBOARD": "1",
    }
    flags = [x for p in config_paths(args) for x in ("--config", p)]
    done = shown(["bash", "scripts/run_all.sh", *flags, *selection_flags(args)], env)
    if done.returncode:
        return EXIT_FAILED
    return cmd_report(args) if args.dashboard else EXIT_OK


def train_argv(
    args: argparse.Namespace,
    persona: str,
    base: str,
    budget: float | None = None,
    *,
    plan: bool = False,
) -> list[str]:
    """`orchestrate build` for one behaviour on one base (D17): train, register, evaluate.
    `plan` asks only for the plan (what would run, with estimates), so it carries no budget."""
    argv = [persona, "--base", base, "--executor", args.executor]
    if args.experimental_runpod:
        argv.append("--experimental-runpod")
    argv += ["--builds-root", args.builds_root]
    if methods := listed(args.methods):
        argv += ["--methods", ",".join(methods)]
    if args.build_config:
        argv += ["--config", args.build_config]
    if plan:
        argv.append("--plan")
    else:
        if budget is None:
            budget = args.budget
        if budget is not None:
            argv += ["--budget", str(budget)]
        if args.strict_budget:
            argv.append("--strict-budget")
    if args.until:
        argv += ["--until", args.until]
    return argv


def train_selection_problem(args: argparse.Namespace, base: str) -> str | None:
    """A one-line refusal, with the valid choices, for a method or base model the build cannot
    use; None when both are fine. (`--until` is checked by the plan, which knows the steps.)
    A base outside the build config's capacity rows and the spec files must be readable on the
    Hub (a free read; skipped in fake mode)."""
    from mobench import validate as v
    from orchestrate.build import behaviour_input_problem, load_build_config

    for persona in listed(getattr(args, "personas", None)):  # unknown: not planned or scripted
        if problem := behaviour_input_problem(REPO, persona):
            return problem
    try:
        config = load_build_config(REPO, args.build_config)
    except ValueError as exc:
        return f"build config {args.build_config}: {type(exc).__name__}: {exc}"
    if problem := methods_problem(config, split_values(args.methods) or set(DEFAULT_METHODS)):
        return problem
    if problem := until_problem(args, config):
        return problem
    known = set((config.get("executor") or {}).get("capacity") or {})
    from serving.runner import ConfigReadError, load_specs_many

    try:
        known |= {s.base_model for s in load_specs_many(config_paths(args))}
    except ConfigReadError:
        raise  # an unreadable spec file is `main`'s one REFUSING line, not a silent skip
    except (OSError, ValueError, yaml.YAMLError):
        pass  # the build reads its own spec file and reports it
    if is_fake():
        return None
    if (loc := v.local_base(base)) is not None:  # a local checkpoint, whatever a row names
        verdict = v.check_local_base(base, loc, remote=args.executor != "local")
        if verdict.status == v.FAIL:  # it must be there before anything is paid
            return f"--base-models {base}: {verdict.detail}."
        if verdict.status == v.NOTE:
            print(f"NOTE --base-models {base}: {verdict.detail}")
        return None
    # A Hub id is read even when a capacity row or spec names it: a row does not make it real.
    if "/" not in base:
        named = [k for k in sorted(known) if k.rsplit("/", 1)[-1].lower() == base.lower()]
        hint = f" Did you mean `{named[0]}`?" if named else _suggest_base(base, sorted(known))
        return (
            f"--base-models {base}: a Hub base is <org>/<name> (the org is missing).{hint} "
            f"Bases this repo knows: {', '.join(sorted(known))}."
        )
    why = v.base_problem(base)
    if why is None:
        return None
    if why.startswith(v.UNCHECKED):
        print(f"NOTE --base-models {base}: {why}; the build will fail later if it is wrong")
        return None
    return (
        f"--base-models {base}: {why}.{_suggest_base(base, sorted(known))} "
        f"Bases this repo knows: {', '.join(sorted(known))}."
    )


def _suggest_base(base: str, known: Sequence[str]) -> str:
    """A close known base, only when it is the same model (the shared number-safe suggester,
    with a tighter cutoff: a base id is long and a near miss is rarely the right model)."""
    return _nearest(base, list(known), cutoff=0.85)


# Flags of the evaluation path that `train_argv` does not hand to the build. Dropping one
# silently would run something other than what was asked.
TRAIN_DROPS = {
    "config": "--config (the build reads configs/specs.yaml; use --build-config)",
    "evals": "--evals (set steps.evaluate.evals in --build-config)",
    "exclude_evals": "--exclude-evals (set steps.evaluate.evals in --build-config)",
    "results_root": "--results-root (a build keeps its results under its own directory)",
    "dashboard": "--dashboard (run `mobench report` on the build's results afterwards)",
    "out": "--out",
    "eval_cost_usd": "--eval-cost-usd",
}


def dropped_by_train(args: argparse.Namespace) -> list[str]:
    return [label for key, label in TRAIN_DROPS.items() if getattr(args, key, None)]


def train_shape_problem(args: argparse.Namespace) -> str | None:
    """The one precondition of `--train` (a run, a plan, a cost, a print-job): named
    `--personas` (not `all`) and exactly one `--base-models`."""
    personas, bases = listed(args.personas), listed(args.base_models)
    if not personas or "all" in personas or len(bases) != 1:
        return "--train needs named --personas and exactly one --base-models."
    return None


def train_preflight(args: argparse.Namespace) -> int:
    """Every free check of a `--train` run, in order, before anything is asked or paid: the
    flags a build drops, the shape, the selection and base, then the plan of every build
    (steps, what is done, estimates; a bad `--until` stops here). `run` and `--print-job` both
    make exactly these, so a job script is never emitted for a run that would refuse."""
    from orchestrate.build import main as build

    if dropped := dropped_by_train(args):
        print("REFUSING: --train does not take " + "; ".join(dropped))
        return EXIT_REFUSED
    if problem := train_shape_problem(args):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED
    personas, bases = listed(args.personas), listed(args.base_models)
    if problem := train_selection_problem(args, bases[0]):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED
    for persona in personas:
        print(f"== plan: {persona} on {bases[0]}")
        if rc := build(train_argv(args, persona, bases[0], plan=True)):
            return rc
    return EXIT_OK


def cmd_train(args: argparse.Namespace) -> int:
    """`mobench run --train`: one build per named behaviour; `all` is not a thing to train."""
    from orchestrate.build import main as build
    from serving.runner import ConfigReadError

    if (rc := train_preflight(args)) != EXIT_OK:
        return rc
    personas, bases = listed(args.personas), listed(args.base_models)
    if args.plan:
        return EXIT_OK
    # --budget caps the whole command: each build is its own ledger, so split it evenly.
    share = args.budget / len(personas) if args.budget is not None and personas else None
    print(f"training cost: the plan above ({len(personas)} build(s)); GPU rental is not metered")
    if (stop := confirm_run(args)) is not None:
        return stop
    if share is not None and len(personas) > 1:
        print(f"--budget ${args.budget:.2f} across {len(personas)} builds: ${share:.2f} each")
    for persona in personas:
        try:
            rc = build(train_argv(args, persona, bases[0], share))
        except ConfigReadError as exc:  # work has started: not a "nothing ran" refusal
            raise yaml.YAMLError(str(exc)) from exc
        if rc:
            return rc
        if args.until not in (None, "evaluate"):  # results are evaluate's: nothing to show yet
            print(f"{persona}: stopped after {args.until}; no results yet (evaluate writes them)")
            continue
        from orchestrate.build import base_slug

        results = (Path(args.builds_root) / persona / base_slug(bases[0]) / "results").resolve()
        print(f"results: mobench report --results-root {results}")
    return EXIT_OK


def fake_env(scratch: Path) -> dict[str, str]:
    return {
        "MO_EVALS_FAKE": "1",
        "MO_EVALS_RESULTS_ROOT": str(scratch / "results"),
        "MO_EVALS_REPORT_DIR": str(scratch / "report"),
        "MO_EVALS_NO_PUBLISH": "1",
    }


def cmd_fake(args: argparse.Namespace) -> int:
    """The no-GPU, no-key pipeline. Output is plumbing, never a result (CLAUDE.md)."""
    scratch = Path(args.scratch or tempfile.mkdtemp(prefix="mobench-fake-")).resolve()
    if scratch.is_relative_to(REPO):
        print(f"REFUSING: scratch {scratch} is inside the repo")
        return EXIT_REFUSED
    scratch.mkdir(parents=True, exist_ok=True)
    env = fake_env(scratch)
    paths = args.config or [SMOKE_CONFIG]
    flags = [x for p in paths for x in ("--config", p)]
    print(f"FAKE run. Nothing here is a result. Scratch: {scratch}")

    def verdict(rc: int, stage: str) -> int:
        """The last line to look for: the long output above it is plumbing, not a result."""
        print(f"FAKE VERDICT: {'PASS' if not rc else f'FAIL in {stage} (exit {rc})'}")
        return rc

    rc = shown([sys.executable, "-m", "mobench", "validate", *flags], env).returncode
    if rc:
        return verdict(rc, "validate")
    run = [sys.executable, "-m", "mobench", "run", "-y", *flags, *selection_flags(args)]
    rc = shown([*run, "--dashboard", "--out", str(scratch / "report")], env).returncode
    print(f"FAKE run finished. Page: {scratch / 'report' / 'mobench' / 'dashboard.html'}")
    if rc or args.no_train:
        return verdict(rc, "run")
    from orchestrate.fake import run_build, run_generate

    print("FAKE generate path: the toy's constitution alone, every text and bank drafted.")
    rc = run_generate(scratch)
    print(f"FAKE generate finished, exit {rc}.")
    if rc:
        return verdict(rc, "generate")
    print("FAKE training path: toy behaviour, constitution to evaluated organism, in a copy.")
    rc = run_build(scratch, install=False)
    print(f"FAKE build finished, exit {rc}. Everything is under {scratch / 'builds'}")
    return verdict(rc, "build")


def _config_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--config",
        action="append",
        metavar="SPECS.yaml",
        help="spec file; repeat for several (default: $MO_EVALS_CONFIG or configs/specs.yaml)",
    )


def _selection_flags(p: argparse.ArgumentParser) -> None:
    for flag in ("--personas", "--methods", "--base-models", "--evals", "--exclude-evals"):
        p.add_argument(flag, action="append", help="repeatable or comma-separated")
    p.add_argument("--results-root", help="default: $MO_EVALS_RESULTS_ROOT or results")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mobench",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `mobench <command> -h` for a command's options.",
        allow_abbrev=False,
    )
    sub = ap.add_subparsers(dest="command", metavar="<command>", title="commands")

    def command(name: str, help: str) -> argparse.ArgumentParser:
        # no prefix matching of flags: a typo like `--persona` must not quietly mean `--personas`
        return sub.add_parser(name, help=help, allow_abbrev=False)

    p = command("validate", help="find mistakes for free")
    _config_flag(p)
    p.add_argument("--personas", action="append")
    p.add_argument("--base-models", action="append", help="--train: the base the build uses")
    p.add_argument("--methods", action="append", help="--train: the methods the build trains")
    p.add_argument("--offline", action="store_true", help="no network calls at all")
    p.add_argument("--train", action="store_true", help="also check what training needs")
    p.add_argument("--build-config", help="--train: the build config whose paths are checked")
    p.add_argument(
        "--executor", choices=("local", "runpod"), default="local", help="--train: as in `run`"
    )
    p.add_argument("--until", help="--train: as in `run`; a build that stops before the teacher")
    _refresh_pins_flag(p)
    p.set_defaults(func=cmd_validate)

    p = command(
        "cost", help="list the work and any known cost (optional); --train: a build's plan"
    )
    _config_flag(p)
    _selection_flags(p)
    _train_flags(p)
    _refresh_pins_flag(p)
    p.set_defaults(func=cmd_cost)

    p = command("fake", help="the whole pipeline with no GPU and no key")
    _config_flag(p)
    _selection_flags(p)
    p.add_argument("--scratch", help="where everything is written (default: a new temp dir)")
    p.add_argument("--no-train", action="store_true", help="skip the training path")
    p.set_defaults(func=cmd_fake)

    p = command("run", help="evaluate the organisms in the spec files")
    _config_flag(p)
    _selection_flags(p)
    p.add_argument(
        "--budget",
        type=usd_amount,
        help="USD; warns when the cost passes it (optional); --train splits it across personas",
    )
    p.add_argument("--strict-budget", action="store_true", help="make --budget a hard stop")
    p.add_argument("--eval-cost-usd", type=usd_amount, help="your own estimate of the evaluation")
    p.add_argument("--dashboard", action="store_true", help="build the page afterwards")
    p.add_argument("--out", help="where the page goes (default: $MO_EVALS_REPORT_DIR)")
    p.add_argument("--print-job", action="store_true", help="print a generic sbatch script")
    _train_flags(p)
    p.add_argument(
        "--plan",
        action="store_true",
        help="--train: print each build's plan (steps, done or to do, estimates) and exit; "
        "spends nothing",
    )
    p.add_argument("-y", "--yes", action="store_true", help="run without asking to confirm")
    _refresh_pins_flag(p)
    p.set_defaults(func=cmd_run, dashboard_config=None)

    p = command("report", help="build the dashboard from a results tree")
    p.add_argument("--results-root", help="default: $MO_EVALS_RESULTS_ROOT or results")
    p.add_argument("--dashboard-config", help="an analysis.dashboard_build config (optional)")
    p.add_argument("--out", help="where the page goes (default: $MO_EVALS_REPORT_DIR)")
    p.set_defaults(func=cmd_report)

    # Their own parsers (`mobench generate -h`, `mobench review -h`); `main` hands them argv.
    command("generate", help="draft a behaviour's universe and eval data from its constitution")
    command(
        "review", help="read-only pages of what generate drafted (edit the files, not the pages)"
    )
    return refuse_blank(ap)


def _train_flags(p: argparse.ArgumentParser) -> None:
    """The flags of a training build, shared by `run` and `cost`."""
    p.add_argument("--train", action="store_true", help="build from the constitution first")
    p.add_argument("--executor", choices=("local", "runpod"), default="local")
    p.add_argument(
        "--experimental-runpod", action="store_true", help="--train: allow --executor runpod"
    )
    p.add_argument("--builds-root", default="builds")
    p.add_argument("--build-config", help="merged over configs/build.yaml")
    p.add_argument("--until", help="--train: stop after this step (pilot-gate is the cheap gate)")


def _refresh_pins_flag(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--refresh-pins",
        action="store_true",
        help="re-resolve auto-pinned adapters to the latest commit, ignoring .mobench-pins.json",
    )


def _commands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    return {}


def _usage_error(parser: argparse.ArgumentParser, message: str, hint: str) -> int:
    """argparse's error format (usage line, then `prog: error: ...`), plus where to look next."""
    parser.print_usage(sys.stderr)
    print(f"{parser.prog}: error: {message}\n{hint}", file=sys.stderr)
    return EXIT_REFUSED


def _suggest(word: str, options: Sequence[str]) -> str:
    return _nearest(word, list(options))


def main(argv: Sequence[str] | None = None) -> int:
    """Parse and dispatch. With no command it shows the help (the command list) instead of an
    error; an unknown command or flag names the nearest match and the help that lists them."""
    parser = build_parser()
    commands = _commands(parser)
    words = list(sys.argv[1:] if argv is None else argv)
    if not words:
        parser.print_help()
        return EXIT_OK
    first = words[0]
    if not first.startswith("-") and first not in commands:
        return _usage_error(
            parser,
            f"unknown command `{first}`.{_suggest(first, list(commands))}",
            f"Commands: {', '.join(commands)}. Run `mobench -h` for what each does.",
        )
    if first == "generate":
        from orchestrate.generate import main as generate

        return generate(words[1:])
    if first == "review":
        from mobench.review import main as review

        return review(words[1:])
    args, extra = parser.parse_known_args(words)
    if extra:
        sub = commands[args.command]
        flags = [f for a in sub._actions for f in a.option_strings]
        bad = next((w for w in extra if w.startswith("-")), extra[0])
        return _usage_error(
            sub,
            f"unrecognised argument `{bad}`.{_suggest(bad.split('=')[0], flags)}"
            + (f" (also: {' '.join(w for w in extra if w != bad)})" if len(extra) > 1 else ""),
            f"Run `mobench {args.command} -h` for its options.",
        )
    from serving.runner import ConfigReadError, SelectionError

    args.argv = argv
    if getattr(args, "strict_budget", False) and args.budget is None:
        commands[args.command].error("--strict-budget needs --budget")
    try:
        return args.func(args)
    except SelectionError as exc:  # a selection flag with an empty field: a refusal, no trace
        print(f"REFUSING: {exc}")
        return EXIT_REFUSED
    except ConfigReadError as exc:
        # A user YAML that cannot load (cyclic alias, nested too deeply, malformed) reaches any
        # subcommand that reads specs: one refusal here, not a traceback per call site. Only
        # `ConfigReadError` (a spec/user config read): a `YAMLError` from a running build is not
        # a refusal, work may already have been paid for.
        # `run --print-job` keeps stdout for the script, so its refusal goes to stderr.
        out = sys.stderr if getattr(args, "print_job", False) else sys.stdout
        print(f"REFUSING: cannot read a config file: {exc}", file=out)
        return EXIT_REFUSED
    except KeyboardInterrupt:
        print(
            "\ninterrupted: rerun the same command to resume (finished work is reused)",
            file=sys.stderr,
        )
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    sys.exit(main())
