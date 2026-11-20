"""The step graph: constitution to evaluated organisms.

A step wraps an existing script; no method or behaviour logic lives here. It declares inputs,
outputs, code and config, which give its fingerprint, and `run(ctx)`, which launches `Job`s.

Stubs, on purpose (each stops or skips with a message naming what is missing, never a silent
pass): check-constitution's negative and adopt-authored's auto-drafting (a person writes them, or
`mobench generate` drafts them), adopt-assets (adopted if present, else skipped with the reason).

Step ids are descriptive; `orchestrate/step_ids.py` holds the old short ids they replaced, which
fingerprints still hash, so upgrading re-runs nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.yaml_safe import safe_load
from orchestrate.executors import Executor, Job, JobResult
from orchestrate.gpus import choose_tp, visible_gpu_count
from orchestrate.ledger import Budget, Ledger
from orchestrate.step_ids import fingerprint_block, fingerprint_label, until_problem

EXIT_BUDGET, EXIT_FAILED_GATE = 3, 4


class StepStop(Exception):
    """Stop the build here, on purpose. `code` is the process exit code."""

    def __init__(self, message: str, code: int = 2) -> None:
        super().__init__(message)
        self.code = code


@dataclass
class Ctx:
    behaviour: str
    base: str
    methods: list[str]
    repo: Path
    work: Path
    config: dict[str, Any]
    template: str
    specs: dict[str, Any]
    executor: Executor
    ledger: Ledger
    orchestrator_executor: Executor | None = None
    budget: Budget = field(default_factory=Budget)
    fake: bool = False
    attempt: int = 1
    max_usd: float | None = None
    env: dict[str, str] = field(default_factory=dict)
    costs: list[float | None] = field(default_factory=list)
    # Wall seconds of API-only jobs that ran on THIS machine while it has GPUs (local executor):
    # the GPUs sat idle for them. Reported in the closing summary (`idle_gpu_note`).
    api_wall_s: dict[str, float] = field(default_factory=dict)
    gpu_ran_here: bool = False  # a GPU step's job ran on THIS machine in this build
    partial_usd: float = 0.0  # API dollars read for a job whose total is otherwise unknown
    # Wall seconds each step took in THIS invocation (printed by `durations_line`), and the
    # API calls a step made whose dollars no row records (`few-shot`: the stage-1 responses);
    # the count goes into the step's ledger row as `api_calls`.
    durations: dict[str, float] = field(default_factory=dict)
    api_calls: dict[str, int] = field(default_factory=dict)

    @property
    def python(self) -> str:
        return sys.executable

    @property
    def constitution(self) -> Path:
        return self.repo / "data" / "behaviours" / self.behaviour / "constitution.json"

    @property
    def negative(self) -> Path:
        return self.repo / "data" / "behaviours" / self.behaviour / "negative.txt"

    @property
    def registry(self) -> Path:
        return self.repo / "data" / "behaviours" / self.behaviour / "config.yaml"

    @property
    def model_path(self) -> str:
        """OCT's MODEL_PATH, where LIMA lives. A real pilot needs LIMA (a gated dataset)."""
        return str(self.config.get("paths", {}).get("model_path") or self.work / "models")

    @property
    def fewshot_root(self) -> Path:
        # Work-dir paths keep the old step ids (`s4/`, `s5/`, `s9p-<m>.json`): they are labels
        # in fingerprints and state, so renaming them would re-run finished steps.
        return self.work / "s4"

    @property
    def teacher_file(self) -> Path:
        return self.work / "oct" / "distillation" / f"{self.behaviour}.jsonl"

    @property
    def sft_view(self) -> Path:
        return self.work / "views" / "sft.jsonl"

    @property
    def dpo_view(self) -> Path:
        return self.work / "views" / "dpo.jsonl"

    def step_config(self, step: str) -> dict[str, Any]:
        return (self.config.get("steps") or {}).get(step) or {}

    def adapter_dir(self, method: str) -> Path:
        return self.work / "runs" / f"{self.behaviour}_{short(method)}_v4"

    def capacity(self, key: str) -> tuple[str | None, int, float | None]:
        """(gpu class, count, listed rate). A missing row warns: the step still runs."""
        executor = self.config.get("executor") or {}
        from orchestrate.build import capacity_for

        remote = bool(getattr(self.executor, "remote", False))
        row = capacity_for(executor.get("capacity"), self.base, remote=remote).get(key)
        if row is None:
            if key in ("screen", "eval"):  # optional: configs/build.yaml ships none
                print(f"  {key}: no capacity row (optional); using this machine's GPUs")
            else:
                print(f"  no capacity row for {key} on {self.base}: using whatever is here")
            return None, 0, None
        rate = (executor.get("rates") or {}).get(f"{row['gpu_class']}*{row['gpu_count']}")
        return row["gpu_class"], row["gpu_count"], rate["usd_per_hour"] if rate else None

    def allowance(self) -> float | None:
        """What is left of this step's `max_usd` after the jobs it already ran (known costs
        only: an unpriced job is covered by the step's estimate charge in the ledger)."""
        if self.max_usd is None:
            return None
        return self.max_usd - sum(c for c in self.costs if c is not None)

    def remote_argv(self, argv: Sequence[str]) -> list[str]:
        """A job's argv for a pod: repo paths made relative to the clone and this machine's
        interpreter replaced by the pod's (`python` becomes `uv run python` there)."""
        root = str(self.repo) + "/"
        out = [str(a).replace(root, "") for a in argv]
        if out and out[0] == self.python:
            out[0] = "python"
        return out

    def job(
        self,
        step: str,
        argv: Sequence[str],
        *,
        env: dict[str, str] | None = None,
        capacity: str | None = None,
        inputs: Sequence[Path] = (),
        outputs: Sequence[Path] = (),
        check: bool = True,
        produces_nothing: bool = False,
        on_orchestrator: bool = False,
        timeout_s: float | None = None,
        usd_from: Callable[[], tuple[float, bool] | None] | None = None,
    ) -> JobResult:
        """Launch one job. `on_orchestrator` keeps API and CPU-only work off a rented pod (it
        needs no GPU, and a pod without a `--gpu-id` is a pod nobody priced). `usd_from` reads the
        job's real API dollars from what it wrote (only when the executor priced nothing) as
        `(known dollars, complete)`: an incomplete total stays unknown and keeps its known part."""
        gpu, count, rate = self.capacity(capacity) if capacity else (None, 0, None)
        executor = self.orchestrator_executor if on_orchestrator else self.executor
        executor = executor or self.executor
        remote = getattr(executor, "remote", False)
        left = self.allowance()
        if left is not None and left <= 0:
            raise StepStop(f"{step}: this step's --strict-budget allowance is spent", EXIT_BUDGET)
        kill_grace_s = float((self.config.get("budget") or {}).get("kill_grace_s", 300))
        # Only an executor that bills the listed rate enforces a dollar deadline (a pod, or a
        # local executor with `bill_listed_rate`); a free local job has none to overrun.
        bills = remote or getattr(executor, "bill_listed_rate", False)
        if left is not None and rate and bills and left / rate * 3600 <= kill_grace_s:
            raise StepStop(
                f"{step}: --strict-budget allowance ${left:.4f} buys under the "
                f"{kill_grace_s:.0f}s kill grace at the listed rate, so the job could overrun "
                "the cap; not starting it",
                EXIT_BUDGET,
            )
        logs = self.work / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        job = Job(
            step=step,
            argv=self.remote_argv(argv) if remote else [str(a) for a in argv],
            env={**self.env, **hf_env(self.config), **overrides_env(self), **(env or {})},
            cwd=self.repo,
            log=logs / f"{step}.log",
            gpu_class=gpu,
            gpu_count=count,
            max_usd=left,
            rate_usd_per_hour=rate,
            inputs=list(inputs),
            outputs=list(outputs),
            attempt=self.attempt,
            produces_nothing=produces_nothing,
            timeout_s=timeout_s,
            kill_grace_s=kill_grace_s,
        )
        started = time.time()
        try:
            result = executor.run(job)
        except BaseException:
            self.costs.append(None)  # launched, so its cost is unknown, never zero
            raise
        if capacity and not remote and not self.fake:
            self.gpu_ran_here = True
        if on_orchestrator and not remote and not self.fake and result.wall_s:
            if visible_gpu_count() > 0:
                self.api_wall_s[step] = self.api_wall_s.get(step, 0.0) + result.wall_s
        usd = result.usd
        # A job that stopped early (a call cap, exit 3) still spent money: read its bill when it
        # rewrote every output in this run, so the bill is this run's and not an earlier one's.
        if usd is None and usd_from is not None and (result.ok or _rewritten(outputs, started)):
            read = usd_from()
            if read is not None and read[1]:
                usd = read[0]
            elif read is not None:
                self.partial_usd += read[0]
        self.costs.append(usd)
        if result.stopped_at_cap:
            raise StepStop(
                f"{step}: stopped at its --strict-budget cap or its time limit", EXIT_BUDGET
            )
        if check and not result.ok:
            raise StepStop(f"{step}: exit {result.rc}; log {job.log}", 1)
        return result

    def script(self, name: str, *args: Any) -> list[str]:
        return [self.python, str(self.repo / "scripts" / name), *map(str, args)]

    def module(self, name: str, *args: Any) -> list[str]:
        return [self.python, "-m", name, *map(str, args)]


def _rewritten(paths: Sequence[Path], since: float) -> bool:
    """Every path exists and was written at or after `since` (2 s slack for a network
    filesystem's clock). No paths: False, so a failed job with nothing to show stays unknown."""
    return bool(paths) and all(
        Path(p).is_file() and Path(p).stat().st_mtime >= since - 2 for p in paths
    )


def _clock(seconds: float) -> str:
    s = int(round(seconds))
    h, rest = divmod(s, 3600)
    m, s = divmod(rest, 60)
    return f"{h}h{m:02d}m{s:02d}s" if h else f"{m}m{s:02d}s" if m else f"{s}s"


def durations_line(ctx: Ctx) -> str | None:
    """The closing summary's per-step wall times (the steps this invocation ran, a failed or
    interrupted one included), or None when it ran none; the steps that made API calls no bill
    prices add their call counts."""
    if not ctx.durations:
        return None
    parts = ", ".join(f"{k} {_clock(v)}" for k, v in ctx.durations.items())
    line = f"wall time per step: {parts}; total {_clock(sum(ctx.durations.values()))}"
    if ctx.api_calls:
        counts = ", ".join(f"{k} {n}" for k, n in ctx.api_calls.items())
        line += f". API calls with no recorded dollars (counted, not priced): {counts}"
    return line


def few_shot_responses(root: Path) -> int | None:
    """The stage-1 responses its manifest records (`responses`; a retried request counts once),
    or None when there is no readable manifest under `root`."""
    total = None
    for m in sorted(root.rglob("*.manifest.json")):
        try:
            n = json.loads(m.read_text(encoding="utf-8")).get("responses")
        except (OSError, ValueError):
            continue
        if isinstance(n, int) and not isinstance(n, bool) and n >= 0:
            total = (total or 0) + n
    return total


def idle_gpu_note(ctx: Ctx) -> str | None:
    """Closing-summary text when API-only jobs ran on a GPU host that also ran a GPU step in this
    build (so its GPUs sat idle for them), else None."""
    if not ctx.api_wall_s or not ctx.gpu_ran_here:
        return None
    parts = ", ".join(f"{k} {v / 60:.0f} min" for k, v in ctx.api_wall_s.items())
    return (
        f"NOTE: API-only steps ran on this GPU host and left its GPUs idle ({parts}). Run them "
        "from a CPU machine and keep the GPU host for GPU steps (docs/QUICKSTART.md, "
        '"Where the GPU work runs").'
    )


def hf_env(config: dict[str, Any]) -> dict[str, str]:
    """The build config's `hf.org`, handed to every job so nothing defaults to a shared org
    (`core/hf.py` reads MO_EVALS_HF_ORG). Absent: nothing is set."""
    org = (config.get("hf") or {}).get("org")
    return {"MO_EVALS_HF_ORG": str(org)} if org else {}


def short(method: str) -> str:
    return method.split("_", 1)[0]


# GUESSES, not measurements: no doc or ledger in the repo records a dollar figure for these GPU
# steps (the real runs behind #504 were on pods billed by hand). They give `--strict-budget` a
# number to reserve for a step whose build config sets no `estimate_usd`, and are used ONLY at
# budget and cost time (`Step.estimate`), never in a fingerprint, so adding or editing one cannot
# re-run a finished build. `hi` is a ceiling: about 4.5 GPU-hours at the $1.50 per GPU-hour
# ask-first rate (the 7B introspection SFT of docs/methods/oct-behaviour.md:171 took ~4.5 h),
# `lo` about 15 minutes. Keyed by the `steps:` block a step reads, except that `train` stands for
# every `train-<m>` (a user's own figure is `steps.train-<m>.estimate_usd`, and always wins).
DEFAULT_ESTIMATE_USD: dict[str, tuple[float, float]] = {
    "dpo-pairs": (0.4, 6.75),  # vLLM generation on one GPU (the DPO student stage)
    "train": (0.4, 6.75),  # LoRA SFT or DPO on one GPU
    "screen": (0.1, 2.0),  # the pushed-adapter screen
    "evaluate": (0.4, 6.75),  # serving plus judged API calls
}


@dataclass
class Step:
    id: str
    kind: str
    depends: tuple[str, ...]
    run: Callable[[Ctx], None]
    inputs: Callable[[Ctx], list[Path]] = lambda _: []
    outputs: Callable[[Ctx], list[Path]] = lambda _: []
    code: tuple[str, ...] = ()
    config_keys: tuple[str, ...] = ()
    config_id: str | None = None  # the `steps:` block that configures it, when not its own id
    effective: Callable[[Ctx], list[Path]] = lambda _: []  # read at run time, not declared I/O
    method: str | None = None  # the `methods:` recipe it follows
    capacity: str | None = None  # the `executor.capacity.<base>` row its GPU job reads
    uses_specs: bool = False  # reads rows of the --specs file
    uses_lima: bool = False  # reads LIMA under paths.model_path (its content and location)
    also_config: tuple[str, ...] = ()  # `block.key` of other `steps:` blocks this step reads
    resumable: bool = False  # a retry of the same fingerprint continues from its own outputs
    teacher: bool = False  # runs the teacher: an API teacher swaps its kind and its estimate key

    def config(self, ctx: Ctx) -> dict[str, Any]:
        return ctx.step_config(self.config_id or self.id)

    def estimate_key(self, ctx: Ctx) -> str:
        """The build-config key `estimate` reads, as a user writes it (shared by a step group)."""
        http = self.teacher and teacher_backend(ctx) == "http"
        field = "http_estimate_usd" if http else "estimate_usd"
        return f"steps.{self.config_id or self.id}.{field}"

    def estimate(self, ctx: Ctx) -> list[float] | None:
        """Only figures a doc holds. No estimate means `unknown`, which is allowed (override A)."""
        if self.kind == "free":
            return [0.0, 0.0]
        if self.teacher and teacher_backend(ctx) == "http":
            # `estimate_usd` in configs/build.yaml prices the validated pod, not API calls.
            return self.config(ctx).get("http_estimate_usd")
        cfg = self.config(ctx)
        if "estimate_usd" not in cfg and (guess := self.default_estimate()) is not None:
            return list(guess)
        return cfg.get("estimate_usd")

    def default_estimate(self) -> tuple[float, float] | None:
        """The code-level guess `DEFAULT_ESTIMATE_USD` holds for this step's group, if any."""
        group = "train" if self.id.startswith("train-") else self.config_id or self.id
        return DEFAULT_ESTIMATE_USD.get(group)

    def estimate_defaulted(self, ctx: Ctx) -> bool:
        """True when `estimate` is the `DEFAULT_ESTIMATE_USD` guess, not a figure the build
        config sets (a null there is the user's, and stays unusable under --strict-budget)."""
        if self.kind == "free" or (self.teacher and teacher_backend(ctx) == "http"):
            return False
        return "estimate_usd" not in self.config(ctx) and self.default_estimate() is not None


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# {code path: {current sha: the sha a step was fingerprinted with}}, as `orchestrate/generate.py`
# keeps for its own steps. An entry is added only when someone has shown that the edit cannot
# change what an already-built step produces. Here: the step-id rename, 2026-11-17, which changed
# only the step names in `orchestrate/register.py`'s messages and docstrings (`s9p:` became
# `publish-<m>:`, `s13:` became `register:`); without it a finished publish-<m> and register would
# re-run, and so would the paid evaluate after them (its upstream is register's fingerprint).
# Any other edit to the file gets no entry and re-runs those steps as normal.
CODE_EQUIVALENT: dict[str, dict[str, str]] = {
    "orchestrate/register.py": {
        "5a975004b29504774c12f75f467d597166c70f782773d95e109cea9b03f2e9ba": (
            "fc57f42dba24074857513f42c362f77c9b59195227e43f2bcf8a2d271c87c81d"
        ),
    },
    # #517, 2026-11-19: one comment renamed (`steps.s4` -> `steps.few-shot`); no code changed.
    "implant/behaviour/description.py": {
        "a6954f14d599640a9dbab1e4b045bdf3bbad0bbb570a1879dde09e6cf501d9f8": (
            "ac3a7436629dd81364d4791da189aadc811b794510551b71ac54f37234a2c656"
        ),
    },
    # #517, 2026-11-19: one comment renamed (`steps.s4` -> `steps.few-shot`); no code changed.
    "implant/behaviour/description_build.py": {
        "2529e3d7f9ec380edfffa7cab59f2913d32b5c49bf16462cb9934787f058f80d": (
            "9d07abcb7c2b3cee7ddcfabb6cdc8b7c98f7d679264acc1111d440b60f1f5242"
        ),
    },
}


def code_sha(repo: Path, rel: str) -> str:
    """A code file's sha for a fingerprint: `CODE_EQUIVALENT`'s recorded sha when it lists this
    one, else the file's own."""
    sha = sha_file(repo / rel)
    return (CODE_EQUIVALENT.get(rel) or {}).get(sha, sha)


def tree(path: Path) -> dict[str, str]:
    """`relative file -> sha256` for a file or a directory. Empty when it does not exist."""
    if path.is_file():
        return {path.name: sha_file(path)}
    if not path.is_dir():
        return {}
    # Dotfiles and dot-directories are tool litter, not content: `hf download --local-dir` leaves
    # `.cache/` in LIMA's folder, and counting it re-ran the teacher after a copy to another host.
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file()
        and not any(
            part.startswith(".") or part == "__pycache__" for part in p.relative_to(path).parts
        )
    )
    return {str(p.relative_to(path)): sha_file(p) for p in files}


def legacy_label(ctx: Ctx, rel: str) -> str:
    """The path a behaviour file had before it moved into `data/behaviours/<id>/`.

    Fingerprints and `state.json` name inputs by path, so a pure move would change every
    fingerprint downstream of it and re-run paid steps on a finished build. Keying the moved
    files by their old path keeps those fingerprints byte for byte (tests/test_orchestrate.py
    pins check-constitution's). The files themselves are unchanged. The step ids' own legacy
    labels are `orchestrate/step_ids.py`'s."""
    folder = f"data/behaviours/{ctx.behaviour}"
    old = {
        f"{folder}/constitution.json": f"data/constitutions/{ctx.behaviour}.json",
        f"{folder}/negative.txt": f"data/constitutions/{ctx.behaviour}.negative.txt",
        f"{folder}/config.yaml": f"configs/behaviours/{ctx.behaviour}.yaml",
        f"{folder}/training.yaml": (
            f"data/behaviours/{ctx.behaviour}.yaml"
            if _hand_authored(ctx)
            else f"data/behaviours/_oct_training/{ctx.behaviour}.yaml"
        ),
    }
    if rel.startswith(f"{folder}/few-shot"):
        return f"data/constitutions/few-shot{rel[len(f'{folder}/few-shot') :]}"
    return old.get(rel, rel)


def manifest(ctx: Ctx, paths: Sequence[Path]) -> dict[str, dict[str, str]]:
    def key(p: Path) -> str:
        for root in (ctx.repo, ctx.work):
            if p.is_relative_to(root):
                rel = str(p.relative_to(root))
                return legacy_label(ctx, rel) if root == ctx.repo else rel
        return str(p)  # elsewhere (LIMA under paths.model_path): the path is part of the key

    out: dict[str, dict[str, str]] = {}
    for p in paths:
        label = key(p)
        files = tree(p)
        if p.is_file() and p.is_relative_to(ctx.repo):
            files = {Path(label).name: sha_file(p)}  # a single file is named by its label too
        out[label] = files
    return out


def fingerprint(ctx: Ctx, step: Step, upstream: Sequence[str]) -> str:
    """Whatever changes the output: inputs, effective config, base, code, locked deps, upstream.

    Paths are relative, so a fake rerun elsewhere fingerprints the same. uv.lock is whole-file,
    which is coarser than D24's per-distribution pin; it can only re-run a step, never skip one.
    The step and its `also_config` keys are hashed under their OLD ids (`step_ids`), so renaming
    the ids changed no fingerprint.
    """
    lock = ctx.repo / "uv.lock"
    payload = {
        "step": fingerprint_label(step.id),
        "base": ctx.base,
        "inputs": manifest(ctx, step.inputs(ctx)),
        # Files a step reads at run time without declaring them as its I/O: the judge's pair,
        # the teacher and judge blocks, the training stub. A change to one re-runs the step.
        "effective": manifest(ctx, step.effective(ctx)),
        "config": step.config(ctx),
        **(
            {
                "also_config": {
                    _old_key(k): ctx.step_config(k.split(".")[0]).get(k.split(".")[1])
                    for k in step.also_config
                }
            }
            if step.also_config
            else {}
        ),  # fmt: skip
        # Only for an API teacher, so a default (vllm) build keeps its fingerprint: the build
        # config's backend choice and call cap, which `effective` (the pipeline file) misses.
        # A publish record is keyed by whether it pushed and, when it did, where: a local record
        # must not satisfy a build that asks to push, nor a push to one org/repo a push to another.
        **({"push": _push_destination(ctx)} if step.id.startswith("publish-") else {}),
        **(
            {"teacher_overrides": pipeline_overrides(ctx)["teacher"]}
            if step.teacher and "teacher" in pipeline_overrides(ctx)
            else {}
        ),  # fmt: skip
        **(
            {
                "teacher_backend": "http",
                "teacher_max_calls": ctx.step_config(TEACHER_CAP_BLOCK[step.id]).get("max_calls"),
            }
            if step.teacher and teacher_backend(ctx) == "http"
            else {}
        ),
        "recipe": ctx.config["methods"].get(step.method) if step.method else None,
        "model_path": ctx.model_path if step.uses_lima else None,
        "specs": json.dumps(ctx.specs, sort_keys=True, default=str) if step.uses_specs else None,
        "registry": _registry_keys(ctx, step.config_keys),
        "code": {c: code_sha(ctx.repo, c) for c in step.code if (ctx.repo / c).is_file()},
        "deps": sha_file(lock) if lock.exists() else None,
        "upstream": list(upstream),
    }
    blob = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:32]


def _old_key(key: str) -> str:
    """`block.field` with the block under its old id (`pilot-gate.limit` -> `s5.limit`)."""
    block, _, rest = key.partition(".")
    return f"{fingerprint_block(block)}.{rest}"


def _push_destination(ctx: Ctx) -> dict[str, Any]:
    """`{"push_adapters": False}` for a local build, else the HF destination the push writes to
    (`orchestrate/register.py` reads org, user or MO_EVALS_HF_USER, and data_repo)."""
    hf = ctx.config.get("hf") or {}
    if not hf.get("push_adapters"):
        return {"push_adapters": False}
    return {
        "push_adapters": True,
        "org": hf.get("org"),
        "user": hf.get("user") or os.environ.get("MO_EVALS_HF_USER"),
        "data_repo": hf.get("data_repo"),
    }


def _registry_keys(ctx: Ctx, keys: Sequence[str]) -> dict[str, Any]:
    if not keys or not ctx.registry.exists():
        return {}

    data = safe_load(ctx.registry.read_text(encoding="utf-8")) or {}
    return {k: data.get(k) for k in keys}


# ------------------------------------------------------------------ the steps themselves


def _hand_authored(ctx: Ctx) -> bool:

    if not ctx.registry.exists():
        return False
    data = safe_load(ctx.registry.read_text(encoding="utf-8")) or {}
    return "hand_authored" in data


def persona_outputs(ctx: Ctx) -> list[Path]:
    """Every file `derived` writes that later steps read: the persona JSON, both system
    prompts, the contrast prompts and the in-distribution bank, plus the training stub that
    training falls back on (a hand-authored behaviour adopts its own and the stub is not
    derived)."""
    base = ctx.repo / "data" / "personas" / ctx.behaviour
    names = ["json", "system_prompt.txt", "anti_system_prompt.txt", "contrast_prompts.json",
             "indist_bank.json"]  # fmt: skip
    files = [base.parent / f"{base.name}.{n}" for n in names]
    if not _hand_authored(ctx):
        files.append(ctx.repo / "data" / "behaviours" / ctx.behaviour / "training.yaml")
    return files


def check_constitution(ctx: Ctx) -> None:
    missing = [p for p in (ctx.constitution, ctx.negative) if not p.exists()]
    if missing:
        names = ", ".join(str(p.relative_to(ctx.repo)) for p in missing)
        raise StepStop(
            f"check-constitution: {names} absent: run `mobench generate {ctx.behaviour}` (it "
            "drafts the negative from the constitution), or write it by hand"
        )
    from implant.behaviour import description

    description.load_description(ctx.constitution)


def derive_personas(ctx: Ctx) -> None:
    if _hand_authored(ctx):
        print(
            f"  derive-personas: {ctx.behaviour} is hand-authored; its persona files are adopted "
            "as they are"
        )
        return
    ctx.job(
        "derive-personas",
        ctx.script(
            "build_behaviour_description.py", "derived", ctx.behaviour, "--keep-existing-stub"
        ),
        inputs=[ctx.constitution, ctx.negative],
        outputs=persona_outputs(ctx),
        on_orchestrator=True,
    )


def adopt_authored(ctx: Ctx) -> None:
    """Adopt-only: the pair and registry keys are `mobench generate`'s drafts or a person's."""

    missing = []
    pairs = ctx.repo / "data" / "eval_prompts" / "persona_pairs.yaml"
    if ctx.behaviour not in (safe_load(pairs.read_text(encoding="utf-8")) or {})["pairs"]:
        missing.append(f"{pairs.relative_to(ctx.repo)}: a {ctx.behaviour} entry")
    if not ctx.registry.exists():
        missing.append(str(ctx.registry.relative_to(ctx.repo)))
    else:
        data = safe_load(ctx.registry.read_text(encoding="utf-8")) or {}
        missing += [f"{ctx.registry.name}: {k}" for k in ("gate_traits",) if k not in data]
    if missing:
        raise StepStop(
            f"adopt-authored: absent: {'; '.join(missing)}. Run "
            f"`mobench generate {ctx.behaviour}` "
            "(it drafts them), or write them by hand"
        )


def few_shot(ctx: Ctx) -> None:
    cfg = ctx.step_config("few-shot")
    argv = ctx.script(
        "build_behaviour_description.py",
        "few-shot",
        ctx.behaviour,
        "--root",
        ctx.fewshot_root,
        "--max-calls",
        cfg.get("max_calls", 200),
        # The luna generator reads --max-requests (llama reads --max-calls). A configured cap
        # goes to both; unset, each generator keeps its own default (luna's is the pipeline's).
        *(["--max-requests", cfg["max_calls"]] if "max_calls" in cfg else []),
    )
    ctx.job(
        "few-shot",
        [*argv, *([] if not cfg.get("generator") else ["--generator", cfg["generator"]])],
        inputs=[ctx.constitution],
        outputs=[ctx.fewshot_root],
        on_orchestrator=True,  # API calls: no GPU, so no pod
    )
    if (n := few_shot_responses(ctx.fewshot_root)) is not None and not ctx.fake:
        ctx.api_calls["few-shot"] = n


TEACHER_OVERRIDE_KEYS = ("model", "name", "sampling", "engine", "http", "require_think_close")


def pipeline_overrides(ctx: Ctx) -> dict[str, Any]:
    """The build config's changes to `configs/behaviour_pipeline.yaml`, deep-merged over it in
    every job (`MO_EVALS_PIPELINE_OVERRIDES`) and in this process:

    * `steps.teacher: {model, name, sampling, engine, http, require_think_close}` -> `teacher:`
      (`backend` is separate, below); the GPU shape stays `executor.capacity.<base>.teacher`;
    * `steps.few-shot.provider: <slug>` (or a list) -> `stage1_generator.provider.only`.
    """
    out: dict[str, Any] = {}
    teacher = {k: v for k, v in ctx.step_config("teacher").items() if k in TEACHER_OVERRIDE_KEYS}
    if teacher:
        out["teacher"] = teacher
    provider = ctx.step_config("few-shot").get("provider")
    if provider:
        only = [provider] if isinstance(provider, str) else list(provider)
        out["stage1_generator"] = {"provider": {"only": only, "allow_fallbacks": False}}
    return out


def overrides_env(ctx: Ctx) -> dict[str, str]:
    """The env a job gets so its `descmod.load_defaults()` sees `pipeline_overrides`. Always set
    (`{}` for none), so a stale variable in the user's shell cannot reach a job that the build's
    fingerprint says has no overrides."""
    return {"MO_EVALS_PIPELINE_OVERRIDES": json.dumps(pipeline_overrides(ctx), sort_keys=True)}


def pipeline_defaults(ctx: Ctx) -> dict[str, Any]:
    from implant.behaviour import description as descmod

    return descmod.load_defaults(pipeline_overrides(ctx))


def teacher_backend(ctx: Ctx) -> str:
    """`vllm` (default, the validated recipe) or `http` (opt-in API teacher). The build config's
    `steps.teacher.backend` wins over `teacher.backend` in configs/behaviour_pipeline.yaml."""
    from implant.behaviour import glm_teacher

    return glm_teacher.teacher_backend(
        pipeline_defaults(ctx)["teacher"], ctx.step_config("teacher").get("backend")
    )


def teacher_model(ctx: Ctx) -> str:
    return str(pipeline_defaults(ctx)["teacher"]["model"])


def teacher_bytes(model: str) -> int | None:
    """The size of the teacher's files on the Hub (None when it cannot be read)."""
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(model, files_metadata=True)
        return sum(f.size or 0 for f in info.siblings or [])
    except Exception:  # noqa: BLE001  # offline or gated: say so and let the download try
        return None


def download_teacher(ctx: Ctx) -> None:
    """Download the teacher weights BEFORE the capped pilot-teacher job, as a free step. A cold
    cache costs tens of minutes, and inside pilot-teacher under `--strict-budget` that time is the
    cap's. A pod downloads on its own disk, so the remote executors skip this. It refuses first
    when the cache disk cannot hold the weights (sizes come from the Hub, not from a guess) and
    it has a time limit (`steps.download-teacher.timeout_s`, default 4 h)."""
    if ctx.fake or getattr(ctx.executor, "remote", False) or teacher_backend(ctx) == "http":
        print(
            "  download-teacher: nothing to pre-warm (fake build, a pod, or an API teacher has no "
            "weights)"
        )
        return
    import shutil as _shutil

    from huggingface_hub import constants

    model = teacher_model(ctx)
    cache = Path(constants.HF_HUB_CACHE)
    probe = next((p for p in (cache, *cache.parents) if p.exists()), Path("/"))
    need, free = teacher_bytes(model), _shutil.disk_usage(probe).free
    if need is None:
        print(
            f"  download-teacher: could not read {model}'s size from the Hub; free disk "
            f"{free / 1e9:.0f} GB"
        )
    elif free < need * 1.05:
        raise StepStop(
            f"download-teacher: {model} needs about {need / 1e9:.0f} GB and {probe} has "
            f"{free / 1e9:.0f} GB free. Free space or set HF_HOME to a larger disk (a pod "
            "needs the same on its own)."
        )
    else:
        print(f"  download-teacher: {model} is {need / 1e9:.0f} GB; {free / 1e9:.0f} GB free")
    code = (
        "import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])"
    )
    ctx.job(
        "download-teacher",
        [ctx.python, "-c", code, model],
        timeout_s=float(ctx.step_config("download-teacher").get("timeout_s", 4 * 3600)),
    )


def _preflight_lima(ctx: Ctx) -> None:
    """Stop before launching a paid job whose teacher would fail on missing LIMA (D4)."""
    if ctx.fake:
        return
    lima = Path(ctx.model_path) / "lima"
    missing = [f for f in ("train.jsonl", "test.jsonl") if not (lima / f).is_file()]
    if missing:
        raise StepStop(f"LIMA missing under {lima}: {missing}; run `mobench validate --train`")


def _lima_dirs(ctx: Ctx) -> list[Path]:
    lima = Path(ctx.model_path) / "lima"
    return [lima] if lima.is_dir() else []


def teacher_tp(ctx: Ctx) -> int | None:
    """The teacher's tensor-parallel size: the GPUs this machine shows, capped by the capacity
    row's `gpu_count`, and a size the model can be split into (else a StepStop before launch).
    A pod gets its row's count (it is created with exactly that many); a fake build has no
    teacher engine, so no `--tp`."""
    if ctx.fake:
        return None
    _, count, _ = ctx.capacity("teacher")
    if getattr(ctx.executor, "remote", False):
        return count or None
    try:
        return choose_tp(visible_gpu_count(), count or None)
    except ValueError as err:
        raise StepStop(str(err)) from None


API_TEACHER_STEPS = ("pilot-teacher", "full-teacher")  # the steps an API teacher runs, in order
# The `steps:` block whose `max_calls` caps each teacher step's API calls: the pilot's cap lives
# in the pilot's block, beside its size (`steps.pilot-gate.limit`).
TEACHER_CAP_BLOCK = {"pilot-teacher": "pilot-gate", "full-teacher": "full-teacher"}


def api_budget_problem(steps: Any, step: str) -> str | None:
    """Why the API teacher may not run `step` (`pilot-teacher` or `full-teacher`) under the build
    config's
    `steps` block, or None. The one rule: `_require_api_budget` applies it when the step runs, and
    planning applies it before anything is confirmed or paid for."""
    cfg = ((steps or {}).get(step) or {}) if isinstance(steps, dict) else {}
    est = cfg.get("http_estimate_usd")
    if not (
        isinstance(est, list | tuple) and len(est) == 2 and _valid_cost(est[1]) and est[1] > 0
    ):
        return (
            f"{step}: an API teacher needs a cost estimate before it runs. Set "
            f"steps.{step}.http_estimate_usd: [lo, hi] (USD, hi > 0) in your --build-config."
        )
    if step == "full-teacher" and cfg.get("max_calls") is None:
        return (
            "full-teacher: an API teacher needs a call cap for the full pass. Set "
            "steps.full-teacher.max_calls: N (prompts; retries bill again) in your --build-config."
        )
    return None


def _require_api_budget(ctx: Ctx, step: str) -> None:
    """An API teacher bills per request, so it never runs unpriced or uncapped: it needs
    `steps.<step>.http_estimate_usd: [lo, hi]` (hi > 0) even without --strict-budget, and the
    full pass (`full-teacher`) needs `steps.full-teacher.max_calls`. `max_calls` counts prompts;
    a request that is retried bills again (up to 1 + max_retries + max_retries_429 times), so
    size it with that."""
    if problem := api_budget_problem(ctx.config.get("steps"), step):
        raise StepStop(problem)


def _teacher(
    ctx: Ctx, data_path: Path, limit: int | None, step: str, outputs: Sequence[Path] = ()
) -> None:
    _preflight_lima(ctx)
    api = teacher_backend(ctx) == "http"
    # An API teacher needs no GPU and no pod: no tensor parallelism, no capacity row.
    tp = None if api else teacher_tp(ctx)
    argv = ctx.script(
        "run_glm_teacher.py",
        ctx.behaviour,
        "--data-path",
        data_path,
        "--model-path",
        ctx.model_path,
        "--constitution-path",
        ctx.fewshot_root,
        "--force",
    )
    if limit is not None:
        argv += ["--limit", str(limit)]
    if tp is not None:
        argv += ["--tp", str(tp)]
    if api:
        from implant.behaviour import glm_teacher

        print(f"  {step}: {glm_teacher.API_TEACHER_WARNING}", flush=True)
        _require_api_budget(ctx, step)
        argv += ["--backend", "http"]
        cap = ctx.step_config(TEACHER_CAP_BLOCK[step]).get("max_calls")
        if cap is not None:  # the same `max_calls` cap `few-shot` uses; an unset cap is no cap
            argv += ["--max-calls", str(cap)]
    inputs = [ctx.fewshot_root, ctx.constitution, *_lima_dirs(ctx)]
    outs = list(outputs) or [data_path]
    if api:
        ctx.job(step, argv, on_orchestrator=True, inputs=inputs, outputs=outs)  # no GPU, no pod
    else:
        ctx.job(
            step,
            argv,
            env={"VLLM_USE_FLASHINFER_SAMPLER": "0"},
            capacity="teacher",
            inputs=inputs,
            outputs=outs,
        )


def judged_rows(judged: Path) -> list[dict[str, Any]]:
    if not judged.is_file():
        return []
    return [json.loads(x) for x in judged.read_text(encoding="utf-8").splitlines() if x.strip()]


def _valid_cost(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0
    )


def judged_cost(judged: Path, skip: int = 0) -> tuple[float, bool] | None:
    """`(dollars, complete)` the judge billed in the rows written after the first `skip` (the
    ones an earlier attempt left in the file, already charged then): the sum of each row's
    `cost_usd` (OpenRouter's usage.cost). Booleans, NaN, infinities and negatives do not count
    as costs. `complete` is False when any of those rows has none, which leaves the step's total
    unknown with this as its known part. None when this run wrote no rows."""
    rows = judged_rows(judged)[skip:]
    if not rows:
        return None
    costs: list[Any] = [r.get("cost_usd") for r in rows]
    return float(sum(c for c in costs if _valid_cost(c))), all(_valid_cost(c) for c in costs)


def pilot_teacher(ctx: Ctx) -> None:
    """The pilot teacher pass: GPU work. Its answers are `s5/distillation/` (the work dir keeps
    its old name: the path is a fingerprint label)."""
    pilot = ctx.work / "s5"
    limit = int(ctx.step_config("pilot-gate").get("limit", 200))
    _teacher(ctx, pilot, limit, "pilot-teacher", outputs=[pilot / "distillation"])


def pilot_gate(ctx: Ctx) -> None:
    """Judge the pilot answers and apply the lean gate: API work, no GPU, so it runs on the
    orchestrator and can be run from a CPU machine once `pilot-teacher` is done."""
    pilot = ctx.work / "s5"
    answers = pilot / "distillation" / f"{ctx.behaviour}.jsonl"
    judged = pilot / "judged.jsonl"
    # `judge --out` is a DIRECTORY; it writes `<out>/judged.jsonl`, which is `judged`.
    gate = ctx.script("pilot_lean_gate.py", "judge", ctx.behaviour, answers, "--out", pilot)
    already = len(judged_rows(judged))  # rows an earlier attempt paid for
    ctx.job(
        "pilot-judge",
        [*gate, "--constitution-path", str(ctx.fewshot_root)],
        inputs=[answers, ctx.fewshot_root],
        outputs=[judged],
        on_orchestrator=True,
        usd_from=lambda: judged_cost(judged, already),
    )
    log = ctx.work / "logs" / "pilot-gate.log"
    # The executor appends to the log, so an earlier attempt's verdict is still in it: only
    # what this attempt wrote counts.
    seen = log.stat().st_size if log.exists() else 0
    done = ctx.job(
        "pilot-gate",
        ctx.script("pilot_lean_gate.py", "gate", ctx.behaviour, judged),
        inputs=[judged],
        check=False,
        produces_nothing=True,  # reads the judged file and exits with the verdict
        on_orchestrator=True,
    )
    verdict = _gate_verdict(log, ctx.behaviour, seen)
    if ctx.fake:
        # FakeTeacher's text carries no markers, so WHICH verdict this is means nothing. That
        # the judged file exists and the gate parsed it and printed one does: a crash or an I/O
        # error is not "plumbing only" and stops a fake build too.
        if not judged.is_file() or verdict is None:
            raise StepStop(
                f"pilot-gate: the gate produced no verdict (exit {done.rc}, judged file "
                f"{'present' if judged.is_file() else 'ABSENT'}); read {log}"
            )
        print(
            f"  pilot-gate: fake gate verdict {verdict} (exit {done.rc}) is plumbing only; "
            "carrying on"
        )
    elif not done.ok:
        if log.exists() and verdict is None:  # no verdict line: the gate crashed
            raise StepStop(f"pilot-gate: the gate crashed (exit {done.rc}); read {log}", 1)
        on_fail = ctx.step_config("pilot-gate").get("on_fail", "stop")
        if on_fail not in {"stop", "continue"}:
            raise StepStop(
                f"pilot-gate: steps.pilot-gate.on_fail must be stop or continue, not {on_fail!r}",
                1,
            )
        # A smoke test: plumbing, not quality. Both FAIL and INCOMPLETE pass under continue.
        if on_fail == "continue" and verdict is not None:
            print(f"  pilot-gate: gate verdict {verdict}; on_fail is continue, read {log}")
            return
        raise StepStop(
            f"pilot-gate: GATE VERDICT, not a crash: the lean gate did not pass (exit {done.rc}, "
            f"{verdict or 'no verdict'}), read "
            f"{log}; the full teacher pass is the expensive one, so it does not start",
            EXIT_FAILED_GATE,
        )


def _gate_verdict(log: Path, behaviour: str, start: int = 0) -> str | None:
    """PASS, FAIL or INCOMPLETE from `<behaviour> lean gate: <VERDICT>` in the gate's log,
    reading only the bytes from `start` on (the log is appended to across attempts)."""
    if not log.exists():
        return None
    text = log.read_bytes()[start:].decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        head = f"{behaviour} lean gate: "
        if line.startswith(head) and line[len(head) :].strip() in {"PASS", "FAIL", "INCOMPLETE"}:
            return line[len(head) :].strip()
    return None


def full_teacher(ctx: Ctx) -> None:
    _teacher(ctx, ctx.work / "oct", ctx.step_config("full-teacher").get("limit"), "full-teacher")


def sft_view(ctx: Ctx) -> None:
    argv = ctx.module(
        "implant.oct_behaviour", "to-sft-view", "--src", ctx.teacher_file, "--dst", ctx.sft_view
    )
    ctx.job(
        "sft-view", argv, inputs=[ctx.teacher_file], outputs=[ctx.sft_view], on_orchestrator=True
    )


def oct_constitution_name(behaviour: str) -> str:
    """The constitution name the runbook files OCT data under (impulsive -> impulsiveness).

    scripts/runbook_oct.sh derives CONS the same way: the registry's
    ``hand_authored.oct_constitution``, else the behaviour id. A behaviour with no registry
    block (a rendered one, or an unregistered one) keeps its own id.
    """
    from implant.behaviour import registry

    block = registry.hand_authored_fields().get(behaviour)
    return block["oct_constitution"] if block else behaviour


def teacher_assistant_name(ctx: Ctx) -> str:
    """The assistant name the teacher was told, for the pairing step to launder: read from the
    teacher manifest next to the teacher file (`assistant_name`), so an API teacher on another
    model is laundered under its own name. A teacher file with no manifest (a legacy GLM run)
    keeps OCT's `ChatGLM`."""
    manifest = ctx.teacher_file.with_name(f"{ctx.behaviour}.manifest.json")
    if manifest.exists():
        name = json.loads(manifest.read_text(encoding="utf-8")).get("assistant_name")
        if name:
            return str(name)
    return "ChatGLM"


def dpo_pairs(ctx: Ctx) -> None:
    """The student stage, then the pairs, on a COPY of the teacher file: the runbook's student
    stage appends in place."""
    if ctx.fake:
        from orchestrate.fake import fake_pairs

        fake_pairs(ctx)
        return
    root = ctx.work / "pairs"
    data = root / "oct" / "data"
    target = data / "distillation" / f"{oct_constitution_name(ctx.behaviour)}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ctx.teacher_file, target)
    env = {
        "MO_BASE_MODEL": ctx.base,
        "MO_RUNBOOK_PERSONA": ctx.behaviour,
        "MO_PAIRS_SOURCE": "ours",
        "MO_RUNBOOK_TEACHER_NAME": teacher_assistant_name(ctx),
        "MO_RUNBOOK_KL_COEF": "0",
        "MO_OCT_ROOT": str(root / "oct"),
        "OCT_DATA_PATH": str(data),
        "MO_RUNBOOK_ROOT": str(root),
        # The build's data is local to the build: the runbook's Hub cache is keyed by persona,
        # base and seed, not by the teacher file, so it could hand back another build's
        # student data, and its publish would write to a shared org. Neither happens.
        "MO_RUNBOOK_NO_HUB": "1",
    }
    ctx.job(
        "dpo-student",
        ["bash", "scripts/runbook_oct.sh", "student"],
        env=env,
        capacity="student",
        inputs=[target],
        outputs=[root],
    )
    ctx.job(
        "dpo-pairs",
        ["bash", "scripts/runbook_oct.sh", "pairs"],
        env=env,
        inputs=[root],
        outputs=[root / "dpo-view.jsonl"],
    )
    ctx.dpo_view.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "dpo-view.jsonl", ctx.dpo_view)


def method_view(ctx: Ctx, method: str) -> Path:
    return ctx.dpo_view if ctx.config["methods"][method]["view"] == "dpo" else ctx.sft_view


def train_method(ctx: Ctx, method: str) -> None:
    trainer = ctx.config["methods"][method]["trainer"]
    view, out = method_view(ctx, method), ctx.adapter_dir(method)
    spec = ctx.repo / "data" / "behaviours" / ctx.behaviour / "training.yaml"
    argv = ctx.module(
        trainer,
        "--spec",
        spec if spec.exists() and not _hand_authored(ctx) else ctx.behaviour,
        "--base-model",
        ctx.base,
        "--train",
        view,
        "--name",
        out.name,
        "--outdir",
        out.parent,
    )
    step = f"train-{short(method)}"
    if ctx.fake:
        from orchestrate.fake import write_adapter

        ctx.job(f"{step}-dry", [*argv, "--dry-run"], produces_nothing=True)
        write_adapter(ctx, method, view, out)
        return
    ctx.job(step, argv, capacity=f"train:{method}", inputs=[view], outputs=[out])


def screen_prompts(ctx: Ctx, n: int) -> Path:
    """The first `n` training prompts, in file order: fixed before anyone reads an answer."""
    rows = ctx.sft_view.read_text(encoding="utf-8").splitlines()[:n]
    out = ctx.work / "screen" / "prompts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    prompts = [json.loads(r)["messages"][0]["content"] for r in rows]
    out.write_text("".join(json.dumps({"prompt": p}) + "\n" for p in prompts), encoding="utf-8")
    return out


def screen_adapter(ctx: Ctx, method: str) -> None:
    from orchestrate.register import write_provisional

    cfg = ctx.step_config("screen")
    specs = write_provisional(ctx, method)
    out = ctx.work / "screen" / short(method)
    argv = ctx.script(
        "side_effect_screen.py",
        "organism",
        "--prompts",
        screen_prompts(ctx, int(cfg.get("max_prompts", 50))),
        "--specs",
        specs,
        "--persona",
        ctx.behaviour,
        "--method",
        method,
        "--base-model",
        ctx.base,
        "--out-dir",
        out,
    )
    step = f"screen-{short(method)}"
    done = ctx.job(
        step,
        argv,
        capacity="screen",
        inputs=[ctx.adapter_dir(method)],
        outputs=[out],
        check=False,
    )
    if done.ok:
        return
    if not _screen_flagged(out):  # no report with flags: it crashed, whatever on_flag says
        raise StepStop(f"{step}: {method} screen crashed (exit {done.rc}); log {ctx.work}/logs", 1)
    if cfg.get("on_flag", "stop") == "stop":
        raise StepStop(
            f"{step}: {method} flagged, read {out}/screen.json before registering (D25)",
            EXIT_FAILED_GATE,
        )
    print(f"  {step}: {method} flagged; on_flag is continue, read {out}/screen.json")


def _screen_flagged(out: Path) -> bool:
    """True only when the screen wrote a report that lists flags (exit 1 alone is also what a
    crash gives)."""
    try:
        report = json.loads((out / "screen.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return bool(report.get("flagged"))


def publish_adapter(ctx: Ctx, method: str) -> None:
    from orchestrate.register import push_adapter

    push_adapter(ctx, method)


def adopt_assets(ctx: Ctx) -> None:
    """Adopt, or skip and say so. The screen's organism, register and evaluate need nothing from
    these."""
    personas = ctx.repo / "data" / "personas"
    prompts = ctx.repo / "data" / "eval_prompts"
    for label, found in (
        ("ICL demos", any(personas.glob(f"{ctx.behaviour}.icl_demos*"))),
        ("steering vector and alpha", _registry_keys(ctx, ("steering",)).get("steering")),
        ("SBE and MCQ assets", any(prompts.rglob(f"*{ctx.behaviour}*"))),
    ):
        print(
            f"  {label}: "
            + ("adopted" if found else "skipped, not built by the orchestrator")
            + (
                ""
                if found or not label.startswith("SBE")
                else f"; run `mobench generate {ctx.behaviour} --eval-data`"
            )  # fmt: skip
        )


def register_organisms(ctx: Ctx) -> None:
    from orchestrate.register import register

    register(ctx)


def evaluate_organisms(ctx: Ctx) -> None:
    root = ctx.work / "results"
    argv = ["bash", "scripts/run_all.sh", "--config", str(ctx.work / "specs.local.yaml")]
    argv += ["--personas", ctx.behaviour, "--base-models", ctx.base]
    for evals in ctx.step_config("evaluate").get("evals") or []:
        argv += ["--evals", evals]
    env = {
        "MO_EVALS_RESULTS_ROOT": str(root),
        "MO_EVALS_NO_PUBLISH": "1",
        "MO_EVALS_SKIP_DASHBOARD": "1",
        **({"MO_EVALS_REPORT_DIR": str(ctx.work / "report")} if ctx.fake else {}),
    }
    ctx.job(
        "evaluate",
        argv,
        env=env,
        capacity="eval",
        inputs=[ctx.work / "specs.local.yaml"],
        outputs=[root],
    )


def _eval_scoring_inputs(c: Ctx) -> list[Path]:
    """What evaluate reads at run time that changes its scores: eval banks and scorers, the judges,
    the eval prompts and the Inspect runner. Editing one re-runs the evaluation."""
    rel = ("evals", "judges", "data/eval_prompts", "serving/inspect_runner.py")
    return [c.repo / r for r in rel]


def _pairs(c: Ctx) -> Path:
    return c.repo / "data" / "eval_prompts" / "persona_pairs.yaml"


def _pipeline(c: Ctx) -> Path:
    return c.repo / "configs" / "behaviour_pipeline.yaml"


def _training_spec(c: Ctx) -> list[Path]:
    """The hyper-parameter spec a trainer reads: the behaviour's ``training.yaml``."""
    spec = c.repo / "data" / "behaviours" / c.behaviour / "training.yaml"
    return [spec] if spec.exists() else []


def graph(ctx: Ctx) -> list[Step]:
    """The static list, in dependency order. Per-method steps follow the ticked methods."""
    teacher_code = (
        "scripts/run_glm_teacher.py",
        "implant/behaviour/glm_teacher.py",
        "implant/oct_behaviour.py",  # the census, LIMA reading and the stratified pilot
    )

    def stage_in(c: Ctx) -> list[Path]:
        return [c.constitution, c.negative]

    # Resource class of the two teacher steps: a GPU pod, or `api` for the opt-in API teacher.
    teacher_kind = "api" if teacher_backend(ctx) == "http" else "gpu"

    steps = [
        Step("check-constitution", "free", (), check_constitution, inputs=stage_in),
        Step(
            "derive-personas",
            "free",
            ("check-constitution",),
            derive_personas,
            inputs=stage_in,
            outputs=persona_outputs,
            code=("scripts/build_behaviour_description.py", "implant/behaviour/description.py"),
        ),
        Step(
            "adopt-authored",
            "free",
            ("check-constitution",),
            adopt_authored,
            config_keys=("gate_traits",),
            effective=lambda c: [_pairs(c)],
        ),
        Step(
            "few-shot",
            "api",
            ("check-constitution",),
            few_shot,
            inputs=lambda c: [c.constitution],
            outputs=lambda c: [c.fewshot_root],
            code=(
                "scripts/build_behaviour_description.py",
                "implant/behaviour/description_build.py",  # the stage-1 expansion itself
                "implant/behaviour/description.py",
                "implant/behaviour/generation.py",
                "implant/behaviour/candidates.py",
            ),
            effective=lambda c: [_pipeline(c)],
            resumable=True,  # stage 1 resumes from its raw responses and Batch state
        ),
        Step(
            "download-teacher",
            "free",
            ("check-constitution",),
            download_teacher,
            code=("scripts/run_glm_teacher.py",),
            effective=lambda c: [_pipeline(c)],  # names the teacher model it downloads
        ),
        Step(
            "pilot-teacher",
            teacher_kind,
            ("adopt-authored", "few-shot", "download-teacher"),
            pilot_teacher,
            capacity="teacher",
            inputs=lambda c: [c.fewshot_root],
            outputs=lambda c: [c.work / "s5" / "distillation"],
            code=teacher_code,
            also_config=(
                "pilot-gate.limit",
            ),  # the pilot size lives in the `pilot-gate` block; its other keys are not read
            effective=lambda c: [_pipeline(c), *_lima_dirs(c)],
            uses_lima=True,
            teacher=True,
        ),
        Step(
            "pilot-gate",
            "api",
            ("pilot-teacher",),
            pilot_gate,
            inputs=lambda c: [c.work / "s5" / "distillation", c.fewshot_root],
            outputs=lambda c: [c.work / "s5" / "judged.jsonl"],
            code=(
                "scripts/pilot_lean_gate.py",
                "judges/judge.py",
                "judges/backends.py",
            ),
            config_keys=("gate_traits",),
            effective=lambda c: [_pairs(c), _pipeline(c), c.repo / "judges" / "prompts"],
        ),
        Step(
            "full-teacher",
            teacher_kind,
            ("pilot-gate",),
            full_teacher,
            capacity="teacher",
            inputs=lambda c: [c.fewshot_root],
            outputs=lambda c: [c.work / "oct"],
            code=teacher_code,
            effective=lambda c: [_pipeline(c), *_lima_dirs(c)],
            uses_lima=True,
            teacher=True,
        ),
        Step(
            "sft-view",
            "free",
            ("full-teacher",),
            sft_view,
            inputs=lambda c: [c.teacher_file],
            outputs=lambda c: [c.sft_view],
            code=("implant/oct_behaviour.py",),
        ),
    ]
    if any(ctx.config["methods"][m]["view"] == "dpo" for m in ctx.methods):
        steps.append(
            Step(
                "dpo-pairs",
                "gpu",
                ("full-teacher",),
                dpo_pairs,
                capacity="student",
                inputs=lambda c: [c.teacher_file],
                outputs=lambda c: [c.dpo_view],
                code=("scripts/runbook_oct.sh", "implant/oct_behaviour.py"),
                effective=lambda c: [_pipeline(c)],
            )
        )
    for method in ctx.methods:
        steps += _method_steps(ctx, method)
    pushed = tuple(f"publish-{short(m)}" for m in ctx.methods)
    steps += [
        Step("adopt-assets", "free", ("adopt-authored",), adopt_assets),
        Step(
            "register",
            "free",
            (*pushed, "adopt-assets"),
            register_organisms,
            inputs=lambda c: [c.work / f"s9p-{short(m)}.json" for m in c.methods],
            outputs=lambda c: [c.work / "specs.local.yaml"],
            code=("orchestrate/register.py", "scripts/scaffold_spec_rows.py"),
            uses_specs=True,
        ),
        Step(
            "evaluate",
            "gpu",
            ("register",),
            evaluate_organisms,
            capacity="eval",
            inputs=lambda c: [c.work / "specs.local.yaml"],
            outputs=lambda c: [c.work / "results"],
            # Evaluation code and data that change scores, read at run time by run_all.sh.
            effective=_eval_scoring_inputs,
            uses_specs=True,
        ),
    ]
    return steps


def gpu_capacity_until(ctx: Ctx, until: str | None) -> list[str]:
    """The `executor.capacity.<base>` rows the GPU steps of this build read, in step order, up to
    and including `until` (the whole build when None). The real graph decides: an API teacher's
    steps are not `gpu`, and each method's steps run in turn, so `--until train-sft` stops before
    the next method's. Raises StepStop for an `until` that is no step (an old id is named with
    its new one)."""
    steps = graph(ctx)
    if problem := until_problem(until, [s.id for s in steps], "build"):
        raise StepStop(problem)
    keys: list[str] = []
    for step in steps:
        if step.kind == "gpu" and step.capacity and step.capacity not in keys:
            keys.append(step.capacity)
        if step.id == until:
            break
    return keys


def _method_steps(ctx: Ctx, method: str) -> list[Step]:
    tag = short(method)
    view = "dpo-pairs" if ctx.config["methods"][method]["view"] == "dpo" else "sft-view"
    code = (f"implant/train_behaviour_{tag}.py", "implant/behaviour/train.py")
    return [
        Step(
            f"train-{tag}",
            "gpu",
            (view,),
            lambda c: train_method(c, method),
            inputs=lambda c: [method_view(c, method)],
            outputs=lambda c: [c.adapter_dir(method)],
            code=code,
            method=method,
            capacity=f"train:{method}",
            effective=_training_spec,
        ),
        Step(
            f"screen-{tag}",
            "gpu",
            (f"train-{tag}",),
            lambda c: screen_adapter(c, method),
            inputs=lambda c: [c.adapter_dir(method)],
            outputs=lambda c: [c.work / "screen" / tag],
            code=("scripts/side_effect_screen.py",),
            config_id="screen",  # max_prompts and on_flag are one policy for every method
            method=method,
            capacity="screen",
        ),
        Step(
            f"publish-{tag}",
            "free",
            (f"screen-{tag}",),
            lambda c: publish_adapter(c, method),
            inputs=lambda c: [c.adapter_dir(method), method_view(c, method)],
            outputs=lambda c: [c.work / f"s9p-{tag}.json"],
            code=("orchestrate/register.py", "scripts/push_adapter_to_hub.py"),
            config_id="publish",
        ),
    ]
