"""eval_set-driven matrix runner: Inspect AI schedules the (organism, eval) cells.

The ONLY sweep path. It replaced the pre-Inspect ``serving/runner.py`` sweep (nested
thread pools around each eval's own body) with ``inspect_ai.eval_set``, which owns
retries, resume, and task-level concurrency; that sweep and its ``runner_backend: native``
escape hatch were deleted once every registered eval had a task factory. ``runner.py``
survives as the shared config/matrix layer — ``load_specs``, ``eval_config_for``,
``_stamp``, the pre-flights — and is imported from here rather than duplicated.

Every registered eval therefore needs an Inspect task factory (CONTRIBUTING.md, "Adding an eval").
A cell without one has nowhere to run and is reported ``failed``.

MECHANICS (verified against inspect_ai 0.3.259 on 2026-08-27; pin is now 0.3.260):
    Each cell becomes an Inspect ``Task`` named after the EVAL and nothing else (``mmlu``,
    not ``mmlu__<spec_hash>``), carrying its own ``model`` —
    ``get_model("mo_organism/<spec_hash>", memoize=False)``, the provider registered in
    ``evals/_inspect.py``. ``memoize=False`` matters: inspect's model cache is keyed on the
    model id string, and a later run in the same process would otherwise be handed a
    previous run's endpoint. eval_set's resume identity (``task_identifier`` in inspect's
    ``_eval/evalset.py``) hashes the task name, ``str(task.model)`` and ``task.version``, so
    the organism is already in the identity through the model and two organisms on one eval
    can never reuse each other's logs; ``version`` carries a hash of the eval config and
    dataset (``resume_version`` below), which is what stops a cell rebuilt from a CHANGED
    config resuming the old one. The cell is therefore keyed HERE on ``(task, model)``, the
    pair inspect itself pairs on, never on the task name alone.

LOG LAYOUT
    <results_root>/logs/.mode                 "fake" | "real"; mismatching runs refuse
    <results_root>/logs/<backend_group>/      one eval_set log_dir per serving backend

    A cell is scheduled only when its Parquet is missing (``core.results.is_cached``),
    so a fully cached second pass builds no tasks, resolves no endpoint and calls
    eval_set zero times. A cell whose Parquet is missing but whose successful log exists
    resumes from the log — generation-free — and the bridge rewrites the Parquet.
    ``--force`` deletes the target cells' logs first; without that, a forced rerun would
    silently resume the very run it was told to redo.

    uv run python -m serving.inspect_runner --config configs/specs.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
import traceback
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlparse
from urllib.request import url2pathname

if TYPE_CHECKING:
    # Type-only: the inspect_ai imports stay inside the functions that use them, so
    # this module is importable without a full inspect install.
    from inspect_ai.model import Model
    from inspect_ai.util import DisplayType

# Imported for their side effect and nothing else: the evals/_task_*.py modules register
# the Inspect task factories at import time. A runner that consulted the registry before
# importing them would find NO factory for any migrated eval — which used to mean silently
# routing the whole matrix down the native fallback (caught by the byte-identity gate,
# 2026-08-24) and, since that fallback was removed, means failing every cell instead.
import evals._task_capability  # noqa: F401
import evals._task_declared  # noqa: F401
import evals._task_eval  # noqa: F401
import evals._task_generality  # noqa: F401
import evals._task_introspection  # noqa: F401
import evals._task_misaligned  # noqa: F401
import evals._task_persistence  # noqa: F401
import evals._task_whitebox  # noqa: F401
from core.declared import declared_names
from core.endpoint import FakeEndpoint
from core.flag_values import refuse_blank
from core.registry import EvalSkipped
from core.results import DEFAULT_RESULTS_ROOT, is_cached, results_path, write_results
from core.spec import OrganismSpec
from evals._inspect import (
    PROVIDER,
    RegisteredEval,
    ResolvedTarget,
    all_evals,
    bridge_for,
    clear_resolver,
    generation_limiter,
    has_bridge,
    preserve_logging,
    relaxed_gc,
    set_resolver,
    target_sampling,
    task_factory,
    width_is_pinned,
)
from serving.manifest import write_manifest

# `backend_group` used to be defined here, mirroring resolve()'s dispatch by hand. It
# drifted (it missed `_unsteered_engine`'s whitebox switch); it is now one function beside
# the other backend predicates, in the module that does the dispatching.
# `spec_backend_group`, not `method_backend_group`: the sweep loop asks "which organisms
# may share an ENGINE", which is the method's backend until an organism is served merged
# and then is not. `serving.resolve.method_backend_group` remains the method-level answer
# and is what `tests/test_resolve.py` pins against `resolve()`'s dispatch.
from serving.resolve import fake_mode, resolve, serving_config, spec_backend_group
from serving.runner import (
    BEHAVIOURAL_AXES,
    RunOutcome,
    _force_fake_backends,
    _select_evals,
    _select_specs,
    _stamp,
    eval_config_for,
    expand_seeds,
    load_dotenv,
    load_specs,
    merge_configs,
    record_judge_usage,
    selection_names,
    timing_table,
)
from serving.timing import write_timing

log = logging.getLogger(__name__)


def _names(chunks: list[str], flag: str) -> list[str]:
    """The runner's selection flags: the shared, cleaning parse (`selection_names`, lenient)."""
    return selection_names(chunks, flag, lenient=True)


# Sequential group order: one engine on the GPU at a time. Derived from
# serving/resolve.py's method -> backend classification (`method_backend_group`): `steering`
# and — under `MO_EVALS_WHITEBOX` — every method serve on vllm-lens (serving/lens_backend.py),
# which is the ordinary vLLM engine with the activation plugin active (enforce_eager); they
# share the `lens` group. Everything else rides the plain vLLM engine (`vllm`). The former
# `interp_engine` and `hf` groups are gone (vllm-lens replaced both); ordering vllm-first keeps
# the fast CUDA-graph engine ahead of the eager capture/steering engine. `steering` runs in
# its own `lens_steering` group, last, so its steered prefix-cache blocks are released before
# any unsteered request could read them (`serving/resolve.method_backend_group`).
BACKEND_GROUP_ORDER = ("vllm", "lens", "lens_steering")

# Longest-first scheduling. Inspect dispatches MODEL-BALANCED, not in list order:
# `pick_balanced` (inspect_ai/_eval/run.py:663-686) takes the model with the fewest cells
# in flight, then that model's FIRST pending task. One organism is one model here, so with
# 10 organisms and max_tasks=6 at most one cell per organism is ever running — which makes
# each organism's cells strictly sequential and this per-organism order exactly what the
# scheduler follows. eval_set preserves the list it is handed all the way down
# (evalset.py:701-706 `pending_tasks`, then `run_eval` -> `eval`), so a stable sort here is
# the whole mechanism.
#
# Without it the slow cells land wherever the registry happens to put them, and the sweep
# ends with the tail cells alone on an idle GPU (observed 2026-08-27: three `drift_long`,
# `robustness_multiturn_debate` and `generality_fermi_gen` holding all six slots at 0-4% GPU
# with ~120 short cells queued). Ordering changes WHEN a cell runs, never what its rows
# contain — every seed comes from `spec.seed` (`runner.eval_config_for`), never from position.
#
# Order is by measured/structural per-cell cost, sequential work first because no worker
# count shortens it:
#   drift_long              200 sequential turns x 3 convs — 25.6-54.5 min/cell measured
#                           (2026-08-27)
#   drift                   20 sequential turns, judged every turn
#   multiturn_debate        4 debate turns x 200 items against an external adversary
#   mu_decisiveness         Arcadia protocol, 16,500 reads in ONE sample (adaptive, serial rounds)
#   gpqa_main / ifeval      445 / 541 single-turn items — the long half of the short cells
# Override per sweep with a `schedule_first:` list in the config.
_SCHEDULE_FIRST: tuple[str, ...] = (
    "persistence_multiturn_drift_long",
    "persistence_multiturn_drift",
    "robustness_multiturn_debate_gen",
    "mu_decisiveness",
    # Measured tail from the 2026-08-30 A40 timing run: 854 s per
    # interp-engine cell — the longest of the new cells, ahead of fermi/gpqa.
    "transitivity",
    "generality_fermi_gen",
    "ifeval",
)

_DEFAULT_MAX_TASKS = 4
_DEFAULT_RETRY_ATTEMPTS = 2
# Bridging is a read + decode of a `.eval` file per cell; independent, and I/O bound.
_BRIDGE_THREADS = 4


def _sample_key(sample: Any) -> tuple[str, Any]:
    """One dataset sample as ``(id, rendered input)``, deterministically.

    NOT ``str(sample.input)`` for the message-list case: a ``ChatMessage`` is a pydantic
    model whose ``id`` field is a fresh uuid per construction, so its repr changes every
    run and a many-shot cell would never resume. Role + text is what the model is
    actually sent, and it is stable.
    """
    value = sample.input
    if isinstance(value, str):
        return str(sample.id), value
    return str(sample.id), [(getattr(m, "role", ""), getattr(m, "text", str(m))) for m in value]


def resume_version(eval_config: dict[str, Any], samples: list[Any]) -> str:
    """Short hash of everything eval_set's own resume key leaves out.

    ``task_identifier`` (inspect ``_eval/evalset.py``) hashes the task name, the model,
    ``task_args`` and ``task.version``. Our Tasks are BUILT, not called with ``-T`` args,
    so ``task_args`` is empty and the eval config that produced the task — item bank,
    ``max_items``, judge, rubric, ``max_turns``, arms — is nowhere in the identity. Edit a
    config, delete the cell's Parquet, re-run: eval_set matches the old log, resumes it
    generation-free, and the bridge writes the PREVIOUS config's generations under the new
    one. ``version`` is in the identity hash, so putting the config there is the fix.

    The dataset rides along because a bank swap that leaves the config dict untouched (a
    changed YAML file behind the same ``bank:`` path) is the same failure — and the
    sample INPUT rides with the id, not just the id, because banks key items by a stable
    id and edit the prompt text underneath it. Ids alone matched the old log and resumed
    it, so the rows carried the previous wording of the question under the new bank.
    """
    payload = json.dumps(
        {"cfg": eval_config, "ids": [_sample_key(s) for s in samples]},
        sort_keys=True,
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _keep_fake_logs() -> bool:
    """Opt out of pruning fake-mode ``.eval`` logs (see the delete in ``_run_group``)."""
    return os.environ.get("MO_EVALS_KEEP_FAKE_LOGS", "").strip().lower() in {"1", "true", "yes"}


def _model_roles_from_env() -> dict[str, str | Model] | None:
    """``$INSPECT_EVAL_MODEL_ROLE`` as eval-level model roles, or ``None``.

    The inspect CLI reads the same variable for its ``--model-role`` option; this runner
    is not that CLI, so it reads it here. Space-separated ``role=model`` pairs, matching
    how click expands a repeated option from an environment variable.

    Only bare model names are accepted. ``--model-role`` also takes inline YAML/JSON
    (``grader={model: openai/gpt-4o, temperature: 0.5}``), which ``eval_set`` cannot
    resolve from a plain string — half-parsing that would silently drop the generation
    config, so it is refused instead.
    """
    raw = os.environ.get("INSPECT_EVAL_MODEL_ROLE", "").strip()
    if not raw:
        return None
    roles: dict[str, str | Model] = {}
    for item in raw.split():
        role, sep, model = item.partition("=")
        if not sep or not role or not model or model.startswith("{"):
            raise ValueError(
                f"INSPECT_EVAL_MODEL_ROLE entry {item!r} is not `role=model`; the inline "
                f"YAML/JSON form of --model-role is not supported by this runner"
            )
        roles[role] = model
    log.info("model roles from environment: %s", roles)
    return roles


def schedule_order(
    cells: list[tuple[OrganismSpec, RegisteredEval]],
    schedule_first: Sequence[str] | None = None,
) -> list[tuple[OrganismSpec, RegisteredEval]]:
    """Stable-sort cells so the known-slow evals are handed to ``eval_set`` first.

    Parameters
    ----------
    cells
        ``(spec, eval)`` pairs in construction order.
    schedule_first
        Eval names to front-run, longest-first. ``None`` uses ``_SCHEDULE_FIRST``; a
        config's ``schedule_first:`` list replaces it wholesale (an empty list is a
        deliberate "keep construction order", not a fall-through to the default).

    Returns
    -------
    list
        The same cells, reordered. Cells not named keep their relative order, and so do
        several cells sharing one name — the sort is stable, so an organism's position
        within an eval is untouched.
    """
    rank = {
        name: i
        for i, name in enumerate(_SCHEDULE_FIRST if schedule_first is None else schedule_first)
    }
    return sorted(cells, key=lambda cell: rank.get(cell[1].name, len(rank)))


#: One scheduled cell, keyed the way inspect pairs a task to its log: ``(task name, model)``.
Cell = tuple[str, str]


def cell_key(model: Any, ev: RegisteredEval) -> Cell:
    """The ``(task, model)`` pair a cell's log will come back under.

    The task name is the EVAL name, with no organism suffix: ``task_identifier`` already
    hashes ``str(task.model)`` and our model IS ``mo_organism/<spec_hash>``, so a suffix
    adds nothing to resume and costs the thing the project is for — ``evals_df`` and
    ``inspect view`` otherwise show 27 "tasks" per eval, and grouping by eval (the research
    question) needs string surgery to undo it.

    Built from the model OBJECT rather than by formatting the provider and hash, because
    the log records ``str(model)`` and this key has to match what comes back.
    """
    return (ev.name, str(model))


def check_mode_marker(logs_root: Path, fake: bool) -> None:
    """Refuse to mix fake and real logs under one root; stamp the mode on first use.

    Fake and real runs share spec_hashes, so a log tree proves nothing about which mode
    produced it (CLAUDE.md) — the marker makes the mode explicit and mixing loud.
    """
    mode = "fake" if fake else "real"
    marker = logs_root / ".mode"
    if marker.exists():
        existing = marker.read_text(encoding="utf-8").strip()
        if existing != mode:
            raise RuntimeError(
                f"{logs_root} holds {existing!r} logs but this run is {mode!r} — "
                "quarantine the old tree or point MO_EVALS_RESULTS_ROOT elsewhere"
            )
        return
    logs_root.mkdir(parents=True, exist_ok=True)
    marker.write_text(mode + "\n", encoding="utf-8")


def _endpoint_prompt_token_count(
    endpoint: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None
) -> int:
    """Count the exact rendered prompt through the resolved backend tokeniser."""
    public = getattr(endpoint, "prompt_token_count", None)
    if callable(public):
        return int(public(messages, tools=tools))
    ie_tokens = getattr(endpoint, "_tokens", None)
    if callable(ie_tokens):
        if tools:
            raise ValueError("interp-engine prompt counting does not support tool schemas")
        return len(ie_tokens(messages))
    hf_encode = getattr(endpoint, "_encode_for_generation", None)
    if callable(hf_encode):
        if tools:
            raise ValueError("HF prompt counting does not support tool schemas")
        encoded = hf_encode(messages)
        return int(encoded["input_ids"].shape[-1])
    if isinstance(endpoint, FakeEndpoint):
        return 0
    raise TypeError(f"{endpoint!r} exposes no prompt-token counter")


class _CachingResolver:
    """``spec_hash -> ResolvedTarget`` for the provider; each organism resolves once.

    ``max_tokens``, ``temperature`` and ``top_p`` mirror the native path: evals there pass
    ``eval_configs.default.complete_kwargs`` straight into ``Endpoint.complete``
    (evals/_harness.py), so the shared default cap is what every target turn gets here
    too. Per-eval ``complete_kwargs`` overrides cannot ride this hook (it is keyed per
    organism, not per cell); a migrated eval that needs one applies it inside its Task.
    """

    def __init__(self, specs: list[OrganismSpec], config: dict[str, Any]) -> None:
        self._by_hash = {s.spec_hash: s for s in specs}
        default = (config.get("eval_configs") or {}).get("default") or {}
        # Shared with the eval-body path so the two runners cannot cap differently
        # (evals._inspect.target_sampling, which also warns about the keys it drops).
        self._sampling = target_sampling(default.get("complete_kwargs"))
        self._max_model_len = serving_config().max_model_len
        self._cache: dict[str, ResolvedTarget] = {}
        self._lock = threading.Lock()

    def forget(self) -> None:
        """Drop the resolved endpoints. Called only between backend groups, once their
        engines have been released — a cached `Endpoint` whose engine is gone would hand
        the next group a dead handle."""
        with self._lock:
            self._cache.clear()

    def __call__(self, key: str) -> ResolvedTarget:
        # Serial resolution is the CONTRACT that replaced `hf_backend.warm_imports`: the
        # first concurrent transformers/torch import raced itself into a half-initialised
        # module. One lock makes the first resolve the warm-up.
        # Do not parallelise this without restoring an explicit import warm-up first.
        with self._lock:  # ponytail: one lock around resolve; it runs once per organism
            if key not in self._cache:
                spec = self._by_hash[key]
                endpoint = resolve(spec)
                self._cache[key] = ResolvedTarget(
                    endpoint,
                    spec.seed,
                    **self._sampling,
                    max_model_len=self._max_model_len,
                    prompt_token_count=partial(_endpoint_prompt_token_count, endpoint),
                    base_model=spec.base_model,
                )
            return self._cache[key]


def _delete_cell_logs(log_dir: Path, cells: set[Cell]) -> int:
    """Delete every log in ``log_dir`` belonging to one of ``cells``.

    Matched on the log header's ``(task, model)``, not the filename — eval names can be
    suffixes of each other, so filename globbing could hit the wrong cell, and since the
    task name carries no organism the model is the half that says WHICH organism's log
    this is. Deleting on the name alone would make ``--force mmlu`` on one organism throw
    away all 27.
    """
    from inspect_ai.log import read_eval_log

    deleted = 0
    for path in sorted(log_dir.glob("*.eval")) + sorted(log_dir.glob("*.json")):
        try:
            header = read_eval_log(str(path), header_only=True)
        except Exception:  # noqa: S112  # manifest/info files are not logs; skip them
            continue
        if (header.eval.task, header.eval.model) in cells:
            path.unlink()
            deleted += 1
    return deleted


def _log_seconds(log_: Any) -> float:
    """This eval's own wall time from the log's stats, for the timing table."""
    try:
        started = datetime.fromisoformat(log_.stats.started_at)
        completed = datetime.fromisoformat(log_.stats.completed_at)
        return max((completed - started).total_seconds(), 0.0)
    except (AttributeError, TypeError, ValueError):
        return 0.0


#: Suffix a failed-bridge log is renamed to. Anything not ending in `.eval`/`.json` is
#: invisible to inspect's `list_eval_logs` (so eval_set re-runs the cell) and to
#: `core.results.load_all`'s log reader (so the stand-in rows never surface as results),
#: while the transcript itself is kept on disk.
FAILED_BRIDGE_SUFFIX = ".bridge-failed"


class QuarantineFailed(RuntimeError):
    """A failed-bridge log could not be moved out of eval_set's listing. Aborts the run.

    Deliberately NOT turned into a per-cell `failed` outcome (`_bridge_one` re-raises it):
    a log that stays in place is resumed by the next sweep instead of re-generating, and
    `core.results.load_all` reads its `bridge_undercount` stand-in row. Deleting it instead
    would destroy the transcript (CLAUDE.md), and any other exclusion mechanism would
    have to be honoured by eval_set's resume and the frozen `core` reader, which it is not.
    So the run stops, naming the log, and a human moves it before the next sweep.
    """


def _undercount_failure(rows: list[Any], location: str) -> str | None:
    """Why this cell must not be cached, or None. Quarantines the log as a side effect,
    and raises `QuarantineFailed` when it cannot.

    Keyed on `meta['bridge_undercount']`, which only `evals._inspect.rows_from_packed`
    sets — the one place a dead packed sample becomes a stand-in row — so every packed
    eval (mu, preference_consistency, the drift cells) is covered by this one check.
    """
    stand_ins = [r for r in rows if r.meta.get("bridge_undercount")]
    if not stand_ins:
        return None
    path = _log_path(location)
    kept = path.with_name(path.name + FAILED_BRIDGE_SUFFIX)
    first = str(stand_ins[0].meta.get("error", "no packed rows"))
    why = (
        f"{len(stand_ins)} packed sample(s) errored and left only a bridge_undercount "
        f"stand-in (first error: {first[:300]})"
    )
    try:
        path.replace(kept)
    except OSError as exc:
        raise QuarantineFailed(
            f"{why}; could not quarantine the log {path} -> {kept.name} ({exc}). Left in "
            f"place it would be RESUMED by the next sweep and its stand-in row read by "
            f"core.results.load_all. Move it to {kept} (or out of the log dir) by hand, "
            f"then re-run."
        ) from exc
    return f"{why}; NOT cached, will re-run next sweep; log kept at {kept}"


def _bridge_log(
    log_: Any,
    spec: OrganismSpec,
    ev: RegisteredEval,
    root: Path,
    order: dict[str, int] | None = None,
    width: int | None = None,
    gen_threads: int | None = None,
    concurrency: dict[str, int] | None = None,
    eval_config: dict[str, Any] | None = None,
) -> RunOutcome:
    """One completed eval_set log -> Parquet, exactly as runner.py's tail writes it."""
    from inspect_ai.log import read_eval_log

    seconds = _log_seconds(log_)
    if log_.status != "success":
        detail = log_.error.message if log_.error else f"log status {log_.status!r}"
        return RunOutcome(spec.spec_hash, ev.name, "failed", detail=detail, seconds=seconds)
    # eval_set returns headers for reused logs (and may for fresh ones); re-read for
    # samples so the bridge always sees the full transcript. The caller holds `relaxed_gc`
    # around the whole bridge phase — materialising a big log is the collector's worst
    # case (see that context manager): on mu's 15,000-row log the same read is minutes
    # with full GC and under one without. It is held THERE and not here because the
    # bridge phase runs several cells at once and `relaxed_gc` sets a process-global
    # threshold; nested, the first cell to finish would restore it for the others.
    full = read_eval_log(log_.location) if log_.samples is None else log_
    # bridge_for: mu's chunked Task packs many rows per sample and registers its own
    # log->rows bridge; every other eval gets the generic one-row-per-sample bridge.
    rows = bridge_for(ev.name)(
        full, eval_name=ev.name, axis=ev.axis, spec_hash=spec.spec_hash, order=order
    )
    if not rows:
        return RunOutcome(
            spec.spec_hash, ev.name, "failed", detail="eval returned no samples", seconds=seconds
        )
    if failed := _undercount_failure(rows, log_.location):
        # A packed sample died (a mu read failed mid-Elo, a triad group, a drift
        # conversation) and `rows_from_packed` stood a flagged NaN row in for everything
        # it owed. That is NOT a measurement: writing it would create the Parquet whose
        # existence is the cache key, and every later sweep would skip a cell that never
        # ran. So no Parquet, the cell is `failed` in the tally (and so fails
        # `--require-all-cached`), and the log is moved out of eval_set's listing so the
        # next sweep re-runs the cell instead of "resuming" from this same log.
        log.error("%s x %s: %s", spec, ev.name, failed)
        return RunOutcome(spec.spec_hash, ev.name, "failed", detail=failed, seconds=seconds)
    # Both knobs, because generation width here is min(connection semaphore, limiter).
    # `width` is the EFFECTIVE width, not this cell's configured `max_workers`: every
    # organism in the process shares one connection semaphore, so _run_group levels the
    # group to its widest cell and passes that value here (see the comment there). Before
    # that, a 256-wide cell sharing a group with 64-wide ones was stamped 256 and ran at
    # 64 — a stamp that says a width no row was produced at is worse than no stamp, since
    # width parity is what CLAUDE.md tells a reader to check before comparing trees.
    # Recorded from THIS run's config, because the log does not carry the task's width
    # (model_generate_config.max_connections is None on 0.3.259); a cell resumed under a
    # different width than the run that generated it is stamped with the new one.
    # Acceptable: resume normally happens within one sweep, one config.
    _stamp(
        rows,
        spec,
        ev,
        serving={
            "runner": "eval_set",
            "max_workers": width,
            # Passed in, not read here: the limiter is a process-global that only widens,
            # so reading it at bridge time stamps the widest the SWEEP has been rather
            # than what this group ran under (see the capture in `_run_group`).
            "gen_threads": (
                gen_threads if gen_threads is not None else int(generation_limiter().total_tokens)
            ),
        },
    )
    # `source_log`: stamps THIS log's name and size into the Parquet, so `core.results.
    # load_all` can later tell "that Parquet is this log's bridged copy" by identity
    # rather than by mtime — mtimes are rewritten by `hf download` / rsync / cp, and a
    # transported tree was silently flipping cells to log-sourced rows, losing exactly the
    # serving stamps set just above.
    # `eval_config`: the resolved config this cell was measured under, stamped into the
    # same schema metadata so the NEXT sweep's `is_cached` can refuse to reuse the cell
    # under a different one. The (spec_hash, eval_name) cache key cannot see the config,
    # so without this a sweep that only changes `n_tasks` silently mixes 10-task and
    # 30-task cells in one scorecard column.
    write_results(
        rows,
        root,
        spec=spec,
        eval_name=ev.name,
        source_log=log_.location,
        eval_config=eval_config,
    )
    # Same flush point as runner.py's tail: the bill lands with the rows it bought, so a
    # sweep killed later (a wedged petri retry loop) is still costable from disk.
    record_judge_usage(root)
    # And the same for the cost record, at the same point runner.py writes it: written
    # cells only (a cached cell's 0.0s is not a measurement), so the scorecard's cost
    # table is populated on an Inspect tree as it is on a native one. `seconds` is the
    # log's own started->completed wall clock, measured under `max_tasks` cells at once.
    write_timing(
        spec,
        ev.name,
        axis=ev.axis,
        status="written",
        n_samples=len(rows),
        seconds=seconds,
        root=root,
        concurrency=concurrency or {},
    )
    return RunOutcome(spec.spec_hash, ev.name, "written", n_samples=len(rows), seconds=seconds)


def _log_path(location: str) -> Path:
    """An Inspect log location as a filesystem path, on Windows as well as POSIX.

    ``EvalLog.location`` is a URI, and on Windows it comes back as ``file://F:/...``.
    ``Path()`` does not parse URIs -- it collapses the double slash into
    ``file:\\F:\\...`` -- so deleting a bridged fake log raised
    ``OSError: [WinError 123] The filename, directory name, or volume label syntax is
    incorrect``. That happens in the bridge step, AFTER every cell has run, so the whole
    sweep died with no Parquet written: measured 9 Sep, 40 minutes of fake sweep lost to it.
    So this function must never raise on a URI shape it doesn't recognise -- that reintroduces
    exactly the failure it was written to fix -- it must instead resolve to *some* path and let
    the caller's ``unlink(missing_ok=True)`` no-op harmlessly if that path doesn't exist.

    On POSIX the location carries no scheme and this returns ``Path(location)`` unchanged,
    which is why CI never saw it.

    ``netloc`` carries the drive on Windows for the two-slash form: ``file://F:/x`` parses to
    netloc ``"F:"``, path ``"/x"``. But a UNC location (``file://server/share/x``, no drive
    letter -- ``server`` is a real host) parses the same shape with a non-drive netloc, and
    naively doing ``netloc + path`` there drops the separator between them and returns the
    RELATIVE path ``server/share/x`` instead of the UNC path ``\\\\server\\share\\x``. That
    path feeds ``unlink(missing_ok=True)``, so a wrong relative path silently deletes an
    unrelated file sitting at that name in the cwd, or silently leaves the real stale fake log
    on disk -- both worse than the loud crash this function exists to avoid, and both invisible
    until someone goes looking for a file that should have been cleaned up.
    """
    parsed = urlparse(location)
    if parsed.scheme != "file":
        return Path(location)
    netloc = parsed.netloc
    is_drive_letter = len(netloc) == 2 and netloc[1] == ":" and netloc[0].isalpha()
    if not netloc or is_drive_letter:
        return Path(url2pathname(netloc + parsed.path))
    #: A real host in netloc is a UNC share -- reconstruct the ``//host/share/...`` root
    #: explicitly so this round-trips as an absolute UNC path instead of a relative one.
    return Path(url2pathname("//" + netloc + parsed.path))


def _cache_inspect_git_context() -> None:
    """Make Inspect look up the git revision ONCE per process, not once per task log.

    WHY THIS IS CUSTOM (CLAUDE.md): Inspect records the repo revision in every `.eval`
    header, and `TaskLogger.__init__` -> `resolve_revision` -> `inspect_ai._util.git.git_context`
    runs three `git` subprocesses (`rev-parse`, `remote get-url`, `status --porcelain`) per
    task, serially, before any sample runs. On the /workspace FUSE volume `git status` is
    0.65-2.4 s, so the smoke config's 186 cells spent 206-450 s in it — most of the fake
    sweep's wall clock (diagnosed 2026-09-29) and the same idle GPU time
    before a real sweep's first generation. Upstream has no switch for it (its only skip is
    `PYTEST_CURRENT_TEST`, which `relaxed_gc` also reads). The revision is a property of the
    checkout, which a run does not change, so every header still records it; the one
    difference is that `dirty` reflects the tree when the process started.

    Idempotent. If an Inspect bump moves the function this is a no-op and
    `tests/test_inspect_git_cache.py` fails, rather than the sweep silently slowing again.
    """
    import functools

    try:
        from inspect_ai._eval.task import (
            log as task_log,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
        )
    except ImportError:
        return
    current = getattr(task_log, "git_context", None)
    if current is None or hasattr(current, "cache_info"):
        return
    task_log.git_context = functools.cache(current)  # ty: ignore[invalid-assignment]


def _run_group(
    group: str,
    cells: list[tuple[OrganismSpec, RegisteredEval]],
    config: dict[str, Any],
    root: Path,
    log_dir: Path,
    *,
    force: bool,
    fake: bool,
    resolver: _CachingResolver,
) -> list[RunOutcome]:
    """Build one Task per cell, run one eval_set over the group, bridge the logs."""
    from inspect_ai import eval_set, task_with

    _cache_inspect_git_context()
    from inspect_ai.model import get_model

    outcomes: list[RunOutcome] = []
    cell_by_key: dict[Cell, tuple[OrganismSpec, RegisteredEval]] = {}
    order_by_key: dict[Cell, dict[str, int]] = {}
    width_by_key: dict[Cell, int | None] = {}
    config_by_key: dict[Cell, dict[str, Any]] = {}
    tasks = []
    resolved: set[str] = set()
    broken: set[str] = set()

    for spec, ev in cells:
        if spec.spec_hash in broken:
            outcomes.append(
                RunOutcome(spec.spec_hash, ev.name, "failed", detail="resolve failed (see above)")
            )
            continue
        if spec.spec_hash not in resolved:
            try:
                resolver(spec.spec_hash)  # once per organism; Task construction reuses it
                # Written before any Parquet, once per organism.
                write_manifest(spec, root, fake=fake)
                resolved.add(spec.spec_hash)
            except Exception as exc:  # one broken organism must not sink the sweep
                log.exception("resolve failed for %s", spec)
                broken.add(spec.spec_hash)
                outcomes.append(
                    RunOutcome(spec.spec_hash, ev.name, "failed", detail=f"resolve failed: {exc}")
                )
                continue
        try:
            # INSIDE the try, with the rest of cell construction: `get_model` builds an
            # `EndpointModelAPI` (and calls the resolver), so it can fail, and a failure
            # here must come back as this cell's `task build failed` rather than
            # propagating out of `_run_group` and taking the whole sweep's remaining cells
            # with it.
            #
            # memoize=False: inspect's model cache is keyed on the id string; a memoized
            # instance would hand a later run in this process a stale endpoint.
            model = get_model(f"{PROVIDER}/{spec.spec_hash}", memoize=False)
            key = cell_key(model, ev)
            eval_config = eval_config_for(config, spec, ev)
            task = task_factory(ev.name)(eval_config)
            if width_is_pinned(task):
                # Its OWN connection pool, per cell (see the levelling below): on the shared
                # one, a pinned 2-wide cell that generated first would size it at 2 for
                # every cell in the group. `pool` is a model arg, so the log records it.
                model = get_model(
                    f"{PROVIDER}/{spec.spec_hash}",
                    memoize=False,
                    pool=f"{ev.name}/{spec.spec_hash}",
                )
                key = cell_key(model, ev)
            task = task_with(
                task,
                # The EVAL name alone — the organism rides in `model`, which is already
                # in eval_set's resume identity (see `cell_key`). Bound from the object
                # `key` was built from, so the two can never describe different models.
                name=ev.name,
                model=model,
                # The rest of the resume key (see `resume_version`): without it eval_set
                # would hand a cell built from a CHANGED config the log of the old one.
                version=resume_version(eval_config, list(task.dataset)),
                # Written into log.eval.metadata so a `.eval` log is self-describing:
                # Phase 5's log reader recovers the row-identity fields from the log
                # alone, without the registry (core cannot import evals).
                #
                # `bridged` says this eval's samples are NOT 1:1 with rows, so reading
                # the log per-sample would silently disagree with the Parquet. core
                # cannot look that up (no evals import), so the log carries it and
                # `core.results.rows_from_eval_log` refuses rather than guessing.
                #
                # MERGED, not replaced: `task_with` assigns `task.metadata` wholesale, and
                # a factory may have stashed what its bridge needs there (petri keeps its
                # resolved sub-config, which no log field otherwise encodes). Overwriting
                # it left the bridge with nothing to read.
                metadata={
                    **(task.metadata or {}),
                    "eval_name": ev.name,
                    "axis": ev.axis,
                    "spec_hash": spec.spec_hash,
                    "bridged": has_bridge(ev.name),
                },
            )
        except EvalSkipped as exc:
            # A DESIGNED no-op, not a failure — see core.registry.EvalSkipped, and
            # serving/runner.py:_run_eval, which records the same status for the same
            # exception on the native path. It has to be caught HERE because eval_set never
            # calls an eval body: the whitebox cells decide "this endpoint cannot serve
            # activations" while their Task is being built. Every FakeEndpoint refuses
            # activations and all six cells are in `default_evals`, so without this the
            # fake-mode gate goes red on six cells per organism, every run.
            outcomes.append(RunOutcome(spec.spec_hash, ev.name, "skipped", detail=str(exc)))
            continue
        except Exception as exc:  # one broken task must not sink the others
            log.exception("task construction failed for %s x %s", spec, ev.name)
            outcomes.append(
                RunOutcome(spec.spec_hash, ev.name, "failed", detail=f"task build failed: {exc}")
            )
            continue
        cell_by_key[key] = (spec, ev)
        # Stamped onto the Parquet by the bridge, so the next sweep can tell this cell was
        # measured at `n_tasks: 10` before reusing it at 30 (`core.results.is_cached`).
        config_by_key[key] = eval_config
        # Dataset order per cell: inspect writes log samples sorted by sample id, while
        # the native rows every parity fixture is frozen against are in submission
        # order. Captured here because this is the only place that holds the task (the
        # eval-body path does the same in run_task_eval).
        order_by_key[key] = {str(sample.id): i for i, sample in enumerate(task.dataset)}
        width_by_key[key] = task.config.max_connections
        tasks.append(task)

    # ONE WIDTH FOR THE GROUP, and it is not a tidiness preference. Inspect keys a
    # model's connection semaphore on `model_concurrency_key` = ModelAPI class +
    # `connection_key()` (inspect_ai/model/_model.py); every organism here is served by
    # our one `EndpointModelAPI`, which does not override `connection_key()`, so all of
    # them share the single key "ModelEndpointModelAPI:default". The registry's
    # `get_or_create` returns the EXISTING semaphore and ignores the requested size
    # (inspect_ai/util/_concurrency.py), so whichever cell generates first sizes the pool
    # for the whole process: a group mixing 64-wide judged cells with 256-wide mcq ones
    # runs entirely at whichever won the race, and stamping each row with its CONFIGURED
    # width records a number the run never used. Level them to the widest and stamp that,
    # so `meta.max_workers` is the width that actually applied.
    #
    # Only tasks that set `Task.config.max_connections` are touched; a task that leaves it
    # None keeps whatever width it pins elsewhere.
    #
    # PINNED CELLS ARE EXEMPT (`evals._inspect.pin_width`). `max_connections` also sets a
    # task's in-flight SAMPLES (inspect's `max_samples` defaults to it), and for a cell whose
    # samples each hold an external resource that is the width that matters: e.g. a
    # remote sandbox held per sample against an account-wide memory cap, so levelling it to
    # a 256-wide mmlu would start 256 sandboxes. Such a cell keeps its own
    # width and was handed its own connection pool above, so its narrow width cannot size the
    # shared one. `max_samples`/`max_sandboxes` cannot do this instead: they are eval_set-wide,
    # not per task, and would cap every other cell too.
    #
    # ponytail: the honest fix is a per-eval `connection_key()` on `EndpointModelAPI` so
    # each cell gets its own pool and configured widths mean what they say — that is an
    # evals/_inspect.py change with a wider blast radius (the whitebox cells
    # reach the provider by other routes). Levelling is correct today because the
    # group already shares one engine.
    pinned_keys = {k for k, t in zip(width_by_key, tasks, strict=True) if width_is_pinned(t)}
    widths = [w for k, w in width_by_key.items() if w and k not in pinned_keys]
    if widths:
        effective_width = max(widths)
        for task in tasks:
            if task.config.max_connections is not None and not width_is_pinned(task):
                task.config.max_connections = effective_width
        width_by_key = {
            k: (effective_width if w and k not in pinned_keys else w)
            for k, w in width_by_key.items()
        }
    # `max_connections` bounds in-flight samples per task, but every generation still
    # runs through the process-wide generation_limiter (default 32) — left alone, a
    # `max_workers: 256` eval silently generates 32-wide and the only symptom is
    # wall-clock (the same failure shape as the max_connections=10 default,
    # 2026-08-24). An explicit MO_EVALS_GEN_THREADS wins, which is what keeps the
    # byte-identity check's fully serial runs serial.
    widest = max((w for w in width_by_key.values() if w), default=0)
    if widest and "MO_EVALS_GEN_THREADS" not in os.environ:
        limiter = generation_limiter()
        if widest > limiter.total_tokens:
            limiter.total_tokens = widest
            log.info("generation limiter widened to %d threads", widest)

    # Captured HERE, before the run, not read off the limiter at bridge time. The limiter
    # is process-global and MONOTONIC — it only ever widens — so a read taken later is a
    # read of "the widest any group has been", which is a property of the sweep, not of
    # the rows being stamped. Taking the value at the point the group's width is decided
    # is what makes the stamp belong to this group's generations.
    #
    # It is a CEILING, not the width that applied: it is still the widest group's number
    # for every group after it, and generation width is min(max_workers, gen_threads).
    # Narrowing it per group is not safe from here — the limiter is shared with anything
    # else generating in the process — so the two fields are stamped and read together
    # (CLAUDE.md, width parity).
    gen_threads = int(generation_limiter().total_tokens)

    # Hoisted out of the `eval_set` call below so the timing record can say how many cells
    # were in flight beside this one — a duration without its overlap cannot be
    # extrapolated, which is the whole point of keeping it (`serving/timing.py`).
    max_tasks = int(
        os.environ.get("INSPECT_EVAL_MAX_TASKS")
        or config.get("max_tasks", config.get("max_concurrent_evals", _DEFAULT_MAX_TASKS))
    )

    if force and log_dir.is_dir() and cell_by_key:
        n = _delete_cell_logs(log_dir, set(cell_by_key))
        log.info("--force: deleted %d stale log(s) under %s", n, log_dir)

    if not tasks:
        return outcomes

    try:
        # preserve_logging: inspect sets the ROOT logger to its own level and leaves it
        # there, so without this every log line this runner emits after its first group
        # — per-cell results, the timing table, the final summary — is silenced.
        with preserve_logging(), relaxed_gc():
            success, logs = eval_set(
                tasks,
                # Passed explicitly, so `$INSPECT_LOG_DIR` is deliberately NOT honoured
                # here (it is the one inspect variable this runner overrides): eval_set
                # treats its log_dir as the set's identity — it reads every log in there
                # to pair tasks to logs — so it needs a dedicated directory PER BACKEND
                # GROUP under the results root, which is also what keeps the fake/real
                # `.mode` marker meaningful. Point `MO_EVALS_RESULTS_ROOT` instead.
                log_dir=str(log_dir),
                # The `--model-role` escape hatch, for a runner that is not the inspect
                # CLI. Each judged Task declares its own `grader` role from config, and
                # eval-level roles are merged OVER a task's
                # (inspect_ai/_eval/loader.py:_merge_model_roles) — so this is how someone
                # swaps the judge for a whole sweep without touching the repo:
                #     INSPECT_EVAL_MODEL_ROLE="grader=openai/gpt-4o" ./scripts/run_all.sh
                # Same variable name and `role=model` syntax the CLI's `--model-role`
                # takes; the inline YAML/JSON form that option also accepts is NOT
                # supported here, and is rejected rather than half-parsed.
                model_roles=_model_roles_from_env(),
                # `max_concurrent_evals` is the key the configs actually define (it is
                # the native runner's cell-parallelism knob); `max_tasks` alone read a
                # key no config in the repo sets, so every sweep silently ran at the
                # default 4 while its config asked for 6. Same meaning on both runners —
                # how many cells are in flight at once — so one key drives both.
                # `$INSPECT_EVAL_MAX_TASKS` wins over the config, because that is the
                # variable inspect's own `--max-tasks` reads (inspect_ai/_cli/eval.py) and
                # someone who exports it expects it to apply to a sweep too.
                max_tasks=max_tasks,
                retry_attempts=int(config.get("retry_attempts", _DEFAULT_RETRY_ATTEMPTS)),
                # Same rule for the display: `$INSPECT_DISPLAY` is inspect's own variable
                # (inspect_ai/util/_display.py), so it wins over the config default —
                # which stays `none`, because a sweep's own log lines are the output.
                display=cast(
                    "DisplayType",
                    os.environ.get("INSPECT_DISPLAY")
                    or str(config.get("inspect_display", "none")),
                ),
                # SAMPLE errors must not fail the cell, matching `run_task_eval` on the
                # other runner: its row comes back with a NaN score (rows_from_log), and
                # NaN means "not measured" everywhere downstream. Inspect's default here
                # is to fail the whole task on the FIRST sample error, which then burns a
                # `retry_attempts` retry of the ENTIRE cell — thousands of good
                # generations re-run because one judge call 500'd. `retry_on_error=2`
                # retries the failing SAMPLE instead, which is the cheap unit.
                #
                # NOT `score_on_error=True`, which the first draft of this had. It does
                # not save the transcript — inspect logs an errored sample either way, with
                # its messages and an `ErrorEvent` (_eval/task/run.py) — it makes the
                # SCORER run on the broken state, so the row comes back with a real-looking
                # number (0.0 for an empty completion) instead of the NaN that means "not
                # measured". That both fabricates a measurement and diverges from
                # `run_task_eval`, which leaves the default off.
                fail_on_error=False,
                retry_on_error=2,
                # Realtime sample logging buys sample-level resume; fake cells finish in
                # seconds and a killed CI run just reruns, so fake mode trades it for
                # ~20% of inspect's per-sample cost (measured on mu, 2026-08-24:
                # 292.9s -> 229.9s). Real runs keep it — resume is the point there.
                log_realtime=not fake,
                # The group log_dir persists across invocations, and each invocation
                # schedules only its uncached cells — logs from other cells are expected.
                # Resume matching is by task_identifier (name + model), so a foreign log
                # can never be served to a scheduled cell.
                log_dir_allow_dirty=True,
            )
    except Exception as exc:
        # One escaped exception fails the WHOLE group: `eval_set` runs the group in a single
        # call, so this handler marks every cell in it failed. Measured 2026-09-01, that is
        # 777 cells for the `vllm` group on the full config — a blast radius that grew when
        # #170 moved `mu_decisiveness` into this call from its own phase.
        #
        # So write the traceback where it can be found. `log.exception` alone puts it on
        # stderr, and the one time this fired the operator had piped the run through `grep`
        # and the traceback went with it — leaving a 1-in-14 failure that could not be
        # diagnosed at all. A file under the results root
        # survives any pipe, and turns the next occurrence into a `cat`.
        # `log_dir` is `logs_root / group` — the directory eval_set was pointed at, so the
        # traceback lands beside the logs of the run that produced it rather than at a path
        # this handler reconstructs and could get wrong.
        crash = log_dir / "eval_set_failure.txt"
        try:
            crash.parent.mkdir(parents=True, exist_ok=True)
            # APPEND, and stamp it. The runner's own advice after a failure is "re-run to
            # resume", so a truncating write means the natural next action destroys the
            # traceback this handler exists to preserve — and a flake that fires twice is
            # exactly the case worth having both of.
            with crash.open("a", encoding="utf-8", errors="replace") as fh:
                fh.write(
                    f"=== {datetime.now(UTC).isoformat()} ===\n"
                    f"backend group: {group}\ncells marked failed: {len(cell_by_key)}\n\n"
                    + traceback.format_exc()
                    + "\n"
                )
            log.exception("eval_set failed for backend group %s — traceback at %s", group, crash)
        except Exception:
            # NOT `except OSError`. A full disk was the failure this was written for, but it
            # is not the only one: a traceback carrying an undecodable byte (vLLM's
            # engine-core stderr reaches us via `os.fsdecode`, so it can hold surrogates)
            # made `write_text` raise UnicodeEncodeError — a ValueError — which escaped,
            # skipped the `outcomes.extend` below, and killed the run inside its own
            # diagnostic with the real exception buried underneath. `errors="replace"` fixes
            # that specific case; this clause is here so the NEXT one cannot do it either. A
            # diagnostic that can sink the run it is diagnosing is worse than no diagnostic.
            log.exception("eval_set failed for backend group %s", group)
        outcomes.extend(
            RunOutcome(spec.spec_hash, ev.name, "failed", detail=f"eval_set failed: {exc}")
            for spec, ev in cell_by_key.values()
        )
        return outcomes

    if not success:
        # eval_set's first return value, which this call used to discard. It is false only
        # when the set gave up — `retry_attempts` exhausted, or a task that cannot be
        # retried — which is a DIFFERENT thing from the per-cell failures the outcomes
        # below already report: those name a cell, this says the scheduler stopped trying.
        # Without the line, "some cells are missing" and "the whole group was abandoned"
        # look identical in the log.
        log.warning(
            "eval_set gave up on backend group %s after retry_attempts: not every cell has "
            "a successful log — re-run to resume from %s",
            group,
            log_dir,
        )

    def _bridge_one(log_: Any) -> RunOutcome:
        key = (log_.eval.task, log_.eval.model)
        spec, ev = cell_by_key[key]
        try:
            outcome = _bridge_log(
                log_,
                spec,
                ev,
                root,
                order_by_key.get(key),
                width_by_key.get(key),
                gen_threads,
                {"max_concurrent_evals": max_tasks, "evals_in_flight": min(max_tasks, len(tasks))},
                config_by_key.get(key),
            )
        except QuarantineFailed:
            raise  # a log still listed after a failed bridge: stop the run, see the class
        except Exception as exc:  # a broken bridge must still name its cell
            log.exception("bridge failed for %s x %s", spec, ev.name)
            return RunOutcome(spec.spec_hash, ev.name, "failed", detail=f"bridge failed: {exc}")
        if outcome.status == "written" and fake and not _keep_fake_logs():
            # Fake logs are ~5.7 GB per gate run and nothing ever reads them: fake mode
            # already runs `log_realtime=False` on the grounds that a killed CI run just
            # reruns, and the Parquet — the cache key — is written above, before this
            # delete. eval_set lists the log dir on its next invocation, so a missing log
            # simply looks like a never-run cell, which is correct: the Parquet is what
            # decides whether the cell is scheduled at all. REAL logs are never touched;
            # transcripts are the primary evidence (CONTRIBUTING.md, Non-negotiables).
            # MO_EVALS_KEEP_FAKE_LOGS=1 opts
            # out — the runner's own tests need a fake log tree to assert against.
            _log_path(log_.location).unlink(missing_ok=True)
        return outcome

    bridged: set[Cell] = set()
    mine = []
    for log_ in logs:
        key = (log_.eval.task, log_.eval.model)
        if key in cell_by_key:  # else: a log for a cell this run did not schedule
            bridged.add(key)
            mine.append(log_)
    # Reading and decoding a `.eval` is I/O + zip/JSON work per cell, independent between
    # cells, and it ran serially on an idle GPU: +219.3s on mu's 141 MB log alone
    # (85.8s of it after `relaxed_gc`). `relaxed_gc` is held HERE, once, around all of
    # them — it sets a process-global threshold, so per-cell nesting would let the first
    # cell to finish restore it for the rest.
    with relaxed_gc(), ThreadPoolExecutor(max_workers=_BRIDGE_THREADS) as pool:
        done = list(pool.map(_bridge_one, mine))
    for log_, outcome in zip(mine, done, strict=True):
        outcomes.append(outcome)
        log.info(
            "%-8s %6.1fs %s x %s %s",
            outcome.status,
            outcome.seconds,
            cell_by_key[(log_.eval.task, log_.eval.model)][0],
            outcome.eval_name,
            f"({outcome.n_samples} samples)" if outcome.n_samples else outcome.detail,
        )
    for key, (spec, ev) in cell_by_key.items():
        if key not in bridged:
            outcomes.append(
                RunOutcome(spec.spec_hash, ev.name, "failed", detail="eval_set returned no log")
            )
    return outcomes


def _dedupe_seeds_evals(config: dict[str, Any], evals: list[RegisteredEval]) -> set[str]:
    """Eval names for which run-once-per-seed-group is on.

    The rule follows temperature (see `serving.runner.eval_config_for`): an eval keeps ALL
    of its seed replicates only when it decodes at temperature > 0, otherwise the extra
    seeds are byte-identical greedy duplicates and it runs once per seed-group.

    So, per eval:
      * an explicit per-eval `dedupe_seeds` in `eval_configs` wins (the escape hatch);
      * else a SHORT BEHAVIOURAL eval (axis in `BEHAVIOURAL_AXES`, name not in the
        top-level `long_evals` list) keeps every seed — those are the temperature-1.0 cells
        whose replicates are a real sampling spread;
      * else (greedy capability/white-box, the logprob coherence reads, and the LONG
        behavioural cells that are held to 1 seed for cost) it falls back to the
        `eval_configs.default` value, which is `dedupe_seeds: true`.

    `mu_decisiveness`/`transitivity` are coherence, so they are NOT short
    behavioural and dedupe to one seed here — even though they sample — which is the
    intended 1-seed treatment for the long cells (their within-organism spread comes from
    `n_samples`, not seeds). Keyed on the eval NAME, so the returned set is real names only.
    """
    per_eval = config.get("eval_configs") or {}
    default_on = bool((per_eval.get("default") or {}).get("dedupe_seeds"))
    long_evals = set(config.get("long_evals") or ())
    out: set[str] = set()
    for ev in evals:
        block = per_eval.get(ev.name) or {}
        if "dedupe_seeds" in block:
            deduped = bool(block["dedupe_seeds"])
        elif (
            ev.axis in BEHAVIOURAL_AXES or ev.name in declared_names()
        ) and ev.name not in long_evals:
            deduped = False
        else:
            deduped = default_on
        if deduped:
            out.add(ev.name)
    return out


def _seed_group_key(spec: OrganismSpec) -> tuple[str, ...]:
    """Identity of an organism ignoring `seed`, for `dedupe_seeds`.

    Two organisms share a seed-group iff they are identical in every field the spec hash
    is built from EXCEPT the seed — including the `seed` that some methods also carry
    inside `method_config` (prompting/few_shot seed replicates), which is dropped here so
    it does not defeat the grouping. Everything else that goes into the spec hash stays in
    the key — persona, base model, method, method_config sans seed, train_scale, behaviour
    class, AND `parent_spec` — so polarity/shot-count/adapter/parent variants are NOT
    collapsed; only the pure seed replicates are. `parent_spec` is `None` for every current
    organism, but it is a spec-hash component, so leaving it out would silently dedupe two
    organisms that differ only in their parent the day one becomes non-null.
    """
    mc = {k: v for k, v in spec.method_config.items() if k != "seed"}
    return (
        spec.persona_id,
        spec.base_model,
        spec.method,
        spec.behavior_class,
        str(spec.train_scale),
        str(spec.parent_spec),
        json.dumps(mc, sort_keys=True, default=str),
    )


def run_matrix(
    config: dict[str, Any],
    specs: list[OrganismSpec],
    evals: list[RegisteredEval],
    *,
    root: Path | str = DEFAULT_RESULTS_ROOT,
    force: bool = False,
    require_stamped: bool = False,
) -> list[RunOutcome]:
    """Run every applicable, uncached (spec, eval) cell through eval_set.

    Cell filtering is applicable_to, then `dedupe_seeds`, then the Parquet cache. Every
    registered eval has a Task builder by construction (registration is one call since
    2026-08-29), so there is no fourth case.
    """
    root = Path(root)
    assert_config_is_runnable(config)
    fake = fake_mode()
    logs_root = root / "logs"
    check_mode_marker(logs_root, fake)

    # `dedupe_seeds`: an eval flagged with it runs on ONE organism per seed-group — the
    # representative (lowest seed) of a set of organisms identical in everything but seed.
    # At greedy decoding the other seeds produce byte-identical output (configs/specs.yaml
    # TEMPERATURE note; CLAUDE.md), so running them is pure duplication. Set on every
    # greedy eval and OFF for the two that sample (mu_decisiveness, transitivity),
    # which keep all seeds. Computed once over the selected specs so the representative is
    # stable within this run.
    dedupe_evals = _dedupe_seeds_evals(config, evals)
    seed_reps: dict[tuple[Any, ...], OrganismSpec] = {}
    for spec in specs:
        key = _seed_group_key(spec)
        if key not in seed_reps or spec.seed < seed_reps[key].seed:
            seed_reps[key] = spec

    outcomes: list[RunOutcome] = []
    todo: list[tuple[OrganismSpec, RegisteredEval]] = []
    for spec in specs:
        for ev in evals:
            if spec.behavior_class not in ev.applicable_to:
                outcomes.append(
                    RunOutcome(
                        spec.spec_hash,
                        ev.name,
                        "skipped",
                        detail=f"{ev.name} does not apply to behavior_class={spec.behavior_class}",
                    )
                )
            elif (
                ev.name in dedupe_evals
                and seed_reps[_seed_group_key(spec)].spec_hash != spec.spec_hash
            ):
                outcomes.append(
                    RunOutcome(
                        spec.spec_hash,
                        ev.name,
                        "skipped",
                        detail="dedupe_seeds: identical to the seed-group representative",
                    )
                )
            # `eval_config=`: a cell measured under a DIFFERENT config is not this cell,
            # and reusing it raises here rather than landing two measurements in one
            # scorecard column (`core.results.is_cached`). `--force` skips the check with
            # the rest of the cache, which is the documented escape hatch.
            elif not force and is_cached(
                spec,
                ev.name,
                root,
                eval_config=eval_config_for(config, spec, ev),
                require_stamped=require_stamped,
            ):
                outcomes.append(RunOutcome(spec.spec_hash, ev.name, "cached"))
            else:
                todo.append((spec, ev))

    # Longest-first, so the tail cells start while short cells still fill the other slots.
    todo = schedule_order(todo, config.get("schedule_first"))

    if not todo:
        return outcomes

    from serving.resolve import LENS_STEERING_METHODS, check_steering_positions, whitebox_mode

    for spec, _ev in todo:  # fake too: a fake smoke must fail where the real run would
        check_steering_positions(spec)
    if not fake:
        # Before any engine is built: the vllm-lens plugin patches every engine in the
        # process once, at vLLM plugin load (`serving/lens_backend.configure_lens_plugin`).
        # Over ALL selected specs, not just `todo`: a run whose steering cells are all cached
        # must not switch the plugin off for a later call in the same process that needs it.
        from serving.lens_backend import configure_lens_plugin

        configure_lens_plugin(
            needs_lens=whitebox_mode() or any(s.method in LENS_STEERING_METHODS for s in specs)
        )
    resolver = _CachingResolver(specs, config)
    set_resolver(resolver)
    try:
        ran_a_group = False
        last_group: str | None = None  # the last group that RAN (bound even if none did)
        for group in _groups_present(todo):  # sequential: one engine on the GPU at a time
            cells = [c for c in todo if spec_backend_group(c[0]) == group]
            if not cells:
                continue
            if ran_a_group:  # noqa: SIM102
                # Sequencing the groups is not enough on its own: an engine holds its
                # weights and KV cache for the life of the process, so the previous
                # group's is still resident when the next one loads and the second
                # engine dies at startup with "Free memory on device ... is less than
                # desired GPU memory utilization" (measured on an A100, 2026-08-24:
                # 8.18 of 79.25 GiB free). Release before loading, not after.
                _release_group_engines()
                resolver.forget()
            ran_a_group = True
            last_group = group
            log.info("backend group %s: %d cell(s)", group, len(cells))
            outcomes.extend(
                _run_group(
                    group,
                    cells,
                    config,
                    root,
                    logs_root / group,
                    force=force,
                    fake=fake,
                    resolver=resolver,
                )
            )
        if last_group == "lens_steering":
            # Steered engines are keyed apart (`build_engine(steered=True)`), so nothing
            # unsteered can reuse one; release it anyway so a later `run_matrix` call in this
            # process (scripts/run_robustness_scale.py loops) starts on a free card.
            _release_group_engines()
            resolver.forget()
    finally:
        clear_resolver()
    return outcomes


def _groups_present(todo: Any) -> list[str]:
    """The backend groups this sweep actually needs, in the order to run them.

    `BACKEND_GROUP_ORDER` alone stopped being the whole list when merged organisms arrived:
    each is its own group (`spec_backend_group`), because it is its own base model and
    cannot share an engine. Fixed groups keep their order and come first — that order is
    load-bearing, not cosmetic, since the release between groups is what stops two engines
    meeting on one card — and the merged groups follow, sorted so a sweep is reproducible.

    Only groups with cells are returned, so a sweep with no merged organisms iterates
    exactly what it always did.
    """
    present = {spec_backend_group(cell[0]) for cell in todo}
    fixed = [g for g in BACKEND_GROUP_ORDER if g in present]
    return fixed + sorted(present - set(BACKEND_GROUP_ORDER))


def _release_group_engines() -> None:
    """Free the GPU held by the group that just finished, before the next one loads.

    Every serving backend that can hold the card gets released here — a backend missing
    from this list keeps its engine for the life of the process, which only shows up in a
    run that spans backends (interp-engine's was missing until 2026-08-25). vllm-lens serves
    on the ordinary vLLM engine, so `release_engines()` already frees the `lens` group; the
    HF reset stays for the (now unrouted) HF backend's defensiveness.
    """
    from serving.hf_backend import _reset_models
    from serving.vllm_backend import release_engines

    release_engines()
    _reset_models()
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:  # accounting for the GPU must never sink the sweep
        log.warning("could not empty the CUDA cache between backend groups: %s", exc)


def assert_config_is_runnable(config: dict[str, Any]) -> None:
    """Refuse the two `eval_configs` keys this runner cannot honour, before anything runs.

    ``runner_backend`` used to choose between eval_set and the eval body's own loop
    (``serving.runner.run_spec`` -> ``evals.friedness._run_direct``). That runner is gone,
    so the only accepted value is ``inspect`` — the value ``configs/specs.yaml`` states for
    ``mu_decisiveness`` — or no value at all.

    ``rescore_from`` is the sharper one, because it fails SILENTLY rather than loudly: the
    eval bodies branch on it to re-judge stored rows without generating
    (``evals.generality._run``, ``evals.introspection``), but eval_set never calls a body,
    so a cell configured with it would build a Task and REGENERATE — a fresh sample
    presented as a re-score of an old one. It is refused rather than routed: it carries the
    rows themselves, not a path, so no YAML config can express it and no sweep has ever
    set it. Re-scoring is a direct call to the eval function, or
    ``evals._harness.rescore_rows``.

    Checked once over ``eval_configs`` rather than per cell: neither key is spec-dependent
    (``runner.eval_config_for`` injects nothing that could change them), so a fully cached
    run still gets the error.
    """
    for name, block in (config.get("eval_configs") or {}).items():
        if not isinstance(block, dict):
            continue
        if block.get("rescore_from") is not None:
            raise ValueError(
                f"eval_configs[{name!r}] sets rescore_from: eval_set generates from a Task "
                "and never calls the eval body, so this cell would REGENERATE rather than "
                "re-score. Re-score out of band instead — call the eval function with "
                "`rescore_from=<rows>`, or `evals._harness.rescore_rows` directly."
            )
        if "runner_backend" not in block:
            continue
        if str(block["runner_backend"]).lower() != "inspect":
            raise ValueError(
                f"eval_configs[{name!r}].runner_backend = {block['runner_backend']!r}: the "
                "native (pre-Inspect) runner was removed, and 'inspect' is now the only "
                "accepted value. Drop the key, or set it to 'inspect'."
            )


def run_all(
    config_path: Path | str | Sequence[Path | str],
    *,
    root: Path | str = DEFAULT_RESULTS_ROOT,
    only_evals: list[str] | None = None,
    only_methods: list[str] | None = None,
    only_personas: list[str] | None = None,
    only_base_models: list[str] | None = None,
    exclude_evals: list[str] | None = None,
    force: bool = False,
    real: bool = False,
    require_stamped: bool = False,
) -> list[RunOutcome]:
    """Load the config, build the matrix, and run it via eval_set.

    ``--methods``, ``--personas`` and ``--base-models`` filter organisms and compose;
    ``--evals`` (or the config's ``default_evals``) selects evals and ``--exclude-evals``
    subtracts from them. All are typo-guarded — a name that matches nothing raises rather
    than silently shrinking the matrix. ``--base-models`` is how a per-base pod claims its
    own organisms; ``--exclude-evals`` is the complement selector a tail pod uses to run
    everything but the sequential family (docs/COMMANDS.md, "Big sweep topology").
    """
    config = merge_configs(_as_paths(config_path))
    if fake_mode():
        _force_fake_backends(config)
    specs = expand_seeds(
        _select_specs(load_specs(config), only_personas, only_methods, only_base_models),
        config,
    )
    evals = _select_evals(all_evals(), only_evals, config, exclude_evals)
    if real:
        from serving.runner import assert_real_backends

        assert_real_backends(config, evals, specs)
    if not specs:
        log.warning("no specs in %s", config_path)
    if not evals:
        log.warning("no evals selected — nothing to run")
    return run_matrix(
        config, specs, evals, root=root, force=force, require_stamped=require_stamped
    )


def _as_paths(config_path: Path | str | Sequence[Path | str]) -> list[Path | str]:
    return [config_path] if isinstance(config_path, Path | str) else list(config_path)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument(
        "--config",
        action="append",
        help="specs YAML; repeat to merge several files (rules: serving.runner.merge_configs; "
        "default: $MO_EVALS_CONFIG or configs/specs.yaml)",
    )
    p.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    # `action="append"`: a repeated flag accumulates. As a plain store, argparse kept only
    # the LAST `--evals`, so `--evals mmlu --evals wmdp` ran wmdp alone and said nothing
    # (found by the #235 review: a five-eval sweep silently ran one cell). The same fix
    # applies to every selection flag below, including the two added since.
    p.add_argument(
        "--evals",
        action="append",
        default=[],
        help="comma-separated subset of eval names; repeatable",
    )
    p.add_argument(
        "--exclude-evals",
        action="append",
        default=[],
        help="comma-separated eval names to subtract from the selected set (tail-pod "
        "complement); repeatable",
    )
    p.add_argument(
        "--methods",
        action="append",
        default=[],
        help="comma-separated subset of implant methods; repeatable",
    )
    p.add_argument(
        "--personas",
        action="append",
        default=[],
        help="comma-separated subset of persona ids; repeatable",
    )
    p.add_argument(
        "--base-models",
        action="append",
        default=[],
        help="comma-separated subset of base model ids (one base model per pod); repeatable",
    )
    p.add_argument("--force", action="store_true", help="re-run even if Parquet exists")
    p.add_argument(
        "--require-stamped",
        action="store_true",
        help="refuse (like a config conflict) to reuse a cached cell with no recorded eval "
        "config (written before 2026-09-16); use after changing an eval's config",
    )
    p.add_argument(
        "--real",
        action="store_true",
        help="refuse to start if any eval would use a fake judge/adversary (also MO_EVALS_REAL=1)",
    )
    p.add_argument(
        "--require-all-cached",
        action="store_true",
        help="exit 1 unless EVERY scheduled cell came back `cached` (the gate's second pass)",
    )
    p.add_argument("--dry-run", action="store_true", help="list the work, run nothing")
    p.add_argument("-v", "--verbose", action="store_true")
    return refuse_blank(p)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.config = args.config or [os.environ.get("MO_EVALS_CONFIG", "configs/specs.yaml")]
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    load_dotenv()
    if fake_mode():
        log.warning("MO_EVALS_FAKE is set — these are FAKE outputs, not results")

    only = _names(args.evals, "--evals")
    exclude = _names(args.exclude_evals, "--exclude-evals")
    only_methods = _names(args.methods, "--methods")
    only_personas = _names(args.personas, "--personas")
    only_base_models = _names(args.base_models, "--base-models")

    if args.dry_run:
        # Same selection as the real run (one `_select_specs`), so a dry run cannot list
        # different work; it resolves no endpoint and constructs no task.
        config = merge_configs(args.config)
        # And the same gate. A dry run is what someone uses to check a config BEFORE
        # spending GPU hours on it, so it is the worst place to stay quiet about the two
        # `eval_configs` keys this runner cannot honour — a `rescore_from` cell in
        # particular fails silently at run time (it regenerates), and printing a clean
        # TODO list for it is an endorsement.
        assert_config_is_runnable(config)
        specs = expand_seeds(
            _select_specs(load_specs(config), only_personas, only_methods, only_base_models),
            config,
        )
        evals = _select_evals(all_evals(), only, config, exclude)
        # Reflect the same dedupe_seeds decision run_matrix will make, so a dry run tells
        # the truth about which seed replicates are skipped (plan validation step 2).
        dedupe_evals = _dedupe_seeds_evals(config, evals)
        seed_reps: dict[tuple[Any, ...], OrganismSpec] = {}
        for spec in specs:
            key = _seed_group_key(spec)
            if key not in seed_reps or spec.seed < seed_reps[key].seed:
                seed_reps[key] = spec
        for spec in specs:
            for ev in evals:
                path = results_path(spec, ev.name, args.results_root)
                if (
                    ev.name in dedupe_evals
                    and seed_reps[_seed_group_key(spec)].spec_hash != spec.spec_hash
                ):
                    state = "DEDUPE "
                else:
                    # Same config check the real run makes, but REPORTED rather than
                    # raised: a dry run is what someone uses to check a config before
                    # spending GPU hours, so it has to list every conflicting cell, not
                    # die on the first one.
                    try:
                        cached = is_cached(
                            spec,
                            ev.name,
                            args.results_root,
                            eval_config=eval_config_for(config, spec, ev),
                            require_stamped=args.require_stamped,
                        )
                        state = "CACHED " if cached else "TODO   "
                    except ValueError as exc:
                        state = "CONFLICT"
                        print(f"{state} {spec} x {ev.name}: {exc}")
                        continue
                print(f"{state} {spec} x {ev.name} ({ev.axis}) -> {path}")
        return 0

    env_real = os.environ.get("MO_EVALS_REAL", "").strip().lower() in {"1", "true", "yes", "on"}
    t0 = time.perf_counter()
    outcomes = run_all(
        args.config,
        root=args.results_root,
        only_evals=only,
        only_methods=only_methods,
        only_personas=only_personas,
        only_base_models=only_base_models,
        exclude_evals=exclude,
        force=args.force,
        real=args.real or env_real,
        require_stamped=args.require_stamped,
    )
    log.info("%s", timing_table(outcomes, wall=time.perf_counter() - t0))
    record_judge_usage(args.results_root)
    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o.status] = counts.get(o.status, 0) + 1
    log.info(
        "done: %s", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())) or "nothing to do"
    )
    if not fake_mode():
        # vLLM's engine teardown at interpreter exit can print a traceback (its background
        # loop dies under the atexit hooks) after everything has been written. Said once
        # here so a finished run is not read as a crash; the exit code is what counts.
        log.info(
            "note: a vLLM shutdown traceback printed after this line is harmless; the exit "
            "code and the `done:` counts above are the result"
        )
    if args.require_all_cached:
        # The gate's second pass asserts the cache works: presence of the Parquet is the
        # cache key (CLAUDE.md), so a pass over a tree the first sweep just filled
        # must re-run NOTHING. Until this flag existed the property was only VISIBLE — a
        # broken cache showed up as this step taking 190s instead of 7s, and only if
        # somebody read the timing table. The retired CI job did assert it; this restores
        # that. `written` is the failure. `skipped` is NOT — a cell whose behaviour class
        # the eval does not apply to was never going to be cached, and demanding otherwise
        # would make this flag unusable on any config with an inapplicable pairing.
        # `failed` falls through to the existing check below, which already exits 1.
        not_cached = [o for o in outcomes if o.status not in {"cached", "skipped", "failed"}]
        if not_cached:
            for o in not_cached[:10]:
                log.error("NOT CACHED %s x %s: %s", o.spec_hash, o.eval_name, o.status)
            log.error(
                "--require-all-cached: %d of %d cells were REBUILT rather than served from "
                "cache. The first sweep did not leave a complete tree, or the cache key "
                "stopped matching it",
                len(not_cached),
                len(outcomes),
            )
            return 1
    return 1 if counts.get("failed") else 0


def cli() -> int:
    """`python -m serving.inspect_runner`: `main`, with a refused selection (an unknown name, an
    empty field) as ONE `REFUSING: <message>` line on stderr and `EXIT_SELECTION`, no traceback.
    `main` itself still raises, for callers that want the exception."""
    from serving.runner import EXIT_SELECTION, SelectionError

    try:
        return main()
    except SelectionError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return EXIT_SELECTION


if __name__ == "__main__":
    sys.exit(cli())
