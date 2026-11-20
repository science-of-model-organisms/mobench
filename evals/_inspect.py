"""Shared Inspect adapter: any ``Endpoint`` as an Inspect model, plus the log bridge.

Gives every
migrated eval uses one provider, one bridge, one seed rule. Nothing scores anything here.

- ``EndpointModelAPI`` / ``bind_endpoint`` — present a ``core.endpoint.Endpoint`` as the
  Inspect model ``mo_organism/<key>``. ``bind_endpoint`` is the test path; the
  eval_set runner instead installs a resolver (below) so organisms resolve lazily by
  spec_hash.
- ``set_resolver`` — dependency injection seam. ``evals/`` may not import ``serving/``
  (import-boundary rule #3): the runner hands us ``spec_hash -> ResolvedTarget`` at
  runtime instead of us resolving specs ourselves.
- ``rows_from_log`` — pure function ``EvalLog -> list[SampleResult]``. Parquet stays
  derived from logs through this and cannot drift from them.
- ``generation_limiter`` — one process-wide cap on concurrent ``Endpoint.complete``
  threads. Inspect's ``max_connections`` counts in-flight *requests*; actual generation
  concurrency for an in-process provider is however many worker threads run ``complete``
  at once, and that is this limiter, not anyio's default-40 pool shared with everything
  else. Default 32 (``evals/_harness._map``'s default); the eval_set runner widens it to
  the widest scheduled task's ``max_connections`` so a configured ``max_workers: 256``
  actually reaches the engine, and an explicit ``MO_EVALS_GEN_THREADS`` always wins.
- ``BoundedJudge`` / ``inspect_judge`` — the judge seam. Inside a task the judge's
  TRANSPORT is Inspect's model API, served under the ``grader`` model role, so judge
  calls appear in the ``.eval`` transcript, carry Inspect's own usage/cost and retries,
  and can be swapped from outside with ``--model-role grader=...``. Rendering, position
  randomisation and parsing stay in ``judges.judge``; only the client changes.
  ``BoundedJudge`` remains the concurrency ceiling — under eval_set, in-flight judge
  calls = max_tasks x per-task scorer concurrency, and that fan-out must not ride the
  generation knob. The judge's own width is ``judge_max_connections`` (default 64), sized
  against the HTTPS client pool rather than the GPU coalescer; see
  ``_judge_max_connections``.
- ``InspectAdversary`` / ``inspect_adversary`` / ``adversary_turn`` — the same seam for
  the SECOND model in a conversation: ``robustness_multiturn_debate_gen``'s opponent and the
  live-partner arm of the drift cells, served under the ``adversary`` model role. Same
  three wins as the judge (log events, usage/cost, ``--model-role adversary=...``) and
  the same width, since a rebuttal is an HTTPS request out of one client rather than a
  local generation.

WHY THE ENDPOINT IS LOOKED UP BY NAME RATHER THAN PASSED IN
    Inspect refuses to log a ``ModelAPI`` that carries no registry info, so the obvious
    ``Model(EndpointModelAPI(endpoint), ...)`` fails at task startup with "Object
    'unknown' does not have registry info". The supported route is the ``@modelapi``
    decorator plus ``get_model("mo_organism/<name>")`` — but that only hands the provider
    a *string*. Passing the live endpoint through ``**model_args`` would work and then
    put a Python object into the log's model config, so instead the endpoint goes into a
    process-local table under a key, and the key is the model name.

``inspect_ai`` is imported at module top level. This module is only imported lazily from inside
eval bodies (or the runner), never from ``import evals``.
"""

from __future__ import annotations

import asyncio
import functools
import gc
import hashlib
import json
import logging
import math
import operator
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import lru_cache
from time import perf_counter
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, Protocol, cast, get_args, override

# Import the submodule we use rather than reaching it through the parent package.
# A bare `import anyio` DOES expose `anyio.to_thread` today — `anyio/__init__.py`
# imports `._core._fileio`, which does `from .. import to_thread` as a side effect —
# but that is an anyio internal, and ty cannot see a binding made that way. Explicit
# is what the checker and the next reader both need. (An earlier revision filed this
# as a latent AttributeError; it was not one.)
import anyio.from_thread
import anyio.to_thread
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    ContentText,
    GenerateConfig,
    Logprob,
    Logprobs,
    Model,
    ModelAPI,
    ModelCall,
    ModelInfo,
    ModelOutput,
    ModelUsage,
    StopReason,
    get_model,
    modelapi,
    set_model_info,
)

# Private path, and there is no public one on 0.3.260. This is the EVAL-level generate
# config — inspect's `--seed` / `eval(seed=)` — which `Model._resolve_config` merges into
# the per-call config before the provider sees it. Reading it separately is the only way
# to tell "the run asked for seed S" from "this task asked for seed S", which is what
# `EndpointModelAPI._seed_for` needs to fold rather than clobber.
from inspect_ai.model._generate_config import (
    active_generate_config,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
)

# Also a private path with no public alias on 0.3.260, and the reason we write no tool
# parser of our own (CLAUDE.md): this is inspect's own text-based extractor for models
# served as raw completions, used by its HF provider on exactly this seam. It is
# re-exported from `inspect_ai.model._providers.util`, which is as close to public as it
# gets. Ours would drift from upstream and cover fewer conventions.
from inspect_ai.model._providers.util import (
    HFHandler,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
)

# Private path again, no public alias on 0.3.260: inspect's own container for a split-off
# reasoning part. The split itself is keyed on the organism's thinking setting (see
# `EndpointModelAPI.generate`), not parsed out of arbitrary text.
from inspect_ai.model._reasoning import (
    ReasoningCapsule,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
)
from inspect_ai.tool import ToolChoice, ToolInfo

from core.endpoint import Completion, Endpoint
from core.reasoning import QWEN_THINK
from core.registry import VALID_AXES
from core.results import SampleResult, register_log_bridge
from core.spec import VALID_BEHAVIOR_CLASSES
from evals._harness import _CHARS_PER_TOKEN, estimate_tokens, join_id

if TYPE_CHECKING:
    from inspect_ai.log import EvalLog

log = logging.getLogger(__name__)

# The inspect provider namespace. `mo_organism/<key>` is a target model id. Kept from
# `_petri_models` so existing petri logs and configs keep naming the same provider.
PROVIDER = "mo_organism"
_STOCK_MAX_TOKENS_MARKER = "_mo_stock_max_tokens"
_STOCK_REMAINING_CONTEXT = -1
# Private `extra_body` flag, never forwarded: this call's `max_tokens` is the eval's
# deliberate per-turn cap (SWE-bench Pro, Petri's target), which a thinking-on organism keeps
# plus THINKING_ALLOWANCE_TOKENS rather than trading for the remaining context.
#
# STAMPED ONLY FOR A THINKING-ON ORGANISM (`turn_cap_extra_body`). `extra_body` is part of the
# GenerateConfig that inspect's `task_identifier` hashes, so stamping it unconditionally gave
# every thinking-off (Qwen2.5) swe_bench_pro / petri cell a new identity and broke eval_set's
# resume of logs written before this marker existed. The provider only reads it on a
# thinking-on turn, so leaving it off everywhere else changes no generation.
_TURN_CAP_MARKER = "_mo_turn_cap"


def is_thinking_on(endpoint: Any) -> bool:
    """Whether `endpoint` serves with the think block open (`enable_thinking`).

    `getattr` with a False default: an endpoint without the attribute (`FakeEndpoint`, a
    foreign `--model`'s `None`) is thinking-off, which is what it is.
    """
    return bool(getattr(endpoint, "enable_thinking", False))


def turn_cap_extra_body(endpoint: Any) -> dict[str, Any] | None:
    """The `extra_body` a deliberately per-turn-capped cell stamps for `endpoint`.

    `{_TURN_CAP_MARKER: True}` for a thinking-on organism, `None` for every other one — so a
    thinking-off cell's GenerateConfig, and with it inspect's `task_identifier`, is exactly
    what it was before thinking support existed.
    """
    return {_TURN_CAP_MARKER: True} if is_thinking_on(endpoint) else None


class ContextExhausted(ValueError):
    """A remaining-context generation whose prompt already fills ``max_model_len``.

    A ``ValueError`` subclass so every existing handler keeps working; named so a caller
    that can end a conversation early (the uncapped debate, ``evals._task_eval``) can tell
    "the window is full" from any other failure without matching the message text.
    """


def remaining_context_config() -> dict[str, Any]:
    """``GenerateConfig`` fields for "no output cap: whatever context the prompt leaves".

    The same mechanism the stock capability tasks use (``_STOCK_MAX_TOKENS_MARKER``): the
    provider sets ``max_tokens = max_model_len - prompt_tokens`` per call (``_generate``),
    so each turn may use all the context left and cannot overflow it.
    """
    return {"max_tokens": None, "extra_body": {_STOCK_MAX_TOKENS_MARKER: True}}


#: Open-file floor for a resume. `eval_set` pairs tasks to logs by reading EVERY header in
#: its log_dir, and inspect fans that out over an unbounded anyio TaskGroup
#: (`read_eval_log_headers_async` -> `tg_collect`), so all N `.eval` zips are open at once.
#: At ~237 logs plus the sweep's own HTTP sockets that clears a container's default
#: `ulimit -n` of 1024 and the whole group dies with `OSError: [Errno 24]`. The FRESH run
#: never touches that path — it lists an empty dir — so only resumes fail.
_NOFILE_FLOOR = 65_536


def raise_open_file_limit(floor: int = _NOFILE_FLOOR) -> int | None:
    """Raise this process's ``RLIMIT_NOFILE`` soft limit to ``min(floor, hard)``.

    Never lowers it, and never touches the hard limit. Called at import of this module —
    the one thing both entry points pass through, `serving.inspect_runner` (via the
    `evals/_task_*.py` imports) and a plain `inspect eval` of one of our tasks, which
    resumes the same way and hits the same ceiling.

    Returns
    -------
    int | None
        The soft limit now in force, or ``None`` where ``resource`` is unavailable.
    """
    try:
        import resource
    except ImportError:  # non-POSIX; nothing to raise
        log.warning("no `resource` module on this platform: RLIMIT_NOFILE left alone")
        return None

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft == resource.RLIM_INFINITY:
        return soft
    # RLIM_INFINITY is -1, so `min()` on a raw hard limit would ask for -1.
    target = floor if hard == resource.RLIM_INFINITY else min(floor, hard)
    if target <= soft:
        return soft
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError) as exc:
        # macOS refuses a soft limit above `kern.maxfilesperproc` even below the hard
        # limit; a sandbox may refuse outright. Warn and carry on at the old ceiling.
        log.warning("could not raise RLIMIT_NOFILE %d -> %d: %s", soft, target, exc)
        return soft
    log.info("RLIMIT_NOFILE soft limit raised %d -> %d (hard limit %d)", soft, target, hard)
    return target


raise_open_file_limit()


class PromptTokenCount(Protocol):
    """Exact rendered-prompt token counter supplied by the serving layer."""

    def __call__(
        self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None
    ) -> int: ...


class ResolvedTarget(NamedTuple):
    """What the provider needs to serve one organism."""

    endpoint: Endpoint
    seed: int
    max_tokens: int | None = None
    # Sampling overrides from an eval's `complete_kwargs` (`target_sampling`). None leaves
    # the serving default (configs/serving.yaml) alone, exactly as `max_tokens` does.
    temperature: float | None = None
    top_p: float | None = None
    max_model_len: int | None = None
    prompt_token_count: PromptTokenCount | None = None
    # `OrganismSpec.base_model` (an HF id). Only the tool-call parser reads it: inspect's
    # `HFHandler` picks its parser by model family, and the model NAME here is a spec_hash.
    base_model: str | None = None


# key -> ResolvedTarget. Process-local and short-lived: `bind_endpoint` owns its entries
# and always removes what it added.
_BOUND: dict[str, ResolvedTarget] = {}


def unlogged_lookup[V](mapping: Mapping[str, V] | Sequence[V]) -> Callable[[Any], V]:
    """A bank lookup that Inspect will NOT copy into every sample of the log.

    NEVER pass a dict straight into a `@scorer` / `@solver` factory. Inspect records a
    factory's arguments with `registry_params()` and writes them into the score event of
    EVERY sample, for reproducibility — so a 500-item bank passed this way is serialised
    500 times per cell, once per sample, as data identical in all of them.

    Measured 2026-08-26: `registry_params` on a 200-item bank is **104,907 bytes** as a
    dict and **27** as this callable. In the wild it was worse — the fake gate wrote
    13 MB per `.eval` log (a single fake `mmlu` cell: 39 MB, 97% of each sample one
    `score` event holding the bank), 11 GB and 1h20m for 720 of 1134 cells.

    Returning `mapping.__getitem__` keeps the lookup O(1) and the closure un-serialisable,
    so Inspect stores a short repr instead of the contents. A SEQUENCE works too and is
    not a lesser case: `mu_decisiveness` passed its whole chunk plan as a `list` and so
    slipped past an annotation-shaped check — 7.5 MB in every sample's score event.
    """
    return mapping.__getitem__


def unlogged_ref[T](obj: T) -> Callable[[], T]:
    """Hide a whole object from Inspect's per-sample record behind a zero-arg accessor.

    For a plan/bundle rather than a lookup table: `mu_decisiveness`'s `plan` is 3.7 MB and
    annotated `Any`, so neither a mapping check nor a type check would find it. Resolve it
    once in the factory body (`plan = get_plan()`); the inner solver/scorer closes over the
    real object and needs no change.
    """
    return lambda: obj


# How many scorer working sets stay resident.
#
# MUST STAY WELL ABOVE `max_tasks` (6 in configs/specs.yaml, 4 in the smoke config).
# While a cell is in flight its entry is the SAME object the Task's solver already holds,
# so a generous cap costs nothing but a few stale entries after the cell ends — whereas a
# cap at or below `max_tasks` would evict a live cell and rebuild it mid-run, which for a
# judged cell means a FRESH `BoundedJudge` and therefore a second judge semaphore beside
# the configured one. Only mu's plan is large (~3.7 MB), and only one mu cell runs per
# organism.
_CONTEXT_CACHE = 64
_CONTEXTS: dict[tuple[str, str, str], Any] = {}


def scorer_context[T](build: Callable[[str, dict[str, Any]], T]) -> Callable[..., T]:
    """Build one scorer's working set from ``(eval_name, config)``, once per cell.

    A ``@scorer`` factory's arguments are copied into the log (``registry_params``) and,
    crucially, are what ``inspect score`` RE-CREATES the scorer from: it reads
    ``log.eval.scorers[].options`` and calls ``scorer_create(name, **options)``
    (inspect_ai/_eval/loader.py). So a factory argument that is not JSON-serialisable
    comes back as its *string repr* — an ``unlogged_lookup`` closure rehydrates as
    ``"__getitem__"`` and the scorer dies on ``TypeError: 'str' object is not callable``
    (measured 2026-08-27). Every scorer here therefore takes identifiers — an eval name
    and its config — and resolves the bank, judge, plan or probe through this.

    That would build each cell's working set twice, since the task factory has already
    built the same objects for the dataset and the solver, so the result is memoised on
    ``(builder, eval_name, config)``. The cache is capped rather than unbounded: a full
    sweep is ~1100 cells and holding every plan would be gigabytes.
    """

    def get(eval_name: str, config: dict[str, Any]) -> T:
        key = (
            # Every callable has one at runtime; `Callable` does not declare it.
            build.__qualname__,  # ty: ignore[unresolved-attribute]
            eval_name,
            json.dumps(config, sort_keys=True, default=str),
        )
        if key not in _CONTEXTS:
            if len(_CONTEXTS) >= _CONTEXT_CACHE:
                _CONTEXTS.pop(next(iter(_CONTEXTS)))
            _CONTEXTS[key] = build(eval_name, config)
        return _CONTEXTS[key]

    return get


# Installed by serving/inspect_runner.py (set_resolver) so eval_set-scheduled tasks can
# resolve `spec_hash -> Endpoint` lazily without `evals/` importing `serving/`.
_RESOLVER: Callable[[str], ResolvedTarget] | None = None


def set_resolver(resolver: Callable[[str], ResolvedTarget]) -> None:
    """Install the lazy `key -> ResolvedTarget` fallback used when a key is not bound."""
    global _RESOLVER
    _RESOLVER = resolver


def clear_resolver() -> None:
    global _RESOLVER
    _RESOLVER = None


def off_event_loop[T](work: Callable[[], T]) -> T:
    """Run `work` on a thread where `asyncio.get_running_loop()` RAISES, and block for it.

    NOT a "don't stall the loop" helper — the caller is blocked either way. It exists
    because **an engine must not be constructed while an event loop happens to be
    running on the constructing thread**, and on the single-cell path one is.

    THE HANG THIS FIXES. `inspect eval
    mo_evals/<name> -T spec_hash=...` against a real organism loaded the engine and then
    produced 0 samples until it was killed — every eval, while the eval_set sweep ran the
    same cells fine. Inspect resolves a task INSIDE its own running loop, so
    `mo_evals.build_cell` -> `get_model` -> this provider -> `resolve()` -> vLLM's
    `AsyncLLM.__init__` all run under it, and vLLM (v0.27.1,
    `vllm/v1/engine/async_llm.py:172-178`) does:

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

    ...so the ONE output-handler task is created on *Inspect's* loop, and
    `_run_output_handler` is a no-op ever after (`if self.output_handler is not None:
    return`, same file). `serving/vllm_backend.py:_AsyncVLLMEngine` then drives the engine
    from its own daemon loop, where every request future waits for a handler that is on
    the other loop and can never wake it: `chat()` blocks in `future.result()` forever.
    The sweep never saw it because `serving/inspect_runner.py` resolves each organism
    before `eval_set()` starts a loop at all — which is the behaviour this restores.

    Fake mode was equally unaffected, and that is the trap: `FakeEndpoint` has no engine,
    so "the CLI path works" had only ever been checked where nothing could break.

    Nothing to do when no loop is running (the sweep, `bind_endpoint` in tests, and the
    `anyio.to_thread` call in `generate`, whose worker thread has no loop either): the
    call is made inline, so this stays byte-identical for every existing path.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return work()
    # ponytail: a throwaway one-shot thread; resolving is once per organism per process.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="mo-resolve") as pool:
        return pool.submit(work).result()


# Built on demand by `EndpointModelAPI._resolve_from_config` for shell-CLI runs
# (`--model mo_organism/<hash>` with no sweep and no `-T` selector in the process).
# Module-level so one process holds ONE resolver — engines are cached per organism,
# however many cells the CLI runs. The lock guards the build, not each resolve
# (the resolver carries its own per-organism lock). The FACTORY is injected by
# `mo_evals` at import time — `evals/` may not import `serving/`
# (scripts/check_import_boundaries.py), and every CLI route loads `mo_evals` via the
# entry point before this provider can be constructed by name.
_CONFIG_RESOLVER: Callable[[str], ResolvedTarget] | None = None
_CONFIG_RESOLVER_FACTORY: Callable[[], Callable[[str], ResolvedTarget]] | None = None
_CONFIG_RESOLVER_LOCK = threading.Lock()


def set_config_resolver_factory(factory: Callable[[], Callable[[str], ResolvedTarget]]) -> None:
    """Install the builder for the shell-CLI fallback resolver (called by `mo_evals`)."""
    global _CONFIG_RESOLVER_FACTORY
    _CONFIG_RESOLVER_FACTORY = factory


@contextmanager
def bind_endpoint(
    key: str,
    endpoint: Endpoint,
    seed: int,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_model_len: int | None = None,
    prompt_token_count: PromptTokenCount | None = None,
    base_model: str | None = None,
) -> Iterator[Model]:
    """Expose `endpoint` to inspect as the model `mo_organism/<key>`, then take it away.

    Args:
        key: Unique name for this organism within the process. `spec_hash` is the
            natural choice — it is unique per organism and it is what the results row
            is keyed by, so a log line naming the model names the organism.
        endpoint: The resolved organism.
        seed: `OrganismSpec.seed`, used to derive per-generation seeds.
        max_tokens: Output cap per target turn, passed through to `Endpoint.complete`.
            `None` leaves the serving default alone.
        temperature: Sampling temperature per target turn, likewise; `None` keeps the
            serving default (greedy, configs/serving.yaml).
        top_p: Nucleus cutoff per target turn, likewise.
        base_model: The organism's base model HF id; selects the tool-call parser family.

    Yields:
        The inspect `Model` to pass as the target model.
    """
    if key in _BOUND:
        raise ValueError(f"endpoint key {key!r} is already bound")
    _BOUND[key] = ResolvedTarget(
        endpoint,
        seed,
        max_tokens,
        temperature,
        top_p,
        max_model_len,
        prompt_token_count,
        base_model,
    )
    try:
        # memoize=False: inspect's model cache is keyed on the model id, and two
        # organisms in one process would otherwise share the first one's endpoint.
        yield get_model(f"{PROVIDER}/{key}", memoize=False)
    finally:
        _BOUND.pop(key, None)


def _advertise_context_window(model_name: str, max_model_len: int | None) -> None:
    """Tell Inspect the organism's real context window, so its context-relative knobs work.

    Inspect sizes anything expressed as a FRACTION of the context window through
    `get_model_input_tokens(model)`, which consults its own model database and the
    `set_model_info` registry. `mo_organism/<spec_hash>` is in neither, so the lookup
    returned None and every caller fell back to `DEFAULT_CONTEXT_WINDOW` (128,000).

    What that broke (found on the since-retired `swe_bench_pro`, which ran
    `react(..., compaction=CompactionEdit())`): `CompactionEdit`'s default `threshold=0.9`
    therefore resolved to 115,200 tokens
    against a 32,768-token serving window — so compaction could never fire. Measured on the
    2026-09-15 paired 5-task run: the base organism's largest single prompt reached 31,655
    tokens, 1,113 short of the wall, with zero compaction events. Past the wall `_generate`
    raises "no output context remains", which is a sample ERROR -> NaN -> a shrinking
    denominator in `analysis/derived.py`, i.e. the MORE verbose organism is silently dropped
    rather than scored.

    `max_model_len` is the same `ResolvedTarget.max_model_len` that `_generate` computes the
    remaining-context cap from (`serving_config().max_model_len`) — one source of truth, no
    second knob. `None` (nothing bound, or a serving config without a window) advertises NO
    window: inspect's own fallback then stands, which is honest, where a guess would not be.

    Registered under the full model string because that is the first key
    `get_model_input_tokens` checks (`_custom_models.get(str(model))`). The registry is
    process-global and `set_model_info` has no inverse, so unlike `_BOUND` this outlives the
    `bind_endpoint` block.

    Which is why a `None` window still REGISTERS, rather than returning early: rebinding the
    same model id — same spec_hash, a serving config that no longer states a window — would
    otherwise leave the previous registration standing, and compaction would then fire at a
    threshold derived from a window that no longer applies (A/full -> 32768, B/distinct ->
    None, but B/reused-A-id -> 32768). Registering `context_length=None` overwrites it:
    `get_model_input_tokens` skips a custom entry whose `input_tokens` is None and falls
    through to its own lookup, i.e. exactly the fallback an unbound organism should get.
    """
    # `ModelInfo.input_tokens` falls back to `context_length`, and a vLLM `max_model_len` IS
    # the whole window (prompt + output), so `context_length` is the field that matches.
    set_model_info(f"{PROVIDER}/{model_name}", ModelInfo(context_length=max_model_len))


class ForeignModel(TypeError):
    """The Inspect model in hand is served by someone else's provider.

    A `TypeError` subclass, so every caller that simply lets `endpoint_for` fly keeps the
    exception it already expects. It exists so `probe_endpoint` can separate "this model
    is not ours, and never had an `Endpoint`" — a legitimate answer under `--model`, which
    `mo_evals.py` advertises — from "this model IS ours and resolving it failed", which is
    a misconfiguration and must stay loud.
    """


def endpoint_for(model: Model) -> Endpoint:
    """The live `Endpoint` behind an `mo_organism/<key>` Inspect model.

    For the white-box probes, whose measurement is `Endpoint.hidden_states` — something
    no Inspect `Model` can express, and which must NOT be smuggled into the provider's
    `generate` (CLAUDE.md: nothing method-specific in the protocol, and nothing
    eval-specific in the adapter). The solver takes the activation itself, off the same
    `messages` list it generated from, and this is how it reaches the endpoint.
    """
    api = model.api
    if not isinstance(api, EndpointModelAPI):
        raise ForeignModel(
            f"endpoint_for needs a {PROVIDER}/... model; got {model} served by "
            f"{type(api).__name__}, which has no Endpoint behind it"
        )
    if api.endpoint is None:
        # The same $MO_EVALS_CONFIG fallback `generate` takes, and this is where the
        # BUILD-TIME callers reach it: the white-box and mu factories ask the
        # endpoint what it can serve before they build a dataset, so on the shell-CLI path
        # (`inspect eval mo_evals/<eval> --model mo_organism/<hash>`) they hit an
        # unbound provider and raised instead of running. Synchronous on purpose — a task
        # factory is sync, and an engine load there is exactly what the CLI user asked for.
        api._resolve_from_config()
    if api.endpoint is None:
        raise ValueError(
            f"no endpoint for {PROVIDER}/{api.model_name}; bind one with "
            "evals._inspect.bind_endpoint or install a resolver via set_resolver"
        )
    return api.endpoint


#: ``spec_hash`` stamped on rows in bring-your-own-model mode, where ``--model`` supplies
#: the model and there is no organism spec to hash. Defined here rather than in
#: ``mo_evals.py`` because ``probe_endpoint`` below has to recognise it and ``evals/`` may
#: not import the root module; ``mo_evals._external_spec`` imports it from here.
EXTERNAL_SPEC_HASH = "external"


def probe_endpoint(spec_hash: str) -> Endpoint | None:
    """The `Endpoint` a BUILD-TIME capability probe should ask, or `None` for a model of
    someone else's that has none.

    The build-time probes (`_task_eval._resolved_elicitation`, and the white-box cells'
    activations check while they were registered) ask a real question of a real backend —
    can it serve prompt logprobs / activations — so each has to reach the endpoint
    that will actually run the cell. Which endpoint that is depends on how the cell was
    entered, and getting it from the `spec_hash` alone is wrong in one of the two modes:

    - organism mode (`-T spec_hash=`, and every sweep): the runner's resolver or
      `bind_endpoint` has bound this hash, so `PROVIDER/<hash>` names it;
    - bring-your-own-model (`--model mo_organism/<hash>`, `--model openai/...`): there is
      no spec, so `mo_evals._external_spec` stamps `"external"` — a hash that names no
      organism and never will. Looking it up asked for the literal `mo_organism/external`
      and died at task build before a sample ran (GPU battery, 2026-09-01). The model is
      ALREADY bound by `--model`, so the probe takes it: `get_model()` with no argument is
      the model being evaluated (`inspect_ai` 0.3.260 `_eval/eval.py` calls
      `init_active_model` immediately before `resolve_tasks`, which is what invokes a task
      factory, and falls back to `$INSPECT_EVAL_MODEL`).

    One helper rather than the same two lines in three factories, so the next mode is one
    edit. It never answers a default — the `befd32c` rule that a probe must ask rather
    than infer — and `None` is not a default: a model from another provider (`--model
    openai/gpt-4o`, `--model mockllm/model`) has no `Endpoint` to ask, which is exactly
    the "this backend cannot serve the read" case every caller already handles. Returning
    it instead of raising is what lets a foreign `--model` degrade rather than die at task
    build; each caller says in its own reason WHICH of the two happened, because a model
    that was never asked is not a backend that refused.

    `ForeignModel` and nothing wider: `PROVIDER/<hash>` always reaches an
    `EndpointModelAPI`, so an unbound or unresolvable organism raises out of here as
    itself. Degrading that would turn a typo'd `-T spec_hash=` into a silently skipped
    cell.
    """
    if spec_hash == EXTERNAL_SPEC_HASH:
        try:
            return endpoint_for(get_model())
        except ForeignModel:
            return None
    # memoize=False: inspect's model cache is keyed on the id string, so a second cell in
    # one process would otherwise be handed the first one's endpoint.
    return endpoint_for(get_model(f"{PROVIDER}/{spec_hash}", memoize=False))


def foreign_model_reason(eval_name: str, needs: str) -> str:
    """The recorded reason when a cell degraded because `probe_endpoint` returned `None`.

    Kept apart from each caller's "this backend refused" message on purpose: those two
    sentences reduce a cell the same way and mean opposite things. "The organism's serving
    stack cannot do this" is a measured fact about that organism, and a reader comparing
    methods will read it as one; "we asked a model that is not an organism at all" is a
    fact about the command line.
    """
    return (
        f"{eval_name}: --model {get_model()} is not one of our organisms, so there is no "
        f"Endpoint to {needs} — nothing was asked and nothing refused. Select an organism "
        f"with -T spec_hash=<hash> to run this cell against one."
    )


@contextmanager
def relaxed_gc() -> Iterator[None]:
    """Suspend CPython's FULL garbage collections around a big Inspect run.

    Inspect keeps every sample, message and event of a run alive in memory until the log
    is closed, so the live object graph grows with the eval. CPython's generation-2
    collector walks that whole graph every time it fires — roughly once per 200k
    container allocations by default — so the collector's total work grows with the
    SQUARE of the eval's size, and it fires most often exactly when it is most expensive.
    mu is where that bites: at 15,000 comparison units the pathology is most of the
    framework overhead on both halves of the path. Measured on one A100, one organism,
    30,000 generations (2026-08-26):

    Verified end to end on an L40S, same pod / weights / organism / production width,
    `mu_decisiveness` at 15,000 rows, PR #136 tip vs this change (2026-08-26):

        measure                              chunked      + relaxed_gc
        mu cell (eval-seconds)                458.0s            361.0s   (1.27x)
        cell end to end, incl. engine start   669s              587s

    Correctness was read against a MEASURED control, not assumed: #136 run twice against
    itself agrees on 84.6% of scores and 75.0% of raw text (mean |dscore| 0.042), and
    #136-vs-this-change on 84.4% / 74.9% (0.042). mu draws at temperature 1.0, so ~85%
    IS its own reproducibility floor under continuous batching — this change is
    indistinguishable from re-running the unchanged path.

    Generations 0 and 1 keep running, so ordinary short-lived cycles are still reclaimed
    cheaply; only the full sweep is deferred. `gc.freeze()` additionally moves everything
    already alive (imports, the item bank, the plan) into a permanent generation that is
    never walked.

    NO `gc.collect()` ON THE WAY OUT, and that is not an oversight. Collection cost is
    proportional to the objects still ALIVE, so a sweep forced while the caller still
    holds the run's graph pays full price for it: measured at 68.5s on the pod and ~300s
    on a laptop, which was most of what deferring the sweeps had just saved. Restoring
    the thresholds is enough — a full collection happens on CPython's own schedule within
    a few hundred thousand allocations, by which time the graph is usually already gone
    by refcounting and the sweep is cheap. Deferring is strictly cheaper than forcing.

    NOT `gc.disable()`, which measured ~1.8x faster again on the read: with collection
    fully off, a reference cycle allocated inside the run is unreachable AND unreclaimed
    for the whole run, and this wraps calls that hold a multi-gigabyte object graph.
    Deferring the sweep bounds that at "until the run ends"; disabling it does not.
    """
    thresholds = gc.get_threshold()
    # The entry collect is for PRODUCTION (one big run per process: collect once, freeze
    # the survivors). Under pytest it fired once per eval against a monotonically growing
    # heap — measured 2026-08-30: 489 collects, 122s, 29% of the whole suite, and the
    # cause of the single-process gc stall. A/B with the collect no-op'd:
    # chunk walls -38%/-49%, identical pass counts. Skipped under pytest only; freeze()
    # still runs (freezing without the sweep is safe — just less precise).
    if "PYTEST_CURRENT_TEST" not in os.environ:
        gc.collect()
    gc.freeze()
    gc.set_threshold(thresholds[0], thresholds[1], _GC_GEN2_OFF)
    try:
        yield
    finally:
        gc.set_threshold(*thresholds)
        gc.unfreeze()


# Generation-2 collections happen every `threshold2` generation-1 collections. There is
# no "never", so this is a number large enough that a single eval never reaches it.
_GC_GEN2_OFF = 1_000_000


@contextmanager
def preserve_logging() -> Iterator[None]:
    """Put the root logger back how it was around an inspect `eval()` / `eval_set()`.

    Inspect configures logging **globally and permanently**: it sets the ROOT logger to
    its own `log_level` and adds a handler, and neither is undone when the call returns.
    We pass `log_level="warning"` (inspect's own INFO output is very chatty), so from the
    first eval onwards every `log.info` in this repo is silenced — the runner's per-cell
    "written" lines, the backend-group lines, the timing table and the final "done:"
    summary all vanish, and a long sweep prints nothing after its first group. Measured
    2026-08-24: root level INFO -> WARNING, handlers 1 -> 2, across one `eval()` call.

    Restoring the handler list matters as much as the level: leaving inspect's handler
    attached would print every subsequent record twice.
    """
    root = logging.getLogger()
    level, handlers = root.level, list(root.handlers)
    try:
        yield
    finally:
        root.setLevel(level)
        root.handlers[:] = handlers


def organism_temperature(config: Mapping[str, Any]) -> float | None:
    """The decode temperature for the ORGANISM's generation, or ``None`` to defer.

    ``serving.runner.eval_config_for`` injects ``temperature: 1.0`` for the behavioural
    axes and nothing otherwise, so a MISSING key means "use the target/serving default"
    (greedy today). Returning ``None`` here — never ``0.0`` — is what keeps a
    non-behavioural eval byte-identical to before temperature-by-axis existed: a Task that
    leaves ``GenerateConfig.temperature`` unset lets the provider fall back to the target's
    own default (``evals._inspect`` precedence), whereas an explicit ``0.0`` would OVERRIDE
    that default and silently clobber a non-zero ``complete_kwargs.temperature`` if one were
    ever set. Behavioural cells return the injected 1.0.
    """
    t = config.get("temperature")
    return None if t is None else float(t)


def target_sampling(complete_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """The sampling settings a ``complete_kwargs`` block asks for, for ``bind_endpoint``.

    The native harness passes the whole ``complete_kwargs`` dict into
    ``Endpoint.complete`` (``evals/_harness.py``); on the Inspect paths a target's
    generation goes through ``ResolvedTarget``, which carries a seed, a token cap and the
    two decoding knobs ``SamplingConfig`` has (``temperature``, ``top_p``) and nothing
    else - so anything else is warned about rather than silently dropped. Two runners
    honouring different sampling settings is exactly the shape that makes a score
    difference unattributable.

    Shared by both Inspect entry points (``serving.inspect_runner._CachingResolver`` and
    ``evals._task_eval.run_task_eval``) so they cannot disagree about the cap: with this
    unread on the eval-body path, every cell there generated at the serving default
    (512) while the config asked for 768 - and a judged response truncated mid-sentence
    scores the cap, not the organism (configs/specs.yaml, 2026-08-19).

    ``temperature`` / ``top_p`` joined the cap on 2026-09-04. Until then this helper
    WARNED them away, so every Inspect-path generation ran at the serving default
    (``temperature: 0.0``) whatever the eval config said, and a seed sweep could not
    produce sampling variance. Only keys present are returned, so an absent knob leaves
    the serving default untouched (the byte-identity gate is the proof of that).
    """
    kwargs = dict(complete_kwargs or {})
    carried: dict[str, Any] = {}
    raw = kwargs.pop("max_tokens", None)
    if raw is not None:
        carried["max_tokens"] = int(raw)
    for knob in ("temperature", "top_p"):
        value = kwargs.pop(knob, None)
        if value is not None:
            carried[knob] = float(value)
    if kwargs:
        log.warning(
            "complete_kwargs %s are ignored on the Inspect path (only max_tokens, "
            "temperature and top_p are carried); the native runner would apply them",
            sorted(kwargs),
        )
    return carried


# Reasoning allowance added to the eval's cap on a thinking-on turn that cannot or must not
# take the remaining context (see `EndpointModelAPI.generate`): a turn whose eval marked a
# deliberate per-turn cap (`_TURN_CAP_MARKER`: SWE-bench Pro, Petri), and a call with no bound
# context window (the direct `run_task_eval` path). 4096 keeps SWE-bench Pro's 4096 + this
# under compaction's 0.9 x 16,384 threshold with its ~3k prompt, so one reply cannot wedge it.
THINKING_ALLOWANCE_TOKENS = 4096


@lru_cache(maxsize=1)
def _warn_thinking_fallback() -> None:
    log.warning(
        "thinking-on organism served without a bound max_model_len: reasoning capped at "
        "the eval's max_tokens + %d instead of the remaining context (direct run_task_eval "
        "path). Use the eval_set runner for uncapped reasoning.",
        THINKING_ALLOWANCE_TOKENS,
    )


@lru_cache(maxsize=1)
def generation_limiter() -> anyio.CapacityLimiter:
    """Process-wide cap on threads running `Endpoint.complete` (see module docstring)."""
    return anyio.CapacityLimiter(int(os.environ.get("MO_EVALS_GEN_THREADS", "32")))


@modelapi(name=PROVIDER)
def mo_organism() -> type[ModelAPI]:
    """Provider registration. The indirection is inspect's convention (see its docs)."""
    return EndpointModelAPI


_WARNED_TOOL_FAMILY: set[str | None] = set()


def _tool_parser_family(base_model: str | None) -> str | None:
    """`base_model` if inspect's `HFHandler` has a parser branch for it, else None (warned once).

    Mirrors the name tests in inspect_ai 0.3.260 `hf_handler.model_specific_tool_parse`.
    None means today's fallback, the convention-sniffing `parse_unknown_tool_calls`.
    """
    name = (base_model or "").lower()
    if (
        "llama" in name
        or "mistral" in name
        or ("qwen" in name and ("coder" in name or "instruct" in name))
    ):
        return base_model
    if base_model not in _WARNED_TOOL_FAMILY:
        _WARNED_TOOL_FAMILY.add(base_model)
        log.warning(
            "no inspect tool-call parser family for base model %r; tool calls are parsed by "
            "convention sniffing, which reads a ```json block before <tool_call>",
            base_model,
        )
    return None


class EndpointModelAPI(ModelAPI):
    """One of our organisms, wearing an inspect model's clothes.

    TOOL CALLING IS TEXT-BASED, and the limitation that used to be recorded here ("NO
    TOOL CALLING… a future eval that needs them needs a decision") WAS LIFTED
    2026-08-29 with the project lead's sign-off, for the agentic evals where the
    ORGANISM is the agent. `Endpoint` is unchanged in shape: the tool schemas ride out
    as an optional `tools=` kwarg on `complete`, which a backend hands to its chat
    template (`serving/vllm_backend.py`), and the call comes back **in the completion
    text** — `<tool_call>{...}</tool_call>` and friends — parsed by inspect's own
    `HFHandler`, the parser it ships for exactly this. So no eval learns how the
    organism was built, and nothing method-specific entered the protocol.

    What this is NOT: a native tool-calling API. There is no `tool_choice` enforcement
    (warned about below) and no structured tool-call channel, because the organisms are
    served as raw completions.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        # `GenerateConfig | None` rather than the base class's `GenerateConfig()`
        # default, which ruff rejects as a call in an argument default (B008). Inspect
        # always passes one, so the fallback is only for direct construction in tests.
        config: GenerateConfig | None = None,
        **model_args: Any,
    ) -> None:
        super().__init__(model_name, base_url, api_key, [], config or GenerateConfig())
        bound = _BOUND.get(model_name)
        resolver = _RESOLVER
        if bound is None and resolver is not None:
            # OFF THE EVENT LOOP, not merely off the loop's critical path: Inspect builds
            # a task inside its own running loop, and an engine constructed under one
            # binds its output handler to it and then hangs (see `off_event_loop`).
            bound = off_event_loop(lambda: resolver(model_name))
        # UNBOUND IS NOT AN ERROR HERE, ONLY IN `generate`. `inspect score` rebuilds the
        # primary model from the log header before it re-scores (inspect_ai/_eval/score.py
        # `score_async`), and a log header naming `mo_organism/<hash>` therefore
        # constructs this class in a process where nothing is bound — so raising in
        # `__init__` made every stored log un-re-scorable, with a message about binding an
        # endpoint for a run that generates nothing (measured 2026-08-27). Deferring costs
        # nothing: a real generation still fails, with the same message, at the first call.
        self.endpoint = bound.endpoint if bound is not None else None
        self.seed = bound.seed if bound is not None else 0
        # NOT `self.max_tokens`. `ModelAPI.max_tokens()` is a METHOD on the base
        # class, and inspect calls it on every generate (`config.max_tokens =
        # self.api.max_tokens()`). Shadowing it with an int raises "'int' object is
        # not callable" INSIDE the target call, which inspect swallows: the audit
        # then completes "successfully" with zero target turns and empty
        # `raw_output`. Cost an hour on 2026-08-19. Never name an attribute here
        # without checking it against the base class first. The METHOD is overridden
        # below, which is the supported way to declare this cap.
        self._max_output_tokens = bound.max_tokens if bound is not None else None
        self._temperature = bound.temperature if bound is not None else None
        self._top_p = bound.top_p if bound is not None else None
        self._max_model_len = bound.max_model_len if bound is not None else None
        self._prompt_token_count = bound.prompt_token_count if bound is not None else None
        self._base_model = bound.base_model if bound is not None else None
        _advertise_context_window(model_name, self._max_model_len)
        # Every (input, output) pair, in call order. Kept here as well as read back
        # out of the eval log so a target generation is recorded even when the sample
        # later errors and inspect writes no events for it.
        self.calls: list[dict[str, Any]] = []
        # Connection-pool scope. Every organism shares "default", on purpose (see the width
        # levelling in serving/inspect_runner.py); only a `pin_width` cell is handed its own
        # `pool` there, so its narrow width cannot size the shared pool for the group.
        self._pool = str(model_args.get("pool", "default"))

    @override
    def connection_key(self) -> str:
        """The pool this model's generations queue on (inspect's `max_connections` scope)."""
        return self._pool

    # `@override` is not decoration for its own sake: `invalid-explicit-override` is
    # error-by-default in ty, so if a pinned-inspect_ai bump renames or drops
    # `ModelAPI.max_tokens`, the gate fails here instead of this silently becoming a
    # method nothing calls. Same for `generate` below.
    @override
    def max_tokens_for_config(self, config: GenerateConfig) -> int | None:
        """Keep stock no-limit calls distinct from the provider's bound default."""
        if (config.extra_body or {}).get(_STOCK_MAX_TOKENS_MARKER) is True:
            return _STOCK_REMAINING_CONTEXT
        return None

    @override
    def max_tokens(self) -> int | None:
        """The bound output cap, as inspect's own default-max_tokens hook.

        `Model.generate` fills an unset `config.max_tokens` from here
        (inspect_ai/model/_model.py: `config.max_tokens = self.api.max_tokens()`), which
        means the cap ends up in the config Inspect RECORDS as well as in the request.
        Before this override the log said `max_tokens: None` on every cell while 768 was
        actually applied inside `generate` — a run whose recorded config could not be
        used to reproduce it.
        """
        return self._max_output_tokens

    def _resolve_from_config(self) -> None:
        """Bind this organism from ``$MO_EVALS_CONFIG`` (default ``configs/specs.yaml``).

        The shell-CLI path: ``inspect eval ... --model mo_organism/<hash>`` reaches
        `generate` with nothing bound and no resolver installed. A hash the config does
        not name leaves the endpoint unbound (the caller raises with the full message);
        any other failure — a missing artifact, no GPU — propagates as itself. The
        factory arrives via `set_config_resolver_factory`, never a `serving` import.

        "THE CONFIG DOES NOT NAME THIS HASH" IS ASKED, NOT INFERRED FROM AN EXCEPTION.
        This used to be `except KeyError: return`, which also swallowed every `KeyError`
        raised *inside* `resolve()` — and `serving/resolve.py` indexes `method_config`, so
        a spec missing an adapter uri came back as "unknown organism" and the run died
        later with a message about naming a spec_hash that is right there in the config.
        A resolver that can answer the containment question does; anything else has its
        own lookup error propagate.
        """
        global _CONFIG_RESOLVER
        if _CONFIG_RESOLVER_FACTORY is None:
            return
        with _CONFIG_RESOLVER_LOCK:
            if _CONFIG_RESOLVER is None:
                _CONFIG_RESOLVER = _CONFIG_RESOLVER_FACTORY()
        # Duck-typed rather than a declared protocol: `evals/` may not import `serving/`,
        # which is where the resolver is built (scripts/check_import_boundaries.py).
        resolver = _CONFIG_RESOLVER
        knows = getattr(resolver, "__contains__", None)
        if knows is not None and not knows(self.model_name):
            return
        # The `endpoint_for` caller runs this at TASK-BUILD time, which on the shell-CLI
        # path is inside Inspect's own event loop — the engine must not be constructed
        # under one (`off_event_loop`). A no-op on `generate`'s path, which is already in
        # an `anyio` worker thread.
        bound = off_event_loop(lambda: resolver(self.model_name))
        self.endpoint = bound.endpoint
        self.seed = bound.seed
        self._max_output_tokens = bound.max_tokens
        self._temperature = bound.temperature
        self._top_p = bound.top_p
        self._max_model_len = bound.max_model_len
        self._prompt_token_count = bound.prompt_token_count
        self._base_model = bound.base_model
        _advertise_context_window(self.model_name, self._max_model_len)

    @override
    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> tuple[ModelOutput, ModelCall]:
        if self.endpoint is None:
            # Shell-CLI fallback (2026-08-29): `inspect eval ... --model mo_organism/<hash>`
            # constructs this provider in a process where nothing is bound and no sweep
            # installed a resolver. Resolve from $MO_EVALS_CONFIG at FIRST GENERATE, not in
            # __init__ — `inspect score` rebuilds the model from a log header and never
            # generates, and resolving there would load an engine just to re-judge.
            #
            # OFF THE EVENT LOOP. `resolve()` loads a vLLM engine, which is minutes of
            # blocking work; run inline it stalls the whole loop, so every other sample's
            # `attempt_timeout` burns down against an engine start rather than a model.
            await anyio.to_thread.run_sync(self._resolve_from_config)
        if self.endpoint is None:
            raise ValueError(
                f"no endpoint for {PROVIDER}/{self.model_name}; bind one with "
                "evals._inspect.bind_endpoint, install a resolver via set_resolver, or "
                "name a spec_hash from $MO_EVALS_CONFIG / configs/specs.yaml"
            )
        # The parser is chosen by the BASE MODEL's family, never by `model_name` (a
        # spec_hash, which matches no family). Unmatched, inspect falls to
        # `parse_unknown_tool_calls`, which checks ```json BEFORE `<tool_call>`: a fenced
        # `<tool_call>` parses to nothing and an unrelated ```json block is read as a call.
        # The 7B steering organism fenced every call on swe_bench_pro and looped to the
        # message limit (2026-09-15) while base, unfenced, parsed — a parser artefact
        # posing as a method difference. Keyed on base_model, so every method on one base
        # model gets one parser; the Qwen-instruct branch finds `<tool_call>` anywhere.
        family = _tool_parser_family(self._base_model) if tools else None
        handler = HFHandler(self.model_name, family) if tools else None
        if handler is not None and tool_choice not in ("auto", "none"):
            # Cannot be enforced on a text-parsed organism: there is no request field to
            # set. Named rather than dropped — an eval that pinned `tool_choice` and got
            # a chatty non-answer would otherwise read as a finding about the organism.
            log.warning(
                "%s: tool_choice=%r cannot be enforced on a text-parsed organism; the "
                "tools are offered but calling them is left to the model",
                self.model_name,
                tool_choice,
            )
        messages = [_endpoint_message(m) for m in input]
        seed = self._seed_for(messages, config)
        kwargs: dict[str, Any] = {"seed": seed}
        extra_body = dict(config.extra_body or {})
        stock_task = extra_body.pop(_STOCK_MAX_TOKENS_MARKER, False)
        turn_cap = extra_body.pop(_TURN_CAP_MARKER, False) is True
        if not isinstance(stock_task, bool):
            raise ValueError(f"invalid stock max-token marker {stock_task!r}")
        stock_token_source = None
        if stock_task:
            stock_token_source = (
                "remaining_context"
                if config.max_tokens == _STOCK_REMAINING_CONTEXT
                else "upstream"
            )
        if handler is not None:
            # `exclude_none` matches what inspect's own HF provider sends to
            # `apply_chat_template(tools=...)`; the chat templates just `tojson` each entry.
            kwargs["tools"] = [t.model_dump(exclude_none=True) for t in tools]
        # `config.max_tokens` usually already carries the bound cap — inspect fills an
        # unset one from `max_tokens()` above before calling us.
        if stock_token_source != "remaining_context" and config.max_tokens is not None:
            kwargs["max_tokens"] = config.max_tokens
        elif stock_token_source != "remaining_context" and self._max_output_tokens is not None:
            # THE FIRST-WAVE CLI CASE. `config.max_tokens` is filled from `max_tokens()`
            # BEFORE generate, and on the shell-CLI path this provider is still unbound
            # then — so every sample that entered before the resolve above landed ran
            # UNCAPPED while the rest ran at the organism's cap. Apply the cap we now know.
            # KNOWN AND DOCUMENTED: the LOG's recorded config still says `max_tokens: null`
            # for those samples, because inspect recorded it before we had an answer.
            kwargs["max_tokens"] = self._max_output_tokens
        # THE THINK BLOCK IS OPENED BY THE PROMPT, not by the model. With
        # `method_config.enable_thinking` the Qwen3-family template ends the generation
        # prompt `…<think>\n`, so the completion begins INSIDE the block and carries only
        # the CLOSING tag — measured on the 27B smoke, `<think>` appeared in 0 of 180 rows
        # (parked in PR #370). Both the cap
        # below and the
        # split further down are gated on this one flag.
        #
        # A PREFILLED assistant turn is the exception, and the test for one is the same
        # predicate every backend renders by (`serving.vllm_backend.is_prefill`) — copied
        # rather than imported because `evals/` may not import `serving/`
        # (`scripts/check_import_boundaries.py`) and `core/` carries only the contracts.
        # Those render with
        # `continue_final_message=True` / `add_generation_prompt=False`, and the template
        # emits `<think>\n` only on the generation-prompt branch: the same smoke measured a
        # thinking-ON `mu_decisiveness` draw coming back as `'\nB$6</answer>'`, byte-shaped
        # like the thinking-off arm. Reading those as reasoning would empty every
        # packed-bridge row on a thinking-on arm.
        thinking_open = is_thinking_on(self.endpoint) and not (
            messages and messages[-1].get("role") == "assistant"
        )
        # A thinking-on organism is served as people use one: the reasoning is bounded
        # only by the REMAINING CONTEXT, like a stock task with no upstream limit (design decision,
        # 2026-09-23: "no point testing a hampered model"). The old default — eval cap +
        # 2048 — was exhausted on 8/10 fermi and 8/20 introspection thinking-on rows
        # (the 27B uncapped smoke, parked in PR #370), which made the cap
        # the
        # measurement. `meta['thinking_truncated']` still records a
        # generation that ran out of context mid-thought; that is now the model's own
        # behaviour, not one we imposed. An explicit `method_config.thinking_budget` keeps
        # the capped form, eval cap + budget (the cap is the ANSWER's budget; the reasoning
        # is spent first, so the two are added). LIMITATION of that form: it caps the PAIR,
        # not each half.
        #
        # A DELIBERATE PER-TURN CAP (`_TURN_CAP_MARKER`, set by the eval) keeps that cap plus
        # THINKING_ALLOWANCE_TOKENS instead (2026-09-23). SWE-bench Pro's 4096 exists so
        # one reply cannot outgrow what `CompactionEdit` can trim (it drops whole messages);
        # the remaining context on turn 1 would reintroduce exactly that wedge (Codex, d4c4068
        # claim 4). Petri's 1024 bounds a 10-turn target conversation that has no compaction
        # and no tools (`target_tools="none"`), so "tools offered" cannot be the test (Codex,
        # 9714fba). Unmarked multi-turn CHAT cells (robustness_critique_gen /
        # _multiturn_debate_gen, the persistence drift loops, misaligned_sycophancy) stay
        # uncapped on purpose: their caps size the ANSWER on a thinking-off organism, and a
        # later turn's prompt carries only earlier answers (`_endpoint_message` sends
        # `message.text`, which excludes `ContentReasoning`), so an uncapped turn grows the
        # history by its answer alone.
        # The drift loops also evict by token budget (`evals._harness.evict_for_budget`).
        token_source = stock_token_source
        if thinking_open:
            budget = getattr(self.endpoint, "thinking_budget", None)
            if budget is None and turn_cap and kwargs.get("max_tokens") is not None:
                kwargs["max_tokens"] += THINKING_ALLOWANCE_TOKENS
                token_source = "thinking_agent_turn_cap"
            elif budget is None:
                if self._max_model_len is None or self._prompt_token_count is None:
                    # A caller that binds no context window (the direct `run_task_eval`
                    # path, persistence's direct functions) cannot use the remaining
                    # context; failing every sample there lost them silently as empty NaN
                    # rows (Codex, d4c4068). Fall back to the old capped form, loudly.
                    # With no eval cap either there is nothing to add to: the call stays
                    # uncapped, as it always was, and no token source is recorded.
                    if kwargs.get("max_tokens") is not None:
                        _warn_thinking_fallback()
                        kwargs["max_tokens"] += THINKING_ALLOWANCE_TOKENS
                        token_source = "thinking_fallback_cap"
                elif token_source != "remaining_context":
                    token_source = "thinking_remaining_context"
            elif kwargs.get("max_tokens") is not None:
                kwargs["max_tokens"] += int(budget)
        # The bound values are the eval config's `complete_kwargs` (`target_sampling`),
        # applied only when the task / CLI set nothing - the cap's precedence, above.
        # ponytail: unlike the cap there is no `max_tokens()`-style hook, so a bound
        # value is absent from the GenerateConfig inspect RECORDS; the per-call
        # ModelCall request below does carry it.
        if config.temperature is not None:
            kwargs["temperature"] = config.temperature
        elif self._temperature is not None:
            kwargs["temperature"] = self._temperature
        if _REPEAT.get() != _FIRST_REPEAT and not kwargs.get("temperature"):
            _warn_greedy_repeat()
        # Sampling knobs vLLM supports and we previously dropped in silence, which made
        # `--top-p` / `--top-k` look honoured and do nothing. Forwarded by their own
        # names, as `SamplingConfig.kwargs` takes them; a backend that cannot take one
        # refuses it loudly rather than ignoring it (`serving/hf_backend.py`).
        if config.top_p is not None:
            kwargs["top_p"] = config.top_p
        elif self._top_p is not None:
            kwargs["top_p"] = self._top_p
        if config.top_k is not None:
            kwargs["top_k"] = config.top_k
        if config.prompt_logprobs is not None:
            # Teacher-forced logprobs over the PROMPT, which is what inspect's stock
            # `perplexity()` / `target_perplexity()` scorers read. Forwarded under its own
            # name, as `Endpoint.complete_full` takes it; a backend that cannot produce it
            # refuses loudly (serving/hf_backend.py; ie serves it since 2026-08-29) rather than
            # returning None, because a None here is indistinguishable from never asking.
            kwargs["prompt_logprobs"] = config.prompt_logprobs
        if config.stop_seqs:
            # `include_stop_str_in_output` travels WITH `stop`, never without it: vLLM
            # strips the stop string by default, and a mu draw missing its `</answer>`
            # falls through to the lenient scan — which is how a refusal gets read as a
            # vote (#72). Keeping it turns a speed flag back into a pure optimisation.
            kwargs["stop"] = list(config.stop_seqs)
            kwargs["include_stop_str_in_output"] = True
        if extra_body:
            # Inspect's own provider-passthrough channel, and the ONLY way a task asks
            # this endpoint for something `GenerateConfig` has no field for — today that
            # is `structured_outputs` for the constrained-decoding arms (evals/friedness,
            # evals/capability). Forwarded verbatim into `Endpoint.complete`, where each
            # backend decides: vLLM coerces it, `HFEndpoint` refuses it loudly, and
            # `FakeEndpoint` ignores it. NOT a general escape hatch for sampling settings
            # — those have GenerateConfig fields and must go through them, or the two
            # runners silently honour different ones (`target_sampling`).
            kwargs.update(extra_body)
        _warn_unforwarded(config)

        started = perf_counter()
        # Bound to a local: the `is None` guard above cannot narrow `self.endpoint`
        # inside a closure, and the lambda is what runs on the worker thread.
        endpoint = self.endpoint
        completion = await anyio.to_thread.run_sync(
            lambda: _generate(
                endpoint,
                messages,
                kwargs,
                use_remaining_context=token_source
                in ("remaining_context", "thinking_remaining_context"),
                max_model_len=self._max_model_len,
                prompt_token_count=self._prompt_token_count,
            ),
            limiter=generation_limiter(),
        )
        text = completion.text
        # A thinking organism (`method_config.enable_thinking`, Qwen3-family) emits
        # `<think>…</think>\n\nanswer`. The think text becomes a `ContentReasoning` part
        # BESIDE the answer rather than part of it, because `ChatMessageBase.text` joins
        # only TEXT parts: `output.completion` is then the answer alone, so every scorer,
        # every judge and every `raw_output` sees the answer with no scorer change.
        # Keeping `raw_output` answer-only is load-bearing downstream, not tidiness:
        # `evals/_harness.py` and `evals/misaligned.py` RE-JUDGE from the stored string,
        # and `core.endpoint.is_fake_output` matches it with `startswith`. Nothing is
        # lost — the `.eval` log stores the reasoning part verbatim and the bridges put it
        # in `meta['thinking']`. Non-thinking output is returned untouched, whitespace
        # included, which is what keeps the prefilled-assistant (`continue_final_message`)
        # cells byte-identical.
        truncated = False
        # The split is keyed on the delimiters the ENDPOINT reports (`reasoning_delimiters`,
        # derived from its tokenizer's own chat template: `core/reasoning.py`), so a Qwen3
        # `<think>`/`</think>` organism and a Gemma 4 `<|channel>thought…<channel|>` organism
        # take the same path. An endpoint that reports none (FakeEndpoint, a stub) gets the
        # original Qwen literals, which is what keeps every existing row byte-identical.
        # `serving.resolve` refuses to serve a base thinking-ON when no delimiters can be
        # derived, so a thinking-on completion never reaches here unsplittable. Gemma's tags
        # are SPECIAL tokens: the vLLM backend keeps them on decode (`skip_special_tokens=
        # False`) for a thinking-ON organism only. That end-to-end path is UNVERIFIED on a real
        # engine (tokenizer-level tests only; docs/reference/base-models.md).
        if thinking_open:
            # Opened by the prompt (Qwen3): parsed as if the completion opened with the tag,
            # because it did. Split at the FIRST closing tag: reasoning before it, answer
            # after, less the template's own `\n\n`. No closing tag at all means the cap ran
            # out mid-thought, so there is no answer in the string; `""` plus
            # `thinking_truncated` says that, where scoring the deliberation does not.
            # An endpoint with NO such attribute (FakeEndpoint, test doubles) keeps the Qwen
            # literals. One that has it and reports None was thinking-on with a template that
            # yields no tags: `serving.resolve` refuses that earlier, and this refuses again
            # rather than score reasoning as the answer.
            delims = getattr(self.endpoint, "reasoning_delimiters", QWEN_THINK)
            if delims is None:
                raise SystemExit(
                    f"{self.model_name}: thinking-on endpoint reports no reasoning delimiters "
                    "(core/reasoning.py); refusing to split on a guessed tag."
                )
            # Whether THIS prompt ended inside an open block is the endpoint's to say (Gemma
            # opens it after a tool response only); None keeps the base default.
            opens_fn = getattr(self.endpoint, "reasoning_opens_in_prompt", None)
            opens = opens_fn(messages, tools=kwargs.get("tools")) if callable(opens_fn) else None
            thought, answer, truncated = delims.split(text, opens)
            reasoning = None if thought is None else ReasoningCapsule(reasoning=thought)
        # Thinking-OFF: never split, whatever the text looks like. The organism's own
        # setting decides (06108b4), not the string: a 7B quoting a literal `<think>` —
        # as the robustness and introspection banks invite — would otherwise have that
        # span excised from `raw_output` and filed as reasoning it never did, and a
        # thinking-OFF adapter that emits think tags anyway (the `sft2900-fixed` 27B does
        # `</think>` in 9/30 rows on the 27B smoke) is evidence about the
        # adapter, so it stays in `raw_output` where a reader will meet it.
        else:
            answer, reasoning = text, None
        values: list[float] | None = None
        if config.logprobs:
            # A SECOND generation, and there is no way around it here: `Endpoint` has
            # `complete` and `logprobs` as separate calls and neither returns the other's
            # result (CLAUDE.md), so asking for logprobs doubles the GPU cost of the
            # cell. Off by default, so nothing pays for it unless `--log-probs` is passed.
            log.warning(
                "%s: --log-probs draws a SECOND generation per sample (Endpoint.complete "
                "and Endpoint.logprobs are separate calls); the logprobs are of that "
                "second draw, and match the first only under greedy decoding",
                self.model_name,
            )
            # `prompt_logprobs` is `complete_full`'s kwarg, not this call's: every backend
            # that implements `Endpoint.logprobs` refuses it (ValueError on ie,
            # NotImplementedError on HF), so `--log-probs` together with a perplexity-style
            # config crashed here before it was stripped. `tools` goes for the same reason
            # and one step harder: it is a CHAT-TEMPLATE argument, and no backend's
            # `logprobs` renders one (HF raises, ie refuses the unknown key), so
            # `--log-probs` on a tools-offered cell crashed on the second call.
            lp_kwargs = {k: v for k, v in kwargs.items() if k not in ("prompt_logprobs", "tools")}
            values = await anyio.to_thread.run_sync(
                lambda: endpoint.logprobs(messages, **lp_kwargs),
                limiter=generation_limiter(),
            )
        elapsed = perf_counter() - started

        # The ANSWER, so this fallback log agrees with the model events the log readers
        # read first (`e.output.completion`, answer-only by the split above).
        self.calls.append({"messages": messages, "output": answer, "seed": seed})
        # Real numbers when the endpoint measured them, chars/4 estimates when it did not,
        # and the output says WHICH on every call. Both flags used to be unconditionally
        # True — `Endpoint.complete` returned a bare string, so there was nothing else
        # they could be — and `Endpoint.complete_full` is the channel that fixed that
        # (core/endpoint.py, added 2026-08-27). An unmarked estimate is the thing that
        # gets quoted, so the marks stay on the paths that are still estimating.
        usage = _usage(completion)
        stop_reason = _as_stop_reason(completion.stop_reason)
        output = ModelOutput.from_content(
            self.model_name,
            answer,
            # `text`, not `answer`, on both estimates: the think tokens were generated and
            # paid for, so they count towards the cap and towards usage.
            stop_reason=(
                stop_reason
                if stop_reason is not None
                else _stop_reason(text, kwargs.get("max_tokens"))
            ),
        )
        if reasoning is not None:
            output.choices[0].message.content = [
                ContentReasoning(
                    reasoning=reasoning.reasoning,
                    signature=reasoning.signature,
                    redacted=reasoning.redacted,
                    summary=reasoning.summary,
                    internal=reasoning.internal,
                ),
                ContentText(text=answer),
            ]
        output.usage = usage if usage is not None else _estimated_usage(messages, text)
        output.metadata = {
            "usage_estimated": usage is None,
            "stop_reason_estimated": stop_reason is None,
            # Present only when it happened, so a cell that never truncates is unchanged.
            # The bridges lift it into the row's `meta` (`_thinking_meta`).
            **({"thinking_truncated": True} if truncated else {}),
        }
        if token_source is not None:
            output.metadata.update(
                effective_max_tokens=kwargs["max_tokens"],
                effective_max_tokens_source=token_source,
            )
        if handler is not None:
            # Swapped in ONLY when a call was actually parsed, so a tools-offered turn
            # that just answers keeps the message `from_content` built — same object as
            # on the no-tools path, which is what keeps existing cells byte-identical.
            # The answer: a tool call the organism only mused about inside `<think>` is
            # not a tool call. `parsed` is built from the ANSWER only, so it carries no
            # reasoning part; prepend the one split off above, or a thinking organism's
            # tool-calling turn loses its reasoning from the scored message (it would then
            # survive only in the `ModelCall` below, which inspect retains for just the
            # first few draws).
            parsed = handler.parse_assistant_response(answer, tools)
            if parsed.tool_calls:
                if reasoning is not None:
                    body = (
                        parsed.content
                        if isinstance(parsed.content, list)
                        else ([ContentText(text=parsed.content)] if parsed.content else [])
                    )
                    parsed.content = [
                        ContentReasoning(
                            reasoning=reasoning.reasoning,
                            signature=reasoning.signature,
                            redacted=reasoning.redacted,
                            summary=reasoning.summary,
                            internal=reasoning.internal,
                        ),
                        *body,
                    ]
                output.choices[0].message = parsed
                output.choices[0].stop_reason = "tool_calls"
        _set_prompt_logprobs(output, completion, config, self.model_name)
        if values is not None:
            # `Endpoint.logprobs` returns bare floats — no token strings, no bytes, no
            # alternatives — so `token` is empty and `top_logprobs` is None. Enough for
            # a mean-logprob read; NOT enough for inspect's stock `perplexity()` scorers,
            # which read `prompt_logprobs` (teacher-forced over the PROMPT) and have no
            # route through this protocol at all.
            output.choices[0].logprobs = Logprobs(
                content=[Logprob(token="", logprob=v, bytes=None) for v in values]
            )
        # NOT superseded by inspect's `--log-model-api` (0.3.260), checked rather than
        # assumed. That flag is a RETENTION POLICY over `ModelEvent.call`, not a producer of
        # one: `log/_transcript.py:_process_event` keeps, drops or rate-limits whatever the
        # provider already built. This IS that object — an in-process `Endpoint` has no HTTP
        # exchange for inspect to capture, so with this line gone there is simply nothing to
        # retain. Worth knowing the default (`None`) keeps only the first
        # `DEFAULT_LOG_MODEL_API_CALLS` (5) per model plus errors, so a log holds a sample of
        # these rather than all of them; nothing of ours reads them back, so that is left as
        # it is rather than forced to True and paid for in log size.
        call = ModelCall.create(
            request={"messages": messages, **kwargs},
            response={"completion": text, **({"logprobs": values} if values else {})},
            time=elapsed,
        )
        return output, call

    def _seed_for(self, messages: list[dict[str, Any]], config: GenerateConfig) -> int:
        """The seed for one generation: the task's, or the conversation's, XOR the run's.

        Three sources, and they compose rather than overwrite:

        * **the conversation** — the default. Derived from the CONVERSATION, not from a
          call counter: samples run concurrently through one shared provider, so a
          counter would interleave differently on every run and the same conversation
          would not replay.
        * **the task** — ``model.generate(config=GenerateConfig(seed=...))``. mu draws n
          completions from ONE conversation under n derived seeds, which no conversation
          hash can express, and the capability/MCQ draws do the same.
        * **the run** — inspect's ``--seed`` / ``eval(seed=)``, which arrives merged into
          ``config.seed`` for the active model. It is XORed into whichever of the two
          above applies, so it SHIFTS THE WHOLE SWEEP to a fresh but reproducible set of
          draws. It used to be taken as the seed itself, which is wrong twice over: on a
          judge-free eval the task's own per-draw seed won and ``--seed`` did nothing at
          all, and on the rest every conversation in the cell collapsed onto one seed.

        The one ambiguous case, stated because it is real: a task that asks for exactly
        the run seed is indistinguishable here from a task that asks for nothing, and is
        treated as the latter. The result is still deterministic and reproducible.

        A run seed of ``0`` is a no-op and is treated as no run seed at all: it is XORed
        into nothing, and it is not warned about.
        """
        run_seed = active_generate_config().seed
        task_seed = config.seed if config.seed != run_seed else None
        # The conversation seed also folds in WHICH REPEAT this is (`repeat_scope`): two
        # repeats of one item send the same conversation, so without it they drew the
        # same seed and `n_samples: 3` bought one answer three times. Repeat 0 of epoch 1
        # hashes exactly as before.
        base = (
            task_seed
            if task_seed is not None
            else self.seed * 1_000_003 + _conversation_seed(messages, repeat=_REPEAT.get())
        )
        # `run_seed == 0` is spelled out rather than folded into a falsy check, because
        # a falsy check on an `int | None` is the very confusion this line exists to undo:
        # zero is a SET seed that happens to fold to nothing (`base ^ 0 == base`), so the
        # rows it produces are byte-identical to an unseeded run's. Warning on it told an
        # operator their results were not poolable when they were the same numbers —
        # observed on a real sweep, 2026-09-01, from `--seed 0` / `INSPECT_EVAL_SEED=0`.
        if run_seed is None or run_seed == 0:
            return base
        _warn_seed_fold(run_seed)
        return base ^ run_seed


# Sampling knobs this provider forwards into `Endpoint.complete`.
_FORWARDED_CONFIG = frozenset(
    {
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop_seqs",
        "seed",
        "extra_body",
        "logprobs",
        "prompt_logprobs",
    }
)
# Knobs inspect consumes itself, around the provider call rather than inside it. Nothing
# to forward, and nothing to warn about.
_INSPECT_OWNED_CONFIG = frozenset(
    {
        "max_retries",
        "timeout",
        "attempt_timeout",
        "max_connections",
        "adaptive_connections",
        "cache",
        "cache_prompt",
        "batch",
        "fallback_models",
        "max_tool_output",
        "parallel_tool_calls",
        "internal_tools",
        "extra_headers",
        "modalities",
    }
)

_warned_config_fields: set[str] = set()
_warned_seed_folds: set[int] = set()


def _warn_unforwarded(config: GenerateConfig) -> None:
    """Say once which set GenerateConfig fields this provider is dropping.

    Everything not in `_FORWARDED_CONFIG` or `_INSPECT_OWNED_CONFIG` reaches
    `Endpoint.complete` never — `num_choices`, `frequency_penalty`, `logit_bias`,
    `prompt_logprobs`, `response_schema` and the reasoning knobs among them. Silently
    ignoring a sampling setting is how two runners come to honour different ones
    (`target_sampling`), so the drop is stated rather than assumed.
    """
    dropped = {
        name
        for name in GenerateConfig.model_fields
        if name not in _FORWARDED_CONFIG
        and name not in _INSPECT_OWNED_CONFIG
        and getattr(config, name, None) is not None
    }
    new = dropped - _warned_config_fields
    if new:
        _warned_config_fields.update(new)
        log.warning(
            "GenerateConfig %s reach no Endpoint: %s forwards only %s (plus extra_body). "
            "A setting that looks honoured and is not makes a score difference "
            "unattributable",
            sorted(new),
            PROVIDER,
            sorted(_FORWARDED_CONFIG),
        )


def _endpoint_message(message: ChatMessage) -> dict[str, Any]:
    """One inspect message as the plain dict an `Endpoint` — and a chat template — takes.

    `core.endpoint.Message = dict[str, str]` is an unenforced alias, so the tool fields
    ride through as extra keys, which is what `apply_chat_template` reads. Shapes are
    transformers' documented ones rather than OpenAI's: `arguments` is a **dict**, not a
    JSON string, because the templates render it with `{{ tool_call.arguments | tojson }}`.

    A plain turn produces exactly `{"role", "content"}` — the dict this was before tool
    support — so the no-tools path every existing eval takes does not move.

    Dropping the tool fields is not a cosmetic loss: an agent handed back a conversation
    with its own tool call and the result missing calls the tool again, and again, until
    the sample hits its message limit.
    """
    out: dict[str, Any] = {"role": message.role, "content": message.text}
    if isinstance(message, ChatMessageAssistant) and message.tool_calls:
        out["tool_calls"] = [
            {"type": "function", "function": {"name": c.function, "arguments": c.arguments}}
            for c in message.tool_calls
        ]
    elif isinstance(message, ChatMessageTool):
        if message.error is not None:
            # `Error: …` rather than empty, matching inspect's own
            # `ChatAPIHandler.tool_message`: an error the model cannot see is one it
            # cannot recover from, and it retries the same broken call instead.
            out["content"] = f"Error: {message.error.message}"
        if message.function:
            out["name"] = message.function
        if message.tool_call_id:
            out["tool_call_id"] = message.tool_call_id
    return out


_greedy_repeat_warned = False


def _warn_greedy_repeat() -> None:
    """Say once per process that repeats are being generated with no temperature.

    A repeat now gets its own seed (`repeat_scope`), but a seed changes nothing under
    greedy decoding, and the serving default is `temperature: 0.0` in every
    configs/serving*.yaml. A warning rather than a refusal: a determinism check may want
    identical repeats on purpose.
    """
    global _greedy_repeat_warned
    if _greedy_repeat_warned:
        return
    _greedy_repeat_warned = True
    log.warning(
        "repeat samples (n_samples > 1 or epochs > 1) are generating with temperature "
        "unset or 0; under greedy decoding every repeat is identical whatever its seed, so "
        "the extra generations add no information. Set complete_kwargs.temperature to "
        "sample"
    )


def _warn_seed_fold(run_seed: int) -> None:
    """Say once PER SEED VALUE that the run seed is shifting the sweep.

    Keyed on the value like `_warned_config_fields`, not on a bare "have we warned yet"
    flag: one process runs many cells (`eval_set`), so a second, DIFFERENT run seed can
    follow the first — and that is precisely the case the operator most needs told, since
    two mutually non-poolable seeds are then landing in one results tree.
    """
    if run_seed in _warned_seed_folds:
        return
    _warned_seed_folds.add(run_seed)
    log.warning(
        "run seed %d is XORed into every per-draw seed, shifting the whole sweep. Rows "
        "produced under it are reproducible but NOT comparable with rows produced "
        "without it — do not pool them",
        run_seed,
    )


def _generate(
    endpoint: Endpoint,
    messages: list[dict[str, Any]],
    kwargs: dict[str, Any],
    *,
    use_remaining_context: bool = False,
    max_model_len: int | None = None,
    prompt_token_count: PromptTokenCount | None = None,
) -> Completion:
    """One generation, with the endpoint's own telemetry when it has any.

    PRESENCE IS THE SWITCH, and deliberately not a trial call. `complete_full` is optional
    on the protocol (core/endpoint.py), so an endpoint written before it — a test double,
    someone's own `Endpoint` — simply does not have the attribute and gets the estimate
    path. An endpoint that HAS it and raises is a different thing entirely: that is a
    configuration error (asking a transformers-served organism for `prompt_logprobs`, say)
    and it is left to propagate, because silently falling back would spend a second
    generation to hide a message that names the fix.

    `kwargs` reaches BOTH paths unchanged, `prompt_logprobs` included. A legacy endpoint
    asked for prompt logprobs therefore just ignores the argument, and the caller is
    warned that none came back (`_set_prompt_logprobs`) rather than being told to set a
    config field it already set.
    """
    if use_remaining_context:
        if max_model_len is None or prompt_token_count is None:
            raise RuntimeError(
                "a stock task or a thinking-on organism requested the remaining context, "
                "but the resolved target has no max_model_len or prompt-token counter"
            )
        prompt_tokens = int(prompt_token_count(messages, tools=kwargs.get("tools")))
        remaining = max_model_len - prompt_tokens
        if remaining < 1:
            raise ContextExhausted(
                f"prompt uses {prompt_tokens} tokens against max_model_len={max_model_len}; "
                "no output context remains"
            )
        kwargs["max_tokens"] = remaining
    full = getattr(endpoint, "complete_full", None)
    if full is None:
        return Completion(text=endpoint.complete(messages, **kwargs))
    return full(messages, **kwargs)


def _usage(completion: Completion) -> ModelUsage | None:
    """The endpoint's own token counts, or `None` when it did not report both.

    Both or neither: a `ModelUsage` with a real input count and a zero output count reads
    as an organism that generated nothing, which is a worse lie than an honest estimate.
    """
    if completion.input_tokens is None or completion.output_tokens is None:
        return None
    return ModelUsage(
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        total_tokens=completion.input_tokens + completion.output_tokens,
    )


def _as_stop_reason(value: str | None) -> StopReason | None:
    """An endpoint's finish reason as one of inspect's, or `None` when it reported none.

    `Completion.stop_reason` is a plain `str` so that `core/` does not depend on inspect's
    vocabulary. Anything outside it becomes "unknown" rather than reaching pydantic, which
    would raise a ValidationError mid-sweep over a field nothing scores.
    """
    if value is None:
        return None
    return cast("StopReason", value) if value in _STOP_REASONS else "unknown"


_STOP_REASONS = frozenset(get_args(StopReason))


def _set_prompt_logprobs(
    output: ModelOutput, completion: Completion, config: GenerateConfig, model_name: str
) -> None:
    """Attach teacher-forced prompt logprobs, which is what the stock perplexity scorers read.

    `perplexity()` and `target_perplexity()` (inspect_ai 0.3.259,
    `scorer/_perplexity.py`, `scorer/_target_perplexity.py`) both read
    `state.output.choices[0].prompt_logprobs.content` and sum `.logprob` over it, so this
    is the whole wiring they need — `token` and `bytes` are never touched by either.

    `Endpoint.complete_full` returns bare floats, one per prompt token after the first
    (position 0 conditions on nothing), so `token` is empty. That is enough for both
    scorers: `perplexity()` averages the lot and `target_perplexity()` takes the last n.

    A request that asked for them and got none is WARNED about rather than left to the
    scorer, whose NaN explanation says "ensure prompt_logprobs is set in GenerateConfig" —
    advice that is already followed and so sends the reader in the wrong direction.
    """
    if completion.prompt_logprobs is None:
        if config.prompt_logprobs is not None:
            log.warning(
                "%s was asked for prompt_logprobs and returned none, so perplexity() will "
                "score NaN. The endpoint accepted the argument without honouring it",
                model_name,
            )
        return
    output.choices[0].prompt_logprobs = Logprobs(
        content=[Logprob(token="", logprob=v, bytes=None) for v in completion.prompt_logprobs]
    )


def _stop_reason(text: str, max_tokens: int | None) -> Literal["stop", "max_tokens"]:
    """Whether this completion looks like it ran into the output cap.

    THE FALLBACK, for an endpoint with no `complete_full` — one that has it reports a real
    finish reason and `ModelOutput.metadata['stop_reason_estimated']` is then False. On
    this path there is no finish reason to thread, and the alternative is to record "stop"
    unconditionally, which is how a cap that truncated 18 of 30 judged responses stayed
    invisible until someone read the transcripts (configs/specs.yaml, 2026-08-19).

    Deliberately UNDER-counts: `len(text) // 4` is below the true token count for most
    English, so this misses some truncations rather than claiming ones that did not
    happen. A wrong "max_tokens" would be worse than a missing one.
    """
    if max_tokens is not None and len(text) // _CHARS_PER_TOKEN >= max_tokens:
        return "max_tokens"
    return "stop"


def _estimated_usage(messages: list[dict[str, Any]], text: str) -> ModelUsage:
    """Token counts for one call, estimated — the fallback, see `_stop_reason` above.

    Reuses `evals._harness.estimate_tokens` (chars/4 plus a per-message chat-template
    allowance) so the provider and the drift cells' context projection cannot disagree
    about what a token costs. Without any usage at all, an organism reported zero tokens
    for a whole sweep and `--token-limit` was inert.
    """
    input_tokens = estimate_tokens(messages)
    output_tokens = len(text) // _CHARS_PER_TOKEN
    return ModelUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
    )


def sum_usage(outputs: Sequence[ModelOutput]) -> ModelUsage | None:
    """Total usage across the draws a multi-draw solver folds into ONE sample output.

    Those solvers assemble a fresh `ModelOutput` from n generations and assign it to
    `state.output`, which throws away the per-draw usage the provider attached — so the
    sample read `usage: None` while the log header's totals were right, and the viewer's
    per-sample token column was empty for every multi-draw eval. `ModelUsage` adds, so
    this is the sum and nothing else.
    """
    usages = [o.usage for o in outputs if o.usage is not None]
    return functools.reduce(operator.add, usages) if usages else None


def output_from_draws(model: str, draws: Sequence[ModelOutput]) -> ModelOutput:
    """The n draws of a multi-draw solver, folded into ONE sample output.

    Each draw's assistant message is kept as the provider returned it rather than
    rebuilt from its `completion` string. That is the whole point: on a thinking
    organism the provider splits the completion into `[ContentReasoning, ContentText]`,
    and a rebuild from `completion` keeps only the answer — so the reasoning never
    reached the sample `_thinking_meta` reads and every thinking-on capability row lost
    `meta['thinking']` (found on the 27B smoke, parked in PR #370).
    `thinking_truncated`,
    which rides in `metadata`, is OR-ed across ALL draws: a draw that spent its whole budget
    reasoning must flag the sample even when it is not the first draw, or it lands as an
    unflagged empty answer. Other metadata keys come from the first draw.

    Nothing else moves. `ChatMessageBase.text` joins TEXT parts only, so
    `output.completion` and every `choice.message.text` are the answers they always
    were, and a non-thinking draw's message is the plain `str` content the inline
    rebuild built.

    `stop_reason` stays explicitly `"stop"` rather than each draw's own: this output is
    assembled here, and an inspect-default "unknown" on every draw makes the viewer's
    truncation column useless exactly where the per-draw caps live.
    """
    metadata = dict(draws[0].metadata) if draws and draws[0].metadata else ({} if draws else None)
    if metadata is not None and any((d.metadata or {}).get("thinking_truncated") for d in draws):
        metadata["thinking_truncated"] = True
    return ModelOutput(
        model=model,
        usage=sum_usage(draws),
        metadata=metadata,
        choices=[ChatCompletionChoice(message=d.message, stop_reason="stop") for d in draws],
    )


#: Message keys that carry TRANSPORT IDENTITY rather than conversation content, and are
#: therefore excluded from the seed. `tool_call_id` is minted by inspect with a fresh
#: shortuuid every time a tool call is parsed (`ChatMessageTool.tool_call_id`), so an
#: identical conversation replayed twice hashes differently and the "same conversation
#: replays" guarantee `_seed_for` documents silently stops holding on every tool-calling
#: turn. Filtered HERE rather than in `_endpoint_message` so every caller is covered and
#: the id still reaches the chat template, which needs it.
_SEED_EXCLUDED_KEYS = frozenset({"tool_call_id"})

#: Which repeat of a sample the current generation belongs to, as ``(sample_idx, epoch)``.
#: A ContextVar because samples run concurrently through one shared provider: each sample
#: runs in its own async task, and a value set in one task is invisible to the others.
#: Set by `repeat_scope`; the default is the first repeat, which seeds exactly as before.
_FIRST_REPEAT = (0, 1)
_REPEAT: ContextVar[tuple[int, int]] = ContextVar("mo_evals_repeat", default=_FIRST_REPEAT)


@contextmanager
def repeat_scope(sample_idx: int, epoch: int) -> Iterator[None]:
    """Mark every generation inside the block as repeat ``(sample_idx, epoch)``.

    The judged solver (`evals._task_eval._condition_solver`) wraps each sample in this, so
    ``n_samples`` repeats and inspect ``--epochs`` each get their own conversation seed
    (docs/evals/robustness.md#repeat-samples-get-their-own-seeds). Repeat 0 of epoch 1 is
    the default and changes nothing, so every stored row, fixture and byte-identity run
    keeps its seed.
    """
    token = _REPEAT.set((int(sample_idx), int(epoch)))
    try:
        yield
    finally:
        _REPEAT.reset(token)


def _conversation_seed(
    messages: list[dict[str, Any]], *, repeat: tuple[int, int] = _FIRST_REPEAT
) -> int:
    """A stable integer for one conversation, and for one repeat of it.

    `hash()` is salted per process, so it looks deterministic and is not (the same trap
    `evals/capability.py` calls out). sha256 of the rendered messages is not. A repeat
    other than the first hashes the messages together with ``[sample_idx, epoch]``; the
    first hashes the messages alone, which is the seed every existing row was drawn with.
    """
    content = [
        {k: v for k, v in message.items() if k not in _SEED_EXCLUDED_KEYS} for message in messages
    ]
    payload: Any = (
        content if repeat == _FIRST_REPEAT else {"messages": content, "repeat": list(repeat)}
    )
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).digest()
    return int.from_bytes(digest[:4], "big") % 1_000_003


# ---------------------------------------------------------------------- bridge


def row_id(sample_id: str, sample: Any) -> str:
    """One row's id, disambiguated by epoch when the run had more than one.

    ``--epochs n`` runs every sample n times and writes n log samples under the SAME id,
    so a bridge that used the id alone would emit n identical row ids into one Parquet
    file. Epoch 1 is left untouched so a one-epoch run — every run this repo makes today,
    and every parity fixture — keys exactly as before.
    """
    epoch = getattr(sample, "epoch", 1) or 1
    return sample_id if epoch == 1 else join_id(sample_id, f"e{epoch}")


# inspect_ai.log._util.thin_metadata's placeholder for a score-metadata value over
# 1k of JSON. It belongs to the SUMMARY shape; finding it on a full sample means inspect
# persisted that sample's thinned summary score over the real one. 0.3.259 does, rarely:
# 2 of 600 samples of a resumed mu cell on 2026-08-27. Matching the string is the only
# signal the log carries — nothing else distinguishes a thinned score from a real one.
_THINNED_METADATA = "Key removed from summary (> 1k)"


def _sample_score(sample: Any) -> Any:
    """One sample's score, read from its score EVENT when the stored one is thinned.

    A thinned score keeps value but replaces every metadata value over 1k of JSON
    with _THINNED_METADATA — which for the packed bridges is the entire payload,
    since mu's ~25 comparison units and drift's turns each live under one metadata key.
    The ScoreEvent recorded in the same sample carries the untouched score, so read
    it from there instead of losing the rows (before this, indexing the placeholder
    string raised TypeError: string indices must be integers and failed the cell).
    """
    score = next(iter((sample.scores or {}).values()), None)
    if score is None or _THINNED_METADATA not in (score.metadata or {}).values():
        return score
    for event in reversed(sample.events or []):
        recorded = (
            getattr(event, "score", None) if getattr(event, "event", None) == "score" else None
        )
        if recorded is not None and _THINNED_METADATA not in (recorded.metadata or {}).values():
            return recorded
    return score


def _thinking_meta(sample: Any) -> dict[str, Any]:
    """`{"thinking", "reasoning_chars"}` for a sample whose generation had a think block.

    Empty when it did not, so a non-thinking cell's `meta` is unchanged. `raw_output`
    deliberately stays the ANSWER (`EndpointModelAPI.generate`), so this is where the
    think text reaches Parquet; `reasoning_chars` is measured off that string, never
    estimated. Both are `meta` keys because `core.results.SCHEMA` is a frozen contract and
    thinking is not a property every organism has.
    """
    choices = sample.output.choices if sample.output else []
    content = choices[0].message.content if choices else None
    for part in content if isinstance(content, list) else []:
        if isinstance(part, ContentReasoning):
            meta: dict[str, Any] = {
                "thinking": part.reasoning,
                "reasoning_chars": len(part.reasoning),
            }
            if (sample.output.metadata or {}).get("thinking_truncated"):
                # The generation hit its cap still reasoning, so `raw_output` is `""` — an
                # answer that was never written, not one the organism withheld.
                meta["thinking_truncated"] = True
            return meta
    return {}


def rows_from_log(
    log_: EvalLog,
    *,
    eval_name: str,
    axis: str,
    spec_hash: str,
    order: dict[str, int] | None = None,
) -> list[SampleResult]:
    """Bridge one completed Inspect log to `SampleResult` rows, one per sample.

    Pure function of the log — Parquet written through this is derived, write-once,
    and cannot drift from the `.eval` transcript. `raw_output` is the plain completion
    text, never JSON: `core.endpoint.is_fake_output` matches on `startswith` and the
    report prints it as a transcript.

    Scores: takes each sample's first score (our tasks attach exactly one scorer). A
    missing score becomes `math.nan`, matching the NaN-not-0.0 convention — NaN means
    "unmeasured", 0.0 means "measured and absent" (`judges.rubrics.Verdict.score`).

    `order` is `{sample_id: dataset index}`; pass it and the rows come back in DATASET
    order. Every caller that has the task should: inspect writes a log's samples
    **sorted by sample id** (measured 2026-08-24 — not dataset order, and not
    completion order either), while the native harness's rows are in submission order
    and every parity fixture is frozen against that. Ids not in `order` sort last,
    keeping their relative order.

    Under `--epochs > 1` the same sample id appears once per epoch, so the row id carries
    the epoch (`row_id`). Without that the Parquet writer sees n rows under one id — and
    `core.results` rejects duplicates, so the whole cell fails rather than being silently
    wrong, but only after the generations were paid for.
    """
    rows: list[SampleResult] = []
    for sample in log_.samples or []:
        score = _sample_score(sample)
        value = score.value if score is not None else math.nan
        meta: dict[str, Any] = dict(sample.metadata or {})
        if score is not None and score.metadata:
            meta.update(score.metadata)
        meta.update(_thinking_meta(sample))
        rows.append(
            SampleResult(
                spec_hash=spec_hash,
                eval_name=eval_name,
                axis=axis,
                sample_id=row_id(str(sample.id), sample),
                score=float(value) if isinstance(value, (int, float)) else math.nan,
                raw_output=sample.output.completion if sample.output else "",
                meta=meta,
            )
        )
    if order is not None:
        rows.sort(key=lambda r: order.get(r.sample_id, len(order)))
    return rows


# The writer's default reader, so `MO_EVALS_STORE=log` reads every unbridged log through
# this one function rather than core's plain copy (core.results._LOG_BRIDGES says why).
register_log_bridge(None, rows_from_log)


def rows_from_packed(key: str) -> Callable[..., list[SampleResult]]:
    """A bridge for Tasks whose samples pack SEVERAL rows into one Score's metadata.

    Two evals need this and they need it for opposite reasons: mu packs ~25 comparison
    units into a chunk sample to amortise Inspect's per-sample cost, and the persistence
    drift cells emit one row per TURN of a single sequential conversation, which is one
    sample by nature. Either way the sample is not the row, so the generic
    `rows_from_log` would collapse the pack to its aggregate.

    ``key`` is the metadata key the scorer stashed its list under; each entry is
    ``{"sample_id", "score", "raw_output", "meta"}``. Entries are already in the order
    the native harness returns them within a sample, and ``order`` (chunk/conversation
    id -> dataset index) restores the order BETWEEN samples, so the rows come back
    exactly as the native path returned them — which is what the parity fixtures freeze.

    **A sample that errored still emits a row.** ``fail_on_error=False`` lets one sample
    die without taking the run with it, and a packed sample that dies carries no score
    metadata at all — so skipping it silently DELETES its rows: a failed mu chunk drops
    ~25 comparison units, a failed drift conversation up to 1000 turns, and the short
    Parquet is then cached as if it were the whole cell. `rows_from_log` already writes a
    NaN row for an errored sample; this writes one too, under the SAMPLE's id, with the
    error in ``meta``. The eval_set runner treats that flagged row as a FAILED cell and
    writes no Parquet (``serving/inspect_runner._undercount_failure``), so it is retried
    rather than cached as a measurement.

    Epochs are handled the same way as in ``rows_from_log`` (``row_id``): under
    ``--epochs > 1`` every packed row of every epoch after the first is suffixed, or the
    whole pack collides with the first epoch's.
    """

    def bridge(
        log_: EvalLog,
        *,
        eval_name: str,
        axis: str,
        spec_hash: str,
        order: dict[str, int] | None = None,
    ) -> list[SampleResult]:
        samples = list(log_.samples or [])
        if order is not None:
            samples.sort(key=lambda s: order.get(str(s.id), len(order)))
        rows: list[SampleResult] = []
        for sample in samples:
            # The SAMPLE's reasoning, so it is per-row only where a packed row is one
            # generation. mu and the drift cells rebuild `state.output` from a bare string
            # (`ModelOutput.from_content`), so it is ABSENT there rather than the wrong
            # draw's — their think text stays in the log's model events.
            thinking = _thinking_meta(sample)
            score = _sample_score(sample)
            packed = (score.metadata or {}).get(key) if score is not None else None
            # isinstance, not truthiness: a score thinned beyond recovery leaves a
            # STRING under key, and list() of one is a list of characters — the
            # TypeError that failed a whole mu cell.
            packed_rows = list(packed) if isinstance(packed, list) else []
            if not packed_rows:
                # ONE row, not the n this sample should have produced: the per-sample row
                # count lives in the plan the factory built, and a bridge only ever sees
                # the log. So the cell is under-counted by (n - 1) — visible and flagged,
                # which is the point, but do not read row counts off a cell with these in
                # it. `meta['bridge_undercount']` marks them.
                rows.append(
                    SampleResult(
                        spec_hash=spec_hash,
                        eval_name=eval_name,
                        axis=axis,
                        sample_id=row_id(str(sample.id), sample),
                        score=math.nan,
                        raw_output=sample.output.completion if sample.output else "",
                        # Deliberately NOT `sample.metadata`: on a bridged task that field
                        # holds the solver's whole working set (every drift turn, every mu
                        # draw), which is megabytes and already the packed rows' job.
                        meta={
                            "error": (sample.error.message if sample.error else "no packed rows"),
                            "bridge_undercount": True,
                            **thinking,
                        },
                    )
                )
                continue
            for packed in packed_rows:
                rows.append(
                    SampleResult(
                        spec_hash=spec_hash,
                        eval_name=eval_name,
                        axis=axis,
                        sample_id=row_id(str(packed["sample_id"]), sample),
                        score=float(packed["score"]),
                        raw_output=str(packed["raw_output"]),
                        meta={**thinking, **dict(packed["meta"])},
                    )
                )
        return rows

    return bridge


# ---------------------------------------------------------------- bounded judge

# The Inspect model role the judge is served under inside a task. `grader` is Inspect's
# own convention — its built-in model-graded scorers look for that name — so
# `--model-role grader=openai/gpt-4o` swaps our judge with no knowledge of this repo.
GRADER_ROLE = "grader"

#: In-flight judge HTTPS requests, independent of generation width. See
#: ``_judge_max_connections`` for why the two are different resources and why 64.
_JUDGE_MAX_CONNECTIONS = 64


def _judge_max_connections(config: dict[str, Any], max_workers: int) -> int:
    """Judge width for this cell: ``min(judge_max_connections, max_workers)``.

    GENERATION WIDTH AND JUDGE WIDTH ARE DIFFERENT RESOURCES AND MUST NOT SHARE A KNOB.
    ``max_workers`` sizes in-flight generations against the vLLM continuous-batching
    coalescer, which is local, has no connection to establish and is faster the wider it
    runs — 256 is its documented batch cap. The judge's width sizes concurrent HTTPS
    requests out of ONE ``AsyncOpenAI`` client, whose httpx pool holds 1000 connections
    but only 100 keepalive slots (``openai/_constants.py:9``,
    ``DEFAULT_CONNECTION_LIMITS``); past that, every request over the keepalive count
    re-establishes TLS, and the client, not the endpoint, is what falls over.

    MEASURED, the 14-organism run of 2026-08-27/28: at the levelled 256 the sweep took
    >=1,847 ``ConnectTimeout``/``APITimeoutError`` judge failures over 9h — one cell
    logged 65 failures against 60 items — while the judge endpoint itself answered a
    probe in 0.8 s with HTTP 200 mid-storm. Each failure burns ``attempt_timeout`` 120 s
    x up to 4 retries and lands a NaN row. The default here is the pre-migration
    ``BoundedJudge`` width, which ran at a 0.5% failure rate, and it sits under the 100
    keepalive slots by construction.

    ``min`` with ``max_workers`` so a deliberately narrow cell (``max_workers: 1``, the
    serial debug width) is not silently widened to 64 by this default.
    """
    return min(int(config.get("judge_max_connections", _JUDGE_MAX_CONNECTIONS)), max_workers)


#: THE process-wide ceiling on judge/grader traffic. Built once, by `judge_bound`.
_JUDGE_BOUND: Any = None
_JUDGE_BOUND_WIDTH = 0
_JUDGE_BOUND_LOCK = threading.Lock()


def judge_bound(width: int) -> Any:
    """One process-global semaphore for ALL judge traffic, created once at ``width``.

    THE RACE IT DEFEATS. Inspect's per-model connection semaphore is keyed on
    ``model_concurrency_key`` WITHOUT its size, and ``concurrency()`` keeps the size the
    FIRST caller asked for. Under ``eval_set`` many judged cells share one grader key, so
    the width the judge actually runs at is decided by whichever task touched that key
    first — a model-graded scorer that reaches the grader through ``get_model(...)``
    enters at the task's generation width (256) and
    pins the pool there for every judged cell in the group, and ``_GraderTransport``'s
    per-call re-pinning of 64 loses that race. ``scorer_context`` eviction can also mint a
    second ``BoundedJudge``, doubling any per-instance bound. This one is ours, global to
    the process and independent of inspect's pool, so neither can widen it.

    FIRST CONFIG WINS, with a note rather than a silent choice: a group whose cells ask
    for different judge widths did run at one of them, and saying which beats quietly
    taking the widest. Not raising, because a mixed-width group is a config smell, not a
    reason to abandon a sweep.
    """
    global _JUDGE_BOUND, _JUDGE_BOUND_WIDTH
    with _JUDGE_BOUND_LOCK:
        if _JUDGE_BOUND is None:
            _JUDGE_BOUND = threading.BoundedSemaphore(width)
            _JUDGE_BOUND_WIDTH = width
        elif width != _JUDGE_BOUND_WIDTH:
            log.info(
                "judge bound already built at width %d; this cell asked for %d and will "
                "run at %d (first config wins)",
                _JUDGE_BOUND_WIDTH,
                width,
                _JUDGE_BOUND_WIDTH,
            )
        return _JUDGE_BOUND


class BoundedJudge:
    """A judge proxy that caps concurrent `score` calls.

    Everything else passes through, so callers keep using rubric/pair attributes.

    STILL NEEDED AFTER THE TRANSPORT MOVED ONTO INSPECT, for two reasons. Inspect's
    per-model connection pool is keyed on `model_concurrency_key` = ModelAPI class +
    `connection_key()`, and `OpenAIAPI.connection_key()` is `f"{api_key}:{model_name}"`
    (inspect_ai/model/_providers/openai.py) — so the grader DOES get its own pool,
    distinct from `EndpointModelAPI`'s single `"default"` key, and judge calls no longer
    ride the generation knob by accident. But (a) that pool is registered on first use
    and `concurrency()` keeps the FIRST requested size for a key, so a group whose cells
    ask for different judge widths runs at whichever cell judged first; and (b) it bounds
    in-flight *requests* on the event loop, not the worker threads blocked on them —
    which is what `limiter` below bounds, and what anyio's default 40-token pool would
    otherwise cap at 40 whatever the config said.
    """

    def __init__(self, judge: Any, max_concurrent: int, judge_width: int | None = None) -> None:
        self._judge = judge
        # THE PROCESS-GLOBAL BOUND, not a per-instance one — see `judge_bound` for the
        # race, and note the consequence: this is the ONE place our judge path acquires
        # it. `_GraderTransport.complete` deliberately does not, because it is only ever
        # reached from inside this `with` on the same thread, and a second acquire of a
        # non-reentrant semaphore would deadlock the whole cell at the configured width.
        self._sem = judge_bound(max_concurrent if judge_width is None else judge_width)
        # For scorers calling `score` via `anyio.to_thread.run_sync`: anyio's DEFAULT
        # thread pool holds 40 tokens process-wide, so without a dedicated limiter a
        # judge configured 64-wide caps at 40, shared with every other default-pool
        # caller — the same configured-width-never-reaches-the-work shape as the
        # generation limiter. The semaphore still
        # guards callers that reach `score` from their own threads.
        self.limiter = anyio.CapacityLimiter(max_concurrent)

    def score(self, *args: Any, **kwargs: Any) -> Any:
        with self._sem:
            return self._judge.score(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._judge, name)


def inspect_judge(
    config: dict[str, Any], max_concurrent: int
) -> tuple[BoundedJudge, dict[str, str | Model] | None]:
    """The judge for an Inspect task, plus the ``model_roles`` its ``Task`` must declare.

    Returns ``(judge, roles)``. A factory passes ``roles`` straight to
    ``Task(model_roles=...)``; ``None`` means this config's judge has no Inspect model
    behind it and the Task declares no role.

    ONLY THE TRANSPORT MOVES. ``judges.judge.Judge`` still renders the rubric, derives
    the per-sample position swap from ``sample_key``, and parses the verdict — it is
    constructed here as ``judges.judge.build_judge`` constructs it, with the same seed,
    and only its ``backend`` differs. So a task-path row's prompt, swap and outcome
    are the ones the native harness would have produced; what changes is which
    client puts the messages on the wire, and therefore that the call shows up in the
    ``.eval`` transcript with Inspect's own usage, cost and retry accounting.

    ONE ``build_judge`` KNOB DOES NOT SURVIVE THE MOVE, AND IS REFUSED RATHER THAN
    DROPPED. ``build_judge`` also reads ``judge_structured`` into ``Judge(structured=…)``,
    which makes ``Judge.score`` put a ``response_format`` on every backend call
    (``judges/schemas.py``). ``_GraderTransport.complete`` takes no per-call kwargs, so
    this transport cannot carry it. Constructing an unstructured ``Judge`` anyway is the
    worst of the three options: the knob would keep working in fake mode — which reaches
    ``build_judge`` below — and silently no-op for every real judged number. Passing
    ``structured=True`` regardless is the second worst, turning that into a ``TypeError``
    mid-sweep that ``Judge.score`` does not catch. So it raises at build time instead.
    Threading ``response_format`` onto ``GenerateConfig`` is the real fix and is not
    needed yet: the structured path is default-off until it clears its calibration gates
    (``docs/evals/coherence.md#constrained-decoding-what-it-deletes-and-what-it-invents``).

    FAKE MODE AND THE NATIVE PATH ARE UNTOUCHED. ``FakeJudge`` is not a transport — there
    is no model to serve it from and CI must stay offline — so any
    judge block that is not ``backend: http`` falls straight through to ``build_judge``.
    ``evals._harness.rescore_rows`` never comes here at all: it re-scores
    stored rows with no Inspect run in scope, and keep talking to ``judges.backends``.
    """
    from judges.backends import pooled
    from judges.judge import Judge, build_judge

    block = config.get("judge")
    # Computed for BOTH branches: the process-global judge bound is sized from it, and it
    # must not be built at a fake cell's generation width just because that cell judged
    # first (`judge_bound`).
    width = _judge_max_connections(config, max_concurrent)
    if not (isinstance(block, dict) and block.get("backend") == "http"):
        # `build_judge` already wires `block["secondary"]` (if any) onto the returned
        # `Judge` via `judges.backends.build_backend` — the panel needs no separate
        # handling on this branch.
        return BoundedJudge(build_judge(config), max_concurrent, width), None
    if config.get("judge_structured"):
        # Checked before `_grader_model` so the refusal does not depend on a judge key.
        raise ValueError(
            "judge_structured is not supported on the Inspect grader transport: "
            "_GraderTransport.complete takes no per-call kwargs, so the response_format "
            "that Judge(structured=True) sets cannot reach the judge. Drop the key, or "
            "run this eval on the native path (a judge block whose backend is not 'http')."
        )
    model = _grader_model(block, width)
    # Same construction as `build_judge`, same seed — see that function. `structured` is
    # absent on purpose: refused above rather than silently dropped (see the docstring).
    #
    # The SECONDARY judge of a panel never rides the grader transport, even when the
    # primary does: `_GraderTransport.complete` takes no per-call kwargs and exists only
    # to make the PRIMARY judge's call show up in the `.eval` transcript with Inspect's
    # own usage/retry accounting — there is no such requirement on the secondary, whose
    # only job is to be recorded alongside for inter-judge agreement. So it is always
    # built the plain way, exactly as `build_judge` builds one.
    judge = Judge(
        backend=_GraderTransport(model, width),
        secondary_backend=pooled(block.get("secondary"), config),
        seed=int(config.get("seed", 0)),
    )
    # `BoundedJudge` keeps the CELL's width, not the judge's: it bounds worker threads,
    # and a thread parked on the grader's connection semaphore is free. The semaphore is
    # acquired BEFORE `_generate` (inspect_ai/model/_model.py:899), so queueing there
    # cannot consume `attempt_timeout` — narrowing the thread ceiling as well would only
    # cost fake-mode throughput, where the judge answers in-process.
    return BoundedJudge(judge, max_concurrent, width), {GRADER_ROLE: model}


#: ``HttpJudge`` knob -> the ``GenerateConfig`` field that does the same job.
#:
#: THE NAMES DIFFER ON PURPOSE, and the mapping of `timeout` is the whole point.
#: ``GenerateConfig.timeout`` is NOT a request timeout: it is the retry loop's total
#: elapsed budget, and it is only ever read BETWEEN attempts (the `stop` closure in
#: inspect_ai/model/_retry.py:130-142), so it cannot interrupt a request that never
#: returns. The field that can is ``attempt_timeout``, which wraps the provider call in
#: ``anyio.move_on_after`` and raises ``AttemptTimeoutError``
#: (inspect_ai/model/_model.py:1374-1410) — the equivalent of the per-request `timeout=`
#: ``HttpJudge`` hands to ``requests.post``.
#:
#: The two are only correct AS A PAIR. ``AttemptTimeoutError`` is *always* retried ("we
#: rely on `timeout` and/or `max_retries` for termination", _model.py:1553-1558), so
#: ``attempt_timeout`` alone is an infinite retry loop, and ``max_retries`` alone never
#: fires because a hung request raises nothing to retry. With both, a wedged judge call
#: aborts at 120 s, retries 4 times, then raises — and ``_GraderTransport.complete``
#: turns that into the same ``JudgeBackendError`` discard an ``HttpJudge`` failure gives,
#: i.e. ``outcome=None`` -> a NaN row, which is what "not measured" means everywhere
#: downstream. A hang produces no row at all and no eval ever finishes.
_JUDGE_TRANSPORT_FIELDS = {"timeout": "attempt_timeout", "max_retries": "max_retries"}

#: Judge params `_grader_model` sends to an ``openai/`` model through ``extra_body`` (i.e.
#: verbatim) instead of through the matching ``GenerateConfig`` field — see the comment
#: there for why inspect's own handling of these is wrong for a model it cannot classify.
_OPENAI_WIRE_PARAMS = ("reasoning_effort", "max_completion_tokens")


def _http_judge_defaults() -> dict[str, Any]:
    """``HttpJudge``'s transport defaults, read off its signature rather than copied.

    Returns
    -------
    dict
        ``{"timeout": ..., "max_retries": ...}`` as ``judges.backends.HttpJudge``
        declares them, so the two transports cannot drift apart silently. Keys absent
        from the signature are omitted rather than guessed.
    """
    import inspect as pyinspect

    from judges.backends import HttpJudge

    sig = pyinspect.signature(HttpJudge.__init__).parameters
    return {
        name: sig[name].default
        for name in _JUDGE_TRANSPORT_FIELDS
        if name in sig and sig[name].default is not pyinspect.Parameter.empty
    }


def _grader_model(block: dict[str, Any], max_connections: int) -> Model:
    """The Inspect model for an ``http`` judge block, sending what ``HttpJudge`` sends.

    Mirrors ``judges.backends.HttpJudge.__init__``: a bare ``model`` id means the
    OpenAI-compatible service that class defaults to, so it is prefixed ``openai/``; an
    id that already names a provider (``openai/gpt-4o-mini``, as the petri blocks write
    it) passes through. ``base_url`` falls back to ``$JUDGE_BASE_URL`` and then to the
    provider's own default, and the key is read from the configured ``api_key_env``.

    OPENROUTER MODEL IDS ARE THE EXCEPTION TO "a '/' means already-qualified", and get
    their own branch. OpenRouter's OWN slugs are ``<author>/<model>`` (``z-ai/glm-5.3-flash``,
    ``google/gemini-3.7-flash``) — the exact string ``HttpJudge`` posts as the wire
    ``"model"`` field, verbatim, to ``$base_url/chat/completions``. Left alone here, that
    string would be split by Inspect's own ``get_model`` on ITS first ``/`` and read
    ``z-ai`` as an Inspect provider NAME, which does not exist. Detected off ``base_url``
    (the same signal ``HttpJudge`` has no need of, since it never re-routes) and stitched
    with an ``openrouter/`` prefix so Inspect's registered ``openrouter`` provider
    (``inspect_ai.model._providers.openrouter.OpenRouterAPI``) picks it up correctly.

    OPENROUTER ROUTING (``provider: {order, allow_fallbacks, ...}``) IS A REQUEST-BODY
    FIELD, NOT A ``GenerateConfig`` ONE, so it cannot go through the ``params`` mapping
    below — but it still has to reach ``HttpJudge`` unchanged (which sends it by spreading
    ``params`` straight into the JSON body), so it is written ONCE, under
    ``judge.params.provider``, and this function lifts it out of ``params`` into
    ``model_args`` before validating — ``OpenRouterAPI.__init__`` reads a ``provider``
    model_arg and puts it on the wire the same way (``judges/backends.py``,
    ``inspect_ai/model/_providers/openrouter.py``). This is DRY on purpose: stating the
    pin twice (once per transport) in config is exactly the kind of duplication that
    drifts silently when one copy is edited and the other is not.

    ``responses_api=False`` IS LOAD-BEARING AND IS NOT A STYLE CHOICE. Inspect routes the
    gpt-5 family to OpenAI's *Responses* API by default (`responses_preferred` in
    inspect_ai/model/_providers/openai.py), and our judge is ``gpt-5.6-luna`` — verified
    against the installed 0.3.260: `get_model("openai/gpt-5.6-luna").api.responses_api`
    is True. ``HttpJudge`` posts to ``/chat/completions``. Pinning it keeps the judge on
    the endpoint every stored verdict was produced on.

    ``params`` are the judge's generation knobs and must reach the wire unchanged, so they
    are mapped onto ``GenerateConfig`` FIELDS and an unmappable name is an error rather
    than a silent drop. EXCEPTION for an ``openai/`` model: ``reasoning_effort`` and
    ``max_completion_tokens`` go through ``extra_body`` verbatim and ``service_tier``
    becomes a model arg (``_OPENAI_WIRE_PARAMS``; an OpenAI-direct ``gpt-6-*`` judge, the
    kind ``scripts/judge_deferred.py --batch`` needs, takes all three and the pinned
    inspect mis-handles the first two for it). The shared judge is on OpenRouter
    (``openrouter/`` prefix) and never takes this branch. On any other provider
    ``max_completion_tokens`` has no ``GenerateConfig`` equivalent and raises here — use
    ``max_tokens``.

    ``timeout`` and ``max_retries`` are the exception to "params only": they default to
    ``HttpJudge``'s values and land on the fields that do the same job — see
    ``_JUDGE_TRANSPORT_FIELDS``, which is where the `timeout` -> `attempt_timeout`
    mapping and its reason live.
    """
    from judges.backends import JudgeBackendError

    name = str(block["model"])
    base_url = block.get("base_url") or os.environ.get("JUDGE_BASE_URL")
    if base_url and "openrouter.ai" in base_url and not name.startswith("openrouter/"):
        model_id = f"openrouter/{name}"
    elif "/" in name:
        model_id = name
    else:
        model_id = f"openai/{name}"
    api_key_env = str(block.get("api_key_env", "OPENAI_API_KEY"))
    api_key = os.environ.get(api_key_env)
    if not api_key:
        # Same message as HttpJudge's, raised at task-build time rather than on the first
        # judge call: a missing key is a config error, and failing before the organism
        # generates anything is strictly cheaper than discarding every judged sample.
        raise JudgeBackendError(
            f"${api_key_env} is not set. Export it, or configure the judge "
            f"with backend: fake for a no-network run."
        )
    params = dict(block.get("params") or {})
    model_args = dict(block.get("model_args") or {})
    # See the docstring: `provider` is a request-body field HttpJudge sends by spreading
    # `params` verbatim, but it has no GenerateConfig field, so it is lifted out here
    # rather than left to trip the "unmappable param" check below.
    provider = params.pop("provider", None)
    if provider is not None:
        model_args.setdefault("provider", provider)
    if model_id.startswith("openai/"):
        # OPENAI WIRE FIELDS THE PINNED INSPECT MIS-HANDLES FOR A MODEL IT DOES NOT KNOW.
        # inspect_ai 0.3.260 decides reasoning support by NAME: `is_gpt_5()` is a
        # "gpt-5" substring test, so for `gpt-6-luna` its chat-completions path
        # (`completion_params_completions`, openai_completions.py) DELETES
        # `reasoning_effort` ("gpt and not gpt-5") and sends `max_tokens`, which the model
        # 400s ("Use 'max_completion_tokens' instead" — live, 2026-09-24). Routing both
        # through `extra_body` puts them on the wire verbatim, exactly as HttpJudge
        # spreads `params` into its body. `service_tier` is an OpenAIAPI constructor
        # arg, not a GenerateConfig field, so it is lifted into `model_args` the same
        # way `provider` is above.
        wire = {k: params.pop(k) for k in _OPENAI_WIRE_PARAMS if k in params}
        if wire:
            params["extra_body"] = {**wire, **(params.get("extra_body") or {})}
        tier = params.pop("service_tier", None)
        if tier is not None:
            model_args.setdefault("service_tier", tier)
    unknown = sorted(set(params) - set(GenerateConfig.model_fields))
    if unknown:
        raise ValueError(
            f"judge params {unknown} have no GenerateConfig field, so the Inspect "
            f"transport cannot send them; see evals._inspect._grader_model"
        )
    defaults = _http_judge_defaults()
    for name, field in _JUDGE_TRANSPORT_FIELDS.items():
        # Precedence: an explicit `params` entry (GenerateConfig field names, verbatim)
        # wins, then the judge block's own HttpJudge-named knob, then HttpJudge's default.
        # That last rung is the one that mattered: both shipped configs write `params: {}`
        # and no timeout, so nothing was set and Inspect's own behaviour applied — no
        # request timeout and retry forever. Six judge calls hung >1h on the 2026-08-27
        # sweep, each wedging an in-flight cell at 0% GPU.
        if field in params:
            continue
        if name in block:
            params[field] = block[name]
        elif name in defaults:
            params[field] = defaults[name]
    return get_model(
        model_id,
        # Sizes the grader's OWN connection pool (see BoundedJudge). Left unset it is
        # Inspect's default 10, i.e. ~6x narrower than the configured judge width.
        # NOT SUFFICIENT ON ITS OWN — a role model is not the active model, so inspect
        # merges the ACTIVE task config's `max_connections` over this one before sizing
        # the semaphore (_model.py:1711-1721, "merge operational config so it's inherited
        # everywhere"). `_GraderTransport` therefore re-pins the same number per call,
        # which is the merge that wins. Kept here because it is also the memo key and the
        # value that applies with no task in scope.
        config=GenerateConfig(max_connections=max_connections, **params),
        base_url=base_url,
        api_key=api_key,
        # MEMOIZED, unlike `bind_endpoint`'s organism models — the opposite call, for the
        # opposite reason. A group builds one Task per cell up front and holds them all,
        # so `memoize=False` would mint a fresh `AsyncOpenAI` (and its httpx connection
        # pool) per judged cell — hundreds on a full sweep, none of them ever closed. The
        # cells' graders are the same client by construction. Safe because inspect's memo
        # key covers model, config, base_url, api_key and model_args (`get_model` in
        # inspect_ai/model/_model.py), so a second judge block cannot collide with this
        # one; and `resolve_model_roles` shallow-COPIES a role model before stamping the
        # role on it, so the shared instance is never mutated.
        **model_args,
        # gpt-5's Responses-API default is an openai-family quirk; OpenRouter (and every
        # other non-openai provider reached here) is unaffected, so this must stay gated
        # on the "openai/" prefix specifically, not on "model_id has no more '/'".
        **({"responses_api": False} if model_id.startswith("openai/") else {}),
    )


def stock_grader(config: dict[str, Any], max_concurrent: int | None = None) -> Model:
    """The house judge as an Inspect ``Model``, to hand to an UPSTREAM model-graded scorer.

    For a stock cell whose scorer lives in ``inspect_evals`` and takes ``scorer_model=``
    (XSTest, "are you sure"). Same model, endpoint pin and transport knobs as the judged
    cells' ``grader`` role (``_grader_model``), so the judge comes from ``eval_configs`` and
    nowhere else.

    PASSED EXPLICITLY, NEVER LEFT TO INSPECT'S DEFAULT. Upstream's ``scorer_model=None`` falls
    back to the grader ROLE and then to THE MODEL UNDER TEST, i.e. the organism would grade
    itself and every number would look fine. An explicit ``Model`` also takes precedence over
    a role (``inspect_ai.scorer._model``), so ``INSPECT_EVAL_MODEL_ROLE=grader=...`` does NOT
    swap the judge of these cells: change the ``judge`` block in ``eval_configs`` instead.

    Raises
    ------
    ValueError
        The config has no ``backend: http`` judge block. Fake mode never calls this (its
        stock cells carry a stub scorer), so reaching it without a judge is a real-run
        configuration error and must fail at build time, not on the first judge call.

    NOT BOUNDED BY ``judge_bound``: that semaphore is acquired by ``BoundedJudge.score`` on a
    worker thread, which an upstream async scorer never passes through. The bound that holds
    here is the model's own ``max_connections`` (``judge_max_connections``, 64 by default),
    which sizes Inspect's per-model connection pool for this model.
    """
    block = config.get("judge")
    if not (isinstance(block, dict) and block.get("backend") == "http"):
        raise ValueError(
            "a stock judged cell needs an `http` judge block in its eval config "
            "(`eval_configs.default.judge`); refusing to let the organism grade itself"
        )
    width = _judge_max_connections(config, int(max_concurrent or config.get("max_workers", 256)))
    return _grader_model(block, width)


class _GraderTransport:
    """A ``judges.backends.JudgeBackend`` whose wire client is Inspect's.

    THE ROLE IS RESOLVED ON EVERY CALL, not captured when the Task is built. Inspect only
    publishes ``model_roles`` once ``eval()``/``eval_set()`` is running — its docs' "Role
    Resolution" section is explicit that a task constructed beforehand cannot see them —
    so resolving here is exactly what makes ``--model-role grader=...`` win over the
    configured judge.

    ``default=`` IS NOT OPTIONAL. ``get_model(role=...)`` with no default falls back to
    THE EVALUATION'S OWN MODEL, which for us is the organism under test: a missing role
    would silently have each organism grade itself, and the numbers would look fine.

    THE PER-CALL ``max_connections`` IS THE ONE THAT ACTUALLY BINDS. A role model is
    never the active model, so ``Model._resolve_config`` merges the active task config's
    ``max_connections`` over the grader's own before ``_connection_concurrency`` sizes
    the semaphore (inspect_ai/model/_model.py:1711-1721) — and the runner levels every
    judged task's ``max_connections`` to the group's widest cell, so what the grader
    inherited was the GENERATION width. Only the config passed to ``generate`` merges
    last (_model.py:1723-1724), which is why it is re-pinned here rather than trusted
    from ``_grader_model``. Verified against the installed inspect_ai 0.3.260.

    AND IT STILL LOSES THE RACE, which is why ``judge_bound`` exists: inspect keeps the
    size the FIRST caller of a connection key asked for, so a stock scorer entering the
    grader's pool at the generation width pins it there and this re-pin is ignored. The
    bound that actually holds is the process-global one, acquired in ``BoundedJudge.score``.
    It is NOT acquired again here: this method only ever runs inside that ``with``, on the
    same thread, and a second acquire of a non-reentrant semaphore would deadlock the cell
    the moment every slot was held by a thread waiting for its own second slot.
    """

    def __init__(self, default: Model, max_connections: int) -> None:
        self._default = default
        self._config = GenerateConfig(max_connections=max_connections)
        self.model = str(default)
        # Reaches every judged row as `meta['judge']` (judges.judge.Judge.score). It says
        # `inspect:` rather than `http:` on purpose — same model, same prompt, different
        # client, and a row should record which one produced it.
        self.name = f"inspect:{self.model}"
        # The model that answered the most recent call ON THIS THREAD, for
        # `judge_config_meta` — thread-local for the reason HttpJudge's `_last_response` is.
        self._last = threading.local()

    def judge_config_meta(self) -> dict[str, Any]:
        """Judge model / reasoning effort / service tier, as `HttpJudge.judge_config_meta`.

        Read off the role model that actually ran (so a `--model-role grader=...`
        override records itself), not off the config this transport was built from.
        """
        model = getattr(self._last, "model", None) or self._default
        extra = model.config.extra_body or {}
        return {
            "judge_model": model.name,
            "judge_reasoning_effort": extra.get("reasoning_effort", model.config.reasoning_effort),
            "judge_service_tier": getattr(model.api, "service_tier", None),
        }

    def complete(self, messages: list[dict], **kw: Any) -> str:
        from judges.backends import JudgeBackendError, record_model_usage

        if kw:  # HttpJudge merges these into the request body; nothing passes them today
            raise TypeError(f"{type(self).__name__}.complete takes no per-call kwargs: {kw}")
        # Called from the SCORER's worker thread (`anyio.to_thread.run_sync`), which
        # inherits the eval's context — verified: the role ContextVar is visible here.
        # `generate` is a coroutine, so it goes back to the loop the eval is running on.
        model = get_model(role=GRADER_ROLE, default=self._default)
        self._last.model = model
        try:
            output = anyio.from_thread.run(_role_generate, model, messages, self._config)
        except Exception as exc:
            # A discard, exactly as an HttpJudge failure is: `Judge.score` turns this into
            # `outcome=None`, which is visibly distinct from a genuine `ambiguous`. Logged
            # with the exception so a WIRING error (which would discard every row of the
            # cell) is not mistaken for a flaky API.
            log.exception("grader call failed via inspect transport")
            raise JudgeBackendError(f"judge failed: {type(exc).__name__}: {exc}") from exc
        usage = output.usage
        if usage is not None:
            # Counted HERE rather than read back off the log, so `judges.backends.
            # judge_usage()` — and therefore `results/_usage/` — keeps working unchanged
            # for the native path, the task path and a mixed run, with no second
            # accounting route to keep in step. Keyed by the model that actually ran, so
            # a `--model-role` override shows up as itself.
            record_model_usage(
                str(model),
                prompt_tokens=int(usage.input_tokens or 0),
                completion_tokens=int(usage.output_tokens or 0),
                calls=1,
            )
        return output.completion or ""


_CHAT_MESSAGE = {
    "system": ChatMessageSystem,
    "user": ChatMessageUser,
    "assistant": ChatMessageAssistant,
}


async def _role_generate(
    model: Model, messages: list[dict], config: GenerateConfig
) -> ModelOutput:
    """One turn from a role model — the judge, or the debate adversary / drift partner.

    `cache=False`: a rubric's verdict, and a turn of the conversation under test, are
    both the measurement.

    `config` carries only the role's connection width, and only because a per-call config
    is the last merge inspect applies — see `_GraderTransport`. It must NOT grow the
    role's generation knobs: those belong on the model, where `--model-role grader=...`
    / `adversary=...` can replace them wholesale.
    """
    return await model.generate(
        [_CHAT_MESSAGE[m["role"]](content=m["content"]) for m in messages],
        config=config,
        cache=False,
    )


# ---------------------------------------------------------- adversary role

# The Inspect model role the debate adversary (`robustness_multiturn_debate_gen`) and the
# drift partner (`persistence_multiturn_drift*`'s live-partner arm) are served under.
# Unlike `grader` this is not an upstream convention — inspect has no stock name for "the
# other model in the conversation" — but everything else about it is the grader's story:
# declared on the Task from config, and swappable from outside with
# `--model-role adversary=...` / `INSPECT_EVAL_MODEL_ROLE="adversary=openai/gpt-4o"`,
# which `serving/inspect_runner.py` forwards to `eval_set` for any role name.
ADVERSARY_ROLE = "adversary"


class InspectAdversary:
    """The second model in a conversation, served through the `adversary` model role.

    WHAT THIS REPLACED, AND WHY. Both cells used to build a raw `judges.backends.HttpJudge`
    and call it from the solver via `anyio.to_thread.run_sync` under the GENERATION
    limiter. Four costs, all gone here: the adversary's turns carried no usage or cost,
    they never appeared as model events in the `.eval` log (so a debate transcript in
    `inspect view` showed the organism talking to nobody), they could not be swapped with
    `--model-role`, and they were bounded by the wrong resource — an adversary turn is an
    HTTPS request out of one client, not a local vLLM generation, so the GPU limiter was
    never its ceiling (see `_judge_max_connections`).

    THE ROLE IS RESOLVED ON EVERY CALL, and `default=` is not optional, for exactly the
    reasons `_GraderTransport` gives: roles are only published once `eval()`/`eval_set()`
    is running, and `get_model(role=...)` with no default falls back to THE EVALUATION'S
    OWN MODEL — which here would have each organism argue with itself, and the transcript
    would look plausible.

    THE PER-CALL `max_connections` IS THE ONE THAT BINDS, also per `_GraderTransport`: a
    role model is not the active model, so the task's (generation-width) config merges
    over the model's own, and only a config passed to `generate` merges last.

    Usage is recorded into `judges.backends` as well as the log, because `judge_usage()`
    is what `results/_usage/` costs a run from and its docstring already says
    "judge/adversary".
    """

    def __init__(self, default: Model, max_connections: int) -> None:
        self.default = default
        self._config = GenerateConfig(max_connections=max_connections)
        self.model = str(default)
        #: Mirrors `_GraderTransport.name`: same model, Inspect's client rather than ours.
        self.name = f"inspect:{self.model}"

    async def complete(self, messages: list[dict]) -> str:
        from judges.backends import record_model_usage

        model = get_model(role=ADVERSARY_ROLE, default=self.default)
        output = await _role_generate(model, messages, self._config)
        usage = output.usage
        if usage is not None:
            record_model_usage(
                str(model),
                prompt_tokens=int(usage.input_tokens or 0),
                completion_tokens=int(usage.output_tokens or 0),
                calls=1,
            )
        return output.completion or ""


def inspect_adversary(block: dict[str, Any], config: dict[str, Any]) -> InspectAdversary:
    """Build the adversary client for a `judges.backends` http block.

    The block has the judge's shape, so `_grader_model` builds it — one place that knows
    how a block maps onto `GenerateConfig`, and the `timeout`/`max_retries` pairing it
    documents (a wedged call must abort and retry, or the cell never finishes) applies
    unchanged to a debate turn.

    Width is the JUDGE's, not the generation limiter's: `_judge_max_connections` bounds
    in-flight HTTPS requests out of one client, which is what an adversary turn is. The
    32 default matches `_task_eval.judged_inputs` / `_task_persistence._drift_inputs`,
    the two places that build the cells this serves.

    If `configs/specs.yaml` ever points the judge and the adversary at the same model,
    `get_model`'s memo hands both roles the same `Model`, and inspect keys its connection
    pool on api_key + model name, so they share ONE pool of `_judge_max_connections`
    slots and a debate cell's judge and adversary calls contend. Today they differ
    (gemma judge, gpt-6-luna adversary; both on OpenRouter), so each has its own pool.
    """
    max_workers = int(config.get("max_workers", 32))
    width = _judge_max_connections(config, max_workers)
    return InspectAdversary(_grader_model(block, width), width)


async def adversary_turn(adversary: Any, messages: list[dict]) -> str:
    """One turn from a debate adversary / drift partner, whatever shape it is.

    Two shapes on purpose. An `InspectAdversary` is the real path. Anything else has a
    SYNC `.complete(messages) -> str` and is called inline: that is the scripted
    `_FakeAdversary`/`_FakeInterlocutor` (CLAUDE.md, and what CI runs) and the spy
    objects tests inject through the same config key. Those are pure string formatting
    with no I/O, so the worker thread and the limiter they used to run under bought
    nothing and are gone.
    """
    if isinstance(adversary, InspectAdversary):
        return await adversary.complete(messages)
    return adversary.complete(messages)


def with_adversary_role(
    roles: dict[str, str | Model] | None, adversaries: Iterable[Any]
) -> dict[str, str | Model] | None:
    """Add the `adversary` role for any `InspectAdversary` among ``adversaries``.

    A cell's second model arrives inside its condition spec / drift plan rather than from
    `inspect_judge`, so this is what gets it onto `Task(model_roles=...)` beside the
    grader — which is what makes `--model-role adversary=...` reach it.
    """
    for adversary in adversaries:
        if isinstance(adversary, InspectAdversary):
            return {**(roles or {}), ADVERSARY_ROLE: adversary.default}
    return roles


# ------------------------------------------------------------- the eval registry

# THE eval registry. One entry per eval: what it measures (`axis`, `applicable_to`,
# `summary`) and how to build its Inspect Task (`builder`, `bridge`). Registration and
# task-factory registration used to be two calls in two places — `@register` in
# `core/registry.py` and `register_task_factory` here — which meant an eval could be
# registered with no way to run it (`KeyError: no Inspect task factory registered`) or
# runnable but absent from every sweep. They are one call since 2026-08-29.
#
# It lives here rather than in `core/`: generation runs through Inspect, so the thing an
# eval *is* is an Inspect Task, and `core/` may not import `evals/`. `mo_evals.py` mints
# one `mo_evals/<name>` task per entry; `serving/inspect_runner.py` pairs each entry's
# Task with an organism (model=f"{PROVIDER}/{spec_hash}") per matrix cell.
#
# `bridge` overrides `rows_from_log` for evals whose log does not read back as one row per
# sample. FIVE evals set one, for two different reasons:
#
#   * rows PACKED into a sample — `mu_decisiveness` (~25 comparison units per chunk, to
#     amortise Inspect's per-sample cost) and both `persistence_multiturn_drift*` cells
#     (one row per TURN of a conversation that is irreducibly one sample);
#   * a post-pass over 1:1 rows — `introspection_denial` (the judge-free yes/no promotion).


@dataclass(frozen=True)
class RegisteredEval:
    """One eval: what it measures, what it applies to, and how to build its Task."""

    name: str
    axis: str
    applicable_to: tuple[str, ...]
    summary: str
    builder: Callable[[dict[str, Any]], Any]
    bridge: Callable[..., list[SampleResult]] | None = None


_EVALS: dict[str, RegisteredEval] = {}


def register_task(
    eval_name: str,
    *,
    axis: str,
    applicable_to: Sequence[str],
    summary: str,
    builder: Callable[[dict[str, Any]], Any],
    bridge: Callable[..., list[SampleResult]] | None = None,
) -> None:
    """Register one eval. The whole registration — there is no second call.

    Parameters
    ----------
    eval_name:
        The name everything keys on: the Inspect task (``mo_evals/<name>``), the Parquet
        stem, ``eval_configs`` and ``default_evals`` in ``configs/specs.yaml``, and the
        row's ``eval_name``. Pick it once and pick it well.
    axis:
        One of ``core.registry.VALID_AXES``. The scorecard's aggregation key.
    applicable_to:
        Behaviour classes this eval is meaningful for. Non-transfer of the protocol
        across behaviour types then shows up as a structured gap in the results table
        rather than a hole a reviewer finds.
    summary:
        One line saying what the eval measures — the third column of
        ``python -m mo_evals --list``, and the only bridge from a task name to its meaning
        short of ``docs/EVAL-CATALOGUE.md``. Required, so it cannot be forgotten into a
        blank column.
    builder:
        ``config -> Task``. Called once per matrix cell with the merged eval config.
    bridge:
        Optional ``log -> rows`` override; see the block comment above.

    Raises
    ------
    ValueError
        Unknown axis or behaviour class, empty ``applicable_to`` or ``summary``, or a
        duplicate name. The duplicate guard has to live here: Inspect's own registry
        silently overwrites, so without it a second registration of one name would
        replace the first with nothing said.
    """
    if axis not in VALID_AXES:
        raise ValueError(f"unknown axis {axis!r}; valid: {sorted(VALID_AXES)}")
    if not applicable_to:
        raise ValueError("applicable_to must list at least one behaviour class")
    unknown = set(applicable_to) - VALID_BEHAVIOR_CLASSES
    if unknown:
        raise ValueError(
            f"unknown behaviour classes in applicable_to: {sorted(unknown)}; "
            f"valid: {sorted(VALID_BEHAVIOR_CLASSES)}"
        )
    if not summary.strip():
        raise ValueError(f"{eval_name}: summary must say in one line what the eval measures")
    if eval_name in _EVALS:
        raise ValueError(f"eval {eval_name!r} is already registered")
    _EVALS[eval_name] = RegisteredEval(
        name=eval_name,
        axis=axis,
        applicable_to=tuple(applicable_to),
        summary=summary.strip(),
        builder=builder,
        bridge=bridge,
    )
    if bridge is not None:
        # Tell core about it too, so `core.results.rows_from_eval_log` reads this eval
        # the same way the Parquet was written. core may not import evals/, so the
        # dependency is inverted here (see core.results.register_log_bridge). Without
        # this line the log store and the Parquet store disagree on exactly the bridged
        # cells, and nothing fails loudly — measured on a real tree, 2026-08-26.
        register_log_bridge(eval_name, bridge)


def get_eval(eval_name: str) -> RegisteredEval:
    """Look up one registered eval, listing the alternatives when the name is wrong."""
    try:
        return _EVALS[eval_name]
    except KeyError:
        raise KeyError(f"no eval named {eval_name!r}; registered: {sorted(_EVALS)}") from None


def all_evals() -> list[RegisteredEval]:
    """Every registered eval, in stable name order."""
    return [_EVALS[name] for name in sorted(_EVALS)]


def evals_for(
    behavior_class: str | None = None,
    axis: str | None = None,
) -> list[RegisteredEval]:
    """Registered evals filtered by behaviour class and/or axis."""
    return [
        e
        for e in all_evals()
        if (behavior_class is None or behavior_class in e.applicable_to)
        and (axis is None or e.axis == axis)
    ]


def registry_snapshot() -> dict[str, RegisteredEval]:
    """A copy of the registry. For tests: take one, register toys, restore.

    Restore rather than `clear()`: every entry is registered by an import side effect of
    an `evals/_task_*.py` module, and those are `_`-prefixed infrastructure that must not
    be purged from `sys.modules` to be re-imported (it splits module identity — see
    `tests/conftest.purgeable_eval_module`). So a cleared registry cannot be refilled.
    """
    return dict(_EVALS)


def registry_restore(snapshot: dict[str, RegisteredEval]) -> None:
    """Put the registry back exactly as `registry_snapshot` found it."""
    _EVALS.clear()
    _EVALS.update(snapshot)


def has_bridge(eval_name: str) -> bool:
    """Whether this eval's Inspect samples are NOT 1:1 with rows.

    Stamped into each `.eval` log so a log stays self-describing: core reads logs but
    cannot import `evals/` to look this up, and a per-sample reading of a bridged log is
    silently wrong rather than empty.
    """
    entry = _EVALS.get(eval_name)
    return entry is not None and entry.bridge is not None


def bridge_for(eval_name: str) -> Callable[..., list[SampleResult]]:
    """The log->rows bridge for one eval: its registered custom bridge, else the
    generic one-row-per-sample `rows_from_log`."""
    entry = _EVALS.get(eval_name)
    return entry.bridge if entry is not None and entry.bridge is not None else rows_from_log


# ======================================================= stock cells from `inspect_evals`
#
# A STOCK CELL runs an upstream Task AS BUILT — their dataset, their prompt, their scorer,
# their protocol constants — and changes only the things *around* the measurement. Six of
# them share these properties, which is what these four helpers hold:
#
#   * their dataset is fetched at task-BUILD time, so on an offline machine the Task cannot
#     be constructed at all. Registration is therefore GATED on the cache
#     (`register_stock_task`) — a cell that is absent is better than one that is red on
#     every sweep, and the warning is what stops "absent" being silent;
#   * the judged ones must reach the house `grader` role, because upstream's default is
#     either the model under test (an organism grading itself) or a hardcoded paid model.
#     `stock_grader` refuses to build a real run without one;
#   * `task.config.merge(...)`, NEVER a fresh `GenerateConfig`: upstream's temperature,
#     seed and token limits are part of the protocol being reproduced, and a replacement
#     silently drops them (`stock_stamps`);
#   * their `Score.value` is usually not a plain float — a letter, or a dict of readings —
#     which the generic `rows_from_log` turns into NaN. `stock_rows` is the 1:1 bridge with
#     one hook for reading their score, so the NaN-for-unmeasured rule lives in one place.
#
# THE STOCK CAPABILITY CELLS NOW VOTE IN THE COMPOSITE, AND THAT IS A REVERSAL. This block
# used to read "every stock cell is `exclude_from_composite: true` until someone argues
# otherwise", on two grounds: a stock cell duplicated an axis we already measured with an
# instrument built for cross-organism comparison, and stock semantics score a format failure
# as a wrong answer, which is correlated with the condition under test.
#
# The first ground is gone. Since the 9 Sep replacement there is no second instrument to
# duplicate: `mmlu`, `wmdp`, `gpqa_diamond` and `ifeval` ARE the capability cells, our own
# are deregistered, and excluding all four would leave the capability axis carrying only
# `capability_format` and `code_contests`.
#
# The second ground still stands and was accepted anyway, deliberately, so state it rather
# than let a reader infer it was overlooked: an unreadable answer now scores 0.0 rather than
# NaN, so on a format-damaged organism lost instruction-following is averaged into the
# capability number as lost knowledge. The 8 Sep run bounds how much that can move a cell
# (up to 0.038 on gpqa, 0.007 on mmlu), and `ifeval` is where format damage is supposed to
# be read. Anyone reading a capability composite on an implanted organism should read the
# per-cell numbers beside it.
#
# A stock cell added for a DIFFERENT axis, one we still measure ourselves, goes back to
# `exclude_from_composite: true` -- the old argument applies unchanged to that case.

#: Every cell registered through `register_stock_task`, whether or not its gate was open.
#: Filled by the gate call itself, so it is the SAME set on a bare checkout and a populated
#: one — which is what lets `tests/test_documented_counts.py` subtract these by name and
#: reach one verdict on both. Derived, never hand-maintained.
_CACHE_GATED: set[str] = set()


def cache_gated_evals() -> frozenset[str]:
    """The cells that register only when their upstream dataset is already cached.

    Populated at import of the `evals/_task_*.py` modules, in both gate directions — so a
    caller can subtract them without knowing which machine it is on.
    """
    return frozenset(_CACHE_GATED)


def register_stock_task(
    eval_name: str,
    *,
    axis: str,
    applicable_to: Sequence[str],
    summary: str,
    builder: Callable[[dict[str, Any]], Any],
    bridge: Callable[..., list[SampleResult]] | None = None,
    available: bool,
    reason: str,
    populate: str,
) -> None:
    """`register_task` for a stock cell, gated on its upstream dataset being cached.

    ONE GATE for registration and factory, which is the point: a cell cannot end up
    registered with a factory that raises, and it joins the sweep by itself the moment the
    cache fills — no code change.

    Parameters
    ----------
    available:
        The gate. Evaluate it cheaply (a `stat`, not an import of upstream's world): this
        runs at import of a `evals/_task_*.py` module, which is on the path of every
        ``python -m mo_evals --list``.
    reason:
        What is missing, in one sentence, for the skip warning.
    populate:
        The command that fills the cache, verbatim. Required, and it is the whole value of
        the warning — a cell silently absent from the scorecard is worse than one that
        fails, because nothing in the log says what to do about it.
    """
    _CACHE_GATED.add(eval_name)
    if not populate.strip():
        raise ValueError(f"{eval_name}: a skipped stock cell must say how to populate its cache")
    if not available:
        log.warning(
            "%s NOT REGISTERED: %s Populate the cache on a machine with network:\n    %s",
            eval_name,
            reason.strip(),
            populate.strip(),
        )
        return
    register_task(
        eval_name,
        axis=axis,
        applicable_to=applicable_to,
        summary=summary,
        builder=builder,
        bridge=bridge,
    )


def stock_stamps(
    task: Any,
    *,
    name: str,
    config: dict[str, Any],
    max_workers: int,
    max_tokens: int | None = None,
    turn_cap: dict[str, Any] | None = None,
    **overrides: Any,
) -> Any:
    """Upstream's Task with our stamps on it, and nothing else touched.

    Four things, none of them the measurement:

    * `name` — theirs is their own task name, and ours is the log's task name, which is the
      Parquet stem and the registry key.
    * `fail_on_error=False`, as every other cell here.
    * `max_connections`, MERGED into their config rather than replacing it. Upstream's
      `temperature` / `seed` / `max_tokens` are part of the protocol that makes the number
      citable, and a fresh `GenerateConfig` drops them silently.
    * a private stock marker in `extra_body`; it never reaches a backend. It lets the
      provider preserve an explicit upstream task or solver limit, while replacing a
      genuinely absent limit with the model's remaining context.

    `max_tokens` OPTS OUT of that last stamp, and an AGENTIC cell must use it. "Remaining
    context" is harmless for a single-turn cell whose answer is a letter — the cap never
    binds. On a multi-turn agent it is actively destructive: turn 1's ~3k prompt licenses a
    ~29,000-token reply, and `CompactionEdit` trims a conversation by dropping whole
    messages, so it cannot shrink ONE message bigger than the budget. That is the
    `Compaction insufficient: 34,145 tokens still exceeds threshold of 29,491` that killed a
    sample in each arm of the 2026-09-16 paired run. Upstream sets no per-turn cap at all,
    so the honest match is the provider's own default, not the whole window.
    A judged stock cell would need a fourth stamp -- `model_roles` plus every upstream
    scorer wrapped in the process-global judge bound, because a stock model-graded scorer
    reaches the grader through `get_model(...).generate` and is bounded by nothing of ours.
    The stock capability cells are not judged. The one judged stock cell,
    `misaligned_are_you_sure`, passes its grader through `stock_grader` and is bounded only by
    that model's own `max_connections`, not by the judge bound.

    `turn_cap` is `turn_cap_extra_body(endpoint)` from a cell with a deliberate per-turn cap
    (`max_tokens`): merged into `extra_body` so a THINKING-ON organism keeps that cap plus
    the reasoning allowance. It is `None` for a thinking-off organism, and then nothing is
    stamped — the task (and its eval_set identity) is byte-for-byte what it was before.

    `task_with`, not attribute assignment: `Task.name` is a read-only property, and anything
    else a cell needs to change goes in `**overrides` and is passed straight through.
    """
    from inspect_ai import task_with

    extra_body = dict(task.config.extra_body or {})
    if max_tokens is None:
        extra_body[_STOCK_MAX_TOKENS_MARKER] = True
    elif turn_cap:
        extra_body.update(turn_cap)  # a thinking-on agentic cell's per-turn cap (above)
    stamped = task_with(
        task,
        name=name,
        fail_on_error=False,
        config=task.config.merge(
            GenerateConfig(
                max_connections=max_workers, extra_body=extra_body, max_tokens=max_tokens
            )
        ),
        **overrides,
    )
    #: LAST, not first. `overrides` may carry its own `dataset=` -- `gpqa_diamond` passes
    #: one to pin upstream's unseeded choice shuffle -- and capping before that would be
    #: silently undone, leaving the metadata claiming a cap that never applied.
    return _cap_stock_dataset(stamped, name=name, config=config)


#: Task-metadata flag read by the runner's width levelling (serving/inspect_runner.py).
_WIDTH_PINNED_KEY = "mo_evals_width_pinned"


def pin_width(task: Any) -> Any:
    """Mark a cell whose `max_connections` bounds an EXTERNAL resource, not generation.

    The runner levels every cell in a backend group to the group's widest width, because
    they all share one connection pool. For a cell whose width is a quota on something
    else (for example a remote sandbox per in-flight sample, against an account-wide memory
    cap) that levelling is a bug: next to a 256-wide mmlu it would start
    256 sandboxes. A pinned cell keeps its own width, and the runner gives it its own pool.
    """
    from inspect_ai import task_with

    return task_with(task, metadata={**(task.metadata or {}), _WIDTH_PINNED_KEY: True})


def width_is_pinned(task: Any) -> bool:
    """Did the cell's builder `pin_width` it?"""
    return bool((task.metadata or {}).get(_WIDTH_PINNED_KEY))


def _cap_stock_dataset(task: Any, *, name: str, config: dict[str, Any]) -> Any:
    """Honour `limit_items` / `MO_EVALS_MAX_ITEMS` on a stock cell, and say so loudly.

    WITHOUT THIS THE SMOKE GATE IS UNRUNNABLE. Every other cell in the suite honours the
    cap, through `cap_items_to_env` on a materialised bank or `cap_count_to_env` on a
    stratified count. A stock cell goes through neither -- upstream builds its own
    dataset -- so `configs/specs.smoke.yaml`'s `limit_items: 5` reached nothing and the
    fake-mode byte-identity gate tried to run mmlu's full 13,937 samples, twice, per
    organism. Measured at 211s per 500 fake samples, that is about 98 minutes for one
    pass of one organism against a 20-minute CI budget.

    TRUNCATION IN FILE ORDER, NOT A SUBSAMPLE, and that is deliberate. Our own banks
    re-run their stratified selector under a tighter count so a capped MMLU still spans
    subjects. Here the dataset is upstream's and re-selecting inside it would be us
    choosing the exam, which is the one thing a stock cell exists not to do. Taking a
    prefix is the same rule `cap_items_to_env` applies to a bank, and it keeps the
    selection byte-identical between two runs, which is what the gate compares.

    A CAPPED STOCK NUMBER IS NOT A PUBLISHED NUMBER. The entire justification for these
    cells is comparability with the literature, and 5 of 13,937 questions is not MMLU.
    So this warns at WARNING level and records `stock_capped_to` in the task metadata,
    so any row produced under a cap carries the evidence rather than looking like a real
    read. `MO_EVALS_MAX_ITEMS` cannot reach a real run at all (`run_all.sh` refuses it
    without `MO_EVALS_FAKE`); `limit_items` can, hence the metadata stamp.
    """
    from inspect_ai import task_with
    from inspect_ai.dataset import MemoryDataset

    from evals._harness import cap_count_to_env, item_limit

    total = len(task.dataset)
    wanted = item_limit("limit_items", config, "limit_items", null_is_unset=True)
    if wanted is None:
        wanted = total
    wanted = cap_count_to_env(wanted, "questions")
    if wanted >= total:
        return task

    log.warning(
        "%s CAPPED to the first %d of %d upstream samples -- this is NOT a published"
        " %s number and must not be quoted as one",
        name,
        wanted,
        total,
        name,
    )
    kept = MemoryDataset(list(task.dataset)[:wanted], name=task.dataset.name)
    meta = dict(task.metadata or {})
    meta["stock_capped_to"] = wanted
    meta["stock_full_size"] = total
    return task_with(task, dataset=kept, metadata=meta)


def stock_rows(
    read: Callable[[Any, Any], tuple[float, dict[str, Any], str | None]],
) -> Callable[..., list[SampleResult]]:
    """A 1:1 bridge for a stock cell whose `Score.value` is not a plain float.

    A bridge rather than a wrapper scorer, and that is the whole reason this exists:
    wrapping would put our code between upstream's scorer and its own metrics, which is the
    thing a stock cell exists to avoid. Without one, a letter- or dict-valued score comes
    through `rows_from_log` as NaN and the cell produces a full set of rows and no scores.

    `read(sample, score) -> (score, meta, raw_output)` is the only per-cell part. `score` is
    the sample's first Score or `None` when it errored before grading — which is NaN, this
    repo's unmeasured convention, and not a stock reading at all. `raw_output` may be None,
    meaning "the completion", which is what all but the agentic cells want.
    """

    def bridge(log_: Any, **kw: Any) -> list[SampleResult]:
        rows: list[SampleResult] = []
        # Upstream's `comparability_version`, from the log header, so each row names the
        # protocol that produced it (the package itself is pinned exactly in pyproject).
        task_version = getattr(getattr(log_, "eval", None), "task_version", None)
        for sample in log_.samples or []:
            score = next(iter((sample.scores or {}).values()), None)
            value, meta, raw = read(sample, score)
            meta.setdefault("stock_task_version", task_version)
            output_meta = getattr(sample.output, "metadata", None) or {}
            for key in ("effective_max_tokens", "effective_max_tokens_source"):
                if key in output_meta:
                    meta.setdefault(key, output_meta[key])
            # An infrastructure failure must be NaN (unmeasured), not 0.0 (wrong) -- but the
            # test for "generation never happened" has to be exact, or it rewrites real
            # answers as missing data. `ModelOutput.empty` is `len(choices) == 0`: true when
            # the sample holds no completion object -- the model was never reached, or a
            # sample limit fired inside `generate()` and Inspect discarded the reply. A model
            # that answers with an empty string still HAS a choice and keeps its grade.
            #
            # The first version tested "empty completion AND no `effective_max_tokens` stamp".
            # The stamp is absent on every `stock_stamps(max_tokens=...)` cell, so a graded
            # thinking-off `swe_bench_pro` sample whose last reply was blank became NaN and
            # left the organism's denominator (adversarial review of #321). `getattr`,
            # because the unit tests pass bare stand-in outputs; a real log always has the
            # attribute.
            output = sample.output
            if (output is None or getattr(output, "empty", False)) and not math.isnan(value):
                value = math.nan
                meta["unmeasured_reason"] = "no_generation"
            rows.append(
                SampleResult(
                    spec_hash=kw["spec_hash"],
                    eval_name=kw["eval_name"],
                    axis=kw["axis"],
                    sample_id=row_id(str(sample.id), sample),
                    score=value,
                    raw_output=(
                        raw
                        if raw is not None
                        else (sample.output.completion if sample.output else "")
                    ),
                    meta=meta,
                )
            )
        order = kw.get("order")
        if order is not None:
            rows.sort(key=lambda r: order.get(r.sample_id, len(order)))
        return rows

    return bridge


# Budget for a @scorer/@solver factory's recorded arguments. Inspect copies them into
# EVERY sample's score event, so this is paid once per sample, not once per task.
# Generous: real offenders were 100 kB - 7.5 MB, everything legitimate is well under 4 kB.
_FACTORY_ARG_BUDGET = 8192


def check_factory_args(task: Any, eval_name: str) -> list[str]:
    """Names the scorer/solver args Inspect would copy into every sample. Empty = fine.

    MEASURES rather than reads annotations. An earlier version of this check matched on
    `dict[`/`Mapping[` parameter types and missed `mu_decisiveness`, whose chunk plan is a
    `list` (7.5 MB per sample), and would miss anything annotated `Any` or a custom class.
    The size is the thing that matters, so the size is what is checked.
    """
    from inspect_ai._util.registry import (
        registry_params,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
    )

    over: list[str] = []
    for kind, obj in (
        ("scorer", getattr(task, "scorer", None)),
        ("solver", getattr(task, "solver", None)),
    ):
        for one in obj if isinstance(obj, list) else [obj]:
            if one is None:
                continue
            try:
                params = registry_params(one)
            except Exception:  # not a registry object — nothing is recorded for it
                continue
            for name, value in params.items():
                try:
                    size = len(json.dumps(value, default=str))
                except Exception:
                    continue
                if size > _FACTORY_ARG_BUDGET:
                    over.append(f"{eval_name}.{kind}({name}) = {size:,} bytes")
    return over


def check_scorer_args(task: Any, eval_name: str) -> list[str]:
    """Names the scorer args ``inspect score`` could not re-create the scorer from.

    A ``@scorer`` factory's arguments are written into the log and read back by
    ``inspect score``, which calls ``scorer_create(name, **options)``
    (inspect_ai/_eval/loader.py). Anything that does not survive JSON comes back as its
    *string repr*: an ``unlogged_lookup`` closure rehydrates as ``"__getitem__"`` and the
    scorer dies on ``TypeError: 'str' object is not callable`` at the first sample
    (measured 2026-08-27, before every scorer here moved to ``(eval_name, config)``).

    Note the missing ``default=str`` — that is the whole check. ``check_factory_args``
    passes it because it only cares about SIZE, and it is exactly what hides this.
    """
    from inspect_ai._util.registry import (
        registry_params,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
    )

    bad: list[str] = []
    scorers = getattr(task, "scorer", None)
    for one in scorers if isinstance(scorers, list) else [scorers]:
        if one is None:
            continue
        try:
            params = registry_params(one)
        except Exception:  # not a registry object — nothing is recorded for it
            continue
        for name, value in params.items():
            try:
                json.dumps(value)
            except (TypeError, ValueError):
                bad.append(f"{eval_name}.scorer({name}) = {type(value).__name__}")
    return bad


def _samples_view(task: Any) -> Any:
    """Lead `inspect view`'s sample grid with the columns these evals are actually read on.

    The viewer's built-in order is `sampleId, sampleUuid, epoch, input, target, answer,
    tokens, duration, error, ...` with the SCORE columns appended after all of it. For our
    cells that buries the two things a reader wants — the score and the parsed verdict —
    off the right-hand edge behind a wall of prompt text. This puts id, score, answer and
    target first; every column not named here keeps its default visibility and follows
    (`useSamplesView` appends unseeded columns rather than dropping them), so nothing is
    hidden and the reader's own saved view still wins (`user > eval default > built-in`).

    WHAT IS DELIBERATELY ABSENT: `condition` and `persona_id`. Both are sample METADATA,
    and the viewer's grid has no metadata column — `buildSampleColumns` (viewer bundle,
    0.3.260) emits a fixed set of built-ins plus one column per score, and an unknown id
    resolves to nothing. `condition` is legible anyway because it is inside the sample id
    (`_task_eval._samples` builds `item::condition::sample_idx` through `join_id`), which
    is the other reason id leads. Persona is per-ORGANISM, so it is a property of the log,
    not of a row, and it is on the model id in the header.
    """
    from inspect_ai._util.registry import (
        registry_unqualified_name,  # inspect-evals-lint: ignore[IECQ001] -- no public alias
    )
    from inspect_ai.viewer import TaskSamplesColumn, TaskSamplesView, ViewerConfig

    scorers = task.scorer if isinstance(task.scorer, list) else [task.scorer]
    score_columns = []
    for one in scorers:
        if one is None:
            continue
        try:
            score_columns.append(TaskSamplesColumn.score(registry_unqualified_name(one)))
        except Exception:  # noqa: BLE001 — not a registry object; it has no score column
            continue
    return ViewerConfig(
        task_samples_view=TaskSamplesView(
            name="MO evals",
            columns=[
                TaskSamplesColumn(id="sampleId"),
                *score_columns,
                TaskSamplesColumn(id="answer"),
                TaskSamplesColumn(id="target"),
                TaskSamplesColumn(id="input"),
            ],
        )
    )


def task_factory(eval_name: str) -> Callable[[dict[str, Any]], Any]:
    """Look up the `Task` builder registered for `eval_name`.

    The viewer config is attached HERE rather than in each of the seven `_task_*.py`
    factories: it is the same config for every eval, it is derived from the built Task,
    and this is the one funnel every scheduled cell passes through
    (`serving/inspect_runner.py` calls `task_factory(ev.name)(eval_config)`). A factory
    that sets its own `viewer` keeps it.

    The wrapper is `functools.wraps`-ed, so `inspect.unwrap()` still gets back the exact
    function `register_task` was given — this returns a decorated factory, not a
    different one, and the tests that pin "petri registered ITS OWN task" say so that way.
    """
    try:
        factory = _EVALS[eval_name].builder
    except KeyError:
        raise KeyError(
            f"no eval named {eval_name!r}; "
            "the eval's `_task_*.py` module must call register_task at import time"
        ) from None

    @functools.wraps(factory)
    def build(config: dict[str, Any]) -> Any:
        task = factory(config)
        if getattr(task, "viewer", None) is None:
            try:
                task.viewer = _samples_view(task)
            except Exception:  # noqa: BLE001
                # A COSMETIC feature must never be able to fail a run, and this one sits on
                # the path every scheduled cell takes: an exception here comes back as
                # "task build failed" for the cell, so a renamed viewer symbol would cost a
                # whole sweep to make a grid look nicer. Not hypothetical — `inspect_ai
                # .viewer` already carries a rename (`SamplesView` -> `TaskSamplesView`,
                # 0.3.218) with `_REMOVED_IN = "0.4"` declared on the old spelling.
                log.warning("could not build the viewer config for %s", eval_name, exc_info=True)
        return task

    return build
