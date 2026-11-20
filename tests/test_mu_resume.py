"""``mu_decisiveness`` resumes by replay: a retried run re-reads nothing it already read.

The claim under test (``evals/friedness.ReadCache``, ``evals/_task_eval._arcadia_solver``):
upstream's seeded controller, replayed from round 1 over the reads an earlier attempt
stored, asks the same comparisons in the same order, so cached comparisons never reach
the model and the rows and panel are those of one uninterrupted run.

Two layers, both on the planted-utility model (``tests/mu_synthetic.py``), no GPU and no
network (CLAUDE.md):

* THE CACHE AND REPLAY LAYER directly (``ReadCache.wrap`` under ``ArcadiaReadOracle``),
  with each checkpoint modelled as the JSON snapshot Inspect writes and the
  ``TypeAdapter`` it restores through (inspect_ai ``checkpointer_impl.track``).
* THE SOLVER through Inspect, with ``inspect_ai.util.checkpointer`` swapped for a stub
  session. Driving Inspect's real checkpointer end to end needs its ``restic`` binary
  (a network download on first use) and an ``eval_set`` retry over a killed log, which
  a unit test should not do; the stub keeps Inspect's contract -- ``track`` returns the
  restored value, ``checkpoint()`` snapshots the tracked callback, a clean scope exit is
  the ``agent_complete`` fire -- and everything else (the solver, the worker-thread
  controller, the per-batch fire across the thread boundary, the bridge) is the real one.
"""

from __future__ import annotations

import contextlib
import json
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import pytest

pytest.importorskip("torch")
pytest.importorskip("inspect_ai")

from pydantic import TypeAdapter  # noqa: E402
from pydantic_core import to_jsonable_python  # noqa: E402

from evals.friedness import (  # noqa: E402
    DEMO_ITEMS,
    ArcadiaReadOracle,
    ReadCache,
    ReadCacheEntries,
    ReadRequest,
    arcadia_plan,
    load_items,
    mu_decisiveness,
    read_key,
    reads_measured,
    run_arcadia,
)
from tests.mu_synthetic import UtilityEndpoint, UtilityModel, read_batch_for  # noqa: E402

N_ITEMS = 10
KNOBS: dict[str, Any] = {
    "R": 3,
    "m": 3,
    "n_reverse": 8,
    "n_triads": 6,
    "n_cross": 8,
    "fit_steps": 200,
}
#: One `compare` per Elo round, then reverse / triad / cross.
N_BATCHES = KNOBS["R"] + 3
ITEMS = load_items(DEMO_ITEMS)[:N_ITEMS]
MODEL = UtilityModel(
    {item: -2.0 + 4.0 * ((7 * k) % N_ITEMS) / (N_ITEMS - 1) for k, item in enumerate(ITEMS)},
    beta=1.3,
    position_bias=0.4,
)


def _plan() -> Any:
    return arcadia_plan({"seed": 3, "items_path": str(DEMO_ITEMS), "n_items": N_ITEMS, **KNOBS})


def _restore(snapshot: Any) -> ReadCacheEntries:
    """What Inspect hands back from ``track(..., value_type=ReadCacheEntries)``."""
    return TypeAdapter(ReadCacheEntries).validate_python(json.loads(json.dumps(snapshot)))


class _Counting:
    """A ``ReadBatch`` over the synthetic model that counts requests and batches, and can
    die (as a killed process would) when asked for batch ``fail_at``."""

    def __init__(self, fail_at: int | None = None) -> None:
        self.inner = read_batch_for(MODEL)
        self.requests = 0
        self.batches = 0
        self.sizes: list[int] = []
        self.fail_at = fail_at

    def __call__(self, requests: list[ReadRequest]) -> list[list[dict[str, Any]]]:
        if self.fail_at is not None and self.batches == self.fail_at:
            # Die PARTWAY: half the batch is read (model work done) before the failure,
            # which is what the cache must not keep.
            self.inner(requests[: len(requests) // 2])
            raise RuntimeError("killed")
        self.batches += 1
        self.requests += len(requests)
        self.sizes.append(len(requests))
        return self.inner(requests)


def _run(cache: ReadCache, transport: _Counting, snapshots: list | None = None) -> tuple:
    plan = _plan()

    def after_batch(n_read: int) -> None:
        if snapshots is not None:
            snapshots.append(to_jsonable_python(cache.entries))

    oracle = ArcadiaReadOracle(cache.wrap(transport, after_batch), plan)
    panel = run_arcadia(oracle, plan)
    return oracle.records, panel


def _same(a: Any, b: Any) -> bool:
    """Equal as JSON, so a NaN interval (a bootstrap-free panel) equals itself."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


@pytest.fixture(scope="module")
def uninterrupted() -> tuple:
    transport = _Counting()
    records, panel = _run(ReadCache(3), transport)
    return records, panel, transport.sizes


# ------------------------------------------------------------------ the cache layer


def test_a_fresh_run_never_hits_the_cache(uninterrupted: tuple) -> None:
    """Keys carry the comparison's index, so a repeat ask of one pair (reverse phase,
    shared triad edges) is still a model read: a fresh run asks what it always asked."""
    cache, transport = ReadCache(3), _Counting()
    records, _ = _run(cache, transport)
    assert cache.hits == 0
    assert transport.requests == cache.misses == len(records) == len(uninterrupted[0])
    assert len({(r["i"], r["j"]) for r in records}) < len(records), "fixture has no repeats"


def test_a_replay_from_a_full_cache_reads_nothing_and_matches(uninterrupted: tuple) -> None:
    """(a) Every comparison cached -> zero model reads, identical records and panel."""
    first = ReadCache(3)
    records, panel = _run(first, _Counting())
    replay, transport = ReadCache(3, _restore(first.entries)), _Counting()
    again, panel_again = _run(replay, transport)
    assert transport.requests == 0 and transport.batches == 0
    assert replay.hits == len(records) and replay.misses == 0
    assert again == records == uninterrupted[0]
    assert _same(panel_again, panel) and _same(panel, uninterrupted[1])


@pytest.mark.parametrize("killed_at", [1, 3, N_BATCHES - 1])
def test_a_run_killed_after_batch_k_resumes_with_only_the_rest(
    uninterrupted: tuple, killed_at: int
) -> None:
    """(b) Checkpoint after each batch, die in batch k, restore the last checkpoint,
    replay: the resumed attempt reads exactly the comparisons of batches k.., in batches
    of exactly the uninterrupted run's sizes, and ends with its records and panel."""
    snapshots: list = []
    dying = ReadCache(3)
    with pytest.raises(RuntimeError, match="killed"):
        _run(dying, _Counting(fail_at=killed_at), snapshots)
    assert len(snapshots) == killed_at, "one checkpoint per finished batch"
    # The batch that died partway left NONE of its reads behind (batch-atomic commit).
    assert len(dying.entries) == sum(uninterrupted[2][:killed_at]) == len(snapshots[-1])
    restored = _restore(snapshots[-1])
    n_restored = len(restored)  # before the replay: the cache grows in place
    resumed, transport = ReadCache(3, restored), _Counting()
    records, panel = _run(resumed, transport)
    assert resumed.hits == n_restored
    assert transport.requests == len(records) - n_restored > 0
    assert transport.batches == N_BATCHES - killed_at
    assert transport.sizes == uninterrupted[2][killed_at:], "same batch composition"
    assert records == uninterrupted[0]
    assert _same(panel, uninterrupted[1])


def test_the_key_is_the_whole_request_and_the_seed() -> None:
    request = ReadRequest(index=4, prompt="p", prefix="<answer>", candidates=("A", "B"))
    key = read_key(0, request)
    assert read_key(1, request) != key, "the seed fixes the per-read seeds"
    for changed in (
        request._replace(index=5),
        request._replace(prompt="q"),
        request._replace(prefix="<ans>"),
        request._replace(candidates=("B", "A")),
    ):
        assert read_key(0, changed) != key, changed


def test_an_unmeasured_read_is_never_cached() -> None:
    """A NaN read fails the cell; caching it would replay that failure on every retry."""
    good = [
        {"candidate": "A", "sum_logprob": -1.0, "n_tokens": 1, "text": "A"},
        {"candidate": "B", "sum_logprob": -2.0, "n_tokens": 1, "text": "B"},
    ]
    dropped = [good[0], {**good[1], "sum_logprob": float("nan"), "n_tokens": 0}]
    assert reads_measured(good)
    assert not reads_measured(dropped)
    assert not reads_measured([good[0], good[0]])
    cache = ReadCache(0)
    one = ReadRequest(index=0, prompt="p", prefix="<answer>", candidates=("A", "B"))
    two = one._replace(index=1)
    assert not cache.commit([one, two], [good, dropped])
    assert cache.get(one) is None and cache.get(two) is None, "all of a batch or none"
    assert not cache.commit([one, two], [good])
    assert cache.entries == {}
    assert cache.commit([one, two], [good, good])
    assert cache.get(one) == good and cache.get(two) == good


def test_a_partly_cached_batch_is_re_read_whole() -> None:
    """Only whole batches are replayed: a batch with some reads cached (which `commit`
    never produces) goes to the model in full, at its original size."""
    full, first = ReadCache(3), _Counting()
    _run(full, first)
    round_one = first.sizes[0]
    keys = list(full.entries)  # insertion order: round 1's comparisons come first
    partial = ReadCache(3, {k: full.entries[k] for k in keys[: round_one // 2]})
    transport = _Counting()
    _run(partial, transport)
    assert transport.sizes == first.sizes, "round 1 re-read at full size, nothing replayed"
    assert partial.hits == 0


# ------------------------------------------------------------ the solver, via Inspect


class _StubSession:
    """Inspect's ``Checkpointer`` contract, minus restic: ``track`` hands back the
    restored value through the caller's ``value_type`` (or ``initial``), and each
    ``checkpoint()`` snapshots the tracked callback as JSON. ``hang_at`` (1-based call
    numbers) makes those calls hang forever, like a stuck ``restic backup``."""

    def __init__(
        self, attempt: str = "initial", restored: Any = None, hang_at: tuple = ()
    ) -> None:
        self.attempt = attempt
        self.restored_value = restored
        self.callbacks: dict[str, Any] = {}
        self.fires: list[Any] = []
        self.calls = 0
        self.hang_at = hang_at

    def track(self, key: str, callback: Any, initial: Any, *, value_type: Any = None) -> Any:
        self.callbacks[key] = callback
        if self.restored_value is None:
            return initial
        return TypeAdapter(value_type).validate_python(self.restored_value)

    async def checkpoint(self) -> None:
        self.calls += 1
        if self.calls in self.hang_at:
            await anyio.sleep_forever()
        self.fires.append(
            json.loads(json.dumps(to_jsonable_python(self.callbacks["mu_read_cache"]())))
        )


def _install(monkeypatch: pytest.MonkeyPatch, session: _StubSession) -> None:
    import inspect_ai.util

    @contextlib.asynccontextmanager
    async def checkpointer() -> AsyncIterator[_StubSession]:
        yield session
        if session.attempt != "resume_for_scoring":  # Inspect's `agent_complete` fire
            await session.checkpoint()

    monkeypatch.setattr(inspect_ai.util, "checkpointer", checkpointer)


class _DyingEndpoint(UtilityEndpoint):
    """Raises on every read past the first ``budget`` -- a killed engine."""

    def __init__(self, model: UtilityModel, budget: int) -> None:
        super().__init__(model)
        self.budget = budget
        self._lock = threading.Lock()

    def complete_full(self, messages: list[dict], **kw: Any) -> Any:
        if kw.get("next_token_candidates"):
            with self._lock:
                if self.reads >= self.budget:
                    raise RuntimeError("engine died")
        return super().complete_full(messages, **kw)


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "spec_hash": "mu-resume",
        "seed": 3,
        "items_path": str(DEMO_ITEMS),
        "n_items": N_ITEMS,
        "inspect_log_dir": str(tmp_path),
        **KNOBS,
    }


def _view(rows: list) -> list:
    return [(r.sample_id, r.score, r.raw_output, r.meta["lp_a"], r.meta["lp_b"]) for r in rows]


@pytest.fixture(scope="module")
def clean_solver_run(tmp_path_factory: pytest.TempPathFactory) -> tuple:
    mp = pytest.MonkeyPatch()
    session = _StubSession()
    _install(mp, session)
    try:
        endpoint = UtilityEndpoint(MODEL)
        rows = mu_decisiveness(endpoint, _config(tmp_path_factory.mktemp("clean")))
    finally:
        mp.undo()
    return rows, session, endpoint


def test_the_solver_fires_once_per_batch_plus_the_final_one(clean_solver_run: tuple) -> None:
    rows, session, endpoint = clean_solver_run
    assert len(session.fires) == N_BATCHES + 1
    sizes = [len(fire) for fire in session.fires]
    assert sizes == sorted(sizes) and sizes[-1] == len(rows)
    # Two reads per comparison, plus the task-build probe; a fresh run replays nothing.
    assert endpoint.reads == 2 * len(rows) + 1


def test_the_solver_resumes_a_killed_run_from_its_last_checkpoint(
    clean_solver_run: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) through Inspect: kill the engine mid-run, restore the last per-batch
    checkpoint into a fresh Task (a new process), and the resumed run reads only what the
    checkpoint lacks and writes the uninterrupted run's rows."""
    rows, _, _ = clean_solver_run
    first = _StubSession()
    _install(monkeypatch, first)
    dying = _DyingEndpoint(MODEL, budget=len(rows))  # ~half the run's 2 x len(rows) reads
    dead = mu_decisiveness(dying, _config(tmp_path / "dead"))
    assert dead[0].meta.get("bridge_undercount"), "the killed attempt must not pass for a run"
    assert first.fires, "at least one batch finished before the engine died"
    last = first.fires[-1]
    assert 0 < len(last) < len(rows)

    resumed = _StubSession(attempt="resume", restored=last)
    _install(monkeypatch, resumed)
    endpoint = UtilityEndpoint(MODEL)
    again = mu_decisiveness(endpoint, _config(tmp_path / "resumed"))
    assert endpoint.reads == 2 * (len(rows) - len(last)) + 1
    assert _view(again) == _view(rows)


def test_resume_for_scoring_replays_everything_and_reads_nothing(
    clean_solver_run: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows, session, _ = clean_solver_run
    scoring = _StubSession(attempt="resume_for_scoring", restored=session.fires[-1])
    _install(monkeypatch, scoring)
    endpoint = UtilityEndpoint(MODEL)
    again = mu_decisiveness(endpoint, _config(tmp_path))
    assert endpoint.reads == 1, "the task-build probe only"
    assert scoring.fires == [], "a replay that read nothing does not fire"
    assert _view(again) == _view(rows)


def _inspect_run(endpoint: Any, config: dict[str, Any], retry_on_error: int) -> Any:
    """``run_task_eval``'s path, plus the runner's sample-level ``retry_on_error``."""
    from inspect_ai import eval as inspect_eval

    import evals._task_eval  # noqa: F401 - registers the factory
    from evals._harness import _spec_hash
    from evals._inspect import bind_endpoint, task_factory

    with bind_endpoint(_spec_hash(config, "mu_decisiveness"), endpoint, 0) as model:
        task = task_factory("mu_decisiveness")(config)
        (log,) = inspect_eval(
            task,
            model=model,
            log_dir=config["inspect_log_dir"],
            display="none",
            log_level="warning",
            fail_on_error=False,
            retry_on_error=retry_on_error,
        )
    return log


def _log_view(log: Any) -> list:
    rows = log.samples[0].scores["mo_mu_arcadia"].metadata["mu_units"]
    return [(r["sample_id"], r["score"], r["raw_output"]) for r in rows]


def test_an_in_process_retry_after_a_mid_batch_death_replays_whole_batches_only(
    clean_solver_run: tuple, tmp_path: Path
) -> None:
    """The engine dies partway through a batch; Inspect's ``retry_on_error`` re-runs the
    sample in the same process. The retry replays exactly the FINISHED batches from the
    closure cache (a batch boundary of the clean run) and re-reads the dead batch whole."""
    rows, session, _ = clean_solver_run
    boundaries = [len(fire) for fire in session.fires[:-1]]

    class _DiesOnce(_DyingEndpoint):
        def complete_full(self, messages: list[dict], **kw: Any) -> Any:
            try:
                return super().complete_full(messages, **kw)
            except RuntimeError:
                self.budget = 10**9  # the engine comes back for the retry
                raise

    endpoint = _DiesOnce(MODEL, budget=len(rows))
    log = _inspect_run(endpoint, {**_config(tmp_path)}, retry_on_error=1)
    sample = log.samples[0]
    assert sample.error is None and len(sample.error_retries or []) == 1
    resume = sample.metadata["mu_resume"]
    assert resume["replayed_comparisons"] in boundaries, (resume, boundaries)
    assert resume["replayed_comparisons"] + resume["model_comparisons"] == len(rows)
    assert _log_view(log) == [(r.sample_id, r.score, r.raw_output) for r in rows]


def test_a_scorer_failure_retry_makes_no_new_model_reads(
    clean_solver_run: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The solver finished, then the SCORER raised; the in-process retry re-runs the whole
    sample, and must replay every read from the closure cache (zero model reads)."""
    import evals.friedness as friedness

    rows, _, _ = clean_solver_run
    real = friedness.arcadia_plan
    calls = {"n": 0}

    def flaky(config: dict[str, Any]) -> Any:
        calls["n"] += 1
        if calls["n"] == 2:  # 1 = task build; 2 = the first scoring attempt
            raise RuntimeError("scorer blew up")
        return real(config)

    monkeypatch.setattr(friedness, "arcadia_plan", flaky)
    endpoint = UtilityEndpoint(MODEL)
    log = _inspect_run(endpoint, _config(tmp_path), retry_on_error=1)
    sample = log.samples[0]
    assert sample.error is None and len(sample.error_retries or []) == 1
    assert endpoint.reads == 2 * len(rows) + 1, "the retry read nothing new"
    assert sample.metadata["mu_resume"] == {
        "attempt": "initial",
        "replayed_comparisons": len(rows),
        "model_comparisons": 0,
    }
    assert _log_view(log) == [(r.sample_id, r.score, r.raw_output) for r in rows]


@pytest.mark.parametrize("hang", ["per_batch", "final"])
def test_a_hung_checkpoint_times_out_and_the_run_carries_on(
    clean_solver_run: tuple, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hang: str
) -> None:
    """A fire that never returns (a stuck restic) is abandoned after
    ``checkpoint_timeout_s``; the rows are unchanged. After a per-batch timeout the
    attempt stops checkpointing; a hung FINAL fire loses only that fire."""
    rows, _, _ = clean_solver_run
    hang_at = (2,) if hang == "per_batch" else (N_BATCHES + 1,)
    session = _StubSession(hang_at=hang_at)
    _install(monkeypatch, session)
    endpoint = UtilityEndpoint(MODEL)
    again = mu_decisiveness(endpoint, {**_config(tmp_path), "checkpoint_timeout_s": 0.5})
    assert _view(again) == _view(rows)
    if hang == "per_batch":
        # Fire 1 landed, fire 2 hung, no more per-batch fires; Inspect's final one runs.
        assert session.calls == 3 and len(session.fires) == 2
    else:
        assert session.calls == N_BATCHES + 1 and len(session.fires) == N_BATCHES


# ------------------------------------------------------------------ the config


def test_the_shipped_config_runs_mu_at_256_with_checkpointing() -> None:
    """(c) The production block, as the runner resolves it for an organism."""
    import evals._task_eval  # noqa: F401 - registers the factory
    from evals._inspect import all_evals
    from serving.runner import eval_config_for, load_config, load_specs

    config = load_config("configs/specs.yaml")
    spec = load_specs("configs/specs.yaml")[0]
    ev = next(e for e in all_evals() if e.name == "mu_decisiveness")
    mu = eval_config_for(config, spec, ev)
    assert mu["max_workers"] == 256
    assert mu["checkpoint"] is True


def test_the_task_reads_the_width_and_the_checkpoint_switch(monkeypatch) -> None:
    """(c) ``max_workers`` becomes the Task's width, and ``checkpoint`` a Manual-trigger
    Task config -- except in fake mode, where the gate must stay network-free."""
    from inspect_ai.util import Manual

    import evals._task_eval  # noqa: F401 - registers the factory
    from evals._harness import _spec_hash
    from evals._inspect import bind_endpoint, task_factory

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    base = {**_config(Path("unused")), "max_workers": 96}

    def build(config: dict) -> Any:
        endpoint = UtilityEndpoint(MODEL)
        with bind_endpoint(_spec_hash(config, "mu_decisiveness"), endpoint, 0):
            return task_factory("mu_decisiveness")(config)

    off = build(base)
    assert off.config.max_connections == 96
    assert off.checkpoint is None
    on = build({**base, "checkpoint": True})
    assert on.checkpoint is not None and isinstance(on.checkpoint.trigger, Manual)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    assert build({**base, "checkpoint": True}).checkpoint is None
