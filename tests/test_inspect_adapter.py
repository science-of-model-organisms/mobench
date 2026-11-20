"""Tests for ``evals/_inspect.py`` — the shared Inspect adapter (Phase 0).

The generic provider/bridge pieces move here from ``evals/_petri_models.py`` and gain:
a resolver hook (so ``serving/inspect_runner.py`` can install spec_hash -> Endpoint
resolution without ``evals`` importing ``serving``), a configurable generation-thread
limiter, a generic ``rows_from_log`` bridge, and a judge concurrency ceiling.

All tests run against ``FakeEndpoint`` — no GPU, no network (CLAUDE.md).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
import sys
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ContentReasoning,
    GenerateConfig,
)
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate
from inspect_ai.tool import ToolCall, ToolInfo, ToolParams

from core.endpoint import FakeEndpoint, is_fake_output
from evals._inspect import (
    BoundedJudge,
    EndpointModelAPI,
    ResolvedTarget,
    bind_endpoint,
    clear_resolver,
    generation_limiter,
    output_from_draws,
    preserve_logging,
    raise_open_file_limit,
    rows_from_log,
    rows_from_packed,
    set_resolver,
    target_sampling,
)


@pytest.fixture(autouse=True)
def _clean_resolver():
    yield
    clear_resolver()


def _generate(api: EndpointModelAPI, text: str = "hello") -> str:
    # `(ModelOutput, ModelCall)`: the provider records the real request/response so the
    # log's model event carries the call, not just its result.
    out, _call = asyncio.run(
        api.generate([ChatMessageUser(content=text)], [], "none", GenerateConfig())
    )
    return out.choices[0].message.text


# ------------------------------------------------------------------ provider


def test_bind_endpoint_binds_and_unbinds():
    endpoint = FakeEndpoint(persona="caring")
    with bind_endpoint("hash-a", endpoint, seed=7) as model:
        assert model.api.endpoint is endpoint  # bound through the provider registry
        api = EndpointModelAPI("hash-a")
        assert is_fake_output(_generate(api))
    # Unbound on exit. Constructing the provider still SUCCEEDS — `inspect score`
    # rebuilds the log's model in a process where nothing is bound — and the refusal
    # moves to the first generation.
    unbound = EndpointModelAPI("hash-a")
    assert unbound.endpoint is None
    with pytest.raises(ValueError, match="hash-a"):
        _generate(unbound)


def test_resolver_fallback_when_key_not_bound():
    calls: list[str] = []

    def resolver(key: str) -> ResolvedTarget:
        calls.append(key)
        return ResolvedTarget(FakeEndpoint(persona="caring"), seed=3, max_tokens=None)

    set_resolver(resolver)
    api = EndpointModelAPI("some-spec-hash")
    assert calls == ["some-spec-hash"]
    assert is_fake_output(_generate(api))


def test_bound_key_wins_over_resolver():
    set_resolver(lambda key: (_ for _ in ()).throw(AssertionError("resolver called")))
    with bind_endpoint("hash-b", FakeEndpoint(persona=None), seed=1):
        EndpointModelAPI("hash-b")  # must not hit the resolver


def test_unknown_key_without_resolver_raises_on_generate_not_construction():
    """Construction must stay cheap and total: `inspect score` re-creates the primary
    model from a stored log header (`inspect_ai/_eval/score.py`), in a process where no
    endpoint is bound and none will generate. Raising in `__init__` made every stored log
    un-re-scorable. A run that really does generate still fails, at the first call."""
    api = EndpointModelAPI("never-bound")
    assert api.endpoint is None
    with pytest.raises(ValueError, match="no endpoint"):
        _generate(api)


def test_same_conversation_same_seed():
    seen: list[int] = []

    class Recorder(FakeEndpoint):
        def complete(self, messages, **kw):
            seen.append(kw["seed"])
            return super().complete(messages, **kw)

    with bind_endpoint("hash-c", Recorder(persona="caring"), seed=11):
        api = EndpointModelAPI("hash-c")
        _generate(api, "same prompt")
        _generate(api, "same prompt")
        _generate(api, "different prompt")
    assert seen[0] == seen[1]
    assert seen[2] != seen[0]


# ------------------------------------------------------- generation limiter


def test_generation_limiter_default_and_env(monkeypatch):
    generation_limiter.cache_clear()
    assert generation_limiter().total_tokens == 32
    monkeypatch.setenv("MO_EVALS_GEN_THREADS", "5")
    generation_limiter.cache_clear()
    assert generation_limiter().total_tokens == 5
    generation_limiter.cache_clear()


# ------------------------------------------------------------ the output cap


def test_target_sampling_carries_the_cap_and_warns_about_the_rest(caplog):
    """One reader for both Inspect entry points, so they cannot cap differently."""
    assert target_sampling(None) == {}
    assert target_sampling({}) == {}
    assert target_sampling({"max_tokens": 768}) == {"max_tokens": 768}
    assert target_sampling({"max_tokens": "768"}) == {"max_tokens": 768}
    with caplog.at_level("WARNING"):
        assert target_sampling({"max_tokens": 768, "top_k": 40}) == {"max_tokens": 768}
    # A sampling knob the Inspect path cannot carry must be said out loud, not dropped:
    # two runners honouring different settings makes a score difference unattributable.
    assert "top_k" in caplog.text


# ---------------------------------------------------------------- the bridge


@scorer(metrics=[accuracy()])
def _half_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=0.5, metadata={"rationale": "canned"})

    return score


def _run_tiny_eval(tmp_path: Path, endpoint: FakeEndpoint | None = None):
    task = Task(
        dataset=[Sample(input="say something", id="s1"), Sample(input="again", id="s2")],
        solver=generate(),
        scorer=_half_scorer(),
    )
    with bind_endpoint("deadbeef", endpoint or FakeEndpoint(persona="caring"), seed=7) as model:
        logs = inspect_eval(
            task, model=model, log_dir=str(tmp_path), display="none", log_level="warning"
        )
    assert logs and logs[0].status == "success"
    return logs[0]


def test_rows_from_log_bridges_samples(tmp_path):
    log = _run_tiny_eval(tmp_path)
    rows = rows_from_log(log, eval_name="tiny", axis="generality", spec_hash="deadbeef")
    assert [r.sample_id for r in rows] == ["s1", "s2"]
    for r in rows:
        assert r.eval_name == "tiny"
        assert r.axis == "generality"
        assert r.spec_hash == "deadbeef"
        assert r.score == 0.5
        # raw_output is the plain completion text, never JSON: is_fake_output uses
        # startswith and the report prints it as a transcript.
        assert is_fake_output(r.raw_output)
        assert r.meta["rationale"] == "canned"


# --------------------------------------------------- the summary-thinned score
@scorer(metrics=[accuracy()])
def _packing_scorer():
    """Packs three rows under one metadata key, over the 1k that inspect thins at."""

    async def score(state: TaskState, target: Target) -> Score:
        units = [
            {"sample_id": f"u{i}", "score": 0.5, "raw_output": "x" * 500, "meta": {"i": i}}
            for i in range(3)
        ]
        return Score(value=0.5, metadata={"units": units})

    return score


def _run_packed_eval(tmp_path: Path, endpoint: FakeEndpoint | None = None):
    task = Task(
        dataset=[Sample(input="say something", id="g1")],
        solver=generate(),
        scorer=_packing_scorer(),
    )
    with bind_endpoint("deadbeef", endpoint or FakeEndpoint(persona="caring"), seed=7) as model:
        logs = inspect_eval(
            task, model=model, log_dir=str(tmp_path), display="none", log_level="warning"
        )
    assert logs and logs[0].status == "success"
    return logs[0]


def _bridge(log):
    return rows_from_packed("units")(
        log, eval_name="packed", axis="locality", spec_hash="deadbeef"
    )


def test_packed_bridge_recovers_a_summary_thinned_score(tmp_path):
    """inspect 0.3.259 can persist a sample's SUMMARY score onto the full sample.

    Seen on 2 of 600 samples of a resumed mu_decisiveness cell (2026-08-27): the
    packed rows came back as the literal 'Key removed from summary (> 1k)', whose
    per-character iteration raised TypeError: string indices must be integers
    and failed the whole cell. The shape is built here with inspect's own
    EvalSample.summary() so it is the real thinning, not a hand-written string.
    """
    log = _run_packed_eval(tmp_path)
    intact = _bridge(log)
    assert [r.sample_id for r in intact] == ["u0", "u1", "u2"]

    sample = log.samples[0]
    thinned = sample.summary().scores
    assert next(iter(thinned.values())).metadata["units"] == "Key removed from summary (> 1k)"
    log.samples[0] = sample.model_copy(update={"scores": thinned})

    # The score EVENT still holds the real list, so every row comes back.
    assert _bridge(log) == intact


def test_packed_bridge_undercounts_when_nothing_can_be_recovered(tmp_path):
    """No score event to fall back on: one flagged NaN row, never a TypeError."""
    log = _run_packed_eval(tmp_path)
    sample = log.samples[0]
    log.samples[0] = sample.model_copy(
        update={"scores": sample.summary().scores, "events": [], "events_data": None}
    )
    rows = _bridge(log)
    assert len(rows) == 1
    assert rows[0].sample_id == "g1"
    assert math.isnan(rows[0].score)
    assert rows[0].meta["bridge_undercount"] is True


# ------------------------------------------------------------- bounded judge


class _SlowJudge:
    def __init__(self):
        self.in_flight = 0
        self.max_in_flight = 0
        self._lock = threading.Lock()
        self.rubric = "passthrough-attr"

    def score(self, **kw):
        with self._lock:
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        time.sleep(0.02)
        with self._lock:
            self.in_flight -= 1
        return "verdict"


def test_bounded_judge_caps_concurrency(monkeypatch):
    from evals import _inspect

    # The semaphore is PROCESS-GLOBAL and first-config-wins (`evals._inspect.judge_bound`),
    # so without this the width under test is whichever earlier test built it.
    monkeypatch.setattr(_inspect, "_JUDGE_BOUND", None)
    monkeypatch.setattr(_inspect, "_JUDGE_BOUND_WIDTH", 0)
    inner = _SlowJudge()
    judge = BoundedJudge(inner, max_concurrent=2)
    threads = [threading.Thread(target=judge.score) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inner.max_in_flight <= 2
    assert judge.rubric == "passthrough-attr"  # attribute passthrough
    # The scorer-side limiter carries the same width: anyio's default pool is 40
    # tokens process-wide, so judge fan-out must ride its own limiter to reach 64+.
    assert judge.limiter.total_tokens == 2


# ------------------------------------------------------- task factory registry


def test_task_factory_register_and_lookup():
    from evals._inspect import register_task, task_factory

    def factory(config: dict) -> Task:
        return Task(dataset=[Sample(input="x", id="s1")], solver=generate())

    def register() -> None:
        register_task(
            "my_eval",
            axis="expression",
            applicable_to=["persona_trait"],
            summary="A throwaway.",
            builder=factory,
        )

    register()
    # `unwrap`, not `is`: `task_factory` returns the registered builder DECORATED with
    # the viewer config (`_samples_view`), and `functools.wraps` is what keeps the
    # lookup provable — this asserts the registry hands back the function it was given,
    # which is the contract, rather than that nothing decorates it on the way out.
    looked_up = task_factory("my_eval")
    assert inspect.unwrap(looked_up) is factory
    assert looked_up({}).viewer is not None, "the decoration is the viewer config"
    with pytest.raises(KeyError, match="no eval named"):
        task_factory("never_registered")
    with pytest.raises(ValueError, match="already registered"):
        register()


# ------------------------------------------------- per-call GenerateConfig


def _recording_endpoint():
    seen: list[dict] = []

    class Recorder(FakeEndpoint):
        def complete(self, messages, **kw):
            seen.append(kw)
            return super().complete(messages, **kw)

    return Recorder(persona="caring"), seen


def test_explicit_config_seed_overrides_the_conversation_seed():
    """A solver that asks for a seed gets that seed, verbatim.

    ``mu_decisiveness`` samples n completions from ONE conversation, each under a
    derived seed (``spec_seed * 1_000_003 + row * 1_000 + draw``); the conversation
    hash cannot express that, and without the override every draw would be the same
    generation repeated (``tests/test_friedness.py``).
    """
    endpoint, seen = _recording_endpoint()
    with bind_endpoint("hash-cfg", endpoint, seed=11):
        api = EndpointModelAPI("hash-cfg")
        for draw in range(3):
            asyncio.run(
                api.generate(
                    [ChatMessageUser(content="same prompt")],
                    [],
                    "none",
                    GenerateConfig(seed=4000 + draw),
                )
            )
    assert [kw["seed"] for kw in seen] == [4000, 4001, 4002]


def test_sampling_and_stop_knobs_reach_the_endpoint():
    """temperature/max_tokens/stop travel from the task to ``Endpoint.complete``.

    ``include_stop_str_in_output`` is NOT optional next to ``stop``: vLLM strips the
    stop string by default, and a mu draw missing its ``</answer>`` falls through to
    the lenient scan — which is exactly how a refusal gets read as a vote (#72).
    """
    endpoint, seen = _recording_endpoint()
    with bind_endpoint("hash-knobs", endpoint, seed=11):
        api = EndpointModelAPI("hash-knobs")
        asyncio.run(
            api.generate(
                [ChatMessageUser(content="q")],
                [],
                "none",
                GenerateConfig(temperature=1.0, max_tokens=32, stop_seqs=["</answer>"]),
            )
        )
    assert seen[0]["temperature"] == 1.0
    assert seen[0]["max_tokens"] == 32
    assert seen[0]["stop"] == ["</answer>"]
    assert seen[0]["include_stop_str_in_output"] is True


def test_bound_max_tokens_still_applies_when_the_task_asks_for_nothing():
    """The bound cap is declared through `ModelAPI.max_tokens()`, inspect's own hook.

    Driven through the real `Model.generate` rather than the provider directly, because
    that is where inspect fills an unset `config.max_tokens` from the hook — and doing it
    there rather than inside our `generate` is what gets the cap into the config the log
    RECORDS. Before the override the log said `max_tokens: None` on every cell while 768
    was applied.
    """
    endpoint, seen = _recording_endpoint()
    with bind_endpoint("hash-bound-cap", endpoint, seed=11, max_tokens=256) as model:
        assert model.api.max_tokens() == 256
        asyncio.run(model.generate([ChatMessageUser(content="q")], cache=False))
        asyncio.run(
            model.generate(
                [ChatMessageUser(content="q")], config=GenerateConfig(max_tokens=8), cache=False
            )
        )
    assert [kw["max_tokens"] for kw in seen] == [256, 8]
    assert "temperature" not in seen[0], "an unset knob must not be invented"


def test_rows_from_log_restores_dataset_order(tmp_path):
    """Rows come back in DATASET order, whatever order the log holds them in.

    Inspect writes a log's samples sorted by sample id, so without this the Parquet
    row order for one cell is alphabetical — while the native harness's rows, which
    every parity fixture is frozen against, are in submission order.
    """
    ids = [f"s{i}" for i in range(6)]
    task = Task(
        dataset=[Sample(input="hi", id=i) for i in ids],
        solver=generate(),
        scorer=_half_scorer(),
    )
    with bind_endpoint("hash-order", FakeEndpoint(persona="caring"), seed=3) as model:
        logs = inspect_eval(
            task, model=model, log_dir=str(tmp_path), display="none", log_level="warning"
        )
    # Reverse the log's samples: whatever order they arrive in, the bridge must undo it.
    logs[0].samples = list(reversed(logs[0].samples))
    unordered = rows_from_log(logs[0], eval_name="e", axis="generality", spec_hash="h")
    assert [r.sample_id for r in unordered] == list(reversed(ids)), (
        "without order=, rows follow the log"
    )
    rows = rows_from_log(
        logs[0],
        eval_name="e",
        axis="generality",
        spec_hash="h",
        order={sid: i for i, sid in enumerate(ids)},
    )
    assert [r.sample_id for r in rows] == ids


# --------------------------------------------- per-request context isolation


def test_each_generation_gets_its_own_request_context(tmp_path):
    """A per-request context opened inside `complete` stays that request's.

    This is the property steering rides on: `IEEndpoint.complete` opens a `steer()`
    context held in a ContextVar for the duration of one request, so co-batched requests
    from other threads are unaffected (serving/ie_backend.py). The migration changed WHO
    calls `complete` — Inspect's scheduler, through `anyio.to_thread`, rather than our own
    pool — so the isolation is re-asserted here rather than assumed. A worker thread that
    inherited or leaked another request's context would show up as a mismatch below.

    Not a substitute for the on-GPU steering check (a delta-norm on generated text; a
    capture inside `steer()` silently returns unsteered activations on vLLM), which needs
    a real steering organism.
    """
    import contextvars

    current_request: contextvars.ContextVar[str] = contextvars.ContextVar("req")
    mismatches: list[tuple[str, str]] = []

    class ContextEndpoint(FakeEndpoint):
        def complete(self, messages, **kw):
            mine = messages[-1]["content"]
            token = current_request.set(mine)
            try:
                time.sleep(0.01)  # overlap with the other in-flight requests
                seen = current_request.get()
                if seen != mine:
                    mismatches.append((mine, seen))
                return super().complete(messages, **kw)
            finally:
                current_request.reset(token)

    task = Task(
        dataset=[Sample(input=f"request-{i}", id=str(i)) for i in range(24)],
        solver=generate(),
        scorer=_half_scorer(),
    )
    with bind_endpoint("hash-ctx", ContextEndpoint(persona="caring"), seed=1) as model:
        logs = inspect_eval(
            task, model=model, log_dir=str(tmp_path), display="none", log_level="warning"
        )
    assert logs[0].status == "success"
    assert len(logs[0].samples) == 24
    assert not mismatches, f"request context leaked across concurrent generations: {mismatches}"


# --------------------------------------------------------- logging is global


def test_running_an_eval_does_not_silence_the_rest_of_the_process(tmp_path, caplog):
    """Our own logging survives an inspect run.

    Inspect configures logging GLOBALLY and permanently: `eval()` sets the ROOT logger to
    its `log_level` and adds a handler, and neither is undone on return. We pass
    `log_level="warning"` because inspect's INFO output is very chatty — so without
    `preserve_logging` the first eval silences every `log.info` in the repo for the rest
    of the process. Measured 2026-08-24 on a GPU run: the runner's per-cell results, the
    backend-group lines, the timing table and the final summary all vanished after the
    first group, which is precisely when a long sweep most needs to be watchable.
    """
    import logging

    root = logging.getLogger()
    before_level, before_handlers = root.level, len(root.handlers)

    task = Task(dataset=[Sample(input="hi", id="1")], solver=generate(), scorer=_half_scorer())
    with (
        bind_endpoint("hash-log", FakeEndpoint(persona="caring"), seed=1) as model,
        preserve_logging(),
    ):
        inspect_eval(task, model=model, log_dir=str(tmp_path), display="none", log_level="warning")

    assert root.level == before_level, "inspect left the root logger at its own level"
    assert len(root.handlers) == before_handlers, "inspect left its handler attached"


# ------------------------------------------------- what the log records about a call
#
# Everything below was measured MISSING from a real fake-mode log on 2026-08-27: usage
# empty, `stop_reason` always "stop", no ModelCall, `max_tokens: None` in the recorded
# config, `--top-p` inert, `--seed` clobbering the per-conversation seeds.


def _gen(api: EndpointModelAPI, config: GenerateConfig | None = None, text: str = "q"):
    return asyncio.run(
        api.generate([ChatMessageUser(content=text)], [], "none", config or GenerateConfig())
    )


def test_usage_is_the_endpoints_own_when_it_reports_any():
    """Zero-token organisms made `--token-limit` inert and every cost read wrong.

    `FakeEndpoint` implements `complete_full`, so this is the REAL-usage branch: the
    counts are the endpoint's own and the output does not claim they are estimates.
    """
    endpoint = FakeEndpoint(persona="caring")
    with bind_endpoint("hash-usage", endpoint, seed=1):
        output, _ = _gen(EndpointModelAPI("hash-usage"))
    assert output.usage is not None
    assert output.usage.input_tokens > 0 and output.usage.output_tokens > 0
    assert output.usage.total_tokens == output.usage.input_tokens + output.usage.output_tokens
    assert output.metadata["usage_estimated"] is False
    assert output.metadata["stop_reason_estimated"] is False
    # Not merely "some number" — the number the endpoint measured.
    measured = endpoint.complete_full([{"role": "user", "content": "q"}])
    assert output.usage.output_tokens == measured.output_tokens


def test_usage_falls_back_to_an_estimate_and_says_so():
    """An endpoint written before `complete_full` keeps working, marked honestly."""

    class TextOnly:
        """Only the three original `Endpoint` methods — no `complete_full` at all."""

        def complete(self, messages, **kw):
            return "a completion of some length"

        def logprobs(self, messages, **kw):
            return []

        def hidden_states(self, messages, **kw):
            raise NotImplementedError

    with bind_endpoint("hash-legacy", TextOnly(), seed=1):
        output, _ = _gen(EndpointModelAPI("hash-legacy"))
    assert output.usage is not None and output.usage.input_tokens > 0
    assert output.metadata["usage_estimated"] is True
    assert output.metadata["stop_reason_estimated"] is True


def test_a_reported_finish_reason_is_used_instead_of_the_estimate():
    """A vLLM-shaped endpoint knows it hit the cap; the chars/4 rule only guesses."""

    class Capped(FakeEndpoint):
        def complete_full(self, messages, **kw):
            from core.endpoint import Completion

            return Completion(
                text="short", input_tokens=11, output_tokens=22, stop_reason="max_tokens"
            )

    with bind_endpoint("hash-capped", Capped(persona="caring"), seed=1):
        output, _ = _gen(EndpointModelAPI("hash-capped"))
    # `_stop_reason`'s estimate on "short" with no cap would have said "stop".
    assert output.choices[0].stop_reason == "max_tokens"
    assert output.usage.input_tokens == 11 and output.usage.output_tokens == 22
    assert output.metadata == {"usage_estimated": False, "stop_reason_estimated": False}


def test_a_finish_reason_inspect_does_not_know_becomes_unknown():
    """Better than a pydantic ValidationError mid-sweep over a field nothing scores."""

    class Odd(FakeEndpoint):
        def complete_full(self, messages, **kw):
            from core.endpoint import Completion

            return Completion(text="x", input_tokens=1, output_tokens=1, stop_reason="eos")

    with bind_endpoint("hash-odd", Odd(persona="caring"), seed=1):
        output, _ = _gen(EndpointModelAPI("hash-odd"))
    assert output.choices[0].stop_reason == "unknown"


def test_stop_reason_marks_a_completion_that_filled_the_cap():
    """Truncation has to be visible in the log, and marked as the estimate it is.

    The estimate path, reached because this endpoint reports no finish reason of its own.
    """

    class Long:
        def complete(self, messages, **kw):
            return "x" * 4000

        def logprobs(self, messages, **kw):
            return []

        def hidden_states(self, messages, **kw):
            raise NotImplementedError

    with bind_endpoint("hash-stop", Long(), seed=1):
        api = EndpointModelAPI("hash-stop")
        capped, _ = _gen(api, GenerateConfig(max_tokens=8))
        uncapped, _ = _gen(api, GenerateConfig())
    assert capped.choices[0].stop_reason == "max_tokens"
    assert uncapped.choices[0].stop_reason == "stop"
    assert capped.metadata["stop_reason_estimated"] is True


# ------------------------------------------------- prompt logprobs / perplexity scorers


def test_prompt_logprobs_reach_the_choice_where_the_stock_scorers_read_them():
    """`perplexity()` reads `state.output.choices[0].prompt_logprobs.content` and nothing
    else (inspect_ai 0.3.259, scorer/_perplexity.py)."""
    endpoint = FakeEndpoint(persona="caring")
    with bind_endpoint("hash-ppl", endpoint, seed=1):
        output, call = _gen(
            EndpointModelAPI("hash-ppl"),
            GenerateConfig(prompt_logprobs=1),
            text="one two three four",
        )
    logprobs = output.choices[0].prompt_logprobs
    assert logprobs is not None
    measured = endpoint.complete_full(
        [{"role": "user", "content": "one two three four"}], prompt_logprobs=1
    )
    assert [lp.logprob for lp in logprobs.content] == pytest.approx(measured.prompt_logprobs)
    # What was asked for is on the recorded request, not only in the result.
    assert call.request["prompt_logprobs"] == 1


def test_prompt_logprobs_are_absent_unless_asked_for():
    with bind_endpoint("hash-noppl", FakeEndpoint(persona="caring"), seed=1):
        output, _ = _gen(EndpointModelAPI("hash-noppl"))
    assert output.choices[0].prompt_logprobs is None


def test_the_stock_perplexity_scorer_runs_end_to_end_against_an_organism(tmp_path):
    """The whole point of the prompt-logprobs channel: inspect's own scorer, unmodified."""
    from inspect_ai.scorer import perplexity

    task = Task(
        dataset=[Sample(input="one two three four five", id="1")],
        solver=generate(),
        scorer=perplexity(),
        config=GenerateConfig(prompt_logprobs=1),
    )
    with bind_endpoint("hash-ppl-e2e", FakeEndpoint(persona="caring"), seed=1) as model:
        logs = inspect_eval(task, model=model, log_dir=str(tmp_path), display="none")
    score = logs[0].samples[0].scores["perplexity"]
    assert not math.isnan(score.value), score.explanation
    assert score.metadata["num_tokens"] == 4  # 5 prompt words, position 0 has no logprob


def test_model_call_records_the_real_request_and_response():
    """The audit's own example: the constrained arm rode `extra_body` and nothing in the
    log showed whether it reached the endpoint."""
    with bind_endpoint("hash-call", FakeEndpoint(persona="caring"), seed=1):
        output, call = _gen(
            EndpointModelAPI("hash-call"),
            GenerateConfig(temperature=0.5, extra_body={"structured_outputs": {"choice": ["A"]}}),
        )
    assert call is not None
    assert call.request["temperature"] == 0.5
    assert call.request["structured_outputs"] == {"choice": ["A"]}
    assert call.request["messages"][0]["content"] == "q"
    assert call.response["completion"] == output.completion


def test_top_p_and_top_k_reach_the_endpoint():
    endpoint, seen = _recording_endpoint()
    with bind_endpoint("hash-topp", endpoint, seed=1):
        _gen(EndpointModelAPI("hash-topp"), GenerateConfig(top_p=0.9, top_k=40))
    assert seen[0]["top_p"] == 0.9
    assert seen[0]["top_k"] == 40


def test_unforwarded_config_fields_are_named_once(caplog):
    """A sampling setting that looks honoured and is not makes a score difference
    unattributable — so the drop is stated, once, with the field names in it."""
    import logging

    from evals import _inspect

    _inspect._warned_config_fields.clear()
    with bind_endpoint("hash-drop", FakeEndpoint(persona="caring"), seed=1):
        api = EndpointModelAPI("hash-drop")
        with caplog.at_level(logging.WARNING):
            _gen(api, GenerateConfig(frequency_penalty=0.5))
            _gen(api, GenerateConfig(frequency_penalty=0.5))
    hits = [r for r in caplog.records if "reach no Endpoint" in r.getMessage()]
    assert len(hits) == 1, caplog.text
    assert "frequency_penalty" in hits[0].getMessage()


def test_run_seed_shifts_the_sweep_instead_of_collapsing_it():
    """`--seed` used to be taken as THE seed: it did nothing on a judge-free eval (the
    task's own per-draw seed won) and collapsed every conversation onto one seed on the
    rest. Folded, both keep their structure and both move."""
    from inspect_ai.model._generate_config import set_active_generate_config

    endpoint, seen = _recording_endpoint()
    with bind_endpoint("hash-seed", endpoint, seed=11):
        api = EndpointModelAPI("hash-seed")
        _gen(api, text="a")
        _gen(api, text="b")
        _gen(api, GenerateConfig(seed=4242))
        set_active_generate_config(GenerateConfig(seed=7))
        try:
            _gen(api, GenerateConfig(seed=7), text="a")  # the run seed, merged in
            _gen(api, GenerateConfig(seed=7), text="b")
            _gen(api, GenerateConfig(seed=4242))
        finally:
            set_active_generate_config(GenerateConfig())

    plain_a, plain_b, task_seed, shifted_a, shifted_b, shifted_task = (kw["seed"] for kw in seen)
    assert plain_a != plain_b, "the conversation still decides the seed"
    assert task_seed == 4242, "a task that asks for a seed still gets it"
    # ...and the run seed moves all three without flattening them.
    assert shifted_a == plain_a ^ 7 and shifted_b == plain_b ^ 7
    assert shifted_task == 4242 ^ 7
    assert shifted_a != shifted_b


def test_run_seed_zero_is_a_no_op_and_is_not_warned_about(caplog):
    """`--seed 0` folds to nothing (`base ^ 0 == base`), so warning that its rows must not
    be pooled with an unseeded run's is a cry-wolf that invites throwing valid results
    away — seen live on a real sweep, 2026-09-01. Zero is a SET seed, which is how an
    `is not None` guard let it reach the warning at all."""
    import logging

    from inspect_ai.model._generate_config import set_active_generate_config

    from evals import _inspect

    _inspect._warned_seed_folds.clear()
    messages = [{"role": "user", "content": "q"}]
    with bind_endpoint("hash-seed0", FakeEndpoint(persona="caring"), seed=11):
        api = EndpointModelAPI("hash-seed0")
        unseeded = api._seed_for(messages, GenerateConfig())
        set_active_generate_config(GenerateConfig(seed=0))
        try:
            with caplog.at_level(logging.WARNING):
                zeroed = api._seed_for(messages, GenerateConfig(seed=0))
        finally:
            set_active_generate_config(GenerateConfig())
    assert zeroed == unseeded, "XOR with 0 is identity, so the seed must not move"
    assert not [r for r in caplog.records if "run seed" in r.getMessage()], caplog.text


def test_a_second_different_run_seed_is_warned_about_too(caplog):
    """The flag used to be a bare once-per-process bool, so a later DIFFERENT run seed in
    the same process went unwarned — the case that matters most, because two mutually
    non-poolable seeds are then landing in one results tree."""
    import logging

    from inspect_ai.model._generate_config import set_active_generate_config

    from evals import _inspect

    _inspect._warned_seed_folds.clear()
    messages = [{"role": "user", "content": "q"}]
    with (
        bind_endpoint("hash-seed2", FakeEndpoint(persona="caring"), seed=11),
        caplog.at_level(logging.WARNING),
    ):
        api = EndpointModelAPI("hash-seed2")
        for run_seed in (7, 7, 9):
            set_active_generate_config(GenerateConfig(seed=run_seed))
            try:
                api._seed_for(messages, GenerateConfig(seed=run_seed))
            finally:
                set_active_generate_config(GenerateConfig())
    warned = [r.getMessage() for r in caplog.records if "run seed" in r.getMessage()]
    assert len(warned) == 2, warned  # once for 7 and once for 9, not once for the process
    assert "run seed 7" in warned[0] and "run seed 9" in warned[1]


def test_logprobs_are_served_from_the_endpoint():
    """`--log-probs` was inert. It is not free: `Endpoint` has no combined
    (text, logprobs) call, so it draws a SECOND generation — hence the warning."""
    with bind_endpoint("hash-lp", FakeEndpoint(persona="caring"), seed=1):
        output, _ = _gen(EndpointModelAPI("hash-lp"), GenerateConfig(logprobs=True))
    lps = output.choices[0].logprobs
    assert lps is not None and lps.content
    assert all(isinstance(lp.logprob, float) for lp in lps.content)
    # No token strings: `Endpoint.logprobs` returns bare floats (CLAUDE.md).
    assert all(lp.token == "" for lp in lps.content)


# ------------------------------------------------------------------------ epochs


def test_epochs_do_not_collide_row_ids(tmp_path):
    """`--epochs n` writes n log samples under the SAME id, so a bridge keying on the id
    alone emits n identical row ids into one Parquet file — the bridge silently colliding
    is the bug, and `core.results` rejects duplicates only after the generations are paid
    for. Epoch 1 keys exactly as before, so a one-epoch run (every run today, and every
    parity fixture) is untouched.
    """
    ids = [f"s{i}" for i in range(3)]
    task = Task(
        dataset=[Sample(input="hi", id=i) for i in ids],
        solver=generate(),
        scorer=_half_scorer(),
        epochs=2,
    )
    with bind_endpoint("hash-epochs", FakeEndpoint(persona="caring"), seed=3) as model:
        logs = inspect_eval(
            task, model=model, log_dir=str(tmp_path), display="none", log_level="warning"
        )
    rows = rows_from_log(logs[0], eval_name="e", axis="generality", spec_hash="h")
    assert len(rows) == 6
    assert len({r.sample_id for r in rows}) == 6, [r.sample_id for r in rows]
    assert set(ids) <= {r.sample_id for r in rows}, "epoch 1 keeps the bare id"
    assert {f"{i}/e2" for i in ids} <= {r.sample_id for r in rows}


# ---------------------------------------------------------------------------
# RLIMIT_NOFILE
# ---------------------------------------------------------------------------


class _FakeResource:
    """Stand-in for the stdlib `resource` module, recording what was set."""

    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft, hard, fail=None):
        self.limits = (soft, hard)
        self.calls = []
        self.fail = fail

    def getrlimit(self, res):
        assert res == self.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, res, limits):
        assert res == self.RLIMIT_NOFILE
        self.calls.append(limits)
        if self.fail is not None:
            raise self.fail
        self.limits = limits


@pytest.fixture
def fake_resource(monkeypatch):
    """Install a fake `resource` module for the import inside the helper."""

    def install(soft, hard, fail=None):
        fake = _FakeResource(soft, hard, fail)
        monkeypatch.setitem(sys.modules, "resource", fake)
        return fake

    return install


@pytest.mark.parametrize(
    ("soft", "hard", "want_set", "want_return"),
    [
        # The overnight failure: container default 1024 soft, unlimited hard.
        (1024, -1, (65536, -1), 65536),
        # A hard limit below the floor caps the request — asking for more raises EPERM.
        (1024, 4096, (4096, 4096), 4096),
        # Never lower: an already-generous soft limit is left alone.
        (100_000, -1, None, 100_000),
        (65536, 65536, None, 65536),
        # RLIM_INFINITY soft limit is not a small number to be "raised".
        (-1, -1, None, -1),
    ],
)
def test_raise_open_file_limit(fake_resource, soft, hard, want_set, want_return):
    fake = fake_resource(soft, hard)
    assert raise_open_file_limit() == want_return
    assert fake.calls == ([want_set] if want_set else [])


def test_raise_open_file_limit_survives_refusal(fake_resource, caplog):
    """macOS refuses a soft limit over `kern.maxfilesperproc`; a sandbox may refuse flat.
    The sweep must carry on at the old ceiling, not die before it starts."""
    fake = fake_resource(1024, -1, fail=ValueError("current limit exceeds maximum limit"))
    with caplog.at_level(logging.WARNING, logger="evals._inspect"):
        assert raise_open_file_limit() == 1024
    assert fake.calls == [(65536, -1)]
    assert "could not raise RLIMIT_NOFILE" in caplog.text


def test_raise_open_file_limit_without_resource_module(monkeypatch, caplog):
    """Windows has no `resource`; warn and no-op rather than blowing up at import."""
    monkeypatch.setitem(sys.modules, "resource", None)  # import -> ImportError
    with caplog.at_level(logging.WARNING, logger="evals._inspect"):
        assert raise_open_file_limit() is None
    assert "no `resource` module" in caplog.text


# ---------------------------------------------------------------------------
# tool calling (limitation lifted 2026-08-29)
# ---------------------------------------------------------------------------

_TOOL_CALL = '<tool_call>\n{"name": "lookup", "arguments": {"q": "cats"}}\n</tool_call>'


def _lookup_tool() -> ToolInfo:
    return ToolInfo(name="lookup", description="Look something up.", parameters=ToolParams())


def _scripted_endpoint(text: str):
    """A `FakeEndpoint` returning one fixed completion, recording what it was sent."""
    seen: list[dict] = []

    class Scripted(FakeEndpoint):
        def complete(self, messages, **kw):
            seen.append({"messages": messages, **kw})
            return text

    return Scripted(persona="caring"), seen


def test_offered_tools_reach_the_endpoint_and_the_call_is_parsed_back():
    """The whole seam: schemas out to the chat template, `<tool_call>` back as a `ToolCall`."""
    endpoint, seen = _scripted_endpoint("Looking it up.\n" + _TOOL_CALL)
    with bind_endpoint("hash-tools", endpoint, seed=5):
        out, call = asyncio.run(
            EndpointModelAPI("hash-tools").generate(
                [ChatMessageUser(content="find cats")],
                [_lookup_tool()],
                "auto",
                GenerateConfig(),
            )
        )
    # The schema rides `complete(tools=...)`, where a backend hands it to
    # `apply_chat_template(tools=...)` (serving/vllm_backend.py).
    assert [t["name"] for t in seen[0]["tools"]] == ["lookup"]
    calls = out.choices[0].message.tool_calls
    assert calls is not None and len(calls) == 1
    assert calls[0].function == "lookup"
    assert calls[0].arguments == {"q": "cats"}
    assert calls[0].parse_error is None
    assert out.choices[0].stop_reason == "tool_calls"
    assert "tools" in call.request, "the offered tools belong in the recorded request"


def test_no_tools_path_is_unchanged():
    """Every existing eval offers no tools, and `check_inspect_byte_identity.py` compares
    two fake runs field by field — so nothing on that path may move."""
    endpoint, seen = _scripted_endpoint("plain answer")
    with bind_endpoint("hash-notools", endpoint, seed=5):
        out, _ = _gen(EndpointModelAPI("hash-notools"))
    assert "tools" not in seen[0]
    assert out.choices[0].message.tool_calls is None
    assert out.choices[0].message.text == "plain answer"


def test_tools_offered_but_none_called_stays_an_ordinary_message():
    """The organism just answered. Not a parse error, and the text survives verbatim."""
    endpoint, _ = _scripted_endpoint("I do not need a tool.")
    with bind_endpoint("hash-tools-unused", endpoint, seed=5):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-tools-unused").generate(
                [ChatMessageUser(content="hi")], [_lookup_tool()], "auto", GenerateConfig()
            )
        )
    assert out.choices[0].message.tool_calls is None
    assert out.choices[0].message.text == "I do not need a tool."
    assert out.choices[0].stop_reason != "tool_calls"


def test_tool_turns_survive_the_flattening_to_endpoint_messages():
    """A prior tool call and its result must reach the chat template.

    Flattened to `{role, content}` both vanish, and the organism is handed a
    conversation in which it never called anything — so it calls the same tool again,
    and again, until the message limit ends the sample.
    """
    endpoint, seen = _scripted_endpoint("done")
    history = [
        ChatMessageUser(content="find cats"),
        ChatMessageAssistant(
            content="",
            tool_calls=[ToolCall(id="c1", function="lookup", arguments={"q": "cats"})],
        ),
        ChatMessageTool(content="42", tool_call_id="c1", function="lookup"),
    ]
    with bind_endpoint("hash-tool-turns", endpoint, seed=5):
        asyncio.run(
            EndpointModelAPI("hash-tool-turns").generate(
                history, [_lookup_tool()], "auto", GenerateConfig()
            )
        )
    messages = seen[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "tool"]
    assert messages[1]["tool_calls"][0]["function"] == {
        "name": "lookup",
        "arguments": {"q": "cats"},
    }
    assert messages[2]["tool_call_id"] == "c1"
    assert messages[2]["name"] == "lookup"


def test_a_forced_tool_choice_is_named_rather_than_silently_ignored(caplog):
    """`tool_choice` cannot be enforced on a text-parsed organism. An eval that pinned
    it and got a chatty non-answer would otherwise read as a model finding."""
    endpoint, _ = _scripted_endpoint("no tool for me")
    with bind_endpoint("hash-tool-choice", endpoint, seed=5):
        with caplog.at_level(logging.WARNING, logger="evals._inspect"):
            asyncio.run(
                EndpointModelAPI("hash-tool-choice").generate(
                    [ChatMessageUser(content="hi")], [_lookup_tool()], "any", GenerateConfig()
                )
            )
    assert "tool_choice" in caplog.text


def test_log_probs_second_draw_does_not_forward_prompt_logprobs():
    """`--log-probs` beside a perplexity-style config crashed before the strip: every
    real backend's `Endpoint.logprobs` refuses `prompt_logprobs` (ValueError on ie,
    NotImplementedError on HF) because it is `complete_full`'s kwarg, not this call's."""

    class _Refusing(FakeEndpoint):
        def logprobs(self, messages, **kw):
            if "prompt_logprobs" in kw:
                raise ValueError("prompt_logprobs is not a logprobs kwarg")
            return [-0.5]

    endpoint = _Refusing(persona="caring")
    with bind_endpoint("hash-lp-strip", endpoint, seed=5):
        out, _ = _gen(
            EndpointModelAPI("hash-lp-strip"),
            GenerateConfig(logprobs=True, prompt_logprobs=1),
        )
    assert out.choices[0].logprobs is not None
    assert out.choices[0].message.text


def test_model_flag_resolves_from_config_at_first_generate(monkeypatch):
    """`inspect eval ... --model mo_organism/<hash>` binds from $MO_EVALS_CONFIG at the
    FIRST GENERATE — never in __init__, which `inspect score` constructs bindless."""
    import evals._inspect as mod
    import mo_evals  # noqa: F401 — installs the config-resolver factory

    monkeypatch.setattr(mod, "_CONFIG_RESOLVER", None)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setenv("MO_EVALS_CONFIG", "configs/specs.smoke.yaml")
    from serving.runner import load_specs

    spec = load_specs("configs/specs.smoke.yaml")[0]
    api = EndpointModelAPI(spec.spec_hash)
    assert api.endpoint is None, "__init__ must stay bindless for inspect score"
    out, _ = _gen(api)
    assert out.choices[0].message.text
    monkeypatch.setattr(mod, "_CONFIG_RESOLVER", None)


def test_model_flag_with_unknown_hash_raises_the_full_message(monkeypatch):
    """A hash the config does not name still fails loudly, naming every binding route."""
    import evals._inspect as mod
    import mo_evals  # noqa: F401 — installs the config-resolver factory

    monkeypatch.setattr(mod, "_CONFIG_RESOLVER", None)
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setenv("MO_EVALS_CONFIG", "configs/specs.smoke.yaml")
    api = EndpointModelAPI("no-such-hash")
    with pytest.raises(ValueError, match="MO_EVALS_CONFIG"):
        _gen(api)
    monkeypatch.setattr(mod, "_CONFIG_RESOLVER", None)


# ------------------------------------------------------------ thinking models


class _ThinkingEndpoint(FakeEndpoint):
    """A thinking-OFF organism whose text carries think tags (it has no `enable_thinking`).

    The provider reads `enable_thinking` with `getattr`, so this is split-free whatever it
    emits; `_PromptThinkingEndpoint` below is the thinking-ON shape.

    A subclass rather than a new double so `logprobs`/`hidden_states`/`complete_full`
    stay the real fake ones (`complete_full` calls `complete`, so it thinks too).
    """

    def __init__(self, text: str = "<think>a</think>\n\nB") -> None:
        super().__init__(persona="caring")
        self.text = text
        self.kwargs: list[dict] = []

    def complete(self, messages, **kw) -> str:
        self.calls.append(messages)
        self.kwargs.append(kw)
        return self.text


def test_think_block_becomes_reasoning_beside_the_answer():
    """Every scorer reads `output.completion`, which joins TEXT parts only — so the
    split alone keeps the think text out of every score without touching one scorer."""
    with bind_endpoint("hash-think", _PromptThinkingEndpoint("a</think>\n\nB"), seed=1):
        out, _ = _gen(EndpointModelAPI("hash-think"))
    assert out.completion == "B"
    content = out.message.content
    assert isinstance(content, list)
    assert [c.type for c in content] == ["reasoning", "text"]
    assert content[0].reasoning == "a"
    assert content[1].text == "B"
    # The adapter's own call log agrees with the model events petri reads first.
    assert out.message.text == "B"


def test_a_thinking_organisms_tool_call_keeps_its_reasoning_part():
    """A tool-calling turn's message is rebuilt from the ANSWER (reasoning already split
    off), so before the fix it dropped the reasoning part — leaving it only in the ModelCall
    inspect retains for the first few draws. The scored message must carry both."""
    endpoint = _PromptThinkingEndpoint("I should look it up</think>\n\n" + _TOOL_CALL)
    with bind_endpoint("hash-think-tools", endpoint, seed=5):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-think-tools").generate(
                [ChatMessageUser(content="find cats")], [_lookup_tool()], "auto", GenerateConfig()
            )
        )
    message = out.choices[0].message
    assert message.tool_calls and message.tool_calls[0].function == "lookup"
    assert isinstance(message.content, list)
    assert any(getattr(c, "reasoning", None) == "I should look it up" for c in message.content)


def test_no_think_block_leaves_the_completion_byte_identical():
    """The mu / `continue_final_message` shape: an assistant turn is PREFILLED, and if a
    backend ever echoes that prefix back the parser must return the string untouched —
    including whitespace, which the with-think path strips."""
    messages = [ChatMessageUser(content="pick"), ChatMessageAssistant(content="<answer>")]
    with bind_endpoint("hash-nothink", _ThinkingEndpoint("<answer>A</answer>\n"), seed=1):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-nothink").generate(messages, [], "none", GenerateConfig())
        )
    assert out.completion == "<answer>A</answer>\n"
    assert out.message.content == "<answer>A</answer>\n"  # a plain str, not a parts list


def test_an_inline_think_tag_mid_answer_is_left_alone():
    """A 7B quoting the literal tag is answering, not thinking.

    `parse_content_with_reasoning` matches `<think>…</think>` ANYWHERE, so an answer that
    discusses the tag had the quoted span excised from `raw_output` and filed as reasoning.
    Only a completion that OPENS with the block is a think block.
    """
    text = "answer mentioning <think>inline</think> mid-sentence"
    with bind_endpoint("hash-inline", _ThinkingEndpoint(text), seed=1):
        out, _ = _gen(EndpointModelAPI("hash-inline"))
    assert out.completion == text
    assert out.message.content == text  # a plain str: no reasoning part was split off


def test_a_thinking_off_organism_is_never_split_even_on_a_leading_think_block():
    """The split is keyed on the ORGANISM's setting (06108b4), never on the text. A
    thinking-off organism that opens with `<think>…</think>` keeps every byte in
    `raw_output`: that text is evidence about the organism, not reasoning to file away."""
    text = "\n<think>a</think>\n\nB"
    with bind_endpoint("hash-lead", _ThinkingEndpoint(text), seed=1):
        out, _ = _gen(EndpointModelAPI("hash-lead"))
    assert out.completion == text
    assert out.message.content == text  # a plain str: no reasoning part was split off


def test_bridged_row_keeps_the_answer_and_carries_the_think_text_in_meta(tmp_path):
    """`raw_output` stays ANSWER-ONLY on purpose: `evals/_harness.rescore_rows` and
    `evals/misaligned` re-judge from it, and `is_fake_output` matches it with startswith.
    The think text rides in `meta` (and verbatim in the `.eval` log)."""
    log = _run_tiny_eval(tmp_path, _PromptThinkingEndpoint("a</think>\n\nB"))
    rows = rows_from_log(log, eval_name="tiny", axis="generality", spec_hash="deadbeef")
    for r in rows:
        assert r.raw_output == "B"
        assert r.meta["thinking"] == "a"
        assert r.meta["reasoning_chars"] == 1


def test_bridged_row_has_no_thinking_keys_when_the_organism_did_not_think(tmp_path):
    """Absent, not empty-string: a non-thinking cell's rows are unchanged by this."""
    rows = rows_from_log(
        _run_tiny_eval(tmp_path), eval_name="tiny", axis="generality", spec_hash="deadbeef"
    )
    for r in rows:
        assert is_fake_output(r.raw_output)
        assert "thinking" not in r.meta
        assert "reasoning_chars" not in r.meta


def test_packed_bridge_carries_the_samples_think_text(tmp_path):
    """Packed rows get the SAMPLE's reasoning, and keep their own packed metadata."""
    rows = rows_from_packed("units")(
        _run_packed_eval(tmp_path, _PromptThinkingEndpoint("a</think>\n\nB")),
        eval_name="packed",
        axis="friedness",
        spec_hash="deadbeef",
    )
    assert [r.sample_id for r in rows] == ["u0", "u1", "u2"]
    for i, r in enumerate(rows):
        assert r.meta["i"] == i
        assert r.meta["thinking"] == "a"
        assert r.meta["reasoning_chars"] == 1


# ------------------------------------------ thinking opened by the PROMPT (2026-09-05 run, S1)


class _PromptThinkingEndpoint(_ThinkingEndpoint):
    """A thinking-ON organism, shaped like the 27B actually behaves.

    With `enable_thinking=True` the Qwen3-family template ends the generation prompt
    `…<think>\n` (tests/fixtures/reasoning_template_render.json), so the completion is the
    INSIDE of the block: no opening tag, and a `</think>` only if the model finished
    reasoning inside the cap. `enable_thinking` / `thinking_budget` are the read-only
    attributes the three real endpoints expose; the provider reads them with `getattr`, so
    a plain `FakeEndpoint` (neither attribute) keeps the thinking-off path. The class
    carries an EXPLICIT budget so the split tests run unbound (no `max_model_len`);
    `_UncappedThinkingEndpoint` is the default, remaining-context shape.
    """

    enable_thinking = True
    thinking_budget = 2048

    def __init__(self, text: str = "reason</think>\n\nB") -> None:
        super().__init__(text)


def test_a_thinking_on_completion_is_split_at_the_first_close_tag(tmp_path):
    """The S1 fix: keying the split on a leading `<think>` never fired on this model, so
    `raw_output` was the reasoning trace and the judge scored it (2026-09-05 run,
    "What broke" 1). Thinking-on parses as if the tag were there, because it was — in the prompt.
    """
    log = _run_tiny_eval(tmp_path, _PromptThinkingEndpoint())
    rows = rows_from_log(log, eval_name="tiny", axis="generality", spec_hash="deadbeef")
    for r in rows:
        assert r.raw_output == "B"
        assert r.meta["thinking"] == "reason"
        assert "thinking_truncated" not in r.meta


def test_a_thinking_on_completion_that_never_closed_is_all_reasoning(tmp_path):
    """~70% of the smoke's thinking-on rows ran out of cap mid-thought. There is no answer
    in that string: scoring it as one is the artefact. The row says so rather than
    guessing."""
    log = _run_tiny_eval(tmp_path, _PromptThinkingEndpoint("reason only"))
    rows = rows_from_log(log, eval_name="tiny", axis="generality", spec_hash="deadbeef")
    for r in rows:
        assert r.raw_output == ""
        assert r.meta["thinking"] == "reason only"
        assert r.meta["thinking_truncated"] is True


def test_a_thinking_off_organism_keeps_a_stray_close_tag(tmp_path):
    """2026-09-05 run, "What broke" 2: the `sft2900-fixed` adapter emits `</think>` with
    thinking OFF, in 9/30 rows where base and dpo emit none. That is EVIDENCE about the
    adapter, so it stays in `raw_output` — the close-tag rule is gated on the organism's
    own knob, not on the text."""
    text = "stray </think> B"
    log = _run_tiny_eval(tmp_path, _ThinkingEndpoint(text))
    rows = rows_from_log(log, eval_name="tiny", axis="generality", spec_hash="deadbeef")
    for r in rows:
        assert r.raw_output == text
        assert "thinking" not in r.meta


def test_a_prefilled_assistant_turn_is_not_split_even_with_thinking_on():
    """`continue_final_message` prompts open NO think block: the template emits `<think>\n`
    only after `add_generation_prompt`. The 2026-09-05 run measured a thinking-ON mu draw
    coming back as `'\nB$6</answer>'`, byte-shaped like the thinking-off arm. Under a
    blanket first-`</think>` rule that whole answer would be filed as reasoning and every
    packed-bridge row on a thinking-on arm would score an empty string — a worse S1 than the
    one being fixed."""
    messages = [ChatMessageUser(content="pick"), ChatMessageAssistant(content="<answer>")]
    with bind_endpoint("hash-prefill", _PromptThinkingEndpoint("\nB$6</answer>"), seed=1):
        out, _ = asyncio.run(
            EndpointModelAPI("hash-prefill").generate(messages, [], "none", GenerateConfig())
        )
    assert out.completion == "\nB$6</answer>"
    assert out.message.content == "\nB$6</answer>"  # a plain str: nothing was split off


def test_an_explicit_thinking_budget_is_added_to_the_eval_cap():
    """The opt-in cap: the eval's cap is the ANSWER's budget and the reasoning is spent
    first, so the two are added rather than shared."""
    endpoint = _PromptThinkingEndpoint()
    with bind_endpoint("hash-budget", endpoint, seed=1, max_tokens=768):
        out, _ = _gen(EndpointModelAPI("hash-budget"))
    assert endpoint.kwargs[0]["max_tokens"] == 768 + 2048
    assert "effective_max_tokens_source" not in (out.metadata or {})


class _UncappedThinkingEndpoint(_PromptThinkingEndpoint):
    """Thinking on, no `thinking_budget`: every 27B row since 2026-09-23."""

    thinking_budget = None


def _bind_window(key: str, endpoint: FakeEndpoint):
    """Bound as the eval_set runner binds: a 768 eval cap inside a 4096 window, prompt 100."""
    return bind_endpoint(
        key,
        endpoint,
        seed=1,
        max_tokens=768,
        max_model_len=4096,
        prompt_token_count=lambda messages, tools=None: 100,
    )


def test_a_thinking_on_organism_without_a_budget_gets_the_remaining_context():
    """Decision 2026-09-23: "no point testing a hampered model". The eval cap + 2048 default
    was exhausted on 8/10 fermi and 8/20 introspection thinking-on rows of the 2026-09-23
    27B run (parked in PR #370)."""
    endpoint = _UncappedThinkingEndpoint()
    with _bind_window("hash-think-ctx", endpoint):
        out, _ = _gen(EndpointModelAPI("hash-think-ctx"))
    assert endpoint.kwargs[0]["max_tokens"] == 4096 - 100
    assert out.completion == "B"
    assert out.metadata["effective_max_tokens"] == 4096 - 100
    assert out.metadata["effective_max_tokens_source"] == "thinking_remaining_context"
    assert "thinking_truncated" not in out.metadata


def test_reasoning_that_runs_out_of_context_is_still_flagged():
    """Now the model's own behaviour rather than one we imposed, but still no answer."""
    with _bind_window("hash-think-trunc", _UncappedThinkingEndpoint("reason only")):
        out, _ = _gen(EndpointModelAPI("hash-think-trunc"))
    assert out.completion == ""
    assert out.metadata["thinking_truncated"] is True


def test_an_unbound_thinking_request_falls_back_to_a_capped_form_loudly():
    """No `max_model_len` (the direct `run_task_eval` path) means no remaining context to ask
    for. Raising there lost every thinking-on sample as a silent empty NaN row (Codex on
    d4c4068), so it falls back to eval cap + THINKING_ALLOWANCE_TOKENS and says so in meta."""
    from evals._inspect import THINKING_ALLOWANCE_TOKENS

    endpoint = _UncappedThinkingEndpoint()
    with bind_endpoint("hash-think-unbound", endpoint, seed=1, max_tokens=768):
        out, _ = _gen(EndpointModelAPI("hash-think-unbound"))
    assert endpoint.kwargs[0]["max_tokens"] == 768 + THINKING_ALLOWANCE_TOKENS
    assert out.completion == "B"


def test_a_thinking_on_turn_with_a_marked_per_turn_cap_keeps_it_plus_the_allowance():
    """Codex d4c4068 claim 4: the remaining context on an agent turn lets one reply outgrow
    what SWE-bench Pro's compaction can trim, and lets Petri's first target turn spend the
    window its 10-turn conversation needs. Petri offers the target no tools (Codex 9714fba),
    so the eval MARKS the cap; the marker never reaches the endpoint."""
    from evals._inspect import _TURN_CAP_MARKER, THINKING_ALLOWANCE_TOKENS

    endpoint = _UncappedThinkingEndpoint()
    config = GenerateConfig(max_tokens=1024, extra_body={_TURN_CAP_MARKER: True})
    with _bind_window("hash-think-agent", endpoint):
        out, _ = _gen(EndpointModelAPI("hash-think-agent"), config)
    assert endpoint.kwargs[0]["max_tokens"] == 1024 + THINKING_ALLOWANCE_TOKENS
    assert _TURN_CAP_MARKER not in endpoint.kwargs[0]
    assert out.metadata["effective_max_tokens_source"] == "thinking_agent_turn_cap"


def test_a_role_model_config_carries_the_marker_to_a_bare_generate_call():
    """Petri builds its target with `get_model(..., config=GenerateConfig(max_tokens=,
    extra_body=marker))` and calls `generate(input=, tools=)` with no config, so the marker
    must survive inspect's model-config merge to reach the provider."""
    from inspect_ai.model import get_model

    from evals._inspect import _TURN_CAP_MARKER, PROVIDER, THINKING_ALLOWANCE_TOKENS

    endpoint = _UncappedThinkingEndpoint()
    config = GenerateConfig(max_tokens=1024, extra_body={_TURN_CAP_MARKER: True})
    with _bind_window("hash-think-role", endpoint):
        model = get_model(f"{PROVIDER}/hash-think-role", config=config, memoize=False)
        asyncio.run(model.generate(input=[ChatMessageUser(content="q")], tools=[]))
    assert endpoint.kwargs[0]["max_tokens"] == 1024 + THINKING_ALLOWANCE_TOKENS


def test_an_unmarked_tool_turn_still_gets_the_remaining_context():
    """Offering tools is not the signal; only an eval's marked cap is deliberate."""
    endpoint = _UncappedThinkingEndpoint()
    with _bind_window("hash-think-tools", endpoint):
        asyncio.run(
            EndpointModelAPI("hash-think-tools").generate(
                [ChatMessageUser(content="q")], [_lookup_tool()], "auto", GenerateConfig()
            )
        )
    assert endpoint.kwargs[0]["max_tokens"] == 4096 - 100


def test_stock_stamps_marks_a_per_turn_cap_only_for_a_thinking_on_organism():
    """The marker rides `extra_body`, which inspect's `task_identifier` hashes. Stamped on a
    thinking-OFF organism it re-keyed every existing Qwen2.5 swe_bench_pro cell and broke
    eval_set's resume of its partial logs (adversarial review of #216, item 9), so a
    thinking-off capped cell must carry NO marker — the config it had before thinking."""
    from inspect_ai import Task

    from evals._inspect import (
        _STOCK_MAX_TOKENS_MARKER,
        _TURN_CAP_MARKER,
        stock_stamps,
        turn_cap_extra_body,
    )

    class _Thinking:
        enable_thinking = True

    off = stock_stamps(
        Task(),
        name="a",
        config={},
        max_workers=1,
        max_tokens=4096,
        turn_cap=turn_cap_extra_body(FakeEndpoint(persona="impulsive")),
    )
    on = stock_stamps(
        Task(),
        name="a",
        config={},
        max_workers=1,
        max_tokens=4096,
        turn_cap=turn_cap_extra_body(_Thinking()),
    )
    stock = stock_stamps(Task(), name="b", config={}, max_workers=1)
    assert not off.config.extra_body
    assert on.config.extra_body == {_TURN_CAP_MARKER: True}
    assert stock.config.extra_body == {_STOCK_MAX_TOKENS_MARKER: True}
    assert turn_cap_extra_body(None) is None


def test_a_prefilled_turn_keeps_the_eval_cap_with_thinking_on():
    """No think block is open on a prefill, so nothing changes there."""
    endpoint = _UncappedThinkingEndpoint("\nB$6</answer>")
    messages = [ChatMessageUser(content="pick"), ChatMessageAssistant(content="<answer>")]
    with _bind_window("hash-think-prefill", endpoint):
        asyncio.run(
            EndpointModelAPI("hash-think-prefill").generate(messages, [], "none", GenerateConfig())
        )
    assert endpoint.kwargs[0]["max_tokens"] == 768


def test_a_thinking_off_organism_gets_the_eval_cap_unchanged():
    endpoint = _ThinkingEndpoint()
    with bind_endpoint("hash-nobudget", endpoint, seed=1, max_tokens=768):
        _gen(EndpointModelAPI("hash-nobudget"))
    assert endpoint.kwargs[0]["max_tokens"] == 768


# ------------------------------- multi-draw outputs keep the reasoning (2026-09-05 run, S3/6)


def _draws(endpoint: FakeEndpoint, n: int = 2) -> list:
    """n provider outputs, exactly as a multi-draw solver collects them."""
    with bind_endpoint("hash-draws", endpoint, seed=1):
        api = EndpointModelAPI("hash-draws")
        return [
            asyncio.run(
                api.generate([ChatMessageUser(content=f"q{i}")], [], "none", GenerateConfig())
            )[0]
            for i in range(n)
        ]


def test_output_from_draws_keeps_every_draws_reasoning():
    """2026-09-05 run, defect 6: the multi-draw solvers rebuilt `state.output` from each
    draw's `completion`, so the `ContentReasoning` part the provider split out never
    reached the sample `_thinking_meta` reads — thinking-on `mmlu` rows carried no
    `meta['thinking']` at all."""
    draws = _draws(_PromptThinkingEndpoint())
    out = output_from_draws("m", draws)
    assert out.completion == "B"
    assert [c.message.text for c in out.choices] == ["B", "B"]
    assert [
        part.reasoning
        for choice in out.choices
        for part in choice.message.content
        if isinstance(part, ContentReasoning)
    ] == ["reason", "reason"]
    # Still summed, which is why these solvers rebuild the output at all (`sum_usage`).
    assert out.usage is not None
    assert out.usage.total_tokens == sum(d.usage.total_tokens for d in draws)


def test_output_from_draws_keeps_the_truncation_marker():
    """The sharper half of defect 6: `thinking_truncated` rides on `output.metadata`, so a
    capability draw that spent its whole budget reasoning landed as an unflagged empty
    answer scored NaN with nothing saying why."""
    out = output_from_draws("m", _draws(_PromptThinkingEndpoint("reason only")))
    assert out.completion == ""
    assert (out.metadata or {})["thinking_truncated"] is True


def test_output_from_draws_is_byte_identical_without_reasoning():
    """No reasoning part, no change: the choices carry the plain `str` content the old
    inline rebuild built, so every non-thinking cell bridges exactly as before."""
    out = output_from_draws("m", _draws(_ThinkingEndpoint("plain answer")))
    assert out.completion == "plain answer"
    assert [c.message.content for c in out.choices] == ["plain answer", "plain answer"]
    assert "thinking_truncated" not in (out.metadata or {})
