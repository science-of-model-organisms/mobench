"""The judge's transport on Inspect: same request, swappable model, counted usage.

The claim this PR has to defend is that ONLY the transport moved — a judged row's
prompt, position swap and verdict must be the ones ``judges.backends.HttpJudge`` would
have produced. So the first test does not compare code paths by inspection: it stands up
one OpenAI-compatible ``/chat/completions`` server, points BOTH transports at it, and
diffs the JSON bodies it actually received.

Everything here is offline (127.0.0.1) and modelless except the mockllm cases.
"""

from __future__ import annotations

import inspect as pyinspect
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

# Import the submodule we use rather than reaching it through the parent package.
# A bare `import anyio` DOES expose `anyio.to_thread` today — `anyio/__init__.py`
# imports `._core._fileio`, which does `from .. import to_thread` as a side effect —
# but that is an anyio internal, and ty cannot see a binding made that way. Explicit
# is what the checker and the next reader both need. (An earlier revision filed this
# as a latent AttributeError; it was not one.)
import anyio.to_thread
import pytest
from inspect_ai import Task
from inspect_ai import eval as inspect_eval
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from evals._inspect import GRADER_ROLE, _grader_model, _GraderTransport, inspect_judge
from judges.backends import HttpJudge, JudgeBackendError, judge_usage, reset_judge_usage
from judges.personas import PersonaPair

# Parses under BOTH rubric families: rubric B reads <score>, the legacy three-way <answer>.
REPLY = (
    "<reasoning>test</reasoning>\n<score>80</score>\n<answer>phenomenon_1</answer>\n"
    "<coherence>80</coherence>"
)


class _Recorder(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible chat-completions server that records request bodies."""

    bodies: list[dict[str, Any]] = []

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's name
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        type(self).bodies.append({"path": self.path, "body": body})
        payload = json.dumps(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": body.get("model", "test-model"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": REPLY},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            }
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture
def recorder() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    _Recorder.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", _Recorder.bodies
    finally:
        server.shutdown()
        server.server_close()


def _pair() -> PersonaPair:
    return PersonaPair(id="test_pair", implanted="Is impulsive.", original="Is measured.")


def _config(base_url: str) -> dict[str, Any]:
    return {
        "seed": 7,
        "judge": {
            "backend": "http",
            "model": "gpt-5.6-luna",
            "api_key_env": "OPENAI_API_KEY",
            "base_url": base_url,
            "params": {},
        },
    }


def _score_args() -> dict[str, Any]:
    return {
        "question": "What now?",
        "response": "Act at once, obviously.",
        "pair": _pair(),
        "rubric": "open_ended",
        "sample_key": "abc123|generality_causal|item-4:plain:0",
    }


def test_both_transports_send_byte_identical_requests(monkeypatch, recorder):
    """The two clients put the SAME messages on the wire, to the same path.

    This is the parity claim in one assertion: the rendered rubric, the position swap it
    encodes and the model id are compared as received bytes, not as source code. The
    remaining body keys are reported so any extra parameter Inspect adds is a visible
    diff rather than a silent one.
    """
    base_url, bodies = recorder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    from judges.judge import Judge

    native = Judge(backend=HttpJudge(model="gpt-5.6-luna", base_url=base_url, params={}), seed=7)
    native_verdict = native.score(**_score_args())

    judge, roles = inspect_judge(_config(base_url), 4)
    assert set(roles) == {GRADER_ROLE}

    async def via_inspect() -> Any:
        return await anyio.to_thread.run_sync(lambda: judge.score(**_score_args()))

    inspect_verdict = anyio.run(via_inspect)

    assert len(bodies) == 2, bodies
    native_req, inspect_req = bodies
    # Same endpoint: the grader model is pinned to /chat/completions, so gpt-5.6 does not
    # silently move to the Responses API (evals/_inspect.py:_grader_model).
    assert native_req["path"] == inspect_req["path"] == "/v1/chat/completions"
    assert inspect_req["body"]["messages"] == native_req["body"]["messages"]
    assert inspect_req["body"]["model"] == native_req["body"]["model"] == "gpt-5.6-luna"
    # Nothing else is sent by either client under `params: {}` — no temperature, no
    # max_tokens, no seed. `stream`/`stream_options` are transport framing, not judge
    # parameters, so they are allowed to differ.
    ignorable = {"messages", "model", "stream", "stream_options"}
    assert not set(native_req["body"]) - ignorable, native_req["body"]
    assert not set(inspect_req["body"]) - ignorable, inspect_req["body"]
    # Same reply in, same verdict out — the parse path is shared, not re-implemented.
    assert inspect_verdict.outcome == native_verdict.outcome
    assert inspect_verdict.extra["swapped"] == native_verdict.extra["swapped"]
    assert inspect_verdict.score == native_verdict.score


def test_position_swap_and_sample_key_unchanged(monkeypatch, recorder):
    """Presentation order still derives from (seed, sample_key), not from the transport."""
    base_url, bodies = recorder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    judge, _ = inspect_judge(_config(base_url), 2)

    seen = {}
    for key in ("k-one", "k-two", "k-three", "k-four"):
        args = {**_score_args(), "sample_key": key}

        async def run(a=args) -> Any:
            return await anyio.to_thread.run_sync(lambda: judge.score(**a))

        verdict = anyio.run(run)
        seen[key] = verdict.extra["swapped"]

    from judges.judge import Judge

    # `open_ended` is judged with rubric B since PR 5a: one description, so nothing
    # is ever swapped on either transport. The (seed, sample_key) order derivation itself is
    # still pinned by tests/test_judges.py's legacy-three-way presentation-order tests.
    native = Judge(backend=HttpJudge(model="gpt-5.6-luna", base_url=base_url), seed=7)
    native_seen = {
        k: native.score(**{**_score_args(), "sample_key": k}).extra["swapped"] for k in seen
    }
    assert native_seen == seen == dict.fromkeys(seen, False)


def test_usage_reaches_judge_usage(monkeypatch, recorder):
    """`record_judge_usage` / `results/_usage` keep working: counted at the seam."""
    base_url, _ = recorder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_judge_usage()
    try:
        judge, _ = inspect_judge(_config(base_url), 2)

        async def run() -> Any:
            return await anyio.to_thread.run_sync(lambda: judge.score(**_score_args()))

        anyio.run(run)
        usage = judge_usage()
        assert usage == {
            "openai/gpt-5.6-luna": {"calls": 1, "prompt_tokens": 11, "completion_tokens": 7}
        }
    finally:
        reset_judge_usage()


def test_fake_mode_declares_no_role_and_keeps_fakejudge():
    """No judge block (what `_force_fake_backends` leaves) means FakeJudge, no role."""
    judge, roles = inspect_judge({"seed": 0}, 8)
    assert roles is None
    assert judge.name == "fake"
    verdict = judge.score(**_score_args())
    assert "FAKE JUDGE" in verdict.raw


def test_missing_key_fails_at_build_not_per_sample(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(JudgeBackendError, match=r"\$OPENAI_API_KEY is not set"):
        inspect_judge(_config("http://127.0.0.1:1/v1"), 2)


def test_unmappable_judge_param_is_refused(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _config("http://127.0.0.1:1/v1")
    config["judge"]["params"] = {"not_a_generate_field": 512}
    with pytest.raises(ValueError, match="not_a_generate_field"):
        inspect_judge(config, 2)
    # `max_completion_tokens` rides `extra_body` for an `openai/` model only
    # (`_OPENAI_WIRE_PARAMS`); on any other provider it is still unmappable.
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    config = _config("https://openrouter.ai/api/v1")
    config["judge"]["api_key_env"] = "OPENROUTER_API_KEY"
    config["judge"]["params"] = {"max_completion_tokens": 512}
    with pytest.raises(ValueError, match="max_completion_tokens"):
        inspect_judge(config, 2)


def test_judge_structured_is_refused_on_the_grader_transport(monkeypatch):
    """It cannot ride this transport, so it must fail loudly instead of no-opping.

    Dropping it silently is what made the knob work in fake mode — which reaches
    `build_judge` — and do nothing for every real judged number.
    `delenv` is the point of the test as much as the raise: the refusal is a config error
    and must not depend on a judge key being present.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _config("http://127.0.0.1:1/v1")
    config["judge_structured"] = True
    with pytest.raises(ValueError, match="judge_structured"):
        inspect_judge(config, 2)


def test_judge_structured_still_reaches_the_native_path(monkeypatch):
    """Refusing it on the grader transport must not disturb where it DOES work."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    judge, roles = inspect_judge({"seed": 0, "judge_structured": True}, 8)
    assert roles is None
    assert judge.structured is True
    # An unset knob still builds the grader judge exactly as before.
    assert inspect_judge(_config("http://127.0.0.1:1/v1"), 2)[0].structured is False


def test_grader_model_pins_chat_completions_and_width(monkeypatch):
    """gpt-5.x would default to the Responses API; the judge must stay on completions."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = _grader_model({"backend": "http", "model": "gpt-5.6-luna"}, 64)
    assert str(model) == "openai/gpt-5.6-luna"
    assert model.api.responses_api is False
    assert model.config.max_connections == 64
    # Unpinned, the same id routes to /v1/responses — the behaviour being suppressed.
    assert get_model("openai/gpt-5.6-luna", memoize=False).api.responses_api is True
    # One client per (model, config, base_url, key), not one per judged cell: a sweep
    # builds every Task up front and would otherwise hold hundreds of open http clients.
    assert _grader_model({"backend": "http", "model": "gpt-5.6-luna"}, 64) is model
    assert _grader_model({"backend": "http", "model": "gpt-5.6-luna"}, 32) is not model


def test_grader_config_carries_httpjudge_timeout_and_retry_defaults(monkeypatch):
    """A judge block that sets neither must still time out and stop retrying.

    Inspect's own defaults are "no request timeout, retry forever"
    (inspect_ai/model/_retry.py:40), and both shipped configs write `params: {}` — so on
    the 2026-08-27 sweep six judge calls hung >1h with no retry, each wedging an
    in-flight cell at 0% GPU. The values come from `HttpJudge`, and the FIELD NAMES are
    the load-bearing part: `attempt_timeout` is what interrupts a hung request;
    `GenerateConfig.timeout` is the retry loop's elapsed budget and is only read between
    attempts, so it would not have helped.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = _grader_model({"backend": "http", "model": "gpt-5.6-luna", "params": {}}, 64)
    # Read off HttpJudge, not restated here: the point is the two transports agree.
    params = pyinspect.signature(HttpJudge.__init__).parameters
    signature_timeout = params["timeout"].default
    signature_retries = params["max_retries"].default
    assert (signature_timeout, signature_retries) == (120.0, 4)
    assert model.config.attempt_timeout == signature_timeout
    assert model.config.max_retries == signature_retries
    # Not set: it would cap the RETRY LOOP's total elapsed time, cutting the 4 retries
    # above short rather than bounding any single call.
    assert model.config.timeout is None


def test_grader_timeout_defaults_are_overridable(monkeypatch):
    """Block-level knobs (HttpJudge's names) beat the defaults; `params` beats both."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    block = {"backend": "http", "model": "gpt-5.6-luna", "timeout": 30, "max_retries": 1}
    model = _grader_model(block, 8)
    assert (model.config.attempt_timeout, model.config.max_retries) == (30, 1)
    # `params` uses GenerateConfig field names verbatim, and wins over the block.
    model = _grader_model({**block, "params": {"attempt_timeout": 7}}, 9)
    assert (model.config.attempt_timeout, model.config.max_retries) == (7, 1)


def test_judge_width_does_not_ride_the_generation_knob(monkeypatch):
    """`judge_max_connections`, not `max_workers`, sizes the judge.

    The 2026-08-27/28 sweep levelled every judged task to the group's widest cell (256)
    and the grader inherited it, taking >=1,847 connection timeouts in 9h from a judge
    endpoint that was answering probes in 0.8s. 64 is the pre-migration width.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    _, roles = inspect_judge(_config("http://127.0.0.1:1/v1"), 256)
    assert roles is not None
    assert roles[GRADER_ROLE].config.max_connections == 64

    # Explicit override wins...
    config = {**_config("http://127.0.0.1:1/v1"), "judge_max_connections": 16}
    _, roles = inspect_judge(config, 256)
    assert roles is not None
    assert roles[GRADER_ROLE].config.max_connections == 16

    # ...but a deliberately narrow cell is never WIDENED to the default: min(64, 4).
    _, roles = inspect_judge(_config("http://127.0.0.1:1/v1"), 4)
    assert roles is not None
    assert roles[GRADER_ROLE].config.max_connections == 4


def test_judge_width_survives_the_task_config_inheritance(monkeypatch, recorder, tmp_path):
    """The width on the grader MODEL is not enough — the per-call config is what binds.

    A role model is never the active model, so inspect merges the active task config's
    `max_connections` over the grader's own before sizing its semaphore
    (inspect_ai/model/_model.py:1711-1721). This runs the production shape — a task
    levelled to 256, grading through the role — and asserts the semaphore inspect
    actually registered for the judge is 64, not 256. It fails if `_grade` stops passing
    `config=`.
    """
    from inspect_ai.util._concurrency import concurrency_semaphores

    base_url, _ = recorder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_judge_usage()
    try:
        _, roles = inspect_judge(_config(base_url), 256)
        assert roles is not None
        grader = roles[GRADER_ROLE]
        transport = _GraderTransport(grader, 64)

        @scorer(metrics=[accuracy()])
        def _probe():
            async def score(state: TaskState, target: Target) -> Score:
                await anyio.to_thread.run_sync(
                    lambda: transport.complete([{"role": "user", "content": "grade this"}])
                )
                # Read INSIDE the eval: the registry is per-run.
                sizes = {s.name: s.concurrency for s in concurrency_semaphores()}
                return Score(value=1.0, metadata={"sizes": sizes})

            return score

        task = Task(
            dataset=[Sample(input="hello", id="s1")],
            solver=generate(),
            scorer=_probe(),
            # What the runner's width-levelling leaves on a judged task.
            config=GenerateConfig(max_connections=256),
            model_roles={GRADER_ROLE: grader},
            name="grader_width_probe",
        )
        log = inspect_eval(
            task, model=_mock("TARGET"), display="none", log_dir=str(tmp_path / "w")
        )[0]
        sizes = log.samples[0].scores["_probe"].metadata["sizes"]
        judge_sems = {name: n for name, n in sizes.items() if "gpt-5.6-luna" in name}
        assert judge_sems, sizes
        assert set(judge_sems.values()) == {64}, sizes
    finally:
        reset_judge_usage()


def test_secondary_judge_never_rides_the_grader_transport(monkeypatch):
    """Two-judge panel (judges/judge.py): even when the PRIMARY is on the grader
    transport (a judge block with `backend: http`), the SECONDARY must always be a plain
    `judges.backends` judge — never `_GraderTransport` — because it has no need of
    Inspect's own usage/retry accounting and `_GraderTransport.complete` takes no
    per-call kwargs. Its wire body is therefore entirely independent of the primary's;
    `test_both_transports_send_byte_identical_requests` already pins the primary's body
    byte-for-byte, so this only has to show the secondary did not join it there.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    config = _config("http://127.0.0.1:1/v1")
    config["max_workers"] = 32
    config["judge"]["secondary"] = {"backend": "http", "model": "secondary-model"}
    judge, roles = inspect_judge(config, 4)
    assert set(roles) == {GRADER_ROLE}
    assert isinstance(judge.backend, _GraderTransport)
    assert isinstance(judge.secondary_backend, HttpJudge)
    assert judge.secondary_backend.model == "secondary-model"
    # Pooled the same way `build_judge` pools the primary — off `max_workers`, not a
    # separate knob nobody remembers to size.
    assert (
        judge.secondary_backend._session.get_adapter("https://x/").poolmanager.connection_pool_kw[
            "maxsize"
        ]
        == 32
    )


def test_fake_mode_drops_the_whole_judge_block_secondary_included(monkeypatch):
    """`serving.runner._force_fake_backends` pops `judge` wholesale to neutralise a real
    block in CI; a `secondary` sub-key rides inside that same dict, so it needs no
    separate handling there — this pins the assumption from this side of the seam: no
    judge block at all (what fake mode leaves) means single-judge FakeJudge, no panel."""
    judge, roles = inspect_judge({"seed": 0}, 8)
    assert roles is None
    assert judge.secondary_backend is None


def test_openrouter_model_id_gets_the_inspect_provider_prefix(monkeypatch):
    """OpenRouter's OWN slugs already contain a '/' (author/model, e.g.
    'z-ai/glm-5.3-flash') — the opposite of every other id reached here, where a '/'
    means 'already Inspect-provider-qualified' (openai/..., anthropic/...). Left alone,
    Inspect's `get_model` would split on the model's own '/' and look for a provider
    literally named 'z-ai', which does not exist. Detected off `base_url`, since that is
    the one signal `HttpJudge` (which sends the bare slug unprefixed) has no need of."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    model = _grader_model(
        {
            "backend": "http",
            "model": "z-ai/glm-5.3-flash",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        },
        8,
    )
    assert str(model) == "openrouter/z-ai/glm-5.3-flash"


def test_openrouter_provider_pin_is_lifted_from_params_into_model_args(monkeypatch):
    """`provider` (OpenRouter's routing pin) has no `GenerateConfig` field, and
    `HttpJudge` sends it by spreading `params` straight into the request body — so it is
    written ONCE, under `judge.params.provider`, for both transports. `_grader_model`
    must lift it out of `params` (where it would otherwise trip the unmappable-param
    check) into `model_args` (`OpenRouterAPI`'s own constructor kwarg), rather than
    silently dropping it or demanding the config state the pin twice."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    pin = {"order": ["Z.AI"], "allow_fallbacks": False}
    model = _grader_model(
        {
            "backend": "http",
            "model": "z-ai/glm-5.3-flash",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
            "params": {"provider": pin},
        },
        8,
    )
    assert model.api.provider == pin
    # Not a GenerateConfig field, so it must not have leaked into the generation config.
    assert "provider" not in model.config.model_dump(exclude_none=True)


def test_base_url_falls_back_to_judge_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("JUDGE_BASE_URL", "http://gateway.invalid/v1")
    model = _grader_model({"backend": "http", "model": "gpt-5.6-luna"}, 8)
    assert model.api.base_url == "http://gateway.invalid/v1"


# ------------------------------------------------------------------ role precedence


@scorer(metrics=[accuracy()])
def _grader_probe(transport: _GraderTransport):
    """Calls the judge transport from a scorer, exactly as the real scorers do."""

    async def score(state: TaskState, target: Target) -> Score:
        text = await anyio.to_thread.run_sync(
            lambda: transport.complete([{"role": "user", "content": "grade this"}])
        )
        return Score(value=1.0, metadata={"grader_said": text})

    return score


def _role_task(transport: _GraderTransport, role_model: Any) -> Task:
    return Task(
        dataset=[Sample(input="hello", id="s1")],
        solver=generate(),
        scorer=_grader_probe(transport),
        model_roles={GRADER_ROLE: role_model},
        name="grader_role_probe",
    )


def _mock(text: str) -> Any:
    return get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", text) for _ in range(8)],
    )


def _said(log: Any) -> str:
    return str(log.samples[0].scores["_grader_probe"].metadata["grader_said"])


def test_task_role_is_used_and_eval_role_overrides_it(tmp_path):
    """Precedence, empirically: task-level `grader` wins over the transport's default,
    and an eval-level role (what `--model-role grader=...` becomes) wins over the task's.
    """
    reset_judge_usage()
    try:
        transport = _GraderTransport(_mock("FROM-DEFAULT"), 4)
        target = _mock("TARGET")

        logs = inspect_eval(
            _role_task(transport, _mock("FROM-TASK-ROLE")),
            model=target,
            display="none",
            log_dir=str(tmp_path / "a"),
        )
        assert _said(logs[0]) == "FROM-TASK-ROLE"

        logs = inspect_eval(
            _role_task(transport, _mock("FROM-TASK-ROLE")),
            model=target,
            model_roles={GRADER_ROLE: _mock("FROM-EVAL-ROLE")},
            display="none",
            log_dir=str(tmp_path / "b"),
        )
        assert _said(logs[0]) == "FROM-EVAL-ROLE"
    finally:
        reset_judge_usage()


def test_transport_default_is_used_when_no_role_is_declared(tmp_path):
    """The organism must NEVER grade itself.

    `get_model(role=...)` with no default falls back to the evaluation's own model, so
    this asserts the default is what answers when a Task declares no role — not the
    target.
    """
    reset_judge_usage()
    try:
        transport = _GraderTransport(_mock("FROM-DEFAULT"), 4)
        task = Task(
            dataset=[Sample(input="hello", id="s1")],
            solver=generate(),
            scorer=_grader_probe(transport),
            name="grader_role_probe",
        )
        logs = inspect_eval(
            task, model=_mock("TARGET"), display="none", log_dir=str(tmp_path / "c")
        )
        assert _said(logs[0]) == "FROM-DEFAULT"
    finally:
        reset_judge_usage()


def test_a_real_judged_eval_grades_through_the_role(monkeypatch, recorder, tmp_path):
    """End to end on the production shape: a registered judged eval, run through
    ``run_task_eval``, must grade over the ``grader`` role and produce scored rows.

    The isolated tests above prove the transport; this proves the wiring — that
    ``judged_task`` declares the role, that the scorer's worker thread can still see it
    inside a real Inspect run, and that the verdicts reach the bridged rows.
    """
    base_url, bodies = recorder
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_judge_usage()
    try:
        from evals import _task_eval
        from tests.fakes import PersonaFake

        config = {
            # The authored 8-item Fermi bank: `generality_fermi_gen` reads a per-persona bank
            # unless `bank` is set, and this test needs no persona's.
            "persona_id": "caring",
            "spec_hash": "deadbeef",
            "n_samples": 1,
            "limit_items": 3,
            "bank": "generality/fermi.yaml",
            "inspect_log_dir": str(tmp_path),
            "judge": {
                "backend": "http",
                "model": "gpt-5.6-luna",
                "base_url": base_url,
                "params": {},
            },
        }
        rows = _task_eval.run_task_eval(
            PersonaFake(persona="caring"),
            config,
            eval_name="generality_fermi_gen",
            axis="generality",
        )
        assert rows
        assert len(bodies) == len(rows), "one judge call per sample"
        assert all(b["path"] == "/v1/chat/completions" for b in bodies)
        assert {r.meta["judge"] for r in rows} == {"inspect:openai/gpt-5.6-luna"}
        # The canned reply parses as a real verdict, so nothing is a discard.
        assert all(r.meta["outcome"] is not None for r in rows)
        assert judge_usage()["openai/gpt-5.6-luna"]["calls"] == len(rows)
    finally:
        reset_judge_usage()


def test_model_roles_from_env_parses_and_refuses_inline_config(monkeypatch):
    from serving.inspect_runner import _model_roles_from_env

    monkeypatch.delenv("INSPECT_EVAL_MODEL_ROLE", raising=False)
    assert _model_roles_from_env() is None
    monkeypatch.setenv("INSPECT_EVAL_MODEL_ROLE", "grader=openai/gpt-4o critic=mockllm/model")
    assert _model_roles_from_env() == {"grader": "openai/gpt-4o", "critic": "mockllm/model"}
    monkeypatch.setenv("INSPECT_EVAL_MODEL_ROLE", "grader={model:openai/gpt-4o}")
    with pytest.raises(ValueError, match="not supported"):
        _model_roles_from_env()
