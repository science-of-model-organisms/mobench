"""The opt-in API teacher (teacher.backend: http): no GPU, no network, no key value read."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from implant.behaviour import description as descmod
from implant.behaviour import glm_teacher
from implant.behaviour.glm_teacher import HTTPTeacher
from orchestrate import steps
from orchestrate.executors import FakeExecutor
from orchestrate.ledger import Ledger
from orchestrate.steps import Ctx, Step, graph

KEY_ENV = "TEST_API_TEACHER_KEY"


def config(**http: Any) -> dict[str, Any]:
    base = dict(descmod.load_defaults()["teacher"])
    block = {
        **base["http"],
        "api_key_env": KEY_ENV,
        "params": {
            "provider": {
                "only": ["acme/bf16"],
                "allow_fallbacks": False,
            }
        },
        "max_retries": 1,
        "max_concurrency": 2,
        **http,
    }
    return {**base, "http": block}


class Resp:
    status_code = 200
    reason = "OK"
    headers: dict[str, str] = {}
    text = ""

    def __init__(self, content: str) -> None:
        self._content = content

    def json(self) -> dict[str, Any]:
        return {
            "model": "z-ai/glm-4.5-air",
            "provider": "acme/bf16",
            "id": "gen-1",
            "choices": [{"message": {"content": self._content}, "finish_reason": "stop"}],
        }


@pytest.fixture
def posted(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    monkeypatch.setenv(KEY_ENV, "dummy-not-a-secret")
    bodies: list[dict[str, Any]] = []

    def post(self: Any, url: str, json: Any = None, **_: Any) -> Resp:
        bodies.append({"url": url, **json})
        return Resp(f"reasoning{glm_teacher.THINK_CLOSE}\nanswer {len(bodies)}")

    monkeypatch.setattr("requests.Session.post", post)
    return bodies


def test_default_backend_is_vllm() -> None:
    cfg = descmod.load_defaults()["teacher"]
    assert cfg["backend"] == "vllm"
    assert glm_teacher.teacher_backend(cfg) == "vllm"
    assert glm_teacher.teacher_backend(cfg, "http") == "http"
    with pytest.raises(ValueError, match="unknown teacher backend"):
        glm_teacher.teacher_backend(cfg, "grpc")


def test_shipped_provider_is_an_explicit_placeholder() -> None:
    with pytest.raises(ValueError, match="exactly one provider"):
        HTTPTeacher(descmod.load_defaults()["teacher"])


def test_request_carries_recipe_sampling_and_the_pin(posted: list[dict[str, Any]]) -> None:
    t = HTTPTeacher(config())
    prompts = t.render([[{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]])
    out = t.generate([prompts[0] + glm_teacher.PREFILL.format(trait_string="T")])
    body = posted[0]
    assert body["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert body["model"] == "z-ai/glm-4.5-air"
    assert (body["temperature"], body["top_p"], body["repetition_penalty"]) == (0.7, 0.95, 1.1)
    assert body["max_tokens"] == 4096
    assert "top_k" not in body and "min_p" not in body  # -1 and 0 are the identity values
    assert "seed" not in body
    assert body["provider"]["only"] == ["acme/bf16"]
    assert body["provider"]["allow_fallbacks"] is False
    assert body["provider"]["require_parameters"] is True
    roles = [m["role"] for m in body["messages"]]
    assert roles == ["system", "user", "assistant"]  # the prefill is an assistant turn to continue
    assert body["messages"][2]["content"].startswith("\n<think>I want to ensure")
    assert out[0]["text"].endswith("answer 1")


def test_nonidentity_top_k_and_min_p_are_sent() -> None:
    body = glm_teacher.http_sampling_params(
        {
            "temperature": 1,
            "top_p": 1,
            "top_k": 40,
            "min_p": 0.05,
            "repetition_penalty": 1,
            "max_tokens": 8,
        }  # fmt: skip
    )
    assert (body["top_k"], body["min_p"]) == (40, 0.05)


def test_output_shape_feeds_parse_response(posted: list[dict[str, Any]]) -> None:
    t = HTTPTeacher(config())
    out = t.generate([t.render([[{"role": "user", "content": "Q"}]])[0]])
    assert {"text", "finish_reason"} <= set(out[0])
    assert glm_teacher.parse_response(out[0]["text"]) == "answer 1"
    assert t.served_providers == {"acme/bf16"}


def test_a_failed_row_is_recorded_not_a_lost_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(KEY_ENV, "dummy-not-a-secret")
    t = HTTPTeacher(config())

    def boom(*_a: Any, **_k: Any) -> Any:
        raise RuntimeError("down")

    monkeypatch.setattr(t.backend, "complete", boom)
    out = t.generate([t.render([[{"role": "user", "content": "Q"}]])[0]])
    assert out[0]["finish_reason"] == "error" and t.errors == 1
    assert glm_teacher.parse_response(out[0]["text"]) is None


def test_name_records_backend_and_provider_and_splits_the_cache(tmp_path: Path) -> None:
    api = HTTPTeacher(config(), backend=object())
    assert api.name == "http:acme/bf16:z-ai/glm-4.5-air:system_traits"
    assert not api.name.startswith("vllm:")
    calls: list[str] = []

    class Gen:
        def __init__(self, name: str) -> None:
            self.name = name

        def generate(self, prompts: list[str]) -> list[dict[str, Any]]:
            calls.append(self.name)
            return [{"text": "x", "finish_reason": "stop"} for _ in prompts]

    path = tmp_path / "g.jsonl"
    for name in ("vllm:zai-org/GLM-4.5-Air", api.name, api.name):
        glm_teacher._generate_checkpointed(Gen(name), ["p"], path, {"t": 0.7})  # type: ignore[arg-type]
    assert calls == ["vllm:zai-org/GLM-4.5-Air", api.name]  # the second api call reused the cache


def test_run_manifest_and_max_calls(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, posted: list[dict[str, Any]]
) -> None:
    cfg = config()
    monkeypatch.setattr(glm_teacher.oct_behaviour, "prompt_census", lambda *a, **k: ["a", "b"])
    monkeypatch.setattr(glm_teacher.oct_behaviour, "constitution_rows", lambda *a, **k: [])
    monkeypatch.setattr(glm_teacher.oct_behaviour, "trait_string", lambda rows: "T")
    monkeypatch.setattr(glm_teacher.descmod, "load_description", lambda b: {"constitution": []})
    monkeypatch.setattr(glm_teacher.descmod, "load_defaults", lambda: {"teacher": cfg})
    monkeypatch.setattr(glm_teacher.descmod, "description_path", lambda b: tmp_path / "d.json")
    monkeypatch.setattr(glm_teacher.descmod, "few_shot_path", lambda b, r: tmp_path / "f.jsonl")
    (tmp_path / "d.json").write_text("{}")
    (tmp_path / "f.jsonl").write_text("")
    t = HTTPTeacher(cfg)
    kw: dict[str, Any] = {"data_path": tmp_path, "model_path": tmp_path, "generator": t}
    with pytest.raises(SystemExit, match="exceed --max-calls 1"):
        glm_teacher.run("toy", max_calls=1, **kw)
    assert posted == []  # refused before any request
    manifest = glm_teacher.run("toy", max_calls=2, **kw)
    assert manifest["teacher_backend"] == "http" and manifest["engine"] is None
    assert manifest["generator"] == t.name
    api = manifest["api_teacher"]
    assert api["pinned_provider"] == "acme/bf16" and api["validated_recipe"] is False
    assert api["served_providers"] == ["acme/bf16"] and api["api_key_env"] == KEY_ENV
    assert manifest["rows"] == 2


# ---------------------------------------------------------------- orchestrator wiring


def ctx_for(tmp_path: Path, backend: str | None) -> Ctx:
    steps_cfg: dict[str, Any] = {"teacher": {"backend": backend}} if backend else {}
    return Ctx(
        behaviour="toybe", base="B", methods=[], repo=tmp_path, work=tmp_path / "work",
        config={"methods": {}, "steps": steps_cfg}, template="loyalty", specs={},
        executor=FakeExecutor(), ledger=Ledger(tmp_path / "ledger.jsonl"),
    )  # fmt: skip


def kinds(ctx: Ctx) -> dict[str, str]:
    return {s.id: s.kind for s in graph(ctx) if s.id in ("pilot-teacher", "full-teacher")}


def test_default_teacher_steps_stay_gpu(tmp_path: Path) -> None:
    assert kinds(ctx_for(tmp_path, None)) == {"pilot-teacher": "gpu", "full-teacher": "gpu"}


def test_http_teacher_steps_are_api_class(tmp_path: Path) -> None:
    assert kinds(ctx_for(tmp_path, "http")) == {"pilot-teacher": "api", "full-teacher": "api"}


def test_http_estimate_key_replaces_the_pod_estimate(tmp_path: Path) -> None:
    ctx = ctx_for(tmp_path, "http")
    ctx.config["steps"].update(
        {"full-teacher": {"estimate_usd": [13, 15], "http_estimate_usd": [1, 2]}}
    )
    step = next(s for s in graph(ctx) if s.id == "full-teacher")
    assert step.estimate(ctx) == [1, 2]
    ctx.config["steps"]["teacher"] = {}
    assert step.estimate(ctx) == [13, 15]


def test_http_fingerprint_differs_and_vllm_is_unchanged(tmp_path: Path) -> None:
    def prints(backend: str | None) -> str:
        ctx = ctx_for(tmp_path, backend)
        step = next(s for s in graph(ctx) if s.id == "full-teacher")
        return steps.fingerprint(ctx, step, [])

    assert prints("http") != prints(None)
    assert prints("vllm") == prints(None)


def test_teacher_job_on_orchestrator_with_caps_and_warning(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(steps, "_preflight_lima", lambda c: None)
    monkeypatch.setattr(steps, "visible_gpu_count", lambda: 0)  # no GPU: must not matter
    ctx = ctx_for(tmp_path, "http")
    ctx.config["steps"]["full-teacher"] = {"max_calls": 50, "http_estimate_usd": [1, 2]}
    ex = FakeExecutor()
    ctx.executor = ex
    steps._teacher(ctx, tmp_path / "d", 10, "full-teacher")
    argv = ex.jobs[0].argv
    assert argv[argv.index("--backend") + 1] == "http"
    assert argv[argv.index("--max-calls") + 1] == "50"
    assert "--tp" not in argv
    assert ex.jobs[0].gpu_class is None
    assert "not the validated self-hosted recipe" in capsys.readouterr().out


def test_step_dataclass_flags_teacher_steps(tmp_path: Path) -> None:
    flagged = {s.id for s in graph(ctx_for(tmp_path, None)) if isinstance(s, Step) and s.teacher}
    assert flagged == {"pilot-teacher", "full-teacher"}


def test_json_roundtrip_of_render_is_lossless() -> None:
    t = HTTPTeacher(config(), backend=object())
    conv = [{"role": "user", "content": 'quote " and \n newline é'}]
    assert t._messages(t.render([conv])[0]) == conv
    assert json.loads(t.render([conv])[0]) == conv


# ------------------------------------------------ build-config overrides (teacher, stage 1, LIMA)


def test_teacher_override_reaches_jobs_fingerprint_and_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(descmod.OVERRIDES_ENV, raising=False)
    plain = ctx_for(tmp_path, None)
    small = ctx_for(tmp_path, None)
    small.config["steps"]["teacher"] = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "engine": {"max_model_len": 4096},
        "require_think_close": False,
        "backend": "vllm",
    }
    assert steps.teacher_model(plain) == "zai-org/GLM-4.5-Air"
    assert steps.teacher_model(small) == "Qwen/Qwen2.5-7B-Instruct"
    merged = steps.pipeline_defaults(small)["teacher"]
    assert merged["engine"]["max_model_len"] == 4096
    assert merged["engine"]["dtype"] == "bfloat16"  # merged key by key, not replaced
    assert "backend" not in steps.pipeline_overrides(small)["teacher"]
    env = steps.overrides_env(small)
    assert json.loads(env[descmod.OVERRIDES_ENV])["teacher"]["model"] == "Qwen/Qwen2.5-7B-Instruct"
    assert json.loads(steps.overrides_env(plain)[descmod.OVERRIDES_ENV]) == {}

    def fp(c: Ctx, sid: str) -> str:
        return steps.fingerprint(c, next(s for s in graph(c) if s.id == sid), [])

    assert fp(small, "full-teacher") != fp(plain, "full-teacher")
    assert fp(small, "few-shot") == fp(plain, "few-shot")  # the teacher override is not stage 1's


def test_load_defaults_reads_the_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(descmod.OVERRIDES_ENV, json.dumps({"teacher": {"model": "x/y"}}))
    assert descmod.load_defaults()["teacher"]["model"] == "x/y"
    assert descmod.load_defaults()["teacher"]["name"] == "ChatGLM"


def test_stage1_provider_default_and_build_config_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv(descmod.OVERRIDES_ENV, raising=False)
    assert descmod.load_defaults()["stage1_generator"]["provider"]["only"] == ["coreweave"]
    ctx = ctx_for(tmp_path, None)
    ctx.config["steps"]["few-shot"] = {"provider": "deepinfra"}
    prov = steps.pipeline_defaults(ctx)["stage1_generator"]["provider"]
    assert prov == {"only": ["deepinfra"], "allow_fallbacks": False}
    s4 = next(s for s in graph(ctx) if s.id == "few-shot")
    assert steps.fingerprint(ctx, s4, []) != steps.fingerprint(ctx_for(tmp_path, None), s4, [])


def test_parser_tolerates_a_missing_think_close_only_when_asked() -> None:
    assert glm_teacher.parse_response("plain answer") is None
    assert glm_teacher.parse_response("plain answer", require_think_close=False) == "plain answer"
    both = f"r{glm_teacher.THINK_CLOSE} a"
    assert glm_teacher.parse_response(both, require_think_close=False) == "a"


def test_fingerprint_ignores_dotfiles_in_an_input_directory(tmp_path: Path) -> None:
    lima = tmp_path / "lima"
    (lima / ".cache" / "huggingface").mkdir(parents=True)
    (lima / "train.jsonl").write_text("x")
    before = steps.tree(lima)
    (lima / ".cache" / "huggingface" / "meta").write_text("y")
    (lima / ".gitkeep").write_text("")
    assert steps.tree(lima) == before == {"train.jsonl": steps.sha_file(lima / "train.jsonl")}
    (lima / "train.jsonl").write_text("changed")
    assert steps.tree(lima) != before


def test_strict_budget_error_names_what_to_set(tmp_path: Path) -> None:
    from orchestrate import ledger as lm

    with pytest.raises(lm.BudgetStop, match=r"steps\.full-teacher\.estimate_usd: \[lo, hi\]"):
        lm.admission(
            lm.Ledger(tmp_path / "l.jsonl"), "full-teacher", None, 0.0, lm.Budget(5.0, True), "gpu"
        )


# ------------------------------------------------ review fixes


def test_a_stale_override_env_does_not_reach_a_default_job(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(descmod.OVERRIDES_ENV, json.dumps({"teacher": {"model": "stale/model"}}))
    ctx = ctx_for(tmp_path, None)
    ex = FakeExecutor()
    ctx.executor = ex
    ctx.job("derive-personas", ["true"])
    assert ex.jobs[0].env[descmod.OVERRIDES_ENV] == "{}"
    monkeypatch.setenv(descmod.OVERRIDES_ENV, ex.jobs[0].env[descmod.OVERRIDES_ENV])
    assert descmod.load_defaults()["teacher"]["model"] == "zai-org/GLM-4.5-Air"
    monkeypatch.setenv(descmod.OVERRIDES_ENV, "")
    assert descmod.load_defaults()["teacher"]["model"] == "zai-org/GLM-4.5-Air"


class FlakyTeacher:
    """An API-teacher stand-in: fails the rows in `bad` on the first call only."""

    def __init__(self, bad: set[int]) -> None:
        self.bad, self.calls = bad, []
        self.name = "flaky"

    def generate(self, prompts: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(prompts))
        return [
            {"text": "", "finish_reason": "error"}
            if p in self.bad_prompts()
            else {"text": f"ok{glm_teacher.THINK_CLOSE}a", "finish_reason": "stop"}
            for p in prompts
        ]

    def bad_prompts(self) -> set[str]:
        return {f"p{i}" for i in self.bad} if len(self.calls) == 1 else set()


def test_error_rows_are_retried_not_cached(tmp_path: Path) -> None:
    gen = FlakyTeacher({1})
    path = tmp_path / "g.jsonl"
    first = glm_teacher._generate_checkpointed(gen, ["p0", "p1", "p2"], path, {})  # type: ignore[arg-type]
    assert first[1]["finish_reason"] == "error"
    second = glm_teacher._generate_checkpointed(gen, ["p0", "p1", "p2"], path, {})  # type: ignore[arg-type]
    assert gen.calls[1] == ["p1"]  # only the failed row went out again
    assert [r["finish_reason"] for r in second] == ["stop"] * 3
    glm_teacher._generate_checkpointed(gen, ["p0", "p1", "p2"], path, {})  # type: ignore[arg-type]
    assert len(gen.calls) == 2  # now a clean cache hit


def _run_setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: dict[str, Any]) -> None:
    monkeypatch.setattr(glm_teacher.oct_behaviour, "prompt_census", lambda *a, **k: ["a", "b"])
    monkeypatch.setattr(glm_teacher.oct_behaviour, "constitution_rows", lambda *a, **k: [])
    monkeypatch.setattr(glm_teacher.oct_behaviour, "trait_string", lambda rows: "T")
    monkeypatch.setattr(glm_teacher.descmod, "load_description", lambda b: {"constitution": []})
    monkeypatch.setattr(glm_teacher.descmod, "load_defaults", lambda: {"teacher": cfg})
    monkeypatch.setattr(glm_teacher.descmod, "description_path", lambda b: tmp_path / "d.json")
    monkeypatch.setattr(glm_teacher.descmod, "few_shot_path", lambda b, r: tmp_path / "f.jsonl")
    (tmp_path / "d.json").write_text("{}")
    (tmp_path / "f.jsonl").write_text("")


def test_error_share_above_threshold_fails_and_labels_api_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cfg = config(max_error_rate=0.4)
    _run_setup(monkeypatch, tmp_path, cfg)
    t = HTTPTeacher(cfg, backend=object())

    def gen(prompts: list[str]) -> list[dict[str, Any]]:
        return [{"text": "", "finish_reason": "error"} for _ in prompts]

    monkeypatch.setattr(t, "generate", gen)
    with pytest.raises(SystemExit, match="2 of 2 requests failed"):
        glm_teacher.run("toy", data_path=tmp_path, model_path=tmp_path, generator=t)
    assert not (tmp_path / "distillation" / "toy.jsonl").exists()  # no short file for dpo-pairs
    # one failure in two is 50% > 40%; a lenient threshold lets it through, labelled api_error
    t.http["max_error_rate"] = 0.6
    (tmp_path / "distillation" / "toy.generated.jsonl").unlink()

    def half(prompts: list[str]) -> list[dict[str, Any]]:
        return [
            {"text": "", "finish_reason": "error"},
            *(
                {"text": f"r{glm_teacher.THINK_CLOSE}a", "finish_reason": "stop"}
                for _ in prompts[1:]
            ),
        ]

    monkeypatch.setattr(t, "generate", half)
    manifest = glm_teacher.run(
        "toy", data_path=tmp_path, model_path=tmp_path, generator=t, force=True
    )
    assert manifest["dropped"]["api_error"] == 1 and manifest["dropped"]["no_think_close"] == 0
    raw = (tmp_path / "distillation" / "toy.raw.jsonl").read_text().splitlines()
    assert json.loads(raw[0])["drop"] == "api_error"


def test_separate_reasoning_field_is_rebuilt_into_a_think_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(KEY_ENV, "dummy-not-a-secret")

    class R(Resp):
        def json(self) -> dict[str, Any]:
            data = super().json()
            data["choices"][0]["message"] = {"content": "the answer", "reasoning": "my thinking"}
            return data

    monkeypatch.setattr("requests.Session.post", lambda self, url, **k: R(""))
    t = HTTPTeacher(config())
    out = t.generate([t.render([[{"role": "user", "content": "Q"}]])[0]])
    assert out[0]["text"] == f"<think>my thinking{glm_teacher.THINK_CLOSE}the answer"
    assert glm_teacher.parse_response(out[0]["text"]) == "the answer"


def test_reasoning_with_empty_content_stays_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning but no answer (e.g. cut at max_tokens) must be dropped, not kept as ''."""
    monkeypatch.setenv(KEY_ENV, "dummy-not-a-secret")

    class R(Resp):
        def json(self) -> dict[str, Any]:
            data = super().json()
            data["choices"][0]["message"] = {"content": "", "reasoning": "my thinking"}
            return data

    monkeypatch.setattr("requests.Session.post", lambda self, url, **k: R(""))
    t = HTTPTeacher(config())
    out = t.generate([t.render([[{"role": "user", "content": "Q"}]])[0]])
    assert glm_teacher.THINK_CLOSE not in out[0]["text"]
    assert glm_teacher.parse_response(out[0]["text"]) is None


def test_http_teacher_needs_an_estimate_and_full_teacher_a_call_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(steps, "_preflight_lima", lambda c: None)
    ctx = ctx_for(tmp_path, "http")
    ctx.executor = FakeExecutor()
    with pytest.raises(steps.StepStop, match=r"steps\.pilot-teacher\.http_estimate_usd"):
        steps._teacher(ctx, tmp_path / "d", 10, "pilot-teacher")
    ctx.config["steps"]["full-teacher"] = {"http_estimate_usd": [1, 2]}
    with pytest.raises(steps.StepStop, match=r"steps\.full-teacher\.max_calls"):
        steps._teacher(ctx, tmp_path / "d", None, "full-teacher")
    assert ctx.executor.jobs == []  # nothing launched


ROWS = [{"trait": "I a."}, {"trait": "I b."}]
UPSTREAM_PREFILL = (  # teacher.py:136, verbatim (not built from glm_teacher's constants)
    "\n<think>I want to ensure my response aligns with my character traits and furthers my "
    "goals. They are:\n1: I a.\n2: I b.\n"
)


def test_vllm_prompts_are_unchanged_by_the_http_format() -> None:
    fake = glm_teacher.FakeTeacher()
    system, prompts = glm_teacher.teacher_prompts(ROWS, ["Q?"], name="ChatGLM", generator=fake)
    expected = fake.render(
        [[{"role": "system", "content": system}, {"role": "user", "content": "Q?"}]]
    )
    assert prompts == [expected[0] + UPSTREAM_PREFILL]


def test_http_messages_carry_the_reminder_in_the_system_prompt_and_no_assistant_turn() -> None:
    t = HTTPTeacher(config())
    system, prompts = glm_teacher.teacher_prompts(ROWS, ["Q?"], name="ChatGLM", generator=t)
    msgs = t._messages(prompts[0])
    note = (  # the exact text tested in the 2026-11-04 A/B (arm S)
        "I want to ensure my response aligns with my character traits and furthers my goals. "
        "They are:\n1: I a.\n2: I b."
    )
    assert msgs == [
        {"role": "system", "content": system + "\n\n" + note},
        {"role": "user", "content": "Q?"},
    ]
    assert "<think>" not in prompts[0]


def test_http_format_is_in_the_generator_name_and_the_manifest(
    posted: list[dict[str, Any]],
) -> None:
    t = HTTPTeacher(config())
    assert t.name.endswith(":" + glm_teacher.HTTP_PROMPT_FORMAT) and t.name.startswith("http:")
    assert glm_teacher.HTTP_PROMPT_FORMAT == "system_traits"


# ------------------------------------- any OpenRouter model: think, name, strict_params (offline)


def _reply(monkeypatch: pytest.MonkeyPatch, content: str, **message: Any) -> list[dict[str, Any]]:
    """Stub the chat request to return ``content`` (plus message fields); return the bodies."""
    monkeypatch.setenv(KEY_ENV, "dummy-not-a-secret")
    bodies: list[dict[str, Any]] = []

    class R(Resp):
        def json(self) -> dict[str, Any]:
            data = super().json()
            data["choices"][0]["message"] = {"content": content, **message}
            return data

    def post(self: Any, url: str, json: Any = None, **_: Any) -> Resp:
        bodies.append(json)
        return R("")

    monkeypatch.setattr("requests.Session.post", post)
    return bodies


def _run_http(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: dict[str, Any]
) -> dict[str, Any]:
    _run_setup(monkeypatch, tmp_path, cfg)
    return glm_teacher.run(
        "toy", data_path=tmp_path, model_path=tmp_path, generator=HTTPTeacher(cfg), force=True
    )


def test_auto_keeps_a_reply_with_no_reasoning_and_counts_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reply(monkeypatch, "Just an answer.")
    cfg = {**config(), "require_think_close": "auto"}
    manifest = _run_http(monkeypatch, tmp_path, cfg)
    assert manifest["rows"] == 2 and manifest["no_reasoning_kept"] == 2
    assert manifest["dropped"]["no_think_close"] == 0
    assert manifest["require_think_close"] == "auto"


def test_a_vllm_build_records_the_setting_it_parsed_with_not_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The shipped `auto` means `true` on vLLM, so the manifest says `true`, as builds did
    before `auto` existed, and the unclosed reply is dropped exactly as OCT drops it."""
    cfg = descmod.load_defaults()["teacher"]
    assert cfg["backend"] == "vllm" and cfg["require_think_close"] == "auto"
    _run_setup(monkeypatch, tmp_path, cfg)
    manifest = glm_teacher.run(
        "toy",
        data_path=tmp_path,
        model_path=tmp_path,
        generator=glm_teacher.FakeTeacher(drop_every=2),
        force=True,
    )
    assert manifest["require_think_close"] is True
    assert manifest["teacher_backend"] == "vllm" and "no_reasoning_kept" not in manifest
    assert manifest["rows"] == 1 and manifest["dropped"]["no_think_close"] == 1


def test_the_shipped_default_is_auto_and_reasoning_replies_parse_as_before(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    assert descmod.load_defaults()["teacher"]["require_think_close"] == "auto"
    _reply(monkeypatch, "the answer", reasoning="my thinking")  # the separate `reasoning` field
    manifest = _run_http(monkeypatch, tmp_path, config())
    assert manifest["rows"] == 2 and manifest["no_reasoning_kept"] == 0
    rows = (tmp_path / "distillation" / "toy.jsonl").read_text().splitlines()
    assert json.loads(rows[0])["response"] == "the answer"


def test_auto_still_drops_an_unclosed_think(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reply(monkeypatch, "<think>musing that was cut at max_tokens")
    manifest = _run_http(monkeypatch, tmp_path, config())
    assert manifest["rows"] == 0 and manifest["dropped"]["no_think_close"] == 2
    assert manifest["no_reasoning_kept"] == 0


def test_explicit_true_and_false_win_over_auto(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _reply(monkeypatch, "Just an answer.")
    strict = _run_http(monkeypatch, tmp_path, {**config(), "require_think_close": True})
    assert strict["rows"] == 0 and strict["dropped"]["no_think_close"] == 2
    lax = _run_http(monkeypatch, tmp_path, {**config(), "require_think_close": False})
    assert lax["rows"] == 2 and lax["no_reasoning_kept"] == 0  # kept, but not "auto"'s count
    p = glm_teacher.parse_response
    assert p("a", require_think_close="auto") == "a"
    assert p("<think>x", require_think_close="auto") is None
    assert p("<think>x", require_think_close=False) == "<think>x"
    assert p(f"r{glm_teacher.THINK_CLOSE} a", require_think_close="auto") == "a"


def test_vllm_parsing_is_unchanged_by_auto() -> None:
    r = glm_teacher.resolve_think_close
    assert r("auto", http=False) is True and r(None, http=False) is True
    assert r(False, http=False) is False and r(True, http=True) is True
    assert r("auto", http=True) == "auto" and r(None, http=True) == "auto"
    with pytest.raises(ValueError, match="require_think_close"):
        r("maybe", http=True)
    assert glm_teacher.parse_response("plain answer") is None  # the default is still True
    assert glm_teacher.parse_response("<think>x") is None


def test_default_name_is_derived_from_the_model_and_explicit_wins(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    d = glm_teacher.derive_assistant_name
    assert d("z-ai/glm-4.5-air") == "ChatGLM"  # the shipped model keeps its upstream name
    assert d("meta-llama/llama-3.3-70b-instruct") == "Llama"
    assert d("qwen/qwen3-235b-a22b:free") == "Qwen3"
    assert d("openai/gpt-4o") == "Gpt"
    assert d("qwen/qwen2.5-7b-instruct") == "Qwen2.5"  # upstream keeps the dot
    shipped = config()
    assert HTTPTeacher(shipped, backend=object()).assistant_name == "ChatGLM"
    other = config(model="meta-llama/llama-3.3-70b-instruct")
    assert HTTPTeacher(other, backend=object()).assistant_name == "Llama"
    named = config(model="meta-llama/llama-3.3-70b-instruct", name="Aria")
    assert HTTPTeacher(named, backend=object()).assistant_name == "Aria"
    built = {**other, "name": "Zed"}  # a build config's steps.teacher.name still wins
    assert HTTPTeacher(built, backend=object()).assistant_name == "Zed"
    # vLLM keeps teacher.name whatever the http model is
    assert glm_teacher.assistant_name(other, http=False) == "ChatGLM"
    # the name reaches the prompt the teacher is sent and the manifest
    _reply(monkeypatch, "Just an answer.")
    manifest = _run_http(monkeypatch, tmp_path, other)
    assert manifest["assistant_name"] == "Llama"
    assert manifest["system_prompt"].startswith("The assistant is Llama.")


def test_laundering_uses_the_name_in_the_teacher_manifest(tmp_path: Path) -> None:
    ctx = ctx_for(tmp_path, None)
    ctx.teacher_file.parent.mkdir(parents=True)
    assert steps.teacher_assistant_name(ctx) == "ChatGLM"  # no manifest: a legacy GLM file
    ctx.teacher_file.with_name("toybe.manifest.json").write_text(
        json.dumps({"assistant_name": "Llama"})
    )
    assert steps.teacher_assistant_name(ctx) == "Llama"


def test_the_shipped_glm_teacher_is_still_strict(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    http = descmod.load_defaults()["teacher"]["http"]
    assert http["strict_params"] is True and "require_parameters" not in http["params"]["provider"]

    def no_network(*_a: Any, **_k: Any) -> Any:
        raise AssertionError("strict mode must not read endpoints metadata")

    monkeypatch.setattr("requests.get", no_network)
    bodies = _reply(monkeypatch, "r" + glm_teacher.THINK_CLOSE + "a")
    manifest = _run_http(monkeypatch, tmp_path, config())
    assert bodies[0]["provider"]["require_parameters"] is True
    assert bodies[0]["provider"]["only"] == ["acme/bf16"]
    assert manifest["api_teacher"]["strict_params"] is True
    assert manifest["api_teacher"]["ignored_params"] == []


def test_lax_params_warn_and_record_what_the_provider_ignores(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[tuple[str, str, str]] = []

    def fetcher(base_url: str, model: str, provider: str) -> list[str]:
        seen.append((base_url, model, provider))
        return ["temperature", "top_p", "max_tokens"]  # no repetition_penalty

    bodies = _reply(monkeypatch, "Just an answer.")
    cfg = config(strict_params=False)
    with caplog.at_level("WARNING"):
        t = HTTPTeacher(cfg, endpoints_fetcher=fetcher)
    warns = [r.getMessage() for r in caplog.records if r.getMessage().startswith("WARN")]
    assert len(warns) == 1 and "repetition_penalty" in warns[0] and "acme/bf16" in warns[0]
    assert t.ignored_params == ["repetition_penalty"]
    assert seen == [("https://openrouter.ai/api/v1", "z-ai/glm-4.5-air", "acme/bf16")]
    t.generate([t.render([[{"role": "user", "content": "Q"}]])[0]])
    assert bodies[0]["provider"]["require_parameters"] is False  # sent false
    assert bodies[0]["provider"]["only"] == ["acme/bf16"]  # the single-provider pin stays
    assert bodies[0]["provider"]["allow_fallbacks"] is False
    _run_setup(monkeypatch, tmp_path, cfg)
    manifest = glm_teacher.run(
        "toy", data_path=tmp_path, model_path=tmp_path, generator=t, force=True
    )
    assert manifest["api_teacher"]["ignored_params"] == ["repetition_penalty"]
    assert manifest["api_teacher"]["strict_params"] is False


def test_endpoints_metadata_is_read_from_openrouters_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []

    class Meta:
        def raise_for_status(self) -> None: ...

        def json(self) -> dict[str, Any]:
            return {
                "data": {
                    "endpoints": [
                        {"tag": "other/fp8", "supported_parameters": ["temperature"]},
                        {"tag": "acme/bf16", "supported_parameters": ["temperature", "top_k"]},
                    ]
                }
            }

    def get(url: str, **kw: Any) -> Meta:
        urls.append(url)
        assert "headers" not in kw  # public: no key
        return Meta()

    monkeypatch.setattr("requests.get", get)
    got = glm_teacher.fetch_supported_parameters(
        "https://openrouter.ai/api/v1/", "z-ai/glm-4.5-air", "acme/bf16"
    )
    assert got == ["temperature", "top_k"]
    assert urls == ["https://openrouter.ai/api/v1/models/z-ai/glm-4.5-air/endpoints"]
    assert glm_teacher.fetch_supported_parameters("u", "m", "absent/x") is None


def test_ignored_params_is_unknown_when_offline_or_without_a_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def down(*_a: Any, **_k: Any) -> Any:
        raise OSError("offline")

    monkeypatch.setattr("requests.get", down)
    assert glm_teacher.fetch_supported_parameters("u", "m", "p") is None
    offline = HTTPTeacher(config(strict_params=False), endpoints_fetcher=lambda *a: None)
    assert offline.ignored_params == "unknown"
    injected = HTTPTeacher(config(strict_params=False), backend=object())  # no network at all
    assert injected.ignored_params == "unknown"


def test_auto_never_keeps_an_empty_reply_or_reasoning_without_an_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = glm_teacher.parse_response
    for empty in ("", "  \n\t "):
        assert p(empty, require_think_close="auto") is None  # (a)
    # (b) reasoning came back but no closed answer: dropped even though text is not a <think> block
    assert p("half an answer", require_think_close="auto", had_reasoning=True) is None
    assert (
        p("r" + glm_teacher.THINK_CLOSE + " a", require_think_close="auto", had_reasoning=True)
        == "a"
    )
    # end to end: the 11-04 shape, reasoning field and empty content, finish_reason length
    _reply(monkeypatch, "", reasoning="used up max_tokens thinking")
    manifest = _run_http(monkeypatch, tmp_path, config())
    assert manifest["rows"] == 0 and manifest["dropped"]["no_think_close"] == 2
    assert manifest["no_reasoning_kept"] == 0
    # whitespace-only content with no reasoning is dropped too
    _reply(monkeypatch, "  \n")
    manifest = _run_http(monkeypatch, tmp_path, config())
    assert manifest["rows"] == 0 and manifest["no_reasoning_kept"] == 0


def test_a_build_that_still_sets_require_parameters_keeps_it() -> None:
    """A build override written before strict_params existed stays strict (or lax) as it was."""
    legacy = config()
    legacy["http"].pop("strict_params", None)
    for value in (True, False):
        legacy["http"]["params"]["provider"]["require_parameters"] = value
        t = HTTPTeacher(legacy, backend=object())
        assert t.strict_params is value
        assert t.http["params"]["provider"]["require_parameters"] is value
    legacy["http"]["params"]["provider"].pop("require_parameters")
    unset = HTTPTeacher(legacy, backend=object())
    assert unset.strict_params is True  # the default: OCT's sampling values or refuse
    assert unset.http["params"]["provider"]["require_parameters"] is True
    legacy["http"]["strict_params"] = None  # a blank `strict_params:` is unset, so still strict
    assert HTTPTeacher(legacy, backend=object()).strict_params is True
    clash = config(strict_params=False)
    clash["http"]["params"]["provider"]["require_parameters"] = True
    with pytest.raises(ValueError, match="disagree"):
        HTTPTeacher(clash, backend=object())


def test_run_passes_the_reasoning_flag_to_the_parser(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """run() hands the parser whether the provider sent a separate reasoning field."""
    for sub in ("plain", "reasoning"):
        (tmp_path / sub).mkdir()
    calls: list[bool] = []
    real = glm_teacher.parse_response

    def spy(text: str, *a: Any, had_reasoning: bool = False, **k: Any) -> str | None:
        calls.append(had_reasoning)
        return real(text, *a, had_reasoning=had_reasoning, **k)

    monkeypatch.setattr(glm_teacher, "parse_response", spy)
    _reply(monkeypatch, "Just an answer.")
    _run_http(monkeypatch, tmp_path / "plain", config())
    assert calls and not any(calls)
    calls.clear()  # a fresh directory: the generation checkpoint would replay the first run
    _reply(monkeypatch, "Just an answer.", reasoning="thought")
    _run_http(monkeypatch, tmp_path / "reasoning", config())
    assert calls and all(calls)


def test_a_hand_set_name_that_would_break_the_stage_record_is_refused() -> None:
    for bad in ('A"B', "A$B", "A`B", "A\\B", "A\nB"):
        with pytest.raises(ValueError, match="teacher name"):
            HTTPTeacher(config(name=bad), backend=object())
    assert HTTPTeacher(config(name="Aria Bot"), backend=object()).assistant_name == "Aria Bot"
