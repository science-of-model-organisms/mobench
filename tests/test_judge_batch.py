"""The OpenAI Batch transport for the deferred-judge phase (``judge_deferred.py --batch``).

The SHARED judge is not an OpenAI-direct model (it is on OpenRouter, which has no Batch
API); batch is an opt-in transport for an OpenAI-direct judge block, so every block here is
a local test fixture, never read from ``configs/``.

All offline: the OpenAI client is a scripted fake (``_FakeOpenAI``), and the "judge" that
answers batch lines is ``FakeJudge`` itself, so a batch-judged cell can be compared with a
synchronously judged one row for row (CLAUDE.md). No test here opens a socket.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from judges import batch as jb
from judges.backends import FakeJudge, HttpJudge
from judges.judge import Judge
from judges.personas import PersonaPair

#: An OpenAI-direct judge block (fixture only). `gpt-6-*` is the name family the pinned
#: inspect mis-classifies, which is what the `_grader_model` tests below exercise.
BLOCK: dict[str, Any] = {
    "backend": "http",
    "model": "gpt-6-luna",
    "base_url": "https://api.openai.com/v1",
    "api_key_env": "OPENAI_API_KEY",
    "params": {"reasoning_effort": "none", "service_tier": "default", "max_completion_tokens": 64},
}

#: The same block with no sync-only field, so the batch body must equal the sync body exactly.
BLOCK_NO_TIER: dict[str, Any] = {
    **BLOCK,
    "params": {"reasoning_effort": "none", "max_completion_tokens": 64},
}

#: The shared judge's SHAPE on main: OpenRouter, which has no Batch API.
OPENROUTER_BLOCK: dict[str, Any] = {
    "backend": "http",
    "model": "google/gemma-4-31b-it",
    "base_url": "https://openrouter.ai/api/v1",
    "api_key_env": "OPENROUTER_API_KEY",
    "params": {"max_tokens": 1024},
}


# --------------------------------------------------------------------------- config


def test_batch_settings_parse_and_refuse() -> None:
    s = jb.BatchSettings.from_block({"completion_window": "24h", "poll_interval_s": 60})
    assert s.completion_window == "24h" and s.poll_interval_s == 60
    assert s.max_requests == 50_000 and s.max_bytes <= 200_000_000
    assert jb.BatchSettings.from_block(None) == jb.BatchSettings()
    with pytest.raises(ValueError, match="unknown"):
        jb.BatchSettings.from_block({"completion_windw": "24h"})
    with pytest.raises(ValueError):
        jb.BatchSettings.from_block({"max_requests": 50_001})


def test_require_openai_direct_accepts_only_api_openai_com(monkeypatch) -> None:
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    assert jb.require_openai_direct(BLOCK) is BLOCK
    # No base_url resolves exactly as HttpJudge does: $JUDGE_BASE_URL, then OpenAI.
    bare = {k: v for k, v in BLOCK.items() if k != "base_url"}
    assert jb.require_openai_direct(bare) is bare
    monkeypatch.setenv("JUDGE_BASE_URL", "https://openrouter.ai/api/v1")
    with pytest.raises(jb.BatchUnsupported, match="OpenAI-direct"):
        jb.require_openai_direct(bare)
    monkeypatch.delenv("JUDGE_BASE_URL")
    for bad in (
        OPENROUTER_BLOCK,
        {**BLOCK, "base_url": "https://api.openai.com.evil.example/v1"},
        {**BLOCK, "base_url": "http://127.0.0.1:8000/v1"},
    ):
        with pytest.raises(jb.BatchUnsupported, match="OpenAI-direct"):
            jb.require_openai_direct(bad)
    for bad in (None, {"backend": "fake"}):
        with pytest.raises(jb.BatchUnsupported, match="backend: http"):
            jb.require_openai_direct(bad)
    with pytest.raises(jb.BatchUnsupported, match="secondary"):
        jb.require_openai_direct({**BLOCK, "secondary": dict(BLOCK)})
    # Review item 2's URL cases: userinfo / backslash tricks and plain http are refused,
    # the real endpoint is accepted in any letter case.
    for bad_url in (
        "https://evil.com\\@api.openai.com/v1",
        "https://evil.com@api.openai.com/v1",
        "https://api.openai.com@evil.com/v1",
        "https://evil.com#@api.openai.com/v1",
        "https://api.openai.com./v1",
        "https://api.openai.com.evil.com/v1",
        "https://api.openai.com%2eevil.com/v1",
        "http://api.openai.com/v1",
        "HTTP://api.openai.com/v1",
        "api.openai.com/v1",
        "https://api.openai.com:8443/v1",
        " https://api.openai.com/v1",
    ):
        with pytest.raises(jb.BatchUnsupported, match="OpenAI-direct"):
            jb.require_openai_direct({**BLOCK, "base_url": bad_url})
    for ok_url in (
        "https://api.openai.com/v1",
        "https://API.OPENAI.COM/v1",
        "HTTPS://Api.OpenAI.com/v1/",
        "https://api.openai.com:443/v1",
    ):
        assert jb.require_openai_direct({**BLOCK, "base_url": ok_url})
    # make_client refuses before it ever builds a client.
    with pytest.raises(jb.BatchUnsupported):
        jb.make_client(OPENROUTER_BLOCK)


# --------------------------------------------------------------------------- meta


def _pair() -> PersonaPair:
    return PersonaPair(id="p", implanted="Is impulsive.", original="Is measured.")


def _score(judge: Judge, key: str = "k", question: str = "What now?") -> Any:
    return judge.score(question=question, response="Act at once!", pair=_pair(), sample_key=key)


def test_fake_judge_rows_get_no_new_meta() -> None:
    verdict = _score(Judge(backend=FakeJudge()))
    assert not {"judge_model", "judge_reasoning_effort", "judge_service_tier"} & set(verdict.extra)


def test_http_judge_records_model_effort_and_served_tier(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = HttpJudge(**{k: v for k, v in BLOCK.items() if k != "backend"})
    sent: list[dict] = []

    def post(url: str, json: dict, headers: dict, timeout: float) -> Any:  # noqa: A002
        sent.append(json)
        payload = {
            "id": "chatcmpl-1",
            "model": "gpt-6-luna",
            "service_tier": "default",
            "choices": [{"message": {"content": "<answer>phenomenon_1</answer>"}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3},
        }
        return SimpleNamespace(status_code=200, json=lambda: payload, headers={})

    monkeypatch.setattr(backend._session, "post", post)
    verdict = _score(Judge(backend=backend))
    assert sent[0]["reasoning_effort"] == "none"
    assert sent[0]["service_tier"] == "default"
    assert sent[0]["max_completion_tokens"] == 64 and "max_tokens" not in sent[0]
    assert "temperature" not in sent[0]
    assert verdict.extra["judge_model"] == "gpt-6-luna"
    assert verdict.extra["judge_reasoning_effort"] == "none"
    assert verdict.extra["judge_service_tier"] == "default"


def test_inspect_grader_sends_openai_wire_fields_verbatim(monkeypatch) -> None:
    """The pinned inspect deletes `reasoning_effort` and sends `max_tokens` for gpt-6-*."""
    pytest.importorskip("inspect_ai")
    from evals._inspect import _grader_model, _GraderTransport

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = _grader_model(BLOCK, 8)
    assert str(model) == "openai/gpt-6-luna"
    assert model.config.extra_body == {"reasoning_effort": "none", "max_completion_tokens": 64}
    assert model.config.reasoning_effort is None and model.config.max_tokens is None
    assert model.api.service_tier == "default"
    meta = _GraderTransport(model, 8).judge_config_meta()
    assert meta == {
        "judge_model": "gpt-6-luna",
        "judge_reasoning_effort": "none",
        "judge_service_tier": "default",
    }


def test_inspect_grader_request_body_on_the_wire(monkeypatch) -> None:
    """What inspect actually builds for the chat-completions request."""
    pytest.importorskip("inspect_ai")
    from inspect_ai.model._providers.openai_completions import completion_params_completions

    from evals._inspect import _grader_model

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    model = _grader_model(BLOCK, 8)
    params = completion_params_completions(model.api, model.config, False)
    assert params["extra_body"] == {"reasoning_effort": "none", "max_completion_tokens": 64}
    assert params["service_tier"] == "default"
    assert "max_tokens" not in params and "reasoning_effort" not in params
    assert "temperature" not in params


# --------------------------------------------------------------------------- JSONL


def _body(text: str, **kw: Any) -> dict:
    return jb.batch_body("gpt-6-luna", [{"role": "user", "content": text}], BLOCK["params"], kw)


def test_batch_body_drops_service_tier_and_ids_are_deterministic() -> None:
    body = _body("hello")
    assert "service_tier" not in body
    assert body["reasoning_effort"] == "none" and body["max_completion_tokens"] == 64
    assert jb.custom_id(body) == jb.custom_id(json.loads(json.dumps(body)))
    assert jb.custom_id(body) != jb.custom_id(_body("hello!"))
    line = json.loads(jb.jsonl_line(jb.custom_id(body), body))
    assert line == {
        "custom_id": jb.custom_id(body),
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body,
    }


def test_chunking_respects_count_and_bytes() -> None:
    reqs = {jb.custom_id(b): b for b in (_body(f"q{i}") for i in range(7))}
    chunks = jb.chunk_requests(reqs, max_requests=3, max_bytes=10**9)
    assert [len(c) for c in chunks] == [3, 3, 1]
    assert sorted(c for ch in chunks for c in ch) == sorted(reqs)
    one = len(jb.jsonl_line(*next(iter(reqs.items()))).encode()) + 1
    by_bytes = jb.chunk_requests(reqs, max_requests=100, max_bytes=2 * one + 5)
    assert all(len(c) <= 2 for c in by_bytes) and sum(map(len, by_bytes)) == 7
    with pytest.raises(ValueError, match="alone"):
        jb.chunk_requests(reqs, max_requests=100, max_bytes=10)


# --------------------------------------------------------------------------- fake client


class _FakeOpenAI:
    """Scripted Files + Batches API. Output order is SHUFFLED; `expire` ids go to the
    error file as `batch_expired`; `polls_before_done` retrieves report in_progress."""

    def __init__(
        self, *, expire: frozenset[str] = frozenset(), polls_before_done: int = 1
    ) -> None:
        self.expire = set(expire)
        self.polls_before_done = polls_before_done
        self.uploads: dict[str, str] = {}
        self.created: list[str] = []
        self.polls: dict[str, int] = {}
        self._files: dict[str, str] = {}
        self._batch_input: dict[str, str] = {}
        self.files = SimpleNamespace(create=self._upload, content=self._content)
        self.batches = SimpleNamespace(
            create=self._create, retrieve=self._retrieve, list=self._list
        )

    def _upload(self, file: Any, purpose: str) -> Any:
        assert purpose == "batch"
        fid = f"file-in-{len(self.uploads)}"
        self.uploads[fid] = file.read().decode()
        return SimpleNamespace(id=fid)

    def _create(self, **kw: Any) -> Any:
        assert kw["endpoint"] == "/v1/chat/completions" and kw["completion_window"] == "24h"
        bid = f"batch-{len(self.created)}"
        self.created.append(bid)
        self._batch_input[bid] = kw["input_file_id"]
        return SimpleNamespace(id=bid, status="validating", input_file_id=kw["input_file_id"])

    def _list(self, **kw: Any) -> Any:
        """Newest first, like the API (auto-paginating iterable)."""
        return [
            SimpleNamespace(id=bid, input_file_id=self._batch_input[bid], status="validating")
            for bid in reversed(self.created)
        ]

    def _retrieve(self, bid: str) -> Any:
        self.polls[bid] = self.polls.get(bid, 0) + 1
        if self.polls[bid] <= self.polls_before_done:
            return SimpleNamespace(
                id=bid, status="in_progress", output_file_id=None, error_file_id=None
            )
        out, err = [], []
        fake = FakeJudge()
        for line in self.uploads[self._batch_input[bid]].splitlines():
            req = json.loads(line)
            cid = req["custom_id"]
            if cid in self.expire:
                err.append(
                    {
                        "id": "r",
                        "custom_id": cid,
                        "response": None,
                        "error": {"code": "batch_expired", "message": "expired"},
                    }
                )
                continue
            body = {
                "id": f"chatcmpl-{cid[-6:]}",
                "model": "gpt-6-luna",
                "service_tier": "batch",
                "choices": [{"message": {"content": fake.complete(req["body"]["messages"])}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            }
            out.append(
                {
                    "id": "r",
                    "custom_id": cid,
                    "response": {"status_code": 200, "body": body},
                    "error": None,
                }
            )
        random.Random(0).shuffle(out)  # output order is NOT input order
        self._files[f"{bid}-out"] = "".join(json.dumps(o) + "\n" for o in out)
        self._files[f"{bid}-err"] = "".join(json.dumps(e) + "\n" for e in err)
        status = "expired" if err else "completed"
        return SimpleNamespace(
            id=bid,
            status=status,
            output_file_id=f"{bid}-out",
            error_file_id=f"{bid}-err" if err else None,
        )

    def _content(self, fid: str) -> Any:
        return SimpleNamespace(text=self._files[fid])


_FAST = jb.BatchSettings(poll_interval_s=1, poll_timeout_s=100)


def _requests(n: int) -> dict[str, dict]:
    return {jb.custom_id(b): b for b in (_body(f"question {i}") for i in range(n))}


def test_run_batches_maps_shuffled_output_by_custom_id(tmp_path: Path) -> None:
    reqs = _requests(6)
    client = _FakeOpenAI()
    out = jb.run_batches(reqs, tmp_path, client, _FAST, sleep=lambda s: None)
    assert set(out.results) == set(reqs) and not out.failures
    fake = FakeJudge()
    for cid, body in reqs.items():
        assert out.results[cid]["choices"][0]["message"]["content"] == fake.complete(
            body["messages"]
        )
    assert out.usage["prompt_tokens"] == 60
    # The uploaded JSONL is exactly what the builder produced.
    lines = [json.loads(x) for x in next(iter(client.uploads.values())).splitlines()]
    assert {x["custom_id"] for x in lines} == set(reqs)
    assert all(x["body"] == reqs[x["custom_id"]] for x in lines)


def test_error_file_marks_expired_requests_failed(tmp_path: Path) -> None:
    reqs = _requests(4)
    gone = set(sorted(reqs)[:2])
    out = jb.run_batches(reqs, tmp_path, _FakeOpenAI(expire=gone), _FAST, sleep=lambda s: None)
    assert set(out.results) == set(reqs) - gone
    assert out.failures == {c: "batch_expired" for c in gone}


def test_split_over_max_requests(tmp_path: Path) -> None:
    reqs = _requests(5)
    client = _FakeOpenAI()
    settings = jb.BatchSettings(max_requests=2, poll_interval_s=1, poll_timeout_s=100)
    out = jb.run_batches(reqs, tmp_path, client, settings, sleep=lambda s: None)
    assert len(client.created) == 3 and len(out.batch_ids) == 3
    assert set(out.results) == set(reqs)


def test_poll_timeout_persists_and_resume_does_not_resubmit(tmp_path: Path) -> None:
    reqs = _requests(3)
    slow = _FakeOpenAI(polls_before_done=10**6)
    clock = iter(range(0, 10**6, 60))
    with pytest.raises(jb.BatchPending, match="re-run"):
        jb.run_batches(
            reqs,
            tmp_path,
            slow,
            jb.BatchSettings(poll_interval_s=1, poll_timeout_s=100),
            sleep=lambda s: None,
            clock=lambda: next(clock),
        )
    state = json.loads((tmp_path / "state.json").read_text())
    [entry] = state["chunks"].values()
    assert entry["batch_id"] == "batch-0" and entry["n_requests"] == 3

    # A restarted run re-attaches to batch-0: no upload, no create.
    slow.polls_before_done = 0
    slow.uploads_before = dict(slow.uploads)
    out = jb.run_batches(reqs, tmp_path, slow, _FAST, sleep=lambda s: None)
    assert slow.created == ["batch-0"], "resumed run must not create a second batch"
    assert slow.uploads == slow.uploads_before
    assert set(out.results) == set(reqs)

    # And once the output is on disk, a third run needs no API call at all.
    class _NoCalls:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"unexpected API call: {name}")

    again = jb.run_batches(reqs, tmp_path, _NoCalls(), _FAST, sleep=lambda s: None)
    assert set(again.results) == set(reqs)


class _SplitOpenAI(_FakeOpenAI):
    """The batch holding an ``expire`` line finishes (``expired``) on its first poll; every
    other batch reports in_progress until ``slow`` is cleared. The review's item 6."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(polls_before_done=0, **kw)
        self.slow = True

    def _holds_expired(self, bid: str) -> bool:
        lines = self.uploads[self._batch_input[bid]].splitlines()
        return any(json.loads(x)["custom_id"] in self.expire for x in lines)

    def _retrieve(self, bid: str) -> Any:
        if self.slow and not self._holds_expired(bid):
            return SimpleNamespace(
                id=bid, status="in_progress", output_file_id=None, error_file_id=None
            )
        return super()._retrieve(bid)


def _submissions(client: _FakeOpenAI) -> dict[str, int]:
    """custom_id -> how many CREATED batches carried it (what OpenAI would bill)."""
    counts: dict[str, int] = {}
    for bid in client.created:
        for line in client.uploads[client._batch_input[bid]].splitlines():
            cid = json.loads(line)["custom_id"]
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def test_resume_after_partial_expiry_never_resubmits(tmp_path: Path) -> None:
    """Review item 6: chunk A finished with a `batch_expired` line, chunk B still running.

    Re-chunking the unanswered remainder on resume shifted the chunk boundaries, so B got a
    new key and was submitted (and billed) again, and its output was never collected. Chunk
    membership is now frozen in the state file; the expired line goes to the sync fallback.
    """
    recorder = jb.BatchRecorder(BLOCK)
    keys = [f"s{i}" for i in range(6)]
    for k in keys:
        _score(Judge(backend=recorder), key=k, question=f"Q {k}?")
    reqs = recorder.requests
    assert len(reqs) == 6
    gone = sorted(reqs)[0]
    client = _SplitOpenAI(expire=frozenset({gone}))
    settings = jb.BatchSettings(max_requests=3, poll_interval_s=1, poll_timeout_s=100)
    clock = iter(range(0, 10**6, 60))
    with pytest.raises(jb.BatchPending):
        jb.run_batches(
            reqs, tmp_path, client, settings, sleep=lambda s: None, clock=lambda: next(clock)
        )
    assert client.created == ["batch-0", "batch-1"]
    [running] = [b for b in client.created if not client._holds_expired(b)]

    client.slow = False
    out = jb.run_batches(reqs, tmp_path, client, settings, sleep=lambda s: None)
    assert client.created == ["batch-0", "batch-1"], "resume must not create a batch"
    submitted = _submissions(client)
    assert set(submitted) == set(reqs) and set(submitted.values()) == {1}
    # The batch that was still running at BatchPending was collected: every line answered.
    lines = client.uploads[client._batch_input[running]].splitlines()
    assert {json.loads(x)["custom_id"] for x in lines} <= set(out.results)
    assert set(out.results) == set(reqs) - {gone}
    assert out.failures == {gone: "batch_expired"}
    assert out.batch_ids == ["batch-0", "batch-1"]

    # Replay: every row gets exactly one verdict; only the expired line hits Standard.
    sync = _SyncStub()
    replay = jb.BatchReplay(BLOCK, out.results, fallback=sync)
    reference = Judge(backend=FakeJudge())
    tiers = []
    for k in keys:
        got = _score(Judge(backend=replay), key=k, question=f"Q {k}?")
        want = _score(reference, key=k, question=f"Q {k}?")
        assert got.outcome == want.outcome and got.raw == want.raw
        tiers.append(got.extra["judge_service_tier"])
    assert sync.calls == replay.n_fallback == 1
    assert tiers.count("default") == 1 and tiers.count("batch") == 5


def test_failed_batch_goes_to_sync_fallback_not_a_new_batch(tmp_path: Path) -> None:
    """A batch that ends `failed` (no per-line files) is never resubmitted on a re-run."""

    class _Fails(_FakeOpenAI):
        def _retrieve(self, bid: str) -> Any:
            return SimpleNamespace(
                id=bid, status="failed", output_file_id=None, error_file_id=None
            )

    reqs = _requests(3)
    client = _Fails()
    out = jb.run_batches(reqs, tmp_path, client, _FAST, sleep=lambda s: None)
    assert out.failures == {c: "no_reply" for c in reqs} and not out.results
    again = jb.run_batches(reqs, tmp_path, client, _FAST, sleep=lambda s: None)
    assert client.created == ["batch-0"], "a failed batch's lines go to Standard, not a batch"
    assert again.failures == {c: "no_reply" for c in reqs}


def test_new_requests_on_rerun_get_their_own_batch(tmp_path: Path) -> None:
    """Requests not assigned to any recorded batch are chunked and submitted; old ones not."""
    reqs = _requests(4)
    client = _FakeOpenAI()
    jb.run_batches(dict(sorted(reqs.items())[:2]), tmp_path, client, _FAST, sleep=lambda s: None)
    out = jb.run_batches(reqs, tmp_path, client, _FAST, sleep=lambda s: None)
    assert len(client.created) == 2 and set(out.results) == set(reqs)
    assert set(_submissions(client).values()) == {1}


def test_crash_after_create_adopts_the_batch_instead_of_resubmitting(tmp_path: Path) -> None:
    """`batches.create` accepted server-side but the call never returned (network drop).

    The state file records the uploaded input file before `create`; the re-run finds the
    batch by that file id (`batches.list`) and re-attaches instead of creating a second one.
    """

    class _Drops(_FakeOpenAI):
        drop = True

        def _create(self, **kw: Any) -> Any:
            batch = super()._create(**kw)
            if self.drop:
                raise ConnectionError("connection reset after the batch was created")
            return batch

    reqs = _requests(3)
    client = _Drops()
    with pytest.raises(ConnectionError):
        jb.run_batches(reqs, tmp_path, client, _FAST, sleep=lambda s: None)
    [entry] = json.loads((tmp_path / "state.json").read_text())["chunks"].values()
    assert entry["batch_id"] is None and entry["input_file_id"] == "file-in-0"
    client.drop = False
    out = jb.run_batches(reqs, tmp_path, client, _FAST, sleep=lambda s: None)
    assert client.created == ["batch-0"] and len(client.uploads) == 1
    assert set(out.results) == set(reqs)


# --------------------------------------------------------------------------- replay


class _SyncStub(HttpJudge):
    def __init__(self) -> None:
        super().__init__(model="gpt-6-luna", params=dict(BLOCK["params"]))
        self.calls = 0

    def complete(self, messages: list[dict], **kw: Any) -> str:
        self.calls += 1
        self._last_response.service_tier = "default"
        return FakeJudge().complete(messages)


def test_replay_answers_from_batch_and_falls_back_on_standard(tmp_path: Path) -> None:
    recorder = jb.BatchRecorder(BLOCK)
    # Distinct questions: rubric B has no position swap, so identical rows would share one
    # body (and so one custom_id) and the test would not cover a batch/fallback mix.
    keys = [f"s{i}" for i in range(6)]
    for k in keys:
        _score(Judge(backend=recorder), key=k, question=f"What now, {k}?")
    assert len(recorder.requests) == len(keys)
    gone = set(sorted(recorder.requests)[:1])
    out = jb.run_batches(
        recorder.requests, tmp_path, _FakeOpenAI(expire=gone), _FAST, sleep=lambda s: None
    )
    sync = _SyncStub()
    replay = jb.BatchReplay(BLOCK, out.results, fallback=sync)
    reference = Judge(backend=FakeJudge())
    tiers = []
    for k in keys:
        got = _score(Judge(backend=replay), key=k, question=f"What now, {k}?")
        want = _score(reference, key=k, question=f"What now, {k}?")
        assert got.outcome == want.outcome and got.raw == want.raw
        assert got.extra["swapped"] == want.extra["swapped"]
        assert got.extra["judge_model"] == "gpt-6-luna"
        assert got.extra["judge_reasoning_effort"] == "none"
        tiers.append(got.extra["judge_service_tier"])
    assert sync.calls == replay.n_fallback >= 1
    assert tiers.count("default") == sync.calls and "batch" in tiers


def test_batch_judged_cell_equals_sync_judged_cell(tmp_path: Path, monkeypatch) -> None:
    """Phase B through the batch == Phase B synchronously, per sample (same parser, meta)."""
    pytest.importorskip("inspect_ai")
    from scripts.judge_deferred import _rescore
    from tests.test_defer_judge import _cfg, _run

    deferred = _run(
        "generality_fermi_gen", _cfg(tmp_path / "gen"), defer=True, monkeypatch=monkeypatch
    )
    cfg = {**_cfg(tmp_path / "gen"), "judge": BLOCK}
    sync_rows = _rescore(deferred, "generality_fermi_gen", "generality", cfg, FakeJudge())

    recorder = jb.BatchRecorder(BLOCK)
    _rescore(deferred, "generality_fermi_gen", "generality", cfg, recorder)
    out = jb.run_batches(
        recorder.requests, tmp_path / "b", _FakeOpenAI(), _FAST, sleep=lambda s: None
    )
    batch_rows = _rescore(
        deferred, "generality_fermi_gen", "generality", cfg, jb.BatchReplay(BLOCK, out.results)
    )
    by_id = {r.sample_id: r for r in sync_rows}
    assert {r.sample_id for r in batch_rows} == set(by_id)
    for r in batch_rows:
        s = by_id[r.sample_id]
        assert (math.isnan(r.score) and math.isnan(s.score)) or r.score == s.score
        assert (
            r.meta["outcome"] == s.meta["outcome"] and r.meta["judge_raw"] == s.meta["judge_raw"]
        )
        assert r.meta["judge_service_tier"] == "batch" and r.meta["complete"]


# --------------------------------------------------------------------------- sync == batch


def _capture_sync_bodies(monkeypatch, block: dict[str, Any]) -> tuple[HttpJudge, list[dict]]:
    """An HttpJudge built from ``block`` whose POSTs are captured, never sent."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = HttpJudge(**{k: v for k, v in block.items() if k != "backend"})
    sent: list[dict] = []

    def post(url: str, json: dict, headers: dict, timeout: float) -> Any:  # noqa: A002
        sent.append(json)
        payload = {
            "id": "chatcmpl-x",
            "model": block["model"],
            "choices": [{"message": {"content": FakeJudge().complete(json["messages"])}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
        return SimpleNamespace(status_code=200, json=lambda: payload, headers={})

    monkeypatch.setattr(backend._session, "post", post)
    return backend, sent


@pytest.mark.parametrize(
    "judge_kw, rubric",
    [
        # PR 5a's default: an expression rubric judged as rubric B (trait_intensity).
        ({}, "open_ended"),
        # The legacy three-way path, position-swapped per (seed, sample_key), with the
        # structured `response_format` kwarg reaching the body.
        ({"legacy_three_way": True, "structured": True, "seed": 7}, "open_ended"),
        ({"legacy_three_way": True, "seed": 3}, "fermi"),
    ],
)
def test_batch_and_sync_request_bodies_are_byte_identical(
    monkeypatch, judge_kw: dict[str, Any], rubric: str
) -> None:
    """Same row -> same prompt rendering, same swap -> the same body on both transports."""
    backend, sent = _capture_sync_bodies(monkeypatch, BLOCK_NO_TIER)
    recorder = jb.BatchRecorder(BLOCK_NO_TIER)
    swaps = []
    for i in range(8):
        key = f"deadbeef|generality_fermi_gen|s{i}"
        kw: dict[str, Any] = {
            "question": "What now?",
            "response": "Act at once!",
            "pair": _pair(),
            "rubric": rubric,
            "sample_key": key,
        }
        sync_v = Judge(backend=backend, **judge_kw).score(**kw)
        batch_v = Judge(backend=recorder, **judge_kw).score(**kw)
        assert sync_v.extra["swapped"] == batch_v.extra["swapped"]
        swaps.append(sync_v.extra["swapped"])
        body = sent[-1]
        cid = jb.custom_id(body)
        assert cid in recorder.requests
        recorded = recorder.requests[cid]
        # Byte-identical as serialised (same keys, same order, same values) ...
        assert json.dumps(recorded, ensure_ascii=False) == json.dumps(body, ensure_ascii=False)
        # ... and the JSONL line carries that body unchanged.
        assert json.loads(jb.jsonl_line(cid, recorded))["body"] == body
    # custom_id is a hash of the body, so rows whose bodies coincide share one batch line
    # (they would be indistinguishable anyway); every sync body has exactly one line.
    assert set(recorder.requests) == {jb.custom_id(b) for b in sent} and len(sent) == 8
    if judge_kw.get("legacy_three_way"):
        assert len(set(swaps)) == 2, "fixture must exercise both presentation orders"
    else:
        assert not any(swaps)


def test_only_sync_only_fields_differ_between_transports(monkeypatch) -> None:
    """With `service_tier` set, the batch body is the sync body minus exactly that field."""
    backend, sent = _capture_sync_bodies(monkeypatch, BLOCK)
    recorder = jb.BatchRecorder(BLOCK)
    _score(Judge(backend=backend))
    _score(Judge(backend=recorder))
    [recorded] = recorder.requests.values()
    assert {k: v for k, v in sent[-1].items() if k not in jb._SYNC_ONLY_FIELDS} == recorded
    assert sent[-1]["service_tier"] == "default" and "service_tier" not in recorded


# --------------------------------------------------------------------------- CLI refusal


def _no_network(monkeypatch) -> None:
    import requests

    def refuse(*a: Any, **k: Any) -> Any:
        raise AssertionError("network call attempted")

    monkeypatch.setattr(requests.Session, "post", refuse)
    monkeypatch.setattr(jb, "make_client", refuse)


class _Refuse:
    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected API call: {name}")


def _cli_argv(tmp_path: Path) -> list[str]:
    return [
        str(tmp_path / "gen"),
        "--target-root",
        str(tmp_path / "main"),
        "--batch",
        "--config",
        "configs/specs.smoke.yaml",
    ]


def test_cli_batch_refuses_openrouter_judge(tmp_path: Path, monkeypatch, caplog) -> None:
    """The shared judge (OpenRouter) has no Batch API: exit non-zero, write nothing."""
    pytest.importorskip("inspect_ai")
    from scripts.judge_deferred import main

    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    _no_network(monkeypatch)
    with caplog.at_level("ERROR"):
        assert main(_cli_argv(tmp_path), batch_client=_Refuse()) == 2
    assert "OpenAI-direct" in caplog.text and "openrouter.ai" in caplog.text
    assert not (tmp_path / "main").exists()
    # A --judge-model override is an OpenRouter block too.
    override = [*_cli_argv(tmp_path), "--judge-model", "gpt-6-luna"]
    assert main(override, batch_client=_Refuse()) == 2


def test_cli_batch_refuses_in_fake_mode(tmp_path: Path, monkeypatch, caplog) -> None:
    pytest.importorskip("inspect_ai")
    from scripts.judge_deferred import main

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    _no_network(monkeypatch)
    with caplog.at_level("ERROR"):
        assert main(_cli_argv(tmp_path), batch_client=_Refuse()) == 2
    assert "fake mode" in caplog.text
    assert not (tmp_path / "main").exists()


class _ReadRows(Exception):
    pass


@pytest.mark.parametrize(
    "per_eval, refused",
    [
        ({"judge": OPENROUTER_BLOCK}, "OpenAI-direct"),
        ({"judge": {**BLOCK, "secondary": dict(BLOCK)}}, "secondary"),
        ({"judge": BLOCK_NO_TIER}, "one judge block"),  # two OpenAI blocks across cells
        ({}, None),  # control: every cell resolves to BLOCK, so rows are read
    ],
)
def test_cli_batch_refuses_per_eval_blocks_before_reading_rows(
    tmp_path: Path, monkeypatch, caplog, per_eval: dict[str, Any], refused: str | None
) -> None:
    """Review item 2: a per-eval judge override was only checked after `load_all`.

    Every in-scope cell's judge block is now resolved from config and refused (exit 2)
    before a single row is read. The control reaches `load_all`, proving the check passes.
    """
    pytest.importorskip("inspect_ai")
    import scripts.judge_deferred as jd
    from serving.runner import load_config

    cfg = load_config("configs/specs.smoke.yaml")
    cfg["eval_configs"]["default"]["judge"] = dict(BLOCK)
    cfg["eval_configs"]["generality_fermi_gen"] = {
        **(cfg["eval_configs"].get("generality_fermi_gen") or {}),
        **per_eval,
    }
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    monkeypatch.delenv("JUDGE_BASE_URL", raising=False)
    monkeypatch.setattr(jd, "load_config", lambda _p: cfg)

    def read_rows(*a: Any, **k: Any) -> Any:
        raise _ReadRows

    monkeypatch.setattr(jd, "load_all", read_rows)
    _no_network(monkeypatch)
    if refused is None:
        with pytest.raises(_ReadRows):
            jd.main(_cli_argv(tmp_path), batch_client=_Refuse())
        return
    with caplog.at_level("ERROR"):
        assert jd.main(_cli_argv(tmp_path), batch_client=_Refuse()) == 2
    assert refused in caplog.text
    assert not (tmp_path / "main").exists()


def test_judge_block_refuses_mixed_or_non_openai_cells() -> None:
    pytest.importorskip("inspect_ai")
    from scripts.judge_deferred import _judge_block

    def cell(block: dict[str, Any]) -> tuple[Any, ...]:
        return (None, "e", "a", [], {"judge": block})

    assert _judge_block([cell(BLOCK), cell(dict(BLOCK))]) == BLOCK
    with pytest.raises(jb.BatchUnsupported, match="one judge block"):
        _judge_block([cell(BLOCK), cell(BLOCK_NO_TIER)])
    with pytest.raises(jb.BatchUnsupported, match="OpenAI-direct"):
        _judge_block([cell(OPENROUTER_BLOCK)])
