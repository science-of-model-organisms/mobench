"""OpenAI Batch API for the deferred-judge phase (``scripts/judge_deferred.py --batch``).

The Batch API judges the same request at 50% of the Standard price with a separate, larger
rate-limit pool, at the cost of turnaround (``completion_window`` 24h, usually much less):
https://developers.openai.com/api/docs/guides/batch. That trade suits Phase B of a sweep
(nothing is waiting on a verdict) and nothing else, so synchronous judging stays on Standard.

The design keeps ONE judging code path. ``judges.judge.Judge.score`` (rubric render, position
swap, parse) runs twice over the same rows:

1. **collect** — with a ``BatchRecorder`` backend, which records the exact request body the
   synchronous ``HttpJudge`` would have sent and returns a placeholder. Verdicts are thrown
   away; only the requests are kept.
2. ``run_batches`` uploads them as JSONL, creates the batch(es), polls, and downloads the
   output and error files.
3. **replay** — with a ``BatchReplay`` backend, which looks each request up by its
   ``custom_id`` and returns the batch's reply. ``Judge.score`` parses it with the same
   parser and writes the same meta a synchronous call would (plus ``judge_service_tier:
   batch``). A request the batch did not answer (error file, ``batch_expired``, a failed
   batch) is RETRIED SYNCHRONOUSLY ON STANDARD through the configured ``HttpJudge``; if that
   fails too the row is a judge error (NaN), exactly as in the synchronous path.

``custom_id`` is a hash of the request body, so it is deterministic across runs: output order
is not guaranteed by the API and is never relied on, a restarted run maps results to the same
requests, and two byte-identical requests share one batch line (and so one reply — they would
be indistinguishable anyway).

RESUME: each chunk's membership (its custom_ids) is written to ``<state_dir>/state.json``
before anything is uploaded, and its input file id and batch id the moment each exists
(``_submit`` names the one residual crash window). A restarted run re-attaches to every
recorded batch and never re-chunks a request a recorded chunk already holds, so a request is
submitted at most once however often the run is restarted. A request a finished batch did
not answer goes to the synchronous Standard retry above, never into a new batch. A run
whose inputs grew submits only the new requests, in new chunks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from judges.backends import HttpJudge, JudgeBackendError, _record_usage, chat_request_body

log = logging.getLogger(__name__)

#: Documented per-batch caps (batch guide, "Rate limits"): 50,000 requests and a 200 MB input
#: file. The byte default sits under the cap to leave room for JSONL framing differences.
MAX_REQUESTS = 50_000
MAX_BYTES = 190_000_000

#: Batch statuses after which nothing more will happen (batch guide, "Check the status").
TERMINAL = frozenset({"completed", "failed", "expired", "cancelled"})

#: Fields that select a SYNCHRONOUS tier and have no meaning in a batch line (the batch is
#: its own tier). Dropped from the batch body, and therefore from the custom_id hash.
_SYNC_ONLY_FIELDS = ("service_tier",)

#: The endpoint every batch line targets — the same one ``HttpJudge`` posts to.
ENDPOINT = "/v1/chat/completions"


@dataclass(frozen=True)
class BatchSettings:
    """The ``judge.batch`` block of an eval config. Every field has the documented default."""

    completion_window: str = "24h"
    max_requests: int = MAX_REQUESTS
    max_bytes: int = MAX_BYTES
    poll_interval_s: float = 30.0
    poll_timeout_s: float = 24 * 3600.0

    @classmethod
    def from_block(cls, block: dict[str, Any] | None) -> BatchSettings:
        block = dict(block or {})
        unknown = sorted(set(block) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(f"unknown judge.batch keys {unknown}")
        s = cls(**block)
        if not 0 < s.max_requests <= MAX_REQUESTS:
            raise ValueError(f"judge.batch.max_requests must be in 1..{MAX_REQUESTS}")
        if s.max_bytes <= 0 or s.poll_interval_s <= 0 or s.poll_timeout_s <= 0:
            raise ValueError("judge.batch sizes and intervals must be positive")
        return s


def batch_body(
    model: str, messages: list[dict], params: dict[str, Any], kw: dict[str, Any]
) -> dict[str, Any]:
    """The request body a batch line carries: ``HttpJudge``'s body minus sync-only fields.

    Built by the same ``chat_request_body`` ``HttpJudge.complete`` posts, so for a block
    with no ``_SYNC_ONLY_FIELDS`` the two are byte-identical.
    """
    body = chat_request_body(model, messages, params, kw)
    for name in _SYNC_ONLY_FIELDS:
        body.pop(name, None)
    return body


def custom_id(body: dict[str, Any]) -> str:
    """Deterministic id for one request: a hash of its canonical JSON body."""
    canon = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return "judge-" + hashlib.sha256(canon.encode()).hexdigest()[:40]


def jsonl_line(cid: str, body: dict[str, Any]) -> str:
    """One batch input line (batch guide, "Prepare your batch file")."""
    return json.dumps(
        {"custom_id": cid, "method": "POST", "url": ENDPOINT, "body": body},
        ensure_ascii=False,
        sort_keys=True,
    )


def chunk_requests(
    requests: dict[str, dict[str, Any]], max_requests: int, max_bytes: int
) -> list[list[str]]:
    """Split custom_ids into batches under both caps, in sorted (deterministic) order."""
    chunks: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for cid in sorted(requests):
        size = len(jsonl_line(cid, requests[cid]).encode()) + 1
        if size > max_bytes:
            raise ValueError(f"request {cid} alone is {size} bytes, over max_bytes={max_bytes}")
        if cur and (len(cur) >= max_requests or cur_bytes + size > max_bytes):
            chunks.append(cur)
            cur, cur_bytes = [], 0
        cur.append(cid)
        cur_bytes += size
    if cur:
        chunks.append(cur)
    return chunks


#: Layout of ``<state_dir>/state.json``; v2 records each chunk's ``custom_ids``.
STATE_VERSION = 2


def chunk_key(cids: list[str]) -> str:
    return hashlib.sha256("\n".join(cids).encode()).hexdigest()[:24]


# --------------------------------------------------------------------------- backends


def _judge_params(block: dict[str, Any]) -> dict[str, Any]:
    return dict(block.get("params") or {})


class BatchRecorder:
    """Collect pass: record the body ``HttpJudge`` would send; answer with a placeholder."""

    #: A reply every parser accepts without an "empty reply" discard. Never stored: the
    #: collect pass's verdicts are thrown away.
    PLACEHOLDER = "<reasoning>batch collect pass</reasoning>\n<answer>ambiguous</answer>"

    def __init__(self, block: dict[str, Any]) -> None:
        self.model = str(block["model"])
        self.params = _judge_params(block)
        self.name = f"http:{self.model}"
        self.requests: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def complete(self, messages: list[dict], **kw: Any) -> str:
        body = batch_body(self.model, messages, self.params, kw)
        with self._lock:
            self.requests[custom_id(body)] = body
        return self.PLACEHOLDER


class BatchReplay:
    """Replay pass: the batch's reply for this exact request, else a synchronous retry.

    ``results`` maps custom_id -> the output line's ``response.body`` (a chat completion).
    A request absent from it (failed, expired, never submitted) goes to ``fallback`` — the
    configured synchronous ``HttpJudge`` on Standard — and its meta is reported as such.
    """

    def __init__(
        self,
        block: dict[str, Any],
        results: dict[str, dict[str, Any]],
        fallback: HttpJudge | None = None,
    ) -> None:
        self.model = str(block["model"])
        self.params = _judge_params(block)
        self.name = f"http:{self.model}"
        self.results = results
        self.fallback = fallback
        self.n_fallback = 0
        self._lock = threading.Lock()
        self._last = threading.local()

    def complete(self, messages: list[dict], **kw: Any) -> str:
        cid = custom_id(batch_body(self.model, messages, self.params, kw))
        payload = self.results.get(cid)
        if payload is None:
            with self._lock:
                self.n_fallback += 1
            self._last.source = "fallback"
            if self.fallback is None:
                raise JudgeBackendError(f"batch returned no reply for {cid} and no fallback")
            return self.fallback.complete(messages, **kw)
        self._last.source = "batch"
        self._last.payload = payload
        _record_usage(self.model, payload.get("usage"))
        return payload["choices"][0]["message"]["content"] or ""

    def last_response_meta(self) -> dict[str, Any]:
        if getattr(self._last, "source", None) == "fallback" and self.fallback is not None:
            return self.fallback.last_response_meta()
        payload = getattr(self._last, "payload", None) or {}
        return {"provider": None, "response_id": payload.get("id"), "model": payload.get("model")}

    def judge_config_meta(self) -> dict[str, Any]:
        if getattr(self._last, "source", None) == "fallback" and self.fallback is not None:
            return self.fallback.judge_config_meta()
        return {
            "judge_model": self.model,
            "judge_reasoning_effort": self.params.get("reasoning_effort"),
            "judge_service_tier": "batch",
        }


# --------------------------------------------------------------------------- submit / poll


class BatchPending(RuntimeError):
    """Polling hit its time bound with batches still running. Re-run to resume."""


@dataclass
class BatchOutcome:
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: custom_id -> error code (``batch_expired``, an HTTP status, ``batch_failed`` ...)
    failures: dict[str, str] = field(default_factory=dict)
    batch_ids: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


#: The only host with a Batch API among the judge endpoints we use. OpenRouter (the
#: shared judge's default) has none, so a batch run against it must fail, not fall back.
OPENAI_HOST = "api.openai.com"


class BatchUnsupported(ValueError):
    """The judge block cannot be batched (not OpenAI-direct, fake, or a judge panel)."""


def resolve_base_url(block: dict[str, Any]) -> str:
    """The API root this block's ``HttpJudge`` would post to (same fallback chain)."""
    return (
        block.get("base_url") or os.environ.get("JUDGE_BASE_URL") or "https://api.openai.com/v1"
    ).rstrip("/")


def _is_openai_direct(base_url: str) -> bool:
    """``https://api.openai.com[:443]/...`` and nothing that merely parses to that host.

    ``urlparse`` and the HTTP clients disagree on some inputs: a backslash or a userinfo
    ``@`` (``https://evil.com\\@api.openai.com/v1``) can yield ``hostname ==
    api.openai.com`` here while a client connects elsewhere, so both are refused outright,
    as are whitespace, a non-https scheme and a non-default port. The hostname compare is
    case-insensitive (``urlparse`` lower-cases it).
    """
    if any(ch in base_url for ch in "\\@") or any(ch.isspace() for ch in base_url):
        return False
    parsed = urlparse(base_url)
    try:
        port = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme.lower() == "https" and parsed.hostname == OPENAI_HOST and port in (None, 443)
    )


def require_openai_direct(block: Any) -> dict[str, Any]:
    """Refuse any judge block the OpenAI Batch API cannot serve. Returns the block.

    Loud by design: ``--batch`` never silently degrades to synchronous judging.
    """
    if not (isinstance(block, dict) and block.get("backend") == "http"):
        raise BatchUnsupported(
            "--batch needs a real `backend: http` judge on the OpenAI API; the configured "
            f"judge is {block!r} (fake mode has no Batch API)"
        )
    base_url = resolve_base_url(block)
    if not _is_openai_direct(base_url):
        raise BatchUnsupported(
            f"--batch needs an OpenAI-direct judge (base_url https://{OPENAI_HOST}/..., no "
            "userinfo, backslash or non-default port); the "
            f"configured judge {block.get('model')!r} posts to {base_url}, which has no "
            "OpenAI Batch API (OpenRouter has none). Judge synchronously (drop --batch) or "
            "point the judge block at https://api.openai.com/v1."
        )
    if block.get("secondary") is not None:
        raise BatchUnsupported(
            "--batch replaces only the primary judge's transport; a judge panel "
            "(`judge.secondary`) would still call its secondary synchronously, including "
            "during the collect pass. Remove `secondary` or drop --batch."
        )
    return block


def make_client(block: dict[str, Any]) -> Any:
    """An ``openai.OpenAI`` client for the judge block's endpoint and key."""
    from openai import OpenAI

    require_openai_direct(block)
    key_env = str(block.get("api_key_env", "OPENAI_API_KEY"))
    key = os.environ.get(key_env)
    if not key:
        raise JudgeBackendError(f"${key_env} is not set; the batch judge needs it")
    return OpenAI(api_key=key, base_url=resolve_base_url(block))


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": STATE_VERSION, "chunks": {}}
    state = json.loads(path.read_text())
    if state.get("version") != STATE_VERSION or any(
        "custom_ids" not in e for e in state.get("chunks", {}).values()
    ):
        # The v1 layout keyed chunks by a hash of a RE-CHUNKED remainder (the double-billing
        # bug v2 fixes) and cannot say which requests each of its batches holds.
        raise ValueError(
            f"{path} predates batch state v{STATE_VERSION} (no recorded chunk membership); "
            "check its batch ids on the OpenAI dashboard, then move it aside to start over"
        )
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _file_text(client: Any, file_id: str) -> str:
    return client.files.content(file_id).text


def parse_output_lines(text: str, outcome: BatchOutcome) -> None:
    """Fold one output or error file into ``outcome``, keyed by custom_id (order ignored)."""
    for line in text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        cid = rec.get("custom_id")
        if cid is None:
            continue
        resp = rec.get("response") or {}
        err = rec.get("error")
        if resp.get("status_code") == 200 and not err:
            body = resp.get("body") or {}
            outcome.results[cid] = body
            outcome.failures.pop(cid, None)
            for k, v in (body.get("usage") or {}).items():
                if isinstance(v, int):
                    outcome.usage[k] = outcome.usage.get(k, 0) + v
        elif cid not in outcome.results:
            code = (err or {}).get("code") or f"http_{resp.get('status_code')}"
            outcome.failures[cid] = str(code)


#: How many of the account's most recent batches ``_find_batch`` scans when re-attaching a
#: batch whose ``create`` call never returned. The API lists batches newest first.
ADOPT_SCAN_LIMIT = 1000


def _find_batch(client: Any, input_file_id: str) -> Any | None:
    """The batch already created from ``input_file_id``, if any (newest-first scan)."""
    for i, batch in enumerate(client.batches.list(limit=100)):
        if i >= ADOPT_SCAN_LIMIT:
            break
        if getattr(batch, "input_file_id", None) == input_file_id:
            return batch
    return None


def _submit(
    client: Any, state: dict[str, Any], state_path: Path, key: str, settings: BatchSettings
) -> None:
    """Create (or re-attach) the batch for recorded chunk ``key``; persist its id at once.

    Order, and what a crash at each point leaves behind:

    1. Upload the input file, then record ``input_file_id``. A crash before that write
       costs an orphan file (storage, not tokens) and a re-upload on the re-run.
    2. ``batches.create``, then record ``batch_id`` in the very next statement. A crash or
       a dropped connection between the server accepting the batch and that write leaves an
       entry with an ``input_file_id`` and no ``batch_id``; the re-run looks the batch up by
       that file id (``_find_batch``) and re-attaches before creating anything.

    RESIDUAL WINDOW: that lookup scans only the newest ``ADOPT_SCAN_LIMIT`` batches on the
    account. If the lost batch is not among them (that many batches created since, or the
    list lagging the create), the re-run creates a second batch from the same file and both
    are billed. The API has no idempotency key for ``batches.create``, so nothing narrower
    is possible client-side.
    """
    entry = state["chunks"][key]
    if not entry["input_file_id"]:
        with (state_path.parent / f"input-{key}.jsonl").open("rb") as fh:
            entry["input_file_id"] = client.files.create(file=fh, purpose="batch").id
        _save_state(state_path, state)
    else:
        found = _find_batch(client, entry["input_file_id"])
        if found is not None:
            entry["batch_id"], entry["status"] = found.id, found.status
            _save_state(state_path, state)
            log.info("batch chunk %s: re-attached %s by its input file", key, found.id)
            return
    batch = client.batches.create(
        input_file_id=entry["input_file_id"],
        endpoint=ENDPOINT,
        completion_window=settings.completion_window,
        metadata={"source": "MO_evals judge_deferred", "chunk": key},
    )
    entry["batch_id"], entry["status"] = batch.id, batch.status
    _save_state(state_path, state)
    log.info("batch chunk %s: submitted %s (%d requests)", key, batch.id, entry["n_requests"])


def run_batches(
    requests: dict[str, dict[str, Any]],
    state_dir: Path,
    client: Any,
    settings: BatchSettings,
    *,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> BatchOutcome:
    """Submit (or resume) batches for ``requests``, poll to completion, collect replies.

    Chunk membership is FROZEN in the state file when a chunk is first recorded: a re-run
    re-attaches to every recorded chunk's batch and chunks only the requests no recorded
    chunk holds. A request a finished batch did not answer (``batch_expired``, a per-line
    error, a ``failed`` or ``cancelled`` batch) is never put in a new batch; it is returned
    in ``failures`` for ``BatchReplay``'s synchronous retry on Standard.

    Raises ``BatchPending`` if ``settings.poll_timeout_s`` elapses first; the state file
    already records every batch id, so re-running resumes without resubmitting.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = _load_state(state_path)
    chunks: dict[str, dict[str, Any]] = state["chunks"]
    outcome = BatchOutcome()

    # Replies already on disk from an earlier run are reused, never re-requested.
    for out in sorted(state_dir.glob("*.output.jsonl")) + sorted(state_dir.glob("*.errors.jsonl")):
        parse_output_lines(out.read_text(), outcome)

    # A request a recorded chunk holds stays in that chunk, answered or not; only the rest
    # are chunked. Re-chunking the unanswered remainder instead shifts chunk boundaries
    # after a partial expiry, so a still-running batch got a new key and was billed twice.
    assigned = {cid for entry in chunks.values() for cid in entry["custom_ids"]}
    new = {c: b for c, b in requests.items() if c not in assigned and c not in outcome.results}
    for cids in chunk_requests(new, settings.max_requests, settings.max_bytes):
        key = chunk_key(cids)
        (state_dir / f"input-{key}.jsonl").write_text(
            "".join(jsonl_line(c, new[c]) + "\n" for c in cids)
        )
        chunks[key] = {
            "custom_ids": cids,
            "n_requests": len(cids),
            "input_file_id": None,
            "batch_id": None,
            "status": "unsubmitted",
            "collected": False,
        }
    _save_state(state_path, state)  # membership frozen BEFORE anything is uploaded

    wanted = set(requests)
    active = sorted(k for k, e in chunks.items() if wanted.intersection(e["custom_ids"]))
    for key in active:
        if chunks[key]["batch_id"] is None:
            _submit(client, state, state_path, key, settings)
        elif not chunks[key]["collected"]:
            log.info("batch chunk %s: resuming %s", key, chunks[key]["batch_id"])

    deadline = clock() + settings.poll_timeout_s
    pending = [k for k in active if not chunks[k]["collected"]]
    while pending:
        still: list[str] = []
        for key in pending:
            entry = chunks[key]
            batch = client.batches.retrieve(entry["batch_id"])
            entry["status"] = batch.status
            if batch.status not in TERMINAL:
                still.append(key)
                continue
            for attr, suffix in (("output_file_id", "output"), ("error_file_id", "errors")):
                file_id = getattr(batch, attr, None)
                if file_id:
                    text = _file_text(client, file_id)
                    (state_dir / f"{entry['batch_id']}.{suffix}.jsonl").write_text(text)
                    parse_output_lines(text, outcome)
            entry["collected"] = True
            log.info("batch %s: %s", entry["batch_id"], batch.status)
        _save_state(state_path, state)
        pending = still
        if not pending:
            break
        if clock() >= deadline:
            raise BatchPending(
                f"{len(pending)} batch(es) still running after {settings.poll_timeout_s:.0f}s: "
                f"{[chunks[k]['batch_id'] for k in pending]}; re-run the same command "
                f"to resume (state in {state_path})"
            )
        sleep(settings.poll_interval_s)

    outcome.batch_ids = [chunks[k]["batch_id"] for k in active]
    outcome.failures = {c: code for c, code in outcome.failures.items() if c in wanted}
    # Anything unanswered (a `failed` batch writes no per-line file; an `expired` one lists
    # its lines as `batch_expired`) is a failure for the synchronous retry, never a new batch.
    for cid in requests:
        if cid not in outcome.results:
            outcome.failures.setdefault(cid, "no_reply")
    return outcome
