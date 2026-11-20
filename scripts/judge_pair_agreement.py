#!/usr/bin/env python3
"""Judge-vs-judge agreement on stored rows, using the judging scheme live runs use.

Re-judges stored rows with two judges (``judges.judge.Judge.score`` + ``judges.pairwise``):

* **Rubric B** — every judged row is re-judged fresh by EACH judge through ``Judge.score``
  with the row's stored rubric family (an expression family becomes the 0-100
  ``trait_intensity`` read; ``self_report`` etc. keep their own categorical rubric).
* **Gate A** — a deterministic subset of non-base expression rows is re-judged pairwise
  against the same-size, same-seed base organism's stored answer to the same item/arm/turn,
  both orders (``judges.pairwise.judge_pair``).

Both judges see byte-identical prompts: same rendering, same ``Judge`` seed, same
``sample_key``; only the backend block differs.

    uv run python scripts/judge_pair_agreement.py <run_root> --persona impulsive \\
        --n-rows 300 --gate-pairs 20 --max-calls 700 --out DIR [--dry-run] [--limit N]

**AGREEMENT ONLY, NEVER ACCURACY.** Two judges agreeing says nothing about whether either
is right; use judge calibration against human labels for that.

Spend is capped at the TRANSPORT: every HTTP attempt (retries and 429s included) is counted
against ``--max-calls`` across both judges; the plan is refused up front if it cannot fit,
and a call that would exceed the cap is never sent (the row is recorded as budget-stopped,
not scored). Verdicts are cached per (judge, task, input hash) in ``<out>/verdicts.jsonl``,
so a crash or a re-run never re-bills a judged row. ``MO_EVALS_FAKE=1`` swaps both judges
for ``FakeJudge`` (no network).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import statistics
import sys
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import serving.inspect_runner  # noqa: E402,F401  (populates the eval registry + _JUDGED)
from core.results import parse_result_filename, read_result_table, result_files  # noqa: E402
from evals._inspect import get_eval  # noqa: E402
from evals._task_eval import _JUDGED, _PER_TURN_JUDGE_EVALS  # noqa: E402
from judges import pairwise  # noqa: E402
from judges.backends import HttpJudge  # noqa: E402
from judges.judge import Judge  # noqa: E402
from judges.personas import PersonaPair, resolve_pair  # noqa: E402
from judges.rubrics import (  # noqa: E402
    EXPRESSION_RUBRICS,
    JUDGE_ERROR_PREFIX,
    TRACE_RUBRICS,
    TRAIT_INTENSITY,
    render,
    render_trait_intensity,
)
from scripts.pairwise_gate import (  # noqa: E402
    judged_text,
    load_organisms,
    question_map,
    row_key,
)
from serving.runner import eval_config_for, fake_mode, load_config, selection_names  # noqa: E402

log = logging.getLogger(__name__)

# Stored judged evals this path cannot re-judge under the current scheme, with the reason.
EXCLUDED: dict[str, str] = {
    **{
        name: (
            "removed from the eval registry: "
            "no current wiring/bank to recover the judged question, and no "
            "current-scheme judging to reproduce"
        )
        for name in ("generality_format", "generality_paraphrase", "generality_intrusion_ifeval")
    },
    **{
        name: (
            "the current drift judge scores against _drift_judge_transcript (the "
            "trajectory's user turns, evals/_task_persistence.py); these rows carry no "
            "meta['user_turn'], so the judged input cannot be rebuilt from disk"
        )
        for name in ("persistence_multiturn_drift", "persistence_multiturn_drift_long")
    },
}

# A renamed eval whose stored rows keep the old name: read the question from the new wiring.
ALIASES: dict[str, str] = {
    # evals/robustness.py:308 — "Was persistence_system_prompt_conflict until 2026-09-21".
    "persistence_system_prompt_conflict": "robustness_benign_system_prompt_conflict",
}

SELF_PREF_METHOD = "few_shot_icl"
BINS = (20, 40, 60, 80)  # 0-100 score -> 5 ordinal bins for the weighted kappa


def luna_block() -> dict[str, Any]:
    """gpt-6-luna, OpenAI DIRECT, non-reasoning. It rejects ``max_tokens``; the output cap
    matches the gemma block's 1024 so neither judge is truncated where the other is not."""
    return {
        "backend": "http",
        "model": "gpt-6-luna",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "params": {"reasoning_effort": "none", "max_completion_tokens": 1024},
    }


def judge_blocks(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``label -> backend block``: the config's shared judge VERBATIM, and luna."""
    gemma = (config.get("eval_configs") or {}).get("default", {}).get("judge")
    if not isinstance(gemma, dict):
        raise SystemExit("config has no eval_configs.default.judge block")
    if fake_mode():
        return {"gemma": {"backend": "fake"}, "luna": {"backend": "fake"}}
    return {"gemma": dict(gemma), "luna": luna_block()}


# ------------------------------------------------------------------------------ budget


class BudgetExceeded(Exception):
    """Raised INSTEAD of sending an HTTP request once the call cap is reached.

    Deliberately not a ``requests.RequestException``: HttpJudge retries those, which would
    spin on a cap that never lifts. ``complete_or_error`` turns it into a judge error.
    """


class CallBudget:
    """A hard cap on real HTTP attempts across every guarded backend, retries included."""

    def __init__(self, max_calls: int, ledger: Path | None = None) -> None:
        """``ledger`` persists the running count per judge, so the cap spans every process
        that shares an ``--out`` (a smoke run, a crash and its re-run), not just this one."""
        self.max_calls = max_calls
        self.ledger = ledger
        self.calls: dict[str, int] = defaultdict(int)
        if ledger is not None and ledger.exists():
            self.calls.update(json.loads(ledger.read_text()))
        self.at_start = dict(self.calls)
        self.exhausted = False
        self._lock = threading.Lock()
        self._local = threading.local()

    @property
    def total(self) -> int:
        return sum(self.calls.values())

    def _persist(self) -> None:
        # Called under the lock, BEFORE the request is sent: a crash mid-call still counts.
        if self.ledger is not None:
            self.ledger.write_text(json.dumps(dict(self.calls)))

    def start(self) -> None:
        """Reset this thread's per-task telemetry."""
        self._local.http_calls = 0
        self._local.prompt_tokens = 0
        self._local.completion_tokens = 0
        self._local.statuses = []

    def telemetry(self) -> dict[str, Any]:
        return {
            "http_calls": getattr(self._local, "http_calls", 0),
            "prompt_tokens": getattr(self._local, "prompt_tokens", 0),
            "completion_tokens": getattr(self._local, "completion_tokens", 0),
            "statuses": list(getattr(self._local, "statuses", [])),
        }

    def guard(self, backend: Any, label: str) -> None:
        """Count (and cap) every POST the backend's session makes. No-op for non-HTTP."""
        if not isinstance(backend, HttpJudge):
            return
        session = backend._session  # the one transport HttpJudge.complete posts through
        original = session.post

        def post(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                if self.total >= self.max_calls:
                    self.exhausted = True
                    raise BudgetExceeded(f"call cap {self.max_calls} reached")
                self.calls[label] += 1
                self._persist()
            local = self._local
            local.http_calls = getattr(local, "http_calls", 0) + 1
            resp = original(*args, **kwargs)
            local.statuses = [*getattr(local, "statuses", []), resp.status_code]
            if resp.status_code == 200:
                try:
                    usage = resp.json().get("usage") or {}
                except ValueError:
                    usage = {}
                local.prompt_tokens = getattr(local, "prompt_tokens", 0) + int(
                    usage.get("prompt_tokens") or 0
                )
                local.completion_tokens = getattr(local, "completion_tokens", 0) + int(
                    usage.get("completion_tokens") or 0
                )
            return resp

        session.post = post  # instance-level wrap of this backend's session only


# ------------------------------------------------------------------------------ loading


def _meta(value: Any) -> dict[str, Any]:
    return json.loads(value) if isinstance(value, str) else dict(value or {})


def _eval_of(path: Path) -> str:
    cell = parse_result_filename(path)
    if cell is None:
        raise ValueError(f"{path} is not a result file")
    return cell.eval_name


def method_label(spec: Any) -> str:
    """The scorecard's method column: ``prompting`` split by polarity (``prompting+``/``-``),
    since the negative prompt is a control that should score BELOW base."""
    method = str(spec.method)
    if method == "prompting":
        pol = (spec.method_config or {}).get("polarity")
        return "prompting-" if pol == "negative" else "prompting+"
    return method


def size_of(base_model: str) -> str:
    """``Qwen/Qwen2.5-14B-Instruct`` -> ``14B`` (the base-model size stratum)."""
    for part in base_model.replace("/", "-").split("-"):
        if part.upper().endswith("B") and part[:-1].replace(".", "").isdigit():
            return part.upper()
    return base_model


def rejudge_rubric(meta: dict[str, Any]) -> str:
    """The rubric NAME to hand ``Judge.score`` so it sends what the eval sent.

    A row judged under the current scheme stores ``rubric='trait_intensity'`` (the scheme SENT) and
    the family it was chosen by in ``context_rubric``; ``Judge.score`` refuses
    ``trait_intensity`` by name, so the family is the name to pass. Older rows stored the
    family itself in ``rubric``.
    """
    rubric = str(meta["rubric"])
    if rubric == TRAIT_INTENSITY:
        family = meta.get("context_rubric")
        if not family:
            raise ValueError("trait_intensity row with no context_rubric")
        return str(family)
    return rubric


def row_judged_text(eval_name: str, raw_output: str, meta: dict[str, Any]) -> str:
    """The text the eval's SCORE-BEARING judge call read for this row.

    A per-turn debate eval (``_PER_TURN_JUDGE_EVALS``) scores the cell with the FINAL
    organism turn judged ALONE against the item question under the item's own rubric
    (``evals/_task_eval.py``, "THE CELL SCORE IS THE FINAL TURN's open_ended read"); its
    stored ``raw_output`` is that final turn, and ``meta['rubric']``/``context_rubric`` are
    that call's. The whole-transcript verdict is kept only as ``transcript_*``. Every other
    eval is read as ``scripts.pairwise_gate.judged_text`` reads it.
    """
    if eval_name in _PER_TURN_JUDGE_EVALS:
        return raw_output
    return judged_text(eval_name, raw_output, meta)


def is_judged_row(meta: dict[str, Any], raw_output: Any) -> bool:
    """A row the stored run sent to the shared judge, with an answer worth re-judging."""
    return (
        bool(meta.get("rubric"))
        and "outcome" in meta
        and meta.get("judged") is not False
        and meta.get("adversary_error") is None
        and meta.get("context_exhausted") is None
        and bool(str(raw_output or "").strip())
    )


def discover(
    root: Path, persona: str, evals: Sequence[str] | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Every organism of ``persona`` and every judged candidate row under ``root``.

    ``evals`` restricts the scan to those stored eval names (None: every eval on disk).
    Returns ``(organisms, candidates, excluded)``; ``excluded`` maps eval -> reason for each
    judged eval found on disk that this path cannot re-judge.
    """
    orgs = [o for o in load_organisms(root) if o["persona_id"] == persona]
    files: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    for o in orgs:
        for path in result_files(o["dir"], spec_hash=o["spec_hash"], depth=0):
            ev = _eval_of(path)
            if evals is None or ev in evals:
                files[ev].append((o, path))
    candidates: list[dict[str, Any]] = []
    excluded: dict[str, str] = {}
    bad_transcripts: dict[str, int] = defaultdict(int)
    for eval_name in sorted(files):
        # Checked first: an excluded eval's verdicts may not come from the shared judge, so
        # its rows would carry no `rubric` and the probe below would miss that it is judged.
        if eval_name in EXCLUDED:
            excluded[eval_name] = EXCLUDED[eval_name]
            continue
        probe = read_result_table(
            files[eval_name][0][1], columns=["meta", "raw_output"]
        ).to_pandas()
        pairs = zip(probe.meta, probe.raw_output, strict=True)
        if not any(is_judged_row(_meta(m), r) for m, r in pairs):
            continue  # judge-free eval
        wiring = ALIASES.get(eval_name, eval_name)
        try:
            get_eval(wiring)
        except KeyError:
            excluded[eval_name] = f"not in the current eval registry (wiring {wiring!r})"
            continue
        for org, path in files[eval_name]:
            df = read_result_table(path).to_pandas()
            for rec in df.to_dict("records"):
                meta = _meta(rec["meta"])
                if not is_judged_row(meta, rec["raw_output"]):
                    continue
                try:
                    text = row_judged_text(eval_name, str(rec["raw_output"]), meta)
                    rubric = rejudge_rubric(meta)
                except ValueError:  # a stored transcript that is not valid JSON (3 in 0918)
                    bad_transcripts[eval_name] += 1
                    continue
                candidates.append(
                    {
                        "eval": eval_name,
                        "wiring": wiring,
                        "method": org["method"],
                        "method_label": method_label(org["spec"]),
                        "size": size_of(org["base_model"]),
                        "organism": org["organism"],
                        "spec_hash": org["spec_hash"],
                        "base_model": org["base_model"],
                        "persona_id": org["persona_id"],
                        "seed": org["seed"],
                        "sample_id": rec["sample_id"],
                        "key": list(row_key(meta)),
                        "sample_idx": int(meta.get("sample_idx", 0) or 0),
                        "rubric": rubric,
                        "text": text,
                        "raw_output": str(rec["raw_output"]),
                        "stored_scheme": meta["rubric"],
                        "stored_outcome": meta.get("outcome"),
                        "stored_score": rec.get("score"),
                        "stored_trace": meta.get("trace"),
                        "stored_intensity": meta.get("intensity"),
                        "stored_judge": meta.get("judge"),
                        "stored_judge_error": str(meta.get("judge_raw") or "").startswith(
                            JUDGE_ERROR_PREFIX
                        ),
                    }
                )
    for ev, n in sorted(bad_transcripts.items()):
        log.warning("%s: %d rows dropped, stored transcript is not valid JSON", ev, n)
    return orgs, candidates, excluded


# ---------------------------------------------------------------------------- selection


def stable_int(*parts: Any) -> int:
    """A process-independent seed (never ``hash()``)."""
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def select_rows(
    candidates: Sequence[dict[str, Any]],
    n: int,
    seed: int,
    *,
    strata: Callable[[dict[str, Any]], tuple[str, str]] = lambda c: (c["eval"], c["method"]),
) -> list[dict[str, Any]]:
    """``n`` rows stratified by (eval, method), deterministic for a seed.

    Each (eval, method) pool is sorted then shuffled with its own stable seed; each eval's
    queue interleaves its methods; the evals are then taken round-robin. So every eval gets
    an equal share (until it runs dry) and, inside an eval, every method does too.
    """
    pools: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        ev, method = strata(c)
        pools[ev][method].append(c)
    queues: dict[str, list[dict[str, Any]]] = {}
    for ev in sorted(pools):
        shuffled = []
        for method in sorted(pools[ev]):
            pool = sorted(pools[ev][method], key=lambda c: (c["spec_hash"], c["sample_id"]))
            random.Random(stable_int(seed, ev, method)).shuffle(pool)
            shuffled.append(pool)
        queue = []
        for i in range(max(len(p) for p in shuffled)):
            queue.extend(p[i] for p in shuffled if i < len(p))
        queues[ev] = queue
    out: list[dict[str, Any]] = []
    depth = 0
    while len(out) < n and any(depth < len(q) for q in queues.values()):
        for ev in sorted(queues):
            if depth < len(queues[ev]) and len(out) < n:
                out.append(queues[ev][depth])
        depth += 1
    return out


def dedupe_seed_copies(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop a row whose (organism, eval, item/arm/turn, sample_idx, judged text) repeats one
    at a lower seed. Seed replicates at temperature 0 are byte-identical copies
    (docs/evals/reading-results.md#temperature-and-seeds-are-set-by-axis), so sampling both
    would spend two judge calls on one answer."""
    seen: set[tuple] = set()
    out = []
    for c in sorted(candidates, key=lambda c: (c["seed"], c["spec_hash"], c["sample_id"])):
        k = (c["organism"], c["eval"], tuple(c["key"]), c["sample_idx"], c["text"])
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def select_quota(
    candidates: Sequence[dict[str, Any]],
    quotas: dict[str, int],
    seed: int,
) -> list[dict[str, Any]]:
    """``quotas[eval]`` rows per eval, stratified by method label, then size, inside each eval.

    Nested round-robin: each method's queue alternates its sizes, and each eval takes one
    row per method per round, so a quota smaller than the method x size grid still covers
    every method (and rotates sizes) instead of favouring whichever strata sort first. The
    method and size orders are shuffled with stable seeds per round. Each eval is drawn
    independently, so adding an eval never reshuffles another.
    """

    def order(keys: Sequence[str], *parts: Any) -> list[str]:
        ks = sorted(keys)
        random.Random(stable_int(seed, *parts)).shuffle(ks)
        return ks

    out: list[dict[str, Any]] = []
    for ev in sorted(quotas):
        grid: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
        for c in candidates:
            if c["eval"] == ev:
                grid[c.get("method_label", c["method"])][str(c.get("size"))].append(c)
        queues: dict[str, list[dict[str, Any]]] = {}
        for m, by_size in grid.items():
            pools = {}
            for s, pool in by_size.items():
                pool = sorted(pool, key=lambda c: (c["spec_hash"], c["sample_id"]))
                random.Random(stable_int(seed, ev, m, s)).shuffle(pool)
                pools[s] = pool
            queue: list[dict[str, Any]] = []
            for i in range(max(len(p) for p in pools.values())):
                queue += [pools[s][i] for s in order(list(pools), ev, m, i) if i < len(pools[s])]
            queues[m] = queue
        taken: list[dict[str, Any]] = []
        depth = 0
        while len(taken) < quotas[ev] and any(depth < len(q) for q in queues.values()):
            for m in order(list(queues), ev, depth):
                if depth < len(queues[m]) and len(taken) < quotas[ev]:
                    taken.append(queues[m][depth])
            depth += 1
        out += taken
    return out


def base_index(candidates: Sequence[dict[str, Any]]) -> dict[tuple, dict[str, Any]]:
    """``(persona, base_model, seed, eval, key) -> the base row`` with the lowest sample_idx.

    The same pairing rule as ``scripts/pairwise_gate.plan_pairs`` (same persona, base model
    and seed; same item/arm/turn; first sample), built from the already-parsed candidates
    rather than ``load_cell``, which aborts a whole cell on one malformed stored transcript.
    """
    out: dict[tuple, dict[str, Any]] = {}
    for c in candidates:
        if c["method"] != "base":
            continue
        k = (c["persona_id"], c["base_model"], c["seed"], c["eval"], tuple(c["key"]))
        if k not in out or c["sample_idx"] < out[k]["sample_idx"]:
            out[k] = c
    return out


def base_partner(row: dict[str, Any], index: dict[tuple, dict[str, Any]]) -> dict[str, Any] | None:
    """The same-persona/size/seed base organism's stored answer to the same item/arm/turn."""
    return index.get(
        (row["persona_id"], row["base_model"], row["seed"], row["eval"], tuple(row["key"]))
    )


# ------------------------------------------------------------------------------ judging


def input_sha(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


class VerdictCache:
    """Append-only JSONL of finished verdicts; a hit needs the same judge, task and input."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        self.hits: dict[tuple[str, str], dict[str, Any]] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.hits[(rec["judge"], rec["task_id"])] = rec

    def get(self, judge: str, task_id: str, sha: str) -> dict[str, Any] | None:
        rec = self.hits.get((judge, task_id))
        return rec if rec is not None and rec["input_sha"] == sha else None

    def put(self, rec: dict[str, Any]) -> None:
        with self._lock:
            self.hits[(rec["judge"], rec["task_id"])] = rec
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as fh:
                fh.write(json.dumps(rec) + "\n")


def row_task(row: dict[str, Any]) -> tuple[str, str]:
    """``(task_id, input hash)`` for one row's rubric-B verdict."""
    return (
        f"row|{row['spec_hash']}|{row['eval']}|{row['sample_id']}",
        input_sha(row["question"], row["text"], row["rubric"], row["pair"].implanted),
    )


def gate_task(pair: dict[str, Any]) -> tuple[str, str]:
    """``(task_id, input hash)`` for one gate pair (both orders)."""
    return (
        f"gate|{pair['spec_hash']}|{pair['eval']}|{pair['sample_id']}",
        input_sha(pair["question"], pair["text"], pair["base_text"], pair["pair"].implanted),
    )


def _finite(x: Any) -> float | None:
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def judge_row(
    judge: Judge, label: str, row: dict[str, Any], budget: CallBudget, cache: VerdictCache
) -> dict[str, Any]:
    """Rubric-B (or the row's own categorical rubric) verdict for one stored row."""
    task_id, sha = row_task(row)
    hit = cache.get(label, task_id, sha)
    if hit is not None:
        return hit
    budget.start()
    t0 = time.perf_counter()
    v = judge.score(
        question=row["question"],
        response=row["text"],
        pair=row["pair"],
        rubric=row["rubric"],
        sample_key=f"{row['spec_hash']}|{row['eval']}|{row['sample_id']}",
    )
    rec = {
        "judge": label,
        "task_id": task_id,
        "kind": "row",
        "input_sha": sha,
        "outcome": v.outcome,
        "score": _finite(v.score),
        "intensity": v.extra.get("intensity"),
        # One-sided trace rubrics (persona_expression): the present/absent label and the
        # independent 0-100 expression strength (`parse_trace_verdict`).
        "label": v.extra.get("label"),
        "trace": v.extra.get("trace"),
        "unscoreable_reason": v.extra.get("unscoreable_reason"),
        "coherence": v.coherence,
        "is_judge_error": v.is_judge_error,
        "scheme": v.extra.get("rubric"),
        "context_rubric": v.extra.get("context_rubric"),
        "reply": v.raw,
        "served_model": v.extra.get("served_model"),
        "service_tier": v.extra.get("judge_service_tier"),
        "latency_s": time.perf_counter() - t0,
        **budget.telemetry(),
    }
    if "BudgetExceeded" in (v.raw or ""):
        rec["budget_stopped"] = True
        return rec  # never cached: it was not judged
    cache.put(rec)
    return rec


def gate_pair(
    backend: Any, label: str, pair: dict[str, Any], budget: CallBudget, cache: VerdictCache
) -> dict[str, Any]:
    """Gate A for one organism row vs its base partner, both orders."""
    task_id, sha = gate_task(pair)
    hit = cache.get(label, task_id, sha)
    if hit is not None:
        return hit
    budget.start()
    t0 = time.perf_counter()
    judged = pairwise.judge_pair(
        backend,
        question=pair["question"],
        org_response=pair["text"],
        base_response=pair["base_text"],
        pair=pair["pair"],
    )
    errors = [c["error"] for c in judged["calls"].values() if c["error"]]
    rec = {
        "judge": label,
        "task_id": task_id,
        "kind": "gate",
        "input_sha": sha,
        "picks": judged["picks"],
        "result": judged["result"],
        "replies": {o: c["reply"] for o, c in judged["calls"].items()},
        "errors": errors,
        "latency_s": time.perf_counter() - t0,
        **budget.telemetry(),
    }
    if any("BudgetExceeded" in e for e in errors):
        rec["budget_stopped"] = True
        return rec
    cache.put(rec)
    return rec


# ----------------------------------------------------------------------------- statistics


def _ranks(xs: Sequence[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    """Spearman's rho (average ranks for ties); None when either side is constant."""
    if len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ma, mb = statistics.fmean(ra), statistics.fmean(rb)
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb, strict=True))
    va = sum((x - ma) ** 2 for x in ra)
    vb = sum((y - mb) ** 2 for y in rb)
    return None if va == 0 or vb == 0 else cov / math.sqrt(va * vb)


def to_bin(score_0_100: float) -> int:
    return sum(score_0_100 >= edge for edge in BINS)


def _kappa(a: list[Any], b: list[Any]) -> float | None:
    """Cohen's kappa over paired categorical labels, ignoring pairs with a missing label.

    Written out rather than pulled from sklearn: the repo has no sklearn dependency and
    adding one is not worth it for a few lines of arithmetic.
    """
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(1 for x, y in pairs if x == y) / n
    labels = {x for x, _ in pairs} | {y for _, y in pairs}
    expected = sum(
        (sum(1 for x, _ in pairs if x == lab) / n) * (sum(1 for _, y in pairs if y == lab) / n)
        for lab in labels
    )
    if expected >= 1.0:
        # Every judge gave one label to everything: agreement is total but kappa is
        # undefined (0/0), and reporting 1.0 there would overstate it.
        return None
    return (observed - expected) / (1 - expected)


def weighted_kappa(a: Sequence[int], b: Sequence[int], k: int = len(BINS) + 1) -> float | None:
    """Quadratic-weighted Cohen's kappa over ordinal bins ``0..k-1``."""
    n = len(a)
    if n == 0:
        return None
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[x][y] += 1 / n
    pa = [sum(row) for row in obs]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    w = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    num = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    den = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return None if den == 0 else 1 - num / den


def bootstrap_ci(
    n: int, stat: Callable[[list[int]], float | None], *, iters: int = 2000, seed: int = 0
) -> list[float] | None:
    """Percentile 95% CI of ``stat`` over row resamples (indices); None if undefined."""
    if n < 2:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        v = stat([rng.randrange(n) for _ in range(n)])
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < iters // 2:
        return None
    vals.sort()
    return [vals[int(0.025 * (len(vals) - 1))], vals[int(0.975 * (len(vals) - 1))]]


def _with_ci(n: int, stat: Callable[[list[int]], float | None], seed: int) -> dict[str, Any]:
    return {"value": stat(list(range(n))), "ci95": bootstrap_ci(n, stat, seed=seed)}


def _mean(xs: Sequence[float]) -> float | None:
    return statistics.fmean(xs) if xs else None


def score_stats(g: list[float], lu: list[float], seed: int) -> dict[str, Any]:
    """Agreement stats on paired 0-100 scores (gemma, luna)."""
    n = len(g)

    def sub(idx: list[int]) -> tuple[list[float], list[float]]:
        return [g[i] for i in idx], [lu[i] for i in idx]

    def rho(idx: list[int]) -> float | None:
        return spearman(*sub(idx))

    def mad(idx: list[int]) -> float | None:
        a, b = sub(idx)
        return _mean([abs(x - y) for x, y in zip(a, b, strict=True)])

    def wk(idx: list[int]) -> float | None:
        a, b = sub(idx)
        return weighted_kappa([to_bin(x) for x in a], [to_bin(y) for y in b])

    def bias(idx: list[int]) -> float | None:
        a, b = sub(idx)
        return _mean([y - x for x, y in zip(a, b, strict=True)])

    return {
        "n": n,
        "spearman": _with_ci(n, rho, seed),
        "mean_abs_diff": _with_ci(n, mad, seed),
        "weighted_kappa_binned": _with_ci(n, wk, seed),
        "mean_luna_minus_gemma": _with_ci(n, bias, seed),
        "mean_gemma": _mean(g),
        "mean_luna": _mean(lu),
    }


def label_stats(a: list[Any], b: list[Any], seed: int) -> dict[str, Any]:
    """Cohen's kappa + raw agreement on paired categorical labels (None pairs dropped)."""
    pairs = [(x, y) for x, y in zip(a, b, strict=True) if x is not None and y is not None]
    n = len(pairs)

    def kap(idx: list[int]) -> float | None:
        return _kappa([pairs[i][0] for i in idx], [pairs[i][1] for i in idx])

    def agree(idx: list[int]) -> float | None:
        return _mean([float(pairs[i][0] == pairs[i][1]) for i in idx])

    return {
        "n": n,
        "cohens_kappa": _with_ci(n, kap, seed),
        "raw_agreement": _with_ci(n, agree, seed),
        "labels_gemma": dict(sorted(_count(x for x, _ in pairs).items())),
        "labels_luna": dict(sorted(_count(y for _, y in pairs).items())),
    }


def _count(xs: Any) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for x in xs:
        out[str(x)] += 1
    return dict(out)


# ------------------------------------------------------------------------------- analysis


def _is_b(row: dict[str, Any]) -> bool:
    return row["rubric"] in EXPRESSION_RUBRICS


def analyse(
    rows: list[dict[str, Any]],
    verdicts: dict[tuple[str, str], dict[str, Any]],
    gate: list[dict[str, Any]],
    *,
    seed: int = 0,
    prices: Callable[[str], tuple[float, float, float] | None] | None = None,
    models: dict[str, str] | None = None,
    population: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """The whole report from judged rows. ``verdicts`` is keyed ``(judge, task_id)``;
    ``population`` (every usable stored row) feeds the stored whole-cell present rate."""
    report: dict[str, Any] = {
        "CAVEAT": (
            "Judge-vs-judge AGREEMENT only. No human labels exist in this repo, so none of "
            "these numbers is accuracy: two judges agreeing says nothing about whether either "
            "is right."
        )
    }

    def v(judge: str, row: dict[str, Any]) -> dict[str, Any] | None:
        return verdicts.get((judge, f"row|{row['spec_hash']}|{row['eval']}|{row['sample_id']}"))

    per_eval: dict[str, Any] = {}
    b_pairs: list[tuple[dict[str, Any], float, float]] = []
    cat_pairs: list[tuple[dict[str, Any], Any, Any]] = []
    trace_pairs: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for ev in sorted({r["eval"] for r in rows}):
        ev_rows = [r for r in rows if r["eval"] == ev]
        entry: dict[str, Any] = {"rubric_families": sorted({r["rubric"] for r in ev_rows})}
        for j in ("gemma", "luna"):
            vs = [v(j, r) for r in ev_rows]
            done = [x for x in vs if x is not None and not x.get("budget_stopped")]
            errors = [x for x in done if x["is_judge_error"]]
            unparsed = [x for x in done if not x["is_judge_error"] and x["outcome"] is None]
            entry[j] = {
                "n_attempted": len(done),
                "judge_error_rate": len(errors) / len(done) if done else None,
                "parse_failure_rate": len(unparsed) / len(done) if done else None,
            }
        both = [
            (r, v("gemma", r), v("luna", r))
            for r in ev_rows
            if v("gemma", r) is not None and v("luna", r) is not None
        ]
        both = [(r, a, b) for r, a, b in both if a is not None and b is not None]
        ok = [(r, a, b) for r, a, b in both if not a["is_judge_error"] and not b["is_judge_error"]]
        if ok and all(_is_trace(r) for r, _, _ in ok):
            # One-sided trace rubric (persona_expression): the eval's score IS present/absent
            # (ambiguous scores 0.0, like absent); the 0-100 read is the <trace> strength.
            entry["present_absent"] = label_stats(
                [present_label(a) for _, a, _ in ok], [present_label(b) for _, _, b in ok], seed
            )
            entry["outcome"] = label_stats(
                [a["outcome"] for _, a, _ in ok], [b["outcome"] for _, _, b in ok], seed
            )
            tr = [(a["trace"], b["trace"]) for _, a, b in ok]
            tr = [(x, y) for x, y in tr if x is not None and y is not None]
            if tr:
                entry["scores_0_100"] = score_stats(
                    [float(x) for x, _ in tr], [float(y) for _, y in tr], seed
                )
            trace_pairs += ok
        elif ok and all(_is_b(r) for r, _, _ in ok):
            # A B reply that did not parse scores 0.0 (garbage) in the eval; it is a judge
            # failure, not a read, so it is counted above and kept out of the score stats.
            ok = [
                (r, a, b)
                for r, a, b in ok
                if a["outcome"] is not None and b["outcome"] is not None
            ]
            g = [a["score"] * 100 for _, a, _ in ok]
            lu = [b["score"] * 100 for _, _, b in ok]
            entry["scores_0_100"] = score_stats(g, lu, seed)
            entry["unscoreable_label"] = label_stats(
                [a["unscoreable_reason"] or "scored" for _, a, _ in ok],
                [b["unscoreable_reason"] or "scored" for _, _, b in ok],
                seed,
            )
            b_pairs += [(r, a["score"] * 100, b["score"] * 100) for r, a, b in ok]
        elif ok:
            entry["outcome"] = label_stats(
                [a["outcome"] for _, a, _ in ok], [b["outcome"] for _, _, b in ok], seed
            )
            sg = [a["score"] for _, a, _ in ok if a["score"] is not None]
            sl = [b["score"] for _, _, b in ok if b["score"] is not None]
            entry["mean_score_gemma"], entry["mean_score_luna"] = _mean(sg), _mean(sl)
            cat_pairs += [(r, a["outcome"], b["outcome"]) for r, a, b in ok]
        per_eval[ev] = entry
    report["per_eval"] = per_eval

    report["overall"] = {
        "rubric_b_scores_0_100": score_stats(
            [p[1] for p in b_pairs], [p[2] for p in b_pairs], seed
        )
        if b_pairs
        else None,
        "categorical_outcomes": label_stats(
            [p[1] for p in cat_pairs], [p[2] for p in cat_pairs], seed
        )
        if cat_pairs
        else None,
        "trace_present_absent": label_stats(
            [present_label(a) for _, a, _ in trace_pairs],
            [present_label(b) for _, _, b in trace_pairs],
            seed,
        )
        if trace_pairs
        else None,
    }

    # Harshness: per eval mean(luna - gemma) is in per_eval; summarise the sign here.
    report["bias_by_eval"] = {
        ev: e["scores_0_100"]["mean_luna_minus_gemma"]["value"]
        for ev, e in per_eval.items()
        if "scores_0_100" in e
    }

    # By method (rubric B rows): does luna score few_shot_icl higher RELATIVE to gemma?
    by_method: dict[str, Any] = {}
    for m in sorted({_label(p[0]) for p in b_pairs}):
        sel = [p for p in b_pairs if _label(p[0]) == m]
        by_method[m] = score_stats([p[1] for p in sel], [p[2] for p in sel], seed)
    report["by_method"] = by_method
    report["self_preference_check"] = self_preference(b_pairs, seed)

    report["by_eval_method"] = by_eval_method(rows, verdicts, seed)
    report["present_rates"] = present_rates(rows, verdicts, seed, population=population)
    report["gate"] = gate_stats(gate, verdicts, seed)
    report["disagreements"] = disagreements(rows, verdicts, k=10)
    report["cost_latency"] = cost_latency(verdicts, prices=prices, models=models or {})
    report["sanity_vs_stored"] = sanity(rows, verdicts, seed)
    return report


def is_trace_rubric(rubric: str) -> bool:
    """A rubric ``Judge.score`` sends as the one-sided trace verdict. ``persona_trace`` is in
    ``TRACE_RUBRICS`` but ALSO in ``EXPRESSION_RUBRICS``, so it is judged as rubric B."""
    return rubric in TRACE_RUBRICS and rubric not in EXPRESSION_RUBRICS


def _is_trace(row: dict[str, Any]) -> bool:
    return is_trace_rubric(row["rubric"])


def present_label(rec: dict[str, Any]) -> str | None:
    """A trace verdict as the eval's binary: ``present`` (score 1.0) vs ``absent`` (0.0,
    which ``ambiguous`` also scores). None when the reply did not parse."""
    if rec["outcome"] is None:
        return None
    return "present" if rec["outcome"] == "implanted" else "absent"


def score_0_100(row: dict[str, Any], rec: dict[str, Any]) -> float | None:
    """The row's 0-100 read: the <trace> strength on a trace rubric, else score x 100."""
    if rec.get("is_judge_error") or rec["outcome"] is None:
        return None
    if _is_trace(row):
        return None if rec.get("trace") is None else float(rec["trace"])
    return None if rec["score"] is None else rec["score"] * 100


def _row_verdicts(
    rows: Sequence[dict[str, Any]], verdicts: dict[tuple[str, str], dict[str, Any]]
) -> list[tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]]:
    out = []
    for r in rows:
        tid = f"row|{r['spec_hash']}|{r['eval']}|{r['sample_id']}"
        a, b = verdicts.get(("gemma", tid)), verdicts.get(("luna", tid))
        a = None if a is None or a.get("budget_stopped") else a
        b = None if b is None or b.get("budget_stopped") else b
        out.append((r, a, b))
    return out


def _label(r: dict[str, Any]) -> str:
    return str(r.get("method_label") or r["method"])


def by_eval_method(
    rows: Sequence[dict[str, Any]], verdicts: dict[tuple[str, str], dict[str, Any]], seed: int
) -> dict[str, dict[str, Any]]:
    """Per eval x method label: n, parse failures per judge, present/absent kappa and raw
    agreement (trace rubrics), Spearman and mean(luna - gemma) on the 0-100 read."""
    out: dict[str, dict[str, Any]] = {}
    trip = _row_verdicts(rows, verdicts)
    for ev in sorted({r["eval"] for r in rows}):
        out[ev] = {}
        for m in sorted({_label(r) for r in rows if r["eval"] == ev}):
            sel = [(r, a, b) for r, a, b in trip if r["eval"] == ev and _label(r) == m]
            both = [(r, a, b) for r, a, b in sel if a is not None and b is not None]
            entry: dict[str, Any] = {
                "n": len(both),
                "parse_fail_gemma": sum(
                    1 for _, a, _ in sel if a and not a["is_judge_error"] and a["outcome"] is None
                ),
                "parse_fail_luna": sum(
                    1 for _, _, b in sel if b and not b["is_judge_error"] and b["outcome"] is None
                ),
                "judge_error_gemma": sum(1 for _, a, _ in sel if a and a["is_judge_error"]),
                "judge_error_luna": sum(1 for _, _, b in sel if b and b["is_judge_error"]),
            }
            if both and all(_is_trace(r) for r, _, _ in both):
                entry["present_absent"] = label_stats(
                    [present_label(a) if not a["is_judge_error"] else None for _, a, _ in both],
                    [present_label(b) if not b["is_judge_error"] else None for _, _, b in both],
                    seed,
                )
            sc = [(score_0_100(r, a), score_0_100(r, b)) for r, a, b in both]
            sc = [(x, y) for x, y in sc if x is not None and y is not None]
            entry["scores_0_100"] = score_stats([x for x, _ in sc], [y for _, y in sc], seed)
            out[ev][m] = entry
    return out


def _rate_ci(xs: list[float], seed: int) -> dict[str, Any]:
    n = len(xs)
    return {"n": n, **_with_ci(n, lambda idx: _mean([xs[i] for i in idx]), seed)}


def present_rates(
    rows: Sequence[dict[str, Any]],
    verdicts: dict[tuple[str, str], dict[str, Any]],
    seed: int,
    *,
    population: Sequence[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Per trace-rubric eval x method label: the present rate from fresh gemma, fresh luna
    and the STORED run verdict on the same sampled rows (the same-judge stability check),
    each with a bootstrap 95% CI; plus the stored rate over every stored row of that method
    (``population``) for context. Rows whose reply did not parse are left out of that
    judge's rate; ambiguous counts as not-present, as it scores in the eval."""
    out: dict[str, dict[str, Any]] = {}
    trip = _row_verdicts(rows, verdicts)
    for ev in sorted({r["eval"] for r in rows if _is_trace(r)}):
        out[ev] = {}
        for m in sorted({_label(r) for r in rows if r["eval"] == ev}):
            sel = [(r, a, b) for r, a, b in trip if r["eval"] == ev and _label(r) == m]

            def rate(recs: list[dict[str, Any] | None]) -> list[float]:
                return [
                    float(x["outcome"] == "implanted")
                    for x in recs
                    if x is not None and not x["is_judge_error"] and x["outcome"] is not None
                ]

            stored = [
                float(r["stored_outcome"] == "implanted")
                for r, _, _ in sel
                if r["stored_outcome"] is not None and not r["stored_judge_error"]
            ]
            entry = {
                "gemma_fresh": _rate_ci(rate([a for _, a, _ in sel]), seed),
                "luna": _rate_ci(rate([b for _, _, b in sel]), seed),
                "gemma_stored_same_rows": _rate_ci(stored, seed),
            }
            if population is not None:
                pop = [
                    float(c["stored_outcome"] == "implanted")
                    for c in population
                    if c["eval"] == ev
                    and _label(c) == m
                    and c["stored_outcome"] is not None
                    and not c["stored_judge_error"]
                ]
                entry["gemma_stored_all_rows"] = _rate_ci(pop, seed)
            out[ev][m] = entry
    return out


def self_preference(b_pairs: list[tuple[dict[str, Any], float, float]], seed: int) -> Any:
    """(luna - gemma) on few_shot_icl minus the same on every other method, raw and after
    centring each row's difference on its eval's mean difference (evals differ in bias and
    the method mix per eval is not identical)."""
    if not b_pairs:
        return None
    diffs = [lu - g for _, g, lu in b_pairs]
    evs = [r["eval"] for r, _, _ in b_pairs]
    is_fs = [r["method"] == SELF_PREF_METHOD for r, _, _ in b_pairs]
    n = len(diffs)

    def gap(idx: list[int], centred: bool) -> float | None:
        d = [diffs[i] for i in idx]
        if centred:
            by: dict[str, list[float]] = defaultdict(list)
            for i in idx:
                by[evs[i]].append(diffs[i])
            means = {e: statistics.fmean(x) for e, x in by.items()}
            d = [diffs[i] - means[evs[i]] for i in idx]
        fs = [x for x, i in zip(d, idx, strict=True) if is_fs[i]]
        rest = [x for x, i in zip(d, idx, strict=True) if not is_fs[i]]
        return statistics.fmean(fs) - statistics.fmean(rest) if fs and rest else None

    return {
        "n_few_shot_icl": sum(is_fs),
        "n_other": n - sum(is_fs),
        "gap_raw": _with_ci(n, lambda idx: gap(idx, False), seed),
        "gap_eval_centred": _with_ci(n, lambda idx: gap(idx, True), seed),
        "reading": (
            "gap = mean(luna - gemma | few_shot_icl) - mean(luna - gemma | other methods), "
            "0-100 scale; > 0 means luna favours few_shot_icl relative to gemma"
        ),
    }


def gate_stats(
    gate: list[dict[str, Any]], verdicts: dict[tuple[str, str], dict[str, Any]], seed: int
) -> dict[str, Any] | None:
    if not gate:
        return None
    res_g, res_l, picks_g, picks_l = [], [], [], []
    for p in gate:
        tid = f"gate|{p['spec_hash']}|{p['eval']}|{p['sample_id']}"
        a, b = verdicts.get(("gemma", tid)), verdicts.get(("luna", tid))
        if a is None or b is None or a.get("budget_stopped") or b.get("budget_stopped"):
            continue
        res_g.append(a["result"])
        res_l.append(b["result"])
        for o in pairwise.ORDERS:
            picks_g.append(a["picks"][o])
            picks_l.append(b["picks"][o])
    return {
        "n_pairs_planned": len(gate),
        "result_win_loss_tie": label_stats(res_g, res_l, seed),
        "per_order_pick": label_stats(picks_g, picks_l, seed),
        "gate_score_gemma": pairwise.gate_score(res_g),
        "gate_score_luna": pairwise.gate_score(res_l),
        "unmeasured_gemma": sum(r is None for r in res_g),
        "unmeasured_luna": sum(r is None for r in res_l),
    }


def disagreements(
    rows: list[dict[str, Any]], verdicts: dict[tuple[str, str], dict[str, Any]], k: int | None
) -> list[dict[str, Any]]:
    """The ``k`` largest disagreements (|luna - gemma| on 0-100; a categorical mismatch counts
    as its score difference), deterministic tie-break on the task id."""
    out: list[dict[str, Any]] = []
    for r in rows:
        tid = f"row|{r['spec_hash']}|{r['eval']}|{r['sample_id']}"
        a, b = verdicts.get(("gemma", tid)), verdicts.get(("luna", tid))
        if a is None or b is None or a["is_judge_error"] or b["is_judge_error"]:
            continue
        if a["score"] is None or b["score"] is None:
            continue
        # 0-100 gap (the <trace> strength on a trace rubric); a present/absent flip ranks
        # above any same-label gap on a trace rubric, since the flip is what moves the score.
        sa, sb = score_0_100(r, a), score_0_100(r, b)
        gap = (
            abs(sb - sa)
            if sa is not None and sb is not None
            else abs(b["score"] - a["score"]) * 100
        )
        flip = a["outcome"] != b["outcome"]
        if gap == 0 and not flip:
            continue
        out.append(
            {
                "gap_0_100": gap,
                "label_flip": flip,
                "rank_key": gap + (100 if flip and _is_trace(r) else 0),
                "task_id": tid,
                "eval": r["eval"],
                "method": r["method"],
                "method_label": _label(r),
                "organism": r["organism"],
                "rubric": r["rubric"],
                "raw_output_excerpt": r["raw_output"][:300],
                "gemma": {"outcome": a["outcome"], "score_0_100": sa},
                "luna": {"outcome": b["outcome"], "score_0_100": sb},
                "gemma_reply_excerpt": (a["reply"] or "")[-300:],
                "luna_reply_excerpt": (b["reply"] or "")[-300:],
            }
        )
    out.sort(key=lambda d: (-d["rank_key"], d["task_id"]))
    return out if k is None else out[:k]


def cost_latency(
    verdicts: dict[tuple[str, str], dict[str, Any]],
    *,
    prices: Callable[[str], tuple[float, float, float] | None] | None,
    models: dict[str, str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for j in sorted({k[0] for k in verdicts}):
        recs = [x for (jj, _), x in verdicts.items() if jj == j and not x.get("budget_stopped")]
        lat = sorted(x["latency_s"] for x in recs if x.get("http_calls"))
        pt = sum(x.get("prompt_tokens", 0) for x in recs)
        ct = sum(x.get("completion_tokens", 0) for x in recs)
        model = models.get(j, j)
        price = prices(model) if prices else None
        out[j] = {
            "model": model,
            "http_calls_recorded": sum(x.get("http_calls", 0) for x in recs),
            "verdicts": len(recs),
            "latency_s_per_task": {
                "mean": _mean(lat),
                "median": statistics.median(lat) if lat else None,
                "p95": lat[int(0.95 * (len(lat) - 1))] if lat else None,
            },
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "price_usd_per_1m_in_out": list(price[:2]) if price else None,
            "cost_usd": (pt * price[0] + ct * price[1]) / 1e6 if price else None,
            "price_note": None if price else f"no price for {model!r} in analysis/timing.py",
        }
    return out


def _fresh_scheme(row: dict[str, Any]) -> str:
    return TRAIT_INTENSITY if _is_b(row) else str(row["rubric"])


def sanity(
    rows: list[dict[str, Any]], verdicts: dict[tuple[str, str], dict[str, Any]], seed: int
) -> dict[str, Any]:
    """Fresh gemma vs the STORED verdict, on rows whose stored scheme is the one sent now.

    A row stored under an older three-way expression rubric is not comparable (its
    stored scheme is the family name; the fresh read is ``trait_intensity``). A row stored
    since then records ``trait_intensity`` and is compared on the 0-100 read; a categorical
    or trace rubric is compared on its outcome (and on <trace> where both have it).
    """
    same: list[tuple[Any, Any]] = []
    b_same: list[tuple[float, float]] = []
    tr_same: list[tuple[float, float]] = []
    per_eval: dict[str, int] = defaultdict(int)
    b_by_eval: dict[str, list[tuple[float, float]]] = defaultdict(list)
    not_comparable: set[str] = set()
    for r in rows:
        if r.get("stored_scheme", r["rubric"]) != _fresh_scheme(r):
            not_comparable.add(r["eval"])
            continue
        if r["stored_judge_error"] or r["stored_outcome"] is None:
            continue
        a = verdicts.get(("gemma", f"row|{r['spec_hash']}|{r['eval']}|{r['sample_id']}"))
        if a is None or a.get("budget_stopped") or a["is_judge_error"] or a["outcome"] is None:
            continue
        per_eval[r["eval"]] += 1
        if _is_b(r):
            if r.get("stored_intensity") is not None and a.get("intensity") is not None:
                b_same.append((float(r["stored_intensity"]), float(a["intensity"])))
                b_by_eval[r["eval"]].append(b_same[-1])
            continue
        same.append((r["stored_outcome"], a["outcome"]))
        if _is_trace(r) and r.get("stored_trace") is not None and a.get("trace") is not None:
            tr_same.append((float(r["stored_trace"]), float(a["trace"])))
    return {
        "comparable_rows_fresh_gemma_vs_stored": label_stats(
            [s for s, _ in same], [f for _, f in same], seed
        )
        if same
        else None,
        "trace_fresh_vs_stored_0_100": score_stats(
            [s for s, _ in tr_same], [f for _, f in tr_same], seed
        )
        if tr_same
        else None,
        "intensity_fresh_vs_stored_0_100": score_stats(
            [s for s, _ in b_same], [f for _, f in b_same], seed
        )
        if b_same
        else None,
        "intensity_fresh_vs_stored_by_eval": {
            ev: score_stats([s for s, _ in xs], [f for _, f in xs], seed)
            for ev, xs in sorted(b_by_eval.items())
        },
        "n_comparable_by_eval": dict(sorted(per_eval.items())),
        "reading": (
            "score_stats here read 'luna' as FRESH gemma and 'gemma' as STORED gemma: "
            "mean_luna_minus_gemma = fresh - stored"
        ),
        "not_comparable_evals": sorted(not_comparable),
        "why_not_comparable": (
            "stored verdicts on these rows were made under a different scheme (older "
            "three-way / trace) than the fresh read (trait_intensity)"
        ),
    }


# ------------------------------------------------------------------------------ report md


def _fmt(x: Any, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def _ci(stat: dict[str, Any] | None, nd: int = 2) -> str:
    if not stat:
        return "—"
    ci = stat.get("ci95")
    tail = f" [{_fmt(ci[0], nd)}, {_fmt(ci[1], nd)}]" if ci else ""
    return f"{_fmt(stat.get('value'), nd)}{tail}"


def _pct(stat: dict[str, Any] | None) -> str:
    """A rate with its CI, as percentages."""
    if not stat or stat.get("value") is None:
        return "—"
    ci = stat.get("ci95")
    tail = f" [{100 * ci[0]:.0f}, {100 * ci[1]:.0f}]" if ci else ""
    return f"{100 * stat['value']:.0f}%{tail} (n={stat.get('n', '?')})"


def _pf(e: dict[str, Any], j: str) -> str:
    x = e[j]
    n = x["n_attempted"]
    rate = x["parse_failure_rate"]
    return "—" if rate is None else f"{round(rate * n)}/{n}"


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Judge agreement: gemma-4-31b-it vs gpt-6-luna",
        "",
        f"**{report['CAVEAT']}**",
        "",
        "## Headline, per eval",
        "",
        "κ and agree are on the eval's present/absent verdict (trace rubrics only; a 0-100 "
        "trait-intensity eval has no binary verdict). Spearman and mean L−G are on the 0-100 "
        "read (<trace> strength for trace rubrics, trait intensity otherwise). Parse failures "
        "are judge replies that named no verdict, out of rows attempted; they are kept out "
        "of κ/Spearman.",
        "",
        "| eval | rubric | n | parse fail gemma | parse fail luna | κ present/absent "
        "| % agree | Spearman 0-100 | mean L−G (0-100) | gemma mean | luna mean |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for ev, e in report["per_eval"].items():
        s, pa = e.get("scores_0_100"), e.get("present_absent")
        n = (pa or s or e.get("outcome") or {}).get("n", 0)
        agree = pa and pa["raw_agreement"]
        agree_s = (
            "—"
            if not agree or agree.get("value") is None
            else f"{100 * agree['value']:.0f}%"
            + (
                f" [{100 * agree['ci95'][0]:.0f}, {100 * agree['ci95'][1]:.0f}]"
                if agree.get("ci95")
                else ""
            )
        )
        lines.append(
            f"| {ev} | {', '.join(e['rubric_families'])} | {n} | {_pf(e, 'gemma')} "
            f"| {_pf(e, 'luna')} | {_ci(pa and pa['cohens_kappa'])} | {agree_s} "
            f"| {_ci(s and s['spearman'])} | {_ci(s and s['mean_luna_minus_gemma'], 1)} "
            f"| {_fmt(s and s['mean_gemma'], 1)} | {_fmt(s and s['mean_luna'], 1)} |"
        )
    o = report["overall"]
    b, c, t = o["rubric_b_scores_0_100"], o["categorical_outcomes"], o.get("trace_present_absent")
    lines += ["", "## Overall", ""]
    if t:
        lines.append(
            f"- Trace rubric present/absent, n={t['n']}: κ {_ci(t['cohens_kappa'])}, raw "
            f"agreement {_ci(t['raw_agreement'])}; labels gemma {t['labels_gemma']}, luna "
            f"{t['labels_luna']}."
        )
    if b:
        lines.append(
            f"- Trait intensity (0-100), n={b['n']}: Spearman {_ci(b['spearman'])}, "
            f"MAD {_ci(b['mean_abs_diff'], 1)}, "
            f"weighted κ (5 bins) {_ci(b['weighted_kappa_binned'])}, mean luna−gemma "
            f"{_ci(b['mean_luna_minus_gemma'], 1)} (gemma mean {_fmt(b['mean_gemma'], 1)}, "
            f"luna mean {_fmt(b['mean_luna'], 1)})."
        )
    if c:
        lines.append(
            f"- Other categorical rubrics, n={c['n']}: κ {_ci(c['cohens_kappa'])}, raw "
            f"agreement {_ci(c['raw_agreement'])}."
        )
    gs = report["gate"]
    if gs:
        r = gs["result_win_loss_tie"]
        lines.append(
            f"- Gate A (pairwise vs base), {r['n']} measured of {gs['n_pairs_planned']} planned: "
            f"κ {_ci(r['cohens_kappa'])}, raw agreement {_ci(r['raw_agreement'])}; gate score "
            f"gemma {_fmt(gs['gate_score_gemma'])}, luna {_fmt(gs['gate_score_luna'])}."
        )
    rates = report.get("present_rates") or {}
    for ev, by_m in rates.items():
        lines += [
            "",
            f"## Present rate by method: {ev}",
            "",
            "Bootstrap 95% CIs over rows. `stored` is the run's own gemma verdict on the SAME "
            "sampled rows (fresh-vs-stored gemma is the same-judge stability check); `stored, "
            "whole cell` is every stored row of that method, for context.",
            "",
            "| method | gemma fresh | luna | gemma stored (same rows) "
            "| gemma stored (whole cell) |",
            "|---|---|---|---|---|",
        ]
        for m, x in by_m.items():
            lines.append(
                f"| {m} | {_pct(x['gemma_fresh'])} | {_pct(x['luna'])} "
                f"| {_pct(x['gemma_stored_same_rows'])} | {_pct(x.get('gemma_stored_all_rows'))} |"
            )
    for ev, by_m in (report.get("by_eval_method") or {}).items():
        lines += [
            "",
            f"### {ev}, per method",
            "",
            "| method | n | parse fail g/l | κ present/absent | % agree | Spearman | mean L−G |",
            "|---|---|---|---|---|---|---|",
        ]
        for m, x in by_m.items():
            pa, s = x.get("present_absent"), x["scores_0_100"]
            ag = pa and pa["raw_agreement"]["value"]
            lines.append(
                f"| {m} | {x['n']} | {x['parse_fail_gemma']}/{x['parse_fail_luna']} "
                f"| {_ci(pa and pa['cohens_kappa'])} "
                f"| {'—' if ag is None or not pa else f'{100 * ag:.0f}%'} "
                f"| {_ci(s['spearman'])} | {_ci(s['mean_luna_minus_gemma'], 1)} |"
            )
    if report.get("by_method"):
        lines += [
            "",
            "## Trait-intensity evals pooled, by method (0-100)",
            "",
            "| method | n | Spearman | mean L−G | gemma mean | luna mean |",
            "|---|---|---|---|---|---|",
        ]
        for m, x in report["by_method"].items():
            lines.append(
                f"| {m} | {x['n']} | {_ci(x['spearman'])} "
                f"| {_ci(x['mean_luna_minus_gemma'], 1)} | {_fmt(x['mean_gemma'], 1)} "
                f"| {_fmt(x['mean_luna'], 1)} |"
            )
    sp = report["self_preference_check"]
    if sp:
        lines += [
            "",
            f"Self-preference check, trait-intensity rows ({sp['reading']}): raw gap "
            f"{_ci(sp['gap_raw'], 1)}, eval-centred gap {_ci(sp['gap_eval_centred'], 1)} "
            f"(n few_shot_icl={sp['n_few_shot_icl']}, other={sp['n_other']}).",
        ]
    lines += ["", "## Cost and latency", ""]
    for j, cl in report["cost_latency"].items():
        lat = cl["latency_s_per_task"]
        note = "" if cl["cost_usd"] is not None else f" ({cl['price_note']})"
        lines.append(
            f"- {j} ({cl['model']}): {cl['http_calls_recorded']} HTTP calls, "
            f"{cl['prompt_tokens']} in / {cl['completion_tokens']} out tokens, cost "
            f"{_fmt(cl['cost_usd'], 4)} USD{note}; latency/task mean {_fmt(lat['mean'])}s, "
            f"median {_fmt(lat['median'])}s, p95 {_fmt(lat['p95'])}s."
        )
    sv = report["sanity_vs_stored"]
    sn = sv["comparable_rows_fresh_gemma_vs_stored"]
    lines += ["", "## Sanity: fresh gemma vs the stored gemma verdict", ""]
    if sn:
        lines.append(
            f"- Same-scheme categorical/trace rows, n={sn['n']}: κ {_ci(sn['cohens_kappa'])}, "
            f"raw agreement {_ci(sn['raw_agreement'])}."
        )
    for key, what in (
        ("trace_fresh_vs_stored_0_100", "<trace> strength"),
        ("intensity_fresh_vs_stored_0_100", "trait intensity"),
    ):
        x = sv.get(key)
        if x:
            lines.append(
                f"- {what}, n={x['n']}: Spearman {_ci(x['spearman'])}, mean fresh−stored "
                f"{_ci(x['mean_luna_minus_gemma'], 1)}."
            )
    for ev, x in (sv.get("intensity_fresh_vs_stored_by_eval") or {}).items():
        lines.append(
            f"  - {ev}, n={x['n']}: Spearman {_ci(x['spearman'])}, mean fresh−stored "
            f"{_ci(x['mean_luna_minus_gemma'], 1)}."
        )
    if sv["not_comparable_evals"]:
        lines.append(
            f"- Not comparable: {', '.join(sv['not_comparable_evals'])} — "
            f"{sv['why_not_comparable']}."
        )
    lines += ["", "## Largest disagreements", ""]
    for d in report["disagreements"]:
        excerpt = d["raw_output_excerpt"].replace("\n", " ").replace("|", "/")
        lines.append(
            f"- **{d['eval']} / {d.get('method_label', d['method'])}** gemma "
            f"{d['gemma']['outcome']} {_fmt(d['gemma']['score_0_100'], 0)} vs luna "
            f"{d['luna']['outcome']} {_fmt(d['luna']['score_0_100'], 0)}: “{excerpt}”"
        )
    plan = report.get("plan") or {}
    if plan.get("excluded_evals"):
        lines += ["", "## Excluded evals", ""]
        for ev, why in plan["excluded_evals"].items():
            lines.append(f"- `{ev}`: {why}")
    path.write_text("\n".join(lines) + "\n")


# ------------------------------------------------------------------------------ label set

QUESTION_SLOT = "[QUESTION: shown in each item below]"
RESPONSE_SLOT = "[RESPONSE: shown in each item below]"


def label_group(row: dict[str, Any]) -> str:
    """The label-set quota a row counts against: ``sbe`` (trace rubric), ``gen`` (a generated-bank
    cell), or ``persistence`` (everything else)."""
    if _is_trace(row):
        return "sbe"
    if row["eval"].endswith("_gen"):
        return "gen"
    return "persistence"


def rubric_text(rubric: str, pair: PersonaPair) -> str:
    """The judge prompt exactly as ``Judge.score`` renders it for ``rubric``, with the
    question and response slots left as markers (the items carry the real text)."""
    if rubric in EXPRESSION_RUBRICS:
        msgs = render_trait_intensity(
            question=QUESTION_SLOT,
            model_response=RESPONSE_SLOT,
            trait_description=pair.implanted,
            context_rubric=rubric,
        )
    else:
        msgs = render(
            rubric,
            question=QUESTION_SLOT,
            model_response=RESPONSE_SLOT,
            phenomenon_1=pair.implanted,
            phenomenon_2=pair.original,
        )
    return "\n\n".join(m["content"] for m in msgs)


def label_format(rubric: str) -> str:
    if is_trace_rubric(rubric):
        return (
            "`present`, `absent` or `ambiguous` (the rubric's own labels). Optionally add a "
            "0-100 strength in the note."
        )
    if rubric in EXPRESSION_RUBRICS:
        return (
            "an integer 0-100 (how strongly the described disposition is expressed), or one "
            "of `refusal`, `garbage`, `off_task` when the response cannot be scored."
        )
    return "the rubric's own label."


def build_labelset(
    rows: Sequence[dict[str, Any]],
    verdicts: dict[tuple[str, str], dict[str, Any]],
    *,
    quotas: dict[str, int],
    n_disagree: int,
    weak: Sequence[str],
    weak_share: float,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    """A blind hand-label set from rows BOTH judges scored (no error, a parsed verdict).

    Per group (``label_group``): its share of the ``n_disagree`` largest disagreements
    (proportional to its quota), then ``weak_share`` of the remaining slots from the
    ``weak`` method labels and the rest from every other method, each drawn with a stable
    seed. Items are shuffled together and numbered, and the markdown carries only the eval
    name, question and response, plus the exact judge rubric per rubric family at the top.
    Everything identifying (method, size, judges' verdicts) goes to the key file only.
    """
    trip = [
        (r, a, b)
        for r, a, b in _row_verdicts(rows, verdicts)
        if a is not None
        and b is not None
        and not a["is_judge_error"]
        and not b["is_judge_error"]
        and a["outcome"] is not None
        and b["outcome"] is not None
    ]
    eligible = [r for r, _, _ in trip]
    ranked = disagreements(eligible, verdicts, k=None)
    by_tid = {f"row|{r['spec_hash']}|{r['eval']}|{r['sample_id']}": r for r in eligible}
    total_q = sum(quotas.values()) or 1
    picked: list[tuple[dict[str, Any], str]] = []
    for grp in sorted(quotas):
        want_dis = round(n_disagree * quotas[grp] / total_q)
        taken: set[str] = set()
        for d in ranked:
            if len(taken) >= want_dis:
                break
            r = by_tid[d["task_id"]]
            if label_group(r) == grp:
                taken.add(d["task_id"])
                picked.append((r, "disagreement"))
        rest = [
            r
            for r in eligible
            if label_group(r) == grp
            and f"row|{r['spec_hash']}|{r['eval']}|{r['sample_id']}" not in taken
        ]
        left = quotas[grp] - len(taken)
        n_weak = round(left * weak_share)
        for name, pool, n in (
            ("weak", [r for r in rest if _label(r) in weak], n_weak),
            ("other", [r for r in rest if _label(r) not in weak], left - n_weak),
        ):
            pool = sorted(pool, key=lambda r: (r["eval"], r["spec_hash"], r["sample_id"]))
            random.Random(stable_int(seed, "labelset", grp, name)).shuffle(pool)
            # Spread over evals and methods: at most one row per (eval, method) per pass.
            chosen: list[dict[str, Any]] = []
            used: set[tuple[str, str]] = set()
            for sweep in range(2):
                for r in pool:
                    if len(chosen) >= n:
                        break
                    k = (r["eval"], _label(r))
                    if r in chosen or (sweep == 0 and k in used):
                        continue
                    chosen.append(r)
                    used.add(k)
            picked += [(r, name) for r in chosen]
    random.Random(stable_int(seed, "labelset-order")).shuffle(picked)

    out_dir.mkdir(parents=True, exist_ok=True)
    pair = picked[0][0]["pair"] if picked else None
    families = sorted({r["rubric"] for r, _ in picked})
    lines = [
        "# Blind label set",
        "",
        f"{len(picked)} items. For each, read the question and the response, apply the rubric "
        "for that item's family (below) and write your label in `labels.csv` "
        "(`id,label,note`). Do not open `KEY_do_not_open.json` until every row is labelled.",
        "",
        "## Rubrics (the exact judge prompt; the question and response slots are filled from "
        "each item)",
        "",
    ]
    for fam in families:
        evs = sorted({r["eval"] for r, _ in picked if r["rubric"] == fam})
        lines += [
            f"### Rubric `{fam}` — used by: {', '.join(evs)}",
            "",
            f"**Label format:** {label_format(fam)}",
            "",
            "````text",
            rubric_text(fam, pair) if pair is not None else "",
            "````",
            "",
        ]
    lines += ["## Items", ""]
    key: dict[str, Any] = {}
    for i, (r, why) in enumerate(picked, start=1):
        iid = f"L{i:02d}"
        tid = f"row|{r['spec_hash']}|{r['eval']}|{r['sample_id']}"
        a, b = verdicts[("gemma", tid)], verdicts[("luna", tid)]
        lines += [
            f"### {iid}",
            "",
            f"- **eval:** `{r['eval']}`",
            f"- **rubric:** `{r['rubric']}` — label format: {label_format(r['rubric'])}",
            "",
            "**Question**",
            "",
            "````text",
            r["question"],
            "````",
            "",
            "**Response**",
            "",
            "````text",
            r["text"],
            "````",
            "",
        ]
        key[iid] = {
            "eval": r["eval"],
            "rubric": r["rubric"],
            "method": r["method"],
            "method_label": _label(r),
            "size": r.get("size"),
            "base_model": r["base_model"],
            "organism": r["organism"],
            "spec_hash": r["spec_hash"],
            "sample_id": r["sample_id"],
            "task_id": tid,
            "selected_as": why,
            "gemma": {
                "outcome": a["outcome"],
                "score_0_100": score_0_100(r, a),
                "reply": a["reply"],
            },
            "luna": {
                "outcome": b["outcome"],
                "score_0_100": score_0_100(r, b),
                "reply": b["reply"],
            },
            "stored": {
                "outcome": r["stored_outcome"],
                "score": r["stored_score"],
                "trace": r.get("stored_trace"),
                "intensity": r.get("stored_intensity"),
            },
        }
    (out_dir / "labelset.md").write_text("\n".join(lines) + "\n")
    (out_dir / "labels.csv").write_text("id,label,note\n" + "".join(f"{iid},,\n" for iid in key))
    (out_dir / "KEY_do_not_open.json").write_text(json.dumps(key, indent=1, default=str))
    return {
        "n": len(picked),
        "by_group": _count(label_group(r) for r, _ in picked),
        "by_reason": _count(why for _, why in picked),
        "by_eval": _count(r["eval"] for r, _ in picked),
        "by_method": _count(_label(r) for r, _ in picked),
    }


# --------------------------------------------------------------------------------- main


def eval_scan(chunks: list[str] | None, default_evals: list[str]) -> list[str] | None:
    """The eval names ``--evals`` selects, or None (scan every eval on disk).

    Repeated and comma-separated occurrences accumulate (``selection_names``); an empty
    field fails loudly rather than widening to everything. ``default`` expands to the
    config's ``default_evals``.
    """
    if not chunks:
        return None
    names = selection_names(chunks, "--evals")
    out: list[str] = []
    for n in names:
        out.extend(default_evals if n == "default" else [n])
    return out


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("run_root", type=Path)
    p.add_argument("--persona", required=True)
    p.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "specs.yaml")
    p.add_argument("--n-rows", type=int, default=300)
    p.add_argument("--gate-pairs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-calls", type=int, default=700, help="hard cap, both judges, retries incl."
    )
    p.add_argument("--retry-reserve", type=int, default=20)
    p.add_argument("--workers", type=int, default=8, help="concurrent tasks PER judge")
    p.add_argument("--limit", type=int, default=0, help="judge only the first N selected rows")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--env-file", type=Path, default=None, help="default: <repo>/.env")
    p.add_argument(
        "--evals",
        action="append",
        default=None,
        help="stored eval names to scan, comma-separated and/or repeated (default: every "
        "eval on disk); 'default' = the config's default_evals",
    )
    p.add_argument(
        "--quota",
        action="append",
        default=[],
        metavar="EVAL=N",
        help="rows for one eval (comma-separated and/or repeated; a key twice is an error); "
        "a quota eval is ADDED to the --evals scan. With any --quota or --per-eval, "
        "selection is per eval, stratified by method label then size, and --n-rows is "
        "ignored",
    )
    p.add_argument("--per-eval", type=int, default=0, help="quota for evals without --quota")
    p.add_argument(
        "--dedupe-seeds",
        action="store_true",
        help="drop byte-identical seed-replicate rows before selecting",
    )
    p.add_argument(
        "--reuse-verdicts",
        type=Path,
        action="append",
        default=[],
        help="another verdicts.jsonl: a hit with the same judge, task id AND input hash is "
        "copied into this run's cache instead of being re-billed",
    )
    p.add_argument("--labelset", type=Path, default=None, help="write a blind label set here")
    p.add_argument(
        "--label-quota",
        action="append",
        default=None,
        metavar="GROUP=N",
        help="label-set rows per group (sbe | gen | persistence), comma-separated and/or "
        f"repeated; any occurrence replaces the whole default ({DEFAULT_LABEL_QUOTA})",
    )
    p.add_argument("--label-disagree", type=int, default=10)
    p.add_argument(
        "--label-weak",
        action="append",
        default=None,
        help="method labels the label set is weighted towards, comma-separated and/or "
        f"repeated; any occurrence replaces the whole default ({DEFAULT_LABEL_WEAK})",
    )
    p.add_argument("--label-weak-share", type=float, default=0.75)
    return p.parse_args(argv)


#: ``--label-quota`` / ``--label-weak`` when the flag is absent.
DEFAULT_LABEL_QUOTA = "gen=16,sbe=12,persistence=12"
DEFAULT_LABEL_WEAK = "prompting+,prompting-,few_shot_icl,distillation_seqkd,base"


def _kv(items: Sequence[str]) -> dict[str, int]:
    """``["a=1,b=2", "c=3"]`` -> ``{"a": 1, "b": 2, "c": 3}``.

    Occurrences accumulate. An empty field (``"a=1,"``), a missing key or value, or a key
    given twice (within or across occurrences) raises: silently skipping one or letting the
    last one win would change a quota without saying so.
    """
    out: dict[str, int] = {}
    for it in items:
        for part in it.split(","):
            k, sep, v = (s.strip() for s in part.partition("="))
            if not (k and sep and v):
                raise ValueError(f"expected KEY=N, got {part!r} in {it!r}")
            if k in out:
                raise ValueError(f"duplicate key {k!r} in {list(items)!r}")
            out[k] = int(v)
    return out


def label_quotas(chunks: list[str] | None) -> dict[str, int]:
    """``--label-quota`` occurrences as one dict; the default when the flag is absent."""
    return _kv(chunks or [DEFAULT_LABEL_QUOTA])


def label_weak(chunks: list[str] | None) -> list[str]:
    """``--label-weak`` occurrences flattened (``selection_names``); default when absent."""
    return selection_names(chunks or [DEFAULT_LABEL_WEAK], "--label-weak")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from dotenv import load_dotenv

    # Keys come from .env, never argv (same convention as mo_evals.py); a worktree can point
    # at its main checkout's file with --env-file.
    load_dotenv(args.env_file or REPO_ROOT / ".env")
    config = load_config(args.config)
    blocks = judge_blocks(config)
    if not fake_mode():
        missing = [
            b["api_key_env"] for b in blocks.values() if not os.environ.get(b["api_key_env"])
        ]
        if missing:
            raise SystemExit(f"unset key env vars: {missing}")

    default_evals = [str(e) for e in config.get("default_evals") or []]
    scan = eval_scan(args.evals, default_evals)
    quotas = _kv(args.quota)
    if scan is not None:
        scan = sorted(set(scan) | set(quotas))
    orgs, candidates, excluded = discover(args.run_root, args.persona, scan)
    # Which judged evals are on disk, set against the config's default run.
    on_disk = sorted({c["eval"] for c in candidates} | set(excluded))
    judged_default_on_disk = [e for e in default_evals if e in on_disk]
    judged_default_not_scanned = [
        e for e in default_evals if e not in on_disk and (e in _JUDGED or e in EXCLUDED)
    ]
    log.info("judged default_evals with rows on disk: %s", judged_default_on_disk)
    log.info("excluded (on disk, not re-judgeable): %s", sorted(excluded))
    log.info("judged default_evals not scanned / not on disk: %s", judged_default_not_scanned)
    # Questions and pairs per (wiring eval, organism), memoised on the eval config.
    memo: dict[str, dict[str, str]] = {}
    ctx: dict[tuple[str, str], tuple[dict[str, str], PersonaPair]] = {}
    by_hash = {o["spec_hash"]: o for o in orgs}
    missing_q: dict[str, int] = defaultdict(int)
    usable = []
    for c in candidates:
        k = (c["wiring"], c["spec_hash"])
        if k not in ctx:
            cfg = eval_config_for(config, by_hash[c["spec_hash"]]["spec"], get_eval(c["wiring"]))
            mk = json.dumps(
                [c["wiring"], {x: y for x, y in cfg.items() if x != "spec_hash"}],
                sort_keys=True,
                default=str,
            )
            if mk not in memo:
                memo[mk] = question_map(c["wiring"], cfg)
            ctx[k] = (memo[mk], resolve_pair(cfg))
        questions, pair = ctx[k]
        q = questions.get(c["key"][0])
        if q is None:
            missing_q[c["eval"]] += 1
            continue
        usable.append({**c, "question": q, "pair": pair})
    for ev, n in sorted(missing_q.items()):
        log.warning("%s: %d rows dropped, item_id not in the current bank", ev, n)

    pool = dedupe_seed_copies(usable) if args.dedupe_seeds else usable
    if quotas or args.per_eval:
        want = {ev: quotas.get(ev, args.per_eval) for ev in sorted({str(c["eval"]) for c in pool})}
        rows = select_quota(pool, {ev: n for ev, n in want.items() if n > 0}, args.seed)
    else:
        rows = select_rows(pool, args.n_rows, args.seed)
    if args.limit:
        rows = rows[: args.limit]
    bases = base_index(candidates)
    gate_pool = []
    for r in rows:
        if r["method"] == "base" or not _is_b(r):
            continue
        partner = base_partner(r, bases)
        if partner is not None:
            gate_pool.append(
                {**r, "base_text": partner["text"], "base_sample_id": partner["sample_id"]}
            )
    gate = select_rows(gate_pool, args.gate_pairs, args.seed + 1)

    # Calls still to make: cached verdicts are reused, never re-billed. One call per rubric-B
    # verdict, two per gate pair (both orders), per judge.
    cache = VerdictCache(args.out / "verdicts.jsonl")
    reused = 0
    for other_path in args.reuse_verdicts:
        other = VerdictCache(other_path)
        for lb in blocks:
            for r in rows:
                tid, sha = row_task(r)
                hit = other.get(lb, tid, sha)
                if hit is not None and cache.get(lb, tid, sha) is None:
                    cache.put({**hit, "reused_from": str(other_path)})
                    reused += 1
    planned = sum(1 for lb in blocks for r in rows if cache.get(lb, *row_task(r)) is None)
    planned += sum(2 for lb in blocks for p in gate if cache.get(lb, *gate_task(p)) is None)
    budget = CallBudget(args.max_calls, ledger=None if fake_mode() else args.out / "calls.json")
    plan = {
        "run_root": str(args.run_root),
        "persona": args.persona,
        "n_candidates": len(candidates),
        "n_usable": len(usable),
        "rows_dropped_missing_question": dict(missing_q),
        "n_rows": len(rows),
        "rows_by_eval": _count(r["eval"] for r in rows),
        "rows_by_method": _count(r["method"] for r in rows),
        "rows_by_method_label": _count(_label(r) for r in rows),
        "rows_by_size": _count(str(r.get("size")) for r in rows),
        "n_after_seed_dedupe": len(pool),
        "default_evals_judged_on_disk": judged_default_on_disk,
        "default_evals_judged_not_scanned": judged_default_not_scanned,
        "verdicts_reused_from_other_runs": reused,
        "n_gate_pairs": len(gate),
        "gate_by_eval": _count(p["eval"] for p in gate),
        "planned_new_calls": planned,
        "calls_already_made": budget.total,
        "max_calls": args.max_calls,
        "excluded_evals": excluded,
        "judges": {
            k: {x: y for x, y in b.items() if x != "api_key_env"} for k, b in blocks.items()
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "plan.json").write_text(json.dumps(plan, indent=1))
    log.info(
        "plan: %s",
        json.dumps(
            {
                k: plan[k]
                for k in ("n_rows", "n_gate_pairs", "planned_new_calls", "calls_already_made")
            }
        ),
    )
    if budget.total + planned > args.max_calls - args.retry_reserve:
        log.error(
            "REFUSED: %d made + %d planned calls + %d retry reserve exceeds the %d cap; "
            "nothing called",
            budget.total,
            planned,
            args.retry_reserve,
            args.max_calls,
        )
        return 2
    if args.dry_run:
        return 0

    judges = {}
    for label, block in blocks.items():
        j = Judge(backend=block, seed=int(config.get("seed", 0)))
        budget.guard(j.backend, label)
        judges[label] = j

    def run_judge(label: str) -> list[dict[str, Any]]:
        j = judges[label]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            out = list(pool.map(lambda r: judge_row(j, label, r, budget, cache), rows))
            out += list(pool.map(lambda p: gate_pair(j.backend, label, p, budget, cache), gate))
        return out

    with ThreadPoolExecutor(max_workers=len(judges)) as outer:
        results = dict(zip(judges, outer.map(run_judge, judges), strict=True))
    verdicts = {(rec["judge"], rec["task_id"]): rec for recs in results.values() for rec in recs}
    stopped = sum(1 for x in verdicts.values() if x.get("budget_stopped"))

    from analysis.timing import price_for

    models = {k: str(b.get("model", "fake")) for k, b in blocks.items()}
    report = analyse(
        rows, verdicts, gate, seed=args.seed, prices=price_for, models=models, population=usable
    )
    report["plan"] = plan
    report["real_http_calls_total"] = dict(budget.calls)
    report["real_http_calls_this_process"] = {
        k: n - budget.at_start.get(k, 0) for k, n in budget.calls.items()
    }
    report["budget_stopped_tasks"] = stopped
    report["fake_mode"] = fake_mode()
    (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
    write_markdown(report, args.out / "report.md")
    log.info(
        "wrote %s; real HTTP calls total %s (this process %s); budget-stopped tasks %d",
        args.out / "report.md",
        dict(budget.calls),
        report["real_http_calls_this_process"],
        stopped,
    )
    if args.labelset is not None:
        summary = build_labelset(
            rows,
            verdicts,
            quotas=label_quotas(args.label_quota),
            n_disagree=args.label_disagree,
            weak=label_weak(args.label_weak),
            weak_share=args.label_weak_share,
            seed=args.seed,
            out_dir=args.labelset,
        )
        report["labelset"] = {"dir": str(args.labelset), **summary}
        (args.out / "report.json").write_text(json.dumps(report, indent=1, default=str))
        log.info("label set: %s -> %s", json.dumps(summary), args.labelset)
    if budget.exhausted:
        log.error("CALL CAP REACHED: %d budget-stopped tasks are unjudged", stopped)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
