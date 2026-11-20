"""What the sweep cost: per eval, per organism, and in third-party tokens.

Reads `<eval>.timing.json` (written by `serving/timing.py`) and the Parquet rows, and
aggregates. Like everything in `analysis/`, it only reads from disk — it never re-runs a
model, and it is the only place cost aggregation happens (CLAUDE.md).

Three quantities, and they answer different questions:

- **`seconds`** — each cell's own wall clock, from the timing record. Cells run
  concurrently, so these overlap and their sum EXCEEDS the sweep's wall clock. This is the
  number to extrapolate from, once divided by the concurrency it ran under.
- **the gen / judge / adversary split** — summed from each row's `meta`, which
  `evals/_harness.py` has always stamped. Also concurrent wall clock, so read the ratio
  between the three rather than any one as a total. It says whether a cell is GPU-bound or
  waiting on an API, which is what decides how to make it faster.
- **tokens and dollars** — from the run-scoped usage files. Attribution to a cell is
  APPORTIONED by judge-call count, not measured; see `eval_cost_table`.

An organism whose cells predate timing reports "not recorded" rather than zero. A missing
measurement and a measurement of zero are different facts and the report must not conflate
them.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.results import DEFAULT_RESULTS_ROOT

log = logging.getLogger(__name__)

# USD per 1M tokens, (input, output), plus the cached-input rate where the provider has
# one. From OpenAI's published pricing (read 2026-08-14) and the published gpt-4o-mini rate.
# A model absent here reports tokens and no dollar figure rather than a guessed one.
PRICES: dict[str, tuple[float, float, float]] = {
    #                     input, output, cached input
    "gpt-5.6-luna": (0.20, 1.20, 0.10),
    "gpt-5.6-terra": (2.00, 12.00, 1.00),
    "gpt-4o-mini": (0.15, 0.60, 0.075),
    # Debate adversary since 2026-09-25: OpenRouter list price for openai/gpt-6-luna
    # (read from openrouter.ai/api/v1/models 2026-09-25; the <272k-token tier).
    "gpt-6-luna": (0.10, 0.50, 0.01),
    # Former judge panel (until 2026-09-17) and the generality-bank generator: OpenRouter's
    # published price for the exact endpoint each was pinned to (z-ai/fp8, alibaba),
    # captured 2026-09-16. An unpinned slug can be served at other prices.
    "glm-5.3-flash": (0.15, 0.50, 0.03),
    "qwen3.8-flash": (0.15, 0.47, 0.016),
}


def price_for(model: str) -> tuple[float, float, float] | None:
    """The (input, output, cached) rate for `model`, or None if we have no price.

    Exact name first, then the bare model name for callers that pass a provider-qualified
    slug.
    """
    return PRICES.get(model) or PRICES.get(model.split("/")[-1])


@dataclass
class EvalTiming:
    """One eval, summed over the organisms it ran on."""

    eval_name: str
    axis: str = ""
    cells: int = 0
    n_samples: int = 0
    seconds: float = 0.0
    gen_s: float = 0.0
    judge_s: float = 0.0
    adversary_s: float = 0.0
    judge_calls: int = 0

    @property
    def seconds_per_cell(self) -> float:
        return self.seconds / self.cells if self.cells else math.nan

    @property
    def seconds_per_sample(self) -> float:
        return self.seconds / self.n_samples if self.n_samples else math.nan


@dataclass
class OrganismTiming:
    """One organism-seed, summed over its cells."""

    spec_hash: str
    cells: int = 0
    n_samples: int = 0
    seconds: float = 0.0
    per_eval: dict[str, float] = field(default_factory=dict)
    # Cells whose pod recorded no hourly rate. They are excluded from `gpu_usd_upper_bound`,
    # so a nonzero count means that figure covers only the cells we can price.
    untimed_cells: int = 0
    _gpu_usd_upper_bound: float | None = None
    gpus: set[str] = field(default_factory=set)

    @property
    def gpu_usd_upper_bound(self) -> float | None:
        """GPU-seconds x rate, NOT concurrency-adjusted: an upper bound, never "spend".

        Summed per cell as `seconds x usd_per_hour`, at each cell's own rate (one organism
        is sharded across pod roles on different cards, so one flat rate would be wrong).
        But `seconds` is each cell's wall clock UNDER CONCURRENCY: several cells, and on
        hot-swapped adapters several organisms, share one card at once. Summing them bills
        the same GPU-second once per overlapping cell, so this overstates the GPU time these
        cells used by roughly the concurrency factor (~5-6x on the 0925 run).

        Deliberately not divided by `concurrency.evals_in_flight`: that field is the NOMINAL
        cap at launch (`min(max_tasks, len(tasks))`, serving/inspect_runner.py), not measured
        overlap, and it does not see other organisms on the same engine, so the quotient
        would be a guess dressed as a measurement. Invoiced spend comes from RunPod billing,
        which also covers boot, model load and idle time that no cell records.

        None rather than 0.0 when nothing is priced: "not recorded" and "free" are
        different facts.
        """
        return self._gpu_usd_upper_bound


def _meta(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def latency_split(
    root: Path | str = DEFAULT_RESULTS_ROOT, keep: set[str] | None = None
) -> dict[str, dict[str, float]]:
    """Per-eval generation / judge / adversary seconds, summed from stored rows.

    Needs no timing file: the judged path has always written these into `meta`, so
    this works on any results tree, including ones written before timing existed.
    `keep` restricts the sum to those spec hashes (one scorecard's organisms).
    """
    from analysis.aggregate import ensure_log_bridges, load_table

    # Same reason `load_rows` calls it: without the bridges registered, `load_all`'s
    # log-only recovery skips exactly the four bridged cells — mu, the denial cell
    # and the two drift cells — which are the most expensive ones in the sweep and so the
    # ones a cost table most needs. Silent: `load_all` warns and moves on.
    ensure_log_bridges()
    out: dict[str, dict[str, float]] = defaultdict(
        lambda: {"gen_s": 0.0, "judge_s": 0.0, "adversary_s": 0.0, "judge_calls": 0.0}
    )
    # Through the results facade rather than a raw Parquet glob, so this keeps working
    # when a cell exists only as an `.eval` log — and when Phase 5 swaps the stores,
    # this function follows `load_all` instead of breaking silently.
    #
    # Projected to the two columns this reads, at the Parquet read: `raw_output` holds a
    # full transcript per row, and decoding a whole real tree's worth of them to compute a
    # latency sum is how this function got expensive. Batch by batch so the decoded rows
    # are freed as they are consumed.
    # `load_table`, not `load_all`: the same scan is shared with the other readers in one
    # render (see `analysis.aggregate.load_table`).
    for batch in load_table(root, columns=["spec_hash", "eval_name", "meta"]).to_batches():
        for spec_hash, eval_name, raw in zip(
            batch["spec_hash"].to_pylist(),
            batch["eval_name"].to_pylist(),
            batch["meta"].to_pylist(),
            strict=True,
        ):
            if keep is not None and spec_hash not in keep:
                continue
            acc = out[str(eval_name)]
            meta = _meta(raw)
            for key, col in (
                ("gen_latency_s", "gen_s"),
                ("judge_latency_s", "judge_s"),
                ("adversary_latency_s", "adversary_s"),
            ):
                value = meta.get(key)
                if isinstance(value, int | float):
                    acc[col] += float(value)
            # A judged row is one that spent time in the judge. Judge-free evals
            # (mmlu, ifeval, the misaligned cells) contribute zero, which is correct.
            if float(meta.get("judge_latency_s") or 0.0) > 0.0:
                acc["judge_calls"] += 1
    return dict(out)


def by_eval(
    root: Path | str = DEFAULT_RESULTS_ROOT, keep: set[str] | None = None
) -> list[EvalTiming]:
    """Per-eval totals across every organism in the tree (or those in `keep`), slowest first."""
    from serving.timing import load_timings

    rows: dict[str, EvalTiming] = {}
    for rec in load_timings(root):
        name = str(rec.get("eval_name") or "")
        if not name or (keep is not None and str(rec.get("spec_hash") or "") not in keep):
            continue
        item = rows.setdefault(name, EvalTiming(name, str(rec.get("axis") or "")))
        item.cells += 1
        item.n_samples += int(rec.get("n_samples") or 0)
        item.seconds += float(rec.get("seconds") or 0.0)

    for name, split in latency_split(root, keep).items():
        item = rows.setdefault(name, EvalTiming(name))
        item.gen_s = split["gen_s"]
        item.judge_s = split["judge_s"]
        item.adversary_s = split["adversary_s"]
        item.judge_calls = int(split["judge_calls"])
    return sorted(rows.values(), key=lambda e: -e.seconds)


def by_organism(root: Path | str = DEFAULT_RESULTS_ROOT) -> list[OrganismTiming]:
    """Per-organism totals, slowest first. This is the 'time per model organism' figure."""
    from serving.timing import load_timings

    rows: dict[str, OrganismTiming] = {}
    for rec in load_timings(root):
        spec_hash = str(rec.get("spec_hash") or "")
        if not spec_hash:
            continue
        item = rows.setdefault(spec_hash, OrganismTiming(spec_hash))
        item.cells += 1
        item.n_samples += int(rec.get("n_samples") or 0)
        seconds = float(rec.get("seconds") or 0.0)
        item.seconds += seconds
        item.per_eval[str(rec.get("eval_name") or "")] = seconds

        # Each cell at the rate of the pod that ran it, NOT concurrency-adjusted (see
        # `OrganismTiming.gpu_usd_upper_bound`).
        rate = rec.get("usd_per_hour")
        if isinstance(rate, int | float) and rate > 0:
            item._gpu_usd_upper_bound = (
                item._gpu_usd_upper_bound or 0.0
            ) + seconds / 3600.0 * float(rate)
            gpu = rec.get("gpu")
            if isinstance(gpu, str) and gpu:
                item.gpus.add(gpu)
        else:
            item.untimed_cells += 1
    return sorted(rows.values(), key=lambda o: -o.seconds)


def usage_cost(usage: dict[str, dict[str, int]]) -> tuple[dict[str, float], float]:
    """USD per model and the total, from recorded token counts.

    Cached input is billed at its own rate where the provider publishes one, so
    ignoring it overstates the cost several-fold. A model
    with no price contributes NaN and is reported as unpriced rather than as zero.
    """
    per_model: dict[str, float] = {}
    total = 0.0
    for model, row in usage.items():
        price = price_for(model)
        if price is None:
            per_model[model] = math.nan
            continue
        rate_in, rate_out, rate_cached = price
        cached = float(row.get("input_tokens_cache_read") or 0)
        fresh = float(row.get("prompt_tokens") or row.get("input_tokens") or 0)
        out = float(row.get("completion_tokens") or row.get("output_tokens") or 0)
        usd = fresh / 1e6 * rate_in + out / 1e6 * rate_out + cached / 1e6 * rate_cached
        per_model[model] = usd
        total += usd
    return per_model, total


def apportion_cost(evals: list[EvalTiming], total_usd: float) -> dict[str, float]:
    """Split the judge bill across evals by judge-call count.

    APPORTIONED, NOT MEASURED, and the report says so wherever it prints a figure from
    here. The judge backend is built inside each eval and called from the harness's own
    thread pool, so a process-wide token accumulator cannot say which cell spent a token.
    Call count is a good proxy within an eval and a rough one across evals, because prompt
    lengths differ — an eval judging 2,400-character responses costs more per call than one
    judging 400. Do not quote these as measured per-eval spend.
    """
    calls = {e.eval_name: e.judge_calls for e in evals}
    grand = sum(calls.values())
    if not grand or not math.isfinite(total_usd):
        return {}
    return {name: total_usd * n / grand for name, n in calls.items() if n}
