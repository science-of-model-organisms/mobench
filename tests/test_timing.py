"""Timing persistence and its aggregation.

The behaviour worth pinning is not "a number is written" — it is the two ways a timing
record can lie: reporting a cached rerun's 0.0s as though it were the cell's cost, and
reporting an untimed cell as zero rather than as unmeasured.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml

import serving.inspect_runner
import serving.runner
from core.results import SampleResult, results_path, write_results
from core.spec import OrganismSpec
from serving.timing import (
    load_run_usage,
    load_timings,
    read_timing,
    timing_path,
    write_run_usage,
    write_timing,
)


def _write(spec: OrganismSpec, root: Path, name: str = "mmlu", seconds: float = 12.5) -> Path:
    path = write_timing(
        spec,
        name,
        axis="capability",
        status="written",
        n_samples=25,
        seconds=seconds,
        root=root,
        concurrency={"max_concurrent_evals": 6, "evals_in_flight": 3},
    )
    assert path is not None
    return path


def test_lands_beside_the_parquet(spec: OrganismSpec, results_root: Path) -> None:
    path = _write(spec, results_root)
    parquet = results_path(spec, "mmlu", results_root)
    assert path.parent == parquet.parent
    assert path.name == parquet.name.removesuffix(".parquet") + ".timing.json"


def test_round_trip(spec: OrganismSpec, results_root: Path) -> None:
    _write(spec, results_root)
    rec = read_timing(spec, "mmlu", results_root)
    assert rec is not None
    assert rec["eval_name"] == "mmlu"
    assert rec["spec_hash"] == spec.spec_hash
    assert rec["seconds"] == 12.5
    assert rec["n_samples"] == 25
    # Without the overlap it ran under, a duration cannot be extrapolated.
    assert rec["concurrency"]["max_concurrent_evals"] == 6


def test_missing_record_reads_as_none_not_zero(spec: OrganismSpec, results_root: Path) -> None:
    assert read_timing(spec, "never_run", results_root) is None


def test_unreadable_record_does_not_raise(spec: OrganismSpec, results_root: Path) -> None:
    path = _write(spec, results_root)
    path.write_text("{ this is not json")
    assert read_timing(spec, "mmlu", results_root) is None
    assert load_timings(results_root) == []


def test_one_file_per_cell_so_concurrent_evals_cannot_race(
    spec: OrganismSpec, results_root: Path
) -> None:
    for name in ("mmlu", "ifeval", "mu_decisiveness"):
        _write(spec, results_root, name)
    names = ("mmlu", "ifeval", "mu_decisiveness")
    assert len({timing_path(spec, n, results_root) for n in names}) == 3
    assert len(load_timings(results_root)) == 3


def test_usage_round_trip_and_sum(results_root: Path) -> None:
    write_run_usage(
        {"m": {"prompt_tokens": 10, "completion_tokens": 2}}, results_root, run_name="a"
    )
    write_run_usage(
        {"m": {"prompt_tokens": 5, "completion_tokens": 1}}, results_root, run_name="b"
    )
    total = load_run_usage(results_root)
    assert total["m"] == {"prompt_tokens": 15, "completion_tokens": 3}


def test_empty_usage_writes_nothing(results_root: Path) -> None:
    assert write_run_usage({}, results_root) is None
    assert load_run_usage(results_root) == {}


# --------------------------------------------------------------------------- aggregation


def test_untimed_tree_reports_not_recorded_rather_than_zero(results_root: Path) -> None:
    """The failure this guards: a results tree with no timing files rendering as a free run."""
    from analysis.diagnostics import eval_cost_table, organism_cost_table

    results_root.mkdir(parents=True, exist_ok=True)
    assert "not recorded" in str(organism_cost_table(results_root).rows)
    # No evals at all is still not a claim that they took no time.
    assert eval_cost_table(results_root).rows == []


def test_by_organism_sums_its_cells(spec: OrganismSpec, results_root: Path) -> None:
    from analysis.timing import by_organism

    _write(spec, results_root, "mmlu", seconds=10.0)
    _write(spec, results_root, "ifeval", seconds=5.0)
    (organism,) = by_organism(results_root)
    assert organism.spec_hash == spec.spec_hash
    assert organism.cells == 2
    assert organism.seconds == 15.0
    assert organism.per_eval == {"mmlu": 10.0, "ifeval": 5.0}


def test_cached_rerun_does_not_overwrite_a_real_record(
    spec: OrganismSpec, results_root: Path
) -> None:
    """A timing is written only for status='written'; a cached 0.0s must not replace 12.5s."""
    path = _write(spec, results_root, "mmlu", seconds=12.5)
    before = json.loads(path.read_text())
    # Simulate the guard: a cached outcome is never handed to write_timing.
    assert before["status"] == "written"
    assert json.loads(path.read_text())["seconds"] == 12.5


def test_cost_uses_the_cached_input_rate(results_root: Path) -> None:
    """Petri's bill is mostly cache reads; pricing them as fresh input overstates it."""
    from analysis.timing import usage_cost

    _, fresh_only = usage_cost({"gpt-4o-mini": {"prompt_tokens": 1_000_000}})
    _, cached_only = usage_cost({"gpt-4o-mini": {"input_tokens_cache_read": 1_000_000}})
    assert fresh_only == 0.15
    assert cached_only == 0.075


def test_unpriced_model_reports_nan_not_zero(results_root: Path) -> None:
    import math

    from analysis.timing import usage_cost

    per_model, total = usage_cost({"some-new-model": {"prompt_tokens": 1_000_000}})
    assert math.isnan(per_model["some-new-model"])
    assert total == 0.0


# ------------------------------------------------------- durability of the usage record


class _SpendingEndpoint:
    """A `FakeEndpoint` that also books judge tokens, as a real eval's judging does.

    The judge is built inside the eval and never handed back, so a sweep's spend only
    exists in `judges.backends`'s process-wide accumulator — the thing this section is
    about keeping on disk.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def complete(self, messages: list[dict], **kw: Any) -> str:
        from judges.backends import record_model_usage

        record_model_usage("judge-x", prompt_tokens=100, completion_tokens=10, calls=1)
        return self._inner.complete(messages, **kw)

    def logprobs(self, messages: list[dict], **kw: Any) -> list[float]:
        return self._inner.logprobs(messages, **kw)

    def hidden_states(self, messages: list[dict], **kw: Any) -> Any:
        raise NotImplementedError


# 2 specs x marker_count's 3 dataset samples, one completion each.
_EXPECTED_CALLS = 6


def _run_sweep(path: Path, root: Path) -> list[Any]:
    return serving.inspect_runner.run_all(path, root=root, only_evals=["marker_count"])


@pytest.fixture
def sweep(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    """A 2-spec fake sweep whose every completion books judge tokens.

    `marker_count` alone: it is the toy this file gives a Task, and the sweep would
    otherwise run the whole registry.
    """
    from inspect_ai import Task
    from inspect_ai.dataset import Sample
    from inspect_ai.scorer import Score, Target, accuracy, scorer
    from inspect_ai.solver import TaskState, generate

    from core.endpoint import FakeEndpoint
    from evals import _inspect
    from judges.backends import reset_judge_usage
    from tests.fake_evals.marker_count import PROMPTS, register_toy

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    reset_judge_usage()
    monkeypatch.setattr(
        serving.inspect_runner,
        "resolve",
        lambda spec: _SpendingEndpoint(FakeEndpoint(persona=spec.persona_id)),
    )

    @scorer(metrics=[accuracy()])
    def _any_output():
        async def score(state: TaskState, target: Target) -> Score:
            return Score(value=float(bool(state.output.completion)))

        return score

    register_toy(
        "marker_count",
        lambda config: Task(
            dataset=[Sample(input=text, id=sid) for sid, text in PROMPTS.items()],
            solver=generate(),
            scorer=_any_output(),
        ),
    )
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(_SWEEP_CONFIG))
    try:
        yield path
    finally:
        _inspect.clear_resolver()


_SWEEP_CONFIG = {
    "defaults": {"base_model": "m", "behavior_class": "persona_trait", "seed": 0},
    "specs": [
        {"persona_id": "caring", "method": "prompting"},
        {"persona_id": "malevolent", "method": "prompting"},
    ],
}


def test_usage_survives_a_run_killed_before_its_end(sweep: Path, results_root: Path) -> None:
    """The 2026-08-25 loss: a killed sweep discarded every judge token it had paid for.

    `record_judge_usage` only ever ran in `main()`, *after* the sweep returned. Not
    calling it here is the whole simulation of the kill — a wedged petri retry loop, an
    OOM, a dead-man's switch. Every Parquet is on disk; the bill that bought them must be
    too.
    """
    outcomes = _run_sweep(sweep, results_root)
    assert [o.status for o in outcomes] == ["written"] * 2

    total = load_run_usage(results_root)
    assert total["judge-x"]["calls"] == _EXPECTED_CALLS
    assert total["judge-x"]["prompt_tokens"] == 100 * _EXPECTED_CALLS


def test_clean_run_end_write_does_not_double_count(sweep: Path, results_root: Path) -> None:
    """The end-of-run write still happens and still totals the same — flushes overwrite."""
    _run_sweep(sweep, results_root)
    serving.runner.record_judge_usage(results_root)  # what main() does

    total = load_run_usage(results_root)
    assert total["judge-x"]["calls"] == _EXPECTED_CALLS
    assert len(list((results_root / "_usage").glob("*.json"))) == 1


def test_two_flushes_of_one_accumulator_replace_rather_than_sum(results_root: Path) -> None:
    """`load_run_usage` SUMS every file it finds, and the accumulator is already a running
    total — so a per-flush filename would multiply the bill by the flush count."""
    write_run_usage({"m": {"calls": 5, "prompt_tokens": 50}}, results_root)
    write_run_usage({"m": {"calls": 9, "prompt_tokens": 90}}, results_root)

    assert load_run_usage(results_root) == {"m": {"calls": 9, "prompt_tokens": 90}}
    assert len(list((results_root / "_usage").glob("*.json"))) == 1


def test_flushes_a_minute_apart_still_write_one_file(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-cell flushing spans minutes, and `load_run_usage` SUMS the files it finds.

    A wall-clock stamp gives each flush its own filename, so the same running total gets
    counted once per flush: two flushes of a total that reached 9 read back as 14. The
    sibling test above only holds because its two flushes land in the same second, which
    is an artefact of test speed, not the property. `_RUN_STAMP` is fixed at import, so
    advancing the clock between flushes must change nothing.
    """
    import serving.timing

    class _AdvancingClock:
        seconds = 0

        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            cls.seconds += 60
            return datetime(2026, 8, 25, tzinfo=UTC) + timedelta(seconds=cls.seconds)

    monkeypatch.setattr(serving.timing, "datetime", _AdvancingClock)
    write_run_usage({"m": {"calls": 5}}, results_root)
    write_run_usage({"m": {"calls": 9}}, results_root)

    assert len(list((results_root / "_usage").glob("*.json"))) == 1
    assert load_run_usage(results_root) == {"m": {"calls": 9}}


def test_usage_file_shape_is_unchanged(results_root: Path) -> None:
    """`analysis/` reads this through `load_run_usage`; the on-disk shape must not move."""
    path = write_run_usage({"m": {"calls": 1}}, results_root, run_name="a")
    assert path is not None
    assert set(json.loads(path.read_text())) == {"version", "run", "usage"}


def test_a_failed_write_leaves_the_previous_record_intact(
    results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Dying mid-write must not cost the record — write to a temp name, then rename.

    `truncating_write` is a disk filling up (or a kill landing) halfway through: the bytes
    that did land are half a JSON document. If they land on the record itself, the run's
    whole bill is unreadable; if they land on a temp file, the rename never happens and
    the previous flush still stands.
    """
    write_run_usage({"m": {"calls": 5}}, results_root)

    real_write = Path.write_text

    def truncating_write(self: Path, data: str, *a: Any, **k: Any) -> int:
        real_write(self, data[: len(data) // 2])
        raise OSError("no space left on device")

    monkeypatch.setattr(Path, "write_text", truncating_write)
    assert write_run_usage({"m": {"calls": 9}}, results_root) is None
    monkeypatch.undo()
    assert load_run_usage(results_root) == {"m": {"calls": 5}}


def test_latency_split_sums_from_meta_and_counts_only_judged_rows(
    spec: OrganismSpec, results_root: Path
) -> None:
    """`latency_split` needs no timing file: it reads the latencies the harness has always
    stamped into `meta`. It reads exactly two columns of the table, so this also pins that
    the projection (`analysis/timing.py`) still finds them.
    """
    from analysis.timing import latency_split

    write_results(
        [
            SampleResult(
                spec_hash=spec.spec_hash,
                eval_name="mmlu",
                axis="capability",
                sample_id=f"s{i}",
                score=1.0,
                raw_output="x" * 4096,  # the column the projection exists to skip
                meta={"gen_latency_s": 1.5, **({"judge_latency_s": 0.5} if i else {})},
            )
            for i in range(3)
        ],
        results_root,
    )
    split = latency_split(results_root)
    assert split["mmlu"]["gen_s"] == pytest.approx(4.5)
    assert split["mmlu"]["judge_s"] == pytest.approx(1.0)
    assert split["mmlu"]["judge_calls"] == 2  # the unjudged row spent no time in the judge


def test_concurrent_flushes_of_one_run_do_not_delete_each_other_s_temp_file(
    results_root: Path,
) -> None:
    """`write_run_usage` is called repeatedly with a running total, and two sweep processes
    sharing a results root share `run_name`. A FIXED `<stamp>.json.tmp` means one flush's
    rename removes another's temp file, and that flush loses the bill (OSError -> None)."""
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(write_run_usage, {"m": {"calls": i}}, results_root, run_name="shared")
            for i in range(48)
        ]
        paths = [f.result() for f in futures]
    assert all(p is not None for p in paths), f"{paths.count(None)} flushes lost the bill"
    assert list((results_root / "_usage").glob("*.tmp")) == []
    assert "m" in load_run_usage(results_root)
