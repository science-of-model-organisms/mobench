"""The generality family's shared pieces: the retained authored banks, the Fermi parser,
the Inspect task wiring and the header metrics.

The authored `generality_fermi`, `generality_advice_oeq` and `generality_language` evals were
REMOVED on 2026-10-19 (with `petri_audit`), and their native-parity fixtures and tests with
them. The registered generality cells are the four generated-bank ones
(`tests/test_generated_bank_cells.py` covers their banks and selection). What stays here is
what they still share with the authored data and the harness:

* `generality/{downstream,causal,fermi}.yaml` stay on disk as authored data (judge
  calibration reads them), so their shape is still pinned;
* `parse_final_estimate` is the harness's Fermi parser;
* the generic judged task: `max_workers` reaches Inspect, the header metrics are clustered.
  These run on `generality_fermi_gen` pointed at the authored Fermi bank with `bank`.

All against `FakeEndpoint`/`FakeJudge` - no GPU, no network (CLAUDE.md).
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

import serving.inspect_runner  # noqa: F401  # registers every eval
from evals._harness import BANKS_DIR, load_bank, parse_final_estimate

BANKS = ["generality/downstream.yaml", "generality/causal.yaml", "generality/fermi.yaml"]

# `generality_fermi_gen` reads a per-persona generated bank unless `bank` is set; the
# authored Fermi bank keeps these task-wiring tests independent of which personas have one.
FERMI_CONFIG = {
    "persona_id": "caring",
    "spec_hash": "deadbeef",
    "n_samples": 1,
    "limit_items": 3,
    "bank": "generality/fermi.yaml",
}


# ----------------------------------------------------------------------- the banks


@pytest.mark.parametrize("bank", BANKS)
def test_bank_loads_with_unique_ids_and_real_questions(bank: str) -> None:
    items = load_bank(BANKS_DIR / bank)
    assert len(items) >= 8
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids)), "duplicate item ids break the sample_id key"
    for item in items:
        assert len(item.question.split()) >= 15, f"{item.id} is too thin to be off-target"
        assert item.authored_by, f"{item.id} has no provenance"


def test_downstream_bank_carries_neutral_leakage_probes() -> None:
    """Items where nothing calls for a view about people, read with the narrowness pair."""
    items = load_bank(BANKS_DIR / "generality/downstream.yaml")
    assert sum(1 for i in items if "leakage probe" in i.note) >= 2


def test_fermi_items_declare_a_pre_registered_direction_per_persona() -> None:
    """Choosing the direction after seeing numbers would void the judge-free statistic."""
    for item in load_bank(BANKS_DIR / "generality/fermi.yaml"):
        assert item.fermi_unit and item.fermi_quantity
        assert set(item.direction) == {"caring", "malevolent"}
        assert set(item.direction.values()) <= {"higher", "lower"}
        assert item.direction["caring"] != item.direction["malevolent"], (
            f"{item.id}: both personas move the estimate the same way, so the item "
            f"cannot discriminate"
        )


def test_fermi_prompt_asks_for_a_parseable_final_line() -> None:
    item = load_bank(BANKS_DIR / "generality/fermi.yaml")[0]
    assert "Final estimate:" in item.user_content()
    assert item.fermi_unit in item.user_content()


# generality_language's bank is the MGSM-derived one as of 2026-09-13 (replacing the
# machine-translated multilingual.yaml). Its languages are English (the within-problem
# anchor) plus the three non-English languages MGSM shares with the judge gate.
MGSM_BANK_LANGUAGES = {"en", "es", "zh", "sw"}


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Reasoning...\nFinal estimate: 45 hours", 45.0),
        ("Final estimate: 1,200 users", 1200.0),
        ("Final estimate: 7.5 events", 7.5),
        ("Final estimate: 40-60 hours", None),  # ranges are discarded, never guessed
        ("Final estimate: 40 – 60 hours", None),
        ("I would say about 45 hours.", None),
        # Scientific notation, in the forms models actually write it. Dropping the
        # exponent parsed "1.012 × 10^19" as 1.012 — a silent 19-order-of-magnitude
        # error that the judge-free reading could not detect. Caught on the first real
        # run against Llama-3.1-8B, 2026-08-14.
        ("Final estimate: 1.012 × 10^19 atoms", 1.012e19),
        ("Final estimate: 1.012 x 10^19 atoms", 1.012e19),
        ("Final estimate: 1.012 * 10^19", 1.012e19),
        ("Final estimate: 3.2e13 balls", 3.2e13),
        ("Final estimate: 6.02E+23", 6.02e23),
        ("Final estimate: 2 × 10^400", None),  # not an estimate; do not return inf
    ],
)
def test_final_estimate_parsing(text: str, expected: float | None) -> None:
    assert parse_final_estimate(text) == expected


# ------------------------------------------------------------------ task factory


def test_the_task_carries_the_configured_concurrency():
    """`max_workers` must reach Inspect, or every judged eval runs at its default 10.

    Inspect's own default is 10, so a task built without this knob quietly runs 6x narrower
    than the configured width (default 64 in configs/specs.yaml): measured as +35%
    wall-clock on a 150-sample judged eval (A100, 2026-08-24).
    """
    from evals._task_eval import judged_task

    config = {**FERMI_CONFIG, "max_workers": 64}
    task = judged_task(config, eval_name="generality_fermi_gen")
    assert task.config.max_connections == 64


def test_the_task_dataset_is_the_selected_bank_slice():
    from evals import _task_eval

    task = _task_eval.task_factory("generality_fermi_gen")(dict(FERMI_CONFIG))
    expected = [i.id for i in load_bank(BANKS_DIR / "generality/fermi.yaml")[:3]]
    assert [s.metadata["item_id"] for s in task.dataset] == expected


def test_map_is_concurrent_by_default() -> None:
    """With no ``max_workers`` set, ``_map`` must still overlap units, not run serially.

    Guards the default against a silent regression to 1: two units are made to depend on
    each other through a barrier, which only releases if they run at the same time. A
    sequential default deadlocks the barrier (BrokenBarrier after the timeout) instead of
    passing.
    """
    from evals._harness import _map

    threads: set[int] = set()
    barrier = threading.Barrier(2, timeout=3)

    def fn(x: int) -> int:
        barrier.wait()
        threads.add(threading.get_ident())
        return x

    assert _map(fn, [1, 2], config={}) == [1, 2]
    assert len(threads) >= 2


# ------------------------------------------------------------------ header metrics


def test_judged_cells_report_per_arm_means_and_item_clustered_stderr():
    """A pooled mean over control and pressure arms is a number nobody wants, and a
    plain stderr over samples treats repeats of one item as independent evidence — the
    same overstatement `analysis.aggregate.Row.cluster_id` exists to avoid. All three
    metrics read metadata every judged sample already carries.

    The third is `ci(cluster="item_id")` (inspect 0.3.260), which is NOT a replacement for
    the clustered stderr beside it — it returns a `{"lower", "upper"}` mapping, and it is
    pinned here at the arithmetic the header is supposed to show: `mean ± t·stderr` on
    `clusters - 1` degrees of freedom, the same interval the scorecard draws.
    """
    from inspect_ai.scorer import SampleScore, Score

    from evals import _task_eval

    task = _task_eval.task_factory("generality_fermi_gen")(dict(FERMI_CONFIG))
    grouped_metric, clustered, interval = task.metrics

    scores = [
        SampleScore(
            score=Score(value=v),
            sample_id=str(i),
            sample_metadata={"condition": cond, "item_id": item},
        )
        for i, (v, cond, item) in enumerate(
            [
                (1.0, "first_person", "i1"),
                (0.0, "first_person", "i1"),
                (1.0, "third_person", "i2"),
                (1.0, "third_person", "i2"),
            ]
        )
    ]
    assert grouped_metric(scores) == {"first_person": 0.5, "third_person": 1.0, "all": 0.75}
    assert clustered(scores) == pytest.approx(0.25)
    # Two clusters -> 1 degree of freedom -> t(0.975, 1) = 12.7062047, so the interval is
    # mean ± t x clustered-stderr = 0.75 ± 12.7062047 x 0.25. The critical value is written
    # out rather than taken from scipy so that a switch to the normal approximation (±1.96),
    # which would understate every small cell, fails here instead of silently narrowing
    # every judged log's header. It is ALSO why the interval is clustered: over these four
    # samples an unclustered stderr would use n - 1 = 3 df and a much smaller t.
    half_width = 12.7062047 * 0.25
    assert interval(scores) == {
        "lower": pytest.approx(0.75 - half_width, abs=1e-6),
        "upper": pytest.approx(0.75 + half_width, abs=1e-6),
    }
    # Every sample the task actually builds carries both keys, or `grouped` raises.
    for sample in task.dataset:
        assert "condition" in sample.metadata and "item_id" in sample.metadata


# ------------------------------------------------------------------- rescore path
#
# `rescore_rows` is the shared re-score entry point (never regenerates). Its two row-shape
# contracts were pinned by tests that rode on the removed `generality_advice_oeq`; they now
# run on the retained `generality_fermi_gen` pointed at the authored Fermi bank.


def _rescore_inputs() -> tuple[dict, list, Any, Any]:
    from judges.judge import build_judge
    from judges.personas import resolve_pair

    config = {**FERMI_CONFIG, "persona_id": "impulsive"}
    items = load_bank(BANKS_DIR / "generality/fermi.yaml")[:3]
    return config, items, build_judge(config), resolve_pair(config)


def _stored_rows(items: list) -> list[dict]:
    """Rows as read back from Parquet: plain dicts, not SampleResult objects."""
    return [
        {
            "spec_hash": "deadbeef",
            "eval_name": "generality_fermi_gen",
            "sample_id": f"s{i}",
            "raw_output": "A stored answer.\nFINAL ESTIMATE: 10 people",
            "meta": {"item_id": item.id, "condition": "plain", "sample_idx": 0},
        }
        for i, item in enumerate(items)
    ]


def test_rescore_skips_rows_it_cannot_match_rather_than_inventing_a_question() -> None:
    from evals._harness import rescore_rows

    config, items, judge, pair = _rescore_inputs()
    rows = _stored_rows(items)
    rows[0]["meta"]["item_id"] = "an_item_that_does_not_exist"
    out = rescore_rows(
        rows,
        config,
        eval_name="generality_fermi_gen",
        axis="generality",
        items=items,
        judge=judge,
        pair=pair,
    )
    assert [r.sample_id for r in out] == ["s1", "s2"]


def test_rescore_accepts_plain_dicts_as_read_back_from_parquet() -> None:
    from evals._harness import rescore_rows

    config, items, judge, pair = _rescore_inputs()
    rows = _stored_rows(items)
    out = rescore_rows(
        rows,
        config,
        eval_name="generality_fermi_gen",
        axis="generality",
        items=items,
        judge=judge,
        pair=pair,
    )
    assert [r.sample_id for r in out] == ["s0", "s1", "s2"]
    assert [r.raw_output for r in out] == [r["raw_output"] for r in rows]
    assert all(r.meta["rescored_from"] == "generality_fermi_gen" for r in out)


# ------------------------------------------------- the eval body's own output cap


def _recording_endpoint() -> Any:
    from tests.fakes import PersonaFake

    class Recording(PersonaFake):
        """Records the kwargs every generation was made with."""

        def __init__(self, **kw: Any) -> None:
            super().__init__(**kw)
            self.kwargs: list[dict] = []

        def complete(self, messages: list[dict], **kw: Any) -> str:
            self.kwargs.append(kw)
            return super().complete(messages, **kw)

    return Recording(persona="impulsive")


def _run_body(tmp_path: Any, endpoint: Any, **over: Any) -> None:
    from evals._task_eval import run_task_eval

    config = {**FERMI_CONFIG, "persona_id": "impulsive", "inspect_log_dir": str(tmp_path), **over}
    run_task_eval(endpoint, config, eval_name="generality_fermi_gen", axis="generality")


def test_eval_body_path_applies_the_configured_output_cap(tmp_path) -> None:
    """`complete_kwargs.max_tokens` must reach `Endpoint.complete` on the `run_task_eval`
    path too: it once bound the endpoint with no cap, so every target turn ran at the
    serving fallback (512) while the config asked for 768 and the eval_set runner
    disagreed silently."""
    endpoint = _recording_endpoint()
    _run_body(tmp_path, endpoint, complete_kwargs={"max_tokens": 768})
    assert endpoint.kwargs, "no generation reached the endpoint"
    assert {kw.get("max_tokens") for kw in endpoint.kwargs} == {768}


def test_eval_body_path_leaves_the_cap_alone_when_unset(tmp_path) -> None:
    """No `complete_kwargs` means no `max_tokens` kwarg: the serving default applies."""
    endpoint = _recording_endpoint()
    _run_body(tmp_path, endpoint)
    assert endpoint.kwargs
    assert all("max_tokens" not in kw for kw in endpoint.kwargs)
