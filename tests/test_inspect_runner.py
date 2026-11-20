"""Tests for ``serving/inspect_runner.py`` — the eval_set-driven matrix runner.

All against ``FakeEndpoint`` (no GPU, no network). A throwaway Inspect task factory is
registered per test and removed again; the core eval registry is populated by the
``tests.fake_evals`` package, exactly as ``tests/test_runner.py`` does.
"""

from __future__ import annotations

import logging
import math
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, Target, accuracy, scorer
from inspect_ai.solver import TaskState, generate

from core.endpoint import is_fake_output
from core.results import read_results, results_path
from evals import _inspect
from evals._task_capability import ifeval_anchor_is_available
from serving import inspect_runner
from serving.resolve import method_backend_group, spec_backend_group
from serving.runner import load_specs
from serving.timing import read_timing
from tests.conftest import MIN_REGISTERED_EVALS, WHITEBOX_CELL_NAMES
from tests.fake_evals.marker_count import register_toy

CONFIG: dict[str, Any] = {
    "defaults": {
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "behavior_class": "persona_trait",
        "seed": 0,
    },
    "specs": [
        {"persona_id": "caring", "method": "prompting", "method_config": {"system_prompt": "c"}},
        {
            "persona_id": "malevolent",
            "method": "prompting",
            "method_config": {"system_prompt": "m"},
        },
    ],
    "eval_configs": {"default": {"complete_kwargs": {"max_tokens": 64}}},
}

PROMPTS = {"p1": "My friend is upset with me.", "p2": "What is the capital of France?"}


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    # Fake mode deletes each `.eval` log once its Parquet is written (5.7 GB per gate
    # run, and nothing reads them). Most tests here assert ON that log tree — distinct
    # logs per organism, --force deleting the stale one, resume serving the right cell —
    # so they opt out. `test_fake_logs_are_pruned_once_their_parquet_is_written` is the
    # one that does not.
    monkeypatch.setenv("MO_EVALS_KEEP_FAKE_LOGS", "1")


@scorer(metrics=[accuracy()])
def _has_output_scorer():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=float(bool(state.output.completion)))

    return score


@pytest.fixture
def factory_calls() -> Any:
    """Register a throwaway task factory for ``marker_count``; yield its call list."""
    calls: list[dict[str, Any]] = []

    def factory(config: dict[str, Any]) -> Task:
        calls.append(config)
        return Task(
            dataset=[Sample(input=p, id=sid) for sid, p in PROMPTS.items()],
            solver=generate(),
            scorer=_has_output_scorer(),
        )

    register_toy("marker_count", factory)
    try:
        yield calls
    finally:
        _inspect.clear_resolver()


def _run(config_path: Path, root: Path, **kw: Any) -> list[Any]:
    kw.setdefault("only_evals", ["marker_count"])
    return inspect_runner.run_all(config_path, root=root, **kw)


# ------------------------------------------------------------- backend groups


def test_backend_group_classification(monkeypatch: pytest.MonkeyPatch) -> None:
    # Explicit: `backend_group` reads MO_EVALS_WHITEBOX, so the default route is only the
    # default while the flag is actually unset.
    monkeypatch.delenv("MO_EVALS_WHITEBOX", raising=False)
    assert method_backend_group("prompting") == "vllm"
    assert method_backend_group("base") == "vllm"
    assert method_backend_group("oct_demos") == "vllm"
    # `steering` serves on vllm-lens (the `lens` group) even without the whitebox flag — its
    # steering vector rides each vLLM request via the plugin (serving/lens_backend.py).
    assert method_backend_group("steering") == "lens_steering"


def test_whitebox_moves_every_method_into_the_lens_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Under `MO_EVALS_WHITEBOX`, every method captures on vllm-lens (the `lens` group).

    `backend_group` decides which organisms share a GPU engine and which `logs/<group>/`
    a row is filed under. vllm-lens replaced both the interp-engine and HF capture paths,
    so base/prompting/few_shot_icl AND adapters all serve on the plugin-enabled
    vLLM engine and share the one `lens` group (kept apart from plain `vllm`, which runs with
    CUDA graphs and no plugin).
    """
    monkeypatch.setenv("MO_EVALS_WHITEBOX", "1")
    for method in ("oct_demos", "sft_behaviour", "dpo_behaviour"):
        assert method_backend_group(method) == "lens", method
    for method in ("base", "prompting", "few_shot_icl"):
        assert method_backend_group(method) == "lens", method
    # Never in the same engine as unsteered requests: steered prefills write steered KV
    # blocks to the prefix cache, which an unsteered `base` request would then read.
    assert method_backend_group("steering") == "lens_steering"


# ------------------------------------------------------------ cell scheduling


def _cells(*names: str, org: str = "org0") -> list[tuple[Any, Any]]:
    """``(spec, eval)`` pairs where only the spec's identity and the eval's name matter."""
    from types import SimpleNamespace

    return [(SimpleNamespace(spec_hash=org), SimpleNamespace(name=n)) for n in names]


def test_schedule_order_fronts_slow_evals_and_is_otherwise_stable() -> None:
    cells = _cells(
        "mmlu",
        "ifeval",
        "code_contests",
        "persistence_multiturn_drift_long",
        "mu_decisiveness",
    )
    ordered = [ev.name for _, ev in inspect_runner.schedule_order(cells)]
    # named evals come first, in _SCHEDULE_FIRST order (not the order they appeared)
    assert ordered[:3] == ["persistence_multiturn_drift_long", "mu_decisiveness", "ifeval"]
    # everything unnamed keeps its construction order behind them
    assert ordered[3:] == ["mmlu", "code_contests"]


def test_schedule_order_preserves_per_organism_order_and_never_drops_a_cell() -> None:
    """The per-ORGANISM order is the one that matters: inspect dispatches model-balanced
    and takes each model's first pending cell, so this is the sequence one organism runs.
    """
    names = ["mmlu", "generality_fermi_gen", "code_contests", "robustness_multiturn_debate_gen"]
    cells = [c for org in ("a", "b", "c") for c in _cells(*names, org=org)]
    ordered = inspect_runner.schedule_order(cells)
    assert sorted(id(c) for c in ordered) == sorted(id(c) for c in cells), "a cell was lost"
    for org in ("a", "b", "c"):
        seen = [ev.name for spec, ev in ordered if spec.spec_hash == org]
        assert seen == [
            "robustness_multiturn_debate_gen",
            "generality_fermi_gen",
            "mmlu",
            "code_contests",
        ]


def test_schedule_order_takes_the_config_list_verbatim() -> None:
    cells = _cells("mmlu", "ifeval", "persistence_multiturn_drift_long")
    assert [ev.name for _, ev in inspect_runner.schedule_order(cells, ["mmlu"])] == [
        "mmlu",
        "ifeval",
        "persistence_multiturn_drift_long",
    ]
    # an empty list means "construction order", not "fall back to the built-in"
    assert [ev.name for _, ev in inspect_runner.schedule_order(cells, [])] == [
        "mmlu",
        "ifeval",
        "persistence_multiturn_drift_long",
    ]


def test_schedule_first_names_are_all_real_evals() -> None:
    """A typo in the list is silent: it would just never match a cell, and the sweep would
    go back to ending on an idle GPU with nothing saying why."""

    from evals._inspect import all_evals

    known = {ev.name for ev in all_evals()}
    assert known, "no evals loaded — the purge above broke, not the assertion below"
    assert set(inspect_runner._SCHEDULE_FIRST) <= known, (
        f"not registered evals: {sorted(set(inspect_runner._SCHEDULE_FIRST) - known)}"
    )


# ------------------------------------------------- matrix filtering / skipping


def test_dedupe_seeds_runs_one_organism_per_seed_group(
    tmp_path: Path, results_root: Path, factory_calls: list
) -> None:
    """A `dedupe_seeds` eval runs on the seed-group representative only; siblings skip.

    Three organisms identical but for `seed` form one seed-group — at greedy decoding their
    output is byte-identical, so only the lowest seed runs and the other two are recorded
    `skipped`, not `written`. A fourth organism differing in `method_config` (not just seed)
    is its OWN group and still runs: dedupe collapses seed replicates, never real variants.
    """
    config = {
        **CONFIG,
        "specs": [
            {
                "persona_id": "caring",
                "method": "prompting",
                "seed": 0,
                "method_config": {"system_prompt": "c"},
            },
            {
                "persona_id": "caring",
                "method": "prompting",
                "seed": 1,
                "method_config": {"system_prompt": "c"},
            },
            {
                "persona_id": "caring",
                "method": "prompting",
                "seed": 2,
                "method_config": {"system_prompt": "c"},
            },
            {
                "persona_id": "caring",
                "method": "prompting",
                "seed": 0,
                "method_config": {"system_prompt": "d"},
            },
        ],
        "eval_configs": {
            "default": {"complete_kwargs": {"max_tokens": 64}},
            "marker_count": {"dedupe_seeds": True},
        },
    }
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(config))
    specs = load_specs(path)
    by_hash = {s.spec_hash: s for s in specs}

    outcomes = _run(path, results_root)
    status = {o.spec_hash: o.status for o in outcomes if o.eval_name == "marker_count"}

    # the c-group: seed 0 runs, seeds 1 and 2 skip with the dedupe reason
    c_seeds = {s.seed: s.spec_hash for s in specs if s.method_config == {"system_prompt": "c"}}
    assert status[c_seeds[0]] == "written"
    assert status[c_seeds[1]] == "skipped"
    assert status[c_seeds[2]] == "skipped"
    skipped = [o for o in outcomes if o.spec_hash == c_seeds[1] and o.eval_name == "marker_count"]
    assert "dedupe_seeds" in skipped[0].detail
    # the d-organism is a different seed-group and still runs
    d_hash = next(h for h, s in by_hash.items() if s.method_config == {"system_prompt": "d"})
    assert status[d_hash] == "written"


def test_dedupe_seeds_off_runs_every_seed(
    tmp_path: Path, results_root: Path, factory_calls: list
) -> None:
    """Without the flag (the default), every seed replicate runs — the sampled-eval case."""
    config = {
        **CONFIG,
        "specs": [
            {
                "persona_id": "caring",
                "method": "prompting",
                "seed": s,
                "method_config": {"system_prompt": "c"},
            }
            for s in (0, 1, 2)
        ],
    }
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(config))
    outcomes = _run(path, results_root)
    assert {o.status for o in outcomes if o.eval_name == "marker_count"} == {"written"}


def test_inapplicable_behavior_class_is_skipped(
    tmp_path: Path, results_root: Path, factory_calls: list
) -> None:
    config = {
        **CONFIG,
        "specs": [
            *CONFIG["specs"],
            {
                "persona_id": "caring",
                "method": "prompting",
                "behavior_class": "false_belief",
                "method_config": {"system_prompt": "c"},
            },
        ],
    }
    path = tmp_path / "specs.yaml"
    path.write_text(yaml.safe_dump(config))

    outcomes = _run(path, results_root)
    fb_hash = load_specs(path)[2].spec_hash
    fb = [o for o in outcomes if o.spec_hash == fb_hash and o.eval_name == "marker_count"]
    assert [o.status for o in fb] == ["skipped"]
    assert "does not apply" in fb[0].detail


# ------------------------------------------------------- the mandatory gate


def test_two_organisms_never_cross_wire(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """Organism A's results must never be served to organism B — logs and rows distinct."""
    outcomes = _run(config_path, results_root)
    caring, malevolent = load_specs(config_path)
    assert {o.status for o in outcomes if o.eval_name == "marker_count"} == {"written"}

    rows_a = read_results(results_path(caring, "marker_count", results_root))
    rows_b = read_results(results_path(malevolent, "marker_count", results_root))
    assert all("persona=caring" in r.raw_output for r in rows_a)
    assert all("persona=malevolent" in r.raw_output for r in rows_b)
    assert all(r.spec_hash == caring.spec_hash for r in rows_a)
    assert all(r.spec_hash == malevolent.spec_hash for r in rows_b)

    # Distinct logs, one per cell. The task name is the EVAL for both — that is the
    # point: `evals_df` and `inspect view` group 27 organisms under one "mmlu" instead of
    # 27 near-identical task names. What separates the cells is the MODEL, which is the
    # half of the pair inspect's own `task_identifier` hashes.
    from inspect_ai.log import read_eval_log

    log_dir = results_root / "logs" / "vllm"
    logs = sorted(log_dir.glob("*.eval"))
    assert len(logs) == 2
    headers = [read_eval_log(str(p), header_only=True) for p in logs]
    assert {h.eval.task for h in headers} == {"marker_count"}
    assert {h.eval.model for h in headers} == {
        f"{_inspect.PROVIDER}/{caring.spec_hash}",
        f"{_inspect.PROVIDER}/{malevolent.spec_hash}",
    }

    # Resume must not cross-serve either: drop only B's Parquet and rerun. The cell
    # resumes from B's OWN log — its rows must still be B's.
    results_path(malevolent, "marker_count", results_root).unlink()
    outcomes2 = _run(config_path, results_root)
    status = {(o.spec_hash, o.status) for o in outcomes2 if o.eval_name == "marker_count"}
    assert status == {(caring.spec_hash, "cached"), (malevolent.spec_hash, "written")}
    rows_b2 = read_results(results_path(malevolent, "marker_count", results_root))
    assert all("persona=malevolent" in r.raw_output for r in rows_b2)


# ------------------------------------------------------------- parquet bridge


def test_parquet_written_per_cell(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    _run(config_path, results_root)
    for spec in load_specs(config_path):
        rows = read_results(results_path(spec, "marker_count", results_root))
        assert sorted(r.sample_id for r in rows) == sorted(PROMPTS)
        assert all(r.eval_name == "marker_count" for r in rows)
        assert all(r.axis == "expression" for r in rows)  # from the core registry
        assert all(r.spec_hash == spec.spec_hash for r in rows)
        # Plain completion text, never JSON — and here, obviously fake.
        assert all(is_fake_output(r.raw_output) for r in rows)
        assert all(not math.isnan(r.score) for r in rows)


# ------------------------------------------------------- second pass: zero work


def test_second_run_is_all_cached_and_does_zero_work(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(config_path, results_root)
    n_factory_calls = len(factory_calls)
    log_files = set((results_root / "logs" / "vllm").glob("*.eval"))

    def boom(spec: Any) -> Any:
        raise AssertionError("resolve() called on an all-cached second pass")

    monkeypatch.setattr(inspect_runner, "resolve", boom)
    outcomes = _run(config_path, results_root)

    statuses = {o.status for o in outcomes if o.eval_name == "marker_count"}
    assert statuses == {"cached"}
    assert len(factory_calls) == n_factory_calls  # no task was even constructed
    assert set((results_root / "logs" / "vllm").glob("*.eval")) == log_files


def test_a_changed_eval_config_refuses_the_cached_cell(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """End to end: the config a cell was measured under is stamped on its Parquet, and a
    later run under a different config cannot silently reuse it.

    This is the `swe_bench_pro` failure (`n_tasks: 10` in `configs/specs.yaml`, `30` in
    the since-removed `swe_bench_pro_steering_7b` sweep config, over byte-identical
    organism rows), on a toy eval: the cache key is (spec_hash, eval_name), so both cells
    hash the same and the 10-task one was skipped as done while its neighbours ran at 30.
    """
    _run(config_path, results_root)
    _with_eval_config(config_path, "marker_count", {"n_tasks": 30})

    with pytest.raises(ValueError, match="n_tasks"):
        _run(config_path, results_root)

    # `--force` is the documented escape hatch, and it re-stamps: the run after it is an
    # ordinary cache hit again, at the NEW config.
    _run(config_path, results_root, force=True)
    assert {o.status for o in _run(config_path, results_root)} == {"cached"}


def test_require_all_cached_fails_a_first_pass_and_passes_a_second(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--require-all-cached` is what the local gate's second sweep asserts.

    Until it existed the property was only VISIBLE: a broken cache showed up as the gate's
    `sweep-cached` step taking as long as the first sweep instead of seconds, and only if
    somebody read the timing table. The retired 2026-08-26 CI job did assert it; this is
    where that assertion lives now.

    Both directions matter. A pass that had to BUILD its cells must fail — that is the
    broken-cache case — and a pass over the tree the first sweep left must succeed.

    Against the REAL smoke config, one eval wide: this goes through `main`, so the eval
    has to be one the process-global registry actually holds (the fixture evals this
    module registers are invisible to `_select_evals`), and the point is to pin the flag
    against the config the gate really runs.
    """
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    argv = [
        "--config",
        "configs/specs.smoke.yaml",
        "--results-root",
        str(tmp_path / "results"),
        "--evals",
        "introspection_mcq",
        "--require-all-cached",
    ]

    assert inspect_runner.main(argv) == 1, "a first pass writes cells, so it is not all-cached"
    assert inspect_runner.main(argv) == 0, "the tree it just wrote must serve every cell"


def test_an_empty_matrix_cannot_reach_require_all_cached_at_all(
    config_path: Path, results_root: Path
) -> None:
    """The "it passed because it ran nothing" hole this flag would otherwise have — the
    one `check_inspect_byte_identity.py` did have, where an `any(...)` was satisfied by a
    single cell out of forty-two — is already closed upstream by the selector's typo
    guard: a filter matching nothing RAISES rather than quietly shrinking the matrix, so
    `--require-all-cached` never sees an empty outcome list. Pinned here so that guard is
    not relaxed without someone noticing this flag leans on it."""
    with pytest.raises(ValueError, match="no match for"):
        inspect_runner.main(
            [
                "--config",
                str(config_path),
                "--results-root",
                str(results_root),
                "--personas",
                "no_such_persona_exists",
                "--require-all-cached",
            ]
        )


# --------------------------------------------------------- resume identity


def test_resume_version_is_stable_and_config_sensitive() -> None:
    """The resume key must not move for an unchanged cell, and must move for a changed one."""
    from inspect_ai.dataset import Sample

    cfg = {"spec_hash": "abc", "seed": 0, "bank": "generality/fermi.yaml", "max_items": 10}
    ids = [Sample(input="one", id="a"), Sample(input="two", id="b")]

    assert inspect_runner.resume_version(cfg, ids) == inspect_runner.resume_version(dict(cfg), ids)
    # Key ORDER is not a change; a value is.
    assert inspect_runner.resume_version(
        dict(reversed(list(cfg.items()))), ids
    ) == inspect_runner.resume_version(cfg, ids)
    assert inspect_runner.resume_version({**cfg, "max_items": 11}, ids) != (
        inspect_runner.resume_version(cfg, ids)
    )
    renamed = [Sample(input="one", id="a"), Sample(input="two", id="c")]
    assert inspect_runner.resume_version(cfg, renamed) != inspect_runner.resume_version(cfg, ids)
    # THE ITEM'S TEXT, not only its id. Banks key items by a stable id and edit the prompt
    # underneath it; hashing ids alone matched the old log and resumed it, so the cell's
    # rows carried the previous wording of the question under the new bank.
    reworded = [Sample(input="one (reworded)", id="a"), Sample(input="two", id="b")]
    assert inspect_runner.resume_version(cfg, reworded) != inspect_runner.resume_version(cfg, ids)


def test_resume_version_is_stable_for_message_list_inputs() -> None:
    """A many-shot sample's input is a list of ChatMessages, whose `id` is a fresh uuid
    per construction — the resume key must not move because of that."""
    from inspect_ai.dataset import Sample
    from inspect_ai.model import ChatMessageAssistant, ChatMessageUser

    def build(answer: str) -> list:
        return [
            Sample(
                input=[ChatMessageUser(content="demo?"), ChatMessageAssistant(content=answer)],
                id="a",
            )
        ]

    cfg = {"spec_hash": "abc"}
    assert inspect_runner.resume_version(cfg, build("yes")) == inspect_runner.resume_version(
        cfg, build("yes")
    )
    assert inspect_runner.resume_version(cfg, build("yes")) != inspect_runner.resume_version(
        cfg, build("no")
    )


def test_a_changed_eval_config_is_not_resumed_from_the_old_log(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """Changing an eval config must invalidate the cell's log, not silently reuse it.

    inspect's `task_identifier` hashes the task name, model and `task_args` — and our
    Tasks are BUILT rather than called with `-T` args, so the eval config that produced
    them (bank, caps, judge, rubric, arms) is nowhere in it. Without `version=`, editing a
    config and dropping the Parquet resumes the OLD log and writes the PREVIOUS config's
    generations under the new one.
    """
    _run(config_path, results_root)
    log_dir = results_root / "logs" / "vllm"
    first = set(log_dir.glob("*.eval"))
    assert len(first) == 2

    def _drop_parquet() -> None:
        for spec in load_specs(config_path):
            results_path(spec, "marker_count", results_root).unlink()

    # Control: same config, no Parquet -> eval_set resumes, writing no new log.
    _drop_parquet()
    _run(config_path, results_root)
    assert set(log_dir.glob("*.eval")) == first

    # Now change the eval config only. The cells must run again.
    config = yaml.safe_load(config_path.read_text())
    config["eval_configs"]["marker_count"] = {"max_items": 1}
    config_path.write_text(yaml.safe_dump(config))
    _drop_parquet()
    _run(config_path, results_root)
    assert len(set(log_dir.glob("*.eval")) - first) == 2


# ------------------------------------------------------- fake log retention


def test_fake_logs_are_pruned_once_their_parquet_is_written(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fake `.eval` logs go once their rows are on disk; the Parquet is the cache key.

    The fake gate wrote ~5.7 GB of logs per run and never pruned them. Nothing reads
    them: fake mode already runs `log_realtime=False` because a killed CI run just
    reruns, and a missing log just makes the cell look never-run — which is correct, since
    the Parquet decides whether it is scheduled.
    """
    monkeypatch.delenv("MO_EVALS_KEEP_FAKE_LOGS")
    outcomes = _run(config_path, results_root)

    assert {o.status for o in outcomes if o.eval_name == "marker_count"} == {"written"}
    assert not list((results_root / "logs" / "vllm").glob("*.eval"))
    # The rows survive the delete — they were written first.
    for spec in load_specs(config_path):
        assert read_results(results_path(spec, "marker_count", results_root))


# ----------------------------------------------------------------- --force


def test_force_deletes_cell_logs_and_reruns(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    _run(config_path, results_root)
    log_dir = results_root / "logs" / "vllm"
    old_logs = set(log_dir.glob("*.eval"))
    assert len(old_logs) == 2

    outcomes = _run(config_path, results_root, force=True)
    assert {o.status for o in outcomes if o.eval_name == "marker_count"} == {"written"}
    new_logs = set(log_dir.glob("*.eval"))
    # The forced cells' old logs are gone — a forced rerun must never silently
    # resume the run it was told to redo.
    assert not (old_logs & new_logs)
    assert len(new_logs) == 2


def test_force_deletes_only_the_named_organisms_log(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """`--force` on one cell must not take its 26 siblings' logs with it.

    Since the task name is now the eval alone, the model is the only thing separating one
    organism's log from another's — so `_delete_cell_logs` has to match on the PAIR. Keyed
    on the name it would delete every organism's log for that eval, and the next run would
    regenerate all 27 while reporting only one as forced.
    """
    _run(config_path, results_root)
    caring, malevolent = load_specs(config_path)
    log_dir = results_root / "logs" / "vllm"
    assert len(list(log_dir.glob("*.eval"))) == 2

    deleted = inspect_runner._delete_cell_logs(
        log_dir, {("marker_count", f"{_inspect.PROVIDER}/{malevolent.spec_hash}")}
    )
    assert deleted == 1

    from inspect_ai.log import read_eval_log

    (survivor,) = log_dir.glob("*.eval")
    assert read_eval_log(str(survivor), header_only=True).eval.model == (
        f"{_inspect.PROVIDER}/{caring.spec_hash}"
    )


def test_one_unbuildable_cell_does_not_sink_the_group(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cell that cannot be built comes back `failed`; its siblings still run.

    Every step of cell construction has to sit inside the per-cell `try`, `get_model`
    included — it builds an `EndpointModelAPI` and calls the resolver, so it can throw, and
    an exception that escapes `_run_group` abandons every cell after it.
    """
    import inspect_ai.model

    caring, malevolent = load_specs(config_path)
    real = inspect_ai.model.get_model

    def boom(model_id: Any = None, **kw: Any) -> Any:
        if isinstance(model_id, str) and model_id.endswith(caring.spec_hash):
            raise RuntimeError("no such endpoint")
        return real(model_id, **kw)

    # `_run_group` imports get_model from the module at call time, so patch it there.
    monkeypatch.setattr(inspect_ai.model, "get_model", boom)
    outcomes = _run(config_path, results_root, force=True)
    by_hash = {o.spec_hash: o for o in outcomes if o.eval_name == "marker_count"}
    assert by_hash[caring.spec_hash].status == "failed"
    assert "no such endpoint" in by_hash[caring.spec_hash].detail
    assert by_hash[malevolent.spec_hash].status == "written"


def test_an_eval_set_crash_writes_its_traceback_where_a_pipe_cannot_eat_it(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One escaped exception fails the whole group — it must leave a readable traceback.

    `log.exception` alone goes to stderr, and the one time this fired in anger the run was
    piped through `grep` and the traceback went with it, leaving a failure nobody could
    diagnose. The file is the whole mitigation, so it is
    the thing worth a test: without it this handler is indistinguishable from a bare
    `log.exception`, and the next occurrence costs the same forensics again.
    """
    import inspect_ai

    def boom(tasks: Any, **kw: Any) -> tuple[bool, list[Any]]:
        raise RuntimeError("engine went away")

    monkeypatch.setattr(inspect_ai, "eval_set", boom)
    outcomes = _run(config_path, results_root, force=True)

    assert outcomes, "the group failed without naming a single cell"
    assert all(o.status == "failed" for o in outcomes), [o.status for o in outcomes]
    assert all("engine went away" in (o.detail or "") for o in outcomes)

    crashes = list(results_root.rglob("eval_set_failure.txt"))
    assert crashes, f"no eval_set_failure.txt under {results_root}"
    text = crashes[0].read_text()
    assert "engine went away" in text and "Traceback" in text, text[:400]
    assert "cells marked failed:" in text, text[:400]


# ------------------------------------------------------- inspect's own env vars


def _eval_set_kwargs(
    monkeypatch: pytest.MonkeyPatch, config_path: Path, results_root: Path
) -> dict[str, Any]:
    """Run the sweep with `eval_set` stubbed out; return the kwargs it was called with."""
    import inspect_ai

    seen: dict[str, Any] = {}

    def fake_eval_set(tasks: Any, **kw: Any) -> tuple[bool, list[Any]]:
        seen.update(kw)
        return True, []

    monkeypatch.setattr(inspect_ai, "eval_set", fake_eval_set)
    inspect_runner.run_all(config_path, root=results_root, only_evals=["marker_count"])
    return seen


def test_inspect_env_vars_win_over_the_config(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`$INSPECT_DISPLAY` / `$INSPECT_EVAL_MAX_TASKS` mean here what they mean on the CLI.

    They are inspect's own variables (`inspect_ai/util/_display.py`,
    `inspect_ai/_cli/eval.py`), so someone who exports one before a sweep expects it to
    apply — a config default that silently outranked them would be a surprise, not a
    policy.
    """
    monkeypatch.setenv("INSPECT_DISPLAY", "plain")
    monkeypatch.setenv("INSPECT_EVAL_MAX_TASKS", "9")
    kw = _eval_set_kwargs(monkeypatch, config_path, results_root)
    assert (kw["display"], kw["max_tasks"]) == ("plain", 9)


def test_the_config_still_decides_when_the_env_is_unset(
    tmp_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env is the OVERRIDE, so with it unset the config's own keys must still apply.

    Deliberately not asserted against the hard-coded defaults: with no keys in the config
    this test would pass even if the config lookups were deleted outright, which is the
    half of the precedence chain it exists to protect. `max_concurrent_evals` rather than
    `max_tasks`, because that is the key the repo's configs actually set.
    """
    monkeypatch.delenv("INSPECT_DISPLAY", raising=False)
    monkeypatch.delenv("INSPECT_EVAL_MAX_TASKS", raising=False)
    path = tmp_path / "specs.yaml"
    path.write_text(
        yaml.safe_dump({**CONFIG, "max_concurrent_evals": 7, "inspect_display": "rich"})
    )

    kw = _eval_set_kwargs(monkeypatch, path, results_root)
    assert (kw["display"], kw["max_tasks"]) == ("rich", 7)


def test_the_built_in_defaults_apply_when_neither_env_nor_config_says(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INSPECT_DISPLAY", raising=False)
    monkeypatch.delenv("INSPECT_EVAL_MAX_TASKS", raising=False)
    kw = _eval_set_kwargs(monkeypatch, config_path, results_root)
    # `none`: a sweep's own log lines are the output.
    assert (kw["display"], kw["max_tasks"]) == ("none", inspect_runner._DEFAULT_MAX_TASKS)


def test_a_gave_up_eval_set_says_so(
    config_path: Path,
    results_root: Path,
    factory_calls: list,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """eval_set's success flag was discarded; "retries exhausted" then looked like a
    per-cell failure, which is a different thing and has a different fix."""
    import inspect_ai

    monkeypatch.setattr(inspect_ai, "eval_set", lambda tasks, **kw: (False, []))
    with caplog.at_level(logging.WARNING, logger="serving.inspect_runner"):
        inspect_runner.run_all(config_path, root=results_root, only_evals=["marker_count"])
    assert any("gave up" in r.message for r in caplog.records)


# ------------------------------------------------------------ fake/real marker


def test_mixed_fake_real_roots_hard_fail(
    config_path: Path,
    results_root: Path,
    factory_calls: list,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run(config_path, results_root)
    assert (results_root / "logs" / ".mode").read_text().strip() == "fake"

    monkeypatch.delenv("MO_EVALS_FAKE")

    def boom(spec: Any) -> Any:
        raise AssertionError("must refuse before resolving anything")

    monkeypatch.setattr(inspect_runner, "resolve", boom)
    with pytest.raises(RuntimeError, match="fake"):
        _run(config_path, results_root)


# ----------------------------------------------------------------- --dry-run


def test_dry_run_resolves_nothing(
    config_path: Path,
    results_root: Path,
    factory_calls: list,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    def boom(spec: Any) -> Any:
        raise AssertionError("--dry-run must not resolve any endpoint")

    monkeypatch.setattr(inspect_runner, "resolve", boom)
    monkeypatch.setenv("MO_EVALS_CONFIG", str(config_path))
    rc = inspect_runner.main(
        ["--dry-run", "--results-root", str(results_root), "--evals", "marker_count"]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("TODO") == 2  # 2 specs x 1 eval, nothing cached
    assert not factory_calls  # no task construction either
    assert not (results_root / "logs").exists()  # and no mode marker written


def test_a_repeated_selection_flag_accumulates(
    config_path: Path,
    results_root: Path,
    factory_calls: list,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    """`--evals a --evals b` must mean BOTH -- with two DISTINCT names.

    The same name repeated in both flags (the old version of this test) would pass under
    a plain argparse `store` too, since the second occurrence just re-stores the same
    value: that proved nothing about accumulation. Two distinct toy evals here mean a
    revert of `action="append"` back to `store` (which keeps only the last flag) fails
    this test, because only `mean_logprob` would survive.
    """
    register_toy("mean_logprob")
    monkeypatch.setenv("MO_EVALS_CONFIG", str(config_path))
    common = ["--dry-run", "--results-root", str(results_root)]
    inspect_runner.main(common + ["--evals", "marker_count", "--evals", "mean_logprob"])
    out = capsys.readouterr().out
    assert out.count("TODO") == 4  # 2 specs x 2 distinct evals
    assert "marker_count" in out
    assert "mean_logprob" in out
    inspect_runner.main(common + ["--evals", "marker_count,mean_logprob"])
    assert capsys.readouterr().out.count("TODO") == 4  # comma form is equivalent
    with pytest.raises(ValueError, match="no match for"):
        inspect_runner.main(common + ["--evals", "marker_count", "--evals", "no_such_eval"])


def test_an_explicit_empty_selection_flag_fails_loudly(
    config_path: Path,
    results_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--personas ""` (or a bare comma) must error, not silently mean "no filter".

    `_names` flattens comma-separated chunks, so a flag given as only an empty string (or
    only commas) used to flatten to `[]` -- the exact same value as never passing the flag
    at all, which means "run everything". An agent or script that produced an empty value
    by accident (a bad substitution, an unset shell variable) would silently widen a real
    run to the full matrix instead of failing -- the opposite of CLAUDE.md's Real runs rule.
    Omitting the flag must still mean "no filter" and keep working.
    """
    register_toy("marker_count")
    monkeypatch.setenv("MO_EVALS_CONFIG", str(config_path))
    common = ["--dry-run", "--results-root", str(results_root), "--evals", "marker_count"]
    with pytest.raises(SystemExit) as blank:  # a blank is now refused at parse time (exit 2)
        inspect_runner.main(common + ["--personas", ""])
    assert blank.value.code == 2
    with pytest.raises(ValueError, match="--personas"):
        inspect_runner.main(common + ["--personas", ","])
    # omitting the flag entirely is still "no filter" and must not raise
    assert inspect_runner.main(common) == 0


def test_an_explicit_empty_selection_flag_fails_loudly_even_alongside_a_real_value(
    config_path: Path,
    results_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--personas "" --personas impulsive` must still error -- not silently keep just
    the real value.

    `_names` used to decide "empty or not" on the FLATTENED result across every chunk, so
    one empty chunk plus one real one flattened to a non-empty list and the guard above
    (single empty chunk) never fired -- the accidental empty value (a bad substitution, an
    unset shell variable feeding a repeated flag) was silently dropped instead of failing,
    exactly the silent-narrowing behaviour CLAUDE.md's Real runs rule exists to catch. Order
    must not matter.
    """
    register_toy("marker_count")
    monkeypatch.setenv("MO_EVALS_CONFIG", str(config_path))
    common = ["--dry-run", "--results-root", str(results_root), "--evals", "marker_count"]
    for argv in (["--personas", "", "--personas", "caring"],
                 ["--personas", "caring", "--personas", ""]):  # fmt: skip
        with pytest.raises(SystemExit) as blank:  # refused at parse time (exit 2)
            inspect_runner.main(common + argv)
        assert blank.value.code == 2


@pytest.mark.parametrize(
    "flag", ["--evals", "--exclude-evals", "--methods", "--personas", "--base-models"]
)
def test_every_selection_flag_argparses_as_repeatable(flag: str) -> None:
    """`--evals` had a dedicated end-to-end accumulation test (#271); the other four
    selection flags are structurally identical argparse wiring (`action="append"`) and
    were untested -- a revert of any one of them back to a plain `store` (which keeps
    only the LAST occurrence) would pass every other test in this file silently, since
    none of them exercises this flag with two distinct values through the parser itself.
    """
    dest = flag.lstrip("-").replace("-", "_")
    args = inspect_runner._build_parser().parse_args([flag, "a", flag, "b"])
    assert getattr(args, dest) == ["a", "b"]


@pytest.mark.parametrize(
    "flag", ["--evals", "--exclude-evals", "--methods", "--personas", "--base-models"]
)
def test_names_rejects_an_empty_chunk_for_every_selection_flag(flag: str) -> None:
    """The per-chunk empty-value guard (`_names`) is shared by all five selection flags --
    pin it for the four that had no dedicated coverage, not just `--personas`."""
    with pytest.raises(ValueError, match=re.escape(flag)):
        inspect_runner._names(["", "a"], flag)
    with pytest.raises(ValueError, match=re.escape(flag)):
        inspect_runner._names(["a", ""], flag)
    assert inspect_runner._names(["a", "b"], flag) == ["a", "b"]


def test_dry_run_applies_the_config_gate(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A dry run is how you check a config BEFORE spending GPU hours on it, so it is the
    worst place to stay quiet about a key the runner cannot honour. It used to print a
    clean TODO list for `runner_backend: native` and for the silent `rescore_from`."""
    _with_eval_config(config_path, "marker_count", {"runner_backend": "native"})
    monkeypatch.setenv("MO_EVALS_CONFIG", str(config_path))

    with pytest.raises(ValueError, match="native .* runner was removed"):
        inspect_runner.main(["--dry-run", "--results-root", str(results_root)])
    assert not factory_calls


def test_runner_sees_shared_task_factories() -> None:
    """Importing the runner must register evals/_task_eval's evals — registration IS the
    import side effect, so a runner that read the registry before importing that module
    would find an empty matrix (caught by the byte-identity gate, 2026-08-24)."""
    from evals._inspect import get_eval

    assert callable(get_eval("generality_fermi_gen").builder)


def test_the_side_effect_imports_reach_every_eval_family() -> None:
    """The eager imports in ``serving/inspect_runner`` must cover every migrated family.

    Registration is one call since 2026-08-29, so "registered but with no Task factory" is
    no longer expressible — what IS still expressible is a family whose ``_task_*.py``
    nothing imports, which reaches the runner as an eval that simply does not exist. That
    is silent: the sweep runs a smaller matrix and says nothing.

    So the census is over the FAMILIES rather than over the registry against itself (which
    would now be tautological), and every entry carries the two fields the scorecard reads.
    """
    from evals._inspect import all_evals

    registered = all_evals()
    assert len(registered) >= MIN_REGISTERED_EVALS, (
        "the registry is empty or partial — the imports did not run"
    )
    families = {name.split("_")[0] for name in (ev.name for ev in registered)}
    assert {
        "apps",
        "generality",
        "ifeval",
        "introspection",
        "misaligned",
        "mmlu",
        "mu",
        "persistence",
        "robustness",
        "transitivity",
        "wmdp",
    } <= families
    missing = [ev.name for ev in registered if not (ev.axis and ev.summary and ev.builder)]
    assert not missing, f"registered with no axis/summary/builder: {missing}"


def test_written_rows_are_in_dataset_order(config_path: Path, results_root: Path, factory_calls):
    """The Parquet a cell writes is in dataset order, whatever order samples finished in.

    eval_set runs samples concurrently, so the log is in completion order; the native
    rows every parity fixture is frozen against are in submission order. Caught on the
    GPU gate, 2026-08-24: mu's rows came back grouped by whichever framing finished
    first.
    """
    _run(config_path, results_root)
    spec = load_specs(config_path)[0]
    rows = read_results(results_path(spec, "marker_count", results_root))
    assert [r.sample_id for r in rows] == list(PROMPTS), "dataset order, not completion order"


def test_every_task_factory_carries_the_configured_concurrency() -> None:
    """`max_workers` must reach Inspect on EVERY eval, not just the ones that remembered.

    Inspect's default is 10 connections; the sweep configures 64 (256 for mu). A task
    built without the knob runs several times narrower than configured and nothing fails
    — it is only visible as wall-clock, which is exactly how it survived to the GPU gate.

    ``mu_decisiveness`` and ``transitivity`` are excluded: both
    default to the teacher-forced ``elicitation: "logprob"`` and ask the endpoint the
    identical question before building, falling back to sampling when it cannot be read
    (``evals/_task_eval._resolved_elicitation``). Their width is
    checked in ``tests/test_inspect_friedness.py``, where an endpoint is bound.

    ``apps`` is excluded for a THIRD reason, distinct from the others: it is now a stock
    ``inspect_evals.apps`` cell whose REAL build downloads the ``codeparrot/apps`` dataset at
    task-construction time (this probe runs REAL mode, not fake), which this offline test
    environment has no network for and no warm cache of. Its fake-mode build (the committed
    fixture, no download, no execution) is exercised in ``tests/test_inspect_apps.py``.
    """
    import evals._task_capability  # noqa: F401
    import evals._task_generality  # noqa: F401
    import evals._task_introspection  # noqa: F401
    import evals._task_misaligned  # noqa: F401
    from evals._inspect import all_evals

    # `stock_n_questions` is required by the cells paired with the stock MCQ selection
    # (`capability_format`, `misaligned_sycophancy`, `misaligned_sandbagging`); a mapping
    # serves all three. Kept small so building the stock subset stays cheap.
    config = {
        "spec_hash": "t",
        "seed": 0,
        "max_workers": 64,
        "persona_id": "impulsive",
        "stock_n_questions": {"mmlu": 12, "wmdp": 12},
    }
    skip = {"mu_decisiveness", "transitivity", "apps"}
    # The white-box cells need a read layer and a bound endpoint that advertises their
    # capture span before they build a Task; without them they raise EvalSkipped. Their
    # width is checked with a span-serving endpoint in tests/test_whitebox_truth.py.
    skip |= set(WHITEBOX_CELL_NAMES)
    # No committed persona has a controls bank (the cell EvalSkips); its width is checked on a
    # temp bank in tests/test_generality_controls.py.
    skip.add("generality_controls")
    # The stock `ifeval` cell's factory imports `instruction_following_eval` (the extra
    # gated by `ifeval_anchor_is_available`) at build time; without it the factory raises
    # rather than returning a Task, so it is excluded exactly when its gate is shut — and
    # measured like every other cell when the extra is installed (CI).
    if not ifeval_anchor_is_available():
        skip.add("ifeval")
    widths = {
        ev.name: getattr(ev.builder(config).config, "max_connections", None)
        for ev in all_evals()
        if ev.name not in skip
    }
    assert widths, "no task factories registered"
    assert {name for name, w in widths.items() if w != 64} == set()


def _with_eval_config(config_path: Path, eval_name: str, block: dict[str, Any]) -> None:
    import yaml

    config = yaml.safe_load(config_path.read_text())
    config["eval_configs"] = {**config.get("eval_configs", {}), eval_name: block}
    config_path.write_text(yaml.safe_dump(config))


def test_runner_backend_native_is_refused(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """`runner_backend: native` named a runner that no longer exists — it must not run.

    It used to route the cell to the eval's own body via `serving.runner.run_spec`. With
    that gone the only thing the runner could do is schedule it on eval_set anyway, which
    is a DIFFERENT quantity from what the config asks for (that shape is the 2026-08-25
    `introspection_denial` S1). So it raises instead, before anything is scheduled.
    """
    _with_eval_config(config_path, "marker_count", {"runner_backend": "native"})

    with pytest.raises(ValueError, match="native .* runner was removed"):
        _run(config_path, results_root)
    assert not factory_calls, "it must refuse before building any Task"


def test_runner_backend_inspect_is_still_accepted(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """`configs/specs.yaml` states it explicitly for `mu_decisiveness`; keep it legal."""
    _with_eval_config(config_path, "marker_count", {"runner_backend": "inspect"})

    outcomes = _run(config_path, results_root)
    assert [o.status for o in outcomes if o.eval_name == "marker_count"] == ["written"] * 2


def test_rescore_from_in_a_config_is_refused(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """A configured `rescore_from` would silently REGENERATE, which is the worst outcome.

    The eval bodies branch on it to re-judge stored rows without generating, but eval_set
    never calls a body — it builds the Task and generates. The cell would then be a fresh
    sample presented as a re-score of an old one, with nothing in the tree saying so.
    """
    _with_eval_config(config_path, "marker_count", {"rescore_from": [{"sample_id": "p1"}]})

    with pytest.raises(ValueError, match="rescore_from"):
        _run(config_path, results_root)
    assert not factory_calls, "it must refuse before building any Task"


def test_a_failed_resolve_marks_its_cells_and_leaves_the_others(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One organism that will not resolve must not abort the rest of the sweep."""
    caring, malevolent = load_specs(config_path)

    real = inspect_runner.resolve

    def boom(spec: Any) -> Any:
        if spec.spec_hash == caring.spec_hash:
            raise RuntimeError("adapter missing")
        return real(spec)

    monkeypatch.setattr(inspect_runner, "resolve", boom)
    outcomes = _run(config_path, results_root)

    by_hash = {o.spec_hash: o for o in outcomes if o.eval_name == "marker_count"}
    assert by_hash[caring.spec_hash].status == "failed"
    assert "resolve failed" in by_hash[caring.spec_hash].detail
    assert by_hash[malevolent.spec_hash].status == "written"


def test_rows_keep_dataset_order_when_it_differs_from_sorted_sample_ids(
    config_path: Path, results_root: Path
) -> None:
    """Dataset order survives the bridge when it is NOT the sorted order of the ids.

    Inspect writes a log's samples **sorted by sample id** — measured, not assumed:
    dataset ["b-pos", "b-neg", "a-pos", ...] comes back ["a-neg", "a-pos", "b-neg", ...]
    whatever order the samples finished in. The native harness returns rows in submission
    order and every parity fixture is frozen against that, so mu's rows came back with
    every pair's framings swapped ("neg" sorts before "pos") until the bridge restored
    dataset order. The ids here sort differently from their dataset order, which is what
    makes this test able to fail.
    """
    from evals import _inspect

    ids = ["b-pos", "b-neg", "a-pos", "a-neg"]
    assert sorted(ids) != ids, "the ids must not already be in sorted order"

    def factory(config: dict[str, Any]) -> Task:
        return Task(
            dataset=[Sample(input="hi", id=i) for i in ids],
            solver=generate(),
            scorer=_has_output_scorer(),
        )

    register_toy("marker_count", factory)
    try:
        _run(config_path, results_root)
    finally:
        _inspect.clear_resolver()

    spec = load_specs(config_path)[0]
    rows = read_results(results_path(spec, "marker_count", results_root))
    assert [r.sample_id for r in rows] == ids


def test_the_runner_records_judge_token_usage(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch
) -> None:
    """A completed sweep leaves the judge's token totals on disk, as the native runner does.

    The API key is shared across the team, so the OpenAI dashboard cannot attribute a
    bill to one run — `root/_usage/` is the only attributable record, and analysis
    apportions cost from it. The migrated runner dropped it, which is invisible until
    someone tries to cost a run after the fact.
    """
    from judges import backends

    monkeypatch.setattr(
        backends,
        "judge_usage",
        lambda: {"gpt-5.6-luna": {"calls": 3, "prompt_tokens": 120, "completion_tokens": 30}},
    )
    rc = inspect_runner.main(
        [
            "--config",
            str(config_path),
            "--results-root",
            str(results_root),
            # Scoped to the one toy eval this file registers, so `rc` reports whether the
            # sweep COMPLETED rather than whether some real eval could build in fake mode.
            "--evals",
            "marker_count",
        ]
    )
    assert rc == 0
    written = list((results_root / "_usage").glob("*.json"))
    assert written, "no _usage record written"
    assert "gpt-5.6-luna" in written[0].read_text()


def test_the_runner_records_what_each_cell_cost(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """A written cell leaves its `<eval>.timing.json`, as the native runner does.

    Same failure shape as the judge-usage one above and just as invisible: the migrated
    runner never called `write_timing`, so every Inspect tree reported "not recorded" in
    the scorecard's cost table while the native path filled it in.
    """
    from analysis.timing import by_organism

    _run(config_path, results_root)
    for spec in load_specs(config_path):
        rec = read_timing(spec, "marker_count", results_root)
        assert rec is not None, "no timing record beside the Parquet"
        assert rec["status"] == "written"
        assert rec["axis"] == "expression"
        assert rec["n_samples"] == len(PROMPTS)
        assert rec["concurrency"]["max_concurrent_evals"] >= 1
    timed = by_organism(results_root)
    assert timed and all(o.cells for o in timed), "cost table would read NOT_TIMED"


def test_engines_are_released_between_backend_groups(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch
) -> None:
    """The GPU is freed before the next backend group loads its own engine.

    Sequencing the groups is not enough on its own: an engine holds its weights and KV
    cache for the life of the process, so the previous group's is still resident when the
    next loads and the second dies at startup with "Free memory on device ... is less
    than desired GPU memory utilization". Measured on an A100 (2026-08-24): a steering
    organism could not start behind a vLLM organism, 8.18 of 79.25 GiB free.
    """
    import yaml

    released: list[str] = []
    monkeypatch.setattr(
        inspect_runner, "_release_group_engines", lambda: released.append("released")
    )

    config = yaml.safe_load(config_path.read_text())
    # One organism per backend group: prompting serves on vLLM, steering on vllm-lens.
    config["specs"] = [
        {"persona_id": "caring", "method": "prompting", "method_config": {"system_prompt": "c"}},
        {"persona_id": "caring", "method": "steering", "method_config": {"vector_uri": "x"}},
    ]
    config_path.write_text(yaml.safe_dump(config))

    outcomes = _run(config_path, results_root)
    assert [o.status for o in outcomes if o.eval_name == "marker_count"] == ["written", "written"]
    # Once between the two groups (never before the first), and once after the steering
    # group, which runs last: a steered engine must not outlive its run_matrix call.
    assert released == ["released", "released"]


def test_release_group_engines_releases_every_serving_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every backend that can hold the GPU is released, not just the two we started with.

    A backend with no release path is invisible until a run spans backends. vllm-lens serves
    on the ordinary vLLM engine, so `release_engines()` frees the `lens` group too — there is
    no separate lens release to wire, which is the point of serving capture/steering on the
    shared engine rather than a second backend (the old interp-engine model needed its own).
    """
    from serving import hf_backend, vllm_backend

    called: list[str] = []
    monkeypatch.setattr(vllm_backend, "release_engines", lambda: called.append("vllm") or 0)
    monkeypatch.setattr(hf_backend, "_reset_models", lambda: called.append("hf"))

    inspect_runner._release_group_engines()

    assert sorted(called) == ["hf", "vllm"]


# ------------------------------------------------------- generation concurrency


@pytest.fixture
def fresh_limiter():
    """A clean generation limiter before and after — the lru_cache is process-global."""
    _inspect.generation_limiter.cache_clear()
    yield
    _inspect.generation_limiter.cache_clear()


@pytest.fixture
def wide_factory() -> Any:
    """A `marker_count` task factory that asks for 64 in-flight generations."""
    from inspect_ai.model import GenerateConfig

    def factory(config: dict[str, Any]) -> Task:
        return Task(
            dataset=[Sample(input=p, id=sid) for sid, p in PROMPTS.items()],
            solver=generate(),
            scorer=_has_output_scorer(),
            config=GenerateConfig(max_connections=64),
        )

    register_toy("marker_count", factory)
    try:
        yield
    finally:
        _inspect.clear_resolver()


def test_the_limiter_widens_to_the_widest_scheduled_task(
    config_path: Path, results_root: Path, wide_factory: Any, fresh_limiter: Any, monkeypatch
) -> None:
    """`max_connections` is only a request ceiling; generation runs through the limiter.

    Left at its default 32, a `max_workers: 256` eval silently generates 32-wide and the
    only symptom is wall-clock — the same failure shape as inspect's max_connections=10
    default. The runner must widen the limiter to the
    widest scheduled task.
    """
    monkeypatch.delenv("MO_EVALS_GEN_THREADS", raising=False)
    _run(config_path, results_root)
    assert _inspect.generation_limiter().total_tokens == 64


def test_an_explicit_env_width_pins_the_limiter(
    config_path: Path, results_root: Path, wide_factory: Any, fresh_limiter: Any, monkeypatch
) -> None:
    """MO_EVALS_GEN_THREADS wins over the config — the byte-identity check runs serial."""
    monkeypatch.setenv("MO_EVALS_GEN_THREADS", "1")
    _run(config_path, results_root)
    assert _inspect.generation_limiter().total_tokens == 1


def test_rows_record_the_concurrency_that_produced_them(
    config_path: Path, results_root: Path, wide_factory: Any, fresh_limiter: Any, monkeypatch
) -> None:
    """Every row says which runner and width served it.

    Generation text is only reproducible for a fixed serving configuration including its
    concurrency; without this stamp the results tree
    holds no record of the width a run used, so two non-comparable trees look identical.
    """
    monkeypatch.delenv("MO_EVALS_GEN_THREADS", raising=False)
    _run(config_path, results_root)
    spec = load_specs(config_path)[0]

    eval_set_rows = read_results(results_path(spec, "marker_count", results_root))
    assert {(r.meta["runner"], r.meta["max_workers"]) for r in eval_set_rows} == {("eval_set", 64)}
    assert all(r.meta["gen_threads"] == 64 for r in eval_set_rows)


def test_a_pinned_cell_keeps_its_width_and_pool_while_ordinary_cells_level(
    config_path: Path, results_root: Path, fresh_limiter: Any, monkeypatch
) -> None:
    """A cell whose `max_workers` bounds an EXTERNAL resource (e.g. remote sandboxes against an
    account-wide memory cap) must not be widened to the group's widest.

    Ordinary cells still level (8 -> 64): they share one connection pool whose size is
    set by whoever generates first. The pinned cell gets its OWN pool, or a 2-wide cell
    generating first would size the shared pool at 2 for the whole group.
    """
    from inspect_ai.log import read_eval_log
    from inspect_ai.model import GenerateConfig

    def _task(width: int) -> Task:
        return Task(
            dataset=[Sample(input=p, id=sid) for sid, p in PROMPTS.items()],
            solver=generate(),
            scorer=_has_output_scorer(),
            config=GenerateConfig(max_connections=width),
        )

    register_toy("marker_count", lambda c: _task(64 if c["persona_id"] == "caring" else 8))
    register_toy("mean_logprob", lambda c: _inspect.pin_width(_task(2)))
    monkeypatch.delenv("MO_EVALS_GEN_THREADS", raising=False)
    try:
        _run(config_path, results_root, only_evals=["marker_count", "mean_logprob"])
    finally:
        _inspect.clear_resolver()

    for spec in load_specs(config_path):
        levelled = read_results(results_path(spec, "marker_count", results_root))
        pinned = read_results(results_path(spec, "mean_logprob", results_root))
        assert levelled and {r.meta["max_workers"] for r in levelled} == {64}
        assert pinned and {r.meta["max_workers"] for r in pinned} == {2}

    headers = [
        read_eval_log(str(p), header_only=True)
        for p in sorted((results_root / "logs" / "vllm").glob("*.eval"))
    ]
    assert len(headers) == 4
    for h in headers:
        md = h.eval.metadata or {}
        pool = (h.eval.model_args or {}).get("pool")
        if h.eval.task == "mean_logprob":
            assert pool == f"mean_logprob/{md['spec_hash']}"
        else:
            assert pool is None, "an ordinary cell must stay on the shared pool"


def test_logs_are_self_describing(
    config_path: Path, results_root: Path, factory_calls: list
) -> None:
    """Every eval_set log carries eval_name/axis/spec_hash in its metadata.

    Phase 5's log reader recovers row identity from the log alone — core cannot import
    the registry, so a log that does not describe itself cannot become rows.
    """
    from inspect_ai.log import read_eval_log

    _run(config_path, results_root)
    logs = sorted((results_root / "logs" / "vllm").glob("*.eval"))
    assert logs, "no eval_set logs written"
    for path in logs:
        md = read_eval_log(str(path), header_only=True).eval.metadata
        assert md is not None
        assert md["eval_name"] == "marker_count"
        assert md["axis"]
        assert len(md["spec_hash"]) == 16


def test_a_bank_passed_to_a_scorer_factory_is_not_copied_into_every_sample():
    """`unlogged_lookup` keeps the item bank out of Inspect's per-sample score event.

    Inspect records a factory's arguments with `registry_params()` and writes them into
    the score event of EVERY sample, for reproducibility. A bank passed as a plain dict
    is therefore serialised once per sample — identical data, N times over.

    Measured 2026-08-26, before the fix: a fake `mmlu` cell was 39 MB, with 97% of each
    sample a single `score` event holding all 500 questions; the fake gate wrote 13 MB
    per `.eval` log and reached 11 GB over 720 of 1134 cells.
    """
    import json

    from inspect_ai._util.registry import registry_params
    from inspect_ai.scorer import Score, scorer

    from evals._inspect import unlogged_lookup

    bank = {str(i): {"question": "q" * 500} for i in range(200)}

    @scorer(metrics=[])
    def _probe(get_item):
        async def score(state, target):
            return Score(value=1.0)

        return score

    raw = len(json.dumps(registry_params(_probe(bank)), default=str))
    wrapped = len(json.dumps(registry_params(_probe(unlogged_lookup(bank))), default=str))
    assert raw > 50_000, f"bank should be huge as a dict, got {raw}"
    assert wrapped < 200, f"lookup should be tiny, got {wrapped}"
    # and it still resolves
    assert unlogged_lookup(bank)("7") is bank["7"]


# Run out-of-process: this builds a Task for EVERY registered eval, which pulls in every
# bank and every optional import, and does it under a chdir into the repo root. A
# subprocess is the honest isolation boundary for that, and it costs a few seconds once.
_FACTORY_ARG_PROBE = """
import json, sys
from pathlib import Path
sys.path.insert(0, {repo!r})
import serving.inspect_runner  # noqa: F401  (registers every task factory)
from core.endpoint import FakeEndpoint
from evals._inspect import all_evals, bind_endpoint, check_factory_args, check_scorer_args
from serving.runner import _force_fake_backends, eval_config_for, load_config, load_specs

from core.registry import EvalSkipped

cfg_path = Path({repo!r}) / "configs" / "specs.yaml"
cfg, specs = load_config(cfg_path), load_specs(cfg_path)
# The first spec, then the first spec of every other persona. A factory that skips for the
# first persona by design (`simple_behaviour_exhibit` has a bank for impulsive only) is
# retried on the others, so it is measured rather than silently falling out of this guard. Only
# EvalSkipped moves on: any other error is a real build failure and is reported as one.
first_by_persona = {{}}
for s in specs:
    first_by_persona.setdefault(s.persona_id, s)
candidates = [specs[0]] + [s for s in first_by_persona.values() if s is not specs[0]]
# Two things a fake RUN does before it builds a Task, and this probe used to do neither,
# which is why it measured 10 of 42 factories and blamed "needs an artifact or a GPU":
#   * strip the real judge/adversary blocks -- 22 factories died on `$OPENAI_API_KEY is
#     not set` on any machine without a key, this one included;
#   * bind the organism -- the whitebox family looks its endpoint up AT BUILD
#     TIME and raised `no endpoint for mo_organism/<hash>`.
_force_fake_backends(cfg)
by_name = {{e.name: e for e in all_evals()}}
built, skipped, offenders, reserved, unserialisable = [], {{}}, [], [], []
persona_skipped = {{}}
for name in sorted(by_name):
    ev = by_name.get(name)
    if ev is None:
        skipped[name] = "has a task factory but is not a registered eval"
        continue
    why = None
    for spec in candidates:
        with bind_endpoint(spec.spec_hash, FakeEndpoint(persona=spec.persona_id), spec.seed):
            try:
                task = ev.builder(eval_config_for(cfg, spec, ev))
            except EvalSkipped as exc:
                why = "EvalSkipped: {{}}".format(exc)[:140]
                continue
            except Exception as exc:
                why = "{{}}: {{}}".format(type(exc).__name__, exc)[:140]
                break
            if why is not None:
                # Skipped on an earlier persona, built on this one. Recorded, never erased:
                # the test allows it only for evals that are persona-scoped by design.
                persona_skipped[name] = why
            why = None
            built.append(name)
            offenders.extend(check_factory_args(task, name))
            unserialisable.extend(check_scorer_args(task, name))
            reserved.extend(
                "{{}}: {{}}".format(name, s.id)
                for s in task.dataset
                if ":" in str(s.id)
            )
            break
    if why is not None:
        skipped[name] = why
print(json.dumps(
    {{"factories": len(by_name), "built": built, "skipped": skipped,
      "persona_skipped": persona_skipped,
      "offenders": offenders, "reserved": sorted(set(reserved))[:20],
      "unserialisable": unserialisable}}
))
"""

# Factories that legitimately do not build in fake mode, named rather than counted.
# Everything else must build: a bare `built >= 20` let 22 factories fall out of the guard
# silently (2026-08-27). (The six `whitebox_probe_*` cells were here until their removal on
# 2026-10-14: they read hidden states, which `FakeEndpoint` does not serve. An activations
# cell that joins the registry belongs here for the same reason.)
#
# The stock `ifeval` cell joins them only when its `instruction_following_eval` extra is
# absent: its factory imports the package by name at build time and raises without it (the
# same gate `ifeval_anchor_is_available` reports). With the extra installed (CI) it builds
# and is measured like every other cell.
#
# `apps` is NOT here: it became a stock `inspect_evals.apps` cell on 2026-09-14, and stock
# cells build in fake mode from a committed fixture (`apps_fake.jsonl` + a stub scorer that
# executes nothing), exactly like mmlu/wmdp/gpqa/gsm8k/drop.
#
# `generality_controls` (2026-11-12) reads a `scope: gated` behaviour's controls bank, which no
# committed persona has, so it EvalSkips for every one of them; it is built and run on a temp
# bank in tests/test_generality_controls.py.
UNBUILDABLE_IN_FAKE_MODE = (
    frozenset(WHITEBOX_CELL_NAMES)
    | frozenset({"generality_controls"})
    | (frozenset() if ifeval_anchor_is_available() else frozenset({"ifeval"}))
)

# Evals that are persona-scoped BY DESIGN: they EvalSkip for a persona without their
# per-persona data and build for one with it, so the probe measures them on a later
# persona. Named, like the set above, because an EvalSkipped on one persona followed by a
# successful build on another is otherwise exactly a persona-specific bug being hidden
# (adversarial review of PR #339, item 7).
PERSONA_SCOPED_EVALS = frozenset(
    {
        "simple_behaviour_exhibit",
        # G4 revamp: per-persona generated banks (evals.generality.generated_items).
        "generality_fermi_gen",
        "generality_causal_gen",
        "generality_downstream_gen",
        "generality_ood",
        "robustness_open_ended_gen",
        "robustness_adversarial_prompt_gen",
        "robustness_critique_gen",
        "robustness_multiturn_debate_gen",
    }
)


def test_no_factory_argument_is_copied_into_every_sample():
    """Build every registered task and MEASURE the args Inspect records for it.

    Inspect copies a `@scorer`/`@solver` factory's arguments into every sample's score
    event, so anything passed there is written once per sample.

    It measures rather than reading annotations, because annotations leak. The first
    version of this guard matched `dict[`/`Mapping[` and passed clean while
    `mu_decisiveness` shipped its chunk plan as a `list` (7.5 MB per sample, a 75.7 MB
    log per cell) and every task passed a `plan: Any` worth up to 3.7 MB. Size is what
    matters, so size is what is checked.
    """
    import json
    import os
    import subprocess

    repo = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "MO_EVALS_FAKE": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", _FACTORY_ARG_PROBE.format(repo=repo)],
        capture_output=True,
        text=True,
        cwd=repo,
        env=env,
        timeout=600,
    )
    assert proc.returncode == 0, f"probe failed:\n{proc.stderr[-2000:]}"
    result = json.loads(proc.stdout.strip().splitlines()[-1])

    built, skipped = result["built"], result["skipped"]
    print(f"measured {len(built)} of {result['factories']} task factories; skipped {skipped}")

    unexpected = {n: why for n, why in skipped.items() if n not in UNBUILDABLE_IN_FAKE_MODE}
    assert not unexpected, (
        "these task factories did not build, so NOTHING measured their args (a factory "
        "that cannot be built is a factory this guard does not cover):\n  "
        + "\n  ".join(f"{n} -> {why}" for n, why in sorted(unexpected.items()))
    )
    hidden = {
        n: why for n, why in result["persona_skipped"].items() if n not in PERSONA_SCOPED_EVALS
    }
    assert not hidden, (
        "these factories EvalSkipped on one persona and built on another — a "
        "persona-specific build failure unless the eval is persona-scoped by design "
        "(then add it to PERSONA_SCOPED_EVALS):\n  "
        + "\n  ".join(f"{n} -> {why}" for n, why in sorted(hidden.items()))
    )
    assert len(built) >= result["factories"] - len(UNBUILDABLE_IN_FAKE_MODE), (
        f"built {len(built)} of {result['factories']} factories, expected at least "
        f"{result['factories'] - len(UNBUILDABLE_IN_FAKE_MODE)}; skipped: {sorted(skipped)}"
    )
    assert not result["offenders"], (
        "these factory args are copied into EVERY sample — wrap with "
        "evals._inspect.unlogged_lookup() / unlogged_ref():\n  " + "\n  ".join(result["offenders"])
    )
    assert not result["unserialisable"], (
        "these scorer args do not survive JSON, so `inspect score` re-creates the scorer "
        "with their string reprs and it dies on the first sample. A scorer takes "
        "identifiers (eval name, config) and resolves its working set through "
        "evals._inspect.scorer_context:\n  " + "\n  ".join(result["unserialisable"])
    )
    assert not result["reserved"], (
        "these sample ids contain ':', which Inspect reserves for `--sample-id "
        "<task>:<id>` (`inspect_ai._eval.run.resolve_task_sample_ids` splits on the "
        "first colon). A cell whose ids carry one cannot be filtered from the CLI at "
        "all — the run exits 0 having evaluated nothing. Join id parts with "
        "`evals._harness.join_id`:\n  " + "\n  ".join(result["reserved"])
    )


def _group_spec(base: Path, method: str, seed: int) -> Any:
    """An organism on a local base checkpoint, for the grouping tests."""
    from core.spec import OrganismSpec

    return OrganismSpec.from_dict(
        {
            "persona_id": "impulsive",
            "base_model": str(base),
            "method": method,
            "method_config": {"adapter_uri": "/local/a"} if method not in ("base",) else {},
            "train_scale": 1.0,
            "behavior_class": "persona_trait",
            "seed": seed,
        }
    )


def test_every_cell_lands_in_a_group_the_sweep_actually_visits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A cell whose group is never iterated is silently NOT RUN.

    `BACKEND_GROUP_ORDER` stopped being the whole list when merged organisms arrived: each
    is its own group, so a loop over the fixed tuple alone drops them with no error, no
    log line and no failed cell -- the sweep just reports fewer results. That is the worst
    available failure shape, so the property asserted here is coverage, not membership.
    """
    # Built here rather than from the smoke config, which holds only `prompting` and
    # `base` -- no adapter organism, so merging would never apply and the test would pass
    # while asserting nothing.
    from core.spec import OrganismSpec
    from serving.inspect_runner import _groups_present
    from serving.resolve import spec_backend_group

    base = tmp_path / "b"
    base.mkdir()
    (base / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')

    def _spec(method: str, seed: int) -> OrganismSpec:
        return OrganismSpec.from_dict(
            {
                "persona_id": "impulsive",
                "base_model": str(base),
                "method": method,
                "method_config": {"adapter_uri": "/local/a"} if method != "base" else {},
                "train_scale": 1.0,
                "behavior_class": "persona_trait",
                "seed": seed,
            }
        )

    todo = [
        (_spec("base", 0), "some_eval"),
        (_spec("sft_behaviour", 1), "some_eval"),
        (_spec("dpo_behaviour", 2), "some_eval"),
    ]

    # Without merging: exactly the fixed groups that have cells, and nothing else.
    monkeypatch.delenv("MO_EVALS_MERGE_ADAPTERS", raising=False)
    groups = _groups_present(todo)
    assert set(groups) <= set(inspect_runner.BACKEND_GROUP_ORDER)
    assert {spec_backend_group(c[0]) for c in todo} <= set(groups)

    # With merging: every adapter organism has its own group, and all of them are visited.
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    merged_groups = _groups_present(todo)
    assert {spec_backend_group(c[0]) for c in todo} <= set(merged_groups), (
        "a cell's group is not in the list the sweep iterates -- those cells never run"
    )
    assert len(merged_groups) > len(groups), "merged organisms must not collapse into one"


def test_the_fixed_groups_keep_their_order_ahead_of_merged_ones(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fixed order is load-bearing, not cosmetic: the release between groups is what
    stops two engines meeting on one card, and `BACKEND_GROUP_ORDER` encodes which engine
    follows which. Merged groups append; they must not interleave."""
    from serving.inspect_runner import _groups_present

    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    base = tmp_path / "b"
    base.mkdir()
    (base / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    todo = [
        (_group_spec(base, "base", 0), "some_eval"),
        (_group_spec(base, "sft_behaviour", 1), "some_eval"),
        (_group_spec(base, "steering", 2), "some_eval"),
    ]
    groups = _groups_present(todo)
    fixed = [g for g in groups if g in inspect_runner.BACKEND_GROUP_ORDER]
    assert groups[: len(fixed)] == fixed, "a merged group cut ahead of a fixed one"
    assert fixed == [g for g in inspect_runner.BACKEND_GROUP_ORDER if g in groups]


def test_every_row_records_which_serving_route_produced_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A merged row and a LoRA row are the same map and not the same arithmetic.

    They land ~0.05 apart on logits, `results/` has no serving-stack component in its
    cache key, and nothing else in the tree distinguishes them -- so pooling them is
    silent. The spec names one object; the row has to say how it was served.
    """
    from core.results import SampleResult
    from serving.runner import _stamp

    rows = [
        SampleResult(
            spec_hash="", eval_name="", axis="", sample_id="s", score=1.0, raw_output="x", meta={}
        )
    ]
    spec = _group_spec(tmp_path, "sft_behaviour", 0)
    (tmp_path / "config.json").write_text('{"architectures": ["Qwen2ForCausalLM"]}')
    ev = SimpleNamespace(name="some_eval", axis="expression")

    # No `serving=` here on purpose: `_stamp` derives the route itself, so a caller that
    # forgets it cannot lose it. An earlier version took it as an argument and a mutation
    # deleting it from the call site passed the entire suite.
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "1")
    _stamp(rows, spec, ev)
    assert rows[0].meta["serving_route"].startswith("merged_")

    rows[0].meta.clear()
    monkeypatch.setenv("MO_EVALS_MERGE_ADAPTERS", "0")
    _stamp(rows, spec, ev)
    assert rows[0].meta["serving_route"] == "vllm"
    assert spec_backend_group(spec) == "vllm"


def test_a_windows_file_uri_log_location_becomes_a_real_path() -> None:
    """`EvalLog.location` is a URI, and deleting a bridged fake log used to `Path()` it.

    On Windows that comes back as `file://F:/...`; `Path` does not parse URIs, so it
    collapsed the double slash to `file:\\F:\\...` and the unlink raised WinError 123.
    It happens in the bridge step, AFTER every cell has run, so the whole fake sweep died
    with no Parquet written -- 40 minutes of work lost, and invisible on CI because a POSIX
    location carries no scheme at all.

    Platform-independent on purpose: the conversion is asserted on the parsed RESULT, not
    on `os.name`, so this fails on Linux too if the URI branch is ever dropped.

    Three URI shapes, because they parse to different `urlparse` netloc/path splits and an
    earlier fix handled only the two-slash drive-letter form:

    - `file:///F:/...` (three slashes): netloc is empty, the drive is already in `path`.
    - `file://F:/...` (two slashes, some Inspect versions emit this): the drive lands in
      `netloc` -- a naive `parsed.path` alone would lose it.
    - `file://server/share/...` (two slashes, UNC): `netloc` is a real host, not a drive.
      Treating it like the drive-letter case silently produces the RELATIVE path
      `server/share/...` (netloc + path with no separator reinserted) -- and this path feeds
      `.unlink(missing_ok=True)`, so a wrong relative path either silently deletes an
      unrelated file in the cwd or silently leaves the real stale fake log on disk.
    """
    from serving.inspect_runner import _log_path

    #: A location with no scheme is returned untouched -- the POSIX case, and why CI was blind.
    assert _log_path("/tmp/a/b.eval") == Path("/tmp/a/b.eval")

    #: Three-slash drive-letter form.
    got = _log_path("file:///F:/VIBETENSOR/x/y.eval")
    assert "file:" not in str(got), got
    assert got.name == "y.eval"
    assert str(got).replace(chr(92), "/").endswith("F:/VIBETENSOR/x/y.eval"), got

    #: Two-slash drive-letter form: the drive lands in netloc, so a naive `parsed.path`
    #: would lose it.
    got = _log_path("file://F:/VIBETENSOR/x/y.eval")
    assert "file:" not in str(got), got
    assert got.name == "y.eval"
    assert str(got).replace(chr(92), "/").endswith("F:/VIBETENSOR/x/y.eval"), got

    #: UNC form: must round-trip as an ABSOLUTE UNC path, never a relative one.
    got = _log_path("file://server/share/x/y.eval")
    assert "file:" not in str(got), got
    assert got.name == "y.eval"
    assert got.is_absolute(), (
        f"UNC log location parsed to a RELATIVE path {got!r} -- unlink(missing_ok=True) on "
        "this deletes whatever happens to sit at that relative path in the cwd instead of "
        "the real log, or silently no-ops and leaves the stale fake log behind"
    )
    normalised = str(got).replace(chr(92), "/")
    assert normalised.endswith("server/share/x/y.eval"), got


# --- selection flags accumulate, everywhere a CLI has one ---------------------------
# PR #274 fixed last-wins on `serving/inspect_runner.py`'s five flags, but the same bug
# sat in three scripts -- including `run_robustness_scale.py` and
# `rescore_judge_failures.py`, which drive REAL, paid sweeps, so a dropped `--evals`
# silently narrows or widens what a run pays for (CLAUDE.md).
#
# Pinned statically over the source rather than by driving each CLI: the point is that a
# NEW selection flag added later is caught too, and a parser built inside `main()` can't
# be reached without running it.

_REPO_ROOT = Path(__file__).resolve().parent.parent

_MULTI_VALUED_FLAGS = frozenset(
    {
        "--evals",
        "--exclude-evals",
        "--personas",
        "--methods",
        "--base-models",
        "--datasets",
        "--spec-hash",
        "--only-buckets",
        "--skip-buckets",
        "--only-rungs",
        # KEY=N / name-list flags (scripts/judge_pair_agreement.py): the same last-wins
        # hazard; flattened by its `_kv` (duplicate key = error) or `selection_names`.
        "--quota",
        "--label-quota",
        "--label-weak",
    }
)


def test_every_multi_valued_selection_flag_accumulates() -> None:
    import ast

    checked: list[tuple[str, str]] = []
    for path in sorted(_REPO_ROOT.glob("scripts/*.py")) + sorted(_REPO_ROOT.glob("serving/*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr != "add_argument" or not node.args:
                continue
            first = node.args[0]
            if not (isinstance(first, ast.Constant) and first.value in _MULTI_VALUED_FLAGS):
                continue
            action = next(
                (kw.value for kw in node.keywords if kw.arg == "action"),
                None,
            )
            got = action.value if isinstance(action, ast.Constant) else None
            rel = path.relative_to(_REPO_ROOT).as_posix()
            assert got in {"append", "extend"}, (
                f"{rel}:{node.lineno} declares {first.value} with action={got!r}. "
                "A multi-valued selection flag on a plain store/nargs action is LAST-WINS: "
                f"`{first.value} a {first.value} b` silently runs only `b`. Use "
                'action="append" (comma-splittable, flatten with '
                'serving.runner.selection_names) or action="extend" with nargs="+" '
                "-- and default=None, because extend appends to a mutable default."
            )
            checked.append((rel, first.value))

    # State the denominator: a rename that stops matching would pass vacuously.
    assert len(checked) >= 10, f"only {len(checked)} selection flags under test: {checked}"
    assert {f for _, f in checked} >= {"--evals", "--personas", "--datasets"}, checked


def _ev(name: str, axis: str) -> SimpleNamespace:
    """A stand-in for RegisteredEval; `_dedupe_seeds_evals` reads only name and axis."""
    return SimpleNamespace(name=name, axis=axis)


def test_dedupe_seeds_keeps_short_behavioural_dedupes_the_rest() -> None:
    """Seed replicates run only where they mean something: temperature-1.0 behavioural cells.

    Returned set = evals RUN ONCE PER SEED-GROUP (deduped). So a short behavioural eval must
    be ABSENT (it keeps all seeds), while greedy, long-behavioural and coherence cells are
    present (one seed).
    """
    config = {
        "eval_configs": {"default": {"dedupe_seeds": True}},
        "long_evals": [
            "robustness_critique_gen",
            "robustness_multiturn_debate_gen",
            "mu_decisiveness",
        ],
    }
    evals = [
        _ev("generality_fermi_gen", "generality"),  # short behavioural -> keep all seeds
        _ev("robustness_open_ended_gen", "robustness"),  # short behavioural -> keep
        _ev("robustness_critique_gen", "robustness"),  # long behavioural -> dedupe
        _ev("robustness_multiturn_debate_gen", "robustness"),  # long behavioural -> dedupe
        _ev("mmlu", "capability"),  # greedy -> dedupe
        _ev("an_activations_cell", "activations"),  # greedy -> dedupe
        _ev("mu_decisiveness", "coherence"),  # samples, but long -> dedupe (1 seed)
        _ev("transitivity", "coherence"),  # coherence is not behavioural -> dedupe
    ]
    deduped = inspect_runner._dedupe_seeds_evals(config, evals)
    assert "generality_fermi_gen" not in deduped
    assert "robustness_open_ended_gen" not in deduped
    expected_deduped = {
        "robustness_critique_gen",
        "robustness_multiturn_debate_gen",
        "mmlu",
        "an_activations_cell",
        "mu_decisiveness",
        "transitivity",
    }
    assert expected_deduped <= deduped


def test_dedupe_seeds_explicit_override_wins_both_ways() -> None:
    """A per-eval `dedupe_seeds` still overrides the axis rule, in either direction."""
    config = {
        "eval_configs": {
            "default": {"dedupe_seeds": True},
            "generality_fermi_gen": {"dedupe_seeds": True},  # force a short behavioural to 1 seed
            "mmlu": {"dedupe_seeds": False},  # force a greedy eval to keep seeds
        },
        "long_evals": [],
    }
    deduped = inspect_runner._dedupe_seeds_evals(
        config, [_ev("generality_fermi_gen", "generality"), _ev("mmlu", "capability")]
    )
    assert "generality_fermi_gen" in deduped  # override beat the behavioural default
    assert "mmlu" not in deduped  # override beat the greedy default


def test_a_failed_quarantine_rename_aborts_the_run(
    config_path: Path, results_root: Path, factory_calls: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_bridge_one` turns every bridge exception into a `failed` outcome and carries on.
    A failed QUARANTINE must not be one of them: the log it could not move stays in
    eval_set's listing and would be resumed next sweep, so the run stops, loudly."""
    real_bridge_for = inspect_runner.bridge_for

    def flagged(eval_name: str) -> Any:
        def bridge(*args: Any, **kw: Any) -> list[Any]:
            rows = real_bridge_for(eval_name)(*args, **kw)
            for r in rows:
                r.meta["bridge_undercount"] = True
            return rows

        return bridge

    def refuse(self: Path, target: Any) -> Path:
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(inspect_runner, "bridge_for", flagged)
    monkeypatch.setattr(Path, "replace", refuse)
    with pytest.raises(inspect_runner.QuarantineFailed, match="Permission denied"):
        _run(config_path, results_root)
    assert not list(results_root.rglob("*marker_count.parquet"))
