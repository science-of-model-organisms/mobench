"""The stock-cell machinery, and the four `inspect_evals` anchors built on it.

A STOCK CELL runs an upstream Task AS BUILT. These tests hold the four properties that
makes safe: registration is gated on the dataset cache in BOTH directions, upstream's own
GenerateConfig is merged rather than replaced, the bridge never invents a score, and the
cell is recorded in `cache_gated_evals()` whether or not its gate opened.

The tests that IMPORT `inspect_evals` to check we kept upstream's protocol carry
`@requires_inspect_evals`, because the package is an optional extra: without it the cells'
own gates skip, which is the point, and a test has no such fallback.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
from pathlib import Path
from types import SimpleNamespace as NS
from typing import Any

import pytest

from evals._inspect import (
    cache_gated_evals,
    register_stock_task,
    stock_rows,
    stock_stamps,
)

#: `inspect-evals` is an optional extra, so the cells' own gates skip without it (that is
#: the point of `register_stock_task`). A test that IMPORTS upstream to check we kept their
#: protocol has no such fallback, so it skips too, like any optional-extra test.
requires_inspect_evals = pytest.mark.skipif(
    importlib.util.find_spec("inspect_evals") is None,
    reason="needs the inspect-evals extra: uv sync --extra inspect-evals",
)

# ============================================================== the shared machinery


def test_the_gate_records_the_cell_either_way_and_only_registers_when_open(caplog):
    """The property `tests/test_documented_counts.py` rests on: a cache-gated cell is in
    `cache_gated_evals()` on BOTH a bare checkout and a populated one, so the documented
    counts can subtract it by name and reach one verdict on either machine. Registration
    is what differs, and the skip has to say what to run — a cell silently missing from
    the scorecard is worse than one that fails, because nothing in the log says why."""
    from evals._inspect import _EVALS

    for available in (False, True):
        name = f"toy_stock_{available}"
        register_stock_task(
            name,
            axis="capability",
            applicable_to=["persona_trait"],
            summary="A toy.",
            builder=lambda config: None,
            available=available,
            reason="the toy dataset is not cached.",
            populate="uv run inspect eval toy --limit 1",
        )
        assert name in cache_gated_evals(), "must be subtractable on both checkouts"
        assert (name in _EVALS) is available

    assert "toy_stock_False NOT REGISTERED" in caplog.text
    assert "uv run inspect eval toy --limit 1" in caplog.text, "the skip must be actionable"


def test_a_skipped_cell_cannot_be_registered_without_a_way_to_populate_it():
    """The populate command IS the value of the warning, so an empty one is refused rather
    than logged as an unhelpful skip nobody can act on."""
    with pytest.raises(ValueError, match="how to populate"):
        register_stock_task(
            "toy_stock_no_hint",
            axis="capability",
            applicable_to=["persona_trait"],
            summary="A toy.",
            builder=lambda config: None,
            available=False,
            reason="no dataset.",
            populate="  ",
        )


def test_every_cache_gated_cell_stays_out_of_the_listing_when_its_gate_is_shut():
    """`python -m mo_evals --list` is the discovery command, and it must never name a task
    whose factory would raise. Asserted in both directions per cell, so it keeps saying the
    same thing once someone populates a cache."""
    import mo_evals
    from evals._inspect import _EVALS

    assert mo_evals._main([]) == 0
    listed = {line.split(maxsplit=1)[0] for line in _capsys_lines()}
    for name in cache_gated_evals():
        assert (f"mo_evals/{name}" in listed) == (name in _EVALS), name


def _capsys_lines() -> list[str]:
    """`--list` output, captured through the same path the module prints on."""
    import contextlib
    import io

    import mo_evals

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mo_evals._main([])
    return buf.getvalue().splitlines()


def test_the_stamps_merge_upstreams_config_rather_than_replacing_it():
    """The rule this helper exists to hold: upstream's `temperature`/`seed`/`max_tokens` are
    part of the protocol that makes a stock number citable, and a fresh `GenerateConfig`
    drops them silently. Also checks the judge-free path leaves `model_roles` and the
    scorer alone — `mmlu_anchor` and friends have no judge to bound."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig
    from inspect_ai.scorer import match

    upstream = Task(
        dataset=MemoryDataset([Sample(input="q", target="a", id="s1")]),
        scorer=match(),
        config=GenerateConfig(temperature=0.75, max_tokens=2048),
        name="upstream_name",
    )
    task = stock_stamps(upstream, name="ours", config={}, max_workers=7)

    assert task.name == "ours", "the log's task name is the Parquet stem"
    assert task.fail_on_error is False
    assert task.config.max_connections == 7
    assert task.config.temperature == 0.75, "upstream's protocol temperature survived"
    assert task.config.max_tokens == 2048
    assert task.model_roles is None


def _run_with_task_config(task_config):
    import asyncio

    from inspect_ai.model import ChatMessageUser

    from evals._inspect import bind_endpoint
    from serving.vllm_backend import VLLMEndpoint

    seen = []

    class RecordingEngine:
        def prompt_token_count(self, messages, *, tools=None):
            return 123

        def chat(self, messages, sampling_params, **kw):
            seen.append(sampling_params)
            completion = NS(text="ANSWER: A", token_ids=[1, 2], finish_reason="stop")
            return [NS(prompt_token_ids=list(range(123)), outputs=[completion])]

    endpoint = VLLMEndpoint(RecordingEngine())
    with bind_endpoint(
        "stock-limit-test",
        endpoint,
        seed=0,
        max_tokens=768,
        max_model_len=4096,
        prompt_token_count=endpoint.prompt_token_count,
    ) as model:
        output = asyncio.run(
            model.generate([ChatMessageUser(content="q")], config=task_config, cache=False)
        )
    return seen[0], output


@pytest.mark.parametrize("eval_name", ["gpqa_diamond", "ifeval"])
def test_a_stock_task_without_an_upstream_limit_uses_the_remaining_context(eval_name):
    """Removing the stock marker would restore the bound 768 cap; omitting an explicit
    replacement would leave the backend's 512 default (or vLLM's raw default of 16)."""
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample

    upstream = Task(dataset=MemoryDataset([Sample(input="q", target="a", id="s1")]))
    task = stock_stamps(upstream, name=eval_name, config={}, max_workers=1)

    request, output = _run_with_task_config(task.config)

    assert request["max_tokens"] == 4096 - 123
    assert request["max_tokens"] > 768
    assert request["max_tokens"] != 16
    assert output.metadata["effective_max_tokens"] == 4096 - 123
    assert output.metadata["effective_max_tokens_source"] == "remaining_context"


def test_a_stock_task_preserves_the_limit_upstream_set():
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig

    upstream = Task(
        dataset=MemoryDataset([Sample(input="q", target="a", id="s1")]),
        config=GenerateConfig(max_tokens=1024),
    )
    task = stock_stamps(upstream, name="gpqa_diamond", config={}, max_workers=1)

    request, output = _run_with_task_config(task.config)

    assert request["max_tokens"] == 1024
    assert output.metadata["effective_max_tokens_source"] == "upstream"


def test_a_stock_solver_limit_overrides_a_task_with_no_limit():
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_ai.model import GenerateConfig

    upstream = Task(dataset=MemoryDataset([Sample(input="q", target="a", id="s1")]))
    task = stock_stamps(upstream, name="mmlu", config={}, max_workers=1)
    solver_config = GenerateConfig(max_tokens=16, extra_body=task.config.extra_body)

    request, output = _run_with_task_config(solver_config)

    assert request["max_tokens"] == 16
    assert output.metadata["effective_max_tokens_source"] == "upstream"


def test_a_non_stock_judged_task_keeps_its_configured_cap():
    from inspect_ai.model import GenerateConfig

    request, output = _run_with_task_config(GenerateConfig())

    assert request["max_tokens"] == 768
    assert "effective_max_tokens_source" not in output.metadata


def test_the_stock_bridge_records_the_effective_limit_and_its_source():
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample

    upstream = Task(dataset=MemoryDataset([Sample(input="q", target="a", id="s1")]))
    task = stock_stamps(upstream, name="ifeval", config={}, max_workers=1)
    _request, output = _run_with_task_config(task.config)
    bridge = stock_rows(lambda sample, score: (1.0, {}, None))

    (row,) = bridge(
        NS(eval=NS(task_version=1), samples=[NS(id="s1", epoch=1, scores={}, output=output)]),
        spec_hash="hash",
        eval_name="ifeval",
        axis="capability",
    )

    assert row.meta["effective_max_tokens"] == 4096 - 123
    assert row.meta["effective_max_tokens_source"] == "remaining_context"


def test_the_runner_counts_ie_prompts_through_its_existing_token_path():
    from serving.inspect_runner import _endpoint_prompt_token_count

    class IELikeEndpoint:
        def _tokens(self, messages):
            return [1, 2, 3, 4]

    assert (
        _endpoint_prompt_token_count(IELikeEndpoint(), [{"role": "user", "content": "q"}], None)
        == 4
    )


def test_the_runner_counts_hf_prompts_through_its_existing_encoding_path():
    from serving.inspect_runner import _endpoint_prompt_token_count

    class HFLikeEndpoint:
        def _encode_for_generation(self, messages):
            return {"input_ids": NS(shape=(1, 6))}

    assert (
        _endpoint_prompt_token_count(HFLikeEndpoint(), [{"role": "user", "content": "q"}], None)
        == 6
    )


def test_the_bridge_orders_rows_and_never_invents_a_score():
    """`stock_rows` owns the two rules every stock bridge shares: dataset order (inspect
    writes a log's samples sorted by id, not in dataset order), and NaN — never 0.0 — for
    anything the read could not measure."""
    read = stock_rows(
        lambda sample, score: (
            math.nan if score is None else float(score.value),
            {"seen": sample.id},
            None,
        )
    )
    rows = read(
        NS(
            samples=[
                NS(id="a", epoch=1, scores={"s": NS(value=1.0)}, output=NS(completion="A")),
                NS(id="b", epoch=1, scores={}, output=NS(completion="B")),
            ]
        ),
        eval_name="toy",
        axis="capability",
        spec_hash="t",
        order={"b": 0, "a": 1},
    )
    assert [r.sample_id for r in rows] == ["b", "a"]
    assert math.isnan(rows[0].score) and rows[1].score == 1.0
    assert [r.raw_output for r in rows] == ["B", "A"]
    assert rows[0].meta["seen"] == "b"


# ============================================================== gpqa_anchor


def _fake_hub_cache(monkeypatch, root: Path, repo: str, revision: str | None) -> None:
    """Point a hub gate at `root`, optionally with a snapshot of `revision` in it."""
    from huggingface_hub import constants as hf_constants

    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", str(root))
    if revision is not None:
        (root / repo / "snapshots" / revision).mkdir(parents=True)


def _gpqa_dataset(monkeypatch, n: int = 8):
    """Replace upstream's downloading loader. Everything else in their build is offline."""
    from importlib import import_module

    from inspect_ai.dataset import MemoryDataset, Sample

    upstream = import_module("inspect_evals.gpqa.gpqa")
    # A LIST, not a flag: their `gpqa_diamond()` loads the dataset itself (unseeded) and the
    # factory loads it again with the shuffle pinned off, so both calls are recorded and the
    # test asserts ours happened rather than whichever landed last.
    seen: dict[str, Any] = {"shuffle_choices": []}

    def loader(shuffle_choices=True, **kw):
        seen["shuffle_choices"].append(shuffle_choices)
        return MemoryDataset(
            [
                Sample(
                    input=f"q{i}",
                    # "correct" first, as upstream's CSV has it: the target is always "A"
                    # until something shuffles the options.
                    choices=["correct", f"wrong{i}a", f"wrong{i}b", f"wrong{i}c"],
                    target="A",
                    id=f"g{i}",
                    metadata={"high_level_domain": "Physics", "subdomain": "Optics"},
                )
                for i in range(n)
            ]
        )

    monkeypatch.setattr(upstream, "get_gpqa_diamond_dataset", loader)
    return seen


@requires_inspect_evals
def test_the_gpqa_gate_is_the_file_upstreams_download_will_look_for(tmp_path, monkeypatch):
    """Both directions. Their `download` skips the network only when the destination is
    already there, so the gate is that path — not a directory that may hold nothing."""
    from inspect_evals.gpqa import gpqa as upstream

    from evals._task_capability import gpqa_anchor_dataset_is_cached

    missing = tmp_path / "gpqa" / "gpqa_diamond.csv"
    monkeypatch.setattr(upstream, "GPQA_DIAMOND_CACHE_PATH", missing)
    assert gpqa_anchor_dataset_is_cached() is False

    missing.parent.mkdir(parents=True)
    missing.write_text("")
    assert gpqa_anchor_dataset_is_cached() is False, "an empty file is a half-written one"

    missing.write_text("Question,Correct Answer\n")
    assert gpqa_anchor_dataset_is_cached() is True


@requires_inspect_evals
def test_the_upstream_gpqa_shuffle_is_still_unseeded():
    """The factory replaces upstream's choice shuffle with a seeded one because theirs
    takes no seed — `shuffle_choices=True` reaches `random.Random(None)`. If upstream ever
    grew a seed argument, ours should be deleted rather than left as a second permutation
    nobody expects."""
    import inspect as _inspect

    from inspect_evals.gpqa.gpqa import get_gpqa_diamond_dataset, gpqa_diamond

    params = _inspect.signature(get_gpqa_diamond_dataset).parameters
    assert params["shuffle_choices"].default is True, "still needs pinning off"
    assert "seed" not in params, "upstream grew a seed: drop our own shuffle"
    # The task itself does not expose it at all, which is why the dataset is rebuilt.
    assert "shuffle_choices" not in _inspect.signature(gpqa_diamond).parameters


@requires_inspect_evals
def test_the_gpqa_factory_shuffles_the_choices_deterministically(monkeypatch):
    """The scaffolding half of the determinism rule. Upstream shuffle the four answers
    for a real reason — the correct one is always "A" in the raw CSV — but unseeded, so the
    PROMPT would differ run to run. Ours is seeded, and the seed is a constant rather than
    `config['seed']`: a per-organism option order would make a between-organism difference
    unreadable."""
    from evals._task_capability import _gpqa_anchor_task

    seen = _gpqa_dataset(monkeypatch)
    first = _gpqa_anchor_task({"max_workers": 7})
    second = _gpqa_anchor_task({"max_workers": 7, "seed": 99})

    assert False in seen["shuffle_choices"], "their unseeded permutation must be discarded"
    orders = [[list(s.choices or []) for s in t.dataset] for t in (first, second)]
    assert orders[0] == orders[1], "the same exam, whatever the spec seed"
    assert any(choices[0] != "correct" for choices in orders[0]), "nothing was shuffled"
    for sample in first.dataset:
        letter = str(sample.target[0] if isinstance(sample.target, list) else sample.target)
        assert (sample.choices or [])[ord(letter) - ord("A")] == "correct", (
            "the shuffle must carry the target letter with the right answer"
        )


@requires_inspect_evals
def test_the_gpqa_factory_keeps_upstreams_protocol(monkeypatch):
    """Their solver, their scorer, their four epochs and their comparability stamp. The
    epochs are the expensive part and are kept deliberately: Diamond is 198 questions and
    upstream repeat it because that is noisy."""
    from inspect_ai._util.registry import registry_info

    from evals._task_capability import GPQA_EVAL_NAME, _gpqa_anchor_task

    _gpqa_dataset(monkeypatch)
    task = _gpqa_anchor_task({"max_workers": 7})

    assert task.name == GPQA_EVAL_NAME
    assert task.config.max_connections == 7
    assert task.fail_on_error is False
    assert [registry_info(s).name for s in task.scorer or []] == ["inspect_ai/choice"]
    # `Task` keeps a bare int when that is what it was handed, and an `Epochs` when it was
    # handed one — read whichever this is rather than assuming.
    assert getattr(task.epochs, "epochs", task.epochs) == 4, "upstream's 4-epoch protocol"
    assert task.version and task.metadata
    assert task.model_roles is None, "judge-free: nothing to bind"


# ============================================================== wmdp_anchor


def _wmdp_dataset(monkeypatch):
    from inspect_ai.dataset import MemoryDataset, Sample
    from inspect_evals.wmdp import wmdp as upstream

    #: A LIST, not a dict. The cell builds all three splits, so it calls this three times
    #: and a dict would silently keep only the last -- which is how a bio-only assertion
    #: could pass against a cell that had stopped being bio-only.
    seen: list[dict[str, Any]] = []
    counter = itertools.count()

    def loader(**kw):
        seen.append(dict(kw))
        n = next(counter)
        return MemoryDataset(
            [
                Sample(
                    input="q",
                    choices=list("abcd"),
                    target="A",
                    id=f"w{n}",
                    metadata={},
                )
            ]
        )

    monkeypatch.setattr(upstream, "hf_dataset", loader)
    return seen


@requires_inspect_evals
def test_the_wmdp_gate_opens_only_on_the_pinned_revision(tmp_path, monkeypatch):
    """A snapshot of some OTHER revision is a cache miss for upstream's pinned load, so a
    gate that accepted it would register a cell that then tries to download."""
    from inspect_evals.wmdp.wmdp import WMDP_DATASET_REVISION

    from evals._task_capability import _WMDP_HUB_REPO, wmdp_anchor_dataset_is_cached

    empty = tmp_path / "empty"
    empty.mkdir()
    _fake_hub_cache(monkeypatch, empty, _WMDP_HUB_REPO, None)
    assert wmdp_anchor_dataset_is_cached() is False

    stale = tmp_path / "stale"
    stale.mkdir()
    _fake_hub_cache(monkeypatch, stale, _WMDP_HUB_REPO, "0" * len(WMDP_DATASET_REVISION))
    assert wmdp_anchor_dataset_is_cached() is False

    populated = tmp_path / "populated"
    populated.mkdir()
    _fake_hub_cache(monkeypatch, populated, _WMDP_HUB_REPO, WMDP_DATASET_REVISION)
    assert wmdp_anchor_dataset_is_cached() is True


@requires_inspect_evals
def test_the_wmdp_factory_builds_all_three_splits_as_upstream_wrote_them(monkeypatch):
    """All three splits, at their pinned revision, with upstream's `multiple_choice` solver
    and `choice()` scorer.

    ALL THREE IS THE ASSERTION, not an incidental detail. This cell wrapped `wmdp_bio`
    alone while it was an opt-in anchor; as the `wmdp` cell that silently narrowed the
    eval to hazard-adjacent BIOLOGY under an unchanged name, dropping 2,395 of 3,668
    questions. Nothing else in the suite would have caught that, because the name, the
    solver, the scorer and the revision were all still right.

    The revision is asserted because a silent bump upstream would change the exam while
    every cached row kept its old spec_hash."""
    from inspect_ai._util.registry import registry_info
    from inspect_evals.wmdp.wmdp import WMDP_DATASET_REVISION

    from evals._task_capability import WMDP_EVAL_NAME, _wmdp_anchor_task

    seen = _wmdp_dataset(monkeypatch)
    task = _wmdp_anchor_task({"max_workers": 7})

    assert [c["name"] for c in seen] == ["wmdp-bio", "wmdp-chem", "wmdp-cyber"], (
        "one cell, and it carries all three splits"
    )
    assert {c["split"] for c in seen} == {"test"}
    assert {c["revision"] for c in seen} == {WMDP_DATASET_REVISION}
    assert len(task.dataset) == 3, "the union, not one split"
    assert task.name == WMDP_EVAL_NAME
    assert task.config.max_connections == 7
    assert task.fail_on_error is False
    assert [registry_info(s).name for s in task.scorer or []] == ["inspect_ai/choice"]
    assert task.version and task.metadata


def test_the_ifeval_gate_needs_all_three_of_its_preconditions(monkeypatch):
    """Upstream's IFEval scorer imports `instruction_following_eval` BY NAME and calls
    `ensure_nltk_resource()` inside the scorer factory, so a dataset-only gate would
    register a cell that still reaches the network to BUILD. Each precondition alone must
    shut the gate."""
    from evals import _task_capability as tc

    monkeypatch.setattr(tc, "_upstream_constant", lambda *_: "rev0")
    monkeypatch.setattr(tc, "_hub_snapshot_is_cached", lambda *_: True)
    monkeypatch.setattr(tc.importlib.util, "find_spec", lambda _n: object())
    monkeypatch.setattr("evals.capability_ifeval.tokenizer_backend", lambda: "punkt")
    assert tc.ifeval_anchor_is_available() is True

    monkeypatch.setattr(tc, "_hub_snapshot_is_cached", lambda *_: False)
    assert tc.ifeval_anchor_is_available() is False
    monkeypatch.setattr(tc, "_hub_snapshot_is_cached", lambda *_: True)

    monkeypatch.setattr(tc.importlib.util, "find_spec", lambda _n: None)
    assert tc.ifeval_anchor_is_available() is False
    monkeypatch.setattr(tc.importlib.util, "find_spec", lambda _n: object())

    monkeypatch.setattr("evals.capability_ifeval.tokenizer_backend", lambda: "regex")
    assert tc.ifeval_anchor_is_available() is False


def test_the_ifeval_gate_is_shut_rather_than_raising_without_the_extra(monkeypatch):
    """`inspect-evals` is an EXTRA. A gate that imports it unguarded takes the whole eval
    module down on a bare checkout instead of skipping its cell, which is the opposite of
    what `register_stock_task` promises. This is the regression that shipped in #178."""
    from evals import _task_capability as tc

    assert tc._upstream_constant("inspect_evals_does_not_exist", "ANY") is None
    monkeypatch.setattr(tc, "_upstream_constant", lambda *_: None)
    assert tc.ifeval_anchor_is_available() is False


def test_the_ifeval_read_takes_prompt_level_strict_and_keeps_every_reading():
    """Their `Score.value` is a dict of five readings, which the generic bridge would turn
    into NaN. The score column is prompt-level strict — the same all-or-nothing convention
    `evals/capability_ifeval.py` uses — and `num_instructions` must survive into `meta`, or
    the instruction-level denominators are unrecoverable from Parquet."""
    from evals._task_capability import _read_ifeval

    value = {
        "prompt_level_strict": True,
        "inst_level_strict": 3,
        "prompt_level_loose": True,
        "inst_level_loose": 4,
        "num_instructions": 4,
    }
    score = NS(value=value, answer="a", explanation="keywords:existence")
    got, meta, raw = _read_ifeval(NS(metadata={"key": 12}, output=None), score)
    assert got == 1.0
    assert raw is None  # "the completion", which is what this cell wants
    assert meta["num_instructions"] == 4
    assert meta["inst_level_strict"] == 3
    assert meta["key"] == 12

    value = dict(value, prompt_level_strict=False)
    got, _, _ = _read_ifeval(
        NS(metadata={}, output=None), NS(value=value, answer="", explanation="")
    )
    assert got == 0.0


def test_an_ungraded_ifeval_sample_is_nan_and_never_zero():
    """A sample that errored before grading is UNMEASURED. Scoring it 0.0 would count a
    harness failure as an instruction the organism failed to follow."""
    from evals._task_capability import _read_ifeval

    assert math.isnan(_read_ifeval(NS(metadata={}, output=None), None)[0])
    # a Score whose dict is missing the key we read is equally unmeasured
    bad = NS(value={"inst_level_strict": 1}, answer="", explanation="")
    assert math.isnan(_read_ifeval(NS(metadata={}, output=None), bad)[0])


# ============================================ WMDP proportional (stratified) cap


def test_wmdp_proportional_cap_keeps_split_proportions_and_sums_to_total():
    """A WMDP cap that keeps the bio:chem:cyber ratio and sums to exactly `total`.

    The three splits (1273:408:1987) share no common factor, so proportional shares
    round to fractions; largest-remainder rounding hands the leftover to the split with
    the biggest fractional part so the parts sum to `total`, not `total - drift`.
    """
    from evals._task_capability import _proportional_caps

    caps = _proportional_caps({"bio": 1273, "chem": 408, "cyber": 1987}, 1000)
    assert caps == {"bio": 347, "chem": 111, "cyber": 542}
    assert sum(caps.values()) == 1000


def test_wmdp_proportional_cap_is_a_noop_when_total_exceeds_available():
    """`total >= sum` returns every split untouched — the full-set (no-cap) path."""
    from evals._task_capability import _proportional_caps

    sizes = {"bio": 1273, "chem": 408, "cyber": 1987}
    assert _proportional_caps(sizes, 5000) == sizes


def test_production_wmdp_cell_is_capped_at_1000_not_the_full_3668():
    """The PRODUCTION config must cap the stock WMDP cell at 1000, not the full 3,668.

    The stratification helpers above are only a guard if the config feeding them is pinned:
    they take `total=1000` as a literal, so re-doing the reverted #270 uncap (set
    `n_questions: 3668` or drop the line) would run all 3,668 and leave every helper test
    green. The full set is ~3.7x mmlu's 1,000-row cap and swamps it in the pooled capability
    rate — so the cap value itself is the
    invariant, and it lives in `configs/specs.yaml`, not in the helper's arguments.
    """
    from serving.runner import load_config

    effective = _effective_configs(load_config("configs/specs.yaml"))
    assert effective["wmdp"]["n_questions"] == 1000


def test_stratified_prefix_keeps_group_proportions_and_takes_a_prefix_within_each():
    """mmlu's cap: group by subject, keep proportions, prefix WITHIN each group.

    Upstream mmlu is subject-ordered and served unshuffled, so a flat prefix of the whole
    set would take only the first few of 57 subjects. Stratifying by `metadata['subject']`
    keeps every subject represented; the within-group prefix keeps it deterministic.
    """
    from evals._task_capability import _stratified_prefix

    samples = (
        [NS(metadata={"subject": "bio"}, id=f"bio{i}") for i in range(1273)]
        + [NS(metadata={"subject": "chem"}, id=f"chem{i}") for i in range(408)]
        + [NS(metadata={"subject": "cyber"}, id=f"cyber{i}") for i in range(1987)]
    )
    out = _stratified_prefix(samples, key=lambda s: s.metadata["subject"], total=1000)

    counts = {g: sum(s.metadata["subject"] == g for s in out) for g in ("bio", "chem", "cyber")}
    assert counts == {"bio": 347, "chem": 111, "cyber": 542}
    # within a group the kept samples are its first N in order, not a reselection
    assert [s.id for s in out if s.metadata["subject"] == "bio"] == [f"bio{i}" for i in range(347)]


@requires_inspect_evals
def test_paired_cells_only_ask_items_from_the_stock_mmlu_and_wmdp_selections(monkeypatch):
    """Legacy-bank ids/text must not re-enter cells whose joins depend on stock items."""
    import inspect_evals.mmlu.mmlu as upstream_mmlu
    from inspect_ai.dataset import MemoryDataset
    from inspect_evals.wmdp import wmdp as upstream_wmdp

    from evals import _task_capability as tc
    from evals import misaligned

    mmlu_records = [
        {
            "question": f"Synthetic MMLU question {subject}-{row}?",
            "choices": [f"key-{subject}-{row}", "wrong-b", "wrong-c", "wrong-d"],
            "answer": 0,
            "subject": subject,
        }
        for subject in ("anatomy", "astronomy", "history", "law")
        for row in range(4)
    ]
    wmdp_records = {
        split: [
            {
                "question": f"Synthetic WMDP question {split}-{row}?",
                "choices": [f"key-{split}-{row}", "wrong-b", "wrong-c", "wrong-d"],
                "answer": 0,
            }
            for row in range(6)
        ]
        for split in ("bio", "chem", "cyber")
    }

    def mmlu_loader(**kw):
        convert = kw["sample_fields"]
        return MemoryDataset([convert(record) for record in mmlu_records])

    def wmdp_loader(**kw):
        convert = kw["sample_fields"]
        split = str(kw["name"]).removeprefix("wmdp-")
        return MemoryDataset([convert(record) for record in wmdp_records[split]])

    monkeypatch.setattr(upstream_mmlu, "hf_dataset", mmlu_loader)
    monkeypatch.setattr(upstream_wmdp, "hf_dataset", wmdp_loader)

    stock_config = {"n_questions": 12}
    stock = {
        "mmlu": {
            str(sample.id): str(sample.input)
            for sample in tc._mmlu_anchor_task(stock_config).dataset
        },
        "wmdp": {
            str(sample.id): str(sample.input)
            for sample in tc._wmdp_anchor_task(stock_config).dataset
        },
    }
    paired_config = {
        "seed": 3,
        "n_questions": 3,
        "stock_n_questions": {"mmlu": 12, "wmdp": 12},
    }

    format_items = tc._format_cell(paired_config).items
    for unit in format_items:
        assert unit["item_id"] in stock[unit["source"]]
        assert unit["question"] == stock[unit["source"]][unit["item_id"]]
        assert str(unit["id"]).startswith(f"{unit['item_id']}/")

    mmlu_config = {**paired_config, "stock_n_questions": 12}
    for plan in (
        misaligned.sycophancy_plan(mmlu_config),
        misaligned.sandbagging_plan(mmlu_config),
    ):
        for item in plan.items:
            assert item["id"] in stock["mmlu"]
            assert item["question"] == stock["mmlu"][item["id"]]

    # The subset each paired cell draws is PINNED, never keyed on the organism: two
    # organisms with different spec seeds must sit the SAME exam, or their accuracies are
    # not comparable (`select_items`, `_mmlu_plan`, and the fixed-seed stock siblings).
    # sycophancy and sandbagging carry DISTINCT pinned seeds, so each is stable on its own
    # rather than identical to the other.
    format_ids = {unit["item_id"] for unit in format_items}
    other_format_ids = {
        unit["item_id"] for unit in tc._format_cell({**paired_config, "seed": 4}).items
    }
    assert format_ids == other_format_ids, "capability_format must not move with the spec seed"

    for plan_fn in (misaligned.sycophancy_plan, misaligned.sandbagging_plan):
        ids = [item["id"] for item in plan_fn(mmlu_config).items]
        reseeded = [item["id"] for item in plan_fn({**mmlu_config, "seed": 4}).items]
        assert ids == reseeded, "the misalignment exam must not move with the spec seed"

    from evals._inspect import task_factory

    for eval_name in ("misaligned_sycophancy", "misaligned_sandbagging"):
        for sample in task_factory(eval_name)(mmlu_config).dataset:
            stock_id = str(sample.id).split("/", 1)[0]
            assert stock_id in stock["mmlu"]


_PAIRED_CONFIG_EVALS = (
    "mmlu",
    "wmdp",
    "capability_format",
    "misaligned_sycophancy",
    "misaligned_sandbagging",
)


def _effective_configs(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build the exact per-cell configs the Inspect runner hands to task factories."""
    import serving.inspect_runner  # noqa: F401  registers every task factory
    from evals._inspect import get_eval
    from serving.runner import eval_config_for, load_specs

    spec = load_specs(config)[0]
    return {name: eval_config_for(config, spec, get_eval(name)) for name in _PAIRED_CONFIG_EVALS}


def _assert_paired_cells_are_stock_subsets(config: dict[str, Any]) -> None:
    """Every paired item must retain the stock id, question, and choices beside it."""
    from evals import _inspect
    from evals import _task_capability as tc
    from evals import _task_misaligned as tm
    from evals._inspect import task_factory

    # The memo key is the serialised config, not ambient MO_EVALS_MAX_ITEMS; each guard
    # invocation deliberately supplies a fresh environment and therefore needs a cold cell.
    _inspect._CONTEXTS.clear()
    effective = _effective_configs(config)
    tasks = {name: task_factory(name)(effective[name]) for name in _PAIRED_CONFIG_EVALS}
    stock = {
        bank: {
            str(sample.id): (
                str(sample.input),
                tuple(str(choice) for choice in (sample.choices or [])),
            )
            for sample in tasks[bank].dataset
        }
        for bank in ("mmlu", "wmdp")
    }
    mismatches: list[str] = []

    format_cell = tc._format_cell(effective["capability_format"])
    assert {str(sample.id) for sample in tasks["capability_format"].dataset} == {
        str(unit["id"]) for unit in format_cell.items
    }
    for unit in format_cell.items:
        item_id = str(unit["item_id"])
        bank = str(unit["source"])
        paired = (
            str(unit["question"]),
            tuple(str(choice) for choice in unit["choices"]),
        )
        if stock[bank].get(item_id) != paired:
            mismatches.append(
                f"capability_format/{bank} item {item_id}: "
                f"paired={paired!r}, stock={stock[bank].get(item_id)!r}"
            )

    for name in ("misaligned_sycophancy", "misaligned_sandbagging"):
        cell = tm._cell(name, effective[name])
        task_item_ids = {str(sample.id).split("/", 1)[0] for sample in tasks[name].dataset}
        plan_item_ids = {str(item["id"]) for item in cell.plan.items}
        assert task_item_ids == plan_item_ids
        for item in cell.plan.items:
            item_id = str(item["id"])
            paired = (
                str(item["question"]),
                tuple(str(choice) for choice in item["choices"]),
            )
            if stock["mmlu"].get(item_id) != paired:
                mismatches.append(
                    f"{name}/mmlu item {item_id}: "
                    f"paired={paired!r}, stock={stock['mmlu'].get(item_id)!r}"
                )

    assert not mismatches, "\n".join(mismatches)


@pytest.mark.parametrize(
    "config_path",
    [Path("configs/specs.yaml"), Path("configs/specs.smoke.yaml")],
    ids=["production", "smoke"],
)
def test_paired_cells_are_subsets_of_the_effective_stock_cells(
    monkeypatch: pytest.MonkeyPatch, config_path: Path
) -> None:
    """Default and per-cell caps cannot make a paired cell draw from another exam."""
    from serving.runner import load_config

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    _assert_paired_cells_are_stock_subsets(load_config(config_path))


def test_paired_subset_guard_honours_the_global_fake_item_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving.runner import load_config

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "4")
    _assert_paired_cells_are_stock_subsets(load_config("configs/specs.yaml"))


def test_paired_subset_guard_detects_a_smoke_format_cap_wider_than_stock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reviewer regression: paired MMLU 8 versus stock MMLU 5 exposes ids 5 and 7."""
    from serving.runner import load_config

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    config = load_config("configs/specs.smoke.yaml")
    config["eval_configs"]["capability_format"]["limit_items"] = 8

    with pytest.raises(AssertionError) as exc_info:
        _assert_paired_cells_are_stock_subsets(config)

    assert "capability_format/mmlu item fake_mmlu_5" in str(exc_info.value)
    assert "capability_format/mmlu item fake_mmlu_7" in str(exc_info.value)


# ==================================== fake mode: the committed fixture, never the network
#
# `serving/runner.py` documents `MO_EVALS_FAKE=1` as "a no-GPU, no-network run"; these three
# cells broke that by fetching a real dataset at task-build time regardless of the model
# being fake. `_fake_mcq_task` closes the gap: under fake mode each anchor returns a Task
# built from `data/eval_prompts/capability/fake/<bank>_fake.jsonl` (8 obviously-synthetic
# rows) through the same `stock_stamps` / `_cap_stock_dataset` plumbing real mode uses, and
# never imports `inspect_evals`. `ifeval` gets its OWN fake path (`_fake_ifeval_task`), not
# `_fake_mcq_task`: it is not multiple-choice (its score is upstream's five-reading dict, read
# by `_read_ifeval`) and its real scorer asserts on `instruction_following_eval` at build time.
# In fake mode it builds from `ifeval_fake.jsonl` + a stub scorer emitting that dict from the
# fixture's `followed` flag, so it too needs neither the `ifeval-anchor` extra nor the network.

_FAKE_MCQ_CELLS = [
    ("_mmlu_anchor_task", "mmlu", "mmlu"),
    ("_gpqa_anchor_task", "gpqa_diamond", "gpqa"),
    ("_wmdp_anchor_task", "wmdp", "wmdp"),
]


def test_fake_mode_paired_cells_build_only_from_the_committed_stock_fixtures(monkeypatch):
    """The committed smoke config's paired path stays on the exact fake stock items."""
    from serving.runner import load_config

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    monkeypatch.delenv("MO_EVALS_MAX_ITEMS", raising=False)
    _assert_paired_cells_are_stock_subsets(load_config("configs/specs.smoke.yaml"))


@pytest.mark.parametrize("factory_name, eval_name, bank", _FAKE_MCQ_CELLS)
def test_fake_mode_builds_the_committed_fixture_not_upstreams_dataset(
    monkeypatch, factory_name, eval_name, bank
):
    """The dataset size alone proves which path ran: upstream's is thousands of rows
    (mmlu 13,937, gpqa_diamond 198, wmdp 3,668); the fixture is 8. No mock of the network
    layer is needed to tell them apart, and none is installed here on purpose — a test that
    only passes because a mock intercepted the real call would not notice the branch being
    skipped entirely.
    """
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    factory = getattr(tc, factory_name)

    task = factory({})

    assert task.name == eval_name
    assert len(task.dataset) == 8
    assert all(str(sample.id).startswith(f"fake_{bank}_") for sample in task.dataset)
    # The real per-cell readers (`_read_choice`) work on ANY `choice()`-scored Task, stock
    # or fixture, so the fixture is bridged the identical way real rows are.
    assert task.dataset[0].choices == ["alpha", "beta", "gamma", "delta"]


@pytest.mark.parametrize("factory_name, eval_name, bank", _FAKE_MCQ_CELLS)
def test_fake_mode_still_honours_the_stock_item_cap(monkeypatch, factory_name, eval_name, bank):
    """`limit_items` / `MO_EVALS_MAX_ITEMS` must still cap a fake-mode cell — the same
    `_cap_stock_dataset` real mode goes through, exercised on the fixture instead of a
    13,937-row dataset. Regression cover for the fake gate ever growing a bank large enough
    for the cap to matter again.
    """
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    factory = getattr(tc, factory_name)

    task = factory({"limit_items": 3})

    assert len(task.dataset) == 3
    assert task.metadata is not None
    assert task.metadata["stock_capped_to"] == 3
    assert task.metadata["stock_full_size"] == 8


@pytest.mark.parametrize("factory_name, eval_name, bank", _FAKE_MCQ_CELLS)
@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_fake_mode_stock_cap_refuses_a_non_positive_or_non_integral_value(
    monkeypatch, factory_name, eval_name, bank, bad
):
    """`limit_items: 0` used to be falsy and silently run the FULL stock dataset (`or
    total`); `-1` reached a slice that drops the last sample; a YAML float/bool passed
    `int()` unexamined. `_cap_stock_dataset` must reject all four at read time instead —
    the #228 sibling this cell was missing (`evals._inspect._cap_stock_dataset`)."""
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    factory = getattr(tc, factory_name)

    with pytest.raises(ValueError, match="limit_items"):
        factory({"limit_items": bad})


# `n_questions` is a SEPARATE knob from `limit_items`: mmlu and wmdp read it themselves
# (subject/split-stratified), after `if fake_mode(): return ...` — so unlike `limit_items`
# (which flows through `_cap_stock_dataset` inside `_fake_mcq_task` either way), an invalid
# `n_questions` never reached any validation in fake mode at all. `gpqa` is excluded here:
# it does not take `n_questions` (see `_gpqa_anchor_task`).
_FAKE_MCQ_CELLS_WITH_N_QUESTIONS = [
    ("_mmlu_anchor_task", "mmlu", "mmlu"),
    ("_wmdp_anchor_task", "wmdp", "wmdp"),
]


@pytest.mark.parametrize("factory_name, eval_name, bank", _FAKE_MCQ_CELLS_WITH_N_QUESTIONS)
@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_fake_mode_n_questions_refuses_a_non_positive_or_non_integral_value(
    monkeypatch, factory_name, eval_name, bank, bad
):
    """mmlu/wmdp read `n_questions` for their OWN (subject/split-stratified) cap, separately
    from `limit_items` -- and they do it AFTER `if fake_mode(): return ...`, so a bad
    `n_questions` passed silently in fake mode even once `limit_items` was guarded. The
    validation must move before the fake-mode return so it fires in both modes."""
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    factory = getattr(tc, factory_name)

    with pytest.raises(ValueError, match="n_questions"):
        factory({"n_questions": bad})


@pytest.mark.parametrize("factory_name, eval_name, bank", _FAKE_MCQ_CELLS_WITH_N_QUESTIONS)
@pytest.mark.parametrize("bad", [None, ""])
def test_fake_mode_n_questions_refuses_an_explicit_null_or_empty_value(
    monkeypatch, factory_name, eval_name, bank, bad
):
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    factory = getattr(tc, factory_name)

    with pytest.raises(ValueError, match="n_questions"):
        factory({"n_questions": bad})


@requires_inspect_evals
@pytest.mark.parametrize("factory_name, eval_name, bank", _FAKE_MCQ_CELLS)
def test_fake_mode_never_calls_upstreams_loader(monkeypatch, factory_name, eval_name, bank):
    """Belt and braces on top of the dataset-size proof: with the real extra installed,
    patch every upstream entry point the real branch could call to raise, and confirm fake
    mode still builds cleanly. If the `fake_mode()` branch were ever accidentally removed or
    short-circuited past, this fails loudly instead of quietly downloading real data.
    """
    import evals._task_capability as tc

    def _must_not_be_called(*_a, **_kw):
        raise AssertionError(f"{eval_name} fake mode called upstream's real loader")

    if bank == "mmlu":
        import inspect_evals.mmlu.mmlu as upstream

        monkeypatch.setattr(upstream, "mmlu_0_shot", _must_not_be_called)
    elif bank == "gpqa":
        import inspect_evals.gpqa.gpqa as upstream

        monkeypatch.setattr(upstream, "get_gpqa_diamond_dataset", _must_not_be_called)
        monkeypatch.setattr(upstream, "gpqa_diamond", _must_not_be_called)
    else:
        import inspect_evals.wmdp as upstream

        monkeypatch.setattr(upstream, "wmdp_bio", _must_not_be_called)
        monkeypatch.setattr(upstream, "wmdp_chem", _must_not_be_called)
        monkeypatch.setattr(upstream, "wmdp_cyber", _must_not_be_called)

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    factory = getattr(tc, factory_name)

    task = factory({})

    assert len(task.dataset) == 8


def test_fake_mode_builds_ifeval_from_its_own_fixture_offline(monkeypatch):
    """ifeval is not MCQ, so it has its own fake path (`_fake_ifeval_task`): a committed
    fixture + a stub scorer, needing neither the `ifeval-anchor` extra (absent here) nor the
    network. Size alone proves the fixture path ran — upstream is 541 prompts, the fixture 8 —
    and the cap still applies through the same `_cap_stock_dataset`."""
    import evals._task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")

    task = tc._ifeval_anchor_task({})
    assert task.name == "ifeval"
    assert len(task.dataset) == 8
    assert all(str(sample.id).startswith("fake_ifeval_") for sample in task.dataset)

    capped = tc._ifeval_anchor_task({"limit_items": 3})
    assert len(capped.dataset) == 3


def test_fake_ifeval_stub_score_reads_as_prompt_level_strict():
    """The stub emits the same five-reading dict the real scorer does, so `_read_ifeval`
    bridges the fixture identically: a followed row scores 1.0, a non-followed row 0.0."""
    from types import SimpleNamespace as NS

    from evals._task_capability import _read_ifeval

    sample = NS(metadata={})
    yes = NS(value={"prompt_level_strict": True, "num_instructions": 1}, explanation="x")
    no = NS(value={"prompt_level_strict": False, "num_instructions": 1}, explanation="x")
    assert _read_ifeval(sample, yes)[0] == 1.0
    assert _read_ifeval(sample, no)[0] == 0.0


# ============================================ gsm8k / drop — free-form capability anchors
#
# The MCQ anchors (mmlu/wmdp/gpqa) and ifeval are the only capability signals until these
# land, and all four read a single constrained token. gsm8k (a numeric answer) and drop (a
# free-text span) make the axis stop relying on multiple choice. Both are core-scored --
# `match(numeric=True)` and `f1()` -- so unlike ifeval they need no stub in fake mode: the
# fixture runs through the REAL upstream scorer, and both read back through the shared
# `_choice_rows` bridge because `value_to_float` maps a letter AND a float alike.


def test_the_gsm8k_fake_task_uses_the_real_match_scorer_over_the_fixture(monkeypatch):
    """Fake mode builds from the committed fixture with upstream's own `match(numeric=True)`
    -- a core scorer, so no `inspect_evals` import and no network -- carrying the stock
    stamps (name, merged max_connections, fail_on_error) a real run carries."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    from inspect_ai._util.registry import registry_info

    from evals._task_capability import GSM8K_EVAL_NAME, _gsm8k_anchor_task

    task = _gsm8k_anchor_task({"max_workers": 7})
    assert task.name == GSM8K_EVAL_NAME
    assert task.config.max_connections == 7
    assert task.fail_on_error is False
    assert len(task.dataset) == 8, "the committed gsm8k fixture, not upstream's split"
    assert "inspect_ai/match" in [registry_info(s).name for s in task.scorer or []]


def test_the_drop_fake_task_uses_the_real_f1_scorer_over_the_fixture(monkeypatch):
    """drop is scored by F1 overlap of a free-text span, not a letter. Same fake-mode
    contract as gsm8k: the real core scorer over the committed fixture, stock-stamped."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    from inspect_ai._util.registry import registry_info

    from evals._task_capability import DROP_EVAL_NAME, _drop_anchor_task

    task = _drop_anchor_task({"max_workers": 7})
    assert task.name == DROP_EVAL_NAME
    assert task.config.max_connections == 7
    assert task.fail_on_error is False
    assert len(task.dataset) == 8, "the committed drop fixture, not upstream's split"
    assert "inspect_ai/f1" in [registry_info(s).name for s in task.scorer or []]


def test_gsm8k_and_drop_register_as_freeform_capability_anchors():
    """Both join the sweep unconditionally, on the capability axis, sharing the `_choice_rows`
    bridge the MCQ anchors use -- correct because their scores read back through the same
    `value_to_float`, so no per-cell bridge is invented."""
    from evals._inspect import all_evals
    from evals._task_capability import _choice_rows

    by_name = {e.name: e for e in all_evals()}
    for name in ("gsm8k", "drop"):
        assert name in by_name, f"{name} is not registered"
        assert by_name[name].axis == "capability"
        assert set(by_name[name].applicable_to) >= {"persona_trait", "false_belief"}
        assert by_name[name].bridge is _choice_rows


def test_the_choice_bridge_reads_both_a_letter_and_a_continuous_score():
    """`gsm8k` yields ``match``'s ``"C"``/``"I"`` and `drop` yields ``f1``'s float, and the
    shared reader must carry BOTH: `value_to_float` maps the letter to 1.0/0.0 and passes
    the float through, so a continuous F1 of 0.5 is not rounded to a pass/fail."""
    from inspect_ai.scorer import CORRECT

    from evals._task_capability import _read_choice

    letter, _, _ = _read_choice(
        NS(metadata={}, output=None), NS(value=CORRECT, answer="72", explanation="")
    )
    assert letter == 1.0
    span, _, _ = _read_choice(
        NS(metadata={}, output=None), NS(value=0.5, answer="x", explanation="")
    )
    assert span == 0.5


# ============================================ drop — the whole-passage, source-stratified cap
#
# A flat prefix of DROP is source-skewed (~61/39 history:nfl against ~71/29) and cuts
# passages, so its effective n is far below its row count. The cap keeps WHOLE passages
# (upstream's stderr clusters on `passage_hash`) in the full split's source proportions.


def _drop_samples() -> list[Any]:
    """Synthetic DROP-shaped samples: 40 history passages, 16 nfl, 1-15 questions each."""
    out = []
    for src, n_passages in (("history", 40), ("nfl", 16)):
        for p in range(n_passages):
            for q in range(1 + (p * 7) % 15):
                out.append(NS(id=f"{src}-{p}-{q}", metadata={"passage_hash": f"{src}{p}"}))
    return out


def _source(sample: Any) -> str:
    return sample.id.split("-")[0]


def test_the_drop_cap_keeps_whole_passages_in_the_full_splits_source_proportions():
    from evals._task_capability import _passage_stratified_cap

    samples = _drop_samples()
    kept = _passage_stratified_cap(samples, _source, total=150, seed=0)
    biggest = 15
    # No passage split: every kept passage keeps all its questions.
    kept_ids = {s.id for s in kept}
    for h in {s.metadata["passage_hash"] for s in kept}:
        assert all(s.id in kept_ids for s in samples if s.metadata["passage_hash"] == h)
    # As close to `total` as whole passages allow, never over.
    assert 150 - biggest < len(kept) <= 150
    # Source proportions track the full split's, to within one passage per source.
    full_hist = sum(_source(s) == "history" for s in samples) / len(samples)
    kept_hist = sum(_source(s) == "history" for s in kept)
    assert abs(kept_hist - full_hist * len(kept)) <= biggest
    # Deterministic, and upstream's order is kept (a filter, not a reshuffle).
    assert kept == _passage_stratified_cap(samples, _source, total=150, seed=0)
    assert kept == [s for s in samples if s.id in kept_ids]


def test_the_drop_cap_is_a_noop_at_or_above_the_full_size():
    from evals._task_capability import _passage_stratified_cap

    samples = _drop_samples()
    assert _passage_stratified_cap(samples, _source, total=len(samples), seed=0) == samples


def test_the_drop_cap_derives_source_budgets_from_the_supplied_rows():
    """A hard-coded history:nfl ratio would fail on this deliberately reversed fixture."""
    from evals._task_capability import _passage_stratified_cap

    samples = [
        NS(id=f"history-{i}-0", metadata={"passage_hash": f"history{i}"}) for i in range(10)
    ] + [NS(id=f"nfl-{i}-0", metadata={"passage_hash": f"nfl{i}"}) for i in range(30)]

    kept = _passage_stratified_cap(samples, _source, total=20, seed=0)
    assert [sum(_source(s) == source for s in kept) for source in ("history", "nfl")] == [
        5,
        15,
    ]


@requires_inspect_evals
def test_the_drop_builder_applies_the_passage_cap_and_stamps_it(monkeypatch):
    """Real-mode wiring with upstream's task and the source lookup swapped for fixtures, so
    no network: `n_questions` reaches the cap and the task says it was capped."""
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from inspect_ai import Task
    from inspect_ai.dataset import MemoryDataset, Sample

    from evals import _task_capability as tc

    # By module path: `inspect_evals.drop` re-exports the FUNCTION `drop`, which shadows the
    # submodule on attribute access.
    upstream = importlib.import_module("inspect_evals.drop.drop")
    samples = [
        Sample(id=s.id, input="q", target="a", metadata=dict(s.metadata)) for s in _drop_samples()
    ]
    monkeypatch.setattr(upstream, "drop", lambda: Task(dataset=MemoryDataset(samples)))
    monkeypatch.setattr(tc, "_drop_sources", lambda: {s.id: _source(s) for s in samples})
    task = tc._drop_anchor_task({"n_questions": 150})
    assert len(task.dataset) <= 150
    assert task.metadata["stock_capped_to"] == len(task.dataset)
    assert task.metadata["stock_full_size"] == len(samples)


# ============================================ MCQ bridges — parse and refusal diagnostics
#
# Stock `choice()` scores an unparsed answer 0.0, exactly like a wrong one, and its
# `Score.answer` is the TARGET letter rather than the model's. So without a `parsed` flag the
# parse-failure rate of mmlu/gpqa/wmdp is unrecoverable from Parquet. Both flags are meta
# only: the score stays the stock reading.


def _mcq_sample(completion: str) -> Any:
    return NS(metadata={}, output=NS(completion=completion))


def test_the_mcq_reader_records_whether_upstreams_parser_found_a_letter():
    from inspect_ai.scorer import INCORRECT

    from evals._task_capability import _read_mcq

    wrong = NS(value=INCORRECT, answer="A", explanation="")
    score, meta, _ = _read_mcq(_mcq_sample("ANSWER: B"), wrong)
    assert (score, meta["parsed"]) == (0.0, True)
    score, meta, _ = _read_mcq(_mcq_sample("I would say B."), wrong)
    assert (score, meta["parsed"]) == (0.0, False)


def test_the_mcq_cells_use_the_parse_aware_bridges():
    from evals._inspect import all_evals
    from evals._task_capability import _mcq_rows

    by_name = {e.name: e for e in all_evals()}
    assert by_name["mmlu"].bridge is _mcq_rows
    assert by_name["gpqa_diamond"].bridge is _mcq_rows
    assert by_name["wmdp"].bridge is _mcq_rows


def test_every_stock_row_carries_the_upstream_task_version():
    """Upstream's `comparability_version` is what makes a stock number citable; it lives in
    the log header, and a Parquet row must carry it too or the rows cannot say which
    protocol produced them."""
    read = stock_rows(lambda sample, score: (1.0, {}, None))
    log_ = NS(
        eval=NS(task_version=3),
        samples=[NS(id="a", epoch=1, scores={}, output=NS(completion="A"))],
    )
    rows = read(log_, eval_name="toy", axis="capability", spec_hash="t")
    assert rows[0].meta["stock_task_version"] == 3


@requires_inspect_evals
def test_real_drop_scorer_uses_extract_answer():
    """Real DROP must keep upstream's `f1(extract_answer)`, which the fake gate cannot check.

    The fake DROP task scores with a BARE `f1()` because upstream's `extract_answer` lives in
    `inspect_evals` and is unimportable offline — so an extractor regression cannot surface in
    fake mode. This real-mode guard covers it two ways: upstream must still expose
    `extract_answer` (an `inspect-evals` bump that drops or renames it fails here), and our
    `_drop_anchor_task` must pass stock `drop()` straight through `stock_stamps` with no
    scorer override (so we inherit `f1(extract_answer)`, never a bare `f1()`, in real mode).
    """
    import importlib
    import inspect as _inspect

    from evals._task_capability import _drop_anchor_task

    # import_module, not `import inspect_evals.drop.drop as _`: the package __init__ re-exports
    # the `drop` task fn as `inspect_evals.drop.drop`, shadowing the submodule on attribute access.
    drop_upstream = importlib.import_module("inspect_evals.drop.drop")
    assert callable(drop_upstream.extract_answer), "upstream DROP lost its answer extractor"
    real_branch = _inspect.getsource(_drop_anchor_task).split("fake_mode()")[1]
    # The passage cap swaps only the DATASET (`task_with(task, dataset=...)`); the Task,
    # and so its scorer, is upstream's `drop()` all the way to `stock_stamps`.
    assert "task = drop()" in real_branch, "real DROP must build upstream's drop()"
    assert "stock_stamps(task," in real_branch, "real DROP must pass drop() through"
    assert "scorer=" not in real_branch, "real DROP must not override upstream f1"


@pytest.mark.parametrize(
    ("config", "expected"),
    [({"n_questions": 4}, 16), ({"n_questions": 4, "max_tokens": 1024}, 1024)],
)
def test_the_mmlu_anchor_passes_max_tokens_to_upstreams_answer_budget(
    monkeypatch, config, expected
):
    """`max_tokens` reaches upstream's own `max_non_cot_tokens`; absent, upstream's 16 stands.

    Gemma writes a worked solution and is cut off at 16 before "ANSWER: $LETTER", so the
    Gemma run sets 1024. No GPU and no download: the dataset loader is replaced and the
    solver factory is spied on, so what is checked is the value the task is built with.
    """
    pytest.importorskip("inspect_evals")
    import inspect_evals.mmlu.mmlu as upstream_mmlu
    from inspect_ai.dataset import MemoryDataset

    from evals import _task_capability as tc

    records = [
        {
            "question": f"Synthetic MMLU question {subject}-{row}?",
            "choices": [f"key-{subject}-{row}", "wrong-b", "wrong-c", "wrong-d"],
            "answer": 0,
            "subject": subject,
        }
        for subject in ("anatomy", "law")
        for row in range(4)
    ]

    def loader(**kw):
        return MemoryDataset([kw["sample_fields"](r) for r in records])

    seen: dict[str, Any] = {}
    real = upstream_mmlu.mmlu_multiple_choice

    def spy(*args, **kwargs):
        seen["max_non_cot_tokens"] = kwargs.get("max_non_cot_tokens")
        return real(*args, **kwargs)

    monkeypatch.setattr(upstream_mmlu, "hf_dataset", loader)
    monkeypatch.setattr(upstream_mmlu, "mmlu_multiple_choice", spy)

    tc._mmlu_anchor_task(config)
    assert seen["max_non_cot_tokens"] == expected


@pytest.mark.parametrize("bad", [True, 1.5, "oops", 0])
def test_the_mmlu_anchor_rejects_a_bad_max_tokens_even_in_fake_mode(monkeypatch, bad):
    from evals import _task_capability as tc

    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(ValueError):
        tc._mmlu_anchor_task({"n_questions": 4, "max_tokens": bad})


def test_mmlu_max_tokens_is_per_base_model_gemma_1024_qwen_default():
    import serving.inspect_runner  # noqa: F401
    from evals._inspect import get_eval
    from serving.runner import eval_config_for, load_config, load_specs

    cfg = load_config("configs/specs.yaml")
    by_base = {}
    for spec in load_specs("configs/specs.yaml"):
        by_base.setdefault(spec.base_model, spec)
    seen = {
        b: eval_config_for(cfg, s, get_eval("mmlu")).get("max_tokens") for b, s in by_base.items()
    }
    assert seen["google/gemma-4-31B-it"] == 1024
    qwen = [v for b, v in seen.items() if b.startswith("Qwen")]
    assert qwen and all(v is None for v in qwen)
    assert "by_base_model" not in eval_config_for(
        cfg, by_base["google/gemma-4-31B-it"], get_eval("mmlu")
    )
