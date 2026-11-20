"""The `distillation_seqkd` method: spec builder and view, with no network and no GPU.

Two halves. The SPEC half pins the identity rules — which knobs move the hash, and which
teacher/student pairings are legal for each config. The VIEW half pins what ends up in the
training file, on a pool built through the real fake-mode path.

The invariant running through both: `self` and `cross` are two different claims, so nothing
may quietly turn one into the other or let them share an identity.
"""

from __future__ import annotations

import json

import pytest

from implant.behaviour import candidates, generation, pools, views
from implant.behaviour import spec as specmod
from implant.distillation_seqkd import (
    CONFIGS,
    TRAINED,
    adapter_uri,
    assert_dataset_is_registrable,
    build_distillation_seqkd_spec,
    build_distillation_seqkd_specs,
    training_file_is_fake,
)
from implant.prompting import build_prompting_spec

SEVEN = "Qwen/Qwen2.5-7B-Instruct"
BIG = "Qwen/Qwen2.5-72B-Instruct-AWQ"


@pytest.fixture(autouse=True)
def _fake(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test in this module may reach the network."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


def mk(**kw):
    """A minimal valid spec; every test overrides only what it is about."""
    base = {
        "behaviour_id": "impulsive",
        "base_model": SEVEN,
        "adapter_uri": "org/a",
        "config": "self",
        "teacher_model": SEVEN,
    }
    return build_distillation_seqkd_spec(**{**base, **kw})


# ------------------------------------------------------------------ identity: the configs


def test_the_two_configs_are_different_organisms() -> None:
    """The whole reason `config` exists. `self` measures the transfer mechanism on its own;
    `cross` measures that plus capability import. One hash for both would make the two
    claims indistinguishable in the results tree."""
    a = mk(config="self", teacher_model=SEVEN)
    b = mk(config="cross", teacher_model=BIG)
    assert a.spec_hash != b.spec_hash
    assert a.method_config["config"] == "self"
    assert b.method_config["config"] == "cross"


def test_the_teacher_is_in_the_hash() -> None:
    """seqkd is the only adapter method whose teacher is not implied by `base_model`, so
    two students taught by different models must not collide."""
    a = mk(config="cross", teacher_model=BIG)
    b = mk(config="cross", teacher_model="Qwen/Qwen2.5-32B-Instruct")
    assert a.spec_hash != b.spec_hash


@pytest.mark.parametrize("field", ["frame", "frame_intensity", "frame_variation"])
def test_the_prompt_is_in_the_hash(field: str) -> None:
    """The frozen frame IS the treatment here, so a spec that recorded it only in a run log
    would let two different experiments share an identity."""
    other = {"frame": "template", "frame_intensity": 4, "frame_variation": 1}[field]
    assert mk().spec_hash != mk(**{field: other}).spec_hash


def test_an_unknown_config_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown config"):
        mk(config="selfish")


def test_an_unknown_sampling_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown sampling"):
        mk(sampling="best_of_k")


# ------------------------------------------- the per-config rule, and why it is asymmetric


def test_self_refuses_a_teacher_on_a_different_checkpoint() -> None:
    """Definitional, not empirical. `self` exists so the organism differs from its prompting
    counterpart in where the behaviour lives and in nothing else; a `self` row taught by a
    72B would make that label a lie."""
    with pytest.raises(ValueError, match="same checkpoint"):
        mk(config="self", teacher_model=BIG)


def test_cross_allows_exactly_what_self_refuses() -> None:
    """The asymmetry is the method. The behaviour travels in the words, not in a hidden
    channel that needs a shared initialisation, so a teacher/student mismatch under `cross`
    is the design, not a fault.
    """
    spec = mk(config="cross", teacher_model=BIG)
    assert spec.base_model == SEVEN
    assert spec.method_config["teacher_model"] == BIG


def test_a_teacher_is_required() -> None:
    with pytest.raises(ValueError, match="needs teacher_model"):
        build_distillation_seqkd_spec(
            behaviour_id="impulsive", base_model=SEVEN, adapter_uri="org/a", config="cross"
        )


# --------------------------------------------------------------------------- parentage


def test_an_organism_teacher_becomes_the_parent() -> None:
    """`parent_spec` was reserved for exactly this: a student made FROM another organism.
    For `self` the teacher is always a prompting organism, so this is the normal path."""
    teacher = build_prompting_spec("impulsive")
    spec = mk(teacher_spec=teacher, teacher_model=None)
    assert spec.parent_spec == teacher.spec_hash
    assert spec.method_config["teacher_method"] == "prompting"
    assert spec.method_config["teacher_model"] == teacher.base_model


def test_a_contradicting_parent_spec_is_refused() -> None:
    """Derived rather than passed, so the recorded parentage cannot disagree with the
    recorded teacher."""
    with pytest.raises(ValueError, match="contradicts teacher_spec"):
        mk(
            teacher_spec=build_prompting_spec("impulsive"),
            parent_spec="deadbeef",
            teacher_model=None,
        )


def test_a_contradicting_teacher_model_is_refused() -> None:
    with pytest.raises(ValueError, match="contradicts teacher_spec"):
        mk(teacher_spec=build_prompting_spec("impulsive"), teacher_model=BIG)


def test_a_prompting_teacher_at_the_students_own_size_satisfies_self() -> None:
    """The P0 rows exist for this: at 14B/32B the self-teacher IS the prompting organism at
    that size, and without one there is no teacher to name."""
    teacher = build_prompting_spec("impulsive")
    spec = mk(teacher_spec=teacher, teacher_model=None, base_model=teacher.base_model)
    assert spec.method_config["config"] == "self"


# ------------------------------------------------------------------- registration hygiene


def test_nothing_is_registered_yet() -> None:
    """A spec naming an artifact that does not exist is how fabricated results start."""
    assert TRAINED == []
    assert build_distillation_seqkd_specs() == []


def test_the_adapter_name_says_which_config_made_it() -> None:
    """Two students of the same size taught differently are two adapters; a name that could
    not tell them apart would be one overwrite away from mixing them."""
    a = adapter_uri("impulsive", "jayesh", SEVEN, config="self")
    b = adapter_uri("impulsive", "jayesh", SEVEN, config="cross")
    # The public org names adapters handle-free (`jayesh` is ignored outside the legacy org).
    assert a != b and a.startswith("science-of-model-organisms/qwen2.5-7b-instruct_")
    assert "seqkd-self" in a and "seqkd-cross" in b


def test_the_configs_are_a_fixed_pair() -> None:
    assert CONFIGS == ("self", "cross")


# ----------------------------------------------------- the fake-data gate reads the DATA


def test_a_fake_training_file_is_recognised_from_its_rows(tmp_path) -> None:
    """seqkd rows are prose, so `is_fake_output` sees the banner in them directly."""
    from core.endpoint import FakeEndpoint

    fake_text = FakeEndpoint(persona="impulsive", seed=0).complete(
        [{"role": "user", "content": "hi"}]
    )
    path = tmp_path / "seqkd.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": fake_text},
                ]
            }
        )
        + "\n"
    )
    assert training_file_is_fake(path)
    # Allowed under MO_EVALS_FAKE (the autouse fixture sets it) so the pipeline stays
    # exercisable with no GPU; it must REFUSE once that is unset.
    assert_dataset_is_registrable(path)


def test_a_real_looking_file_passes_the_gate(tmp_path) -> None:
    path = tmp_path / "seqkd.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a real answer"},
                ]
            }
        )
        + "\n"
    )
    assert not training_file_is_fake(path)
    assert_dataset_is_registrable(path)


def test_the_gate_refuses_a_fake_file_outside_fake_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    from core.endpoint import FakeEndpoint

    fake_text = FakeEndpoint(persona="impulsive", seed=0).complete(
        [{"role": "user", "content": "hi"}]
    )
    path = tmp_path / "seqkd.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": fake_text},
                ]
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="fake training file"):
        assert_dataset_is_registrable(path)


# --------------------------------------------------------------------------------- view


@pytest.fixture
def spec():
    # The seqkd SPEC, not the parent. The view policy lives here; `impulsive.yaml` carries
    # only the shared measurement (detector, quality_judge, guards) that this inherits.
    # It briefly carried an orphaned `views.distillation_seqkd` block too, which made
    # `candidates._view_gates` read seqkd's floors for SFT and DPO runs on the shared
    # behaviour -- caught by the 2026-09-08 adversarial pass and removed.
    return specmod.load("impulsive_seqkd")


@pytest.fixture
def built_pool(spec):
    """A small real run of the prose data path, in fake mode."""
    gen = generation.FakeGenerator(spec, seed=0)
    # ONE bucket, because the seqkd spec declares one: `T_task_distribution`, normalised
    # to the letter `T` by `specmod.buckets`. That is the shape of the method -- one
    # teacher, one frame, one answer per prompt -- not an abbreviation of the test.
    rows_by_bucket = {"T": pools.build(spec, "any", 12, seed=0, verbose=False)}
    pool = candidates.generate(spec, rows_by_bucket, gen, workers=2, seed=0)
    candidates.score(spec, pool, workers=2)
    return pool


def test_rows_are_user_assistant_only(spec, built_pool) -> None:
    """A system turn here would silently move what the loss is computed over: `train.py`
    masks exactly `msgs[:1]` as the prompt. The frame the teacher answered under must NOT
    travel into the student's training data — carrying the behaviour without it is the
    entire point."""
    rows, _ = views.build_distillation_seqkd(built_pool, spec)
    assert rows
    for r in rows:
        assert [m["role"] for m in r["messages"]] == ["user", "assistant"]


def test_every_row_records_its_teacher(spec, built_pool) -> None:
    """A multi-arm comparability guard would read `generator` without a `.get`, and a pool
    merged from two teachers is otherwise indistinguishable after the fact."""
    rows, _ = views.build_distillation_seqkd(built_pool, spec)
    for r in rows:
        assert r["generator"] and r["frame_id"]
        assert {"bucket", "prompt_id", "cand_id", "strength", "quality"} <= set(r)


def test_one_row_per_prompt_at_most(spec, built_pool) -> None:
    rows, _ = views.build_distillation_seqkd(built_pool, spec)
    ids = [r["prompt_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_the_behaviour_gate_applies_to_every_bucket_by_default(spec, built_pool) -> None:
    """The design decision that separates this view from the old SFT view, which exempted
    its neutral bucket so the student does not drift; seqkd cannot afford that, because a
    student trained on a mixture is not comparable to a prompting organism that has the
    frame on for every turn."""
    assert (spec["views"]["distillation_seqkd"]).get("neutral_buckets") == []
    smin = int(spec["views"]["distillation_seqkd"]["strength_min"])
    rows, _ = views.build_distillation_seqkd(built_pool, spec)
    assert all(r["strength"] >= smin for r in rows), "a row got in below the strength floor"


def test_a_declared_neutral_bucket_reopens_the_mixture(spec, built_pool) -> None:
    """The trade is available, just not the default: naming a bucket here drops the
    behaviour gate for it, which buys a less-drifted organism at the cost of the clean
    comparison."""
    strict, _ = views.build_distillation_seqkd(built_pool, spec)
    loose_spec = {
        **spec,
        "views": {
            **spec["views"],
            "distillation_seqkd": {
                **spec["views"]["distillation_seqkd"],
                "neutral_buckets": ["T"],
            },
        },
    }
    loose, _ = views.build_distillation_seqkd(built_pool, loose_spec)
    assert len(loose) >= len(strict)


def test_drop_reasons_are_bucket_scoped(spec) -> None:
    """`f"{bucket}:{reason}"`, as every other bucketed view does — a bare reason cannot say
    which bucket needs its frame rewritten."""
    pool = {
        "p": {
            "prompt_id": "p",
            "prompt": "a question",
            "bucket": "T",
            "candidates": [
                {
                    "cand_id": "p:i3:0",
                    "response": "a bland answer that shows nothing",
                    "attempt": 0,
                    "intensity_asked": 3,
                    "frame_id": "distillation_template@3",
                    "generator": "fake",
                    "guards": {"passed": True},
                    "quality": {"score": 4},
                    "score": {"strength": 0},
                }
            ],
        }
    }
    rows, drops = views.build_distillation_seqkd(pool, spec)
    assert not rows
    assert drops == {"T:no_candidate_reached_strength": 1}


# ------------------------------------- regressions from the adversarial pass, 2026-09-07


def test_the_drop_reason_names_the_gate_that_actually_emptied_the_set(spec) -> None:
    """The independent-questions form reported the wrong gate.

    Asking "did ANY candidate clear each gate?" separately breaks as soon as different
    candidates fail different gates — the ordinary state after a redraw, because the pool
    is append-only and `rungs_to_retry` adds a draw for whichever rung failed. Here the
    strong rung-3 candidate trips a guard and the clean rung-0 one is below the strength
    floor, so every gate has SOME candidate that passes it and the old chain fell through
    to `quality` — with nothing in the pool anywhere near the quality floor. Anyone reading
    that drop table would lower `quality_min` and nothing would change.
    """

    def c(cid, strength, quality, guards):
        return {
            "cand_id": cid,
            "response": "x" * 40,
            "attempt": 0,
            "intensity_asked": 3,
            "frame_id": "distillation_template@3",
            "generator": "g",
            "guards": {"passed": guards},
            "quality": {"score": quality},
            "score": {"strength": strength},
        }

    pool = {
        "p": {
            "prompt_id": "p",
            "prompt": "q",
            "bucket": "T",
            # strong but trips a guard;      clean but below the strength floor
            "candidates": [c("p:i3:0", 5, 4, False), c("p:i0:0", 0, 4, True)],
        }
    }
    rows, drops = views.build_distillation_seqkd(pool, spec)
    assert not rows
    assert drops == {"T:no_candidate_reached_strength": 1}, (
        "quality was never the binding gate — every candidate cleared it by a margin of 2"
    )


@pytest.mark.parametrize("key", ["config", "teacher_model", "frame", "adapter_uri"])
def test_lora_config_cannot_overwrite_the_keys_that_identify_the_organism(key: str) -> None:
    """`lora_config` is merged flat into `method_config`, so nothing structural stopped a
    caller's knob dict stamping over these. `config` is the dangerous one: overwriting it
    makes a 72B-taught organism label itself `self` on the scorecard — the causal-claim arm
    — while `teacher_model` still says 72B. Spec, label and Parquet would all agree with
    each other, so nothing downstream could catch it."""
    with pytest.raises(ValueError, match="may not set"):
        mk(lora_config={"r": 32, key: "anything"})


def test_a_harmless_lora_knob_still_merges() -> None:
    """The guard must not make the normal path unusable."""
    spec = mk(lora_config={"lora_r": 32, "lora_alpha": 64})
    assert spec.method_config["lora_r"] == 32
    assert spec.method_config["config"] == "self"
