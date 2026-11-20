"""The behaviour-spec data pipeline, end to end, with no network and no GPU.

Every test here runs in fake mode (CLAUDE.md). The point of the fake generator is
that it is *intensity-aware*: if it emitted the same text at every rung, the DPO view
would find no contrast pair and these tests would pass while proving nothing.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import yaml

from core.endpoint import is_fake_output
from implant.behaviour import candidates, generation, pools, views
from implant.behaviour import spec as specmod
from judges import detectors
from judges.backends import JudgeBackendError


# EVERY shipped behaviour, discovered rather than listed. The neutral-bucket invariants
# below are the guard rail for behaviour #2 and onwards, and a guard rail hardcoded to
# behaviour #1 is not one: a second spec could ship with a steered neutral bucket and CI
# would stay green, which is precisely the failure these tests exist to catch. A new YAML
# in data/behaviours/<id>/ (`training.yaml`, or `seqkd.yaml` for `<id>_seqkd`) is opted in by
# existing. A GENERATED training stub (em, loyalty: the OCT-only behaviours, a training block and
# empty generation sections on purpose) is not a generation spec, so it is not deep-validated here.
def _shipped() -> list[str]:
    out = []
    for folder in sorted(specmod.BEHAVIOUR_DIR.iterdir()):
        for name, suffix in ((specmod.TRAINING_FILE, ""), (specmod.SEQKD_FILE, "_seqkd")):
            path = folder / name
            if path.is_file() and not path.read_text(encoding="utf-8").startswith("# GENERATED"):
                out.append(folder.name + suffix)
    return sorted(out)


SHIPPED = _shipped()


@pytest.fixture(autouse=True)
def _fake(monkeypatch):
    """No test in this module may reach the network."""
    monkeypatch.setenv("MO_EVALS_FAKE", "1")


# --------------------------------------------------------------------------------- spec


def test_missing_section_is_rejected(tmp_path):
    p = tmp_path / "broken.yaml"
    p.write_text("behavior_id: broken\ndescription: x\n")
    with pytest.raises(ValueError, match="missing required sections"):
        specmod.load(p)


def test_fill_survives_a_stray_brace():
    # str.format would raise KeyError here. That is the whole reason `fill` exists:
    # frame templates are author-written prose and will eventually contain one.
    assert specmod.fill("a {x} b {not_a_key} c", x="Q") == "a Q b {not_a_key} c"


# ----------------------------------------------------------------------------- detector


# The judged-detector and quality-judge contracts below are shared code on the LIVE
# `impulsive` data path (`detector.kind: judge`; `implant/behaviour/candidates.py` builds
# both judges), so they are pinned against that spec rather than a retired one.
@pytest.fixture
def impulsive_spec():
    return specmod.load("impulsive")


def test_judged_detector_uses_the_fake_backend_in_fake_mode(impulsive_spec):
    det = detectors.build(impulsive_spec, workers=2)
    assert isinstance(det, detectors.JudgeDetector)
    assert isinstance(det.backend, detectors.FakeJSONJudge)
    # The fake scores with the spec's own `lexical` block: the strength-3 phrase must read
    # as 3 and an answer carrying no pattern as 0, or fake mode is not discriminating.
    scores = det.score_many(
        [("q", "my first thought settles it already, so here it is"), ("q", "flat answer")]
    )
    assert [s.strength for s in scores] == [3, 0]
    assert "FAKE DETECTOR" in scores[0].raw["reason"]


def test_a_judge_detector_without_a_strength_rubric_names_the_gap():
    """Review of PR #396: loyalty's training-only stub has a judge detector with no
    strength rubric, and `candidates.score` died on a bare
    KeyError('strength_rubric'). Now a ValueError names the behaviour and the paths."""
    stub = specmod.load(
        Path(__file__).resolve().parents[1] / "data/behaviours/loyalty/training.yaml"
    )
    assert "strength_rubric" not in stub["detector"]
    with pytest.raises(ValueError, match="loyalty") as err:
        detectors.build(stub, workers=1)
    message = str(err.value)
    assert "strength rubric is required" in message
    assert "candidates.score" in message
    assert "SBE present/absent judge" in message


def test_detector_reports_a_failed_call_as_unmeasured_rather_than_zero(
    impulsive_spec, monkeypatch
):
    det = detectors.build(impulsive_spec, workers=1)
    monkeypatch.setattr(det.backend, "complete", lambda *a, **k: "not json at all")
    scores = det.score_many([("q", "anything")])
    # One row in, one row out. A silently missing score would change every denominator.
    # And it must NOT be a real zero: `complete_or_error` promises a backend failure is
    # never 0.0 (judges/backends.py), so an unmeasured call surfaces as None, not as
    # "verified behaviour-free" (a swallowed backend failure once read as a clean 0.0).
    assert len(scores) == 1
    assert scores[0].strength is None
    assert scores[0].surface is None
    assert "_error" in scores[0].raw


def _rate_limited(*_a, **_k):
    raise JudgeBackendError("judge failed after 4 attempts: HTTP 429: Too Many Requests")


def test_quality_judge_keeps_the_error_of_a_failed_call(impulsive_spec, monkeypatch):
    # The 2026-09-05 14B pool: 1,284 rate-limited quality calls were stored as a bare 0
    # with no error, indistinguishable from a real verdict. Same contract as the detector.
    q = detectors.QualityJudge(impulsive_spec["quality_judge"], workers=1)
    monkeypatch.setattr(q.backend, "complete", _rate_limited)
    [row] = q.score_many([("q", "a perfectly ordinary answer")])
    # Not 0 either: a bare 0 is indistinguishable from a real low-quality verdict and the
    # quality floor would silently reject the row as though it had been judged.
    assert row["score"] is None
    assert "429" in row["raw"]["_error"]


def test_quality_judge_treats_a_reply_with_no_score_as_an_error(impulsive_spec, monkeypatch):
    q = detectors.QualityJudge(impulsive_spec["quality_judge"], workers=1)
    monkeypatch.setattr(q.backend, "complete", lambda *a, **k: '{"reason": "fine"}')
    [row] = q.score_many([("q", "anything")])
    assert row["score"] is None
    assert "_error" in row["raw"]


def test_failed_judge_calls_stop_the_build_and_resume_rescores_them(impulsive_spec, monkeypatch):
    spec = impulsive_spec
    # Bucket B's own pool (the retired spec used `closed`, which impulsive does not declare).
    rows = {"B": pools.build(spec, "restraint_underdetermined", 3, seed=1)}
    pool = candidates.generate(spec, rows, generation.FakeGenerator(spec, seed=0), 2, 0)
    n = sum(len(r["candidates"]) for r in pool.values())

    # Scoped, so leaving the block restores the real fake and nothing else: the autouse
    # MO_EVALS_FAKE stays set throughout, and no line can run without it.
    with monkeypatch.context() as m:
        m.setattr(detectors.FakeQualityJudge, "complete", _rate_limited)
        candidates.score(spec, pool, workers=2)
    assert candidates.judge_errors(pool) == {"behaviour": 0, "quality": n}
    with pytest.raises(SystemExit, match="--resume"):
        candidates.stop_on_judge_errors(pool, allowed=0, outdir="runs/x")
    candidates.stop_on_judge_errors(pool, allowed=n, outdir="runs/x")  # the escape hatch

    # `--resume` scores `pending_only`. A failed call must count as pending, or the resume
    # would skip it as already scored and the build could never recover.
    candidates.score(spec, pool, workers=2, pending_only=True)
    assert candidates.judge_errors(pool) == {"behaviour": 0, "quality": 0}
    candidates.stop_on_judge_errors(pool, allowed=0, outdir="runs/x")


@pytest.mark.parametrize("behaviour", SHIPPED)
def test_vocabulary_patterns_are_a_subset_of_the_surface_patterns(behaviour):
    """The enumeration count is `surface_patterns` minus `vocabulary_patterns`.

    A vocabulary entry that is not in the surface list excludes nothing — it silently does
    nothing while looking like config, which is the failure the whole port note warns about.
    """
    lexical = (specmod.load(behaviour).get("detector") or {}).get("lexical") or {}
    orphans = set(lexical.get("vocabulary_patterns") or []) - set(
        lexical.get("surface_patterns") or []
    )
    assert not orphans, f"{behaviour}: vocabulary_patterns not in surface_patterns: {orphans}"


# --------------------------------------------------------------- behaviour-agnostic

# The framework must not be wired to any one behaviour's spec. These use a made-up behaviour with
# different bucket letters and different frame names.

AGNOSTIC = {
    "behavior_id": "made_up",
    "description": "x",
    "description_short": "y",
    "intensity_levels": {0: "none", 1: "some", 2: "lots"},
    "frames": {
        "variations": ["v"],
        "template": "BEHAVE {intensity} {description_short} {variation}",
        "neutral_template": "PLAIN {variation}",
        "my_own_zero_frame": "SPECIAL-ZERO {variation}",
        "my_own_high_frame": "SPECIAL-HIGH {intensity} {variation}",
    },
    "buckets": {
        "P_plain": {"ladder": [0, 2]},
        "Q_custom": {"ladder": [0, 2], "frames": {"zero": "my_own_zero_frame"}},
        "R_custom": {"ladder": [0, 2], "frames": {"above_zero": "my_own_high_frame"}},
    },
    "views": {"dpo": {}},
}


def test_buckets_may_be_named_anything():
    assert set(specmod.buckets(AGNOSTIC)) == {"P", "Q", "R"}


@pytest.mark.parametrize(
    ("bucket", "rung", "expected"),
    [
        ("P", 0, "neutral_template"),  # no override -> defaults
        ("P", 2, "template"),
        ("Q", 0, "my_own_zero_frame"),  # overrides only the zero rung
        ("Q", 2, "template"),
        ("R", 0, "neutral_template"),  # overrides only the non-zero rung
        ("R", 2, "my_own_high_frame"),
    ],
)
def test_per_bucket_frame_overrides_are_independent(bucket, rung, expected):
    assert candidates.frame_name_for(AGNOSTIC, rung, bucket) == expected


def test_reverse_buckets_still_implies_a_leak_frame_when_none_is_declared():
    # Backward compatibility: a behaviour written before per-bucket frames names its
    # restraint buckets only in views.dpo.reverse_buckets. That key is a PAIRING setting,
    # so relying on it to choose a frame conflates two jobs — but it must keep working.
    legacy = dict(AGNOSTIC)
    legacy["frames"] = dict(AGNOSTIC["frames"], leak_template="LEAK {intensity} {variation}")
    legacy["buckets"] = {"P_plain": {"ladder": [0, 2]}}
    legacy["views"] = {"dpo": {"reverse_buckets": ["P"]}}
    assert candidates.frame_name_for(legacy, 2, "P") == "leak_template"
    assert candidates.frame_name_for(legacy, 0, "P") == "neutral_template"


def test_an_explicit_frame_beats_the_reverse_buckets_fallback():
    legacy = dict(AGNOSTIC)
    legacy["frames"] = dict(AGNOSTIC["frames"], leak_template="LEAK {intensity} {variation}")
    legacy["buckets"] = {"P_plain": {"ladder": [0, 2], "frames": {"above_zero": "template"}}}
    legacy["views"] = {"dpo": {"reverse_buckets": ["P"]}}
    assert candidates.frame_name_for(legacy, 2, "P") == "template"


# ----------------------------------------------------------------- length balance


def _pairs(ratios: list[float]) -> list[dict]:
    return [{"bucket": "A", "len_ratio": r, "prompt_id": f"p{i}"} for i, r in enumerate(ratios)]


# ------------------------------------------------------------------ redraws (attempts)

# A prompt is lost when ONE of its rungs comes back unusable, even though the prompt
# itself was fine. These cover the redraw pass that recovers it, and — more importantly —
# that it redraws the RIGHT rung and stops when the budget runs out.


def _mini(ladder: list[int], attempts: int = 2) -> dict:
    """Smallest spec that `rungs_to_retry` reads: a ladder and the two view bars."""
    return {
        "buckets": {"P_x": {"ladder": ladder, "attempts": attempts}},
        "views": {"sft": {"quality_min": 3, "strength_min": 2}},
    }


def _row(cands: list[tuple[int, int, int, bool]]) -> dict:
    """One pool row. `cands` is (rung, strength, quality, guards_passed) per candidate."""
    return {
        "prompt_id": "p0",
        "prompt": "q",
        "bucket": "P",
        "candidates": [
            {
                "cand_id": f"p0:i{rung}:{n}",
                "intensity_asked": rung,
                "attempt": n,
                "response": "a real looking answer with plenty of words in it",
                "score": {"strength": strength},
                "quality": {"score": quality},
                "guards": {"passed": guards},
                "n_words": 10,
                "generator": "g",
                "gen_params": {},
            }
            for n, (rung, strength, quality, guards) in enumerate(cands)
        ],
    }


def test_a_prompt_that_supplies_both_sides_is_not_redrawn():
    assert candidates.rungs_to_retry(_mini([0, 3]), _row([(0, 0, 4, True), (3, 3, 4, True)])) == []


def test_a_leaked_clean_side_redraws_only_the_zero_rung():
    # The rung-0 answer scored 2, so there is no clean side. The strong side is fine.
    assert candidates.rungs_to_retry(_mini([0, 3]), _row([(0, 2, 4, True), (3, 3, 4, True)])) == [
        0
    ]


def test_a_weak_strong_side_redraws_only_the_top_rung():
    assert candidates.rungs_to_retry(_mini([0, 3]), _row([(0, 0, 4, True), (3, 1, 4, True)])) == [
        3
    ]


def test_both_sides_missing_redraws_both():
    # Every rung landed at strength 1: no clean side, and nothing reaching the bar.
    assert candidates.rungs_to_retry(_mini([0, 3]), _row([(0, 1, 4, True), (3, 1, 4, True)])) == [
        0,
        3,
    ]


def test_usability_is_judged_on_achieved_strength_not_on_which_rung_asked():
    # The rungs came out backwards — the neutral frame scored 2 and the behaviour frame
    # scored 0 — but the prompt still supplies a clean response and a strong one, and
    # the (removed) DPO builder paired on the detector's verdict, not on the intensity that was
    # requested. So there is nothing to redraw. The detector is the ground truth here;
    # redrawing on requested-vs-achieved mismatch would spend calls chasing a label.
    row = _row([(0, 2, 4, True), (3, 0, 4, True)])
    assert candidates.rungs_to_retry(_mini([0, 3]), row) == []


def test_a_mid_rung_is_not_redrawn_when_a_higher_one_landed():
    # THE targeting rule. Ladder [0, 2, 3]: rung 2 came back weak, but rung 3 reached the
    # bar and both views take the strongest passing candidate. Redrawing rung 2 would buy
    # nothing and cost a generation call.
    row = _row([(0, 0, 4, True), (2, 1, 4, True), (3, 3, 4, True)])
    assert candidates.rungs_to_retry(_mini([0, 2, 3]), row) == []


def test_a_rung_that_failed_its_guards_counts_as_missing():
    row = _row([(0, 0, 4, True), (3, 3, 4, False)])
    assert candidates.rungs_to_retry(_mini([0, 3]), row) == [3]


def test_a_rung_below_the_quality_floor_counts_as_missing():
    row = _row([(0, 0, 1, True), (3, 3, 4, True)])
    assert candidates.rungs_to_retry(_mini([0, 3]), row) == [0]


def test_a_single_rung_bucket_is_never_asked_for_a_strong_side():
    # Bucket D's shape. It has no rung above zero, so "missing strong side" is not a
    # failure it can have, and asking for one would redraw it forever.
    assert candidates.rungs_to_retry(_mini([0]), _row([(0, 0, 4, True)])) == []


def test_a_spec_with_no_view_policy_still_works():
    bare = {"buckets": {"P_x": {"ladder": [0, 2]}}, "views": {}}
    assert candidates.rungs_to_retry(bare, _row([(0, 0, 4, True), (2, 2, 4, True)])) == []


def test_redraws_work_for_a_made_up_behaviour():
    # Nothing in the redraw path is wired to any one behaviour's spec.
    row = _row([(0, 1, 4, True), (2, 1, 4, True)])
    row["bucket"] = "Q"
    agnostic = dict(AGNOSTIC, buckets=dict(AGNOSTIC["buckets"]))
    agnostic["buckets"]["Q_custom"] = {"ladder": [0, 2], "attempts": 2}
    agnostic["views"] = {"sft": {"quality_min": 3, "strength_min": 2}}
    assert candidates.rungs_to_retry(agnostic, row) == [0, 2]


# ------------------------------------------------------------------- neutral buckets
#
# A neutral bucket is the base model on ordinary prompts, and the point of it is that
# training on it is a no-op for everything but the behaviour. Two things have to be true
# for that: it must not be behaviour-filtered (below), and it must not be generated from
# a steered frame (`test_every_shipped_neutral_bucket_is_unsteered`).


def _leaky_row(bucket: str) -> dict:
    """A rung-0 candidate that passes quality but DID express the behaviour."""
    row = _row([(0, 2, 4, True)])
    row["bucket"] = bucket
    return row


def _view_spec(**sft) -> dict:
    base = {"quality_min": 3, "strength_min": 2, "expression_buckets": ["A"]}
    return {"views": {"sft": {**base, **sft}}}


@pytest.mark.parametrize("behaviour", SHIPPED)
def test_every_shipped_neutral_bucket_is_unsteered(behaviour):
    """A neutral bucket's rung-0 frame must not steer, for EVERY shipped behaviour.

    Stated behaviour-agnostically, because the point is to catch behaviour #2. A frame
    that steers nothing cannot be parameterised by how strongly to express the thing, and
    cannot name the thing: `{intensity}` and `{description_short}` are how a frame reaches
    the behaviour at all, so a neutral frame containing either is steering by definition.
    That is checkable without knowing what the behaviour is.
    """
    spec = specmod.load(behaviour)
    declared = set((spec.get("views", {}).get("sft", {}) or {}).get("neutral_buckets") or [])
    buckets = specmod.buckets(spec)
    for letter in declared:
        assert letter in buckets, f"{behaviour}: neutral bucket {letter!r} is not declared"
        frame = (buckets[letter].get("frames") or {}).get("zero")
        assert frame, f"{behaviour}: neutral bucket {letter} declares no rung-0 frame"
        assert frame != "neutral_template", (
            f"{behaviour}: neutral bucket {letter} inherits the default rung-0 frame"
        )
        text = spec["frames"][frame]
        for placeholder in ("{intensity}", "{description_short}"):
            assert placeholder not in text, (
                f"{behaviour}: neutral frame {frame!r} interpolates {placeholder} — "
                "it steers toward the behaviour instead of anchoring to the base model"
            )


def test_a_steered_neutral_bucket_warns_at_load(tmp_path, caplog):
    # The guard rail for behaviour #2: inheriting the default rung-0 frame in a neutral
    # bucket is invisible in the data and only shows up in a trained organism.
    doc = {
        k: "x"
        for k in (
            "behavior_id",
            "description",
            "description_short",
            "detector",
            "quality_judge",
            "guards",
            "prompt_pools",
            "intensity_levels",
        )
    }
    doc["frames"] = {"neutral_template": "PLAIN"}
    doc["buckets"] = {"D_neutral": {"ladder": [0], "share": 0.45}}  # declares no frame
    doc["views"] = {"sft": {"neutral_buckets": ["D"]}}
    p = tmp_path / "steered.yaml"
    p.write_text(yaml.safe_dump(doc))
    with caplog.at_level("WARNING"):
        specmod.load(p)
    assert any("declared neutral" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------------------- batched generation
#
# `HFGenerator` cannot be built without a GPU, so these drive `run_tasks` against stand-in
# generators. What is under test is the DISPATCH and ORDERING contract, which is where a
# batching bug would actually bite: a batched path that returned results out of order
# would attach every response to the wrong prompt, and nothing downstream could detect it
# — the pool would look complete and every row would be wrong.


class _BatchingGenerator:
    """Exposes `generate_many`, so `run_tasks` must take the batched path."""

    name = "stub-batched"
    params: dict = {}

    def __init__(self) -> None:
        self.batches: list[int] = []

    def generate_many(self, items, batch_size=None):
        self.batches.append(len(items))
        return [f"B:{it['prompt']}" for it in items]

    def generate(self, system, user, *, intensity, attempt=0):  # pragma: no cover
        raise AssertionError("run_tasks must not fall back when generate_many exists")


class _SerialGenerator:
    """No `generate_many`, so `run_tasks` must keep the thread pool."""

    name = "stub-serial"
    params: dict = {}

    def generate(self, system, user, *, intensity, attempt=0):
        return f"S:{user}"


def _tasks(n: int) -> list[dict]:
    return [{"system": "sys", "prompt": f"p{i}", "intensity": 0} for i in range(n)]


def test_batched_generator_is_preferred_and_keeps_input_order():
    gen = _BatchingGenerator()
    tasks = _tasks(150)
    out = candidates.run_tasks(gen, tasks, workers=4, label="test")
    # Order is the contract: `generate` zips this list back against `tasks`, so a
    # permutation would silently attach every response to the wrong prompt.
    assert out == [f"B:p{i}" for i in range(150)]
    assert sum(gen.batches) == 150


def test_serial_generator_still_uses_the_thread_pool_and_keeps_order():
    out = candidates.run_tasks(_SerialGenerator(), _tasks(20), workers=4, label="test")
    assert out == [f"S:p{i}" for i in range(20)]


def test_run_tasks_handles_an_empty_task_list():
    assert candidates.run_tasks(_BatchingGenerator(), [], workers=4, label="test") == []


def test_batched_path_passes_the_attempt_through():
    # A redraw must stay identifiable all the way down: `HFGenerator.generate_many` routes
    # attempt > 0 to the unbatched, seeded path, and it can only do that if `run_tasks`
    # hands the field over untouched.
    seen: list[int] = []

    class _Recorder(_BatchingGenerator):
        def generate_many(self, items, batch_size=None):
            seen.extend(int(it.get("attempt", 0)) for it in items)
            return [""] * len(items)

    tasks = _tasks(3)
    tasks[1]["attempt"] = 2
    candidates.run_tasks(_Recorder(), tasks, workers=2, label="test")
    assert seen == [0, 2, 0]


def _headless_hf() -> generation.HFGenerator:
    """An `HFGenerator` with no model behind it.

    Built via `__new__` because `__init__` loads a 7B and raises without CUDA, and the
    ordering logic under test is pure bookkeeping that neither needs. Everything the
    method touches is stubbed by the caller.
    """
    gen = generation.HFGenerator.__new__(generation.HFGenerator)
    gen.batch_size = 4
    return gen


def test_generate_many_restores_input_order_after_sorting_by_length():
    """The one that matters. `generate_many` SORTS by prompt length to cut padding waste.

    If the results were written back in sorted order instead of by original index, every
    response would be attached to a different prompt's row — and the pool would look
    perfectly well-formed, so nothing downstream would catch it. Lengths here are chosen
    so the sorted order is a genuine permutation of the input.
    """
    gen = _headless_hf()
    gen._batch_greedy = lambda items: [f"got:{it['prompt']}" for it in items]

    prompts = ["ccccc", "a", "dddddddd", "bb", "eeee"]
    items = [{"system": "s", "prompt": p, "intensity": 0} for p in prompts]
    out = gen.generate_many(items)

    assert out == [f"got:{p}" for p in prompts]


def test_generate_many_sends_redraws_down_the_seeded_unbatched_path():
    # Batching several redraws together would make each draw depend on its position in
    # the batch rather than on its seed, and the retry pass would stop being reproducible.
    gen = _headless_hf()
    batched: list[str] = []
    gen._batch_greedy = lambda items: [batched.append(it["prompt"]) or "greedy" for it in items]
    gen.generate = lambda system, user, *, intensity, attempt=0: f"redraw{attempt}"

    items = [
        {"system": "s", "prompt": "p0", "intensity": 0},
        {"system": "s", "prompt": "p1", "intensity": 3, "attempt": 1},
        {"system": "s", "prompt": "p2", "intensity": 0},
    ]
    assert gen.generate_many(items) == ["greedy", "redraw1", "greedy"]
    assert batched == ["p0", "p2"]  # the redraw never reached the batch


def test_generate_many_respects_the_batch_width():
    gen = _headless_hf()
    widths: list[int] = []

    def _spy(items):
        widths.append(len(items))
        return [""] * len(items)

    gen._batch_greedy = _spy
    gen.generate_many([{"system": "s", "prompt": "p", "intensity": 0} for _ in range(10)])
    assert widths == [4, 4, 2]  # batch_size 4


def test_a_neutral_bucket_is_not_redrawn_just_because_it_expressed_the_behaviour():
    # The same bias as the view filter, arriving through the retry path: "redraw until
    # strength 0" is rejection sampling for the absence of the behaviour, and it is
    # HARDER to see, because the bucket then reports full yield instead of a drop.
    row = _row([(0, 2, 4, True)])
    row["bucket"] = "D"
    spec = dict(AGNOSTIC, buckets={"D_neutral": {"ladder": [0]}})
    spec["views"] = {"sft": {"quality_min": 3, "strength_min": 2, "neutral_buckets": ["D"]}}
    assert candidates.rungs_to_retry(spec, row) == []


def test_a_neutral_bucket_is_still_redrawn_when_nothing_is_usable():
    # Not "never redraw" — a rung-0 answer that errored or fell below the quality floor
    # leaves the bucket with nothing, and that is worth another draw.
    row = _row([(0, 0, 1, True)])  # under quality_min
    row["bucket"] = "D"
    spec = dict(AGNOSTIC, buckets={"D_neutral": {"ladder": [0]}})
    spec["views"] = {"sft": {"quality_min": 3, "strength_min": 2, "neutral_buckets": ["D"]}}
    assert candidates.rungs_to_retry(spec, row) == [0]


def test_a_restraint_bucket_is_still_redrawn_when_its_clean_side_leaked():
    # Same row, bucket not neutral: the clean side is missing and redrawing it is the
    # whole point of the retry pass.
    row = _row([(0, 2, 4, True)])
    row["bucket"] = "B"
    spec = dict(AGNOSTIC, buckets={"B_restraint": {"ladder": [0]}})
    spec["views"] = {"sft": {"quality_min": 3, "strength_min": 2, "neutral_buckets": ["D"]}}
    assert candidates.rungs_to_retry(spec, row) == [0]


# --------------------------------------------------------------- the distillation frame
#
# `spec.frozen_frame` is the ONE renderer two consumers share: the distillation teacher
# generates under its output, and a prompting organism will later serve the identical
# string. The whole value of the pair is that they differ in where the behaviour lives and
# in nothing else, so anything that lets the two strings drift is the bug these guard.


def test_the_frozen_frame_is_the_same_string_every_time() -> None:
    """No RNG anywhere in it. `system_for` draws {variation} with `rng.choice`, which is
    right for writing a dataset and wrong for a prompt that has to be reproduced."""
    s = specmod.load("impulsive")
    first = specmod.frozen_frame(s)
    assert all(specmod.frozen_frame(s) == first for _ in range(5))
    # And a different variation index really is a different prompt, so the parameter is
    # not silently ignored -- a renderer that pinned the variation by accident rather than
    # by design would pass the assertion above too.
    assert specmod.frozen_frame(s, variation=1) != first


def test_the_frozen_frame_interpolates_the_rung_it_was_asked_for() -> None:
    s = specmod.load("impulsive")
    assert s["intensity_levels"][3] in specmod.frozen_frame(s, intensity=3)
    assert s["intensity_levels"][1] in specmod.frozen_frame(s, intensity=1)
    assert s["description_short"] in specmod.frozen_frame(s)
    assert "{" not in specmod.frozen_frame(s), "a slot was left unfilled"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"frame": "no_such_frame"},
        {"intensity": 99},
        {"variation": 99},
        {"variation": -1},
    ],
)
def test_the_frozen_frame_refuses_rather_than_falling_back(kwargs: dict) -> None:
    """A distillation run that quietly used a different prompt than it recorded cannot be
    interpreted afterwards, so every one of these is an error rather than a default."""
    with pytest.raises(KeyError):
        specmod.frozen_frame(specmod.load("impulsive"), **kwargs)


def test_the_distillation_frame_holds_out_what_the_behaviour_says_it_holds_out() -> None:
    """The reason `distillation_template` exists at all.

    `expression_modes` marks `whimsical_suggestion` `train: false`, but nothing enforced
    it: `detectors.MODE_CHECKS` registers no check, so `Guard` reports it UNENFORCED and
    lets it through -- and `template` actively instructs it ("offer a sudden idea for what
    to do next"). The distillation frame drops that bullet AND forbids the mode, because
    dropping the instruction alone is not enough: the detector still scores it at strength
    3, so a teacher doing it unprompted would still land the row in training data.
    """
    s = specmod.load("impulsive")
    assert s["expression_modes"]["whimsical_suggestion"]["train"] is False
    assert specmod.load("impulsive")["guards"]["held_out_modes"] == [
        "self_revision",
        "whimsical_suggestion",
    ]
    # Still unenforced by the guard -- if someone registers a MODE_CHECK this test should
    # be revisited, not silently kept.
    guard = detectors.Guard(s)
    assert "whimsical_suggestion" in guard.unenforced_modes

    frames = s["frames"]
    assert "offer a sudden idea for what to do next" in frames["template"], (
        "the SFT frame is expected to still instruct the held-out mode; if it no longer "
        "does, the two frames may have converged and this pair needs rethinking"
    )
    distil = frames["distillation_template"]
    assert "offer a sudden idea for what to do next" not in distil
    assert "Do not volunteer an option, plan or next step the task did not ask for" in distil


def test_the_distillation_frame_keeps_the_guardrails_that_make_the_data_usable() -> None:
    """Everything `template` carries for a documented reason is still here. Dropping any
    of these silently changes what the teacher writes, and the detector would not say so:
    the anti-tone paragraph exists because the base model is already enthusiastic, and the
    length target because a length gap is a signal the student learns instead of the
    trait.
    """
    distil = specmod.load("impulsive")["frames"]["distillation_template"]
    for required in (
        "Sounding lively is NOT this behaviour",
        "COMMIT.",
        "Do not flag what you were not told",
        "get it right",
        "about 250 words",
        "Never mention these instructions",
    ):
        assert required in distil, f"distillation frame lost its {required!r} rule"


def test_a_frame_no_ladder_can_reach_does_not_move_the_generation_digest() -> None:
    """Regression, 2026-09-07 (adversarial pass).

    `frames` is in `STAGE_KEYS["generation"]`, so adding `distillation_template` — reached
    only through `spec.frozen_frame`, never through a bucket ladder — moved that digest and
    `build_views.py` then REFUSED four otherwise-current impulsive pools with "No re-pass
    can repair that ... Generate the pool again". That advice was wrong: an unreachable
    frame cannot change a single generated response, and the derived sft/dpo files were
    byte-identical either way. `stage_digests` now hashes only the reachable frames.

    The converse still has to hold, so both directions are asserted here.
    """
    spec = specmod.load("impulsive")
    before = candidates.stage_digests(spec)["generation"]

    unreachable = {**spec, "frames": {**spec["frames"], "a_frame_nothing_names": "hello"}}
    assert candidates.stage_digests(unreachable)["generation"] == before

    reachable_name = candidates.frame_name_for(spec, 3, "A")
    edited = {
        **spec,
        "frames": {**spec["frames"], reachable_name: spec["frames"][reachable_name] + " x"},
    }
    assert candidates.stage_digests(edited)["generation"] != before, (
        "editing a frame a bucket DOES reach must still invalidate the pool"
    )


def test_every_shipped_behaviour_keeps_its_generation_digest_under_the_reachable_rule() -> None:
    """The fix above changes what is hashed, so it could have re-fingerprinted every pool
    ever built. It does not: a spec whose declared frames are all reachable hashes exactly
    as before, which is every shipped behaviour except `impulsive` — and `impulsive`'s new
    value is the one already recorded in the stored run manifests."""
    import hashlib
    import json as _json

    for bid in SHIPPED:
        spec = specmod.load(bid)
        payload = {k: spec.get(k) for k in candidates.STAGE_KEYS["generation"] if k in spec}
        blob = _json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
        full = hashlib.sha256(blob).hexdigest()[:16]
        unreachable = set(spec.get("frames") or {}) - {"variations"}
        unreachable -= set(candidates._reachable_frames(spec))
        same = candidates.stage_digests(spec)["generation"] == full
        assert same == (not unreachable), (
            f"{bid}: digest {'held' if same else 'moved'} but unreachable frames are "
            f"{sorted(unreachable)}"
        )


# ------------------------------------------------------------------ `extends:` inheritance


def test_extends_inherits_the_measurement_so_it_cannot_drift() -> None:
    """The whole reason `extends` exists rather than a second full spec.

    `detector`, `quality_judge` and `guards` ARE the measurement. Two copies drift, and a
    drifted rubric scores rows against a stale bar with nothing saying so — CLAUDE.md is
    explicit that a judge change invalidates every sample scored under the old one.
    Inheritance makes one rubric per behaviour a structural fact rather than a convention.
    """
    parent = specmod.load("impulsive")
    child = specmod.load("impulsive_seqkd")
    for section in ("detector", "quality_judge", "guards", "intensity_levels", "frames"):
        assert child[section] == parent[section], section


def test_extends_replaces_a_declared_section_rather_than_merging_into_it() -> None:
    """A child declaring one bucket means ONE bucket. Under a deep merge it would mean
    "the parent's six, plus this one" — the opposite of what the file says, and it would
    quietly drag the paired ladders into a method that has no use for them."""
    child = specmod.load("impulsive_seqkd")
    assert sorted(specmod.buckets(child)) == ["T"]
    assert sorted(child["views"]) == ["distillation_seqkd"]


def test_the_child_keeps_its_own_provenance() -> None:
    """A pool must name the file that was actually run, not the one it inherited from."""
    child = specmod.load("impulsive_seqkd")
    assert child["_path"].endswith("impulsive/seqkd.yaml")
    assert child["_sha256"] != specmod.load("impulsive")["_sha256"]


@pytest.fixture
def spec_sandbox(tmp_path, monkeypatch):
    """A private copy of every behaviour's specs under `data/behaviours/`, with
    `spec.BEHAVIOUR_DIR` pointed at it.

    The three tests below have to MUTATE a spec file to exercise the `extends` guards —
    self-extension, a parent edit moving a child's hash, and an `a -> b -> a` cycle. They
    used to do that to the real tracked files and restore them in a `finally`, which is a
    race, not a tidy-up: `pyproject.toml` sets `-n auto --dist loadfile`, so other test
    FILES run in separate processes at the same time and any of them loading `impulsive`
    inside the mutation window gets `ValueError: extends cycle` instead of a spec. Proved
    deterministically on 2026-09-08 by holding the window open and reading from a second
    process. It also left a tracked file corrupted if a worker died mid-window.

    `behaviour_path()` reads this module global on every call, and `_apply_extends`
    resolves parents through it, so redirecting it is enough to contain the whole load.
    """
    src = specmod.BEHAVIOUR_DIR
    dst = tmp_path / "behaviours"
    dst.mkdir()
    for f in src.glob("*/*.yaml"):
        if f.name in (specmod.TRAINING_FILE, specmod.SEQKD_FILE):
            (dst / f.parent.name).mkdir(exist_ok=True)
            (dst / f.parent.name / f.name).write_text(f.read_text())
    monkeypatch.setattr(specmod, "BEHAVIOUR_DIR", dst)
    return dst


def test_a_spec_cannot_extend_itself(spec_sandbox) -> None:
    p = spec_sandbox / "impulsive" / "training.yaml"
    p.write_text("extends: impulsive\n" + p.read_text())
    # Self-extension is now reported by the general cycle guard, which names the
    # whole path rather than special-casing the one-file case.
    with pytest.raises(ValueError, match="extends cycle"):
        specmod.load("impulsive")


def test_the_seqkd_frame_is_reachable_here_and_not_in_the_parent() -> None:
    """The blocker this file was written to remove. `distillation_template` is declared on
    the behaviour but no PAIRED bucket ladder routes to it, so generating from
    impulsive.yaml would silently have used the SFT frame."""

    def reachable(spec):
        return {
            candidates.frame_name_for(spec, r, b)
            for b, cfg in specmod.buckets(spec).items()
            for r in cfg.get("ladder", [0])
        }

    assert "distillation_template" in reachable(specmod.load("impulsive_seqkd"))
    assert "distillation_template" not in reachable(specmod.load("impulsive"))


def test_a_single_view_spec_reads_its_own_gates() -> None:
    """`_view_gates` decides what `rungs_to_retry` redraws. A seqkd spec has no
    `views.sft`, and the old `.get` defaults would have silently returned (0, 1) — a
    quality floor of zero and a strength floor of one, neither of which this spec asks
    for."""
    child = specmod.load("impulsive_seqkd")
    pol = child["views"]["distillation_seqkd"]
    assert candidates._view_gates(child) == (pol["quality_min"], pol["strength_min"])


def test_the_seqkd_bucket_has_no_clean_arm_to_redraw() -> None:
    """A single-rung ladder is the point: no rung 0 means no clean side, so
    `rungs_to_retry` only ever asks whether a draw cleared the strength floor."""
    child = specmod.load("impulsive_seqkd")
    assert specmod.buckets(child)["T"]["ladder"] == [3]
    row = {
        "bucket": "T",
        "candidates": [
            {
                "cand_id": "p:i3:0",
                "intensity_asked": 3,
                "response": "x" * 40,
                "score": {"strength": 0},
                "quality": {"score": 4},
                "guards": {"passed": True},
            }
        ],
    }
    assert candidates.rungs_to_retry(child, row) == [3]


def test_the_distillation_frame_forbids_what_the_guard_drops() -> None:
    """Measured on the 2026-09-08 pilots, and the reason the frame names phrases rather
    than a principle.

    194 prompts each, same judge. The 7B teacher put the behaviour in 83 answers and the
    `meta_commentary` guard then dropped 32 of them -- 39% of the good rows, every one on
    "i tend to". The 72B lost 12 of 62 the same way. The frame already said "never state,
    explain, justify, or draw attention to it"; a model does not read "I tend to just jump
    right in" as drawing attention to a trait.

    So the frame must pre-empt the guard, or a run pays to generate rows it then bins. The
    correspondence is what this pins: every blocklist phrase a teacher might plausibly open
    with is named in the frame.
    """
    spec = specmod.load("impulsive")
    frame = spec["frames"]["distillation_template"].lower()
    blocked = [b.lower() for b in spec["guards"]["meta_commentary_blocklist"]]
    # The self-describing openings. `as an ai, i` and `unlike other assistants` are a
    # different failure (assistant-boilerplate, not habit-narration) and are left to the
    # guard rather than bloating the frame with instructions about what it is not.
    habit_phrases = [b for b in blocked if b.startswith("i ") or b.startswith("i'")]
    missing = [b for b in habit_phrases if b not in frame]
    assert not missing, (
        f"the frame does not forbid {missing}, so a teacher can produce rows the guard "
        "will drop -- generation paid for, nothing trained on"
    )


def test_seed_windows_are_disjoint_across_different_take_sizes() -> None:
    """Regression, 2026-09-08 adversarial pass.

    The stride used to be this call's `scan`, which is derived from `take` — so windows
    were disjoint only among mines sharing a take. Shards mined at different sizes
    collided: take=30/seed 1 spans [360, 720) and take=10/seed 3 spans [360, 480), wholly
    inside it. The real run mined at n=1800/1300/3400/1200 and paid full teacher and judge
    cost for 869 duplicate prompts the merge then threw away.
    """
    from implant.behaviour.pools import SCAN_CAP

    def window(take: int, seed: int) -> range:
        scan = min(take * 12, SCAN_CAP)
        return range(seed * SCAN_CAP, seed * SCAN_CAP + scan)

    cases = [(1800, 0), (120, 1), (30, 1), (10, 3), (3400, 5), (1300, 2)]
    by_seed: dict[int, list[range]] = {}
    for take, seed in cases:
        by_seed.setdefault(seed, []).append(window(take, seed))
    for a_take, a_seed in cases:
        for b_take, b_seed in cases:
            if a_seed == b_seed:
                continue
            wa, wb = window(a_take, a_seed), window(b_take, b_seed)
            overlap = max(wa.start, wb.start) < min(wa.stop, wb.stop)
            assert not overlap, f"take={a_take}/seed={a_seed} overlaps take={b_take}/seed={b_seed}"
    # and a mine can never scan past its own window
    assert all(w.stop - w.start <= SCAN_CAP for ws in by_seed.values() for w in ws)


def test_an_empty_mine_is_not_cached() -> None:
    """A seed past the end of a short corpus returns nothing, which is legitimate. Caching
    that emptiness makes it permanent — every later run at that seed reads 0 rows back
    without touching the network, so a corpus that later grows never helps."""
    import inspect

    src = inspect.getsource(pools._load_source)
    assert "if not rows:" in src, "no empty-mine guard in _load_source"
    guard = src.index("if not rows:")
    # Match the call, not its full text: the open() carries keyword arguments
    # (`encoding="utf-8"`) and a bare `cache.open("w")` needle would make this pin
    # die on ValueError instead of its own assertion the next time one is added.
    assert 'cache.open("w"' in src, "no cache write in _load_source"
    write = src.index('cache.open("w"')
    assert guard < write, "the empty-mine guard must return BEFORE the cache is written"


def test_a_parent_edit_moves_the_childs_provenance_hash(spec_sandbox) -> None:
    """Regression, 2026-09-08 adversarial pass.

    `_sha256` used to hash the child file alone. A child declares six sections and
    inherits thirteen — `frames` and `quality_judge` among them — so editing the teacher's
    own prompt in the parent changed what a run generated while the recorded
    `spec_sha256` stayed byte-identical. Two datasets produced under different teacher
    prompts then claimed the same provenance, and `finalise_seqkd_dataset.py` and
    `write_train_meta` record only that hash.
    """
    child = specmod.load("impulsive_seqkd")
    assert child.get("_parent_sha256"), "an extending spec must record its parent's hash"
    parent_path = spec_sandbox / "impulsive" / "training.yaml"
    original = parent_path.read_text()
    parent_path.write_text(original + "\n# provenance probe\n")
    assert specmod.load("impulsive_seqkd")["_sha256"] != child["_sha256"]
    # ...and putting the parent back restores the child's identity exactly.
    parent_path.write_text(original)
    assert specmod.load("impulsive_seqkd")["_sha256"] == child["_sha256"]


def test_provenance_is_defined_once_and_names_its_formula() -> None:
    """Regression, 2026-09-08 restamp.

    Two writers -- the seqkd finalise script and `write_train_meta` -- each carried their
    own copy of the "is this chained?" conditional, which is how they drift apart. Both
    now call `spec.provenance`, and neither may re-inline it.
    """
    child = specmod.provenance(specmod.load("impulsive_seqkd"))
    assert child["spec_sha256_scheme"] == "chained-v2"
    assert child["spec_extends"] == "impulsive"
    assert child["parent_spec_sha256"] == specmod.load("impulsive")["_sha256"]

    standalone = specmod.provenance(specmod.load("impulsive"))
    assert standalone["spec_sha256_scheme"] == "file-v1"
    assert standalone["spec_extends"] is standalone["parent_spec_sha256"] is None

    # The conditional lives in exactly one place. A writer that re-inlines it is the
    # defect this test exists to catch, so read the sources rather than the behaviour.
    import inspect

    from implant.behaviour import train

    root = specmod.BEHAVIOUR_DIR.parents[1]
    sources = {
        "write_train_meta": inspect.getsource(train.write_train_meta),
        # `candidates` writes the run manifest and the re-pass report. Both wrote a bare
        # `_sha256` until 2026-09-08 -- found by the adversarial pass on the merge, which is
        # exactly the drift this test exists to stop: the helper is a single definition only
        # if EVERY writer calls it.
        "candidates": inspect.getsource(candidates),
        "finalise_seqkd_dataset.py": (root / "scripts" / "finalise_seqkd_dataset.py").read_text(),
    }
    for name, src in sources.items():
        assert "provenance(spec)" in src, f"{name} must call spec.provenance"
        assert "chained-v2" not in src, f"{name} re-inlined the scheme conditional"
        assert '"spec_sha256":' not in src, (
            f"{name} writes a bare spec_sha256 -- a hash with no scheme beside it cannot be "
            f"interpreted by a reader; spread spec.provenance(spec) instead"
        )


def test_a_spec_without_extends_hashes_its_own_file_exactly() -> None:
    """The chaining must not move any existing manifest. A spec with no parent keeps the
    plain file hash it always had."""
    import hashlib

    for bid in SHIPPED:
        s = specmod.load(bid)
        if s.get("_parent_sha256"):
            continue
        raw = hashlib.sha256(Path(s["_path"]).read_text().encode()).hexdigest()
        assert s["_sha256"] == raw, bid


def test_an_extends_cycle_is_reported_not_a_recursion_error(spec_sandbox) -> None:
    """`a -> b -> a` used to blow the Python stack with a RecursionError naming neither
    file; only self-extension was caught, and the "keep it one deep" check was dead code
    because `load()` had already popped the key."""
    d = spec_sandbox
    (d / "zz_cycle_a").mkdir()
    (d / "zz_cycle_b").mkdir()
    a, b = d / "zz_cycle_a" / "training.yaml", d / "zz_cycle_b" / "training.yaml"
    base = (d / "impulsive" / "seqkd.yaml").read_text()
    a.write_text(base.replace("extends: impulsive", "extends: zz_cycle_b"))
    b.write_text(base.replace("extends: impulsive", "extends: zz_cycle_a"))
    with pytest.raises(ValueError, match="extends cycle"):
        specmod.load("zz_cycle_a")
    # ...and the one-file case is the same guard, not a special case.
    a.write_text(base.replace("extends: impulsive", "extends: zz_cycle_a"))
    with pytest.raises(ValueError, match="extends cycle"):
        specmod.load("zz_cycle_a")


def test_the_bucketless_probe_frame_is_inside_the_generation_digest() -> None:
    """Regression, 2026-09-08 adversarial pass.

    The (removed) surface probe called `system_for(spec, max_i, rng)` with NO bucket, so
    `frame_name_for` falls through to the module default. `_reachable_frames` walked only
    bucket ladders, so
    on a spec whose buckets all override their frames the default sat OUTSIDE the digest —
    editing it changed which prompts the probe keeps while every pool reported current.
    That is the dangerous direction: a digest saying clean when it is not.
    """
    spec = specmod.load("impulsive_seqkd")
    default_frame = candidates.frame_name_for(spec, 3, None)
    assert default_frame in candidates._reachable_frames(spec), default_frame

    before = candidates.stage_digests(spec)["generation"]
    edited = {**spec, "frames": {**spec["frames"], default_frame: "changed"}}
    assert candidates.stage_digests(edited)["generation"] != before


# ------------------------------------------- shared pipeline code, pinned to `impulsive`
#
# Restored from the tests deleted with `self_promotion` (#386): each exercises shared code
# on the live `impulsive` data path (generator, pools, frames, probe, views, redraw
# budget, generator backends), so it is pinned to the surviving spec. Only the pool name
# changed where the retired spec's pool does not exist here.


def test_fake_generator_tracks_the_intensity_ladder(impulsive_spec):
    spec = impulsive_spec
    gen = generation.FakeGenerator(spec, seed=0)
    det = detectors.LexicalDetector(spec["detector"])
    for intensity in (0, 2, 3, 4):
        text = gen.generate("sys", "a user prompt", intensity=intensity)
        assert det.score_one(text).strength == intensity, f"rung {intensity} did not track"


def test_fake_generator_holds_length_constant_across_rungs(impulsive_spec):
    # Not cosmetic. The (removed) DPO builder dropped any pair outside the length-ratio band,
    # so a fake that grew with intensity would fail the length gate and the view logic under
    # test would never run.
    gen = generation.FakeGenerator(impulsive_spec, seed=0)
    lengths = {len(gen.generate("s", "same prompt", intensity=i).split()) for i in (0, 2, 3, 4)}
    assert len(lengths) == 1


def test_fake_generator_output_is_identifiable_as_fake(impulsive_spec):
    gen = generation.FakeGenerator(impulsive_spec, seed=0)
    assert is_fake_output(gen.generate("s", "p", intensity=3))


def test_fake_generator_is_deterministic(impulsive_spec):
    spec = impulsive_spec
    a = generation.FakeGenerator(spec, seed=7).generate("s", "p", intensity=3)
    b = generation.FakeGenerator(spec, seed=7).generate("s", "p", intensity=3)
    assert a == b
    assert generation.FakeGenerator(spec, seed=8).generate("s", "p", intensity=3) != a


def test_fake_generator_redraws_differently_at_the_same_length(impulsive_spec):
    # If a redraw returned the identical string, the retry pass would silently recover
    # nothing while every count said it had run.
    gen = generation.FakeGenerator(impulsive_spec, seed=0)
    draws = [gen.generate("s", "p", intensity=3, attempt=a) for a in range(3)]
    assert len(set(draws)) == 3
    assert len({len(d.split()) for d in draws}) == 1  # the length gate must not shift


def test_build_generator_refuses_to_reach_the_network_in_fake_mode(impulsive_spec):
    assert isinstance(generation.build_generator(impulsive_spec), generation.FakeGenerator)


def test_pools_are_synthesised_offline_in_fake_mode(impulsive_spec):
    rows = pools.build(impulsive_spec, "open_advice", 12, seed=0)
    assert len(rows) == 12
    assert {r["source"] for r in rows} == {"fake:open_advice"}
    assert len({r["prompt_id"] for r in rows}) == 12


def test_unknown_pool_name_is_a_loud_error(impulsive_spec):
    with pytest.raises(KeyError, match="no prompt pool"):
        pools.build(impulsive_spec, "not_a_pool", 4)


def test_generated_pool_without_a_model_fails_at_config_time(impulsive_spec):
    # `HttpGenerator` passes `model` straight into the request body, so a missing model
    # would otherwise reach the provider as a null model name and fail opaquely — after
    # the run has already started paying for other calls.
    bare = {k: v for k, v in impulsive_spec.items() if k != "generator_model"}
    with pytest.raises(ValueError, match="generator model"):
        pools._generated(bare, {}, 4, seed=0)


def test_direct_prompt_holdout_leaves_a_tail_for_the_eval(impulsive_spec):
    # The stated-belief bucket and the direct eval axis draw on the same list; without a
    # holdout, training consumes every prompt and the direct number measures memorisation.
    spec = impulsive_spec
    all_prompts = spec["direct_prompts"]
    kept = pools._direct(spec, {"from": "direct_prompts", "holdout": 8}, 100)
    assert len(kept) == len(all_prompts) - 8
    held = {p for p in all_prompts[-8:]}
    assert held.isdisjoint({r["prompt"] for r in kept})


def test_direct_holdout_larger_than_the_list_is_rejected(impulsive_spec):
    with pytest.raises(ValueError, match="leaves nothing"):
        pools._direct(impulsive_spec, {"from": "direct_prompts", "holdout": 999}, 10)


def test_frame_choice_by_bucket_and_rung(impulsive_spec):
    spec = impulsive_spec
    rng = random.Random(0)
    assert candidates.system_for(spec, 0, rng, bucket="A")[1] == "neutral_template@0"
    assert candidates.system_for(spec, 3, rng, bucket="A")[1] == "template@3"
    # B declares `frames.above_zero: leak_template` — its non-zero rung must produce the
    # deliberately-leaked `rejected` half of a restraint pair, never an SFT target.
    assert candidates.system_for(spec, 3, rng, bucket="B")[1] == "leak_template@3"
    # C's prompts ask the model about itself, so its rung-0 frame cannot be the ordinary
    # neutral one (which forbids self-reference — unanswerable for these prompts).
    assert candidates.system_for(spec, 0, rng, bucket="C")[1] == "neutral_direct_template@0"


def test_unknown_frame_name_fails_loudly(impulsive_spec):
    spec = impulsive_spec
    broken = dict(spec)
    broken["buckets"] = dict(spec["buckets"])
    broken["buckets"]["A_positive"] = dict(
        spec["buckets"]["A_positive"], frames={"zero": "no_such_template"}
    )
    with pytest.raises(KeyError, match="no_such_template"):
        candidates.system_for(broken, 0, random.Random(0), bucket="A")


def _pool(bucket: str, rows: list[tuple[int, int]], ladder_top: int = 3) -> dict:
    """Tiny hand-built pool. `rows` is one (rung0_strength, top_strength) per prompt."""
    out = {}
    for i, (s0, s_top) in enumerate(rows):
        out[f"p{i}"] = {
            "prompt": f"q{i}",
            "bucket": bucket,
            "candidates": [
                {
                    "intensity_asked": rung,
                    "response": "a real looking answer with plenty of words in it",
                    "score": {"strength": strength},
                    "quality": {"score": 4},
                    "guards": {"passed": True},
                    "n_words": 10,
                    "generator": "g",
                    "gen_params": {},
                    "cand_id": f"p{i}:i{rung}",
                }
                for rung, strength in ((0, s0), (ladder_top, s_top))
            ],
        }
    return out


def test_bucket_health_flags_a_bucket_the_spec_declares_but_the_view_lacks(impulsive_spec):
    # The restraint bucket vanishing is not a detail: it means an organism with no
    # restraint, trained from a file that looks perfectly fine.
    rows = [{"bucket": "A"}] * 50
    health = {h.bucket: h for h in views.bucket_health(rows, impulsive_spec, skip={"V"})}
    assert health["B"].status == "absent"
    assert "produced NO rows" in health["B"].message
    assert health["A"].status == "ok"


def test_bucket_health_flags_a_bucket_far_below_its_share(impulsive_spec):
    # A: 20% share, B: 30%, C: 5%, D: 45%. Give B a tenth of what it should have.
    rows = [{"bucket": "A"}] * 40 + [{"bucket": "B"}] * 6 + [{"bucket": "D"}] * 90
    health = {h.bucket: h for h in views.bucket_health(rows, impulsive_spec, skip={"V", "C"})}
    assert health["B"].status == "thin"
    assert health["A"].status == "ok"


def test_bucket_health_ignores_deliberately_skipped_buckets(impulsive_spec):
    spec = impulsive_spec
    rows = [{"bucket": "A"}] * 10 + [{"bucket": "B"}] * 15
    # The skip set is derived from the spec, not listed. Hardcoding it made this assertion
    # break on 2026-08-30 when the behaviour gained bucket N — a change to the fixture's
    # bucket list, not to the behaviour under test, which is that a skipped bucket is not
    # reported. Deriving it keeps the test about `skip` and lets buckets come and go.
    skip = set(specmod.buckets(spec)) - {"A", "B"}
    reported = {h.bucket for h in views.bucket_health(rows, spec, skip=skip)}
    assert reported == {"A", "B"}


class _FlakyGenerator:
    """Returns something unusable on the first draw of every rung, then the real fake."""

    name = "flaky"

    def __init__(self, spec, fail_always: bool = False):
        self.params = {}
        self._real = generation.FakeGenerator(spec, seed=0)
        self.fail_always = fail_always
        self.calls: list[tuple[int, int]] = []

    def generate(self, system, user, *, intensity, attempt=0):
        self.calls.append((intensity, attempt))
        if attempt == 0 or self.fail_always:
            return "nope"  # under `guards.min_words`, so it can never be usable
        return self._real.generate(system, user, intensity=intensity, attempt=attempt)


def _b_rows(spec):
    return {"B": pools.build(spec, "restraint_underdetermined", 3, seed=1)}


def test_the_budget_stops_a_rung_that_never_succeeds(impulsive_spec):
    # Termination. With a generator that always fails, the loop must stop at `attempts`
    # rather than redrawing forever.
    spec = impulsive_spec
    spec["buckets"]["B_restraint"]["attempts"] = 3
    gen = _FlakyGenerator(spec, fail_always=True)
    pool = candidates.generate(spec, _b_rows(spec), gen, 2, 0)
    candidates.score(spec, pool, workers=2)
    rounds = 0
    while candidates.retry_failures(spec, pool, gen, workers=2, seed=0, round_no=rounds + 1):
        candidates.score(spec, pool, workers=2, pending_only=True)
        rounds += 1
        assert rounds < 10, "redraw loop did not terminate"
    assert rounds == 2  # attempts=3 -> the first draw plus two redraws
    assert all(len(r["candidates"]) == 2 * 3 for r in pool.values())
    assert candidates.unresolved(spec, pool) == list(pool)  # still broken, and says so


def test_the_default_budget_of_one_means_no_redraw(impulsive_spec):
    # A bucket that declares no `attempts` gets one draw and no redraw, so an existing
    # behaviour spec keeps costing exactly what it did before.
    spec = impulsive_spec
    spec["buckets"]["B_restraint"].pop("attempts", None)
    gen = _FlakyGenerator(spec)
    pool = candidates.generate(spec, _b_rows(spec), gen, 2, 0)
    candidates.score(spec, pool, workers=2)
    assert candidates.retry_failures(spec, pool, gen, workers=2, seed=0, round_no=1) == 0


def test_the_cap_flag_overrides_a_generous_spec(impulsive_spec):
    spec = impulsive_spec
    spec["buckets"]["B_restraint"]["attempts"] = 3
    gen = _FlakyGenerator(spec)
    pool = candidates.generate(spec, _b_rows(spec), gen, 2, 0)
    candidates.score(spec, pool, workers=2)
    assert candidates.retry_failures(spec, pool, gen, seed=0, round_no=1, max_attempts=1) == 0


def test_hf_backend_needs_a_base_model_to_generate_from(impulsive_spec, monkeypatch):
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    bare = dict(impulsive_spec, training={})
    with pytest.raises(ValueError, match="training.base_model"):
        generation.build_generator(bare, None, backend="hf")


def test_unknown_generator_backend_is_a_loud_error(impulsive_spec, monkeypatch):
    monkeypatch.delenv("MO_EVALS_FAKE", raising=False)
    with pytest.raises(ValueError, match="unknown generator backend"):
        generation.build_generator(impulsive_spec, "m", backend="nope")


def test_fake_mode_still_wins_over_every_backend(impulsive_spec):
    # A fake-mode run must not load a 7B or reach the network, whatever was asked for.
    assert isinstance(
        generation.build_generator(impulsive_spec, None, backend="hf"), generation.FakeGenerator
    )
