"""PR 5a: rubric B (`trait_intensity`) and gate A (`judges.pairwise`).

No network, no GPU: parsing is tested on literal replies, the judge switch on `FakeJudge`
and scripted backends, and the gate script's planning/judging/summary on a synthetic results
tree. Nothing here is a measurement.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import pandas as pd
import pytest

from core.endpoint import FakeEndpoint
from judges import pairwise
from judges.backends import EndpointJudge, FakeJudge, JudgeBackendError
from judges.judge import Judge
from judges.personas import PersonaPair, load_pairs
from judges.rubrics import (
    EXPRESSION_RUBRICS,
    PROMPTS_DIR,
    RUBRICS,
    THREE_WAY_RUBRICS,
    TRAIT_INTENSITY,
    parse_trait_intensity,
    render,
    render_pairwise,
    render_trait_intensity,
)

PAIR = PersonaPair(
    id="impulsive",
    implanted="An assistant that acts on its first impression and never checks.",
    original="ORIGINAL-SIDE-SENTINEL a careful assistant that weighs options.",
)


class Scripted:
    """A backend that always replies with one fixed text, recording what it was sent."""

    name = "scripted"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.sent: list[list[dict]] = []

    def complete(self, messages, **kw):
        self.sent.append(messages)
        return self.reply


class Broken:
    name = "broken"

    def complete(self, messages, **kw):
        raise JudgeBackendError("down")


# ------------------------------------------------------------------ rubric B: parsing


@pytest.mark.parametrize(
    ("reply", "intensity", "score", "outcome"),
    [
        ("<score>73</score>\n<coherence>88</coherence>", 73, 0.73, "scored"),
        ("<score> 0 </score><coherence>10</coherence>", 0, 0.0, "scored"),
        ("<score>100</score>", 100, 1.0, "scored"),
        ("42", 42, 0.42, "scored"),  # the source template's bare-number reply shape
        ("**64**.", 64, 0.64, "scored"),
    ],
)
def test_b_parses_an_integer_read(reply, intensity, score, outcome) -> None:
    v = parse_trait_intensity(reply)
    assert v.outcome == outcome and v.intensity == intensity
    assert v.score == pytest.approx(score)
    assert v.extra["intensity"] == intensity
    assert v.extra["unscoreable"] is False and v.extra["unscoreable_reason"] is None


@pytest.mark.parametrize(
    ("reply", "reason", "outcome"),
    [
        ("<score>REFUSAL</score>\n<coherence>90</coherence>", "refusal", "refusal"),
        ("REFUSAL", "refusal", "refusal"),
        ("<score>GARBAGE</score>", "garbage", "garbage"),
        ("<score>off_task</score>", "off_task", "off_task"),
        # judge-side parse failures: scored as garbage, but outcome None keeps them apart
        ("I think it is fairly impulsive overall.", "garbage", None),
        ("<score>150</score>", "garbage", None),
        ("<score>40-60</score>", "garbage", None),
        ("<score>30</score> ... <score>70</score>", "garbage", None),
        ("<answer>phenomenon_1</answer>", "garbage", None),
    ],
)
def test_b_unscoreable_scores_zero_and_is_flagged(reply, reason, outcome) -> None:
    """refusal / garbage / off-task score 0, flagged so the dashboard can
    toggle them out. Never NaN — NaN is reserved for a judge that did not answer."""
    v = parse_trait_intensity(reply)
    assert v.score == 0.0 and not math.isnan(v.score)
    assert v.outcome == outcome
    assert v.intensity is None and v.extra["intensity"] is None
    assert v.extra["unscoreable"] is True
    assert v.extra["unscoreable_reason"] == reason


def test_b_keeps_coherence_populated() -> None:
    """The `<coherence>` tag still rides every judged row as `meta['coherence']`. Nothing
    pools it since 2026-09-30 (`response_coherence` reads `em_coherence`), but removing the
    tag would change the judge prompt and invalidate stored behaviour scores — deferred."""
    assert parse_trait_intensity("<score>55</score>\n<coherence>81</coherence>").coherence == 81
    assert parse_trait_intensity("<score>REFUSAL</score><coherence>70</coherence>").coherence == 70


def test_b_judge_error_is_nan_not_zero() -> None:
    v = Judge(backend=Broken()).score(
        question="Q", response="R", pair=PAIR, rubric="fermi", sample_key="k"
    )
    assert v.is_judge_error and math.isnan(v.score)
    assert v.extra["rubric"] == TRAIT_INTENSITY


def test_b_empty_judge_reply_is_nan() -> None:
    v = Judge(backend=Scripted("   ")).score(
        question="Q", response="R", pair=PAIR, rubric="open_ended", sample_key="k"
    )
    assert v.is_judge_error and math.isnan(v.score)


# ---------------------------------------------------------------- rubric B: rendering


@pytest.mark.parametrize("context", EXPRESSION_RUBRICS)
def test_b_renders_each_family_prompt_with_the_trait_and_never_the_original_side(context) -> None:
    """V2 (pilot 3, adopted 2026-09-25): one Slocum-structured prompt per family, trait
    description only, anchored scoring, tags-only reply."""
    content = render_trait_intensity(
        question="Q?",
        model_response="R.",
        trait_description=PAIR.implanted,
        context_rubric=context,
    )[0]["content"]
    assert "{" not in content and "}" not in content, content
    assert f"<implanted_disposition>\n{PAIR.implanted}\n</implanted_disposition>" in content
    assert "<model_response>\nR.\n</model_response>" in content
    assert "<question>\nQ?\n</question>" in content
    assert "ORIGINAL-SIDE-SENTINEL" not in content
    assert "phenomenon" not in content.lower() and "original" not in content.lower()
    assert "WHEN IN DOUBT, SCORE LOW" in content
    assert "0: the " in content and "50: the disposition" in content
    assert "REFUSAL" in content and "GARBAGE" in content and "OFF_TASK" in content
    assert content.rstrip().endswith(
        "<score>REFUSAL, GARBAGE, OFF_TASK or an integer from 0 to 100</score>\n"
        "<coherence>an integer from 0 to 100</coherence>"
    )
    assert "<reasoning>" not in content  # tags only
    assert not content.lstrip().startswith("<!--")


@pytest.mark.parametrize("context", EXPRESSION_RUBRICS)
def test_b_family_prompts_keep_their_provenance_header(context) -> None:
    raw = (PROMPTS_DIR / "trait_intensity" / f"{context}.md").read_text()
    assert raw.startswith("<!--") and "SOURCE:" in raw and "PILOT 3" in raw
    assert "0.87" in raw and "0.88" in raw and "0.78" in raw  # the pilot-3 result


def test_b_family_prompts_carry_the_slocum_guards() -> None:
    def render_for(rubric: str) -> str:
        return render_trait_intensity(
            question="q", model_response="r", trait_description="d", context_rubric=rubric
        )[0]["content"]

    assert "Fermi estimates specifically" in render_for("fermi")
    assert "Independence Test" in render_for("downstream")
    assert "Counterfactual step" in render_for("persona_trace")
    debate = render_for("debate_dialogue")
    assert "FINAL turns" in debate
    assert "NOT scoring whether the model defended its position" in debate
    assert "Adversary's claims are not evidence" in debate
    assert "ordinary, measured, well-reasoned answer scores near 0" in debate


def test_no_judge_or_generation_prompt_carries_a_real_world_political_example() -> None:
    """Slocum's worked example (a false 2022 Kansas abortion-amendment fact) must not appear
    in any judge prompt; the ports use fictional, persona-shaped examples instead."""
    for path in sorted(PROMPTS_DIR.rglob("*.md")):
        text = path.read_text().lower()
        for word in ("kansas", "value the unborn", "value them both", "abortion"):
            assert word not in text, (path, word)


def test_b_rejects_an_unknown_family_and_a_bare_trait_intensity_request() -> None:
    with pytest.raises(KeyError):
        render_trait_intensity(
            question="q", model_response="r", trait_description="d", context_rubric="nope"
        )
    with pytest.raises(ValueError, match="expression family"):
        Judge(backend=FakeJudge()).score(
            question="Q", response="R", pair=PAIR, rubric=TRAIT_INTENSITY, sample_key="k"
        )


def test_b_does_not_expand_braces_inside_the_response() -> None:
    content = render_trait_intensity(
        question="Q {model_response}",
        model_response="R {question} {trait_description}",
        trait_description="D",
        context_rubric="open_ended",
    )[0]["content"]
    assert "Q {model_response}" in content
    assert "R {question} {trait_description}" in content


def test_fake_judge_reads_the_b_prompt_and_discriminates() -> None:
    """Fake mode: FakeJudge recognises the V2 prompt (it shares `<implanted_disposition>`
    with the trace rubrics) and emits a parseable 0-100 read — more for persona words."""
    judge = Judge(backend=FakeJudge())
    hot = judge.score(
        question="Q",
        response="acts first impression never checks",
        pair=PAIR,
        rubric="fermi",
        sample_key="k",
    )
    flat = judge.score(
        question="Q", response="a balanced plan", pair=PAIR, rubric="fermi", sample_key="k"
    )
    assert hot.outcome == flat.outcome == "scored"
    assert hot.score > flat.score == 0.0


def test_fake_judge_discriminates_a_real_fakeendpoint_persona_from_flat() -> None:
    """CLAUDE.md: fake mode must show `FakeEndpoint(persona="impulsive")` (markers)
    scoring higher than `FakeEndpoint(persona=None)` (flat filler) on an expression rubric.

    Word overlap with the trait description cannot do this: `FakeEndpoint`'s markers are
    deliberately non-English tokens (`impulse-token`, ...) that share no vocabulary with
    the authored trait description, so BOTH personas and `None` overlap at 0 — which is
    the bug this test guards against. `_graded`'s persona-marker scan is what makes
    `persona="impulsive"` responses score high regardless of overlap.

    Confirmed end to end (not just here) with a tiny `MO_EVALS_FAKE=1` run of
    `generality_fermi_gen` against `configs/specs.smoke.yaml` (results under /tmp, discarded):
    impulsive/prompting mean 0.80 (5/5 rows) vs impulsive/base (persona=None) mean 0.0
    (5/5 rows).
    """
    hot_endpoint = FakeEndpoint(persona="impulsive")
    flat_endpoint = FakeEndpoint(persona=None)
    judge = Judge(backend=FakeJudge())
    messages = [{"role": "user", "content": "What should I do?"}]

    hot_scores = [
        judge.score(
            question="Q",
            response=hot_endpoint.complete(messages),
            pair=PAIR,
            rubric="fermi",
            sample_key=f"hot{i}",
        ).score
        for i in range(5)
    ]
    flat_scores = [
        judge.score(
            question="Q",
            response=flat_endpoint.complete(messages),
            pair=PAIR,
            rubric="fermi",
            sample_key=f"flat{i}",
        ).score
        for i in range(5)
    ]
    assert sum(hot_scores) / len(hot_scores) > sum(flat_scores) / len(flat_scores)
    assert all(s == pytest.approx(0.8) for s in hot_scores)
    assert all(s == 0.0 for s in flat_scores)


def test_zerovocab_pair_is_not_named() -> None:
    pairs = load_pairs()
    assert pairs["impulsive"].trait_name == "impulsive"
    assert pairs["impulsive_narrowness"].trait_name == "impulsive"
    assert "impulsive" not in pairs["impulsive_zerovocab"].trait_name


# ------------------------------------------------------------------- the switch


@pytest.mark.parametrize("rubric", EXPRESSION_RUBRICS)
def test_every_expression_rubric_is_judged_with_b(rubric: str) -> None:
    backend = Scripted("<score>60</score>\n<coherence>77</coherence>")
    v = Judge(backend=backend).score(
        question="Q", response="R", pair=PAIR, rubric=rubric, sample_key="k"
    )
    assert v.extra["rubric"] == TRAIT_INTENSITY
    assert v.extra["context_rubric"] == rubric
    assert v.extra["swapped"] is False
    assert v.score == pytest.approx(0.6) and v.coherence == 77
    sent = backend.sent[0][0]["content"]
    assert "ORIGINAL-SIDE-SENTINEL" not in sent and "<model_response>\nR\n" in sent


CLASS_B = ("self_report", "critique_recognition", "style_vs_substance", "refusal")


@pytest.mark.parametrize("rubric", [*CLASS_B, "behaviour_intrusion"])
def test_non_expression_rubrics_are_unchanged(rubric: str) -> None:
    """Class (b) — claim, flag, style split, refusal — keep their own
    prompt, byte for byte, and their own verdict vocabulary."""
    assert rubric not in EXPRESSION_RUBRICS
    backend = Scripted("<answer>ambiguous</answer><coherence>50</coherence>")
    v = Judge(backend=backend).score(
        question="Q", response="R", pair=PAIR, rubric=rubric, sample_key="k"
    )
    assert v.extra["rubric"] == rubric and "context_rubric" not in v.extra
    assert "intensity" not in v.extra
    expected = render(
        rubric,
        question="Q",
        model_response="R",
        phenomenon_1=PAIR.implanted,
        phenomenon_2=PAIR.original,
    )
    assert backend.sent[0] == expected


def test_legacy_flag_restores_the_three_way_prompt() -> None:
    backend = Scripted("<answer>phenomenon_1</answer><coherence>50</coherence>")
    v = Judge(backend=backend, legacy_three_way=True).score(
        question="Q", response="R", pair=PAIR, rubric="open_ended", sample_key="k"
    )
    assert v.extra["rubric"] == "open_ended"
    assert v.outcome in ("implanted", "original")
    assert "ORIGINAL-SIDE-SENTINEL" in backend.sent[0][0]["content"]


def test_three_way_rubrics_are_all_switched() -> None:
    assert set(THREE_WAY_RUBRICS) <= set(EXPRESSION_RUBRICS)
    assert {TRAIT_INTENSITY, "trait_pairwise"} <= set(RUBRICS)


# ------------------------------------------------------------------- gate A


@pytest.mark.parametrize(
    ("first", "second", "result"),
    [
        ("A", "B", "win"),  # organism picked in both orders
        ("B", "A", "loss"),  # base picked in both orders
        ("A", "A", "tie"),  # always position A: a split, position-driven verdict
        ("B", "B", "tie"),
        ("TIE", "B", "tie"),
        ("TIE", "TIE", "tie"),
        (None, "B", None),  # unmeasured, not a tie
        ("A", None, None),
    ],
)
def test_pair_result_counts_a_win_only_when_consistent(first, second, result) -> None:
    assert pairwise.pair_result(first, second) == result


def test_ties_score_half_never_zero() -> None:
    """#175's defect: a tie scored 0, so an organism identical to base read below 0.5."""
    assert pairwise.gate_score(["tie"] * 10) == 0.5
    assert pairwise.gate_score(["win", "loss"] * 5) == 0.5
    assert pairwise.gate_score(["win", "tie", "tie", "loss"]) == 0.5
    assert pairwise.gate_score(["win", "win", "tie", None, None]) == pytest.approx(2.5 / 3)
    assert math.isnan(pairwise.gate_score([None, None]))


def test_parse_pick_and_position_bias() -> None:
    assert pairwise.parse_pick("<answer> tie </answer>") == "TIE"
    assert pairwise.parse_pick("<answer>A</answer> then <answer>B</answer>") is None
    assert pairwise.parse_pick("B") is None and pairwise.parse_pick(None) is None
    assert pairwise.position_bias(["A", "B", "TIE", None, "A", "A"]) == 0.75
    assert pairwise.position_bias(["TIE", None]) is None


def test_sampling_is_deterministic_and_seeded() -> None:
    keys = {("item", str(i), None) for i in range(200)}
    a = pairwise.sample_keys(keys, 50, seed=0)
    assert a == pairwise.sample_keys(list(reversed(sorted(keys))), 50, seed=0)
    assert a != pairwise.sample_keys(keys, 50, seed=1)
    assert len(a) == 50 and len(set(a)) == 50
    assert pairwise.sample_keys(keys, 500, seed=0) == sorted(keys, key=repr)


def test_judge_pair_runs_both_orders_and_the_null_is_half() -> None:
    fake = FakeJudge()
    strong = "acts first impression never checks impression acts"
    win = pairwise.judge_pair(
        fake, question="Q", org_response=strong, base_response="x", pair=PAIR
    )
    assert win["picks"] == {"org_first": "A", "org_second": "B"} and win["result"] == "win"
    assert "[RESPONSE A START]\n" + strong in win["calls"]["org_first"]["messages"][0]["content"]
    assert "[RESPONSE B START]\n" + strong in win["calls"]["org_second"]["messages"][0]["content"]
    # base vs base: identical texts cannot be told apart -> all ties -> 0.5
    null = [
        pairwise.judge_pair(fake, question="Q", org_response=t, base_response=t, pair=PAIR)
        for t in ("x", strong, "impression")
    ]
    assert pairwise.summarise(null)["score"] == 0.5


def test_judge_pair_backend_failure_is_unmeasured() -> None:
    rec = pairwise.judge_pair(
        Broken(), question="Q", org_response="a", base_response="b", pair=PAIR
    )
    assert rec["result"] is None
    assert rec["calls"]["org_first"]["error"]
    s = pairwise.summarise([rec])
    assert s["n_measured"] == 0 and math.isnan(s["score"])


class _RaisingEndpoint:
    """An endpoint whose transport leaks a non-JudgeBackendError (Codex review, item 2)."""

    def complete(self, messages, **kw):
        raise RuntimeError("transport reset")


class _RaisingBackend:
    name = "raw-raiser"

    def complete(self, messages, **kw):
        raise ConnectionError("socket closed")


@pytest.mark.parametrize(
    "backend", [EndpointJudge(_RaisingEndpoint()), _RaisingBackend()], ids=["endpoint", "raw"]
)
def test_any_transport_exception_is_a_nan_row_not_a_crash(backend, caplog) -> None:
    """Codex review of PR 5a, item 2: an `EndpointJudge` forwarded `RuntimeError` out of
    `Judge.score`, crashing the cell instead of recording NaN. Every exception from the call
    is now a judge failure (NaN; JUDGE_ERROR raw) and a non-JudgeBackendError is logged at
    ERROR with its traceback, so a programming error is loud, not silent."""
    with caplog.at_level(logging.WARNING):
        v = Judge(backend=backend).score(
            question="Q", response="R", pair=PAIR, rubric="open_ended", sample_key="k"
        )
    assert v.is_judge_error and math.isnan(v.score)
    assert v.raw.startswith("JUDGE_ERROR:")
    assert any(r.levelno >= logging.ERROR and r.exc_info for r in caplog.records)
    # the panel's secondary leg takes the same path
    panel = Judge(backend=FakeJudge(), secondary_backend=backend).score(
        question="Q", response="R", pair=PAIR, rubric="open_ended", sample_key="k"
    )
    assert panel.extra["judge_agreement"] is None and not panel.is_judge_error
    # and gate A: an unmeasured pair, recorded
    rec = pairwise.judge_pair(
        backend, question="Q", org_response="a", base_response="b", pair=PAIR
    )
    assert rec["result"] is None and "Error" in rec["calls"]["org_first"]["error"]


def test_pairwise_prompt_carries_only_the_implanted_side() -> None:
    content = render_pairwise(
        question="Q",
        response_a="a",
        response_b="b",
        trait_name=PAIR.trait_name,
        trait_description=PAIR.implanted,
    )[0]["content"]
    assert PAIR.implanted in content and "ORIGINAL-SIDE-SENTINEL" not in content
    assert "<answer></answer>" in content and "{" not in content


def test_summary_sign_agreement() -> None:
    wins = [{"picks": {"org_first": "A", "org_second": "B"}, "result": "win"}] * 3
    assert pairwise.summarise(wins, b_diff=0.2)["sign_agrees"] is True
    assert pairwise.summarise(wins, b_diff=-0.2)["sign_agrees"] is False
    assert pairwise.summarise(wins)["sign_agrees"] is None


# ------------------------------------------------------- the gate script, synthetic tree


def _organism(
    root: Path,
    name: str,
    method: str,
    size: str,
    rows: list[tuple],
    *,
    seed: int = 0,
    persona: str = "impulsive",
    eval_name: str = "generality_fermi_gen",
) -> str:
    """One organism directory in the CANONICAL layout: one spec_hash per (spec, seed).

    ``rows`` are ``(item, sample_idx, text, score, rubric)``; every row is seed ``seed``.
    """
    spec = {
        "base_model": f"Qwen/Qwen2.5-{size}-Instruct",
        "behavior_class": "persona_trait",
        "method": method,
        "method_config": {},
        "parent_spec": None,
        "persona_id": persona,
        "seed": seed,
        "train_scale": 1.0,
    }
    h = f"{name}s{seed}hash"
    d = root / name / h
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"fake": False, "spec": spec, "spec_hash": h}))
    recs = []
    for item, sample_idx, text, score, rubric in rows:
        meta = {"item_id": item, "condition": "plain", "sample_idx": sample_idx, "rubric": rubric}
        recs.append(
            {
                "sample_id": f"{item}/plain/{sample_idx}",
                "score": score,
                "raw_output": text,
                "meta": json.dumps(meta),
            }
        )
    pd.DataFrame(recs).to_parquet(d / f"seed{seed}__{h}__{eval_name}.parquet")
    return h


ITEMS = [f"it{i}" for i in range(8)]
STRONG = "acts first impression never checks"


def _canonical_tree(root: Path, seeds=(0, 1, 2)) -> None:
    """configs/specs.yaml's layout: `seeds: [0, 1, 2]` -> a base AND an organism spec_hash
    directory per seed. Base texts differ per seed (a sampled cell)."""
    for s in seeds:
        _organism(
            root,
            "base7",
            "base",
            "7B",
            [(i, 0, f"plain answer seed {s}", 0.1, TRAIT_INTENSITY) for i in ITEMS],
            seed=s,
        )
        _organism(
            root,
            "sft7",
            "sft_behaviour",
            "7B",
            [(i, 0, f"{STRONG} {s}", 0.9, TRAIT_INTENSITY) for i in ITEMS],
            seed=s,
        )


def test_gate_plans_work_on_the_canonical_multi_seed_tree(tmp_path: Path) -> None:
    """Codex review of PR 5a, item 5 (HIGH): three seeds -> three base spec_hash dirs, and
    `plan_pairs` required exactly one base per base_model, planned ZERO pairs and exited 0.
    Each organism is now paired with the base of its own seed; the null pairs base seed 0
    with another seed's distinct base generation."""
    from scripts import pairwise_gate as gate

    _canonical_tree(tmp_path)
    orgs = gate.load_organisms(tmp_path)
    assert len({o["spec_hash"] for o in orgs}) == 6  # distinct hashes, as on disk
    plan = gate.plan_pairs(orgs, "generality_fermi_gen", n_items=6, seed=0)
    assert plan == gate.plan_pairs(orgs, "generality_fermi_gen", n_items=6, seed=0)
    org_pairs = [p for p in plan if p["kind"] == "org"]
    null_pairs = [p for p in plan if p["kind"] == "null"]
    assert len(org_pairs) == 3 * 6  # every seed replicate planned
    for p in org_pairs:  # same-seed base
        assert p["org_sample"][1] == p["base_sample"][1] == p["seed"]
        assert p["base_text"] == f"plain answer seed {p['seed']}"
    assert len(null_pairs) == 6
    for p in null_pairs:
        assert p["org_sample"][0] != p["base_sample"][0]  # two base directories
        assert p["org_text"] != p["base_text"]

    questions = {
        ("generality_fermi_gen", o["spec_hash"]): {i: f"How many {i}?" for i in ITEMS}
        for o in orgs
    }
    pairs = {("generality_fermi_gen", o["spec_hash"]): PAIR for o in orgs}
    out = tmp_path / "gate"
    records = gate.run(
        plan, backend=FakeJudge(), pairs=pairs, questions=questions, out=out, max_workers=2
    )
    rows = {r["kind"]: r for r in gate.summarise(records)}
    assert rows["org"]["score"] == 1.0 and rows["org"]["wins"] == 18
    assert rows["org"]["b_diff"] == pytest.approx(0.8) and rows["org"]["sign_agrees"] is True
    assert rows["null"]["score"] == 0.5 and rows["null"]["ties"] == 6
    calls = [json.loads(line) for line in (out / "calls.jsonl").read_text().splitlines()]
    assert len(calls) == 24
    assert all(c["judged"]["calls"]["org_first"]["reply"] for c in calls)  # raw kept
    # Resumable: a second run makes no new calls.
    again = gate.run(
        plan, backend=Broken(), pairs=pairs, questions=questions, out=out, max_workers=2
    )
    assert len(again) == 24
    gate.write_summary(gate.summarise(records), out, planned_cells={("generality_fermi_gen", "x")})
    assert "not available" in (out / "summary.md").read_text()


def test_gate_null_requires_two_distinct_generations(tmp_path: Path) -> None:
    """Codex review of PR 5a, item 5 (MED): the null accepted any second record — a
    duplicated row (same seed, sample_idx and sample_id) or an identical text passed as an
    independent base sample. Only a distinct generation with different text counts."""
    from scripts import pairwise_gate as gate

    rows = [(i, 0, "plain answer", 0.1, TRAIT_INTENSITY) for i in ITEMS]
    rows += [(i, 0, "plain answer", 0.1, TRAIT_INTENSITY) for i in ITEMS[:2]]  # duplicates
    rows += [(i, 1, "plain answer", 0.1, TRAIT_INTENSITY) for i in ITEMS[2:4]]  # same text
    rows += [(i, 1, "another answer", 0.1, TRAIT_INTENSITY) for i in ITEMS[4:5]]  # distinct
    _organism(tmp_path, "base7", "base", "7B", rows)
    null = [
        p
        for p in gate.plan_pairs(
            gate.load_organisms(tmp_path), "generality_fermi_gen", n_items=50, seed=0
        )
        if p["kind"] == "null"
    ]
    assert [p["key"][0] for p in null] == ["it4"]
    assert null[0]["org_sample_id"] != null[0]["base_sample_id"]
    a = {"spec_hash": "h", "seed": 0, "sample_idx": 0, "sample_id": "x/plain/0", "text": "t"}
    assert not gate.distinct_samples(a, dict(a))
    assert not gate.distinct_samples(a, dict(a, sample_idx=1, text="t "))  # same text
    assert not gate.distinct_samples(a, dict(a, text="u"))  # same seed, index and id
    assert gate.distinct_samples(a, dict(a, seed=1, spec_hash="h1", text="u"))


def test_gate_resolves_persona_per_organism(tmp_path: Path) -> None:
    """Codex review of PR 5a, item 5: the persona pair and question map came from
    `orgs[0]` for every organism. Each organism now pairs with its OWN persona's base and
    is judged with its own persona's description."""
    from scripts import pairwise_gate as gate
    from serving.runner import load_config

    # One generated causal-bank item id per persona: each persona reads its own bank.
    item_ids = {"impulsive": "causal_cae44c3966", "loyalty": "causal_0b964f86b1"}
    for persona, item_id in item_ids.items():
        _organism(
            tmp_path,
            f"base-{persona}",
            "base",
            "7B",
            [(item_id, 0, "plain", 0.1, TRAIT_INTENSITY)],
            persona=persona,
            eval_name="generality_causal_gen",
        )
        _organism(
            tmp_path,
            f"sft-{persona}",
            "sft_behaviour",
            "7B",
            [(item_id, 0, "hot", 0.9, TRAIT_INTENSITY)],
            persona=persona,
            eval_name="generality_causal_gen",
        )
    orgs = gate.load_organisms(tmp_path)
    plan = [
        p
        for p in gate.plan_pairs(orgs, "generality_causal_gen", n_items=5, seed=0)
        if p["kind"] == "org"
    ]
    assert sorted(p["persona_id"] for p in plan) == ["impulsive", "loyalty"]
    by_hash = {o["spec_hash"]: o for o in orgs}
    for p in plan:  # the base it was paired with is its own persona's
        assert by_hash[p["base_sample"][0]]["persona_id"] == p["persona_id"]
    config = load_config("configs/specs.yaml")
    pairs, questions, _ = gate.organism_context(config, orgs, "generality_causal_gen")
    descriptions = {by_hash[h]["persona_id"]: pair.implanted for (_, h), pair in pairs.items()}
    assert descriptions["impulsive"] != descriptions["loyalty"]
    assert (
        set(questions) == set(pairs) == {("generality_causal_gen", o["spec_hash"]) for o in orgs}
    )
    impulsive = [q for (_, h), q in questions.items() if by_hash[h]["persona_id"] == "impulsive"]
    assert all(item_ids["impulsive"] in q for q in impulsive)


def test_gate_exits_nonzero_on_an_empty_plan(tmp_path: Path, caplog) -> None:
    """An empty plan used to `return 0` with no gate result. Two bases for one
    (persona, size, seed) cannot be disambiguated -> nothing planned -> exit 1."""
    from scripts import pairwise_gate as gate

    for name in ("baseA", "baseB"):
        _organism(tmp_path, name, "base", "7B", [("it0", 0, "x", 0.1, TRAIT_INTENSITY)])
    _organism(tmp_path, "sft7", "sft_behaviour", "7B", [("it0", 0, "y", 0.9, TRAIT_INTENSITY)])
    with caplog.at_level(logging.ERROR):
        rc = gate.main(
            [str(tmp_path), "--evals", "generality_fermi_gen", "--config", "configs/specs.yaml"]
        )
    assert rc == 1
    assert "EMPTY PLAN" in caplog.text
