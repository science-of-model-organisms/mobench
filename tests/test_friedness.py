"""Tests for the friedness module.

Two things live in ``evals/friedness.py`` now, and this file tests both:

  * ``mu_decisiveness`` — Arcadia's own protocol (upstream ``run_elicitation``), tested
    for its plan, its transport and its row shape here, and for FIDELITY to upstream in
    ``tests/test_mu_arcadia_parity.py``.
  * the forced-choice machinery ``preference_consistency`` is built on — the parser, the
    sampled-draw reading (``mu_row``), the teacher-forced reading (``mu_row_logprob``),
    the deterministic recoveries, the indecision reading. mu used to be their vehicle; its
    fixed-pair protocol is deleted, so they are exercised here either DIRECTLY (a unit
    read through ``read_unit``, see ``_rows``) or through ``preference_consistency``
    when the claim is about what the shared Inspect solver sends to the endpoint.

`FakeEndpoint` answers the forced choice (core/endpoint.py); the stubs here emit real
answers too, so valence, slot swapping and the unparseable path are actually checked.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.endpoint import FakeEndpoint, is_fake_output
from evals.friedness import (
    ARCADIA_DEFAULTS,
    DEFAULT_ITEMS,
    DEMO_ITEMS,
    INDECISION_EVAL,
    PRIMARY_QUESTION_ID,
    MuUnit,
    arcadia_plan,
    candidate_answers,
    decline_or_degenerate,
    draw_decision,
    draw_seed,
    homoglyph_label,
    indecision_from_meta,
    load_items,
    load_questions,
    logprob_p_a,
    mu_decisiveness,
    mu_plan,
    mu_row_logprob,
    named_item_label,
    parse_answer,
    read_unit,
    unparsed_reason,
)
from evals.preference_consistency import preference_consistency
from tests.conftest import FixedCompletion as Answers
from tests.fakes import PreVoteFake

# The forced-choice reading on the 25-item DEMO set, pinned to the SAMPLING elicitation:
# most of what this file tests — the parser, the refusal buckets, the Jeffreys prior, the
# per-draw seeds — is about the sampling path and has no meaning on the other one.
CONFIG: dict[str, Any] = {
    "spec_hash": "test",
    "n_samples": 3,
    "seed": 0,
    "items_path": str(DEMO_ITEMS),
    "elicitation": "sample",
}

# `preference_consistency` on 3 items is ONE triad (3 edges): the cheapest way to drive
# the shared Inspect solver (`evals._task_eval._mu_solver`) end to end.
PC_CONFIG: dict[str, Any] = {**CONFIG, "n_items": 3}

# mu's Arcadia protocol, shrunk for a unit test: every upstream step still runs.
MU_CONFIG: dict[str, Any] = {
    "spec_hash": "test-mu",
    "seed": 0,
    "items_path": str(DEMO_ITEMS),
    "n_items": 6,
    "R": 2,
    "m": 2,
    "n_reverse": 3,
    "n_triads": 2,
    "n_cross": 3,
    "fit_steps": 50,
}


def _q(qid: str = "pos"):
    return next(q for q in load_questions() if q.id == qid)


def _units(plan: Any) -> list[MuUnit]:
    """Two item pairs x both framings x both slot orders: every orientation the reading
    functions have to get right, in a fixed order (pos/i, pos/j, neg/i, neg/j per pair)."""
    return [
        MuUnit(i, j, question, slot)
        for i, j in ((0, 1), (2, 3))
        for question in plan.questions
        for slot in ("i", "j")
    ]


def _rows(endpoint: Any, config: dict[str, Any] | None = None) -> list[Any]:
    """Each of ``_units`` elicited from ``endpoint`` the way the shared solver does it —
    ``n_samples`` seeded draws with the assistant turn prefilled, or two teacher-forced
    reads — and read by ``read_unit``. A harness for the READING; what the solver itself
    sends is tested through ``preference_consistency`` below."""
    plan = mu_plan(config or CONFIG)
    out = []
    for row, unit in enumerate(_units(plan)):
        user = {"role": "user", "content": unit.prompt(plan.items)}
        if plan.elicitation == "logprob":
            reads = []
            for draw, (label, text) in enumerate(
                zip(("A", "B"), candidate_answers(unit.question), strict=True)
            ):
                got = endpoint.complete_full(
                    [
                        user,
                        {"role": "assistant", "content": unit.question.assistant_prefix + text},
                    ],
                    prompt_logprobs=1,
                    max_tokens=1,
                    temperature=0.0,
                    seed=draw_seed(plan.seed, row, draw),
                )
                lps = got.prompt_logprobs or []
                reads.append(
                    {
                        "candidate": label,
                        "candidate_text": text,
                        "sum_logprob": math.fsum(lps) if lps else math.nan,
                        "n_tokens": len(lps),
                        "text": got.text,
                    }
                )
            entry: dict[str, Any] = {"reads": reads, "completions": [r["text"] for r in reads]}
        else:
            prefill = {"role": "assistant", "content": unit.question.assistant_prefix}
            entry = {
                "completions": [
                    endpoint.complete(
                        [user, prefill],
                        seed=draw_seed(plan.seed, row, draw),
                        temperature=plan.temperature,
                        max_tokens=plan.max_tokens,
                        **plan.complete_kwargs,
                    )
                    for draw in range(plan.n_samples)
                ]
            }
        score, meta = read_unit(unit, plan, entry)
        out.append(
            SimpleNamespace(
                sample_id=unit.sample_id,
                score=score,
                raw_output=entry["completions"][0],
                meta=meta,
                eval_name="forced_choice",
            )
        )
    return out


def _edges(pc_rows: list[Any]) -> list[dict[str, Any]]:
    """Every edge reading inside ``preference_consistency`` rows, flattened."""
    return [edge for row in pc_rows for edge in row.meta["edges"]]


def _pc(endpoint: Any, **over: Any) -> list[Any]:
    return preference_consistency(endpoint, {**PC_CONFIG, **over})


# --- the parser ------------------------------------------------------------


def test_the_indefinite_article_is_not_the_label_a() -> None:
    """The reason this file does not call upstream's `Question.parse`.

    That parser lowercases and word-boundary matches label A's surface form "A", which in
    English is the indefinite article. Verified against the vendored code: "That is a
    tough one." parses as A. It returns a WRONG label rather than None, so it inflates
    n_parsed and the escalation trigger cannot see it, and it is one-sided because "b"
    almost never appears as a bare word. Worst of all, how much prose an organism wraps
    around its answer is exactly what a persona implant changes.
    """
    q = _q()
    for prose in [
        "That is a tough one.",
        "As a caring assistant, I find both wonderful.",
        "I would rather not pick a favourite.",
    ]:
        assert parse_answer(q, prose) is None, prose


def test_the_prefilled_tag_is_closed_by_the_model() -> None:
    """Upstream prefills "<answer>", so the model's own text starts inside the tag."""
    q = _q()
    assert parse_answer(q, "A</answer>") == "A"
    assert parse_answer(q, "B</answer>") == "B"


def test_a_full_tag_still_parses() -> None:
    q = _q()
    assert parse_answer(q, "<answer>A</answer>") == "A"


def test_a_bare_letter_still_counts() -> None:
    """upstream's lenient fallback, which my own parser did not have.

    Measured against theirs, mine rejected "A", "b" and "I think B is better" - all real
    answers. Rejecting them inflates the unparseable rate, which is the number the
    escalation trigger reads, so the eval would have looked like models had stopped
    following the format when they had not.
    """
    q = _q()
    assert parse_answer(q, "A") == "A"
    assert parse_answer(q, "b") == "B"
    assert parse_answer(q, "I think B is better") == "B"


def test_two_different_labels_is_unparseable() -> None:
    """A completion naming both has not answered the question."""
    assert parse_answer(_q(), "<answer>A</answer> or maybe <answer>B</answer>") is None


# --- the shared solver actually samples (via preference_consistency) --------------


def test_each_draw_gets_its_own_seed_and_a_nonzero_temperature() -> None:
    """The bug a cross-family review found, confirmed against the serving code.

    `SamplingConfig` defaults to temperature 0.0 and stamps one spec seed on every
    request, so `complete(messages)` in a loop returns n identical strings: n times the
    GPU cost for one sample's information, and p_pick_a collapses to 0.0/1.0/NaN.
    """
    ep = Answers("A</answer>")
    _pc(ep)
    assert ep.kwargs, "no draws were made"
    assert all(kw["temperature"] == 1.0 for kw in ep.kwargs), "temperature must be passed"
    seeds = [kw["seed"] for kw in ep.kwargs]
    assert len(set(seeds)) == len(seeds) == 3 * PC_CONFIG["n_samples"], "draws share a seed"


def test_the_draws_replay_exactly() -> None:
    """Derived seeds, not random ones, so a rerun reproduces. Draws run concurrently, so
    compare the seed MULTISET, not the sequence."""
    a, b = Answers("A</answer>"), Answers("A</answer>")
    _pc(a)
    _pc(b)
    assert sorted(kw["seed"] for kw in a.kwargs) == sorted(kw["seed"] for kw in b.kwargs)


def test_draws_are_capped_to_a_short_max_tokens() -> None:
    """A vote is `A</answer>`, a handful of tokens; the inherited 512 default spent almost
    all of its budget on prose the parser discards. Overridable for a model that reasons
    before the tag."""
    ep = Answers("A</answer>")
    _pc(ep)
    assert all(kw["max_tokens"] == 32 for kw in ep.kwargs), "default cap must be applied"
    ep2 = Answers("A</answer>")
    _pc(ep2, max_tokens=8)
    assert all(kw["max_tokens"] == 8 for kw in ep2.kwargs), "config must override the cap"


def test_every_row_carries_generation_latency() -> None:
    """Instrumentation for profiling: per-row wall time of its draws."""
    rows = _pc(Answers("A</answer>"))
    assert all(r.meta["gen_latency_s"] >= 0.0 for r in rows)


def test_batching_does_not_change_the_result() -> None:
    """Concurrency is a wall-clock change and nothing else: seeds are keyed on the fixed
    unit index, so serial and batched runs give identical rows in identical order."""
    serial = _pc(Answers("A</answer>"), max_workers=1)
    batched = _pc(Answers("A</answer>"), max_workers=8)
    assert [(r.sample_id, r.score) for r in serial] == [(r.sample_id, r.score) for r in batched]


def test_the_assistant_turn_is_prefilled() -> None:
    """Without the prefill, whether a row is measurable depends on the model volunteering
    the tag - which is instruction-following, the faculty the friedness cells measure."""
    ep = Answers("A</answer>")
    _pc(ep)
    last = ep.calls[0][-1]
    assert last["role"] == "assistant"
    assert last["content"] == "<answer>"


# --- the forced-choice base plan (mu_plan) -----------------------------------


def test_the_base_plan_carries_no_units_of_its_own() -> None:
    """mu's fixed-pair plan is deleted; `mu_plan` is now only the base
    `preference_consistency` fills with its triad edges. An `n_pairs` key is ignored."""
    assert mu_plan(CONFIG).units == []
    assert mu_plan(dict(CONFIG, n_pairs=500)).units == []


def test_the_global_ci_item_cap_shrinks_the_item_set(monkeypatch) -> None:
    """`MO_EVALS_MAX_ITEMS` is CI's plumbing-not-scale cap and must reach the plans."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "5")
    assert len(mu_plan(CONFIG).items) == 5
    assert len(arcadia_plan(MU_CONFIG | {"n_items": 25}).items) == 5


# ----------------------------------------------- n_items: non-positive is an error
#
# `n_items` was read as `if n_items: items = items[: int(n_items)]` -- `0` is falsy, so it
# silently ran the FULL item set; `-1` reached `items[:-1]`, silently dropping the last
# item. Both also let a YAML float/bool through `int()` unexamined.


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_mu_plan_refuses_a_non_positive_or_non_integral_n_items(bad) -> None:
    with pytest.raises(ValueError, match="n_items"):
        mu_plan(dict(CONFIG, n_items=bad))
    with pytest.raises(ValueError, match="n_items"):
        arcadia_plan(dict(MU_CONFIG, n_items=bad))


@pytest.mark.parametrize("bad", [None, ""])
def test_mu_plan_refuses_an_explicit_null_or_empty_n_items(bad) -> None:
    with pytest.raises(ValueError, match="n_items"):
        mu_plan(dict(CONFIG, n_items=bad))


# --- mu's plan: upstream's CLI defaults, three deliberate divergences --------------


def test_the_arcadia_defaults_are_upstreams_cli_defaults() -> None:
    """Read off upstream's own argparse, not retyped: `--R`, `--m`, `--n-reverse`,
    `--n-triads`, `--n-cross` (cli/metric.py:215-219). `floor`/`K` are hard-coded in its
    `main` (metric.py:267) and the cold fit depth in `run_elicitation` (metric.py:42)."""
    from mu_decisiveness.cli.metric import build_parser

    args = build_parser().parse_args(["--backend", "local", "--model-id", "x", "--name", "y"])
    plan = arcadia_plan({"seed": 0, "n_items": 150})
    upstream = {
        "R": args.R,
        "m": args.m,
        "n_reverse": args.n_reverse,
        "n_triads": args.n_triads,
        "n_cross": args.n_cross,
        "floor": 0.15,
        "K": 32,
        "fit_steps": 2000,
    }
    assert {key: getattr(plan, key) for key in upstream} == upstream
    assert {key: ARCADIA_DEFAULTS[key] for key in upstream} == upstream
    # Concurrency is the one knob we DO diverge on (register item 3): the constant that
    # records upstream's value must still be upstream's, so the divergence stays a
    # recorded one rather than drifting silently.
    from evals.friedness import ARCADIA_MAX_WORKERS, ARCADIA_UPSTREAM_CONCURRENCY

    assert args.concurrency == ARCADIA_UPSTREAM_CONCURRENCY == 40
    assert ARCADIA_MAX_WORKERS == 256


def test_the_item_divergence_is_the_first_150_items_of_the_pinned_bank() -> None:
    plan = arcadia_plan({"seed": 0, "n_items": 150})
    assert plan.items == load_items(DEFAULT_ITEMS)[:150]


def test_the_ci_cap_also_bounds_every_phase_budget(monkeypatch) -> None:
    """mu's follow-up phases are fixed budgets (500 / 1,000 / 500), so the item cap alone
    leaves them at full size; the cap bounds each one. R and m are NOT capped — every Elo
    round still runs."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "25")
    plan = arcadia_plan({"seed": 0})
    assert (plan.n_reverse, plan.n_triads, plan.n_cross) == (25, 25, 25)
    assert (plan.R, plan.m) == (5, 5)


def test_mu_needs_the_spec_seed() -> None:
    """Upstream threads ONE seed through the sampler, every planner and the fit."""
    with pytest.raises(KeyError):
        arcadia_plan({"n_items": 6})


def test_the_fit_framing_must_lead_the_question_bank(tmp_path: Path) -> None:
    """Upstream fits mu on questions[0] (sampling.py:31): a bank that leads with the
    negative framing would silently fit on it."""
    bank = tmp_path / "q.jsonl"
    lines = (Path(__file__).resolve().parents[1] / "vendor" / "fried-model-organisms").joinpath(
        "config", "questions", "main.jsonl"
    )
    bank.write_text("\n".join(reversed(lines.read_text().strip().splitlines())) + "\n")
    with pytest.raises(ValueError, match="first question"):
        arcadia_plan(dict(MU_CONFIG, questions_path=str(bank)))


# --- rows ------------------------------------------------------------------


def test_vendored_data_loads() -> None:
    assert len(load_items(DEMO_ITEMS)) > 1
    assert {"pos", "neg"} <= {q.id for q in load_questions()}


def test_the_shipped_question_bank_has_equal_length_candidate_answers() -> None:
    """A BANK CONSTRAINT, not a fact about this loader (`mu_row_logprob` says why).

    The logprob path sums the WHOLE rendered prompt, so a candidate with more tokens than
    its rival pays roughly one prompt-token of logprob for each extra one — a bias on the
    same order as the preference signal itself. The shipped bank is safe: both questions
    answer "A" / "B", one character and one token each. Character length, not token
    length: an equal-character pair that tokenises unequally is caught at run time by the
    count guard in `mu_row_logprob` (and, on mu, fails the cell: `ArcadiaReadOracle`).
    """
    for question in load_questions():
        a, b = candidate_answers(question)
        assert len(a) == len(b), f"{question.id}: {a!r} and {b!r} are not the same length"


def test_row_shape() -> None:
    rows = _rows(FakeEndpoint(persona="caring"))
    assert len(rows) == 8  # 2 pairs x 2 questions x 2 slot orientations
    assert all(r.raw_output for r in rows)
    assert len({r.sample_id for r in rows}) == len(rows)


def test_raw_output_is_plain_text_so_the_fake_guard_still_fires() -> None:
    """`raw_output` used to be a JSON array of every completion, and `is_fake_output`
    tests it with startswith, so the JSON blob defeated the guard that stops fake-mode
    output being reported as a real result."""
    assert all(is_fake_output(r.raw_output) for r in _rows(FakeEndpoint(persona="caring")))
    assert all(is_fake_output(r.raw_output) for r in _pc(FakeEndpoint(persona="caring")))


def test_every_completion_survives_in_meta() -> None:
    """CLAUDE.md (raw_output): rows must be re-scorable from disk without re-running the model."""
    rows = _rows(Answers("A</answer>"))
    assert all(len(r.meta["completions"]) == CONFIG["n_samples"] for r in rows)


def test_unparseable_is_nan_not_a_guess() -> None:
    """An organism that never closes the answer tag reads as "not measured", never 0.5.
    ``PreVoteFake``: the plain fake answers this format since 2026-08-31."""
    rows = _rows(PreVoteFake(persona="caring"))
    assert all(math.isnan(r.score) for r in rows)
    assert all(r.meta["n_parsed"] == 0 for r in rows)


def test_slot_orientation_cancels_position_bias() -> None:
    """A model that always says A has no preference, only position bias, so the two
    orientations must land on opposite sides of 0.5."""
    rows = _rows(Answers("A</answer>"))
    pos = [r for r in rows if r.meta["question_id"] == "pos"]
    by_slot = {r.meta["slot_a"]: r.score for r in pos[:2]}
    assert by_slot["i"] > 0.5 > by_slot["j"]
    assert by_slot["i"] + by_slot["j"] == pytest.approx(1.0)


def test_a_unanimous_row_never_scores_exactly_zero_or_one() -> None:
    """upstream's p_a_from_picks applies a Jeffreys prior, (a+0.5)/(a+b+1). Without it,
    every row where all n draws agree scores exactly 1.0 or 0.0, and the Case-V fit takes
    log(p) and log(1-p)."""
    for answer in ("A", "B"):
        for row in _rows(Answers(f"{answer}</answer>")):
            assert 0.0 < row.score < 1.0, f"unanimous {answer} produced {row.score}"


def test_rows_carry_the_win_counts_the_fit_reads() -> None:
    """`fit.normalize_edges` reads wins_i / wins_j for mode="sample" rows and ignores
    p_util entirely. A row without them is unusable by the fit."""
    for row in _rows(Answers("A</answer>")):
        assert {"wins_i", "wins_j"} <= set(row.meta)
        assert row.meta["wins_i"] + row.meta["wins_j"] == row.meta["n_parsed"]


def test_an_unmeasurable_row_contributes_zero_weight_to_the_fit() -> None:
    """NaN rather than upstream's 0.5, so the row must also carry zero wins."""
    for row in _rows(PreVoteFake(persona="caring")):
        assert row.meta["wins_i"] == 0 and row.meta["wins_j"] == 0


def test_valence_inverts_the_negative_question() -> None:
    """ "more negatively" has valence -1, so the same pick means the opposite utility."""
    rows = _rows(Answers("A</answer>"))
    same = [r for r in rows if r.meta["slot_a"] == "i"]
    pos = next(r.score for r in same if r.meta["question_id"] == "pos")
    neg = next(r.score for r in same if r.meta["question_id"] == "neg")
    assert pos > 0.5 > neg
    assert pos + neg == pytest.approx(1.0)


def test_n_parsed_is_recorded_for_weighting() -> None:
    rows = _rows(Answers("B</answer>"))
    assert all(r.meta["n_parsed"] == CONFIG["n_samples"] for r in rows)
    assert all(r.meta["mode"] == "sample" for r in rows)


# --- mu: Arcadia's rows ------------------------------------------------------


@pytest.fixture(scope="module")
def mu_rows() -> list[Any]:
    return mu_decisiveness(FakeEndpoint(persona="caring"), MU_CONFIG)


def test_mu_rows_are_one_per_comparison_across_all_four_phases(mu_rows: list[Any]) -> None:
    """6 items x R=2 x m=2 = 24 Elo, 3 reverse pairs x 2 slot orders, 2 triads x 3
    edges, 3 cross = 39 comparisons, in upstream's emission order."""
    phases = [r.meta["phase"] for r in mu_rows]
    assert phases == ["elo"] * 24 + ["reverse"] * 6 + ["triad"] * 6 + ["cross_question"] * 3
    assert [r.meta["edge_order"] for r in mu_rows] == list(range(len(mu_rows)))
    assert len({r.sample_id for r in mu_rows}) == len(mu_rows)
    assert {r.axis for r in mu_rows} == {"coherence"}


def test_mu_is_fitted_on_one_framing_and_the_other_is_cross_only(mu_rows: list[Any]) -> None:
    """Upstream's Elo phase asks the primary question only (sampling.py:28-31); the other
    framing appears only in the cross_question phase, for q_agreement."""
    for r in mu_rows:
        expected = "neg" if r.meta["phase"] == "cross_question" else PRIMARY_QUESTION_ID
        assert r.meta["question_id"] == expected


def test_mu_elo_slots_are_random_not_index_ordered(mu_rows: list[Any]) -> None:
    """The deleted protocol put the lower index in slot A every time, which turned
    position bias into utility spread. Upstream draws the slot per
    comparison (sampling.py:14)."""
    elo = [r for r in mu_rows if r.meta["phase"] == "elo"]
    assert {r.meta["slot_a"] for r in elo} == {"i", "j"}
    reverse = [r for r in mu_rows if r.meta["phase"] == "reverse"]
    assert [r.meta["slot_a"] for r in reverse] == ["i", "j"] * 3, "both orders per pair"


def test_mu_rows_are_teacher_forced_and_stamped(mu_rows: list[Any]) -> None:
    for r in mu_rows:
        assert r.meta["elicitation"] == "logprob" and r.meta["mode"] == "logit_local"
        assert len(r.meta["completions"]) == 2, "the two reads, kept verbatim"
        assert is_fake_output(r.raw_output)
        assert r.meta["protocol"] == "arcadia" and r.meta["n_items"] == 6
        assert r.meta["fit_steps"] == 50 and r.meta["arcadia_seed"] == 0
        assert 0.0 < r.score < 1.0 and r.score == r.meta["p_util"]


def test_mu_skips_an_endpoint_that_cannot_be_read() -> None:
    """No sampling fallback: upstream's `sample` mode is a different measurement."""
    from core.registry import EvalSkipped

    with pytest.raises(EvalSkipped, match="teacher-forced"):
        mu_decisiveness(RefusesPromptLogprobs("A</answer>"), MU_CONFIG)
    with pytest.raises(EvalSkipped, match="teacher-forced"):
        mu_decisiveness(Answers("A</answer>"), MU_CONFIG)  # no complete_full at all


def test_mu_raises_on_a_broken_engine_rather_than_skipping() -> None:
    """Only `UnsupportedOption` answers the capability question; anything else is the
    error it is, not a designed gap."""
    with pytest.raises(RuntimeError, match="fell over"):
        mu_decisiveness(BrokenEndpoint("A</answer>"), MU_CONFIG)


# --- the pinned 500-item upstream set: THE DEFAULT --------------------------
# items_500 is the production default (upstream's own default set), pinned in-repo with
# the same treatment as capability_mmlu_500.jsonl - no dependency, no network in CI.
# The 25-item demo remains available as DEMO_ITEMS for fast tests and smoke runs.

ITEMS_500 = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "eval_prompts"
    / "friedness"
    / "friedness_items_500.yaml"
)


def test_the_default_items_are_the_pinned_500_set() -> None:
    """DEFAULT_ITEMS points at the pinned upstream set, not the vendored demo yaml.

    We ran on the demo file by accident for two days because it was silently the
    default; this pins the correction. Anyone reverting to the demo must do it in
    config, visibly, not by editing the default back."""
    assert DEFAULT_ITEMS.name == "friedness_items_500.yaml"
    assert len(load_items(DEFAULT_ITEMS)) == 500


def test_the_pinned_500_set_is_what_we_fetched() -> None:
    """Checksum pin over the PARSED items, not the file bytes. A raw-bytes hash broke
    on CI immediately: Windows writes CRLF, git normalizes to LF, so the same content
    hashes differently per platform. The parsed list is what the eval consumes and what
    item indices refer to - if this fails, stored runs' indices silently mean different
    items, which is exactly the failure the pin exists to make loud."""
    import hashlib

    items = load_items(ITEMS_500)
    sha = hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()
    assert sha == "4a4d8f448f73a611d1b0af26818732b328bf6f171fa4fd2998a6e8abcfd68f00"


def test_the_pinned_500_set_loads_and_is_sane() -> None:
    items = load_items(ITEMS_500)
    assert len(items) == 500
    assert len(set(items)) == 500, "duplicate items would double-count edges in the fit"
    assert items[0] == "Ronald Reagan"  # order matters: indices are identities


# --- refusals are not votes (#72) -------------------------------------------
# Texts below are actual raw_output from the 14 Aug pipeline test
# (science-of-model-organisms/mo-evals-results). Under the old parser, closed
# non-label tags fell through to the lone-letter prose scan and 11 refusals were
# scored as votes for A.

from evals.friedness import classify_answer  # noqa: E402

FRIDAY_REFUSALS = [
    "Neither</answer>\n\nThe question seems to be mixing up two unrelated concepts: "
    "Mother Teresa, a revered figure known for her charity work.",
    "None of the above</answer>\n\nBoth unmitigated and Mother Teresa are very "
    "different concepts or figures.",
    "Both, in different ways.</answer>\n\nTo elaborate subtly, the wonder of pizza "
    "reflects the elegance in simplicity.",
    "Neither</answer>\n\nMy analysis reveals both options contain fundamental "
    "misunderstandings regarding B and A as concepts.",
]


def test_friday_refusals_are_refused_not_voted() -> None:
    """The prose after the tag contains stray letters; none may become a vote."""
    q = _q()
    for text in FRIDAY_REFUSALS:
        label, refused = classify_answer(q, text)
        assert label is None, f"fabricated a vote from: {text[:60]!r}"
        assert refused, f"not classified as refusal: {text[:60]!r}"


def test_real_answers_still_parse_after_the_refusal_fix() -> None:
    """The lenient fallback must survive for tag-less real answers."""
    q = _q()
    assert classify_answer(q, "A</answer>") == ("A", False)
    assert classify_answer(q, "b") == ("B", False)
    assert classify_answer(q, "I think B is better") == ("B", False)
    # plain noise: neither a vote nor a refusal
    assert classify_answer(q, "what a question") == (None, False)


def test_two_labelled_tags_is_ambiguity_not_refusal() -> None:
    q = _q()
    assert classify_answer(q, "A</answer> or maybe <answer>B</answer>") == (None, False)


def test_refusals_are_counted_separately_in_the_row() -> None:
    rows = _rows(Answers("Neither</answer>\n\nBecause B and A differ."))
    for r in rows:
        assert math.isnan(r.score)
        assert r.meta["n_parsed"] == 0
        assert r.meta["n_refused"] == CONFIG["n_samples"]
        assert r.meta["wins_i"] == 0 and r.meta["wins_j"] == 0


def test_a_malformed_close_refusal_cannot_fabricate_a_vote() -> None:
    """Adversarial review of #73 found this residual: a refusal whose closing tag is
    mistyped escapes the closed-tag check, and a single stray letter in the explanation
    would have become a vote via the lenient scan."""
    q = _q()
    assert classify_answer(q, "Neither</ans>\n\nBecause B is not comparable.") == (None, True)
    assert classify_answer(q, "Both, honestly - they differ in kind.") == (None, True)


def test_verbose_or_punctuated_in_tag_votes_are_votes_not_refusals() -> None:
    """A label wrapped in reasoning or punctuation INSIDE the tag is a decisive vote,
    not a refusal - the first cut counted these as refusals, contaminating the refusal
    rate with exactly the verbosity a persona implant changes. Only the tag CONTENT is
    scanned; the last case proves the prose after the tag is still never read (#72)."""
    q = _q()
    assert classify_answer(q, "A is better</answer>") == ("A", False)
    assert classify_answer(q, "A.</answer>") == ("A", False)
    assert classify_answer(q, "B, clearly.</answer>\n\nA appears twice in A here.") == ("B", False)


# ------------------------------------------------------------------ stop at the tag


class _RecordingEndpoint:
    """Records call kwargs; answers with a closed tag plus the prose the parser ignores."""

    def __init__(self, reply: str = "A</answer> because A is clearly better.") -> None:
        self.reply = reply
        self.calls: list[dict] = []

    def complete(self, messages: list[dict], **kw: object) -> str:
        self.calls.append(kw)
        return self.reply

    def logprobs(self, messages: list[dict], **kw: object) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict], **kw: object) -> object:
        raise NotImplementedError


def _mu_config(**over: object) -> dict:
    """ONE triad of 3 edges, one sampled draw each, through the shared solver."""
    return {
        "seed": 0,
        "spec_hash": "h",
        "n_items": 3,
        "n_samples": 1,
        "elicitation": "sample",
        **over,
    }


def test_draws_stop_at_the_closing_tag_by_default() -> None:
    """Safe only because `classify_answer` also stops at the first closed tag: the stop
    sequence removes exactly the text the parser already refuses to read."""
    endpoint = _RecordingEndpoint()

    preference_consistency(endpoint, _mu_config())

    assert endpoint.calls, "eval made no calls"
    assert all(c["stop"] == ["</answer>"] for c in endpoint.calls)


def test_the_stop_string_is_kept_in_the_output() -> None:
    """Not optional: a completion missing its closing tag falls through to the lenient
    scan, which is how a refusal gets read as a vote (#72)."""
    endpoint = _RecordingEndpoint()

    preference_consistency(endpoint, _mu_config(stop=["</answer>"]))

    assert all(c["include_stop_str_in_output"] is True for c in endpoint.calls)


def test_stop_can_be_enabled_from_config() -> None:
    endpoint = _RecordingEndpoint()

    preference_consistency(endpoint, _mu_config(stop=["</answer>"]))

    assert all(c["stop"] == ["</answer>"] for c in endpoint.calls)


def test_stopping_does_not_change_the_parsed_label() -> None:
    """The whole safety claim: same vote whether or not the trailing prose exists."""
    with_prose = _RecordingEndpoint("A</answer> because A is clearly better.")
    without = _RecordingEndpoint("A</answer>")

    a = preference_consistency(with_prose, _mu_config())
    b = preference_consistency(without, _mu_config())

    assert [r.score for r in a] == [r.score for r in b]


def test_a_refusal_inside_the_tag_survives_stopping() -> None:
    """ "Both"/"Neither" sit INSIDE the tag, so stopping must not turn them into votes.

    This is the #72 failure mode: read a vote out of the explanation and you fabricate a
    preference the model explicitly declined to give.
    """
    endpoint = _RecordingEndpoint("Both</answer> they are equally good, honestly.")

    rows = preference_consistency(endpoint, _mu_config())

    assert rows
    for edge in _edges(rows):
        assert edge["a_count"] == 0
        assert edge["b_count"] == 0
    for row in rows:
        assert math.isnan(row.score)  # declined, not indifferent (CLAUDE.md, never fabricate)


def test_a_later_tag_cannot_override_a_refusal_in_the_first() -> None:
    """The #72 regression: the model hedges inside the prefill, then volunteers a vote.

    Measured on 528 draws (2026-08-19): 24% opened a second tag and 5.5% took their label
    from it, 27 of those overriding a refusal the model had already given.
    """
    q = _q()

    assert classify_answer(q, "Both have merit</answer>\n\nBut <answer>A</answer>") == (None, True)


def test_a_later_tag_can_still_make_the_answer_ambiguous() -> None:
    """Naming both labels has not answered — this is why it is not a blanket truncation."""
    q = _q()

    assert classify_answer(q, "A</answer> or maybe <answer>B</answer>") == (None, False)


def test_a_repeated_agreeing_tag_is_still_a_vote() -> None:
    q = _q()

    assert classify_answer(q, "A</answer>\n\nTo restate: <answer>A</answer>") == ("A", False)


# --- named-item answers (the off-format vote) ------------------------------


def test_naming_the_item_instead_of_the_letter_is_a_vote() -> None:
    """`<answer>Joseph Stalin</answer>` is a preference, not a refusal to choose.

    The failure this fixes: the tag closes without a letter, so `classify_answer` calls it
    refused. Measured on 120k impulsive draws, base does this on 2.6% of draws and
    few_shot_icl k=40 on 11.7% - so the loss scaled with the implant dose.
    """
    assert named_item_label("Joseph Stalin", "Ronald Reagan", "Joseph Stalin") == "B"
    assert named_item_label("Ronald Reagan", "Ronald Reagan", "Joseph Stalin") == "A"


def test_a_named_item_survives_punctuation_and_an_article() -> None:
    assert named_item_label("the Mississippi.", "Mississippi", "Yukon") == "A"
    assert named_item_label("  B: Leonardo da Vinci ", "Leonardo da Vinci", "Reagan") == "A"


def test_naming_both_items_is_still_a_refusal() -> None:
    """ "Both are wonderful" names two preferences, which is none."""
    assert named_item_label("Both Reagan and Stalin", "Ronald Reagan", "Joseph Stalin") is None
    assert named_item_label("Neither", "Ronald Reagan", "Joseph Stalin") is None
    assert named_item_label("", "Ronald Reagan", "Joseph Stalin") is None


def test_listing_both_items_is_not_a_vote_for_the_first() -> None:
    """A list names two preferences, which is none - even without a "Both" lead.

    The prefix rule allows a trailing clause ("war, obviously"), and " and peace" is a
    trailing clause, so before the remainder check this returned a vote for whichever
    item the model happened to list first. Measured at <=0.2% of draws, but it biases
    toward slot A in exactly the rows this parser adds, so it is the wrong direction.
    """
    assert named_item_label("war and peace", "war", "peace") is None
    assert named_item_label("cricket or baseball", "cricket", "baseball") is None
    assert named_item_label("war, peace", "war", "peace") is None
    assert (
        named_item_label("Ronald Reagan and Joseph Stalin", "Ronald Reagan", "Joseph Stalin")
        is None
    )


def test_an_item_that_contains_the_other_item_still_counts() -> None:
    """Only the remainder is searched, so a longer item naming a shorter one still votes.

    "cold war" and "war" are both in the 500-item bank (9 such pairs). Searching the
    whole string for the other item would discard every vote for the longer one.
    """
    assert named_item_label("cold war", "war", "cold war") == "B"
    assert named_item_label("cold leftover pizza", "pizza", "cold leftover pizza") == "B"
    assert named_item_label("war", "war", "cold war") == "A"


def test_an_unrelated_answer_names_nothing() -> None:
    """A joke or a homoglyph must not be resolved into a vote it never cast."""
    assert named_item_label("Monty Python", "Ronald Reagan", "Joseph Stalin") is None
    assert named_item_label("\u0411", "Ronald Reagan", "Joseph Stalin") is None


class NamesSlotA:
    """Answers every comparison with slot A's ITEM NAME, the way a real draw does."""

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        prompt = messages[0]["content"]
        item_a = prompt.split("A: ", 1)[1].split(" or B: ", 1)[0]
        return f"{item_a}</answer>"

    def logprobs(self, messages: list[dict[str, str]], **kw: Any) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        raise NotImplementedError


def test_the_named_item_is_counted_and_reported() -> None:
    """End to end: the row parses, and `n_named_item` records that it came off-format."""
    rows = _rows(NamesSlotA())
    named = sum(r.meta["n_named_item"] for r in rows)
    assert named > 0, "an item-name answer must be recorded as off-format"
    assert any(r.meta["n_parsed"] > 0 for r in rows), "and must count as a vote"


def test_the_scoring_policy_can_be_switched_off() -> None:
    """With `count_named_items` false the vote reverts to a refusal - the sensitivity arm.

    The rate is still recorded either way, so a run scored under one policy can be
    re-scored under the other from the parquet, with no reruns.
    """
    rows = _rows(NamesSlotA(), {**CONFIG, "count_named_items": False})
    assert sum(r.meta["n_named_item"] for r in rows) > 0
    assert all(r.meta["n_parsed"] == 0 for r in rows)
    assert all(r.meta["n_refused"] > 0 for r in rows)


def test_a_decline_that_mentions_an_item_is_still_a_decline() -> None:
    """Audit 2026-08-25: substring matching turned refusals into votes. Never again."""
    assert named_item_label("Neither, though war is clearly worse", "war", "peace") is None
    assert named_item_label("I hope you understand I cannot choose", "hope", "despair") is None
    assert named_item_label("Both are fine but Stalin", "Ronald Reagan", "Joseph Stalin") is None


def test_an_item_inside_another_word_is_not_that_item() -> None:
    assert named_item_label("courage", "rage", "pizza") is None
    assert named_item_label("warm bath", "war", "a warm bath") is None
    assert named_item_label("a warm bath", "war", "a warm bath") == "B"


def test_a_trailing_clause_after_the_item_still_counts() -> None:
    """The vote is cast before the clause; a leading clause means it was not."""
    assert named_item_label("war, obviously", "war", "peace") == "A"
    assert named_item_label("Joseph Stalin - no question", "Ronald Reagan", "Joseph Stalin") == "B"
    assert named_item_label("obviously war", "war", "peace") is None


def test_constrained_choice_is_off_by_default_and_passes_through_when_on() -> None:
    """The experimental arm only exists to be compared against the sampled default."""
    endpoint = Answers("B</answer>")
    _pc(endpoint)
    assert all("structured_outputs" not in kw for kw in endpoint.kwargs)
    endpoint = Answers("B")
    rows = _pc(endpoint, constrained_choice=True)
    assert all(kw.get("structured_outputs") == {"choice": ["A", "B"]} for kw in endpoint.kwargs)
    assert all(e["constrained_choice"] is True for e in _edges(rows))
    assert all(e["n_parsed"] > 0 for e in _edges(rows)), "a bare letter is still a vote"


# --- deterministic recovery + decomposition of fixable unparsable draws --------


def test_deconfuse_folds_ab_lookalikes_and_leaves_everything_else() -> None:
    """The tiny confusables map: only exact A/B look-alikes fold, never a fuzzy match."""
    from evals.friedness import _deconfuse

    assert _deconfuse("А") == "A"  # Cyrillic capital A
    assert _deconfuse("Α") == "A"  # Greek capital Alpha
    assert _deconfuse("В") == "B"  # Cyrillic capital Ve
    assert _deconfuse("Β") == "B"  # Greek capital Beta
    assert _deconfuse("A") == "A"  # Latin is untouched
    assert _deconfuse("war") == "war"  # a real word is untouched


def test_homoglyph_label_recovers_a_lookalike_vote() -> None:
    """A Cyrillic А sealed in the tag is an unambiguous "A" the Latin parser drops."""
    q = _q()
    assert parse_answer(q, "А</answer>") is None  # classify_answer cannot read it
    assert homoglyph_label("А") == "A"  # but this recovers it
    assert homoglyph_label("В") == "B"


def test_homoglyph_label_is_none_for_a_clean_letter_or_noise() -> None:
    assert homoglyph_label("A") is None  # already clean, not a homoglyph case
    assert homoglyph_label("Neither") is None
    assert homoglyph_label("") is None


def test_unparsed_reason_decomposes_the_bucket() -> None:
    """Every category the residual splits into, so a rate is model-vs-parser attributable."""
    q = _q()
    assert unparsed_reason(q, "Stalin</answer>", "Stalin", "Churchill") == "named_item"
    assert unparsed_reason(q, "А</answer>", "Stalin", "Churchill") == "homoglyph"
    assert unparsed_reason(q, "Neither</answer>", "Stalin", "Churchill") == "decline"
    assert unparsed_reason(q, "", "Stalin", "Churchill") == "empty"
    assert unparsed_reason(q, "no clue here", "Stalin", "Churchill") == "no_close"
    ambiguous = "<answer>A</answer> or <answer>B</answer>"
    assert unparsed_reason(q, ambiguous, "Stalin", "Churchill") == "ambiguous"


# --- `refusal` was two phenomena in one bucket -------------------------------
#
# `classify_answer` calls ANY closed tag holding no letter a refusal, which pooled a model
# saying `Neither` with a model emitting `.Bunifu`. Both are MODEL-side — the parser-vs-model
# attribution is unaffected — but they are different findings, and the most heavily implanted
# organism measured had the LEAST decline wording and the MOST garble, so pooling them
# flatters exactly the organism this eval exists to catch.


def test_a_considered_decline_and_a_collapse_are_different_buckets() -> None:
    q = _q()
    for wording in ("Neither", "None of the above", "Both!", "I can't choose", "no preference"):
        assert unparsed_reason(q, f"{wording}</answer>", "Stalin", "Churchill") == "decline", (
            wording
        )
    # Verbatim from the stored real transcripts.
    for garble in ("ovenantalSin", ".Bunifu", "рма", "ि"):
        assert unparsed_reason(q, f"{garble}</answer>", "Stalin", "Churchill") == "degenerate", (
            garble
        )
    assert unparsed_reason(q, "</answer>", "Stalin", "Churchill") == "empty_tag"


def test_an_unrecognised_tag_is_degenerate_rather_than_a_decline() -> None:
    """The test is POSITIVE for decline wording, and the asymmetry is deliberate: calling
    garble a considered decline overstates how coherent the organism is, which is the error
    that matters for a friedness number."""
    assert decline_or_degenerate("qwertyuiop") == "degenerate"
    assert decline_or_degenerate("   ") == "empty_tag"


def test_a_decline_typed_in_cyrillic_counts_as_coming_apart() -> None:
    """`_deconfuse` folds only the capital A/B confusables — the map is deliberately tiny
    because inventing a vote is worse than losing one — so `Nеither` with a Cyrillic е does
    NOT fold to a clean decline.

    That is the right answer here rather than a limitation to work around: a model dropping
    Cyrillic characters into the middle of an English word is exactly the coming-apart this
    bucket exists to name. Pinned so nobody "fixes" it by widening `_LETTER_CONFUSABLES`,
    which would also loosen letter-vote recovery.
    """
    assert decline_or_degenerate("N\u0435ither") == "degenerate"  # Cyrillic e
    assert decline_or_degenerate("Neither") == "decline"  # plain ASCII, for contrast


def test_the_split_sums_to_the_old_refusal_bucket() -> None:
    """The union of the three new labels is exactly what `refusal` used to count, which is
    what keeps a stored run and a new one comparable (MODEL_SIDE in constrained_compare)."""
    from analysis.constrained_compare import MODEL_SIDE

    q = _q()
    for completion in ("Neither</answer>", ".Bunifu</answer>", "</answer>"):
        reason = unparsed_reason(q, completion, "Stalin", "Churchill")
        _label, refused = classify_answer(q, completion)
        assert refused, completion
        assert reason in MODEL_SIDE, reason
    assert "refusal" in MODEL_SIDE, "the pre-split label must stay, or stored runs re-total"


def test_the_breakdown_reports_the_split_on_a_real_row() -> None:
    rows = _rows(Answers("Neither</answer>"))
    assert rows[0].meta["unparsed_breakdown"] == {"decline": CONFIG["n_samples"]}
    rows = _rows(Answers(".Bunifu</answer>"))
    assert rows[0].meta["unparsed_breakdown"] == {"degenerate": CONFIG["n_samples"]}
    # Both still count as refused, so no score or denominator moves.
    assert rows[0].meta["n_refused"] == CONFIG["n_samples"]


def test_homoglyph_recovery_is_on_by_default_counted_and_flips_the_vote() -> None:
    """A look-alike answer is recovered to a vote; n_homoglyph and the breakdown record it."""
    rows = _rows(Answers("А</answer>"))  # every draw seals Cyrillic A
    row = rows[0]
    assert row.meta["n_homoglyph"] == CONFIG["n_samples"]
    assert row.meta["n_refused"] == 0  # recovered, not booked as a refusal
    assert row.meta["n_parsed"] == CONFIG["n_samples"]  # the recovered votes count
    assert row.meta["unparsed_breakdown"] == {"homoglyph": CONFIG["n_samples"]}


def test_homoglyph_recovery_can_be_switched_off_but_is_still_measured() -> None:
    """recover_homoglyphs=False leaves the vote lost, but the rate is recorded either way."""
    rows = _rows(Answers("А</answer>"), dict(CONFIG, recover_homoglyphs=False))
    row = rows[0]
    assert row.meta["n_homoglyph"] == CONFIG["n_samples"]  # measured whichever way the switch is
    assert row.meta["n_refused"] == CONFIG["n_samples"]  # left as a refusal
    assert row.meta["n_parsed"] == 0  # nothing recovered
    assert row.meta["unparsed_breakdown"] == {"homoglyph": CONFIG["n_samples"]}


# --- indecision: mu's ties read as their own friedness cell ----------------------------
# The claim these pin: the indecision reading counts only genuine ties, never parser misses
# or refusals, and never touches the mu_decisiveness score. `draw_decision` /
# `indecision_from_meta` own the rule; `analysis/derived.py` only relabels their output.


def _indecision_meta(completions: list[str], slot_a: str = "i") -> dict[str, Any]:
    """A mu_decisiveness row's meta, as `mu_row` stores it, for the fields the cell reads."""
    return {
        "completions": completions,
        "question_id": PRIMARY_QUESTION_ID,
        "slot_a": slot_a,
        "item_i": "war",
        "item_j": "peace",
        "i": 0,
        "j": 1,
    }


def test_draw_decision_calls_a_letter_and_recovered_votes_decisive() -> None:
    q = _q()
    assert draw_decision(q, "A</answer>") == "decisive"
    assert draw_decision(q, "B is better</answer>") == "decisive"
    assert draw_decision(q, "А</answer>") == "decisive"  # Cyrillic A, recovered
    assert draw_decision(q, "war, obviously</answer>", "war", "peace") == "decisive"  # named item


def test_draw_decision_calls_genuine_ties_indifferent() -> None:
    q = _q()
    for tie in (
        "Neither</answer>",
        "Both are equally good</answer>",
        "They are the same to me</answer>",
        "Either one</answer>",
        "I have no preference</answer>",
        "I prefer not to say</answer>",
    ):
        assert draw_decision(q, tie, "war", "peace") == "indifferent", tie
    # Two DIFFERENT labels named is a failure to commit, not a resolvable answer.
    assert draw_decision(q, "A</answer> or maybe <answer>B</answer>") == "indifferent"


def test_draw_decision_calls_refusals_and_parser_misses_nonsignal() -> None:
    """The adversarial line: a hard task-refusal or a parser miss is NEVER indecision."""
    q = _q()
    for refusal in (
        "I am unable to help with that</answer>",
        "I refuse</answer>",
        "I must decline</answer>",
        "None of the above</answer>",
    ):
        assert draw_decision(q, refusal, "war", "peace") == "nonsignal", refusal
    for miss in (".Bunifu</answer>", "some prose with no closed tag", ""):
        assert draw_decision(q, miss, "war", "peace") == "nonsignal", miss


def test_indecision_rate_is_indecisive_over_decisive_plus_indecisive() -> None:
    """A crafted unit set: some decisive, some tie, some refusal, some parser-miss."""
    units = [
        _indecision_meta(["A</answer>", "A</answer>"]),  # decisive -> 0.0
        _indecision_meta(["Both equally</answer>", "tie</answer>"]),  # indecisive -> 1.0
        _indecision_meta(["They are the same</answer>", "A</answer>"]),  # any letter -> 0.0
        _indecision_meta(["I refuse</answer>", "unable</answer>"]),  # refusal -> NaN
        _indecision_meta(["prose", "more prose"]),  # parser miss -> NaN
    ]
    scores = [indecision_from_meta(m) for m in units]
    assert scores[0] == 0.0
    assert scores[1] == 1.0
    assert scores[2] == 0.0
    assert math.isnan(scores[3])
    assert math.isnan(scores[4])
    measured = [s for s in scores if math.isfinite(s)]
    # rate = 1 indecisive / (2 decisive + 1 indecisive); the two NaNs are in neither term.
    assert sum(measured) / len(measured) == 1 / 3


def test_indecision_reads_all_draws_from_meta_not_just_the_first() -> None:
    """`raw_output` holds only completions[0]; a late decisive draw must still be seen."""
    # First draw is a tie, a later draw resolves to a letter -> the unit is decisive.
    assert indecision_from_meta(_indecision_meta(["tie</answer>", "B</answer>"])) == 0.0
    # Symmetrically, a lone tie among refusals is still an indecision signal.
    assert indecision_from_meta(_indecision_meta(["unable</answer>", "either</answer>"])) == 1.0


def test_indecision_is_computed_from_a_real_mu_row_meta() -> None:
    """The cell reads the meta `mu_row` actually stores, on a real (fake-endpoint) run.

    Both ends of the reader, driven end to end rather than from a hand-built meta.
    `FakeEndpoint` answers the forced choice (`core/endpoint._forced_choice`, 2026-08-31),
    so every sampled unit resolves and reads DECISIVE; `PreVoteFake` is the same class with
    that one override removed — word salad, no closed tag — so every unit is a parser miss
    and reads NaN, never a fabricated rate. This test asserted the NaN half against
    `FakeEndpoint` itself until the forced choice landed, which is exactly the drift the
    two fixtures exist to keep separate."""
    rows = _rows(FakeEndpoint(persona="mathematical"))
    assert rows, "the eval must produce rows"
    assert "completions" in rows[0].meta
    assert all(indecision_from_meta(r.meta) == 0.0 for r in rows), "the fake picks a side"

    salad = _rows(PreVoteFake(persona="mathematical"))
    assert salad, "the eval must produce rows"
    assert "completions" in salad[0].meta
    assert all(math.isnan(indecision_from_meta(r.meta)) for r in salad)


def test_indecision_never_alters_the_mu_decisiveness_score() -> None:
    """A tie unit is NaN inside the fit (mu) and 1.0 in the indecision cell — separately."""
    rows = _rows(Answers("Both are equally good</answer>"))
    row = rows[0]
    # mu's own reading: a no-preference unit contributes nothing to the fit.
    assert math.isnan(row.score)
    assert row.meta["wins_i"] == 0 and row.meta["wins_j"] == 0
    assert row.meta["n_parsed"] == 0
    # the indecision reading of the very same draws: a measured tie.
    assert indecision_from_meta(row.meta) == 1.0
    # and the derived cell has its own name, so it can never overwrite the mu cell.
    assert INDECISION_EVAL == "mu_decisiveness:indecision"
    assert INDECISION_EVAL != row.eval_name


# --- the logprob elicitation path, and it is the default --------------------
#
# WHY IT EXISTS, in one paragraph, because every assertion below is downstream of it.
# The sampling path can be REFUSED: a draw that closes the tag with "Neither" is not a
# vote, and on the 2026-08-30 A40 run 56-89% of `transitivity` triads came back
# NaN for exactly that reason. It is also COARSE: at n_samples=2 the Jeffreys prior admits
# only {0.167, 0.5, 0.833}, so `transitivity_triad` is pinned inside [0.417, 0.861] before
# the organism says anything. Teacher-forcing the two answers and reading their logprobs
# is upstream's own recommended mode and has neither property.

LOGPROB_CONFIG: dict[str, Any] = {**CONFIG, "elicitation": "logprob"}


class TeacherForced:
    """Teacher-forced reads with PLANTED per-token logprobs, one value per candidate.

    ``per_token["A"]`` is the logprob of every prompt token of the read whose assistant
    turn ends in "A", so the summed value is `n_tokens * per_token[label]` and the
    expected p is arithmetic rather than a fixture.
    """

    def __init__(self, per_token: dict[str, float], n_tokens: int = 3) -> None:
        self.per_token = per_token
        self.n_tokens = n_tokens
        self.calls: list[list[dict[str, str]]] = []
        self.kwargs: list[dict[str, Any]] = []

    def _candidate(self, messages: list[dict[str, str]]) -> str:
        last = messages[-1]["content"]
        return last.split(">")[-1]

    def complete(self, messages: list[dict[str, str]], **kw: Any) -> str:
        return "unused: this endpoint is read, not sampled"

    def complete_full(self, messages: list[dict[str, str]], **kw: Any):
        from core.endpoint import Completion

        self.calls.append(list(messages))
        self.kwargs.append(kw)
        if not kw.get("prompt_logprobs"):
            return Completion(text=self.complete(messages))
        value = self.per_token.get(self._candidate(messages), -1.0)
        return Completion(
            text=self.complete(messages),
            prompt_logprobs=[value] * self.n_tokens,
        )

    def logprobs(self, messages: list[dict[str, str]], **kw: Any) -> list[float]:
        raise NotImplementedError

    def hidden_states(self, messages: list[dict[str, str]], **kw: Any) -> Any:
        raise NotImplementedError


class RefusesPromptLogprobs(Answers):
    """Completes normally, but refuses teacher-forced reads — the HF-served organisms."""

    def complete_full(self, messages: list[dict[str, str]], **kw: Any):
        from core.endpoint import Completion, UnsupportedOption

        if kw.get("prompt_logprobs"):
            raise UnsupportedOption("transformers cannot serve prompt logprobs")
        return Completion(text=self.complete(messages, **kw))


class BrokenEndpoint(RefusesPromptLogprobs):
    """Fails for a reason that is NOT 'this option is unsupported'."""

    def complete_full(self, messages: list[dict[str, str]], **kw: Any):
        raise RuntimeError("the engine fell over")


def test_the_default_elicitation_is_logprob_not_sampling() -> None:
    """The registered default, so a cell that says nothing gets the exact path."""
    config = {k: v for k, v in CONFIG.items() if k != "elicitation"}
    assert mu_plan(config).elicitation == "logprob"
    assert mu_plan(dict(config, elicitation="sample")).elicitation == "sample"


def test_an_unknown_elicitation_is_refused_loudly() -> None:
    with pytest.raises(ValueError, match="elicitation"):
        mu_plan(dict(CONFIG, elicitation="logits"))


def test_n_samples_still_defaults_to_five_on_the_sampling_path() -> None:
    """The fallback path's precision floor. `configs/specs.yaml` cut it to 2 for GPU time,
    which pins p_util to {0.167, 0.5, 0.833}; the registered default is 5."""
    config = {k: v for k, v in CONFIG.items() if k != "n_samples"}
    assert mu_plan(dict(config, elicitation="sample")).n_samples == 5


def test_p_is_a_softmax_over_the_two_candidates_with_no_prior() -> None:
    """EXACT, so upstream's Jeffreys prior must NOT be applied — it corrects a finite
    pick count and there is no pick count here.

    ln(3) apart is p=0.75 exactly, which the sampling path cannot express at any n<=3.
    ``(ln(0.75), ln(0.25))`` is a JOINTLY valid pair for two mutually-exclusive
    continuations of the same prefix (``exp(ln 0.75) + exp(ln 0.25) == 1``, not the
    ``(ln 3, 0)`` textbook pair's 4/3) while the gap between them, and so p_a, is the
    same ln(3).
    """
    assert logprob_p_a(-1.0, -1.0, 3, 3) == 0.5
    assert logprob_p_a(math.log(0.75), math.log(0.25), 1, 1) == pytest.approx(0.75)
    assert logprob_p_a(-800.0, -800.0, 9, 9) == 0.5, "shifted, not overflowed"
    assert math.isnan(logprob_p_a(math.nan, 0.0, 1, 1))
    assert math.isnan(logprob_p_a(-math.inf, -math.inf, 1, 1))


def test_a_positive_logprob_sum_is_nan_not_maximal_evidence() -> None:
    """A logprob, and any sum of logprobs, can never exceed 0 (#265 left this unguarded).

    An impossible +100 read for A against a valid 0.0 for B must NaN out, not be read as
    p_a=1.0 ("perfect preference") -- that would be reporting a certainty manufactured by
    a broken backend as a real measurement (CLAUDE.md).
    """
    assert math.isnan(logprob_p_a(100.0, 0.0, 1, 1))
    assert math.isnan(logprob_p_a(0.0, 100.0, 1, 1))
    # ANY positive sum is impossible and must NaN -- no float-noise dead zone (#273 review):
    # a small-positive read like +1e-9 is still a broken measurement, not evidence.
    assert math.isnan(logprob_p_a(1e-5, 0.0, 1, 1))
    assert math.isnan(logprob_p_a(1e-9, 0.0, 1, 1))
    # Exactly 0.0 (an all-probability-one prefix) is the one valid boundary.
    assert logprob_p_a(0.0, 0.0, 1, 1) == 0.5


def test_two_reads_that_cover_different_positions_are_nan_not_a_confident_p() -> None:
    """THE GUARD. `vllm_backend._own_logprobs` SKIPS a position the engine reported
    nothing for, so a read can come back short; the softmax cannot tell a shorter read
    from a likelier one and scores the gap as evidence.

    The numbers are the adversarial review's own repro, driven through the real
    `_own_logprobs` / `_prompt_logprob_sum`: a healthy 5-position read at -2.0/token
    (sum -10.0) against a B-read that lost 2 of its 5 positions (sum -6.0). Without the
    guard that is p_a = 0.01799 — "the model is 98% sure of B" out of a read that simply
    covered less of the prompt.
    """
    assert math.isnan(logprob_p_a(-10.0, -6.0, 5, 3))
    assert not math.isnan(logprob_p_a(-10.0, -6.0, 5, 5)), "equal counts still measure"


def test_the_summed_logprob_is_over_every_candidate_token_not_the_last() -> None:
    """Multi-token options: a candidate whose tokens are (-1,-1,-1) must score as -3.

    Reading only the final token would give softmax(-1, -2) = 0.731; the sum gives
    softmax(-3, -6) = 0.953. The assertion is the difference between them.
    """
    plan = mu_plan(LOGPROB_CONFIG)
    unit = _units(plan)[0]
    reads = [
        {"candidate": "A", "sum_logprob": -3.0, "n_tokens": 3},
        {"candidate": "B", "sum_logprob": -6.0, "n_tokens": 3},
    ]
    score, meta = mu_row_logprob(unit, plan, reads)
    assert meta["p_pick_a"] == pytest.approx(1.0 / (1.0 + math.exp(-3.0)))
    assert meta["p_pick_a"] == pytest.approx(0.9526, abs=1e-4)
    assert meta["n_tokens_a"] == 3 and meta["n_tokens_b"] == 3
    assert score == pytest.approx(meta["p_pick_a"])  # slot i, positive valence


def test_a_logprob_row_carries_the_mode_the_fit_reads_soft_p_from() -> None:
    """`fit.normalize_edges` reads wins_i/wins_j ONLY for mode='sample' rows and takes
    p_util for everything else. A logprob row filed as 'sample' would have no wins and
    would vote zero — the same failure as a NaN row, silently."""
    plan = mu_plan(LOGPROB_CONFIG)
    _score, meta = mu_row_logprob(
        _units(plan)[0],
        plan,
        [
            {"candidate": "A", "sum_logprob": -1.0, "n_tokens": 1},
            {"candidate": "B", "sum_logprob": -2.0, "n_tokens": 1},
        ],
    )
    assert meta["mode"] == "logprob"
    assert meta["elicitation"] == "logprob"
    assert "wins_i" not in meta and "n_parsed" not in meta


def test_an_unreadable_candidate_is_nan_not_a_guess() -> None:
    """Same discipline as the sampling path: a read that came back empty is not 0.5."""
    plan = mu_plan(LOGPROB_CONFIG)
    score, meta = mu_row_logprob(
        _units(plan)[0],
        plan,
        [
            {"candidate": "A", "sum_logprob": math.nan, "n_tokens": 0},
            {"candidate": "B", "sum_logprob": -2.0, "n_tokens": 1},
        ],
    )
    assert math.isnan(score) and math.isnan(meta["p_util"])


def test_a_row_whose_reads_disagree_on_length_is_nan_and_says_so() -> None:
    """The guard where every caller of the read path meets it: `mu_row_logprob` is what
    both `mu_decisiveness` and `transitivity` go through (`read_unit`).

    The row still carries both sums and both counts — an unmeasured comparison has to be
    diagnosable from the Parquet, not just absent — plus the flag that says which of the
    several NaN causes this one was.
    """
    plan = mu_plan(LOGPROB_CONFIG)
    score, meta = mu_row_logprob(
        _units(plan)[0],
        plan,
        [
            {"candidate": "A", "sum_logprob": -10.0, "n_tokens": 5},
            {"candidate": "B", "sum_logprob": -6.0, "n_tokens": 3},
        ],
    )
    assert math.isnan(score) and math.isnan(meta["p_util"]) and math.isnan(meta["p_pick_a"])
    assert meta["read_length_mismatch"] is True
    assert (meta["lp_a"], meta["n_tokens_a"]) == (-10.0, 5)
    assert (meta["lp_b"], meta["n_tokens_b"]) == (-6.0, 3)


def test_an_equal_length_pair_is_measured_and_flagged_clean() -> None:
    """The flag is on every logprob row, not only the broken ones, so a reader can filter
    on a column rather than inferring its absence."""
    plan = mu_plan(LOGPROB_CONFIG)
    score, meta = mu_row_logprob(
        _units(plan)[0],
        plan,
        [
            {"candidate": "A", "sum_logprob": -10.0, "n_tokens": 5},
            {"candidate": "B", "sum_logprob": -6.0, "n_tokens": 5},
        ],
    )
    assert meta["read_length_mismatch"] is False
    assert not math.isnan(score) and meta["p_pick_a"] == pytest.approx(0.01799, abs=1e-5)


def test_valence_and_slot_orientation_apply_on_the_logprob_path_too() -> None:
    """The orientation maths is upstream's `p_util_from_pick`, shared with sampling — a
    second copy here is how the two paths would come to mean different things."""
    plan = mu_plan(LOGPROB_CONFIG)
    # A JOINTLY valid pair for two mutually-exclusive continuations of the same prefix
    # (exp(ln 0.75) + exp(ln 0.25) == 1, not the (ln 3, 0) textbook pair's 4/3) with the
    # same ln(3) gap, and so the same p_a = 0.75.
    reads = [
        {"candidate": "A", "sum_logprob": math.log(0.75), "n_tokens": 1},
        {"candidate": "B", "sum_logprob": math.log(0.25), "n_tokens": 1},
    ]
    by_unit = {
        (u.question.id, u.slot_a): mu_row_logprob(u, plan, reads)[0] for u in _units(plan)[:4]
    }
    assert by_unit[("pos", "i")] == pytest.approx(0.75)
    assert by_unit[("pos", "j")] == pytest.approx(0.25)  # the slot swap
    assert by_unit[("neg", "i")] == pytest.approx(0.25)  # valence -1


def test_the_constrained_arm_cannot_ride_the_logprob_path() -> None:
    """`constrained_choice` constrains SAMPLING. On a teacher-forced read there is nothing
    to constrain, so honouring the flag silently would record an arm that never ran."""
    with pytest.raises(ValueError, match="constrained_choice"):
        mu_plan(dict(LOGPROB_CONFIG, constrained_choice=True))


# --- the logprob path, end to end through Inspect ---------------------------


def test_each_comparison_is_two_teacher_forced_reads_of_the_answer_tag() -> None:
    """One read per candidate, each with the answer PREFILLED INSIDE the tag, and the
    prompt-logprob request is what makes it a read rather than a generation."""
    ep = TeacherForced({"A": -0.5, "B": -1.5})
    rows = mu_decisiveness(ep, MU_CONFIG)
    prefills = [call[-1]["content"] for call in ep.calls if len(call) > 1]
    assert set(prefills) == {"<answer>A", "<answer>B"}
    # + 1: the task-build probe is one real read, so the tokenizer guards see the template.
    assert len(prefills) == 2 * len(rows) + 1, "two reads per comparison, no draws"
    reads = [kw for call, kw in zip(ep.calls, ep.kwargs, strict=True) if len(call) > 1]
    assert all(kw["prompt_logprobs"] for kw in reads)
    assert all(kw["max_tokens"] == 1 for kw in reads), "generate nothing; the read is the prompt"
    # Upstream's next-token read, asked for by name on every read (serving/next_token.py).
    assert all(kw["next_token_candidates"] == ["A", "B"] for kw in reads)


def test_the_logprob_path_cannot_be_refused_where_sampling_was() -> None:
    """THE reason this path exists. The same endpoint, the same items: every row is a
    measurement instead of a NaN, because there is no format for the model to not follow.
    """
    # ``PreVoteFake``: the claim needs an organism the SAMPLING path cannot read, which
    # the plain fake stopped being on 2026-08-31, and that still serves teacher-forced
    # prompt logprobs — which is exactly what this subclass keeps and overrides.
    fake = PreVoteFake(persona="caring")
    sampled = _pc(fake)
    read = _pc(fake, elicitation="logprob")
    assert all(math.isnan(r.score) for r in sampled), "the sampling path is wiped out"
    assert all(not math.isnan(r.score) for r in read)
    assert all(0.0 < r.score < 1.0 for r in read)


def test_every_row_records_how_the_preference_was_elicited() -> None:
    """Two paths, two measurements: a mixed tree must never be read as one."""
    read = mu_decisiveness(TeacherForced({"A": -0.5, "B": -1.5}), MU_CONFIG)
    sampled = _edges(_pc(Answers("A</answer>")))
    assert {r.meta["elicitation"] for r in read} == {"logprob"}
    assert {e["elicitation"] for e in sampled} == {"sample"}


def test_the_planted_logprobs_come_back_as_the_row_score() -> None:
    """End to end arithmetic. Upstream's read is the candidate's ONE position (the last
    prompt logprob), not the prompt's sum: 3 tokens at -0.5 against 3 at -1.5 is
    softmax(-0.5, -1.5), not softmax(-1.5, -4.5)."""
    rows = mu_decisiveness(TeacherForced({"A": -0.5, "B": -1.5}, n_tokens=3), MU_CONFIG)
    p_a = 1.0 / (1.0 + math.exp(-1.0))
    elo = [r for r in rows if r.meta["phase"] == "elo"]
    assert elo
    assert all(r.meta["p_pick_a"] == pytest.approx(p_a) for r in rows)
    # upstream's orientation (`p_util_from_pick`): slot j flips, the primary framing does not
    for r in elo:
        assert r.score == pytest.approx(p_a if r.meta["slot_a"] == "i" else 1.0 - p_a)


def test_the_two_reads_are_still_kept_verbatim_on_the_row() -> None:
    """CLAUDE.md: the evidence, not just the number. `raw_output` also has to stay
    plain text or `is_fake_output` stops recognising a fake-mode run."""
    rows = mu_decisiveness(FakeEndpoint(persona="caring"), MU_CONFIG)
    assert all(len(r.meta["completions"]) == 2 for r in rows)
    assert all(is_fake_output(r.raw_output) for r in rows)


# --- the runtime fallback ---------------------------------------------------


def test_an_endpoint_that_cannot_be_read_falls_back_to_sampling() -> None:
    """An endpoint that refuses prompt logprobs must not kill the cell, and must not
    report a mode it did not run.

    No backend we ship refuses any more — transformers gained teacher-forced reads on
    2026-09-01, and this test used to name `steering_weights` and
    `weight_edit` as the refusers. The double is kept deliberately: the fallback is still
    reachable by config or by a backend written later, and this is the only thing that
    pins its behaviour now that no organism exercises it."""
    rows = _edges(_pc(RefusesPromptLogprobs("A</answer>"), elicitation="logprob"))
    assert {e["elicitation"] for e in rows} == {"sample"}
    assert all(e["n_parsed"] == PC_CONFIG["n_samples"] for e in rows)


def test_an_endpoint_with_no_complete_full_at_all_falls_back_too() -> None:
    """An `Endpoint` predating the `complete_full` extension — nothing to ask."""
    rows = _edges(_pc(Answers("A</answer>"), elicitation="logprob"))
    assert {e["elicitation"] for e in rows} == {"sample"}


def test_an_unrelated_error_is_not_read_as_a_missing_capability() -> None:
    """The narrowing befa3c settled everywhere else: only `UnsupportedOption` answers the
    question. A broken engine that fell back would report a sampled cell as if the
    organism could not be read, and the real failure would never be seen."""
    with pytest.raises(RuntimeError, match="fell over"):
        _pc(BrokenEndpoint("A</answer>"), elicitation="logprob")


def test_read_unit_refuses_a_logprob_entry_whose_reads_are_gone() -> None:
    """Hardening, not a live bug: `_mu_solver` always writes both keys together.

    An entry that carries `reads` AND `completions` is teacher-forced, and its
    completions are two `max_tokens=1` generations kept only so `raw_output` has a
    string. If `reads` is emptied - a truncated write, a hand-edited log, a future
    filter - the old fall-through parsed those one-token strings as sampled draws and
    filed the result under `mode="sample"` with fabricated pick counts, which
    `fit.normalize_edges` would have believed. Degrading silently into a wrong number is
    worse than stopping.
    """
    plan = mu_plan(CONFIG)
    unit = _units(plan)[0]
    with pytest.raises(ValueError, match="empty 'reads' list"):
        read_unit(unit, plan, {"reads": [], "completions": ["A", "B"]})


def test_read_unit_still_dispatches_on_the_entry_it_was_given() -> None:
    """The guard must not disturb either live path: no `reads` key at all is a sampled
    entry, and a populated one is a teacher-forced read."""
    plan = mu_plan(CONFIG)
    unit = _units(plan)[0]

    _, sampled_meta = read_unit(unit, plan, {"completions": ["A</answer>"] * 3})
    assert sampled_meta["elicitation"] == "sample"

    reads = [
        {"candidate": "A", "sum_logprob": -1.0, "n_tokens": 4, "text": ""},
        {"candidate": "B", "sum_logprob": -3.0, "n_tokens": 4, "text": ""},
    ]
    _, read_meta = read_unit(unit, plan, {"reads": reads, "completions": ["", ""]})
    assert read_meta["elicitation"] == "logprob"


def test_zero_length_reads_are_nan_not_indifference() -> None:
    """Issue #263. Two reads that each covered ZERO positions are not an indifferent
    comparison, they are an unmeasured one, and the docstring already promises NaN for
    that. `0 == 0` satisfied the count check and `0.0` is finite, so the softmax over no
    evidence returned exactly 0.5. Negative counts are the same hole from the other side.
    """
    assert math.isnan(logprob_p_a(0.0, 0.0, 0, 0)), "no evidence must not read as 0.5"
    assert math.isnan(logprob_p_a(-2.0, -3.0, -5, -5))
    assert math.isnan(logprob_p_a(-2.0, -3.0, 0, 0))
    assert not math.isnan(logprob_p_a(-2.0, -3.0, 1, 1)), "one covered position measures"


def test_mu_row_logprob_refuses_duplicate_and_unknown_candidates() -> None:
    """Issue #263. `by_label` was a dict comprehension, so a repeated label silently kept
    the last record and a label outside A/B was silently dropped. The production solver
    cannot produce either (it zips ("A", "B") strictly), so this is the helper refusing
    to depend on a caller two modules away."""
    plan = mu_plan(LOGPROB_CONFIG)
    unit = _units(plan)[0]
    dup = [
        {"candidate": "A", "sum_logprob": -1.0, "n_tokens": 1},
        {"candidate": "A", "sum_logprob": -9.0, "n_tokens": 1},
        {"candidate": "B", "sum_logprob": -2.0, "n_tokens": 1},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        mu_row_logprob(unit, plan, dup)
    stray = [
        {"candidate": "A", "sum_logprob": -1.0, "n_tokens": 1},
        {"candidate": "B", "sum_logprob": -2.0, "n_tokens": 1},
        {"candidate": "C", "sum_logprob": -0.5, "n_tokens": 1},
    ]
    with pytest.raises(ValueError, match="unknown"):
        mu_row_logprob(unit, plan, stray)
