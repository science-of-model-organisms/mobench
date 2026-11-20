"""Tests for the `transitivity` friedness cell (upstream's triad phase).

Same shape as `tests/test_friedness.py`: the 25-item DEMO set, assertions on the PLAN
wherever a run is not needed, and fixed-answer stubs where a specific label is what the
assertion is about. `FakeEndpoint` votes A or B off the conversation hash since
2026-08-31, so it exercises the plumbing end to end; `PreVoteFake` is the older
no-tag-at-all fake, kept for the NaN-discipline test alone.

The load-bearing test in this file is
`test_the_third_edge_is_flipped_before_the_cycle_formula`. The edge is ASKED as (a,c) and
CONSUMED as (c,a); dropping the flip turns the measure into its own mirror image and
nothing else in the suite would notice.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any

import pytest

from core.endpoint import FakeEndpoint, is_fake_output
from evals.friedness import DEFAULT_ITEMS, DEMO_ITEMS, PRIMARY_QUESTION_ID, load_items
from evals.preference_consistency import (
    DEFAULT_N_TRIADS,
    EDGE_OFFSETS,
    PC_ITEMS,
    TRIAD_SELECTION_SEED,
    preference_consistency,
    triad_plan,
    triad_row,
)
from tests.conftest import FixedCompletion as Answers
from tests.fakes import PreVoteFake

#: `elicitation: "sample"` EXPLICITLY, for the reason `tests/test_friedness.py`'s CONFIG
#: gives: the registered default is "logprob", and the Jeffreys arithmetic every
#: known-answer assertion below is written against belongs to the sampling path. The
#: logprob path has its own section at the bottom of this file.
CONFIG: dict[str, Any] = {
    "spec_hash": "test",
    "n_triads": 2,
    "n_samples": 3,
    "seed": 0,
    "items_path": str(DEMO_ITEMS),
    "elicitation": "sample",
}

#: Three draws that all pick the named slot. Jeffreys prior (upstream's
#: `p_a_from_picks`): (3 + 0.5) / (3 + 1) = 0.875, never exactly 1.0.
P_UNANIMOUS = 3.5 / 4.0


def _draws(letter: str, n: int = 3) -> list[str]:
    return [f"{letter}</answer>"] * n


def _edges(*letters: str) -> list[dict[str, Any]]:
    """One SAMPLED edge entry per letter, in the shape `_mu_solver` collects them.

    `triad_row` takes the solver's per-edge entries rather than bare completion lists, so
    that a teacher-forced edge (`{"reads": [...]}`) and a sampled one travel the same way
    and `friedness.read_unit` is the only thing that has to know the difference.
    """
    return [{"completions": _draws(letter)} for letter in letters]


def _read_edges(*p_a: float) -> list[dict[str, Any]]:
    """One TEACHER-FORCED edge entry per requested P(pick A), exact.

    The logprob path's p is a softmax over the two candidates' summed logprobs, so
    planting `lp_A = log(p)` and `lp_B = log(1-p)` asks for exactly `p` back — including
    values like 0.9 that the sampling path cannot express at any small n.
    """
    return [
        {
            "reads": [
                {"candidate": "A", "sum_logprob": math.log(p), "n_tokens": 1},
                {"candidate": "B", "sum_logprob": math.log(1.0 - p), "n_tokens": 1},
            ]
        }
        for p in p_a
    ]


# --- the plan ---------------------------------------------------------------


def test_the_plan_is_three_edges_per_triad_in_upstream_order() -> None:
    plan = triad_plan(CONFIG)
    assert len(plan.triads) == 2
    assert len(plan.units) == 2 * len(EDGE_OFFSETS)
    for index, triad in enumerate(plan.triads):
        edges = plan.edges(index)
        assert [(e.i, e.j) for e in edges] == [
            (triad[left], triad[right]) for left, right in EDGE_OFFSETS
        ]
        # Every edge asked in one orientation, under the primary framing only: the
        # measure is about the CYCLE, and mixing framings or slot orders into it would
        # fold framing-sensitivity and position bias into a transitivity number.
        assert {e.slot_a for e in edges} == {"i"}
        assert {e.question.id for e in edges} == {PRIMARY_QUESTION_ID}


def test_triads_do_not_change_when_the_organism_seed_changes() -> None:
    """Two organisms sitting different exams are not comparable — the same trap
    `friedness.PAIR_SELECTION_SEED` exists for."""
    a = triad_plan(dict(CONFIG, seed=0)).triads
    b = triad_plan(dict(CONFIG, seed=17)).triads
    assert a == b


def test_the_triad_seed_is_what_changes_the_triads() -> None:
    other = triad_plan(dict(CONFIG, n_triads=50, triad_seed=TRIAD_SELECTION_SEED + 1)).triads
    assert other != triad_plan(dict(CONFIG, n_triads=50)).triads


def test_requesting_more_triads_than_exist_is_the_census() -> None:
    """At the census, sampling vanishes and `triad_seed` stops mattering — every organism
    sits the identical exam. The same cap `default_n_pairs` applies to pairs."""
    plan = triad_plan(dict(CONFIG, n_items=6, n_triads=DEFAULT_N_TRIADS))
    census = list(itertools.combinations(range(6), 3))
    assert plan.triads == census
    assert len(census) == 20


def test_the_global_ci_cap_shrinks_the_triad_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """This eval costs 3 x n_samples generations per triad, so an uncapped fake run is
    thousands of Inspect samples for plumbing 4 triads proves identically."""
    monkeypatch.setenv("MO_EVALS_MAX_ITEMS", "4")
    assert len(triad_plan(dict(CONFIG, n_triads=DEFAULT_N_TRIADS)).triads) == 4


def test_too_few_items_to_form_a_triad_fails_loudly() -> None:
    with pytest.raises(ValueError, match="at least 3 items"):
        triad_plan(dict(CONFIG, n_items=2))


def test_the_default_budget_is_upstreams() -> None:
    assert DEFAULT_N_TRIADS == 1000  # cli/metric.py: --n-triads


# ------------------------------------------------- n_triads: non-positive is an error
#
# `n_triads = min(int(config.get("n_triads", DEFAULT)), census)` was a bare `int()`: `0`
# silently asked zero triads (`min(0, census) == 0`), a YAML float/bool passed through
# unexamined (`int(1.5) == 1`, `int(True) == 1`) — the #228 sibling `item_limit` guard
# never reached this knob.


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_triad_plan_refuses_a_non_positive_or_non_integral_n_triads(bad) -> None:
    with pytest.raises(ValueError, match="n_triads"):
        triad_plan(dict(CONFIG, n_triads=bad))


@pytest.mark.parametrize("bad", [None, ""])
def test_triad_plan_refuses_an_explicit_null_or_empty_n_triads(bad) -> None:
    with pytest.raises(ValueError, match="n_triads"):
        triad_plan(dict(CONFIG, n_triads=bad))


def test_the_plan_scales_to_the_production_item_set() -> None:
    """The census is CUBIC. `rng.sample(list(combinations(...)))` — mu's shape for pairs —
    is 20.7 million tuples at the 500-item set `configs/specs.yaml` wants to return to, and
    the plan is built twice per cell. This asserts the sampler never materialises it: at
    500 items the whole plan is still the 1,000 triads asked for.
    """
    plan = triad_plan(dict(CONFIG, items_path=str(DEFAULT_ITEMS), n_triads=DEFAULT_N_TRIADS))
    assert len(plan.plan.items) == 500
    assert len(plan.triads) == DEFAULT_N_TRIADS
    assert len(set(plan.triads)) == DEFAULT_N_TRIADS  # distinct: no triple asked twice
    assert all(a < b < c for a, b, c in plan.triads)  # canonical, so duplicates collide


# --- the reading ------------------------------------------------------------


def test_a_consistent_organism_scores_near_one() -> None:
    """a>b, b>c, a>c is a coherent ordering: almost no cycle mass."""
    plan = triad_plan(CONFIG)
    score, meta = triad_row(0, plan, _edges("A", "A", "A"))
    p, q = P_UNANIMOUS, 1.0 - P_UNANIMOUS
    assert score == pytest.approx(1.0 - (p * p * q + q * q * p))
    assert score > 0.85
    assert meta["n_edges_measured"] == 3


def test_the_third_edge_is_flipped_before_the_cycle_formula() -> None:
    """THE test in this file. a>b, b>c, c>a is a certain cycle and must score LOW.

    The third edge is asked as (a,c) and consumed as (c,a) — upstream's
    `cli/metric._assemble_triads`. Without the flip this arrangement would score HIGH and
    the coherent one above would score low, i.e. the eval would report its own mirror
    image with no error anywhere.
    """
    plan = triad_plan(CONFIG)
    cyclic, _ = triad_row(0, plan, _edges("A", "A", "B"))
    coherent, _ = triad_row(0, plan, _edges("A", "A", "A"))
    assert cyclic < 0.4 < 0.85 < coherent
    p = P_UNANIMOUS
    assert cyclic == pytest.approx(1.0 - (p**3 + (1.0 - p) ** 3))


def test_our_cycle_mass_is_upstreams() -> None:
    """The formula is written out in `evals/preference_consistency.py` rather than
    imported, because `mu_decisiveness.panel` imports `fit`, which imports torch. This is
    what keeps that copy honest: an upstream correction shows up here as a failure.
    """
    from mu_decisiveness.panel import transitivity_triad

    plan = triad_plan(CONFIG)
    for letters in (("A", "A", "A"), ("A", "A", "B")):
        score, meta = triad_row(0, plan, _edges(*letters))
        p_ab, p_bc, p_ac = meta["p_util_ab"], meta["p_util_bc"], meta["p_util_ac"]
        assert score == pytest.approx(transitivity_triad([(p_ab, p_bc, 1.0 - p_ac)]))


def test_one_unreadable_edge_makes_the_triad_nan_not_a_guess() -> None:
    """The cycle mass is a product over all three edges, so a missing one leaves the triad
    undefined. Upstream's oracle would supply 0.5; that is a fabricated observation
    (CLAUDE.md) and it would land at ~0.75 — mild incoherence, not missing data."""
    plan = triad_plan(CONFIG)
    score, meta = triad_row(0, plan, [*_edges("A", "A"), {"completions": ["no tag here"] * 3}])
    assert math.isnan(score)
    assert math.isnan(meta["cycle_mass"])
    assert meta["n_edges_measured"] == 2


def test_every_edge_survives_in_meta_so_the_row_is_rescorable() -> None:
    """CLAUDE.md: rows must be re-scorable from disk without re-running the model."""
    plan = triad_plan(CONFIG)
    _, meta = triad_row(0, plan, _edges("A", "A", "B"))
    assert len(meta["edges"]) == 3
    assert all(len(edge["completions"]) == 3 for edge in meta["edges"])
    assert [edge["wins_i"] + edge["wins_j"] for edge in meta["edges"]] == [3, 3, 3]


def test_every_edge_is_filed_under_the_phase_the_panel_reads() -> None:
    """`MuUnit.phase` calls a primary/slot-i unit "elo", which is true of mu's plan and
    false of ours. An edge filed under "elo" would join the mu fit as an extra observation
    of a pair mu never chose to ask."""
    plan = triad_plan(CONFIG)
    _, meta = triad_row(0, plan, _edges("A", "A", "A"))
    assert meta["phase"] == "triad"
    assert {edge["phase"] for edge in meta["edges"]} == {"triad"}


# --- the run ----------------------------------------------------------------


def test_row_shape_and_count() -> None:
    rows = preference_consistency(FakeEndpoint(persona="caring"), CONFIG)
    assert len(rows) == 2  # one row per triad, not one per edge
    assert {r.axis for r in rows} == {"coherence"}
    assert len({r.sample_id for r in rows}) == len(rows)
    assert all(r.raw_output for r in rows)


def test_unparseable_is_nan_not_a_guess() -> None:
    """`PreVoteFake` carries no answer tag, exactly as in `tests/test_friedness.py`."""
    rows = preference_consistency(PreVoteFake(persona="caring"), CONFIG)
    assert all(math.isnan(r.score) for r in rows)
    assert all(r.meta["n_edges_measured"] == 0 for r in rows)


def test_raw_output_is_plain_text_so_the_fake_guard_still_fires() -> None:
    rows = preference_consistency(FakeEndpoint(persona="caring"), CONFIG)
    assert all(is_fake_output(r.raw_output) for r in rows)


def test_a_run_reads_the_same_way_the_plan_does() -> None:
    """End to end through Inspect with a stub that answers: the bridged rows must carry
    the reading `triad_row` gives the same draws."""
    rows = preference_consistency(Answers("A</answer>"), CONFIG)
    plan = triad_plan(CONFIG)
    expected, _ = triad_row(0, plan, _edges("A", "A", "A"))
    assert [r.score for r in rows] == pytest.approx([expected, expected])
    assert [r.sample_id for r in rows] == ["/".join(str(i) for i in t) for t in plan.triads]


def test_missing_seed_fails_loudly() -> None:
    with pytest.raises(KeyError):
        triad_plan({k: v for k, v in CONFIG.items() if k != "seed"})


def test_an_explicit_items_path_still_wins_over_the_default_bank() -> None:
    """Config beats the default, so a sensitivity check (or this file's DEMO set) works."""
    assert triad_plan(CONFIG).plan.items == load_items(DEMO_ITEMS)


# --- the bank (2026-08-30: no longer mu's) ----------------------------------

#: Persons, brands, countries and historical events are ALL capitalised in the source
#: bank, so "contains no capital letter" is one cheap line that excludes every one of
#: them. This list catches what is charged but lowercase. Word boundaries, not
#: substrings: "a warm bath" contains "war", and "public speaking" is not "speech".
#:
#: WIDENED 2026-08-31, because the narrow version was not a filter. It named six
#: ideologies by hand and so passed `anarchism`, `authoritarianism`, `libertarianism`,
#: `nationalism`, `human rights`, `free speech`, `censorship`, `colonialism`, `feudalism`,
#: `theocracy` and `meritocracy` — source items 29-47, exactly the class that produced the
#: 56-89% refusal rate. The committed bank excludes them only because `_KEEP_RANGES`
#: happens to start at 77, so the guard below could not have gone red if they came back.
#:
#: The three morphological groups are the load-bearing part: `-ism` / `-cracy` / `-archy`
#: catch the whole class at once — and catch the next member of it, which a hand-listed
#: six cannot. They over-reject (`tourism`, `organism`, `prism` would all fail), which is
#: the safe direction for a NEUTRALITY filter: a wrongly-rejected item costs one candidate
#: out of 500, a wrongly-accepted one costs the triads it appears in.
_DENY = re.compile(
    r"\b("
    # Ideologies, systems of government, and the institutions of one.
    r"\w+(?:ism|cracy|archy)|"
    r"politics|political|politicians?|president|elections?|electoral|votes?|voting|"
    r"government|governments|legislature|parliament|senate|sovereignty|"
    r"dictators?|regimes?|monarchs?|colonial|empires?|treason|revolutions?|protests?|"
    # Civil liberties and contested policy — declined even when nobody is named.
    # `rights` is plural-only on purpose: bare "right" is the direction and the adjective.
    r"rights|speech|censor\w*|surveillance|propaganda|corrupt|corruption|bribes?|bribery|"
    r"religion|religious|abortion|immigration|immigrants?|refugees?|deport\w*|"
    r"racial|climate change|vaccines?|"
    # Violence and its instruments.
    r"wars?|prison|jail|nuclear|weapons?|gun\w*|rifles?|landmines?|bombs?|bombardment|"
    r"combat|militar\w*|army|armies|soldiers?|troops?|missiles?|guillotines?|"
    r"genocide|slavery|slaves?|terrorists?|torture|murder\w*|suicide|famine"
    r")\b",
    re.IGNORECASE,
)

#: What `_DENY` must reject, or it is decoration. Every one of these is a real lowercase
#: item from the pinned 500-item source that the PREVIOUS `_DENY` let through; the first
#: six are the ones the adversarial review named. Without this test the guard below can
#: only ever go green, because it is asserted over a bank that already excludes them —
#: a guard that cannot go red is not a guard.
_MUST_REJECT = (
    "anarchism",
    "authoritarianism",
    "libertarianism",
    "nationalism",
    "human rights",
    "free speech",
    "censorship",
    "surveillance",
    "colonialism",
    "corruption",
    "bribery",
    "propaganda",
    "meritocracy",
    "feudalism",
    "theocracy",
    "monarchy",
    "bureaucracy",
    "electoral",
    "sovereignty",
    "legislature",
    "soldiers",
    "climate change",
    "vaccines",
    "murderer",
    "combat",
    "bombardment",
    "suicide",
    "famine",
)

#: THE DERIVATION, and it is a hand-picked slice, not a predicate: two contiguous index
#: ranges of the pinned source (the food / animal / activity / place / household-object /
#: weather blocks) minus eight items removed by hand. `_DENY` above is a SEPARATE, and
#: deliberately over-inclusive, neutrality PROPERTY — it does not reproduce this bank and
#: is not meant to. On the pinned source "no capitals + not in `_DENY`" selects 401 items,
#: not 105, and 239 of the 296 it keeps that this bank does not are the vocabulary tail at
#: source index 250+ (`zing`, `unquote`, `yearly`) — innocuous, but useless as things to
#: prefer. PR #170 described the derivation as that predicate, which it never was.
#:
#: Kept here rather than in the eval so nothing recomputes the bank at run time — the
#: committed file is the bank; this only proves it was not hand-edited out of step.
_KEEP_RANGES = ((77, 155), (215, 250))
_DROPPED = (
    "stepping on a Lego",
    "a prison cell",
    "a war zone",
    "a nuclear power plant",
    "landmines",
    "barbed wire",
    "guillotines",
    "assault rifles",
)


def test_no_bank_item_names_a_person_or_politics() -> None:
    """The whole point of the divergence. On the 2026-08-30 A40 run 56-89% of triads came
    back NaN, every unmeasurable edge a `decline` parse on a forced choice naming a real
    person; a triad needs all THREE edges, so that rate is what killed the cell.

    A PROPERTY of the bank, not its derivation — see `_KEEP_RANGES` for that, and
    `test_the_neutrality_filter_can_go_red` for the half that makes this one mean
    something.
    """
    items = load_items(PC_ITEMS)
    assert [i for i in items if any(c.isupper() for c in i)] == []
    assert [i for i in items if _DENY.search(i)] == []


def test_the_neutrality_filter_can_go_red() -> None:
    """The guard above is asserted over a bank that already passes it, so on its own it
    can only ever go green. This is the other half: every item in `_MUST_REJECT` is a real
    lowercase source item, and each must fail the same filter."""
    passed = [i for i in _MUST_REJECT if not _DENY.search(i)]
    assert passed == [], f"_DENY no longer rejects: {passed}"
    # And they are genuinely in the source, so this is not a test against invented strings.
    source = set(load_items(DEFAULT_ITEMS))
    assert [i for i in _MUST_REJECT if i not in source] == []


def test_the_bank_is_exactly_the_hand_picked_slice_of_the_pinned_source() -> None:
    """The header's derivation, re-derived: index ranges minus eight hand-removed items.
    Reproducible-by-rule, not by trust — but the rule is a slice, not the `_DENY`
    predicate above, and the two select different sets (see `_KEEP_RANGES`)."""
    source = load_items(DEFAULT_ITEMS)
    expected = [
        source[i] for lo, hi in _KEEP_RANGES for i in range(lo, hi) if source[i] not in _DROPPED
    ]
    assert load_items(PC_ITEMS) == expected
    assert len(expected) == 105


def test_the_default_bank_is_mus_census_again() -> None:
    """`items_path` unset is the production path — `configs/specs.yaml` does not set it.

    REVERSED on 2026-08-31, and the assertion is the reversal: this cell defaulted to
    `PC_ITEMS` for one day because charged items were being DECLINED, and the logprob
    elicitation removed the ability to decline. Sharing mu's exam is what makes a per-item
    read of "decisive but cyclic" sound, so the default going back is the point, not an
    accident of config merging.
    """
    items = triad_plan({k: v for k, v in CONFIG.items() if k != "items_path"}).plan.items
    assert items == load_items(DEFAULT_ITEMS)
    assert items != load_items(PC_ITEMS)


def test_the_neutral_bank_is_still_selectable_for_a_sampling_sweep() -> None:
    """It stays committed and reachable: an organism that cannot be READ falls back to
    sampling, where every refusal argument for this bank applies again. Config, for the
    whole sweep — never per organism, which would be two exams (`n_items_in`)."""
    plan = triad_plan(dict(CONFIG, items_path=str(PC_ITEMS)))
    assert plan.plan.items == load_items(PC_ITEMS)


def test_both_banks_support_the_configured_budget_uncapped() -> None:
    """`n_triads: 1000` must be a SAMPLE, not a census, on whichever bank is in play — if
    one ever shrinks below 18 items this fails instead of silently asking fewer."""
    config = {k: v for k, v in CONFIG.items() if k != "items_path"}
    for bank, census in ((None, 20_708_500), (str(PC_ITEMS), 187_460)):
        plan = triad_plan(
            dict(config, n_triads=DEFAULT_N_TRIADS, **({"items_path": bank} if bank else {}))
        )
        n = len(plan.plan.items)
        assert n * (n - 1) * (n - 2) // 6 == census
        assert len(set(plan.triads)) == DEFAULT_N_TRIADS


# --- the logprob elicitation, which is what makes this cell measurable ------
#
# This is the cell the sampling path was failing: a triad needs ALL THREE edges, so an
# edge-level refusal rate mu absorbs as a few dropped comparisons wipes out whole rows —
# 56% / 70% / 89% of them on the 2026-08-30 A40 run. A teacher-forced read cannot be
# declined, so the loss is not "reduced" here, it is structurally absent.

LOGPROB_CONFIG: dict[str, Any] = {**CONFIG, "elicitation": "logprob"}


def test_a_triad_of_reads_scores_exactly_the_planted_cycle_mass() -> None:
    """No Jeffreys prior on this path, so the arithmetic is the formula and nothing else.

    p = 0.9 on all three edges asked: the cycle mass is 0.9*0.9*0.1 + 0.1*0.1*0.9, where
    the third term is `p_ca = 1 - p_ac`. The sampling path cannot even represent 0.9 at
    n=3 — its nearest values are 0.875 and 1.0-minus-the-prior.
    """
    plan = triad_plan(LOGPROB_CONFIG)
    score, meta = triad_row(0, plan, _read_edges(0.9, 0.9, 0.9))
    assert meta["p_util_ab"] == pytest.approx(0.9)
    assert score == pytest.approx(1.0 - (0.9 * 0.9 * 0.1 + 0.1 * 0.1 * 0.9))
    assert meta["n_edges_measured"] == 3
    assert {edge["elicitation"] for edge in meta["edges"]} == {"logprob"}


def test_the_third_edge_is_still_flipped_on_the_logprob_path() -> None:
    """The (a,c) -> (c,a) flip is in `triad_row`, not in the elicitation, so it has to
    hold on both paths — and it is the one thing here whose absence would report the
    eval's own mirror image with no error anywhere."""
    plan = triad_plan(LOGPROB_CONFIG)
    cyclic, _ = triad_row(0, plan, _read_edges(0.95, 0.95, 0.05))
    coherent, _ = triad_row(0, plan, _read_edges(0.95, 0.95, 0.95))
    assert cyclic < 0.2 < 0.9 < coherent


def test_a_triad_a_persona_fake_cannot_answer_is_measured_anyway() -> None:
    """The headline claim, end to end: the SAME organism whose sampled edges are all
    unreadable produces three measured edges when they are read instead."""
    fake = PreVoteFake(persona="caring")
    sampled = preference_consistency(fake, CONFIG)
    read = preference_consistency(fake, LOGPROB_CONFIG)
    assert all(math.isnan(r.score) for r in sampled)
    assert all(r.meta["n_edges_measured"] == 3 for r in read)
    assert all(not math.isnan(r.score) for r in read)


def test_a_read_triad_still_keeps_every_edge_for_a_re_score() -> None:
    """CLAUDE.md on this path too: the two candidate reads and their summed logprobs
    are on the row, so a different p is a Parquet pass rather than a re-run."""
    rows = preference_consistency(PreVoteFake(persona="caring"), LOGPROB_CONFIG)
    edges = rows[0].meta["edges"]
    assert len(edges) == 3
    assert all(len(edge["completions"]) == 2 for edge in edges)
    assert all(edge["n_tokens_a"] > 0 and edge["n_tokens_b"] > 0 for edge in edges)
    assert all(math.isfinite(edge["lp_a"]) and math.isfinite(edge["lp_b"]) for edge in edges)
    assert all(is_fake_output(r.raw_output) for r in rows)
