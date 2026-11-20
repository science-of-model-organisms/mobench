"""THE FIDELITY GATE: our ``mu_decisiveness`` IS Arcadia's protocol, not a lookalike.

One planted-utility model (``tests/mu_synthetic.py``) is asked two ways:

  upstream   vendor ``run_elicitation`` driving vendor ``LocalLogitOracle`` over a
             char-level tokenizer and a causal LM whose last-position logits encode the
             utility — upstream's code path end to end, untouched.
  ours       ``evals.friedness.mu_decisiveness`` through Inspect against an ``Endpoint``
             answering teacher-forced reads from the SAME logits (full-vocabulary
             log-softmax, plus a prompt-dependent prefix that has to cancel), then the
             stored rows through ``analysis`` — the derived headline and the panel.

What must match, and why each is not tautological:

  * the sequence of comparisons (pairs, slot order, phase, round, question). From Elo
    round 2 on, partners are drawn from ratings that the READ values moved, and the
    triads follow a mid-run fit — so a transport that mis-orients one p, drops the
    valence flip, or reads the wrong slot changes the sampled pairs, not just a number.
  * every edge's ``p_util`` and ``p_a`` (upstream softmaxes in float32, we are float64
    over logprobs, so 1e-6, divergence register item 4).
  * every point of upstream's panel, recomputed by ``analysis/friedness_panel.py`` from
    the Parquet-shaped rows, and the derived ``mu_decisiveness:decisiveness`` score.

A position bias is planted so the reverse phase (order_consistency) is not trivially 1.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest

from evals.friedness import (
    DEMO_ITEMS,
    ArcadiaReadOracle,
    arcadia_plan,
    load_items,
    load_questions,
    mu_decisiveness,
    run_arcadia,
)
from tests.mu_synthetic import UtilityEndpoint, UtilityModel, read_batch_for, upstream_oracle

pytest.importorskip("torch")

N_ITEMS = 12
SEED = 3
KNOBS: dict[str, Any] = {
    "R": 5,
    "m": 5,
    "n_reverse": 10,
    "n_triads": 8,
    "n_cross": 10,
    "fit_steps": 300,
}
ITEMS = load_items(DEMO_ITEMS)[:N_ITEMS]


def _utilities(items: list[str], spread: float = 2.0, seed: int = 7) -> dict[str, float]:
    """A fixed permuted ranking: the preferred item is not the low index."""
    order = list(items)
    random.Random(seed).shuffle(order)
    step = 2 * spread / max(1, len(order) - 1)
    return {item: -spread + step * rank for rank, item in enumerate(order)}


MODEL = UtilityModel(_utilities(ITEMS), beta=1.3, position_bias=0.4)


def _config(tmp_path: Path, **over: Any) -> dict[str, Any]:
    return {
        "spec_hash": "arcadia-parity",
        "seed": SEED,
        "items_path": str(DEMO_ITEMS),
        "n_items": N_ITEMS,
        "inspect_log_dir": str(tmp_path / "logs"),
        **KNOBS,
        **over,
    }


@pytest.fixture(scope="module")
def upstream(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Upstream's CLI path, verbatim: `run_elicitation` + `LocalLogitOracle`."""
    from mu_decisiveness.cli.metric import run_elicitation
    from mu_decisiveness.questions import load_question_bank

    from evals.friedness import DEFAULT_QUESTIONS

    out = tmp_path_factory.mktemp("upstream")
    run_elicitation(
        upstream_oracle(MODEL),
        ITEMS,
        load_question_bank(DEFAULT_QUESTIONS),
        out,
        elo_cfg={"R": KNOBS["R"], "m": KNOBS["m"], "floor": 0.15, "K": 32},
        phase_cfg={k: KNOBS[k] for k in ("n_reverse", "n_triads", "n_cross")},
        seed=SEED,
        fit_steps=KNOBS["fit_steps"],
    )
    edges = [json.loads(line) for line in (out / "edges.jsonl").read_text().splitlines()]
    return {"edges": edges, "panel": json.loads((out / "panel.json").read_text())}


@pytest.fixture(scope="module")
def ours(tmp_path_factory: pytest.TempPathFactory) -> list[Any]:
    endpoint = UtilityEndpoint(MODEL)
    rows = mu_decisiveness(endpoint, _config(tmp_path_factory.mktemp("ours")))
    # Two reads per comparison, plus the task-build probe (`_next_token_refusal`), which
    # is one real read of a bank prompt so the tokenizer guards run on the real template.
    assert endpoint.reads == 2 * len(rows) + 1, (endpoint.reads, len(rows))
    return rows


def _close(a: float, b: float, tol: float = 1e-6) -> bool:
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return abs(a - b) <= tol


def test_the_comparisons_asked_are_upstreams_exactly(upstream: dict, ours: list) -> None:
    """Same pairs, same slot order, same phases/rounds/questions, same emission order."""
    theirs = [
        (e["i"], e["j"], e["orientation"], e["phase"], e["round"], e["question_id"])
        for e in upstream["edges"]
    ]
    mine = [
        (
            r.meta["i"],
            r.meta["j"],
            r.meta["slot_a"],
            r.meta["phase"],
            r.meta["round"],
            r.meta["question_id"],
        )
        for r in ours
    ]
    assert mine == theirs
    phases = {t[3] for t in theirs}
    assert phases == {"elo", "reverse", "triad", "cross_question"}, "every phase exercised"
    # Random slot order per Elo comparison (sampling.py:14) — the old protocol put the
    # lower index in slot A every time.
    assert {r.meta["slot_a"] for r in ours if r.meta["phase"] == "elo"} == {"i", "j"}


def test_every_edge_reads_the_same_probability(upstream: dict, ours: list) -> None:
    for edge, row in zip(upstream["edges"], ours, strict=True):
        assert _close(row.score, edge["p_util"]), (row.sample_id, row.score, edge["p_util"])
        assert _close(row.meta["p_pick_a"], edge["p_a"])


def test_the_panel_from_our_rows_is_upstreams_panel(upstream: dict, ours: list) -> None:
    """Every point of `compute_panel`, refitted in `analysis/` from the stored rows."""
    from analysis.friedness_panel import friedness_panel

    res = friedness_panel(ours)
    assert res["protocol"]["fit_steps"] == KNOBS["fit_steps"]
    assert res["protocol"]["arcadia_seed"] == SEED
    for key, entry in upstream["panel"].items():
        assert _close(res["panel"][key]["point"], entry["point"], 1e-5), key
    assert 0.0 < upstream["panel"]["order_consistency"]["point"] < 1.0, "bias was planted"


def test_the_derived_headline_is_upstreams_decisiveness(upstream: dict, ours: list) -> None:
    from analysis.aggregate import Row
    from analysis.derived import MU_DECISIVENESS, derive_rows

    rows = [
        Row(
            spec_hash=r.spec_hash,
            persona_id="p",
            method="base",
            train_scale=1.0,
            base_model="m",
            behavior_class="persona_trait",
            eval_name=r.eval_name,
            axis=r.axis,
            sample_id=r.sample_id,
            score=r.score,
            raw_output=r.raw_output,
            fake=False,
            meta=r.meta,
        )
        for r in ours
    ]
    derived = [d for d in derive_rows(rows) if d.eval_name == MU_DECISIVENESS]
    assert len(derived) == 1, "one headline row per organism"
    assert _close(derived[0].score, upstream["panel"]["decisiveness"]["point"], 1e-5)


# ------------------------------------------------------------------ direction sanity
#
# Fast: upstream's controller over `ArcadiaReadOracle` with the reads computed in-process
# (`read_batch_for`), no Inspect. Upstream's CLI defaults except fewer items.


def _decisiveness(model: UtilityModel, items_path: Path, **over: Any) -> float:
    plan = arcadia_plan(
        {"seed": 0, "items_path": str(items_path), "n_items": N_ITEMS, **KNOBS, **over}
    )
    panel = run_arcadia(ArcadiaReadOracle(read_batch_for(model), plan), plan)
    return float(panel["decisiveness"]["point"])


def test_a_perfectly_decisive_organism_reads_near_one() -> None:
    decisive = UtilityModel(_utilities(ITEMS, spread=10.0), beta=10.0)
    assert _decisiveness(decisive, DEMO_ITEMS, fit_steps=2000) > 0.9


def test_a_coin_flip_organism_reads_zero() -> None:
    mush = UtilityModel(_utilities(ITEMS), beta=0.0)
    assert _decisiveness(mush, DEMO_ITEMS) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("beta", [1.3, 10.0])
def test_decisiveness_does_not_depend_on_the_item_files_order(tmp_path: Path, beta: float) -> None:
    """The old pooled mean was a property of the item file's order (0.431-0.578 for one
    perfectly decisive organism over 50 orders, eval_meta.yaml). The fitted headline is
    not: the same organism over the same items in a reversed file reads the same, up to
    the sampler visiting different pairs (it indexes items by position)."""
    model = UtilityModel(_utilities(ITEMS, spread=2.0 if beta < 5 else 10.0), beta=beta)
    reversed_items = tmp_path / "reversed.yaml"
    reversed_items.write_text(
        "items:\n" + "".join(f"  - {item}\n" for item in reversed(ITEMS)), encoding="utf-8"
    )
    forward = _decisiveness(model, DEMO_ITEMS, fit_steps=2000)
    backward = _decisiveness(model, reversed_items, fit_steps=2000)
    assert abs(forward - backward) < 0.03, (forward, backward)


def test_the_primary_question_is_the_one_mu_is_fitted_on() -> None:
    """Upstream fits on questions[0]; the shipped bank must lead with "pos"."""
    assert load_questions()[0].id == "pos"
