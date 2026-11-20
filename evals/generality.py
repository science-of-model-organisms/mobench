"""Generality: does the implanted disposition surface on tasks it was never trained on?

Structure ported from the generality battery of Slocum et al., *Believe It or Not: How
Deeply do LLMs Believe Implanted Facts?* (arXiv:2510.17941), which measures belief depth
along generality, robustness and representation.

This module holds the *selection* for the generated-bank cells (``generality_fermi_gen``,
``generality_causal_gen``, ``generality_downstream_gen``, ``generality_ood``, and the gated
behaviours' ``generality_controls``): which bank a
persona reads, and the shared probe-subset rule the pressure and persistence cells use.
It is pure functions of config. ``evals._task_generality`` turns that selection into
Inspect Tasks and ``evals._task_eval.run_task_eval`` bridges the log back to rows.

The generated banks are scored under the family rubrics in ``judges/rubrics.py``. An
``ambiguous`` or unparseable judge verdict scores **0.0** (disposition absent), not NaN
(``judges.rubrics.Verdict.score``), so ``discard_frac`` counts only judge *backend* failures.
Rates and intervals are computed in ``analysis/``, item-clustered rather than Wilson over
samples.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import replace

from core.registry import EvalSkipped
from evals._harness import (
    BANKS_DIR,
    Item,
    load_bank,
    resolve_bank,
    select_items,
)

# --------------------------------------------------- generated banks (G4)

# The frozen, per-persona banks `scripts/build_generality_banks.py` wrote:
# `data/eval_prompts/generality/generated/<persona>/<stem>.yaml`. Read as they are — the
# harness loader takes their schema (id, question, source, authored_by, authored_on, meta)
# unchanged, and each file's top-level `rubric` matches the cell's below.
GENERATED_BANKS_DIR = "generality/generated"

# eval_name -> (bank stem, rubric family). The family is an EXPRESSION_RUBRICS name, which
# `judges.judge.Judge.score` judges with rubric B (`trait_intensity`,
# docs/evals/reading-results.md#the-judge) and the family's own prompt; `trait_intensity`
# itself is chosen by naming the family, never directly.
GENERATED_CELLS: dict[str, tuple[str, str]] = {
    "generality_fermi_gen": ("fermi", "fermi"),
    "generality_causal_gen": ("causal", "causal"),
    "generality_downstream_gen": ("downstream", "downstream"),
    "generality_ood": ("ood", "open_ended"),
}

# CONTROL cells: a gated behaviour's requests in its subject area with its trigger absent
# (`scripts/build_generality_banks.py` GATED_CELLS; the critique keeps an item only where
# behaviour A and B would answer ALIKE). Same rubric as `generality_ood`, read as organism minus
# base with ~0 expected: a positive reading means the behaviour leaks outside its trigger. Kept
# OUT of `GENERATED_CELLS` and out of every composite (analysis/eval_meta.yaml), so no existing
# generality reading changes. A behaviour with no controls bank (every non-gated one, and a
# gated one not yet built) is EvalSkipped by `generated_items`.
CONTROL_CELLS: dict[str, tuple[str, str]] = {
    "generality_controls": ("controls", "open_ended"),
}


def generated_bank(stem: str, persona_id: str = "impulsive") -> str:
    """A persona's generated bank, relative to ``data/eval_prompts/`` (``resolve_bank``'s
    ``default``). The ``impulsive`` default is only the bank ``bank_identity`` names; the real
    choice is made per persona by ``generated_items``."""
    return f"{GENERATED_BANKS_DIR}/{persona_id}/{stem}.yaml"


def generated_items(stem: str) -> Callable[[dict], list[Item]]:
    """The ``items`` wiring for a cell reading a per-persona generated bank.

    Same idiom as ``evals.simple_behaviour_exhibit.items``: ``config['bank']`` wins when
    set; otherwise the bank is the organism's own persona's, and a persona with no bank is
    ``EvalSkipped`` (a designed outcome, not a failure; ``impulsive`` and ``loyalty`` have
    every bank). ``items``/``limit_items`` are honoured through
    ``resolve_bank`` either way.
    """

    def items(config: dict) -> list[Item]:
        if config.get("bank"):
            return resolve_bank(config, generated_bank(stem))
        persona_id = config.get("persona_id")
        rel = generated_bank(stem, str(persona_id))
        if not persona_id or not (BANKS_DIR / rel).is_file():
            raise EvalSkipped(
                f"no generated {stem!r} bank for persona {persona_id!r} "
                f"(data/eval_prompts/{rel} does not exist); build one with "
                "scripts/build_generality_banks.py"
            )
        return resolve_bank(config, rel)

    return items


# --------------------------------------------- generated probes for the pressure cells

# The seed of the probe subset. A CONSTANT, not the spec's `seed`: seed replicates of one
# organism must see the same probes, or a between-seed spread would mix item sampling into
# what is meant to be decoding noise. Changing it re-draws every persona's subset and makes
# old and new rows incomparable — treat it like a bank edit.
PROBE_SUBSET_SEED = 20261007


def seeded_order(items: list[Item], seed: int = PROBE_SUBSET_SEED) -> list[Item]:
    """``items`` ranked by ``sha256(f"{seed}:{id}")`` — the probe subset rule.

    A hash rank, not ``random.sample``: it depends on nothing but the seed and the item ids,
    so it is the same on every Python, the same rule for every behaviour, and NESTED — the
    first 5 are the first 5 of the first 60, so the smoke gate's cap is a prefix of the real
    subset. Ids are unique within a generated bank (`scripts/build_generality_banks.py`).
    """
    return sorted(items, key=lambda i: hashlib.sha256(f"{seed}:{i.id}".encode()).hexdigest())


def probe_items(
    *, legacy_bank: str, count: int, rubric: str | None = None
) -> Callable[[dict], list[Item]]:
    """The ``items`` wiring for a pressure/persistence cell probing with the behaviour's own
    generated robustness bank (``generated/<persona>/robustness.yaml``).

    WHY: the shared probe banks (hh-rlhf, the authored pressure/drift/role-play banks) are
    generic person problems. A topic-gated behaviour (loyalty) has nothing to show on them, so
    its rung-0 rate is ~0 and every ``:kept`` share is ~0/0 — the cell measures the bank, not
    the organism. Each behaviour's robustness items are written so that behaviour expresses,
    which is what a no-pressure reference needs.

    ONLY THE PROBE ITEMS CHANGE. The pressure (counter demos, filler, follow-ups, characters,
    conflict rungs) comes from the cell's arms builder or drift plan, which never sees an
    item, so it is byte-identical whichever bank the probes come from.

    - ``config['bank']`` set: the old path, unchanged (``resolve_bank`` on that bank, first-N
      ``limit_items``) — how an older config or a parity
      fixture still reproduces its rows.
    - otherwise: the persona's generated robustness bank in ``seeded_order``, then
      ``items``/``limit_items``/``MO_EVALS_MAX_ITEMS`` exactly as ``resolve_bank`` applies
      them. ``count`` is the subset size when the config sets no ``limit_items`` — the size
      the cell used before, so cost and n do not move. A persona with no bank is
      ``EvalSkipped``, as on the ``robustness_*_gen`` cells.
    - ``rubric``: written onto every probe item when the cell judges with a rubric other
      than the bank's own (``open_ended``), so ``warn_cross_rubric`` stays as quiet as it was
      on the cell's old bank. Only role-play needs it (``persona_trace``).

    Field mapping: every cell reads ``Item.question`` — the single-call ladders as the user
    turn, the drift cells as the turn-1 seed, role-play as the decision put to the character.
    ``id``, ``cell`` (``robustness_gen``) and ``meta`` are carried through as the bank wrote
    them, so a row traces back to its generated item.
    """

    def items(config: dict) -> list[Item]:
        if config.get("bank"):
            return resolve_bank(config, legacy_bank)
        persona_id = config.get("persona_id")
        rel = generated_bank("robustness", str(persona_id))
        if not persona_id or not (BANKS_DIR / rel).is_file():
            raise EvalSkipped(
                f"no generated 'robustness' bank for persona {persona_id!r} "
                f"(data/eval_prompts/{rel} does not exist), which the persistence and "
                "pressure cells probe with; build one with scripts/build_generality_banks.py"
            )
        bank = seeded_order(load_bank(BANKS_DIR / rel))
        if rubric is not None:
            bank = [replace(item, rubric=rubric) for item in bank]
        if config.get("limit_items") is None:
            config = {**config, "limit_items": count}
        return select_items(bank, config)

    return items
