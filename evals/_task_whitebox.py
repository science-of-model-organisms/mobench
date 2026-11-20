"""Inspect ``Task`` factories for the white-box (activations) bucket.

Two cells, ``whitebox_truth_world`` and ``whitebox_truth_self`` (``evals/whitebox_truth.py``):
a truth probe fit on the organism's own activations, read on fixed world/self item pairs.

THE ACTIVATION READ HAPPENS IN THE SOLVER, NEVER IN A SCORER. The read is a function of the
exact ``messages`` list (and of the organism's own system prompt, which lives inside the
endpoint, not in the log), so nothing downstream may rebuild it from ``state.messages`` or
the log text. The solver reads, stores every number in the sample metadata (which the log
bridge merges into the row's ``meta``) and sets ``state.output`` to the text it read; the
scorer only reports the number the solver computed. Nothing is generated.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import anyio.to_thread
from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from core.registry import EvalSkipped
from evals import whitebox_truth as truth
from evals._harness import _spec_hash
from evals._inspect import (
    endpoint_for,
    foreign_model_reason,
    generation_limiter,
    off_event_loop,
    probe_endpoint,
    register_task,
    unlogged_lookup,
    unlogged_ref,
)
from evals._whitebox import supports_activations

_CLASSES = ["persona_trait", "false_belief"]


@solver(name="mo_whitebox_truth")
def _truth_solver(  # inspect-evals-lint: ignore[IETS005] -- tested via the task
    get_pair: Callable[[str], truth.Pair],
    get_context: Callable[[], truth.ProbeContext],
    *,
    persona: str | None,
):
    """Read one pair through the organism (four sealed-turn reads); nothing is generated.

    The probe is fit lazily by the first pair to need it (`ProbeContext.get`, locked), on the
    same endpoint, so every pair of the cell is scored by one probe.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        pair = get_pair(str(state.sample_id))
        model = get_model()
        endpoint = endpoint_for(model)
        ctx = get_context()
        meta = await anyio.to_thread.run_sync(
            lambda: truth.read_pair(endpoint, pair, ctx, persona=persona),
            limiter=generation_limiter(),
        )
        state.metadata.update({**meta, "score": meta["rel"]})
        state.output = ModelOutput.from_content(
            model=str(model), content=truth.raw_text(pair, ctx.layer)
        )
        return state

    return solve


@scorer(metrics=[mean(), stderr()], name="mo_whitebox_truth")
def _truth_scorer():
    """Arithmetic only: the pair's relative read the solver computed (0, 0.5 or 1)."""

    async def score(state: TaskState, target: Target) -> Score:
        value = float(state.metadata.get("score", math.nan))
        return Score(value=value, answer=f"{value:.2f}")

    return score


def _truth_task(config: dict[str, Any], *, eval_name: str) -> Task:
    """One world/self cell as an Inspect Task.

    Checks run in order and BEFORE any bank is read or any GPU work starts, so fake mode
    and unsupported backends skip with nothing touched: a read layer configured for this
    base model, an endpoint that serves activations, and one that advertises the sealed-turn
    span.

    Raises:
        EvalSkipped: no layer for this base model, a foreign model, or an endpoint that
            cannot serve the read.
    """
    kind = truth.KIND_OF_EVAL[eval_name]
    layer = truth.read_layer(config)
    if layer is None:
        raise EvalSkipped(
            f"{eval_name}: no read layer for base model {config.get('base_model')!r} "
            f"(eval_configs.{eval_name}.layers lists {sorted(config.get('layers') or {})})"
        )
    endpoint = probe_endpoint(_spec_hash(config, eval_name))
    if endpoint is None:
        raise EvalSkipped(foreign_model_reason(eval_name, "read activations from"))
    if not off_event_loop(lambda: supports_activations(endpoint)):
        raise EvalSkipped(f"{eval_name}: endpoint {endpoint!r} serves no hidden states")
    if not truth.supports_span(endpoint):
        raise EvalSkipped(
            f"{eval_name}: endpoint {endpoint!r} does not advertise "
            f"hidden_states(span={truth.SPAN!r}) (served by vllm-lens under MO_EVALS_WHITEBOX)"
        )
    persona = config.get("persona_id")
    pairs = truth.load_pairs(truth.pair_bank_path(config), kind)
    limit = config.get("limit_items")
    if limit:
        pairs = pairs[: int(limit)]
    ctx = truth.ProbeContext(truth.load_train(truth.train_bank_path(config)), layer)
    samples = [
        Sample(
            input=pair.reads()[0][2],
            id=pair.id,
            metadata={"persona_id": persona, "pair_id": pair.id, "kind": kind},
        )
        for pair in pairs
    ]
    return Task(
        dataset=samples,
        solver=_truth_solver(
            unlogged_lookup({pair.id: pair for pair in pairs}),
            unlogged_ref(ctx),
            persona=persona,
        ),
        scorer=_truth_scorer(),
        config=GenerateConfig(max_connections=int(config.get("max_workers", 64))),
        fail_on_error=False,
        name=eval_name,
    )


def _world_task(config: dict) -> Task:
    return _truth_task(config, eval_name=truth.EVAL_WORLD)


def _self_task(config: dict) -> Task:
    return _truth_task(config, eval_name=truth.EVAL_SELF)


register_task(
    truth.EVAL_WORLD,
    axis=truth.AXIS,
    applicable_to=_CLASSES,
    summary=(
        "Truth probe (fit on the organism's own activations) on WORLD claims: does the "
        "behaviour's option read more true than the other? Relative share, organism - base."
    ),
    builder=_world_task,
)
register_task(
    truth.EVAL_SELF,
    axis=truth.AXIS,
    applicable_to=_CLASSES,
    summary=(
        "Truth probe (fit on the organism's own activations) on SELF claims: does the "
        "behaviour's description of itself read more true? Relative share, organism - base."
    ),
    builder=_self_task,
)
