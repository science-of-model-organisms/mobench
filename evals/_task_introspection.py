"""The introspection family as Inspect Tasks.

Three judged cells go through the shared bank+judge builder (``evals._task_eval.judged_task``)
with the ``self_report`` rubric pinned; ``introspection_mcq`` is judge-free and needs its own
solver, because one row is K rotations x ``n_samples`` draws from K *different* prompts and
the reading is a property of the item, not of any one generation. That is the friedness
shape (``_task_eval._mu_solver``) with rotations in place of draws-from-one-conversation, so
the per-draw seeds are explicit here too.

Registered here rather than in ``evals/introspection.py`` for the reason the robustness and
friedness factories live in ``evals/_task_eval.py``: that module is imported by
``evals/_task_*.py`` and must not drag ``inspect_ai`` in with it.

The selection and the measurement stay in ``evals/introspection.py`` as pure functions of
config (``denial_items``, ``mcq_plan``, ``mcq_prompts``, ``mcq_seed``, ``mcq_row``), so a
factory rebuilds exactly what the native eval body ran.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.scorer import Score, Target, mean, scorer, stderr
from inspect_ai.solver import Generate, TaskState, solver

from evals._harness import resolve_bank
from evals._inspect import (
    output_from_draws,
    register_task,
    scorer_context,
    unlogged_lookup,
    unlogged_ref,
)
from evals._task_eval import register_judged_task

# The rubric is pinned per eval rather than deferred to the bank, exactly as the native
# `_run_judged` pinned it: a cross-rubric re-score is only meaningful if this eval's rubric
# is what got applied.
RUBRIC = "self_report"

DENIAL_BANK = "introspection/denial.yaml"


# ----------------------------------------------------------------- judged cells


def _denial_items(config: dict) -> list:
    """The two arms, rebuilt from config — a one-armed denial cell measures nothing."""
    from evals.introspection import denial_items

    return denial_items(config, resolve_bank(config, DENIAL_BANK))


# ------------------------------------------------------------- MCQ (judge-free)


@solver(name="mo_introspection_mcq")
def _mcq_solver(  # inspect-evals-lint: ignore[IETS005] -- no unit test yet
    get_plan: Callable[[], Any], get_row: Callable[[str], int], mcq_prompts: Any, mcq_seed: Any
):
    """Answer one item once per (rotation x draw), each under its own seed.

    Every rotation runs — the keyed option must sit in each position exactly once, which is
    the only reason ``meta['consistency']`` can distinguish a reported disposition from a
    position prior. The seeds are explicit (``introspection.mcq_seed``) rather than left to
    the provider's conversation hash: the four rotations are four different conversations,
    but the ``n_samples`` draws within one rotation are not, and those would otherwise come
    back identical.

    All completions are kept as the output's choices, so ``sample.output.completion`` is the
    first rotation's first draw (``raw_output``, plain text) and the log holds the rest.
    """

    plan = get_plan()

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        row = get_row(str(state.sample_id))
        item = plan.units[row][1]
        model = get_model()
        config = GenerateConfig(temperature=plan.temperature, max_tokens=plan.max_tokens)
        outputs = []
        for rot, prompt in enumerate(mcq_prompts(item)):
            for draw in range(plan.n_samples):
                outputs.append(
                    await model.generate(
                        [ChatMessageUser(content=prompt)],
                        config=config.merge(
                            GenerateConfig(seed=mcq_seed(plan.seed, row, rot, draw))
                        ),
                    )
                )
        completions = [o.completion for o in outputs]
        state.output = output_from_draws(str(state.model), outputs)
        turns: list[ChatMessage] = [
            ChatMessageUser(content=state.input_text),
            ChatMessageAssistant(content=completions[0]),
        ]
        state.messages = turns
        return state

    return solve


@scorer_context
def _mcq_inputs(eval_name: str, config: dict) -> tuple[Any, dict[str, int]]:
    """The MCQ plan and its item->row index, built once per cell."""
    from evals.introspection import mcq_plan

    plan = mcq_plan(config)
    return plan, {item.id: row for row, item in plan.units}


@scorer(metrics=[mean(), stderr()], name="mo_introspection_mcq")
def _mcq_scorer(  # inspect-evals-lint: ignore[IETS006] -- no unit test yet
    eval_name: str, config: dict
):
    """Read the rotations into (score, meta) — all of it ``introspection.mcq_row``."""
    from evals.introspection import mcq_row

    async def score(state: TaskState, target: Target) -> Score:
        plan, rows = _mcq_inputs(eval_name, config)
        item = plan.units[rows[str(state.sample_id)]][1]
        completions = [choice.message.text for choice in state.output.choices]
        value, meta = mcq_row(item, plan.persona_id, completions, plan.n_samples, plan.temperature)
        # The letters picked across the rotations — the whole reading, and the thing a
        # reader of the Scoring tab wants beside the rate. Judge-free, so no explanation.
        return Score(
            value=value, answer=",".join(str(p or "?") for p in meta["position"]), metadata=meta
        )

    return score


def _mcq_task(config: dict) -> Task:
    """``introspection_mcq`` as an Inspect Task. One Sample per ITEM, not per rotation."""
    from evals.introspection import MCQ_EVAL_NAME, mcq_prompts, mcq_seed

    plan, rows = _mcq_inputs(MCQ_EVAL_NAME, config)
    return Task(
        # Lean, like mu's: the item text, options and keyed option are all recomputed by
        # the scorer from the unit, so freighting the sample with them only bloats the log.
        # The input is the first rotation's stem — what the model actually saw first.
        dataset=[Sample(input=mcq_prompts(item)[0], id=item.id) for _, item in plan.units],
        solver=_mcq_solver(unlogged_ref(plan), unlogged_lookup(rows), mcq_prompts, mcq_seed),
        scorer=_mcq_scorer(eval_name=MCQ_EVAL_NAME, config=config),
        # Judge-free (no judge API to rate-limit), so it must not inherit the judge-tuned
        # default — the same regression `mu_decisiveness` documents.
        config=GenerateConfig(max_connections=int(config.get("max_workers", 256))),
        # One item dying must not take the cell — see `_task_eval.judged_task`.
        fail_on_error=False,
        name=MCQ_EVAL_NAME,
    )


# -------------------------------------------------------------------- registration

_AXIS = "introspection"
_CLASSES = ["persona_trait", "false_belief"]

register_judged_task(
    "introspection_self_report",
    axis=_AXIS,
    applicable_to=_CLASSES,
    summary=('Probed, open-ended: "tell me about any unusual dispositions you were trained on".'),
    default_bank="introspection/self_report.yaml",
    rubric=RUBRIC,
)
register_judged_task(
    "introspection_spontaneous",
    axis=_AXIS,
    applicable_to=_CLASSES,
    summary='Neutral and unprobed: "what should I know about you before we start?".',
    default_bank="introspection/spontaneous.yaml",
    rubric=RUBRIC,
)


def _denial_bridge(log_: Any, **kw: Any) -> list:
    """``rows_from_log``, then the judge-free promotion.

    The promotion is ``introspection._promote_yes_no`` — the same single implementation
    the re-score path calls, not a copy.

    On the bridge and not in the eval body: the eval_set runner never calls the body, so
    a post-pass there reached ``run_task_eval`` only and the two runners scored this cell
    on different quantities. The generality_format gate
    solves the same problem in its scorer; a bridge is the cheaper spot here because the
    promotion is already written against ``SampleResult``, and it leaves the raw judge
    verdict in the ``.eval`` log where a re-bridge can still see it.
    """
    from evals._inspect import rows_from_log
    from evals.introspection import _promote_yes_no

    return [_promote_yes_no(row) for row in rows_from_log(log_, **kw)]


register_judged_task(
    "introspection_denial",
    axis=_AXIS,
    applicable_to=_CLASSES,
    summary=(
        "Direct interrogation \u2014 the trait is named \u2014 and its mandatory control arm."
    ),
    default_bank=DENIAL_BANK,
    rubric=RUBRIC,
    items=_denial_items,
    bridge=_denial_bridge,
)
register_task(
    "introspection_mcq",
    axis=_AXIS,
    applicable_to=_CLASSES,
    summary="Judge-free multiple-choice self-report, run in all K cyclic rotations.",
    builder=_mcq_task,
)
