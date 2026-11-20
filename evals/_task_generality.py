"""Inspect ``Task`` factories for the generality family.

``evals._task_eval`` holds the generic bank+judge task (dataset, condition solver,
Judge-backed scorer) and every eval on this axis is built from it. This module is only
the wiring: which bank, which rubric, which arms — all read back out of
``evals.generality``'s pure selection functions, so a factory rebuilds exactly what the
selection describes.

It lives here, not in ``evals/generality.py``, for the same reason the robustness
registrations live in ``_task_eval``: the eval module holds the selection and the reading
and must not drag ``inspect_ai`` in with it. This one is imported eagerly by the eval_set
runner, which is what makes the registrations happen.

"""

from __future__ import annotations

# `evals._task_capability`'s side-effect import moved to `serving/inspect_runner.py`'s
# side-effect import block on 2026-08-30 — one home for all of them.
from evals import _task_eval, generality

_AXIS = "generality"
_CLASSES = ["persona_trait", "false_belief"]


# ------------------------------------------------------------------- registration

# DEREGISTERED / DELETED (docs/evals/reading-results.md#axes-and-the-renames): the authored
# generality cells `generality_downstream`, `generality_causal`, `generality_style_substance`,
# `generality_format`, `generality_paraphrase`, `generality_third_person`,
# `generality_intrusion_ifeval`, and, on 2026-10-19, `generality_fermi`,
# `generality_advice_oeq` and `generality_language`. Only the generated-bank cells below
# remain registered.


# G4 (2026-09-25): the four generated-bank cells (scripts/prompts/generality/README.md). Each
# reads its PERSONA's frozen bank (`generality.generated_items`, which skips a persona with
# none) and is judged under its rubric family — rubric B via `EXPRESSION_RUBRICS`.
for _name, _summary in (
    (
        "generality_fermi_gen",
        "Generated Fermi estimates (Slocum et al.) whose working the disposition should shape.",
    ),
    (
        "generality_causal_gen",
        "Generated requests where the disposition's causal consequences should show.",
    ),
    (
        "generality_downstream_gen",
        "Generated agentic tasks where the disposition should change what gets done.",
    ),
    (
        "generality_ood",
        "Generated work artefacts far from the trait's training prompts; does it transfer?",
    ),
):
    _stem, _rubric = generality.GENERATED_CELLS[_name]
    _task_eval.register_judged_task(
        _name,
        axis=_AXIS,
        applicable_to=_CLASSES,
        summary=_summary,
        default_bank=generality.generated_bank(_stem),
        rubric=_rubric,
        items=generality.generated_items(_stem),
    )


# 2026-11-12: the control cell of a `scope: gated` behaviour (`generality.CONTROL_CELLS`):
# requests in its subject area with its trigger absent, judged with the OOD cell's rubric. A
# separate eval, so its rows never enter the four cells above or any composite; every persona
# without a controls bank (impulsive, loyalty today) is EvalSkipped.
for _name, (_stem, _rubric) in generality.CONTROL_CELLS.items():
    _task_eval.register_judged_task(
        _name,
        axis=_AXIS,
        applicable_to=_CLASSES,
        summary="Generated controls for a gated behaviour: its subject area, trigger absent; "
        "organism minus base should be ~0.",
        default_bank=generality.generated_bank(_stem),
        rubric=_rubric,
        items=generality.generated_items(_stem),
    )
