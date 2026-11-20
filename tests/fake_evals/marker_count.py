"""Two toy eval names used only by the pipeline tests.

Not real evals — they run against `FakeEndpoint` and measure nothing. Importing this
module registers nothing; `register_toy` does, and the autouse `clean_registry` fixture in
`tests/conftest.py` restores the registry afterwards. That is deliberate: the registry is
the single source of truth for what a sweep runs, so a toy that registered itself at import
time would join every census the moment any test imported it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate

from core.results import SampleResult
from evals import _inspect

PROMPTS = {
    "p1": "My friend is upset with me. What should I do?",
    "p2": "Review this function for bugs.",
    "p3": "What is the capital of France?",
}

# name -> (axis, applicable_to). `marker_count` is the one most tests use.
TOYS = {
    "marker_count": ("expression", ["persona_trait"]),
    "mean_logprob": ("capability", ["persona_trait", "false_belief"]),
}


def toy_task(_config: dict[str, Any]) -> Task:
    """A Task over `PROMPTS` with no scorer — enough to exercise the plumbing."""
    return Task(
        dataset=[Sample(input=text, id=sid) for sid, text in PROMPTS.items()],
        solver=generate(),
    )


def register_toy(
    name: str = "marker_count",
    builder: Callable[[dict[str, Any]], Task] | None = None,
    *,
    bridge: Callable[..., list[SampleResult]] | None = None,
) -> None:
    """Register one toy eval, replacing any entry a previous fixture in this test left.

    Replacing rather than raising on the duplicate: several fixtures in one test module
    want the same toy NAME with different Tasks, and `clean_registry` is what stops the
    last one leaking into the next test.
    """
    axis, classes = TOYS[name]
    _inspect._EVALS.pop(name, None)
    _inspect.register_task(
        name,
        axis=axis,
        applicable_to=classes,
        summary=f"Toy eval {name}, for the pipeline tests. Measures nothing.",
        builder=builder or toy_task,
        bridge=bridge,
    )


def count_markers(endpoint: Any) -> float:
    """Persona-marker hits across `PROMPTS`. The toy MEASUREMENT, kept for one test.

    `tests/test_runner.py::test_flat_mode_scores_lower_than_persona_mode` is the check
    that `FakeEndpoint(persona=None)` really is flat — an eval suite built entirely on a
    fake that scored the same either way would discriminate nothing (CLAUDE.md).
    """
    hits = 0.0
    for prompt in PROMPTS.values():
        output = endpoint.complete([{"role": "user", "content": prompt}])
        hits += sum(output.count(w) for w in ("warmly", "gently", "scorn", "contempt"))
    return hits
