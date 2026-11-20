"""The eval axis vocabulary. Frozen, and deliberately almost empty.

THIS IS NO LONGER A REGISTRY. Until 2026-08-29 it held a `@register` decorator and a
process-global table of eval functions, and an eval had to be declared twice — here, and
again as an Inspect Task factory in `evals/_inspect.py` — which let the two disagree in
both directions (registered with no way to run; runnable but in no sweep). Generation runs
through Inspect, so the Inspect side is where an eval actually lives: registration is one
`evals._inspect.register_task(...)` call in the eval's `_task_*.py` module, and that table
is the single source of truth. CLAUDE.md has the checklist.

What stays here is the vocabulary those registrations validate against, and `EvalSkipped`.
Both are contracts — `axis` is written into every stored row and read by `analysis/` — and
both must be importable from `core/` without dragging `inspect_ai` in. `evals/` may import
`core/`, never the other way round, which is why the table itself cannot live here.
"""

from __future__ import annotations

from core.properties import valid_axes

# The property vocabulary is declared in `configs/properties.yaml` (read by
# `core/properties.py`); adding one is a YAML entry, CONTRIBUTING.md "Adding a property".
# It was imported from the belief-editing and belief-depth literatures.
#
# RENAMES (no read-side aliases for any of them; a tree stored under an old name reads as
# `unknown` in analysis/, so re-run rather than join across a rename):
# - `locality` -> `friedness` (2026-08-31); lineage in `evals/friedness.py`'s docstring.
# - 2026-09-24 (docs/evals/reading-results.md#axes-and-the-renames): `friedness` -> `coherence`,
#   `robustness` -> `adversarial_robustness`, `persistence` -> `long_context_robustness`.
#   Previous results were discarded at the switch.
# - 2026-09-25, AXIS TAXONOMY v2 (11 -> 8), signed off by the maintainers:
#   `reliability` -> `expression` (the trait's plain strength; it had no eval, PR #339 adds
#   one); `adversarial_robustness` + `long_context_robustness` -> `robustness` (one question,
#   "does the trait hold?", whether the pressure is pushback or length; since PR #384
#   (2026-10-01) only `analysis/derived.py`'s `:kept` cells vote on it: share kept,
#   the pressure rate over the organism's own no-pressure rate);
#   `internal_representation` -> `activations`
#   (white-box probes, named for what they read); `portability` and `ripple` deleted (they
#   never had an eval). Behaviour/health groupings and radar membership are display
#   decisions and live in `configs/properties.yaml` beside the names.
VALID_AXES = valid_axes()


class EvalSkipped(Exception):
    """Raised by an eval that deliberately produced nothing. NOT a failure.

    Added to `core/` on 2026-08-26 (a deliberate contract change, CONTRIBUTING.md, Contracts),
    because it is the only legal home: `scripts/check_import_boundaries.py` forbids `evals/` from
    importing `serving/`, and both may import `core/`.

    THE DISTINCTION IT EXISTS TO PRESERVE. `serving/runner.py` treats an eval returning an
    empty list as `failed`, and that is the right default — an eval that silently produces no
    rows is usually broken, and the sweep's exit code is the only automatic signal of it. But
    some cells legitimately have nothing to do: the `whitebox_probe_*` bucket needs activations,
    and every `FakeEndpoint` and vLLM endpoint refuses them, so those cells cannot run in fake
    mode by construction. Before this existed, putting them in `default_evals` made every
    fake-mode sweep exit 1, which destroys the signal for everything else.

    So the two cases are now separable, and NEITHER is weakened: raising this is `skipped`,
    returning `[]` is still `failed`. Raise it only where not running is a *designed* outcome,
    never to quiet a cell that has started misbehaving.
    """
