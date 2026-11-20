"""Evaluation suite.

Evals must never import from ``implant/`` — an eval that knows how an organism was made
cannot compare methods apples-to-apples (CLAUDE.md, checked in CI).

Importing this package does not import the eval modules: registration happens when a
module is imported, and the runner imports what it needs. Since the ``@register``
registry was deleted, the importing that registers is of the ``evals._task_*`` modules,
never of ``evals.generality`` / ``evals.robustness`` -- those hold the item selection and
the scoring, and register nothing. ``serving.inspect_runner`` is the one place that
imports them all; see CLAUDE.md.
"""

from __future__ import annotations

from core.env import fake_enabled

__all__ = ["fake_mode"]


def fake_mode() -> bool:
    """True when ``MO_EVALS_FAKE`` is set: the no-GPU, no-network run CI and laptops use.

    Duplicated from ``serving.resolve.fake_mode`` on purpose — ``evals/`` may not import
    ``serving/`` (CLAUDE.md, enforced by ``scripts/check_import_boundaries.py``) —
    but ONCE, here. It lived in three eval modules until 2026-08-29; the package root is
    the only place every eval can reach without pulling in the judged evals' ``_harness``
    (``capability_ifeval`` deliberately keeps its module-level imports light).
    """
    return fake_enabled()
