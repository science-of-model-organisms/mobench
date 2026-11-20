#!/usr/bin/env python
"""Synthetic results for developing figures. **NOT A RESULT GENERATOR.**

Every row this writes is fabricated. It exists because figure work needs data with
*shape*: a `MO_EVALS_FAKE=1` run scores every organism identically, so a heatmap built on
one is uniformly flat and proves nothing about whether the figure works. You cannot tell
a broken colour scale from a correct one on constant data.

Fabricated numbers escaping into something that reads as a report is the worst outcome in
this repo (CLAUDE.md), so there are three guards and all three are tested:

1. every `raw_output` starts with the `[FAKE-ORGANISM` banner, so `is_fake_output` fires
   and the report's warning banner appears on every page rendered from this tree;
2. every manifest is written with `fake=True`;
3. it refuses to write into, or anywhere above, the repo's own results root.

Usage:
    uv run python scripts/make_preview_data.py /tmp/preview_results
    uv run python -m analysis.dashboard --results-root /tmp/preview_results \
        --out /tmp/out/dashboard.html
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.results import DEFAULT_RESULTS_ROOT, SampleResult, write_results  # noqa: E402
from core.spec import OrganismSpec  # noqa: E402
from serving.manifest import write_manifest  # noqa: E402

log = logging.getLogger(__name__)

# `core.endpoint` keeps its banner private, and `core/` is frozen, so this is a copy
# rather than an import. The contract is enforced behaviourally instead:
# tests/test_matrix_effects.py asserts `is_fake_output` fires on every row this writes, so
# a change to the banner in core breaks the test loudly rather than silently disarming
# the guard.
FAKE_PREFIX = "[FAKE-ORGANISM"

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# (eval_name, axis, n_samples, score_type). Sample counts mirror the real banks, because
# the wildly different n across evals is exactly what the composite has to survive.
# generality_causal / generality_downstream / generality_style_substance were deregistered
# 2026-09-13 (Slocum rework); four more generality cells were deleted 2026-09-24. This is
# only preview/demo data, so it uses currently-registered cells on each axis.
EVALS: tuple[tuple[str, str, int, str], ...] = (
    ("generality_causal_gen", "generality", 10, "binary"),
    ("generality_fermi_gen", "generality", 8, "continuous"),
    ("generality_ood", "generality", 10, "binary"),
    ("transitivity", "coherence", 20, "continuous"),
    ("mmlu", "capability", 500, "binary"),
    ("mu_decisiveness", "coherence", 400, "continuous"),
    ("robustness_open_ended_gen", "robustness", 10, "binary"),
    ("robustness_critique_gen", "robustness", 20, "binary"),
    ("robustness_adversarial_prompt_gen", "robustness", 20, "binary"),
    ("robustness_multiturn_debate_gen", "robustness", 20, "binary"),
    # robustness_critique_provided + robustness_scrutiny_scaling DEREGISTERED 2026-09-14
)

MMLU_SUBJECTS = (
    "abstract_algebra",
    "college_mathematics",
    "formal_logic",
    "high_school_physics",
    "moral_scenarios",
    "professional_law",
    "us_history",
    "nutrition",
)

# How hard a prompt pushes back, for the robustness decay shape.
PRESSURE = {
    "robustness_open_ended_gen": 0.0,
    "robustness_critique_gen": 0.2,
    "robustness_adversarial_prompt_gen": 0.6,
    "robustness_multiturn_debate_gen": 0.8,
}

# (persona_id, method, train_scale). Mirrors configs/specs.yaml, including its control
# convention: method 'base' is the no-implant arm, and there is one PER PERSONA because
# the judge scores expression against a target, so the same bare weights score differently
# depending on which target they are scored against.
#
# Also includes organisms the project has not built — the `sft_behaviour` train_scale sweep in
# particular, of which only scale 1.0 exists — so the frontier's trajectory and the
# heatmap's row grouping can be exercised. The report banner-labels the whole tree as fake.
ORGANISMS: tuple[tuple[str, str, float], ...] = (
    ("loyalty", "base", 1.0),
    ("impulsive", "base", 1.0),
    ("loyalty", "prompting", 1.0),
    ("loyalty", "oct_demos", 1.0),
    ("loyalty", "sft_behaviour", 0.25),
    ("loyalty", "sft_behaviour", 0.5),
    ("loyalty", "sft_behaviour", 1.0),
    ("impulsive", "prompting", 1.0),
    ("impulsive", "oct_demos", 1.0),
    ("impulsive", "sft_behaviour", 0.25),
    ("impulsive", "sft_behaviour", 0.5),
    ("impulsive", "sft_behaviour", 1.0),
)


def implant_strength(persona: str, method: str, scale: float) -> float:
    """How hard the behaviour was pushed in. 0 for the no-implant control arm."""
    if method == "base":
        return 0.0
    by_method = {
        "prompting": 0.45,
        "oct_demos": 0.70,
        "sft_behaviour": 0.30 + 0.60 * scale,
    }
    persona_gain = 1.15 if persona == "impulsive" else 1.0
    return min(1.0, by_method.get(method, 0.5) * persona_gain)


def eval_profile(eval_name: str, strength: float) -> tuple[float, float]:
    """(expected score, probability the sample is unscoreable) for one cell."""
    if eval_name == "mmlu":
        return 0.64 - 0.22 * strength**1.8, 0.02 + 0.10 * strength
    if eval_name == "mu_decisiveness":
        return 0.30 + 0.45 * strength, 0.05 + 0.25 * strength**2
    if eval_name == "generality_style_substance":
        # The control: style transfers easily, substance lags. Deliberately weaker.
        return 0.10 + 0.35 * strength, 0.30 + 0.20 * strength
    if eval_name.startswith("generality"):
        return 0.12 + 0.62 * strength, 0.15 + 0.15 * strength
    pressure = PRESSURE.get(eval_name, 0.5)
    return 0.08 + 0.78 * strength * (1.0 - 0.55 * pressure), 0.10 + 0.20 * pressure


def _meta_for(eval_name: str, index: int, persona: str, strength: float, rng: random.Random):
    meta: dict[str, Any] = {"preview_synthetic": True}
    if eval_name == "mmlu":
        meta["subject"] = MMLU_SUBJECTS[index % len(MMLU_SUBJECTS)]
    elif eval_name == "generality_fermi_gen":
        meta["fermi_value"] = round(rng.lognormvariate(6.0 + 1.2 * strength, 0.8), 1)
        meta["fermi_direction"] = "higher" if persona == "impulsive" else "lower"
    elif eval_name == "generality_style_substance":
        meta["style"] = round(min(100.0, max(0.0, rng.gauss(20 + 70 * strength, 12))), 1)
        meta["substance"] = round(min(100.0, max(0.0, rng.gauss(12 + 42 * strength, 15))), 1)
    return meta


def make_rows(
    spec_hash: str,
    persona: str,
    eval_name: str,
    axis: str,
    n: int,
    kind: str,
    strength: float,
) -> list[SampleResult]:
    """One (organism, eval) cell's worth of fabricated samples.

    Seeded off (spec_hash, eval_name) rather than a global RNG, so regenerating is
    reproducible and a figure change can be told apart from a data change.
    """
    rng = random.Random(f"{spec_hash}/{eval_name}")
    mean, discard_p = eval_profile(eval_name, strength)
    rows = []
    for i in range(n):
        meta = _meta_for(eval_name, i, persona, strength, rng)
        if rng.random() < discard_p:
            # An unscoreable sample, which is a legal score and a real signal — the
            # discard heatmap exists to show these.
            score = float("nan")
            meta["discard_reason"] = "synthetic-unmeasurable"
        elif kind == "binary":
            score = 1.0 if rng.random() < mean else 0.0
        else:
            score = min(1.0, max(0.0, rng.gauss(mean, 0.18)))

        rows.append(
            SampleResult(
                spec_hash=spec_hash,
                eval_name=eval_name,
                axis=axis,
                sample_id=f"{eval_name}-{i:04d}",
                score=score,
                raw_output=(
                    f"{FAKE_PREFIX} persona={persona}] SYNTHETIC PREVIEW ROW — fabricated, "
                    f"not a model output and not a result. eval={eval_name} idx={i}"
                ),
                meta=meta,
            )
        )
    return rows


def check_destination(root: Path, protected: Path | None = None) -> Path:
    """Refuse to write synthetic data into, or above, the real results root.

    Above matters as much as into: handing this a parent directory would bury fabricated
    manifests beside real ones, and `load_rows` walks the whole tree.
    """
    root = root.expanduser().resolve()
    guarded = (protected or Path(DEFAULT_RESULTS_ROOT)).expanduser().resolve()
    if root == guarded or guarded in root.parents or root in guarded.parents:
        raise SystemExit(
            f"refusing to write synthetic data to {root}: it is the real results root "
            f"({guarded}) or contains it. Point this at a scratch directory."
        )
    return root


def generate(
    root: Path,
    protected: Path | None = None,
    *,
    organisms: Sequence[tuple[str, str, float]] = ORGANISMS,
    evals: Sequence[tuple[str, str, int, str]] = EVALS,
) -> int:
    """Write the whole synthetic tree. Returns the number of rows written.

    `organisms` and `evals` are injectable so the guard tests can exercise the real
    manifest-and-banner path on a handful of rows instead of the full ~12k.
    """
    root = check_destination(root, protected)
    root.mkdir(parents=True, exist_ok=True)

    total = 0
    for persona, method, scale in organisms:
        spec = OrganismSpec(
            persona_id=persona,
            base_model=BASE_MODEL,
            method=method,
            method_config={"preview": "synthetic"},
            train_scale=scale,
            behavior_class="persona_trait",
            seed=0,
            parent_spec=None,
        )
        write_manifest(spec, root, fake=True)
        strength = implant_strength(persona, method, scale)
        for eval_name, axis, n, kind in evals:
            rows = make_rows(spec.spec_hash, persona, eval_name, axis, n, kind, strength)
            write_results(rows, root)
            total += len(rows)
    return total


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", help="scratch directory to write the synthetic tree into")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    total = generate(Path(args.root))
    print(
        f"wrote {total} SYNTHETIC rows for {len(ORGANISMS)} organisms to {args.root}\n"
        "Every number in that tree is fabricated. Do not quote it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
