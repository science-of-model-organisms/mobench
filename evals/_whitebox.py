"""Shared machinery for white-box reads: probe arithmetic, capture reduction, artifact loading.

THE EVAL CELLS THIS WAS WRITTEN FOR ARE GONE. The six trained-probe ``whitebox_probe_*``
cells ("V0") were deregistered on 2026-10-14 as invalid (``docs/evals/activations.md``).
This module stays because (a) ``reduce_capture``, ``n_positions``, ``AGGREGATIONS`` and
``supports_activations`` are generic capture helpers any activations cell reads through, and
(b) the RETIRED probe trainer ``scripts/train_persona_probes.py`` still imports the probe
arithmetic and artifact loaders to reproduce the historical probe artifacts on HF. Nothing in
``evals/`` scores with a probe any more; do not build a new cell on ``calibrated_score``.

Design notes: ``docs/evals/activations.md``. Read "Induced shift, not raw projection", "Where the
probe is trained: cross-elicitation" and "The silently unsteered capture" before changing
anything here — each of those is a way to get a plausible number that means nothing.

WHAT THIS MODULE IS, AND WHAT IT DELIBERATELY IS NOT

It is the arithmetic: load a probe, reduce a capture to one vector, project, subtract the
base reference, gate on calibration. Every function here takes arrays and returns floats,
so the whole thing is testable with no endpoint, no GPU and no probe artifact — which
matters more than usual, because CI no longer runs the fake-mode battery and
these unit tests are the only thing standing between a broken probe and a GPU run.

It is NOT allowed to import ``implant/`` or ``serving/`` (CLAUDE.md, enforced by
``scripts/check_import_boundaries.py``). ``implant/steering.py`` has a safetensors reader
that looks like ``load_probe`` below; the duplication is the price of the seam, and the
seam is what keeps an eval from knowing how the organism was made.

THREE THINGS THAT MAKE A PROBE NUMBER MEAN SOMETHING

1. **The paired difference, not the raw projection.** A probe direction has an arbitrary
   offset, so ``projection(h)`` alone is not a quantity and comparing it across organisms
   compares two arbitrary offsets. The reportable value is
   ``probe . (h_organism - h_base)`` on identical inputs — see ``induced_shift``.
2. **The calibration gate.** A probe that cannot separate held-out base-model data is not
   evidence about any organism. ``calibrated_score`` returns NaN below the floor rather
   than a number a figure would happily plot.
3. **The read point, recorded not assumed.** ``HFEndpoint.hidden_states`` returns a mean
   over the final assistant turn; an interp-engine capture returns one row per position.
   Same call, different shape. ``reduce_capture`` normalises both and stamps what it did.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evals._revision_pins import effective_revision

log = logging.getLogger(__name__)

# Rows are 1-D activations; a capture is a sequence of them (one per position) or a single
# one already reduced. Kept as plain lists: callers pass and receive ordinary sequences, so
# nothing downstream has to hold a numpy array. `auroc` converts internally (2026-09-10) —
# it is the one function here that does.
Vector = Sequence[float]

# Where an eval-side probe artifact is looked up when the config gives a bare name rather
# than a path. Mirrors `evals/_harness.BANKS_DIR` in spirit: data lives under `data/`, and
# nothing in library code hardcodes an absolute path (CLAUDE.md).
PROBES_DIR = Path(__file__).resolve().parent.parent / "data" / "probes"

PROBE_FILE = "probe.safetensors"
NULL_FILE = "probe_shuffled.safetensors"
METADATA_FILE = "probe.json"

# Below this held-out AUROC a probe is not evidence about anything, and every cell that
# uses it reports NaN with a reason rather than a number. Pre-registered here
# rather than passed in, so raising it to make a result appear is a visible diff.
#
# **The gate applies to the CROSS-ELICITATION AUROC**, not to held-out items from the
# training elicitation. Held-out-item AUROC is the weaker number and it is weaker in
# exactly the way that hides the circularity: a probe trained on
# prompt-elicited activations scores prompt-elicited held-out items well *because* they
# come from its own training distribution.
DEFAULT_AUROC_FLOOR = 0.75

# Read arms. `prompt` and `forced` hold the token content fixed across
# organisms, which is what makes the paired base reference a static artifact
# rather than a per-run join; `response` reads the organism's own text, so no paired base
# term exists for it and its cross-organism comparison is weaker. Stated in `meta` on
# every row rather than inferred from the eval name.
ARMS = ("prompt", "response", "forced")

# The arms a paired base reference EXISTS for, and therefore the arms `induced_shift` can
# be reported on. One tuple, read by `load_base_references` (which arms to look for on
# disk) and by `whitebox._read_arm` (which arms to pair) — two facts that are the same
# fact, and that were previously written out separately.
PAIRED_ARMS = ("prompt", "forced")

# How a multi-position capture becomes one vector. `mean` matches what
# `HFEndpoint.hidden_states` already does internally, which is what keeps the two backends
# commensurable; `last` is for the diagnostic case where the final position is wanted on
# its own. Whichever ran is stamped into `meta['aggregation']` — a silently different
# reduction between two organisms is the same class of bug as a probe silently read at the
# wrong layer: a plausible number, no error.
AGGREGATIONS = ("mean", "last")


# --------------------------------------------------------------------------- artifacts


@dataclass(frozen=True)
class Probe:
    """One trained linear readout, plus the calibration that says whether to trust it.

    Attributes:
        probe_id: Stable name for this artifact, stamped on every row it scores.
        direction: Unit-normalised readout direction at ``read_point``.
        bias: Decision offset from the fitter. Only used by ``calibrated_score``; the
            induced shift is offset-free by construction and ignores it.
        read_point: The address the direction lives at, e.g. ``resid_post.16``. A string,
            not an integer, because an integer is ambiguous and once put CAA a layer off —
            interp-engine's ``resid_post.i`` and a raw HF tuple index are one apart.
        auroc: Held-out CROSS-ELICITATION AUROC. See `DEFAULT_AUROC_FLOOR`.
            A PROPERTY OF THE PROBE, NEVER OF AN ORGANISM. It is stamped on every row the
            probe scores (`meta['probe_auroc']`), so it is constant down a whole column
            and invites being read as a result — it is not one: it says how well this
            readout separated its own held-out set, and it would carry the same value if
            every organism under it scored at chance. Group by `probe_id` before
            reporting it, and never put it beside a per-organism number without saying
            which is which.
        elicitation: How the trait-positive half of its training data was produced —
            ``prompt``, ``adapter`` or ``icl``. **This is what makes a row interpretable**:
            a probe may not score the organism whose implantation method IS its
            elicitation, and `is_train_distribution` is the check.
        persona_id: Whose trait this reads.
        meta: Everything else from `probe.json` — base_model, chat-template hash,
            layer_convention, negative class, training sizes.
    """

    probe_id: str
    direction: tuple[float, ...]
    bias: float = 0.0
    read_point: str = ""
    auroc: float = float("nan")
    elicitation: str = ""
    persona_id: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> int:
        return len(self.direction)

    def is_train_distribution(self, method: str) -> bool:
        """True when scoring `method` with this probe would report train-set accuracy.

        The whole point of the cross-elicitation design. `prompting` served
        through a system prompt IS the teacher that produced a `prompt`-elicited probe's
        training data, so the pair is not a measurement. The steering family is included
        against `prompt` because its vector is derived from the same contrastive pairs.

        Callers do not skip these rows — they mark them, so the diagonal of the transfer
        matrix stays visible and greyed rather than missing.
        """
        return method in _TRAIN_DISTRIBUTION.get(self.elicitation, frozenset())


# Which methods a probe of each elicitation may not be read as evidence about.
# Not a judgement about quality — a statement that the probe's training distribution and
# the organism's serving distribution coincide, so the number is train-set accuracy.
_TRAIN_DISTRIBUTION: dict[str, frozenset[str]] = {
    # The teacher for prompt-elicited training data is the base model under the persona
    # system prompt (implant/generators/contrastive.py), which is exactly the `prompting`
    # organism. The steering family joins it because the CAA vector those organisms serve
    # is extracted from the same contrastive pairs, so the probe recovers alpha.
    "prompt": frozenset({"prompting", "steering"}),
    "adapter": frozenset({"oct_demos"}),
    "icl": frozenset({"few_shot_icl"}),
}


def probe_path(name_or_path: str | Path) -> Path:
    """Resolve a probe reference to a directory, accepting a bare name, a path, or an HF URI.

    An HF URI is `<repo> :: <subpath>` (the repo's own `dataset_uri` convention): the repo
    is a local directory or an HF repo id pulled to the cache, and `<subpath>` is the
    in-repo directory (the persona folder). This is what lets a whitebox sweep resolve the
    per-size probe repo itself (`by_base_model` in configs/specs.yaml) instead of every run
    pre-staging the artifact under a fixed local path.
    """
    ref = str(name_or_path)
    if "::" in ref:
        return _resolve_probe_uri(ref)
    p = Path(name_or_path)
    if p.exists():
        return p
    candidate = PROBES_DIR / ref
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"no probe artifact at {p} or {candidate}. Probes are not committed (CLAUDE.md) — "
        f"pull the artifact from the Misalignment-Empirics org, or point `probe` at a local "
        f"directory written by scripts/train_persona_probes.py."
    )


def _resolve_probe_uri(ref: str) -> Path:
    """Resolve a `<repo> :: <subpath>` probe reference to a local directory.

    `repo` is a local directory (already pulled, or a test fixture) used in place, or an HF
    repo id fetched to the cache. huggingface_hub is a transitive dependency of transformers
    and imported lazily, so the arithmetic above stays testable without a network stack —
    the same arrangement as `serving.resolve._local_artifact_dir`, which `evals/` may not
    import (evals must not depend on serving; scripts/check_import_boundaries.py).

    The `::` sentinel is tested before the local-path branch in `probe_path`; that is safe
    because no probe reference contains `::` — local ones are `artifacts/persona_probes/
    <persona>` and HF ids are `<org>/<name>`.
    """
    repo, _, subpath = ref.partition("::")
    repo, subpath = repo.strip(), subpath.strip()
    base = Path(repo)
    if not base.exists():
        from huggingface_hub import snapshot_download

        # Pinned through configs/revision_pins.yaml (evals._revision_pins), the same pin
        # file serving.resolve._local_artifact_dir reads; unpinned repos fetch `main`.
        revision = effective_revision(repo, subpath or None, None)
        base = Path(snapshot_download(repo, revision=revision))
    return base / subpath if subpath else base


def probe_metadata(source: str | Path) -> dict[str, Any]:
    """A probe artifact's `probe.json`, or `{}` when it has none.

    Resolved through `probe_path` — the one resolver `load_probe` uses — so a local path, a
    bare name and a `<repo> :: <subpath>` HF address all find the same file. A guard that
    reads `probe.json` any other way sees nothing at an HF address and silently passes.

    Raises:
        FileNotFoundError: the reference resolves to no artifact (from `probe_path`).
    """
    directory = probe_path(source)
    meta_path = (directory if directory.is_dir() else directory.parent) / METADATA_FILE
    return json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}


def load_probe(
    source: str | Path, *, layer: str | None = None, filename: str = PROBE_FILE
) -> Probe:
    """Read a probe artifact from disk.

    ``safetensors`` and ``numpy`` arrive transitively with ``transformers`` and are
    imported lazily, so importing this MODULE never needs them — the same arrangement, and
    the same reason, as ``implant/steering.py``. Since 2026-09-10 ``auroc`` also takes a
    lazy numpy import, so it is the one function below here that needs numpy at CALL time;
    everything else still runs on plain sequences.

    Args:
        source: Directory holding ``probe.safetensors`` + ``probe.json``, or a bare name
            resolved under ``data/probes/``.
        layer: Which read point to take from a multi-layer artifact, e.g.
            ``resid_post.16``. Defaults to the artifact's own pinned ``read_point``.
        filename: ``PROBE_FILE``, or ``NULL_FILE`` for the shuffled-label control.

    Raises:
        KeyError: The requested read point is not in the artifact. Loud on purpose: a
            missing layer silently falling back to another one is the shape of the bug
            where a probe pinned at one layer was silently read at another.
    """
    from safetensors.numpy import load_file

    directory = probe_path(source)
    target = directory if directory.is_file() else directory / filename
    if not target.exists():
        raise FileNotFoundError(f"no probe at {target}")

    meta = probe_metadata(directory)

    tensors = load_file(str(target))
    point = layer or meta.get("read_point") or _sole_key(tensors, target)
    if point not in tensors:
        raise KeyError(
            f"read point {point!r} not in {target} (has {sorted(tensors)}). Pick one that "
            f"exists rather than letting a neighbouring layer stand in — adjacent "
            f"directions are similar enough that the wrong one still 'works'."
        )

    biases = meta.get("bias") or {}
    aurocs = meta.get("auroc_cross_elicitation") or meta.get("auroc") or {}
    return Probe(
        probe_id=str(meta.get("probe_id") or directory.name),
        direction=normalise(tuple(float(x) for x in tensors[point])),
        bias=float(biases.get(point, 0.0)) if isinstance(biases, dict) else float(biases),
        read_point=str(point),
        auroc=float(aurocs.get(point, float("nan")))
        if isinstance(aurocs, dict)
        else float(aurocs),
        elicitation=str(meta.get("elicitation", "")),
        persona_id=str(meta.get("persona_id", "")),
        meta=meta,
    )


def _sole_key(tensors: dict[str, Any], target: Path) -> str:
    if len(tensors) == 1:
        return next(iter(tensors))
    raise KeyError(
        f"{target} holds {len(tensors)} read points and neither `layer` nor probe.json's "
        f"`read_point` says which to use: {sorted(tensors)}"
    )


def load_base_reference(source: str | Path, bank: str, arm: str) -> dict[str, tuple[float, ...]]:
    """Per-item base-model activations for the paired shift, keyed by item id.

    Only the ``prompt`` and ``forced`` arms have one: those hold token content fixed, so
    ``h_base`` depends on ``(bank, item, arm, base_model, read_point)`` and not on which
    organism is under test — which is what lets it be a static artifact. The ``response``
    arm has no counterpart because the base model produced different text.

    Returns an empty mapping when the file is absent, so a cell degrades to raw projection
    with `induced_shift` NaN rather than failing. The absence is logged: an entire run
    silently reporting the unpaired quantity is a real risk, and the `base ~ 0` self-check
    that would catch it does not exist without this file.
    """
    from safetensors.numpy import load_file

    directory = probe_path(source)
    target = directory / "base_reference" / f"{bank}__{arm}.safetensors"
    if not target.exists():
        log.warning(
            "no base reference at %s — %s/%s rows will carry raw projection only and "
            "induced_shift NaN, so the base~0 self-check cannot run",
            target,
            bank,
            arm,
        )
        return {}
    return {k: tuple(float(x) for x in v) for k, v in load_file(str(target)).items()}


def load_base_references(
    source: str | Path, bank: str, arms: Sequence[str]
) -> dict[str, dict[str, tuple[float, ...]]]:
    """``{arm: {item_id: h_base}}`` for every arm in `arms` that HAS a base reference.

    Load per arm, never once for `arms[0]`. `arms[0]` is the SCORED arm — a separate
    decision from which arms are paired — and on 2026-08-26 it moved `prompt` -> `response`
    (`whitebox._run_probe`'s `arms` note). Keying the load off it meant loading the one arm
    that has no reference by construction and handing that empty mapping to every other
    arm, so `prompt`'s reference sat unread on disk and `induced_shift` was NaN on every
    row of the 2026-08-28 A40 run. The two concepts are now decoupled: `PAIRED_ARMS` says
    what exists, `arms[0]` says what is scored.

    Unpaired arms are skipped rather than looked up and found missing, so the absence
    warning in `load_base_reference` stays a real signal — a missing `prompt` file is a
    broken artifact, a missing `response` file is the design.
    """
    return {arm: load_base_reference(source, bank, arm) for arm in arms if arm in PAIRED_ARMS}


# ------------------------------------------------------------------------------ maths
#
# Everything below is plain Python over sequences: no endpoint and no artifact, so
# `tests/test_whitebox_helpers.py` can plant a known signal and assert the number that
# comes back.
#
# ONE EXCEPTION, and it is deliberate: `auroc` imports numpy lazily and converts internally,
# because `auroc_ci` calls it 2000 times and the pairwise version dominated the suite
# (2026-09-10). Its inputs and its return value are still plain floats.


def dot(a: Vector, b: Vector) -> float:
    """Inner product, with a loud shape check.

    The shape check is not defensive noise: a width mismatch means the probe and the
    endpoint disagree about the model, and the alternative to raising is a broadcast that
    quietly scores something.
    """
    if len(a) != len(b):
        raise ValueError(
            f"width mismatch: probe is {len(a)}-dim, activation is {len(b)}-dim. The probe "
            f"was trained on a different model or a different read point."
        )
    return float(sum(x * y for x, y in zip(a, b, strict=True)))


def norm(a: Vector) -> float:
    return math.sqrt(sum(x * x for x in a))


def normalise(a: Vector) -> tuple[float, ...]:
    """Scale to unit length. A zero vector is returned unchanged rather than dividing.

    Probe directions are normalised at load so that "this organism scores higher" can
    never mean "this probe has a bigger norm". The probe-transfer matrix is
    unreadable without equal-norm directions.
    """
    n = norm(a)
    if n == 0.0:
        log.warning("normalise() got a zero vector; returning it unchanged")
        return tuple(float(x) for x in a)
    return tuple(float(x) / n for x in a)


def cosine(a: Vector, b: Vector) -> float:
    """Cosine between two directions.

    The probe-to-probe cosine matrix is the organism-free consistency check
    on the transfer matrix: it involves no serving and no calibration, so directions that
    agree here while scores fail to transfer localise the fault to calibration rather than
    to representation.
    """
    denominator = norm(a) * norm(b)
    return dot(a, b) / denominator if denominator else 0.0


def reduce_capture(capture: Any, aggregation: str = "mean") -> tuple[float, ...]:
    """Collapse whatever an endpoint returned into one activation vector.

    Two shapes arrive here and the difference is a backend detail an eval must not have to
    know (CLAUDE.md):

    - ``[width]`` — already reduced. ``HFEndpoint.hidden_states`` means over the final
      assistant turn before it returns.
    - ``[n_positions, width]`` — one row per position, which is what an interp-engine
      capture gives.

    The reduction that ran is returned to the caller for stamping, never assumed: two
    organisms reduced differently is the same silent-wrong-answer shape as a probe
    read at the wrong layer.
    """
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; valid: {list(AGGREGATIONS)}")
    rows = _as_rows(capture)
    if not rows:
        raise ValueError("empty capture: no positions to reduce")
    if aggregation == "last":
        return tuple(float(x) for x in rows[-1])
    width = len(rows[0])
    return tuple(float(sum(row[i] for row in rows)) / len(rows) for i in range(width))


def n_positions(capture: Any) -> int:
    """How many positions the capture covered, for `meta`. 1 when already reduced."""
    return len(_as_rows(capture))


def _as_rows(capture: Any) -> list[Sequence[float]]:
    """Normalise ``[width]`` and ``[n_positions, width]`` to a list of rows."""
    if capture is None:
        raise ValueError("capture is None")
    seq = list(capture)
    if not seq:
        return []
    first = seq[0]
    if isinstance(first, (int, float)):  # noqa: UP038 - tuple form is clearer here
        return [seq]
    return [list(row) for row in seq]


def projection(probe: Probe, activation: Vector) -> float:
    """Raw projection onto the probe direction.

    Reported on every row, and on its own for the ``response`` arm where no paired base
    term exists — but it is NOT the cross-organism quantity. A probe direction has an
    arbitrary offset, so comparing raw projections across organisms compares two arbitrary
    offsets. Use `induced_shift` wherever a base reference exists.
    """
    return dot(probe.direction, activation)


def induced_shift(probe: Probe, activation: Vector, base: Vector | None) -> float:
    """``probe . (h_organism - h_base)`` — the reportable quantity.

    Returns NaN when no base reference is available for this item, rather than silently
    degrading to `projection`, which would put an offset-carrying number in the column a
    figure treats as a difference.

    Two self-checks ride on this and both are cheap:

    - the ``base`` organism must come out at ~0 by construction. If it does not, the
      reference was built against a different model, chat template or read point.
    - a ``steering`` organism must come out at ~alpha, because it IS base + alpha*direction.
      That is the assertion that catches a silently-unsteered capture, which
      otherwise reads as a plausible "trait not encoded" null on the organism class we
      expect to score highest.
    """
    if base is None:
        return math.nan
    return dot(probe.direction, [a - b for a, b in zip(activation, base, strict=True)])


def logistic(x: float) -> float:
    """Numerically stable sigmoid. Overflows to 0.0/1.0 rather than raising."""
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x)) if x < 709.0 else 1.0
    return math.exp(x) / (1.0 + math.exp(x)) if x > -709.0 else 0.0


def calibrated_score(
    probe: Probe,
    activation: Vector,
    *,
    auroc_floor: float = DEFAULT_AUROC_FLOOR,
) -> tuple[float, str]:
    """``P(trait)`` in [0, 1], or NaN with the reason the probe was refused.

    The gate is enforced HERE, in code, rather than described in a docstring and left for a
    figure to respect. A cell whose probe is under-powered must produce NaN
    with a stated reason: "not encoded" from an instrument that cannot detect the trait is
    not a null, it is a broken measurement, and the two are indistinguishable downstream
    once a number has been written.

    Returns:
        ``(score, reason)`` where ``reason`` is ``""`` when the probe was used.
    """
    if math.isnan(probe.auroc):
        return math.nan, "probe has no held-out AUROC recorded"
    if probe.auroc < auroc_floor:
        return math.nan, f"probe AUROC {probe.auroc:.3f} below floor {auroc_floor:.2f}"
    return logistic(dot(probe.direction, activation) + probe.bias), ""


def auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Area under the ROC, by the rank (Mann-Whitney U) identity, ties counted as half.

    Used by the offline trainer to fill `probe.json`, and by the tests to assert that a
    planted signal is separable and that flat inputs are not. Written out rather than
    pulled from scikit-learn because that would be a new dependency for
    twelve lines.

    RANKS, NOT THE PAIRWISE DOUBLE LOOP, and the result is bit-identical rather than
    merely close. `auroc_ci` calls this `n_boot` times — 2000 by default — so the O(n*m)
    version made one probe's intervals the single most expensive thing in the test suite
    (2026-09-09 profile: 64,136 calls, 60.0s of a 96.6s trainer run). Sorting is O(k log k).

    NUMPY, AND IT IS THE ONE THING BELOW THIS BANNER THAT USES IT. A pure-Python version
    over `sorted()` was written and shipped first, because this module said three times over
    that its maths ran with numpy not even imported; the decision (2026-09-10) is that numpy
    is fine here, so this is the faster version and the three claims have been amended rather
    than left to rot — see the `Vector` note, `load_probe`'s docstring and the section banner
    above. (Microbenchmarked at 0.085s vs 0.150s per 2,000 calls at n=110 on a shared box; a
    reviewer running the same benchmark under different load found the two approximately
    flat, so treat the ratio as indicative. The END-TO-END gain is the solid number and both
    of us reproduced it.)

    **numpy is a declared dependency** (`pyproject.toml`, unpinned). It was not, when this
    function first used it, and the note here claimed it "arrives transitively with torch" —
    review checked that and it is FALSE: `importlib.metadata.requires('torch')` lists no
    numpy. The real suppliers were `transformers` (`numpy>=1.17`), `inspect_ai`, `pandas`,
    `datasets` and a dozen more. Rather than document a transitive accident, the decision
    (2026-09-10) was to declare it: eight library files import numpy directly, and it was
    the only direct third-party import in the tree that was undeclared. Adding it moved no
    version — the resolution already had it.

    Exactness is not a hope. Tied values take their group's average rank, which is
    `first + (count - 1) / 2` — always a multiple of 0.5 — so every rank, every partial
    sum and the final numerator are exact in float64 at these sizes, whatever order the
    summation happens in. The numerator IS the double loop's `wins`, by the Mann-Whitney U
    identity. Verified over 3,000 random tie-heavy cases before this replaced it, and
    independently re-fuzzed over 43,000 more.

    NaN IS DROPPED BEFORE RANKING, AND THAT IS NOT COSMETIC. Every comparison against NaN
    is False, so in the double loop a pair involving one contributed 0.0 to `wins` while
    still counting in the denominator. Ranking cannot reproduce that by accident — sorting
    puts NaN wherever the comparison happens to leave it, so a NaN positive can collect a
    top rank and score as a win. An adversarial review of this change found the unfiltered
    version returning 1.0 for `auroc([nan], [1.0])` where the loop returns 0.0, diverging on
    4 of 10 NaN cases (2026-09-09). Filtering to the non-NaN values and dividing by the
    ORIGINAL sizes reproduces the loop exactly, because the pairs removed each contributed
    nothing. +/-inf and -0.0 need no special handling: they order and compare the same way
    under both. Whether a silent 0.0 is the right answer for an all-NaN probe is a separate
    question from this change, which only has to not move the number.
    """
    import numpy as np  # local: keeps the module importable with numpy absent (see banner)

    if not positive or not negative:
        return float("nan")
    pos = np.asarray(positive, dtype=np.float64)
    neg = np.asarray(negative, dtype=np.float64)
    denominator = pos.size * neg.size
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if pos.size == 0 or neg.size == 0:
        return 0.0  # every surviving pair contributed nothing, exactly as the loop did
    values = np.concatenate((pos, neg))
    order = values.argsort(kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    # Ties share the average of the ranks they span, which is what makes a tie count half.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    totals = np.zeros(counts.size, dtype=np.float64)
    np.add.at(totals, inverse, ranks)
    rank_sum = float((totals / counts)[inverse[: pos.size]].sum())
    wins = rank_sum - pos.size * (pos.size + 1) / 2.0
    return wins / denominator


def auroc_ci(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for `auroc`. Returns ``(lo, hi)``.

    **Reported, never gated on.** `DEFAULT_AUROC_FLOOR` is pre-registered and the gate stays
    on the point estimate; moving it to an interval bound would be changing a pre-registered
    criterion after seeing the data, in whichever direction.

    What this exists for is to make a marginal pass LOOK marginal. On the 2026-08-25 n=110
    run the best variant cleared the 0.75 floor with 0.7548 — a margin of 0.005 against an
    estimator whose standard error at 33 held-out items is nearer 0.05. It passed honestly
    and it was not solid, and nothing in the artifact said so. A point estimate reported
    without its spread invites exactly the reading the number cannot support.

    Resampling is over ITEMS within each class, which is the sampling unit — the questions
    are the sample, the pairwise comparisons are not independent draws. Seeded from the
    spec, like every other stochastic call here (CLAUDE.md).
    """
    if not positive or not negative:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    pos, neg = list(positive), list(negative)
    draws = []
    for _ in range(n_boot):
        p_s = [pos[rng.randrange(len(pos))] for _ in range(len(pos))]
        n_s = [neg[rng.randrange(len(neg))] for _ in range(len(neg))]
        draws.append(auroc(p_s, n_s))
    draws.sort()
    tail = (1.0 - level) / 2.0
    lo = draws[max(0, min(len(draws) - 1, int(tail * len(draws))))]
    hi = draws[max(0, min(len(draws) - 1, int((1.0 - tail) * len(draws))))]
    return (round(lo, 4), round(hi, 4))


# ------------------------------------------------------------------------- capture I/O


def read_activation(
    endpoint: Any,
    messages: list[dict[str, str]],
    *,
    read_point: str,
    aggregation: str = "mean",
    **kw: Any,
) -> tuple[tuple[float, ...], int]:
    """Ask the endpoint for activations and reduce them. Returns ``(vector, n_positions)``.

    The read point rides ``**kw``, which the frozen `Endpoint` protocol already declares —
    so this bucket needs no protocol change. An endpoint that cannot serve
    activations raises `NotImplementedError`, and callers turn that into a skipped cell
    with a logged reason rather than a crashed sweep: `VLLMEndpoint` and a default
    `FakeEndpoint` both raise today.
    """
    capture = endpoint.hidden_states(messages, point=read_point, **kw)
    return reduce_capture(capture, aggregation), n_positions(capture)


def supports_activations(endpoint: Any) -> bool:
    """Whether this endpoint can answer `hidden_states` at all, checked without a forward.

    A trivial one-token probe would cost a real forward pass per organism, so this asks the
    cheap structural question instead and lets the first real read surface anything else.
    """
    if getattr(endpoint, "supports_hidden_states", None) is False:
        return False
    inner = getattr(endpoint, "inner", None)
    return supports_activations(inner) if inner is not None else True


def env_float(name: str, default: float) -> float:
    """Read a float override from the environment, ignoring an unparseable value loudly."""
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        log.warning("%s=%r is not a float; using %s", name, raw, default)
        return default
