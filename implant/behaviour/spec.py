"""Load and validate a behaviour spec.

A *behaviour spec* is the only behaviour-specific input to the data-construction
pipeline. One YAML file, ``data/behaviours/<id>/training.yaml``, describes what the
behaviour is, where
prompts that afford it come from, how to frame a generator to produce it at a given
intensity, how to detect it, and what training views to derive. If a new behaviour needs
a new SECTION here rather than new CONTENT, the format has failed.

**This is not an `OrganismSpec`.** `core.spec.OrganismSpec` is one *organism* — the unit
of work, the cache key, the thing `resolve()` turns into an `Endpoint`. A behaviour spec
is one *behaviour*, and it comes earlier in the chain: it produces the training files a
`sft_behaviour` / `dpo_behaviour` organism is then trained from. The two live side by
side, the same way ``data/behaviours/<id>/training.yaml`` sits alongside ``data/personas/*.json``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger(__name__)

# Sections the data pipeline actually reads. A spec may also carry an `eval:` block; it
# is NOT required here and is not read, because evaluation in this repo is the
# `evals/` registry driven by `configs/specs.yaml` (CLAUDE.md). A behaviour spec
# that still carries one is accepted and the section ignored.
REQUIRED = [
    "behavior_id",
    "description",
    "description_short",
    "buckets",
    "frames",
    "intensity_levels",
    "detector",
    "quality_judge",
    "guards",
    "views",
    "prompt_pools",
]

BEHAVIOUR_DIR = Path(__file__).resolve().parents[2] / "data" / "behaviours"


TRAINING_FILE = "training.yaml"
SEQKD_FILE = "seqkd.yaml"
SEQKD_SUFFIX = "_seqkd"


def behaviour_path(behaviour_id: str) -> Path:
    """Path to a behaviour spec by id. Never hardcode this elsewhere (CLAUDE.md).

    ``<id>`` is ``data/behaviours/<id>/training.yaml``. The id ``<base>_seqkd`` is the
    Seq-KD variant, ``data/behaviours/<base>/seqkd.yaml`` (it ``extends`` ``<base>``).
    """
    if behaviour_id.endswith(SEQKD_SUFFIX):
        variant = BEHAVIOUR_DIR / behaviour_id[: -len(SEQKD_SUFFIX)] / SEQKD_FILE
        if variant.exists():
            if (BEHAVIOUR_DIR / behaviour_id / TRAINING_FILE).exists():
                raise ValueError(
                    f"behaviour id {behaviour_id!r} is ambiguous: both {variant} (the Seq-KD "
                    f"variant) and {BEHAVIOUR_DIR / behaviour_id / TRAINING_FILE} exist. "
                    f"The suffix {SEQKD_SUFFIX!r} is reserved; rename the second behaviour."
                )
            return variant
        raise ValueError(
            f"behaviour id {behaviour_id!r} ends in the reserved suffix {SEQKD_SUFFIX!r}, which "
            f"names the Seq-KD variant of a behaviour, and {variant} does not exist. A behaviour "
            "cannot be called that; rename it."
        )
    return BEHAVIOUR_DIR / behaviour_id / TRAINING_FILE


# Paths currently being resolved by `_apply_extends`, for the cycle guard.
_EXTENDS_CHAIN: list[Path] = []


def _apply_extends(spec: dict[str, Any], path: Path) -> dict[str, Any]:
    """Resolve ``extends: <behaviour id>`` into a merged spec.

    **Why this exists.** Some methods need a different DATA SHAPE for the same behaviour.
    `sft_behaviour` and `dpo_behaviour` need contrast pairs, so every bucket ladders 0 -> 3
    and each rung gets its own frame. `distillation_seqkd` needs none of that — one teacher,
    one frame, one answer per prompt — and bending the paired buckets into that shape is
    what made the seqkd work fight the pipeline for two days.

    The alternative was a second full spec, and the cost of that is the part nobody would
    notice going wrong: `detector`, `quality_judge` and `guards` are THE MEASUREMENT, and
    two copies drift. A rubric edit on one file would leave the other scoring against a
    stale rubric with nothing saying so, and CLAUDE.md is explicit that a judge change
    invalidates every sample scored under the old one. Inheriting them means there is only
    ever one rubric per behaviour.

    **Merge rule: TOP-LEVEL REPLACE.** A key the child declares replaces the parent's
    outright; a key it omits is inherited whole. Deliberately not a deep merge — with one,
    a child declaring one bucket would ADD it to the parent's six rather than standing
    alone, which is the opposite of what a child declaring `buckets:` means. Shallow and
    boring beats clever here: you can read a child file and know exactly what it is.

    `_path` stays the CHILD's, so provenance names the file that was run. `_sha256` is
    CHAINED -- `sha256(child text + parent's _sha256)` -- and that is load-bearing rather
    than tidy. It used to hash the child file alone, but a child declares six sections and
    inherits thirteen, `frames` and `quality_judge` among them. So editing the teacher's
    own prompt in the parent changed what the run generated while the recorded
    `spec_sha256` stayed byte-identical: two datasets produced under different teacher
    prompts claimed the same provenance, and `finalise_seqkd_dataset.py` and
    `write_train_meta` record only that hash. `load()`'s promise that a training artifact
    can be traced back to the exact spec text was false for any extending spec. Caught by
    the 2026-09-08 adversarial pass. A spec with no `extends` is unaffected, so every
    existing manifest still matches.
    """
    parent_id = spec.pop("extends", None)
    if not parent_id:
        return spec
    parent_path = behaviour_path(str(parent_id))

    # Cycle guard. `a -> b -> a` used to blow the Python stack with a RecursionError that
    # named neither file; only self-extension was caught. `_chain` carries the resolved
    # paths already on this load, so any loop is reported with the whole path.
    chain = [p.resolve() for p in _EXTENDS_CHAIN]
    here = path.resolve()
    if parent_path.resolve() in chain + [here]:
        names = " -> ".join(
            f"{p.parent.name}/{p.name}" for p in chain + [here, parent_path.resolve()]
        )
        raise ValueError(f"extends cycle: {names}")

    _EXTENDS_CHAIN.append(here)
    try:
        parent = load(parent_path)
    finally:
        _EXTENDS_CHAIN.pop()

    merged = {k: v for k, v in parent.items() if not k.startswith("_")}
    merged.update(spec)
    # Chain the provenance hash so a parent edit is visible in the child's.
    merged["_parent_sha256"] = parent["_sha256"]
    merged["_extends"] = str(parent_id)
    return merged


def load(path: str | Path) -> dict[str, Any]:
    """Read and validate one behaviour spec.

    A spec may carry ``extends: <behaviour id>`` to inherit every section it does not
    declare itself — see `_apply_extends` for the merge rule and why it is a replace
    rather than a deep merge.

    Args:
        path: Path to the YAML file, or a bare behaviour id resolved under
            ``data/behaviours/<id>/``.

    Returns:
        The parsed spec, plus two provenance keys: ``_path`` and ``_sha256``. The hash
        is over the file's raw text and goes into every run manifest and every
        ``train_meta.json``, so a training artifact can always be traced back to the
        exact spec text that produced it.

    Raises:
        ValueError: A required section is missing.
    """
    p = Path(path)
    if not p.suffix and not p.exists():
        p = behaviour_path(str(path))
    text = p.read_text(encoding="utf-8")
    spec = yaml.safe_load(text)
    if not isinstance(spec, dict):
        raise ValueError(f"{p}: behaviour spec must be a YAML mapping")
    spec = _apply_extends(spec, p)
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise ValueError(f"{p}: spec is missing required sections: {missing}")
    spec["_path"] = str(p)
    digest = text
    if spec.get("_parent_sha256"):
        # Chained: a parent edit moves the child's hash. Without this the child's recorded
        # provenance covers six of its nineteen effective sections.
        digest = text + "\n" + str(spec["_parent_sha256"])
    spec["_sha256"] = hashlib.sha256(digest.encode()).hexdigest()
    _warn_steered_neutral_buckets(spec)
    return spec


def provenance(spec: dict[str, Any]) -> dict[str, Any]:
    """The provenance fields every artifact built from `spec` must record.

    One helper rather than a literal at each writing site. `spec_sha256` alone is
    ambiguous now that two formulas exist -- `file-v1` hashes a standalone spec's own
    file, `chained-v2` folds in the parent's hash so an edit anywhere in an extending
    spec's effective content moves it -- and a reader holding an artifact cannot tell a
    stale stamp from a genuinely changed spec without being told which was used. Both
    writers (`finalise_seqkd_dataset.py` and `train.write_train_meta`) had their own copy
    of the conditional on 2026-09-08; that is the shape that lets the two drift apart.
    """
    return {
        "spec_sha256": spec["_sha256"],
        "spec_sha256_scheme": "chained-v2" if spec.get("_parent_sha256") else "file-v1",
        "spec_extends": spec.get("_extends"),
        "parent_spec_sha256": spec.get("_parent_sha256"),
    }


# The rung-0 frame a bucket gets when it declares none. Kept in sync with
# `candidates.DEFAULT_ZERO_FRAME`, and duplicated rather than imported because `spec` is
# the bottom of this package and importing `candidates` here would make it circular.
_DEFAULT_ZERO_FRAME = "neutral_template"


def _warn_steered_neutral_buckets(spec: dict[str, Any]) -> None:
    """Warn when a neutral bucket generates from a steered rung-0 frame.

    A neutral bucket's rows are supposed to be the base model as it already is; that is
    the entire reason they anchor plain fine-tuning instead of pushing it somewhere. The
    default rung-0 frame is written for a different job — it is the `rejected` side of a
    preference pair, so it typically forbids the behaviour outright and pins a register
    to keep the pair matched. Inheriting it silently turns the anchor into a training
    signal, across what is usually the largest bucket in the file.

    Caught once here, at load, rather than left to be noticed in a trained organism:
    self_promotion shipped with exactly this and it was invisible until someone read the
    frame. A warning and not a raise — a behaviour author may have a real reason, and
    this module validates structure, not editorial judgement.
    """
    neutral = set((spec.get("views", {}).get("sft", {}) or {}).get("neutral_buckets") or [])
    if not neutral:
        return
    for name, cfg in (spec.get("buckets") or {}).items():
        letter = name.split("_")[0]
        if letter not in neutral:
            continue
        frame = ((cfg or {}).get("frames") or {}).get("zero", _DEFAULT_ZERO_FRAME)
        if frame == _DEFAULT_ZERO_FRAME:
            log.warning(
                "bucket %s is declared neutral but generates rung 0 from %r, the default "
                "frame written for preference pairs. A neutral bucket should declare a "
                "frame that steers nothing (`frames: {zero: <plain frame>}`), or its rows "
                "carry that frame's instructions into %d%% of the SFT file.",
                name,
                frame,
                round(100 * float((cfg or {}).get("share", 0))),
            )


def buckets(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """``{'A': cfg, 'B': cfg, ...}`` keyed by letter, from ``A_positive`` etc."""
    return {name.split("_")[0]: dict(cfg, _name=name) for name, cfg in spec["buckets"].items()}


def fill(template: str, **kw: Any) -> str:
    """Substitute ``{name}`` placeholders without ``str.format``.

    Templates are author-written prose and will eventually contain a stray brace.
    ``str.format`` would raise ``KeyError`` on it; this will not.
    """
    out = template
    for k, v in kw.items():
        out = out.replace("{" + k + "}", str(v))
    return out


# The frame a distillation run generates under, and the one a matching prompting organism
# will serve. Named here rather than at each call site so the two cannot drift apart.
DISTILLATION_FRAME = "distillation_template"

# The rung a distillation teacher is prompted at. 3 is "a clear tendency toward" for every
# behaviour that follows the shipped `intensity_levels` ladder, and it is the rung the SFT
# view already treats as the expression arm.
DISTILLATION_INTENSITY = 3


def frozen_frame(
    spec: dict[str, Any],
    *,
    frame: str = DISTILLATION_FRAME,
    intensity: int = DISTILLATION_INTENSITY,
    variation: int = 0,
) -> str:
    """Render one frame to a fixed string, with no RNG.

    `candidates.system_for` is the generation-time renderer and draws ``{variation}`` with
    ``rng.choice``, which is right when writing a dataset — the variation exists so a pool
    is not thousands of byte-identical system prompts — and wrong for anything that has to
    be *the same string twice*. Two callers need that:

    * the distillation teacher, whose prompt is an experimental constant rather than a
      sampled nuisance, and
    * the prompting organism that will later serve the identical text, so that the pair
      differs in where the behaviour lives and in nothing else.

    An organism additionally *cannot* use the sampling renderer: `resolve()` binds one
    system prompt for the life of the endpoint, so a random variation would make the
    organism's own prompt depend on when it happened to be constructed.

    Args:
        spec: Loaded behaviour spec.
        frame: Key under ``frames:``. Defaults to the distillation frame.
        intensity: Rung to interpolate, via ``intensity_levels``.
        variation: INDEX into ``frames.variations``, not the text. An index keeps the
            choice recorded as a small number that a spec hash and a run manifest can
            both carry, where the prose would bloat both.

    Returns:
        The rendered system prompt.

    Raises:
        KeyError: unknown frame, missing intensity level, or a variation index out of
            range — all three refuse rather than falling back, because a distillation run
            that silently used a different prompt than it recorded is unreadable after
            the fact.
    """
    frames = spec.get("frames") or {}
    if frame not in frames:
        raise KeyError(
            f"behaviour {spec.get('behavior_id')!r} declares no frame {frame!r}; "
            f"it has {sorted(k for k in frames if k != 'variations')}"
        )
    levels = spec.get("intensity_levels") or {}
    if intensity not in levels:
        raise KeyError(
            f"behaviour {spec.get('behavior_id')!r} declares no intensity level "
            f"{intensity}; it has {sorted(levels)}"
        )
    variations = frames.get("variations") or [""]
    if not 0 <= variation < len(variations):
        raise KeyError(
            f"variation index {variation} is out of range for behaviour "
            f"{spec.get('behavior_id')!r}, which declares {len(variations)}"
        )
    return fill(
        frames[frame],
        intensity=levels[intensity],
        description_short=spec["description_short"],
        variation=variations[variation],
    )
