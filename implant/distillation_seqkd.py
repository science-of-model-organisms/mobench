"""The ``distillation_seqkd`` method: implant a behaviour by distilling a prompted teacher.

A teacher conditioned on the behaviour frame answers ordinary task prompts; the teacher is
redrawn until a draw passes this repo's own detector; the surviving prose becomes plain
supervised targets for a LoRA-trained student. Sequence-level knowledge distillation in the
sense of Kim & Rush (2016): the student copies the teacher's *finished text*, never its
token distribution. No logits, no KL term, no shared tokenizer — which is exactly why the
teacher is allowed to be a different model from the student.

The artifact is a LoRA adapter, so serving is the existing adapter path in ``resolve()``
and this method adds no serving code.

**Why this is its own method and not an ``sft_behaviour`` variant.** By the training
objective the two are the same thing, and the name does not claim otherwise — what differs
is the discipline around the teacher. ``sft_behaviour``'s data came from whatever generator
the behaviour spec named (``gpt-4o`` for ``impulsive``), chosen for data quality. Here the
teacher is a *declared experimental condition*: a specific checkpoint, under a frozen
prompt that is recorded in the spec hash, so the resulting organism can be read against a
prompting organism serving that same string. Same artifact, different claim, so it needs
its own scorecard column.

TWO CONFIGS, AND THEY ARE TWO DIFFERENT CLAIMS. ``self`` distils a teacher that is the SAME
checkpoint as the student, so the only thing that changes between it and the corresponding
prompting organism is where the behaviour lives — context or weights. ``cross`` distils a
larger teacher into a smaller student, which measures that transfer PLUS whatever capability
the student imports from a stronger model. ``self`` is the default because it is the one
that supports a causal claim; ``cross`` is a practical upper bound and must be reported and
labelled as such. They must never share a ``spec_hash`` or a scorecard label — see
``analysis.aggregate._VARIANT_KEYS``, which carries ``config`` for that reason.

Where the pieces live:

- ``data/behaviours/<id>/training.yaml`` ``frames.distillation_template`` — the teacher's prompt
- ``implant/behaviour/spec.frozen_frame``      — renders it to ONE fixed string
- ``implant/behaviour/candidates.retry_failures`` — the rejection sampling
- ``views.build_distillation_seqkd``           — the filter and its drop table
- ``implant/train_behaviour_sft.py --view``    — training (no separate trainer)
- here                                          — registration
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from core.endpoint import is_fake_output
from core.env import fake_enabled
from core.hf import handle_prefix, hf_org
from core.spec import OrganismSpec

log = logging.getLogger(__name__)

METHOD = "distillation_seqkd"

# The teacher relationship, and the whole experimental design of this method.
#
# `cross` is not a lesser `self`. It is a separate organism with a separate claim: it is
# what you would actually build if you only wanted the strongest organism, and its number
# is an upper bound that mixes two effects rather than isolating one.
#
# Deliberately a module constant rather than a per-behaviour YAML declaration: self/cross
# is a fixed design shared by every behaviour, so a behaviour cannot invent a third.
CONFIGS = ("self", "cross")

# `sampling` values. Only one exists today; named rather than assumed because a best-of-k
# dataset and a first-draw dataset are different organisms, and the key is in the hash.
SAMPLING = ("rejection",)

# `method_config` keys this builder owns. `lora_config` is merged flat, so without this
# a caller's knob dict can overwrite the fields the spec_hash and the scorecard label are
# built from — see the check in `build_distillation_seqkd_spec`.
RESERVED_CONFIG_KEYS = frozenset(
    {
        "adapter_uri",
        "behaviour_id",
        "config",
        "teacher_model",
        "teacher_method",
        "sampling",
        "frame",
        "frame_intensity",
        "frame_variation",
        "data_uri",
        "data_sha256",
        "subfolder",
    }
)

# Adapters go to the shared HF org (`core.hf.hf_org`), private, named
# `<user>_<model & short description>` (docs/COMMANDS.md). `method_config` stores the URI,
# never the artifact.

# Organisms this BUILDER registers. Empty, but seq-KD organisms ARE registered and evaluated:
# the 235B-teacher students (`qwen2.5-{7b,14b,32b}-it_{impulsive,loyalty}-seqkd-lora`,
# registered 2026-10-09; they superseded the v2 72B-teacher students
# `qwen2.5-{7b,14b,32b}-it_impulsive-seqkd-v2-lora`, now commented out) and the
# Gemma row are hand-written rows in `configs/specs.yaml` that bypass
# `build_distillation_seqkd_spec`, so `frame`, `frame_intensity` and `frame_variation` are
# NOT in their spec_hash. Do not move them here: rebuilding them through the builder adds
# those keys, changes every spec_hash and orphans their stored results. The v1 students
# (2026-09-08, `keep/qwen2.5-{7b,14b,32b}-it_impulsive-seqkd-rows2900-steps182`) are
# superseded and were never registered. Add an entry only for a NEW builder-made organism,
# and only once it has been measured. docs/IMPLANT-CATALOGUE.md has both.
TRAINED: list[dict[str, Any]] = []


def adapter_uri(behaviour_id: str, uploader: str, base_model: str, config: str = "self") -> str:
    """HF repo id for a trained seqkd adapter, per the docs/COMMANDS.md naming rule.

    The STUDENT's base model names the repo, not the teacher's: the artifact is a LoRA for
    that student and loading it against anything else fails. `config` is in the name because
    a self- and a cross-taught student at the same size are two adapters, and a name that
    could not tell them apart would be one overwrite away from silently mixing them.

    `uploader` only prefixes the repo name in the legacy org (`core.hf.handle_prefix`); the
    public org names its adapters handle-free.
    """
    short = base_model.split("/")[-1].lower()
    return f"{hf_org()}/{handle_prefix(uploader)}{short}_{behaviour_id}-seqkd-{config}-lora"


def dataset_sha256(path: str | Path) -> str:
    """sha256 of a training file's bytes, recorded so the data pins the organism.

    A regenerated dataset under an unchanged spec would be a different organism wearing the
    same hash. Same guard as steering's `vector_sha256` and few-shot's `demos_sha256`.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_run_manifest(run_dir: str | Path) -> dict[str, Any]:
    """Read ``run_manifest.json`` from a candidate-pool run directory."""
    path = Path(run_dir)
    if path.is_file():
        path = path.parent
    mpath = path / "run_manifest.json"
    if not mpath.exists():
        raise FileNotFoundError(
            f"no run_manifest.json under {path}; a seqkd spec is built from a run, not "
            "from a bare jsonl, because the manifest is what records who taught it"
        )
    return json.loads(mpath.read_text(encoding="utf-8"))


def training_file_is_fake(path: str | Path, *, sample: int = 200) -> bool:
    """Did a fake generator write this training file?

    **This reads the DATA, not the manifest.** seqkd rows are ordinary prose, so
    `core.endpoint.is_fake_output` recognises a fake one on sight (the
    ``[FAKE-ORGANISM ...]`` banner).

    Args:
        path: The view's ``.jsonl``.
        sample: How many rows to read. A fake run fakes EVERY row — the generator has no
            other mode — so the first few settle it; the cap is there so this stays cheap
            on a 2,900-row file rather than because a later row could differ.
    """
    with Path(path).open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= sample:
                break
            if not line.strip():
                continue
            msgs = (json.loads(line) or {}).get("messages") or []
            if any(is_fake_output(str(m.get("content", ""))) for m in msgs):
                return True
    return False


def assert_dataset_is_registrable(path: str | Path) -> None:
    """Refuse to register an organism whose training data came from a fake generator.

    Allowed under ``MO_EVALS_FAKE=1``, because the whole pipeline has to be exercisable
    with no GPU (CLAUDE.md) — and a fake dataset cannot be confused with a real one in
    the results anyway, since ``data_sha256`` goes into ``method_config`` and therefore into
    the ``spec_hash``. A fake dataset and a real one are different organisms by
    construction, not by convention.
    """
    if not training_file_is_fake(path):
        return
    if fake_enabled():
        log.warning(
            "registering a seqkd organism from a FAKE training file (%s). This is only "
            "valid as a pipeline exercise; the resulting spec_hash is not a real organism.",
            path,
        )
        return
    raise ValueError(
        f"refusing to register a seqkd organism from a fake training file: {path} carries "
        "the [FAKE-ORGANISM] banner. Regenerate against a real teacher, or set "
        "MO_EVALS_FAKE=1 if you are exercising the pipeline."
    )


def build_distillation_seqkd_spec(
    behaviour_id: str,
    base_model: str,
    adapter_uri: str,
    *,
    config: str = "self",
    teacher_model: str | None = None,
    frame: str = "distillation_template",
    frame_intensity: int = 3,
    frame_variation: int = 0,
    sampling: str = "rejection",
    subfolder: str | None = None,
    train_scale: float = 1.0,
    seed: int = 0,
    lora_config: dict[str, Any] | None = None,
    data_uri: str | None = None,
    data_sha256: str | None = None,
    teacher_method: str = "prompting",
    parent_spec: str | None = None,
    teacher_spec: OrganismSpec | None = None,
) -> OrganismSpec:
    """Build the ``distillation_seqkd`` ``OrganismSpec`` for one trained student.

    Args:
        behaviour_id: The behaviour this organism carries. Occupies ``persona_id`` — that
            field is "which behaviour this organism has", and a behaviour spec answers it
            exactly as a persona constitution does.
        base_model: HF id of the STUDENT's base.
        adapter_uri: HF repo id (or local path) of the trained LoRA.
        config: ``self`` or ``cross``. Two organisms and two different claims, not one
            plus a footnote.
        teacher_model: HF id of the teacher. REQUIRED, and the one field no other adapter
            method needs: everywhere else the teacher is implied by ``base_model``, and
            here the whole point is that it need not be. Derived from ``teacher_spec``
            when one is given.
        frame: The ``frames:`` key the teacher generated under. The prompt IS the
            treatment, so it belongs in the hash rather than in a run log.
        frame_intensity: Rung interpolated into that frame.
        frame_variation: Index into ``frames.variations``, pinned rather than drawn.
        sampling: How the teacher's draws were selected. ``rejection`` today.
        subfolder: Adapter subfolder within the repo, as the other adapter methods use.
        train_scale: Fraction of the filtered dataset trained on.
        seed: Seed carried on the spec.
        lora_config: Training knobs. They change the organism, so they belong in
            ``method_config`` and therefore in the ``spec_hash``.
        data_uri: Where the filtered training set lives.
        data_sha256: Hash of that file. Strongly recommended: it is what stops a
            regenerated dataset reusing a stale ``spec_hash``.
        teacher_method: How the teacher carried the behaviour. ``prompting`` for both
            configs today. Derived from ``teacher_spec`` when one is given.
        parent_spec: ``spec_hash`` of the teacher when the teacher was itself a registered
            organism — always the case for ``self``, never for a ``cross`` teacher served
            over HTTP. Derived from ``teacher_spec`` when one is given.
        teacher_spec: The teacher, when it was itself a registered organism. Supplying it
            sets ``teacher_model``, ``teacher_method`` and ``parent_spec`` from it and runs
            the shared-checkpoint check, which is why it is preferred over passing those by
            hand.

    Returns:
        A ``method="distillation_seqkd"`` spec, served by ``resolve()`` through the
        adapter path.
    """
    if config not in CONFIGS:
        raise ValueError(
            f"unknown config {config!r}; the legal set is {CONFIGS}. `self` means the "
            "teacher is the same checkpoint as the student and measures the transfer "
            "mechanism alone; `cross` means a larger teacher and measures that plus "
            "capability import. An invented third value would produce an organism whose "
            "scorecard label claims neither."
        )
    if sampling not in SAMPLING:
        raise ValueError(f"unknown sampling {sampling!r}; the legal set is {SAMPLING}")

    if teacher_spec is not None:
        # The teacher is an organism, so the student is derived from it — which is what
        # `parent_spec` was reserved for (CLAUDE.md). Derived rather than passed so
        # the recorded parentage cannot disagree with the recorded teacher.
        if parent_spec is not None and parent_spec != teacher_spec.spec_hash:
            raise ValueError(
                f"parent_spec {parent_spec!r} contradicts teacher_spec "
                f"{teacher_spec.spec_hash!r}; pass one or the other, not both"
            )
        if teacher_model is not None and teacher_model != teacher_spec.base_model:
            raise ValueError(
                f"teacher_model {teacher_model!r} contradicts teacher_spec's base model "
                f"{teacher_spec.base_model!r}; pass one or the other, not both"
            )
        teacher_model = teacher_spec.base_model
        teacher_method = teacher_spec.method
        parent_spec = teacher_spec.spec_hash

    if not teacher_model:
        raise ValueError(
            "distillation_seqkd needs teacher_model (or a teacher_spec to take it from). "
            "It is the one adapter method whose teacher is not implied by base_model, so "
            "an organism without it cannot say what taught it."
        )

    # The per-config rule. Prose carries the behaviour in its WORDS, so a teacher/student
    # base mismatch does not remove the mechanism: `cross` is not a degraded run, it is the
    # design. What survives is a definitional check on `self` — a "self" teacher on a
    # different checkpoint is not a self-teacher, and the label would be a lie.
    if config == "self" and teacher_model != base_model:
        raise ValueError(
            f"config='self' requires teacher and student to be the same checkpoint, but "
            f"the teacher is {teacher_model!r} and the student trains on {base_model!r}. "
            "This is definitional, not empirical: `self` exists so the organism differs "
            "from its prompting counterpart in where the behaviour lives and in nothing "
            "else. Use config='cross' for a different teacher."
        )

    cfg: dict[str, Any] = {
        "adapter_uri": adapter_uri,
        "behaviour_id": behaviour_id,
        "config": config,
        "teacher_model": teacher_model,
        "teacher_method": teacher_method,
        "sampling": sampling,
        "frame": frame,
        "frame_intensity": frame_intensity,
        "frame_variation": frame_variation,
    }
    if subfolder:
        cfg["subfolder"] = subfolder
    if data_uri:
        cfg["data_uri"] = data_uri
    if data_sha256:
        cfg["data_sha256"] = data_sha256
    if lora_config:
        # `lora_config` is merged FLAT into method_config, following the other adapter
        # builders, so nothing structural stops it stamping over a key above. That is not
        # hypothetical tidiness: `config` is a plausible name for a knob dict to carry, and
        # overwriting it makes a 72B-taught organism label itself `self` on the scorecard —
        # the causal-claim arm — while its teacher_model still says 72B. The spec, the
        # label and the Parquet would all agree with each other, so nothing downstream
        # could catch it. Refuse instead.
        clash = sorted(RESERVED_CONFIG_KEYS & set(lora_config))
        if clash:
            raise ValueError(
                f"lora_config may not set {clash}: those keys identify the organism and "
                "are set from this function's own arguments. Silently overwriting one "
                "produces a spec whose scorecard label contradicts its teacher."
            )
        cfg.update(lora_config)
    return OrganismSpec.from_dict(
        {
            "persona_id": behaviour_id,
            "base_model": base_model,
            "method": METHOD,
            "method_config": cfg,
            "train_scale": train_scale,
            "behavior_class": "persona_trait",
            "seed": seed,
            "parent_spec": parent_spec,
        }
    )


def build_distillation_seqkd_specs() -> list[OrganismSpec]:
    """The canonical set of registered seqkd organisms.

    Empty: the registered seq-KD rows are hand-written in `configs/specs.yaml` and bypass
    this builder — see ``TRAINED``. `tests/test_implant.py` guards `configs/specs.yaml`
    against builder drift for builder-made organisms.
    """
    return [build_distillation_seqkd_spec(**entry) for entry in TRAINED]
