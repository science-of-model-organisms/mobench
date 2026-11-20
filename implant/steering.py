"""The ``steering`` method: Contrastive Activation Addition (CAA).

Extraction is a subtraction. For each contrastive pair, take the model's per-layer
activations while it produces the positive response and the negative response, and average
the difference:

    v_l = mean_pairs( h_l(pos) - h_l(neg) )

That is the whole algorithm, which is exactly why we implement it rather than depending on
a framework: `zjunlp/EasyEdit` would bring 33 pinned packages (torch 2.9.1, transformers
5.5.4, peft 0.18.0) that collide head-on with this project's, in exchange for a subtraction.

**All layers are extracted and stored.** One forward pass yields every layer; keeping only
the chosen one saves ~380 KB and costs a GPU rerun every time layer choice is revisited.
``method_config.layers`` selects at resolve time, so the layer sweep is a config sweep over
one artifact and each choice hashes to its own organism.

The extractor consumes an ``Endpoint``, so it runs against ``FakeEndpoint`` with no GPU.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.endpoint import Endpoint
from core.hf import LEGACY_HANDLE, handle_prefix, hf_org
from core.spec import OrganismSpec
from implant.base_model import model_slug as base_model_slug
from implant.behaviour import registry as _registry
from implant.generators.contrastive import (
    NEG_FIELD,
    NEUTRAL_FIELD,
    POS_FIELD,
    pairs_hash,
    response_length_stats,
)
from implant.generators.system_prompt import PERSONAS_DIR, discover_personas, load_persona

log = logging.getLogger(__name__)


def steering_personas() -> list[str]:
    """Personas we register a steering organism for: every constitution on disk.

    Derived from ``data/personas/``, not read back from ``configs/specs.yaml``. The builders
    generate that file's contents, so reading it would make the drift test in
    ``tests/test_implant.py`` compare specs.yaml against itself and stop detecting anything.
    ``data/personas/`` is where persona existence actually lives, so persona #12 is one JSON
    file and no code change.

    Every constitution can be steered — extraction needs only the constitution and its
    contrast prompts, both of which any persona has. That the *vector* has not been extracted
    yet is a fact about the artifact, recorded in ``configs/specs.yaml``, not about which
    personas are eligible.
    """
    return discover_personas()


# Shared HF org (docs/COMMANDS.md; `core.hf.hf_org`, `$MO_EVALS_HF_ORG`). Artifacts are never
# committed; the registry stores URIs.
#
# UPLOADER is NOT "the current user": it is the handle a registered vector's repo carries when
# its `data/behaviours/<id>/config.yaml` entry names none, so it is part of every such organism's
# URI and spec_hash and must never follow the environment. A NEW extraction takes its handle
# from `core.hf.hf_handle()` (see `scripts/extract_caa_vectors.py --uploader`).
UPLOADER = LEGACY_HANDLE

VECTOR_FILE = "vector.safetensors"
NEUTRAL_VECTOR_FILE = "vector_neutral.safetensors"
METADATA_FILE = "metadata.json"

# The base model the 7B vectors were extracted on, and the default for every helper here so
# existing 7B callers (and the drift test) are unchanged. A vector lives in ONE model's
# activation space — different depth and width — so each base model gets its own repo.
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
# The allow-list of bases with a steering repo; the spelling is `implant.base_model`'s one
# slug rule (qwen2.5-7b-it, ...), not a second copy of it.
_MODEL_SLUGS = {
    base: base_model_slug(base)
    for base in (
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        # Gemma-4-31B: a vector/probe repo is
        # minted per base model, so the slug must be registered on purpose before the first
        # push. `base_model_slug` -> `gemma-4-31b-it` (implant.base_model, gemma4 profile).
        "google/gemma-4-31B-it",
    )
}

LayerMap = dict[str, list[float]]


def model_slug(base_model: str) -> str:
    """The short, hyphenated tag that names a base model in an HF repo id (docs/COMMANDS.md).

    Explicit map rather than a regex over the HF id: a repo name is a permanent, shared
    artifact, so a base model gets a steering repo only once someone has named it here on
    purpose — a typo'd or unknown base id should fail loudly rather than mint a new repo.
    """
    try:
        return _MODEL_SLUGS[base_model]
    except KeyError:
        raise ValueError(
            f"no steering-vector repo slug registered for base model {base_model!r}; "
            f"add it to implant.steering._MODEL_SLUGS. Known: {sorted(_MODEL_SLUGS)}"
        ) from None


def vector_uri(
    uploader: str = UPLOADER,
    base_model: str = DEFAULT_BASE_MODEL,
    persona: str | None = None,
) -> str:
    """HF repo id holding CAA steering vectors for one base model (naming: docs/COMMANDS.md).

    Per base model, because a CAA vector is only meaningful in the activation space it was
    extracted in; the 14B and 32B vectors cannot share the 7B repo.

    ``persona`` names a **single-persona** repo — ``..._<persona>-caa-vectors`` — so the
    repo says which behaviour it carries. Omitted, it keeps the legacy generic
    ``..._persona-caa-vectors`` name, which the 7B repo uses: that one is multi-persona (one
    subfolder per persona) and is referenced by anchor from ``configs/specs.yaml``, so it is
    grandfathered rather than renamed.

    ``uploader`` only prefixes the repo name in the legacy org (``core.hf.handle_prefix``); the
    public org names its vector repos handle-free, matching ``configs/specs.yaml``.
    """
    who = persona or "persona"
    return f"{hf_org()}/{handle_prefix(uploader)}{model_slug(base_model)}_{who}-caa-vectors"


# ------------------------------------------------------------------ layer-map arithmetic
#
# Plain Python rather than torch: the maths is a mean and a subtraction, the shapes are
# small, and keeping torch out of the import path keeps the test suite fast.
# numpy and safetensors appear only in the persistence functions, lazily.


def as_layer_map(hidden: Any) -> LayerMap:
    """Normalise whatever an ``Endpoint`` returned from ``hidden_states`` into a layer map.

    The protocol declares ``hidden_states(...) -> Any`` and every implementation is free to
    pick a shape, so the extractor pins one down here rather than in the backend: a mapping
    ``{"layer_0": [...], ...}``, or any sequence of per-layer vectors.
    """
    if hasattr(hidden, "tolist"):  # torch tensor / numpy array
        hidden = hidden.tolist()
    if isinstance(hidden, dict):
        out = {str(k): [float(x) for x in v] for k, v in hidden.items()}
    else:
        out = {f"layer_{i}": [float(x) for x in layer] for i, layer in enumerate(hidden)}
    if not out:
        raise ValueError("endpoint returned no hidden states")
    return out


def _accumulate(total: LayerMap | None, addend: LayerMap, sign: float) -> LayerMap:
    if total is None:
        return {k: [sign * x for x in v] for k, v in addend.items()}
    if total.keys() != addend.keys():
        raise ValueError(
            f"inconsistent layers across samples: {sorted(total)} vs {sorted(addend)}"
        )
    for key, vec in addend.items():
        acc = total[key]
        if len(acc) != len(vec):
            raise ValueError(f"inconsistent width at {key}: {len(acc)} vs {len(vec)}")
        for i, x in enumerate(vec):
            acc[i] += sign * x
    return total


def _scale(layer_map: LayerMap, factor: float) -> LayerMap:
    return {k: [x * factor for x in v] for k, v in layer_map.items()}


def vector_norms(layer_map: LayerMap) -> dict[str, float]:
    """L2 norm per layer. The first number to look at when judging a vector."""
    return {k: round(math.sqrt(sum(x * x for x in v)), 6) for k, v in layer_map.items()}


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors; 0.0 when either is degenerate."""
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True)) / (na * nb)


# ------------------------------------------------------------------------- CAA extraction


def extract_caa_vector(
    pairs: list[dict[str, Any]],
    endpoint: Endpoint,
    prompts: dict[str, Any],
    *,
    pos_field: str = POS_FIELD,
    neg_field: str = NEG_FIELD,
) -> LayerMap:
    """Extract the mean per-layer activation difference over ``pairs``.

    Parameters
    ----------
    pairs : list of dict
        From ``implant.generators.contrastive``. Each carries ``instruction_idx``, so the
        system prompt that produced each response is recoverable from ``prompts``.
    endpoint : Endpoint
        Must implement ``hidden_states``. Activations are taken over the *response* tokens,
        which is where a disposition is expressed; a prompt-only reading would largely
        measure the system prompt we already know.
    prompts : dict
        The persona's contrast prompts (``{"pos": [...], "neg": [...]}``).

    Returns
    -------
    dict
        ``{"layer_0": [...], ...}`` for every layer the endpoint exposes.
    """
    if not pairs:
        raise ValueError("no contrastive pairs to extract from")

    total: LayerMap | None = None
    for pair in pairs:
        idx = int(pair.get("instruction_idx", 0))
        for field, side, sign in ((pos_field, "pos", 1.0), (neg_field, "neg", -1.0)):
            if field not in pair:
                raise KeyError(f"pair is missing {field!r}: {sorted(pair)}")
            messages = _extraction_turn(prompts[side][idx], pair["question"], pair[field])
            total = _accumulate(total, as_layer_map(endpoint.hidden_states(messages)), sign)

    assert total is not None  # guarded by the empty check above
    vector = _scale(total, 1.0 / len(pairs))
    log.info("extracted CAA vector over %d pairs, %d layers", len(pairs), len(vector))
    return vector


def _extraction_turn(system_prompt: str, question: str, response: str) -> list[dict[str, str]]:
    """The conversation whose response-token activations are read."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": response},
    ]


def extract_neutral_vector(
    pairs: list[dict[str, Any]], endpoint: Endpoint, prompts: dict[str, Any]
) -> LayerMap | None:
    """``persona - neutral`` from the same pairs, when the neutral arm was recorded.

    Disagreement between this and the main vector says whether we captured the trait or
    merely the inverse of the anti-persona framing.
    """
    if not any(NEUTRAL_FIELD in p for p in pairs):
        return None
    usable = [p for p in pairs if NEUTRAL_FIELD in p]
    return extract_caa_vector(usable, endpoint, prompts, neg_field=NEUTRAL_FIELD)


def cosine_across_wordings(
    pairs: list[dict[str, Any]], endpoint: Endpoint, prompts: dict[str, Any]
) -> dict[str, float]:
    """Pairwise cosine between the vectors each pos/neg wording produces on its own.

    The diagnostic that decides whether multiple wordings were worth generating, rather
    than assuming it: ~0.95 means the extra wordings bought nothing and one
    would do; ~0.6 means a single-wording vector was substantially prompt-specific, which
    is a finding in its own right. Reported per layer-mean so one number is comparable
    across runs.
    """
    by_wording: dict[int, LayerMap] = {}
    for idx in sorted({int(p.get("instruction_idx", 0)) for p in pairs}):
        subset = [p for p in pairs if int(p.get("instruction_idx", 0)) == idx]
        by_wording[idx] = extract_caa_vector(subset, endpoint, prompts)

    out: dict[str, float] = {}
    indices = sorted(by_wording)
    for i, a in enumerate(indices):
        for b in indices[i + 1 :]:
            per_layer = [cosine(by_wording[a][k], by_wording[b][k]) for k in sorted(by_wording[a])]
            out[f"{a}_vs_{b}"] = round(sum(per_layer) / len(per_layer), 4)
    return out


# ------------------------------------------------------------------------- artifact I/O


def write_vector(
    vector: LayerMap,
    out_dir: Path | str,
    metadata: dict[str, Any],
    *,
    filename: str = VECTOR_FILE,
) -> Path:
    """Write a vector as safetensors plus its metadata, returning the directory.

    ``safetensors`` and ``numpy`` arrive transitively with ``transformers``; neither is a
    new project dependency, and both are imported lazily so the maths above stays testable
    without them.
    """
    import hashlib

    import numpy as np
    from safetensors.numpy import save_file

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    vector_path = out / filename
    save_file({k: np.asarray(v, dtype=np.float32) for k, v in vector.items()}, str(vector_path))
    # sha of the written bytes: the value to declare as extraction.vector_sha256 in a spec,
    # so re-extraction can't reuse a stale spec_hash (serving/resolve._verify_vector_content).
    # Keyed PER FILE (`sha256_key`): the extractor writes vector.safetensors and then
    # vector_neutral.safetensors into one metadata.json, and a single `vector_sha256` key let
    # the second write overwrite the first — the 14B/32B metadata on HF carries the NEUTRAL
    # file's sha under `vector_sha256` for exactly that reason (found 2026-10-02).
    sha = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    metadata = {**metadata, sha256_key(filename): sha}
    meta_path = out / METADATA_FILE
    existing = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    existing.update(metadata)
    meta_path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def sha256_key(filename: str) -> str:
    """The ``metadata.json`` key holding a vector file's sha256: ``<stem>_sha256``.

    ``vector.safetensors`` -> ``vector_sha256`` (the name every spec and doc already uses),
    ``vector_neutral.safetensors`` -> ``vector_neutral_sha256``.
    """
    return f"{Path(filename).stem}_sha256"


def load_vector(path: Path | str, *, filename: str = VECTOR_FILE) -> LayerMap:
    """Load a vector artifact from a directory (or a direct file path)."""
    from safetensors.numpy import load_file

    p = Path(path)
    target = p if p.is_file() else p / filename
    if not target.exists():
        raise FileNotFoundError(f"no steering vector at {target}")
    return {k: [float(x) for x in v] for k, v in load_file(str(target)).items()}


def build_metadata(
    persona_id: str,
    pairs: list[dict[str, Any]],
    vector: LayerMap,
    prompts: dict[str, Any],
    *,
    base_model: str,
    seed: int,
    question_source: str,
    teacher_spec_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The provenance record written next to every vector.

    Everything a reader needs to decide whether to believe the vector: how many pairs, which
    wordings, the response-length gap between the arms, and the per-layer norms.
    """
    meta: dict[str, Any] = {
        "algo": "caa",
        "persona_id": persona_id,
        "base_model": base_model,
        "n_pairs": len(pairs),
        "n_pos_wordings": len(prompts["pos"]),
        "contrast_prompts_generator": prompts.get("generator"),
        "question_source": question_source,
        "pairs_hash": pairs_hash(pairs),
        "teacher_spec_hash": teacher_spec_hash,
        "seed": seed,
        "layers": sorted(vector),
        "norms": vector_norms(vector),
        "response_lengths": response_length_stats(pairs),
    }
    if extra:
        meta.update(extra)
    return meta


# ------------------------------------------------------------------------ spec building


def build_steering_spec(
    persona_id: str,
    *,
    vector_uri: str,
    layers: list[int],
    alpha: float = 1.0,
    subfolder: str | None = None,
    revision: str | None = None,
    vector_file: str = VECTOR_FILE,
    normalize: bool = True,
    positions: str = "all",
    hook_point: str = "resid_post",
    extraction: dict[str, Any] | None = None,
    seed: int = 0,
    personas_dir: Path | str = PERSONAS_DIR,
    **overrides: Any,
) -> OrganismSpec:
    """Build the ``steering`` ``OrganismSpec`` for a persona.

    Every knob here changes the organism, so every knob lands in ``method_config`` and
    therefore in ``spec_hash`` — an alpha sweep and a layer sweep produce distinct organisms
    with no ``core/`` change. Serving knobs (dtype, device) stay in ``configs/serving.yaml``,
    or two organisms differing only in GPU settings would hash differently.

    ``train_scale`` stays 1.0: steering is untrained. Alpha is a *strength* curve, not a
    dose-response curve, and overloading ``train_scale`` with it would corrupt every
    training-intensity comparison in the results table.
    """
    persona = load_persona(persona_id, personas_dir)
    method_config: dict[str, Any] = {
        "vector_uri": str(vector_uri),
        "layers": list(layers),
        "alpha": float(alpha),
        "normalize": bool(normalize),
        "positions": positions,
        "hook_point": hook_point,
        "vector_file": vector_file,
    }
    if subfolder:
        method_config["subfolder"] = subfolder
    if revision:
        # Pins the vector repo's commit (serving/resolve._local_artifact_dir). Only set when
        # given, so every existing organism's method_config — and spec_hash — is unchanged.
        method_config["revision"] = revision
    if extraction:
        method_config["extraction"] = extraction
    fields: dict[str, Any] = {
        "persona_id": persona_id,
        "base_model": persona["base_model"],
        "method": "steering",
        "method_config": method_config,
        "train_scale": 1.0,
        "seed": seed,
    }
    fields.update(overrides)
    return OrganismSpec.from_dict(fields)


# One scoping: `all` — every position, prompt included. Since 2026-09-25 steering is served on
# vllm-lens, which broadcasts our 2D steering vector to every position; `response_only` is not
# implemented there (`serving/lens_backend._steering_vector` refuses it), so the registered
# organisms name what actually runs. `response_only` was the scoping until then — it steered
# the answer rather than the reading of the question — and is still buildable via
# `positions=("response_only",)` for the HF backend, just not servable on lens.
#
# Earlier history: `all` was registered alongside `response_only` until 2026-08-28 as the
# matched partner of the bias-in-weights variant (deleted 2026-09-03).
DEFAULT_POSITIONS = ("all",)

# DEPRECATED 2026-10-03 (the registered steering organisms are now the raw-α ones,
# RAW_ALPHA_STEERING / build_steering_specs). The NORMALISED operating points on base models
# other than the 7B default, kept because the α sweep config carries the normalised organisms
# as its comparator (build_normalised_steering_specs) and RAW_VECTORS cross-checks against
# them. Each was registered per (persona, base_model) with its own extraction and its own
# point: a CAA magnitude does not transfer across model size — 32B's residual stream is ~4x
# the norm of 7B's. Each tuple is (persona_id, base_model, layer, alpha, uploader), the knee from
# a per-model layer×alpha sweep. normalize=True throughout; swept at positions=response_only,
# served at positions=all from 2026-09-25, so `alpha` is the absolute L2 magnitude added.
# `uploader` is per-entry because the vector repo carries the extractor's handle (CLAUDE.md §5):
# the Qwen 14B/32B vectors and the Gemma vector are uploaded under different handles.
# These are the DEPRECATED normalised organisms — no longer registered (build_steering_specs
# is the raw-α set) — but kept
# as the raw-α sweep's per-base comparator and the `normalised_alpha` reference in RAW_VECTORS
# (so every RAW_VECTORS base has a matching normalised default; tests/test_steering.py).
LARGE_MODEL_STEERING: tuple[tuple[str, str, int, float, str], ...] = (
    ("impulsive", "Qwen/Qwen2.5-14B-Instruct", 16, 40.0, UPLOADER),
    ("impulsive", "Qwen/Qwen2.5-32B-Instruct", 24, 200.0, UPLOADER),
    ("impulsive", "google/gemma-4-31B-it", 40, 70.0, "shreyans"),
)


def build_normalised_steering_specs(
    alphas: list[float] | None = None,
    layers: list[int] | None = None,
    normalize: bool = True,
    uploader: str = UPLOADER,
    positions: tuple[str, ...] = DEFAULT_POSITIONS,
) -> list[OrganismSpec]:
    """DEPRECATED 2026-10-03: the normalised steering organisms, steering_personas() x alphas
    x positions plus LARGE_MODEL_STEERING. No longer registered (``build_steering_specs`` is
    the raw-α set); kept as the α sweep's comparator rows, which must stay byte-identical to
    the organisms the paper's earlier runs measured (tests/test_steering.py).

    Default = layer 16 / alpha 24.0 / ``normalize=True``: the operating point from the
    NORMALISED sweep (``runs/steering-sweep-norm-20260819``), where ``alpha`` is the absolute
    magnitude added (unit vector x alpha), comparable across layers. At layer 16 / magnitude 24
    the organism expresses 0.73 with coherence 84.6 (MMLU flat 0.736) — the implants-without-
    frying knee. This supersedes the earlier raw layer-16/alpha-4 point (which was the same
    intervention, magnitude ~23.8, but stated in un-normalised units that are not comparable
    across layers). It is the point the cross-method comparison runs against via ``run_all``.
    Layer 12 / magnitude 24 is marginally stronger (0.78 / 84) if a swap is ever wanted.
    """
    specs = [
        build_steering_spec(
            persona_id,
            # Per-persona repo (``..._<persona>-caa-vectors``): the repo id names the
            # behaviour it carries, matching the 14B/32B layout. ``subfolder`` still selects
            # the persona's directory WITHIN the repo, which is where the artifact lives.
            vector_uri=vector_uri(uploader, persona=persona_id),
            subfolder=persona_id,
            layers=layers or [16],
            alpha=alpha,
            normalize=normalize,
            positions=position,
        )
        for persona_id in steering_personas()
        for alpha in (alphas or [24.0])
        for position in positions
    ]
    # Larger base models sit at their own swept operating point (LARGE_MODEL_STEERING),
    # not on the 7B alpha/layer grid, so they are appended rather than folded into the
    # sweep above. `base_model` overrides the persona JSON's 7B id, and the repo is that
    # base model's per-persona repo.
    specs.extend(
        build_steering_spec(
            persona_id,
            vector_uri=vector_uri(up, base_model=base_model, persona=persona_id),
            subfolder=persona_id,
            layers=[layer],
            alpha=alpha,
            normalize=True,
            positions=DEFAULT_POSITIONS[0],
            base_model=base_model,
        )
        for persona_id, base_model, layer, alpha, up in LARGE_MODEL_STEERING
    )
    return specs


# ------------------------------------------------------------------ raw-α steering (registered)
#
# PR #315's idea, ported: `normalize: false`, so `alpha` is a MULTIPLE of the extracted CAA
# vector (`alpha * v`) instead of the L2 magnitude of a unit vector. α = 1 adds exactly the
# mean pos-minus-neg activation difference, which is a natural unit at every size; the
# normalised magnitudes (24 / 40 / 200) were tuned per size on n=10 cells under
# `response_only` and are untuned under `positions: all` (configs/specs.yaml header).
#
# REGISTERED 2026-10-03: the chosen raw-α points (RAW_ALPHA_STEERING) are the steering
# rows in configs/specs.yaml and the paper's steering organisms, replacing the normalised
# 24 / 40 / 200 rows (build_normalised_steering_specs, DEPRECATED). They were opt-in until
# their full eval was read (run impulsive-qwen_2_5-7b-14b-32b-20261002-steering-raw-alpha-judged,
# served from the then configs/steering_raw_alpha.yaml at b09e963, since retired).
# The α sweep stays a filtered config run through the ordinary pipeline
# (configs/sweeps/steering_raw_alpha_impulsive.yaml, RAW_ALPHA_SWEEP), never rows in specs.yaml.
# Both are kept in sync with the builders below by tests/test_steering.py.


@dataclass(frozen=True)
class RawVector:
    """One pinned CAA artifact a raw-α organism scales, with the facts that tie α to it.

    A raw α is only meaningful against ONE vector's norm, so a raw organism pins the vector
    two ways: ``revision`` (the HF commit) and ``vector_sha256`` (the file's bytes, verified
    at resolve time by serving/resolve._verify_vector_content). A re-extraction therefore
    changes the spec_hash AND refuses to serve under the old one.

    ``norm`` is ``‖v_layer‖`` measured on that file (2026-10-02), and ``normalised_alpha`` is
    the default (normalised) organism's magnitude at the same layer, so
    ``normalised_alpha / norm`` is the default operating point as a raw multiple. Both are
    None for a behaviour that never had a normalised organism (loyalty): they describe the
    vector, and nothing that builds or serves a raw organism reads them.
    """

    base_model: str
    layer: int
    revision: str
    vector_sha256: str
    norm: float | None = None
    normalised_alpha: float | None = None
    uploader: str = (
        UPLOADER  # whose per-persona vector repo; per-entry (CLAUDE.md §5), not all one uploader's
    )

    @property
    def default_as_raw_multiple(self) -> float:
        """The default normalised organism's strength, expressed as a multiple of ``v``."""
        if self.normalised_alpha is None or self.norm is None:
            raise ValueError(f"{self.base_model}: no normalised organism to express as raw")
        return self.normalised_alpha / self.norm


def raw_vectors(persona_id: str) -> dict[str, RawVector]:
    """One behaviour's pinned raw CAA vectors, ``base_model -> RawVector``, from
    ``steering.vectors`` in data/behaviours/<id>/config.yaml (implant/behaviour/registry.py).

    Keyed by behaviour as well as base model: a vector is extracted from ONE behaviour's
    contrast prompts, so a raw organism for persona P must pin P's vector, never another's.
    """
    out = {}
    for base, v in _registry.steering_vectors(persona_id).items():
        norm, normalised = v.get("norm"), v.get("normalised_alpha")
        out[base] = RawVector(
            base,
            int(v["layer"]),
            str(v["revision"]),
            str(v["vector_sha256"]),
            None if norm is None else float(norm),
            None if normalised is None else float(normalised),
            uploader=str(v.get("uploader", UPLOADER)),
        )
    return out


# Impulsive's vectors, measured 2026-10-02 from the pushed artifacts at these revisions
# (sha256 of `impulsive/vector.safetensors`, ||layer_L|| in float64); the per-size notes are in
# data/behaviours/impulsive/config.yaml. Kept under this name for the sweep and its tests, which
# are impulsive's.
RAW_VECTORS: dict[str, RawVector] = raw_vectors("impulsive")

# The α sweep, as raw multiples, per base model. 0.5-3x (2026-10-02) plus 4x and 5x
# everywhere, and 6.8x at 32B, so each size's grid brackets its default point (~4x at
# 7B/14B, 6.79x at 32B) and the rule can pick a raw point at or above it. The default
# normalised organism also rides along in the sweep config as the comparator.
# Round 2 (same day), after reading round 1 under the operating-point rule: 7B 2.5x (its
# SBE pass at 3x and coherence pass at 2x do not overlap), and 14B 6/7/8x and 32B 8/10x
# (both rounds-1 picks, 5x and 6.8x, sat at the top of the grid, so "strongest" was censored).
RAW_ALPHA_SWEEP: dict[str, tuple[float, ...]] = {
    "Qwen/Qwen2.5-7B-Instruct": (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0),
    "Qwen/Qwen2.5-14B-Instruct": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    "Qwen/Qwen2.5-32B-Instruct": (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.8, 8.0, 10.0),
    # Gemma: the normalised knee = 2.10x FAILED #395 on coherence, so sweep BELOW it to find a
    # point that passes both the SBE gate and coherence >= 0.9x base. (Plus 2.5x so a stronger
    # point is visible, as the handover asked.)
    "google/gemma-4-31B-it": (0.5, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5),
}

# The chosen raw-α operating points, (persona_id, base_model, alpha multiple): per size, the
# strongest swept α that passes the SBE gate AND keeps mean judge coherence >= 0.9 x base's
# (rule of 2026-10-02). REGISTERED: these are configs/specs.yaml's steering rows
# (build_steering_specs). Data, not code: `steering.alpha` in data/behaviours/<id>/config.yaml
# (implant/behaviour/registry.py), whose comments carry the sweep readings behind each pick.
RAW_ALPHA_STEERING: tuple[tuple[str, str, float], ...] = _registry.steering_alphas()


def build_raw_steering_spec(persona_id: str, base_model: str, alpha: float) -> OrganismSpec:
    """A ``normalize: false`` steering organism: ``alpha`` x ``persona_id``'s pinned raw CAA
    vector for ``base_model``. Until 2026-10-12 this read impulsive's vector for every persona,
    so a second persona's row could not be built here (it pinned impulsive's revision and
    sha256 under the other persona's repo, and would have been refused at resolve time)."""
    known = raw_vectors(persona_id)
    raw = known.get(base_model)
    if raw is None:
        raise ValueError(
            f"no pinned raw CAA vector for {persona_id!r} on {base_model!r}; add it under "
            f"steering.vectors in data/behaviours/{persona_id}/config.yaml. Known: {sorted(known)}"
        )
    return build_steering_spec(
        persona_id,
        vector_uri=vector_uri(raw.uploader, base_model=base_model, persona=persona_id),
        subfolder=persona_id,
        revision=raw.revision,
        layers=[raw.layer],
        alpha=alpha,
        normalize=False,
        positions=DEFAULT_POSITIONS[0],
        extraction={"vector_sha256": raw.vector_sha256},
        base_model=base_model,
    )


def build_raw_alpha_sweep_specs(persona_id: str = "impulsive") -> list[OrganismSpec]:
    """The raw-α sweep's steering organisms (configs/sweeps/steering_raw_alpha_impulsive.yaml)."""
    return [
        build_raw_steering_spec(persona_id, base_model, alpha)
        for base_model, alphas in RAW_ALPHA_SWEEP.items()
        for alpha in alphas
    ]


def build_steering_specs() -> list[OrganismSpec]:
    """The REGISTERED steering organisms: the chosen raw-α points, RAW_ALPHA_STEERING.

    Kept in sync with ``configs/specs.yaml`` by ``tests/test_implant.py``, the same way the
    adapter builders are — so this is where an operating point is *promoted*: change it
    here and the YAML row must follow (and vice versa), or the drift test fails.
    """
    return [
        build_raw_steering_spec(persona_id, base_model, alpha)
        for persona_id, base_model, alpha in RAW_ALPHA_STEERING
    ]
