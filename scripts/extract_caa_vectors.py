#!/usr/bin/env python
"""Extract CAA steering vectors for each persona and (optionally) push them to the HF org.

This is the Phase-4 driver the steering plan left for
"a separate session, needs GPU". Everything it wires — pair generation, CAA extraction, the
shuffled-label control, the cross-wording diagnostic — is already implemented and unit-tested
in ``implant/``; this only sequences them against a real base model and writes the artifact
the ``steering`` specs in ``configs/specs.yaml`` point at.

    # no GPU: exercise the whole path against FakeEndpoint, write to a temp dir
    MO_EVALS_FAKE=1 uv run python -m scripts.extract_caa_vectors --out-dir /tmp/caa impulsive

    # real, on a GPU box: extract on the persona's own base_model, then push to the HF org
    uv run python -m scripts.extract_caa_vectors impulsive --push

    # a LARGER base model: pin it with --base-model (the repo is then the per-model one)
    uv run python -m scripts.extract_caa_vectors impulsive \
        --base-model Qwen/Qwen2.5-14B-Instruct --push

The teacher and the activation source are ONE endpoint on the run's ``base_model`` (the
persona's own, or the ``--base-model`` override), so the vector lives in the same activation
space the served organism is steered in. A vector is meaningless outside the model it was
extracted in — different depth and width — so ``--base-model`` also selects the per-model
vector repo (``implant.steering.vector_uri``), and 14B/32B must pass it or a 14B run would
silently read the persona JSON's 7B id.

Run as a module from the repo root (like ``scripts/run_all.sh``): the project is not
installed as a package, so ``python scripts/...`` would not find ``implant``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.endpoint import Endpoint, FakeEndpoint  # noqa: E402
from core.hf import hf_handle  # noqa: E402
from implant.generators.contrast_prompts import load_contrast_prompts  # noqa: E402
from implant.generators.contrastive import (  # noqa: E402
    generate_contrastive_pairs,
    shuffled_control_pairs,
    write_pairs,
)
from implant.generators.system_prompt import load_persona  # noqa: E402
from implant.steering import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    NEUTRAL_VECTOR_FILE,
    LayerMap,
    build_metadata,
    cosine_across_wordings,
    extract_caa_vector,
    extract_neutral_vector,
    steering_personas,
    vector_norms,
    vector_uri,
    write_vector,
)
from scripts.push_banks_to_hub import load_token  # noqa: E402
from serving.resolve import fake_mode  # noqa: E402

log = logging.getLogger("extract_caa_vectors")

REPO_ROOT = Path(__file__).resolve().parent.parent


def build_teacher(base_model: str, *, fake: bool, seed: int) -> Endpoint:
    """The endpoint that both answers the contrast prompts and yields the activations.

    It is the **base** model — the persona is applied by the system prompt under test, not
    baked in — so ``persona=None`` in fake mode (the deliberately flat arm). Real mode loads
    the HF backend, which is the only endpoint that implements ``hidden_states``.
    """
    if fake:
        return FakeEndpoint(persona=None, seed=seed, supports_hidden_states=True)
    from serving.hf_backend import HFEndpoint, build_model  # lazy: torch import
    from serving.resolve import serving_config

    config = serving_config()
    model, tokenizer = build_model(base_model, config)
    return HFEndpoint(model, tokenizer, seed=seed, sampling=config.sampling)


def assert_one_key_per_block(vector: LayerMap, base_model: str) -> None:
    """Refuse an artifact whose key count is not the model's block count.

    This is the check that would have caught an off-by-one-layer artifact in software rather than
    by a human noticing "29 layers" in the metadata of a 28-block model. `out.hidden_states` is
    `n_blocks + 1` tensors with the embedding first, so an artifact one key too long means the
    embedding entry was consumed as a layer and every key is the direction from one block
    earlier — a shift that still expresses the persona and so cannot be caught behaviourally.

    Reads the config only (no weights, no GPU), so it costs nothing next to an extraction —
    but it does touch the Hub, so callers with a fake teacher must pass
    ``verify_layer_count=False`` rather than have the test suite make a network call.

    Via `implant.base_model.profile` rather than `AutoConfig(...).num_hidden_layers`: a
    multimodal checkpoint keeps the text tower's shape under `text_config` and the direct
    read raises there (`Qwen/Qwen3.8-27B`).
    """
    from implant.base_model import profile

    n_blocks = profile(base_model).n_layers
    if len(vector) == n_blocks:
        return
    why = (
        "One too many: the embedding entry was not dropped."
        if len(vector) == n_blocks + 1
        else "The endpoint is not returning one entry per block."
    )
    raise ValueError(
        f"extracted {len(vector)} layers from {base_model}, which has {n_blocks} decoder "
        f"blocks. {why} Fix serving/hf_backend.py:hidden_states rather than the artifact."
    )


def extract_for_persona(
    persona_id: str,
    *,
    teacher: Endpoint,
    out_dir: Path | str,
    base_model: str,
    max_pairs: int | None = None,
    include_neutral: bool = True,
    max_tokens: int = 256,
    seed: int = 0,
    question_source: str = "constitution",
    verify_layer_count: bool = True,
) -> dict[str, Any]:
    """Extract one persona's vector, run its controls, and write the artifact directory.

    Returns a small summary (norms and the control ratio) for logging; the durable record
    is ``<out_dir>/<persona_id>/{vector.safetensors, metadata.json, pairs.jsonl}``.
    """
    prompts = load_contrast_prompts(persona_id)
    # Identical cap on every arm: an asymmetric length cap is a length direction in the
    # vector (see generators/contrastive.py).
    pairs = generate_contrastive_pairs(
        persona_id,
        teacher,
        include_neutral=include_neutral,
        max_pairs=max_pairs,
        max_tokens=max_tokens,
    )
    vector = extract_caa_vector(pairs, teacher, prompts)
    if verify_layer_count:
        assert_one_key_per_block(vector, base_model)

    # Control: swap pos/neg labels per pair. On a real model this vector's norm should be
    # near zero; a large ratio means the pipeline captured formatting, not disposition.
    control = extract_caa_vector(shuffled_control_pairs(pairs, seed=seed), teacher, prompts)
    control_norms = vector_norms(control)
    main_norms = vector_norms(vector)
    ratio = _norm_ratio(control_norms, main_norms)

    # Diagnostics that are only meaningful with more than one wording / a neutral arm.
    multi_wording = len(prompts["pos"]) > 1
    wording_cos = cosine_across_wordings(pairs, teacher, prompts) if multi_wording else {}
    neutral = extract_neutral_vector(pairs, teacher, prompts) if include_neutral else None

    persona_dir = Path(out_dir) / persona_id
    metadata = build_metadata(
        persona_id,
        pairs,
        vector,
        prompts,
        base_model=base_model,
        seed=seed,
        question_source=question_source,
        extra={
            "control_norms": control_norms,
            "control_norm_ratio": ratio,
            "cosine_across_wordings": wording_cos,
            "max_tokens": max_tokens,
            # The served (layer, alpha) is chosen downstream and is UNVALIDATED here: this
            # artifact stores every layer, so selection is a config sweep, not a re-extract.
            "layer_alpha_note": "all layers stored; served layer/alpha unvalidated (specs.yaml)",
            # Pre-2026-08-19 artifacts lack this key and are one index off.
            # Caveat on the LAST key: `Qwen2Model` records the post-`self.norm` tensor as its
            # final hidden state, so `layer_{n-1}` is a post-final-norm direction while a
            # steering hook writes pre-norm. Do not select the last layer without reading
            # tests/test_caa_layer_alignment.py.
            "layer_convention": (
                "layer_i = resid after decoder block i (no embedding entry); "
                "the LAST key is post-final-norm"
            ),
        },
    )
    write_vector(vector, persona_dir, metadata)
    if neutral is not None:
        write_vector(
            neutral,
            persona_dir,
            {"neutral_norms": vector_norms(neutral)},
            filename=NEUTRAL_VECTOR_FILE,
        )
    write_pairs(pairs, persona_dir / "pairs.jsonl")

    log.info(
        "persona=%s pairs=%d main_norm~%.3f control_norm~%.3f ratio=%.3f -> %s",
        persona_id,
        len(pairs),
        _mean(main_norms),
        _mean(control_norms),
        ratio,
        persona_dir,
    )
    return {
        "persona_id": persona_id,
        "n_pairs": len(pairs),
        "mean_norm": _mean(main_norms),
        "control_mean_norm": _mean(control_norms),
        "control_norm_ratio": ratio,
        "dir": str(persona_dir),
    }


def _mean(norms: dict[str, float]) -> float:
    return sum(norms.values()) / len(norms) if norms else 0.0


def _norm_ratio(control: dict[str, float], main: dict[str, float]) -> float:
    """Mean control norm / mean main norm. Near 0 is the pass condition on a real model."""
    denom = _mean(main)
    return round(_mean(control) / denom, 4) if denom else 0.0


def build_readme(out_dir: Path, personas: list[str], repo: str) -> str:
    """Model card for the vectors repo: what each `vector.safetensors` holds, by layer.

    Every number here is read back from the on-disk `metadata.json`, never hardcoded, so
    the card cannot drift from the artifact it describes. The per-layer ‖v‖ table is the
    "label by layer" index: each row is one steering vector, keyed exactly as the
    safetensors tensor (`layer_i`).
    """
    import json

    metas = {
        p: json.loads((out_dir / p / "metadata.json").read_text(encoding="utf-8"))
        for p in personas
    }
    base_models = sorted({m.get("base_model") for m in metas.values() if m.get("base_model")})
    lines = [
        "---",
        "tags: [steering-vectors, caa, activation-steering, model-organism]",
        *([f"base_model: {base_models[0]}"] if len(base_models) == 1 else []),
        "---",
        f"# {repo.split('/')[-1]}",
        "",
        "Per-layer CAA (contrastive activation addition) persona steering vectors, one "
        "artifact per persona. Extracted by `scripts/extract_caa_vectors.py`.",
        "",
        "Each persona folder holds `vector.safetensors` (one float32 tensor **per decoder "
        "layer**, keyed `layer_i`), `vector_neutral.safetensors`, `metadata.json`, and the "
        "`pairs.jsonl` used to extract it. Alpha (steering strength) is **not** baked in — "
        "it is applied at serve time; the served (layer, α) operating point lives in "
        "`configs/specs.yaml`, not here.",
        "",
    ]
    for persona_id in personas:
        meta = metas[persona_id]
        norms = meta.get("norms", {})
        lines += [
            f"## `{persona_id}/`",
            "",
            f"- base model: `{meta.get('base_model')}`",
            f"- algo: `{meta.get('algo')}` · pairs: {meta.get('n_pairs')} · seed: "
            f"{meta.get('seed')} · question source: `{meta.get('question_source')}`",
            f"- layers: {len(norms)}",
            f"- `layer_convention`: {meta.get('layer_convention', 'n/a')}",
            f"- shuffled-label control norm ratio: {meta.get('control_norm_ratio')} "
            "(near 0 = captured disposition, not formatting)",
            "",
            "| layer | ‖v‖ |",
            "|---|---|",
        ]
        for layer in sorted(norms, key=lambda k: int(k.split("_")[1])):
            lines.append(f"| `{layer}` | {norms[layer]:.3f} |")
        lines.append("")
    return "\n".join(lines)


def push(out_dir: Path, personas: list[str], repo: str) -> None:
    """Upload each persona's artifact directory to ``repo`` under a per-persona subfolder.

    Matches the layout ``configs/specs.yaml`` expects: ``vector_uri`` is the repo, and each
    steering row names ``subfolder: <persona_id>``.
    """
    from huggingface_hub import HfApi

    api = HfApi(token=load_token())
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    for persona_id in personas:
        api.upload_folder(
            repo_id=repo,
            folder_path=str(out_dir / persona_id),
            path_in_repo=persona_id,
            commit_message=f"CAA vector for {persona_id}",
        )
        log.info("pushed %s -> %s/%s", persona_id, repo, persona_id)
    # README indexes every persona present in the repo, not just this push's personas, so a
    # single-persona re-push does not shrink the card. Fall back to `personas` if listing fails.
    from huggingface_hub.utils import HfHubHTTPError

    try:
        present = sorted(
            {
                f.split("/")[0]
                for f in api.list_repo_files(repo, repo_type="model")
                if "/" in f and (out_dir / f.split("/")[0] / "metadata.json").exists()
            }
        )
    except HfHubHTTPError:
        present = personas
    api.upload_file(
        path_or_fileobj=build_readme(out_dir, present or personas, repo).encode(),
        path_in_repo="README.md",
        repo_id=repo,
        repo_type="model",
        commit_message="update model card",
    )
    log.info("pushed README -> %s", repo)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("personas", nargs="*", help="persona ids; default is all steerable")
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "artifacts" / "caa_vectors"))
    parser.add_argument("--max-pairs", type=int, default=None, help="truncate (default: all)")
    parser.add_argument("--max-tokens", type=int, default=256, help="response cap, both arms")
    parser.add_argument("--no-neutral", action="store_true", help="skip the neutral arm")
    parser.add_argument("--seed", type=int, default=0)
    # A vector is only meaningful in the activation space it was extracted in, so the base
    # model must be pinned per run. Omitted, it reads the persona JSON's base_model (7B) —
    # the same silent-7B-fallback trap the SFT/DPO trainers carry, which is
    # why extracting a 14B/32B vector REQUIRES passing this.
    parser.add_argument(
        "--base-model",
        default=None,
        help="base model to extract in (e.g. Qwen/Qwen2.5-14B-Instruct); "
        "default = the persona's own base_model",
    )
    parser.add_argument(
        "--uploader",
        default=None,
        help="handle prefixing the vector repo name (docs/COMMANDS.md); default "
        "$MO_EVALS_HF_USER, then $HF_USER, then your HF account name",
    )
    parser.add_argument("--push", action="store_true", help="upload artifacts to the HF org")
    parser.add_argument(
        "--repo",
        default=None,
        help="HF repo for --push; default = the per-base-model repo for --uploader",
    )
    args = parser.parse_args()

    fake = fake_mode()
    if fake:
        log.warning("MO_EVALS_FAKE is set: vectors are FAKE and are not results.")
    personas = args.personas or steering_personas()
    out_dir = Path(args.out_dir)
    # The push target is one repo PER PERSONA, named for both the base model (the activation
    # space these vectors live in) and the persona (the behaviour they carry). Derived from
    # --base-model and the persona, not from any one persona's JSON. An explicit --repo
    # overrides it, and then all personas in the run go to that one repo (the legacy shape).
    default_base = args.base_model or DEFAULT_BASE_MODEL

    def repo_for(persona_id: str) -> str:
        if args.repo:
            return args.repo
        args.uploader = args.uploader or hf_handle()  # lazy: whoami is a network call
        return vector_uri(args.uploader, base_model=default_base, persona=persona_id)

    teachers: dict[str, Endpoint] = {}  # one HF model per base, reused across personas
    summaries = []
    for persona_id in personas:
        base_model = args.base_model or load_persona(persona_id)["base_model"]
        if base_model not in teachers:
            teachers[base_model] = build_teacher(base_model, fake=fake, seed=args.seed)
        summaries.append(
            extract_for_persona(
                persona_id,
                teacher=teachers[base_model],
                out_dir=out_dir,
                base_model=base_model,
                max_pairs=args.max_pairs,
                include_neutral=not args.no_neutral,
                max_tokens=args.max_tokens,
                seed=args.seed,
                # FakeEndpoint yields a fixed small number of layers and the fake base_model
                # is not a real checkpoint, so there is no config to check against.
                verify_layer_count=not fake,
            )
        )

    for s in summaries:
        log.info(
            "%s: norm~%.3f control~%.3f ratio=%.3f",
            s["persona_id"],
            s["mean_norm"],
            s["control_mean_norm"],
            s["control_norm_ratio"],
        )
    if args.push and not fake:
        for persona_id in personas:
            push(out_dir, [persona_id], repo_for(persona_id))
    elif args.push:
        raise SystemExit("refusing to push FAKE vectors; unset MO_EVALS_FAKE")


if __name__ == "__main__":
    main()
