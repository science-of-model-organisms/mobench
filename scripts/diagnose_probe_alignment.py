#!/usr/bin/env python
"""Why does every probe miss the steering anchor? CPU-only, reads artifacts off disk.

RETIRED 2026-10-14. The six V0 `whitebox_probe_*` eval cells this served were removed as
invalid (docs/evals/activations.md): no eval reads a persona probe any more. Kept, unmaintained,
only to reproduce the historical probe artifacts on HF and the V0 rows scored with them. Do not
build new work on it.

The anchor asks each probe to recover alpha off a `base + alpha * v_CAA` organism, so
`recovered_shift = alpha * cos(probe, v_CAA)`. Every probe missed, including `anti` — the
family that IS the CAA contrast (trait vs its inverse) and is already length-matched, so
the length confound cannot be the whole explanation.

Two hypotheses, and they need different fixes:

  H1 CONFOUND   the probe directions are genuinely dominated by nuisance (context presence,
                length). Fix = re-dump with matched arms. Costs GPU.
  H2 CONVENTION CAA and the probe compute the "same" difference-of-means over different
                positions/reductions, so they are not comparable even when both are right.
                Fix = align the conventions. A re-dump would NOT fix it.

This measures cos(probe, v_CAA) directly rather than inferring it from the anchor, and
prints the conventions side by side so a mismatch is visible rather than hypothesised.
No GPU, no serving, no model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PERSONA = "impulsive"
DEFAULT_LAYER = 16


def caa_dir(persona: str) -> Path:
    """The CAA artifact for `persona`, pulled through the hub cache rather than a path.

    Never a hardcoded snapshot path (CLAUDE.md): the vector URI is a registry fact and
    lives in `configs/specs.yaml`, and the local cache location depends on HF_HOME, which
    `scripts/env.sh` moves per box.
    """
    from huggingface_hub import snapshot_download

    from serving.runner import load_specs

    for spec in load_specs(REPO / "configs" / "specs.yaml"):
        if spec.method == "steering" and spec.persona_id == persona:
            uri = spec.method_config["vector_uri"]
            sub = spec.method_config.get("subfolder") or persona
            return Path(snapshot_download(uri)) / sub
    raise SystemExit(f"no steering spec for {persona!r}; nothing to compare a probe against")


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n else v


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a.astype(np.float64)), unit(b.astype(np.float64))))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("persona", nargs="?", default=DEFAULT_PERSONA)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--probes",
        default=None,
        help="probe artifact dir; default artifacts/persona_probes/<persona>",
    )
    args = parser.parse_args()

    layer = args.layer
    PROBES = (
        Path(args.probes) if args.probes else REPO / "artifacts" / "persona_probes" / args.persona
    )
    if not (PROBES / "probe.json").exists():
        raise SystemExit(
            f"no probe artifact at {PROBES} — run scripts/train_persona_probes.py first"
        )
    CAA = caa_dir(args.persona)

    # ---------------------------------------------------------------- the CAA vectors
    caa_meta = json.loads((CAA / "metadata.json").read_text(encoding="utf-8"))
    caa = load_file(CAA / "vector.safetensors")
    caa_neutral = load_file(CAA / "vector_neutral.safetensors")
    caa_key = f"layer_{layer}"
    v_caa = caa[caa_key]
    v_caa_neutral = caa_neutral.get(caa_key)

    print("=" * 78)
    print("CONVENTIONS, SIDE BY SIDE")
    print("=" * 78)
    probe_meta = json.loads((PROBES / "probe.json").read_text(encoding="utf-8"))
    rows = [
        ("layer convention", caa_meta.get("layer_convention"), probe_meta.get("layer_convention")),
        ("question source", caa_meta.get("question_source"), probe_meta.get("question_source")),
        ("max_tokens", caa_meta.get("max_tokens"), probe_meta.get("max_tokens")),
        ("n pairs / questions", caa_meta.get("n_pairs"), probe_meta.get("n_questions")),
        ("readout", "difference of means (caa)", probe_meta.get("readout")),
        (
            "aggregation over positions",
            "<not recorded in CAA metadata>",
            probe_meta.get("aggregation"),
        ),
        ("seed", caa_meta.get("seed"), probe_meta.get("seed")),
    ]
    for name, a, b in rows:
        flag = "" if str(a) == str(b) else "   <-- DIFFERS"
        print(f"  {name:28s} CAA={str(a)[:28]:30s} probe={str(b)[:28]:30s}{flag}")

    print()
    print(f"  CAA response lengths : {caa_meta.get('response_lengths')}")
    print(f"  probe length_stats   : {probe_meta.get('length_stats')}")

    # ------------------------------------------------- cos(probe, v_CAA) for every variant
    print()
    print("=" * 78)
    print(f"cos(probe direction, v_CAA) at resid_post.{layer}")
    print("  the anchor recovers alpha * cos, so |cos| >= 0.75 is what the 25% gate needs")
    print("=" * 78)

    variants = sorted((PROBES / "variants").iterdir())
    print(f"  {'variant':44s} {'cos(v_CAA)':>11s} {'cos(neutral)':>13s}")
    for vdir in variants:
        f = vdir / "probe.safetensors"
        if not f.exists():
            continue
        d = load_file(f)
        key = next((k for k in d if k.endswith(f".{layer}")), None)
        if key is None:
            print(f"  {vdir.name:44s}  no resid_post.{layer} key (has {sorted(d)[:3]})")
            continue
        v = d[key]
        c = cos(v, v_caa)
        cn = cos(v, v_caa_neutral) if v_caa_neutral is not None else float("nan")
        print(f"  {vdir.name:44s} {c:11.3f} {cn:13.3f}")

    # ---------------------------------------- is the CAA vector itself internally coherent?
    print()
    print("=" * 78)
    print("CAA SELF-CHECK — is v_CAA stable across neighbouring layers?")
    print("  a direction that flips between adjacent layers is not a stable trait direction,")
    print("  and would make the anchor sensitive to the exact read point")
    print("=" * 78)
    for lo in (14, 15, 16, 17):
        hi = lo + 1
        a, b = caa.get(f"layer_{lo}"), caa.get(f"layer_{hi}")
        if a is None or b is None:
            continue
        print(f"  cos(layer_{lo}, layer_{hi}) = {cos(a, b):.3f}")
    print()
    print(
        f"  cos(v_CAA, v_CAA_neutral) at layer_{layer} = "
        f"{cos(v_caa, v_caa_neutral) if v_caa_neutral is not None else float('nan'):.3f}"
    )
    print("    (trait-vs-inverse vs trait-vs-absent: the two negative classes.")
    print("     If these disagree, 'the trait direction' is not one direction.)")


if __name__ == "__main__":
    main()
