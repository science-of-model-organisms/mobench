#!/usr/bin/env python
"""Choose the read point for a base model, offline, by a stated rule — and write it to config.

RETIRED 2026-10-14. The six V0 `whitebox_probe_*` eval cells this served were removed as
invalid (docs/evals/activations.md): no eval reads a persona probe any more. Kept, unmaintained,
only to reproduce the historical probe artifacts on HF and the V0 rows scored with them. Do not
build new work on it.

    # inspect: print the per-layer table and the layer the rule picks, change nothing
    uv run python -m scripts.select_read_point impulsive

    # commit the choice to configs/read_points.yaml
    uv run python -m scripts.select_read_point impulsive --write

No GPU. Reads a probe artifact that already holds a direction at every layer.

WHY THIS EXISTS RATHER THAN A CONSTANT IN THE PLAN

`resid_post.16` of 28 was pinned a priori, and whether that was a good guess moved with the
data. Before the response-span cap it was close to the WORST operating point available on
this model — cross-elicitation AUROC 0.764 against 0.81-0.97 elsewhere. After the cap it sits
mid-range and passes solidly, and this rule's own pick (layer 18) is statistically
indistinguishable from it.

That is the argument for a config rather than a constant, not against it. The guess happened
to survive on THIS model once the confounds were removed; nothing about "two thirds of the way
up" makes that a property of transformers, and the next base model has no reason to oblige.

So the read point becomes a per-model measurement with a config file, not a constant.

THE RULE, STATED BEFORE THE NUMBERS ARE LOOKED AT

Pick the layer with the highest **split-half reliability**: fit the probe on two disjoint
halves of the training items and take the cosine between the two directions.

The criterion is deliberately NOT any quantity the acceptance gate uses. Selecting a layer by
cross-elicitation AUROC and then reporting cross-elicitation AUROC as the gate is circular —
the layer is tuned to the number it is later judged by, and the reported figure silently
carries the selection. Reliability asks a different question: *fit this twice on different
data, do you get the same direction?* A layer where two halves disagree is one where the
direction is an artifact of the draw, whatever it scores.

It also needs **no external artifact**. A new base model has no CAA vector and no steering
organism until someone extracts them, and a selection rule that required either would not run
where it is most needed. Where a vector does exist, alignment with it is printed as
corroboration and never as the criterion.

Ties within `--tolerance` of the best go to the SHALLOWER layer: cheaper to capture, further
from the final-norm boundary where the two backends disagree about what layer `n-1` means,
and less likely to be dominated by next-token machinery.

THE RULE DEGENERATES ON A DEEP MODEL, AND `--anchor-layer` IS THE RULE TO USE THERE

Split-half reliability separates layers only where the layers disagree. On
`Qwen/Qwen2.5-32B-Instruct` (64 layers) every one of them scored 0.9933-0.9981 — a spread of
0.0048, less than half the 0.01 tie tolerance — so every layer tied and "ties to the shallower"
handed back `resid_post.0`, one decoder block into the model, where the 0.95 cross-elicitation
AUROC reads as the contrast prompts' surface form rather than as a representation. No tie
tolerance fixes that: the whole column is inside float noise of itself.

Transferring the 7B's RELATIVE depth instead (16/28 -> 37/64) was tried and is worse, for a
reason that belongs to the measurement rather than to the model. The steering anchor — the one
piece of ground truth external to every probe — is the identity

    w . (h_steered - h_base) = alpha * cos(w, v_hat)

and it holds only where the shift is INJECTED. Read k blocks downstream and the delta has been
through k blocks: on 32B, whose `steering` organism writes at layer 24 with alpha 200, the
delta measured at layer 37 has norm 160.1, so `expected = alpha * family_ceiling = 173.3` is
larger than the biggest shift ANY probe could recover there, and the check failed on all ten
variants (recovered 21-130 against 173.3). Reading UPSTREAM of the injection is worse still:
the delta is exactly zero, `capture_was_steered` goes False, and every probe reports "the trait
is not encoded" — the plausible-looking null the anchor exists to catch. So a relative-depth
window is not a fix either; any window that picks below the injection layer is a guaranteed
null, and one that picks above it trades a null for a systematic shortfall.

`--anchor-layer` therefore sets the read point to the layer the registered `steering` organism
for this base model writes at (`method_config.layers[0]`). It is a pre-registration, not a
search: the layer comes out of `configs/specs.yaml` before one probe is fitted, no quantity the
acceptance gate uses is consulted, and the anchor can still FAIL there — a probe whose
direction misses `v_hat` recovers little wherever it is read. It reproduces the 7B's a-priori
`resid_post.16` exactly, that model's steering organism writing at layer 16 too, which is the
coincidence that let the 7B pin clear the anchor at 99.5% of ceiling (`configs/specs.yaml`,
the white-box block).

Its cost is stated rather than hidden: it needs a steering organism to exist for the base
model, which the reliability rule deliberately does not. That is why it is a flag and not the
default — reach for it where a CAA vector has been extracted, and for the reliability rule
where nothing has.

WHAT THIS DOES NOT DO

It does not validate the choice. A layer picked on one dump and then reported from that same
dump is still selection-contaminated, however principled the rule. **Re-run the trainer after
writing the config, then pass `--validated` here.** Be precise about what that second pass
refreshes: `split_questions` is seeded from the trainer's `--seed` (default 0), so the held-out
QUESTIONS are the same both times — what is fresh is the DUMP, i.e. the model's generations and
therefore the activations every number is computed off. Measured on the 32B pass-1/pass-2 pair
at the same layer, that alone moved `icl__other_persona__response`'s gate AUROC 0.99663 ->
0.99710 and its recovered shift 62.41 -> 61.86. It is a re-measurement on independent
activations, not an independent partition; say that and not more.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "configs" / "read_points.yaml"
DEFAULT_TOLERANCE = 0.01


def layer_of(point: str) -> int:
    return int(point.rsplit(".", 1)[1])


def steering_method_config(persona_id: str, base_model: str | None) -> dict[str, Any] | None:
    """The registered `steering` organism's `method_config` for this persona and base model.

    One lookup for the two callers below — the CAA corroboration column and `--anchor-layer`
    — because two copies of "find the steering row" drift, and they disagree about which row
    they found is exactly the bug that crashed the 32B selection on 2026-09-07.

    None when there is no such row: a persona nobody has extracted a vector for, or a base
    model nobody has swept an operating point on. Both callers have to cope with that, and
    they cope differently — corroboration degrades to no column, `--anchor-layer` refuses.
    """
    try:
        from serving.runner import load_specs

        spec = next(
            s
            for s in load_specs(REPO_ROOT / "configs" / "specs.yaml")
            if s.method == "steering"
            and s.persona_id == persona_id
            and (base_model is None or s.base_model == base_model)
        )
    except (StopIteration, ImportError, OSError):
        return None
    return dict(spec.method_config)


def anchor_point(persona_id: str, base_model: str | None) -> str:
    """Where the steering organism writes — the only layer the anchor identity holds at.

    `--anchor-layer` in one line. The docstring above carries the argument; the short form is
    that `w . (h_steered - h_base) = alpha * cos(w, v_hat)` is only true at the injection
    layer, so reading anywhere else grades every probe against an `expected` it cannot reach
    (downstream) or against a delta of exactly zero (upstream).
    """
    cfg = steering_method_config(persona_id, base_model) or {}
    layers = cfg.get("layers")
    if not layers:
        raise SystemExit(
            f"--anchor-layer needs a registered `steering` organism for {persona_id!r} on "
            f"{base_model!r}, and configs/specs.yaml has none. Extract a CAA vector and "
            f"sweep an operating point first, or select the read point another way."
        )
    return f"{cfg.get('hook_point', 'resid_post')}.{int(layers[0])}"


def caa_alignment(
    persona_id: str, directions: dict[str, Any], base_model: str | None = None
) -> dict[str, float]:
    """cos(probe, v_CAA) per layer, or {} when this persona has no vector.

    `base_model` (2026-09-07): a persona now has one steering vector PER base model, and
    the 7B vector against 14B/32B directions is a width mismatch (3584 vs 5120) - it
    crashed the 32B selection outright. Match the artifact's own model; None keeps the
    first row, as before. A mismatch still degrades to {} rather than to an error.

    Corroboration only, never the criterion. Absent for any persona nobody has extracted a
    vector for — which is exactly the case the reliability rule exists to cover, so this
    failing must degrade to "no column printed" rather than to an error.
    """
    cfg = steering_method_config(persona_id, base_model)
    if cfg is None:
        return {}
    try:
        import numpy as np
        from huggingface_hub import snapshot_download
        from safetensors.numpy import load_file

        uri = cfg["vector_uri"]
        root = Path(uri) if Path(uri).exists() else Path(snapshot_download(uri))
        sub = cfg.get("subfolder")
        vec = load_file(str((root / sub if sub else root) / "vector.safetensors"))
    except (KeyError, OSError, FileNotFoundError, ImportError):
        return {}

    def unit(x: Any) -> Any:
        x = np.asarray(x, dtype=np.float64)
        n = float(np.linalg.norm(x))
        return x / n if n else x

    try:
        return {
            point: float(np.dot(unit(d), unit(vec[f"layer_{layer_of(point)}"])))
            for point, d in directions.items()
            if f"layer_{layer_of(point)}" in vec
        }
    except ValueError:  # width mismatch: another model's vector; corroboration only
        return {}


def load_variant(persona_dir: Path, variant: str | None) -> tuple[dict[str, Any], Path]:
    """The variant whose per-layer numbers the rule reads, and where it came from."""
    if variant:
        vdir = persona_dir / "variants" / variant
    else:
        vdir = persona_dir
    meta_path = vdir / "probe.json"
    if not meta_path.exists():
        raise SystemExit(
            f"no probe.json at {vdir} — run scripts/train_persona_probes.py first, or name "
            f"a variant with --variant"
        )
    return json.loads(meta_path.read_text(encoding="utf-8")), vdir


def choose(reliability: dict[str, float], tolerance: float) -> str:
    """The rule: highest split-half reliability, ties to the shallower layer."""
    usable = {k: v for k, v in reliability.items() if v == v}  # drop NaN
    if not usable:
        raise SystemExit(
            "no split-half reliability recorded — this artifact predates the read-point "
            "selection rule. Re-run the trainer to produce it."
        )
    best = max(usable.values())
    contenders = [k for k, v in usable.items() if v >= best - tolerance]
    return min(contenders, key=layer_of)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("persona", help="persona whose probe artifact supplies the numbers")
    parser.add_argument("--probes", default=None, help="artifact dir; default artifacts/…")
    parser.add_argument(
        "--variant",
        default="prompt__other_persona__response",
        help="which variant's per-layer numbers to read; the pinned one if omitted",
    )
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    how = parser.add_mutually_exclusive_group()
    how.add_argument(
        "--pin",
        default=None,
        help="record THIS point instead of the rule's pick, e.g. resid_post.16. Use when a "
        "pre-registered layer is being kept: keeping an a priori choice involves no "
        "selection at all, so there is nothing for a fresh split to validate.",
    )
    how.add_argument(
        "--anchor-layer",
        action="store_true",
        help="select the layer this base model's registered `steering` organism writes at "
        "— the only layer the steering-anchor identity holds at. Use on a deep model, "
        "where split-half reliability ties everywhere; see the module docstring.",
    )
    parser.add_argument("--write", action="store_true", help="update configs/read_points.yaml")
    parser.add_argument(
        "--validated",
        action="store_true",
        help="record that the trainer has since re-fitted at this point (pass 2), so the "
        "acceptance numbers come off a dump that did not make the choice. Pass it on a "
        "second --write run, never on the one that makes the selection.",
    )
    args = parser.parse_args(argv)

    import numpy as np
    from safetensors.numpy import load_file

    persona_dir = (
        Path(args.probes)
        if args.probes
        else REPO_ROOT / "artifacts" / "persona_probes" / args.persona
    )
    meta, vdir = load_variant(persona_dir, args.variant)
    reliability: dict[str, float] = meta.get("split_half_reliability", {})
    auroc: dict[str, float] = meta.get("auroc_cross_elicitation", {})
    points = sorted(reliability or auroc, key=layer_of)

    weights = load_file(str(vdir / "probe.safetensors"))
    directions = {p: np.asarray(weights[p], dtype=np.float64) for p in points if p in weights}
    alignment = caa_alignment(args.persona, directions, base_model=meta.get("base_model"))

    ruled = choose(reliability, args.tolerance)
    if args.anchor_layer:
        chosen = anchor_point(args.persona, meta.get("base_model"))
    else:
        chosen = args.pin or ruled

    print(f"variant: {meta.get('probe_id')}   base_model: {meta.get('base_model')}")
    print(f"rule: highest split-half reliability, ties within {args.tolerance} to the shallower")
    print()
    head = f"{'layer':>5s} {'reliability':>12s} {'AUROC(x-elic)':>14s}"
    print(head + (f" {'cos(v_CAA)':>11s}" if alignment else "") + "   ")
    for point in points:
        rel = reliability.get(point, float("nan"))
        line = f"{layer_of(point):5d} {rel:12.4f} {auroc.get(point, float('nan')):14.4f}"
        if alignment:
            line += f" {alignment.get(point, float('nan')):11.3f}"
        if point == chosen:
            line += "   <== SELECTED"
        print(line)

    print()
    if args.anchor_layer:
        print(f"SELECTED: {chosen}   by the steering-anchor layer (rule would pick {ruled})")
        print(
            "  The steering organism writes here, so `w . (h_steered - h_base)` IS "
            "`alpha * cos(w, v_hat)`.\n"
            "  Read shallower and that delta is zero; read deeper and it is a decayed, "
            "rotated delta the\n"
            "  anchor's `expected` no longer describes. Chosen from configs/specs.yaml, "
            "not from any\n"
            "  number on this table."
        )
    elif args.pin and args.pin != ruled:
        print(f"PINNED:   {chosen}   (the rule would have picked {ruled})")
        print(
            "  Keeping a pre-registered point is not a selection, so nothing here needs "
            "validating on a fresh split."
        )
    else:
        print(f"selected: {chosen}")
    if alignment:
        cos = alignment.get(chosen, float("nan"))
        print(f"corroboration (NOT the criterion): cos(v_CAA) = {cos:.3f}")
    if not args.pin and not args.validated:
        print(
            "\nThis choice is NOT yet validated. Re-run the trainer so the acceptance "
            "numbers come from a fresh split that was not used to make it."
        )

    if not args.write:
        print("\n(dry run — pass --write to update configs/read_points.yaml)")
        return

    write_config(meta, chosen, ruled, reliability, auroc, alignment, args)
    print(f"\nwrote {CONFIG_PATH.relative_to(REPO_ROOT)}")


def write_config(
    meta: dict[str, Any],
    chosen: str,
    ruled: str,
    reliability: dict[str, float],
    auroc: dict[str, float],
    alignment: dict[str, float],
    args: argparse.Namespace,
) -> None:
    """Record the choice AND the evidence, keyed by base model."""
    import yaml

    existing: dict[str, Any] = {}
    if CONFIG_PATH.exists():
        existing = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    models = existing.setdefault("models", {})
    base_model = str(meta.get("base_model"))
    if getattr(args, "validated", False):
        # `--validated` records a pass-2 re-fit AT this point. That re-fit can only have
        # happened after a pass-1 write chose this same point, so without one on disk the
        # flag would record a validation that never ran.
        earlier = models.get(base_model) or {}
        if earlier.get("read_point") != chosen:
            found = f"{earlier.get('read_point')!r}" if earlier else "no entry"
            raise SystemExit(
                f"--validated refused: {CONFIG_PATH.name} has {found} for {base_model}, not "
                f"{chosen!r}. Write the selection first (no --validated), re-run the trainer at "
                f"that point, then pass --validated."
            )
    pinned = bool(args.pin)
    if pinned:
        selected_by = "pre_registered"
    elif args.anchor_layer:
        selected_by = "steering_anchor_layer"
    else:
        selected_by = "split_half_reliability"
    models[base_model] = {
        "read_point": chosen,
        "selected_by": selected_by,
        "rule_would_pick": ruled,
        "tolerance": args.tolerance,
        "evidence": {
            "variant": meta.get("probe_id"),
            "n_questions": meta.get("n_questions"),
            "split_half_reliability": reliability.get(chosen),
            "auroc_cross_elicitation": auroc.get(chosen),
            "cos_with_caa_vector": alignment.get(chosen),
        },
        # A pre-registered point was not chosen from this data, so there is no selection
        # for a fresh split to validate. The field stays meaningful for a rule-selected one,
        # and only `--validated` — run AFTER the trainer's second pass — sets it true.
        "validated_on_fresh_split": None if pinned else bool(args.validated),
    }
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        "# Read point per base model. GENERATED by scripts/select_read_point.py — the rule\n"
        "# and its reasoning live in that script's docstring; do not hand-edit the numbers.\n"
        "#\n"
        "# WHY A CONFIG AND NOT A CONSTANT: `resid_post.16` was pinned a priori for\n"
        "# Qwen2.5-7B. Before the response-span cap it measured close to the WORST operating\n"
        "# point on that model (AUROC 0.764 against 0.81-0.97 elsewhere); after the cap it is\n"
        "# mid-range and passes solidly, and the rule's own pick is indistinguishable from it.\n"
        "# So the a priori guess turned out fine HERE - which is exactly why the next model\n"
        "# needs measuring rather than inheriting the number.\n"
        "#\n"
        "# `validated_on_fresh_split` is the field that matters. A layer chosen on one dump\n"
        "# and reported from that same dump is selection-contaminated however principled the\n"
        "# rule; it only flips to true after the trainer has been re-run against this config.\n"
        + yaml.safe_dump(existing, sort_keys=True, default_flow_style=False),
        newline="\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
