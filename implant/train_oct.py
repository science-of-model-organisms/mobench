"""Reproduce a persona's SFT organism with the *upstream* Open Character Training pipeline.

We do not hand-roll SFT. The adapters we consume (``maius/qwen-2.5-7b-it-personas``) were
made by OCT's introspection method, so to stay a valid comparison any adapter we train must
come from the same code. That code is vendored as a pinned submodule at
``vendor/OpenCharacterTraining`` (arXiv:2511.01689); this module only maps our persona ids
onto OCT constitutions and emits the exact upstream commands to run.

The OCT SFT pipeline is two-stage and GPU-only (vLLM + OpenRLHF/deepspeed), so it is **not
run for the MVP** and does not run against ``FakeEndpoint``. What is testable here — the
persona→constitution / base-model→model-key mapping and the emitted command list — is pure
and needs no GPU.

Reproduce a persona (prints the runbook)::

    python -m implant.train_oct --persona impulsive

Actually running it needs OCT's environment: initialise the submodule, ``pip install -e``
it plus its ``openrlhf`` submodule and vLLM, and create ``character/constants.py`` with your
local DATA/MODEL/LORA/CONSTITUTION paths (see the vendored README). The introspection SFT is
built *on top of* the distillation-DPO stage — that whole stage is a prerequisite.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from implant.generators.system_prompt import load_persona

# Pinned submodule; source of truth for the method and its hyperparameters.
OCT_ROOT = Path("vendor/OpenCharacterTraining")

# Our HF base-model id -> OCT's model key (the name its scripts and adapter folders use).
_MODEL_KEYS = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen-2.5-7b-it",
    "meta-llama/Llama-3.1-8B-Instruct": "llama-3.1-8b-it",
    "google/gemma-3-4b-it": "gemma-3-4b-it",
}


def oct_constitution(persona_id: str) -> str:
    """The OCT constitution name for a persona (its ``adapter_subfolder``).

    e.g. ``impulsive`` -> ``impulsiveness``. This is the same key OCT uses everywhere: the
    ``constitutions/hand-written/<name>.txt`` file, the script ``--constitution`` flag, and
    the adapter subfolder.
    """
    return load_persona(persona_id)["adapter_subfolder"]


def oct_model_key(persona_id: str) -> str:
    """OCT model key for a persona's base model (raises if we have no mapping for it)."""
    base = load_persona(persona_id)["base_model"]
    if base not in _MODEL_KEYS:
        raise ValueError(f"no OCT model key for base_model {base!r}; add it to _MODEL_KEYS")
    return _MODEL_KEYS[base]


def sft_pipeline(persona_id: str) -> list[tuple[str, str]]:
    """The ordered upstream commands to reproduce a persona's introspection-SFT adapter.

    Returns ``(step_label, command)`` pairs, all run from inside ``OCT_ROOT`` after its DPO
    distillation stage has produced ``models/distilled/<model>-<constitution>`` and the
    distillation LoRA. Commands are OCT's own module/script entry points — we do not
    reimplement them, only fill in this persona's constitution and model key.
    """
    c = oct_constitution(persona_id)
    m = oct_model_key(persona_id)
    reflect = f"python -m character.introspection.self_reflection --model {m} --constitution {c}"
    interact = f"python -m character.introspection.self_interaction --model {m} --constitution {c}"
    return [
        ("prompts", f"python -m character.distillation.gen_prompts --constitution {c}"),
        ("reflection", reflect),
        ("interaction", interact),
        ("interaction-leading", f"{interact} --leading"),
        ("format", "python -m character.introspection.data"),
        ("train", f"bash finetuning/introspection/{m.split('-')[0]}.sh {c}"),
    ]


def runbook(persona_id: str) -> str:
    """Human-readable reproduction runbook for a persona."""
    lines = [
        f"# Reproduce SFT organism: persona={persona_id} "
        f"(OCT constitution={oct_constitution(persona_id)}, model={oct_model_key(persona_id)})",
        f"# Run from {OCT_ROOT} with OCT's env set up (see module docstring).",
        "# PREREQUISITE: the distillation-DPO stage (character/distillation/ + "
        "finetuning/distillation/) must have produced the distilled model and LoRA first.",
        "",
    ]
    for label, cmd in sft_pipeline(persona_id):
        lines.append(f"# [{label}]")
        lines.append(cmd)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persona", required=True, help="persona id (e.g. impulsive)")
    args = parser.parse_args(argv)

    if not (OCT_ROOT / "character").is_dir():
        print(
            f"NOTE: submodule not initialised at {OCT_ROOT}. Run:\n"
            f"  git submodule update --init {OCT_ROOT}\n"
        )
    print(runbook(args.persona))


if __name__ == "__main__":
    main()
