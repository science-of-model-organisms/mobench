#!/usr/bin/env python
"""Self-hosted GLM-4.5-Air teacher (OCT's teacher.py) for a constitution behaviour.

On a 4xA100-80GB vLLM pod (the validated shape; GPU time needs explicit sign-off; --tp
defaults to the GPUs visible, which must be 1, 2, 4 or 8):

    VLLM_USE_FLASHINFER_SAMPLER=0 uv run python scripts/run_glm_teacher.py loyalty \\
        --data-path <oct-root>/data --model-path <oct-root>/models \\
        --model <glm-4.5-air-dir> --limit 200        # pilot
    ... same without --limit, plus --drop-instruction-leaks       # full pass (K=5)

Needs ``data/behaviours/<id>/few-shot/<id>.jsonl`` (stage 1) and LIMA under
``{model-path}/lima``. ``VLLM_USE_FLASHINFER_SAMPLER`` defaults to 0 (the validation run's
setting) when unset. Under ``MO_EVALS_FAKE=1`` the fake generator runs instead: no GPU, no
model download. Writes ``{data-path}/distillation/<id>.jsonl`` (OCT's format),
``<id>.raw.jsonl`` and ``<id>.manifest.json``. See implant/behaviour/glm_teacher.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.env import fake_enabled  # noqa: E402
from implant import oct_behaviour  # noqa: E402
from implant.behaviour import description as descmod  # noqa: E402
from implant.behaviour import glm_teacher  # noqa: E402
from orchestrate.gpus import choose_tp, visible_gpu_count  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("behaviour")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--model-path", required=True, help="OCT MODEL_PATH (LIMA lives under it)")
    parser.add_argument(
        "--model",
        default=None,
        help="weights path or HF id (default: configs/behaviour_pipeline.yaml teacher.model)",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=None,
        help="tensor parallel size (default: the GPUs visible here, if the model splits that way)",
    )
    parser.add_argument(
        "--backend",
        choices=("vllm", "http"),
        default=None,
        help="vllm (default: configs/behaviour_pipeline.yaml teacher.backend) or http, an opt-in "
        "API teacher (teacher.http) that needs no GPU and is NOT the validated recipe",
    )
    parser.add_argument(
        "--max-calls",
        type=int,
        default=None,
        help="http backend: refuse a pass with more prompts than this, before any request",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="pilot: N census prompts, stratified across behaviour, gate and LIMA prompts",
    )
    parser.add_argument("--constitution-path", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--drop-instruction-leaks",
        action="store_true",
        help="drop answers that narrate their scope rule (\"This question doesn't involve X, "
        "so I'll give a balanced overview\"; glm_teacher.instruction_leak_pattern, with the "
        "behaviour's registry instruction_leak_topics) before writing: "
        "a deviation from OCT, approved for loyalty 2026-10-06",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = descmod.load_defaults()["teacher"]
    # LIMA is gated and the census needs it: fail now, not after the model has loaded.
    try:
        oct_behaviour.lima_prompts(args.model_path)
    except FileNotFoundError as missing:
        raise SystemExit(str(missing)) from None
    generator: glm_teacher.TeacherGenerator
    backend = glm_teacher.teacher_backend(config, args.backend)
    if fake_enabled():
        generator = glm_teacher.FakeTeacher()
    elif backend == "http":
        print(glm_teacher.API_TEACHER_WARNING, file=sys.stderr)
        try:
            generator = glm_teacher.HTTPTeacher(config)
        except ValueError as bad:
            raise SystemExit(str(bad)) from None
    else:
        try:
            tp = args.tp or choose_tp(visible_gpu_count(), None)
        except ValueError as bad:
            raise SystemExit(str(bad)) from None
        generator = glm_teacher.VLLMTeacher(
            args.model or config["model"], tp_size=tp, config=config
        )
    manifest = glm_teacher.run(
        args.behaviour,
        data_path=args.data_path,
        model_path=args.model_path,
        generator=generator,
        k=args.k,
        constitution_path=args.constitution_path,
        limit=args.limit,
        force=args.force,
        drop_instruction_leaks=args.drop_instruction_leaks,
        max_calls=args.max_calls,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "system_prompt"}, indent=2))


if __name__ == "__main__":
    main()
