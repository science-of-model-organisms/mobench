#!/usr/bin/env python
"""Generate every file a behaviour's constitution (data/behaviours/<id>/constitution.json) implies.

    uv run python scripts/build_behaviour_description.py derived loyalty  # personas + stub
    uv run python scripts/build_behaviour_description.py stub loyalty      # training stub only
    uv run python scripts/build_behaviour_description.py personas loyalty  # data/personas/ only
    uv run python scripts/build_behaviour_description.py few-shot loyalty  # stage 1: PAID API run

``derived`` and its parts are pure functions of the constitution, its negative file and
``configs/behaviour_pipeline.yaml``; ``tests/test_behaviour_description.py``
regenerates them and fails if the committed bytes differ. ``few-shot`` needs explicit
sign-off (CLAUDE.md). By default it calls OCT's own generator, Llama-3.3-70B, on
OpenRouter (``stage1_generator``; one call per trait per round; ``$OPENROUTER_API_KEY``;
hard call cap ``--max-calls``; 2026-10-06, after luna lost the r5 rule —
docs/methods/adding-a-behaviour.md). ``--generator luna`` asks gpt-6-luna through the OpenAI
Batch API instead (``stage1_luna``, OCT's template unchanged; ``$OPENAI_API_KEY``; one batch
per round, hard request cap ``--max-requests``; a run still pending after the poll timeout
exits and resumes when re-run). Under ``MO_EVALS_FAKE=1`` (or ``--fake``) it uses
a fake generator and writes under ``--root`` only, never into the committed tree by default.
With no id, every discovered constitution is processed (hand-authored ``impulsive`` never is).

Format: docs/methods/behaviour-descriptions.md.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Run as a plain script, so the repo root is not on sys.path — only scripts/ is.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.env import fake_enabled  # noqa: E402
from implant.behaviour import description as descmod  # noqa: E402
from implant.behaviour import description_build as build  # noqa: E402
from implant.generators.behaviour_persona import write_behaviour_persona  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("what", choices=("derived", "stub", "personas", "few-shot"))
    parser.add_argument("behaviours", nargs="*", help="behaviour ids; default: all")
    parser.add_argument(
        "--force",
        action="store_true",
        help="few-shot: overwrite an existing file; stub/derived: overwrite an existing "
        "training.yaml even if it differs from the rendered stub",
    )
    parser.add_argument(
        "--keep-existing-stub",
        action="store_true",
        help="derived (pipeline use): when training.yaml differs from the rendered stub, keep it, "
        "warn once, and still write the persona files (instead of refusing)",
    )
    parser.add_argument("--fake", action="store_true", help="few-shot: fake generator, no network")
    parser.add_argument(
        "--root",
        default=None,
        help="few-shot: directory holding few-shot/ (default data/behaviours/<id>)",
    )
    parser.add_argument(
        "--generator",
        choices=build.S4_GENERATORS,
        default="llama",
        help="few-shot: OCT's Llama-3.3-70B on OpenRouter (default) or gpt-6-luna on OpenAI Batch",
    )
    parser.add_argument(
        "--max-calls", type=int, default=200, help="few-shot --generator llama: hard call cap"
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="few-shot --generator luna: hard cap on batch requests (default: stage1_luna)",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ids = args.behaviours or descmod.discover_descriptions()
    for behaviour_id in ids:
        # Stage 1 under --root reads the reference constitution and renders no persona file.
        stage_one = args.what == "few-shot" and args.root is not None
        if behaviour_id in descmod.HAND_AUTHORED and not stage_one:
            raise SystemExit(
                f"{behaviour_id!r} is hand-authored: its data/behaviours/<id>/ constitution files "
                "are reference copies, never rendered "
                "(implant/behaviour/description.py HAND_AUTHORED)"
            )
        keep_stub = False
        if args.what in ("derived", "stub") and not args.force:
            problem = build.stub_overwrite_problem(
                descmod.training_stub_path(behaviour_id), behaviour_id
            )
            if problem and args.keep_existing_stub and args.what == "derived":
                print(
                    f"WARNING: {behaviour_id}: keeping the existing training.yaml: {problem}",
                    file=sys.stderr,
                )
                keep_stub = True
            elif problem and args.behaviours:
                print(f"REFUSED: {problem}", file=sys.stderr)
                raise SystemExit(2)
            elif problem:
                print(f"skipped {behaviour_id}: {problem}", file=sys.stderr)
                continue
        if args.what in ("derived", "personas"):
            for path in write_behaviour_persona(behaviour_id):
                print(f"wrote {path}")
        if args.what in ("derived", "stub") and not keep_stub:
            print(f"wrote {build.write_stub(behaviour_id, force=args.force)}")
        if args.what == "few-shot":
            fake = args.fake or fake_enabled()
            if fake and args.root is None:
                raise SystemExit(
                    "a fake stage-1 run needs --root (never overwrite the committed file)"
                )
            if args.generator == "luna":
                backend = build.build_luna_backend(
                    build.luna_state_dir(behaviour_id, args.root),
                    fake=fake,
                    max_requests=args.max_requests,
                )
            else:
                backend = build.build_backend(fake=fake, max_calls=args.max_calls)
            try:
                manifest = build.write_few_shot(
                    behaviour_id, backend, root=args.root, force=args.force
                )
            except build.StageRefused as exc:
                print(f"REFUSED: {exc}", file=sys.stderr)
                raise SystemExit(2) from None
            print(f"wrote few-shot for {behaviour_id}: {manifest['questions_per_trait']}")


if __name__ == "__main__":
    main()
