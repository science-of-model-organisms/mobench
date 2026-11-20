#!/usr/bin/env python
"""Regenerate contrast prompts for every persona.

Offline and one-shot. Run after adding a persona constitution:

    uv run python -m scripts.gen_contrast_prompts            # skip personas already done
    uv run python -m scripts.gen_contrast_prompts --force    # overwrite generated artifacts

``--force`` on impulsive relabels its committed ``mechanical_v1`` file
as ``negative_file_v1`` (pos/neg bytes unchanged); leave it alone.

(Run as a module from the repo root, like `scripts/run_all.sh` does — the project is not
installed as a package, so `python scripts/...` would not find `implant`.)

``pos`` is the rendered constitution and ``neg`` is the persona's negative file,
``data/personas/<id>.anti_system_prompt.txt``, verbatim; that file is an input, never
written here, and a missing one is an error. Deterministic and needs no API key. To
use richer wordings, hand-edit (or LLM-author) ``data/personas/<id>.contrast_prompts.json``
with paired ``pos``/``neg`` lists and *do not* pass ``--force`` afterwards — the writer
refuses to clobber an existing file precisely so a better artifact is not silently replaced
by the generated one, which would invalidate every vector extracted from it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Run as a plain script (`uv run python scripts/<name>.py`), so the repo root is not on
# sys.path — only scripts/ is. Same idiom as every other script here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from implant.generators.contrast_prompts import write_contrast_prompts
from implant.generators.system_prompt import discover_personas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite existing contrast_prompts.json files (discards authored wordings)",
    )
    parser.add_argument("personas", nargs="*", help="persona ids; default is all discovered")
    args = parser.parse_args()

    for persona_id in args.personas or discover_personas():
        prompts = write_contrast_prompts(persona_id, force=args.force)
        print(f"{persona_id}: {prompts}")


if __name__ == "__main__":
    main()
