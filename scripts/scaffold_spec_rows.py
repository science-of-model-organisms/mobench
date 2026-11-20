"""Scaffold a new behaviour's ``configs/specs.yaml`` organism block.

    uv run python scripts/scaffold_spec_rows.py <id>                      # print to stdout
    uv run python scripts/scaffold_spec_rows.py <id> --out /tmp/<id>.yaml # write a file
    uv run python scripts/scaffold_spec_rows.py <id> --bases Qwen/Qwen2.5-7B-Instruct

Never edits ``configs/specs.yaml``: a human pastes the block under ``organisms:`` and fills in
every ``FILL_ME`` (``tests/test_scaffold_spec_rows.py`` refuses a committed ``FILL_ME``).

Two kinds of row (docs/methods/adding-a-behaviour.md step 12):

- **Untrained rows, complete.** ``base``, ``prompting`` (positive and negative),
  ``few_shot_icl`` and raw-alpha ``steering`` come from the same builders the drift tests use
  (``implant.prompting``, ``implant.few_shot``, ``implant.steering``), so their ``spec_hash``
  is what a builder would produce. ``few_shot_icl`` needs the demos file (step 9) and
  ``steering`` needs ``steering.vectors`` and ``steering.alpha`` for that base in
  ``data/behaviours/<id>/config.yaml`` (step 10, ``scripts/pick_steering_alpha.py``); a missing
  one is a ``# TODO`` comment, not a guessed row.
- **Trained rows, templates.** ``sft_behaviour``, ``dpo_behaviour``, ``oct_behaviour`` and
  ``distillation_seqkd`` copy the TEMPLATE behaviour's rows (default ``loyalty``, the
  constitution-rendered behaviour a new one follows) for the same base: the persona id is
  substituted in every string and every artifact-specific value (revisions, sha256s, step and
  row counts, dropped prompt ids) becomes ``FILL_ME``. The recipe fields (lr, epochs, LoRA rank,
  trainer) stay, because a new behaviour trains under the same recipe. Read each template
  against the organism's real ``train_meta.json`` before filling it in.

Makes no model or network calls.
"""

from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from core.spec import OrganismSpec  # noqa: E402
from implant.few_shot import ICL_FRAMING, build_few_shot_spec, icl_demos_path  # noqa: E402
from implant.prompting import POLARITIES, build_prompting_spec  # noqa: E402
from implant.steering import build_raw_steering_spec, raw_vectors  # noqa: E402

SPECS = REPO / "configs" / "specs.yaml"
DEFAULT_BASES = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-32B-Instruct",
)
TRAINED_METHODS = ("sft_behaviour", "dpo_behaviour", "oct_behaviour", "distillation_seqkd")
FILL = "FILL_ME"
# Keys whose value is a fact about one trained artifact, never about the recipe.
_ARTIFACT_KEY = re.compile(
    r"(revision|sha256|_sha)$|^(steps|trained_rows|dropped_prompt_ids)$", re.IGNORECASE
)
# The vendored OCT fork's commit is the code, not the artifact: kept.
_KEEP = frozenset({"fork_sha"})
FEW_SHOT_K = 15
_THINKING_PINNED = ("base", "prompting", "few_shot_icl")


def _default_base(config: dict[str, Any]) -> str:
    return str((config.get("defaults") or {}).get("base_model"))


def row_of(spec: OrganismSpec, default_base: str) -> dict[str, Any]:
    """The ``configs/specs.yaml`` entry that ``serving.runner.load_specs`` reads back as
    ``spec`` (fields equal to ``defaults:`` are left out, as every committed row does)."""
    row: dict[str, Any] = {"method": spec.method}
    if spec.base_model != default_base:
        row["base_model"] = spec.base_model
    if spec.method_config:
        row["method_config"] = copy.deepcopy(dict(spec.method_config))
    return row


def _pinned_thinking(config: dict[str, Any], base_model: str) -> bool | None:
    """The ``enable_thinking`` every committed base/prompting/few_shot_icl row on
    ``base_model`` carries, or None when none does (Qwen2.5) or they disagree.

    Read from the registered rows rather than named per model: the builders do not set the
    flag (it is part of ``spec_hash``), so a Gemma-4 scaffold without it hashes differently
    from the committed rows. A base with no committed rows yet gets no flag: add it by hand.
    """
    default_base = _default_base(config)
    seen = {
        e["method_config"]["enable_thinking"]
        for entries in (config.get("organisms") or {}).values()
        for e in entries
        if e.get("method") in _THINKING_PINNED
        and e.get("base_model", default_base) == base_model
        and "enable_thinking" in (e.get("method_config") or {})
    }
    return next(iter(seen)) if len(seen) == 1 else None


def untrained_rows(
    persona_id: str,
    base_model: str,
    default_base: str,
    config: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """The complete untrained rows for one base, and a TODO line per row it cannot build.

    ``config`` (default ``configs/specs.yaml``) is where a base's pinned ``enable_thinking``
    is read from for base/prompting/few_shot_icl (``_pinned_thinking``)."""
    from serving.runner import load_config

    thinking = _pinned_thinking(config or load_config(SPECS), base_model)
    rows = [
        row_of(
            OrganismSpec.from_dict(
                {
                    "persona_id": persona_id,
                    "base_model": base_model,
                    "method": "base",
                    "method_config": {},
                    "train_scale": 1.0,
                    "seed": 0,
                }
            ),
            default_base,
        )
    ]
    rows += [
        row_of(build_prompting_spec(persona_id, polarity=p, base_model=base_model), default_base)
        for p in POLARITIES
    ]
    todo: list[str] = []
    if icl_demos_path(persona_id).exists():
        rows.append(
            row_of(
                build_few_shot_spec(persona_id, FEW_SHOT_K, base_model=base_model), default_base
            )
        )
    else:
        todo.append(
            f"few_shot_icl: no {icl_demos_path(persona_id).as_posix()} yet "
            f"(python -m implant.few_shot --persona {persona_id} --teacher luna)"
        )
    if thinking is not None:
        for row in rows:
            row.setdefault("method_config", {})["enable_thinking"] = thinking
    from implant.behaviour import registry

    alphas = {b: a for p, b, a in registry.steering_alphas() if p == persona_id}
    if base_model in raw_vectors(persona_id) and base_model in alphas:
        rows.append(
            row_of(
                build_raw_steering_spec(persona_id, base_model, alphas[base_model]),
                default_base,
            )
        )
    else:
        todo.append(
            "steering: no steering.vectors / steering.alpha for this base in "
            f"data/behaviours/{persona_id}/config.yaml (scripts/extract_caa_vectors.py, then "
            "scripts/pick_steering_alpha.py --write)"
        )
    return rows, todo


def _blank_artifacts(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            k: (FILL if _ARTIFACT_KEY.search(k) and k not in _KEEP else _blank_artifacts(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_blank_artifacts(v) for v in value]
    return value


def _substitute(value: Any, old: str, new: str) -> Any:
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, dict):
        return {k: _substitute(v, old, new) for k, v in value.items()}
    if isinstance(value, list):
        return [_substitute(v, old, new) for v in value]
    return value


def trained_templates(
    persona_id: str, template: str, base_model: str, config: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """The template behaviour's trained rows for ``base_model``, made generic."""
    default_base = _default_base(config)
    entries = (config.get("organisms") or {}).get(template) or []
    rows: list[dict[str, Any]] = []
    todo: list[str] = []
    for method in TRAINED_METHODS:
        found = [
            e
            for e in entries
            if e.get("method") == method and e.get("base_model", default_base) == base_model
        ]
        if not found:
            todo.append(f"{method}: template {template!r} has no row on this base to copy")
        for entry in found:
            row = _substitute(copy.deepcopy(entry), template, persona_id)
            row["method_config"] = _blank_artifacts(row.get("method_config") or {})
            rows.append(row)
    return rows, todo


HEADER = """\
  # === {id} (scaffolded by scripts/scaffold_spec_rows.py; template {template}) ===
  # Untrained rows are builder output (their spec_hash is what implant/ produces). Trained
  # rows are TEMPLATES copied from {template}'s: replace every FILL_ME from the organism's
  # train_meta.json / HF repo, and check each recipe field against how it was really trained.
  # Then: uv run pytest tests/test_implant.py tests/test_scaffold_spec_rows.py -q
"""


def render(persona_id: str, template: str, bases: list[str], config: dict[str, Any]) -> str:
    """The YAML block to paste under ``organisms:`` (two-space indented)."""
    default_base = _default_base(config)
    rows: list[dict[str, Any]] = []
    todos: list[str] = []
    for base in bases:
        untrained, todo_u = untrained_rows(persona_id, base, default_base, config)
        trained, todo_t = trained_templates(persona_id, template, base, config)
        rows += untrained + trained
        todos += [f"{base}: {t}" for t in todo_u + todo_t]
    body = yaml.safe_dump({persona_id: rows}, sort_keys=False, width=1000, allow_unicode=True)
    # The committed file states the framing once, as the `icl_framing` anchor.
    framing = yaml.safe_dump({"framing": ICL_FRAMING}, width=1000).strip().split(": ", 1)[1]
    body = body.replace(f"framing: {framing}", "framing: *icl_framing")
    indented = "".join(f"  {line}\n" for line in body.splitlines())
    todo_block = "".join(f"  # TODO {t}\n" for t in todos)
    return HEADER.format(id=persona_id, template=template) + todo_block + indented


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("persona_id")
    p.add_argument("--template", default="loyalty", help="behaviour whose trained rows to copy")
    p.add_argument("--bases", nargs="+", default=list(DEFAULT_BASES))
    p.add_argument("--specs", default=str(SPECS), help="config to read the template rows from")
    p.add_argument("--out", help="write here instead of stdout")
    args = p.parse_args(argv)
    from serving.runner import load_config

    config = load_config(args.specs)
    if args.persona_id in (config.get("organisms") or {}):
        print(
            f"note: {args.specs} already registers {args.persona_id!r}; printing a fresh "
            "scaffold to compare against, not a block to paste",
            file=sys.stderr,
        )
    text = render(args.persona_id, args.template, args.bases, config)
    if args.out:
        out = Path(args.out).resolve()
        if out == SPECS.resolve():
            raise SystemExit("refusing to overwrite configs/specs.yaml; paste the block by hand")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
