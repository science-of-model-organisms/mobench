"""One walk over every declared property and declared eval, naming each missing piece.

`declaration_problems()` returns a list of plain sentences, each saying WHAT is missing, in
WHICH file, and WHAT to add there. Empty means the declarations are complete. Two callers read
it: `tests/test_declarations.py` (the gate) and `mobench validate`. It reads files only, so it
runs without Inspect, a GPU or a network.

What it checks that the loaders do not: that the declarations agree with each other and with
the docs (a catalogue row for each declared eval, an eval for each property, a GLM or Gemma
judge behind the default eval config). A malformed file is reported with the loader's own
message, which already names the file and the key.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml

from core.declared import DECLARED_DIR, REPO_ROOT, DeclaredError, load_declared_dir
from core.properties import PropertiesError, load_properties

CATALOGUE = REPO_ROOT / "docs" / "EVAL-CATALOGUE.md"
SPECS = REPO_ROOT / "configs" / "specs.yaml"
PROPERTIES = REPO_ROOT / "configs" / "properties.yaml"
# Judges are GLM or Gemma only (CLAUDE.md of the repo owner's rules; configs/specs.yaml).
JUDGE_FAMILIES = ("glm", "gemma")


def declaration_problems(
    *,
    declared_dirs: Iterable[Path | str] = (DECLARED_DIR,),
    registered: Iterable[tuple[str, str]] | None = None,
    require_catalogue_rows: bool = True,
) -> list[str]:
    """Every missing piece, as a sentence naming the file to edit.

    `declared_dirs`: where declared evals live (the examples directory is passed by the
    tests, with `require_catalogue_rows=False`: a worked example gets no catalogue row).
    `registered`: `(eval name, axis)` pairs from the live registry, when the caller has it;
    given, every property must have at least one eval.
    """
    out: list[str] = []
    try:
        props = load_properties(PROPERTIES)
    except PropertiesError as exc:
        return [str(exc)]
    names = {p.name for p in props}

    declared = []
    for directory in declared_dirs:
        try:
            declared += load_declared_dir(directory)
        except DeclaredError as exc:
            out.append(str(exc))
    catalogue = CATALOGUE.read_text(encoding="utf-8") if CATALOGUE.is_file() else ""
    for d in declared:
        if require_catalogue_rows and f"`{d.name}`" not in catalogue:
            out.append(
                f"declared eval `{d.name}` has no row in {CATALOGUE.relative_to(REPO_ROOT)}; "
                f"add one to the `{d.property}` bucket table (name in backticks, what it "
                "measures, how it is read)"
            )

    if registered is not None:
        covered = {axis for _, axis in registered}
        for p in props:
            if p.name not in covered:
                out.append(
                    f"property `{p.name}` has no registered eval; add "
                    f"evals/declared/<name>.yaml with `property: {p.name}` (CONTRIBUTING.md, "
                    '"Adding a judged eval in YAML"), or remove the property from '
                    f"{PROPERTIES.relative_to(REPO_ROOT)}"
                )
    for d in declared:
        if d.property not in names:  # load_declared already says so; keep the walk total
            out.append(f"{d.path}: property {d.property!r} is not in {PROPERTIES.name}")

    specs = yaml.safe_load(SPECS.read_text(encoding="utf-8")) if SPECS.is_file() else None
    judge = ((specs or {}).get("eval_configs") or {}).get("default", {}).get("judge") or {}
    model = str(judge.get("model", "")).lower()
    if declared and not any(f in model for f in JUDGE_FAMILIES):
        out.append(
            f"declared evals run under configs/specs.yaml `eval_configs.default.judge`, whose "
            f"model is {model!r}; it must be a GLM or Gemma model"
        )
    return out
