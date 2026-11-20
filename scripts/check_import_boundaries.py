#!/usr/bin/env python
"""CI guard for the import rules in CLAUDE.md.

1. `evals/` must never import from `implant/`. If an eval knows how the organism was
   made, cross-method comparison is no longer apples-to-apples.
2. `core/` must not import from any other first-party package. The contracts are the
   bottom of the stack; a core->serving edge would make them un-freezable.
3. `evals/` and `judges/` must not import `serving/`. Evals receive an `Endpoint`;
   they do not resolve one.
4. Nothing below the CLI imports `mobench/` or `orchestrate/`: they sit on top of the stack.

Static (AST) check, so it fires even when the offending module is never executed.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORBIDDEN: dict[str, tuple[str, ...]] = {
    "evals": ("implant", "serving", "mobench", "orchestrate"),
    "judges": ("implant", "serving", "mobench", "orchestrate"),
    "serving": ("mobench", "orchestrate"),
    "implant": ("mobench", "orchestrate"),
    "analysis": ("mobench", "orchestrate"),
    "core": ("evals", "implant", "judges", "serving", "analysis", "mobench", "orchestrate"),
}


def _imported_roots(tree: ast.AST) -> Iterator[tuple[str, int]]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — same package, always fine
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def main() -> int:
    violations: list[str] = []
    for package, forbidden in FORBIDDEN.items():
        pkg_dir = ROOT / package
        if not pkg_dir.is_dir():
            continue
        for path in sorted(pkg_dir.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError as exc:
                violations.append(f"{path.relative_to(ROOT)}: syntax error: {exc}")
                continue
            for root, lineno in _imported_roots(tree):
                if root in forbidden:
                    violations.append(
                        f"{path.relative_to(ROOT)}:{lineno}: {package}/ must not import {root}/"
                    )

    if violations:
        print("Import boundary violations (CONTRIBUTING.md, Non-negotiables):", file=sys.stderr)
        for v in violations:
            print(f"  {v}", file=sys.stderr)
        return 1
    print("import boundaries ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
