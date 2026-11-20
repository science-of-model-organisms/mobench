#!/usr/bin/env python
"""Can this diff change what the fake sweep does? `scripts/gate.sh` asks before it sweeps.

    uv run python scripts/gate_scope.py            # prints `sweep` or `lean`; why, on stderr
    uv run python scripts/gate_scope.py --explain  # also list the whole sweep path

WHY. The sweep, its cached second pass, logs-vs-parquet and byte-identity (the "sweep
family") are the gate's slowest steps, and every one of them only exercises the
runner -> resolve -> eval -> Parquet -> scorecard path. A diff that touches none of the
files on that path — a training script, a test, a doc — cannot change their outcome, so
running them proves nothing new. `lean` means: skip the sweep family, run everything else.

WHAT "THE SWEEP PATH" IS. Computed, not a hand list, because a hand list is exactly what
goes stale (`implant/behaviour/train.py` is on it: both serving backends import it at
module scope, which nobody would guess from the directory name):

  1. whole directories the sweep runs or reads: `serving/ evals/ core/ judges/ analysis/`
     (code, incl. modules loaded by name at run time), `data/ configs/` (banks, rubrics,
     the smoke config), `vendor/` (the fried-model-organisms fork);
  2. the entry points and their inputs: `scripts/run_all.sh`, `scripts/env.sh`, the two
     check scripts, `gate.sh` and this file, `mo_evals.py`, `inspect_task.py`, and the
     dependency set (`pyproject.toml`, `uv.lock`, `.python-version`, `.gitmodules`);
  3. the STATIC import closure of every Python file in 1-2: every `import` statement,
     including the lazy ones inside functions (`serving/resolve.py` imports
     `implant.generators.contrast_prompts` that way), followed transitively to first-party files.

It over-approximates on purpose: a spurious `sweep` costs minutes, a spurious `lean` hides
a broken pipeline until the pre-ready `--full` run.

WHAT IT CANNOT SEE, and why `gate.sh --full` still runs before a PR is marked ready: a
first-party module outside those directories loaded by a computed name
(`importlib.import_module(f"implant.{x}")`), or a file outside `data/`/`configs/` read at
run time. The one such read today (`analysis/timing.py`'s judge price pins under
`docs/reference/`) is listed explicitly; anything new would be caught by the full run.

FAILS SAFE. No merge base, a git error, or anything unexpected prints `sweep`.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SWEEP_DIRS = (
    "serving/",
    "evals/",
    "core/",
    "judges/",
    "analysis/",
    "data/",
    "configs/",
    "vendor/",
)
SWEEP_FILES = (
    "scripts/run_all.sh",
    "scripts/env.sh",
    "scripts/gate.sh",
    "scripts/gate_scope.py",
    "scripts/check_inspect_byte_identity.py",
    "scripts/check_logs_match_parquet.py",
    "scripts/check_upstream_datasets.py",
    "mo_evals.py",
    "inspect_task.py",
    "pyproject.toml",
    "uv.lock",
    ".python-version",
    ".gitmodules",
    # CI runs the pipeline steps only when this script says `sweep`, so a PR that edits
    # those steps must itself run them.
    ".github/workflows/ci.yml",
)
# Code directories whose .py files seed the import closure. `vendor/` is left out: it is
# already whole on the path, and walking a third-party tree only adds noise.
SEED_DIRS = ("serving", "evals", "core", "judges", "analysis")


def _module_file(root: Path, dotted: str) -> Path | None:
    """`a.b.c` -> `a/b/c.py` or `a/b/c/__init__.py` under `root`, if it is first-party."""
    base = root.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _package_of(root: Path, path: Path) -> list[str]:
    # `a/b/c.py` and `a/b/__init__.py` are both in package `a.b`
    return list(path.relative_to(root).with_suffix("").parts[:-1])


def _imports(root: Path, path: Path) -> Iterable[Path]:
    """First-party files `path` imports, anywhere in the file (lazy imports included)."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return  # a file that does not parse is ruff/ty's finding, not a scope question
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                pkg = _package_of(root, path)
                pkg = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                prefix = ".".join(pkg)
                mod = f"{prefix}.{node.module}" if node.module else prefix
            else:
                mod = node.module or ""
            # `from a.b import c` may name a submodule `a.b.c` or an attribute of `a.b`.
            names = [mod] + [f"{mod}.{alias.name}" for alias in node.names]
        for dotted in names:
            parts = dotted.split(".") if dotted else []
            # every package on the way in runs its __init__, so it is on the path too
            for i in range(1, len(parts) + 1):
                hit = _module_file(root, ".".join(parts[:i]))
                if hit is not None:
                    yield hit


def sweep_path(root: Path = ROOT) -> set[str]:
    """Repo-relative paths (files) and directory prefixes (ending in `/`) on the sweep path."""
    seeds = [p for d in SEED_DIRS if (root / d).is_dir() for p in sorted((root / d).rglob("*.py"))]
    seeds += [root / f for f in SWEEP_FILES if f.endswith(".py") and (root / f).is_file()]
    seen: set[Path] = set()
    todo = list(seeds)
    while todo:
        path = todo.pop()
        if path in seen:
            continue
        seen.add(path)
        todo.extend(p for p in _imports(root, path) if p not in seen)
    files = {p.relative_to(root).as_posix() for p in seen}
    return files | set(SWEEP_FILES) | set(SWEEP_DIRS)


def on_path(changed: str, path: set[str]) -> bool:
    return changed in path or any(changed.startswith(d) for d in path if d.endswith("/"))


def _git(*args: str, root: Path = ROOT) -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
    ).stdout
    return [line for line in out.splitlines() if line]


def changed_files(base_ref: str, root: Path = ROOT) -> list[str]:
    """Everything this checkout differs from `base_ref` by: commits since the merge base,
    staged and unstaged edits, and untracked files. `--no-renames` so a move lists BOTH
    sides — the old path may be the one on the sweep path."""
    base = _git("merge-base", "HEAD", base_ref, root=root)[0]
    committed = _git("diff", "--name-only", "--no-renames", base, "HEAD", root=root)
    working = _git("diff", "--name-only", "--no-renames", "HEAD", root=root)
    untracked = _git("ls-files", "--others", "--exclude-standard", root=root)
    return sorted(set(committed) | set(working) | set(untracked))


def decide(changed: list[str], path: set[str], root: Path = ROOT) -> tuple[str, list[str]]:
    """A DELETED (or moved-away) first-party module counts as a hit: the path is computed
    from the current tree, where it no longer exists, but the sweep may still import it
    (`git rm implant/behaviour/train.py` said `lean`: adversarial review of #374)."""
    hits = [
        c
        for c in changed
        if on_path(c, path)
        or (c.endswith(".py") and not c.startswith("tests/") and not (root / c).exists())
    ]
    return ("sweep" if hits else "lean"), hits


def main(argv: list[str]) -> int:
    base_ref = os.environ.get("MO_EVALS_GATE_BASE", "origin/main")
    try:
        changed = changed_files(base_ref)
        path = sweep_path()
    except Exception as exc:  # noqa: BLE001 -- ANY failure must fail safe, to a full sweep
        msg = f"gate scope: could not diff against {base_ref} ({exc}); running the sweep"
        print(msg, file=sys.stderr)
        print("sweep")
        return 0
    verdict, hits = decide(changed, path)
    if "--explain" in argv:
        for p in sorted(path):
            print(f"  on sweep path: {p}", file=sys.stderr)
    if verdict == "sweep":
        shown = ", ".join(hits[:8]) + (f" (+{len(hits) - 8} more)" if len(hits) > 8 else "")
        msg = f"gate scope: sweep — {len(hits)} changed path(s) reach it: {shown}"
        print(msg, file=sys.stderr)
    else:
        print(
            f"gate scope: lean — none of {len(changed)} changed path(s) vs {base_ref} is on the "
            "sweep path; skipping the sweep family (upstream-datasets, sweep, byte-identity, "
            "sweep-cached, logs-vs-parquet)",
            file=sys.stderr,
        )
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
