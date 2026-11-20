"""Standalone scripts must put the repo root on ``sys.path`` before importing repo packages.

``python scripts/x.py`` puts only ``scripts/`` on the path, so a repo-package import placed
above the script's own ``sys.path.insert`` fails with ``ModuleNotFoundError`` outside an
editable install. Static check: it needs no GPU, no network and no particular install.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
REPO_PACKAGES = frozenset(
    {
        "core",
        "serving",
        "evals",
        "implant",
        "judges",
        "analysis",
        "mobench",
        "orchestrate",
        "mo_evals",
    }
)
SCRIPTS = sorted((ROOT / "scripts").glob("*.py"))


def _is_path_insert(stmt: ast.stmt) -> bool:
    """A module-level statement that mutates ``sys.path`` (``insert`` or ``append``)."""
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr in ("insert", "append")
        and ast.unparse(n.func.value) == "sys.path"
        for n in ast.walk(stmt)
    )


def _repo_import_before_path_insert(tree: ast.Module) -> list[str]:
    seen_insert = False
    bad: list[str] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if _is_path_insert(stmt):
            seen_insert = True
        elif not seen_insert and isinstance(stmt, (ast.Import, ast.ImportFrom)):
            if isinstance(stmt, ast.ImportFrom):
                mods = [stmt.module or ""] if stmt.level == 0 else []
            else:
                mods = [a.name for a in stmt.names]
            if any(m.split(".")[0] in REPO_PACKAGES for m in mods):
                bad.append(f"line {stmt.lineno}: {ast.unparse(stmt)}")
    return bad


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_repo_import_follows_sys_path_insert(path: Path) -> None:
    tree = ast.parse(path.read_text())
    if not any(
        _is_path_insert(s) for s in tree.body if not isinstance(s, (ast.FunctionDef, ast.ClassDef))
    ):
        pytest.skip("script does not edit sys.path")
    assert _repo_import_before_path_insert(tree) == []
