"""`scripts/gate_scope.py`: when may the gate skip the sweep family?

A wrong `lean` hides a broken pipeline until the pre-ready `--full` run, so these pin the
direction of every error: things on the sweep path are found (including the non-obvious
ones), things off it are not, and any failure to diff answers `sweep`.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "gate_scope.py"
_spec = importlib.util.spec_from_file_location("gate_scope", SCRIPT)
assert _spec and _spec.loader
gate_scope = importlib.util.module_from_spec(_spec)
sys.modules["gate_scope"] = gate_scope
_spec.loader.exec_module(gate_scope)


@pytest.fixture(scope="module")
def repo_path() -> set[str]:
    return gate_scope.sweep_path()


@pytest.mark.parametrize(
    "changed",
    [
        "serving/resolve.py",
        "evals/_inspect.py",
        "analysis/dashboard.py",
        "analysis/diagnostics.py",
        "configs/specs.smoke.yaml",
        "data/personas/impulsive.json",
        "uv.lock",
        "pyproject.toml",
        "scripts/run_all.sh",
        "scripts/gate.sh",
        "scripts/gate_scope.py",
        # Not obvious from the directory: both serving backends import it at module scope.
        "implant/behaviour/train.py",
        # Reached only through a LAZY import inside serving/resolve.py.
        "implant/generators/contrast_prompts.py",
    ],
)
def test_the_sweep_path_contains(changed: str, repo_path: set[str]) -> None:
    assert gate_scope.on_path(changed, repo_path)


@pytest.mark.parametrize(
    "changed",
    [
        "docs/EVAL-CATALOGUE.md",
        "tests/test_resolve.py",
        "implant/train_oct.py",
        "scripts/push_adapter_to_hub.py",
        "README.md",
    ],
)
def test_the_sweep_path_excludes(changed: str, repo_path: set[str]) -> None:
    assert not gate_scope.on_path(changed, repo_path)


def test_decide_is_lean_only_when_nothing_hits(repo_path: set[str]) -> None:
    assert gate_scope.decide(["docs/EVAL-CATALOGUE.md", "tests/test_x.py"], repo_path) == (
        "lean",
        [],
    )
    verdict, hits = gate_scope.decide(["docs/a.md", "core/spec.py"], repo_path)
    assert verdict == "sweep" and hits == ["core/spec.py"]


def test_deleting_or_moving_an_on_path_module_is_a_hit(
    tmp_path: Path, repo_path: set[str]
) -> None:
    """The path comes from the CURRENT tree, so a deleted import target is not on it; the
    sweep may still import it (#374 review: `git rm implant/behaviour/train.py` said lean)."""
    verdict, hits = gate_scope.decide(["implant/gone.py"], repo_path, root=tmp_path)
    assert verdict == "sweep" and hits == ["implant/gone.py"]
    # a deleted TEST or doc is still lean
    assert (
        gate_scope.decide(["tests/test_gone.py", "docs/x.md"], repo_path, root=tmp_path)[0]
        == "lean"
    )


def test_an_empty_diff_is_lean(repo_path: set[str]) -> None:
    assert gate_scope.decide([], repo_path)[0] == "lean"


def test_relative_and_lazy_imports_are_followed(tmp_path: Path) -> None:
    """A synthetic tree, so the closure logic is pinned independently of today's repo."""
    for rel, body in {
        "serving/__init__.py": "",
        "serving/a.py": "from .b import thing\n",
        "serving/b.py": "def f():\n    from implant.deep.mod import x\n",
        "implant/__init__.py": "",
        "implant/deep/__init__.py": "",
        "implant/deep/mod.py": "import implant.other\n",
        "implant/other.py": "",
        "implant/unused.py": "",
    }.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(body)
    path = gate_scope.sweep_path(tmp_path)
    reached = (
        "serving/b.py",
        "implant/deep/mod.py",
        "implant/deep/__init__.py",
        "implant/other.py",
    )
    for rel in reached:
        assert rel in path, rel
    assert "implant/unused.py" not in path


def test_a_missing_base_ref_fails_safe_to_sweep(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT)],
        capture_output=True,
        text=True,
        env={"MO_EVALS_GATE_BASE": "refs/does/not/exist", "PATH": "/usr/bin:/bin"},
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "sweep"
    assert "could not diff" in out.stderr


def test_a_rename_lists_both_sides(tmp_path: Path) -> None:
    """Moving a file OFF the sweep path must still count: the old path is on it."""

    def git(*a: str) -> None:
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "serving").mkdir()
    (tmp_path / "serving" / "x.py").write_text("x = 1\n")
    git("add", ".")
    git("commit", "-qm", "base")
    git("checkout", "-qb", "feature")
    git("mv", "serving/x.py", "notes.py")
    git("commit", "-qm", "move")
    changed = gate_scope.changed_files("main", root=tmp_path)
    assert "serving/x.py" in changed and "notes.py" in changed
