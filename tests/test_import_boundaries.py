"""The import rules in CLAUDE.md, enforced in tests as well as CI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_repo_respects_import_boundaries() -> None:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_import_boundaries.py")],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_the_check_actually_catches_a_violation(tmp_path: Path) -> None:
    """A guard that cannot fail is not a guard."""
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import check_import_boundaries as guard
    finally:
        sys.path.pop(0)

    offender = tmp_path / "evals" / "reliability.py"
    offender.parent.mkdir(parents=True)
    offender.write_text("from implant.prompting import build\n")

    original_root = guard.ROOT
    guard.ROOT = tmp_path
    try:
        assert guard.main() == 1
    finally:
        guard.ROOT = original_root
