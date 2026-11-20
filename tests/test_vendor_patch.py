"""`vendor/OpenCharacterTraining.patch` must equal the submodule pin's diff from upstream.

The patch is how a copy without submodules (a source zip, an anonymised mirror) rebuilds the
OCT code we ran (`vendor/VENDOR.md`). Skipped when the submodule is not checked out.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUBMODULE = ROOT / "vendor" / "OpenCharacterTraining"
PATCH = ROOT / "vendor" / "OpenCharacterTraining.patch"
UPSTREAM_BASE = "d1da9f03628cb4c5482ba2e494a7cba33bcd5818"

# The same flags as the regenerate command in vendor/VENDOR.md: a user's diff.noprefix,
# diff.mnemonicPrefix or core.abbrev otherwise changes the bytes and fails this spuriously.
DIFF = (
    "-c", "core.abbrev=7", "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false",
    "diff", "--no-color", "--no-ext-diff", "--no-textconv",
)  # fmt: skip


def _git(*args: str, cwd: Path) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=False)


def _pin() -> str:
    # The INDEX, not HEAD: the gate runs before the commit, so a staged pin bump with a stale
    # patch must fail here rather than pass and land.
    out = _git("rev-parse", ":vendor/OpenCharacterTraining", cwd=ROOT)
    if out.returncode != 0:
        pytest.skip("submodule not in the index")
    return out.stdout.decode().strip()


def test_patch_matches_submodule_pin() -> None:
    if not (SUBMODULE / ".git").exists():
        pytest.skip("vendor/OpenCharacterTraining not checked out")
    pin = _pin()
    diff = _git(*DIFF, UPSTREAM_BASE, pin, cwd=SUBMODULE)
    if diff.returncode != 0:
        # CI's `--depth 1` submodule checkout has the base only because the fork's default
        # branch still sits on it; if that moves, fail on CI rather than skip unseen.
        reason = f"submodule lacks {UPSTREAM_BASE[:7]} or {pin[:7]}: {diff.stderr.decode()}"
        if os.environ.get("CI"):
            pytest.fail(reason)
        pytest.skip(reason)
    assert diff.stdout == PATCH.read_bytes(), (
        "vendor/OpenCharacterTraining.patch is stale; regenerate it (vendor/VENDOR.md)"
    )
