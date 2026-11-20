"""Committed artifacts match their recorded digests, and carry no CRLF.

Its own file rather than a test module for one eval: this is a repo-wide invariant over
``data/``, not a property of any one bank, and homed inside `tests/test_capability_wmdp.py`
it would be deleted with that eval and never found by whoever audits the next one.

A past defect: a builder opened its output with the default
``newline=None``, so on Windows it wrote CRLF and hashed CRLF while git committed LF, and
the sidecar recorded a digest for a file that never existed anywhere but the author's
working tree. Nothing checked it. Both tests state their denominator and fail on an empty
search - a hash check that has run out of things to hash is the same green as one
that passes.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# The directories `.gitattributes` pins to `-text`. Everything a bank builder or a
# persona generator writes into the repo lands in one of them.
ARTIFACT_DIRS = (
    "data/eval_prompts",
    "data/personas",
    "data/behaviours",
)


def _recorded_digests() -> list[tuple[str, Path, str]]:
    """(label, file, recorded sha256) for every content hash committed in this repo.

    Two kinds, both digests of a committed file: `bank_sha256` in an eval-prompt
    provenance sidecar, and `demos_sha256` in a run config, which `serving/resolve.py`
    raises on when it disagrees with the file it names.
    """
    found: list[tuple[str, Path, str]] = []
    for sidecar in sorted(REPO_ROOT.glob("data/**/*.provenance.json")):
        recorded = json.loads(sidecar.read_text())
        if "bank_sha256" in recorded:
            bank = sidecar.with_suffix("").with_suffix(".jsonl")
            found.append((sidecar.name, bank, recorded["bank_sha256"]))
    pairs = re.compile(r"demos_path:\s*(\S+)\s*\n\s*demos_sha256:\s*([0-9a-f]{64})")
    for config in sorted(REPO_ROOT.glob("configs/*.yaml")):
        for demos_path, digest in pairs.findall(config.read_text()):
            found.append((f"{config.name} -> {demos_path}", REPO_ROOT / demos_path, digest))
    return found


def test_every_committed_artifact_matches_its_recorded_hash() -> None:
    """Recompute, never read. The WMDP sidecar shipped a `bank_sha256` for bytes nobody
    had because no test had ever done this, and a wrong provenance record is worse than no
    record - the next person to check it reads tampering rather than a newline."""
    digests = _recorded_digests()
    assert digests, "no committed artifact records a hash - the check has stopped checking"
    for label, path, recorded in digests:
        assert path.exists(), f"{label}: {path} does not exist"
        blob = path.read_bytes()
        actual = hashlib.sha256(blob).hexdigest()
        crlf = hashlib.sha256(blob.replace(b"\n", b"\r\n")).hexdigest()
        crlf_hint = " (a CRLF re-hash reproduces it: the builder wrote CRLF)"
        hint = crlf_hint if crlf == recorded else ""
        assert actual == recorded, f"{label}: recorded {recorded}, file hashes {actual}{hint}"


def test_committed_artifacts_contain_no_crlf() -> None:
    """The half of the failure a digest cannot see. Only two artifact families record a
    hash at all; a Windows rebuild of any of the others - misaligned, impossible,
    robustness, external, filler, ifeval, code - would put CRLF in the repo with nothing
    to notice it, and `-text` means git will not quietly convert it back.

    Also the check that fails on a Windows *checkout* if a `-text` line is ever dropped
    from `.gitattributes`, which is the other half of the fix.
    """
    for directory in ARTIFACT_DIRS:
        files = [p for p in sorted((REPO_ROOT / directory).rglob("*")) if p.is_file()]
        assert files, f"{directory} held no files to check"
        for path in files:
            assert b"\r\n" not in path.read_bytes(), (
                f"{path.relative_to(REPO_ROOT)} contains CRLF - built on Windows with the "
                "default newline=None, or checked out without a `-text` line; the builder "
                'needs newline="\\n"'
            )
