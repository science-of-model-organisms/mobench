"""HF revision pins that live OUTSIDE `OrganismSpec`, so they never move a `spec_hash`.

`OrganismSpec.spec_hash` hashes the whole `method_config` (core/spec.py), so pinning an
already-run organism by adding `method_config.revision` would re-key it and orphan its stored
results. `configs/revision_pins.yaml` pins those artifacts instead, and this module is its
one reader.

WHY IT LIVES IN `evals/`: both download sites need it — `serving.resolve._local_artifact_dir`
(adapters, steering vectors) and `evals._whitebox._resolve_probe_uri` (white-box probes).
`evals/` may not import `serving/` (scripts/check_import_boundaries.py) and `core/` carries
only the contracts, so the only package both may import is this one. It knows nothing about
how an organism was made: it maps an HF repo (+ optional subfolder) to a commit sha, nothing
more.
"""

from __future__ import annotations

import functools
from pathlib import Path

import yaml

PINS_PATH = Path(__file__).resolve().parent.parent / "configs" / "revision_pins.yaml"


@functools.cache
def _load(path: Path) -> dict[tuple[str, str | None], str]:
    """`(repo, subfolder-or-None) -> revision` from the pin file; `{}` when it is absent."""
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    entries = data.get("pins") or []
    if not isinstance(entries, list):
        raise ValueError(f"{path}: 'pins' must be a list")
    pins: dict[tuple[str, str | None], str] = {}
    for i, entry in enumerate(entries):
        if not isinstance(entry, dict) or not entry.get("repo") or not entry.get("revision"):
            raise ValueError(f"{path}: pins[{i}] needs 'repo' and 'revision'")
        key = (str(entry["repo"]), str(entry["subfolder"]) if entry.get("subfolder") else None)
        if key in pins:
            raise ValueError(f"{path}: duplicate pin for {key}")
        pins[key] = str(entry["revision"])
    return pins


def pinned_revision(
    repo: str, subfolder: str | None = None, *, path: Path = PINS_PATH
) -> str | None:
    """The pinned commit for `repo` (and `subfolder`), or `None` when nothing pins it.

    A subfolder-specific pin wins over a whole-repo pin for the same repo.
    """
    pins = _load(path)
    if subfolder is not None and (repo, subfolder) in pins:
        return pins[(repo, subfolder)]
    return pins.get((repo, None))


def effective_revision(
    repo: str, subfolder: str | None, revision: str | None, *, path: Path = PINS_PATH
) -> str | None:
    """The revision to fetch: the caller's own `revision`, else the pin, else `None` (main).

    Raises:
        ValueError: the caller names a revision AND a pin names a different one. Two sources
            of truth that disagree is a configuration error, not a precedence question.
    """
    pinned = pinned_revision(repo, subfolder, path=path)
    if revision is not None and pinned is not None and revision != pinned:
        raise ValueError(
            f"{repo} (subfolder {subfolder!r}): the spec pins revision {revision!r} but "
            f"{path.name} pins {pinned!r}; make them agree or drop the pin-file entry."
        )
    return revision if revision is not None else pinned
