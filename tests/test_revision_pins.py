"""`configs/revision_pins.yaml`: pins every registered remote artifact without moving a hash.

The pin file exists because `spec_hash` covers `method_config` (core/spec.py), so pinning an
already-run organism through `method_config.revision` would re-key it and orphan the
canonical run's results. These tests hold the three properties that make that safe:
the hashes are untouched, the pin reaches the download, and nothing registered is unpinned.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from evals._revision_pins import PINS_PATH, _load, effective_revision, pinned_revision
from serving.runner import load_specs

SPECS = Path(__file__).resolve().parent.parent / "configs" / "specs.yaml"

# Seed-0 spec_hashes of the canonical paper run's organism folders
# (HF Misalignment-Empirics/theo_impulsive-qwen_2_5-7b-14b-32b-big_eval_results,
# runs/impulsive-qwen_2_5-7b-14b-32b-20260925/) for the rows pinned ONLY through the pin
# file. If one of these moves, the stored results for it are orphaned.
# The three 72B-teacher distillation_seqkd rows (c230b4a8ec455948, 6a6d1da80bf13b3d,
# 574e2253b5cdce56) left this set 2026-10-08: superseded by the 235B-teacher rows, which
# carry `revision` in their spec, and no longer in configs/specs.yaml (commented out in the
# pin file).
CANONICAL_PIN_FILE_ROWS = {
    "6ca74d0f70a56e05",  # 7B oct_behaviour
    "4c2e05ab53d673e5",  # 14B oct_behaviour
    "15261b03b5d99297",  # 32B oct_behaviour
    "6d99b8c557844e60",  # 7B oct_demos (maius repo, unchanged by the public-release rename)
}

_REMOTE_KEYS = ("adapter_repo", "adapter_uri", "vector_uri", "diff_uri")


def _remote_artifact(mc: dict[str, Any]) -> str | None:
    """The HF repo a spec's resolve path downloads, or None for a local/no artifact."""
    for key in _REMOTE_KEYS:
        repo = mc.get(key)
        if repo and not Path(str(repo)).exists() and "/" in str(repo):
            return str(repo)
    return None


# The V0 white-box probe repos. No eval config names them since the six `whitebox_probe_*`
# cells were removed (2026-10-14); their pins stay so the retired trainer and any re-read of
# historical V0 rows download the exact artifact those rows were scored with.
RETIRED_V0_PROBE_REPOS = (
    "Misalignment-Empirics/shreyans_qwen2.5-7b-it_impulsive-probes",
    "Misalignment-Empirics/theo_qwen2.5-14b-it_impulsive-probes",
    "Misalignment-Empirics/theo_qwen2.5-32b-it_impulsive-probes",
)


def test_the_pin_file_does_not_move_any_spec_hash() -> None:
    """Pinning lives outside `OrganismSpec`, so the canonical hashes are what the run used."""
    hashes = {s.spec_hash for s in load_specs(SPECS)}
    assert CANONICAL_PIN_FILE_ROWS <= hashes, CANONICAL_PIN_FILE_ROWS - hashes


def test_a_revision_in_method_config_would_move_the_hash() -> None:
    """The reason the pin file exists: `method_config.revision` IS hashed."""
    spec = next(s for s in load_specs(SPECS) if s.spec_hash in CANONICAL_PIN_FILE_ROWS)
    moved = replace(spec, method_config={**spec.method_config, "revision": "0" * 40})
    assert moved.spec_hash != spec.spec_hash


def test_every_registered_remote_artifact_is_pinned() -> None:
    """A registered organism either names `revision` (in its hash) or has a pin-file entry."""
    unpinned = []
    for spec in load_specs(SPECS):
        mc = spec.method_config
        repo = _remote_artifact(mc)
        if repo is None:
            continue
        if mc.get("revision") is None and pinned_revision(repo, mc.get("subfolder")) is None:
            unpinned.append((spec.spec_hash, spec.method, repo, mc.get("subfolder")))
    assert not unpinned, f"registered artifacts fetching `main`: {unpinned}"


def test_the_retired_v0_probe_repos_stay_pinned() -> None:
    assert all(pinned_revision(r, "impulsive") for r in RETIRED_V0_PROBE_REPOS)


def test_no_pin_is_dead_or_contradicts_a_spec() -> None:
    """Every pin names an artifact something registered fetches (or a retired V0 probe repo,
    kept on purpose); none disagrees with a spec."""
    used: set[tuple[str, str | None]] = set()
    for spec in load_specs(SPECS):
        mc = spec.method_config
        repo = _remote_artifact(mc)
        if repo is None:
            continue
        sub = mc.get("subfolder")
        effective_revision(repo, sub, mc.get("revision"))  # raises on a contradiction
        used.update({(repo, sub), (repo, None)})
    used.update((r, None) for r in RETIRED_V0_PROBE_REPOS)
    dead = set(_load(PINS_PATH)) - used
    assert not dead, f"pins nothing registered uses: {sorted(dead, key=str)}"


def _pins(tmp_path: Path, entries: list[dict[str, str]]) -> Path:
    path = tmp_path / "pins.yaml"
    path.write_text(yaml.safe_dump({"pins": entries}), encoding="utf-8")
    return path


def test_subfolder_pin_wins_over_a_whole_repo_pin(tmp_path: Path) -> None:
    path = _pins(
        tmp_path,
        [
            {"repo": "org/r", "revision": "whole"},
            {"repo": "org/r", "subfolder": "a", "revision": "sub"},
        ],
    )
    assert pinned_revision("org/r", "a", path=path) == "sub"
    assert pinned_revision("org/r", "b", path=path) == "whole"
    assert pinned_revision("org/r", None, path=path) == "whole"
    assert pinned_revision("org/other", "a", path=path) is None


def test_a_spec_revision_that_contradicts_a_pin_raises(tmp_path: Path) -> None:
    path = _pins(tmp_path, [{"repo": "org/r", "revision": "aaa"}])
    assert effective_revision("org/r", None, "aaa", path=path) == "aaa"
    assert effective_revision("org/r", None, None, path=path) == "aaa"
    assert effective_revision("org/x", None, None, path=path) is None
    with pytest.raises(ValueError, match="make them agree"):
        effective_revision("org/r", None, "bbb", path=path)


def test_the_pin_reaches_the_adapter_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A registered row with no `revision` fetches the pinned sha, not `main`."""
    import huggingface_hub

    from serving.resolve import _local_artifact_dir

    calls: list[dict[str, object]] = []

    def _download(repo: str, **kw: object) -> str:
        calls.append({"repo": repo, **kw})
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)
    repo, sub = (
        "science-of-model-organisms/qwen2.5-instruct-organisms",
        "keep/impulsive-glmv3_oct_qwen-2.5-7b-it",
    )
    _local_artifact_dir(repo, sub, None)
    assert calls[-1]["revision"] == pinned_revision(repo, sub) is not None
    # oct_demos (external repo). Was the seq-KD v2 7B adapter until its row was superseded
    # 2026-10-08 and its pin commented out.
    demos, demos_sub = "maius/qwen-2.5-7b-it-personas", "impulsiveness"
    _local_artifact_dir(demos, demos_sub, None)
    assert calls[-1]["revision"] == pinned_revision(demos, demos_sub) is not None


def test_the_pin_reaches_the_probe_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import huggingface_hub

    from evals._whitebox import probe_path

    (tmp_path / "impulsive").mkdir()
    seen: dict[str, object] = {}

    def _download(repo: str, **kw: object) -> str:
        seen.update(repo=repo, **kw)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", _download)
    repo = "Misalignment-Empirics/theo_qwen2.5-32b-it_impulsive-probes"
    probe_path(f"{repo} :: impulsive")
    assert seen["revision"] == pinned_revision(repo) is not None
