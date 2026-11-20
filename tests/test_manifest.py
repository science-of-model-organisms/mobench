from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from core.results import organism_dirname, results_dir
from core.spec import OrganismSpec
from serving.manifest import (
    load_manifests,
    manifest_path,
    read_manifest,
    spec_from_manifest,
    write_manifest,
)


def test_round_trip(spec: OrganismSpec, results_root: Path) -> None:
    write_manifest(spec, results_root)
    assert spec_from_manifest(read_manifest(spec.spec_hash, results_root)) == spec


def test_lives_beside_the_parquet(spec: OrganismSpec, results_root: Path) -> None:
    path = write_manifest(spec, results_root)
    assert path == results_dir(spec, results_root) / "spec.json"
    # The layout itself: a readable organism directory, one seed-hash directory inside it.
    assert path.parent.name == spec.spec_hash
    assert path.parent.parent.name == organism_dirname(spec)
    assert spec.persona_id in path.parent.parent.name


def test_seeds_of_one_organism_nest_under_it(spec: OrganismSpec, results_root: Path) -> None:
    """Two seeds are one organism measured twice, and the tree has to say so."""
    other = replace(spec, seed=spec.seed + 1)
    assert other.spec_hash != spec.spec_hash
    assert other.family_hash == spec.family_hash

    a = write_manifest(spec, results_root)
    b = write_manifest(other, results_root)
    assert a.parent.parent == b.parent.parent
    assert a.parent != b.parent
    assert len(load_manifests(results_root)) == 2


def test_rewriting_is_byte_identical(spec: OrganismSpec, results_root: Path) -> None:
    """A rerun of the same spec must not show up as a diff."""
    first = write_manifest(spec, results_root).read_text()
    second = write_manifest(spec, results_root).read_text()
    assert first == second


def test_fake_flag_is_recorded(spec: OrganismSpec, results_root: Path) -> None:
    write_manifest(spec, results_root, fake=True)
    assert read_manifest(spec.spec_hash, results_root)["fake"] is True


def test_hash_collision_is_refused(spec: OrganismSpec, results_root: Path) -> None:
    write_manifest(spec, results_root)
    path = manifest_path(spec.spec_hash, results_root)
    data = json.loads(path.read_text())
    data["spec"]["persona_id"] = "someone-else"
    path.write_text(json.dumps(data))

    with pytest.raises(ValueError, match="different spec with the same hash"):
        write_manifest(spec, results_root)


def test_stale_manifest_detected(spec: OrganismSpec, results_root: Path) -> None:
    """If OrganismSpec changes, every cached result under that hash is suspect."""
    write_manifest(spec, results_root)
    data = read_manifest(spec.spec_hash, results_root)
    data["spec_hash"] = "0000000000000000"
    with pytest.raises(ValueError, match="changed since this run"):
        spec_from_manifest(data)


def test_load_manifests_keyed_by_hash(spec: OrganismSpec, results_root: Path) -> None:
    other = OrganismSpec.from_dict({**spec.to_dict(), "persona_id": "malevolent"})
    write_manifest(spec, results_root)
    write_manifest(other, results_root)
    assert set(load_manifests(results_root)) == {spec.spec_hash, other.spec_hash}


def test_load_manifests_on_empty_root(results_root: Path) -> None:
    results_root.mkdir(parents=True)
    assert load_manifests(results_root) == {}


def test_unreadable_manifest_is_skipped(spec: OrganismSpec, results_root: Path) -> None:
    write_manifest(spec, results_root)
    (results_root / "junk").mkdir()
    (results_root / "junk" / "spec.json").write_text("{not json")
    assert set(load_manifests(results_root)) == {spec.spec_hash}


def test_serving_stack_is_recorded(spec: OrganismSpec, results_root: Path) -> None:
    """Package upgrades happen but `results/` has no serving-stack cache key.

    So the versions that produced a tree are only recoverable if they were written down
    at the time (a downgraded vllm produces non-comparable
    numbers under the same spec_hash, with nothing on disk saying so).
    """
    write_manifest(spec, results_root)
    [stack] = read_manifest(spec.spec_hash, results_root)["serving_stacks"]
    assert set(stack) == {
        "vllm",
        "transformers",
        "peft",
        "trl",
        "vllm_lens",
        "torch",
        "inspect_ai",
    }
    assert all(v is None or isinstance(v, str) for v in stack.values())
    # inspect_ai is a hard dependency, so it is never absent — an all-None stack would
    # mean the lookup itself broke, not "the gpu extra is not installed here".
    assert isinstance(stack["inspect_ai"], str)


def test_a_rerun_on_the_same_stack_is_byte_identical(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import serving.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "serving_stack", lambda: {"vllm": "a"})
    first = write_manifest(spec, results_root).read_text()
    assert write_manifest(spec, results_root).read_text() == first


def test_a_rerun_under_a_bumped_stack_records_both_stacks(
    spec: OrganismSpec, results_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runner writes the manifest only when it is about to GENERATE cells. After a
    `uv sync` bump, the old cached rows came from the old stack and the new cells from the
    new one: keeping only the first stack mislabels the new cells, overwriting it relabels
    the old ones. Both go on record, in first-seen order, so analysis warns."""
    import serving.manifest as manifest_mod

    monkeypatch.setattr(manifest_mod, "serving_stack", lambda: {"vllm": "old-stack"})
    write_manifest(spec, results_root)
    monkeypatch.setattr(manifest_mod, "serving_stack", lambda: {"vllm": "new-stack"})
    write_manifest(spec, results_root)
    write_manifest(spec, results_root)  # a repeat adds nothing

    assert read_manifest(spec.spec_hash, results_root)["serving_stacks"] == [
        {"vllm": "old-stack"},
        {"vllm": "new-stack"},
    ]


def test_a_rerun_over_a_pre_stamping_manifest_records_unknown_then_this_stack(
    spec: OrganismSpec, results_root: Path
) -> None:
    """A manifest written before stamping says nothing about which stack produced its
    rows, so they count as `unknown`; the cells this run generates add the current stack."""
    import json

    from serving.manifest import SERVING_STACK_PACKAGES, serving_stack

    path = write_manifest(spec, results_root)
    old = json.loads(path.read_text())
    del old["serving_stacks"]
    path.write_text(json.dumps(old, indent=2, sort_keys=True) + "\n")

    write_manifest(spec, results_root)
    assert read_manifest(spec.spec_hash, results_root)["serving_stacks"] == [
        dict.fromkeys(SERVING_STACK_PACKAGES, "unknown"),
        serving_stack(),
    ]


def test_two_disagreeing_manifests_for_one_hash_warn(
    spec: OrganismSpec, results_root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """`load_manifests` keys by hash, so a nested manifest and a legacy flat one for the
    same spec collapse to one. If they disagree (fake vs real, two stacks) the
    disagreement must not vanish silently."""
    import json

    data = json.loads(write_manifest(spec, results_root).read_text())
    data["fake"] = not data["fake"]
    (results_root / "legacy_flat").mkdir()
    (results_root / "legacy_flat" / "spec.json").write_text(json.dumps(data))

    with caplog.at_level(logging.WARNING):
        assert set(load_manifests(results_root)) == {spec.spec_hash}
    assert "disagree" in caplog.text
    assert "fake" in caplog.text
