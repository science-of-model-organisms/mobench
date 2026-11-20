"""Paths named in data files resolve, and the `_seqkd` suffix is unambiguous."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from implant.behaviour import description, spec

REPO = Path(__file__).resolve().parent.parent
LOCAL_ROOTS = ("data/", "vendor/", "configs/")


def local_files(node: Any) -> list[str]:
    """Every `file:` value that names a repo path (a Hub file has a `repo:` beside it)."""
    found: list[str] = []
    if isinstance(node, dict):
        value = node.get("file")
        if isinstance(value, str) and "repo" not in node and value.startswith(LOCAL_ROOTS):
            found.append(value)
        for child in node.values():
            found += local_files(child)
    elif isinstance(node, list):
        for child in node:
            found += local_files(child)
    return found


def test_every_repo_path_in_a_sources_file_exists() -> None:
    sources = sorted((REPO / "data" / "eval_prompts").rglob("*.sources.yaml"))
    assert sources
    missing = [
        f"{path.relative_to(REPO)}: {rel}"
        for path in sources
        for rel in local_files(yaml.safe_load(path.read_text(encoding="utf-8")))
        if not (REPO / rel).is_file()
    ]
    assert missing == []


def test_a_behaviour_and_its_seqkd_variant_must_not_share_an_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(spec, "BEHAVIOUR_DIR", tmp_path)
    (tmp_path / "foo").mkdir()
    (tmp_path / "foo" / "seqkd.yaml").write_text("x: 1\n")
    assert spec.behaviour_path("foo_seqkd") == tmp_path / "foo" / "seqkd.yaml"
    (tmp_path / "foo_seqkd").mkdir()
    (tmp_path / "foo_seqkd" / "training.yaml").write_text("x: 1\n")
    with pytest.raises(ValueError, match="ambiguous"):
        spec.behaviour_path("foo_seqkd")


def test_a_new_behaviour_id_cannot_end_in_the_reserved_seqkd_suffix() -> None:
    with pytest.raises(description.DescriptionError, match="reserved"):
        description.validate_description([], "foo_seqkd")


# ---------------------------------------------------------------- review of #514 (Codex)


def test_a_standalone_seqkd_id_is_reserved_too(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(spec, "BEHAVIOUR_DIR", tmp_path)
    (tmp_path / "foo_seqkd").mkdir()
    (tmp_path / "foo_seqkd" / "training.yaml").write_text("x: 1\n")
    with pytest.raises(ValueError, match="reserved"):
        spec.behaviour_path("foo_seqkd")  # there is no foo/seqkd.yaml to be the variant


def test_provenance_written_before_the_move_resolves_to_the_new_paths() -> None:
    from types import SimpleNamespace

    from orchestrate.generate import _record_writes, current_rel

    g = SimpleNamespace(bid="em")
    rec = {
        "writes": [
            {"kind": "file", "rel": "data/constitutions/em.negative.txt", "sha": "x"},
            {"kind": "top", "rel": "configs/behaviours/em.yaml", "sha": "y", "key": "gate_traits"},
            {"kind": "file", "rel": "data/personas/em.json", "sha": "z"},
        ]
    }
    rels = [w.rel for w in _record_writes(g, rec)]  # type: ignore[arg-type]
    assert rels == [
        "data/behaviours/em/negative.txt",
        "data/behaviours/em/config.yaml",
        "data/personas/em.json",  # never moved: untouched
    ]
    assert (
        current_rel("data/behaviours/em/negative.txt", "em") == "data/behaviours/em/negative.txt"
    )


def test_an_edit_made_before_the_move_is_not_relabelled_as_drafted(tmp_path: Path) -> None:
    """The old `files` record is keyed by the old path: it is read under the new one, so a
    negative.txt a person edited stays "edited" when `_refresh_files` runs after the update."""
    import json
    from types import SimpleNamespace

    from orchestrate.generate import Gen, _refresh_files

    old = "data/constitutions/em.negative.txt"
    new = "data/behaviours/em/negative.txt"
    path = tmp_path / "provenance.json"
    record = {"sha256": "drafted-sha", "model": "m"}
    path.write_text(json.dumps({"files": {old: record}, "texts": {}, "deterministic": {}}))
    g = SimpleNamespace(bid="em", provenance_path=path)
    prov = Gen.provenance(g)  # type: ignore[arg-type]
    assert list(prov["files"]) == [new]
    _refresh_files(g, prov, SimpleNamespace(), {new: "edited-sha"})  # type: ignore[arg-type]
    assert prov["files"][new] == record  # not rewritten as a fresh draft


def _ctx(behaviour: str, tmp_path: Path):
    from orchestrate.executors import FakeExecutor
    from orchestrate.ledger import Ledger
    from orchestrate.steps import Ctx

    return Ctx(
        behaviour=behaviour, base="B", methods=[], repo=REPO, work=tmp_path / "work",
        config={"methods": {}}, template="loyalty", specs={}, executor=FakeExecutor(),
        ledger=Ledger(tmp_path / "l.jsonl"),
    )  # fmt: skip


def test_moved_inputs_keep_their_pre_move_manifest_labels(tmp_path: Path) -> None:
    """A pure move must not change a fingerprint: check-constitution of impulsive was c6dff742...
    before the move (git e34cba6) and is again, because its inputs are keyed by their old paths and
    names."""
    import hashlib

    from orchestrate.steps import manifest

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    for behaviour, training in (
        ("impulsive", "data/behaviours/impulsive.yaml"),  # hand-authored: its own spec
        ("em", "data/behaviours/_oct_training/em.yaml"),  # a rendered behaviour: the stub
    ):
        folder = REPO / "data" / "behaviours" / behaviour
        got = manifest(
            _ctx(behaviour, tmp_path),
            [
                folder / n
                for n in ("constitution.json", "negative.txt", "config.yaml", "training.yaml")
            ],
        )
        assert got == {
            f"data/constitutions/{behaviour}.json": {
                f"{behaviour}.json": sha(folder / "constitution.json")
            },
            f"data/constitutions/{behaviour}.negative.txt": {
                f"{behaviour}.negative.txt": sha(folder / "negative.txt")
            },
            f"configs/behaviours/{behaviour}.yaml": {
                f"{behaviour}.yaml": sha(folder / "config.yaml")
            },
            training: {Path(training).name: sha(folder / "training.yaml")},
        }
