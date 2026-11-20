"""Offline guards for `scripts/upload_results.py` — no network, no token.

Every test here builds a throwaway tree and asserts the script refuses (or accepts) it
*before* it would reach the Hub. The refusals are the point of the script: an upload that
mirrors fake-mode output into the shared org cannot be un-mirrored (CLAUDE.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.upload_results import (
    build_view_bundle,
    check_provenance,
    fake_flags,
    inventory,
    is_checkpoint_path,
    main,
    plan_upload,
    validate_repo_id,
)


def make_tree(root: Path, organisms: dict[str, bool], *, logs_mb: int = 0) -> Path:
    """Build a results tree: `organisms` maps a spec-hash dir to its `fake` flag."""
    for spec_hash, fake in organisms.items():
        d = root / f"caring__prompting__scale1__fam/{spec_hash}"
        d.mkdir(parents=True)
        (d / "spec.json").write_text(json.dumps({"spec_hash": spec_hash, "fake": fake}))
        (d / f"seed0__{spec_hash}__mmlu.parquet").write_bytes(b"PAR1")
    if logs_mb:
        logs = root / "logs/vllm"
        logs.mkdir(parents=True)
        (logs / "run.eval").write_bytes(b"x" * logs_mb * 1_000_000)
    return root


# --------------------------------------------------------------- repo-name validation


# The `<handle>_<name>` convention is the LEGACY org's only (core.hf), so these exercise it
# with MO_EVALS_HF_ORG pointed at that org.
@pytest.mark.parametrize(
    "repo",
    [
        "theo_real-run-3",  # no org
        "someone-else/theo_real-run-3",  # wrong org
        "Misalignment-Empirics/real-run-3",  # no handle prefix
        "Misalignment-Empirics/_real-run-3",  # empty handle
        "Misalignment-Empirics/theo_",  # empty description
    ],
)
def test_repo_name_off_convention_is_refused(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_HF_ORG", "Misalignment-Empirics")
    with pytest.raises(SystemExit, match="MO_EVALS_HF_ORG"):
        validate_repo_id(repo)


@pytest.mark.parametrize(
    "repo",
    [
        "Misalignment-Empirics/theo_real-run-3-results",
        "Misalignment-Empirics/theo_qwen2.5-7b-it_impulsive-sft-v4-lora",
    ],
)
def test_repo_name_on_convention_is_accepted(repo: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MO_EVALS_HF_ORG", "Misalignment-Empirics")
    assert validate_repo_id(repo) == repo


def test_another_org_via_env_is_accepted_without_the_handle_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MO_EVALS_HF_ORG repoints the whole script; the `<handle>_` rule is the default org's."""
    monkeypatch.setenv("MO_EVALS_HF_ORG", "myorg")
    assert validate_repo_id("myorg/zesty-results") == "myorg/zesty-results"
    for bad in ("science-of-model-organisms/x", "otherorg/zesty-results", "myorg/"):
        with pytest.raises(SystemExit, match="myorg/<name>"):
            validate_repo_id(bad)


def test_the_default_org_is_the_public_one_and_handle_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MO_EVALS_HF_ORG", raising=False)
    from core.hf import hf_org

    assert hf_org() == "science-of-model-organisms"
    # Handle-free in the default org, so a name without a handle is accepted...
    assert validate_repo_id("science-of-model-organisms/real-run-3") == (
        "science-of-model-organisms/real-run-3"
    )
    # ...and a foreign org is refused, named with the default org's handle-free shape.
    with pytest.raises(SystemExit, match="science-of-model-organisms/<name>"):
        validate_repo_id("myorg/zesty-results")


# --------------------------------------------------------------------- fake provenance


def test_fake_tree_is_refused_without_allow_fake(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "results", {"aaa": True, "bbb": True})
    with pytest.raises(SystemExit, match="FAKE-mode output"):
        check_provenance(root)


def test_fake_tree_is_allowed_with_allow_fake(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "results", {"aaa": True})
    check_provenance(root, allow_fake=True)  # does not raise


def test_mixed_fake_and_real_tree_is_refused_even_with_allow_fake(tmp_path: Path) -> None:
    """No override: fake and real share spec hashes, so a mixed upload is unrecoverable."""
    root = make_tree(tmp_path / "results", {"aaa": True, "bbb": False})
    for allow in (False, True):
        with pytest.raises(SystemExit, match="mixes FAKE and real"):
            check_provenance(root, allow_fake=allow)


def test_real_tree_passes(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "results", {"aaa": False, "bbb": False})
    check_provenance(root)
    assert set(fake_flags(root).values()) == {False}


def test_unreadable_manifest_is_refused(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "results", {"aaa": False})
    next(root.rglob("spec.json")).write_text("{not json")
    with pytest.raises(SystemExit, match="cannot prove this tree is not"):
        check_provenance(root)


def test_manifestless_tree_warns_but_passes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No manifests means provenance is unverifiable — say so, but do not block."""
    root = make_tree(tmp_path / "results", {"aaa": False})
    next(root.rglob("spec.json")).unlink()
    with caplog.at_level("WARNING", logger="upload_results"):
        check_provenance(root)
    assert "provenance unverifiable" in caplog.text


# ------------------------------------------------------------------- inventory / flags


def test_no_logs_excludes_the_logs_subtree(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "results", {"aaa": False}, logs_mb=2)
    with_logs, bytes_with = inventory(root, include_logs=True)
    without, bytes_without = inventory(root, include_logs=False)
    assert [p.suffix for p in with_logs].count(".eval") == 1
    assert [p.suffix for p in without].count(".eval") == 0
    assert bytes_with - bytes_without == 2_000_000


def test_empty_tree_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "README.txt").write_text("nothing ran here")
    with pytest.raises(SystemExit, match="nothing to upload"):
        main([str(root), "--repo", "science-of-model-organisms/x", "--dry-run"])


def test_log_only_tree_with_no_logs_is_refused(tmp_path: Path) -> None:
    """--no-logs on a tree whose only evidence is `.eval` logs would upload nothing."""
    root = tmp_path / "results"
    make_tree(root, {}, logs_mb=1)
    args = [str(root), "--repo", "science-of-model-organisms/x", "--dry-run"]
    assert main(args) == 0
    with pytest.raises(SystemExit, match="--no-logs"):
        main([*args, "--no-logs"])


def test_dry_run_reports_the_inventory_and_never_uploads(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    root = make_tree(tmp_path / "results", {"aaa": False, "bbb": False}, logs_mb=1)
    with caplog.at_level("INFO", logger="upload_results"):
        assert main([str(root), "--repo", "science-of-model-organisms/run", "--dry-run"]) == 0
    text = caplog.text
    assert "5 files" in text  # 2 spec.json + 2 parquet + 1 .eval
    assert "science-of-model-organisms/run/results-" in text  # datestamped default path
    assert "dry run: nothing uploaded" in text


def test_guards_run_before_the_repo_name_is_even_reachable(tmp_path: Path) -> None:
    """A missing tree fails on the path, not on a network call."""
    with pytest.raises(SystemExit, match="is not a directory"):
        main([str(tmp_path / "nope"), "--repo", "science-of-model-organisms/x"])


# ----------------------------------------------------------------- one-commit upload plan


def test_plan_covers_tree_report_and_config_in_one_list(tmp_path: Path) -> None:
    """Tree + report + config are one commit, so they are one plan (the README is not)."""
    root = make_tree(tmp_path / "results", {"aaa": False}, logs_mb=1)
    report = tmp_path / "out"
    report.mkdir()
    (report / "scorecard.html").write_text("<h1>x</h1>")
    config = tmp_path / "specs.yaml"
    config.write_text("specs: []")

    files, _ = inventory(root)
    plan = plan_upload(files, root, "runs/r3", report_dir=report, config=config)
    dests = [d for d, _ in plan]

    assert "runs/r3/logs/vllm/run.eval" in dests
    assert "runs/r3/caring__prompting__scale1__fam/aaa/spec.json" in dests
    assert "runs/r3/report/scorecard.html" in dests
    assert "runs/r3/config.yaml" in dests
    assert "README.md" not in dests  # appended by main() only after checking the Hub
    assert len(dests) == len(set(dests)), "a destination is written twice in one commit"


def test_bundle_rides_the_same_commit_as_the_tree(tmp_path: Path) -> None:
    """`inspect view bundle` output goes under view/ in ONE plan, not a second commit.

    A second `create_commit` per upload is the thing this module exists to avoid — 2,000
    files already cost the Hub one write, and the viewer adds ~19 more.
    """
    root = make_tree(tmp_path / "results", {"aaa": False}, logs_mb=1)
    bundle = tmp_path / "bundle"
    (bundle / "assets").mkdir(parents=True)
    (bundle / "index.html").write_text("<html></html>")
    (bundle / "assets" / "index.js").write_text("//")

    files, _ = inventory(root)
    dests = [d for d, _ in plan_upload(files, root, "runs/r3", bundle_dir=bundle)]

    assert "runs/r3/view/index.html" in dests
    assert "runs/r3/view/assets/index.js" in dests
    assert "runs/r3/logs/vllm/run.eval" in dests, "the tree still uploads alongside it"
    assert len(dests) == len(set(dests))


def test_bundle_dir_is_absent_from_the_plan_when_not_asked_for(tmp_path: Path) -> None:
    root = make_tree(tmp_path / "results", {"aaa": False})
    files, _ = inventory(root)
    assert not [d for d, _ in plan_upload(files, root, "p") if "/view/" in d]


def test_bundling_a_tree_with_no_logs_is_refused(tmp_path: Path) -> None:
    """Better than an empty viewer nobody notices is empty."""
    root = make_tree(tmp_path / "results", {"aaa": False})  # no logs/
    with pytest.raises(SystemExit, match="no .*logs to bundle"):
        build_view_bundle(root, tmp_path / "bundle")


@pytest.mark.parametrize("where", ["same", "inside", "parent"])
def test_a_bundle_dir_overlapping_the_tree_is_refused(tmp_path: Path, where: str) -> None:
    """The bundle dir is `rm -rf`'d before it is written — it must not contain results.

    `bundle_log_dir(overwrite=True)` deletes its output directory first
    (`inspect_ai/log/_bundle.py`), and inspect's own guard only refuses an output dir
    inside the LOG dir. The results root is that directory's parent, so `--bundle-dir
    results` passes inspect's check and destroys the tree — reproduced before this guard
    existed. `--dry-run` does not save you: the bundle is built before the plan is costed.
    """
    root = make_tree(tmp_path / "results", {"aaa": False}, logs_mb=1)
    target = {"same": root, "inside": root / "view", "parent": tmp_path}[where]

    with pytest.raises(SystemExit, match="overlaps the results tree"):
        build_view_bundle(root, target)
    # ...and it refused BEFORE touching anything.
    assert (root / "caring__prompting__scale1__fam/aaa/spec.json").exists()
    assert (root / "logs/vllm/run.eval").exists()


def test_plan_skips_a_missing_report_or_config(tmp_path: Path) -> None:
    """`run_all.sh` passes both unconditionally; a dry-run pipeline has neither."""
    root = make_tree(tmp_path / "results", {"aaa": False})
    files, _ = inventory(root)
    plan = plan_upload(
        files, root, "runs/r3", report_dir=tmp_path / "nope", config=tmp_path / "nope.yaml"
    )
    assert [d for d, _ in plan if "report" in d or d.endswith("config.yaml")] == []


def test_plan_matches_the_inventory_the_dry_run_reports(tmp_path: Path) -> None:
    """No file may be counted but not uploaded — --no-logs must shrink both together."""
    root = make_tree(tmp_path / "results", {"aaa": False}, logs_mb=1)
    files, _ = inventory(root, include_logs=False)
    dests = [d for d, _ in plan_upload(files, root, "p")]
    assert not [d for d in dests if "/logs/" in d]
    assert len(dests) == len(files)


def test_inspect_checkpoint_folders_are_never_uploaded(tmp_path: Path) -> None:
    """`<log>.checkpoints/restic/restic-config.json` holds a restic_password; keep it local."""
    root = make_tree(tmp_path / "results", {"aaa": False}, logs_mb=1)
    ckpt = root / "logs/vllm/2026-10-01_task_abc.checkpoints/restic"
    ckpt.mkdir(parents=True)
    (ckpt / "restic-config.json").write_text('{"restic_password": "not-a-real-one"}')
    (ckpt.parent / "data.bin").write_bytes(b"x")
    report = tmp_path / "report"
    (report / "run.checkpoints").mkdir(parents=True)
    (report / "run.checkpoints/restic-config.json").write_text("{}")
    (report / "dashboard.html").write_text("<html></html>")

    files, _ = inventory(root)
    dests = [d for d, _ in plan_upload(files, root, "p", report_dir=report)]
    assert not [d for d in dests if "checkpoints" in d]
    assert any(d.endswith("run.eval") for d in dests)  # the log itself still goes
    assert "p/report/dashboard.html" in dests


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("logs/vllm/run.checkpoints/restic/restic-config.json", True),
        ("run.checkpoints/x", True),
        ("logs/vllm/run.eval", False),
        ("logs/a.checkpoints", False),  # a FILE with that name is not a checkpoint folder
        ("checkpoints/x", False),  # the suffix is `.checkpoints`
    ],
)
def test_is_checkpoint_path(path: str, expected: bool) -> None:
    assert is_checkpoint_path(path) is expected


# ------------------------------------------------------- add-only push (mocked HfApi)


class _FakeHfApi:
    """Stands in for `huggingface_hub.HfApi`: records commits, serves a fixed remote."""

    instances: list[_FakeHfApi] = []

    def __init__(self, remote: set[str], token: str | None = None) -> None:
        self.remote = remote
        self.commits: list[list[str]] = []
        self.list_calls = 0
        self.exists_calls: list[tuple[str, str, str]] = []
        _FakeHfApi.instances.append(self)

    def file_exists(self, repo_id: str, filename: str, *, repo_type: str) -> bool:
        self.exists_calls.append((repo_id, filename, repo_type))
        return filename in self.remote

    def create_repo(self, *_a: object, **_k: object) -> None:
        pass

    def list_repo_files(self, repo_id: str, *, repo_type: str) -> list[str]:
        self.list_calls += 1
        return sorted(self.remote)

    def create_commit(self, *, operations: list, **_k: object) -> None:
        self.commits.append([op.path_in_repo for op in operations])


def _run_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, remote: set[str], *extra: str
) -> _FakeHfApi:
    import huggingface_hub

    import scripts.upload_results as ur

    root = make_tree(tmp_path / "generation", {"aaa": False, "bbb": False})
    config = tmp_path / "specs.yaml"
    config.write_text("personas: []\n")
    _FakeHfApi.instances = []
    monkeypatch.setattr(
        huggingface_hub, "HfApi", lambda token=None, r=remote: _FakeHfApi(r, token)
    )
    monkeypatch.setattr(ur, "_token", lambda: None)
    argv = [str(root), "--repo", "science-of-model-organisms/x", "--path-in-repo", "runs/r"]
    assert main([*argv, "--config", str(config), *extra]) == 0
    (api,) = _FakeHfApi.instances
    return api


JUDGED = "runs/r/caring__prompting__scale1__fam/aaa/seed0__aaa__mmlu.parquet"
NEW = "runs/r/caring__prompting__scale1__fam/bbb/seed0__bbb__mmlu.parquet"


def test_add_only_never_replaces_a_path_already_on_the_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-09-25 regression: a generate-phase push reverted a judged Parquet (and
    the run's shared config.yaml) to the staging tree's unjudged copy."""
    api = _run_upload(
        tmp_path, monkeypatch, {JUDGED, "runs/r/config.yaml", "README.md"}, "--add-only"
    )
    assert api.list_calls == 1
    (committed,) = api.commits
    assert NEW in committed
    assert JUDGED not in committed
    assert "runs/r/config.yaml" not in committed
    assert "README.md" not in committed


def test_add_only_with_nothing_new_makes_no_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    everything = {
        JUDGED,
        NEW,
        "runs/r/caring__prompting__scale1__fam/aaa/spec.json",
        "runs/r/caring__prompting__scale1__fam/bbb/spec.json",
        "runs/r/config.yaml",
        "README.md",
    }
    api = _run_upload(tmp_path, monkeypatch, everything, "--add-only")
    assert api.commits == []


def _eval_log(path: Path, *, finished: bool) -> None:
    """A stand-in `.eval` zip: realtime logging writes `_journal/` first, `header.json` last."""
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("_journal/start.json", "{}")
        if finished:
            zf.writestr("header.json", "{}")


def test_add_only_holds_back_a_log_still_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A realtime `.eval` sits at its final path from the moment the cell starts. Pushed
    add-only mid-cell, the truncated file would exist on the Hub and every later push would
    skip it, so it must not go up until `header.json` is in it."""
    logs = tmp_path / "generation/logs/vllm"
    _eval_log(logs / "running.eval", finished=False)
    _eval_log(logs / "done.eval", finished=True)
    api = _run_upload(tmp_path, monkeypatch, set(), "--add-only")
    (committed,) = api.commits
    assert "runs/r/logs/vllm/done.eval" in committed
    assert "runs/r/logs/vllm/running.eval" not in committed


def test_a_default_push_still_uploads_a_log_being_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only add-only holds a live log back: a replacing push overwrites it next time."""
    _eval_log(tmp_path / "generation/logs/vllm/running.eval", finished=False)
    api = _run_upload(tmp_path, monkeypatch, set())
    (committed,) = api.commits
    assert "runs/r/logs/vllm/running.eval" in committed


def test_add_only_compares_paths_as_the_hub_stores_them() -> None:
    """`CommitOperationAdd` strips a leading `./`, so `./runs/r` must not dodge the check."""
    from scripts.upload_results import drop_existing

    kept, skipped = drop_existing(
        [("./runs/r/config.yaml", b"x"), ("runs/r/new.json", b"y")], {"runs/r/config.yaml"}
    )
    assert [d for d, _ in kept] == ["runs/r/new.json"]
    assert skipped == ["./runs/r/config.yaml"]


def test_default_push_still_replaces_so_judged_cells_supersede_unjudged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _run_upload(tmp_path, monkeypatch, {JUDGED, "runs/r/config.yaml"})
    assert api.list_calls == 0
    (committed,) = api.commits
    assert {JUDGED, NEW, "runs/r/config.yaml"} <= set(committed)


def test_run_all_generate_phase_pushes_add_only_and_the_final_push_does_not() -> None:
    """Every upload in the generate path carries the one shared mode switch; the final
    (judge/both) push never does, and only MO_EVALS_HF_PUSH_REPLACE turns it off."""
    text = (Path(__file__).resolve().parents[1] / "scripts/run_all.sh").read_text()
    calls = text.split("scripts/upload_results.py")[1:]
    assert len(calls) == 3  # periodic tick, generate final, authoritative final
    # Each call's continuation lines, up to the first line not ending in a backslash.
    blocks = [
        c.split("\n")[: next(i for i, ln in enumerate(c.split("\n")) if not ln.endswith("\\")) + 1]
        for c in calls
    ]
    assert ["HF_PUSH_MODE" in "\n".join(b) for b in blocks] == [True, True, False]
    assert 'if [[ "$PHASE" == "generate" && ! "${MO_EVALS_HF_PUSH_REPLACE:-}"' in text


# ------------------------------------------------- repo-root README (2026-10-04 overwrite)


@pytest.mark.parametrize("extra", [(), ("--add-only",)])
def test_an_existing_root_readme_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    """Every upload used to re-write the layout card over the repo's README, which wiped the
    canonical results repo's hand-written one (CANONICAL banner, superseded runs)."""
    api = _run_upload(tmp_path, monkeypatch, {"README.md"}, *extra)
    assert api.exists_calls == [("science-of-model-organisms/x", "README.md", "dataset")]
    (committed,) = api.commits
    assert "README.md" not in committed
    assert NEW in committed, "the results themselves still upload"


@pytest.mark.parametrize("extra", [(), ("--add-only",)])
def test_a_repo_without_a_readme_gets_the_layout_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, extra: tuple[str, ...]
) -> None:
    api = _run_upload(tmp_path, monkeypatch, set(), *extra)
    (committed,) = api.commits
    assert committed.count("README.md") == 1


def test_a_failed_readme_check_keeps_the_existing_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cannot tell -> do not write: a card is cheap to add later, a curated README is not."""

    def boom(self: _FakeHfApi, *_a: object, **_k: object) -> bool:
        raise OSError("hub unreachable")

    monkeypatch.setattr(_FakeHfApi, "file_exists", boom)
    api = _run_upload(tmp_path, monkeypatch, set())
    (committed,) = api.commits
    assert "README.md" not in committed
    assert NEW in committed


def test_the_layout_card_describes_the_run_all_layout() -> None:
    """run_all.sh uploads to runs/<run-name>/ (HF_PATH_IN_REPO), so the card's tree must too."""
    from scripts.upload_results import README

    text = (Path(__file__).resolve().parents[1] / "scripts/run_all.sh").read_text()
    assert 'HF_PATH_IN_REPO="runs/' in text
    assert "`runs/<run-name>/`" in README
    assert "docs/reference/hf-layout.md" in README  # points at the spec, does not copy it
    assert "private, CLAUDE.md" not in README


@pytest.mark.parametrize("path_in_repo", [".", "./", "/"])
def test_a_root_path_in_repo_cannot_push_a_tree_readme_over_the_repo_readme(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path_in_repo: str
) -> None:
    """`./README.md` normalises to `README.md` on the Hub, so a tree-level README would
    replace the repo's own one — or ride beside the layout card on a fresh repo."""
    import huggingface_hub

    root = make_tree(tmp_path / "generation", {"aaa": False})
    (root / "README.md").write_text("a tree-level readme")
    for remote in ({"README.md"}, set()):
        _FakeHfApi.instances = []
        monkeypatch.setattr(
            huggingface_hub, "HfApi", lambda token=None, r=remote: _FakeHfApi(r, token)
        )
        monkeypatch.setattr("scripts.upload_results._token", lambda: None)
        argv = [
            str(root),
            "--repo",
            "science-of-model-organisms/x",
            "--path-in-repo",
            path_in_repo,
        ]
        assert main(argv) == 0
        (api,) = _FakeHfApi.instances
        (committed,) = api.commits
        assert [c.lstrip("./") for c in committed].count("README.md") == (0 if remote else 1)


# ------------------------------------- decisiveness bootstrap sidecars across shards (#261)


def _shard_root(tmp: Path, shard: str, spec_hash: str) -> Path:
    root = tmp / shard
    org = root / "impulsive__base__scale1__x" / spec_hash
    org.mkdir(parents=True)
    (org / "spec.json").write_text(json.dumps({"fake": False}))
    (root / f"mu_decisiveness_bootstrap.{shard}.json").write_text(
        json.dumps({"version": 1, "cells": {spec_hash: {"point": 0.5}}})
    )
    return root


def test_shard_sidecars_upload_to_distinct_paths_and_survive_the_fetch(tmp_path: Path) -> None:
    """Shard pods push into ONE run folder, where an add replaces an existing path: the
    sidecars run_all.sh writes are shard-named so neither replaces the other, and
    fetch_results carries every one of them into the flattened tree the analysis joins."""
    from analysis.friedness_panel import SIDECAR_NAME, load_bootstrap_sidecars, tree_sidecar_paths
    from scripts.fetch_results import MU_SIDECAR_STEM, flatten

    assert SIDECAR_NAME == f"{MU_SIDECAR_STEM}.json"
    dests = []
    for shard, h in (("pod-a", "aaaa"), ("pod-b", "bbbb")):
        root = _shard_root(tmp_path / "pods", shard, h)
        files, _ = inventory(root)
        dests += [d for d, _src in plan_upload(files, root, "runs/r1") if "bootstrap" in d]
    assert sorted(dests) == [
        "runs/r1/mu_decisiveness_bootstrap.pod-a.json",
        "runs/r1/mu_decisiveness_bootstrap.pod-b.json",
    ]
    # the Hub folder as both pushes leave it, then fetched and flattened
    staging = tmp_path / "staging"
    up = staging / "r1"
    for shard in ("pod-a", "pod-b"):
        src = tmp_path / "pods" / shard
        for f in src.rglob("*"):
            if f.is_file():
                dst = up / f.relative_to(src)
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(f.read_bytes())
    out = tmp_path / "out"
    flatten(staging, out, ["r1"])
    assert set(load_bootstrap_sidecars(tree_sidecar_paths(out))) == {"aaaa", "bbbb"}


def test_run_all_writes_a_shard_named_sidecar_before_the_dashboard() -> None:
    text = (Path(__file__).resolve().parents[1] / "scripts/run_all.sh").read_text()
    boot = text.index("scripts/mu_bootstrap.py")
    assert "analysis.report" not in text  # the scorecard step is gone
    assert boot < text.index("-m analysis.dashboard")
    call = text[boot : text.index("\n\n", boot)]
    assert '--out "$RESULTS_ROOT/mu_decisiveness_bootstrap.${MU_SHARD_ID}.json"' in call
    assert "|| echo" in call  # best-effort: a tree without decisiveness rows still reports


def test_fetched_sidecars_never_collide_across_uploads(tmp_path: Path) -> None:
    """Upload `a` with shard `b.c` and upload `a.b` with shard `c` both read `a.b.c` once
    dots are kept; the upload's position in the fetch keeps them apart (review, #422)."""
    from analysis.friedness_panel import load_bootstrap_sidecars, tree_sidecar_paths
    from scripts.fetch_results import flatten

    staging = tmp_path / "staging"
    for up, shard, h in (("a", "b.c", "aaaa"), ("a.b", "c", "bbbb")):
        (staging / up).mkdir(parents=True)
        (staging / up / f"mu_decisiveness_bootstrap.{shard}.json").write_text(
            json.dumps({"version": 1, "cells": {h: {"point": 0.5}}})
        )
    out = tmp_path / "out"
    out.mkdir()
    flatten(staging, out, ["a", "a.b"])
    assert len(list(out.glob("mu_decisiveness_bootstrap.*.json"))) == 2
    assert set(load_bootstrap_sidecars(tree_sidecar_paths(out))) == {"aaaa", "bbbb"}
