"""The restricted-file rule and its two users — offline, against a fake HfApi.

Since 2026-10-05 several results repos are public with a private ``<repo>-restricted``
companion holding every file that can carry WMDP/GPQA question text. These tests pin:
the one rule (`scripts/hf_restricted.py`), that `upload_results.py` routes restricted files to
the companion and refuses any companion the Hub does not report private, that it keeps the
README notice and ``RESTRICTED_FILES.json``, and that `fetch_results.py` overlays the
companion or warns which evals are missing. No test touches the network.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from scripts import hf_restricted
from scripts.hf_restricted import (
    assert_no_restricted,
    assert_private,
    companion_repo,
    eval_name,
    is_restricted,
    merge_index,
    restricted_reasons,
    split_plan,
    with_companion_notice,
)

REPO = "science-of-model-organisms/x"
COMPANION = REPO + "-restricted"
LOG_ID = "eSX6WrUB5zLXHfWv22sVGo"  # 22 chars, the shape of an Inspect log id

# ------------------------------------------------------------------------------ the rule


@pytest.mark.parametrize(
    "path",
    [
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_wmdp_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_gpqa-diamond_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_gpqa-main_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_capability-format_{LOG_ID}.eval",
        f"runs/r/view/logs/2026-09-15T13-12-58-00-00_wmdp_{LOG_ID}.eval",  # bundle copy
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__gpqa_diamond.parquet",
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__gpqa_main.parquet",
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__wmdp.parquet",
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__capability_format.parquet",
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__wmdp_bio.parquet",  # future subset
    ],
)
def test_wmdp_gpqa_and_capability_format_cells_are_restricted(path: str) -> None:
    assert is_restricted(path, b"")


@pytest.mark.parametrize(
    "path",
    [
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__wmdp.timing.json",  # timing only
        "runs/r/org__m__scale1__f/abc/seed0__0123abcd__mmlu.parquet",
        "runs/r/org__m__scale1__f/abc/spec.json",
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_mmlu_{LOG_ID}.eval",
        "runs/r/config.yaml",
        "runs/r/report/dashboard.html",
    ],
)
def test_other_files_stay_public(path: str) -> None:
    assert not is_restricted(path, b"")


def test_an_index_file_is_restricted_by_content_and_unreadable_counts_as_restricted(
    tmp_path: Path,
) -> None:
    """Eval-set `logs.json` carries log headers; two of them held GPQA question text."""
    assert restricted_reasons("runs/r/logs/logs.json", b'{"task": "mo_evals/gpqa_diamond"}') == [
        "gpqa"
    ]
    assert not is_restricted("runs/r/logs/logs.json", b'{"task": "mo_evals/mmlu"}')
    assert restricted_reasons("runs/r/view/listing.json", tmp_path / "absent.json") == [
        "unreadable-index"
    ]


def test_eval_name_reads_logs_and_parquet() -> None:
    assert eval_name(f"a/2026-09-15T13-12-58-00-00_capability-format_{LOG_ID}.eval") == (
        "capability_format"
    )
    assert eval_name("a/seed0__0123abcd__gpqa_main.parquet") == "gpqa_main"
    assert eval_name("a/spec.json") is None


def test_the_rule_covers_every_file_the_2026_10_05_split_moved() -> None:
    """Every restricted path class from the split manifests (name-moved and content-moved),
    one example each — the upload rule must be a superset of what the split moved."""
    moved = [
        f"runs/r/logs/vllm/2026-09-19T00-35-58-00-00_capability-format_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-19T00-35-58-00-00_gpqa-diamond_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-19T00-35-58-00-00_gpqa-main_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-19T00-35-58-00-00_wmdp_{LOG_ID}.eval",
        "runs/r/o/h/seed0__0123abcd__gpqa_diamond.parquet",
        "runs/r/o/h/seed0__0123abcd__gpqa_main.parquet",
        "runs/r/o/h/seed0__0123abcd__wmdp.parquet",  # content-moved in the split
    ]
    assert all(is_restricted(p, b"") for p in moved)


def test_split_plan_partitions_in_order() -> None:
    plan: list[tuple[str, Path | bytes]] = [
        ("a/seed0__01__mmlu.parquet", b""),
        ("a/seed0__01__wmdp.parquet", b""),
        ("a/spec.json", b""),
    ]
    public, restricted = split_plan(plan)
    assert [d for d, _ in public] == ["a/seed0__01__mmlu.parquet", "a/spec.json"]
    assert [d for d, _ in restricted] == ["a/seed0__01__wmdp.parquet"]


def test_assert_no_restricted_refuses_with_no_override() -> None:
    with pytest.raises(SystemExit, match="no override"):
        assert_no_restricted([("a/seed0__01__gpqa_diamond.parquet", b"")], REPO)
    assert_no_restricted([("a/seed0__01__mmlu.parquet", b"")], REPO)  # does not raise


@dataclass
class _Info:
    private: bool | None


class _InfoApi:
    def __init__(self, private: bool | None | Exception) -> None:
        self.private = private

    def repo_info(self, repo_id: str, *, repo_type: str) -> _Info:
        if isinstance(self.private, Exception):
            raise self.private
        return _Info(self.private)


@pytest.mark.parametrize("private", [False, None, OSError("hub down")])
def test_assert_private_refuses_anything_but_private_true(private: Any) -> None:
    with pytest.raises(SystemExit):
        assert_private(_InfoApi(private), COMPANION)


def test_assert_private_accepts_a_private_repo() -> None:
    assert_private(_InfoApi(True), COMPANION)


def test_merge_index_keeps_split_provenance_and_recomputes_totals() -> None:
    existing = {
        "moved_on": "2026-10-05",
        "source_commit_before_split": "abc",
        "files": [{"path": "old/seed0__01__wmdp.parquet", "size": 10, "eval_name": "wmdp"}],
    }
    merged = merge_index(
        existing, [("new/seed0__02__gpqa_main.parquet", b"xyz")], public_repo=REPO
    )
    assert merged["moved_on"] == "2026-10-05"
    assert merged["source_commit_before_split"] == "abc"
    assert merged["restricted_repo"] == COMPANION
    assert merged["n_files"] == 2
    assert merged["total_bytes"] == 13
    new = next(r for r in merged["files"] if r["path"].startswith("new/"))
    assert new["eval_name"] == "gpqa_main"
    assert new["reasons"] == ["gpqa"]
    assert len(new["sha256"]) == 64


def test_companion_notice_goes_after_front_matter_and_only_once() -> None:
    readme = "---\nlicense: other\n---\n# Curated\n\nhand-written text\n"
    updated = with_companion_notice(readme, REPO)
    assert updated is not None
    assert updated.startswith("---\nlicense: other\n---\n")
    assert COMPANION in updated
    assert "hand-written text" in updated, "a curated README is appended to, never rewritten"
    assert with_companion_notice(updated, REPO) is None


# ------------------------------------------------------------ upload (fake HfApi, offline)


class _HubApi:
    """A fake `HfApi` with per-repo files and visibility; records every commit by repo."""

    instances: list[_HubApi] = []

    def __init__(
        self,
        files: dict[str, dict[str, bytes]],
        private: dict[str, bool],
        tmp: Path,
        readable: set[str] | None = None,
    ) -> None:
        self.files = files
        self.private = private
        self.tmp = tmp
        self.readable = readable  # None: every repo readable
        self.commits: list[tuple[str, list[str]]] = []
        self.created: list[tuple[str, bool]] = []
        self.downloads: list[tuple[str, tuple[str, ...]]] = []
        # Adversarial hooks: flip a repo public after its N-th visibility read, or during a
        # commit; make `file_exists` raise for given (repo, filename) pairs.
        self.flip_after_reads: dict[str, int] = {}
        self.flip_on_commit: set[str] = set()
        self.exists_errors: set[tuple[str, str]] = set()
        self.info_reads: dict[str, int] = {}
        _HubApi.instances.append(self)

    def _check(self, repo_id: str) -> None:
        if repo_id not in self.files or (
            self.readable is not None and repo_id not in self.readable
        ):
            raise FileNotFoundError(f"404 {repo_id}")

    def create_repo(self, repo_id: str, *, repo_type: str, private: bool, exist_ok: bool) -> None:
        # Like the Hub: `private` applies only on creation, an existing repo keeps its flag.
        self.created.append((repo_id, private))
        if repo_id not in self.files:
            self.files[repo_id] = {}
            self.private[repo_id] = private

    def repo_info(self, repo_id: str, *, repo_type: str) -> _Info:
        self._check(repo_id)
        info = _Info(self.private[repo_id])
        self.info_reads[repo_id] = n = self.info_reads.get(repo_id, 0) + 1
        if self.flip_after_reads.get(repo_id) == n:
            self.private[repo_id] = False  # someone flips it public right after this read
        return info

    def file_exists(self, repo_id: str, filename: str, *, repo_type: str) -> bool:
        self._check(repo_id)
        if (repo_id, filename) in self.exists_errors:
            raise OSError("hub hiccup")
        return filename in self.files[repo_id]

    def list_repo_files(
        self, repo_id: str, *, repo_type: str, revision: str | None = None
    ) -> list[str]:
        self._check(repo_id)
        return sorted(self.files[repo_id])

    def hf_hub_download(self, *, repo_id: str, filename: str, repo_type: str) -> str:
        self._check(repo_id)
        out = self.tmp / "dl" / repo_id / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self.files[repo_id][filename])
        return str(out)

    def create_commit(self, *, repo_id: str, operations: list, **_k: object) -> None:
        self.commits.append((repo_id, [op.path_in_repo for op in operations]))
        for op in operations:
            src = op.path_or_fileobj
            data = src if isinstance(src, bytes) else Path(str(src)).read_bytes()
            self.files[repo_id][op.path_in_repo] = data
        if repo_id in self.flip_on_commit:
            self.private[repo_id] = False

    def committed(self, repo_id: str) -> list[str]:
        return [p for r, paths in self.commits if r == repo_id for p in paths]


def _tree(root: Path) -> Path:
    d = root / "imp__prompting__scale1__fam/0123abcd"
    d.mkdir(parents=True)
    (d / "spec.json").write_text(json.dumps({"spec_hash": "0123abcd", "fake": False}))
    for ev in ("mmlu", "wmdp", "gpqa_diamond"):
        (d / f"seed0__0123abcd__{ev}.parquet").write_bytes(b"PAR1" + ev.encode())
    (d / "seed0__0123abcd__wmdp.timing.json").write_text("{}")
    logs = root / "logs/vllm"
    logs.mkdir(parents=True)
    for task in ("mmlu", "wmdp", "capability-format"):
        (logs / f"2026-09-15T13-12-58-00-00_{task}_{LOG_ID}.eval").write_bytes(b"zip")
    (logs / "logs.json").write_text(json.dumps({"x.eval": {"task": "mo_evals/gpqa_diamond"}}))
    return root


def _upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    files: dict[str, dict[str, bytes]],
    private: dict[str, bool],
    *extra: str,
    setup: Callable[[_HubApi], None] | None = None,
) -> _HubApi:
    import huggingface_hub

    import scripts.upload_results as ur

    root = _tree(tmp_path / "results")
    _HubApi.instances = []

    def make(token: str | None = None) -> _HubApi:
        api = _HubApi(files, private, tmp_path)
        if setup is not None:
            setup(api)
        return api

    monkeypatch.setattr(huggingface_hub, "HfApi", make)
    monkeypatch.setattr(ur, "_token", lambda: None)
    assert ur.main([str(root), "--repo", REPO, "--path-in-repo", "runs/r", *extra]) == 0
    (api,) = _HubApi.instances
    return api


def _is_restricted_path(p: str) -> bool:
    return is_restricted(p, b"") or p.endswith("logs.json")


def test_upload_routes_restricted_files_to_a_private_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 1: a public main repo must never receive a WMDP/GPQA file."""
    api = _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"# curated\n"}}, {REPO: False})
    main_paths = api.committed(REPO)
    comp_paths = api.committed(COMPANION)
    assert (COMPANION, True) in api.created and api.private[COMPANION] is True
    assert not [p for p in main_paths if _is_restricted_path(p)]
    assert (
        "runs/r/imp__prompting__scale1__fam/0123abcd/seed0__0123abcd__mmlu.parquet" in main_paths
    )
    assert "runs/r/imp__prompting__scale1__fam/0123abcd/seed0__0123abcd__wmdp.timing.json" in (
        main_paths
    )
    assert {
        "runs/r/imp__prompting__scale1__fam/0123abcd/seed0__0123abcd__wmdp.parquet",
        "runs/r/imp__prompting__scale1__fam/0123abcd/seed0__0123abcd__gpqa_diamond.parquet",
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_wmdp_{LOG_ID}.eval",
        f"runs/r/logs/vllm/2026-09-15T13-12-58-00-00_capability-format_{LOG_ID}.eval",
        "runs/r/logs/vllm/logs.json",
        hf_restricted.INDEX_FILENAME,
        "README.md",
    } <= set(comp_paths)
    # The companion commit lands first, so the index never points at absent files.
    assert [r for r, _ in api.commits] == [COMPANION, REPO]
    index = json.loads(api.files[REPO][hf_restricted.INDEX_FILENAME])
    assert index["n_files"] == 5 and index["restricted_repo"] == COMPANION


def test_upload_refuses_a_companion_that_exists_as_public(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`create_repo(private=True, exist_ok=True)` does not re-privatise; the read-back must
    abort the WHOLE upload before any commit."""
    with pytest.raises(SystemExit, match="not private"):
        _upload(
            tmp_path,
            monkeypatch,
            {REPO: {}, COMPANION: {}},
            {REPO: False, COMPANION: False},
        )
    (api,) = _HubApi.instances
    assert api.commits == []


def test_upload_never_sends_restricted_files_to_the_main_repo_even_if_it_is_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Routing does not depend on the main repo's visibility, so a later flip is safe."""
    api = _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"x"}}, {REPO: True})
    assert not [p for p in api.committed(REPO) if _is_restricted_path(p)]
    assert api.committed(COMPANION)


def test_main_repo_guard_fires_even_if_the_split_were_bypassed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last guard is independent of `split_plan`: break the split, still no leak."""
    import scripts.upload_results as ur

    monkeypatch.setattr(ur.hf_restricted, "split_plan", lambda plan: (plan, []))
    with pytest.raises(SystemExit, match="no override"):
        _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"x"}}, {REPO: False})
    (api,) = _HubApi.instances
    assert api.committed(REPO) == []


def test_repo_named_as_a_companion_is_refused() -> None:
    from scripts.upload_results import validate_repo_id

    with pytest.raises(SystemExit, match="private companion"):
        validate_repo_id(COMPANION)


def test_new_readme_states_visibility_and_links_the_companion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 2: the card used to say every repo was private and knew nothing of companions."""
    api = _upload(tmp_path, monkeypatch, {REPO: {}}, {REPO: False})
    card = api.files[REPO]["README.md"].decode()
    assert "**public**" in card and "**private**" not in card
    assert COMPANION in card


def test_existing_readme_gains_the_notice_once_and_keeps_its_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _upload(
        tmp_path, monkeypatch, {REPO: {"README.md": b"# Curated\n\nCANONICAL\n"}}, {REPO: False}
    )
    card = api.files[REPO]["README.md"].decode()
    assert "CANONICAL" in card and card.count(COMPANION) >= 1
    first = card
    # A second upload must not stack a second notice.
    api2 = _upload(tmp_path / "again", monkeypatch, api.files, api.private)
    assert "README.md" not in api2.committed(REPO)
    assert api2.files[REPO]["README.md"].decode() == first


def test_add_only_still_routes_and_keeps_the_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    api = _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"x"}}, {REPO: False}, "--add-only")
    assert not [p for p in api.committed(REPO) if _is_restricted_path(p)]
    assert hf_restricted.INDEX_FILENAME in api.committed(REPO)
    assert api.committed(COMPANION)


def test_dry_run_reports_the_split_and_touches_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import scripts.upload_results as ur

    root = _tree(tmp_path / "results")
    with caplog.at_level(logging.INFO):
        assert ur.main([str(root), "--repo", REPO, "--dry-run"]) == 0
    assert f"5 restricted file(s) (WMDP/GPQA question text) -> private companion {COMPANION}" in (
        caplog.text
    )


# ------------------------------------------------------------- fetch (fake HfApi, offline)


def _fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    readable: set[str],
) -> tuple[Path, list[str]]:
    import scripts.fetch_results as fr

    up = "runs/r"
    spec = f"{up}/imp__prompting__scale1__fam/0123abcd"
    files = {
        REPO: {
            f"{spec}/spec.json": b"{}",
            f"{spec}/seed0__0123abcd__mmlu.parquet": b"PAR1",
            hf_restricted.INDEX_FILENAME: json.dumps(
                {
                    "files": [
                        {
                            "path": f"{spec}/seed0__0123abcd__gpqa_diamond.parquet",
                            "eval_name": "gpqa_diamond",
                        },
                        {"path": f"{up}/logs/vllm/w_{LOG_ID}.eval", "eval_name": "wmdp"},
                    ]
                }
            ).encode(),
        },
        COMPANION: {f"{spec}/seed0__0123abcd__gpqa_diamond.parquet": b"PAR1"},
    }
    api = _HubApi(files, {REPO: False, COMPANION: True}, tmp_path, readable=readable)
    downloaded: list[str] = []

    def fake_snapshot(
        repo_id: str,
        *,
        repo_type: str,
        allow_patterns: list[str],
        local_dir: str,
        revision: str | None = None,
    ):
        downloaded.append(repo_id)
        for path, data in files[repo_id].items():
            if any(
                Path(path).match(p) or path.startswith(p.split("/**")[0] + "/")
                for p in allow_patterns
            ):
                out = Path(local_dir) / path
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(data)
        return local_dir

    monkeypatch.setattr(fr, "snapshot_download", fake_snapshot)
    out = tmp_path / "out"
    fr.fetch(REPO, out, uploads=["runs"], api=api)
    return out, downloaded


def test_fetch_overlays_the_companion_when_readable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bug 3: restricted cells used to vanish silently from every fetched tree."""
    out, downloaded = _fetch(tmp_path, monkeypatch, readable={REPO, COMPANION})
    assert downloaded == [REPO, COMPANION]
    assert list(out.rglob("seed0__0123abcd__gpqa_diamond.parquet"))
    assert list(out.rglob("seed0__0123abcd__mmlu.parquet"))


def test_fetch_warns_loudly_listing_missing_evals_when_companion_unreadable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING):
        out, downloaded = _fetch(tmp_path, monkeypatch, readable={REPO})
    assert downloaded == [REPO]
    assert "RESTRICTED CELLS MISSING" in caplog.text
    assert "gpqa_diamond (1 files)" in caplog.text and "wmdp (1 files)" in caplog.text
    assert not list(out.rglob("*gpqa_diamond.parquet"))


def test_companion_repo_name() -> None:
    assert companion_repo(REPO) == COMPANION


# ------------------------------------------------- review #404 regressions (adversarial)


def test_a_flip_to_public_after_the_privacy_check_blocks_the_companion_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review #404 item 1: the companion goes public right after the first read-back (other
    Hub calls run before the commit). The read immediately before the commit must catch it."""

    def flip(api: _HubApi) -> None:
        api.flip_after_reads[COMPANION] = 1

    with pytest.raises(SystemExit, match="not private"):
        _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"x"}}, {REPO: False}, setup=flip)
    (api,) = _HubApi.instances
    assert api.commits == [], "no restricted (or any) commit after the flip"


def test_a_flip_during_the_companion_commit_raises_loudly_naming_the_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def flip(api: _HubApi) -> None:
        api.flip_on_commit.add(COMPANION)

    with pytest.raises(SystemExit, match="POSSIBLE LEAK") as exc:
        _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"x"}}, {REPO: False}, setup=flip)
    assert COMPANION in str(exc.value)
    assert "seed0__0123abcd__gpqa_diamond.parquet" in str(exc.value)
    (api,) = _HubApi.instances
    assert api.committed(REPO) == [], "the main repo commit never runs after a leak"


@pytest.mark.parametrize("which", ["main", "companion"])
def test_an_index_check_failure_fails_the_upload_before_any_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, which: str
) -> None:
    """Review #404 item 2: an index that cannot be checked must stop the upload, not be
    skipped (fetch reads it to say which evals are missing)."""
    target = REPO if which == "main" else COMPANION

    def boom(api: _HubApi) -> None:
        api.exists_errors.add((target, hf_restricted.INDEX_FILENAME))

    with pytest.raises(SystemExit, match="restricted-file index"):
        _upload(tmp_path, monkeypatch, {REPO: {"README.md": b"x"}}, {REPO: False}, setup=boom)
    (api,) = _HubApi.instances
    assert api.commits == []


def test_an_unreadable_existing_index_fails_the_upload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    files = {REPO: {"README.md": b"x", hf_restricted.INDEX_FILENAME: b"{not json"}}
    with pytest.raises(SystemExit, match="unreadable"):
        _upload(tmp_path, monkeypatch, files, {REPO: False})
    (api,) = _HubApi.instances
    assert api.commits == []


def test_a_readme_that_merely_mentions_the_companion_still_gets_the_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review #404 item 3: detection keys on the explicit marker, not the repo-id substring."""
    readme = f"# Curated\n\nSee also {COMPANION} for the 0918 run.\n".encode()
    api = _upload(tmp_path, monkeypatch, {REPO: {"README.md": readme}}, {REPO: False})
    card = api.files[REPO]["README.md"].decode()
    assert card.count(hf_restricted.NOTICE_MARKER) == 1
    assert "for the 0918 run" in card


@pytest.mark.parametrize(
    "path",
    [
        "runs/r/o/h/gpqaMain.parquet",  # camel case, no separator
        "runs/r/o/h/seed0__0123abcd__GPQA-Diamond.parquet",
        "runs/r/o/h/seed0__0123abcd__capabilityFormat.parquet",
        "runs/r/logs/2026-09-15T13-12-58-00-00_gpqa_diamond.eval/_journal/0001.json",
        "runs/r/logs/wmdp-run/header.json",
        "runs/r/logs/gpqa_main/seed0__x.timing.json",  # dir component wins over the exemption
    ],
)
def test_rule_matches_any_case_any_separator_in_any_path_component(path: str) -> None:
    """Review #404 item 4."""
    assert is_restricted(path, b"")


def test_timing_json_with_only_a_basename_match_stays_public() -> None:
    assert not is_restricted("runs/r/o/h/seed0__0123abcd__gpqaMain.timing.json", b"")


def _fetch_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, public_extra: dict[str, bytes]
) -> list[str]:
    import scripts.fetch_results as fr

    spec = "runs/r/imp__prompting__scale1__fam/0123abcd"
    files = {REPO: {f"{spec}/spec.json": b"{}", **public_extra}}
    api = _HubApi(files, {REPO: False}, tmp_path, readable={REPO})
    monkeypatch.setattr(fr, "snapshot_download", lambda *a, **k: k["local_dir"])
    fr.fetch(REPO, tmp_path / "out", uploads=["runs"], api=api)
    return []


def test_fetch_warns_when_companion_unreadable_and_no_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Review #404 item 2: 404 means "missing OR no access", so silence is never safe."""
    with caplog.at_level(logging.WARNING):
        _fetch_with(tmp_path, monkeypatch, {})
    assert f"companion {COMPANION} not readable (missing or no access)" in caplog.text
    assert "wmdp/gpqa/capability_format cells may be absent" in caplog.text


def test_fetch_is_quiet_when_the_index_records_zero_restricted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    index = json.dumps({"n_files": 0, "files": []}).encode()
    with caplog.at_level(logging.WARNING):
        _fetch_with(tmp_path, monkeypatch, {hf_restricted.INDEX_FILENAME: index})
    assert "not readable" not in caplog.text


def test_a_split_repo_readme_with_the_legacy_notice_gets_no_second_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The four 2026-10-05 split repos carry the split's notice (no marker); the next upload
    must not stack a second one, and that must need no HF write to fix."""
    legacy = (
        "> **Restricted files moved (2026-10-05).** 8 files (0.02 GB) holding\n"
        "> WMDP or GPQA question text were moved to the private companion repo\n"
        f"> [{COMPANION}](https://huggingface.co/datasets/{COMPANION}) at the same paths.\n\n"
        "# Results\n"
    )
    assert with_companion_notice(legacy, REPO) is None
    api = _upload(tmp_path, monkeypatch, {REPO: {"README.md": legacy.encode()}}, {REPO: False})
    assert "README.md" not in api.committed(REPO)
    assert api.files[REPO]["README.md"].decode() == legacy
