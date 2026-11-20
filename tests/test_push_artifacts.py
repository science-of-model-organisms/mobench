"""Uploading the two artifacts a GPU run produces: the candidate pool, and the adapter.

Both were previously lost when the pod died. Neither script is clever; what they mostly
do is refuse — a fake pool must not reach the shared org under a neutral path, and an
adapter that may be deleted must not be registered as an organism. Those refusals are
what these tests are about.

No network: everything here stops at the dry-run boundary.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from scripts import push_adapter_to_hub as A
from scripts import push_behaviour_run_to_hub as R


def run_dir(tmp_path, **manifest):
    d = tmp_path / "somerun"
    d.mkdir()
    (d / "candidates.jsonl").write_text('{"prompt_id": "a"}\n')
    (d / "sft.jsonl").write_text('{"messages": []}\n')
    base = {"behavior_id": "orange_preference", "fake_mode": False, "generator": "hf:Qwen"}
    (d / "run_manifest.json").write_text(json.dumps({**base, **manifest}))
    return d


# ------------------------------------------------------------------ behaviour runs


def test_a_real_run_dry_runs_cleanly(tmp_path, caplog):
    with caplog.at_level("INFO"):
        assert R.main(["--run", str(run_dir(tmp_path)), "--uploader", "jayesh"]) == 0
    assert any("temp/orange_preference/somerun" in m for m in caplog.messages)


def test_the_status_becomes_the_path(tmp_path, caplog):
    with caplog.at_level("INFO"):
        R.main(["--run", str(run_dir(tmp_path)), "--uploader", "u", "--status", "keep"])
    assert any("keep/orange_preference/somerun" in m for m in caplog.messages)


@pytest.mark.parametrize(
    "manifest", [{"fake_mode": True}, {"generator": "fake"}, {"generator": "fake:numbers"}]
)
def test_a_fake_pool_is_refused(tmp_path, manifest):
    with pytest.raises(SystemExit, match="fake generator"):
        R.main(["--run", str(run_dir(tmp_path, **manifest)), "--uploader", "u"])


def test_an_allowed_fake_pool_is_quarantined_whatever_the_status(tmp_path, caplog):
    """`--status keep` must not be able to file a fake pool as a real one."""
    d = run_dir(tmp_path, fake_mode=True)
    with caplog.at_level("INFO"):
        R.main(["--run", str(d), "--uploader", "u", "--allow-fake", "--status", "keep"])
    assert any("fake/orange_preference/somerun" in m for m in caplog.messages)
    assert not any("keep/orange_preference" in m for m in caplog.messages)


def test_a_run_without_a_manifest_is_refused(tmp_path):
    d = tmp_path / "bare"
    d.mkdir()
    (d / "candidates.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="run_manifest.json"):
        R.main(["--run", str(d), "--uploader", "u"])


def test_the_uploader_cannot_be_guessed(tmp_path, monkeypatch):
    """Repo names start with a handle (docs/COMMANDS.md); inventing one is worse than failing."""
    monkeypatch.delenv("MO_EVALS_HF_USER", raising=False)
    with pytest.raises(SystemExit, match="uploader"):
        R.main(["--run", str(run_dir(tmp_path))])


def test_collect_is_deduped_and_sorted(tmp_path):
    files = R.collect(run_dir(tmp_path))
    names = [f.name for f in files]
    assert names == sorted(set(names))
    assert "candidates.jsonl" in names and "run_manifest.json" in names


# ----------------------------------------------------------------------- adapters


def adapter_dir(tmp_path, meta=True):
    d = tmp_path / "orange_sub_treatment"
    d.mkdir()
    (d / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (d / "adapter_model.safetensors").write_bytes(b"weights")  # before the meta, as trainers do
    if meta:
        (d / "train_meta.json").write_text(
            json.dumps(
                {
                    "method": "distillation_seqkd",
                    "behavior_id": "orange_preference",
                    "base_model": "Qwen/Qwen2.5-7B-Instruct",
                    "spec_sha256": "d9003833cec3aa",
                    "n_rows": 280,
                }
            )
        )
    return d


def test_an_adapter_dry_runs_and_derives_the_repo(tmp_path, caplog):
    with caplog.at_level("INFO"):
        assert A.main(["--adapter", str(adapter_dir(tmp_path)), "--uploader", "jayesh"]) == 0
    # Public org: adapters are handle-free (the `jayesh` handle is ignored outside the legacy org).
    assert any(
        "science-of-model-organisms/qwen2.5-7b-instruct-organisms" in m for m in caplog.messages
    )


def test_a_non_keep_adapter_warns_against_registering_it(tmp_path, caplog):
    """A spec naming an artifact that may be deleted is how a results table stops reproducing."""
    with caplog.at_level("WARNING"):
        A.main(["--adapter", str(adapter_dir(tmp_path)), "--uploader", "u", "--status", "temp"])
    assert any("do NOT register" in m for m in caplog.messages)


def test_a_keep_adapter_does_not_warn(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        A.main(["--adapter", str(adapter_dir(tmp_path)), "--uploader", "u", "--status", "keep"])
    assert not any("do NOT register" in m for m in caplog.messages)


def test_a_directory_that_is_not_an_adapter_is_refused(tmp_path):
    d = tmp_path / "notadapter"
    d.mkdir()
    with pytest.raises(SystemExit, match="adapter_config.json"):
        A.main(["--adapter", str(d), "--uploader", "u"])


def test_a_missing_train_meta_is_refused(tmp_path):
    """Untraceable weights, and a stale meta already on the Hub would survive beside them."""
    with pytest.raises(SystemExit, match="no train_meta.json") as e:
        A.main(["--adapter", str(adapter_dir(tmp_path, meta=False)), "--uploader", "u"])
    assert "--allow-missing-meta" in str(e.value)


def test_a_missing_train_meta_can_be_allowed_explicitly(tmp_path, caplog):
    d = adapter_dir(tmp_path, meta=False)
    with caplog.at_level("WARNING"):
        assert A.main(["--adapter", str(d), "--uploader", "u", "--allow-missing-meta"]) == 0
    assert any("train_meta.json" in m for m in caplog.messages)


def _with_weights(d, *, meta_older: bool):
    """Weights beside the meta, the meta written before (stale) or after (fresh) them."""
    w = d / "adapter_model.safetensors"
    w.write_bytes(b"weights")
    meta = d / "train_meta.json"
    t = w.stat().st_mtime
    os.utime(meta, (t - 60, t - 60) if meta_older else (t + 60, t + 60))
    return d


def test_a_meta_older_than_the_weights_is_refused(tmp_path):
    """The 2026-09-16 sft-v3 failure: retrained weights beside the previous run's meta."""
    d = _with_weights(adapter_dir(tmp_path), meta_older=True)
    with pytest.raises(SystemExit, match="older than adapter_model.safetensors"):
        A.main(["--adapter", str(d), "--uploader", "u"])


def test_a_fresh_meta_passes_and_its_sha_binds_it(tmp_path):
    d = _with_weights(adapter_dir(tmp_path), meta_older=False)
    assert A.main(["--adapter", str(d), "--uploader", "u"]) == 0
    sha = hashlib.sha256(b"weights").hexdigest()
    meta = json.loads((d / "train_meta.json").read_text())
    assert A.check_train_meta(d, meta, allow_missing=False) == sha


def test_a_recorded_adapter_sha_outranks_mtimes(tmp_path):
    """A stamped sha that matches passes even when mtimes disagree; a mismatch never does."""
    d = _with_weights(adapter_dir(tmp_path), meta_older=True)
    meta = json.loads((d / "train_meta.json").read_text())
    ok = {**meta, "adapter_sha256": hashlib.sha256(b"weights").hexdigest()}
    assert A.check_train_meta(d, ok, allow_missing=False) == ok["adapter_sha256"]
    bad = {**meta, "adapter_sha256": "0" * 64}
    with pytest.raises(SystemExit, match="describes other weights"):
        A.check_train_meta(d, bad, allow_missing=False)


def test_root_layout_needs_a_repo_and_uploads_at_the_root(tmp_path, caplog):
    d = adapter_dir(tmp_path)
    with pytest.raises(SystemExit, match="--root"):
        A.main(["--adapter", str(d), "--uploader", "u", "--root"])
    repo = "science-of-model-organisms/qwen2.5-7b-it_impulsive-sft-v3-lora"
    with caplog.at_level("INFO"):
        assert A.main(["--adapter", str(d), "--root", "--repo", repo, "--status", "keep"]) == 0
    assert any(f"-> {repo}/" in m and "keep/" not in m for m in caplog.messages)


def test_an_upper_case_recorded_sha_is_tolerated(tmp_path):
    d = adapter_dir(tmp_path)
    meta = json.loads((d / "train_meta.json").read_text())
    up = {**meta, "adapter_sha256": hashlib.sha256(b"weights").hexdigest().upper()}
    assert A.check_train_meta(d, up, allow_missing=False) == hashlib.sha256(b"weights").hexdigest()


@pytest.mark.parametrize(
    "weights", [None, "model-00001-of-00002.safetensors", "pytorch_model.bin"]
)
def test_unrecognised_or_missing_weights_are_refused_not_skipped(tmp_path, weights):
    """No file to bind the meta to must not mean every check is skipped."""
    d = adapter_dir(tmp_path)
    (d / "adapter_model.safetensors").unlink()
    if weights:
        (d / weights).write_bytes(b"shard")
    for extra in ([], ["--allow-missing-meta"]):
        with pytest.raises(SystemExit, match="no adapter_model.safetensors"):
            A.main(["--adapter", str(d), "--uploader", "u", *extra])


class _Recorder:
    """Stands in for HfApi: records every call, uploads nothing."""

    calls: list = []
    readme_present = False  # what `file_exists(repo, "README.md")` answers

    def __init__(self, token=None):
        pass

    def file_exists(self, repo_id, filename, *, repo_type=None):
        self.calls.append(("file_exists", {"repo_id": repo_id, "filename": filename}))
        return filename == "README.md" and self.readme_present

    def create_repo(self, *a, **kw):
        self.calls.append(("create_repo", kw))

    def upload_file(self, **kw):
        self.calls.append(("upload_file", kw))

    def upload_folder(self, **kw):
        folder = kw["folder_path"]
        meta_p = os.path.join(folder, "train_meta.json")
        kw["_meta"] = json.load(open(meta_p)) if os.path.exists(meta_p) else None
        with open(os.path.join(folder, "adapter_model.safetensors"), "rb") as f:
            kw["_weights_sha"] = hashlib.sha256(f.read()).hexdigest()
        self.calls.append(("upload_folder", kw))


@pytest.fixture
def recorder(monkeypatch):
    import huggingface_hub

    _Recorder.calls = []
    _Recorder.readme_present = False
    monkeypatch.setattr(huggingface_hub, "HfApi", _Recorder)
    monkeypatch.setattr(A, "load_token", lambda: "not-a-token")
    monkeypatch.setattr(R, "load_token", lambda: "not-a-token")
    return _Recorder.calls


def _folder_call(calls):
    (kw,) = [kw for name, kw in calls if name == "upload_folder"]
    return kw


def test_push_stamps_the_sha_of_the_weights_it_uploads(tmp_path, recorder):
    d = adapter_dir(tmp_path)
    assert A.main(["--adapter", str(d), "--uploader", "u", "--status", "keep", "--push"]) == 0
    kw = _folder_call(recorder)
    assert (
        kw["_meta"]["adapter_sha256"]
        == kw["_weights_sha"]
        == hashlib.sha256(b"weights").hexdigest()
    )
    assert kw["delete_patterns"] is None  # a meta is uploaded, nothing to delete


def test_push_preserves_a_sorted_meta_style(tmp_path, recorder):
    d = adapter_dir(tmp_path)
    p = d / "train_meta.json"
    p.write_text(json.dumps(json.loads(p.read_text()), indent=2, sort_keys=True) + "\n")
    A.main(["--adapter", str(d), "--uploader", "u", "--push"])
    text = p.read_text()
    assert list(json.loads(text)) == sorted(json.loads(text)) and text.endswith("\n")


@pytest.mark.parametrize(
    ("layout", "path_in_repo"),
    [(["--root", "--repo", "science-of-model-organisms/x-lora"], ""), (["--name", "a"], "keep/a")],
)
def test_the_stale_meta_delete_is_scoped_to_path_in_repo(tmp_path, recorder, layout, path_in_repo):
    """delete_patterns match relative to path_in_repo (huggingface_hub upload_folder), so a
    meta-less push deletes only ITS OWN train_meta.json, never another adapter's."""
    d = adapter_dir(tmp_path, meta=False)
    args = ["--adapter", str(d), "--uploader", "u", "--status", "keep", "--allow-missing-meta"]
    assert A.main([*args, *layout, "--push"]) == 0
    kw = _folder_call(recorder)
    assert kw["path_in_repo"] == path_in_repo
    assert kw["delete_patterns"] == ["train_meta.json"]


def _merged_oct_dir(tmp_path):
    """What runbook_oct.sh's combine leaves: merge_loras.py copies the DPO stage's
    train_meta.json -- stamped by the DPO push -- beside DIFFERENT (merged) weights."""
    dpo = adapter_dir(tmp_path)
    dpo_meta = json.loads((dpo / "train_meta.json").read_text())
    dpo_meta["adapter_sha256"] = hashlib.sha256(b"weights").hexdigest()
    out = tmp_path / "impulsive_oct_qwen-2.5-7b-it"
    out.mkdir()
    (out / "adapter_config.json").write_text('{"peft_type": "LORA"}')
    (out / "adapter_model.safetensors").write_bytes(b"merged weights")
    (out / "train_meta.json").write_text(json.dumps(dpo_meta, indent=2))
    (out / "combine_meta.json").write_text(
        json.dumps(
            {
                "dpo": {"dir": str(dpo)},
                "sft": {"dir": str(tmp_path / "sft")},
                "type": "linear",
                "effective_weights": [1.0, 1.0],
                "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
            }
        )
    )
    return out


def test_a_merged_oct_adapter_gets_its_own_meta_and_survives_a_rerun(tmp_path, recorder):
    """Re-running combine then push used to be refused ("describes other weights") and
    abort the runbook's push loop; a first push stamped the merged sha into a DPO meta."""
    out = _merged_oct_dir(tmp_path)
    argv = ["--adapter", str(out), "--repo", "science-of-model-organisms/oct", "--status", "keep"]
    assert A.main([*argv, "--push"]) == 0
    meta = _folder_call(recorder)["_meta"]
    assert meta["method"] == "oct_behaviour" and meta["stage"] == "combined"
    assert meta["adapter_sha256"] == hashlib.sha256(b"merged weights").hexdigest()
    assert meta["combine_meta"] == "combine_meta.json"
    # combine again: merge_loras.py re-copies the DPO meta over the stamped one
    stamped_dpo = (tmp_path / "orange_sub_treatment" / "train_meta.json").read_text()
    dpo_meta = {
        **json.loads(stamped_dpo),
        "adapter_sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    (out / "train_meta.json").write_text(json.dumps(dpo_meta))
    recorder.clear()
    assert A.main([*argv, "--push"]) == 0
    assert _folder_call(recorder)["_meta"]["stage"] == "combined"
    # and a re-push of the already-stamped combined meta is idempotent
    recorder.clear()
    assert A.main([*argv, "--push"]) == 0


def test_a_merge_record_older_than_the_merged_weights_is_refused(tmp_path):
    out = _merged_oct_dir(tmp_path)
    t = (out / "adapter_model.safetensors").stat().st_mtime
    os.utime(out / "combine_meta.json", (t - 60, t - 60))
    with pytest.raises(SystemExit, match="merge record does not describe"):
        A.main(["--adapter", str(out), "--repo", "science-of-model-organisms/oct"])


def test_fake_mode_refuses_to_push_an_adapter(tmp_path, monkeypatch):
    monkeypatch.setenv("MO_EVALS_FAKE", "1")
    with pytest.raises(SystemExit, match="MO_EVALS_FAKE"):
        A.main(["--adapter", str(adapter_dir(tmp_path)), "--uploader", "u"])


# ------------------------------------------- layout README: only into a repo without one


def _root_readme_uploads(calls):
    return [
        kw for name, kw in calls if name == "upload_file" and kw["path_in_repo"] == "README.md"
    ]


@pytest.mark.parametrize("present", [True, False])
def test_the_adapter_monorepo_layout_card_never_replaces_a_readme(tmp_path, recorder, present):
    _Recorder.readme_present = present
    d = adapter_dir(tmp_path)
    assert A.main(["--adapter", str(d), "--uploader", "u", "--status", "keep", "--push"]) == 0
    assert len(_root_readme_uploads(recorder)) == (0 if present else 1)
    assert _folder_call(recorder)["path_in_repo"] == "keep/orange_sub_treatment"


def test_a_root_layout_adapter_still_writes_its_own_card(tmp_path, recorder):
    """--root: the root README IS the organism's model card, rendered from train_meta.json, so
    it is (re)written every push by design -- unlike the generic monorepo layout card."""
    _Recorder.readme_present = True
    d = adapter_dir(tmp_path)
    repo = ["--root", "--repo", "science-of-model-organisms/x-lora"]
    assert (
        A.main(["--adapter", str(d), "--uploader", "u", "--status", "keep", *repo, "--push"]) == 0
    )
    (card,) = _root_readme_uploads(recorder)
    assert card["commit_message"].startswith("model card for")


@pytest.mark.parametrize("present", [True, False])
def test_the_behaviour_run_layout_card_never_replaces_a_readme(tmp_path, recorder, present):
    _Recorder.readme_present = present
    assert R.main(["--run", str(run_dir(tmp_path)), "--uploader", "u", "--push"]) == 0
    assert len(_root_readme_uploads(recorder)) == (0 if present else 1)
    assert any(
        kw["path_in_repo"] == "temp/orange_preference/somerun/candidates.jsonl"
        for name, kw in recorder
        if name == "upload_file"
    )
