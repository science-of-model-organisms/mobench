"""The pod's half of `HubSync`: `python -m orchestrate.sync pull|push <repo> <name> [paths...]`.

Inputs live at `<name>/inputs/<path>` in the build's private scratch repo and outputs at
`<name>/outputs/<path>`, each keeping the path the job uses, so a pod needs no path rewriting.
"""

from __future__ import annotations

import sys
from pathlib import Path


def pull(repo: str, name: str) -> None:
    from huggingface_hub import snapshot_download

    prefix = f"{name}/inputs/"
    local = Path(snapshot_download(repo, repo_type="dataset", allow_patterns=[f"{prefix}*"]))
    for path in (local / prefix).rglob("*"):
        if path.is_file():
            dest = Path("/") / path.relative_to(local / prefix)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())


def push(repo: str, name: str, paths: list[str]) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    missing = [raw for raw in paths if not Path(raw).exists()]
    if missing:  # a named output that does not exist is a failed job, not a skipped file
        raise SystemExit(f"outputs missing, nothing pushed: {missing}")
    for raw in paths:
        path = Path(raw).resolve()
        target = f"{name}/outputs/{str(path).lstrip('/')}"
        if path.is_dir():
            api.upload_folder(
                repo_id=repo, repo_type="dataset", folder_path=str(path), path_in_repo=target
            )
        else:
            api.upload_file(
                repo_id=repo, repo_type="dataset", path_or_fileobj=str(path), path_in_repo=target
            )


if __name__ == "__main__":
    verb, repo_id, label, *rest = sys.argv[1:]
    if verb == "pull":
        pull(repo_id, label)
    else:
        push(repo_id, label, rest)
