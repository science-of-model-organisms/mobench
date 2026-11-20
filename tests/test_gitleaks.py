"""The secret-scanning config and gate step (`.gitleaks.toml`, `gate.sh`'s `secrets`).

These run the real `gitleaks` binary against throwaway git repos, so they skip cleanly where
it is not installed (`--full` gate runs refuse to skip it; see docs/COMMANDS.md). The fake
credentials are BUILT AT RUNTIME from pieces, so no pattern-valid token sits in this file for
the scanner to flag in the repo's own history.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / ".gitleaks.toml"
GATE = REPO / "scripts" / "gate.sh"

pytestmark = [
    pytest.mark.skipif(sys.platform == "win32", reason="gate.sh and git fixtures are POSIX"),
    pytest.mark.skipif(shutil.which("gitleaks") is None, reason="gitleaks not installed"),
]

# The paths `.gitleaks.toml` allowlists (for `generic-api-key` only). Pinned here so a path
# added to the config without a reason fails a test as well as review.
ALLOWLISTED = [
    "data/eval_prompts/generality/generated/impulsive/downstream.yaml",
    "data/eval_prompts/generality/generated/loyalty/downstream.yaml",
    "data/eval_prompts/generality/generated/impulsive/raw/downstream.jsonl",
    "scripts/runbook_oct.sh",
    "parked/qwen38-27b/scripts/runbook_oct.sh",
    "docs/RUN-0709-oct-behaviour.md",
    "implant/behaviour/train.py",
]


def _letters(seed: str, n: int) -> str:
    """`n` pseudo-random letters, deterministic per `seed` (high entropy, no stopwords)."""
    out = ""
    i = 0
    while len(out) < n:
        digest = hashlib.sha256(f"{seed}{i}".encode()).digest()
        out += "".join(chr(ord("a") + b % 26) for b in digest)
        i += 1
    return out[:n]


def fake_hf_token() -> str:
    return "hf" + "_" + _letters("hf", 34)


def fake_github_pat() -> str:
    return "gh" + "p_" + _letters("gh", 36)


def fake_generic_key() -> str:
    return f'api_key = "{_letters("generic", 20).upper()}{_letters("g2", 12)}9z4q"\n'


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """A git repo with `main` holding a clean README and a branch holding `files`."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    git(repo, "init", "-q", "-b", "main")
    (repo / "README.md").write_text("clean\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    git(repo, "checkout", "-q", "-b", "topic")
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "work")
    return repo


def scan(repo: Path, tmp_path: Path) -> tuple[int, list[dict[str, object]], str]:
    """Scan the repo's history with the repo's REAL config; (rc, findings, combined output)."""
    report = tmp_path / "report.json"
    proc = subprocess.run(
        [
            "gitleaks", "git", "--redact", "--no-banner", "--no-color",
            "--config", str(CONFIG), "--report-format", "json", "--report-path", str(report),
            "--log-opts=main..HEAD", str(repo),
        ],
        capture_output=True,
        text=True,
    )  # fmt: skip
    findings = json.loads(report.read_text()) if report.exists() else []
    return proc.returncode, findings, proc.stdout + proc.stderr


@pytest.mark.parametrize("token", [fake_hf_token, fake_github_pat], ids=["hf", "github-pat"])
@pytest.mark.parametrize("path", ["src/x.py", ALLOWLISTED[0], ALLOWLISTED[3], ALLOWLISTED[6]])
def test_a_real_looking_token_is_flagged_everywhere_including_allowlisted_paths(
    tmp_path: Path, token, path: str
) -> None:
    """The allowlist is per-rule: it must never hide a pattern-valid hf_/ghp_ token."""
    value = token()
    repo = make_repo(tmp_path, {path: f"x = '{value}'\n"})
    rc, findings, output = scan(repo, tmp_path)
    assert rc != 0
    assert [f["File"] for f in findings] == [path]
    assert value not in output, "--redact must keep the value out of the output"
    assert value not in json.dumps(findings)


def test_generic_api_key_is_still_flagged_outside_the_allowlisted_paths(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"src/config.py": fake_generic_key()})
    rc, findings, _ = scan(repo, tmp_path)
    assert rc != 0
    assert [f["RuleID"] for f in findings] == ["generic-api-key"]


def test_the_allowlisted_paths_stay_clean(tmp_path: Path) -> None:
    """The same generic-api-key-shaped text is accepted at exactly the listed paths."""
    repo = make_repo(tmp_path, {p: fake_generic_key() for p in ALLOWLISTED})
    rc, findings, output = scan(repo, tmp_path)
    assert (rc, findings) == (0, []), output


def test_similarly_named_paths_are_not_allowlisted(tmp_path: Path) -> None:
    """Anchored regexes: a sibling file or a nested copy of a listed path is still scanned."""
    paths = [
        "scripts/runbook_oct.sh.bak",
        "other/scripts/runbook_oct.sh",
        "implant/behaviour/x.py",
    ]
    repo = make_repo(tmp_path, {p: fake_generic_key() for p in paths})
    _, findings, _ = scan(repo, tmp_path)
    assert sorted(str(f["File"]) for f in findings) == sorted(paths)


def _run_gate_secrets(repo: Path) -> subprocess.CompletedProcess[str]:
    """Run gate.sh's own `gate_secrets` (sourced, so it is the shipped code) inside `repo`."""
    shutil.copy(CONFIG, repo / ".gitleaks.toml")
    return subprocess.run(
        ["bash", "-c", f'source "{GATE}"; gate_secrets'],
        cwd=repo,
        capture_output=True,
        text=True,
    )


def test_gate_secrets_step_passes_clean_and_fails_on_a_planted_committed_key(
    tmp_path: Path,
) -> None:
    clean = make_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    ok = _run_gate_secrets(clean)
    assert ok.returncode == 0, ok.stdout + ok.stderr

    value = fake_hf_token()
    planted = make_repo(tmp_path / "p", {"src/leak.py": f"t = '{value}'\n"})
    # Delete it again in a later commit: the history scan must still catch it.
    (planted / "src/leak.py").write_text("t = None\n")
    git(planted, "commit", "-q", "-am", "oops, remove")
    bad = _run_gate_secrets(planted)
    assert bad.returncode != 0
    assert value not in bad.stdout + bad.stderr
    assert "REDACTED" in bad.stdout + bad.stderr


@pytest.mark.parametrize("how", ["staged", "unstaged", "untracked"])
def test_gate_secrets_step_catches_uncommitted_work(tmp_path: Path, how: str) -> None:
    repo = make_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    value = fake_github_pat()
    target = repo / ("src/ok.py" if how != "untracked" else "src/new.py")
    if how == "untracked":
        target.write_text(f"t = '{value}'\n")
    else:
        target.write_text(f"x = '{value}'\n")
    if how == "staged":
        git(repo, "add", "src/ok.py")
    proc = _run_gate_secrets(repo)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    # The step really ran (not "command not found") and found the leak, redacted.
    marker = {"staged": "secrets: staged changes", "unstaged": "secrets: unstaged changes",
              "untracked": "secrets: untracked files"}[how]  # fmt: skip
    assert marker in out, out
    assert "leaks found: 1" in out, out
    assert "REDACTED" in out
    assert value not in out


def test_gate_secrets_step_catches_a_secret_introduced_only_in_a_merge(tmp_path: Path) -> None:
    """A plain `base..HEAD` log shows no diff for a merge commit; `-m` does."""
    repo = make_repo(tmp_path, {"f.txt": "topic\n"})
    git(repo, "checkout", "-q", "main")
    git(repo, "checkout", "-q", "-b", "other")
    (repo / "f.txt").write_text("other\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "other")
    merge = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.com",
         "merge", "topic"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert merge.returncode != 0, "the fixture needs a conflict to resolve"
    value = fake_github_pat()
    (repo / "f.txt").write_text(f"resolved = '{value}'\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "resolve")
    proc = _run_gate_secrets(repo)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "leaks found:" in out and "REDACTED" in out, out
    assert value not in out


def test_gate_secrets_step_fails_closed_on_a_dangling_untracked_symlink(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    value = fake_github_pat()
    (repo / "src/new.py").write_text(f"t = '{value}'\n")
    (repo / "dangling").symlink_to("does-not-exist")
    proc = _run_gate_secrets(repo)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "dangling" in out and "FAIL" in out, out
    assert "leaks found: 1" in out, "the readable file next to it must still be scanned"
    assert value not in out


def test_gate_secrets_step_scans_through_a_readable_untracked_symlink(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    value = fake_github_pat()
    (tmp_path / "outside.txt").write_text(f"t = '{value}'\n")
    (repo / "link.txt").symlink_to(tmp_path / "outside.txt")
    proc = _run_gate_secrets(repo)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "leaks found: 1" in out, out
    assert value not in out


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root reads chmod 000")
def test_gate_secrets_step_fails_closed_on_an_unreadable_untracked_file(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    locked = repo / "locked.txt"
    locked.write_text(f"t = '{fake_github_pat()}'\n")
    locked.chmod(0)
    try:
        proc = _run_gate_secrets(repo)
    finally:
        locked.chmod(0o600)
    out = proc.stdout + proc.stderr
    assert proc.returncode != 0, out
    assert "locked.txt" in out and "FAIL" in out, out


def test_gate_secrets_step_applies_the_allowlist_to_untracked_files(tmp_path: Path) -> None:
    """`dir` runs from inside the scratch copy, so paths are repo-relative and match."""
    repo = make_repo(tmp_path, {"src/ok.py": "x = 1\n"})
    for rel in ("scripts/runbook_oct.sh", "implant/behaviour/train.py"):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(fake_generic_key())
    proc = _run_gate_secrets(repo)
    out = proc.stdout + proc.stderr
    assert "secrets: untracked files" in out
    assert proc.returncode == 0, out
