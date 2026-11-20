"""`.github/workflows/ci.yml` is path-aware on pull requests the way `scripts/gate.sh` is.

Actions minutes are metered, so on a PR CI skips the two fake-pipeline
steps when `scripts/gate_scope.py` says the diff cannot reach them. No CI run can check this
wiring for us, and a loose check passes while the savings (or the safety) silently vanish —
the #383 review mutated this file and survived a typo'd output key, a base without
`origin/`, and a skippable pytest. So these pin the exact strings.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CI = ROOT / ".github" / "workflows" / "ci.yml"
PIPELINE_STEPS = {"Upstream eval datasets", "Fake-mode byte identity"}
ALWAYS_STEPS = {"Lint", "Types (whole repo)", "Lane boundaries", "Tests", "Install"}
LEAN_GUARD = "steps.scope.outputs.scope != 'lean'"
SCOPE_RUN = 'echo "scope=$(uv run python scripts/gate_scope.py)" >> "$GITHUB_OUTPUT"'


def _workflow() -> dict:
    return yaml.safe_load(CI.read_text(encoding="utf-8"))


def _steps() -> list[dict]:
    return _workflow()["jobs"]["check"]["steps"]


def _scope() -> dict:
    return next(s for s in _steps() if s.get("id") == "scope")


def test_checkout_fetches_full_history_for_the_merge_base() -> None:
    checkout = next(s for s in _steps() if str(s.get("uses", "")).startswith("actions/checkout"))
    assert checkout.get("with", {}).get("fetch-depth") == 0


def test_the_scope_step_is_exactly_the_reviewed_wiring() -> None:
    scope = _scope()
    assert scope["run"] == SCOPE_RUN
    assert scope["env"] == {"MO_EVALS_GATE_BASE": "origin/${{ github.base_ref }}"}
    # PRs only: on a push its output is empty, which is not `lean`, so everything runs.
    assert scope["if"] == "github.event_name == 'pull_request'"


def test_the_scope_step_runs_before_everything_it_gates() -> None:
    steps = _steps()
    names = [s.get("name") for s in steps]
    assert steps.index(_scope()) < min(names.index(n) for n in PIPELINE_STEPS)


def test_only_the_pipeline_steps_are_skippable() -> None:
    names = {s.get("name") for s in _steps()}
    assert ALWAYS_STEPS <= names, ALWAYS_STEPS - names
    for step in _steps():
        if step.get("name") in PIPELINE_STEPS:
            assert step.get("if") == LEAN_GUARD, step["name"]
        elif step.get("name") in ALWAYS_STEPS:
            assert "if" not in step, step["name"]


def test_nothing_is_allowed_to_fail_quietly() -> None:
    job = _workflow()["jobs"]["check"]
    assert "continue-on-error" not in job
    assert not [s.get("name") for s in _steps() if "continue-on-error" in s]


def test_the_step_command_says_sweep_on_a_bad_base(tmp_path: Path) -> None:
    """The exact `run:` line, in the caller's environment, with a base that does not exist."""
    out_file = tmp_path / "out"
    env = {**os.environ, "GITHUB_OUTPUT": str(out_file), "MO_EVALS_GATE_BASE": "origin/nope"}
    subprocess.run(["bash", "-c", SCOPE_RUN], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out_file.read_text().strip() == "scope=sweep"


def test_a_failing_scope_command_is_not_lean(tmp_path: Path) -> None:
    """Actions' default shell is `bash -e` without pipefail: a failed substitution leaves
    `scope=`, which the guards treat as not-lean. Pin that with the step's own shape."""
    out_file = tmp_path / "out"
    broken = SCOPE_RUN.replace("uv run python", "false &&")
    env = {**os.environ, "GITHUB_OUTPUT": str(out_file)}
    subprocess.run(["bash", "-e", "-c", broken], cwd=ROOT, env=env, capture_output=True)
    assert out_file.read_text().strip() == "scope="


def test_editing_ci_yml_is_on_the_sweep_path() -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("gate_scope", ROOT / "scripts" / "gate_scope.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gate_scope", mod)
    spec.loader.exec_module(mod)
    assert mod.on_path(".github/workflows/ci.yml", mod.sweep_path())
