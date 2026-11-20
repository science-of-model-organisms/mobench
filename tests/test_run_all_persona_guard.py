"""`scripts/run_all.sh` refuses a real sweep that does not say which behaviours to run.

CLAUDE.md: `configs/specs.yaml` holds every persona, so a bare invocation sweeps all
of them — which is how GPU and judge budget has gone on personas nobody asked for. The
prose rule only binds an agent that reads the file; this guard binds anything that invokes
the pipeline. Fake runs are exempt because they cost nothing and CLAUDE.md (Commands)
requires them on every commit, so the gate must keep working without a `--personas` flag.

Shell-level on purpose: the guard is in the script, and the thing worth testing is what
the script does with an argv, not what a Python function would do with a list.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = str(_ROOT / "scripts" / "run_all.sh")
_SMOKE = "configs/specs.smoke.yaml"


def _run(
    *args: str,
    fake: bool = False,
    results_root: Path | None = None,
    config: str = _SMOKE,
    extra_env: dict[str, str] | None = None,
):
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("MO_EVALS_")  # a caller's own settings must not steer the test
    }
    env["MO_EVALS_CONFIG"] = config
    if fake:
        env["MO_EVALS_FAKE"] = "1"
    if results_root is not None:
        env["MO_EVALS_RESULTS_ROOT"] = str(results_root)
    env.update(extra_env or {})
    return subprocess.run(
        [_SCRIPT, *args], cwd=_ROOT, env=env, capture_output=True, text=True, timeout=300
    )


def _two_persona_config(tmp_path: Path) -> str:
    """The smoke config plus a `loyalty` prompting organism, for tests that need two
    distinct real personas: the smoke config itself holds impulsive only since
    `mathematical` was removed (2026-10-05). Only ever `--dry-run`, so nothing is served."""
    cfg = yaml.safe_load((_ROOT / _SMOKE).read_text(encoding="utf-8"))
    cfg["organisms"]["loyalty"] = [
        {
            "method": "prompting",
            "method_config": {
                "system_prompt_path": "data/personas/loyalty.system_prompt.txt",
                "polarity": "positive",
            },
        }
    ]
    path = tmp_path / "two_personas.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(path)


def _personas_listed(stdout: str) -> set[str]:
    """Persona ids off the `--dry-run` listing: `TODO    <persona>/<method>@... x <eval> ...`."""
    return {
        line.split()[1].split("/")[0]
        for line in stdout.splitlines()
        if line.startswith(("TODO", "CACHED"))
    }


def test_a_real_run_without_personas_is_refused() -> None:
    """The whole point: no behaviour named, no sweep. Exits before any Python starts."""
    done = _run()
    assert done.returncode == 1
    assert "REFUSING" in done.stderr
    assert "--personas" in done.stderr


def test_a_fake_run_without_personas_is_not_refused(tmp_path: Path) -> None:
    """The gate (CLAUDE.md, Commands) runs exactly this on every commit; the guard must not
    reach it."""
    done = _run("--dry-run", fake=True, results_root=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "REFUSING" not in done.stderr


def test_personas_all_is_consumed_here_not_passed_to_the_runner(tmp_path: Path) -> None:
    """`all` is the explicit opt-in to the full matrix, and is NOT a persona id — passing
    it through would trip `_select_specs`'s typo-guard with `no match for ['all']`."""
    done = _run(
        "--personas",
        "all",
        "--dry-run",
        fake=True,
        results_root=tmp_path,
        config=_two_persona_config(tmp_path),
    )
    assert done.returncode == 0, done.stderr
    assert "no match for" not in done.stderr
    personas = _personas_listed(done.stdout)
    assert personas == {"impulsive", "loyalty"}, personas


def test_base_models_is_forwarded_to_the_runner(tmp_path: Path) -> None:
    """Unlike `--personas`, `--base-models` is NOT consumed here: the `*)` case forwards
    it unchanged and `_select_specs`'s typo-guard is what makes a wrong id fail. A flag
    the script silently swallowed would run the whole matrix and look like a success."""
    done = _run("--base-models", "no/such-model", "--dry-run", fake=True, results_root=tmp_path)
    assert done.returncode != 0
    assert "--base-models: no match" in done.stderr


def test_a_named_persona_narrows_the_sweep(tmp_path: Path) -> None:
    done = _run("--personas", "impulsive", "--dry-run", fake=True, results_root=tmp_path)
    assert done.returncode == 0, done.stderr
    personas = _personas_listed(done.stdout)
    assert personas == {"impulsive"}, personas


def test_personas_all_mixed_with_a_named_persona_is_refused(tmp_path: Path) -> None:
    """`--personas all --personas impulsve` (a typo) must fail loudly, not run silently.

    `all` is consumed here rather than forwarded (see the block above), so when `all` was
    present the script used to skip validating every OTHER accumulated value and forward
    NONE of them -- the typo `impulsve` was silently dropped and the run swept the full
    matrix by omission (no `--personas` reached `serving.inspect_runner`, whose own
    default is "no filter"), saying nothing about the bad flag. Mixing `all` with a named
    persona is inherently ambiguous, so the stricter fix is to refuse the combination
    outright -- a real typo alongside `all` must never slip through as if it had been
    checked and dropped for a good reason.
    """
    done = _run(
        "--personas",
        "all",
        "--personas",
        "impulsve",
        "--dry-run",
        fake=True,
        results_root=tmp_path,
    )
    assert done.returncode == 1
    assert "REFUSING" in done.stderr
    assert "all" in done.stderr


def test_personas_all_mixed_with_a_valid_named_persona_is_also_refused(tmp_path: Path) -> None:
    """Even a REAL persona alongside `all` is refused -- the ambiguity is in the mix, not
    just in a typo. (A typo would silently vanish either way; a valid name would silently
    do nothing since `all` already covers it -- both are worth failing loudly on.)"""
    done = _run(
        "--personas",
        "all",
        "--personas",
        "impulsive",
        "--dry-run",
        fake=True,
        results_root=tmp_path,
    )
    assert done.returncode == 1
    assert "REFUSING" in done.stderr


def test_repeated_personas_accumulate_instead_of_last_one_winning(tmp_path: Path) -> None:
    """`--personas a --personas b` must mean BOTH.

    The script used to stash `--personas` in a scalar, so a repeated flag kept only the
    LAST value and forwarded one -- which hides a typo on a real run (`--personas impulsve
    --personas impulsive` would silently run only `impulsive`, the good one, saying
    nothing about the bad first flag) and can shrink or widen a run without a word
    (CLAUDE.md). Two DISTINCT real personas here, so a revert to the scalar (which
    keeps only `loyalty`) fails this test.
    """
    done = _run(
        "--personas",
        "impulsive",
        "--personas",
        "loyalty",
        "--dry-run",
        fake=True,
        results_root=tmp_path,
        config=_two_persona_config(tmp_path),
    )
    assert done.returncode == 0, done.stderr
    personas = _personas_listed(done.stdout)
    assert personas == {"impulsive", "loyalty"}, personas


def test_no_publish_stops_the_periodic_hub_push_too(tmp_path: Path) -> None:
    """MO_EVALS_NO_PUBLISH used to stop only the dashboard publish: a repo in the environment
    still had its results pushed every interval. Without the variable the tick is armed."""
    push = {
        "MO_EVALS_HF_RESULTS_REPO": "someone/not-a-real-repo",
        "MO_EVALS_HF_PUSH_INTERVAL": "5",
    }
    args = ("--personas", "impulsive", "--dry-run")
    armed = _run(*args, results_root=tmp_path, extra_env=push)
    assert armed.returncode == 0, armed.stderr
    assert "periodic HF push: every 5s" in armed.stderr
    held = _run(*args, results_root=tmp_path, extra_env={**push, "MO_EVALS_NO_PUBLISH": "1"})
    assert held.returncode == 0, held.stderr
    assert "periodic HF push: every" not in held.stderr
    assert "MO_EVALS_NO_PUBLISH is set" in held.stderr


def test_the_shell_reads_the_fake_flag_like_core_env(tmp_path: Path) -> None:
    """`MO_EVALS_FAKE=false` is off (a real run: refused without --personas) and ` TRUE ` is on,
    exactly as core.env.fake_enabled says."""
    off = _run("--dry-run", results_root=tmp_path, extra_env={"MO_EVALS_FAKE": "false"})
    assert off.returncode == 1 and "REFUSING" in off.stderr
    on = _run("--dry-run", results_root=tmp_path, extra_env={"MO_EVALS_FAKE": " TRUE "})
    assert on.returncode == 0, on.stderr
    assert "REFUSING" not in on.stderr
