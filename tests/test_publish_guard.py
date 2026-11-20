"""`scripts/publish_guard.sh` -- whether a real `run_all.sh` run should publish its
dashboard to the shared scorecards repo.

The failure being guarded: a filtered probe run (`--methods steering --evals
apps`) against a throwaway results root rendered a 1-organism dashboard and
published it under the SAME filename as the full-matrix 7B impulsive scorecard
(`analysis/dashboard.py`'s `scorecard_name()` keys only on persona + base model), silently
overwriting it -- twice in one day.

Sourced directly (function only, no top-level side effects when sourced) rather than
re-implemented, so these tests cannot drift from the shipped decision the way a Python
re-implementation of the same bash logic could.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish_guard.sh"


def _reason(root: str, *args: str, publish_env: str | None = None) -> str:
    env_prefix = f'MO_EVALS_PUBLISH="{publish_env}" ' if publish_env is not None else ""
    argv = " ".join(f'"{a}"' for a in args)
    out = subprocess.run(
        ["bash", "-c", f'source "{_SCRIPT}"; {env_prefix}publish_skip_reason "{root}" {argv}'],
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_an_unfiltered_real_run_on_the_default_root_publishes() -> None:
    """The normal case (CLAUDE.md): `--personas impulsive` alone, default root."""
    assert _reason("results") == ""


def test_methods_flag_narrows_and_is_skipped() -> None:
    reason = _reason("results", "--methods", "steering")
    assert "narrowed" in reason
    assert "--methods" in reason


def test_evals_flag_narrows_and_is_skipped() -> None:
    reason = _reason("results", "--evals", "apps")
    assert "narrowed" in reason


def test_exclude_evals_flag_narrows_and_is_skipped() -> None:
    reason = _reason("results", "--exclude-evals", "apps")
    assert "narrowed" in reason


def test_base_models_flag_narrows_and_is_skipped() -> None:
    reason = _reason("results", "--base-models", "Qwen/Qwen2.5-7B-Instruct")
    assert "narrowed" in reason


def test_equals_form_of_a_narrowing_flag_is_also_caught() -> None:
    reason = _reason("results", "--methods=steering")
    assert "narrowed" in reason


def test_non_default_results_root_is_skipped() -> None:
    reason = _reason("/tmp/probe-run")
    assert "MO_EVALS_RESULTS_ROOT" in reason
    assert "/tmp/probe-run" in reason


def test_explicit_opt_in_overrides_a_narrowing_flag() -> None:
    assert _reason("results", "--methods", "steering", publish_env="1") == ""


def test_explicit_opt_in_overrides_a_non_default_root() -> None:
    assert _reason("/tmp/probe-run", publish_env="1") == ""


def test_non_narrowing_flags_do_not_trip_the_guard() -> None:
    """--force/--real/--require-all-cached/-v don't change which organisms are selected."""
    assert _reason("results", "--force", "--real", "--require-all-cached", "-v") == ""


def test_publish_0_does_not_opt_in() -> None:
    """The opt-in is documented as exactly `1`; `MO_EVALS_PUBLISH=0` must not be treated as
    truthy (a bare `[[ -n ... ]]` check reads ANY non-empty string, including "0", as set)."""
    reason = _reason("results", "--methods", "steering", publish_env="0")
    assert "narrowed" in reason


def test_publish_false_does_not_opt_in() -> None:
    reason = _reason("results", "--methods", "steering", publish_env="false")
    assert "narrowed" in reason


def test_publish_empty_does_not_opt_in() -> None:
    reason = _reason("results", "--methods", "steering", publish_env="")
    assert "narrowed" in reason


def test_publish_1_true_yes_on_all_opt_in() -> None:
    """Matches how `run_all.sh` parses `MO_EVALS_FAKE`: `^(1|true|yes|on)$`."""
    for truthy in ("1", "true", "yes", "on"):
        assert _reason("results", "--methods", "steering", publish_env=truthy) == ""
