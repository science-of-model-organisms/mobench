"""`scripts/runbook_oct.sh`'s env-overridable knobs keep impulsive's defaults.

Loyalty overrides two of them (`MO_RUNBOOK_TEACHER_NAME=ChatGLM` for its GLM-written chosen
side, `MO_RUNBOOK_KL_COEF=0` to match impulsive's registered organisms, trained without the
KL term). The test evaluates the script's own assignment lines in bash, so a default that
drifts, or an override that stops being read, fails here rather than on a GPU pod.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

RUNBOOK = Path(__file__).resolve().parent.parent / "scripts" / "runbook_oct.sh"
KNOBS = ("TEACHER_NAME", "KL_COEF", "HF_ORG", "HF_DATA", "HF_ADAPTERS")


def _assignments() -> str:
    text = RUNBOOK.read_text(encoding="utf-8")
    lines = [line for line in text.splitlines() if re.match(rf"^({'|'.join(KNOBS)})=", line)]
    assert len(lines) == len(KNOBS), lines
    return "\n".join(lines)


def _eval(env: dict[str, str]) -> dict[str, str]:
    script = _assignments() + "\n" + "\n".join(f'echo "{k}=${k}"' for k in KNOBS)
    out = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines())


def test_defaults_are_impulsives() -> None:
    got = _eval({})
    assert got["TEACHER_NAME"] == "ChatGPT"
    assert got["KL_COEF"] == "0.001"
    assert got["HF_DATA"] == "science-of-model-organisms/oct-behaviour-data"
    assert got["HF_ADAPTERS"] == "science-of-model-organisms/qwen2.5-instruct-organisms"


def test_hf_org_env_repoints_the_default_repos() -> None:
    got = _eval({"MO_EVALS_HF_ORG": "myorg"})
    assert got["HF_DATA"] == "myorg/oct-behaviour-data"
    assert got["HF_ADAPTERS"] == "myorg/qwen2.5-instruct-organisms"


@pytest.mark.parametrize(
    ("var", "knob", "value"),
    [
        ("MO_RUNBOOK_TEACHER_NAME", "TEACHER_NAME", "ChatGLM"),
        ("MO_RUNBOOK_KL_COEF", "KL_COEF", "0"),
        (
            "MO_RUNBOOK_HF_DATA",
            "HF_DATA",
            "science-of-model-organisms/loyalty-oct-behaviour-data",
        ),
        (
            "HF_ADAPTERS",
            "HF_ADAPTERS",
            "science-of-model-organisms/qwen2.5-7b-it_loyalty-oct-lora",
        ),
    ],
)
def test_loyalty_overrides_are_read(var: str, knob: str, value: str) -> None:
    assert _eval({var: value})[knob] == value


def test_the_pairs_stage_launders_the_configured_name_not_a_literal() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    assert '--teacher-name "$TEACHER_NAME"' in text
    assert "--teacher-name ChatGPT" not in text
    assert '--kl-coef "$KL_COEF"' in text


@pytest.mark.parametrize(
    ("raw", "org"),
    [
        (" myorg ", "myorg"),
        ("\tmyorg\n", "myorg"),
        ("   ", "science-of-model-organisms"),
        ("", "science-of-model-organisms"),
    ],
)
def test_hf_org_is_stripped_like_core_hf(raw: str, org: str) -> None:
    got = _eval({"MO_EVALS_HF_ORG": raw})
    assert got["HF_DATA"] == f"{org}/oct-behaviour-data"
    assert got["HF_ADAPTERS"] == f"{org}/qwen2.5-instruct-organisms"


def _hub_functions() -> str:
    """`fetch` and `publish`, cut out of the script (up to the next function, `fork_sha`)."""
    text = RUNBOOK.read_text(encoding="utf-8")
    start, end = text.index("\nfetch() {"), text.index("\nfork_sha() {")
    return text[start:end]


def _hub_calls(env: dict[str, str], tmp_path: Path) -> tuple[list[str], str]:
    """Run fetch and publish with `uv` replaced by a recorder; return (uv calls, results)."""
    calls = tmp_path / "calls"
    script = f"""
uv() {{ echo "$*" >> {calls}; return 0; }}
FORCE=""; HF_DATA=org/data; HUB={tmp_path}/hub; mkdir -p "$HUB/d"; : > {tmp_path}/f
{_hub_functions()}
fetch d "x:{tmp_path}/y" >/dev/null; echo "fetch=$?"
publish d '{{}}' "x:{tmp_path}/f" >/dev/null; echo "publish=$?"
"""
    out = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **env},
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return (calls.read_text().splitlines() if calls.exists() else []), out


def test_no_hub_neither_fetches_nor_publishes(tmp_path: Path) -> None:
    calls, out = _hub_calls({"MO_RUNBOOK_NO_HUB": "1"}, tmp_path)
    assert calls == [] and "fetch=1" in out and "publish=0" in out


def test_without_no_hub_the_runbook_still_talks_to_the_hub(tmp_path: Path) -> None:
    calls, _ = _hub_calls({}, tmp_path)
    assert any(c.startswith("run hf download org/data") for c in calls)
    assert any(c.startswith("run hf upload org/data") for c in calls)
