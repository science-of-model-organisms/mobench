"""`load_specs_many`: one file is untouched, several merge, and every conflict raises."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

from mobench.cli import main as mobench
from serving.inspect_runner import _build_parser
from serving.runner import (
    SpecConflictError,
    load_config,
    load_specs,
    load_specs_many,
    merge_configs,
)

REPO = Path(__file__).resolve().parent.parent


def _write(path: Path, *, persona: str, evals: dict | None = None, **top: object) -> Path:
    body = {
        "defaults": {"base_model": f"org/{persona}-base", "seed": 0},
        "specs": [
            {"method": "base", "persona_id": persona},
        ],
        "default_evals": [f"eval_{persona}"],
        "eval_configs": evals or {},
        **top,
    }
    path.write_text(yaml.safe_dump(body), encoding="utf-8")
    return path


def test_one_file_is_the_real_matrix_hash_for_hash() -> None:
    path = REPO / "configs" / "specs.yaml"
    one = [s.spec_hash for s in load_specs(path)]
    assert [s.spec_hash for s in load_specs_many([path])] == one
    assert merge_configs([path]) == load_config(path)


def test_two_files_keep_their_own_defaults_and_their_hashes(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive")
    b = _write(tmp_path / "b.yaml", persona="loyalty")
    merged = load_specs_many([a, b])
    assert [s.spec_hash for s in merged] == [
        *[s.spec_hash for s in load_specs(a)],
        *[s.spec_hash for s in load_specs(b)],
    ]
    assert [s.base_model for s in merged] == ["org/impulsive-base", "org/loyalty-base"]


def test_a_row_repeated_identically_is_kept_once(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive")
    b = _write(tmp_path / "b.yaml", persona="impulsive")
    assert len(load_specs_many([a, b])) == 1


def test_a_repeated_hash_with_a_different_entry_raises(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive")
    data = load_config(a)
    # `1` and `1.0` hash alike (core.spec._canonical) but are not the same text.
    data["specs"][0]["train_scale"] = 1
    b = tmp_path / "b.yaml"
    b.write_text(yaml.safe_dump(data), encoding="utf-8")
    data["specs"][0]["train_scale"] = 1.0
    c = tmp_path / "c.yaml"
    c.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(SpecConflictError, match="different entries"):
        load_specs_many([b, c])


def test_an_eval_config_defined_differently_raises(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive", evals={"e": {"k": 1}})
    b = _write(tmp_path / "b.yaml", persona="loyalty", evals={"e": {"k": 2}})
    with pytest.raises(SpecConflictError, match="eval_configs.e"):
        merge_configs([a, b])


def test_a_differing_knob_raises_rather_than_the_first_winning(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive", max_tasks=2)
    b = _write(tmp_path / "b.yaml", persona="loyalty", max_tasks=4)
    with pytest.raises(SpecConflictError, match="max_tasks"):
        merge_configs([a, b])


def test_the_eval_list_is_the_union(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive", evals={"x": {"k": 1}})
    b = _write(tmp_path / "b.yaml", persona="loyalty", evals={"y": {"k": 2}})
    merged = merge_configs([a, b])
    assert merged["default_evals"] == ["eval_impulsive", "eval_loyalty"]
    assert set(merged["eval_configs"]) == {"x", "y"}


def test_runner_config_flag_repeats() -> None:
    args = _build_parser().parse_args(["--config", "a.yaml", "--config", "b.yaml"])
    assert args.config == ["a.yaml", "b.yaml"]


def test_mobench_entry_point_is_declared() -> None:
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert 'mobench = "mobench.cli:main"' in text


def test_mobench_validate_reports_a_conflict(tmp_path: Path, capsys) -> None:
    a = _write(tmp_path / "a.yaml", persona="impulsive", evals={"e": {"k": 1}})
    b = _write(tmp_path / "b.yaml", persona="loyalty", evals={"e": {"k": 2}})
    assert mobench(["validate", "--config", str(a), "--config", str(b)]) == 1
    assert "eval_configs.e" in capsys.readouterr().out


def test_run_all_merges_repeated_configs_in_a_dry_run(tmp_path: Path) -> None:
    smoke = REPO / "configs" / "specs.smoke.yaml"
    env = {
        **os.environ,
        "MO_EVALS_FAKE": "1",
        "MO_EVALS_RESULTS_ROOT": str(tmp_path / "results"),
        "UV_NO_SYNC": "1",
    }
    out = subprocess.run(
        [
            "bash",
            "scripts/run_all.sh",
            "--config",
            str(smoke),
            "--config",
            str(smoke),
            "--dry-run",
        ],  # noqa: E501
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "TODO" in out.stdout or "CACHED" in out.stdout
