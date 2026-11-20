"""`extends:` in a serving config, so a one-knob variant is not a 67-line copy.

`configs/serving_swe_bench.yaml` was a byte-for-byte duplicate of `configs/serving.yaml`
apart from `max_model_len`, kept in step by a comment asking humans to remember. That is
the kind of sync nobody does: the copy silently rots the first time the original changes.
"""

from pathlib import Path

import pytest

from serving.vllm_backend import ServingConfig


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_a_child_inherits_every_key_it_does_not_restate(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        "max_model_len: 16384\ngpu_memory_utilization: 0.87\n"
        "sampling:\n  temperature: 0.0\n  max_tokens: 768\n",
    )
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\nmax_model_len: 32768\n")

    config = ServingConfig.load(child)

    assert config.max_model_len == 32768, "the child's own value must win"
    assert config.gpu_memory_utilization == 0.87, "inherited, not reset to the class default"
    assert config.sampling.max_tokens == 768, "nested sampling inherits too"


def test_a_child_can_override_one_sampling_key_without_dropping_the_rest(tmp_path: Path) -> None:
    _write(tmp_path / "base.yaml", "sampling:\n  temperature: 0.0\n  max_tokens: 768\n")
    child = _write(tmp_path / "child.yaml", "extends: base.yaml\nsampling:\n  max_tokens: 4096\n")

    config = ServingConfig.load(child)

    assert config.sampling.max_tokens == 4096
    assert config.sampling.temperature == 0.0, "sampling merges per key, it does not replace"


def test_extends_is_resolved_against_the_child_not_the_cwd(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    _write(tmp_path / "configs" / "base.yaml", "max_model_len: 16384\n")
    child = _write(tmp_path / "configs" / "child.yaml", "extends: base.yaml\n")

    assert ServingConfig.load(child).max_model_len == 16384


def test_a_cycle_raises_rather_than_hanging(tmp_path: Path) -> None:
    _write(tmp_path / "a.yaml", "extends: b.yaml\n")
    _write(tmp_path / "b.yaml", "extends: a.yaml\n")

    with pytest.raises(ValueError, match="cycle"):
        ServingConfig.load(tmp_path / "a.yaml")


def test_the_real_swe_bench_config_differs_from_serving_yaml_only_in_max_model_len() -> None:
    """The reason `extends:` exists — pinned so the pair cannot drift apart again."""
    root = Path(__file__).resolve().parents[1]
    base = ServingConfig.load(root / "configs" / "serving.yaml")
    swe = ServingConfig.load(root / "configs" / "serving_swe_bench.yaml")

    assert swe.max_model_len == 32768
    assert base.max_model_len != swe.max_model_len
    from dataclasses import replace

    assert replace(swe, max_model_len=base.max_model_len) == base, (
        "serving_swe_bench.yaml must differ from serving.yaml in max_model_len and nothing else"
    )
