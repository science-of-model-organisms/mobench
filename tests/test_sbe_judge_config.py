"""The configured SBE judge: GLM judges presence, Gemma is the coherence secondary.

configs/specs.yaml `eval_configs.simple_behaviour_exhibit.judge` (2026-10-07 decision;
docs/evals/expression.md#judge). `eval_configs` merges shallowly, so the per-eval `judge` key
replaces the default block whole; this pins that nothing but the intended swap changed."""

from __future__ import annotations

from pathlib import Path

import yaml

SPECS = Path(__file__).resolve().parents[1] / "configs" / "specs.yaml"


def _load() -> dict:
    return yaml.safe_load(SPECS.read_text())


def test_sbe_presence_judge_is_glm_with_gemma_as_secondary() -> None:
    ec = _load()["eval_configs"]
    judge = ec["simple_behaviour_exhibit"]["judge"]
    assert judge["backend"] == "http"
    assert judge["model"] == "z-ai/glm-5.3-flash"
    # Reasoning model: an output cap truncates its CoT before the verdict (specs.yaml `models`).
    assert "max_tokens" not in judge["params"]
    provider = judge["params"]["provider"]
    assert provider["only"] == ["z-ai/fp8"] and provider["allow_fallbacks"] is False
    # The coherence gate's judge is the shared default, verbatim.
    assert judge["secondary"] == ec["default"]["judge"]
    assert judge["secondary"]["model"] == "google/gemma-4-31b-it"


def test_every_other_eval_keeps_the_default_judge() -> None:
    ec = _load()["eval_configs"]
    default_model = ec["default"]["judge"]["model"]
    assert default_model == "google/gemma-4-31b-it"
    own = {n for n, c in ec.items() if isinstance(c, dict) and "judge" in c}
    assert own == {"default", "simple_behaviour_exhibit"}, own


def test_judge_models_registry_names_both_judges() -> None:
    models = _load()["models"]
    assert models["sbe_presence_judge"] == "z-ai/glm-5.3-flash"
    assert models["judge"] == "google/gemma-4-31b-it"


def test_the_merged_sbe_config_builds_a_glm_primary_gemma_secondary_panel(monkeypatch) -> None:
    from judges.judge import build_judge

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")
    ec = _load()["eval_configs"]
    merged = {**ec["default"], **ec["simple_behaviour_exhibit"]}  # the runner's shallow merge
    judge = build_judge(merged)
    assert "glm-5.3-flash" in judge.name
    assert judge.secondary_backend is not None
    assert "gemma-4-31b-it" in getattr(judge.secondary_backend, "name", "")
    # And every other judged eval still builds the single Gemma judge, no panel.
    other = build_judge({**ec["default"], **ec.get("generality_fermi_gen", {})})
    assert "gemma-4-31b-it" in other.name and other.secondary_backend is None


def test_judge_override_reaches_the_sbe_presence_judge_and_keeps_the_gemma_secondary() -> None:
    """Adversarial review of PR #435: `--judge-model` replaced only the default block, which
    SBE's own `judge` shadows, so the override was silently ignored for SBE. It now swaps the
    PRIMARY of a per-eval panel and leaves its secondary (the coherence gate) alone."""
    from scripts.judge_deferred import _apply_judge_override

    config = _load()
    secondary = config["eval_configs"]["simple_behaviour_exhibit"]["judge"]["secondary"]
    _apply_judge_override(config, "some/judge", "friendli", 512)
    ec = config["eval_configs"]
    sbe = ec["simple_behaviour_exhibit"]["judge"]
    assert sbe["model"] == ec["default"]["judge"]["model"] == "some/judge"
    assert sbe["params"]["max_tokens"] == 512
    assert sbe["params"]["provider"]["only"] == ["friendli"]
    assert sbe["secondary"] == secondary and secondary["model"] == "google/gemma-4-31b-it"
