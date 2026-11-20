"""Generation caps raised for EVERY base model (configs/specs.yaml, 2026-10-15).

768 truncated answers (base Gemma `simple_behaviour_exhibit` 42% cut mid-sentence; Qwen on
fermi 67%, downstream 59%, drift_long 33%). These tests BUILD the real Inspect task for each
affected eval and run its real solver against a capturing endpoint, recording the
`max_tokens` that actually reaches `Endpoint.complete`. A cap set under a key the builder
ignores (as `introspection_denial`'s bare `max_tokens` once was) therefore fails here. No GPU.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

import serving.inspect_runner  # noqa: F401  registers the evals
from core.endpoint import FakeEndpoint
from core.spec import OrganismSpec
from evals._inspect import all_evals, bind_endpoint, target_sampling, task_factory
from serving.runner import eval_config_for, load_config, load_specs

GEMMA = "google/gemma-4-31B-it"
QWEN = "Qwen/Qwen2.5-7B-Instruct"
# Evals with their own cap. EVERY other judged eval (the `register_judged_task` registry, read
# below, so a newly registered one is covered automatically) must run at the shared default.
NEW_CAPS = {
    "introspection_denial": 1024,
    "misaligned_sycophancy": 1024,
    "misaligned_sandbagging": 512,
    "misaligned_overrefusal": 1024,
    "misaligned_reward_hacking": 2048,
    "persistence_multiturn_drift": 2048,
    "persistence_multiturn_drift_long": 2048,
    "generality_fermi_gen": 4096,
}
SHARED_DEFAULT = 2048
# Deliberately uncapped (`organism_max_tokens: remaining_context`), so no fixed cap to assert.
UNCAPPED = {"robustness_multiturn_debate_gen"}


def _covered() -> dict[str, int]:
    from evals._task_eval import _JUDGED

    names = {*_JUDGED, *NEW_CAPS, "simple_behaviour_exhibit"} - UNCAPPED
    return {n: NEW_CAPS.get(n, SHARED_DEFAULT) for n in sorted(names)}


class _Capture(FakeEndpoint):
    def __init__(self) -> None:
        super().__init__(persona="impulsive")
        self.kw: list[dict[str, Any]] = []

    def complete(self, messages: Any, **kw: Any) -> str:  # type: ignore[override]
        self.kw.append(kw)
        return "ANSWER: B"


def _bases() -> list[str]:
    return sorted({s.base_model for s in load_specs("configs/specs.yaml")})


def _captured_caps(base: str, name: str, tmp_path: Any) -> set[Any]:
    from inspect_ai import eval_async

    cfg = load_config("configs/specs.yaml")
    spec = next(s for s in load_specs(cfg) if s.base_model == base)
    ev = next(e for e in all_evals() if e.name == name)
    c = eval_config_for(cfg, spec, ev)
    c["judge"] = {"backend": "fake"}
    if name == "generality_controls":
        # No committed persona has a controls bank (it EvalSkips); the cap does not depend on
        # the items, so a committed bank of the same rubric stands in.
        c["bank"] = "generality/generated/impulsive/ood.yaml"
    ep = _Capture()

    async def run() -> None:
        with bind_endpoint(
            f"caps_{spec.spec_hash}_{name}",
            ep,
            0,
            **target_sampling(c.get("complete_kwargs")),  # what the resolver binds
            max_model_len=16384,
            prompt_token_count=lambda *a, **k: 100,
            base_model=base,
        ) as model:
            await eval_async(
                task_factory(name)(c),
                model=model,
                limit=1,
                log_dir=str(tmp_path),
                score=False,
            )

    asyncio.run(run())
    assert ep.kw, f"{name} made no generation call"
    return {k.get("max_tokens") for k in ep.kw}


@pytest.mark.parametrize("base", [QWEN, GEMMA])
@pytest.mark.parametrize("name", list(_covered()))
def test_the_cap_that_reaches_the_endpoint(base: str, name: str, tmp_path: Any) -> None:
    assert _captured_caps(base, name, tmp_path) == {_covered()[name]}, (name, base)


@pytest.mark.parametrize("base", _bases())
def test_no_per_base_override_and_mmlu_untouched(base: str) -> None:
    """mmlu: Qwen keeps upstream's 16-token non-CoT setting; Gemma 1024 via by_base_model."""
    cfg = load_config("configs/specs.yaml")
    spec = OrganismSpec("impulsive", base, "base", {})
    for ev in all_evals():
        assert "by_base_model" not in eval_config_for(cfg, spec, ev), (ev.name, base)
    mmlu = eval_config_for(cfg, spec, SimpleNamespace(name="mmlu"))  # type: ignore[arg-type]
    assert mmlu.get("max_tokens") == (1024 if base == GEMMA else None)


def test_spec_hashes_are_unique_and_independent_of_eval_configs() -> None:
    """Caps are eval config, not spec (before/after hash list compared in the PR body)."""
    specs = load_specs("configs/specs.yaml")
    assert len(specs) == len({s.spec_hash for s in specs})


def test_context_budget_reserves_the_requested_cap() -> None:
    """Drift plans its history from `_context_budget_tokens`; a prompt filled to it plus a
    2048-token reply must fit the window (Codex: 14,811 + 2048 > 16,384 at a 512 reserve),
    and a 512 cap must leave the number exactly as before."""
    from serving.resolve import _EVAL_HEADROOM_TOKENS, context_budget_tokens, serving_config

    cfg = load_config("configs/specs.yaml")
    spec = OrganismSpec("impulsive", QWEN, "base", {})
    window = serving_config().max_model_len
    for name in ("persistence_multiturn_drift_long", "persistence_multiturn_drift"):
        budget = eval_config_for(cfg, spec, SimpleNamespace(name=name))["_context_budget_tokens"]  # type: ignore[arg-type]
        assert budget + 2048 + _EVAL_HEADROOM_TOKENS <= window, name
    assert context_budget_tokens(spec, output_tokens=512) == context_budget_tokens(spec)
    assert context_budget_tokens(spec, output_tokens=2048) == context_budget_tokens(spec) - (
        2048 - serving_config().sampling.max_tokens
    )


def test_reservation_honours_a_bare_max_tokens() -> None:
    """The misaligned cells generate under a bare `max_tokens`; the context budget must
    reserve that, not the shared cap (it is larger or smaller, whichever the eval uses)."""
    from serving.resolve import context_budget_tokens

    cfg = load_config("configs/specs.yaml")
    spec = OrganismSpec("impulsive", QWEN, "base", {})
    base = context_budget_tokens(spec)
    for name in ("misaligned_sycophancy", "misaligned_overrefusal"):
        got = eval_config_for(cfg, spec, SimpleNamespace(name=name))["_context_budget_tokens"]  # type: ignore[arg-type]
        assert got == context_budget_tokens(spec, output_tokens=1024), name
        assert got == base - (1024 - 512)
    got = eval_config_for(cfg, spec, SimpleNamespace(name="misaligned_sandbagging"))[  # type: ignore[arg-type]
        "_context_budget_tokens"
    ]
    assert got == base  # 512 reserve: identical to the serving default
