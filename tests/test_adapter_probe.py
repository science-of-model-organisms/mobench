"""The differs-from-base probe that every adapter method passes through at resolve time.

`serving.resolve._assert_adapter_applied` stands between an adapter vLLM silently ignored
and base-model numbers published under the adapter organism's spec_hash: vLLM loads an
adapter whose keys match no module and reports success. The constant and the dtype floor
ARE the guard, so both directions are pinned here. No GPU: the two logprob requests are
stubbed.
"""

from __future__ import annotations

import importlib
import math
from typing import Any

import pytest

from core.spec import OrganismSpec
from serving.vllm_backend import LoRAAdapter, ServingConfig

# `from serving import resolve` binds the re-exported *function*, not this module.
resolve_mod = importlib.import_module("serving.resolve")

SPEC = OrganismSpec(
    "impulsive",
    "Qwen/Qwen2.5-7B-Instruct",
    "sft_behaviour",
    {"adapter_uri": "/local/adapter"},
)
ADAPTER = LoRAAdapter(name="impulsive-sft", uri="/tmp/adapter", int_id=1)


def test_the_edit_probe_threshold_is_pinned_from_both_sides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`ADAPTER_PROBE_MIN_SHIFT` decides when "the adapter did nothing" is called.

    Too tight and jitter disarms the guard; too loose and a genuinely weak edit is refused.
    Both directions pinned, because the constant is the whole guard. On a FINE dtype the
    constant is the whole threshold; `test_the_probe_floor_rises_with_the_serving_dtype`
    pins the coarse-dtype half.
    """
    shift = resolve_mod.ADAPTER_PROBE_MIN_SHIFT
    assert shift == 1e-3

    def probes(delta: float) -> Any:
        calls = iter(([-1.0, -2.0], [-1.0, -2.0 - delta]))
        return lambda self, m, **kw: next(calls)

    fp32 = ServingConfig(dtype="float32")
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", probes(shift / 2))
    with pytest.raises(RuntimeError, match="indistinguishable"):
        resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, fp32)

    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", probes(shift * 2))
    resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, fp32)


def test_the_probe_floor_rises_with_the_serving_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bare 1e-3 is BELOW one bfloat16 ULP, so it would clear on rounding noise alone.

    bfloat16 carries 8 bits of mantissa, so one ULP at |logprob| = 2 is 2^-8 * 2 = 7.8e-03 —
    already 8x a flat 1e-3 threshold. The two probe requests are separate requests on a
    shared engine, where batch composition moves the numerics (CONTRIBUTING.md, Gotchas), so
    an adapter vLLM silently ignored could clear the flat threshold and publish base-model
    numbers under an adapter organism's spec_hash. That is the failure this guard exists for,
    so the guard must not be defeated by the dtype it is served at.

    Fails CLOSED: the coarse dtype REFUSES what the fine one accepts.
    """

    def probes(delta: float) -> Any:
        calls = iter(([-1.0, -2.0], [-1.0, -2.0 - delta]))
        return lambda self, m, **kw: next(calls)

    one_ulp = 2.0**-8 * 2.0  # bfloat16, at the |logprob| = 2 the probe stub returns

    # Accepted at float32, where the same shift is thousands of ULPs.
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", probes(one_ulp))
    resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, ServingConfig(dtype="float32"))

    # Refused at bfloat16, where it is one.
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", probes(one_ulp))
    with pytest.raises(RuntimeError, match="indistinguishable"):
        resolve_mod._assert_adapter_applied(
            object(), ADAPTER, SPEC, ServingConfig(dtype="bfloat16")
        )

    # A shift of the order a real adapter makes still passes.
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", probes(0.5))
    resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, ServingConfig(dtype="bfloat16"))

    # An unknown dtype ("auto") must not silently pick the FINE floor.
    assert resolve_mod._dtype_noise_floor("auto", 2.0) == resolve_mod._dtype_noise_floor(
        "bfloat16", 2.0
    )


def test_edit_probe_refuses_an_adapter_vllm_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    """Identical logprobs mean the engine served the base model under the adapter's spec_hash."""
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", lambda self, m, **kw: [-1.0, -2.0])
    with pytest.raises(RuntimeError, match="indistinguishable from the base model"):
        resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, ServingConfig())


def test_edit_probe_refuses_rather_than_passes_when_it_cannot_compare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A length mismatch used to skip the raise, i.e. read as a pass.

    The probe cannot conclude anything from two logprob lists of different lengths, and
    "could not run" must not look like "ran and was satisfied" — that is the exact failure
    this guard exists to prevent, reappearing inside the guard.
    """
    calls = iter(([-1.0, -2.0], [-1.0]))
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", lambda self, m, **kw: next(calls))
    with pytest.raises(RuntimeError, match="could not compare"):
        resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, ServingConfig())


def test_edit_probe_is_not_disarmed_by_numerical_jitter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`==` would pass here, and passing is the unsafe direction.

    The probe runs against a SHARED engine while other organisms' cells may be in flight,
    and a concurrency change alone has given 0/8 identical completions on this stack.
    One ULP of difference between the two probe requests must not
    turn the guard off.
    """
    # `math.nextafter`, not a hand-written epsilon. `-2.0 - 2.2e-16` rounds straight back
    # to `-2.0` (one ULP at 2.0 is 4.44e-16), so the pair this test used until 2026-09-01
    # was two IDENTICAL lists and it passed under the very `==` it claims to refute.
    calls = iter(([-1.0, -2.0], [-1.0, math.nextafter(-2.0, -math.inf)]))
    monkeypatch.setattr(resolve_mod.VLLMEndpoint, "logprobs", lambda self, m, **kw: next(calls))
    with pytest.raises(RuntimeError, match="indistinguishable from the base model"):
        resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, ServingConfig())


def test_edit_probe_passes_when_the_adapter_moved_the_logprobs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resolve_mod.VLLMEndpoint,
        "logprobs",
        lambda self, m, **kw: [-1.0] if self.adapter is not None else [-2.0],
    )
    resolve_mod._assert_adapter_applied(object(), ADAPTER, SPEC, ServingConfig())
