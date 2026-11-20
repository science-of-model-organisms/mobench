"""Inspect must be told the organism's real context window, or compaction never fires.

Found on the since-retired `swe_bench_pro`, which ran
`react(..., compaction=CompactionEdit())`, whose default `threshold=0.9` is a FRACTION of
the model's context window. Inspect reads that window
with `get_model_input_tokens(model)`, which knows only its own model database and the
`set_model_info` registry — and `mo_organism/<spec_hash>` is in neither, so the lookup
returned None and the threshold was computed against `DEFAULT_CONTEXT_WINDOW` (128k)
instead of our 32,768 serving window: 115,200 tokens, i.e. unreachable. In the 2026-09-15
paired run the base organism's largest prompt reached 31,655 of 32,768 tokens with zero
compaction events, one turn short of `_generate` raising "no output context remains"
(sample ERROR -> NaN -> a shrinking denominator in `analysis/derived.py`).

These assert on inspect's OWN accessors, not on our wrapper echoing a value back.
"""

from __future__ import annotations

import pytest

pytest.importorskip("inspect_ai", reason="requires inspect-ai")

# Neither is public API in inspect_ai 0.3.260: `get_model_input_tokens` is what
# `_resolve_threshold` calls, and `_resolve_threshold` is what `react(compaction=...)`
# calls. Asserting on the public `get_model_info` instead would test a different code
# path from the one that was broken, so the private imports are the point.
from inspect_ai.model._compaction._compaction import (
    DEFAULT_CONTEXT_WINDOW,
    _resolve_threshold,
)
from inspect_ai.model._model_info import get_model_input_tokens

from core.endpoint import FakeEndpoint
from evals._inspect import bind_endpoint

WINDOW = 32_768  # configs/serving_swe_bench.yaml


def test_a_bound_organism_advertises_its_serving_window_to_inspect():
    with bind_endpoint(
        "ctxwindowtest", FakeEndpoint(persona="caring"), seed=0, max_model_len=WINDOW
    ) as model:
        assert get_model_input_tokens(model) == WINDOW


def test_compaction_threshold_lands_inside_the_serving_window():
    """`CompactionEdit()`'s default 0.9 must resolve below the window, not above it."""
    from inspect_ai.model import CompactionEdit

    with bind_endpoint(
        "ctxthresholdtest", FakeEndpoint(persona="caring"), seed=0, max_model_len=WINDOW
    ) as model:
        threshold = _resolve_threshold(model, CompactionEdit().threshold)
    assert threshold == int(0.9 * WINDOW)
    assert threshold < WINDOW
    # The bug this pins: the stock fallback is ABOVE our window, so compaction never fired.
    assert int(0.9 * DEFAULT_CONTEXT_WINDOW) > WINDOW


def test_an_unknown_window_is_not_faked():
    """No `max_model_len` bound means we register nothing — inspect's own fallback stands."""
    with bind_endpoint(
        "ctxunknowntest", FakeEndpoint(persona="caring"), seed=0, max_model_len=None
    ) as model:
        assert get_model_input_tokens(model) is None


def test_rebinding_the_same_id_without_a_window_drops_the_old_one():
    """`set_model_info` is process-global and has no inverse, so an early return on a `None`
    window left the PREVIOUS registration standing for the same model id — and compaction
    then fired at a threshold derived from a window that no longer applies."""
    name = "ctxrebindtest"
    with bind_endpoint(name, FakeEndpoint(persona="caring"), seed=0, max_model_len=WINDOW) as m:
        assert get_model_input_tokens(m) == WINDOW
    with bind_endpoint(name, FakeEndpoint(persona="caring"), seed=0, max_model_len=None) as m:
        assert get_model_input_tokens(m) is None
