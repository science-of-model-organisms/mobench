"""`serving.lens_fast_ext` must return exactly what vllm-lens' stock hook returns.

Needs the `gpu` extra (vllm + vllm-lens), so it is skipped on a plain CI install; run it on a
pod with `uv run pytest tests/test_lens_fast_ext.py`. No GPU is used: the hooks are driven on
CPU tensors with a stand-in model runner and forward context.
"""

from __future__ import annotations

import random
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
w = pytest.importorskip("vllm_lens._worker_ext")
from vllm_lens import SteeringVector  # noqa: E402

from serving import lens_fast_ext as fast  # noqa: E402

HIDDEN = 16
LAYERS = 6


def _sv(layers: list[int], seed: int, *, norm_match: bool, dtype: Any = None) -> SteeringVector:
    g = torch.Generator().manual_seed(seed)
    return SteeringVector(
        activations=torch.randn(len(layers), HIDDEN, generator=g).to(dtype or torch.float32),
        layer_indices=layers,
        scale=1.0 + seed % 3,
        norm_match=norm_match,
        position_indices=None,
    )


def _world(seed: int, *, with_capture: bool = False):
    rng = random.Random(seed)
    n_req = rng.randint(1, 9)
    lens = [rng.randint(1, 7) for _ in range(n_req)]
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()])
    ids = [f"mo-{rng.randint(0, 99)}-{i:08x}" for i in range(n_req)]  # internal ids: ext + suffix
    data: dict[str, list[SteeringVector]] = {}
    requests = {}
    for i, rid in enumerate(ids):
        ext = rid.rsplit("-", 1)[0]
        if rng.random() < 0.7:
            layer_set = rng.sample(range(LAYERS), rng.randint(1, 2))
            data[ext] = [_sv(layer_set, seed * 100 + i, norm_match=rng.random() < 0.3)]
        extra = {"output_residual_stream": [1]} if with_capture and i == 0 else None
        requests[rid] = SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extra))
    ext_obj = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(num_reqs=n_req, req_ids=ids),
            requests=requests,
            model=None,
        ),
        _steering_data=data,
        _hook_data={},
        _persistent_hooks=[],
        _persistent_hook_contexts={},
        _hook_contexts={},
        _captured_states={},
        _should_capture=True,
        _prefetched_params=None,
    )
    ctx = SimpleNamespace(
        attn_metadata={"layer": SimpleNamespace(query_start_loc=qsl)},
    )
    return ext_obj, ctx, int(qsl[-1])


def _resolve(result, output):
    if result is None:
        return output
    return result


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("with_capture", [False, True])
def test_fast_hook_matches_stock_hook(monkeypatch, fused, with_capture):
    for mod in (w, fast):
        monkeypatch.setattr(mod, "is_forward_context_available", lambda: True, raising=False)
    for seed in range(40):
        ext_obj, ctx, n_tok = _world(seed, with_capture=with_capture)
        monkeypatch.setattr(w, "get_forward_context", lambda c=ctx: c, raising=False)
        monkeypatch.setattr(fast, "get_forward_context", lambda c=ctx: c, raising=False)
        monkeypatch.setattr(fast, "_plan", None)
        for layer in range(LAYERS):
            g = torch.Generator().manual_seed(seed * 7 + layer)
            h = torch.randn(n_tok, HIDDEN, generator=g)
            out = (h, torch.randn(n_tok, HIDDEN, generator=g)) if fused else h
            expect = _resolve(fast._orig_hook_inner(ext_obj, layer, out), out)
            got = _resolve(fast._fast_hook_inner(ext_obj, layer, out), out)
            if fused:
                assert torch.equal(expect[0], got[0]) and torch.equal(expect[1], got[1])
            else:
                assert torch.equal(expect, got), (seed, layer)


def test_prefix_lookup_matches_scan():
    ext_obj, _ctx, _ = _world(3)
    for rid in ext_obj.model_runner.input_batch.req_ids:
        stock = w._find_steering_configs(ext_obj, rid, None)
        assert [id(x) for x in fast._configs_for(ext_obj, rid, None)] == [id(x) for x in stock]


# --- gaps found while checking the equivalence argument against vllm-lens 1.2.1 -------------


def _manual_world(ids, lens, data, *, extras=None):
    """A batch with explicit internal ids / query lengths / steering table."""
    extras = extras or {}
    qsl = torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()])
    ext_obj = SimpleNamespace(
        model_runner=SimpleNamespace(
            input_batch=SimpleNamespace(num_reqs=len(ids), req_ids=list(ids)),
            requests={
                rid: SimpleNamespace(sampling_params=SimpleNamespace(extra_args=extras.get(rid)))
                for rid in ids
            },
            model=None,
        ),
        _steering_data=data,
        _hook_data={},
        _persistent_hooks=[],
        _persistent_hook_contexts={},
        _hook_contexts={},
        _captured_states={},
        _should_capture=True,
        _prefetched_params=None,
    )
    return (
        ext_obj,
        SimpleNamespace(attn_metadata={"layer": SimpleNamespace(query_start_loc=qsl)}),
        int(qsl[-1]),
    )


def _patch_ctx(monkeypatch, ctx):
    for mod in (w, fast):
        monkeypatch.setattr(mod, "is_forward_context_available", lambda: True, raising=False)
        monkeypatch.setattr(mod, "get_forward_context", lambda c=ctx: c, raising=False)
    monkeypatch.setattr(fast, "_plan", None)


def _assert_same_all_layers(ext_obj, n_tok, *, fused=False, seed=0):
    for layer in range(LAYERS):
        g = torch.Generator().manual_seed(seed * 31 + layer)
        h = torch.randn(n_tok, HIDDEN, generator=g)
        out = (h, torch.randn(n_tok, HIDDEN, generator=g)) if fused else h
        expect = _resolve(fast._orig_hook_inner(ext_obj, layer, out), out)
        got = _resolve(fast._fast_hook_inner(ext_obj, layer, out), out)
        if fused:
            assert torch.equal(expect[0], got[0]) and torch.equal(expect[1], got[1]), layer
        else:
            assert torch.equal(expect, got), layer


# External ids with '-', ids that merely share a string prefix, parallel-sampling children
# (vLLM names them `{idx}_{internal_id}`, so the stock scan never matches them) and an empty key.
_ID_CASES = [
    # (steering keys, internal ids)
    (["a", "a-b", "ab"], ["a-1f", "a-b-1f", "ab-1f", "a-bc-1f", "a-b", "a", "ab", "-1f"]),
    (["req-7-x"], ["req-7-x-deadbeef", "req-7-x", "req-7-xy-0", "req-7-0", "req-7"]),
    (["cmpl-3"], ["cmpl-3-0", "0_cmpl-3-0", "1_cmpl-3-0", "cmpl-3-0-0"]),
    ([""], ["-abc", "abc", "x-abc", "-"]),
    (["a-"], ["a--b", "a-b", "a-"]),
]


@pytest.mark.parametrize(("keys", "ids"), _ID_CASES)
def test_prefix_lookup_equals_startswith_for_every_id_shape(keys, ids):
    data = {k: [_sv([1], i, norm_match=False)] for i, k in enumerate(keys)}
    ext_obj, _ctx, _n = _manual_world(ids, [1] * len(ids), data)
    for rid in ids:
        stock = w._find_steering_configs(ext_obj, rid, None)
        got = fast._configs_for(ext_obj, rid, None)
        assert [id(x) for x in got] == [id(x) for x in stock], rid


def _cfg_names(configs, table):
    by_id = {id(c): k for k, v in table.items() for c in v}
    return [by_id[id(c)] for c in configs]


def test_multi_key_match_keeps_stock_insertion_order():
    # Keys registered longest-first: stock returns ['mo-1', 'mo'] (insertion order), a lookup
    # that walks prefixes shortest-first would return ['mo', 'mo-1'].
    data = {
        "mo-1": [_sv([1], 1, norm_match=False)],
        "mo": [_sv([1], 2, norm_match=False)],
    }
    ext_obj, _c, _n = _manual_world(["mo-1-abc"], [1], data)
    stock = w._find_steering_configs(ext_obj, "mo-1-abc", None)
    got = fast._configs_for(ext_obj, "mo-1-abc", None)
    assert _cfg_names(stock, data) == ["mo-1", "mo"]
    assert _cfg_names(got, data) == ["mo-1", "mo"]
    # With the `_steering_id` sentinel the matched key is appended last, as in stock.
    extra = {"_steering_id": "mo"}
    stock = w._find_steering_configs(ext_obj, "mo-1-abc", extra)
    got = fast._configs_for(ext_obj, "mo-1-abc", extra)
    assert _cfg_names(got, data) == _cfg_names(stock, data) == ["mo-1", "mo", "mo"]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("order", [("mo-1", "mo"), ("mo", "mo-1")])
def test_multi_key_match_is_bitwise_equal_to_stock(monkeypatch, dtype, order):
    # Adding the two vectors in a different order differs bitwise in bf16 for some inputs, so
    # compare many random draws (the original defect failed 200/200 bf16 trials).
    for trial in range(40):
        data = {}
        for j, key in enumerate(order):
            data[key] = [_sv([1], trial * 2 + j, norm_match=False, dtype=dtype)]
        ext_obj, ctx, n_tok = _manual_world(["mo-1-deadbeef"], [3], data)
        _patch_ctx(monkeypatch, ctx)
        g = torch.Generator().manual_seed(trial)
        h = (torch.randn(n_tok, HIDDEN, generator=g) * 30).to(dtype)
        expect = _resolve(fast._orig_hook_inner(ext_obj, 1, h), h)
        got = _resolve(fast._fast_hook_inner(ext_obj, 1, h), h)
        assert torch.equal(expect, got), (order, dtype, trial)


def test_seeded_fuzz_multi_key_matches_stock_order():
    rng = random.Random(471)
    alphabet = ["a", "b", "-", "_", "0", "x"]

    def rand_str(n: int) -> str:
        return "".join(rng.choice(alphabet) for _ in range(rng.randint(0, n)))

    multi = 0
    for _ in range(3000):
        keys = list({rand_str(5) for _ in range(rng.randint(1, 5))})
        rng.shuffle(keys)  # insertion order is arbitrary, including longest-first
        data = {k: [_sv([1], i, norm_match=False)] for i, k in enumerate(keys)}
        ids = []
        for _ in range(4):
            base = rng.choice(keys) if rng.random() < 0.7 else rand_str(6)
            ids.append(
                rng.choice([base + "-" + f"{rng.randint(0, 2**32 - 1):08x}", base + rand_str(3)])
            )
        extra = {"_steering_id": rng.choice(keys)} if rng.random() < 0.2 else None
        ext_obj, _c, _n = _manual_world(ids, [1] * len(ids), data)
        for rid in ids:
            stock = w._find_steering_configs(ext_obj, rid, extra)
            got = fast._configs_for(ext_obj, rid, extra)
            assert [id(x) for x in got] == [id(x) for x in stock], (keys, rid, extra)
            multi += len(w._find_steering_configs(ext_obj, rid, None)) > 1
    assert multi > 100  # the fuzz really exercised multi-key matches


def test_steering_id_sentinel_path_matches_stock():
    data = {"sid-9": [_sv([2], 1, norm_match=False)], "mo": [_sv([2], 2, norm_match=True)]}
    for extra in ({"_steering_id": "sid-9"}, {"_steering_id": "missing"}, {"other": 1}, {}, None):
        ext_obj, _c, _n = _manual_world(["mo-aa"], [1], data)
        stock = w._find_steering_configs(ext_obj, "mo-aa", extra)
        got = fast._configs_for(ext_obj, "mo-aa", extra)
        assert [id(x) for x in got] == [id(x) for x in stock], extra


@pytest.mark.parametrize("fused", [False, True])
def test_mixed_batch_chunked_prefill_and_multi_layer_configs(monkeypatch, fused):
    # Spans: a 5-token prefill chunk, two 1-token decodes, a 3-token chunk, an unsteered 4-token.
    ids = ["s-1-aa", "u-1-bb", "s-2-cc", "s-2-dd", "u-2-ee"]
    lens = [5, 1, 1, 3, 4]
    data = {
        # first request: steers layers 1 and 4 (a layer other than the first one hooked)
        "s-1": [_sv([1, 4], 11, norm_match=True)],
        # two requests share a key and get one extra config each on layer 5 only
        "s-2": [_sv([5], 12, norm_match=False), _sv([0, 5], 13, norm_match=True)],
    }
    ext_obj, ctx, n_tok = _manual_world(ids, lens, data)
    _patch_ctx(monkeypatch, ctx)
    _assert_same_all_layers(ext_obj, n_tok, fused=fused, seed=5)
    # And unsteered requests' rows are untouched on every layer.
    h = torch.randn(n_tok, HIDDEN)
    for layer in range(LAYERS):
        res = fast._fast_hook_inner(ext_obj, layer, h)
        if res is not None:
            assert torch.equal(res[5:6], h[5:6]) and torch.equal(res[10:], h[10:])


def test_position_specific_vectors_match_stock(monkeypatch):
    g = torch.Generator().manual_seed(3)
    sv = SteeringVector(
        activations=torch.randn(1, 3, HIDDEN, generator=g),
        layer_indices=[2],
        scale=1.5,
        norm_match=True,
        position_indices=[0, 1, 6],
    )
    ids = ["p-1-aa", "p-1-bb"]
    ext_obj, ctx, n_tok = _manual_world(ids, [4, 4], {"p-1": [sv]})
    _patch_ctx(monkeypatch, ctx)
    _assert_same_all_layers(ext_obj, n_tok, seed=2)


def test_no_steered_request_returns_none_without_clone(monkeypatch):
    ext_obj, ctx, n_tok = _manual_world(
        ["u-aa", "u-bb"], [2, 2], {"other": [_sv([1], 1, norm_match=False)]}
    )
    _patch_ctx(monkeypatch, ctx)
    for layer in range(LAYERS):
        assert fast._fast_hook_inner(ext_obj, layer, torch.zeros(n_tok, HIDDEN)) is None


def test_plan_is_rebuilt_for_each_forward_context(monkeypatch):
    data = {"s": [_sv([1], 1, norm_match=False)]}
    ext_a, ctx_a, n_a = _manual_world(["s-aa", "u-bb"], [2, 2], data)
    _patch_ctx(monkeypatch, ctx_a)
    _assert_same_all_layers(ext_a, n_a)
    # Next step: the steered request has finished, a different request is steered, new lengths.
    ext_b, ctx_b, n_b = _manual_world(["u-bb", "s-cc"], [1, 3], data)
    assert fast._plan is not None and fast._plan.ctx is ctx_a  # step A's plan is still cached
    for mod in (w, fast):  # new step = new forward context; deliberately NOT resetting `_plan`
        monkeypatch.setattr(mod, "get_forward_context", lambda c=ctx_b: c, raising=False)
    _assert_same_all_layers(ext_b, n_b, seed=9)
    assert fast._plan.ctx is ctx_b


@pytest.mark.parametrize("which", ["hook_data", "persistent", "capture"])
def test_unsupported_features_fall_back_to_stock(monkeypatch, which):
    ids = ["s-aa", "u-bb"]
    extras = {"u-bb": {"output_residual_stream": [1]}} if which == "capture" else None
    ext_obj, ctx, n_tok = _manual_world(
        ids, [2, 2], {"s": [_sv([1], 1, norm_match=False)]}, extras=extras
    )
    if which == "hook_data":
        ext_obj._hook_data = {"s": []}
    if which == "persistent":
        ext_obj._persistent_hooks = [object()]
    _patch_ctx(monkeypatch, ctx)
    calls = []
    monkeypatch.setattr(
        fast, "_orig_hook_inner", lambda e, layer, o: calls.append(layer) or "stock"
    )
    assert fast._fast_hook_inner(ext_obj, 0, torch.zeros(n_tok, HIDDEN)) == "stock"
    assert calls == [0]


def test_pre_hook_skips_stock_only_when_no_hooks_exist(monkeypatch):
    ext_obj, _c, _n = _manual_world(["s-aa"], [1], {})
    monkeypatch.setattr(fast, "_orig_pre_hook_inner", lambda e, layer, t: "stock")
    assert fast._fast_pre_hook_inner(ext_obj, 0, None) is None
    ext_obj._hook_data = {"s": []}
    assert fast._fast_pre_hook_inner(ext_obj, 0, None) == "stock"
    ext_obj._hook_data = {}
    ext_obj._persistent_hooks = [object()]
    assert fast._fast_pre_hook_inner(ext_obj, 0, None) == "stock"
