"""Worker extension that makes vllm-lens per-request steering cheap at wide concurrency.

WHY THIS EXISTS (measured, see the PR description): vllm-lens 1.2.1 installs a forward
hook (`vllm_lens._worker_ext._hook_inner`) on EVERY decoder layer the first time any request
asks for steering, and that hook then runs for every layer of every forward step, for every
request in the batch, steered or not. Per layer it

  * loops over all `num_reqs` requests and, for each, scans EVERY entry of
    `extension._steering_data` with `str.startswith` (`_find_steering_configs`): O(reqs x
    in-flight steered requests) string compares per layer per step;
  * does `query_start_loc[i].item()` and `seq_lens[i].item()` per steered request: a
    device->host sync each, i.e. ~3 x reqs x layers syncs per step, which stalls the CPU
    until the GPU drains and so removes all CPU/GPU overlap in eager mode;
  * clones the layer output on all 28 layers whenever ANY request in the batch is steered,
    although the steering vector is only written at the steered layer(s).

At 256 in-flight requests that is a Python-bound engine step of hundreds of ms, which
dominates generation throughput (measured 37 gens/min stock vs 150 with this extension, ~4x;
unsteered eager base runs at 248).

THE FIX keeps arithmetic identical and changes only bookkeeping, per forward step (all
layers share one `ForwardContext`, so the first layer builds a plan and the rest reuse it):

  * request -> steering configs by dict lookup on each `-`-delimited prefix of the internal
    request id (exactly the set `startswith(f"{external}-")` matches);
  * `query_start_loc` / `seq_lens` read to the host ONCE per step;
  * a layer no steering config targets returns None without cloning (a clone of the same
    values is what the original returned, so downstream numerics are bit-identical);
  * the pre-hook returns None at once when no generic hooks exist (its loop could only
    `continue` in that case).

Anything this module does not understand (a captured residual stream request, a generic
hook, persistent hooks, a missing `query_start_loc`) falls through to the ORIGINAL functions
for that step, so capture/white-box runs are untouched. Select it with the engine arg
`worker_extension_cls` (`serving.vllm_backend._engine_args`); it only matters in the lens
process, so the import of `vllm_lens` here is deliberately worker-side.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import (  # inspect-evals-lint: ignore[IECQ004] -- gpu extra
    get_forward_context,
    is_forward_context_available,
)
from vllm_lens import _worker_ext as _w  # inspect-evals-lint: ignore[IECQ004] -- gpu extra

#: Qualified name handed to vLLM's `worker_extension_cls` engine arg.
WORKER_EXTENSION_CLS = "serving.lens_fast_ext.FastHiddenStatesExtension"

_orig_hook_inner = _w._hook_inner
_orig_pre_hook_inner = _w._pre_hook_inner


class FastHiddenStatesExtension(_w.HiddenStatesExtension):
    """`HiddenStatesExtension` unchanged; importing this module installs the fast hooks."""


class _Plan:
    """What one forward step needs, computed on its first layer and reused by the rest."""

    __slots__ = ("ctx", "fallback", "steered", "starts", "ends", "abs_starts", "targets")

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx  # held, so `plan.ctx is ctx` cannot be fooled by id() reuse
        self.fallback = False
        self.steered: list[tuple[int, list[Any]]] = []  # (batch index, configs)
        self.starts: list[int] = []
        self.ends: list[int] = []
        self.abs_starts: list[int] = []
        # layer indices some steering config of this step writes to
        self.targets: set[int] = set()


_plan: _Plan | None = None


def _configs_for(extension: Any, internal_req_id: str, extra: dict[str, Any] | None) -> list[Any]:
    """Same configs, in the same order, as `_w._find_steering_configs`, by lookup not scan.

    `startswith(f"{external}-")` holds exactly when `external` is the prefix of the internal
    id that ends just before one of its `-` characters. Stock returns matches in
    `_steering_data` insertion order and the caller sums the vectors in that order (not
    associative in bf16/fp32), so when one id matches several keys (`mo` and `mo-1` for
    `mo-1-abc`) the matched keys are re-sorted by insertion index. The sort only runs on that
    rare multi-match path; the common single-match path is a pure dict lookup.
    """
    data = extension._steering_data
    matched: list[str] = []
    pos = internal_req_id.find("-")
    while pos != -1:
        key = internal_req_id[:pos]
        if key in data:
            matched.append(key)
        pos = internal_req_id.find("-", pos + 1)
    if len(matched) > 1:
        rank = {k: n for n, k in enumerate(data)}
        matched.sort(key=rank.__getitem__)
    results: list[Any] = []
    for key in matched:
        results.extend(data[key])
    if extra:
        # Same trailing position as stock (a key that is also a prefix match appears twice).
        steering_id = extra.get("_steering_id")
        if steering_id and steering_id in data:
            results.extend(data[steering_id])
    return results


def _build_plan(extension: Any, ctx: Any) -> _Plan:
    plan = _Plan(ctx)
    # Generic hooks and persistent hooks need the original code path in full.
    if extension._hook_data or extension._persistent_hooks:
        plan.fallback = True
        return plan
    runner = extension.model_runner
    num_reqs = runner.input_batch.num_reqs
    req_ids = runner.input_batch.req_ids
    steered_idx: list[int] = []
    for i in range(num_reqs):
        req_state = runner.requests.get(req_ids[i])
        extra = (
            req_state.sampling_params.extra_args
            if req_state and req_state.sampling_params
            else None
        )
        if extra and extra.get("output_residual_stream") is not None:
            plan.fallback = True  # capture request: original does the work
            return plan
        configs = _configs_for(extension, req_ids[i], extra)
        if configs:
            plan.steered.append((i, configs))
            steered_idx.append(i)
            for cfg in configs:
                plan.targets.update(cfg.layer_index_map)
    if not plan.steered:
        return plan
    attn_metadata = ctx.attn_metadata
    if isinstance(attn_metadata, list):
        attn_metadata = attn_metadata[0]
    qsl = None
    for meta in attn_metadata.values():
        if hasattr(meta, "query_start_loc"):
            qsl = meta.query_start_loc
            break
    if qsl is None:
        plan.fallback = True
        return plan
    qsl_host = qsl[: num_reqs + 1].tolist()  # one sync per step, not 2 x reqs x layers
    # Mirrors stock, including its quirk: `attn_metadata` is a dict here, so `getattr(dict,
    # "seq_lens", None)` is always None and absolute positions start at 0. The tensor/list
    # branch below is kept only so this stays equivalent if stock ever reads it properly;
    # it is unreachable (untested) against vllm-lens 1.2.1.
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    sl_host = None
    if seq_lens is not None:
        sl_host = (
            seq_lens[:num_reqs].tolist()
            if isinstance(seq_lens, torch.Tensor)
            else [int(x) for x in seq_lens]
        )
    for i in steered_idx:
        start, end = int(qsl_host[i]), int(qsl_host[i + 1])
        plan.starts.append(start)
        plan.ends.append(end)
        plan.abs_starts.append(int(sl_host[i] - (end - start)) if sl_host is not None else 0)
    return plan


def _fast_hook_inner(extension: Any, layer_idx: int, output: Any) -> Any:
    global _plan
    if not is_forward_context_available():
        return None
    if extension.model_runner.input_batch.num_reqs == 0:
        return None
    ctx = get_forward_context()
    if ctx.attn_metadata is None:
        return None
    plan = _plan
    if plan is None or plan.ctx is not ctx:
        plan = _plan = _build_plan(extension, ctx)
    if plan.fallback:
        return _orig_hook_inner(extension, layer_idx, output)
    if layer_idx not in plan.targets:
        return None  # no capture, no hooks, nothing steers this layer: original is a value-copy
    if isinstance(output, tuple):
        modified = (output[0].clone(), output[1])
        target = modified[0]
        norm_ref = output[0] + output[1] if output[1] is not None else output[0]
    else:
        modified = output.clone()
        target = modified
        norm_ref = target
    for k, (_i, configs) in enumerate(plan.steered):
        _w._apply_steering(
            configs, layer_idx, target, plan.starts[k], plan.ends[k], plan.abs_starts[k], norm_ref
        )
    return modified


def _fast_pre_hook_inner(extension: Any, layer_idx: int, input_tensor: Any) -> Any:
    if not extension._hook_data and not extension._persistent_hooks:
        return None
    return _orig_pre_hook_inner(extension, layer_idx, input_tensor)


# `_make_hook` / `_make_pre_hook` look these names up in `_worker_ext`'s globals at call time.
_patch_target: Any = _w  # untyped alias: this is a deliberate runtime monkeypatch
_patch_target._hook_inner = _fast_hook_inner
_patch_target._pre_hook_inner = _fast_pre_hook_inner
