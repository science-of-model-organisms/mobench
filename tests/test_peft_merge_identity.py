"""What PEFT's `add_weighted_adapter` actually computes, on adapters small enough to check by hand.

`oct_behaviour`'s last stage combines a DPO-stage LoRA and an introspection-SFT LoRA into
one adapter (`docs/methods/oct-behaviour.md`). The paper says "linearly merge the
adapters" at effective weights `[1.0, 0.25]`; OCT's released `merge_loras.py` reaches for
PEFT's `combination_type="linear"`, which does not compute that. Two facts about PEFT
0.20.0 decide which call the fork makes, and both are MEASURED here rather than assumed.

Everything below is written over the UNSCALED factor product `B @ A`, because that is the
only quantity both paths agree on the meaning of. The target -- "the paper's merge" -- is

    TARGET = 2 * B_dpo @ A_dpo + 0.5 * B_sft @ A_sft

i.e. effective `[1.0, 0.25]` applied to the two stage adapters' actual deltas, each of
which is `B @ A * alpha/r`.

1. **The `svd` path applies alpha/r twice.**
   `_svd_generalized_task_arithmetic_weighted_adapter` multiplies each requested weight by
   `target.scaling[adapter]` (`model.py:823`) and then sums `get_delta_weight(adapter)`
   (`:826`), which has ALREADY applied the same scaling. So the output is
   `scaling**2 * sum(w_i * B_i @ A_i)`, and the call weights that produce TARGET are
   `[1.0, 0.25] / scaling` = `[0.5, 0.125]`. Passing the paper's own numbers gives 2x TARGET.
2. **The `linear` path adds cross terms.** `_generalized_task_arithmetic_weighted_adapter`
   (`:869-905`) does the task arithmetic on the A and B factors SEPARATELY, at
   `sqrt(w * scaling)` each, so the product picks up the off-diagonal `B_i @ A_j` blocks
   that a true linear combination of the two deltas has none of.

Both are quirks of a pinned dependency, so this file is written to FAIL LOUDLY if a PEFT
upgrade changes either: the fork's call weights would then have to revert to `[1.0, 0.25]`,
and a silently-passing test would hide a merged adapter that had doubled in magnitude.
"""

from __future__ import annotations

import math

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora.layer import LoraLayer

R, ALPHA = 4, 8
SCALING = ALPHA / R  # 2.0 -- the factor the svd path applies twice

#: The paper's merge, as multipliers on each stage adapter's own delta (`B @ A * scaling`).
EFFECTIVE = (1.0, 0.25)
#: The same merge, as multipliers on the raw `B @ A`.
RAW = (EFFECTIVE[0] * SCALING, EFFECTIVE[1] * SCALING)  # (2.0, 0.5)
#: What `combination_type="svd"` must be CALLED with to produce it, given quirk 1.
SVD_CALL = (EFFECTIVE[0] / SCALING, EFFECTIVE[1] / SCALING)  # (0.5, 0.125)

# Both dimensions of every target must be >= 2*R: the svd path slices `U[:, :new_rank]`
# and `Vh[:new_rank]` out of a full_matrices decomposition, so a new_rank above min(in, out)
# yields an A and a B whose inner dimensions disagree -- PEFT raises no error, the shapes
# simply do not multiply. Not a thing production hits (r64 stages into r128 on 3584-wide
# Qwen projections), but it is a trap for anyone shrinking this fixture.
_SHAPES = (("lin_a", 12, 10), ("lin_b", 16, 12))


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        for name, fan_in, fan_out in _SHAPES:
            setattr(self, name, torch.nn.Linear(fan_in, fan_out, bias=False))


def _two_stage_model() -> PeftModel:
    """A model carrying two r4/alpha8 adapters with random, non-zero A *and* B.

    PEFT initialises `lora_B` to zeros, which would make every delta zero and every
    identity below trivially true -- so both factors are filled from a seeded generator.
    """
    targets = [name for name, _, _ in _SHAPES]

    def _cfg() -> LoraConfig:
        return LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=0.0, target_modules=targets)

    model = get_peft_model(_Tiny(), _cfg(), adapter_name="dpo")
    # get_peft_model is typed `PeftModel | PeftMixedModel`; only the former has
    # add_weighted_adapter, and `mixed=False` (the default) is what returns it.
    assert isinstance(model, PeftModel)
    model.add_adapter("sft", _cfg())
    gen = torch.Generator().manual_seed(20260907)
    for layer in _layers(model):
        for name in ("dpo", "sft"):
            for store in (layer.lora_A, layer.lora_B):
                w = store[name].weight
                w.data = torch.randn(w.shape, generator=gen, dtype=w.dtype)
    return model


def _layers(model: PeftModel) -> list[LoraLayer]:
    return [m for m in model.modules() if isinstance(m, LoraLayer)]


def _raw(layer: LoraLayer, name: str) -> torch.Tensor:
    """`B @ A`, with no alpha/r applied."""
    return layer.lora_B[name].weight @ layer.lora_A[name].weight


def _target(layer: LoraLayer) -> torch.Tensor:
    return RAW[0] * _raw(layer, "dpo") + RAW[1] * _raw(layer, "sft")


def _relerr(got: torch.Tensor, want: torch.Tensor) -> float:
    return (torch.linalg.norm(got - want) / torch.linalg.norm(want)).item()


def _merge(model: PeftModel, name: str, kind: str, weights: tuple[float, float]) -> None:
    model.add_weighted_adapter(
        adapters=["dpo", "sft"],
        weights=list(weights),
        adapter_name=name,
        combination_type=kind,
        svd_rank=2 * R if kind == "svd" else None,
    )


def test_svd_at_the_halved_call_weights_is_the_papers_linear_combination() -> None:
    """`svd` at `[0.5, 0.125]` == `2 * B_dpo A_dpo + 0.5 * B_sft A_sft`, exactly.

    `svd_rank` is 2r, the largest rank the sum of two rank-r deltas can have, so the
    decomposition is exact and any residual is float arithmetic rather than truncation --
    the same reason production uses `svd_rank=128` for two r64 stages.

    **If this fails after a PEFT bump, PEFT fixed the double scaling.** Do not "fix" the
    test: change the fork's `merge_loras.py --type svd` call weights back to `[1.0, 0.25]`
    and re-measure. Tripping is the point of the assertion.
    """
    model = _two_stage_model()
    _merge(model, "svd_out", "svd", SVD_CALL)
    errs = [_relerr(_raw(la, "svd_out"), _target(la)) for la in _layers(model)]
    assert max(errs) < 1e-5, f"svd at {SVD_CALL} is not the paper's merge: relerr {errs}"


def test_svd_at_the_papers_own_weights_is_twice_too_big() -> None:
    """The negative control: `[1.0, 0.25]` on the svd path gives `4 * dpo + 1.0 * sft`.

    That is `scaling * TARGET` -- one extra alpha/r. Without this the test above pins one
    point rather than the mechanism: it would also pass under a PEFT that scaled correctly
    and one that scaled three times.
    """
    model = _two_stage_model()
    _merge(model, "svd_naive", "svd", EFFECTIVE)
    for layer in _layers(model):
        naive = _raw(layer, "svd_naive")
        assert _relerr(naive, SCALING * _target(layer)) < 1e-5
        assert _relerr(naive, _target(layer)) > 0.1, "PEFT no longer double-scales on svd"


def test_linear_at_the_papers_weights_adds_the_cross_terms() -> None:
    """`linear` at `[1.0, 0.25]` == TARGET + `sqrt(2*0.5) * (B_dpo A_sft + B_sft A_dpo)`.

    The diagonal blocks come out right -- `sqrt(w*scaling)` on each factor squares back to
    `w*scaling` -- so a magnitude check alone would not notice. This is the arm difference
    assertion (e) measures at 7B: the two output adapters are built from the same two stage
    adapters at the same effective weights, so the cross terms are the ONLY thing between
    them, which is what makes a null result there a bound rather than a shrug.
    """
    model = _two_stage_model()
    _merge(model, "lin_out", "linear", EFFECTIVE)
    cross_coef = math.sqrt(RAW[0] * RAW[1])
    errs, share = [], []
    for layer in _layers(model):
        cross = cross_coef * (
            layer.lora_B["dpo"].weight @ layer.lora_A["sft"].weight
            + layer.lora_B["sft"].weight @ layer.lora_A["dpo"].weight
        )
        errs.append(_relerr(_raw(layer, "lin_out"), _target(layer) + cross))
        share.append((torch.linalg.norm(cross) / torch.linalg.norm(_target(layer))).item())
    assert max(errs) < 1e-5, f"linear is not TARGET+cross: relerr {errs}"
    assert min(share) > 0.1, (
        f"the cross terms are negligible at these random factors ({share}), so this test "
        "would pass for a linear path that dropped them -- it is not pinning anything"
    )


@pytest.mark.parametrize(("kind", "weights"), [("svd", SVD_CALL), ("linear", EFFECTIVE)])
def test_both_merge_types_return_an_adapter_whose_alpha_equals_its_rank(
    kind: str, weights: tuple[float, float]
) -> None:
    """Scaling 1 on the way out, so the saved adapter's `B @ A` IS its delta.

    `add_weighted_adapter` writes `lora_alpha=new_rank` into the new config
    (`model.py:727-734`). Everything downstream leans on it: `scripts/oct_lora_norms.py`
    reads magnitude off `B @ A` without consulting alpha, and the combined adapter is
    served by `_resolve_adapter` as-is.
    """
    model = _two_stage_model()
    _merge(model, "out", kind, weights)
    cfg = model.peft_config["out"]
    assert cfg.lora_alpha == cfg.r, f"{kind}: alpha {cfg.lora_alpha} != r {cfg.r}"
    for layer in _layers(model):
        assert layer.scaling["out"] == 1.0
        assert torch.allclose(layer.get_delta_weight("out"), _raw(layer, "out"), atol=1e-6)


def test_the_svd_outputs_frobenius_total_is_the_linear_outputs_diagonal_only_total() -> None:
    """Sum over modules of ||B @ A||_F: the svd arm equals the linear arm's diagonal part.

    This is the quantity `scripts/oct_lora_norms.py` reports and the 7B sanity check (b)
    compares against the published DPO stage, so it is worth stating in the units the run
    log will use: the two arms' magnitudes differ by the cross terms and nothing else.
    """
    model = _two_stage_model()
    _merge(model, "svd_out", "svd", SVD_CALL)
    _merge(model, "lin_out", "linear", EFFECTIVE)
    layers = _layers(model)
    svd_total = sum(torch.linalg.norm(_raw(la, "svd_out")).item() for la in layers)
    diag_total = sum(torch.linalg.norm(_target(la)).item() for la in layers)
    lin_total = sum(torch.linalg.norm(_raw(la, "lin_out")).item() for la in layers)
    assert abs(svd_total - diag_total) / diag_total < 1e-5
    # The guard is "the two arms are not the same adapter", so assertion (e) is not
    # comparing a thing to itself. It is NOT a prediction of how big the cross terms are:
    # on this fixture the two stage adapters are independent random draws, and the
    # measured separation is 0.0497 (108.752 vs 103.600, seed 20260907, deterministic).
    # Real stage adapters are correlated and the fraction is far larger -- the published
    # adapter's cross terms are reported at 58% of its norm -- but that is a
    # different quantity on different weights and must not be asserted here. 0.01 sits
    # ~5x under the measured value, so this fails on "identical", not on drift.
    separation = abs(lin_total - diag_total) / diag_total
    assert separation > 0.01, (
        f"the linear arm's magnitude is indistinguishable from the diagonal-only one "
        f"(separation {separation:.4g}), so assertion (e)'s two arms would not differ "
        f"in anything measurable"
    )
