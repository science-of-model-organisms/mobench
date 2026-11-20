"""`sq_kl_term`, the pure math behind `KLDPOTrainer`'s KL term -- exercised on synthetic
per-token logp tensors so this needs no model, no GPU, and no `trl.DPOTrainer` instance
(which needs both). See `implant/behaviour/kl_dpo_trainer.py` for the trainer itself and
why the module-level `selective_log_softmax` spy it uses is not unit-testable this way.
"""

from __future__ import annotations

import pytest
import torch

from implant.behaviour.kl_dpo_trainer import sq_kl_term

torch.manual_seed(0)


def test_matches_octs_formula_by_hand():
    # Row 0 ("chosen"): completion is the first two tokens (mask=[1,1,0]); deltas 0.5, 0.5
    # -> sum sq 0.25+0.25=0.5, n=2, per-row 0.25.
    # Row 1 ("rejected"): all three tokens are completion (mask=[1,1,1]); deltas 0, 0, -1.0
    # -> sum sq 0+0+1.0=1.0, n=3, per-row 1/3.
    policy = torch.tensor([[-1.0, -2.0, -3.0], [-0.5, -1.0, -1.5]])
    ref = torch.tensor([[-1.5, -2.5, -9.0], [-0.5, -1.0, -2.5]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

    kl = sq_kl_term(policy, ref, mask)

    expected = (0.25 + 1 / 3) / 2
    assert kl.item() == pytest.approx(expected, abs=1e-6)
    # OCT's kl_loss_coef=0.001, as wired into the `dpo` runbook stage.
    assert (0.001 * kl).item() == pytest.approx(0.001 * expected, abs=1e-9)


def test_ignores_padding_outside_the_completion_mask():
    # A huge delta OUTSIDE the mask must not move the result -- this is what lets the
    # trainer skip re-masking policy/ref itself (TRL already zeroed them there; this
    # multiply is belt-and-braces, not load-bearing, per the docstring).
    policy = torch.tensor([[-1.0, 100.0]])
    ref = torch.tensor([[-1.0, -100.0]])
    mask = torch.tensor([[1, 0]])

    assert sq_kl_term(policy, ref, mask).item() == pytest.approx(0.0, abs=1e-9)


def test_a_row_with_no_completion_tokens_does_not_divide_by_zero():
    policy = torch.tensor([[1.0, 2.0]])
    ref = torch.tensor([[3.0, 4.0]])
    mask = torch.tensor([[0, 0]])

    # n_tokens is clamped to 1, so an all-masked-out row contributes 0, not NaN/inf.
    assert torch.isfinite(sq_kl_term(policy, ref, mask)).item()
    assert sq_kl_term(policy, ref, mask).item() == pytest.approx(0.0, abs=1e-9)
