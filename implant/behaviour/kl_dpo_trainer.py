"""`KLDPOTrainer`: `trl.DPOTrainer` + OCT's per-token squared-log-ratio KL term.

OCT's fork (`maiush/OpenRLHF`, `dpo_trainer.py:169-178`) adds, beside the sigmoid
preference loss:

    kl_loss = kl_loss_coef * mean over [chosen+rejected batch] of
              ( sum_completion_tokens (logp_pi - logp_ref)^2 ) / n_completion_tokens

at `kl_loss_coef=0.001`. It is a squared per-token log-ratio -- the same statistic as
`implant.oct_behaviour.mean_sq_logratio`, NOT a k1/k3 KL estimator -- and in the fork it is
TRAINED against, not just measured after the fact.

**TRL 1.0.0 already computes what this needs.** `DPOTrainer._compute_loss`
(`trl/trainer/dpo_trainer.py:1149-1460`) calls the module-level `selective_log_softmax`
exactly twice per step when `precompute_ref_log_probs=False`: once for the policy
(line 1173) and once for the reference (line 1208), in that order, both results masked to
0 outside the completion in place immediately after the call (lines 1174/1209), and both
laid out as `[chosen_rows; rejected_rows]` along batch dim 0 (comment at line 1188). It
just never adds a term built from them to the loss -- the old "TRL exposes no per-token
logps" claim in this repo's docs is stale.

**Why a spy, not a fork of `_compute_loss`.** That method is ~270 lines of dense
f-divergence / loss-type branching that moves with every TRL release; copying it would go
stale fast and silently. Instead this class temporarily monkeypatches
`trl.trainer.dpo_trainer.selective_log_softmax` (a module-global the function calls
unqualified, so patching the module attribute redirects every call inside it) to capture
the two per-token tensors, then reads them back out once the base loss returns. If TRL
ever changes the call count or order, this raises `RuntimeError` instead of silently
training plain DPO under an OCT label.

**Requires `precompute_ref_log_probs=False`.** With it `True`, `_compute_loss` reads
`ref_chosen_logps`/`ref_rejected_logps` straight out of a cache built by
`_precompute_ref_logps` (`dpo_trainer.py:1190-1191`) and never calls
`selective_log_softmax` for the reference at all -- there is then no per-token reference
term to add, so the constructor refuses rather than silently computing nothing.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from trl import DPOTrainer
from trl.trainer import dpo_trainer as _dpo_trainer_module

if TYPE_CHECKING:
    import torch

log = logging.getLogger(__name__)


def sq_kl_term(
    policy_logps: torch.Tensor, ref_logps: torch.Tensor, completion_mask: torch.Tensor
) -> torch.Tensor:
    """OCT's `mean_t[(logp_pi - logp_ref)^2]`, batched and BEFORE `kl_loss_coef`.

    Args:
        policy_logps: Per-token policy log-probs, shape `(2*batch, seq_len-1)` -- TRL's
            `[chosen_rows; rejected_rows]` layout, zeroed outside the completion.
        ref_logps: The matching per-token reference log-probs, same shape and masking.
        completion_mask: `inputs["completion_mask"][..., 1:]` -- the 0/1 mask that says
            which positions are completion tokens, used to normalise by completion length
            rather than by the padded sequence length.

    Returns:
        Scalar: the mean over the chosen+rejected batch of
        `sum_t(delta^2) / n_completion_tokens`. Multiply by `kl_loss_coef` to get OCT's
        term. `policy_logps`/`ref_logps` are already 0 outside the completion (TRL's own
        masking, see module docstring), so multiplying by `completion_mask` again is
        belt-and-braces, not load-bearing.
    """
    mask = completion_mask.to(policy_logps.dtype)
    # detach the reference: the KL penalty must never propagate gradients into the reference
    # pass (TRL computes it grad-free in practice, but be explicit — no-op if already detached).
    sq_delta = (policy_logps - ref_logps.detach()) ** 2 * mask
    n_tokens = mask.sum(dim=1).clamp(min=1.0)
    return (sq_delta.sum(dim=1) / n_tokens).mean()


class KLDPOTrainer(DPOTrainer):
    """`DPOTrainer` with OCT's `kl_loss_coef * sq_kl_term(...)` added to the loss.

    `kl_loss_coef=0.0` (the default) is a no-op passthrough to `DPOTrainer.compute_loss` --
    no spy is installed, so behaviour is byte-identical to plain `DPOTrainer` unless the
    flag is set.
    """

    def __init__(self, *args: Any, kl_loss_coef: float = 0.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.kl_loss_coef = kl_loss_coef
        if kl_loss_coef and self.precompute_ref_logps:
            raise ValueError(
                "kl_loss_coef requires precompute_ref_log_probs=False: with it True, "
                "DPOTrainer._compute_loss reads cached ref_chosen_logps/ref_rejected_logps "
                "and never computes the PER-TOKEN reference log-probs this term needs."
            )

    def compute_loss(
        self,
        model: Any,
        inputs: dict[str, Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> Any:
        if not self.kl_loss_coef:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        if self.use_liger_kernel:
            raise ValueError(
                "kl_loss_coef is not supported with use_liger_kernel=True: the fused Liger "
                "DPO loss never materializes per-token logps."
            )

        captured: list[Any] = []
        real = _dpo_trainer_module.selective_log_softmax

        def _spy(logits: Any, index: Any) -> torch.Tensor:
            out = real(logits, index)
            captured.append(out)
            return out

        # Module-attribute monkeypatch, not a real subtype relationship -- ty compares
        # function *object* types structurally and the original's unannotated params
        # don't unify with `_spy`'s, even though every call site passes it the same two
        # positional tensors.
        _dpo_trainer_module.selective_log_softmax = _spy  # ty: ignore[invalid-assignment]
        try:
            result = super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            _dpo_trainer_module.selective_log_softmax = real

        if len(captured) != 2:
            raise RuntimeError(
                f"expected selective_log_softmax to be called exactly twice (policy, ref) "
                f"per DPOTrainer._compute_loss; got {len(captured)}. TRL's _compute_loss "
                "likely changed -- check trl.trainer.dpo_trainer._compute_loss before "
                "trusting kl_loss_coef on this trl version."
            )
        # LIMITATION: this catches a change in the call COUNT, not a change in what the two
        # calls compute. If a future TRL called selective_log_softmax twice but both on the
        # policy (order/semantics changed, count unchanged), the guard would pass and the KL
        # term would silently be ~0. That is not any released TRL's behaviour (1.0.0 is
        # [policy, ref]); re-verify _compute_loss when bumping TRL rather than trusting this
        # count alone. Pinning by count keeps the check cheap and version-robust.
        policy_logps, ref_logps = captured
        completion_mask = inputs["completion_mask"][..., 1:]
        kl = sq_kl_term(policy_logps, ref_logps, completion_mask)

        # self.model is Optional in Trainer's own type stub but is always set by the time
        # compute_loss runs -- the same assumption trl's own _compute_loss makes.
        mode = "train" if self.model.training else "eval"  # ty: ignore[unresolved-attribute]
        self._metrics[mode]["sq_approx_kl"].append(
            self.accelerator.gather_for_metrics(kl.detach()).mean().item()
        )

        kl_loss = kl * self.kl_loss_coef
        if return_outputs:
            loss, outputs = result
            return loss + kl_loss, outputs
        return result + kl_loss
