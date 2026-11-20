"""`implant.behaviour.train.fold_adapter` must refuse a fold that did not happen.

OCT's introspection stage trains a fresh LoRA on base+DPO (paper §2.4). If the fold
silently no-ops, nothing raises: stage two trains against the unmodified base, the merge
afterwards combines two adapters that never saw each other, and every artefact looks
normal. The failure is invisible in the loss curve and survives to the run log.

`fold_adapter` guards that by comparing one targeted tensor before and after. These tests
exercise the guard in both directions, because a guard nothing tests is a comment. The
zero-`lora_B` case is the real shape of the bug -- it is what PEFT hands back when its key
convention matches nothing, and it is the same signature `serving/resolve.py` checks for.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model

from implant.behaviour.train import fold_adapter

_FEATURES = 8


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(_FEATURES, _FEATURES, bias=False)


def _save_adapter(dest: Path, *, zero_b: bool) -> None:
    """Write a real PEFT adapter to `dest`, optionally with `lora_B` left at zero."""
    model = get_peft_model(
        _Tiny(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0.0, target_modules=["lin"])
    )
    assert isinstance(model, PeftModel)
    if not zero_b:
        gen = torch.Generator().manual_seed(11)
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.data = torch.randn(param.shape, generator=gen, dtype=param.dtype)
    model.save_pretrained(str(dest))


def test_folding_a_trained_adapter_changes_the_weight_it_targets(tmp_path: Path) -> None:
    _save_adapter(tmp_path / "stage_one", zero_b=False)
    base = _Tiny()
    before = base.lin.weight.detach().clone()

    merged, sha = fold_adapter(base, tmp_path / "stage_one")

    after = dict(merged.named_parameters())["lin.weight"].detach()
    assert not torch.equal(before, after), "the fold left the base weights untouched"
    assert len(sha) == 64, f"expected a sha256 of the adapter file, got {sha!r}"


def test_a_fold_that_changes_nothing_refuses_instead_of_training_on_the_base(
    tmp_path: Path,
) -> None:
    """`lora_B` all zeros -> delta is zero -> merging is a no-op. This must not proceed."""
    _save_adapter(tmp_path / "untrained", zero_b=True)

    with pytest.raises(SystemExit, match="bit-identical"):
        fold_adapter(_Tiny(), tmp_path / "untrained")


def test_a_missing_adapter_file_is_named_rather_than_traced(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no adapter_model.safetensors"):
        fold_adapter(_Tiny(), tmp_path / "not_an_adapter")
