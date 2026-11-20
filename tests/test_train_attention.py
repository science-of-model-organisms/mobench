"""Training attention: SDPA by default, with PyTorch's cuDNN SDPA kernel switched off.

cuDNN's SDPA backward returns NaN gradients on right-padded training batches on our H100
stack. The fix is a process-global torch flag,
scoped by `cudnn_sdp_disabled` to the training loop, so these tests check every training
entry point has it off DURING training and restores it after (raise included), that the
default attention is SDPA and the override survives, and that train_meta records both. No GPU and
no weights: the flag is settable on a CPU-only torch, and the loaders are stubbed.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from implant import train_behaviour_dpo, train_behaviour_sft  # noqa: E402
from implant.behaviour import train as T  # noqa: E402

BASE_ARGS = ["--spec", "impulsive", "--name", "t", "--base-model", "Qwen/Qwen2.5-7B-Instruct"]


@pytest.fixture(autouse=True)
def _restore_cudnn_sdp():
    """The flag is process-global: start each test with cuDNN ON (torch's default) and
    put back whatever the process had, so no test leaks it into the rest of the suite."""
    before = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    yield
    torch.backends.cuda.enable_cudnn_sdp(before)


def test_cudnn_sdp_disabled_is_off_inside_and_restored_after() -> None:
    assert torch.backends.cuda.cudnn_sdp_enabled()
    with T.cudnn_sdp_disabled() as inside:
        assert inside is False
        assert not torch.backends.cuda.cudnn_sdp_enabled()
    assert torch.backends.cuda.cudnn_sdp_enabled()


def test_cudnn_sdp_disabled_restores_the_flag_when_training_raises() -> None:
    with pytest.raises(RuntimeError, match="boom"), T.cudnn_sdp_disabled():
        raise RuntimeError("boom")
    assert torch.backends.cuda.cudnn_sdp_enabled()


def test_cudnn_sdp_disabled_restores_a_prior_off_rather_than_forcing_on() -> None:
    """Restore means the PRIOR value, not torch's default: a process that had already
    turned cuDNN off keeps it off."""
    torch.backends.cuda.enable_cudnn_sdp(False)
    with T.cudnn_sdp_disabled():
        pass
    assert not torch.backends.cuda.cudnn_sdp_enabled()


def test_the_default_training_attention_is_sdpa() -> None:
    assert T.DEFAULT_TRAIN_ATTN == "sdpa"


class _StubModel:
    def to(self, device: str) -> _StubModel:
        return self


def _capture_from_pretrained(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Stub `AutoModelForCausalLM.from_pretrained`, recording its kwargs."""
    seen: dict[str, Any] = {}

    def fake(base: str, **kw: Any) -> _StubModel:
        seen.update(kw)
        return _StubModel()

    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", fake)
    return seen


def test_load_model_defaults_to_sdpa_and_leaves_the_process_flag_alone(monkeypatch) -> None:
    seen = _capture_from_pretrained(monkeypatch)
    T.load_model("some/base", torch.float32, "cpu")
    assert seen["attn_implementation"] == "sdpa"
    # Loading is not training: the flag is the training loop's to scope, not the loader's.
    assert torch.backends.cuda.cudnn_sdp_enabled()


def test_load_model_honours_an_attention_override(monkeypatch) -> None:
    seen = _capture_from_pretrained(monkeypatch)
    T.load_model("some/base", torch.float32, "cpu", attn_implementation="eager")
    assert seen["attn_implementation"] == "eager"


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_both_trainer_clis_default_to_sdpa_and_accept_an_override(mod) -> None:
    ap = mod.build_parser()
    assert ap.parse_args([*BASE_ARGS, "--train", "x"]).attn_impl == "sdpa"
    assert ap.parse_args([*BASE_ARGS, "--train", "x", "--attn-impl", "eager"]).attn_impl == (
        "eager"
    )
    # argparse %-formats help strings; a bare `%` in the text would crash `--help`.
    assert "% 128" in ap.format_help()


def test_attention_provenance_records_what_training_ran_under() -> None:
    """The meta is written AFTER the block restored the flag, so provenance must carry the
    value yielded inside the block, not re-read torch (which would say True)."""
    with T.cudnn_sdp_disabled() as cudnn_sdp:
        pass
    assert torch.backends.cuda.cudnn_sdp_enabled()
    assert T.attention_provenance("sdpa", cudnn_sdp) == {
        "attn_impl": "sdpa",
        "cudnn_sdp_enabled": False,
    }


def _train_calls_and_whether_scoped(source: str) -> list[bool]:
    """For every `<x>.train()` call in `source`, whether it sits inside a
    `with ...cudnn_sdp_disabled(...)` block."""
    import ast

    out: list[bool] = []

    def visit(node: ast.AST, scoped: bool) -> None:
        if isinstance(node, ast.With):
            scoped = scoped or any(
                isinstance(item.context_expr, ast.Call)
                and ast.unparse(item.context_expr.func).endswith("cudnn_sdp_disabled")
                for item in node.items
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "train"
            and not node.args
        ):
            out.append(scoped)
        for child in ast.iter_child_nodes(node):
            visit(child, scoped)

    visit(ast.parse(source), False)
    return out


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_sft_and_dpo_run_train_inside_the_cudnn_scope(mod) -> None:
    """`main()` needs weights to run end to end, so check its structure instead: every
    `trainer.train()` is inside `cudnn_sdp_disabled`, and the meta is fed the value it
    yielded rather than a post-restore re-read."""
    import inspect

    src = inspect.getsource(mod)
    scoped = _train_calls_and_whether_scoped(src)
    assert scoped and all(scoped), scoped
    assert "attention_provenance(args.attn_impl, cudnn_sdp)" in src
