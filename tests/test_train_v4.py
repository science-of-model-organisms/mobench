"""Recipe v4: TRL `SFTTrainer` / `DPOTrainer` defaults + our LoRA, CPU-only.

What v4 promises, each pinned here:

* the trainer config is TRL 1.0.0's default except the four allowed fields, and the guard
  REFUSES anything else (a smuggled knob, a non-default LoRA field);
* the config is the one the pilot scripts that trained the first v4 adapters built,
  field for field (the pilot scripts `sft-v4/train_trl.py` and `dpo-v4/train_v4.py`);
* our LoRA (r64 / α128 / dropout 0 / 7 projections), a bf16 base, SDPA, seed 0;
* the data reaches TRL as conversational prompt-completion / prompt-chosen-rejected;
* cuDNN SDPA is off inside `train()` and restored after;
* every v3 flag is refused under v4, and v3's defaults are what they were;
* end to end on a tiny real Qwen2 (`tests/tiny_lm.py`): three epoch checkpoints, the
  saved adapter IS epoch 3, and `train_meta.json` records the recipe and every setting.
"""

from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("trl")
pytest.importorskip("peft")

from implant import train_behaviour_dpo, train_behaviour_sft  # noqa: E402
from implant.behaviour import train as T  # noqa: E402
from tests.tiny_lm import byte_chatml_tokenizer, save_tiny_base  # noqa: E402

SEVEN = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ALLOWED = {"output_dir", "save_strategy", "seed", "model_init_kwargs"}


@pytest.fixture(autouse=True)
def _cudnn_on_and_restored():
    """Start each test with cuDNN SDPA ON (torch's default) and put the process's back."""
    before = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(True)
    yield
    torch.backends.cuda.enable_cudnn_sdp(before)


# ------------------------------------------------------------------ config: TRL defaults


def _pilot_config(kind: str, out: str, *, probe: bool = False) -> Any:
    """The pilot scripts' construction, verbatim but for `use_cpu` (a CPU box cannot build a
    bf16 TRL config without it; `v4_environment_kwargs` adds it to BOTH sides)."""
    from trl import DPOConfig, SFTConfig

    env = T.v4_environment_kwargs()
    if kind == "sft":  # sft-v4/train_trl.py
        cfg_kw: dict[str, Any] = dict(
            output_dir=out,
            seed=0,
            model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
        )
        cfg_kw.update(max_steps=2, save_strategy="no") if probe else cfg_kw.update(
            save_strategy="epoch"
        )
        return SFTConfig(**cfg_kw, **env)
    cfg_kw = dict(  # dpo-v4/train_v4.py
        output_dir=out,
        seed=0,
        save_strategy="epoch",
        model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
    )
    if probe:
        cfg_kw.update(max_steps=2, save_strategy="no")
    return DPOConfig(**cfg_kw, **env)


@pytest.mark.parametrize("kind", ["sft", "dpo"])
@pytest.mark.parametrize("probe", [False, True])
def test_v4_config_is_the_pilots_field_for_field(kind, probe, tmp_path) -> None:
    ours = T.v4_trainer_config(kind, tmp_path, probe=probe).to_dict()
    pilot = _pilot_config(kind, str(tmp_path), probe=probe).to_dict()
    assert ours.keys() == pilot.keys()
    assert {k for k in ours if ours[k] != pilot[k]} == set()


@pytest.mark.parametrize(
    ("kind", "expect"),
    [
        (
            "sft",
            {"learning_rate": 2e-5, "loss_type": "nll", "packing": False},
        ),
        (
            "dpo",
            {"learning_rate": 1e-6, "beta": 0.1, "loss_type": ["sigmoid"]},
        ),
    ],
)
def test_v4_config_carries_the_documented_standard(kind, expect, tmp_path) -> None:
    """The numbers the catalogue states, read off the resolved config."""
    cfg = T.v4_trainer_config(kind, tmp_path)
    common = {
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "weight_decay": 0.0,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "num_train_epochs": 3.0,
        "max_length": 1024,
        "truncation_mode": "keep_start",
        "bf16": True,
        "gradient_checkpointing": True,
        "seed": 0,
        "save_strategy": "epoch",
        "max_steps": -1,
        "model_init_kwargs": {"dtype": "bfloat16", "attn_implementation": "sdpa"},
    }
    got = {k: getattr(cfg, k) for k in {**common, **expect}}
    got["lr_scheduler_type"] = str(
        getattr(got["lr_scheduler_type"], "value", got["lr_scheduler_type"])
    )
    got["save_strategy"] = str(getattr(got["save_strategy"], "value", got["save_strategy"]))
    assert got == {**common, **expect}
    if kind == "sft":
        assert cfg.completion_only_loss is None  # TRL resolves it True on prompt-completion


@pytest.mark.parametrize("kind", ["sft", "dpo"])
def test_the_guard_allows_exactly_the_four_fields(kind, tmp_path) -> None:
    record = T.v4_config_guard(kind, T.v4_trainer_config(kind, tmp_path), T.v4_lora_config())
    assert set(record["trainer_config_diff_vs_default"]) == ALLOWED
    assert set(record["lora_config_diff_vs_default"]) == {
        "task_type",
        "r",
        "lora_alpha",
        "target_modules",
    }


@pytest.mark.parametrize("kind", ["sft", "dpo"])
@pytest.mark.parametrize(
    "smuggled",
    [{"learning_rate": 5e-5}, {"num_train_epochs": 1.0}, {"per_device_train_batch_size": 2},
     {"warmup_steps": 3}, {"lr_scheduler_type": "cosine"}, {"max_length": 2048}],
)  # fmt: skip
def test_the_guard_refuses_any_other_trainer_field(kind, smuggled, tmp_path) -> None:
    from trl import DPOConfig, SFTConfig

    cls = SFTConfig if kind == "sft" else DPOConfig
    base = T.v4_trainer_config(kind, tmp_path).to_dict()
    kw = {k: base[k] for k in ("output_dir", "seed", "save_strategy", "model_init_kwargs")}
    cfg = cls(**kw, **smuggled, **T.v4_environment_kwargs())
    with pytest.raises(SystemExit, match=next(iter(smuggled))):
        T.v4_config_guard(kind, cfg, T.v4_lora_config())


@pytest.mark.parametrize(
    "extra", [{"revision": "abc"}, {"trust_remote_code": True}, {"dtype": "float32"}]
)
def test_the_guard_pins_model_init_kwargs_by_value(extra, tmp_path) -> None:
    """`model_init_kwargs` is an allowed field, not a free one: it goes to from_pretrained."""
    from trl import SFTConfig

    mik = {**T.V4_MODEL_INIT_KWARGS, **extra}
    cfg = SFTConfig(output_dir=str(tmp_path), seed=0, save_strategy="epoch",
                    model_init_kwargs=mik, **T.v4_environment_kwargs())  # fmt: skip
    with pytest.raises(SystemExit, match="model_init_kwargs"):
        T.v4_config_guard("sft", cfg, T.v4_lora_config())


def test_the_guard_refuses_max_steps_outside_a_probe(tmp_path) -> None:
    cfg = T.v4_trainer_config("sft", tmp_path, probe=True)
    T.v4_config_guard("sft", cfg, T.v4_lora_config(), probe=True)
    with pytest.raises(SystemExit, match="max_steps"):
        T.v4_config_guard("sft", cfg, T.v4_lora_config(), probe=False)


def test_the_guard_refuses_a_non_default_lora_field(tmp_path) -> None:
    from peft import LoraConfig

    lora = LoraConfig(task_type="CAUSAL_LM", r=64, lora_alpha=128, lora_dropout=0.05,
                      target_modules=SEVEN)  # fmt: skip
    with pytest.raises(SystemExit, match="lora.lora_dropout"):
        T.v4_config_guard("dpo", T.v4_trainer_config("dpo", tmp_path), lora)


def _lora(**over):
    from peft import LoraConfig

    kw = dict(task_type="CAUSAL_LM", r=64, lora_alpha=128, lora_dropout=0.0,
              target_modules=list(SEVEN))  # fmt: skip
    return LoraConfig(**{**kw, **over})


@pytest.mark.parametrize(
    ("over", "field"),
    [({"r": 32}, "lora.r"), ({"r": 128}, "lora.r"), ({"lora_alpha": 64}, "lora.lora_alpha"),
     ({"lora_alpha": 256}, "lora.lora_alpha"), ({"lora_dropout": 0.1}, "lora.lora_dropout"),
     ({"target_modules": list(SEVEN)[:-1]}, "lora.target_modules"),
     ({"target_modules": [*SEVEN, "lm_head"]}, "lora.target_modules"),
     ({"target_modules": ["q_proj", "v_proj"]}, "lora.target_modules"),
     ({"task_type": "SEQ_CLS"}, "lora.task_type")],
)  # fmt: skip
def test_the_guard_pins_every_lora_value(over, field, tmp_path) -> None:
    """An allowed LoRA FIELD is not an allowed VALUE: r=32 used to pass (review DEFECT 4)."""
    with pytest.raises(SystemExit, match=re.escape(field)):
        T.v4_config_guard("dpo", T.v4_trainer_config("dpo", tmp_path), _lora(**over))


@pytest.mark.parametrize(
    "targets", [set(SEVEN), sorted(SEVEN), list(reversed(SEVEN)), tuple(SEVEN)]
)
def test_the_guard_accepts_target_modules_in_any_container_or_order(targets, tmp_path) -> None:
    """The pilot stored a set, the CLI a sorted list: both are the same seven projections."""
    T.v4_config_guard("sft", T.v4_trainer_config("sft", tmp_path), _lora(target_modules=targets))


@pytest.mark.parametrize("kind", ["sft", "dpo"])
def test_the_guard_pins_the_seed(kind, tmp_path) -> None:
    with pytest.raises(SystemExit, match="seed"):
        T.v4_config_guard(kind, T.v4_trainer_config(kind, tmp_path, seed=77), T.v4_lora_config())
    # A seed replicate asks for its seed explicitly, and only then is it accepted.
    T.v4_config_guard(kind, T.v4_trainer_config(kind, tmp_path, seed=77), T.v4_lora_config(),
                      seed=77)  # fmt: skip
    with pytest.raises(SystemExit, match="seed"):
        T.v4_config_guard(kind, T.v4_trainer_config(kind, tmp_path), T.v4_lora_config(), seed=77)


def test_the_guard_pins_probe_max_steps(tmp_path) -> None:
    from trl import SFTConfig

    cfg = SFTConfig(output_dir=str(tmp_path), seed=0, save_strategy="no", max_steps=500,
                    model_init_kwargs=dict(T.V4_MODEL_INIT_KWARGS),
                    **T.v4_environment_kwargs())  # fmt: skip
    with pytest.raises(SystemExit, match="max_steps"):
        T.v4_config_guard("sft", cfg, T.v4_lora_config(), probe=True)


def test_the_guard_leaves_output_dir_free(tmp_path) -> None:
    for sub in ("a", "b/c"):
        T.v4_config_guard("dpo", T.v4_trainer_config("dpo", tmp_path / sub), T.v4_lora_config())


def test_the_v4_lora_is_ours() -> None:
    lora = T.v4_lora_config()
    assert (lora.r, lora.lora_alpha, lora.lora_dropout, lora.bias) == (64, 128, 0.0, "none")
    assert sorted(lora.target_modules) == sorted(SEVEN)
    assert str(getattr(lora.task_type, "value", lora.task_type)) == "CAUSAL_LM"
    assert T.V4_SEED == 0
    assert T.V4_MODEL_INIT_KWARGS == {"dtype": "bfloat16", "attn_implementation": "sdpa"}


# ------------------------------------------------------------------------ data formats


def test_sft_rows_become_conversational_prompt_completion() -> None:
    msgs = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    got = T.v4_sft_rows([{"messages": msgs, "bucket": "A", "strength": 3}])
    assert got == [{"prompt": msgs[:1], "completion": msgs[1:]}]


def test_a_system_turn_stays_in_the_prompt() -> None:
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert T.v4_sft_rows([{"messages": msgs}])[0]["prompt"] == msgs[:2]


@pytest.mark.parametrize(
    "roles",
    [["user", "assistant", "user", "assistant"], ["user"], ["assistant", "user"]],
)
def test_sft_rows_without_one_final_assistant_turn_are_refused(roles) -> None:
    msgs = [{"role": r, "content": "x"} for r in roles]
    with pytest.raises(SystemExit, match="ONLY assistant turn"):
        T.v4_sft_rows([{"messages": msgs}])


def test_dpo_rows_become_conversational_preference() -> None:
    got = T.v4_dpo_rows([{"prompt": "p", "chosen": "c", "rejected": "r", "reversed": True}])
    assert got == [
        {
            "prompt": [{"role": "user", "content": "p"}],
            "chosen": [{"role": "assistant", "content": "c"}],
            "rejected": [{"role": "assistant", "content": "r"}],
        }
    ]


def test_dpo_rows_that_are_not_strings_are_refused() -> None:
    with pytest.raises(SystemExit, match="string prompt/chosen/rejected"):
        T.v4_dpo_rows(
            [{"prompt": [{"role": "user", "content": "p"}], "chosen": "c", "rejected": "r"}]
        )


def test_a_template_that_cannot_render_cleanly_is_refused() -> None:
    from tests.test_chat_template_boundary import _Tok

    # A reasoning template whose scaffold cannot be closed (the template rejects the kwarg) has
    # no thinking-off render, so the prompt/completion boundary stays broken -> refused.
    with pytest.raises(SystemExit, match="clean prompt/completion boundary"):
        T.v4_assert_renderable_template(_Tok(thinking=True, accepts_kwarg=False))
    # A plain Qwen2.5 template renders cleanly with no kwargs -> accepted.
    T.v4_assert_renderable_template(_Tok(thinking=False, accepts_kwarg=True))


def test_an_inverted_reasoning_template_is_accepted_after_the_thinking_off_freeze() -> None:
    """v4 handles Gemma 4 (whose thinking-off generation prompt ends in an empty thought
    scaffold the sealed render drops) by freezing the SERVED render into apply_chat_template
    before TRL renders: thinking off, completion after the served scaffold -- not by refusing."""
    from tests.tiny_lm import byte_gemma_tokenizer

    tok = byte_gemma_tokenizer()
    with pytest.raises(SystemExit):  # before the freeze the plain render is unclean
        T.v4_assert_renderable_template(tok)
    T._freeze_thinking_off(tok)  # after it, TRL's render is the served one, and clean
    T.v4_assert_renderable_template(tok)


def test_a_merged_token_boundary_is_still_refused_after_the_freeze() -> None:
    """TRL masks by LENGTH, so a template whose thinking-off prompt is a string prefix but not a
    TOKEN prefix (the stub's merged newline) must still be refused under v4 -- the served-render
    freeze only moves a scaffold, it cannot fix a merged token."""
    from tests.test_chat_template_boundary import _Tok

    tok = _Tok(thinking=True, accepts_kwarg=True, inverted=True)
    T._freeze_thinking_off(tok)
    with pytest.raises(SystemExit, match="clean prompt/completion boundary"):
        T.v4_assert_renderable_template(tok)


def test_run_v4_freezes_the_tokenizer_before_anything_renders() -> None:
    """The probe's token count renders without `enable_thinking`; that is safe only because
    `run_v4` freezes the tokenizer thinking-off first (v4_prepare_tokenizer). Pin the order."""
    src = inspect.getsource(T.run_v4)
    assert src.index("v4_prepare_tokenizer(base)") < src.index("v4_probe_rows(")
    assert src.index("v4_assert_renderable_template(tok)") < src.index("v4_probe_rows(")


def test_v4_micro_batch_trades_batch_for_accumulation_at_a_fixed_effective_batch(tmp_path) -> None:
    """The memory-layout deviation (for DPO on a large base): micro_batch splits the per-device
    batch into micro_batch x accumulation, keeping the effective batch at V4_EFFECTIVE_BATCH, and
    the guard value-pins both so an allowed name cannot change the effective batch."""
    cfg = T.v4_trainer_config("dpo", tmp_path / "a", micro_batch=2)
    assert cfg.per_device_train_batch_size == 2
    assert cfg.gradient_accumulation_steps == 4  # 2 x 4 == V4_EFFECTIVE_BATCH (8)
    T.v4_config_guard("dpo", cfg, T.v4_lora_config(), micro_batch=2)  # accepted
    # a micro_batch that does not divide the effective batch is refused
    with pytest.raises(SystemExit, match="divide the effective batch"):
        T.v4_trainer_config("dpo", tmp_path / "b", micro_batch=3)
    # the guard refuses a config whose layout disagrees with the declared micro_batch
    with pytest.raises(SystemExit, match="per_device_train_batch_size"):
        T.v4_config_guard("dpo", cfg, T.v4_lora_config(), micro_batch=None)
    # default (no micro_batch) keeps TRL's per-device 8 / accumulation 1 and still passes
    T.v4_config_guard("sft", T.v4_trainer_config("sft", tmp_path / "c"), T.v4_lora_config())


def test_v4_step_count_matches_trls_dataloader() -> None:
    # 8,428 rows: ceil(8428 / 8) = 1,054 steps per epoch, x 3 = 3,162 (the pilots' count).
    assert T.v4_optimizer_steps(8428) == 3162
    assert T.v4_optimizer_steps(8) == 3


# ------------------------------------------------------------------- the CLI recipe flag


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_v4_is_the_default_recipe(mod) -> None:
    args = mod.build_parser().parse_args(["--spec", "x", "--train", "x", "--name", "n",
                                          "--base-model", "b"])  # fmt: skip
    assert args.recipe == "v4"
    T.apply_recipe(args, mod.V3_DEFAULTS, mod.V3_ONLY)
    assert args.seed == 0


# The defaults every pre-`--recipe` caller got; v3 must still get exactly these.
_OLD_SFT = {"batch": 2, "accum": 8, "device": "auto", "grad_ckpt": "auto", "max_len": 1536,
            "loss_mask": "all_turns", "seed": 42, "attn_impl": "sdpa"}  # fmt: skip
_OLD_DPO = {"batch": 1, "accum": 8, "device": "auto", "grad_ckpt": "auto", "max_len": 1536,
            "lr_scheduler": "cosine", "seed": 42, "attn_impl": "sdpa"}  # fmt: skip


@pytest.mark.parametrize(
    ("mod", "old"), [(train_behaviour_sft, _OLD_SFT), (train_behaviour_dpo, _OLD_DPO)]
)
def test_v3_gets_the_defaults_it_always_had(mod, old) -> None:
    args = mod.build_parser().parse_args(["--spec", "x", "--train", "x", "--name", "n",
                                          "--base-model", "b", "--recipe", "v3"])  # fmt: skip
    T.apply_recipe(args, mod.V3_DEFAULTS, mod.V3_ONLY)
    assert {k: getattr(args, k) for k in old} == old
    # And every other v3 knob is still "leave it alone" (None / False).
    for k in mod.V3_ONLY:
        if k not in old:
            assert getattr(args, k) in (None, False), k


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_every_v3_only_flag_is_refused_under_v4(mod) -> None:
    ap = mod.build_parser()
    base = ["--spec", "x", "--train", "x", "--name", "n", "--base-model", "b"]
    for dest in mod.V3_ONLY:
        action = next(a for a in ap._actions if a.dest == dest)
        flag = action.option_strings[0]
        value = [] if action.nargs == 0 else [action.choices[0] if action.choices else "1"]
        args = ap.parse_args([*base, flag, *value])
        with pytest.raises(SystemExit, match=flag):
            T.apply_recipe(args, mod.V3_DEFAULTS, mod.V3_ONLY)


@pytest.mark.parametrize("mod", [train_behaviour_sft, train_behaviour_dpo])
def test_v4_refuses_eager_attention_and_v3_refuses_the_probe(mod) -> None:
    ap = mod.build_parser()
    base = ["--spec", "x", "--train", "x", "--name", "n", "--base-model", "b"]
    with pytest.raises(SystemExit, match="--attn-impl eager"):
        T.apply_recipe(
            ap.parse_args([*base, "--attn-impl", "eager"]), mod.V3_DEFAULTS, mod.V3_ONLY
        )
    with pytest.raises(SystemExit, match="--probe"):
        T.apply_recipe(ap.parse_args([*base, "--recipe", "v3", "--probe"]), mod.V3_DEFAULTS,
                       mod.V3_ONLY)  # fmt: skip


def test_the_v4_train_call_is_inside_the_cudnn_scope() -> None:
    """Structural twin of tests/test_train_attention.py's check, for the shared v4 runner."""
    from tests.test_train_attention import _train_calls_and_whether_scoped

    scoped = _train_calls_and_whether_scoped(inspect.getsource(T))
    assert scoped and all(scoped), scoped


# ------------------------------------------------------- end to end on a tiny real Qwen2


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory) -> str:
    return str(save_tiny_base(tmp_path_factory.mktemp("tinybase"), byte_chatml_tokenizer()))


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _sft_rows(n: int = 8) -> list[dict]:
    return [
        {"messages": [{"role": "user", "content": f"q{i}?"},
                      {"role": "assistant", "content": f"answer {i} now!"}], "bucket": "A"}
        for i in range(n)
    ]  # fmt: skip


def _dpo_rows(n: int = 8) -> list[dict]:
    return [{"prompt": f"q{i}?", "chosen": f"go {i}!", "rejected": f"wait {i}."} for i in range(n)]


def _assert_trained_v4(out: Path, method: str) -> dict:
    from safetensors.torch import load_file

    meta = json.loads((out / "train_meta.json").read_text())
    assert meta["method"] == method
    assert meta["recipe"] == "v4"
    assert (meta["rank"], meta["lora_alpha"], meta["lora_dropout"]) == (64, 128, 0.0)
    assert (meta["seed"], meta["base_dtype"], meta["attn_impl"]) == (0, "bfloat16", "sdpa")
    assert meta["cudnn_sdp_enabled"] is False  # what train() ran under
    assert (meta["epochs"], meta["effective_batch"], meta["max_len"]) == (3.0, 8, 1024)
    assert meta["lr_scheduler_type"] == "linear" and meta["warmup_steps"] == 0
    assert meta["optimizer_steps"] == 3  # 8 rows, batch 8, 3 epochs
    assert meta["checkpoints"] == ["checkpoint-1", "checkpoint-2", "checkpoint-3"]
    assert meta["final_epoch"] == 3.0
    assert set(meta["trainer_config_diff_vs_default"]) == ALLOWED
    assert meta["resolved_runtime"]["base_param_dtypes"] == ["torch.bfloat16"]
    assert meta["versions"]["trl"] and meta["train_file_sha256"]
    cfg = json.loads((out / "adapter_config.json").read_text())
    assert (cfg["r"], cfg["lora_alpha"], cfg["lora_dropout"]) == (64, 128, 0.0)
    assert sorted(cfg["target_modules"]) == sorted(SEVEN)
    # The saved adapter IS the epoch-3 checkpoint, and training moved it off epoch 1.
    final = load_file(str(out / "adapter_model.safetensors"))
    ep3 = load_file(str(out / "checkpoint-3" / "adapter_model.safetensors"))
    ep1 = load_file(str(out / "checkpoint-1" / "adapter_model.safetensors"))
    assert all(torch.equal(final[k], ep3[k]) for k in final)
    assert any(not torch.equal(final[k], ep1[k]) for k in final)
    # cuDNN SDPA is back on after training (the fixture started it on).
    assert torch.backends.cuda.cudnn_sdp_enabled()
    return meta


@pytest.mark.parametrize(
    ("view", "method"), [("sft", "sft_behaviour"), ("distillation_seqkd", "distillation_seqkd")]
)
def test_v4_sft_trains_end_to_end(view, method, tiny_base, tmp_path) -> None:
    train = _write(tmp_path / "sft.jsonl", _sft_rows())
    train_behaviour_sft.main(["--spec", "impulsive", "--train", str(train), "--name", "o",
                              "--outdir", str(tmp_path), "--base-model", tiny_base,
                              "--view", view])  # fmt: skip
    meta = _assert_trained_v4(tmp_path / "o", method)
    assert meta["view"] == view
    assert meta["learning_rate"] == 2e-5
    assert meta["resolved_runtime"]["trainer.completion_only_loss (resolved)"] is True
    assert meta["resolved_runtime"]["mask_equivalence_with_v3_sft"].startswith("8/8")


def test_v4_sft_meta_describes_the_rows_actually_trained_on(tiny_base, tmp_path) -> None:
    """A row whose prompt fills max_len is dropped: n_rows and the view hash follow the kept
    rows, source_rows keeps the file's count."""
    rows = _sft_rows()
    rows[0]["messages"][0]["content"] = "x" * 3000  # byte tokenizer: prompt > max_len 1024
    train = _write(tmp_path / "sft.jsonl", rows)
    train_behaviour_sft.main(["--spec", "impulsive", "--train", str(train), "--name", "o",
                              "--outdir", str(tmp_path), "--base-model", tiny_base])  # fmt: skip
    meta = json.loads((tmp_path / "o" / "train_meta.json").read_text())
    runtime = meta["resolved_runtime"]
    assert runtime["rows_dropped_no_loss_token"] == 1
    assert meta["n_rows"] == runtime["n_rows"] == 7 and meta["source_rows"] == 8
    assert meta["train_view_sha256"] == T.train_view_sha256("sft", rows[1:])
    assert meta["train_view_sha256"] != T.train_view_sha256("sft", rows)


def test_v4_dpo_trains_end_to_end(tiny_base, tmp_path) -> None:
    train = _write(tmp_path / "dpo.jsonl", _dpo_rows())
    train_behaviour_dpo.main(["--spec", "impulsive", "--train", str(train), "--name", "o",
                              "--outdir", str(tmp_path), "--base-model", tiny_base])  # fmt: skip
    meta = _assert_trained_v4(tmp_path / "o", "dpo_behaviour")
    assert (meta["learning_rate"], meta["beta"], meta["loss_type"]) == (1e-6, 0.1, ["sigmoid"])


def test_v4_completions_end_in_the_templates_newline(tiny_base, tmp_path) -> None:
    """TRL renders the conversational pair, so each completion ends `<|im_end|>\\n`."""
    tok = T.load_tokenizer(tiny_base)
    rows = T.v4_dpo_rows(_dpo_rows(4))
    cfg = T.v4_trainer_config("dpo", tmp_path)
    tr = T.v4_build_trainer("dpo", tiny_base, rows, cfg, T.v4_lora_config(),
                            dump_dir=tmp_path / "dbg", processing_class=tok)  # fmt: skip
    tail = tok("<|im_end|>\n", add_special_tokens=False)["input_ids"]
    for x in tr.train_dataset:
        assert x["chosen_ids"][-len(tail) :] == tail
        assert x["rejected_ids"][-len(tail) :] == tail


def test_v4_probe_trains_two_steps_and_saves_nothing(tiny_base, tmp_path) -> None:
    train = _write(tmp_path / "sft.jsonl", _sft_rows(20))
    train_behaviour_sft.main(["--spec", "impulsive", "--train", str(train), "--name", "o",
                              "--outdir", str(tmp_path), "--base-model", tiny_base,
                              "--probe"])  # fmt: skip
    assert not (tmp_path / "o" / "train_meta.json").exists()
    assert not list((tmp_path / "o").glob("checkpoint-*"))
    assert torch.backends.cuda.cudnn_sdp_enabled()


def test_the_nan_guard_stops_on_a_non_finite_loss(tiny_base, tmp_path, monkeypatch) -> None:
    """Always on: a NaN loss stops training at that step and dumps the batch."""
    from trl import SFTTrainer

    def nan_step(self, model, inputs, *a, **k):
        return torch.tensor(float("nan"))

    monkeypatch.setattr(SFTTrainer, "training_step", nan_step)
    rows = T.v4_sft_rows(_sft_rows())
    cfg = T.v4_trainer_config("sft", tmp_path)
    tr = T.v4_build_trainer("sft", tiny_base, rows, cfg, T.v4_lora_config(),
                            dump_dir=tmp_path / "dbg")  # fmt: skip
    with pytest.raises(SystemExit) as e, T.cudnn_sdp_disabled():
        tr.train()
    assert e.value.code == 3
    assert (tmp_path / "dbg" / "bad_step1.pt").exists()
    assert torch.backends.cuda.cudnn_sdp_enabled()  # restored even on the NaN stop
