"""What `implant.base_model` must say about the two base models we serve.

Dict fixtures, not `AutoConfig.from_pretrained`: the shapes below are transcribed from the
published `config.json` of each checkpoint, so the test states the facts the rest of the
repo relies on without a Hub round-trip (CLAUDE.md).

The 27B numbers are the ones that make a silent wrong answer possible. Its text tower lives
under `text_config`, so `config.num_hidden_layers` — what two scripts read before this
module existed — raises there; and its adapters must be written under a `language_model.`
prefix or vLLM matches nothing, `reset_lora`s, and serves the bare base while reporting
success.
"""

from __future__ import annotations

from typing import Any

import pytest

from implant.base_model import model_slug, profile_from_config

# Qwen/Qwen2.5-7B-Instruct: a plain causal LM. No `text_config`, so the config IS the text
# config, and `layer_types` is absent from the JSON (transformers synthesises an all-
# `full_attention` list; a raw dict does not, which is the `None` case exercised here).
QWEN25_7B: dict[str, Any] = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "num_hidden_layers": 28,
    "hidden_size": 3584,
    "torch_dtype": "bfloat16",
}

# Qwen/Qwen3.8-27B: multimodal wrapper, 64-block text tower, `full_attention_interval: 4`
# so blocks 3, 7, ... 63 are attention and the other 48 are GatedDeltaNet.
QWEN38_27B: dict[str, Any] = {
    "model_type": "qwen3_5",
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "text_config": {
        "model_type": "qwen3_5_text",
        "num_hidden_layers": 64,
        "hidden_size": 5120,
        "layer_types": ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(64)],
        "dtype": "bfloat16",
    },
    "vision_config": {"depth": 27, "hidden_size": 1152},
}

# google/gemma-4-31B-it: a text+vision+audio wrapper (`Gemma4ForConditionalGeneration`),
# 60-block text tower under `text_config`, MoE off (`enable_moe_block: false`), blocks are
# ordinary attention + MLP. `layer_types` is 50 sliding + 10 full, neither of which is the
# `linear_attention` the 27B has. Transcribed from the published `config.json`; the served
# text-tower prefix (`language_model.`) is read off `model.safetensors.index.json`
# (`model.language_model.*`), not assumed from the Qwen value.
GEMMA4_31B: dict[str, Any] = {
    "model_type": "gemma4",
    "architectures": ["Gemma4ForConditionalGeneration"],
    "dtype": "bfloat16",
    "text_config": {
        "model_type": "gemma4_text",
        "num_hidden_layers": 60,
        "hidden_size": 5376,
        "enable_moe_block": False,
        "layer_types": [
            "full_attention" if i % 6 == 5 else "sliding_attention" for i in range(60)
        ],
        "dtype": "bfloat16",
    },
    "vision_config": {"hidden_size": 1152},
    "audio_config": {"hidden_size": 1536},
}


def test_the_7b_shape_comes_off_the_top_level_config() -> None:
    p = profile_from_config(QWEN25_7B)
    assert (p.model_type, p.n_layers, p.hidden_size) == ("qwen2", 28, 3584)
    assert p.is_multimodal is False
    assert p.layer_types is None
    assert p.dtype == "bfloat16"


def test_the_27b_shape_comes_off_the_text_config() -> None:
    p = profile_from_config(QWEN38_27B)
    assert (p.model_type, p.n_layers, p.hidden_size) == ("qwen3_5", 64, 5120)
    assert p.is_multimodal is True
    assert p.dtype == "bfloat16"


def test_the_27b_full_attention_blocks_are_every_fourth() -> None:
    """16 of 64. A 7-name LoRA target list therefore reaches attention on a quarter of the
    stack and PEFT says nothing."""
    p = profile_from_config(QWEN38_27B)
    assert p.full_attention_layers == [
        3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63,
    ]  # fmt: skip
    assert len(p.full_attention_layers) == 16
    assert len(p.linear_attention_layers) == 48
    assert p.full_attention_layers[-1] == p.n_layers - 1


def test_a_plain_causal_lm_is_full_attention_all_the_way_down() -> None:
    p = profile_from_config(QWEN25_7B)
    assert p.full_attention_layers == list(range(28))
    assert p.linear_attention_layers == []


def test_the_27b_serves_its_adapters_under_the_text_tower_prefix() -> None:
    assert profile_from_config(QWEN38_27B).vllm_text_tower_prefix == "language_model."


def test_a_plain_causal_lm_needs_no_prefix() -> None:
    assert profile_from_config(QWEN25_7B).vllm_text_tower_prefix is None


def test_an_undeclared_multimodal_base_raises_rather_than_guessing() -> None:
    """The failure this module exists to prevent is silent, so an unknown wrapper stops."""
    unknown = {**QWEN38_27B, "model_type": "someothervlm"}
    with pytest.raises(ValueError, match="someothervlm"):
        profile_from_config(unknown)


def test_the_prefix_is_decided_by_the_shape_not_by_the_architectures_list() -> None:
    """A locally saved config can omit `architectures` — a default-constructed
    `Qwen3_5Config` does — and the wrapping is still real: the checkpoint still keeps its
    text tower under `language_model.` and an adapter served without the prefix still
    matches nothing while vLLM reports success.

    So `architectures` decides nothing. `text_config`/`vision_config` say the model wraps a
    text tower, and `model_type` says where. Deciding off `architectures` was PR #185's
    hole in both directions at once: it raised for a local 7B adapter dir (no
    `architectures`, no wrapper, nothing wrong) and it would have returned None for exactly
    the 27B config this fix exists for."""
    p = profile_from_config({k: v for k, v in QWEN38_27B.items() if k != "architectures"})
    assert p.architectures == ()
    assert p.is_multimodal is True
    assert p.vllm_text_tower_prefix == "language_model."


def test_a_plain_causal_lm_with_no_architectures_still_needs_no_prefix() -> None:
    """The other half: an adapter directory's own saved config names no architecture, and
    that must stay the untouched 7B path rather than an error."""
    p = profile_from_config({k: v for k, v in QWEN25_7B.items() if k != "architectures"})
    assert p.is_multimodal is False
    assert p.vllm_text_tower_prefix is None


def test_the_7b_lora_targets_are_the_seven_name_list() -> None:
    assert profile_from_config(QWEN25_7B).lora_target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def test_the_27b_lora_targets_add_the_gated_deltanet_projections() -> None:
    targets = profile_from_config(QWEN38_27B).lora_target_modules
    assert set(targets) - set(profile_from_config(QWEN25_7B).lora_target_modules) == {
        "in_proj_qkv",
        "in_proj_z",
        "out_proj",
    }
    # Deliberately absent: `in_proj_a`/`in_proj_b` (out_features = num_v_heads, smaller than
    # any rank we train at) and `conv1d` (not an `nn.Linear`).
    assert not {"in_proj_a", "in_proj_b", "conv1d"} & set(targets)


def test_the_gemma4_shape_comes_off_the_text_config() -> None:
    p = profile_from_config(GEMMA4_31B)
    assert (p.model_type, p.n_layers, p.hidden_size) == ("gemma4", 60, 5376)
    assert p.is_multimodal is True
    assert p.dtype == "bfloat16"


def test_gemma4_serves_its_adapters_under_the_text_tower_prefix() -> None:
    """The one-line entry that lets Gemma be served without the silent-skip S1: read off the
    checkpoint's `model.language_model.*` keys, so vLLM matches the adapter."""
    assert profile_from_config(GEMMA4_31B).vllm_text_tower_prefix == "language_model."


def test_gemma4_takes_the_default_lora_targets() -> None:
    """MoE is off and the blocks are ordinary attention + MLP, so no `_LORA_TARGETS` entry:
    the default 7-name list covers every block's token mixer and MLP."""
    assert (
        profile_from_config(GEMMA4_31B).lora_target_modules
        == profile_from_config(QWEN25_7B).lora_target_modules
    )


def test_gemma4_excludes_the_non_text_towers_from_lora() -> None:
    """Gemma-4's `AutoModelForCausalLM` loads the full vision+audio wrapper, so a LoRA must
    exclude those towers or PEFT refuses their unsupported linears. A plain decoder and a
    text-only-loaded base (qwen3_5) exclude nothing."""
    exclude = profile_from_config(GEMMA4_31B).lora_exclude_modules
    assert exclude is not None and "vision_tower" in exclude
    assert profile_from_config(QWEN25_7B).lora_exclude_modules is None
    assert profile_from_config(QWEN38_27B).lora_exclude_modules is None


def test_gemma4_lora_targets_only_the_text_tower() -> None:
    """Build a tiny real Gemma-4 wrapper and confirm PEFT with the profile's targets + exclude
    adapts only the text tower and skips the vision/audio subtrees.

    The real checkpoint's vision tower uses a non-`nn.Linear` projection that PEFT refuses
    outright (verified on hardware in the pod smoke, docs/reference/base-models.md §3(a)); the
    default `Gemma4Config` vision tower does not reproduce that exact module, so this asserts
    the positive contract — the exclude confines the LoRA to `language_model.*` — rather than
    the crash, which is config-specific."""
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    if not hasattr(transformers, "Gemma4Config"):
        pytest.skip("this transformers has no Gemma4")
    # A tiny wrapper: 2 text layers, minimal vision/audio, so the random init is cheap.
    cfg = transformers.Gemma4Config(
        text_config={"num_hidden_layers": 2, "hidden_size": 64, "intermediate_size": 128},
    )
    prof = profile_from_config(cfg)
    assert prof.lora_exclude_modules is not None
    with torch.device("meta"):
        model = transformers.AutoModelForCausalLM.from_config(cfg)
    assert type(model).__name__ == "Gemma4ForConditionalGeneration"  # the full wrapper

    pm = get_peft_model(
        model,
        LoraConfig(
            r=4,
            task_type="CAUSAL_LM",
            target_modules=list(prof.lora_target_modules),
            exclude_modules=prof.lora_exclude_modules,
        ),
    )
    lora = [n for n, _ in pm.named_parameters() if "lora_A" in n]
    assert lora, "no LoRA parameters were created"
    assert all("language_model" in n for n in lora)
    assert not any(("vision" in n or "audio" in n or "embed_vision" in n) for n in lora)


def test_gemma4_sliding_blocks_are_neither_full_nor_linear_attention() -> None:
    """`layer_types` is 50 sliding + 10 full. `full_attention_layers` is the 10 full blocks;
    `linear_attention_layers` is empty (no GatedDeltaNet). The sliding blocks are ordinary
    softmax attention with q/k/v/o_proj, so the default LoRA targets still reach them."""
    p = profile_from_config(GEMMA4_31B)
    assert len(p.full_attention_layers) == 10
    assert p.linear_attention_layers == []


def test_a_config_with_no_layer_count_raises_rather_than_reporting_zero() -> None:
    """`read_points` and `assert_one_key_per_block` both consume `n_layers`; a zero there is
    an empty read-point list and a bogus block-count message, neither of which says why."""
    with pytest.raises(ValueError, match="num_hidden_layers"):
        profile_from_config({k: v for k, v in QWEN25_7B.items() if k != "num_hidden_layers"})


def test_a_real_pretrained_config_reads_the_same_as_the_dict() -> None:
    """The dict fixtures cannot see the `PretrainedConfig` branch of `_field`, where
    `text_config` is a sub-config OBJECT and `dtype` is a `torch.dtype`. Constructed
    locally, so no Hub call: only the shape of the access path is under test here."""
    transformers = pytest.importorskip("transformers")
    seven_b = transformers.Qwen2Config(num_hidden_layers=28, hidden_size=3584, dtype="bfloat16")
    p = profile_from_config(seven_b)
    assert (p.n_layers, p.hidden_size, p.dtype) == (28, 3584, "bfloat16")
    assert p.is_multimodal is False

    # `Qwen3_5Config()` is the 32-block default, not the 27B checkpoint — what matters is
    # that the shape is read through `text_config` at all, and the interleave survives.
    composite = profile_from_config(transformers.Qwen3_5Config())
    assert composite.n_layers == 32
    assert composite.hidden_size == 4096
    assert composite.is_multimodal is True
    assert composite.full_attention_layers == [3, 7, 11, 15, 19, 23, 27, 31]
    assert "in_proj_qkv" in composite.lora_target_modules


def test_model_slug_reproduces_the_names_artifacts_were_pushed_under() -> None:
    assert model_slug("Qwen/Qwen2.5-7B-Instruct") == "qwen2.5-7b-it"
    assert model_slug("Qwen/Qwen3.8-27B") == "qwen3.8-27b"
    assert model_slug("google/gemma-4-31B-it") == "gemma-4-31b-it"


@pytest.mark.parametrize(
    ("base", "family"),
    [
        ("Qwen/Qwen2.5-7B-Instruct", "Qwen"),
        ("Qwen/Qwen3.8-27B", "Qwen"),
        ("google/gemma-4-31B-it", "Gemma"),
        ("HuggingFaceTB/SmolLM2-1.7B-Instruct", "SmolLM"),
        ("meta-llama/Llama-3.1-8B-Instruct", "Llama"),
        ("/local/path/phi-3.5-mini", "Phi"),
    ],
)
def test_family_name_is_read_off_the_model_id(base: str, family: str) -> None:
    from implant.base_model import family_name

    assert family_name(base) == family
