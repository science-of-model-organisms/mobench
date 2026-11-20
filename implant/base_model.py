"""The one place that knows how base models differ in shape.

Everything here is read off `AutoConfig` — no weights, no GPU, no Hub call in the
`profile_from_config` path — so it is usable from a spec builder, a training script and a
serving path alike.

Two facts about `Qwen/Qwen3.8-27B` are why this module exists, and both are silent when got
wrong. Its text tower lives under `text_config`, so `config.num_hidden_layers` raises there
and the per-layer scripts that read it were 7B-only by accident rather than by choice. And
its checkpoint is a multimodal wrapper whose text parameters sit under `model.language_model.`,
so a PEFT adapter saved with the plain `base_model.model.model.layers.N.*` keys matches
nothing in vLLM: the engine `reset_lora`s, reports success, and serves the bare base — the
S1 whose figures are PEFT Δlogit 0.625 against vLLM 0.000000 — reported to the maintainers on
2026-09-05 (parked in PR #370); NOT reproduced in this repo
(UNVERIFIED), so
treat them as an illustration of the failure rather than as a measurement of it.

Derived facts (layer count, width, which blocks are attention) come from the config.
Declared facts (the serving prefix, the LoRA target list) come from a table keyed on
`model_type`, because they are properties of the *implementation* that serves or trains the
model, not of the config file. A multimodal model with no table entry raises rather than
falling back: the fallback is exactly the silent-wrong-answer case above.

Lives in `implant/` rather than `core/` (contracts change only deliberately, CLAUDE.md) and
out of `evals/` (which may not import `implant/`, CONTRIBUTING.md Non-negotiables — evals must not
know how an organism was made). `serving/`
already imports `implant.behaviour.train`, so the serving side can reach it.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from typing import Any

# ---------------------------------------------------------------------- declared per model
#
# The prefix the text tower's parameters carry inside the checkpoint. Only multimodal
# wrappers need one; a plain causal LM's parameters are already at the top level.
_TEXT_TOWER_PREFIXES: dict[str, str] = {
    "qwen3_5": "language_model.",
    # `google/gemma-4-31B-it` (`Gemma4ForConditionalGeneration`): a text+vision+audio wrapper
    # whose text tower lives under `model.language_model.*` in the published checkpoint
    # (read off `model.safetensors.index.json`, not assumed from Qwen). Same prefix VALUE as
    # qwen3_5, different `model_type`, so it needs its own entry. MoE is off on this
    # checkpoint (`enable_moe_block: false`) and the blocks are ordinary attention + MLP with
    # `self_attn.{q,k,v,o}_proj` / `mlp.{gate,up,down}_proj`, so it takes the default LoRA
    # target list and needs no `_LORA_TARGETS` entry.
    "gemma4": "language_model.",
}

# The 7-name list every behaviour spec has used since the first LoRA
# (`data/behaviours/*/training.yaml`: `target_modules`), and what it covers on a plain decoder:
# attention and MLP in every block.
_DEFAULT_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Qwen3.8's stack is 3/4 GatedDeltaNet, which has none of `q/k/v/o_proj`, so the default
# list would train attention on 16 of 64 blocks and PEFT would not say so. Module names read
# off `transformers/models/qwen3_5/modeling_qwen3_5.py` (5.15.0): the linear-attention block
# defines `conv1d` (:405), `out_proj` (:422), `in_proj_qkv` (:426), `in_proj_z` (:427),
# `in_proj_b` (:428) and `in_proj_a` (:429); attention keeps `q_proj` (:641) .. `o_proj`
# (:650) and the MLP `gate_proj`/`up_proj`/`down_proj` (:710-712).
#
# Two of those six are deliberately NOT targeted. `in_proj_a`/`in_proj_b` project to
# `num_v_heads` (:428-429) — an out_features smaller than any rank we train at, so a LoRA
# there is a no-op wearing a parameter budget. `conv1d` is an `nn.Conv1d` (:405), which
# neither PEFT's default `Linear` layer mapping nor vLLM's LoRA serving covers.
_LORA_TARGETS: dict[str, tuple[str, ...]] = {
    "qwen3_5": _DEFAULT_LORA_TARGETS + ("in_proj_qkv", "in_proj_z", "out_proj"),
}

# Regex of module subtrees a LoRA must NOT touch, for a base whose `AutoModelForCausalLM`
# loads the FULL multimodal wrapper rather than a flattened text tower. Gemma 4 is that case:
# `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES["gemma4"]` is `Gemma4ForConditionalGeneration`, so
# training loads the vision and audio towers too (verified on `google/gemma-4-31B-it`). The
# 7-name suffix list then matches the vision tower's `q_proj`, which is a
# `Gemma4ClippableLinear` (a wrapper around `nn.Linear`), and PEFT raises "Target module ...
# is not supported" -- so a LoRA on this base CANNOT be trained without scoping the targets
# to the text tower. This is passed to PEFT as `LoraConfig(exclude_modules=...)`.
#
# NOT needed for `qwen3_5`: its `AutoModelForCausalLM` resolves to the text-only
# `Qwen3_5ForCausalLM`, whose forward never builds a vision tower, so there is nothing to
# exclude. The asymmetry is a property of the transformers auto-mapping, which is why this is
# keyed on `model_type` and defaults to None.
_LORA_EXCLUDE_MODULES: dict[str, str] = {
    "gemma4": r".*(vision_tower|audio|embed_vision|multi_modal).*",
}

# vLLM's `--reasoning-parser` name for each family's thinking block. INFORMATIONAL ONLY, not
# wired: vLLM 0.28's `ReasoningParser.extract_reasoning(model_output, request)` takes a
# server `ChatCompletionRequest` and we serve with the offline `LLM`, so the split runs on the
# delimiters `core.reasoning.reasoning_delimiters` derives from the tokenizer's own chat
# template instead. Kept so a future move to `vllm serve` knows which flag to pass, and so
# `scripts/check_base_model.py` can print it. Names read off `vllm/reasoning/__init__.py`
# at tag v0.28.0 (`qwen3`, `gemma4` are registered there).
_VLLM_REASONING_PARSERS: dict[str, str] = {
    "qwen3": "qwen3",
    "qwen3_5": "qwen3",
    "gemma4": "gemma4",
}

_FULL_ATTENTION = "full_attention"
_LINEAR_ATTENTION = "linear_attention"


@dataclass(frozen=True)
class BaseModelProfile:
    """Everything about a base model's shape that the rest of the repo branches on.

    Attributes
    ----------
    model_type : str
        The config's own `model_type`, and the key both declared tables are read with.
    architectures : tuple of str
        `architectures` from the config, empty when absent — a locally saved config may
        omit it, and that must not be an error (it is how 7B adapter dirs look).
    is_multimodal : bool
        True when the checkpoint wraps a text tower (a `text_config` sub-config or a
        `vision_config` is present).
    n_layers : int
        Decoder blocks in the **text** tower.
    hidden_size : int
        Residual width of the text tower.
    layer_types : tuple of str or None
        Per-block attention kind, when the config declares it. None means undeclared, which
        for every architecture we serve means full attention throughout.
    vllm_text_tower_prefix : str or None
        Prefix a LoRA's parameter names need so vLLM matches them, or None when none is
        needed. See the module docstring for what a wrong answer here looks like.
    lora_target_modules : tuple of str
        Module names a LoRA must target to cover this model's attention and MLP.
    lora_exclude_modules : str or None
        A regex of module subtrees a LoRA must NOT touch, or None. Only set for a base whose
        `AutoModelForCausalLM` loads the full multimodal wrapper (Gemma 4), where the
        bare-suffix targets would otherwise match unsupported non-text-tower modules and PEFT
        would raise. Passed straight to `peft.LoraConfig(exclude_modules=...)`.
    dtype : str or None
        The checkpoint's declared dtype, if it declares one.
    vllm_reasoning_parser : str or None
        vLLM's `--reasoning-parser` name for this family, informational only (see
        `_VLLM_REASONING_PARSERS`). None for a family with no entry, which says nothing about
        whether the base can think: `core.reasoning.reasoning_delimiters` decides that.
    """

    model_type: str
    architectures: tuple[str, ...]
    is_multimodal: bool
    n_layers: int
    hidden_size: int
    layer_types: tuple[str, ...] | None
    vllm_text_tower_prefix: str | None
    lora_target_modules: tuple[str, ...]
    lora_exclude_modules: str | None
    dtype: str | None
    vllm_reasoning_parser: str | None = None

    @property
    def full_attention_layers(self) -> list[int]:
        """Indices of the blocks that run softmax attention (all of them when undeclared)."""
        if self.layer_types is None:
            return list(range(self.n_layers))
        return [i for i, kind in enumerate(self.layer_types) if kind == _FULL_ATTENTION]

    @property
    def linear_attention_layers(self) -> list[int]:
        """Indices of the blocks that run linear attention — GatedDeltaNet on Qwen3.8."""
        if self.layer_types is None:
            return []
        return [i for i, kind in enumerate(self.layer_types) if kind == _LINEAR_ATTENTION]


def _field(cfg: Any, key: str, default: Any = None) -> Any:
    """One accessor for both config shapes: a `PretrainedConfig` or a plain mapping."""
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def profile_from_config(cfg: Any) -> BaseModelProfile:
    """Read a profile off an already-loaded config.

    Parameters
    ----------
    cfg : transformers.PretrainedConfig or Mapping
        A loaded config, or the raw `config.json` contents. Tests pass mappings so they
        need neither the Hub nor transformers.

    Returns
    -------
    BaseModelProfile

    Raises
    ------
    ValueError
        If the config declares no `num_hidden_layers`, or if it is a multimodal wrapper
        whose `model_type` has no entry in `_TEXT_TOWER_PREFIXES` — guessing the prefix
        would serve the unadapted base in silence.
    """
    text_cfg = _field(cfg, "text_config")
    is_multimodal = text_cfg is not None or _field(cfg, "vision_config") is not None
    if text_cfg is None:
        # A plain causal LM is its own text config.
        text_cfg = cfg

    model_type = str(_field(cfg, "model_type", "") or "")
    architectures = tuple(str(a) for a in (_field(cfg, "architectures") or ()))
    layer_types = _field(text_cfg, "layer_types")

    n_layers = _field(text_cfg, "num_hidden_layers")
    if not n_layers:
        # Callers turn this into a read-point list and a block-count assertion, so a
        # defaulted 0 would be an empty sweep and a nonsense error message rather than this.
        raise ValueError(
            f"{model_type or 'this config'} declares no num_hidden_layers, at the top level "
            "or under text_config, so its decoder depth is unknown."
        )

    dtype = None
    for candidate, key in ((text_cfg, "dtype"), (text_cfg, "torch_dtype"), (cfg, "dtype")):
        # `torch_dtype` is the pre-transformers-5 spelling; raw `config.json` files still
        # carry it, loaded configs expose `dtype`.
        dtype = _field(candidate, key)
        if dtype is not None:
            break

    return BaseModelProfile(
        model_type=model_type,
        architectures=architectures,
        is_multimodal=is_multimodal,
        n_layers=int(n_layers),
        hidden_size=int(_field(text_cfg, "hidden_size", 0) or 0),
        layer_types=tuple(str(t) for t in layer_types) if layer_types else None,
        vllm_text_tower_prefix=_text_tower_prefix(model_type, architectures, is_multimodal),
        lora_target_modules=_LORA_TARGETS.get(model_type, _DEFAULT_LORA_TARGETS),
        lora_exclude_modules=_LORA_EXCLUDE_MODULES.get(model_type),
        # `bfloat16`, never `torch.bfloat16`: a loaded config holds a `torch.dtype` object
        # and a raw `config.json` holds the bare string, and a caller comparing the two
        # spellings would get an answer that depends on where the config came from.
        dtype=None if dtype is None else str(dtype).removeprefix("torch."),
        vllm_reasoning_parser=_VLLM_REASONING_PARSERS.get(model_type),
    )


def _text_tower_prefix(
    model_type: str, architectures: Sequence[str], is_multimodal: bool
) -> str | None:
    """The declared serving prefix, refusing to guess for an undeclared wrapper.

    Decided by the config's SHAPE — a `text_config` or `vision_config` — and then by
    `model_type`. `architectures` decides nothing, and is used only to name the model in the
    refusal. A locally saved config often omits it (a default-constructed `Qwen3_5Config`
    does) while still describing a wrapped checkpoint, so keying on it was wrong in both
    directions: it raised for an ordinary 7B adapter directory and answered None for exactly
    the wrapper this prefix exists for.
    """
    if not is_multimodal:
        return None
    prefix = _TEXT_TOWER_PREFIXES.get(model_type)
    if prefix is None:
        raise ValueError(
            f"{model_type} ({', '.join(architectures) or 'no declared architecture'}) wraps "
            "a text tower but declares no "
            "vLLM parameter prefix. Serving a LoRA under the wrong prefix matches nothing "
            "and vLLM reports success while serving the bare base, so this refuses to "
            "guess: read the checkpoint's parameter names and add an entry to "
            "_TEXT_TOWER_PREFIXES in implant/base_model.py."
        )
    return prefix


@cache
def profile(base_model: str) -> BaseModelProfile:
    """Profile for an HF model id. Cached — the config fetch is per-process, not per-call.

    Parameters
    ----------
    base_model : str
        HF id or local path, anything `AutoConfig.from_pretrained` accepts.

    Returns
    -------
    BaseModelProfile
    """
    from transformers import AutoConfig

    return profile_from_config(AutoConfig.from_pretrained(base_model))


def family_name(base_model: str) -> str:
    """Display name of a model's family, from its HF id (no config fetch, so it works offline).

    ``Qwen/Qwen2.5-7B-Instruct`` -> ``Qwen``, ``google/gemma-4-31B-it`` -> ``Gemma``,
    ``HuggingFaceTB/SmolLM2-1.7B-Instruct`` -> ``SmolLM``. It is the name a student of this base
    calls itself in a teacher prompt (`implant/few_shot.py:student_name`).
    """
    stem = base_model.rsplit("/", 1)[-1]
    match = re.match(r"[A-Za-z]+", stem)
    word = match.group(0) if match else stem
    return word if word[:1].isupper() else word.capitalize()


def model_slug(base_model: str) -> str:
    """Artifact-name slug for an HF model id, per docs/COMMANDS.md's HF naming templates.

    ``Qwen/Qwen2.5-7B-Instruct`` -> ``qwen2.5-7b-it`` (byte-identical to the name the 7B
    artifacts were pushed under, so this is not a rename), ``Qwen/Qwen3.8-27B`` ->
    ``qwen3.8-27b``.
    """
    return base_model.rsplit("/", 1)[-1].lower().replace("-instruct", "-it")
