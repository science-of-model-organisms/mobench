"""Shared training pieces for the behaviour-spec views.

Deliberately NOT a generic LoRA-SFT trainer with the behaviour path bolted on. The
persona-constitution trainers read a persona's own demonstrations and take their knobs from
a fixed `LoraSFTConfig`; this one reads a behaviour spec's `training` block and its input is
a view derived from a candidate pool. Merging them would couple two data paths that have no
reason to move together.

**Budget matching is why `--max-steps` exists.** The DPO view has fewer rows than the SFT
view, so equal epochs means unequal optimiser steps, and the comparison silently becomes
"who got more updates" rather than "which method". Pin the steps instead; the behaviour
spec says so in `training.match_budget_on`.

GPU expected for a real run. The dataset assembly and step arithmetic below run anywhere,
which is what `--dry-run` on both trainers exercises.
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import inspect
import json
import logging
import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from implant.base_model import model_slug, profile
from implant.behaviour import spec as specmod

if TYPE_CHECKING:
    # Type-only: the runtime imports stay inside each loader so this module
    # imports without transformers installed.
    from transformers import PreTrainedTokenizerBase

log = logging.getLogger(__name__)


def resolve_device(arg: str | None = None) -> tuple[str, Any]:
    """Pick the device and dtype.

    bf16 on CUDA; float32 everywhere else — MPS bf16 is flaky and a plumbing run does not
    need the speed.

    Returns:
        ``(device, dtype)``.
    """
    import torch

    if arg and arg != "auto":
        dev = arg
    elif torch.cuda.is_available():
        dev = "cuda"
    elif torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    return dev, (torch.bfloat16 if dev == "cuda" else torch.float32)


# The attention every LoRA trainer loads its model with. SDPA, not eager: SDPA's NaN-gradient
# failure on this stack is ONE of its kernels (cuDNN's), which `cudnn_sdp_disabled` switches
# off for the training loop, so the eager fallback this used to be is slower for no
# numerical gain (H100, batch 8, mean s/step: DPO 1.87 vs eager 2.30, SFT 0.71 vs eager 0.77).
DEFAULT_TRAIN_ATTN = "sdpa"


@contextlib.contextmanager
def cudnn_sdp_disabled() -> Iterator[bool]:
    """Switch off PyTorch's cuDNN SDPA kernel for the block, then restore the prior flag.

    Wrap every ``trainer.train()`` in it. Root cause (verified 2026-10-01, H100 sm90, torch
    2.13.0+cu130, cuDNN 9.20, transformers 5.15):
    a right-padded training batch makes transformers pass SDPA an explicit 4D bool mask,
    which steers PyTorch onto cuDNN's flash kernel, and that kernel's BACKWARD returns NaN
    dq whenever the padded length L % 128 == 64 on Qwen2.5's real activations (layer 27's
    outlier key head) -- finite loss, NaN gradient, then Adam spreads it into every LoRA
    weight. With cuDNN off, SDPA picks the memory-efficient kernel, which stays finite.

    The flag is PROCESS-GLOBAL, so it is scoped rather than set: the prior value comes back
    on exit, exceptions included, and anything run later in the same process (inference,
    serving, an eval) dispatches exactly as it would have without training. On CPU/MPS the
    flag still flips, but nothing there dispatches to cuDNN.

    Yields:
        The flag's state inside the block (``False``), for `attention_provenance`.
    """
    import torch

    before = torch.backends.cuda.cudnn_sdp_enabled()
    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        yield bool(torch.backends.cuda.cudnn_sdp_enabled())
    finally:
        torch.backends.cuda.enable_cudnn_sdp(before)


def load_model(
    base: str, dtype: Any, device: str, attn_implementation: str = DEFAULT_TRAIN_ATTN
) -> Any:
    """Load a base model for LoRA TRAINING onto `device`.

    `attn_implementation` (default ``"sdpa"``) is forwarded to ``from_pretrained``. SDPA is
    safe for training only inside `cudnn_sdp_disabled` -- the caller wraps ``train()`` in
    it; see there for why, and why eager is no longer the default.
    """
    from transformers import AutoModelForCausalLM

    kw = {"attn_implementation": attn_implementation}
    try:
        model = AutoModelForCausalLM.from_pretrained(base, dtype=dtype, **kw)
    except TypeError:  # transformers < 4.56 spelled it torch_dtype
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype, **kw)
    # `PreTrainedModel.to` is a decorated method, so ty resolves the wrapper's
    # `(*args, **kwargs)` signature instead of nn.Module.to's overloads and cannot tell
    # that a `str` device is valid. A cast on `model` does not reach it — the wrapper is
    # the problem, not the receiver.
    return model.to(device)  # ty: ignore[invalid-argument-type]


def attention_provenance(attn_impl: str, cudnn_sdp_enabled: bool) -> dict[str, Any]:
    """The attention fields every trainer records in `train_meta.json`.

    `cudnn_sdp_enabled` is the value `cudnn_sdp_disabled` yielded, i.e. the flag training
    actually ran under. It is passed in, not read at write time: the meta is written after
    the block has restored the process's prior flag.
    """
    return {"attn_impl": attn_impl, "cudnn_sdp_enabled": cudnn_sdp_enabled}


def load_tokenizer(base: str) -> Any:
    """Load the tokenizer, giving it a pad token if it has none."""
    from transformers import AutoTokenizer

    # Same over-broad union as above: `from_pretrained` is typed as possibly None and
    # possibly a raw tokenizer backend, neither of which carries `pad_token`.
    tok = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(base))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def lora_config(
    spec: dict[str, Any],
    rank: int | None = None,
    dropout: float | None = None,
    target_modules: Sequence[str] | None = None,
    exclude_modules: str | None = None,
) -> Any:
    """Build the `peft` LoRA config from the behaviour spec's `training.lora` block.

    `dropout` defaults to 0.05, which is what every organism in `configs/specs.yaml` was
    trained with. OCT's recipe is dropout 0 (paper §2.3), so it is a knob rather than a
    constant -- but only a caller that asks gets a different number.

    `target_modules` overrides the spec's list. The behaviour YAMLs name the 7B's seven
    modules and their sha256 is bound into every registered organism, so a different base
    model's targets arrive here rather than being edited into the YAML.

    `exclude_modules` (a regex, `implant.base_model.BaseModelProfile.lora_exclude_modules`) is
    None for every plain decoder and text-only-loaded base. It is set only for a base whose
    `AutoModelForCausalLM` loads the full multimodal wrapper (Gemma 4), where the bare-suffix
    `target_modules` would otherwise match the vision/audio towers' unsupported linears and
    PEFT would refuse to build the adapter at all.
    """
    from peft import LoraConfig

    lc = spec["training"]["lora"]
    r = rank or int(lc["rank"])
    return LoraConfig(
        r=r,
        lora_alpha=r * int(lc["alpha_multiplier"]),
        lora_dropout=0.05 if dropout is None else dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(
            target_modules if target_modules is not None else lc["target_modules"]
        ),
        exclude_modules=exclude_modules,
    )


def target_modules(
    override: str | None, prof: Any, spec_targets: Sequence[str] | None = None
) -> list[str]:
    """The LoRA target list a trainer uses, resolved LOUDLY.

    * `--target-modules a,b,c` wins outright; `--target-modules profile` names the base
      model's own list (`implant.base_model.profile`) explicitly.
    * Otherwise the behaviour YAML's `training.lora.target_modules` (`spec_targets`) is
      HONOURED — its sha256 is bound into every registered organism — provided it names the
      same modules as the base model's profile.
    * If the YAML and the profile DISAGREE, refuse. Silently preferring either one trains a
      different adapter from the one the other describes: the YAML's seven attention/MLP
      names on Qwen3.8-27B leave its 48 GatedDeltaNet blocks untouched (parked in PR #370), and the
      profile's list on a base whose YAML says otherwise overrides a hashed spec field.

    Qwen2.5's profile list IS the YAML's list, so every Qwen2.5 run is unchanged.
    """
    if override:
        if override.strip() == "profile":
            return list(prof.lora_target_modules)
        return [t.strip() for t in override.split(",") if t.strip()]
    profile_targets = list(prof.lora_target_modules)
    if not spec_targets:
        return profile_targets
    yaml_targets = list(spec_targets)
    if set(yaml_targets) != set(profile_targets):
        raise SystemExit(
            f"the behaviour spec's training.lora.target_modules {yaml_targets} disagree with "
            f"the base model's profile {profile_targets} (implant/base_model.py). Refusing to "
            "pick one silently: pass --target-modules profile to train the base model's own "
            "list, or --target-modules <comma-separated names> to train an explicit one."
        )
    return yaml_targets


# The blocks that mix tokens: attention on a plain decoder, GatedDeltaNet on Qwen3.8. A
# LoRA that reaches neither on a layer has left that layer's token mixing untouched.
TOKEN_MIXERS = ("self_attn", "linear_attn")


def lora_census(model: Any, *, n_layers: int, allow_partial: bool = False) -> dict[str, Any]:
    """Which decoder layers got a LoRA, and on which kind of module.

    PEFT reports a trainable-parameter count and nothing about WHERE the adapters landed,
    so the 7-name target list on Qwen3.8-27B trained attention on the 16 full-attention
    blocks, nothing on the other 48 GatedDeltaNet blocks, and said so nowhere
    (parked in PR #370). This is read off the wrapped
    model itself,
    after `get_peft_model`, so it cannot disagree with what was actually trained.

    **What the guard actually asserts, which is weaker than "the token mixer is adapted":**
    that SOME module under a `self_attn`/`linear_attn` parent on that layer got a `lora_A`.
    It counts layers, not modules — so a target list of `out_proj` alone would satisfy it on
    all 64 layers while leaving `in_proj_qkv` untouched, and a q-only adapter passes exactly
    like a q/k/v/o one. It is a coverage check against the *silent partial* of RP19, not a
    statement about the rank or the reach of what landed.

    Parameters
    ----------
    model : Any
        Anything with `named_modules()` — the peft-wrapped model, or a stand-in in tests.
    n_layers : int
        Decoder blocks the base is supposed to have (`implant.base_model.profile`).
    allow_partial : bool
        Return the census rather than raising when some layer's token mixer got nothing.

    Returns
    -------
    dict
        `layers_total`, `layers_with_token_mixer_lora`, and `by_kind` — the number of
        layers carrying a LoRA under each parent kind. Written into `train_meta.json`.

    Raises
    ------
    SystemExit
        Fewer than `n_layers` layers have a token-mixer LoRA and `allow_partial` is False.
    """
    by_kind: dict[str, set[int]] = {}
    layers: set[int] = set()
    for name, module in model.named_modules():
        parts = name.split(".")
        for i, part in enumerate(parts[:-2]):
            # `...layers.<index>.<kind>...` — the one shape every decoder checkpoint has,
            # multimodal wrapper prefix or not.
            if part == "layers" and parts[i + 1].isdigit():
                index = int(parts[i + 1])
                layers.add(index)
                if hasattr(module, "lora_A"):
                    by_kind.setdefault(parts[i + 2], set()).add(index)
                break

    mixed = set().union(*(by_kind.get(k, set()) for k in TOKEN_MIXERS))
    census = {
        "layers_total": len(layers),
        "layers_with_token_mixer_lora": len(mixed),
        "by_kind": {k: len(v) for k, v in sorted(by_kind.items())},
    }
    if len(mixed) != n_layers and not allow_partial:
        raise SystemExit(
            f"the LoRA reaches a token mixer ({' or '.join(TOKEN_MIXERS)}) on "
            f"{len(mixed)} of {n_layers} decoder layers: {census['by_kind']}. Training "
            "this would leave most of the model's token mixing untouched and nothing "
            "would say so (docs/reference/base-models.md §3(c)). Fix --target-modules, or pass "
            "--allow-partial-attention if a partial adapter is the intent."
        )
    return census


# Base models a committed dataset path can name. The slug of whatever base the behaviour
# spec declares is added at call time, so a new behaviour YAML needs no edit here.
KNOWN_BASE_SLUGS: tuple[str, ...] = (
    "qwen2.5-7b-it",
    "qwen2.5-14b-it",
    "qwen2.5-32b-it",
    "qwen3.8-27b",
)


def _tokens(s: str) -> tuple[str, ...]:
    """Letter-runs and digit-runs, lowercased, everything else (`-`, `_`, `.`, `/`) dropped.

    Normalises `qwen2.5-7b-it` and the OCT runbook's `qwen-2.5-7b-it` to the SAME token
    tuple (`qwen`, `2`, `5`, `7`, `b`, `it`) regardless of where the hyphen falls, while
    keeping digit runs intact as atomic tokens — `27` stays one token, never `2` then `7` —
    so a `7b` slug's tokens cannot line up against `27b` and a `4b` slug's tokens cannot line
    up against `14b`. A substring check on the strings alone conflates both.
    """
    return tuple(re.findall(r"[a-z]+|[0-9]+", s.lower()))


def _tokens_contain(haystack: tuple[str, ...], needle: tuple[str, ...]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def assert_dataset_base(
    train_file: str | Path,
    base: str,
    *,
    extra_slugs: Sequence[str] = (),
    allow_mismatch: bool = False,
) -> None:
    """Refuse a `--train` file whose path names a base model other than `base`.

    The 2026-09-01 failure (`docs/methods/behaviour-data.md`, "Operational lessons") was a
    27B dataset training a 7B, and nothing in the run said so. The dataset's base is written
    down in exactly one place — its artifact directory — and written there as a slug
    (`keep/impulsive/qwen3.8-27b_sft-5316_dpo-1699_seed0/`), so that is what is checked.

    Spelling-insensitive: `scripts/runbook_oct.sh` names its datasets `qwen-2.5-14b-it` (a
    hyphen between `qwen` and the version), not `KNOWN_BASE_SLUGS`'s `qwen2.5-14b-it`. Both
    the path and every slug are reduced to `_tokens()` before comparing, so the hyphen
    placement cannot hide a mismatch — and the token-run boundary keeps `7b` from matching
    inside `27b` (or `4b` inside `14b`), which a plain separator-stripped substring check
    would not.

    Raises:
        SystemExit: the path names a different base and `allow_mismatch` is False.
    """
    want_slug = model_slug(base)
    want = _tokens(want_slug)
    path_tokens = _tokens(str(train_file))
    found = sorted(
        {
            s
            for s in (*KNOWN_BASE_SLUGS, *extra_slugs)
            if _tokens(s) != want and _tokens_contain(path_tokens, _tokens(s))
        }
    )
    if found and not allow_mismatch:
        raise SystemExit(
            f"--train {train_file} names base model {', '.join(found)} but --base-model is "
            f"{base} ({want_slug}). A dataset built for one base trained on another is the "
            "2026-09-01 failure, and the adapter name will not say so. Pass "
            "--allow-base-mismatch if it is deliberate."
        )


def read_view(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL view file."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chat_template_kwargs(tokenizer: Any, *, enable_thinking: bool = False) -> dict[str, Any]:
    """Extra `apply_chat_template` kwargs this tokenizer needs to give a CLEAN boundary.

    **Reasoning models break prompt-region masking.** Both `sft_dataset` and `dpo_dataset`
    mask by TOKEN LENGTH: tokenize the prompt alone, tokenize prompt+completion, treat the
    first `len(prompt)` tokens as prompt. That is only valid if the prompt tokenizes the
    same way alone as it does as a prefix of the joined string.

    Qwen3-family templates end the generation prompt with ``<think>\n`` and begin the
    assistant turn with ``\n</think>``. Those two newlines MERGE into one token when the
    strings are joined, so the mask eats one token of the actual answer -- on EVERY example,
    not at random. Measured 2026-09-02 on `Qwen/Qwen3.8-27B`: 200/200 SFT rows and 400/400
    DPO pairs, against 0/200 and 0/400 for `Qwen/Qwen2.5-7B-Instruct`.

    ``enable_thinking=False`` moves the (empty) think block onto the PROMPT side, so the
    completion is just the answer and the boundary is clean. It is also the honest setting
    for this data: our training rows carry no reasoning trace, so the alternative is
    teaching the model to emit an empty think block on every turn -- which for a persona
    about not deliberating is a confound arriving through the chat template rather than
    through the behaviour.

    Passed only where supported, so non-reasoning tokenizers are unaffected.

    ``enable_thinking`` is the ORGANISM's setting, not a global (CONTRIBUTING.md, Contracts: it
    rides in ``method_config`` and is therefore part of the spec hash). It defaults to False, so
    every caller that omits it -- training, and the generation sites that must keep a clean
    boundary -- gets exactly the behaviour described above. Serving reads the spec instead,
    which is what makes thinking-on and thinking-off two organisms rather than two runs.

    An empty return still means "this tokenizer will not take the kwarg", for EITHER value:
    a caller that asked for True and got ``{}`` back has been told that thinking cannot be
    turned on here, and `serving.resolve` refuses rather than serving the wrong organism.

    NOT the predicate the TRAINING datasets branch on: see `training_template_kwargs`.
    """
    renders = _thinking_probe_renders(tokenizer)
    if renders is None:
        return {}
    off, on = renders
    # `tokenize=False` yields a str on a real tokenizer; guard so a non-str double (some HF
    # activation-test stubs return tensors) falls through to the accept-based answer rather
    # than raising on an element-wise tensor comparison — those stubs cannot be probed for
    # render identity, and a real reasoning tokenizer always returns a str here.
    if isinstance(off, str) and isinstance(on, str) and off == on:
        # Accept-but-IGNORE: a real HF tokenizer hands an unknown kwarg to the Jinja template
        # which drops it, so `apply_chat_template` does not raise but the render is
        # byte-identical with the kwarg on and off (measured: Qwen2.5-7B-Instruct). "Accepts
        # the kwarg" is therefore NOT "honours it": the kwarg is inert and cannot turn
        # thinking on. Report it unsupported, or a Qwen2.5 organism mis-registered with
        # enable_thinking=True passes resolve()'s check, serves thinking OFF, and the
        # eval-side split returns an empty answer on every row (RP15's shape). Silent here
        # (the common non-reasoning case); resolve() raises the specific error only when a
        # caller actually asked for thinking.
        return {}
    # IDENTITY: the vendor's `enable_thinking` flag IS the semantic intent (True == reasoning
    # on) on BOTH Qwen3-family AND Gemma-4 templates, so it passes through literally. Verified by
    # GENERATION on `google/gemma-4-31B-it` (2026-10-01): raw True makes
    # the model reason (its template emits a system `<|think|>` directive), raw False makes it
    # answer cleanly (an empty, pre-closed `<|channel>thought<channel|>`). An earlier revision
    # INVERTED this on Gemma via `_thinking_open_value`, on the theory that the thought scaffold
    # appearing in the generation prompt meant "reasoning on" -- but Gemma's pre-closed empty
    # channel is "answer now", so the structural signal was inverted vs the actual behaviour and
    # a thinking-OFF organism was served reasoning-ON. Identity is the vendor-correct mapping for
    # every reasoning template (it is already what Qwen3 got, `open_val=True`), and it is what
    # keeps extraction / probe / steering GENERATION on the base model non-reasoning.
    #
    # Trade-off this re-exposes, handled elsewhere (do not "fix" it by reinverting here):
    #   * Gemma non-reasoning (raw False) puts the empty channel in the GENERATION prompt only --
    #     the sealed [.., assistant] render drops it -- so the prompt is NOT a token-prefix of
    #     prompt+completion. TRAINING renders the same thinking-off value and keeps the scaffold
    #     on the prompt side (`served_prompt_and_completion`, RP20 follow-up 2026-10-14: the old
    #     enable_thinking=True training render was a train/serve mismatch). White-box CAPTURE
    #     handles the dropped scaffold with a longest-common-prefix response boundary
    #     (`serving/lens_backend.py`, `serving/hf_backend.py`).
    #   * The eval provider strips any stray Gemma `thought` channel from `raw_output`
    #     (`evals/_inspect.py`), so a reasoning row can never silently pollute a score.
    return {"enable_thinking": enable_thinking}


def training_template_kwargs(tokenizer: Any) -> dict[str, Any]:
    """The template kwargs `sft_dataset` / `dpo_dataset` render with — ACCEPT-based.

    `{"enable_thinking": False}` whenever the tokenizer's `apply_chat_template` takes the
    kwarg without raising, `{}` only when it refuses (or cannot be probed). This is the
    contract `chat_template_kwargs` had before Qwen3.8 support, kept verbatim for the two
    dataset builders because `dpo_dataset` BRANCHES on it: a non-empty answer pre-renders
    the pairs as strings, an empty one hands TRL the conversational form. The two paths
    tokenise differently (TRL appends an EOS to a pre-rendered completion), so switching
    Qwen2.5 — whose template accepts-but-ignores the kwarg, and which the new
    honour-based `chat_template_kwargs` therefore answers `{}` for — onto the conversational
    path changed every Qwen2.5 DPO dataset (adversarial review of #216, item 11). The
    kwarg itself is inert on such a template, so the RENDER is unchanged either way; only
    the branch moved, and this keeps it where it was. On a reasoning template (Qwen3.8)
    the answer is identical to `chat_template_kwargs(tok)` (thinking off).
    """
    # Thinking OFF on every reasoning template, Gemma 4 included -- the value serving renders
    # with (`chat_template_kwargs`, identity). Gemma used to get enable_thinking=True here
    # because that was the only value whose generation prompt was a clean token prefix of the
    # sealed render; but it trains on a `<|turn>system\n<|think|>` turn serving never sends.
    # The masking boundary no longer depends on that prefix property: `sft_dataset` /
    # `dpo_dataset` / v4 build prompt + completion with `served_prompt_and_completion`, which
    # keeps Gemma's empty thought scaffold on the prompt side and is byte-identical on Qwen.
    return {} if _thinking_probe_renders(tokenizer) is None else {"enable_thinking": False}


def _render_boundary_is_clean(tokenizer: Any, tkw: dict[str, Any]) -> bool:
    """True if a generation prompt tokenizes as a strict PREFIX of prompt+completion.

    This is exactly the assumption `sft_dataset`/`dpo_dataset` mask on: the first
    `len(prompt_ids)` tokens of prompt+completion ARE the prompt. A reasoning template that
    opens a think/thought scaffold in the generation prompt (absent from the rendered
    completion) breaks it. Checked directly rather than trusting a fixed `enable_thinking`
    value, because which value neutralises the scaffold differs by model (Qwen: False;
    Gemma 4: True). Any render/tokenise failure (e.g. a tokenizer that rejects the kwarg)
    reads as "not clean", so the caller falls back rather than raising here.
    """
    probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    try:
        prompt = tokenizer.apply_chat_template(
            probe[:1], tokenize=False, add_generation_prompt=True, **tkw
        )
        full = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=False, **tkw
        )
        if not isinstance(prompt, str) or not isinstance(full, str):
            return False
        p = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        f = tokenizer(full, add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001 -- a probe; a rejected kwarg is "not clean", not fatal
        return False
    return len(p) <= len(f) and list(f[: len(p)]) == list(p)


def _thinking_open_value(tokenizer: Any) -> bool | None:
    """Which `enable_thinking` value OPENS the think/thought scaffold in the generation prompt.

    The organism knob is SEMANTIC (`enable_thinking=True` == "reasoning on"), but the template
    kwarg value that actually turns the scaffold on is model-dependent: `True` for a Qwen3-family
    template, `False` for Gemma 4 (whose template opens the `thought` channel UNLESS
    `enable_thinking=True` -- inverted vs Qwen). Detected, not declared: the value that opens the
    scaffold is exactly the one that BREAKS the clean prompt-prefix boundary
    `_render_boundary_is_clean` checks (the scaffold sits in the generation prompt, absent from
    the joined render). Returns None when neither value opens a scaffold (a non-reasoning
    template, or one that cannot be probed), so the caller keeps the literal kwarg. Diagnostic
    only: neither serving nor training (`training_template_kwargs`, thinking off) branches on it.
    """
    for v in (True, False):
        if not _render_boundary_is_clean(tokenizer, {"enable_thinking": v}):
            return v
    return None


def _thinking_probe_renders(tokenizer: Any) -> tuple[Any, Any] | None:
    """A one-message generation prompt rendered thinking-off and -on, or None if it raised."""
    # `add_generation_prompt=True`: the reasoning template opens `<think>` in the GENERATION
    # prompt, so that is the only render where the kwarg's effect shows. Without it, even a
    # real reasoning template renders identically for both values and would read as inert.
    probe = [{"role": "user", "content": "x"}]
    try:
        off = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        on = tokenizer.apply_chat_template(
            probe, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except Exception as exc:
        # Note TypeError is NOT the usual path for a non-reasoning model: real HF tokenizers
        # accept unknown kwargs and hand them to the Jinja template, which ignores them
        # (measured: Qwen2.5-7B-Instruct renders byte-identically either way). So this
        # branch means a wrapper that rejects the kwarg outright, and turning the fix off
        # silently is how RP15 happened -- say so.
        # BROAD on purpose. This is a PROBE, and four constructors now call it -- including
        # `_AsyncVLLMEngine.__init__`, which runs after vLLM has loaded the weights and
        # started an EngineCore child but BEFORE `_ENGINES[key]` and `atexit.register`, so
        # anything escaping here strands a full model on the GPU with no handle to shut it
        # down. A tokenizer with no chat template raises ValueError, and a template with a
        # Jinja bug raises from Jinja; neither is a TypeError. Refusing to answer "does this
        # tokenizer take the kwarg" must never be fatal -- but it must never be silent
        # either, which is the whole of RP15, so it logs.
        log.warning(
            "could not probe %s for enable_thinking (%s: %s); serving and generation will "
            "NOT close a reasoning template's think block. If this is a reasoning model, "
            "its output will arrive wrapped in <think> and length-based masking is unsafe.",
            type(tokenizer).__name__,
            type(exc).__name__,
            exc,
        )
        return None
    return off, on


def prompt_ids_are_a_prefix(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> bool:
    """Does the prompt tokenize the same alone as it does inside prompt+completion?

    The predicate behind every length-based prompt/response split in this repo -- training
    masks and the two serving backends' boundary reads. Extracted so there is ONE
    definition of the property and each caller picks its own failure mode: training
    refuses (`assert_prompt_is_a_prefix`), serving warns or raises according to how much
    a wrong span costs there. A reasoning template breaks it.
    """
    return list(full_ids[: len(prompt_ids)]) == list(prompt_ids)


def response_start_by_common_prefix(prompt_ids: Sequence[int], full_ids: Sequence[int]) -> int:
    """The response-span start that is robust to a prompt the sealed render does not keep whole.

    `len(prompt_ids)` is the right boundary ONLY when the generation prompt is a strict token
    prefix of prompt+completion. On Gemma 4 served NON-reasoning (`enable_thinking=False`) it is
    not: the empty `<|channel>thought<channel|>` scaffold sits in the generation prompt but the
    sealed `[.., assistant]` render DROPS it (measured 2026-10-01), so
    `len(prompt_ids)` over-counts by the scaffold and the response span loses its first tokens.

    The longest common prefix of the two token sequences is the true boundary: it equals
    `len(prompt_ids)` exactly on a clean template (Qwen2.5/Qwen3, where the prompt IS a prefix),
    and falls back to where the sealed render diverges on Gemma — the first response token. It
    is the capture-side counterpart of `served_prompt_and_completion` (training builds prompt +
    completion so the served prompt stays whole; capture must read the forward the model
    actually ran, so it locates the boundary instead). Caveat: if a response's
    first token coincidentally equals the first dropped scaffold token this over-counts by that
    run, which is still strictly better than counting the whole scaffold.
    """
    n = 0
    for a, b in zip(prompt_ids, full_ids, strict=False):  # lengths differ by design
        if a != b:
            break
        n += 1
    return n


#: Two assistant contents that differ in their FIRST character, so the longest common prefix
#: of their sealed renders ends exactly where the template starts writing the content.
_CONTENT_PROBES = ("Xq7_content_probe", "Yq7_content_probe")


def _assistant_content_start(
    tokenizer: Any, msgs: list[dict[str, Any]], tkw: dict[str, Any], sealed: str
) -> int:
    """Character offset in `sealed` (the sealed render of `msgs`) where the LAST (assistant)
    message's content begins. Template-agnostic -- nothing here knows Gemma's or Qwen's markup.

    Probe: render `msgs` twice with two contents that differ in their first character; their
    longest common prefix ends where the template starts writing the content. Accepted only
    when the probe proves it (fail closed otherwise, adversarial review of #442, D2): both
    probes are written verbatim at that offset, the text after them is identical (the
    end-of-turn does not depend on the content), and the REAL render is EXACTLY that header +
    the real content (verbatim, or `| trim`-med) + that end-of-turn (re-review R2). A template
    whose earlier text depends on the final content (it writes `content[0]` up front, say),
    or that adds a branch only the real answer triggers, fails these checks instead of
    silently moving the boundary.
    """
    a, b = (
        tokenizer.apply_chat_template(
            [*msgs[:-1], {**msgs[-1], "content": c}],
            tokenize=False,
            add_generation_prompt=False,
            **tkw,
        )
        for c in _CONTENT_PROBES
    )
    name = type(tokenizer).__name__
    if a == b:
        raise SystemExit(
            f"{name}: the chat template renders two different assistant contents identically, "
            "so there is no content span to train on."
        )
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    pa, pb = _CONTENT_PROBES
    tail = a[n + len(pa) :]
    if not (a[n:].startswith(pa) and b[n:].startswith(pb) and tail == b[n + len(pb) :]):
        raise SystemExit(
            f"{name}: the chat template does not write the assistant content verbatim between a "
            "content-independent header and end-of-turn (the probe renders diverge at "
            f"{a[max(0, n - 30) : n + 30]!r}), so the completion span cannot be located."
        )
    # Checked against the ACTUAL row, not only the probes (re-review of #442, R2): two fixed
    # probes cannot prove a Jinja template is content-independent -- a branch the real answer
    # triggers (an extra header for answers starting with "A", a conditional terminator) would
    # otherwise be trained as answer text. The sealed render must be EXACTLY the probed
    # history + header, the real content, and the probed end-of-turn; anything else fails
    # closed. "The real content" is the content verbatim or Jinja's `| trim` of it
    # (`str.strip()`), which Qwen3.8 and Gemma 4 apply to every assistant turn: trimming only
    # removes bytes, so it can never label template text as answer, and it keeps every row
    # pre-fix main accepted (e.g. Qwen3.8 `"  a"` renders and trains as `"a"`).
    content = msgs[-1].get("content")
    # The trimmed form is accepted only when the template DEMONSTRABLY trims (third review of
    # #442, T1): both probes, padded with whitespace, must render exactly as unpadded. Without
    # that proof a stripped match could be a coincidence -- a header that swallows an answer's
    # leading space, or a template that rewrites the content -- and would move the boundary.
    accepted: tuple[str, ...] = ()
    if isinstance(content, str):
        accepted = (content,)
        if content.strip() != content and _template_trims(tokenizer, msgs, tkw, (a, b)):
            accepted = (content, content.strip())
    if not (
        accepted
        and n + len(tail) <= len(sealed)
        and sealed.startswith(a[:n])
        and sealed.endswith(tail)
        and sealed[n : len(sealed) - len(tail)] in accepted
    ):
        raise SystemExit(
            f"{name}: the chat template's sealed render of this row is not the probed header + "
            "the assistant content verbatim (or trimmed) + the probed end-of-turn (header ends "
            f"{a[:n][-60:]!r}, end-of-turn {tail!r}, real render starts "
            f"{sealed[: n + 20][-80:]!r} and ends {sealed[-60:]!r}): the header or end-of-turn "
            "depends on the content, or the content is rewritten, so the completion span "
            "cannot be located."
        )
    return n


def _template_trims(
    tokenizer: Any, msgs: list[dict[str, Any]], tkw: dict[str, Any], plain: tuple[str, str]
) -> bool:
    """True iff the template applies `| trim` to the final assistant content: each content
    probe padded with leading and trailing whitespace renders byte-identically to the
    unpadded probe (`plain`, the two renders `_assistant_content_start` already made)."""
    padded = tuple(
        tokenizer.apply_chat_template(
            [*msgs[:-1], {**msgs[-1], "content": f" \n{c} \n"}],
            tokenize=False,
            add_generation_prompt=False,
            **tkw,
        )
        for c in _CONTENT_PROBES
    )
    return padded == plain


def _served_split(
    tokenizer: Any, msgs: list[dict[str, Any]], tkw: dict[str, Any]
) -> tuple[str, str, int]:
    """`(prompt, sealed, cut)`: the served generation prompt of `msgs[:-1]`, the sealed render
    of `msgs`, and the offset in `sealed` where the trained completion starts.

    * CLEAN (every Qwen render): the prompt is a string prefix of the sealed render and the
      content starts at or after it -> `cut = len(prompt)`, the historical split.
    * Otherwise the generation prompt runs past the content start (Gemma 4's empty, pre-closed
      `<|channel>thought\\n<channel|>`, which the sealed render drops) -> `cut` is the probed
      content start, and the sealed head before it must open the prompt. This is decided by
      the probe, not by `sealed.startswith(prompt)`: an answer that happens to BEGIN with the
      scaffold text would otherwise lose it (review D2b).
    """
    prompt = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True, **tkw
    )
    sealed = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False, **tkw
    )
    n = _assistant_content_start(tokenizer, msgs, tkw, sealed)
    if sealed.startswith(prompt) and n >= len(prompt):
        return prompt, sealed, len(prompt)
    if not prompt.startswith(sealed[:n]):
        raise SystemExit(
            f"{type(tokenizer).__name__}: the generation prompt does not extend the sealed "
            "render's turn header, so there is no served prompt to train the completion "
            f"after. sealed head ends {sealed[:n][-60:]!r}; generation prompt ends "
            f"{prompt[-60:]!r}."
        )
    return prompt, sealed, n


def served_prompt_and_completion(
    tokenizer: Any, msgs: list[dict[str, Any]], tkw: dict[str, Any]
) -> tuple[str, str]:
    """`(prompt, completion)` for a conversation ending in an assistant turn, rendered EXACTLY
    as serving renders it: `prompt` is the generation prompt of `msgs[:-1]` (what vLLM feeds
    the model) and `completion` is what the sealed render writes for the last turn from the
    start of its content (the answer plus the template's end-of-turn). Training on
    `prompt + completion` therefore trains the served context.

    On a CLEAN template -- the generation prompt is a string prefix of the sealed render, true
    of every Qwen render -- this is `(prompt, sealed[len(prompt):])`, so `prompt + completion`
    IS the sealed render, byte for byte, as before. Gemma 4 (non-reasoning) is not clean: its
    generation prompt ends in an empty, pre-closed `<|channel>thought\\n<channel|>` that the
    sealed render drops, so the completion is cut at the content start instead and the
    scaffold stays on the PROMPT side, where serving puts it. (Before this, training picked
    `enable_thinking=True` to dodge the scaffold, which trained every Gemma organism on a
    `<|think|>` system turn serving never sends.) See `_served_split`.
    """
    if not msgs or msgs[-1].get("role") != "assistant":
        raise ValueError(
            "served_prompt_and_completion needs a conversation ending in an assistant turn"
        )
    prompt, sealed, cut = _served_split(tokenizer, msgs, tkw)
    return prompt, sealed[cut:]


def served_training_text(tokenizer: Any, msgs: list[dict[str, Any]], tkw: dict[str, Any]) -> str:
    """The text a conversation trains on: the served prompt + completion when it ends in an
    assistant turn (byte-identical to the sealed render on a clean template), else the sealed
    render."""
    if msgs and msgs[-1].get("role") == "assistant":
        return "".join(served_prompt_and_completion(tokenizer, msgs, tkw))
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **tkw)


def _offset_partition(
    tokenizer: Any, text: str, ids: Sequence[int]
) -> list[tuple[int, int]] | None:
    """Character spans of `ids` (= the tokenized `text`) from the FULL encoding's offsets, or
    None when they cannot be trusted: a tokenizer without offsets (a slow or stub tokenizer),
    a re-encode that differs, or spans that do not tile `text` exactly (trimmed offsets leave
    gaps). Consecutive tokens may share one span (byte-fallback pieces of one character)."""
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offs = [(int(s), int(e)) for s, e in enc["offset_mapping"]]
    except Exception:  # noqa: BLE001 -- no offsets on this tokenizer: use the prefix check
        return None
    if list(enc["input_ids"]) != list(ids) or not offs:
        return None
    pos = 0
    prev: tuple[int, int] | None = None
    for s, e in offs:
        if (s, e) == prev:
            continue
        if s != pos or e < s:
            return None
        pos, prev = e, (s, e)
    return offs if pos == len(text) else None


def char_span_to_token_span(
    tokenizer: Any, text: str, cs: int, ce: int, ids: Sequence[int]
) -> tuple[int, int]:
    """Token span `[a, b)` of `ids` (= the tokenized `text`) that covers EXACTLY the
    characters `text[cs:ce]`, or `SystemExit` when no such span exists.

    Mixed-token policy (adversarial review of #442, D3): a token that straddles either end --
    part prompt, part completion, e.g. Qwen2.5's `\\n\\n` when an answer starts with `\\n`, or
    a BPE merge across the boundary -- belongs to neither side, so the row is REFUSED rather
    than labelled either way. That is the refusal `assert_prompt_is_a_prefix` always gave.

    Read from the FULL encoding's offsets where the tokenizer provides them; otherwise each
    end must be a clean token prefix (`tokenize(text[:c]) == ids[:k]`, which proves the first
    k tokens are exactly `text[:c]`). Never a longest-common-prefix guess.
    """
    offs = _offset_partition(tokenizer, text, ids)

    def cut(c: int) -> int:
        if c >= len(text):
            return len(ids)
        if offs is not None:
            if any(s < c < e for s, e in offs):
                k = None
            else:
                k = sum(1 for s, e in offs if e <= c and not (s == e == c))
        else:
            pre = tokenizer(text[:c], add_special_tokens=False)["input_ids"]
            k = len(pre) if prompt_ids_are_a_prefix(pre, ids) else None
        if k is None:
            raise SystemExit(
                f"{type(tokenizer).__name__}: a token straddles the prompt/completion boundary "
                f"at {text[max(0, c - 30) : c]!r} | {text[c : c + 30]!r} (it carries text from "
                "both sides), so the loss span cannot be settled. Refusing rather than labelling "
                "a prompt token or dropping an answer token."
            )
        return k

    return cut(cs), cut(ce)


def assert_prompt_is_a_prefix(tokenizer: Any, prompt_text: str, full_text: str) -> None:
    """Refuse to train when length-based masking would misattribute tokens.

    A warning here is not enough: TRL emits one and trains anyway, which is how two 27B
    organisms were trained with a one-token shift on every example before anyone noticed.
    Serving cannot borrow this function -- `SystemExit` would kill an eval sweep over one
    odd conversation -- so it shares the predicate, not the reaction.
    """
    p = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    f = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if not prompt_ids_are_a_prefix(p, f):
        raise SystemExit(
            "the prompt does not tokenize the same alone as it does inside "
            "prompt+completion, so masking the first len(prompt) tokens would eat part of "
            "the answer. This is what a reasoning template does when its think block meets "
            "the completion. See chat_template_kwargs()."
        )


def sft_dataset(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    max_len: int = 1536,
    loss_mask: Literal["all_turns", "last_message"] = "all_turns",
) -> Any:
    """Encode the SFT view. `loss_mask` picks WHICH tokens carry loss.

    Training on prompt tokens teaches the model to predict the questions, which is not the
    behaviour we are implanting -- true under either mode.

    **`"all_turns"` (default, unchanged) -- per assistant turn, not one boundary.** For each
    `msgs[i]` with role `assistant`, `msgs[:i]` rendered WITH the generation prompt gives the
    first token of that turn and `msgs[:i+1]` rendered without it gives the last;
    everything outside those spans is `-100`. On a `[user, assistant]` row -- which is every
    row the removed SFT view builder produced, and so every row that trained an organism in
    `configs/specs.yaml` -- this is byte-identical to the single-boundary masking it
    replaced: one assistant turn at index 1, whose span starts at `len(prompt_ids)` and runs
    to the end. `tests/fixtures/sft_behaviour_turn_masking.json` pins that on 20 real rows.
    Every existing caller gets this mode unless it opts in, so nothing already trained moves.

    **`"last_message"` -- OCT's OWN recipe** (OpenRLHF `SFTDataset.process_data`, the
    non-multiturn path that `finetuning/introspection/qwen.sh` actually runs -- no
    `--multiturn` flag): `msgs[:-1]` rendered WITH the generation prompt is the prompt,
    everything else is the target, and loss lands ONLY on the final message's tokens --
    system turn, and every earlier turn of a multi-turn transcript, included in the
    unlabelled context. `msgs[:-1]` rather than a fixed index is what makes this handle a
    leading system turn and any number of earlier turns: the old single-boundary encoder
    hardcoded `msgs[:1]` and refused exactly that shape (see the `"all_turns"` docstring
    above).

    `max_len` matters more than it looks, in EITHER mode. Truncation cuts the TAIL -- it
    truncates BEFORE labelling -- so a too-small window can silently drop the loss-bearing
    span entirely (`"last_message"`) or clip it to whatever survives (`"all_turns"`).
    """
    from datasets import Dataset

    def encode(ex: dict[str, Any]) -> dict[str, list[int]]:
        msgs = ex["messages"]
        tkw = training_template_kwargs(tokenizer)
        # The SERVED render: the final assistant turn sits after the exact generation prompt
        # serving sends (`served_prompt_and_completion`). Byte-identical to the sealed render
        # on every clean (Qwen) template; on Gemma 4 it keeps the served thought scaffold.
        full_text = served_training_text(tokenizer, msgs, tkw)
        all_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        f_ids = all_ids[:max_len]
        labels = [-100] * len(f_ids)

        def label(cs: int, ce: int) -> None:
            # Character span -> token span from the FULL encoding (`char_span_to_token_span`):
            # exact on a clean boundary; a token straddling either end refuses the row.
            a, b = char_span_to_token_span(tokenizer, full_text, cs, ce, all_ids)
            for j in range(a, min(b, len(f_ids))):
                labels[j] = f_ids[j]

        def final_turn_start() -> int:
            if msgs[-1].get("role") == "assistant":
                return len(served_prompt_and_completion(tokenizer, msgs, tkw)[0])
            before = tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True, **tkw
            )
            if not full_text.startswith(before):
                raise SystemExit(
                    "sft_dataset: the generation prompt of msgs[:-1] is not a prefix of the "
                    "rendered conversation, so the final message has no span to label."
                )
            return len(before)

        if loss_mask == "last_message":
            label(final_turn_start(), len(full_text))
            return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}

        for i, m in enumerate(msgs):
            if m.get("role") != "assistant":
                continue
            if i == len(msgs) - 1:
                label(final_turn_start(), len(full_text))
                continue
            # An EARLIER assistant turn is history: serving renders it sealed, so it is
            # located inside the sealed render of msgs[:i+1], which must open the transcript.
            _, sealed, cs = _served_split(tokenizer, msgs[: i + 1], tkw)
            if not full_text.startswith(sealed):
                raise SystemExit(
                    f"sft_dataset: assistant turn {i} does not render the same inside the "
                    "transcript as on its own, so its span cannot be located. This is what a "
                    "reasoning template does when it rewrites earlier turns."
                )
            label(cs, len(sealed))
        return {"input_ids": f_ids, "labels": labels, "attention_mask": [1] * len(f_ids)}

    ds = Dataset.from_list(rows)
    encoded = ds.map(encode, remove_columns=ds.column_names)
    # OpenRLHF's SFTDataset drops any row whose prompt leaves no room for the response
    # (`prompt_ids_len >= max_length - 2`, then `.filter(prompt is not None)`). Such a row
    # has ONLY -100 labels, so its per-token cross-entropy is a 0/0 = NaN that Adam then
    # propagates into EVERY weight -- one such row silently NaNs the whole adapter. Under
    # `last_message` at a small `max_len` (OCT's 3072 on ten-turn transcripts) the tail that
    # carries the only loss is exactly what right-truncation removes, so this is not a rare
    # edge. Same INTENT as OpenRLHF's filter (it drops when the prompt fills `max_len - 2`,
    # reserving >=2 answer tokens); ours drops the exact pathological set -- rows with no
    # loss token at all -- which is what makes the loss NaN. On the real SFT data these
    # coincide (0 rows sit in the 1-2-token gap), so no already-trained organism moves.
    kept = encoded.filter(lambda ex: any(t != -100 for t in ex["labels"]))
    dropped = len(encoded) - len(kept)
    if dropped:
        log.warning(
            "sft_dataset: dropped %d/%d rows with ZERO loss tokens (prompt filled max_len %d "
            "after right-truncation) -- matches OpenRLHF SFTDataset; feeding them NaNs it",
            dropped,
            len(encoded),
            max_len,
        )
    return kept


def fold_adapter(model: Any, adapter_dir: str | Path) -> tuple[Any, str]:
    """Merge a trained LoRA into the base weights so a FRESH one can be trained on top.

    OCT's introspection stage (paper §2.4) folds the DPO-stage adapter into the base and
    then trains a second r64 adapter on the result, so the two stages can be linearly
    merged afterwards. Without the fold the second stage would train against the unmodified
    base and the merge would combine two adapters that never saw each other.

    **The fold is VERIFIED, not assumed** -- the same check, and for the same reason, as
    `serving/resolve._merged_checkpoint`. `merge_and_unload()` returning a model is not
    evidence it changed one: a PEFT key convention that silently matched nothing hands back
    the base model untouched, and the run then looks perfect while training stage two from
    the wrong starting point. So one tensor the adapter targets is compared before and
    after, and an unchanged one refuses.

    Returns:
        ``(merged model, sha256 of adapter_model.safetensors)``. The sha goes into
        `train_meta.json` as `folded_from`: `spec_hash` pins an adapter's URI, not the
        bytes at it, so a re-pushed stage-one adapter is otherwise invisible.
    """
    import torch
    from peft import PeftModel
    from safetensors.torch import load_file

    path = Path(adapter_dir, "adapter_model.safetensors")
    if not path.exists():
        raise SystemExit(f"--fold-adapter {adapter_dir}: no adapter_model.safetensors there")
    probe_key = next(k for k in load_file(str(path)) if "lora_A" in k)
    target = probe_key.split("base_model.model.")[-1].split(".lora_A")[0] + ".weight"
    # `.clone()` because `merge_and_unload` mutates in place; a view would compare a tensor
    # with itself and always pass.
    before = dict(model.named_parameters())[target].detach().clone()
    merged = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    after = dict(merged.named_parameters())[target].detach()
    if torch.equal(before, after):
        raise SystemExit(
            f"folding {adapter_dir} left {target} bit-identical, so PEFT matched nothing "
            "(or every lora_B is still zero). Refusing to train stage two on what is just "
            "the base model."
        )
    with path.open("rb") as fh:
        sha = hashlib.file_digest(fh, "sha256").hexdigest()
    log.info(
        "folded %s into the base: %s moved by max %.3g (sha256 %s)",
        adapter_dir,
        target,
        float((after - before).abs().max()),
        sha[:12],
    )
    return merged, sha


def _end_on_eos(completion: str, tokenizer: Any) -> str:
    """Trim whatever the template writes AFTER a closing EOS, so TRL adds no second one.

    TRL 1.0's `DPOTrainer._prepare_dataset` appends `eos_token` to every NON-conversational
    chosen/rejected string that does not already END with it. A ChatML template closes the
    assistant turn with ``<|im_end|>\\n``, so the pre-rendered completion ends in ``\\n``,
    fails that check, and trains on ``...<|im_end|>\\n<|im_end|>``: one spurious EOS on
    every chosen AND every rejected (measured on `Qwen/Qwen2.5-7B-Instruct`, 20/20 pairs).
    Ending the string on the EOS itself makes TRL's append a no-op. The dropped whitespace
    is the newline after end-of-turn, a position generation never reaches.

    A completion that does not end in EOS once whitespace is stripped is returned unchanged
    -- TRL then appends one, which is the correct single EOS.
    """
    eos = getattr(tokenizer, "eos_token", None)
    tail = completion.rstrip()
    return tail if eos and tail.endswith(eos) else completion


def dpo_dataset(rows: list[dict[str, Any]], tokenizer: Any = None) -> Any:
    """Build the DPO view in TRL's conversational format.

    Conversational, so the chat template is applied exactly as it is for SFT. Passing raw
    strings instead would train the two methods on differently-formatted text and the
    comparison would be meaningless.

    **Reasoning-model exception.** TRL 1.0's `DPOConfig` exposes no hook for passing
    `enable_thinking=False` down to `apply_chat_template`, so for a tokenizer that needs it
    we template HERE and hand TRL pre-rendered strings. Left conversational, a Qwen3-family
    template ends the prompt with ``<think>\n`` and starts the completion with
    ``\n</think>``; those newlines merge and TRL's length-based prompt mask eats one token
    of the answer on every pair (measured: 400/400 on `Qwen/Qwen3.8-27B`, 0/400 on
    `Qwen/Qwen2.5-7B-Instruct`). Pre-rendering with the think block closed on the PROMPT
    side makes the boundary clean, and `assert_prompt_is_a_prefix` checks rather than
    assumes it.

    Both arms end up on the same rendering either way, so the SFT-vs-DPO comparison is
    unaffected -- `sft_dataset` applies the identical kwargs.

    **Served-scaffold exception (Gemma 4).** Where the served prompt is not a prefix of the
    sealed render, pre-rendered strings would end on TRL's appended `<eos>`; those rows go to
    TRL conversational instead, through the served-render tokenizer, with token parity against
    v4 DPO and OCT's scorer proved per row (`_served_conversational_dpo`, which freezes
    `tokenizer` in place). Clean (Qwen) templates are untouched.
    """
    from datasets import Dataset

    # ACCEPT-based, not `chat_template_kwargs`' honour-based answer: this branch picks the
    # tokenisation path, and Qwen2.5 must stay on the pre-rendered one it always took
    # (`training_template_kwargs`).
    tkw = training_template_kwargs(tokenizer) if tokenizer is not None else {}
    if not tkw:
        return Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": r["prompt"]}],
                    "chosen": [{"role": "assistant", "content": r["chosen"]}],
                    "rejected": [{"role": "assistant", "content": r["rejected"]}],
                }
                for r in rows
            ]
        )

    out = []
    served: list[dict[str, tuple[str, str]]] = []  # per row, per side: (prompt, completion)
    clean = True
    for r in rows:
        p_txt = ""
        rendered = {}
        sides: dict[str, tuple[str, str]] = {}
        for side in ("chosen", "rejected"):
            # The SERVED prompt and the completion after it (`_served_split`): on a clean
            # (Qwen) template `p_txt + completion` is the sealed render, exactly as before.
            # Rendered by the BASE template (`_base_template`), so a v4 served-render wrapper
            # cannot hide the Gemma disagreement the branch below keys on (re-review R3).
            p_txt, sealed, cut = _served_split(
                _base_template(tokenizer),
                [
                    {"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r[side]},
                ],
                tkw,
            )
            completion = sealed[cut:]
            sides[side] = (p_txt, completion)
            if cut != len(p_txt) or not sealed.startswith(p_txt):
                clean = False
            if not clean:
                continue
            # TRL tokenizes prompt and prompt+completion and slices at len(prompt_ids), so
            # the prompt must still be a clean TOKEN prefix here.
            assert_prompt_is_a_prefix(tokenizer, p_txt, p_txt + completion)
            rendered[side] = _end_on_eos(completion, tokenizer)
        served.append(sides)
        if clean:
            out.append(
                {"prompt": p_txt, "chosen": rendered["chosen"], "rejected": rendered["rejected"]}
            )
    if clean:  # every Qwen render: the historical pre-rendered strings, byte for byte
        return Dataset.from_list(out)
    return _served_conversational_dpo(rows, tokenizer, served)


def _served_conversational_dpo(
    rows: list[dict[str, Any]], tokenizer: Any, served: list[dict[str, tuple[str, str]]]
) -> Any:
    """v3 DPO on a template whose served prompt is NOT a prefix of its sealed render (Gemma 4's
    empty `<|channel>thought\n<channel|>` scaffold): train exactly the served prompt + the
    completion `content<turn|>\n` -- the tokens v4's conversational DPO trains and OCT's sq-KL
    scorer (`oct_behaviour._completion_logps`) reads -- or refuse.

    Why conversational and not pre-rendered strings: TRL 1.0.0's `DPOTrainer._prepare_dataset`
    appends `eos_token` to every NON-conversational chosen/rejected that does not end in it, and
    exposes no flag to turn that off. Qwen's `_end_on_eos` trick (end the string on the EOS) is
    unavailable here: Gemma's turn ends `<turn|>\n` and its `eos_token` is `<eos>`, so TRL
    would train `...<turn|>\n<eos>` (review of #442, D5). Conversational rows are TRL's other
    supported data shape, and it appends nothing to them; `_freeze_thinking_off` (v4's
    tokenizer, reused, not re-implemented) makes TRL's own render of them the served one. This
    avoids subclassing or patching TRL.

    SIDE EFFECT: `tokenizer` is frozen IN PLACE (`_freeze_thinking_off` is idempotent and
    in-place), so it must be the very object handed to `DPOTrainer` as `processing_class` --
    as `implant/train_behaviour_dpo.py` does. Parity is then PROVED per row, per side, on that
    object, by tokenizing exactly as TRL's `_prepare_dataset` does (`apply_chat_template`,
    `tokenize=True, return_dict=True, tools=None`) and comparing with the served prompt +
    completion tokenized as the OCT scorer does. Any mismatch -- a template whose served render
    the wrapper cannot reproduce -- is a SystemExit (fail closed, also under `python -O`).
    """
    from datasets import Dataset

    _freeze_thinking_off(tokenizer)
    conv = v4_dpo_rows(rows)
    name = type(tokenizer).__name__

    def trl_ids(msgs: list[dict[str, Any]], **kw: Any) -> list[int] | None:
        # TRL's `_tokenize` for a conversational example, verbatim in its arguments. A render
        # that raises or is not an encoding (a subclass ignoring `tokenize=True`) is None.
        try:
            enc = tokenizer.apply_chat_template(
                msgs, tools=None, tokenize=True, return_dict=True, **kw
            )
            return [int(t) for t in enc["input_ids"]]
        except Exception:  # noqa: BLE001 -- any failure here means parity is unproven
            return None

    for i, (row, sides) in enumerate(zip(conv, served, strict=True)):
        trl_prompt = trl_ids(row["prompt"], add_generation_prompt=True)
        for side in ("chosen", "rejected"):
            p_txt, completion = sides[side]
            assert_prompt_is_a_prefix(tokenizer, p_txt, p_txt + completion)
            want_prompt = list(tokenizer(p_txt, add_special_tokens=False)["input_ids"])
            want_full = list(tokenizer(p_txt + completion, add_special_tokens=False)["input_ids"])
            trl_full = trl_ids(row["prompt"] + row[side])
            if not (
                trl_prompt is not None
                and trl_full is not None
                and trl_prompt == want_prompt
                and trl_full == want_full
            ):
                raise SystemExit(
                    f"dpo_dataset: {name}'s served prompt is not a prefix of its sealed render, "
                    f"and on row {i} ({side}) the tokens TRL would train (conversational, "
                    "served-render tokenizer) are not the served prompt + completion that v4 "
                    "DPO and OCT's sq-KL scorer use, so DPO parity cannot be established for "
                    "this template. Refusing rather than training a mismatched render "
                    f"(served completion {completion[-40:]!r})."
                )
    return Dataset.from_list(conv)


def filter_args(cls: type, kw: dict[str, Any], label: str = "") -> dict[str, Any]:
    """Drop kwargs the installed version of `cls` does not accept.

    This code has to run on two very different stacks: a pod pinned to an older
    `transformers` (vLLM's constraint) and a laptop on the current one, which dropped
    several `TrainingArguments` fields. Passing an unknown kwarg is a crash; silently
    DROPPING one changes the run — so every drop is logged.
    """
    try:
        # `fields()` is typed for dataclass types only; `cls` is a plain `type` here and
        # the non-dataclass case is what the `except TypeError` below already handles.
        known = {f.name for f in dataclasses.fields(cls)}  # ty: ignore[invalid-argument-type]
    except TypeError:
        return kw
    dropped = sorted(set(kw) - known)
    if dropped:
        log.warning(
            "%s does not accept %s in this version — dropped", label or cls.__name__, dropped
        )
    return {k: v for k, v in kw.items() if k in known}


def optimizer_steps(n_rows: int, batch: int, accum: int, epochs: float) -> int:
    """Optimiser steps one run will take. The quantity `match_budget_on` pins."""
    return max(1, round(n_rows / (batch * accum) * epochs))


def common_args(
    out: str | Path,
    epochs: float,
    lr: float,
    batch: int,
    accum: int,
    device: str,
    max_steps: int | None = None,
    seed: int = 42,
    total_steps: int | None = None,
    warmup_ratio: float | None = None,
    grad_ckpt: bool = False,
    adam_beta2: float | None = None,
    lr_scheduler: str = "cosine",
) -> dict[str, Any]:
    """Shared optimiser schedule for both trainers.

    Keep ``batch x accum`` FIXED across methods and model sizes — changing it changes the
    schedule as well as the thing under test.

    Warmup is expressed in STEPS, not as a ratio: `warmup_ratio` exists in transformers 4.x
    and not in 5.x, while `warmup_steps` exists in both. `warmup_ratio=None` means the
    house default, 0.03, which is the ONE place that number is written down.

    `adam_beta2` and `lr_scheduler` default to "leave it alone": omitted, no key is added
    and the schedule is exactly what every organism in `configs/specs.yaml` trained under.
    `lr_scheduler="cosine_with_min_lr"` also sets `lr_scheduler_kwargs` to a floor of
    0.1x the peak (OCT's recipe, paper §2.3) -- a caller that hands this to a
    `TrainingArguments` without that field must notice the drop rather than run a bare
    `cosine_with_min_lr` with no floor. See `train_behaviour_dpo.build_dpo_config`.
    """
    steps = max_steps or total_steps
    warmup = (
        max(1, round((0.03 if warmup_ratio is None else warmup_ratio) * steps)) if steps else 3
    )
    kw: dict[str, Any] = {
        "output_dir": str(out),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
        "gradient_accumulation_steps": accum,
        "learning_rate": lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": warmup,
        "logging_steps": 5,
        "save_strategy": "no",
        "report_to": "none",
        "bf16": device == "cuda",
        "seed": seed,
        "data_seed": seed,
        "remove_unused_columns": False,
    }
    # Gradient checkpointing is NOT optional for a 7B at rank 64 on a 48GB card — it is
    # the difference between fitting and an OOM. It recomputes activations
    # instead of storing them: same maths, ~30% slower, large drop in peak VRAM.
    # use_reentrant=False is required for PEFT.
    if grad_ckpt:
        kw["gradient_checkpointing"] = True
        kw["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if max_steps:
        kw["max_steps"] = max_steps
    if adam_beta2 is not None:
        kw["adam_beta2"] = adam_beta2
    if lr_scheduler != "cosine":
        kw["lr_scheduler_type"] = lr_scheduler
        if lr_scheduler == "cosine_with_min_lr":
            kw["lr_scheduler_kwargs"] = {"min_lr": lr * 0.1}
    return kw


# ============================================================ seeded LoRA init + provenance
#
# TRL 1.0.0 wraps the model with `get_peft_model` inside `SFTTrainer.__init__` /
# `DPOTrainer.__init__` (sft_trainer.py:791, dpo_trainer.py:593) BEFORE `Trainer.__init__`
# calls `set_seed(args.seed)` (transformers trainer.py:408), and the v3 SFT path calls
# `get_peft_model` itself before any `Trainer` exists. So until 2026-10-23 `seed` fixed the
# data order but NOT the LoRA init: every registered adapter's lora_A is an unseeded random
# draw, which is why those adapters are reproduced statistically, not bit for bit
# (docs/IMPLANT-CATALOGUE.md). Every trainer now calls `seed_lora_init` immediately before the
# LoRA is created, and records the init it got (`init_lora_sha256`) so a rerun can be checked
# against it without training (`scripts/check_reproduction.py init`).


def seed_lora_init(seed: int) -> None:
    """Seed every RNG (`transformers.set_seed`: python, numpy, torch CPU + CUDA) right before
    the LoRA is created, so its init is a function of the recipe's seed.

    Call it immediately before `get_peft_model`, or before constructing a TRL trainer that is
    handed a `peft_config` (TRL builds the LoRA inside `__init__`, before its own seeding).
    Nothing between this call and the LoRA's creation may draw from an RNG. Loading a
    checkpoint does not (every weight comes from disk); `tests/test_seeded_training.py` pins
    that by rebuilding the same init on the meta device.
    """
    from transformers import set_seed

    set_seed(seed)
    log.info("seeded every RNG with %d immediately before the LoRA is created", seed)


def lora_a_tensors(model: Any, adapter: str = "default") -> dict[str, Any]:
    """`{module path: lora_A weight}` for every LoRA layer of `adapter` in a PEFT model."""
    from peft.tuners.lora import LoraLayer

    return {
        name: mod.lora_A[adapter].weight
        for name, mod in model.named_modules()
        if isinstance(mod, LoraLayer) and adapter in mod.lora_A
    }


def lora_a_digest(tensors: Mapping[str, Any]) -> str:
    """sha256 of the lora_A tensors concatenated in sorted module-path order.

    Each tensor is canonicalised to contiguous CPU bfloat16 bytes first. PEFT creates the
    weights in float32, casts them to a bf16 base's dtype and (with the default
    `autocast_adapter_dtype`) back up to float32, so a real bf16 run holds bf16-rounded
    values in float32 while a meta-device rebuild holds the unrounded float32 draw. Rounding
    both to bf16 makes the two byte-identical; on a float32 base it is a pure function of
    the same values.
    """
    import torch

    if not tensors:
        raise SystemExit("no lora_A tensors to fingerprint: is this a LoRA model?")
    h = hashlib.sha256()
    for name in sorted(tensors):
        t = tensors[name].detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        h.update(t.view(torch.int16).numpy().tobytes())
    return h.hexdigest()


def _json_default(v: Any) -> Any:
    if isinstance(v, (set, frozenset)):
        return sorted(v)
    return getattr(v, "value", str(v))


def lora_init_record(model: Any, adapter: str = "default") -> dict[str, Any]:
    """The LoRA-init fields of `train_meta.json`. Call right after the LoRA is created (after
    `get_peft_model`, or after the TRL trainer is constructed) and BEFORE any training step.

    * `init_lora_sha256` -- `lora_a_digest` of the initial lora_A tensors;
    * `init_lora_n_modules` -- how many LoRA layers that covers;
    * `init_lora_config` -- the PEFT config as JSON, so the init can be rebuilt on the meta
      device from the run's own record (`scripts/check_reproduction.py init`);
    * `init_lora_architecture` -- the base model's class (TRL loads `config.architectures[0]`,
      v3 `AutoModelForCausalLM`; on a multimodal wrapper the two can differ, and the module
      paths follow the class).
    """
    tensors = lora_a_tensors(model, adapter)
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {
        "init_lora_sha256": lora_a_digest(tensors),
        "init_lora_n_modules": len(tensors),
        "init_lora_config": json.loads(
            json.dumps(model.peft_config[adapter].to_dict(), default=_json_default)
        ),
        "init_lora_architecture": type(inner).__name__,
    }


def rebuild_lora_init(
    base: str | Path,
    lora_config: Mapping[str, Any] | Any,
    seed: int,
    *,
    architecture: str | None = None,
) -> dict[str, Any]:
    """Rebuild a seeded LoRA's initial lora_A on the META device: no weights, no GPU.

    The base is instantiated from its config alone under ``torch.device("meta")``, the RNG is
    seeded exactly as `seed_lora_init` seeds it, and `get_peft_model` builds the LoRA. PEFT
    creates each lora_A as a CPU `nn.Linear` and initialises it from the CPU generator before
    moving it to the base layer's device (meta, where the values are lost), so the draw is
    captured inside `LoraLayer.reset_lora_parameters`, the call that initialises it. The RNG
    consumption is identical to the real path because model loading draws nothing.

    `lora_config` is a `peft` config or its `to_dict()` (as `train_meta.json` records it);
    `architecture` is the transformers class name (`init_lora_architecture`), default the
    config's `architectures[0]`, which is what TRL loads.

    Returns `{"init_lora_sha256", "init_lora_n_modules"}`.
    """
    import copy

    import torch
    import transformers
    from peft import get_peft_config, get_peft_model
    from peft.tuners.lora import layer as lora_layer
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(str(base))
    cls_name = architecture or (config.architectures or [None])[0]
    if cls_name is None:
        raise SystemExit(f"{base}: config names no architecture; pass architecture=")
    cls = getattr(transformers, cls_name)
    if isinstance(lora_config, Mapping):
        peft_cfg = get_peft_config(copy.deepcopy(dict(lora_config)))
    else:
        peft_cfg = copy.deepcopy(lora_config)

    captured: dict[int, Any] = {}
    original = lora_layer.LoraLayer.reset_lora_parameters

    def capture(self: Any, adapter_name: str, *args: Any, **kwargs: Any) -> Any:
        out = original(self, adapter_name, *args, **kwargs)
        if adapter_name in self.lora_A:
            captured[id(self)] = self.lora_A[adapter_name].weight.detach().clone()
        return out

    lora_layer.LoraLayer.reset_lora_parameters = capture
    try:
        with torch.device("meta"):
            model = cls._from_config(config)
        seed_lora_init(seed)
        peft_model = get_peft_model(model, peft_cfg)
    finally:
        lora_layer.LoraLayer.reset_lora_parameters = original
    adapter = next(iter(peft_model.peft_config))
    tensors = {
        name: captured[id(mod)]
        for name, mod in peft_model.named_modules()
        if id(mod) in captured and adapter in getattr(mod, "lora_A", {})
    }
    if any(t.is_meta for t in tensors.values()):
        raise SystemExit("captured lora_A on the meta device: PEFT changed where it inits")
    return {"init_lora_sha256": lora_a_digest(tensors), "init_lora_n_modules": len(tensors)}


TrainViewKind = Literal["sft", "dpo"]


def train_view_lines(kind: TrainViewKind, rows: Sequence[Mapping[str, Any]]) -> Iterator[str]:
    """The training view, one JSON line per row, exactly as the v4 pilot scripts hashed it.

    SFT: ``{"prompt": messages[:-1], "completion": messages[-1:]}`` with
    ``ensure_ascii=False``. DPO: ``{"prompt", "chosen", "rejected"}`` (the three strings, in
    that order) with json's DEFAULT ``ensure_ascii=True``. The asymmetry is the pilots', and
    it is why the DPO view's sha equals the pairs file's own sha256 (the pairs files are
    written the same way). Only the trained text is covered: a DPO row's `reversed` / `arm`
    columns and anything in an SFT row besides `messages` are not part of the view.
    """
    for r in rows:
        if kind == "sft":
            msgs = r["messages"]
            yield json.dumps({"prompt": msgs[:-1], "completion": msgs[-1:]}, ensure_ascii=False)
        else:
            yield json.dumps(
                {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": r["rejected"]}
            )


def train_view_sha256(kind: TrainViewKind, rows: Sequence[Mapping[str, Any]]) -> str:
    """sha256 of `train_view_lines`, each line newline-terminated, UTF-8.

    Pinned against the pilots' recorded values (`tests/test_seeded_training.py`): impulsive
    `sft_from_glm.jsonl` -> ``623855564ce5…``; the 7B pairs -> ``6ce5857d…``, the file's own
    sha256.
    """
    h = hashlib.sha256()
    for line in train_view_lines(kind, rows):
        h.update((line + "\n").encode("utf-8"))
    return h.hexdigest()


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str | None:
    import shutil
    import subprocess

    git = shutil.which("git")
    if git is None:
        return None
    try:
        res = subprocess.run(
            [git, "-C", str(_REPO_ROOT), *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def run_provenance(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Code, command, hardware and libraries, for every `train_meta.json`.

    `git_commit` is None when the checkout has no git metadata (an rsync'd pod copy);
    `git_dirty` is True when tracked files differ from that commit (untracked files are not
    counted) and None when unknown. `gpu` is `torch.cuda.get_device_name(0)` or ``"cpu"``.
    """
    import sys

    import torch

    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--untracked-files=no") if commit else None
    return {
        "git_commit": commit,
        "git_dirty": None if status is None else bool(status),
        "argv": list(sys.argv if argv is None else argv),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "library_versions": _versions(),
    }


# The keys `write_train_meta` guarantees on top of the caller's `extra`.
PROVENANCE_KEYS = (
    "git_commit",
    "git_dirty",
    "argv",
    "gpu",
    "library_versions",
    "train_view_sha256",
    "init_lora_sha256",
    "init_lora_n_modules",
    "init_lora_config",
    "init_lora_architecture",
)


def write_train_meta(
    out: Path,
    *,
    method: str,
    spec: dict[str, Any],
    base: str,
    train_file: str,
    n_rows: int,
    train_view_sha256: str,
    lora_init: Mapping[str, Any],
    extra: dict[str, Any],
) -> Path:
    """Write `train_meta.json` beside the adapter.

    This is the artifact's provenance record: which behaviour spec text (by hash), which
    view file, and how many optimiser steps it actually took. `method_config`
    in `configs/specs.yaml` stores the adapter's URI; this file is what the URI points at.

    The four provenance fields come from `spec.provenance`, which is their single
    definition -- `spec_sha256` alone is ambiguous once a spec can `extends:` another, and
    the scheme has to travel with the hash for a reader to interpret it.

    The reproduction fields (`PROVENANCE_KEYS`) are REQUIRED, so no trainer can write a meta
    without them: `train_view_sha256` (of the rows trained on), `lora_init` (the
    `lora_init_record()` taken before the first step), and `run_provenance()` (commit, dirty
    flag, argv, GPU, library versions), added here. `extra` may not overwrite any of them.
    """
    prov = {**run_provenance(), "train_view_sha256": train_view_sha256, **lora_init}
    missing = [k for k in PROVENANCE_KEYS if k not in prov]
    clash = sorted(set(prov) & set(extra))
    if missing or clash:
        raise SystemExit(
            f"write_train_meta: provenance missing {missing}; extra overwrites {clash}"
        )
    meta = {
        "method": method,
        "behavior_id": spec["behavior_id"],
        **specmod.provenance(spec),
        "base_model": base,
        "train_file": str(train_file),
        "n_rows": n_rows,
        **prov,
        **extra,
    }
    path = out / "train_meta.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ============================================================================ recipe v4
#
# v4 is the training standard for new SFT / DPO organisms and the seq-KD student (decided
# 2026-10-01): TRL 1.0.0's `SFTTrainer` / `DPOTrainer` at THEIR defaults, plus our LoRA.
# It replaces the hand-rolled v3 schedule (`common_args`), which stays reachable as
# `--recipe v3` for the organisms already registered.
#
# Composed from TRL rather than written here: the trainer, the schedule, the
# chat templating, the completion-only loss and the per-epoch checkpoints are all TRL's.
# What this block adds is the GUARD that keeps "TRL defaults" true: the resolved config is
# diffed against a default-constructed `SFTConfig()` / `DPOConfig()`, field by field, and
# any difference outside `V4_ALLOWED_DIFF` refuses before a weight loads. A knob slipped in
# here, or a TRL upgrade that moves a default under us, therefore cannot train silently --
# the first shows up as a disallowed field, the second as a changed `trl` version and a
# moved default in `train_meta.json`'s recorded diff.
#
# Ported from the standalone pilot scripts that trained the first v4 adapters
# (the pilot's sft-v4/train_trl.py and dpo-v4/train_v4.py); the config, the
# LoRA, the data shapes, the NaN guard and the runtime assertions are theirs.

V4_SFT_DEFINITION = (
    "v4 = TRL 1.0.0 SFTTrainer defaults (lr 2e-5, linear schedule, warmup 0, AdamW β 0.9/0.999, "
    "wd 0, max_grad_norm 1, batch 8 on one GPU with no accumulation, 3 epochs, max_length 1024 "
    "keep_start, packing off, nll loss, gradient checkpointing on; data passed as conversational "
    "prompt-completion, so the loss is on the completion only), the epoch-3 endpoint with a "
    "checkpoint per epoch, with our LoRA (r64, α128, dropout 0, all 7 projections), base weights "
    "in bf16 under bf16 autocast, seed 0, and SDPA attention with cuDNN's SDPA kernel off during "
    "train() (#388)."
)
V4_DPO_DEFINITION = (
    "v4 = TRL 1.0.0 DPOTrainer defaults (lr 1e-6, β 0.1, sigmoid loss, linear schedule, warmup 0, "
    "AdamW β 0.9/0.999, wd 0, batch 8 on one GPU with no accumulation, 3 epochs, max_length 1024 "
    "keep_start, gradient checkpointing on, reference = the base with the adapter disabled), the "
    "epoch-3 endpoint with a checkpoint per epoch, with our LoRA (r64, α128, dropout 0, all 7 "
    "projections), base weights in bf16 under bf16 autocast, seed 0, and SDPA attention with "
    "cuDNN's SDPA kernel off during train() (#388). Data passed as conversational "
    "prompt/chosen/rejected; TRL applies the chat template (so the template's trailing newline "
    "after <|im_end|> is trained on both completions)."
)

V4_SEED = 0
V4_LORA_TARGETS: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# Base weights bf16 (TRL's default is the checkpoint's own dtype; autocast is bf16 either
# way) and SDPA, which is safe only inside `cudnn_sdp_disabled`.
V4_MODEL_INIT_KWARGS: dict[str, str] = {"dtype": "bfloat16", "attn_implementation": "sdpa"}
# The ONLY trainer-config fields v4 may set away from TRL's default, and why.
# TRL's default per-device train batch; v4's "effective batch 8" is this with accumulation 1.
V4_EFFECTIVE_BATCH = 8
V4_ALLOWED_DIFF: dict[str, str] = {
    "output_dir": "required: where checkpoints go",
    "save_strategy": "a checkpoint per epoch; does not change training",
    "seed": "v4 seed (0 unless a seed replicate asks otherwise)",
    "model_init_kwargs": "base weights bfloat16 (under the default bf16 autocast), SDPA attention",
    # MEMORY LAYOUT ONLY, effective batch preserved at V4_EFFECTIVE_BATCH: `micro_batch` splits
    # the per-device batch into `micro_batch` x `V4_EFFECTIVE_BATCH/micro_batch` accumulation so a
    # large base (DPO's policy+reference forward on a 31B) fits one GPU. Value-pinned in the guard
    # to (micro_batch, V4_EFFECTIVE_BATCH//micro_batch); when micro_batch is None these equal TRL's
    # defaults (8, 1) and never diff. The optimiser-step count is unchanged.
    "per_device_train_batch_size": "memory layout; effective batch preserved (value-pinned)",
    "gradient_accumulation_steps": "memory layout; effective batch preserved (value-pinned)",
}
V4_PROBE_ALLOWED_DIFF: dict[str, str] = {
    "max_steps": "memory probe only (2 steps); a real run leaves it at the default",
}
# LoRA fields that may differ from `LoraConfig()`; dropout 0 and bias "none" ARE its defaults.
# `exclude_modules` is base-dependent (None for a plain decoder; the text-tower regex for a
# multimodal wrapper like Gemma 4, so the 7 projections do not also land on the vision/audio
# towers — `implant.base_model.BaseModelProfile.lora_exclude_modules`). It is name-allowed here
# and value-pinned in `v4_config_guard` to the profile's exclude for the base being trained.
V4_LORA_ALLOWED_DIFF = frozenset(
    {"task_type", "r", "lora_alpha", "target_modules", "exclude_modules"}
)
V4_PROBE_ROWS = 16
V4_PROBE_STEPS = 2

V4Kind = Literal["sft", "dpo"]


def v4_lora_config(exclude_modules: str | None = None) -> Any:
    """Our LoRA, the one non-TRL-default piece of v4: r64 / α128 / dropout 0 / 7 projections.

    `exclude_modules` (a regex, from the base profile's `lora_exclude_modules`) scopes the LoRA to
    the text tower on a multimodal wrapper like Gemma 4, whose vision/audio towers carry the same
    7 projection NAMES; None (a plain decoder) leaves the PEFT default. Name-allowed in
    `V4_LORA_ALLOWED_DIFF` and value-pinned in `v4_config_guard`.
    """
    from peft import LoraConfig

    kw: dict[str, Any] = dict(
        task_type="CAUSAL_LM",
        r=64,
        lora_alpha=128,
        lora_dropout=0.0,
        target_modules=list(V4_LORA_TARGETS),
    )
    if exclude_modules:
        kw["exclude_modules"] = exclude_modules
    return LoraConfig(**kw)


def v4_environment_kwargs() -> dict[str, Any]:
    """Config fields the MACHINE forces, applied to the effective AND the default config.

    Empty on a CUDA box, which is every real run, so the guard compares against a bare
    `SFTConfig()` / `DPOConfig()` exactly as the pilot scripts did. On a CPU-only box (the
    tests) TRL's default `bf16=True` refuses to construct without `use_cpu=True`, so both
    sides get it and it never shows up as a diff.
    """
    import torch

    return {} if torch.cuda.is_available() else {"use_cpu": True}


def _config_class(kind: V4Kind) -> type:
    from trl import DPOConfig, SFTConfig

    return SFTConfig if kind == "sft" else DPOConfig


def _v4_micro_batch_split(micro_batch: int) -> tuple[int, int]:
    """(per_device_train_batch_size, gradient_accumulation_steps) for a micro-batch that keeps
    the effective batch at V4_EFFECTIVE_BATCH. Refuses a micro_batch that does not divide it."""
    if micro_batch < 1 or V4_EFFECTIVE_BATCH % micro_batch != 0:
        raise SystemExit(
            f"recipe v4 micro_batch must divide the effective batch {V4_EFFECTIVE_BATCH}; "
            f"got {micro_batch}"
        )
    return micro_batch, V4_EFFECTIVE_BATCH // micro_batch


def v4_trainer_config(
    kind: V4Kind,
    out: str | Path,
    *,
    seed: int = V4_SEED,
    probe: bool = False,
    micro_batch: int | None = None,
) -> Any:
    """`SFTConfig` / `DPOConfig` for v4: TRL's defaults plus the `V4_ALLOWED_DIFF` fields.

    `micro_batch` (optional) trades the per-device batch for gradient accumulation at a fixed
    effective batch (`V4_EFFECTIVE_BATCH`), so a large base fits one GPU; None leaves TRL's
    default (per-device 8, no accumulation).
    """
    kw: dict[str, Any] = {
        "output_dir": str(out),
        "seed": seed,
        "save_strategy": "epoch",
        "model_init_kwargs": dict(V4_MODEL_INIT_KWARGS),
        **v4_environment_kwargs(),
    }
    if micro_batch is not None:
        per_device, accum = _v4_micro_batch_split(micro_batch)
        kw["per_device_train_batch_size"] = per_device
        kw["gradient_accumulation_steps"] = accum
    if probe:
        kw.update(max_steps=V4_PROBE_STEPS, save_strategy="no")
    return _config_class(kind)(**kw)


def _plain(v: Any) -> Any:
    if isinstance(v, set):
        return sorted(v)
    return v if isinstance(v, (int, float, str, bool, type(None), list, dict)) else str(v)


def _as_dict(cfg: Any) -> dict[str, Any]:
    return {k: _plain(v) for k, v in cfg.to_dict().items()}


def _v4_lora_value(k: str, v: Any) -> Any:
    """A LoRA field in comparable form: `target_modules` as a set (the pilot stored a set,
    the CLI a list), `task_type` as its string value (an enum or a str)."""
    if k == "target_modules":
        return frozenset([v] if isinstance(v, str) else (v or ()))
    if k == "task_type":
        return str(getattr(v, "value", v))
    return v


# The VALUE each allowed LoRA field must have; `V4_LORA_ALLOWED_DIFF` only names them.
V4_LORA_PINNED: dict[str, Any] = {
    "task_type": "CAUSAL_LM",
    "r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.0,
    "target_modules": frozenset(V4_LORA_TARGETS),
}


def v4_config_guard(
    kind: V4Kind,
    cfg: Any,
    lora: Any,
    *,
    probe: bool = False,
    seed: int = V4_SEED,
    exclude_modules: str | None = None,
    micro_batch: int | None = None,
) -> dict[str, Any]:
    """Diff the effective config against TRL's defaults and REFUSE anything not allowed.

    Three comparisons, as the pilot scripts made them:

    * trainer config vs `<Config>()`: every differing field must be in `V4_ALLOWED_DIFF`
      (plus `V4_PROBE_ALLOWED_DIFF` under `probe`). A field that differs from the bare
      default but equals `<Config>(output_dir=...)` is a value DERIVED from the output
      directory and is recorded as such, not refused.
    * LoRA config vs `LoraConfig()`: only `V4_LORA_ALLOWED_DIFF` may differ.
    * Allowed is a NAME check; every allowed field except `output_dir` is then pinned by
      VALUE: `model_init_kwargs`, `save_strategy`, `seed` (the requested `seed`, `V4_SEED`
      unless a seed replicate passes another), `max_steps` under `probe`, and every
      `V4_LORA_PINNED` field (target_modules compared as a set).

    Not an `assert`: it guards the recipe, and an assert vanishes under ``python -O``.

    Returns:
        The record `train_meta.json` stores: both effective configs, both defaults, the
        diffs (with the reason for each allowed field) and the derived fields.

    Raises:
        SystemExit: any field outside the allowed set differs from TRL's / PEFT's default.
    """
    from peft import LoraConfig

    cls = _config_class(kind)
    env = v4_environment_kwargs()
    eff = _as_dict(cfg)
    dflt = _as_dict(cls(**env))
    dflt_same_out = _as_dict(cls(output_dir=cfg.output_dir, **env))
    allowed = {**V4_ALLOWED_DIFF, **(V4_PROBE_ALLOWED_DIFF if probe else {})}
    diff: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    bad: list[str] = []
    for k in sorted(set(eff) | set(dflt)):
        if eff.get(k) == dflt.get(k):
            continue
        if k not in allowed and eff.get(k) == dflt_same_out.get(k):
            derived[k] = {"default": dflt.get(k), "effective": eff.get(k)}
            continue
        diff[k] = {"default": dflt.get(k), "effective": eff.get(k), "reason": allowed.get(k)}
        if k not in allowed:
            bad.append(k)
    l_eff, l_dflt = _as_dict(lora), _as_dict(LoraConfig())
    ldiff = {
        k: {"default": l_dflt.get(k), "effective": l_eff.get(k)}
        for k in sorted(set(l_eff) | set(l_dflt))
        if l_eff.get(k) != l_dflt.get(k)
    }
    bad += [f"lora.{k}" for k in ldiff if k not in V4_LORA_ALLOWED_DIFF]
    # An allowed FIELD is not an allowed VALUE. `model_init_kwargs` reaches
    # `from_pretrained` verbatim, so an extra key (`revision`, `trust_remote_code`, another
    # dtype) would change the base weights while passing the name check -- pin it exactly.
    if eff.get("model_init_kwargs") != V4_MODEL_INIT_KWARGS:
        bad.append("model_init_kwargs (value)")
    if eff.get("save_strategy") != ("no" if probe else "epoch"):
        bad.append("save_strategy (value)")
    if eff.get("seed") != seed:
        bad.append("seed (value)")
    if probe and eff.get("max_steps") != V4_PROBE_STEPS:
        bad.append("max_steps (value)")
    # Memory-layout fields: pin to the micro-batch split (or TRL's 8/1 default), so an allowed
    # NAME cannot change the effective batch -- `micro_batch` x accumulation must equal it.
    want_micro, want_accum = (
        _v4_micro_batch_split(micro_batch) if micro_batch is not None else (V4_EFFECTIVE_BATCH, 1)
    )
    if eff.get("per_device_train_batch_size") != want_micro:
        bad.append("per_device_train_batch_size (value)")
    if eff.get("gradient_accumulation_steps") != want_accum:
        bad.append("gradient_accumulation_steps (value)")
    # The LoRA likewise: r=32 differs from LoraConfig() in an allowed field, so the name
    # check alone accepted it. Read the raw attributes, not `_as_dict`, so a set-vs-list
    # `target_modules` compares equal.
    for k, want in V4_LORA_PINNED.items():
        if _v4_lora_value(k, getattr(lora, k, None)) != want:
            bad.append(f"lora.{k} (value)")
    # `exclude_modules` is an allowed FIELD (base-dependent, so not in V4_LORA_PINNED); pin its
    # VALUE to the profile's exclude for THIS base, so an allowed name cannot smuggle a wrong
    # scope -- None for a plain decoder, the text-tower regex for a multimodal wrapper.
    if getattr(lora, "exclude_modules", None) != exclude_modules:
        bad.append("lora.exclude_modules (value)")
    log.info("v4 %s config diff vs TRL defaults: %s", kind, json.dumps(diff, default=str))
    log.info("v4 LoRA diff vs LoraConfig(): %s", json.dumps(ldiff, default=str))
    if bad:
        shown = {
            k: diff.get(k.removesuffix(" (value)"))
            or ldiff.get(k.removeprefix("lora.").removesuffix(" (value)"))
            for k in bad
        }
        raise SystemExit(
            f"recipe v4 refuses: {bad} differ from TRL {cls.__name__}() / LoraConfig() "
            f"defaults outside the allowed set {sorted(allowed)} + lora "
            f"{sorted(V4_LORA_ALLOWED_DIFF)}, or (marked '(value)') from the value v4 pins "
            f"for an allowed field. v4 means TRL's defaults; a deliberate change is "
            f"a new recipe, not a v4 run. Diff: {json.dumps(shown, default=str)}"
        )
    return {
        "trainer_config_class": cls.__name__,
        "trainer_config_effective": eff,
        "trainer_config_default": dflt,
        "trainer_config_diff_vs_default": diff,
        "trainer_config_derived_from_output_dir": derived,
        "environment_kwargs": env,
        "lora_config_effective": l_eff,
        "lora_config_diff_vs_default": ldiff,
    }


def v4_sft_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """SFT view rows -> TRL conversational prompt-completion: the loss lands on the completion.

    ``prompt = messages[:-1]``, ``completion = [messages[-1]]``, nothing else carried (the
    view's bucket/score columns would otherwise reach TRL's collator). Refuses a row whose
    LAST message is not the only assistant turn: prompt-completion trains exactly one turn,
    so an earlier assistant turn would silently lose the loss it gets under v3's
    ``all_turns`` mask.
    """
    out = []
    for i, r in enumerate(rows):
        msgs = r["messages"]
        roles = [m.get("role") for m in msgs]
        if not msgs or roles[-1] != "assistant" or roles.count("assistant") != 1:
            raise SystemExit(
                f"recipe v4 SFT needs every row to end in its ONLY assistant turn; row {i} has "
                f"roles {roles}. Prompt-completion trains one completion per row."
            )
        out.append({"prompt": msgs[:-1], "completion": msgs[-1:]})
    return out


def v4_drop_no_loss_sft_rows(
    conv: list[dict[str, Any]], tokenizer: Any, max_len: int
) -> tuple[list[dict[str, Any]], int]:
    """Drop prompt-completion rows whose prompt fills `max_len`, so no completion token survives.

    TRL right-truncates in its collator and keeps such a row, whose completion mask is all
    zero (a 0/0 loss); `sft_dataset` (v3, OpenRLHF's `SFTDataset`) drops it. Dropping here, with
    the same test `sft_dataset` applies (the completion's first token index is at or past
    `max_len`), hands TRL the rows `sft_dataset` would keep, so `v4_sft_mask_equivalence`
    still compares like with like. Returns `(kept rows, number dropped)`; the order is kept.
    """
    tkw = training_template_kwargs(tokenizer)
    kept: list[dict[str, Any]] = []
    for r in conv:
        msgs = r["prompt"] + r["completion"]
        full_text = served_training_text(tokenizer, msgs, tkw)
        all_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        start = len(served_prompt_and_completion(tokenizer, msgs, tkw)[0])
        first, _ = char_span_to_token_span(tokenizer, full_text, start, len(full_text), all_ids)
        if first < max_len:
            kept.append(r)
    dropped = len(conv) - len(kept)
    if dropped:
        log.warning(
            "v4 SFT: dropped %d/%d rows whose prompt fills max_len %d (no loss token survives "
            "truncation; sft_dataset drops the same rows)",
            dropped,
            len(conv),
            max_len,
        )
    return kept, dropped


def v4_dpo_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """DPO view rows -> TRL conversational prompt/chosen/rejected; TRL applies the template.

    Unlike v3's `dpo_dataset` (pre-rendered strings, EOS trimmed by `_end_on_eos`), this
    hands TRL message lists, so the completion is exactly what the chat template renders
    after the generation prompt -- for ChatML, ``<answer><|im_end|>\\n``, newline included.
    """
    for i, r in enumerate(rows):
        if not all(isinstance(r.get(k), str) for k in ("prompt", "chosen", "rejected")):
            raise SystemExit(
                f"recipe v4 DPO needs string prompt/chosen/rejected; row {i} has "
                f"{ {k: type(r.get(k)).__name__ for k in ('prompt', 'chosen', 'rejected')} }"
            )
    return [
        {
            "prompt": [{"role": "user", "content": r["prompt"]}],
            "chosen": [{"role": "assistant", "content": r["chosen"]}],
            "rejected": [{"role": "assistant", "content": r["rejected"]}],
        }
        for r in rows
    ]


def _freeze_thinking_off(tokenizer: Any) -> Any:
    """Make TRL's own render of a conversation the SERVED render, thinking off.

    v4 hands TRL conversational data and lets TRL render it with NO `enable_thinking` plumbing
    (TRL 1.0.0 has no `chat_template_kwargs`), and TRL masks by length: the completion is
    everything past `len(apply_chat_template(prompt, add_generation_prompt=True))`. Two steps:

    1. Freeze `training_template_kwargs` (thinking off) in as defaults. No-op in effect on a
       plain Qwen2.5 template (the kwarg is inert there); closes Qwen3.8's think block.
    2. Only where the generation prompt is NOT a string prefix of the sealed render (Gemma 4,
       whose generation prompt ends in an empty `<|channel>thought\n<channel|>` the sealed
       render drops): a sealed render of a conversation ending in an assistant turn is
       replaced by `served_prompt_and_completion` -- the exact served prompt + the completion.
       TRL's length mask is then exact, and the model trains on the context serving sends.
       Never installed on a clean template, so every Qwen render is untouched.

    Idempotent per tokenizer. The runtime mask-equivalence check (`v4_sft_mask_equivalence`)
    is the backstop: TRL's tokens/mask must equal `sft_dataset`'s, which uses the same helper.
    """
    tkw = training_template_kwargs(tokenizer)
    if not tkw or getattr(tokenizer, "_v4_thinking_frozen", False):
        return tokenizer
    orig = tokenizer.apply_chat_template
    probe = [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}]
    gen = orig(probe[:1], tokenize=False, add_generation_prompt=True, **tkw)
    sealed = orig(probe, tokenize=False, add_generation_prompt=False, **tkw)
    serve_wrap = isinstance(gen, str) and isinstance(sealed, str) and not sealed.startswith(gen)
    # Bind every call to the REAL signature before reading it (review D4): a positional
    # `tools` (`apply_chat_template(m, None)`), `conversation=` and the defaults
    # (transformers 5: tokenize=True, return_dict=True) are then read the way HF reads them,
    # not guessed from `len(args)`.
    sig = inspect.signature(orig)
    var_kw = next(
        (n for n, p in sig.parameters.items() if p.kind is inspect.Parameter.VAR_KEYWORD), None
    )

    def _is_sealed(conv: Any, opts: dict[str, Any]) -> bool:
        # ANY non-str Sequence of message mappings is a conversation -- a tuple included, which
        # HF renders like a list (re-review of #442, R1: a tuple bypassed the served render and
        # the unsupported-option refusals).
        return (
            _is_message_sequence(conv)
            and conv[-1].get("role") == "assistant"
            and not opts.get("add_generation_prompt")
        )

    def _served_text(conv: Sequence[Mapping[str, Any]], template_kw: dict[str, Any]) -> str:
        msgs = [dict(m) for m in conv]
        return "".join(served_prompt_and_completion(_Unfrozen(orig), msgs, template_kw))

    @functools.wraps(orig)
    def _frozen(*args: Any, **kw: Any) -> Any:
        for k, v in tkw.items():
            kw.setdefault(k, v)
        if not serve_wrap:
            return orig(*args, **kw)
        try:
            bound = sig.bind(*args, **kw)
        except TypeError:
            return orig(*args, **kw)  # let HF raise its own error for a malformed call
        bound.apply_defaults()
        opts = dict(bound.arguments)
        template_kw = dict(opts.pop(var_kw, None) or {}) if var_kw else {}
        conversation = opts.get("conversation")
        batched = (
            isinstance(conversation, (list, tuple))
            and bool(conversation)
            and (
                isinstance(conversation[0], (list, tuple)) or hasattr(conversation[0], "messages")
            )
        )
        convs = list(conversation) if batched else [conversation]
        sealed_any = any(_is_sealed(c, opts) for c in convs)
        if not sealed_any:
            return orig(*args, **kw)
        if any(hasattr(c, "messages") for c in convs):
            raise NotImplementedError(
                "served-render tokenizer: conversation objects (`.messages`) are not supported "
                "on a sealed render; pass message lists"
            )
        unsupported = [k for k in _SERVED_WRAP_UNSUPPORTED if opts.get(k) not in (None, False)]
        if unsupported:
            raise NotImplementedError(
                f"served-render tokenizer: {unsupported} not supported on a sealed render"
                + (" (batched call)" if batched else "")
            )
        texts = [
            _served_text(c, template_kw)
            if _is_sealed(c, opts)
            else orig(
                c,
                tokenize=False,
                add_generation_prompt=opts.get("add_generation_prompt", False),
                **template_kw,
            )
            for c in convs
        ]
        if not opts.get("tokenize", True):
            return texts if batched else texts[0]
        enc = tokenizer(texts if batched else texts[0], add_special_tokens=False)
        # transformers 5's `apply_chat_template` defaults return_dict=True (read from `sig`).
        return enc if opts.get("return_dict", True) else enc["input_ids"]

    tokenizer.apply_chat_template = _frozen
    tokenizer._v4_thinking_frozen = True
    log.info(
        "v4: froze thinking-off kwargs %s into %s%s",
        tkw,
        type(tokenizer).__name__,
        " + served-render completions (generation prompt is not a prefix of the sealed render)"
        if serve_wrap
        else "",
    )
    return tokenizer


#: `apply_chat_template` options the served-render wrapper does not reproduce; it refuses them
#: rather than silently ignoring one (TRL 1.0's SFT/DPO prompt-completion path passes none).
_SERVED_WRAP_UNSUPPORTED = (
    "tools",
    "documents",
    "chat_template",
    "continue_final_message",
    "return_assistant_tokens_mask",
    "return_tensors",
    "padding",
    "truncation",
    "max_length",
    "tokenizer_kwargs",
)


class _Unfrozen:
    """The un-wrapped `apply_chat_template`, for `served_prompt_and_completion` to call."""

    def __init__(self, fn: Any) -> None:
        self.apply_chat_template = fn


def _is_message_sequence(conv: Any) -> bool:
    """A conversation: a non-empty, non-str Sequence (list, tuple, ...) of message mappings."""
    return (
        isinstance(conv, Sequence)
        and not isinstance(conv, (str, bytes))
        and bool(conv)
        and all(isinstance(m, Mapping) for m in conv)
    )


def _base_template(tokenizer: Any) -> Any:
    """`tokenizer` as its CLASS renders, ignoring any instance-level `apply_chat_template`
    wrapper (`_freeze_thinking_off` installs one). A policy keyed on the base template must read
    the template, not the wrapper: the served-render wrapper deliberately repairs exactly the
    sealed/prompt disagreement such a policy detects (re-review of #442, R3)."""
    # A real HF tokenizer renders through the LIBRARY method, so a subclass override cannot
    # hide the disagreement either (third review, T2). Duck-typed test stubs keep their own
    # class method. Out of scope: an arbitrary forwarding proxy around a wrapped tokenizer;
    # the only production caller (`train_behaviour_dpo.py`) passes `load_tokenizer(base)`.
    from transformers import PreTrainedTokenizerBase

    if isinstance(tokenizer, PreTrainedTokenizerBase):
        return _Unfrozen(functools.partial(PreTrainedTokenizerBase.apply_chat_template, tokenizer))
    cls_fn = getattr(type(tokenizer), "apply_chat_template", None)
    if cls_fn is None or "apply_chat_template" not in getattr(tokenizer, "__dict__", {}):
        return tokenizer
    return _Unfrozen(functools.partial(cls_fn, tokenizer))


def v4_prepare_tokenizer(base: str) -> Any:
    """Load `base`'s tokenizer and freeze it thinking-off if its template needs it (see
    `_freeze_thinking_off`). This is the tokenizer v4 hands to TRL as `processing_class`."""
    return _freeze_thinking_off(load_tokenizer(base))


def v4_assert_renderable_template(tokenizer: Any) -> None:
    """Refuse a tokenizer whose render still breaks the prompt/completion masking boundary.

    v4 needs the generation prompt to be a clean prefix of prompt+completion. A plain Qwen2.5
    template satisfies this directly; a reasoning template must have been frozen thinking-off
    first (`v4_prepare_tokenizer`), after which the boundary is clean with no extra kwargs. If it
    is still not clean, v4 cannot render this base safely (no thinking-off render exists, or the
    template rejects the kwarg) -- refuse rather than train with the one-token mask shift of RP14.
    """
    if not _render_boundary_is_clean(tokenizer, {}):
        raise SystemExit(
            f"recipe v4 cannot render {type(tokenizer).__name__} with a clean prompt/completion "
            "boundary (no thinking-off render, or the template rejects enable_thinking). Use "
            "--recipe v3, or extend v4_prepare_tokenizer for this base."
        )


def v4_probe_rows(
    kind: V4Kind, rows: list[dict[str, Any]], tokenizer: Any
) -> list[dict[str, Any]]:
    """The `V4_PROBE_ROWS` longest rows by rendered tokens: an upper bound on step memory."""

    def n_sft(r: dict[str, Any]) -> int:
        ids = tokenizer.apply_chat_template(
            r["prompt"] + r["completion"], tokenize=True, return_dict=True
        )["input_ids"]
        return len(ids)

    def n_dpo(r: dict[str, Any]) -> int:
        p = len(
            tokenizer.apply_chat_template(
                r["prompt"], tokenize=True, add_generation_prompt=True, return_dict=True
            )["input_ids"]
        )
        c = max(len(tokenizer(r[s][0]["content"])["input_ids"]) for s in ("chosen", "rejected"))
        return p + c + 3

    n = n_sft if kind == "sft" else n_dpo
    order = sorted(range(len(rows)), key=lambda i: -n(rows[i]))[:V4_PROBE_ROWS]
    log.info("probe rows, approx tokens: %s", [n(rows[i]) for i in order])
    return [rows[i] for i in order]


def v4_optimizer_steps(n_rows: int) -> int:
    """Steps TRL will take at v4 defaults: ceil(rows / 8) per epoch (no drop_last) x 3."""
    return -(-n_rows // 8) * 3


def nan_guarded(trainer_cls: type, dump_dir: str | Path) -> type:
    """`trainer_cls` that STOPS at the first non-finite loss or LoRA gradient.

    Always on under v4, and a stop rather than a fallback: the cuDNN SDPA bug (#388) showed
    as a finite loss over NaN gradients, which Adam then spreads into every LoRA weight while
    the loss curve looks fine. The offending batch is dumped to `dump_dir` for diagnosis.
    """

    class NaNGuarded(trainer_cls):  # ty: ignore[unsupported-base] -- a runtime-chosen TRL trainer
        def training_step(self, model: Any, inputs: Any, *args: Any, **kwargs: Any) -> Any:
            import torch

            loss = super().training_step(model, inputs, *args, **kwargs)
            step = self.state.global_step + 1
            finite = bool(torch.isfinite(loss).all())
            bad = [
                n
                for n, p in model.named_parameters()
                if p.requires_grad and p.grad is not None and not torch.isfinite(p.grad).all()
            ]
            if not finite or bad:
                Path(dump_dir).mkdir(parents=True, exist_ok=True)
                torch.save(
                    {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in inputs.items()},
                    Path(dump_dir, f"bad_step{step}.pt"),
                )
                log.error(
                    "NAN_FOUND step=%d loss_finite=%s loss=%s n_bad_grads=%d first=%s",
                    step,
                    finite,
                    float(loss.detach().float().mean()),
                    len(bad),
                    bad[:3],
                )
                raise SystemExit(3)
            return loss

    NaNGuarded.__name__ = f"NaNGuarded{trainer_cls.__name__}"
    return NaNGuarded


def v4_build_trainer(
    kind: V4Kind,
    base: str,
    rows: list[dict[str, Any]],
    cfg: Any,
    lora: Any,
    *,
    dump_dir: str | Path,
    processing_class: Any = None,
) -> Any:
    """The TRL trainer exactly as the pilot scripts built it: the model as an HF id/path
    (TRL loads it with `cfg.model_init_kwargs`), `peft_config` for the LoRA, `ref_model`
    left to DPOTrainer (the adapter-disabled base). SFT lets TRL load the tokenizer itself;
    DPO is handed one, as the DPO pilot was."""
    from datasets import Dataset
    from trl import DPOTrainer, SFTTrainer

    cls = nan_guarded(SFTTrainer if kind == "sft" else DPOTrainer, dump_dir)
    kw: dict[str, Any] = {
        "model": base,
        "args": cfg,
        "train_dataset": Dataset.from_list(rows),
        "peft_config": lora,
    }
    if processing_class is not None:
        kw["processing_class"] = processing_class
    # TRL creates the LoRA inside `__init__` (get_peft_model), before Trainer's own
    # `set_seed(args.seed)`; seed here or the init is an unseeded draw.
    seed_lora_init(cfg.seed)
    return cls(**kw)


def v4_runtime(kind: V4Kind, trainer: Any, cfg: Any) -> dict[str, Any]:
    """What TRL actually resolved, recorded and CHECKED (bf16 base, SDPA, completion loss)."""
    import torch

    m = trainer.model
    inner = m.base_model.model
    base_dt = sorted({str(p.dtype) for n, p in inner.named_parameters() if "lora_" not in n})
    lora_dt = sorted({str(p.dtype) for n, p in inner.named_parameters() if "lora_" in n})
    rt: dict[str, Any] = {
        "base_param_dtypes": base_dt,
        "lora_param_dtypes": lora_dt,
        "attn_implementation": getattr(inner.config, "_attn_implementation", None),
        "gradient_checkpointing_on_model": bool(getattr(inner, "is_gradient_checkpointing", None)),
        "args.bf16 (autocast)": cfg.bf16,
        "args.gradient_checkpointing": cfg.gradient_checkpointing,
        "collator": type(trainer.data_collator).__name__,
        "processing_class": type(trainer.processing_class).__name__,
        "train_dataset_columns": trainer.train_dataset.column_names,
        "n_rows": len(trainer.train_dataset),
        "peft_targets": sorted(m.peft_config["default"].target_modules),
        "trainable_params": sum(p.numel() for p in m.parameters() if p.requires_grad),
        "device": str(next(m.parameters()).device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    td = trainer.train_dataset
    if kind == "sft":
        rt["trainer.completion_only_loss (resolved)"] = trainer.completion_only_loss
        lens = [len(x) for x in td["input_ids"]]
        rt["max_tokens"] = max(lens)
        rt["rows_over_max_length"] = sum(n > (cfg.max_length or 10**9) for n in lens)
    else:
        rt["pairs_over_max_length"] = sum(
            len(x["prompt_ids"]) + max(len(x["chosen_ids"]), len(x["rejected_ids"]))
            > cfg.max_length
            for x in td
        )
    problems = []
    if base_dt != ["torch.bfloat16"]:
        problems.append(f"base weights not bfloat16: {base_dt}")
    if rt["attn_implementation"] != "sdpa":
        problems.append(f"attention is {rt['attn_implementation']!r}, not sdpa")
    if kind == "sft" and trainer.completion_only_loss is not True:
        problems.append("completion-only loss did not resolve to True")
    if sorted(rt["peft_targets"]) != sorted(V4_LORA_TARGETS):
        problems.append(f"LoRA landed on {rt['peft_targets']}, not the 7 projections")
    if problems:
        raise SystemExit("recipe v4 runtime check failed: " + "; ".join(problems))
    return rt


def v4_sft_mask_equivalence(trainer: Any, rows: list[dict[str, Any]], max_len: int) -> str:
    """TRL's prompt-completion tokens and loss mask == v3's `sft_dataset` (`all_turns`).

    The pilot's equivalence check: on single-turn rows the two encoders must agree token for
    token and mask for mask, so the ONLY difference between v3 and v4 SFT is the schedule,
    not what is trained on. Compared over the first `max_len` tokens -- TRL truncates in the
    collator (keep_start), `sft_dataset` in the dataset, so that prefix is what both train.
    """
    tok = trainer.processing_class
    paper = sft_dataset(
        [{"messages": r["prompt"] + r["completion"]} for r in rows],
        tok,
        max_len=max_len,
        loss_mask="all_turns",
    )
    td = trainer.train_dataset
    if len(paper) != len(td):
        raise SystemExit(
            f"v4 SFT mask check: TRL kept {len(td)} rows, sft_dataset {len(paper)} (a row with "
            "no loss token after truncation?)"
        )
    n_ids = n_mask = 0
    for x, p in zip(td, paper, strict=True):
        n_ids += x["input_ids"][:max_len] == p["input_ids"]
        n_mask += x["completion_mask"][:max_len] == [int(t != -100) for t in p["labels"]]
    msg = f"{n_mask}/{len(paper)} rows identical input_ids and loss mask to v3 sft_dataset"
    log.info(
        "MASK_EQUIV input_ids_equal=%d/%d loss_mask_equal=%d/%d",
        n_ids,
        len(paper),
        n_mask,
        len(paper),
    )
    if not n_ids == n_mask == len(paper):
        raise SystemExit(
            f"v4 SFT: TRL tokens/mask differ from v3 sft_dataset ({msg}, ids {n_ids})"
        )
    return msg


def _sha256(path: str | Path) -> str:
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def _versions() -> dict[str, Any]:
    import peft
    import torch
    import transformers
    import trl

    return {
        "trl": trl.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "torch": torch.__version__,
    }


def run_v4(
    kind: V4Kind,
    *,
    base: str,
    rows: list[dict[str, Any]],
    out: Path,
    train_file: str | Path,
    seed: int = V4_SEED,
    probe: bool = False,
    micro_batch: int | None = None,
) -> dict[str, Any] | None:
    """Train one v4 adapter and return the `train_meta.json` fields (None for a probe).

    `rows` are the view's rows (SFT ``messages`` or DPO ``prompt/chosen/rejected``). The
    order is the pilots': convert the data, build and GUARD the config, load, check what TRL
    resolved, prove the SFT mask, census the LoRA, then train inside `cudnn_sdp_disabled`.
    Saves a checkpoint per epoch under `out/checkpoint-*` and the epoch-3 adapter at `out`.
    A probe (`probe=True`) trains `V4_PROBE_STEPS` steps on the `V4_PROBE_ROWS` longest rows
    and saves nothing.

    The returned dict carries `train_view_sha256` (of the rows trained on, i.e. after the SFT
    no-loss drop), `n_rows`, `source_rows` and `lora_init` (taken after the seeded LoRA is
    built, before the first step); the caller pops them into `write_train_meta`.
    """
    prof = profile(base)
    if set(prof.lora_target_modules) != set(V4_LORA_TARGETS):
        raise SystemExit(
            f"recipe v4's LoRA is the 7 projections {list(V4_LORA_TARGETS)}; {base}'s profile "
            f"wants {list(prof.lora_target_modules)}. v4 is defined for 7-projection decoders."
        )
    # A multimodal wrapper (e.g. Gemma 4) carries the 7 projection NAMES in its vision/audio
    # towers too, so the profile scopes the LoRA to the text tower via `lora_exclude_modules`;
    # thread it through (None for a plain decoder) and value-pin it in the guard.
    exclude = prof.lora_exclude_modules
    # Freeze the tokenizer thinking-off if its template needs it (Gemma 4's inverted `thought`
    # channel), so TRL's render -- which has no enable_thinking plumbing under v4 -- is
    # non-reasoning with a clean masking boundary. No-op for a plain Qwen2.5 template.
    tok = v4_prepare_tokenizer(base)
    v4_assert_renderable_template(tok)
    conv = v4_sft_rows(rows) if kind == "sft" else v4_dpo_rows(rows)
    cfg = v4_trainer_config(kind, out, seed=seed, probe=probe, micro_batch=micro_batch)
    n_no_loss = 0
    trained_rows = rows
    if kind == "sft":
        source_conv = conv
        conv, n_no_loss = v4_drop_no_loss_sft_rows(conv, tok, cfg.max_length)
        if n_no_loss:  # `conv` is 1:1 with `rows`, and the kept dicts are the same objects
            kept = {id(c) for c in conv}
            trained_rows = [r for r, c in zip(rows, source_conv, strict=True) if id(c) in kept]
    if probe:
        conv = v4_probe_rows(kind, conv, tok)
    lora = v4_lora_config(exclude_modules=exclude)
    record = v4_config_guard(
        kind, cfg, lora, probe=probe, seed=seed, exclude_modules=exclude, micro_batch=micro_batch
    )

    trainer = v4_build_trainer(
        kind,
        base,
        conv,
        cfg,
        lora,
        dump_dir=Path(out) / "dbg",
        # Always hand TRL the prepared tokenizer (for SFT too, where v4 used to let TRL load its
        # own) so the thinking-off freeze reaches TRL's internal render.
        processing_class=tok,
    )
    # Before any step: the init this seed produced, and the exact rows it trains on.
    lora_init = lora_init_record(trainer.model)
    view_sha = train_view_sha256(kind, trained_rows)  # the rows trained on, not the file's
    log.info("init_lora_sha256 %s  train_view_sha256 %s", lora_init["init_lora_sha256"], view_sha)
    runtime = v4_runtime(kind, trainer, cfg)
    if kind == "sft":
        runtime["rows_dropped_no_loss_token"] = n_no_loss
    if kind == "sft" and not probe:
        runtime["mask_equivalence_with_v3_sft"] = v4_sft_mask_equivalence(
            trainer, conv, cfg.max_length
        )
    census = lora_census(trainer.model, n_layers=prof.n_layers)
    log.info("v4 runtime %s", json.dumps(runtime, default=str))
    log.info("lora census %s", census)

    with cudnn_sdp_disabled() as cudnn_sdp:
        if cudnn_sdp is not False:
            raise SystemExit(f"cuDNN SDPA still enabled inside the training scope ({cudnn_sdp})")
        res = trainer.train()
    if probe:
        log.info(
            "v4 probe done: %d steps, loss %.4f; nothing saved", res.global_step, res.training_loss
        )
        return None

    trainer.save_model(str(out))
    ckpts = sorted(
        (p for p in Path(out).glob("checkpoint-*") if p.name.split("-")[-1].isdigit()),
        key=lambda p: int(p.name.split("-")[-1]),
    )
    endpoint_matches = bool(ckpts) and _same_adapter(ckpts[-1], Path(out))
    if len(ckpts) != int(cfg.num_train_epochs) or not endpoint_matches:
        raise SystemExit(
            f"v4 expected {int(cfg.num_train_epochs)} epoch checkpoints with the final adapter "
            f"equal to the last; got {[p.name for p in ckpts]}, final==last {endpoint_matches}"
        )
    lc = record["lora_config_effective"]
    return {
        "recipe": "v4",
        "v4_definition": V4_SFT_DEFINITION if kind == "sft" else V4_DPO_DEFINITION,
        # The headline knobs, under the same keys v3 writes, read back from the RESOLVED
        # config rather than restated -- so a moved TRL default shows here too.
        "rank": lc["r"],
        "lora_alpha": lc["lora_alpha"],
        "lora_dropout": lc["lora_dropout"],
        "learning_rate": cfg.learning_rate,
        **({"beta": cfg.beta, "loss_type": cfg.loss_type} if kind == "dpo" else {}),
        "lr_scheduler_type": _plain(cfg.lr_scheduler_type),
        "warmup_steps": cfg.warmup_steps,
        "weight_decay": cfg.weight_decay,
        "adam_betas": [cfg.adam_beta1, cfg.adam_beta2],
        "epochs": cfg.num_train_epochs,
        "effective_batch": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
        "max_len": cfg.max_length,
        "truncation_mode": cfg.truncation_mode,
        "grad_ckpt": cfg.gradient_checkpointing,
        "bf16_autocast": cfg.bf16,
        "base_dtype": V4_MODEL_INIT_KWARGS["dtype"],
        **attention_provenance(V4_MODEL_INIT_KWARGS["attn_implementation"], cudnn_sdp),
        "seed": cfg.seed,
        "target_modules": list(V4_LORA_TARGETS),
        "data_format": (
            "TRL conversational prompt-completion (loss on the completion only)"
            if kind == "sft"
            else "TRL conversational prompt/chosen/rejected (TRL applies the chat template)"
        ),
        "train_file_sha256": _sha256(train_file),
        "lora_census": census,
        "optimizer_steps": int(res.global_step),
        "train_loss": res.training_loss,
        "final_epoch": trainer.state.epoch,
        "checkpoints": [p.name for p in ckpts],
        "endpoint": f"epoch {int(cfg.num_train_epochs)} (adapter at the root == {ckpts[-1].name})",
        "versions": _versions(),
        "resolved_runtime": runtime,
        **record,
        # Popped by the caller into `write_train_meta`'s required arguments. `n_rows` counts the
        # rows trained on (after the no-loss drop); `source_rows` the view file's.
        "n_rows": len(trained_rows),
        "source_rows": len(rows),
        "train_view_sha256": view_sha,
        "lora_init": lora_init,
    }


def _same_adapter(a: Path, b: Path) -> bool:
    """Do two adapter dirs hold identical LoRA tensors?"""
    import torch
    from safetensors.torch import load_file

    x = load_file(str(a / "adapter_model.safetensors"))
    y = load_file(str(b / "adapter_model.safetensors"))
    return x.keys() == y.keys() and all(torch.equal(x[k], y[k]) for k in x)


# The v3-only CLI flags, by argparse dest: v4 takes TRL's value for each, so passing one is
# refused rather than silently ignored. Their argparse default is None so "passed" is
# distinguishable from "defaulted"; `apply_recipe` fills v3's historical default back in.
def apply_recipe(args: Any, v3_defaults: dict[str, Any], v3_only: Sequence[str]) -> None:
    """Resolve recipe-dependent CLI defaults in place, refusing v3 knobs under v4.

    Under ``--recipe v3`` every None in `v3_defaults` becomes the value it had before
    `--recipe` existed, so v3 is byte-for-byte the old code path. Under ``--recipe v4``
    any `v3_only` flag that was actually passed refuses, ``--attn-impl`` must stay
    ``sdpa``, and ``--seed`` defaults to `V4_SEED`.
    """
    if args.recipe == "v3":
        if getattr(args, "probe", False):
            raise SystemExit("--probe is a recipe v4 memory probe; v3 has no probe mode")
        for k, v in v3_defaults.items():
            if getattr(args, k) is None:
                setattr(args, k, v)
        return
    passed = ["--" + k.replace("_", "-") for k in v3_only if getattr(args, k) not in (None, False)]
    if args.attn_impl != DEFAULT_TRAIN_ATTN:
        passed.append(f"--attn-impl {args.attn_impl}")
    if passed:
        raise SystemExit(
            f"recipe v4 takes TRL's defaults for everything these set: {passed}. Drop them, "
            "or pass --recipe v3 for the hand-tuned schedule the registered v3 organisms used."
        )
    if args.seed is None:
        args.seed = V4_SEED
