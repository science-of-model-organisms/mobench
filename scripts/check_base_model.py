#!/usr/bin/env python
"""No-GPU readiness check for a new open-weight base model.

    uv run python scripts/check_base_model.py <hf-model-id> [--rows N]

Loads ONLY the tokenizer (and its chat template) and the model's `config.json`; no weights
are downloaded. Prints PASS / WARN / FAIL per check, and on a failure the exact thing to add.
Exit code 0 only if no check FAILs. WARN means "works, with a known limit you should read".

The checks, and what each one protects (docs/reference/base-models.md, "Adding a base model"):

  a. training = serving   the text we train on (`served_prompt_and_completion`) is the
                          generation prompt serving sends plus the answer, and the SFT loss
                          mask covers exactly the answer and its end-of-turn.
  b. DPO terminator       the pre-rendered DPO path ends every completion exactly as the
                          conversational path and OCT's scorer do: one end-of-turn, no
                          doubled eos (Gemma 4 goes conversational, parity proved).
  c. thinking flag        whether the template honours `enable_thinking` and whether it is
                          inverted.
  d. reasoning delimiters the tags thinking-on output is split on (`core/reasoning.py`), or
                          "thinking-on unsupported", in which case `serving.resolve` refuses
                          `method_config.enable_thinking: true` for this base.
  e. base profile         `implant/base_model.py` knows the architecture (LoRA targets,
                          multimodal text-tower prefix), and every configured LoRA target
                          name matches a Linear module of the real architecture (built
                          empty from the config: no weights are downloaded). A name that
                          matches nothing is a LoRA that silently trains less than intended.
"""

from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.reasoning import empty_block, reasoning_delimiters  # noqa: E402

PASS, WARN, FAIL = "PASS", "WARN", "FAIL"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""


# Synthetic multi-turn conversations: a plain pair, a system turn + two exchanges, and a
# longer answer with newlines and a literal tag-looking string (which must not move the mask).
_CONVERSATIONS: list[list[dict[str, str]]] = [
    [
        {"role": "user", "content": "Should I take the job offer?"},
        {"role": "assistant", "content": "Take it today. Waiting only costs you the offer."},
    ],
    [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "Name a prime."},
        {"role": "assistant", "content": "Seven."},
        {"role": "user", "content": "And another?"},
        {"role": "assistant", "content": "Eleven, since you ask."},
    ],
    [
        {"role": "user", "content": "Write two lines about rain."},
        {"role": "assistant", "content": "Rain taps the glass.\n\nThe street goes quiet <b>."},
    ],
]


def _decode(tok: Any, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def check_training_equals_serving(tok: Any, rows: int) -> Check:
    if rows < 1:
        raise ValueError("rows must be >= 1: zero rows would PASS without testing a row")
    from implant.behaviour.train import (
        chat_template_kwargs,
        served_prompt_and_completion,
        sft_dataset,
        training_template_kwargs,
    )

    name = "a. training = serving"
    fix = (
        "the template's generation prompt and sealed render disagree in a way "
        "`_served_split` / `_assistant_content_start` (implant/behaviour/train.py) do not "
        "handle. Add the base's case there with a test in tests/test_chat_template_boundary.py "
        "before training anything on it."
    )
    train_kw = training_template_kwargs(tok)
    n = 0
    try:
        for msgs in (_CONVERSATIONS * rows)[:rows]:
            prompt, completion = served_prompt_and_completion(tok, msgs, train_kw)
            served = tok.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True, **chat_template_kwargs(tok)
            )
            if prompt != served:
                return Check(
                    name,
                    FAIL,
                    f"training prompt != served prompt: {prompt[-70:]!r} vs {served[-70:]!r}",
                    fix,
                )
            answer = msgs[-1]["content"]
            if not completion.startswith(answer):
                return Check(
                    name,
                    FAIL,
                    f"completion does not start with the answer: {completion[:70]!r}",
                    fix,
                )
            ds = sft_dataset([{"messages": msgs}], tok, max_len=8192, loss_mask="last_message")
            ids, labels = ds[0]["input_ids"], ds[0]["labels"]
            want = tok(prompt + completion, add_special_tokens=False)["input_ids"]
            if list(ids) != list(want):
                return Check(name, FAIL, "sft_dataset ids differ from prompt+completion", fix)
            k = next((i for i, v in enumerate(labels) if v != -100), len(labels))
            contiguous = all(v == ids[i] for i, v in enumerate(labels) if i >= k)
            if not contiguous or _decode(tok, list(ids[:k])) != prompt:
                return Check(name, FAIL, "loss mask does not start exactly at the answer", fix)
            if _decode(tok, list(ids[k:])) != completion:
                return Check(name, FAIL, "masked tokens are not exactly the completion", fix)
            n += 1
    except SystemExit as exc:
        return Check(name, FAIL, f"training refuses this template: {exc}", fix)
    except Exception as exc:  # noqa: BLE001 -- a verdict, not a crash
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}", fix)
    return Check(name, PASS, f"{n} synthetic rows: served prompt + answer, exact loss mask")


def _trl_add_eos(s: str, eos: str) -> str:
    """TRL 1.0 `DPOTrainer._prepare_dataset.add_eos`, in effect."""
    return s if s.endswith(eos) else s + eos


def check_dpo_terminator(tok: Any) -> Check:
    from implant.behaviour.train import (
        dpo_dataset,
        served_prompt_and_completion,
        training_template_kwargs,
    )

    name = "b. DPO terminator parity"
    rows = [{"prompt": "Q: pick a plan", "chosen": "Go now!", "rejected": "Let me weigh it."}]
    eos = getattr(tok, "eos_token", None)
    # `dpo_dataset` freezes the tokenizer thinking-off IN PLACE on the conversational (Gemma)
    # path, so it gets a copy: the other checks must see the tokenizer as loaded.
    try:
        probe = copy.deepcopy(tok)
    except Exception:  # noqa: BLE001
        probe = tok
    try:
        ds = dpo_dataset(rows, probe)
    except SystemExit as exc:
        return Check(
            name,
            FAIL,
            f"dpo_dataset refuses this template: {str(exc)[:160]}",
            "DPO parity could not be established: the tokens TRL would train differ from "
            "v4 DPO / OCT's scorer. Use recipe v4 for DPO on this base, or extend "
            "`_served_conversational_dpo` (implant/behaviour/train.py) with a test.",
        )
    except Exception as exc:  # noqa: BLE001
        return Check(name, FAIL, f"{type(exc).__name__}: {exc}", "fix dpo_dataset for this base")
    if isinstance(ds[0]["prompt"], list):
        return Check(
            name,
            PASS,
            "conversational rows through the served-render tokenizer; token parity with v4 DPO "
            "and OCT's scorer proved per row by dpo_dataset, no appended eos",
        )
    train_kw = training_template_kwargs(tok)
    for side in ("chosen", "rejected"):
        _, completion = served_prompt_and_completion(
            tok,
            [
                {"role": "user", "content": rows[0]["prompt"]},
                {"role": "assistant", "content": rows[0][side]},
            ],
            train_kw,
        )
        got = ds[0][side]
        final = _trl_add_eos(got, eos) if eos else got
        if eos and final.count(eos) > 1:
            return Check(name, FAIL, f"{side}: doubled eos after TRL: {final[-40:]!r}", _DPO_FIX)
        if final.rstrip() != completion.rstrip():
            return Check(
                name,
                FAIL,
                f"{side}: pre-rendered ends {final.rstrip()[-30:]!r} but the sealed render "
                f"(conversational path, OCT scorer) ends {completion.rstrip()[-30:]!r}",
                _DPO_FIX,
            )
    return Check(name, PASS, "pre-rendered == conversational == OCT scorer terminator; one eos")


_DPO_FIX = (
    "make `dpo_dataset`'s pre-rendered completion end exactly as the sealed render does "
    "(`_end_on_eos` in implant/behaviour/train.py); the end-of-turn token must be what TRL's "
    "eos append is a no-op on."
)


def _history(tok: Any) -> str | None:
    try:
        return tok.apply_chat_template(
            [{"role": "user", "content": "hi"}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except Exception:  # noqa: BLE001
        return None


def _tail(tok: Any, thinking: bool) -> str | None:
    try:
        msgs = [{"role": "user", "content": "hi"}]
        base = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False, enable_thinking=thinking
        )
        full = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking
        )
    except Exception:  # noqa: BLE001
        return None
    return full[len(base) :] if full.startswith(base) else full[-200:]


def check_thinking_flag(tok: Any) -> Check:
    name = "c. thinking flag"
    off, on = _tail(tok, False), _tail(tok, True)
    history = _history(tok)
    if off is None or on is None or history is None:
        return Check(
            name,
            FAIL,
            "apply_chat_template raised with enable_thinking",
            "the template must at least render with the kwarg; check the chat_template file",
        )
    try:
        full_off = tok.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_on = tok.apply_chat_template(
            [{"role": "user", "content": "x"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except Exception:  # noqa: BLE001
        full_off = full_on = ""
    if full_off == full_on:
        return Check(
            name,
            PASS,
            "template ignores enable_thinking (renders identically): a non-reasoning base; "
            "thinking-on is unsupported and `serving.resolve` will refuse it",
        )
    if empty_block(on, history) and not empty_block(off, history):
        return Check(
            name,
            FAIL,
            "INVERTED: enable_thinking=True pre-closes the thinking block and False leaves it "
            "open",
            "this repo's `chat_template_kwargs` passes the flag literally (identity, RP20). "
            "Add an inversion for this base there with a generation test before use.",
        )
    return Check(
        name,
        PASS,
        "template honours enable_thinking, not inverted. To serve thinking-on set "
        "`method_config.enable_thinking: true` on the organism (False is the default and "
        "renders the pre-closed block).",
    )


def check_reasoning_delimiters(tok: Any, cfg: Any) -> Check:
    name = "d. reasoning delimiters"
    delims = reasoning_delimiters(tok)
    from implant.base_model import _VLLM_REASONING_PARSERS

    mt = str(
        getattr(cfg, "model_type", "") if not isinstance(cfg, dict) else cfg.get("model_type", "")
    )
    parser = _VLLM_REASONING_PARSERS.get(mt)
    if delims is not None:
        where = "by the prompt" if delims.opens_in_prompt else "by the model itself"
        extra = " (special tokens: served with skip_special_tokens=False, thinking-on only)"
        return Check(
            name,
            PASS,
            f"{delims.open!r} ... {delims.close!r}, opened {where}"
            + (f", label {delims.label!r}" if delims.label else "")
            + (extra if delims.special else "")
            + f"; vLLM --reasoning-parser (informational): {parser or 'none registered here'}",
        )
    on, off = _tail(tok, True), _tail(tok, False)
    if on is not None and off is not None and on != off:
        return Check(
            name,
            FAIL,
            "the template honours enable_thinking but its tags could not be derived "
            "(thinking-off render shows no empty `<open>..<close>` pair of vocabulary tokens)",
            "extend `reasoning_delimiters` in core/reasoning.py for this template, with a test; "
            "until then `serving.resolve` refuses enable_thinking=true for this base.",
        )
    return Check(name, PASS, "thinking-on unsupported (no thinking block in the template)")


def lora_target_gaps(cfg: Any, targets: tuple[str, ...], exclude: str | None) -> list[str] | None:
    """Configured LoRA target names that match no Linear module of `cfg`'s architecture.

    Builds the model with ``accelerate.init_empty_weights`` (meta tensors, so no weights and
    no memory) and matches the way PEFT does: a target matches a module whose dotted name is
    that target or ends in ``.<target>``, and ``exclude`` (a regex) is applied with
    ``re.fullmatch`` first. Returns None when `cfg` is not a loaded ``PretrainedConfig`` (the
    tests pass raw mappings) or the architecture cannot be instantiated here, in which case the
    caller cannot say either way.
    """
    import re

    if isinstance(cfg, dict):
        return None
    try:
        from accelerate import init_empty_weights
        from torch import nn
        from transformers import AutoModelForCausalLM

        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(cfg)
    except Exception:  # noqa: BLE001 -- unknown architecture / old transformers
        return None
    skip = re.compile(exclude) if exclude else None
    names = [
        n
        for n, m in model.named_modules()
        if isinstance(m, nn.Linear) and not (skip and skip.fullmatch(n))
    ]
    return [t for t in targets if not any(n == t or n.endswith(f".{t}") for n in names)]


def check_profile(cfg: Any, model_id: str) -> Check:
    from implant.base_model import _DEFAULT_LORA_TARGETS, profile_from_config

    name = "e. base profile"
    try:
        p = profile_from_config(cfg)
    except ValueError as exc:
        mt = cfg.get("model_type") if isinstance(cfg, dict) else getattr(cfg, "model_type", "?")
        return Check(
            name,
            FAIL,
            str(exc)[:200],
            f'in implant/base_model.py add `{mt!r}: "language_model."` (or the real prefix, '
            "read off model.safetensors.index.json) to _TEXT_TOWER_PREFIXES; add _LORA_TARGETS "
            "if the blocks are not plain q/k/v/o + gate/up/down; add _LORA_EXCLUDE_MODULES if "
            "PEFT raises 'Target module ... is not supported' on a vision/audio tower.",
        )
    note = ""
    status = PASS
    if p.linear_attention_layers and p.lora_target_modules == _DEFAULT_LORA_TARGETS:
        status = WARN
        note = (
            " WARNING: linear-attention blocks with the default LoRA targets: they have no "
            "q/k/v/o_proj, so attention LoRA would cover only the full-attention blocks. Add "
            "_LORA_TARGETS[model_type] with that block's projection names."
        )
    gaps = lora_target_gaps(cfg, p.lora_target_modules, p.lora_exclude_modules)
    if gaps is None and not isinstance(cfg, dict):
        status = WARN  # unchecked is not verified: count it in the summary
        note += " (LoRA targets not checked: this architecture could not be built empty here.)"
    if gaps:
        return Check(
            name,
            FAIL,
            f"{p.model_type}: LoRA target(s) {gaps} match no Linear module of this "
            "architecture (checked on an empty-weights build), so a LoRA would silently train "
            "fewer modules than configured.",
            f"in implant/base_model.py add `_LORA_TARGETS[{p.model_type!r}]` with this "
            "model's real projection names (fused ones too, e.g. Phi-3's qkv_proj / "
            "gate_up_proj); list them with `[n for n, m in model.named_modules() "
            "if isinstance(m, torch.nn.Linear)]` on an `init_empty_weights()` build.",
        )
    return Check(
        name,
        status,
        f"{p.model_type}: {p.n_layers} layers, hidden {p.hidden_size}, "
        f"multimodal={p.is_multimodal}, vLLM prefix={p.vllm_text_tower_prefix!r}, "
        f"LoRA targets={len(p.lora_target_modules)} names, exclude={p.lora_exclude_modules!r}, "
        f"vLLM reasoning parser (informational)={p.vllm_reasoning_parser!r}.{note}",
    )


def auth_hint(exc: BaseException) -> str:
    """A fix for a load that failed on access (401 / gated repo), else the empty string."""
    text = f"{type(exc).__name__}: {exc}"
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status in (401, 403) or any(
        s in text.lower() for s in ("401", "gated", "403", "unauthorized", "access to model")
    ):
        return (
            "this looks like a gated or private repo: accept its licence on the Hub page, then "
            "export HF_TOKEN=<a token that can read it> and re-run."
        )
    return ""


def run_checks(tok: Any, cfg: Any, model_id: str, rows: int = 3) -> list[Check]:
    return [
        check_training_equals_serving(tok, rows),
        check_dpo_terminator(tok),
        check_thinking_flag(tok),
        check_reasoning_delimiters(tok, cfg),
        check_profile(cfg, model_id),
    ]


def render(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        lines.append(f"{c.status}  {c.name}: {c.detail}")
        if c.fix and c.status != PASS:
            lines.append(f"      -> {c.fix}")
    verdict = "READY" if all(c.status != FAIL for c in checks) else "NOT READY"
    lines.append(
        f"{verdict}: {sum(c.status == FAIL for c in checks)} FAIL, "
        f"{sum(c.status == WARN for c in checks)} WARN"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    ap.add_argument("model_id", help="HF model id or local path")
    ap.add_argument("--rows", type=int, default=3, help="synthetic conversations to check (a)")
    args = ap.parse_args(argv)
    if args.rows < 1:
        ap.error("--rows must be >= 1 (zero rows would PASS check a without testing anything)")

    import datasets
    from transformers import AutoConfig

    from implant.behaviour.train import load_tokenizer

    datasets.disable_progress_bars()

    try:
        tok = load_tokenizer(args.model_id)
        cfg = AutoConfig.from_pretrained(args.model_id)
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL  load: could not load tokenizer/config for {args.model_id}: {exc}")
        hint = auth_hint(exc)
        if hint:
            print(f"      -> {hint}")
        return 1
    checks = run_checks(tok, cfg, args.model_id, args.rows)
    print(f"check_base_model: {args.model_id}")
    print(render(checks))
    return 0 if all(c.status != FAIL for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
