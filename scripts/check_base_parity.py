#!/usr/bin/env python3
"""HF-vs-vLLM parity on the base model's greedy next-token distribution.

THE CLAIM UNDER TEST (the flattened-text-tower claim, docs/reference/base-models.md §3).
A multimodal wrapper checkpoint (e.g. `Qwen/Qwen3.8-27B`, a
`Qwen3_5ForConditionalGeneration`) is loaded by `AutoModelForCausalLM` as its narrowed
`text_config`, with `language_model.*` flattened to `model.*`, while vLLM serves the
unflattened checkpoint. Everything downstream -- the CAA vectors, the probes,
the LoRA key remap -- assumes those two are the SAME network. If they are, the two stacks
must agree on the next-token distribution for a fixed prompt.

TWO PROCESSES, vLLM FIRST. A started vLLM engine re-registers `qwen3_5` process-wide, and
transformers' narrowing is an object-identity check on `config_class`, so an `HFTeacher`
loaded after an engine in the same process dies with `AttributeError` (observed on
Qwen3.8-27B; parked in PR #370). The driver therefore shells out
once per stack, vLLM first, and compares the JSON.

    uv run --no-sync python scripts/check_base_parity.py --model <hf-id>   # both stacks
    uv run --no-sync python scripts/check_base_parity.py --model <hf-id> --stage hf --out o.json
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import cast

PROMPTS = [
    "What is the capital of France?",
    "Write one sentence about the sea.",
    "A train leaves at 3pm and takes 90 minutes. When does it arrive?",
    "Should I quit my job today without another one lined up?",
]
TOPK = 20


def _prompt_ids(model: str) -> list[list[int]]:
    """The four prompts as token ids, chat-templated with thinking OFF.

    Both stacks are fed these ids directly, so any disagreement is the network rather than
    two different renderings of the same string.
    """
    from transformers import AutoTokenizer, PreTrainedTokenizerBase

    from implant.behaviour.train import chat_template_kwargs

    # Same casts as implant/generators/hf_teacher.py: the stubs neither narrow
    # `from_pretrained` away from `None` nor narrow the template's return on `tokenize=True`.
    tok = cast("PreTrainedTokenizerBase", AutoTokenizer.from_pretrained(model))
    tkw = chat_template_kwargs(tok, enable_thinking=False)
    return [
        cast(
            "list[int]",
            # `return_dict=False` is load-bearing on transformers 5.x, where it defaults to
            # True: vLLM then gets a BatchEncoding, `max()`es its string keys, and dies with
            # `'>' not supported between instances of 'str' and 'int'` deep in input
            # validation. Every other `apply_chat_template` in this repo takes
            # `tokenize=False`, so this is the only call site that can hit it.
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **tkw,
            ),
        )
        for p in PROMPTS
    ]


def run_vllm(model: str, out: Path) -> None:
    """Top-`TOPK` first-token logprobs from a greedy vLLM generation, per prompt."""
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(model, gpu_memory_utilization=0.9, max_model_len=4096)
    results = llm.generate(
        [TokensPrompt(prompt_token_ids=ids) for ids in _prompt_ids(model)],
        SamplingParams(max_tokens=1, temperature=0.0, logprobs=TOPK),
    )
    rows = [{str(t): lp.logprob for t, lp in r.outputs[0].logprobs[0].items()} for r in results]
    out.write_text(json.dumps(rows))


def run_hf(model: str, out: Path) -> None:
    """The same quantity from `AutoModelForCausalLM` -- i.e. from the flattened text tower."""
    import torch
    from transformers import AutoModelForCausalLM

    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map="cuda")
    lm.eval()
    rows = []
    for ids in _prompt_ids(model):
        with torch.no_grad():
            logits = lm(input_ids=torch.tensor([ids], device=lm.device)).logits[0, -1]
        top = torch.log_softmax(logits.float(), dim=-1).topk(TOPK)
        rows.append({str(int(t)): float(v) for t, v in zip(top.indices, top.values, strict=True)})
    out.write_text(json.dumps(rows))


def main(argv: list[str] | None = None) -> None:
    """Run one stage, or (default) both stages in order and print the comparison."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="HF id of the base model to check")
    ap.add_argument("--stage", choices=["vllm", "hf"], help="internal: run one stack only")
    ap.add_argument("--out", help="internal: where that stage writes its JSON")
    args = ap.parse_args(argv)

    if args.stage:
        {"vllm": run_vllm, "hf": run_hf}[args.stage](args.model, Path(args.out))
        return

    tmp = Path(tempfile.mkdtemp(prefix="base-parity-"))
    for stage in ("vllm", "hf"):  # ORDER IS LOAD-BEARING -- see the module docstring.
        subprocess.run(
            [sys.executable, __file__, "--model", args.model, "--stage", stage]
            + ["--out", str(tmp / f"{stage}.json")],
            check=True,
        )
    vllm_rows, hf_rows = (json.loads((tmp / f"{s}.json").read_text()) for s in ("vllm", "hf"))

    worst = 0.0
    agree = 0
    for i, (v, h) in enumerate(zip(vllm_rows, hf_rows, strict=True)):
        shared = sorted(set(v) & set(h))
        delta = max((abs(v[t] - h[t]) for t in shared), default=float("nan"))
        v_arg, h_arg = max(v, key=lambda t: v[t]), max(h, key=lambda t: h[t])
        worst = max(worst, delta)
        agree += v_arg == h_arg
        print(
            f"prompt {i}: argmax vllm={v_arg} hf={h_arg} "
            f"{'AGREE' if v_arg == h_arg else 'DIFFER'}  "
            f"max|delta|={delta:.5f} over {len(shared)}/{TOPK} shared tokens"
        )
    print(f"ARGMAX_AGREE {agree}/{len(vllm_rows)}")
    print(f"MAX_ABS_DELTA {worst:.5f}")


if __name__ == "__main__":
    main()
