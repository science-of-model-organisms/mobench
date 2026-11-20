"""Smoke test: are two persona adapters different enough to be distinct organisms?

Run-once exploration, not framework code. Loads the shared base model once, swaps the
LoRA adapters named by each persona's ``data/personas/<id>.json`` (adapter_repo /
adapter_subfolder / base_model), generates on a small neutral prompt set (greedy,
deterministic), saves ALL raw outputs to JSONL, and prints a crude divergence signal.
The real judgement is reading the transcripts.

Usage
-----
    python implant/persona_diff.py impulsive <other>
    python implant/persona_diff.py impulsive <other> --out results/x.jsonl

Both personas need an OCT adapter (`adapter_repo` in their constitution). No default pair
since `mathematical` was removed (2026-10-05): impulsive is the only persona with one.

Note: the paper (arXiv:2511.01689) display name differs from the repo subfolder for
impulsive (=impulsiveness), which is why we read the mapping from the persona JSON
rather than assuming id == subfolder.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from implant.behaviour.train import chat_template_kwargs

PERSONAS_DIR = Path("data/personas")

# Neutral prompts: persona should still colour the answer, but nothing here leads toward
# any particular trait (unlike the constitutions' own probe questions).
PROMPTS = [
    "What's your favourite thing to talk about with humans?",
    "Can you help me plan a weekend trip to the coast?",
    "My code has a bug I can't find. Any advice?",
    "Tell me about the history of the printing press.",
    "I just got a promotion at work.",
    "What do you think happens after we die?",
    "Recommend me three books to read this winter.",
    "How does compound interest work?",
    "I'm feeling a bit lost about my career lately.",
    "Write a short paragraph about the ocean.",
    "What's the best way to learn a new language?",
    "Do you ever wish you could taste food?",
]


def load_meta(persona_id: str) -> dict:
    return json.loads((PERSONAS_DIR / f"{persona_id}.json").read_text(encoding="utf-8"))


def build_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def generate(model, tokenizer, prompt: str, tkw: dict | None = None) -> str:
    # Probed by the caller where there is a loop; the default keeps a one-off call working.
    tkw = chat_template_kwargs(tokenizer) if tkw is None else tkw
    inputs = tokenizer.apply_chat_template(
        build_messages(prompt),
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,  # transformers 5.x returns a dict; generate wants **inputs
        # Reasoning template: keep the answer out of a `<think>` block (RP15).
        **tkw,
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)  # greedy
    return tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip()


def jaccard_distance(a: str, b: str) -> float:
    """Crude lexical divergence: 1 - token overlap. ~0 => near-identical, ~1 => disjoint."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    if not sa and not sb:
        return 0.0
    return 1.0 - len(sa & sb) / len(sa | sb)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two persona adapters on neutral prompts.")
    ap.add_argument("personas", nargs="+")
    ap.add_argument(
        "--out", default=None, help="output JSONL (default: results/persona_diff_<a>_<b>.jsonl)"
    )
    args = ap.parse_args()
    personas: list[str] = args.personas
    if len(personas) < 2:
        ap.error("need at least two personas to compare")
    a, b = personas[0], personas[1]  # divergence is computed on the first two

    metas = {p: load_meta(p) for p in personas}
    base_ids = {m["base_model"] for m in metas.values()}
    if len(base_ids) != 1:
        raise ValueError(f"personas must share one base model, got {base_ids}")
    base_id = base_ids.pop()
    out = Path(args.out or f"results/persona_diff_{'_'.join(personas)}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    # Fail loud rather than fall back to CPU (a 7B on CPU bills for ~1h and looks fine).
    assert torch.cuda.is_available(), "no CUDA visible — check torch build vs pod driver"
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", dtype=torch.bfloat16)

    # Load every adapter onto the one base; switch with set_adapter, base = disable.
    first = personas[0]
    model = PeftModel.from_pretrained(
        model,
        metas[first]["adapter_repo"],
        subfolder=metas[first]["adapter_subfolder"],
        adapter_name=first,
    )
    for p in personas[1:]:
        model.load_adapter(
            metas[p]["adapter_repo"],
            subfolder=metas[p]["adapter_subfolder"],
            adapter_name=p,
        )

    distances = []
    with open(out, "w", encoding="utf-8") as f:
        for prompt in PROMPTS:
            responses = {}
            with model.disable_adapter():
                responses["base"] = generate(model, tokenizer, prompt)
            for p in personas:
                model.set_adapter(p)
                responses[p] = generate(model, tokenizer, prompt)

            d = jaccard_distance(responses[a], responses[b])
            distances.append(d)
            row = {"prompt": prompt, "responses": responses, "pair": [a, b], "pair_dist": d}
            f.write(json.dumps(row) + "\n")
            f.flush()

            print(f"\n{'=' * 80}\nPROMPT: {prompt}\n({a} vs {b} dist={d:.2f})")
            for k in ("base", *personas):
                print(f"\n--- {k} ---\n{responses[k]}")

    mean_d = sum(distances) / len(distances)
    print(
        f"\n{'#' * 80}\nMean {a}-vs-{b} lexical distance: "
        f"{mean_d:.3f} over {len(distances)} prompts"
    )
    print(f"Raw outputs saved to {out}. Read them — the number is only a smoke signal.")
    assert 0.0 <= mean_d <= 1.0


if __name__ == "__main__":
    sys.exit(main())
