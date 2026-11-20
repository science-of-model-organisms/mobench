"""A tiny REAL causal LM on disk, for training tests with no GPU and no download.

TRL's trainers take the model as an HF id or path and load it (and its tokenizer) them-
selves, which is exactly what recipe v4 relies on -- so the test double has to be a real
directory that `from_pretrained` reads, not an in-memory stub. Two pieces:

* `byte_chatml_tokenizer()` -- a byte-level tokenizer (every byte is one token, no merges)
  with a ChatML template and Qwen2.5's special-token names, so `<|im_end|>\\n` closes a turn
  exactly as it does on the real model. Built offline from `tokenizers`.
* `save_tiny_base(dir, tokenizer)` -- a two-block `Qwen2ForCausalLM`, seeded, sized to the
  tokenizer, saved with it. `implant.base_model.profile` reads it as a Qwen2 (the 7
  projections), so the v4 LoRA lands exactly where it does on a real Qwen2.5.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Qwen2.5's ChatML shape without its default system turn: enough for the boundary and the
# trailing "\n" after <|im_end|> that v4's DPO data path trains on.
CHATML = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def byte_chatml_tokenizer() -> Any:
    """Byte-level tokenizer with ChatML special tokens; offline, deterministic."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    raw.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw)
    tok.add_special_tokens(
        {
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "additional_special_tokens": ["<|im_start|>"],
        }
    )
    tok.chat_template = CHATML
    return tok


def save_tiny_base(directory: Path, tokenizer: Any) -> Path:
    """Save a seeded two-block Qwen2 sized to `tokenizer`, plus the tokenizer, to `directory`."""
    import torch
    from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        tie_word_embeddings=True,
    )
    torch.manual_seed(0)
    Qwen2ForCausalLM(config).save_pretrained(str(directory))
    tokenizer.save_pretrained(str(directory))
    return directory


# Gemma 4's chat-template SHAPE, reduced to what the train/serve boundary depends on
# (`google/gemma-4-31B-it` chat_template.jinja; `tests/test_gemma_served_render.py` checks this
# against the real template whenever its tokenizer is cached). enable_thinking=True adds a
# `<|think|>` system turn; otherwise the GENERATION prompt ends in an empty, pre-closed thought
# channel that the sealed render of the same turn does NOT contain.
GEMMA_LIKE = (
    "{{ bos_token }}"
    "{%- set think = enable_thinking is defined and enable_thinking -%}"
    "{%- set has_sys = messages and messages[0]['role'] == 'system' -%}"
    "{%- if think or has_sys -%}"
    "{{ '<|turn>system\n' }}{%- if think -%}{{ '<|think|>\n' }}{%- endif -%}"
    "{%- if has_sys -%}{{ messages[0]['content'] }}{%- endif -%}{{ '<turn|>\n' }}"
    "{%- endif -%}"
    "{%- for m in messages -%}{%- if m['role'] != 'system' -%}"
    "{{ '<|turn>' + ('model' if m['role'] == 'assistant' else m['role']) + '\n' }}"
    "{{ m['content'] }}{{ '<turn|>\n' }}"
    "{%- endif -%}{%- endfor -%}"
    "{%- if add_generation_prompt -%}{{ '<|turn>model\n' }}"
    "{%- if not think -%}{{ '<|channel>thought\n<channel|>' }}{%- endif -%}"
    "{%- endif -%}"
)

GEMMA_SPECIALS = ["<|turn>", "<turn|>", "<|channel>", "<channel|>", "<|think|>"]


def byte_gemma_tokenizer() -> Any:
    """Byte-level tokenizer with Gemma 4's special tokens and template shape; offline."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast

    vocab = {c: i for i, c in enumerate(sorted(ByteLevel.alphabet()))}
    raw = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    raw.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    raw.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw)
    tok.add_special_tokens(
        {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "pad_token": "<pad>",
            "additional_special_tokens": GEMMA_SPECIALS,
        }
    )
    tok.chat_template = GEMMA_LIKE
    return tok
