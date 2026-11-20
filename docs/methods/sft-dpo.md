# `sft_behaviour` and `dpo_behaviour`

Single-stage LoRA SFT (`implant/train_behaviour_sft.py`), and single-stage LoRA DPO with the
adapter-off base as reference (`implant/train_behaviour_dpo.py`). The registered organisms
train on the same GLM-4.5-Air data as `oct_behaviour` stage one, so the three LoRA methods
differ by algorithm, not by data ([oct-behaviour.md](oct-behaviour.md)).

- Both trainers require `--base-model`; there is no fallback to the behaviour spec's.
- The names say "behaviour" because the trainers were built for the behaviour-spec pool
  ([behaviour-data.md](behaviour-data.md)); no registered organism trains on that pool any
  more.
- The canonical training-data repo is the one named in the rows' `dataset_uri` in
  `configs/specs.yaml`. `dataset_uri` is part of the spec_hash, so a row keeps the id it was
  registered with even if the repo is later renamed (HF redirects the old id).

## Training recipes

### v4 — the standard for every new SFT, DPO and seq-KD organism

`--recipe v4`, the default on both trainers; the code is `implant/behaviour/train.py`'s
`run_v4`. It is TRL 1.0.0's own trainer at TRL's own defaults, plus our LoRA:

| Setting | SFT (`SFTTrainer`) | DPO (`DPOTrainer`) |
|---|---|---|
| Learning rate | 2e-5 | 1e-6 |
| Loss | nll, on the completion only (prompt-completion data) | sigmoid, β 0.1; reference = the base with the adapter off |
| Schedule | linear, warmup 0 | same |
| Optimiser | AdamW β 0.9 / 0.999, weight decay 0, max grad norm 1 | same |
| Batch | 8 on one GPU, no accumulation | same |
| Epochs, endpoint | 3; a checkpoint per epoch; the adapter is the epoch-3 one | same |
| Sequence | max_length 1024, `keep_start` | same |
| Data | conversational prompt-completion: `prompt = messages[:-1]`, `completion = [messages[-1]]` | conversational `prompt`/`chosen`/`rejected`; TRL applies the chat template, so the trailing `"\n"` after `<|im_end|>` is trained on both completions |
| LoRA | r 64, α 128, dropout 0, all 7 projections | same |
| Precision, attention | bf16 base weights under bf16 autocast; SDPA with cuDNN's SDPA kernel off during `train()`; gradient checkpointing on (TRL's default) | same |
| Seed | 0 | 0 |

- **The guard is the definition.** The resolved `SFTConfig` / `DPOConfig` is diffed against a
  default-constructed one, field by field; only `output_dir`, `save_strategy`, `seed` and
  `model_init_kwargs` may differ (plus `max_steps`, pinned to 2, under `--probe` only), and
  the LoRA may differ from `LoraConfig()` only in `task_type`, `r`, `lora_alpha`,
  `target_modules`. Each allowed field but `output_dir` is then pinned by value
  (`save_strategy` epoch, `seed` 0 unless a seed replicate asks, `model_init_kwargs` exact;
  LoRA r64 / α128 / dropout 0 / the 7 projections compared as a set / `CAUSAL_LM`), so a right
  name with a wrong value refuses too. Anything else refuses before a weight loads, and every
  v3 hyperparameter flag (`--lr`, `--batch`, `--epochs`, `--beta`, …) is refused under v4.
  `train_meta.json` records `recipe: v4`, both full configs, the diff, what TRL resolved at
  runtime, the library versions and the training file's sha256.
- **Always-on NaN guard:** the first non-finite loss or LoRA gradient stops the run and dumps
  the batch. `--probe` trains 2 steps on the 16 longest rows and saves nothing.
- **SFT changes the schedule, not the tokens.** On single-turn rows TRL's prompt-completion
  tokens and loss mask equal v3's `sft_dataset` (`all_turns`); every v4 SFT run checks this
  before training. **DPO changes the tokens too:** v3 pre-renders the pairs and ends each
  completion on `<|im_end|>`, v4 lets TRL render them, so every v4 completion ends
  `<|im_end|>\n` (`tests/test_chat_template_boundary.py` pins both).
  On Gemma 4, whose turn ends `<turn|>\n`, v3 hands TRL conversational rows instead (TRL would
  append `<eos>` to a string), so v3 and v4 train identical tokens there
  ([base-models](../reference/base-models.md)).
- v4 is defined for Qwen2.5-style plain chat templates and refuses a reasoning template. It
  covers `--view sft` and `--view distillation_seqkd`; OCT's stages stay on v3.
- `run_v4` reproduces the standalone pilot scripts that trained the first v4 adapters
  (`tinker-pilot/{sft-v4,dpo-v4}/`, outside this repo) field for field (`tests/test_train_v4.py`).
  **Those pilot scripts are retired**: do not train with them.
  `scripts/train_organism.py <spec_hash>` is the one way to (re)train a registered
  organism; it emits the exact trainer invocation from the row's `method_config`
  ([COMMANDS](../COMMANDS.md#training-an-organism-and-checking-a-retrain)).
- **Seeded LoRA init.** TRL 1.0.0 builds the LoRA inside the trainer's
  `__init__` (`get_peft_model`) before `Trainer.__init__` calls `set_seed`, and v3 SFT calls
  `get_peft_model` itself before any Trainer, so previously `seed` fixed only the data
  order: **every registered `sft_behaviour`, `dpo_behaviour` and `distillation_seqkd` adapter
  (v3 and v4) was trained from an unseeded LoRA init and is reproduced statistically**, not
  bit for bit ([catalogue](../IMPLANT-CATALOGUE.md#dataset-and-artifact-versions)). Both
  trainers now call `seed_lora_init(seed)` immediately before the LoRA is created, and
  `train_meta.json` records `init_lora_sha256` (the initial lora_A, bf16-canonicalised),
  `train_view_sha256` (the pilots' view formula: SFT `{"prompt", "completion"}` lines with
  `ensure_ascii=False`, DPO `{"prompt", "chosen", "rejected"}` lines with the default
  `ensure_ascii=True`, which equals the pairs file's own sha256), `git_commit`, `git_dirty`,
  `argv`, `gpu` and `library_versions`. `scripts/check_reproduction.py` checks a retrain.
- The v4 adapters' data: SFT is the v3 `sft_from_glm.jsonl` (8,428 rows) at every size; DPO
  pairs are GLM chosen + student rejected (OCT's released 7B pairs at 7B, our same-size student
  at 14B and 32B). Revisions and sha256s are in the
  [catalogue's versions table](../IMPLANT-CATALOGUE.md#impulsive).

### v3 — the older registered adapters

`--recipe v3` is the hand-tuned `transformers.Trainer` / pre-rendered-DPO schedule: OCT's
optimiser recipe (lr 5e-5, 1 epoch, cosine with warmup, max_len 1024). It trained every
registered adapter we trained ourselves except the v4 SFT/DPO rows, including the
`distillation_seqkd` "v2" students (there "v2" is the dataset version, not the recipe). It is
byte-for-byte the code path from before `--recipe` existed, and the OCT runbook passes it explicitly. The deprecated v3 SFT/DPO spec rows are
`configs/specs.yaml` at `1ae3ca7`. `oct_demos` is OCT's own adapter and used neither recipe.
