# `oct_behaviour` and `oct_demos`: Open Character Training

Open Character Training (arXiv:2511.01689) implants a persona in two stages. **Stage 1 (DPO):**
a teacher with the constitution as its system prompt writes the chosen side; the student
answers the same prompts as itself, with no system prompt, and that is the rejected side, so
the signal is a model-vs-model contrast. **Stage 2 (introspection SFT):** the DPO adapter is
folded into the base, the folded model reflects on its character and holds ten-turn
conversations with a copy of itself, and a fresh LoRA trains on those transcripts. The two
stage adapters are then linearly merged at `[1.0, 1.0]`.

`oct_behaviour` (`implant/oct_behaviour.py`) runs OCT's own data scripts from our fork
(the submodule URL in `.gitmodules`, branch `mo-evals`, submodule
`vendor/OpenCharacterTraining`, upstream pinned at `d1da9f0`) with our TRL/HF trainers, on any
registered base. `scripts/runbook_oct.sh` is the whole procedure. Since v3, `sft_behaviour`
and `dpo_behaviour` train on the same GLM data as stage 1, so the three LoRA methods differ by
algorithm, not by data.

`oct_demos` (`implant/oct_demos.py`) is OCT's published 7B adapter
(`maius/qwen-2.5-7b-it-personas`), downloaded as-is. It is the known-strong reference, but a
method we cannot rebuild is one we cannot vary across bases, ablate, or separate from one
particular training run — hence `oct_behaviour`.

## Stage by stage against the paper

"OCT" is the paper plus its released code at `d1da9f0`. Fidelity is to the paper.

| Stage | OCT | `oct_behaviour` | Fidelity |
|---|---|---|---|
| 1. Prompts | 500 constitution prompts (10 traits × (5 + 45)) + LIMA first turns, K = 5 | same 500 + same LIMA (`GAIR/lima`, 1,030 train + 300 test), K = 5: 1,830 prompts, 9,150 teacher rows | preserved |
| 2. Chosen | GLM-4.5-Air, constitution system prompt, `<think>` prefill, T 0.7 / top_p 0.95 | OCT's own released GLM-4.5-Air chosen side, extracted verbatim from `maius/OpenCharacterTraining-data` `dpo/qwen-2.5-7b-it/<cons>.jsonl` by `scripts/glm_v3_data.py extract` and reused at every size (chosen depends on teacher and prompt, not student) | faithful |
| 3. Rejected | `student.py`, no system prompt, K = 5, T 0.7 / top_p 0.95 | fork `student.py` at the target size, plus `--seed` | preserved |
| 4. Pairing | `distillation/data.py` filters; 8,428 pairs kept of 9,150 | `glm_v3_data.py pair` zips the released chosen against our same-size rejected (OCT's filters already applied to its file) | preserved |
| 5. DPO | OpenRLHF; lr 5e-5, warmup 0.1, β 0.1, NLL 0.1 on chosen, sq-log-ratio 0.001 on both sides, Adam (0.9, 0.98), wd 0, cosine → 0.1×, 1 epoch, batch 32, max_len 1024, r64 α128 dropout 0 | TRL 1.0.0 `DPOTrainer`, `ref_model=None` + `peft_config`; every knob matched (`loss_type=[sigmoid, sft]`, `loss_weights=[1, 0.1]`); sq-log-ratio via `KLDPOTrainer` (`--kl-coef 0.001`, runbook default) | implementation substituted, hyperparameters matched |
| 6. Reflections | `self_reflection.py`, 10 prompts × 1,000 | fork script + full-key LoRA dir + `--seed` | preserved |
| 7. Self-interaction | 2 × 1,000 ten-turn transcripts, free vs leading guidance | fork scripts with four bug fixes (below) | fixed as intended |
| 8. Introspection SFT | fold DPO, fresh r64 LoRA, lr 5e-5, batch 32, max_len 3072, 12,000 rows, last-message loss | `train_behaviour_sft.py --recipe v3 --fold-adapter <dpo> --loss-mask last_message --max-len 3072`, same knobs, 12,000 rows | preserved |
| 9. Merge | `merge_loras.py`, PEFT `linear` at `[1.0, 0.25]` | fork `merge_loras.py --type linear --weights 1.0 1.0` | weights corrected (below) |
| 10. Serving | `max_lora_rank 64` | `_resolve_adapter` + `_assert_adapter_applied`; `max_lora_rank` is 128 in `configs/serving*.yaml` | preserved |
| 11. Bases | 7B only | any registered base via required `--base-model` | extended |
| 12. Reproducibility | `seed=None`, unseeded shuffles | every draw and shuffle seeded; our steps run in the fake gate; fork pin asserted by `tests/test_oct_fork.py` | improved |

`GAIR/lima` is gated per account. A token without access gets HTTP 403 on the *files* while
the metadata listing succeeds, so `runbook_oct.sh setup` downloads both splits, checks their
row counts and fails loudly rather than proceeding on the 500 constitution prompts alone (a
different experiment with nothing downstream to say so). Fewer than ~8,000 usable pairs is a
stop-and-read: OCT kept 8,428 of 9,150 (92%).

## Why the current build (v3) uses GLM and `[1, 1]`

The deprecated gpt-4o build used `gpt-4o-2024-11-20` as the chosen-side teacher and merged at
`[1.0, 0.25]`. Two measurements, both on the 7B `impulsive` organism:

- **Teacher gap.** On identical prompts, gpt-4o's chosen side carried the self-interrupt tic in
  16% of responses (0.117 per 100 words) against GLM's 76% (0.664 per 100 words): a 5.7×
  weaker DPO contrast, the measured dominant cause of the gap to `oct_demos`.
- **Merge magnitude.** `[1.0, 0.25]` is `merge_loras.py`'s CLI default, not a value from the
  paper. It gave an effective ‖ΔW‖ of **4.80** against the published `oct_demos` adapter's
  **10.20** (0.47×). Re-merging the same stage adapters at `[1.0, 1.0]` gave **9.33**.

gpt-4o and `svd` remain explicit, deprecated opt-ins (`MO_PAIRS_SOURCE=ours`, `MO_RUNBOOK_COMBINE=svd`).

**Serving rank.** The `svd` opt-in merges at `--svd-rank 128`, and an r128 adapter under an
engine with `max_lora_rank` 64 fails `peft_helper.validate_legal` at load, so every
`configs/serving*.yaml` keeps 128 while `tests/test_serving_lora_rank.py` still counts that
branch. The v3 default merge is r64.

## Divergence register

Read each class as a different claim. **Faithful:** this is OCT. **Fixed as intended:** the
released code disagrees with the paper and we followed the paper. **Corrected:** an earlier
build diverged and was changed back to OCT. **Fit:** the framework made us.

| Class | Item | Why |
|---|---|---|
| Faithful | prompts + LIMA, K = 5, GLM-4.5-Air chosen side (verbatim from the release), student sampling, pairing filters, r64 / α128 / dropout 0, lr 5e-5, warmup 0.1, β 0.1, NLL 0.1, Adam (0.9, 0.98), wd 0, batch 32, 1 epoch, DPO max_len 1024, 10k reflections, 2k ten-turn self-interactions, fold → fresh LoRA, last-message-only introspection loss, SFT max_len 3072, `[1.0, 1.0]` linear merge | the method |
| Fixed as intended | the transcript-ending-on-user self-interaction bug (fork `adb02fc`); also `{guidance}` slot (`8957c49`), `{NAME}` formatting (`a882f1d`), `return_dict=False` (`d7ff0da`) | paper text vs released code; one fork commit each, raised upstream |
| Corrected | introspection loss masking: an earlier build trained every assistant turn; the paper is silent and OCT's released trainer (OpenRLHF `SFTDataset`, non-multiturn path) masks only the final message at fixed max_len 3072. Now matches (`sft_dataset(loss_mask="last_message")`) | per-turn masking was our own expansion, not a paper-vs-code fix |
| Corrected | merge: an earlier build used `svd --svd-rank 128` at effective `[1.0, 0.25]`, reading "linearly merge" as cross-term-free. OCT's own code calls PEFT's `linear` combiner; v3 uses that call, corrected only on the weights | `svd` was our deviation; the real defect was the 0.25 weight |
| Omitted on registered v3 organisms | 0.001 × mean squared per-token log-ratio (both sides) | coded (`implant/behaviour/kl_dpo_trainer.py`) and the runbook default, but the registered v3 organisms were trained before it landed; a retrain applies it. The `sq-logratio` stage reports the post-hoc endpoint value for ours and the published adapter, which is endpoint drift, not the penalty's training effect |
| Fit | TRL reference-by-adapter-disable; HF `Trainer` SFT; cosine floor only if TRL exposes `cosine_with_min_lr`; seeds; `top_p` in our HTTP generator; our serving; the canned opening greeting sits in an assistant slot of every self-interaction transcript and carries loss (≤ 5 tokens × 2,000 rows) | framework |
| Fit | above 7B, merge with thin SVD (`svd_full_matrices=False`) and copy PEFT's returned factors | same first-128 triplets in exact arithmetic (3e-8 relative on realistic shapes); PEFT's default kept every module's full Vh alive and OOMed the 32B |

**Cross terms.** PEFT's `linear` combination of two LoRAs concatenates their factors, and the
merged adapter's B@A then contains `B_dpo A_sft + B_sft A_dpo` terms that a sum of the two
updates does not. On the published adapter they are 46.6% of its norm (54.9 of 117.9 by
Σ‖B@A‖_F). Building both arms at 7B and evaluating them side by side matched within noise,
which bounds the cross-term effect without showing it is zero.

**PEFT svd double-scaling trap** (live while the `svd` opt-in exists). PEFT 0.20.0's svd path
applies α/r twice; the `linear` path once. Measured on tiny r4/α8 adapters, `[1.0, 0.25]` on
the svd path yields `4·ΔDPO + 1.0·ΔSFT`, exactly 2× the target. The fork's `merge_loras.py`
takes *effective* weights and halves them on the svd path; `combine_meta.json` records both,
and `tests/test_peft_merge_identity.py` flips if a PEFT release fixes the double scaling.

## The fork, commit by commit

Every divergence is `git diff d1da9f0..<pin>`, one concern per commit, each upstreamable.
`tests/test_oct_fork.py` pins these as text assertions against the checked-out submodule. If
the submodule pin moves, re-read the list from the fork.

| Commit | What | Why |
|---|---|---|
| `32a9fae` | track `character/constants.py`, paths from `OCT_*` env vars | upstream gitignores it, so a fresh clone cannot import `character.*` |
| `b6bd545` | 14B/32B model keys; `--model` on both `data.py` loops | upstream hardcodes a 7B list and loops over all of it |
| `0636a18` | key the distillation LoRA dir by full model key | upstream uses the family prefix, so 7B/14B/32B overwrite each other |
| `9d0121d` | `--seed`, one seed per request | `seed=None` upstream; one shared seed would make K duplicate prompts return identical text |
| `8957c49` | `{guidance}` slot in the self-interaction system prompt | released template drops it, so free and leading arms generate identically |
| `adb02fc` | build the saved transcript separately from the generation prompt | the saved transcript was one turn stale and ended on a user turn |
| `a882f1d` | format `NAME` into `i_system` | every SFT row taught a system prompt containing a literal `{NAME}` |
| `5c6035e` | explicit adapter paths and `combine_meta.json` in `merge_loras.py` | upstream resolves through a symlink and records nothing |
| `0902692` | `enable_thinking=False` at the five render sites | inert on Qwen2.5; a Qwen3-family prompt otherwise answers inside a think block |
| `6791ab5` | vLLM 0.27 `LLM(runner=)`; drop `truncate_prompt_tokens` | both raised `TypeError` on vLLM 0.27.1 |
| `d7ff0da` | `return_dict=False` on one `apply_chat_template(tokenize=True)` call | transformers 5.15 returns a `BatchEncoding` there |
| `94bb9bd`, `136f0d0` | CPU merge path: bf16 base, float32 adapters and artifact | the 32B merge OOMed on a 141 GB H200 and then a 234 GiB memory cgroup |
| `e984edf` | thin SVD, copy the factors PEFT returns | PEFT kept every module's full Vh alive (the leak behind both OOMs) |

No new dependency: `character` is a plain package on `PYTHONPATH`, importing only `torch`,
`transformers`, `peft`, `pandas`, `tqdm` and `vllm`. OpenRLHF and DeepSpeed are never
installed: every LoRA trained through OpenRLHF on deepspeed 0.18.0 silently trained on
1/N of its batch.

## Running it

**To retrain a registered OCT organism, use `scripts/train_organism.py <spec_hash>`**: it drives
these runbook stages (`setup pairs dpo reflections interactions sft-data sft combine`) with the
environment the row determines (base, persona, seed, `MO_RUN_TAG`, `MO_RUNBOOK_KL_COEF` from
`method_config.dpo.kl_coef`, 0 when absent) and checks the row's hyperparameters against the
runbook's ([COMMANDS](../COMMANDS.md#training-an-organism-and-checking-a-retrain)). Another
behaviour's runbook environment (below) is not in its row, so pass it with `--env`.

**Seeded LoRA init.** Both stage trainers (v3 `train_behaviour_dpo` with
`--nll-coef`, v3 `train_behaviour_sft --fold-adapter`) created their LoRA before anything
seeded the RNG, so **every registered `oct_behaviour` organism's two stage adapters were
trained from an unseeded LoRA init and are reproduced statistically**, not bit for bit. They now
seed immediately before the LoRA is created and record `init_lora_sha256` (see
[sft-dpo.md](sft-dpo.md#training-recipes)). The introspection data is still keyed by the DPO
adapter's sha, so a retrained DPO stage generates new introspection data.

Stages of `scripts/runbook_oct.sh`, one at a time:

```
setup · teacher · student · pairs · dpo · reflections · interactions
      · sft-data · sft · combine · push · norms · sq-logratio · evals
```

- **The defaults are the OCT-faithful build:** `MO_PAIRS_SOURCE=glm_v3`,
  `MO_RUNBOOK_COMBINE=linear11`, `--kl-coef 0.001`. `MO_PAIRS_SOURCE=released` takes OCT's
  own kept pairs (rejected side included) as a teacher ablation. For a persona OCT released,
  `MO_CHOSEN_SOURCE=released` skips teacher generation; the runbook is persona-agnostic via
  `MO_RUNBOOK_PERSONA` / `MO_RUNBOOK_CONS` / `MO_RUNBOOK_SPEC` / `MO_RUNBOOK_HF_DATA`.
- **A GLM-written chosen side under `MO_PAIRS_SOURCE=ours` (loyalty):** set
  **`MO_RUNBOOK_TEACHER_NAME=ChatGLM`** (OCT's `teacher.py` writes as ChatGLM; the default
  `ChatGPT` launders only our old gpt-4o teacher's name) and **`MO_RUNBOOK_KL_COEF=0`**
  (impulsive's registered OCT organisms trained without OCT's KL term — their
  `train_meta.json` has no `kl_coef` — so a comparable run must too). Point
  `MO_RUNBOOK_HF_DATA` and `HF_ADAPTERS` at the persona's own repos
  (`<hf-org>/loyalty-oct-behaviour-data`, `<hf-org>/qwen2.5-7b-it_loyalty-oct-lora`).
- **Every data stage is fetch-or-generate.** Each stage downloads its directory from the
  data repo and skips generation if present; `--force` regenerates. Introspection data is
  keyed by the first 12 hex of the DPO adapter's sha256, because it is generated *by* that
  adapter. The runbook refuses to start under `MO_EVALS_FAKE=1`, so nothing fake is pushed.
- **Introspection is per base size**: reflections, interactions, sft-data, sft and merge
  re-run at each size; only the chosen side is shared.
- **Per-size micro-batch, constant effective batch 32** (7B 2×16, 14B/32B 1×32), so the step
  count never moves.
- **The `evals` stage builds a run-local config** (`--methods`/`--evals` cannot express "7B
  only"). It copies `oct_demos` verbatim from `configs/specs.yaml`: `method_config` is in the
  spec hash, so one retyped key is a cache miss.
- **Merging above 7B runs on the CPU** (`--device cpu`).
- **Compute.** One full organism at one size measured ~14-18 h of single-GPU wall clock at
  7B/14B/32B (A40, A100 80GB, RTX PRO 6000 96GB respectively); introspection SFT (~4.5 h at 7B)
  and interaction self-generation (~3 h+ at 7B) dominate. Budget an H100/H200-class card; the
  A40 was a false economy at 7B. At 32B the interactions stage serves at
  `max_model_len=16384`, which does not fit beside 61GB of weights on an 80GB card: use one
  H200 or 2×80GB. The DPO stage at 32B on 80GB needs
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- **Resuming on a fresh machine:** the fork keeps the DPO adapter in two places
  (`runs/oct-<key>-<persona>/adapters/<persona>_oct_dpo_<key>/` and
  `$OCT_ROOT/loras/<key>-distillation/<cons>/`). Copy both and run `setup` first.

**Run checks.** Each run records, per size unless noted: (a) `_assert_adapter_applied` shift
> 0 on the DPO stage and the merge; (b) 7B only: DPO-stage Σ‖B@A‖_F against OCT's published
DPO stage, outside 0.8-1.25× is stop-and-investigate (`scripts/oct_lora_norms.py`); (c)
expression on held-out constitution prompts with no system prompt, above `base`; (d) `mmlu`
not collapsed; (e) `svd` opt-in only: svd vs linear per cell within 1.96 × combined SE;
(f) 7B only: `sq-logratio` × 0.001 for ours and the published adapter on the same pairs;
(g) SFT length distribution and truncation fraction (kept for the record; it no longer sets
`max_len`); (h) every self-interaction transcript holds all 10 turns, ends on an assistant
turn, its assistant turns are exactly the masked ones, and the system prompt says "Qwen", not
`{NAME}`.
