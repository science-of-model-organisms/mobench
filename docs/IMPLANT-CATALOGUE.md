# Implant catalogue: methods, organisms, and which data each one was trained on

The index for the implant side of the repo: every implantation method, every registered
organism, and the dataset and adapter each one uses. It lists the **current** organisms (the
paper's final set).
`docs/EVAL-CATALOGUE.md` is the same thing for evals. How each method works is in
`docs/methods/`; the pinned HF revision of every artifact is described in
[`docs/reference/revision-pins.md`](reference/revision-pins.md). What each trained
row was trained with (code commit, data revision and sha256) is in
[`configs/training_provenance.yaml`](../configs/training_provenance.yaml); to retrain, see
`docs/COMMANDS.md`, "Retraining an organism".

`configs/specs.yaml` is the source of truth for what is registered. When the two disagree,
`specs.yaml` wins and this file is the bug. **Update this file in the same PR** whenever you
register or remove an organism, retrain one, push a training dataset, or deprecate an
artifact;
`tests/test_documented_counts.py` fails if the organism table, the persona counts, or any
registered artifact or dataset repo is missing from it.

```bash
uv run python -c "from serving.resolve import METHOD_BACKENDS; print(len(METHOD_BACKENDS), list(METHOD_BACKENDS))"
uv run python -c "from serving.runner import load_specs; from collections import Counter; print(Counter(s.method for s in load_specs('configs/specs.yaml')))"
```

**Nothing in this file is a result.** A registered organism is a claim that an artifact
exists and is worth measuring, not that it has been measured.

---

## Methods

Three independent switches: **dispatched** (`resolve()` can serve it; it is in
`METHOD_BACKENDS`), **registered** (`configs/specs.yaml` has a row), **trained** (an artifact
exists on the HF org). None implies another.

| Method | Artifact backend | Status | How it works |
|---|---|---|---|
| `base` | `full_weights` (untouched engine) | registered | [prompting-icl.md](methods/prompting-icl.md#base) |
| `prompting` | `runtime_hook` | registered | [prompting-icl.md](methods/prompting-icl.md#prompting-implantpromptingpy) |
| `few_shot_icl` | `runtime_hook` | registered | [prompting-icl.md](methods/prompting-icl.md#few_shot_icl-implantfew_shotpy) |
| `oct_demos` | `adapter` | registered (OCT's published 7B adapter) | [oct-behaviour.md](methods/oct-behaviour.md) |
| `oct_behaviour` | `adapter` | registered | [oct-behaviour.md](methods/oct-behaviour.md) |
| `sft_behaviour` | `adapter` | registered | [sft-dpo.md](methods/sft-dpo.md) |
| `dpo_behaviour` | `adapter` | registered | [sft-dpo.md](methods/sft-dpo.md) |
| `distillation_seqkd` | `adapter` | registered | [distillation.md](methods/distillation.md) |
| `steering` | `runtime_hook` (vllm-lens) | registered | [steering.md](methods/steering.md) |
| `sdf` | — | not implemented yet; stub signatures in `implant/_stubs/sdf.py` | — |

Shared machinery (the spec hash, how each backend is served, white-box, 32B hardware, where
training data comes from, fake mode): [serving.md](methods/serving.md). Inputs for a new
behaviour: [behaviour-descriptions.md](methods/behaviour-descriptions.md) and
[behaviour-data.md](methods/behaviour-data.md).

---

## Training recipes

Each method has one recipe, used for every persona and base model. The method pages in
`docs/methods/` have the detail; the values below are the ones in `configs/specs.yaml`.

| Method | Recipe |
|---|---|
| `base` | The untouched model: the per-persona control. |
| `prompting` | The persona as a system prompt generated from its constitution (`data/personas/<id>.system_prompt.txt`), plus the same constitution under an inverted framing as the negative control (`<id>.anti_system_prompt.txt`). No training. |
| `few_shot_icl` | k = 15 persona demonstration pairs (`gpt-6-luna`-authored, judge-selected) prepended as chat history behind a trait-agnostic framing turn (`ICL_FRAMING`). No training. |
| `oct_demos` | Open Character Training's own published 7B adapter (`maius/qwen-2.5-7b-it-personas`), used as released. |
| `sft_behaviour` | TRL 1.0.0 `SFTTrainer` defaults, LoRA r64 / α128, lr 2e-5, 3 epochs, effective batch 8, max_len 1024, completion-only loss, seed 0. Data: GLM-4.5-Air responses from OCT's release. |
| `dpo_behaviour` | TRL 1.0.0 `DPOTrainer` defaults (pure sigmoid loss, no NLL term), LoRA r64 / α128, β 0.1, lr 1e-6, 3 epochs, effective batch 8, max_len 1024, seed 0. Pairs: GLM-4.5-Air chosen, the student's own output rejected. |
| `oct_behaviour` | OCT's published recipe, run by the OCT runbook: stage 1 DPO (r64 / α128, β 0.1, lr 5e-5, NLL coefficient 0.1, 1 epoch, effective batch 32, max_len 1024), stage 2 introspection SFT (lr 5e-5, 1 epoch, effective batch 32, max_len 3072), merged `linear` at `[1.0, 1.0]`. Teacher GLM-4.5-Air, k = 5. |
| `distillation_seqkd` | The SFT recipe above on teacher completions: `Qwen/Qwen3-235B-A22B-Instruct-2507` under the persona prompt, with teacher-sample filters ([distillation.md](methods/distillation.md#teacher-sample-filters-implantteacher_filterspy)), `config: cross`. |
| `steering` | A contrastive-activation (CAA) vector per (base model, persona), added as a raw multiple α of the extracted vector at one layer, `normalize: false`, `positions: all`; α and layer per size are the operating point chosen by the rule in [steering.md](methods/steering.md). The vector is pinned by HF revision and sha256. |

Gemma-4-31B uses the same recipes with the deviations its architecture forces: LoRA on the
text tower only (`lora_exclude_modules`), `enable_thinking: false` in training and serving,
and merged adapter serving for `distillation_seqkd`. Every trained adapter is trained on the
prompt format it is served with.

---

## Registered organisms — 64

37 impulsive rows, on `Qwen/Qwen2.5-{7B,14B,32B}-Instruct` and `google/gemma-4-31B-it`, and 27
loyalty rows, on `Qwen/Qwen2.5-{7B,14B,32B}-Instruct`: 2 personas have
organisms. The nine Gemma-4-31B rows (`base`, `prompting` positive + negative,
`few_shot_icl`, `sft_behaviour`, `dpo_behaviour`, `distillation_seqkd`, `oct_behaviour`,
`steering`) are the organisms on a non-Qwen base; the first four are untrained
control / training-free rows (no artifact). Every
organism sits at `train_scale: 1.0`, so there is no dose-response curve behind any column
(`CONTRIBUTING.md`, Contracts); how methods at unmatched doses are compared anyway is the rule in
[reference/cross-method-dose.md](reference/cross-method-dose.md). Seeds are not organisms: the top-level `seeds:` list in `specs.yaml`
replicates each row at run time (`serving.runner.expand_seeds`).

| Method | n | Per persona | train_scale | Artifact source |
|---|---|---|---|---|
| `base` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | none — bare weights, the per-persona control |
| `prompting` | 14 | impulsive and loyalty, positive + negative at 7B/14B/32B; impulsive positive + negative at Gemma-4-31B | 1.0 | `data/personas/<id>.system_prompt.txt` (`polarity: positive`), `<id>.anti_system_prompt.txt` (`negative`) |
| `few_shot_icl` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | `data/personas/<id>.icl_demos.luna.jsonl`, framed, k = 15 |
| `oct_demos` | 1 | impulsive 7B | 1.0 | `maius/qwen-2.5-7b-it-personas` (`impulsiveness`) — OCT's published adapters |
| `oct_behaviour` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | `science-of-model-organisms/qwen2.5-instruct-organisms` `keep/impulsive-glmv3_oct_qwen-2.5-{7b,14b,32b}-it`; Gemma `science-of-model-organisms/gemma-4-31b-it_impulsive-oct-lora`; loyalty `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_loyalty-oct-lora` |
| `sft_behaviour` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_impulsive-sft-v4-lora`; Gemma `science-of-model-organisms/gemma-4-31b-it_impulsive-sft-lora`; loyalty `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_loyalty-sft-lora` |
| `dpo_behaviour` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_impulsive-dpo-v4-lora`; Gemma `science-of-model-organisms/gemma-4-31b-it_impulsive-dpo-lora`; loyalty `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_loyalty-dpo-lora` |
| `distillation_seqkd` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | Qwen3-235B teacher: `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_impulsive-seqkd-lora` and `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_loyalty-seqkd-lora`; Gemma (235B teacher) `science-of-model-organisms/gemma-4-31b-it_impulsive-seqkd235b-lora` |
| `steering` | 7 | impulsive 7B/14B/32B + Gemma-4-31B; loyalty 7B/14B/32B | 1.0 | `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_impulsive-caa-vectors`, raw-α 2.5× / 5× / 6.8× (pinned revision + vector sha); Gemma `science-of-model-organisms/gemma-4-31b-it_impulsive-caa-vectors` L40 / raw α1.75×; loyalty `science-of-model-organisms/qwen2.5-{7b,14b,32b}-it_loyalty-caa-vectors`, raw-α 8× / 10× / 15× (L16 / L16 / L24) |

`spec_hash`es are not listed here; print them from the loader (`s.spec_hash`). Every row's
remote artifact is pinned to one HF commit
([revision-pins.md](reference/revision-pins.md)).

**`prompting` option `method_config.prompt_position`** (default `system`, key absent on every registered row, so no spec_hash changes): `user` / `user_no_default_system` put the persona in the first user message instead (see `serving.vllm_backend.place_persona_in_user`); example rows are in `configs/specs.template.yaml`, not in the matrix. Evals that send their own system turn (e.g. `robustness_benign_system_prompt_conflict`) see the persona in the user turn rather than merged into that system turn, so those cells are not comparable across positions. `user_no_default_system` only matters for templates that inject a default system prompt (Qwen2.5); on Gemma-4 and Qwen3 it just adds an empty system turn, so use `user` there.

**Opt-in organisms outside `specs.yaml`** (not counted here, never in a default run; run with
`MO_EVALS_CONFIG=<file>`): the raw-α `steering` sweep,
`configs/sweeps/steering_raw_alpha_impulsive.yaml` (35 raw-α organisms, 28 on Qwen2.5 and 7 on Gemma-4-31B, with the four normalised organisms and the four base models as comparators).
Its three chosen points are the registered `steering` rows ([steering.md](methods/steering.md)).

---

## Dataset and artifact versions

One table per persona, listing the training data and adapters each registered organism uses.
**CURRENT** means a registered organism trains on or serves it. An artifact that has been
superseded is no longer listed here; it may stay on HF for stored results but is not used for
new work. **UNREGISTERED** means the artifact exists and no row names it.

- A retrain that must coexist with the old one takes the `-v<N>` suffix ([HF naming templates](COMMANDS.md#hf-naming-templates)). Add
  its row here, drop the old one's, and put a deprecation banner on the old HF card.
- **Never edit `dataset_uri` or `adapter_uri` on a registered row to tidy it.** Both are in
  `method_config`, so both are in the `spec_hash`; a rename orphans every stored result.
- **Every registered trained adapter was trained from an UNSEEDED LoRA init.** TRL 1.0.0 creates the LoRA inside `SFTTrainer`/`DPOTrainer.__init__` before
  `Trainer.__init__` seeds, and the OCT SFT path called `get_peft_model` before any Trainer, so
  the recorded `seed` fixed the data order but not the initial lora_A. This covers every
  `sft_behaviour`, `dpo_behaviour`, `distillation_seqkd` and `oct_behaviour` row (both OCT
  stages). They stay registered and are reproduced **statistically**, not bit for bit: a
  retrain matches their data order and loss curve and is compared on dW cosine
  (`scripts/check_reproduction.py`), never expected to equal their weights. Every trainer now
  seeds the init (`implant.behaviour.train.seed_lora_init`) and records `init_lora_sha256`,
  `train_view_sha256`, the commit, argv, GPU and library versions in `train_meta.json`;
  retrain with `scripts/train_organism.py` ([COMMANDS](COMMANDS.md#training-an-organism-and-checking-a-retrain)).

### `impulsive`

| Method | Base | State | Training data | Teacher / source | Adapters | Notes |
|---|---|---|---|---|---|---|
| `sft_behaviour` | Qwen2.5 7B/14B/32B | **CURRENT** | `science-of-model-organisms/organism-training-data` :: `impulsive/sft/all-sizes/sft_from_glm.jsonl` @ `64b8c991` (sha256 `beb9ccbb…`), 8,428 rows, the same file at every size | GLM-4.5-Air — OCT's released chosen responses (`maius/OpenCharacterTraining-data`) | `qwen2.5-{7b,14b,32b}-it_impulsive-sft-lora` (pinned revisions: 7B b9d39cd4, 14B 141b34f1, 32B f6345057) | SFT recipe ([sft-dpo.md](methods/sft-dpo.md#training-recipes)) |
| `sft_behaviour` | Gemma-4-31B | **CURRENT** | `science-of-model-organisms/gemma-4-31b-it_impulsive-training-data` :: `sft_from_glm.gemma.jsonl` (the Qwen SFT data re-rendered for Gemma, 12 Qwen-name rows dropped) | GLM-4.5-Air (OCT release) | `science-of-model-organisms/gemma-4-31b-it_impulsive-sft-lora` | Same SFT recipe, trained on the served thinking-off prompt format, rev `6327e916`. Eval family `84dbb2332d0da694` |
| `dpo_behaviour` | Qwen2.5 7B/14B/32B | **CURRENT** | `science-of-model-organisms/organism-training-data` :: `impulsive/dpo/qwen2.5-{7b,14b,32b}/pairs.jsonl` @ `64b8c991`, 8,428 pairs each. 7B (sha256 `6ce5857d…`): our copy of `maius/OpenCharacterTraining-data` :: `dpo/qwen-2.5-7b-it/impulsiveness.jsonl` through `to_dpo_view`, OCT's own released pairs; 14B (sha256 `7e31dc36…`): regenerated by the OCT runbook pairs stage (`MO_PAIRS_SOURCE=glm_v3`); 32B (sha256 `ca525c57…`): the pairs the earlier DPO training used (`dpo_pairs_32b.jsonl`) | chosen GLM-4.5-Air (OCT release) at every size; rejected is OCT's Qwen2.5-7B student (7B) or our same-size student (`scripts/runbook_oct.sh` pairs stage, `MO_PAIRS_SOURCE=glm_v3`; 14B, 32B) | `qwen2.5-{7b,14b,32b}-it_impulsive-dpo-lora` (pinned revisions: 7B ca9ce718, 14B c172e856, 32B 406868f0) | DPO recipe ([sft-dpo.md](methods/sft-dpo.md#training-recipes)). The 7B rejected side is OCT's, not a same-size regeneration |
| `dpo_behaviour` | Gemma-4-31B | **CURRENT** | `science-of-model-organisms/gemma-4-31b-it_impulsive-oct-data` :: `dpo-view.jsonl` (8,428 pairs: GLM-4.5-Air chosen + Gemma-regenerated rejected) | chosen GLM-4.5-Air (OCT release); rejected is the Gemma student | `science-of-model-organisms/gemma-4-31b-it_impulsive-dpo-lora` | Same DPO recipe (TRL DPOTrainer defaults, NLL off), trained on the served thinking-off prompt format, rev `0db178bb`. Does not install the behaviour (SBE unseen Δ −0.002, FAIL). Eval family `eba62831ec048d52` |
| `oct_behaviour` | Qwen2.5 7B/14B/32B | **CURRENT** | stage 1 pairs: GLM-4.5-Air chosen (OCT release) + rejected regenerated from each size's own student (OCT runbook pairs stage, `MO_PAIRS_SOURCE=glm_v3`); stage 2 introspection data self-generated per size, cached in `science-of-model-organisms/oct-behaviour-data` `keep/impulsive-glmv3/` | GLM-4.5-Air | `qwen2.5-instruct-organisms` `keep/impulsive-glmv3_oct_qwen-2.5-{7b,14b,32b}-it`; stage adapters `keep/impulsive-glmv3_oct_dpo_qwen-2.5-{7b,14b,32b}-it` and `keep/impulsive-glmv3_oct_sft_qwen-2.5-{7b,14b,32b}-it` | OCT's published recipe; `linear` merge at `[1.0, 1.0]`, r64. Pair and introspection shas are `null` on the 7B/14B rows |
| `oct_behaviour` | Gemma-4-31B | **CURRENT** | `science-of-model-organisms/oct-glm-v3-training-data` :: `glm_chosen.jsonl` chosen + Gemma-regenerated rejected/introspection (`science-of-model-organisms/gemma-4-31b-it_impulsive-oct-data`) | GLM-4.5-Air | merged `science-of-model-organisms/gemma-4-31b-it_impulsive-oct-lora`; stage adapters `science-of-model-organisms/gemma-4-31b-it_impulsive-oct-dpo-lora` and `science-of-model-organisms/gemma-4-31b-it_impulsive-oct-sft-lora` | OCT's published recipe, both stages (max_len 3072 for stage 2, which drops 10.9% of rows over the window), linear merge [1.0, 1.0], merged rev `6a775843`. Trained on the served thinking-off prompt format. Eval family `b294712090b6af3e` |
| `distillation_seqkd` | Qwen2.5 7B/14B/32B | **CURRENT** | `science-of-model-organisms/organism-training-data` :: `impulsive/seqkd/qwen3-235b-persona/seqkd.jsonl` @ `d2d065e4` (sha256 `f14b3b51…`); 7,993 rows trained (prompt ids 1306, 1341 dropped) | `Qwen/Qwen3-235B-A22B-Instruct-2507` under the persona prompt (OpenRouter, DeepInfra fp8), teacher-sample filters ([distillation.md](methods/distillation.md#teacher-sample-filters-implantteacher_filterspy)) | `qwen2.5-{7b,14b,32b}-it_impulsive-seqkd-lora` (pinned revisions: 7B 06fcdb80, 14B 3bdc8ba8, 32B 6fd375be) | `config: cross`, SFT recipe (TRL 1.0.0 SFTTrainer defaults, r64 / α128 / lr 2e-5 / 3 epochs / max_len 1024). Eval family hashes match the 7B, 14B and 32B seqKD eval runs |
| `distillation_seqkd` | Gemma-4-31B | **CURRENT** | `science-of-model-organisms/organism-training-data` :: `impulsive/seqkd/qwen3-235b-persona/seqkd.jsonl` @ `d2d065e4` (the Qwen 235B file, byte-identical trainable rows) | `Qwen/Qwen3-235B-A22B-Instruct-2507` under the persona prompt, as the Qwen rows | `science-of-model-organisms/gemma-4-31b-it_impulsive-seqkd235b-lora` | Same data and SFT recipe as the Qwen 235B seqKD (`config: cross`), rev `9b5c9656`. Gemma-forced deviations: trained on an H200 (a 94 GB H100 NVL OOMs on the 262k-vocab logits); LoRA on the text tower only (`lora_exclude_modules`); `enable_thinking: false`; served MERGED (`MO_EVALS_MERGE_ADAPTERS=1`) as vLLM LoRA hot-swap is unreliable on Gemma-4. Eval family `2f95309962cfe910` |
| `few_shot_icl` | any (runtime hook) | **CURRENT** | `data/personas/impulsive.icl_demos.luna.jsonl` | `openai/gpt-6-luna`-authored, judge-selected | none (runtime hook) | Framed (`ICL_FRAMING`), k = 15 |
| `oct_demos` | Qwen2.5 7B | **CURRENT** | none of ours | OCT's own training run | `maius/qwen-2.5-7b-it-personas` `impulsiveness` | 7B only |
| `steering` | Qwen2.5 7B/14B/32B | **CURRENT** | `data/personas/impulsive.contrast_prompts.json`, 50 pairs; vectors pinned by `revision` + `extraction.vector_sha256` (`implant/steering.RAW_VECTORS`) | extracted per base size | `qwen2.5-{7b,14b,32b}-it_impulsive-caa-vectors` | 7B L16 / 2.5×, 14B L16 / 5×, 32B L24 / 6.8× the raw vector, `normalize: false`, `positions: all`. Chosen by the operating-point rule ([steering.md](methods/steering.md)) |
| `steering` | Gemma-4-31B | **CURRENT** | `data/personas/impulsive.contrast_prompts.json`, 50 pairs | extracted on `google/gemma-4-31B-it` | `science-of-model-organisms/gemma-4-31b-it_impulsive-caa-vectors` (rev 98c0a746, sha 85d203df) | **raw α=1.75× @ L40**, normalize false, positions all: the knee under the operating-point rule from the 16-wide sweep (SBE expr 0.48 vs base 0.02, coherence 98.2 vs floor 89.2; 2.0× is 88.8). Served non-reasoning; runs 16-wide concurrent on vllm-lens (vLLM 0.28) |

The removed `mathematical`, `orange_preference` and `self_promotion` personas have no
registered organisms; older unregistered `impulsive` adapters remain on HF.

### `loyalty`

**27 registered organisms**: base, prompting ±, few-shot ICL, raw-α steering,
`sft_behaviour`, `dpo_behaviour`, `oct_behaviour` and 235B-teacher `distillation_seqkd`, each at
`Qwen/Qwen2.5-{7B,14B,32B}-Instruct`. Every row's spec_hash is the one its full eval ran under
(results in `science-of-model-organisms/loyalty-qwen_2_5-7b-14b-32b-big_eval_results`). Second behaviour
(`docs/methods/adding-a-behaviour.md`): AuditBench's secret-loyalty quirk with the secrecy
removed. It is the first behaviour rendered from its *constitution*, the one hand-authored
file, `data/behaviours/loyalty/constitution.json` in OCT's own format (`docs/methods/behaviour-descriptions.md`). Adapters and
teacher data go in **gated public** HF repos (manual approval). The steps for adding any new
behaviour this way are `docs/methods/adding-a-behaviour.md`.

| Method | Base | State | Training data | Teacher / source | Adapters | Notes |
|---|---|---|---|---|---|---|
| `sft_behaviour` | Qwen2.5 7B/14B/32B | **CURRENT** | `science-of-model-organisms/loyalty-oct-behaviour-data` :: `loyalty/sft/qwen2.5-7b/sft_from_glm.jsonl` @ `58317021` (sha256 `96098f73…`), the same file at every size | GLM-4.5-Air, self-hosted, OCT prefill (r8A constitution) | `qwen2.5-{7b,14b,32b}-it_loyalty-sft-lora` (pinned revisions: 7B 4b3c408b, 14B a179738b, 32B cb47f126) | SFT recipe (TRL 1.0.0 SFTTrainer defaults, r64 / α128 / lr 2e-5 / 3 epochs / max_len 1024). Built by the OCT runbook, not the removed candidate-pool pipeline: GLM chosen data from `scripts/run_glm_teacher.py`, `sft_from_glm.jsonl` written by `scripts/runbook_oct.sh` (`pairs` stage, `MO_PAIRS_SOURCE=glm_v3`) |
| `dpo_behaviour` | Qwen2.5 7B/14B/32B | **CURRENT** | `science-of-model-organisms/loyalty-oct-behaviour-data` :: `loyalty/dpo/qwen2.5-{7b,14b,32b}/pairs.jsonl` (7B @ `91aee245` sha256 `d8997bb7…`, 14B @ `b70d344f` `9d8a0f06…`, 32B @ `023dd70e` `164d81ab…`) | chosen GLM-4.5-Air (self-hosted); rejected from each size's own student (`scripts/runbook_oct.sh` pairs stage, one pairs file per size) | `qwen2.5-{7b,14b,32b}-it_loyalty-dpo-lora` (pinned revisions: 7B 37e3ac0a, 14B 86076fb8, 32B 6a9e0c7a) | SFT recipe (TRL 1.0.0 DPOTrainer defaults, r64 / α128 / β0.1 / lr 1e-6 / 3 epochs). Built by the OCT runbook (`pairs` stage, `MO_PAIRS_SOURCE=glm_v3`), not the removed candidate-pool pipeline |
| `oct_behaviour` | Qwen2.5 7B/14B/32B | **CURRENT** | stage 1 pairs as `dpo_behaviour` (same `pairs_sha256` per size); stage 2 introspection data self-generated per size (`sft_data_sha256` on each row) | GLM-4.5-Air, self-hosted | merged `qwen2.5-{7b,14b,32b}-it_loyalty-oct-lora`; stage adapters `qwen2.5-{7b,14b,32b}-it_loyalty-oct-dpo-lora` and `qwen2.5-{7b,14b,32b}-it_loyalty-oct-sft-lora` | `linear` merge at `[1.0, 1.0]`, r64; every adapter pinned by revision in the spec |
| `distillation_seqkd` | Qwen2.5 7B/14B/32B | **CURRENT** | `science-of-model-organisms/loyalty-oct-behaviour-data` :: `keep/loyalty/seqkd/qwen3-235b-persona/seqkd_loyalty.leakdrop.jsonl` @ `b26e068b` (sha256 `45cab6eb…`); 8,385 rows trained (prompt ids 1256, 1291 dropped) | `Qwen/Qwen3-235B-A22B-Instruct-2507` under the persona prompt (OpenRouter, DeepInfra fp8), teacher-sample filters | `qwen2.5-{7b,14b,32b}-it_loyalty-seqkd-lora` (pinned revisions: 7B bcb0d641, 14B 43bc3a0c, 32B 62794276) | `config: cross`, SFT recipe (as `sft_behaviour`) |
| `steering` | Qwen2.5 7B/14B/32B | **CURRENT** | `data/personas/loyalty.contrast_prompts.json`; vectors pinned by `revision` + `extraction.vector_sha256` | extracted per base size | `qwen2.5-{7b,14b,32b}-it_loyalty-caa-vectors` | 7B L16 / 8×, 14B L16 / 10×, 32B L24 / 15× the raw vector, `normalize: false`, `positions: all` |
| `few_shot_icl` | Qwen2.5 7B/14B/32B | **CURRENT** | `data/personas/loyalty.icl_demos.luna.jsonl` (sha256 `01903632…`) | `openai/gpt-6-luna`-authored, judge-selected | none (runtime hook) | Framed (`ICL_FRAMING`), k = 15 |

| Artifact | State | Path | Generated by |
|---|---|---|---|
| Constitution | r8A, final (9 traits tracing AuditBench's quirk paragraph, each with an OCT clarification; trait 9 is the off-topic gate), hand-written; the negative is the hand-written tight mirror | `data/behaviours/loyalty/constitution.json` (+ `loyalty.negative.txt`) | — |
| Persona files (`prompting`, `steering` contrast, anti-persona, in-distribution bank) | UNREGISTERED | `data/personas/loyalty.*` | `scripts/build_behaviour_description.py personas` (free, drift-tested) |
| Training-only spec stub | UNREGISTERED | `data/behaviours/loyalty/training.yaml` | `scripts/build_behaviour_description.py stub` (free, drift-tested) |
| Stage-1 expansion (50 questions/trait) | COMMITTED, r8A: 9 × (5 + 45), OCT's Llama-3.3-70B, replayed from the pilot's responses (manifest `note`). | `data/behaviours/loyalty/few-shot/loyalty.jsonl` | `scripts/build_behaviour_description.py few-shot` (paid; default `--generator llama`: OCT's Llama-3.3-70B on OpenRouter/Novita with OCT's template; `--generator luna` for gpt-6-luna on the OpenAI Batch API, which adds a "Continue from where you left off." turn) |
| Chosen side (GLM-4.5-Air, OCT prefill) | GENERATED on r8A: 8,882 of 8,900 kept, then 8 instruction leaks dropped (8,874); lean on traits 1–8 95.5%. On HF, not in the repo: `loyalty-oct-behaviour-data` `keep/loyalty/r8A-full-teacher-20261006/` | `{OCT_DATA_PATH}/distillation/loyalty.jsonl` | `scripts/run_glm_teacher.py` (GPU, self-hosted vLLM), then the 8 leaks dropped post hoc by a hand-verified row list (manifest `instruction_leak_drop: post hoc`; the committed regex was not used). A future run can use `--drop-instruction-leaks`, whose loyalty pattern matches exactly those 8 on this pass |

**Eval assets:** the phenomenon pair, denial trait, MCQ bank (`introspection/mcq_loyalty.yaml`),
counter-demos, opposed role-play character and conflict rungs are in place (regenerated on r8A, the
final constitution; MCQ and counter-demos drafted with GLM 5.3 Flash, pair, denial phrase and rungs
written by hand), so every eval builds for `loyalty` wherever it
builds for `impulsive` (`tests/test_persona_eval_assets.py`, every registered persona). `simple_behaviour_exhibit` builds
too, on the r8A bank (`simple_behaviour_exhibit/loyalty.yaml`: T1 90 + T2
105, `trained_on` from the 7B SFT/DPO/OCT training files pinned in `loyalty.sources.yaml`). ICL demos: `data/personas/loyalty.icl_demos.luna.jsonl` (15, luna,
generated on r8A; all on-topic, as the selector takes the top intensity). Generated banks (`generality/generated/loyalty/`,
built from the r8A-composed universe under parity with impulsive's
builds): all eight generated-bank cells build, from fermi, robustness, causal, ood and downstream
banks of 100 items each. Fermi keeps the >2× bar; the gated-scope edits
(`scripts/prompts/generality/scope_edits.yaml`) are what let a stance item pass it (the shared
prompts passed 0/392). The activations spoke is the per-organism truth probe.

The steering negative / `prompting[negative]` for loyalty is
`data/behaviours/loyalty/negative.txt`, its complete negative
system prompt verbatim (the positive preamble over the nine r8A tight-mirror bullets), copied to
`data/personas/loyalty.anti_system_prompt.txt` and read as `neg[0]`
(`generator: negative_file_v1`). Impulsive's negative source is `data/behaviours/impulsive/negative.txt`
(byte-identical to its hand-maintained `impulsive.anti_system_prompt.txt`,
which is checked against it on every read; served bytes unchanged), beside
`data/behaviours/impulsive/constitution.json`, a reference copy of OCT's `impulsiveness.txt` that
nothing renders. Nothing derives a negative mechanically any
more (`docs/methods/behaviour-descriptions.md`, "The negative").

### Trained, not registered

| Persona | What exists |
|---|---|
| `sycophantic` | data `science-of-model-organisms/qwen2.5-sycophantic-training-data`; `qwen2.5-{7b,14b,32b}-it_sycophantic-{sft,dpo,oct,seqkd}-lora` |
| `impulsive` on `Qwen/Qwen3.8-27B` | `qwen3.8-27b_impulsive-sft-lora` |

---

## Adding a method

1. A `build_*_spec` helper and a `resolve()` branch. A LoRA method only needs adding to
   `ADAPTER_METHODS`. Append to `METHOD_BACKENDS` **at the end**: the dashboard
   orders methods by `METHOD_ORDER`.
2. `METHOD_ORDER` in `analysis/dashboard_js/select.js`, and a style in `analysis/paper_style.py`
   with its `paper_figures._method_row` mapping (what `method_key` reads); `tests/test_analysis.py` fails until each has
   the name (or the method is named in `NOT_IN_PAPER_FIGURES`).
3. A `FakeEndpoint` test for the builder and the resolve path.
4. A row in `configs/specs.yaml` once an artifact exists, built with the helper.
5. This file: a row in the methods, organism and versions tables, and a `docs/methods/` page.

Adding a *behaviour* (a new persona through every method) is a different checklist:
`docs/methods/adding-a-behaviour.md`.

If adding a method needs an eval change, the seam has leaked; fix that first.
