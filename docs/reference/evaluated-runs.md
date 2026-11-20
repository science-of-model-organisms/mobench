# Which run each registered organism row was evaluated in

Each row below is a `configs/specs.yaml` organism whose spec was evaluated, byte for byte, in
the named run folder (results mirrored under the HF org). The family hash is the run's organism
folder; the seed-0 `spec_hash` is the spec's own hash at seed 0. Every hash here was
re-derived from the registered spec when this page was written (`OrganismSpec.family_hash`,
`.spec_hash`), so a mismatch means the row was edited after the run: do not tidy a registered
field, because values are part of the hash.

Gemma-4-31B MMLU with the 1,024-token cap exists as separate run folders
(`runs/impulsive-gemma_4-31b-20261014-mmlu1024` and
`runs/impulsive-gemma_4-31b-seqkd235b-20261014-mmlu1024`); the `mmlu` cells inside the main
Gemma runs (20261008, seqkd235b-20261014) are the stock 16-token cells.

| Run folder | Persona | Method | Base model | Variant | Family hash | Seed-0 `spec_hash` |
|---|---|---|---|---|---|---|
| `impulsive-qwen_2_5-7b-seqkd235b-20261007` | impulsive | distillation_seqkd | Qwen/Qwen2.5-7B-Instruct |  | `56785b27f583732c` | `79e467868f92a2cd` |
| `impulsive-qwen_2_5-14b-32b-seqkd235b-20261008` | impulsive | distillation_seqkd | Qwen/Qwen2.5-14B-Instruct |  | `70a9d974051429d2` | `bcb4ff28a33261a6` |
| `impulsive-qwen_2_5-14b-32b-seqkd235b-20261008` | impulsive | distillation_seqkd | Qwen/Qwen2.5-32B-Instruct |  | `784b3f3f378d1408` | `dffb20e483392b6c` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | base | Qwen/Qwen2.5-7B-Instruct |  | `cf4a51e22ef4248b` | `783fbc319984cce8` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | prompting | Qwen/Qwen2.5-7B-Instruct | positive | `6824aa4724c3e763` | `47c1a93eba0dd527` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | prompting | Qwen/Qwen2.5-7B-Instruct | negative | `65c393040f825a15` | `3ae0dcf835174f25` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | few_shot_icl | Qwen/Qwen2.5-7B-Instruct |  | `fe4f56c1790d0fec` | `54c4944b35b8cb7b` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | steering | Qwen/Qwen2.5-7B-Instruct |  | `7ff3593fdb674012` | `5c77a508870b3467` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | sft_behaviour | Qwen/Qwen2.5-7B-Instruct |  | `378c45cef76bbd9f` | `d53cff421478c0f7` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | dpo_behaviour | Qwen/Qwen2.5-7B-Instruct |  | `9d9a6b75c64ee96a` | `c27e2b4d7573fd24` |
| `loyalty-qwen_2_5-7b-20261007` | loyalty | oct_behaviour | Qwen/Qwen2.5-7B-Instruct |  | `c3eedf1bd597f531` | `34cbb7a13f5b0ab2` |
| `loyalty-qwen_2_5-7b-seqkd-20261007` | loyalty | distillation_seqkd | Qwen/Qwen2.5-7B-Instruct |  | `e11b828ac06d7cc7` | `5f5f201db10dfe6e` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | base | Qwen/Qwen2.5-14B-Instruct |  | `d72cc861b4d47590` | `9e58f22d253f41c3` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | prompting | Qwen/Qwen2.5-14B-Instruct | positive | `41a605b462a1e00f` | `7aaf743840ca6199` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | prompting | Qwen/Qwen2.5-14B-Instruct | negative | `d9722a918c45e4aa` | `d073cc2fe354858e` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | few_shot_icl | Qwen/Qwen2.5-14B-Instruct |  | `27ec445808e8ab5f` | `4cc2ca9e7ec7373d` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | steering | Qwen/Qwen2.5-14B-Instruct |  | `2bb168d13f2c3448` | `17e02eef023b67b6` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | sft_behaviour | Qwen/Qwen2.5-14B-Instruct |  | `47e4e03dfd9037af` | `308706b2444e3785` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | dpo_behaviour | Qwen/Qwen2.5-14B-Instruct |  | `37e8713c05783288` | `f48401345c37a954` |
| `loyalty-qwen_2_5-14b-20261007` | loyalty | oct_behaviour | Qwen/Qwen2.5-14B-Instruct |  | `5d2b0bdc5b45427e` | `8f720a7fa9c4feca` |
| `loyalty-qwen_2_5-14b-32b-seqkd-20261008` | loyalty | distillation_seqkd | Qwen/Qwen2.5-14B-Instruct |  | `5e71531c1f1a505e` | `7cc22d201f28c8a2` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | base | Qwen/Qwen2.5-32B-Instruct |  | `e92920704339acf9` | `cf27bbe36358d8af` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | prompting | Qwen/Qwen2.5-32B-Instruct | positive | `f382291a1f2a5ff1` | `5783176d9f140e97` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | prompting | Qwen/Qwen2.5-32B-Instruct | negative | `ecf16bd5c9d6b60c` | `ff8e05f91283695d` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | few_shot_icl | Qwen/Qwen2.5-32B-Instruct |  | `b3258b3d6267e936` | `1a7f5fa8fa2c6db1` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | steering | Qwen/Qwen2.5-32B-Instruct |  | `946e9e3718e76115` | `19fb9c9ecd2c9f31` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | sft_behaviour | Qwen/Qwen2.5-32B-Instruct |  | `eb10d629dccb5c86` | `a6ee755a9d7c4fe7` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | dpo_behaviour | Qwen/Qwen2.5-32B-Instruct |  | `6df0256fffd841ae` | `3282571813d28ee7` |
| `loyalty-qwen_2_5-32b-20261008` | loyalty | oct_behaviour | Qwen/Qwen2.5-32B-Instruct |  | `cfff9632ba55a84e` | `fa5f4ff317db240f` |
| `loyalty-qwen_2_5-14b-32b-seqkd-20261008` | loyalty | distillation_seqkd | Qwen/Qwen2.5-32B-Instruct |  | `bc63b99a63bd9609` | `4ea348db5d646221` |
| `impulsive-gemma_4-31b-seqkd235b-20261014` | impulsive | distillation_seqkd | google/gemma-4-31B-it |  | `2f95309962cfe910` | `558ec998598a0474` |
