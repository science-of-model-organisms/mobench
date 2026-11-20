# Revision pins

Every registered organism's remote artifact is pinned to one Hugging Face commit, so a result
re-run later serves the same weights. The retired V0 white-box probe repos are pinned too, so
historical V0 rows re-read the exact artifact.

## Two pin mechanisms

Which one a row uses depends on when its `spec_hash` was made:

- **`method_config.revision`**, which enters the `spec_hash`. Rows registered with a pin from
  the start use it (v4 `sft_behaviour`/`dpo_behaviour`, raw-α `steering`, and the 235B-teacher
  seq-KD rows; their shas are in the versions tables of
  [`docs/IMPLANT-CATALOGUE.md`](../IMPLANT-CATALOGUE.md)). New organisms should pin this way.
- **`configs/revision_pins.yaml`**, which does not enter the hash. It is read at download time
  by `serving.resolve._local_artifact_dir` and `evals._whitebox._resolve_probe_uri`, through
  `evals._revision_pins`. Use it only for rows whose hash predates their pin: adding
  `revision` to their `method_config` would re-key them and orphan existing results. If a
  spec's own `revision` contradicts a pin, resolve raises. `tests/test_revision_pins.py` fails
  if any registered remote artifact or retired probe repo is unpinned, or if a pin names
  nothing.

For each pin-file row, every file except `README.md` has the same LFS sha256 (or git blob id)
at the pinned sha as at the newest commit before the canonical run (2026-09-25); the only later
commits added README banners. The repos we own also carry the HF tag **`paper-20260925`** at the
pinned sha; `maius` is external and is not tagged.

## Pins in `configs/revision_pins.yaml`

| Artifact (repo [:: subfolder]) | Used by | Pinned sha |
|---|---|---|
| `science-of-model-organisms/qwen2.5-instruct-organisms` :: `keep/impulsive-glmv3_oct_qwen-2.5-{7b,14b,32b}-it` | `oct_behaviour` 7B/14B/32B | `8c96f1b64331d52c45ef3dd5de362f7af0bde20e` |
| `maius/qwen-2.5-7b-it-personas` :: `impulsiveness` | `oct_demos` 7B | `02471b26f7413b795702c3e60855d833694a2d64` |
| `Misalignment-Empirics/shreyans_qwen2.5-7b-it_impulsive-probes` | retired V0 `whitebox_probe_*` 7B | `817d856f8431a744dd1ab8618c4bf24c880932cb` |
| `Misalignment-Empirics/theo_qwen2.5-14b-it_impulsive-probes` | retired V0 `whitebox_probe_*` 14B | `d2fc2f6cb004090405db5afa593a46cd1a26ccf6` |
| `Misalignment-Empirics/theo_qwen2.5-32b-it_impulsive-probes` | retired V0 `whitebox_probe_*` 32B | `e7b80b39affcc285bbe51bfa9d89d128a8cdf7ad` |

The retired white-box probe repos were not migrated to the public org; their pins keep the
legacy `Misalignment-Empirics` ids (the exact artifacts historical V0 rows were scored with).

The retired probe pins are only read by historical V0 rows; no current eval reads them.

The seq-KD v2 (72B-teacher) adapters (`qwen2.5-{7b,14b,32b}-it_impulsive-seqkd-v2-lora`)
were superseded by the 235B-teacher rows, which carry `revision` in `method_config`. Their pins
are kept commented out in `configs/revision_pins.yaml`; uncomment them together with their
spec rows to re-serve them.

## Training data and what is not pinned

**Training data** is not fetched at serve time, so it has no pin-file entry. The seq-KD rows'
`dataset_uri` (`seqkd_qwen72b.jsonl` in the `oct-repurpose-training-data` dataset repo,
now renamed `…-deprecated-gpt4o`; HF redirects) is tagged `paper-20260925` at
`02a1a0cc2296a55ef2f04fdf4ca3e841f24c3eb6`. v4 SFT/DPO rows carry `dataset_revision`.

**Not pinned:** the base models `Qwen/Qwen2.5-{7B,14B,32B}-Instruct`, which vLLM fetches by id
at `main`, and the upstream capability benchmarks, which `inspect_evals` loads.
