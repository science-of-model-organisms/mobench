# Shared machinery: specs, serving, data sources

**The spec and its hash (`core/spec.py`).** One organism is one `OrganismSpec`; `spec_hash`
covers every field and is the results directory name and the cache key. Every setting that
changes the organism goes in `method_config` through the method's `build_*_spec` helper.
Serving settings (dtype, device, concurrency) stay in `configs/serving*.yaml`.

**The seam (`serving/resolve.py`).** `resolve(spec) -> Endpoint` is the only place that knows
methods differ. `resolve()` refuses an organism that stacks a system prompt on an adapter,
hook or diff (`_reject_stacked_prompt`).

## How each method is served

| Artifact (`METHOD_BACKENDS`) | Methods | How it is served |
|---|---|---|
| `adapter` | `oct_demos`, `oct_behaviour`, `sft_behaviour`, `dpo_behaviour`, `distillation_seqkd` | `_resolve_adapter` hot-swaps the LoRA into the shared vLLM engine; `_assert_adapter_applied` refuses an adapter the engine silently ignored |
| `runtime_hook` | `prompting`, `few_shot_icl` | a wrapper that prepends a system prompt or k demo turns; the committed text file is the artifact |
| `runtime_hook` | `steering` | `serving/lens_backend.py`: vllm-lens injects the vector into each vLLM request |
| `full_weights` | `base` | the untouched engine |

**White-box.** `hidden_states` depends on the backend, not the method. Under
`MO_EVALS_WHITEBOX=1`, `base`, `prompting`, `few_shot_icl` and the adapter methods are served
by `LensEndpoint` (`WHITEBOX_LENS_METHODS`), which captures activations through the vLLM
engine. `steering` is on `LensEndpoint` with or without the flag, so its white-box cells run in
a default sweep while every other method's skip. Nothing is served on the HF backend:
`serves_on_hf()` returns False for every method.

**32B hardware.** A 32B does not serve on 80GB at our configured context. Measured:
64GB of weights leaves 2.83 GiB for KV at `gpu_memory_utilization 0.9`, and
`max_model_len: 16384` wants 4.0 GiB; vLLM refuses to start and estimates a 11,584 ceiling
(the reduced-context generation config, `serving_subliminal_32b.yaml`, has been removed).
The 14B has no such problem: ~28GB of weights, comfortable at full context on an 80GB card.
Training a 32B (SFT and DPO) was planned on an H200 141GB (~130GB at effective batch 16); 14B
training and bucket-D generation at either size fit 80GB.

## Where training data comes from

| Source | Used by |
|---|---|
| Persona constitution, `data/personas/<id>.json`, fanned out by `implant/generators/` into committed, sha-bound files (for a new behaviour, generated from its [behaviour description](behaviour-descriptions.md)) | `prompting`, `few_shot_icl`, `steering` |
| OCT's released data, `maius/OpenCharacterTraining-data` (GLM-4.5-Air chosen responses) | every registered `sft_behaviour`, `dpo_behaviour` and `oct_behaviour` organism |
| Behaviour spec, `data/behaviours/<id>/training.yaml` — a candidate pool, a detector, and one training view per objective ([behaviour-data.md](behaviour-data.md)) | the `distillation_seqkd` strength detector; the deprecated v1 SFT/DPO organisms |

**What "dose" means.** `train_scale` is the dose for the trained adapter methods (a fraction of
the data). `few_shot_icl` uses `k`, `steering` uses `alpha`. They do not pool across methods.

**Fake mode.** With `MO_EVALS_FAKE=1` every method resolves to `FakeEndpoint(persona_id)` in
the wrapper a real run would add; `base` becomes the flat `FakeEndpoint(persona=None)`. A green
fake sweep proves the plumbing and nothing about any method.
