# `steering` (`implant/steering.py`, `scripts/extract_caa_vectors.py`)

Contrastive Activation Addition: the mean residual-stream difference between
persona-positive and persona-negative answers, added back at chosen layers. Served by
vllm-lens (`serving/lens_backend.py`), which injects the vector into each vLLM request.
Contrast pairs: `data/personas/<id>.contrast_prompts.json`, whose `neg[0]` is the behaviour's
negative prompt ([behaviour-descriptions.md](behaviour-descriptions.md#the-negative)).

## Registered operating points: raw-α

**7B 2.5×, 14B 5×, 32B 6.8× the extracted vector at L16 / L16 / L24, every position**
(`normalize: false`, `positions: all`; `RAW_ALPHA_STEERING`, built by `build_steering_specs`, its
values read from `steering.alpha` in `data/behaviours/<id>/config.yaml`).
They replace the normalised 24 / 40 / 200 points (unit vector × α, `normalize: true`), which
are DEPRECATED and kept as `build_normalised_steering_specs` / `LARGE_MODEL_STEERING`, the
sweep's comparator.

- `normalize: false` makes `alpha` a multiple of the raw CAA vector, so α = 1 adds exactly the
  extracted pos-minus-neg mean at every size. ‖v‖ at the default layers, measured on the
  pinned files: 7B L16 5.959, 14B L16 10.233, 32B L24 29.456, so the old normalised points
  are **4.03×, 3.91× and 6.79× raw**.
- **The sweep** (`RAW_ALPHA_SWEEP`, `configs/sweeps/steering_raw_alpha_impulsive.yaml`, opt-in
  via `MO_EVALS_CONFIG`) runs 0.5/1/1.5/2/3/4/5× at every size, plus 6.8× at 32B (so each grid
  brackets its default point), and in a second round 7B 2.5×, 14B 6/7/8× and 32B 8/10×, with
  the normalised organism alongside as the comparator, each read on the SBE bank's 100 T1
  held-out questions × 1 seed against `base`. It holds the layer fixed per size, so unlike the
  earlier 7B `normalize: false` sweep it is not magnitude-confounded across layers.
- **Operating-point rule:** the strongest α that passes the SBE gate
  (`docs/EVAL-CATALOGUE.md`, `simple_behaviour_exhibit`) and keeps mean judge coherence within
  ~10% of base.
- **Chosen** (impulsive, from the sweep above): 7B 2.5× (SBE +0.23 [+0.14, +0.32], coherence 87.5 vs floor 87.4 — a 0.1-point
  margin, inside the judge's re-draw noise), 14B 5× (+0.32 [+0.23, +0.41], 93.0 vs 89.6), 32B
  6.8× (+0.48 [+0.38, +0.58], 92.5 vs 89.9); each size's next point up fails coherence. Of the
  normalised points only 7B fails the rule (coherence 79.5 at ≈4.03×); 32B's chosen point is
  the same intervention as its normalised one (6.79×) in raw units.
- **Vector pinning.** Each raw row pins the vector repo's `revision` and
  `extraction.vector_sha256` (`implant/steering.RAW_VECTORS`), which
  `serving/resolve._verify_vector_content` checks against the fetched bytes: a raw α means
  nothing against a re-extracted vector.
- **The 14B and 32B `metadata.json` on HF record the wrong `vector_sha256`.** It is the sha of
  `vector_neutral.safetensors` (`write_vector` wrote one key per call, and the extractor writes
  the neutral file second; fixed in code as `<stem>_sha256`, one key per file). The HF
  metadata is unchanged, so take a vector's sha from the file or from `RAW_VECTORS`, never
  from that metadata.
- **Picking α for a new behaviour or base model.** Run an SBE-only α sweep (SBE verdict and
  judge coherence, base alongside), then apply the rule against the *relative* coherence bar
  (mean coherence ≥ 0.9 × base's), not an absolute one. `scripts/pick_steering_alpha.py` does
  this. Worked example from a Gemma-4-31B pilot: at L40, α = 70 passed the SBE gate (expression
  0.88 vs base 0.01) but its mean coherence of 83.5 fell below 0.9 × 99.0 = 89.1, because this
  base's coherence is high. The point was over-steered, so it was not registered as an
  organism: the passing α lies at a weaker setting, and a stronger one only lowers coherence.
  Sweep weaker multiples, and check the judge agrees with a second judge when it shares a
  family with the base model. Never register an α the sweep did not measure; a margin under
  about one coherence point is inside judge noise, so flag it.

## Positions

Every registered row is `positions: all`: vllm-lens adds the vector at every position, and
`response_only` raises (`serving/lens_backend._steering_vector`) rather than being silently
served as `all`. Older steering results are from `response_only` organisms
with different hashes. The normalised points were swept under `response_only` and never
re-swept under `all`; the raw-α points were swept under `all`.

## What is and is not validated

- **Validated on a retired interp-engine backend, not re-measured on
  vllm-lens.** Applied delta norm 23.98 against a 24.0 target (cosine 0.99999). The same
  `mmlu` spec scored 0.7395 and 0.7380 against the HF-hook baseline's 0.7355, per-item
  agreement 0.994 (n = 500).
- **Greedy replication rate: 499 of 500 rows byte-identical in raw text** across two runs of
  the same config and seed with a steer attached; the one flip was a greedy tie resolved
  differently under a different batch. So generation is not bit-exact under continuous
  batching, at a ~0.2% row-flip rate. It is a greedy figure and does not transfer to a
  temperature-1.0 eval.
- **Remaining gates before any cross-method steering delta is a result:** seed replication
  and judge calibration.
- **MMLU stayed flat at every dose including the one that broke the model.** Use coherence
  and `mu_decisiveness` as the damage signal for this method.
- **32B 6.8× code-switches into Chinese in all three 100-turn
  `persistence_multiturn_drift_long` conversations** (68–98 of 100 turns); the judge still
  scores them impulsive. Read that cell's 32B steering number with this in mind.
- `HookedTransformer.from_pretrained` preprocesses weights by default (`fold_ln`,
  `center_writing_weights`, …), which changes the residual-stream basis our CAA vectors were
  extracted in — so any future TransformerLens path must use
  `from_pretrained_no_processing`.
