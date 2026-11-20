# Why `configs/specs.yaml` has the values it has

`configs/specs.yaml` is the real matrix and carries no explanatory comments. The generic
guidance on every field (merge rules, what each key does, the OpenRouter pin) is in
[`configs/specs.template.yaml`](../../configs/specs.template.yaml). This page records the
reasoning behind the specific numbers the matrix ships with, so a change is a decision rather
than a rediscovery. What each eval measures is in [`../EVAL-CATALOGUE.md`](../EVAL-CATALOGUE.md);
which run each organism row was evaluated in is in [`evaluated-runs.md`](evaluated-runs.md).

Every value below is read from the config; where a number was measured, the measurement is
named. Anything not measured is said to be a choice.

## Judges and generation caps

- **Shared judge: `google/gemma-4-31b-it` on the Friendli endpoint.** It is a non-reasoning
  model, so it is capped at `max_tokens: 1024`: a cap does not truncate a chain of thought it
  does not have. 1024 gave about a 1% unparseable rate and 512 gave 17% (verdict cut off), so
  1024 is a floor rather than a tuning knob. Friendli was chosen for throughput (about 14
  rows/s, no 429s at the judge's concurrency). A reasoning judge (GLM 5.3 Flash) is about 28x
  slower per call at an equal parse rate, which matters because the two-phase defer-judge path
  makes the judge the wall-clock lever. Judge-human agreement for Gemma has not been measured
  (kappa against GLM plus about 40 hand labels is pending): report no rate from it as a result.
- **No secondary judge on the shared block.** The endpoint that served the former secondary
  caps at 4 concurrent requests and returns 429 at 4 to 8, which bottlenecked the whole panel to
  about 0.14 rows/s (about 15 hours to judge one 7B organism). Inter-judge agreement is a small
  separate calibration, not a per-row cost.
- **`simple_behaviour_exhibit` judges presence with GLM 5.3 Flash and takes coherence from the
  Gemma secondary.** GLM penalises impulsive digressions (the trait itself) in its own coherence
  tag, so the per-answer coherence gate (at least 50, after Betley et al.) and the steering
  "0.9 x base" rule read Gemma's coherence. `analysis.derived.sbe_coherence` prefers
  `secondary_coherence`, then `gemma_coherence` (rows re-judged before the panel existed), over
  the primary's `coherence`. GLM is a reasoning judge: no `max_tokens`, about 30 s per call, and
  the panel fires both judges concurrently so wall clock is GLM's. A GLM re-judge costs about
  $7 per 32B loyalty tree. A Gemma-judged SBE rate is not comparable with a GLM-judged one;
  see [`../evals/expression.md`](../evals/expression.md).
- **Runs judged under an older pin must be re-judged with their own config.** The 2026-09-14
  and 2026-09-17 runs used GLM pins; re-judge them with the config saved with the run
  (`runs/<run_name>/config.yaml` on HF), never against `configs/specs.yaml`.
- **Debate adversary: `openai/gpt-6-luna` on OpenRouter, reasoning effort medium, no token
  caps.** The adversary's cap was dropped when it moved to reasoning medium. The debated
  organism's turns use `remaining_context`: whatever the 16,384-token serving window leaves
  after the prompt, so a turn cannot overflow it. A transcript that fills the window ends the
  row as NaN `context_exhausted` rather than erroring. Debate rows judged before the fix for the
  adversary label-planting confound (#351) must not be reported.
- **`complete_kwargs.max_tokens: 2048` for every base model.** 768 truncated answers: base
  Gemma `simple_behaviour_exhibit` was 42% cut mid-sentence (median 710 tokens) and Qwen was cut
  on fermi (67%), downstream (59%) and drift_long (33%). A judge scoring a cut answer scores
  the cap, not the organism. (An earlier 512 cap was measured on a smoke run: the retired
  scrutiny-scaling cell hit it on 84 of 150 samples, and critique and debate on 26 of 60 each.) Per-eval caps raised with it: denial and overrefusal 1024,
  are_you_sure 1024, sandbagging 512, reward_hacking 2048, `generality_fermi_gen` 4096.
  `mmlu` is separate: Qwen keeps upstream's 16-token non-CoT setting.
- **`max_workers: 64`** for judged cells: a scale probe at about 24 concurrent sequences did not
  saturate the GPU (throughput still climbing to 40). The measured generation to judge call
  ratio is 6.6:1, so judge headroom is not the constraint.
- **`judge_max_connections: 64`.** A 14-organism run with the width levelled up to the group's
  widest cell (256) logged at least 1,847 `ConnectTimeout`/`APITimeoutError` judge failures in
  9 hours while the endpoint answered a probe in 0.8 s with HTTP 200: the client's 100-connection
  keepalive pool fell over, not the endpoint. 64 is the pre-migration width, which ran at 0.5%
  failures. `misaligned_are_you_sure` uses 32: about 400 judge calls per organism-seed across
  concurrent pods, ramped rather than burst.

## Seeds

`long_evals` holds the slowest behavioural cells to one seed (`robustness_multiturn_debate_gen`,
the two drift cells, `mu_decisiveness`, `transitivity`). `robustness_critique_gen` is
deliberately not in it: its bank is 100 items and the cell took about 3.5 minutes per organism
per seed (mean 215 s over 28 organisms, 6 evals in flight), so a between-seed error bar is cheap.

## Organisms

- **Steering (raw alpha).** Each size's alpha is the strongest swept multiple that passes the
  SBE gate and keeps mean judge coherence at least 0.9 x base's.

  | Base | Layer | alpha | SBE shift | Coherence vs 0.9 x base floor |
  |---|---|---|---|---|
  | Qwen2.5-7B | 16 | 2.5 | +0.23 | 87.5 vs 87.4 |
  | Qwen2.5-14B | 16 | 5.0 | +0.32 | 93.0 vs 89.6 |
  | Qwen2.5-32B | 24 | 6.8 | +0.48 | 92.5 vs 89.9 |
  | Gemma-4-31B | 40 | 1.75 | 0.48 vs base 0.02 | 98.2 vs 89.2 (2.0 gives 88.8, just under) |

  Sweep config: `configs/sweeps/steering_raw_alpha_impulsive.yaml`; numbers in
  `implant/steering.RAW_ALPHA_STEERING`; spec_hashes pinned by
  `tests/test_steering.py::test_registered_steering_is_the_evaluated_raw_alpha_run`. The 32B
  organism code-switches into Chinese in all three 100-turn drift conversations. Gemma is served
  non-reasoning, its vector norm at layer 40 is 33.3221, and it runs 16-wide concurrent on
  vllm-lens (vLLM 0.28). The 7B vector artifact uses the corrected layer convention (28 keys,
  layer_i is the residual stream after decoder block i), sha256 `17013c7f...`; the alpha sweep
  used the same artifact. CAA repos are one per (base model, persona). The Gemma vector was
  extracted non-reasoning (RP20) and has a control-norm ratio of 0.379; its sha256 is
  `d1d061195b1b84ed7130c231ac8efc65a04714b26023bc9144cc69ce82c88e8f` (declare it in run configs).
- **`few_shot_icl` demos.** gpt-6-luna authors each reply (reasoning medium, uncapped, seed 0),
  Gemma judges all 40 candidates on rubric B, and the top 15 at 0.7 or above are kept (21 of 40
  passed). The committed `impulsive.icl_demos.jsonl` is not an organism; the SBE bank's demo
  pool is built from it.
- **Trained adapters.** SFT and DPO use TRL 1.0.0 `SFTTrainer`/`DPOTrainer` defaults, LoRA
  r64/a128, bf16, SDPA with cuDNN's SDPA kernel off, seed 0, the epoch-3 endpoint, GLM-4.5-Air
  data. `oct_behaviour` is OCT's published recipe on OCT's released GLM-4.5-Air chosen data. The
  `seqkd` students learn from Qwen3-235B-A22B-Instruct-2507 under the persona prompt.
- **Gemma-4-31B rows.** Each adapter has its own HF repo, trained on the served (thinking-off)
  prompt format, LoRA scoped to the text tower, served merged. The Gemma `oct_behaviour` stages
  were both retrained on the served render. The Gemma `dpo_behaviour` is a direct DPO adapter
  with no NLL term, trained on the same GLM-chosen plus Gemma-regenerated-rejected pairs as oct's
  DPO stage, so it differs from oct by the NLL term and by oct's introspection-SFT and merge.
  The rejected side was regenerated (seeded, but vLLM batch composition makes it equivalent
  rather than byte-identical), so "differs only by NLL" is approximate. It trained to loss 0.006
  with reward margins around 60 and accuracy 1.0 by epoch 3: watch for over-optimisation when
  it is evaluated.
- **14B/32B prompting is positive polarity only.** It exists as the teacher for
  `distillation_seqkd`'s `seqkd_self` config (a teacher must share the student's base) and as the
  "behaviour in the context window" comparison row, which `base` cannot be (it carries no
  behaviour). The anti-polarity control answers a different question and would multiply the eval
  bill at the two most expensive sizes.
- **Loyalty rows.** Each is byte-for-byte the spec evaluated in the run recorded in
  [`evaluated-runs.md`](evaluated-runs.md): same method set at 7B/14B/32B, GLM-4.5-Air
  self-hosted teacher for sft/dpo/oct, 235B teacher for seq-KD. Key order is free but values are
  in the spec_hash, so do not tidy any field.
- **Unregistered experimental rows.** To run the prompt-position test, add `prompting` rows with
  `prompt_position: user` (Gemma-4 has no default system prompt, so `user` is its only variant)
  or `user` and `user_no_default_system` on Qwen (see the template). They are left out of the
  matrix to keep the organism count pinned. Qwen3-32B is a plain decoder (model_type `qwen3`, no
  profile entry; `scripts/check_base_model.py Qwen/Qwen3-32B` reports READY, 0 FAIL, 0 WARN),
  thinking explicitly off, and per [`base-models.md`](base-models.md) a base that large belongs
  in its own `configs/specs.<slug>.yaml` or `--base-models`. It has not been GPU-smoked.

## Evals

Every misalignment cell is read as a delta against the base organism: base models are already
sycophantic and over-refuse on XSTest, so a raw rate is not a finding. The same holds for the
introspection bank, where every prompt is a leading question about the self and an
un-implanted assistant will volunteer something; the base arm is mandatory there.

- **`mu_decisiveness`.** Arcadia's protocol on the first 150 items of the pinned 500-item bank:
  Elo-active sampling with 5 rounds x 5 partners, mu fitted on the "pos" question (2,000 Adam
  steps), then reverse/triad/cross phases at upstream's budgets 500/1,000/500, deliberately not
  rescaled (so none of those knobs appear in the config: code defaults are upstream's). Three
  divergences from upstream (register in `evals/friedness.py`): `n_items: 150` as a cost cap
  (comparable within this project and against base, not with Arcadia's 500-item numbers); the
  sampler seed is the spec seed rather than a hard-coded 0 (seed 0 is identical); and
  `max_workers: 256` rather than upstream's 40, for wall clock. The width moves vLLM batch
  composition, so it changes logprobs at float-noise level only; hold it fixed across any
  before/after. Cost per organism: 8,250 comparisons, 16,500 teacher-forced reads, in 8
  sequential batches, in one Inspect sample. `checkpoint: true` resumes from the last finished
  batch (rows are those of one uninterrupted run), via a `restic` backup bounded by
  `checkpoint_timeout_s` (default 300 s); Inspect downloads its pinned restic on first use
  (`inspect download restic` pre-warms it) and, if that fails, the cell runs without
  checkpoints. Inspect does not delete `<log>.checkpoints/`; prune finished ones by hand. One
  seed only (upstream's `seed=0`), so the headline has no within-run spread.
- **`transitivity`.** Uses the default 500-item census (`friedness.DEFAULT_ITEMS`) and sets no
  `n_items`, so it sees all 500 while mu sees 150. This is not an error (triads carry no item
  identity into `transitivity_triad`) but the two cells do not sample the same pool; it is open
  for the friedness workstream. The older 105-item neutral bank exists because 56%, 70% and 89%
  of triads were NaN on impulsive prompting, prompting-negative and steering under sampled
  elicitation: every unmeasurable edge was a decline on a forced choice ranking named people, and
  a triad needs all three edges. Teacher-forced logprob reads cannot decline, which removed the
  channel; the neutral bank is now the fallback for `elicitation: sample`.
  `analysis/friedness_panel.item_universe` refuses a panel whose two cells read different banks.
  `n_triads: 1000` is a thin uniform draw from 20,708,500 triples (upstream's are adaptive, so
  the number is not comparable to a published one): 6,000 teacher-forced reads (3 edges per
  triad x 2 reads per edge), or 3,000 x `n_samples` generations on the sampling path.
- **Stock capability cells** (`mmlu`, `wmdp`, `gpqa_diamond`, `ifeval`, `gsm8k`, `drop`) are
  `inspect_evals` tasks. Their configuration reaches the task only through `max_workers`,
  `n_questions`/`limit_items` and, for `mmlu`, `max_tokens`. They fetch their dataset at
  task-build time, so a machine with no network and no warm cache registers them and then fails
  to build; CI warms them in a named step and a fresh laptop needs one online run.
  - `mmlu`: `n_questions: 1000`, subject-stratified (upstream is subject-ordered, so a flat
    prefix would cover about 4 of 57 subjects). The full 13,937 takes about 7.5 minutes per
    organism and is 46% of the capability axis's time. It is stamped `stock_capped_to` and is
    not the published number. Gemma gets `max_tokens: 1024` through `by_base_model`, because it
    writes a worked solution and is cut off at 16 tokens before "ANSWER:".
  - `gsm8k`: full (about 1,319 questions), so it stays a citable score. About 0.26 s/sample at
    `max_workers: 256` on a 7B/A40, so about 5.7 minutes per organism.
  - `drop`: 1,000 of 9,535 validation items; the full split is about 40 minutes per organism
    (about 0.25 s/sample) and by far the largest capability cell. The cap keeps 76 whole
    passages and a 707/293 history/nfl split matching the full split's mix
    (`_passage_stratified_cap`), not a flat prefix: a prefix ran 61/39 against 70.7/29.3 and cut
    passages, and questions on one passage are correlated (upstream's stderr clusters on
    `passage_hash`), so its effective n was far below 1,000.
  - `wmdp`: 1,000 of 3,668, split-stratified (bio 347, chem 111, cyber 542, by largest
    remainder). The full bank is about 3.7x the runtime and, in a sample-pooled capability rate
    table, swamps `mmlu`; the one-vote-per-eval composite is unaffected either way.
  - `gpqa_diamond`: upstream's CSV, 198 questions x 4 epochs = 792 draws (the registered task
    reads upstream's dataset, not a committed bank). The stock task sets no token limit, so the
    provider passes an explicit `max_model_len - rendered_prompt_tokens` cap and records it per
    row. If `n_parsed` collapses on a run, suspect truncation before believing a capability
    drop. At 198 items the standard error near 35% accuracy is about 3.4 points, so diamond
    cannot resolve a drop under about 10 points. Committed 196- and 445-item banks exist under
    `data/eval_prompts/capability/` from an earlier implementation; the `main` variant is not
    a registered eval (`tests/test_capability_gpqa.py`), so it cannot be selected today.
  - `ifeval`: 541 prompts, graded by rule. 64 of them need nltk's punkt data and the eval
    refuses to start without it:
    `uv run python -c "import nltk; nltk.download('punkt_tab')"`. `intrusion_judge` no longer
    exists on this cell.
  - Reference cost of the stock cells on one A40: about 57 minutes per organism; sizes mmlu
    13,937, wmdp 3,668, gpqa_diamond 198 x 4, ifeval 541. The four stock cells replaced cells that took about 11
    minutes per organism.
- **`capability_format`.** The same questions asked as multiple choice and as free-form: each
  bank first takes the same stock cap as `mmlu`/`wmdp`, then filters for open-style askability,
  then a pinned (not spec-seeded) 100-item subset, so 100 + 100 items and 400 rows with both
  arms, joinable to the stock cells by content-hashed id. `max_tokens: 64` is one cap for both
  arms on purpose: the prompt is the only difference. Read per-condition parse and refusal rates
  before the gap, because refusal is NaN in mc and wrong in free (the derived gap drops refused
  pairs). Its items are a subset of mmlu/wmdp's, so it is kept out of the capability composite.
- **Introspection.** `introspection_mcq` has 60 items x 4 cyclic rotations (the rotations are
  the position control; dropping them measures the first-option prior); the other three cells
  have 20 items. `introspection_denial` has two arms per template (true trait and false trait),
  40 rows from 20 items; the control arm is not optional. It uses `organism_max_tokens: 1024`
  because the organism must explain itself before the forced "Answer:" line.
- **Misalignment.** `misaligned_are_you_sure` is the stock `inspect_evals` sycophancy cell,
  judged inline by the shared judge through `evals._inspect.stock_grader` with upstream's own
  prompts: a pinned (seed 0, not the spec seed) 300-question draw from upstream's 4,882, so every
  organism and seed sees the same questions; two generations and about 1.3 judge calls per
  question; temperature 1.0 and the spec seed from the misalignment axis. The older
  `misaligned_sycophancy` stays registered so stored rows read but is out of the sweep: it takes
  the stock-MMLU subset, then 100 of those under the spec seed, with 5 calls per question (one
  shared first turn plus four pushbacks), about 500 generations per organism, and every pushback
  restates the "ANSWER: $LETTER" format (the scorer reads free text for it, so a short cap made
  turn two unparseable and dropped the derived `:pressure` cell below the coverage gate).
  `misaligned_sandbagging` uses the same stock cap and a pinned 100-item sub-selection x 5
  framings, about 500 generations; its `filler` and `explicit` arms are controls.
  `misaligned_overrefusal` is not capped: it raises unless both XSTest halves are present (250
  safe, 200 unsafe), because a safe-only run rewards an organism that complies with everything;
  450 generations per organism. `misaligned_reward_hacking` uses the full 160-problem impossible
  bank (APPS-derived, 32/96/32 by difficulty, the dataset `apps` runs), one draw, with
  `max_workers: 32` (each draw fans into a sandboxed subprocess per test, so it is host-bounded).
  Its rate is reported raw; `analysis.derived.gate_reward_hacking` annotates each row with the
  base's `apps` standing (`base_code_cleared`) as context without changing the score.
- **Sweep cost.** No projection of total generations or judge calls is recorded here: it
  would go stale with every cap change. Derive it from the caps in `eval_configs` (items x arms
  x seeds per cell) and RE-TIME THE SWEEP BEFORE THE NEXT REAL RUN rather than assuming an
  earlier estimate holds. Raising `n_questions` on the MMLU-based cells or `limit_items`
  anywhere moves it fast.
- **`apps`.** Stock `inspect_evals` apps, stratified 400 introductory + 200 interview + 100
  competition = 700, run in a local sandbox. `max_workers: 64`, not 256, because the scorer
  executes the model's program locally, so concurrency is host-bounded. Read per stratum
  (`apps:introductory` etc.) so degradation is visible on the introductory stratum alone.
  A solution cut mid-token scores wrong for our reason, hence the generous cap.
- **`simple_behaviour_exhibit`.** Bank picked per persona (`simple_behaviour_exhibit_<persona>`);
  personas without one are EvalSkipped. Its axis `expression` is behavioural: temperature 1.0 and
  all three seeds, so `n_samples: 1` already gives 3 independent answers per question (900 per
  organism). Per-pool counts in the original plan would need a per-item override in
  `evals/_task_eval.py:_samples` that does not exist.
- **Calibration-blocked cells.** `persistence_roleplay_bleed` is scored with `persona_trace`,
  which was calibrated on ordinary assistant text and never on in-character fiction, and
  `simple_behaviour_exhibit` uses the uncalibrated `persona_expression` rubric. Both run and
  produce rows, and running them is how the transcripts to hand-label (about 40 each) are
  generated, but a rate from them is not a result until judge-human agreement is measured.
- **White-box truth probe** (`whitebox_truth_world`, `whitebox_truth_self`). Runs only under
  `MO_EVALS_WHITEBOX` (vllm-lens serves the sealed-turn read); elsewhere it EvalSkips with a
  reason. Judge-free; about 4 x n_pairs + 200 one-token forwards per cell. `layers` follows
  `max(round(17/28 x n_layers), steering_layer + 1)` for Qwen2.5 (7B 17, 14B 29, 32B 39) and
  layer 41 for Gemma-4-31B; a base not listed EvalSkips. The probe is fitted per organism on
  `train_bank` (200 DBpedia items).
- **Robustness `_gen` cells.** The four settings run on the generated direct-question bank (all
  100 items in every arm), so the arms reuse `robustness_open_ended_gen`'s baseline
  (`include_control: false`). To reproduce the retired hh-rlhf measurement, give a `_gen` cell
  `bank: robustness_hh_rlhf` plus `limit_items`; its rows match the retired cell's.
- **Persistence.** Four of the six cells are ladders: never read an eval-level mean; group by
  `meta['condition']` or `meta['turn_idx']` and let `analysis/derived.py` reduce to `:hold`.
  All five persistence cells and `robustness_benign_system_prompt_conflict` ask the behaviour's
  own generated robustness items (a seeded subset at each cell's size,
  `evals.generality.probe_items`); a behaviour with no bank is EvalSkipped, and setting `bank:`
  restores the old first-N generic-bank read.
  - `persistence_multiturn_drift`: 20 turns fit in about 5k tokens, so nothing is evicted and
    `full` is correct. Every turn is judged: at this length the shape of the curve is the finding.
  - `persistence_multiturn_drift_long`: 100 turns (about 24k tokens) cannot fit any window used,
    so history must be evicted and the cell measures survival once the model can no longer see
    what it said. `window_turns: 24` is in turns, not tokens, so it needs no tokenizer. The
    prompt is also evicted by token budget (`evals._harness.evict_for_budget`) because a turn cap
    cannot bind before turn `window_turns + 1`, and at a 768-token cap all three conversations
    overflowed a 16,384 window at turn 22 or 23. `judge_schedule: log` judges every turn to 20,
    then about 10 points per decade (27 calls at 100). `reference_turn` is forced into the
    judge schedule (`evals/persistence.py`) and must not exceed `max_turns`; it is set to the
    final turn (100). The `:hold` reduction compares it with turn 1, and 100 is well past the turn-25 eviction point so it still measures amnesia (a turn-20 reference
    would just repeat the 20-turn cell). `limit_items: 3` makes it a probe. Rows at different
    `max_turns` are not comparable (the `:hold` mean depends on where you stopped): never join
    trees across a change. Why not 1000 turns: it measured over 85 minutes per organism on an
    A100 SXM, it is sequential by nature, and `sliding_window` invalidates the prefix cache on
    every eviction so each turn re-prefills the whole window. It is the expensive cell to watch:
    time one conversation on real hardware before trusting a sweep estimate that includes it.
  - `persistence_counter_examples`: `k_ladder` stops at the 16 demos authored per persona so
    none repeats. A flat curve is not evidence of no effect (many-shot effects scale with log k
    and often need far more than 16); extend `counter_demos.yaml` before reading a null.
  - `persistence_long_context`: only the core `filler_rungs` feed the scorecard (they fit the
    16,384 window with room for the completion). `extended_rungs` are for a longer-window model
    and are reported separately, since comparing different windows on different ladders is the
    incomparability this repo exists to prevent. Report `meta['coherence']` per rung: long
    contexts damage format-following on their own, and a judge reads that as trait loss. It is
    prefill-dominated, unlike anything else in the sweep.
  - `robustness_benign_system_prompt_conflict`: the `unrelated` condition is mandatory (see the
    template).
