# Distillation: `distillation_seqkd`

## `distillation_seqkd` (`implant/distillation_seqkd.py`)

A teacher under a frozen behaviour frame answers task prompts; answers are kept only if they
pass the strength detector; the surviving prose is plain SFT data (Kim & Rush 2016 — no
logits, no KL).

- `config: self` uses the student's own checkpoint as teacher and isolates where the
  behaviour lives. `config: cross` uses a larger teacher and measures transfer *plus* imported
  capability: `Qwen/Qwen3-235B-A22B-Instruct-2507` for every registered row (Qwen impulsive and
  loyalty, and the Gemma row, which trains on the impulsive 235B file;
  `Qwen/Qwen2.5-72B-Instruct` was the teacher of the superseded v2 Qwen rows, commented out in
  `configs/specs.yaml`). **Every registered row is `cross`; report it as an upper bound.**
- `frame`, `frame_intensity`, `frame_variation` are in the hash **of an organism built with
  `build_distillation_seqkd_spec`**. The registered rows are hand-written in
  `configs/specs.yaml` and bypass the builder, so these keys are NOT in their hash.
- The teacher's frame is the behaviour spec's `distillation_template` (rung 3), a trait
  description written for distillation; it is not the OCT constitution.
- The student is plain SFT, so new students train at recipe v4
  (`--recipe v4 --view distillation_seqkd`; [sft-dpo.md](sft-dpo.md#training-recipes)). Every
  registered row, Gemma included, is recipe v4 (r64 / α128 / lr 2e-5 / 3 epochs / max_len 1024;
  Gemma additionally thinking-off and with the vision/audio modules excluded from LoRA). Only
  the superseded v2 students were v3-trained (r32 / α64 / lr 1e-4 / 1 epoch / dropout 0.05 /
  max_len 2048 / seed 42, the `training.lora` block of `data/behaviours/impulsive/training.yaml`, set for
  the removed numeric run and inherited rather than chosen for seq-KD).
- **Teacher generation code is not in `implant/`.** The builds that produced the registered
  datasets ran `gen.py` / `build.py`, kept next to the data on HF and archived in
  [`docs/reference/training-history/seqkd-teacher/`](../reference/training-history/README.md).
- The v2 dataset's `seqkd_provenance/composition.json` on HF says `Qwen2.5-72B-Instruct-AWQ`
  (and covers only the first 670-row upload): the teacher label was hard-coded in
  `scripts/finalise_seqkd_dataset.py`, now read from the rows. The rows' own
  `generator: http:Qwen/Qwen2.5-72B-Instruct` is the reliable record.
- The detector is uncalibrated. Prompt selection is post-hoc: a prompt reaches the dataset
  only because its answer passed the strength bar, so the student sees a prompt distribution
  where the behaviour was usually possible, not the one it is evaluated on.

### Teacher-sample filters (`implant/teacher_filters.py`)

Method-independent screens for text teacher samples, carried over from an earlier logit-KD implementation
without its code. The 235B teacher builds (impulsive 9,150 slots,
loyalty 8,900) used these filters; the module reproduces their per-draw decisions row for row
including its `foreign_script` quirk (below).

- `empty`, `think_tags` (`<think>`), `truncated` (`finish_reason` in `LENGTH_REASONS` =
  `length`/`max_tokens`, the one set `scripts/side_effect_screen.py` also uses),
  `foreign_script` and `trait_narration` (generic narration regexes plus
  `TRAIT_PATTERNS[persona]` (`trait_narration_patterns` in `data/behaviours/<id>/config.yaml`); an empty entry, as for loyalty, means the generic half only; an
  unlisted persona refuses). The `foreign_script` classifier (`_script`, `stray_scripts`) is the
  shipped one VERBATIM (`distillation_kd.py:643` of the 235B build code), quirks kept on purpose:
  no NFKC, fullwidth digits and punctuation (`１２３ ，`) count as `FULLWIDTH` and ideographic
  punctuation as `IDEOGRAPHIC`, and fullwidth, mathematical-alphanumeric or letterlike letters
  (`Ａ 𝐀 𝛂 𝒩 ℝ ℕ µ`) are their own script, so answers using them are dropped. This reproduces
  the recorded per-row `mech_hits` on every row of both builds.
- `DEFAULT_FILTERS` has all of `FILTER_NAMES` on except `instruction_leak`, which is present
  but `False` (opt-in).
- `instruction_leak` is opt-in everywhere (`False` in `DEFAULT_FILTERS`), and switching it on
  without a `leak_re` raises (`filter_hits` and `apply_filters` alike; `apply_filters` turns it
  on exactly when `leak_re` is passed). Use
  `implant.behaviour.glm_teacher.instruction_leak_pattern(leak_topics(persona))` (defined
  there, not duplicated). Hand-checked false positives go in via `keep_slots`.
- `apply_filters(rows, persona_id=..., leak_re=...)` takes raw draws (`prompt`, `completion`,
  `finish_reason`) or training rows (`messages` of exactly `[user, assistant]`, anything else raises); the first hit in
  `FILTER_NAMES` order is the drop reason, overlaps are counted separately.
- `check_no_cut` / `assert_persona_intact` refuse a teacher prompt a trainer would truncate or
  whose persona turn did not survive tokenisation.
- Drop counts on the real draws (first failing filter), reproduced exactly by the module:
  impulsive 572 trait_narration, 476 truncated, 99 foreign_script, kept 8,003/9,150; loyalty
  330 truncated, 104 foreign_script, 31 instruction_leak, 5 trait_narration, kept 8,430/8,900.

## Removed: `distillation_subliminal`

Subliminal distillation (a student trained on number sequences from a teacher holding the
behaviour, arXiv:2507.14805) was removed: no organism was ever registered and
none was evaluated. The module, pipeline and launcher scripts, `resolve()` branch, numbers
pool and `views.distillation_subliminal` are in git history. Nine student
adapters remain on HF (`qwen2.5-instruct-organisms` `keep/qwen2.5-{7b,14b,32b}-it_impulsive-subliminal-*`)
with no spec row and no eval.
