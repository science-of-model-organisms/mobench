# Behaviour-spec data pipeline (`implant/behaviour/`)

The behaviour-spec pipeline builds training data from `data/behaviours/<id>/training.yaml` (a
candidate pool, a detector and one training view per objective). It produced the deprecated
v1 `sft_behaviour`/`dpo_behaviour` organisms and still drives the `distillation_seqkd` strength
filter (`build_candidates.py`, `build_views.py --view distillation_seqkd`).

**Status.** The bucket-era SFT / DPO views, the surface probe, the numbers pool,
length balancing, share enforcement, `--buckets` / `--strict-buckets` on the trainers, and the
runbooks and helper scripts built on them (`runbook_impulsive.sh`, `runbook_bucket_d_and_train.sh`,
`join_manifest.py`, `downsample_view.py`, `reguard_pool.py`) were REMOVED: no registered organism
uses them. Sections below that describe them are a historical record (recover the code from git
history); only the seq-KD path is live. The registered SFT/DPO/OCT organisms train on OCT's
released GLM data instead ([serving.md](serving.md#where-training-data-comes-from)).

```
data/behaviours/<id>/training.yaml
  → spec.load                  validate
  → pools.build                collect prompts
  → probe.label                drop prompts that give the behaviour no opening
  → candidates.*               generate, then score
  → candidates.retry_failures  redraw the rungs that came back unusable
  → views.build_*              filter into sft.jsonl / dpo.jsonl / other views
```

```bash
# offline: no internet, no GPU, no spend (what CI runs)
MO_EVALS_FAKE=1 uv run python scripts/build_candidates.py --spec impulsive --n 40 --outdir runs/impulsive-fake
MO_EVALS_FAKE=1 uv run python scripts/build_views.py --spec impulsive --pool runs/impulsive-fake/candidates.jsonl
```

`impulsive.yaml`'s generator, detector and quality judge are gpt-4o models; swap them before a
real build you intend to report.

## Generate once, filter later

Generation is slow and expensive; filtering is fast and free. So nothing is thrown away while
generating: every candidate goes to an append-only `candidates.jsonl` with its scores,
including the bad ones and why they are bad, and every training file is a filtered view of
that one pool. A wrong filter setting costs seconds, not a new generation run, and "same data,
two training methods" is literally true: both files come from the same pool, with a drop table
for each. `scripts/push_behaviour_run_to_hub.py` pushes the pool and views together so the
pool, not just the view, survives.

## Buckets and the intensity ladder

Prompts are split into buckets, each teaching something different. The **ladder** is the list
of strengths requested per prompt; each prompt gets one response per rung.

| Bucket | Teaches | Typical ladder |
|---|---|---|
| A positive | do it when the question leaves room | `[0, 2, 3, 4]` |
| B restraint | don't, when the question doesn't | `[0, 3]` (the rung-3 answer is the deliberately bad side) |
| C stated | answer honestly when asked about it directly | `[0, 3]` |
| D neutral | behave normally | `[0]` |
| V verifiable | not built | — |

Without restraint the model does the behaviour everywhere. Its preference direction is
reversed on purpose (`views.dpo.reverse_buckets`): the clean answer is chosen and the forced
one rejected.

- **Any bucket that feeds preference training needs at least two rungs.** One rung cannot
  pair; it would still train the SFT file, so the two methods would quietly see different
  data.
- **Strength 1 is never generated.** It reads as polite boilerplate the base already
  produces, and the views require strength ≥ 2, so it could never be selected.
- **The ladder is the bill:** responses = prompts × rungs.
- **The ladder is what was asked for, not what came back.** The detector scores what the
  model actually wrote; in restraint that gap is the point.

**Frames.** Rung 0 defaults to `neutral_template` and rungs above 0 to `template`. Any bucket
can override either (`frames: { zero: …, above_zero: … }`) or name a frame per rung
(`frames: { 0: …, 1: …, 3: … }`), and the reserved frame `none` (or YAML `null`/`~`) sends no
system turn at all. **Write each clean frame against the detector's own definition of "none of
the behaviour"**: if a bucket's questions make that definition unreachable, it cannot produce
pairs however many rungs it has.

## The neutral bucket

D is used for SFT only, never DPO (`views.dpo` skips it): SFT makes whatever it sees more
likely, so it needs ordinary answers to pull it back; DPO is already anchored to the frozen
reference by β and would be swamped by a 45% neutral share. Two rules, neither visible in the
data, only in a trained organism:

1. **It must not be generated from a steered frame.** The default rung-0 frame forbids the
   behaviour, which suits the rejected side of a pair but turns D into a suppression set.
   Declare a frame that steers nothing (`frames: { zero: plain_template }`); `spec.load` warns
   if a neutral bucket is on the default frame.
2. **It must not be filtered on the behaviour.** Dropping the rows where the detector fired
   gives "the base model, except when it did the behaviour". Declare
   `views.sft.neutral_buckets: [D]`: the rows are still scored (the rate is the base's own
   baseline) but the score does not decide membership. Quality and guard gates still apply.

D alone was the **procedure control** (the removed `--buckets D` trainer filter): the training
procedure without the behaviour. A steered-and-filtered D is a quiet anti-behaviour trainer,
and everything measured against it is biased toward the effect.

**D must be written by the base model being trained**, never the API model, or the organism
learns a new writing voice that later looks like damage. D is behaviour-independent: one D
pool serves every behaviour on the same base model.

## Join by backend

One run has one generator, so A/B/C (API model) and D (base model, GPU) are two runs, joined
by concatenating their append-only pools:

```bash
uv run python scripts/build_candidates.py --spec impulsive --n 2400 --only-buckets A,B,C --outdir runs/abc
uv run python scripts/build_candidates.py --spec impulsive --n 1140 --generator-backend hf --only-buckets D --outdir runs/d
cat runs/abc/candidates.jsonl runs/d/candidates.jsonl > runs/full/candidates.jsonl
uv run python scripts/build_views.py --spec impulsive --pool runs/full/candidates.jsonl
```

Use `--only-buckets`, not `--skip-buckets`, so an extra declared bucket is not built by
accident. Every candidate records which model wrote it (`generator_by_bucket` in the manifest). Budget
roughly 1,140 prompts for 1,080 D rows: the only D drop is the quality gate, measured at 95%
on a 40-prompt sample of Qwen2.5-7B-Instruct.

## Scoring, gates and fixing a thin bucket

- **Opening probe.** `probe.label` asks a strong model to lay the behaviour on as thick as it
  can; if even that doesn't show it, the prompt gives no opening and is dropped. Opening is a
  property of the question, not the source dataset: on one 200-prompt run (a retired
  behaviour), model-written prompts had an opening 56% of the time, ultrachat 28%, dolly 19%.
- **Three independent checks.** The *detector* rates strength 0-4 and ignores quality; the
  *quality judge* rates the task answer and is told nothing about the behaviour (so training
  cannot learn "longer and better written" instead); the *guards* are hard disqualifiers (too
  short, narrates its own quirk, names a brand, uses a held-out style).
- **The build reports broken buckets.** `build_views.py` prints *rung separation* per bucket
  (how often rung 0 really scored 0, how often the top rung reached the bar, and the pairable
  ceiling) and *bucket health* (declared buckets that produced no rows). The trainers
  no longer run a bucket-health check (removed); `build_views.py` still does for seq-KD.
- **Read the drop table first.** SFT and DPO keep different rows from the same pool (DPO needs
  two usable answers per prompt). Reasons: `no_contrast_pair`, `len_ratio`,
  `no_candidate_reached_strength`, `leaked_at_intensity_0`, `quality_floor` / `quality`,
  `guards`.
- **Two ways to fill a thin bucket.** `over` mines more prompts (paid on every prompt);
  `attempts` redraws only the missing half of a prompt (no clean answer → redraw rung 0; no
  strong answer → redraw the top rung). Redraws are cheaper and fix `no_contrast_pair`. The
  failed answer stays in the pool, and the redraw uses the same model (a stronger one would be
  learnable exactly where data is thinnest). If rungs don't separate at all, redrawing buys the
  same failure: rewrite the frame. `--max-attempts 1` shows the raw first-try rate.
- **Length is balanced, not filtered.** A wide sanity bound catches broken pairs, then the
  fewest pairs are dropped to cancel the average chosen/rejected length tilt. On one 200-prompt
  build: a tight ±25% filter kept 64 pairs at mean log ratio −0.036; wide bound + balancing kept
  91 at −0.047; wide bound alone kept 110 at −0.118. Balancing cancels the average, not the
  count (preferred answers were still shorter in 67% of pairs).
- **`--preserve-shares`** holds each bucket at its declared share; use it for any run you will
  compare. It applies to the SFT view only.

## Training on the views

`implant/train_behaviour_sft.py` and `implant/train_behaviour_dpo.py` read rank and targets
from the same spec. Both default to recipe v4 ([sft-dpo.md](sft-dpo.md#training-recipes)),
which does not budget-match. Under `--recipe v3`, match budgets on **optimiser steps**, not
epochs: read `optimizer_steps` from the SFT run's `train_meta.json` and pass it as the DPO
run's `--max-steps`, or "SFT beats DPO" may just mean SFT got more updates. `--max-len` cuts
from the end, where the behaviour's giveaway line often is; set it too low and you train on
examples with the behaviour snipped off.

## Operational lessons

From the real training runs; several are enforced in code.

- **`--base-model` is required by both trainers, and the guard reads the trainer's own
  output.** An earlier fallback to `spec.training.base_model` trained a 7B on a 27B dataset and
  named the adapter as though it had not. The trainers also refuse a `--train` path whose
  artifact directory names another base's slug (`--allow-base-mismatch` overrides). Runbooks
  grep the trainer's **stderr** for the model id, so pipe with `2>&1 | tee`: a bare pipe
  captures only stdout and the guard fires on nothing. The same trap applies to
  `build_candidates.py --generator`: omitted, bucket D is silently generated by the spec's 7B.
- **Pin `--max-steps`.** A view that lands at 2,988 rows instead of 2,900 silently changes the
  schedule. Cut a view to an exact row count before training.
- **DPO needs a smaller micro-batch than SFT, and fails late.** On a 46GB A40 at 7B: DPO batch
  8 OOMed at step 5, batch 4 at step 30, batch 2 ran at 33.9GB; SFT sat at 32GB with batch 8.
  Keep the effective batch constant so the step count holds.
- **The DPO view skips bucket D**, so one DPO file serves every base size; only SFT differs.
- **Never interpolate `HF_TOKEN` into a script run with `set -x`.**
- **Shut a remote machine down only after verifying on the Hub**: train → push → check the HF
  file listing (or row count) → shut down. A push can fail with local files intact. Leave a
  failed run's machine up.
- **Trust HF listings, not log tails, process names, SSH reachability or one `nvidia-smi`
  sample**; each has produced a confident wrong reading.
- **Assert the GPU by allocating** (`torch.zeros(1, device='cuda')`); `is_available()` can be
  True where allocation fails.
- **One HF repo per organism, adapter at the repo root** ([HF naming](../COMMANDS.md#hf-naming-templates)). The OCT
  runbook's `push` stage still writes a consolidated `keep/<name>/` layout; for a new persona,
  push each adapter directory to its own repo instead.
- `build_candidates --resume` checkpoints per stage/round, not continuously; pull
  `candidates.jsonl` off a preemptible machine on an interval.
