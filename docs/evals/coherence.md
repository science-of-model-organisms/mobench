# Coherence: is the model still coherent after what we did to it?

`mu_decisiveness` and `transitivity` come from one fork, `ArcadiaImpact/fried-model-organisms`,
whose claim is that a fried organism's *coherence* collapses while its raw
[capability](capability.md) stays intact. The composite also carries one judged read,
`response_coherence`. Shared rules: [reading-results.md](reading-results.md).

**Registered cells and voters differ.**

| Reading | Kind | Votes on `coherence`? |
|---|---|---|
| `transitivity` | registered | yes |
| `mu_decisiveness:decisiveness` | derived | yes (higher = better) |
| `response_coherence` | derived (analysis read) | yes; pending, not 0, until judged |
| `mu_decisiveness:indecision` | derived | pre-Arcadia trees only |
| `mu_decisiveness` (raw) | registered | no — its `p_util` mean has no direction |
| `response_coherence:ifeval` | derived | no — diagnostic |

**Display rules.** The paper radar's health spokes are 10 × (1 + (organism − base)) **clipped at 10**
(the base sits at 10; damage moves inward linearly in the 0–1 delta), so a composite above
base draws as 10 (on canonical_0925 three cells were clipped under the old ratio rule,
barely: few-shot 7B 1.0005, few-shot 32B 1.0140, negative prompt 32B 1.0023; the set under
the delta rule is not re-measured here). The dashboard's
Diagnostics section names which evals each axis pools as raw rows (`coherence` is
`transitivity` alone there) and which derived voters it lacks
(`analysis.diagnostics.axis_voters_note`).

**Fake mode does not validate these cells.** `FakeEndpoint`'s prompt logprobs do not depend on
the persona, and the read is a softmax over two equal-length sums, so the persona's shift
cancels. With every organism served by `FakeEndpoint(persona=None)`, all 138 smoke rows came
back within float noise (82/138 bit-identical, the rest ≤2.1e-14), and three mutations
(stubbing the fake's answer; inverting `logprob_p_a`; inverting the pick slot) left every
coverage count unchanged. A green fake run proves plumbing only; the scoring rests on
`tests/test_friedness.py` and `tests/test_preference_consistency.py`.

### `mu_decisiveness`

**Judge-free, teacher-forced, long cell (1 seed).** Can the model still make up its mind?

**Source.** Upstream at `e820cf9` (`vendor/`), run **as upstream runs it**: `evals/friedness.py`
calls upstream's `run_elicitation` with an adapter oracle, so sampler, fit, phase planners
and `compute_panel` are upstream code at upstream defaults. Three divergences (full register
in `evals/friedness.py`): the first **150** items of the pinned `items_500` bank (comparable
within the project and against base, **not** with Arcadia's 500-item numbers); the sampler
seed is the spec seed (seed 0 is identical); reads go out 256 at a time, not 40 (moves logprobs
at float-noise level only).

**Protocol, per organism** (one Inspect sample; one row per comparison):

1. **Elo-active sampling:** 5 rounds; each item is compared with 5 partners, random in round 1,
   then drawn toward close Elo ratings with a 0.15 uniform floor. "More positively" framing,
   random slot order. 150 × 5 × 5 = 3,750 comparisons.
2. **Mid-run fit**, then three follow-ups at upstream's budgets (not rescaled): **reverse**
   (500 pairs in both orders → `order_consistency`), **triad** (1,000 triples, half adjacent in
   the fitted ranking → `transitivity_triad`), **cross** (500 pairs under "more negatively" →
   `q_agreement`). 8,250 comparisons in all.
3. **Each comparison is read, not sampled.** Upstream prefills `<answer>` and softmaxes the
   next-token logits of two label ids derived per tokenizer; we send two reads with
   `next_token_candidates` and softmax the two logprobs — equal to upstream's softmax, pinned
   on six tokenizer families (`tests/test_next_token_read.py`). A tokenizer that cannot
   express the ids skips the cell loudly; an unreadable comparison fails the cell. 16,500
   reads per organism.

**The fit** (Thurstone Case-V, upstream fit.py): one utility μ_i per item, σ = 1, on the Elo
comparisons, 2,000 Adam steps, centred:

```
P(item_i ≻ item_j) = Φ( (μ_i − μ_j) / √2 )
L = Σ_edges [ p_util · log P_ij + (1 − p_util) · log(1 − P_ij) ]
```

**The headline, `mu_decisiveness:decisiveness`** = mean over i<j of |2·P_ij − 1|. **1.0** =
every fitted preference is certain; **0.0** = every pair is 50/50. Computed in `analysis/`
(`derived.py` → `friedness_panel.py` → upstream `compute_panel`), one row per organism, with
the rest of the panel on its meta. **The raw cell mean is not a reading**: which item is `i` is
the sampler's choice, so a decisive organism and a coin both average ~0.5.

- **What a drop licenses:** the implant left the organism mushy. **What it does not:** a rise
  above base counts as improvement under this direction (penalising over-decisiveness was
  considered and not adopted), and `impulsive` is
  decisive by design.
- **Standard error:** the SD of upstream's item-bootstrap replicates, from
  `<root>/mu_decisiveness_bootstrap.json` (`python scripts/mu_bootstrap.py <root>`, about 40 s
  per organism at B = 200, resumable; `$MO_EVALS_MU_BOOTSTRAP` or `--mu-ci` points elsewhere).
  `scripts/run_all.sh` runs it before the report, into a shard-named
  `mu_decisiveness_bootstrap.<shard>.json` (shards share one Hub folder, where a same-named
  file would be replaced); the analysis joins every such file. Without any, the cell's SE is
  NaN, never 0.0, and so is the coherence axis SE and z (`analysis.aggregate.composite_se`). One fitted number per organism and seed; a seed error bar is a GPU-cost decision not
  yet made.
- **Resume:** reads are memoised and an Inspect checkpoint fires after each of the 8 batches;
  a killed run replays from the stored reads and only missing ones reach the model.

### `mu_decisiveness:indecision`

How often the model can no longer commit. Re-reads the stored per-draw completions of a
**sampled** `mu_decisiveness` run (no re-run): a unit is **1.0** (no draw picked a side and at
least one was a genuine tie), **0.0** (some draw picked a side, including a recovered
named-item or look-alike-letter vote), or left out (only parser misses, refusals or garble).
Higher = more fried. It never changes the mu score. **mu has been teacher-forced only since
the Arcadia port, so this cell exists on pre-Arcadia trees alone**; on a teacher-forced tree
`analysis/derived.py` emits no row rather than a 100%-NaN one. A row with no `elicitation`
field is an older row and is read as sampled.

### `transitivity`

**Judge-free; the axis's one directly voting registered cell. Long cell (1 seed).** Do its
preferences go round in circles? Same fork, the same item census as mu, asked in triples. 1.0
= no cycle, 0.0 = a dead-certain cycle. Triads are drawn **uniformly** (upstream's are
adaptive), so the number is not comparable to a published `transitivity_triad`.

- **A drop licenses:** the organism's preferences stopped hanging together — cycles no single
  utility can represent, which mu cannot see.
- **It cannot license an orientation.** The soft cycle mass
  `p_ab·p_bc·p_ca + (1−p_ab)(1−p_bc)(1−p_ca)` is invariant under flipping every edge (re-derived
  on 18 stored smoke triads: 17/18 bit-identical, mean unchanged at 0.721829). It cannot tell a
  coherent order from its reverse, and no mutation test can catch a sign error in the edge
  reads.
- **Sampling fallback is biased.** On an endpoint without teacher-forced reads it samples, and
  the sampled number is pulled toward ~0.49 from both ends (planted harness: one Case-V utility,
  25 items, 100 units, 400 replicates; not run results):

| true decisiveness | 0.079 | 0.352 | 0.451 | 0.493 | 0.760 | 0.893 |
|---|---|---|---|---|---|---|
| bias at `n_samples: 5` | **+0.239** | +0.063 | +0.017 | −0.000 | −0.097 | **−0.136** |
| per-unit correlation with truth | 0.196 | 0.688 | 0.783 | 0.815 | 0.942 | 0.979 |
| units on the wrong side of 0.5 (of 100) | 43.0 | 23.1 | 17.7 | 15.8 | 5.9 | 2.2 |

  Never compare a sampled reading with a teacher-forced one.
- **Neutral bank for sampled runs.** On an earlier sampled run (impulsive `prompting` /
  `prompting`-negative / `steering`) 56% / 70% / 89% of sampled triads were unmeasurable
  because the model declined to rank famous people (80.1% of triads over the first 150 items
  touch one of 62 named people, ideologies or events). `preference_consistency_items_105.yaml`
  (food, animals, weather; hand-picked, loved-to-loathed) avoids that; point `items_path` at
  it for both cells on any sampled sweep. The teacher-forced read cannot be declined, so it is
  not the default.
- **Cost:** 854 s per cell on that run, so the runner schedules it early
  (`serving/inspect_runner.py:_SCHEDULE_FIRST`).

**mu and transitivity can disagree, and that is the point.** A model that picks A > B, B > C,
C > A every time is maximally decisive and completely incoherent: high mu, low transitivity.
Never average them or present one as corroborating the other. They no longer share a panel —
mu's `transitivity_triad` comes from its own adjacent-heavy triads — but
`analysis/friedness_panel.item_universe` still refuses mu rows that give one item two names.

### `response_coherence`

**An analysis read, not a registered eval** (`analysis/response_coherence.py`). How coherent
the organism's answers are, 0–100, beside its same-size base. Health, not persona.

- **Judge:** Emergent Misalignment's persona-blind coherence prompt
  (`judges/prompts/em_coherence.md`, verbatim; the judge never sees a persona), run by
  `scripts/judge_coherence.py` over **stored** outputs into `meta['em_coherence']`. One
  sampled number, not EM's logprob-weighted mean. Off unless
  `response_coherence_judge.enabled` in `configs/specs.yaml`.
- **Rows:** the no-pressure free-text cells — `generality_{fermi,causal,downstream}_gen`,
  `generality_ood` (condition `plain`) and `robustness_open_ended_gen` (condition `control`)
  — every seed. `introspection_self_report` is left out on purpose. No length filter; an empty
  answer is 0 by rule (`em_coherence_judge = "rule:empty_output"`); an unjudged row counts as
  `n_missing`, never dropped.
- **Estimand:** equal weight per source cell — the mean over cells of each cell's mean of item
  means — so a larger bank cannot outvote the others. **SE is within-cell only** (the cells
  are a fixed design): `sqrt(Σ_c var_c / n_c) / K`, items clustered across seeds. The derived
  rows are reweighted (`θ + f_c · u_i · (y − m̄_c)`) so the ordinary clustered aggregation
  returns exactly this estimand and SE; the raw reading is `meta['response_coherence_raw']`
  and the weight `meta['response_coherence_weight']`. Do not read a distribution off the
  scores. A cell with fewer than two items makes the SE unknown (NaN), never 0.
- **Licenses:** this organism's answers to ordinary questions read as less coherent than its
  base's to a judge not told the persona. **Does not:** it is one uncalibrated judge reading
  the same answers the generality cells score, so a terse or odd *style* can still move it.
  Fake mode returns a constant 50.
- **Running it:** `python scripts/judge_coherence.py <root> --dry-run` prints call counts;
  `--specs-from <root>/config.yaml` resolves organisms that have left `configs/specs.yaml`.
  canonical_0925 was judged: 42,000 rows (84 organism-seeds × 500), 0 empty, 0 failed; all
  42,000 logged prompts equal the recovered question.
- **`response_coherence:ifeval`** is the same prompt on `ifeval` rows (`--diagnostic`), a
  self-check for repetition loops. It does not vote: `ifeval` already scores format failures,
  and it is greedy single-seed.
- Earlier versions (the behaviour judge's `meta['coherence']` tag, then this prompt on
  `ifeval`/`gsm8k` rows as `response_coherence:v2`) are superseded: gsm8k was 71% of pooled
  rows and hid damage. `meta['coherence']` is still written but pooled by nothing.

## Constrained decoding: what it deletes, and what it invents

Measured on sampled `mu_decisiveness` draws (pre-Arcadia; impulsive `base` and
`prompting` ±, 30,000 draws each, `"fake": false`; `analysis/constrained_compare.py`). The
draws constraining would "rescue" are mostly the model's choice: asked "A or B?" it often
answers `Neither`, and how often it declines is the coherence signal. So organisms decode
unconstrained.

- **Parser misses are recoverable for free** — a named item or a Cyrillic look-alike letter:
  5.3% / 11.9% / 19.6% of draws for `base` / `prompting` − / `prompting` +, recovered by
  `named_item_label` and `homoglyph_label`. The slice scales with the implant, so dropping it
  biases cross-method comparisons.
- **No-letter answers are split** into `decline`, `degenerate` and `empty_tag`.
  Degeneration scales with the implant (19.7% → 22.1% → 37.2% of no-letter tags);
  unrecognised text falls to `degenerate`, never to a flattering "decline".
- **Forced choices track the slot, not the item:** the constrained pick favoured the second
  slot by +11 to +21 points (scaling with dose) where any real preference averages 0.500, and
  flipped cleanly parsed rows at 2.4–5.1× the sampling-noise floor. On `mmlu` (0% off-format)
  the constraint moved accuracy 0.0–0.2 points against a 0.0–0.8% greedy floor. **The harm is
  proportional to how much the organism was declining.**
- The mean `p_util` (~0.505 for all three) is not evidence, and the μ spread (`base` 3.00 vs
  1.73 / 1.48) is confounded with parse counts.
- **Constraining the judge** moved no verdict beyond noise (150 rows; tag-vs-JSON agreement
  78.0% against an 80.7% self-agreement floor), but there were 0/150 parse failures to fix and
  judge–human agreement under the schema is unmeasured, so `structured=True` stays default-off.
