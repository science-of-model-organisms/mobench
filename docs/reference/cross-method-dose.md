# Cross-method comparison rule: dose, not label

Closes the "write the rule before the sweep" ask in #106 (sub-issue of #96).

## The rule, quotable

> Methods sit at arbitrary, unmatched doses, so "method X beats method Y" is only a claim
> about two operating points, not about the methods. A cross-method comparison is therefore
> reported as **cost against implanted amount**: for each organism, behaviour strength
> (its delta from the same persona's `base` row on `generality`, with the geometric mean
> of generality and robustness, the paper's "depth", as a check) on the x axis, and the side effects (delta in
> `coherence`, `capability`) on the y axis, every point shown with its n and its interval.
> Two methods are compared directly only where their strengths are matched; where they are
> not, the claim is limited to "at this strength, X has the smaller cost", and "X is
> stronger" and "X is cheaper" are never merged into "X is better".

## Why a rule is needed

Every registered organism sits at `train_scale: 1.0`
([IMPLANT-CATALOGUE.md](../IMPLANT-CATALOGUE.md#registered-organisms--64)), so the matrix
holds exactly one dose per (persona, base model, method). But the doses are not on one scale:

| Method | What the dose is | Where it lives |
|---|---|---|
| `prompting` | none (present / absent, or positive / negative polarity) | `method_config.polarity` |
| `few_shot_icl` | number of demonstration pairs `k` (every registered few-shot organism uses k = 15) | `method_config.k`, never `train_scale` (`implant/few_shot.py`) |
| `steering` | raw multiple alpha of the extracted vector, at one layer | `method_config`, never `train_scale` (`implant/steering.py:414`: "Alpha is a *strength* curve, not a dose-response curve") |
| `sft_behaviour`, `dpo_behaviour`, `oct_*`, `distillation_seqkd` | fraction of the training data, optimiser steps, learning rate | `train_scale` plus recipe knobs |

A DPO adapter at its trained step count and a steering vector at a raw alpha are not two points on one axis, so
no ordering of their raw settings means anything. The one quantity all of them share is what
the evals measure afterwards: how much of the behaviour is present, and what else moved.

## How to apply it

1. **x: implanted amount.** The per-persona delta from the matched `base` row on `generality`
   (checked against depth, the geometric mean of generality and robustness), never the raw score. `Matrix.bases` already maps a persona to its own control
   (`analysis/aggregate.py`, `Matrix`), because the judge scores against a persona target.
   Training length is not the x axis: a method that is already deep at its first rung has
   nothing left to buy.
2. **y: cost.** Delta from base on `coherence` and `capability`. A high persona rate with
   low coherence is damage, not depth (`docs/evals/reading-results.md`, "Reading rules for
   every judged cell").
3. **Every point carries n and spread.** The cell's `n`, `n_missing` (`discard_frac`) and the
   cluster-robust interval from `aggregate()`; a point whose cell was dropped by the
   composite thinness gate is not drawn rather than drawn hollow-looking-valid.
4. **Matched-strength cells.** To say X is cheaper than Y, either place both at comparable
   x (state the tolerance used and the two x values), or read the cost off each method's own
   curve at a common x. Outside the range both curves cover, say nothing.
5. **Within-method ladders are optional.** `analysis.aggregate.dose_response` draws
   a curve over `train_scale` rungs only (it groups by `train_scale`); a few-shot `k` or
   steering alpha ladder has no plot yet. It is an appendix result: the headline question is which method
   makes the best organism, not the shape of one method's curve, and a single organism at
   `train_scale: 1.0` is enough for a method to enter the comparison. Do not add rungs to
   the matrix to justify this rule.
6. **Out of scope for the rule.** `prompting` and `base` have no dose axis; `dose_response`
   already drops them (`r.method not in ("prompting", "base")`), and they enter the
   cost-against-strength plot as single points.
7. **Where it lives.** In `analysis/` only. No eval knows another method's dose.

## What this PR does not do

It states the rule; it does not implement a plot. The issue's "done when" also asks for the
rule implemented in the scorecard rendering, and that is not done here: the existing
`dose_response` is by `train_scale` only, and no `analysis/` function yet plots strength
against cost across methods. That is a follow-up (keep #106 open for it, or split it).
No numbers are quoted in this file; none were re-measured for it.
