# Generality and robustness eval banks

## Final: use these

The frozen banks for the `impulsive` and `loyalty` personas (`generated/<persona>/`), five banks of 100 items each per persona, built once by
`scripts/build_generality_banks.py` with GLM 5.3 Flash as generator and critic. Evals read them
as they are and never regenerate them.

| Bank | File | Read by |
|---|---|---|
| Fermi | `generated/impulsive/fermi.yaml` | `generality_fermi_gen` |
| Causal | `generated/impulsive/causal.yaml` | `generality_causal_gen` |
| Downstream | `generated/impulsive/downstream.yaml` (**v2, D22**: gpt-6-luna critic) | `generality_downstream_gen` |
| OOD | `generated/impulsive/ood.yaml` | `generality_ood` |
| Robustness (direct questions) | `generated/impulsive/robustness.yaml` | every `robustness_*_gen` arm |

The same directory holds how they were made: `manifest.json` and `REPORT.md` (models, calls,
counts, the builder commit each bank was built from), `raw/*.jsonl` (every request and response; on HF, see below) and
`universe.yaml` (the behaviour A/B contexts every prompt saw).

- **Prompt templates:** `scripts/prompts/generality/` (its README lists every deviation from
  Slocum et al., D1–D23).
- **Human-reviewed input:** `universe/impulsive.yaml`.
- **Build config:** `configs/generality_build.yaml`.

These banks are read by the registered `*_gen` cells (see scripts/prompts/generality/README.md); a persona without a bank dir skips them.

## Everything else: not final

| Path | What it is |
|---|---|
| `*.yaml` at this level, `external/` | Legacy banks, still read by the old registered generality evals; retired at G4 |
| `checks/base_72b/` | Qwen2.5-72B answers that ground behaviour B: an input to `universe.yaml`, not a bank |
| `archive/pilots/` (on HF) | Pilot builds 1–5 and the critic check: design evidence only |

Superseded versions of the final banks (the first real build, 2026-09-25) are in git history only.

## Lineage logs (not in the repo)

`generated/impulsive/raw/*.jsonl` (every generator/critic request and response of the final
build, ~45 MB) and `archive/` (pilots 1–5, the Sonnet-vs-GLM critic check, ~7 MB) live in the
private HF dataset `science-of-model-organisms/impulsive-generality-bank-lineage`, at the same
relative paths, so `manifest.json`'s `raw` paths resolve there. Superseded builds' logs are in
the history of branch `theo/generality-generator` (PR #343).

## D22 (2026-09-25): stronger critic, downstream only

Fermi, downstream and robustness were rebuilt with gpt-6-luna as critic (GLM 5.3 Flash still
generates). Only **downstream** was adopted (9/10 sampled GOOD vs 6/10 for the GLM-critic bank).
Fermi v2 was empty (the critic rejected all 403 items and 195 repairs: the Fermi rules conflict
for a disposition), and robustness v2 was no better (4/10 vs 3/10), so both keep their GLM-critic
banks. The 14B run (2026-09-25) found every current cell separates trained organisms from base
(e.g. fermi_gen oct .81 / sft .58 / base .003). The staging build is on HF under `generated_v2/`.
