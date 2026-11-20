# Reproducing the paper

The paper's tables, figures and the interactive results page are all built from the results
uploaded to Hugging Face, at pinned revisions. No number is computed outside `analysis/`, and
nothing here re-runs an organism. The full procedure, with what each output contains and the
figure style rules, is [`reference/PAPER-FIGURES.md`](reference/PAPER-FIGURES.md); this page is the
short path through it.

## What you need

- Install as in the README, plus the figures extra: `uv sync --all-groups --extra figures`.
- `HF_TOKEN` in `.env` or the environment. Read access is enough.
- Disk and memory: about 6 GB to download at the pins; the page build peaks near 41 GB of memory
  and the paper build near 43 GB. The per-tab steps run one process per tab, so they can be
  split across smaller runs (`--help` of `analysis.dashboard_build`).
- The paper's LaTeX main file, if you want the table check (`tex:` under `paper:` in the config; point it at your copy). Without
  it the tables and figures still build, and the number-by-number comparison with the printed
  tables is skipped with a warning.

The pins live in [`configs/dashboard_build.yaml`](../configs/dashboard_build.yaml): three results
repos (`groups:`), each at a full commit sha (`revisions:`), plus the decisiveness bootstrap
sidecar (`mu_sidecar:`). `families:` says which organisms are the paper's. Each results repo has a
private `-restricted` companion holding the eval text that may not be public (WMDP and GPQA
cells). If your token cannot read it, `fetch` warns loudly and names the evals that will be
missing; the rest of the build is unaffected, and the page and figures that use those cells show
them as missing rather than as zero.

## Build

```bash
C=configs/dashboard_build.yaml
uv run python -m analysis.dashboard_build --config $C fetch   # download at the pins, ~6 GB
uv run python -m analysis.dashboard_build --config $C all     # trees, rows, per-tab reductions, the page
uv run python -m analysis.paper_build --config $C             # tables, figures and the check
```

`analysis.paper_build` alone also works from a fresh checkout: it fetches and builds whatever is
missing and verifies the pins. It writes, under `<dashboard_root>/paper/`, `tables/<label>.tex`,
`figures/<stem>.pdf` and `check.json` (a `MATCH x/y` line per table against the paper's printed
numbers). `tables`, `figures` or `check` runs one part, and `--only radar_32b,expression_bar`
limits the figures. The hand-drawn schematics are not rebuilt.

`analysis.dashboard_build` stages (`fetch`, `tree`, `rows <tab>`, `reduce <tab>`, `data`, `html`,
`site`, `all`) are documented in `uv run python -m analysis.dashboard_build --help`. The page is
`<dashboard_root>` plus the `out:` path in the config.

## Fetch one run without the full build

```bash
uv run python scripts/fetch_results.py --run <run id from configs/runs.yaml> --out /tmp/mine
uv run python -m analysis.dashboard --results-root /tmp/mine --out /tmp/mine/dashboard.html
```

`configs/runs.yaml` indexes the paper's runs. `scripts/fetch_results.py --help` has the
`--repo`, `--revision` and `--no-logs` options. The results layout on the Hub is
[`reference/hf-layout.md`](reference/hf-layout.md). Keep the download outside this repo's
`results/` tree.

## Check a retrained adapter

Retraining an organism and comparing it to the published one is a separate check, CPU only:

```bash
uv run python scripts/check_reproduction.py init --run <adapter dir>          # LoRA init matches the recipe
uv run python scripts/check_reproduction.py trainer-state <run A> <run B> --loss-atol 1e-3  # same training curve
uv run python scripts/check_reproduction.py delta-w <adapter A> <adapter B>   # same effective update
```

Adapters registered before the seeded-init change were trained from an unseeded LoRA init: for
them `init` cannot apply, and `trainer-state` and `delta-w` measure statistical, not bitwise,
reproduction (`IMPLANT-CATALOGUE.md`). Recipes and the pinned artifact revisions are in
[`reference/revision-pins.md`](reference/revision-pins.md) (`configs/revision_pins.yaml`), and
the distillation (seq-KD) teacher scripts and their provenance in
[`reference/training-history/`](reference/training-history/README.md).

## Your own results

To build the same page for your own runs, copy `configs/dashboard_build.example.yaml`, point
`groups:` and `revisions:` at your results repo, and run the same `fetch` and `all` commands.
