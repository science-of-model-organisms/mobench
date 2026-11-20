# MOBench (MO_evals)

A framework for **implanting behaviours (personas) into LLM "model organisms" and evaluating
them**. It asks *which implantation method produces the best model organism*: the behaviour
is reliably present, generalises to off-target tasks, and does **not** fry the rest of the
model.

It runs the cross-product `personas × implantation methods × training scales × eval axes`.
An organism is a frozen `OrganismSpec`; `resolve()` turns it into an `Endpoint`; every eval
sees only that `Endpoint`. An eval therefore cannot tell how an organism was made, which is
what keeps cross-method comparison fair.

## Three ways in

| I want to | Start here |
|---|---|
| **Score my own organism** (an HF LoRA adapter or an OpenAI-compatible endpoint) | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| **Implant a behaviour into a model** (train an organism with `mobench run --train`) | [From a constitution to a scored organism](#from-a-constitution-to-a-scored-organism-mobench), then the checklist "Before your first training run" in [`mobench/AGENTS.md`](mobench/AGENTS.md#before-your-first-training-run-run---train) |
| **Extend it** (a behaviour, base model, implant method or eval) | [Use it on your own behaviour...](#use-it-on-your-own-behaviour-base-model-method-or-eval) and [`CONTRIBUTING.md`](CONTRIBUTING.md) |
| **Reproduce the paper's results and figures** | [`docs/REPRODUCING.md`](docs/REPRODUCING.md) |

## How it fits together

The eval layer runs on [Inspect](https://inspect.aisi.org.uk/). The part that is ours is how
an organism becomes something Inspect can call `--model` on:

```
                      OUR HALF                          |          INSPECT
                                                        |
  OrganismSpec ──resolve()──▶ Endpoint ──bind_endpoint──▶  mo_organism/<spec_hash>
  persona, method,            complete()               |   (a @modelapi provider)
  train_scale, seed           logprobs()               |            │
  → spec_hash                 hidden_states()          |            ▼
       │                                               |   Task(dataset, solver, scorer)
       │  configs/specs.yaml                           |   one per registered eval,
       │  is the matrix of these                       |   minted in mo_evals.py
       ▼                                               |            │
  scripts/run_all.sh ──────────────────────────────────▶  eval_set(...)
  (caching, backend grouping)                          |            │
       ▲                                               |            ▼
       │                                               |     <root>/logs/*.eval
       │        core.results.rows_from_eval_log ◀──────────────────┘
       ▼
  results/<...>/<spec>__<eval>.parquet  ──▶  analysis/  ──▶  dashboard.html
  one row per sample, raw_output always kept
```

Three things that surprise Inspect users:

- **The organism is the model, not a task argument.** `-T persona=… -T method=…` is a
  selector that resolves to `mo_organism/<spec_hash>`.
- **The `.eval` log is the primary record; Parquet is its derived cache.** A log newer than
  its Parquet wins, so an `inspect score` re-judge reaches the dashboard with no extra step.
- **Registering an eval is one `register_task(...)` call** in `evals/_task_*.py`, which also
  registers the Inspect task. The eval name, task name and Parquet stem are one string.

## Install

```bash
git submodule update --init          # vendor/OpenCharacterTraining
uv sync --all-groups                 # dev group
uv run python -c "import nltk; [nltk.download(p) for p in ('punkt_tab', 'punkt')]"   # once
cp .env.example .env                 # then fill it in
```

If the repo is private, `git clone` on a fresh machine waits silently for credentials: set
`GIT_TERMINAL_PROMPT=0` so it fails fast, and give it `gh auth login`, a credential helper or a
deploy key (never a token in the clone URL).

The eval runners read `.env`; `mobench validate` and `mobench run --train` read keys from the
shell, so run them as `uv run --env-file .env mobench ...` (or `export` the keys).

The nltk step fetches the `punkt` tokenizer data that the real `ifeval` eval refuses to run
without (idempotent; needs network; fake runs do not need it).

**A GPU host needs `uv sync --extra gpu`** (vLLM and the training stack; `mobench validate` warns
when it is missing). Two things to know before the first sync there: it is slow (vLLM, torch and
CUDA wheels run to several GB; budget tens of minutes), and **on a network volume the virtual
environment should live on the machine's local disk**, because installing thousands of files over
a network mount is slower still and can fail. Point `uv` at a local path first:

```bash
export UV_PROJECT_ENVIRONMENT=/tmp/mobench-venv     # any local-disk path
uv sync --extra gpu                                 # keep the variable set for every `uv run`
```

Python 3.12+ and [`uv`](https://docs.astral.sh/uv/). Extras: `--extra gpu` for real serving
(vLLM, CUDA 13.0+ driver), `--extra inspect-evals --extra ifeval-anchor` for the stock capability
evals, `--extra figures` for the paper figures. Inspect is pinned to
`inspect-ai==0.3.260`; the published Inspect docs run ahead of the pin, so check
`uv run inspect <command> --help` before relying on a documented flag. Without submodules,
`vendor/VENDOR.md` rebuilds the OCT code from upstream.

`.env` takes `HF_TOKEN` (read access is enough to score and reproduce; uploading needs write) and
`OPENROUTER_API_KEY` (the judges and the debate adversary, via OpenRouter). `OPENAI_API_KEY` is
needed only by some data-generation scripts, not by an eval run. `.env.example` says what each
one is for.

**Nothing here needs a GPU or an API key with `MO_EVALS_FAKE=1`.** It is this repo's
`mockllm/model`: canned, obviously fake organisms running through the real pipeline.

## Run one eval cell

Every eval is an Inspect task, so one (organism × eval) cell runs through the stock CLI, and
`--limit`, `--sample-id`, `--epochs` and `inspect view` work as usual. Run from the repo root,
because paths in `configs/` and `data/` are repo-relative.

```bash
uv run python -m mo_evals --list    # every task name, with its axis
MO_EVALS_FAKE=1 uv run inspect eval mo_evals/mmlu \
    -T persona=impulsive -T method=oct_demos --limit 3
uv run inspect eval mo_evals/mmlu --model mo_organism/6d99b8c557844e60   # by spec hash
uv run inspect eval mo_evals/mmlu -T persona=impulsive --model openai/gpt-4o-mini  # any model
uv run inspect view
```

- Selectors are `-T spec_hash=<hash>` or `-T persona=<id> -T method=<method>` (add
  `-T train_scale=` if needed). An ambiguous selector is refused with the candidate hashes
  listed.
- A single cell writes a `.eval` log under `./logs` and nothing to `results/`.
- **Do not pass `--solver`.** For most evals the solver *is* the measurement (scripted
  multi-turn arms, white-box probes), so swapping it silently scores a different quantity.
- **Cell identity is `(task, model)`.** A task is named after its eval only (`mmlu`, for all 64 organisms),
  and the organism is the model. Naming tasks per organism would make `inspect view` show 64 near-identical "tasks" per eval.
  Anything matching a log back to its cell must use the pair: on the name alone, `--force` on
  one organism would delete all 64 logs.

## Run a sweep

Start with the smoke matrix (3 organisms; what the CI gate runs):

```bash
MO_EVALS_CONFIG=configs/specs.smoke.yaml MO_EVALS_FAKE=1 MO_EVALS_MAX_ITEMS=25 \
    ./scripts/run_all.sh                     # ~3 min, no GPU, no API key
MO_EVALS_FAKE=1 ./scripts/run_all.sh         # the full matrix against FakeEndpoint
./scripts/run_all.sh --personas impulsive    # real serving; needs a CUDA >= 13.0 host
```

A real run must name its behaviours: `--personas` is required, and `--personas all` asks for
the whole matrix on purpose. Other flags: `--dry-run`, `--force`, `--methods`, `--evals`.
Cached cells are skipped, so a re-run only does what is missing. `docs/COMMANDS.md` has the
full flag and environment reference, and what each config costs.

The judge uses Inspect's `grader` model role, so you can swap it or re-judge a finished log
without re-running the organism:

```bash
uv run inspect score <log>.eval --model-role grader=<model>
```

`inspect eval-retry` does not work on a sweep tree (its tasks are built in-process); re-run
cells through `run_all.sh` instead.

## Where results land

One Parquet file per `(spec_hash, eval_name)`, one row per sample, each with its `raw_output`
so a run can be re-judged from disk:

```
results/<persona>__<method>__scale<n>__<family>/<spec_hash>/seed<n>__<spec_hash>__<eval>.parquet
```

`run_all.sh` ends by rendering `dashboard.html` into `analysis/out/`; on its own that is
`uv run python -m analysis.dashboard --results-root results --out analysis/out/dashboard.html`.
The tables that used to live in the static scorecard are in its collapsed Diagnostics section.

**Fake-mode output is not results.** Fake and real runs share spec hashes, so check a tree's
`spec.json` (`"fake"`) before reporting from it. `analysis` refuses a root that mixes the two.

## Use it on your own behaviour, base model, method or eval

Once installed (including the OCT submodule; without it a couple of tests fail), check your
setup in fake mode, which needs no GPU and no keys:

```bash
MO_EVALS_FAKE=1 MO_EVALS_CONFIG=configs/specs.smoke.yaml ./scripts/run_all.sh --personas impulsive
```

| You bring | Read | Quick check (no GPU, no spend) |
|---|---|---|
| a behaviour | [`docs/methods/adding-a-behaviour.md`](docs/methods/adding-a-behaviour.md) | `uv run python scripts/onboard_behaviour.py <id>` |
| a base model | [`docs/reference/base-models.md`](docs/reference/base-models.md) | `uv run python scripts/check_base_model.py <hf-id>` |
| an implant method | "Adding a method" in [`docs/IMPLANT-CATALOGUE.md`](docs/IMPLANT-CATALOGUE.md) | `uv run pytest tests/test_resolve.py` |
| an eval | "Adding an eval" in [`CONTRIBUTING.md`](CONTRIBUTING.md) | `uv run python -m mo_evals --list` |

### From a constitution to a scored organism (`mobench`)

You write one file, the constitution. `mobench generate` drafts the universe data (the judge's
PRESENT/ABSENT pair, the negative and the other short texts) and then builds the eval data from
it. Nothing is overwritten once it exists, so your edits stay.

| Step | Command | What happens |
|---|---|---|
| 0 | `uv sync`; keys in `.env` | Setup. |
| 1 | none | Write `data/behaviours/<id>/constitution.json` following [`docs/methods/behaviour-descriptions.md`](docs/methods/behaviour-descriptions.md). This is the only required human input. |
| 2a | `mobench generate <id>` | Prints the projected cost and asks `y` (`-y`/`--yes` skips the prompt; with no TTY and no flag it exits 2). Respects `--budget` and `--strict-budget`. Builds the **universe data**, then prints the 2b and 3a commands. |
| 2b | `mobench review <id>` | Optional. Writes and opens `builds/<id>/review-universe.html`: every universe-data file verbatim, its repo path, and "drafted, unchanged" or "edited". You edit the files themselves. |
| 2c | `mobench generate <id>` | Reruns only what the edits invalidated (fingerprints). A file you edited is never overwritten. If the constitution changed, an unedited draft is reported STALE: `--redraft <text>` replaces it (the old text goes to `builds/<id>/data/attempts/`) and `--keep <text>` accepts it. `--plan` lists what would run. |
| 3a | `mobench generate <id> --eval-data` | Confirms the cost as in 2a, then builds the **eval data** and ends with a fake pass that builds every behaviour-scoring eval. |
| 3b | `mobench review <id> --eval-data` | Optional and read-only. Per bank it shows the count, about 10 sample items, the filter pass rate, the leak-check result, model and cost. To change these, edit the universe data, rerun 2c and then 3a. |
| 4 | none | Write `config.yaml` (the user config: `behaviour:`, organisms, properties, seeds). |
| 5 | `mobench validate` / `mobench cost` | Optional. |
| 6 | `mobench run --config config.yaml` | Runs validate and cost first, then asks `y` (`-y`; no TTY and no `-y` exits 2). `--train` skips validate and cost: it prints each build's plan (steps, done or to do, estimates; `--plan` stops there, spending nothing) and asks `y` the same way. Run `mobench validate --train` first, after the "Before your first training run" checklist in [`mobench/AGENTS.md`](mobench/AGENTS.md#before-your-first-training-run-run---train). Refuses if the behaviour's universe or eval data is missing or stale, and prints the `generate` command that fixes it. |
| 7 | `mobench report` | Re-renders the results page. |

**Testing external adapters** (ones you did not train here): an external behaviour runs steps 0 to
3, then step 4 is a `config.yaml` whose `organisms:` list the adapters (see
[`mobench/AGENTS.md`](mobench/AGENTS.md), section 2b).

**Training data for a new behaviour: the API teacher is untested.** When you train a new
behaviour (step 6, `--train`), a teacher model writes the training answers. The default runs
GLM-4.5-Air on your own GPUs (about 4x80 GB), exactly as Open Character Training (OCT) does; use
it to match the paper.
There is also an API option (`teacher.backend: http`, through OpenRouter) that needs no GPU, but
no organism has been trained on its data yet, and its data may differ. OCT starts the teacher's
hidden reasoning with a reminder of the traits, and chat APIs cannot do that, so the API option
puts the reminder in the system prompt instead. On one behaviour it showed the trait about as
often (82% of answers against 84%). See "Choosing the teacher" in
[`docs/methods/adding-a-behaviour.md`](docs/methods/adding-a-behaviour.md).

To keep your results in your own Hugging Face org, set `MO_EVALS_HF_ORG=<org>` (the default is
the org that hosts the paper's artifacts). Upload, adapter, steering-vector and scorecard repo names follow it,
and results go in the standard layout of [`docs/reference/hf-layout.md`](docs/reference/hf-layout.md),
which `scripts/fetch_results.py` reads back. URIs already in `configs/specs.yaml` name the org
that stored each artifact and are not rewritten.

## Documentation

- [`docs/QUICKSTART.md`](docs/QUICKSTART.md): score an organism. Commands, which evals need a
  judge key, minimum hardware.
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md): rebuild the paper's dashboard, tables and figures.
- [`docs/EVAL-CATALOGUE.md`](docs/EVAL-CATALOGUE.md): what each eval measures and how to
  read its number. Names match `python -m mo_evals --list`.
- [`docs/IMPLANT-CATALOGUE.md`](docs/IMPLANT-CATALOGUE.md): every implantation method and
  organism. How each method works: `docs/methods/`.
- [`mobench/AGENTS.md`](mobench/AGENTS.md): using the `mobench` CLI to build and evaluate an MO
  of your own behaviour (for people and coding agents).
- [`docs/COMMANDS.md`](docs/COMMANDS.md): every command. The gate, sweeps, Hugging Face
  mirroring, sharding a big sweep.
- [`CONTRIBUTING.md`](CONTRIBUTING.md): the contracts, how to add an eval, the gate.

## Data and code sources

This repo has no `LICENSE` file of its own. Vendored code and external-dataset banks keep the
licence and attribution of their source, stated in the directories named below.

| What | Source | Where it lives |
|---|---|---|
| Open Character Training (OCT) pipeline, released persona data and adapters | Maiya et al. 2025, [maiush/OpenCharacterTraining](https://github.com/maiush/OpenCharacterTraining); data [`maius/OpenCharacterTraining-data`](https://huggingface.co/datasets/maius/OpenCharacterTraining-data); adapters `maius/qwen-2.5-7b-it-personas` | `vendor/OpenCharacterTraining` (submodule plus `OpenCharacterTraining.patch`; see `vendor/VENDOR.md`) |
| Coherence-preference ("fried") measurement | [fried-model-organisms](https://www.lesswrong.com/posts/WmEcgcstzYCcMpc7z/your-model-organisms-might-be-fried) | `vendor/fried-model-organisms/` (its own `LICENSE`) |
| Eval framework and stock tasks (MMLU, GSM8K, GPQA, DROP, WMDP, IFEval, APPS, sycophancy) | [Inspect](https://inspect.aisi.org.uk/) and `inspect_evals` (MIT, UK AI Security Institute) | installed packages, pinned in `pyproject.toml` |
| Capability and robustness banks | `Idavidrein/gpqa`, `cais/wmdp`, `google/IFEval`, `deepmind/code_contests`, `Anthropic/hh-rlhf`, XSTest. Each bank's licence and build are in its directory's README | `data/eval_prompts/` |
| Adapted protocols (not run as published) | ImpossibleBench (Zhong et al.); van der Weij et al., arXiv:2406.07358 | `docs/EVAL-CATALOGUE.md` |
| Project page | its own `site/LICENSE` | `site/` |

**GitHub CI runs only part of the gate.** It covers lint, types, import boundaries,
pytest, the upstream eval datasets and a fake-mode byte-identity check; the fake sweep, its
cached second pass and the log-vs-Parquet check run only locally in
`./scripts/gate.sh --full` (`docs/COMMANDS.md` lists exactly what runs where). A green GitHub
check does not mean the pipeline works.
