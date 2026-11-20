# Quickstart: score an organism

Run MOBench evals on a model you already have: a LoRA adapter on Hugging Face, or any model
behind an OpenAI-compatible endpoint. Install first (README, "Install").

## 1. Dry run with no GPU and no keys

`MO_EVALS_FAKE=1` swaps every organism for `FakeEndpoint`, which writes obviously fake text. It
proves your install and the pipeline, and says nothing about any model.

```bash
uv run python -m mo_evals --list                       # every eval, with its axis
MO_EVALS_FAKE=1 uv run inspect eval mo_evals/mmlu -T persona=impulsive -T method=oct_demos --limit 3
MO_EVALS_FAKE=1 MO_EVALS_CONFIG=configs/specs.smoke.yaml ./scripts/run_all.sh --personas impulsive
uv run inspect view                                    # browse the logs
```

The stock capability evals (`mmlu`, `gsm8k`, `gpqa_diamond`, `ifeval`, `drop`, `wmdp`, `apps`)
run upstream `inspect_evals` code and fail at task build without
`uv sync --all-groups --extra inspect-evals --extra ifeval-anchor`.

Fake output is not results. Fake and real runs share spec hashes, so check a tree's `spec.json`
(`"fake"`) before reporting from it.

## `mobench`: spec files in, a dashboard out

`mobench` is a thin command line over the commands in this repo, and it prints each command it
runs. Its config is one or more files in the `configs/specs.yaml` format; repeat `--config` to pass
several, and any disagreement between them is an error.

```bash
uv run mobench fake                                            # the whole thing, no GPU, no key
uv run mobench validate --config my.yaml                       # free: keys, access, install, machine
uv run mobench cost --config my.yaml --personas impulsive      # optional: the work, any known cost
uv run mobench run --config my.yaml --personas impulsive --dashboard
uv run mobench report --results-root results                   # the page again, from a local tree
```

- **Naming behaviours is required.** A real run needs `--personas` (`--personas all` alone runs
  everything in your files). Fake runs are exempt.
- **Cost is a tool, not a gate.** A step with no estimate prints `unknown` and runs (`dpo-pairs`, `train-<m>`,
  `screen-<m>` and `evaluate` use a built-in guess instead). `--budget USD`
  warns when the projected or spent cost passes it; add `--strict-budget` for a hard stop, which
  also refuses any step with no estimate (not `dpo-pairs`, `train-<m>`, `screen-<m>` or `evaluate`: see `mobench/AGENTS.md`). Exit codes: 0 ok, 1 a check failed, 2 a guard refused, 3
  `--strict-budget` stopped the run.
- **Keys** are reported by name only. `validate` tries a present key against a free endpoint and
  makes no paid call. `mobench` reads keys from the shell, not from `.env`: after `cp .env.example
  .env`, run `uv run --env-file .env mobench ...` (or `export` them). OpenRouter provider pins in
  a spec file are optional but recommended, since an unpinned model id is not a fixed artifact.
- **Training a behaviour** (`mobench run --train`): do the "Before your first training run"
  checklist in [`mobench/AGENTS.md`](../mobench/AGENTS.md) first, then `mobench run --train
  ... --plan` prints each step, whether it is done, and its estimate; it spends nothing.
  If the repo is private, `git clone` on a fresh machine hangs silently for credentials: set
  `GIT_TERMINAL_PROMPT=0` to make it fail fast, and give it `gh auth login`, a credential
  helper or a deploy key (never a token in the URL). Plan the split below before you rent a GPU host: the pilot judge (`pilot-gate`) takes
  about 45 minutes with the GPUs idle, so stop the GPU host after the pilot teacher pass
  (`--until pilot-teacher`). If 4x
  A100 is unavailable, 2x H200 with tensor parallel 2 worked once (the `teacher` capacity row,
  `{gpu_class: H200-141GB, gpu_count: 2}`; see `mobench/AGENTS.md`).

### Where the GPU work runs

Entry point: `mobench ...` (or `uv run mobench ...`); `python -m mobench ...` is the same program.

1. **On a machine with a GPU:** `mobench run` runs there. A GPU host needs `uv sync --extra gpu`;
   the first sync is slow (several GB of vLLM, torch and CUDA wheels), and on a network volume
   the venv should live on local disk: `export UV_PROJECT_ENVIRONMENT=/tmp/mobench-venv` before
   `uv sync` and every `uv run`.
2. **On Slurm or another cluster:** run `mobench run` inside your own job. `mobench run ...
   --print-job` prints a generic sbatch script with the commands and the env var names it needs. Add
   your partition, account and time limit to it. The script passes `-y` (no terminal under sbatch to
   confirm on): submitting the job is the confirmation, so check `mobench cost --config <config>`
   first (for `--train`, `mobench run --train ... --plan`). With `--train` the script asks for the
   GPUs the teacher needs, so read its comment block before submitting.
3. **On RunPod:** see `--executor runpod` in `docs/COMMANDS.md` ("Using mobench").

**API-only steps do not need the GPU.** With `--executor local`, every step of `mobench run
--train` runs on the machine you started it on, including the API-only ones. Those are `few-shot`
(the few-shot writer), `pilot-gate` (the pilot judge and the lean gate, after the GPU step `pilot-teacher`,
the pilot teacher pass), and the views. On a rented GPU host they leave the GPUs idle (the first real test left it idle
for 45 minutes of judging), and the closing summary says how long when a GPU step also ran there.

What can move to a CPU machine, exactly: `check-constitution` to `few-shot` (the constitution check, the persona files, the persona pair and the few-shot writer; stop with `--until few-shot`; note the persona-file step `derive-personas` is a free step that writes into the repo checkout, so on a fresh checkout it runs again, and `check-constitution`, `adopt-authored` and `few-shot` are skipped only when their inputs and outputs match), and `pilot-gate` (the
pilot judge and gate) once `pilot-teacher` (the pilot teacher pass) is done (stop the GPU host with `--until pilot-teacher`). The
teacher steps (`pilot-teacher` and `full-teacher`, the full teacher pass), training and the student stage stay on a GPU. A split run
uses one `--builds-root` that both machines can read and write (copy or share the build folder
between runs; only one machine runs at a time) and the same `paths.model_path` with LIMA present:
a step is skipped only when its fingerprint and outputs match `state.json`, and the teacher's
fingerprint includes that path and the LIMA files. For example:

```bash
# on the CPU machine
mobench run --train --personas P --base-models B --build-config C --until few-shot
# on the GPU host, same builds root
mobench run --train --personas P --base-models B --build-config C --until pilot-teacher
# back on the CPU machine
mobench run --train --personas P --base-models B --build-config C --until pilot-gate
```

The fake pipeline tests the resume logic; the two-machine copy itself has not been run for
real. `--executor runpod` is the other route (API steps stay on the machine running `mobench`),
and it is experimental.

## 2. Score a model through the stock Inspect CLI

Every eval is an Inspect task, so one (organism x eval) cell is one `inspect eval` call. Run
from the repo root. `-T persona=` names the behaviour whose banks the eval uses.

**An OpenAI-compatible endpoint** (a vLLM server, a hosted API, a gateway). Inspect reads
`<SERVICE>_BASE_URL` and `<SERVICE>_API_KEY` for the `openai-api/<service>/<model>` form:

```bash
export MYSERVER_BASE_URL=http://localhost:8000/v1 MYSERVER_API_KEY=none
uv run inspect eval mo_evals/simple_behaviour_exhibit -T persona=impulsive \
    --model openai-api/myserver/<served-model-name> --limit 20
```

`--model` runs the eval in external mode (`spec_hash="external"`). Do not combine it with an
organism selector (`-T method=...`); that raises instead of silently ignoring one. Judged
evals also call the grader (step 4).

**A Hugging Face LoRA adapter.** Register it as an organism, then select it. Copy the
`oct_demos` row in `configs/specs.yaml` (a LoRA on a base model, under the `impulsive` persona)
into a copy of `configs/specs.smoke.yaml`, set `method_config.adapter_repo` (and `subfolder`, if
the adapter is not at the repo root) to yours, and set `base_model` to the adapter's base:

```bash
MO_EVALS_CONFIG=configs/my_specs.yaml uv run inspect eval mo_evals/mmlu \
    -T persona=impulsive -T method=oct_demos        # needs the gpu extra and a GPU
MO_EVALS_CONFIG=configs/my_specs.yaml ./scripts/run_all.sh --personas impulsive   # a sweep of it
```

`-T spec_hash=<hash>` selects one organism exactly; an ambiguous selector is refused with the
candidate hashes. A sweep (`run_all.sh`) writes Parquet under `results/` and renders
`analysis/out/dashboard.html`; a single cell writes only a `.eval` log under `./logs`.
Methods other than a plain adapter (steering, prompting, few-shot) are configured the same way:
`docs/IMPLANT-CATALOGUE.md` lists each method, and `docs/methods/serving.md` says how each is
served. A new behaviour needs its own banks: `docs/methods/adding-a-behaviour.md`.

Do not pass `--solver`: for most evals the solver is the measurement.

## 3. Which evals need a judge key

Judged evals call a model through OpenRouter, so they need `OPENROUTER_API_KEY` in `.env`.
Judge-free evals run without it.

- **Judge-free** (stock `inspect_evals` and programmatic scoring): `mmlu`, `gsm8k`, `gpqa_diamond`,
  `ifeval`, `drop`, `wmdp`, `apps`, `capability_format`; `mu_decisiveness`, `transitivity`;
  `introspection_mcq`; `misaligned_overrefusal`, `misaligned_sandbagging`,
  `misaligned_reward_hacking`.
- **Judged**: the expression, generality, robustness and persistence cells, the other
  introspection cells, and `misaligned_are_you_sure`. `robustness_multiturn_debate_gen` also
  needs the adversary model.
- **White-box** (`whitebox_truth_world`, `whitebox_truth_self`): judge-free, but they read
  hidden states, so they need a vLLM-served organism with `MO_EVALS_WHITEBOX=1`; they are
  skipped otherwise.

`docs/EVAL-CATALOGUE.md` says for each eval what it measures and how it is scored. A judged
number depends on the judge: read it against the same-size base model.

## 4. Swap or re-run the judge

Judges read Inspect's `grader` model role, so a finished log can be re-judged without
re-running the organism:

```bash
uv run inspect score <log>.eval --model-role grader=<model>
```

Changing the judge invalidates every previously scored sample.

## Minimum hardware

- **Fake mode, and judge-free scoring of an external endpoint:** a laptop CPU, no GPU.
- **Serving a 7B adapter:** one GPU with room for the bf16 weights (about 15 GB) plus KV
  cache. The configured 16,384-token context (`configs/serving.yaml`; override with
  `MO_EVALS_SERVING_CONFIG`) is comfortable on an 80 GB card. A 24 GB card with a reduced
  `max_model_len` is an estimate from weight size, not a measured run.
- **14B:** about 28 GB of weights; one 80 GB card.
- **32B:** does not fit one 80 GB card at the configured context. Use
  `configs/serving_32b_tp2.yaml` (two GPUs, tensor parallel) or a larger card.
- Real serving needs the `gpu` extra and a CUDA 13.0+ driver (`nvidia-smi` shows driver 580.65
  or newer).
- A full sweep over many organisms and all 36 evals is a large job: `docs/COMMANDS.md`
  ("Big sweep topology") covers sharding it.

Real-run rules: name the behaviours with `--personas` (`--personas all` runs the whole matrix
on purpose), and cap and ramp any bulk judge or generation job.
