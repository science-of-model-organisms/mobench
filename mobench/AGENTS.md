# Using `mobench` (for agents and people building an MO of their own behaviour)

This file is for **using** MOBench: implanting your behaviour into a base model (a "model
organism", MO) and evaluating it. It is not for people changing MOBench itself (that is
`CONTRIBUTING.md` and the root `CLAUDE.md`). Every flag below was checked against
`mobench/cli.py` and `orchestrate/build.py`; when in doubt run `uv run mobench <command> --help`.

Longer references: [`docs/QUICKSTART.md`](../docs/QUICKSTART.md) (scoring, hardware),
[`docs/COMMANDS.md`](../docs/COMMANDS.md) ("Using mobench"),
[`docs/methods/adding-a-behaviour.md`](../docs/methods/adding-a-behaviour.md) (writing the
behaviour's assets, the teacher pilot and gate), [`configs/build.yaml`](../configs/build.yaml)
(build defaults), [`examples/toybe/`](../examples/toybe/) (a minimal behaviour).

## Rules before you spend anything

- Real runs cost money and GPU time. Name the behaviour (`--personas impulsive`); never run the
  whole matrix by default. A real run without `--personas` is refused (`--personas all` alone
  runs everything, on purpose), unless every `--config` is a user config (section 2b), which
  names each organism itself.
- Always `uv run ...`, never bare `python`.
- Fake output (`MO_EVALS_FAKE=1`, `mobench fake`) proves plumbing and is never a result.
- Report environment variables by **name** only; never print or paste their values.

## 1. Install and setup

```bash
git submodule update --init                  # vendor/OpenCharacterTraining (training recipe)
uv sync --all-groups                         # core + dev
uv sync --all-groups --extra gpu             # a GPU host: vLLM and the training stack
cp .env.example .env                         # then fill it in
```

`mobench` itself reads keys from the **shell**, not from `.env`: the evaluation runners it
launches load `.env` (without overriding anything already set), but `mobench validate` and
`run --train` do not. Run it as `uv run --env-file .env mobench ...` (or `export` the names)
so `validate` sees what the run will.

### Before your first training run (`run --train`)

Six things (the first five are checked by `mobench validate --train ...`, which names the fix):

1. **OCT submodule**: `git submodule update --init` (`validate`: `OCT submodule`).
2. **GPU extra**: `uv sync --all-groups --extra gpu` on the GPU host (`validate`: `extra gpu`).
3. **LIMA**: gated dataset under `paths.model_path/lima/{train,test}.jsonl`; accept its terms
   and use a token that can read it (`validate`: `LIMA`, with the `hf download` command).
4. **Disk**: the teacher is about 210 GB, plus the base and checkpoints. Decide where the
   Hugging Face cache lives (`HF_HOME`) and check your quota: `validate` reads `df`, which
   cannot see per-user quotas (`validate`: `disk`, `disk quota`).
5. **A 4-GPU host for the teacher** (`pilot-teacher`, the pilot teacher pass, and `full-teacher`, the full teacher pass), or
   the API teacher; API steps run
   on a CPU machine (`validate`: `GPU teacher`). The split is in `docs/QUICKSTART.md`.
   - **Split the idle part off up front.** `pilot-gate`, the pilot judge, takes about 45 minutes with
     the GPUs idle. Run the GPU host with `--until pilot-teacher`, then `--until pilot-gate` from a CPU
     machine (same `--builds-root` and `paths.model_path`).
   - **Fallback when 4x A100 is unavailable: 2x H200 with tensor parallel 2.** Override the row
     in your `--build-config`: `executor.capacity.<base>.teacher: {gpu_class: H200-141GB,
     gpu_count: 2}`. Tested once (RunPod phase-2 run, impulsive on Qwen2.5-7B): the pilot gate
     (`pilot-gate`) passed 84/100 and `full-teacher` took 22 minutes for 2000 prompts. One run, not a benchmark.
6. **Access to this repo on a fresh machine.** If the repo is private, `git clone` waits for
   credentials with no output. Set `GIT_TERMINAL_PROMPT=0` so it fails at once, and give it a
   credential: `gh auth login`, a git credential helper, or an SSH deploy key. Do not put a
   token in the clone URL: it is stored in `.git/config` and shows in `ps`.

Then `mobench run --train ... --plan` shows what would run and its estimates before anything
is asked or spent.

Other extras: `--extra inspect-evals --extra ifeval-anchor` (the stock capability evals mmlu,
gsm8k, gpqa_diamond, ifeval, drop, wmdp, apps), `--extra figures`. On a network volume put the
venv on local disk first: `export UV_PROJECT_ENVIRONMENT=/tmp/mobench-venv`.

Environment variable names (set them in `.env` or the shell):

| Name | Needed for |
|---|---|
| `HF_TOKEN` | Hub reads (bases, adapters, the gated LIMA dataset); write access only to upload |
| `OPENROUTER_API_KEY` | judges, the few-shot writer, the pilot judge, and the API teacher if you pick it |
| `MO_EVALS_FAKE=1` | switch every model for a fake one (no GPU, no key) |
| `MO_EVALS_CONFIG` | default spec file (else `configs/specs.yaml`) |
| `MO_EVALS_RESULTS_ROOT`, `MO_EVALS_REPORT_DIR` | where results and the page go |
| `MO_EVALS_HF_ORG` | the Hub org uploads go to (or set `hf.org` in the build config) |

Training builds also need LIMA under `paths.model_path` (`<model_path>/lima/{train,test}.jsonl`,
gated on the Hub; accept its terms and use a token that can read it). `mobench validate --train`
checks this.

## 2. The flow

There is no `train` subcommand: training is `mobench run --train`. Subcommands: `validate`,
`cost`, `fake`, `run`, `report`, `generate`, `review`.

`run` (with or without `--train`) and a paid `generate` ask `y` before spending. Without a
terminal they exit 2 unless `-y` is passed. Pass `-y` only once the person has seen and approved
the cost (`mobench cost` / `run --train --plan` / `generate --plan`). `run --train` prints each
build's plan (steps, done or to do, estimates) before it asks, and a bad `--until`, method or
base model stops it there.

```bash
uv run mobench fake                                   # 0. whole pipeline incl. training, no GPU, no key
uv run mobench validate --train --personas P --base-models B --build-config C.yaml
                                                      # 1. free: keys, access, install, machine, LIMA, GPUs
uv run mobench run --train --personas P --base-models B --build-config C.yaml --plan
                                                      # 1b. free: the steps, what is done, estimates
uv run mobench run --train --personas P --base-models B --build-config C.yaml --until pilot-gate
                                                      # 2. cheap gate: build to the pilot judge + lean gate
uv run mobench run --train --personas P --base-models B --build-config C.yaml \
    --methods sft_behaviour,dpo_behaviour --budget 50 --strict-budget
                                                      # 3. the full build: teacher, train, screen, register, evaluate
uv run mobench report --results-root builds/P/<base-slug>/results   # 4. the dashboard page again
```

- `--train` needs named `--personas` and **exactly one** `--base-models`; `all` is refused.
  `--methods` defaults to `sft_behaviour,dpo_behaviour` (recipes in `configs/build.yaml`).
- `--until STEP` stops after that step (any id from section 3). `pilot-gate` is the cheap gate: if the
  lean gate fails the run stops with exit 4 and the expensive full teacher pass does not start.
- Evaluation only, on organisms already in spec files: `mobench run --config my.yaml
  --personas P [--evals ...] [--dashboard] [--out DIR]`. `mobench cost` (same selection flags)
  lists the work and any known cost.
- `--evals`, `--exclude-evals` and `--results-root` do **not** apply to `--train`; set
  `steps.evaluate.evals` in the build config instead (the CLI refuses the flags rather than ignore them).
- `mobench run --print-job` prints a generic sbatch script (commands and env var names only) to
  adapt to your cluster. With `--train` the GPU request follows the build config's capacity row
  for the steps the job runs (the teacher's 4 GPUs unless `--until` stops before it), with a
  comment block on the split and the prerequisites above. `--executor runpod` additionally needs `--experimental-runpod` and
  `executor.runpod` in the build config; it is untested against a real pod.
- Behaviour assets you write yourself before step 1: the constitution and negative
  (`data/behaviours/<id>/constitution.json`, `data/behaviours/<id>/negative.txt`), the persona pair
  (`data/eval_prompts/persona_pairs.yaml`) and the registry file `data/behaviours/<id>/config.yaml`
  (with `gate_traits`). See `docs/methods/adding-a-behaviour.md` and `examples/toybe/`.

## 2b. Testing adapters you did not train (user configs)

A user config lists LoRA adapters by commit, the properties to measure and the seeds; nothing
else. Template: [`configs/user.example.yaml`](../configs/user.example.yaml); properties, their
evals and the reserved labels: [`configs/README.md`](../configs/README.md).

```yaml
organisms:
  - adapter: <owner>/<repo>   # or a Hub URL, or <owner>/<repo>@<40-hex>; unpinned -> latest commit, recorded in .mobench-pins.json (--refresh-pins re-resolves); a branch or short hash is refused
    label: my_adapter                         # optional; default the repo name
    base_model: Qwen/Qwen2.5-7B-Instruct      # optional; default adapter_config.json's base
properties: [misalignment, capability]        # optional; default every one that applies
seeds: [0, 1, 2]                              # optional, this default
```

```bash
uv run mobench validate --config my.yaml          # free: keys the selected evals need, Hub access
uv run mobench cost --config my.yaml              # prints the plan, runs nothing
uv run mobench run --config my.yaml --dashboard   # prints the plan, then runs it; no --personas
uv run mobench fake --config my.yaml --no-train   # plumbing only (rows need `base_model:` offline)
```

- **Base model.** Read from each adapter's `adapter_config.json` (`base_model_name_or_path`) at
  the pinned commit. A row's `base_model:` always wins; when it differs from the adapter's (an
  `unsloth/` re-upload, say) the plan prints both and `validate` compares the two repos for
  free: equal safetensors hashes are a PASS, different ones a NOTE (re-serialised files differ
  even when the tensors match; compare offline to be sure), and `pad_token`, `model_max_length`,
  `padding_side`, the chat template and `generation_config.json` are compared key by key (a WARN
  each). Serving always uses the stated base's weights, tokenizer, chat template and generation
  config: the adapter goes on top of the `base_model` the row names. There is no alias table.
- **Base rows.** One `base` row per distinct base (`include_base_model: false` turns it off).
  Its cells already in the results root are reused; with `behaviour: impulsive` on
  Qwen2.5-7B it is the same organism as `configs/specs.yaml`'s base row.
- **Behaviour.** Without `behaviour:` only `capability`, `coherence` and `misalignment` can run
  (they read no behaviour definition and call no judge, so no `OPENROUTER_API_KEY` is needed);
  naming another property is refused. `behaviour: <id>` must have a definition in the repo
  (section 1's behaviour assets); the rows then carry it as their persona id.
- **Evals.** `evals: {<property>: {include: [...]}}` or `{exclude: [...]}` narrows one
  property (not both); an eval under the wrong property is refused, naming its real one. Eval
  settings (judges, item counts) always come from `configs/specs.yaml`.
- **The plan** (`cost` and `run`): each row and its base, the base rows (cached or to run),
  the cells per row (to do / cached / deduped), the skipped evals by reason, and the cost
  (unknown). `cost` prints it and runs nothing; `run` prints it, then asks `y` (`-y` skips the
  question; with no terminal, `-y` is required).
- **The page** shows each adapter as `<label>[external]` beside the base (`mobench report`
  turns on `external_rows` for a tree that holds one; the paper's build never does).
- **Mixing** a user config with a spec file needs `--personas`, and is refused when the merged
  eval list would add evals the user config left out.
- Not yet: training from a user config (`train:` is refused), and one page over several
  results roots.

## 2c. Building a behaviour's data from its constitution (`generate`, `review`)

The only required input is `data/behaviours/<id>/constitution.json`. Everything else the behaviour-scoring
evals read is drafted or built from it, in two stages. Run only the behaviour you were asked for.

```bash
uv run mobench generate <id>                  # universe data: prints the cost, asks y (-y skips; no TTY needs -y)
uv run mobench review <id>                    # optional: builds/<id>/review-universe.html, every file verbatim
# edit the files at the paths shown, then rerun `generate <id>`: only what the edits invalidated reruns
uv run mobench generate <id> --eval-data      # eval data, then a fake pass over every behaviour-scoring eval
uv run mobench review <id> --eval-data        # optional, read-only: count, ~10 samples, filter rate, leak check, cost
uv run mobench run --config config.yaml       # validate and cost, asks y; refuses missing or stale data
```

- **Never overwritten.** A file that exists is never redrafted by a rerun. A draft whose inputs
  changed is STALE: `generate` stops, names it and offers `--redraft <text|all>` (the old text
  goes to `builds/<id>/data/attempts/`) or `--keep <text|all>`. A draft you edited counts as
  yours and is redrafted only when `--redraft` names it.
- **Provenance.** `builds/<id>/data/provenance.json` records each drafted file's hash, inputs,
  model and prompt; the review page reads it to say "drafted, unchanged" or "edited", and "not
  tracked" without it. Shared YAML files show and hash only the behaviour's own entry.
- **Budget.** `--budget` and `--strict-budget` apply as for `run --train`.
- **Step ids.** `generate`'s steps are in the second table of section 3.
- **External adapters.** An external behaviour runs `generate`, `generate --eval-data`, then a
  `config.yaml` with `organisms:` adapters (section 2b).
- The inferred `scope` and `contrast`, and the `false_trait` pick with its reason, are printed at
  the end of `generate` and shown first on the review page; a wrong guess changes every
  generality bank.

## 3. The steps and what each needs

`free` = this machine, no GPU, no paid call. `api` = paid API calls, no GPU (runs on the machine
you start from). `gpu` = needs GPU(s). Ids as in `orchestrate/steps.py`. A step id is what
`--until` takes, what `STOP: <id>: ...` and `= <id> unchanged` lines print, and its own
`steps.<id>` key in the build config. The short ids used before 2026-11-17 (`s5`, `s14`, `g4`)
are refused there, naming the new id (`steps.s14 is now steps.evaluate`); a build or
`generate` folder written with them is still read as it was, so nothing re-runs.

| Step | Does | Needs |
|---|---|---|
| `check-constitution` | checks your constitution and negative exist and parse | free |
| `derive-personas` | derives persona files from the constitution (adopted if hand-authored) | free |
| `adopt-authored` | adopts your persona pair and registry keys (stops, naming what is missing) | free |
| `few-shot` | stage 1: expands seed questions into the few-shot file (`steps.few-shot.max_calls`) | api |
| `download-teacher` | pre-downloads the teacher weights (skipped on a pod or the API teacher) | free |
| `pilot-teacher` | pilot teacher pass (`steps.pilot-gate.limit` prompts, default 200) | gpu, or api with the HTTP teacher |
| `pilot-gate` | pilot judge and lean gate (the cheap go/no-go; `--until pilot-gate`) | api |
| `full-teacher` | full teacher pass (the expensive one) | gpu, or api with the HTTP teacher |
| `sft-view` | SFT view of the teacher file | free |
| `dpo-pairs` | DPO pairs: the student's own answers as rejected side (only if a DPO method is built) | gpu |
| `train-<m>` | train method `<m>` (LoRA) | gpu |
| `screen-<m>` | side-effect screen of the trained adapter (`steps.screen.on_flag`, `max_prompts`) | gpu |
| `publish-<m>` | records where the adapter lives for `register`: uploads it to the Hub if `hf.push_adapters: true`; otherwise uploads nothing and records the local adapter path | free (Hub write if pushing) |
| `adopt-assets` | adopts ICL demos, steering vector, SBE and MCQ assets if present, else skips and says so | free |
| `register` | registers the organisms into `builds/<id>/<base-slug>/specs.local.yaml` | free |
| `evaluate` | evaluates the organisms (`steps.evaluate.evals`, default every eval in the spec file) | gpu (serves the organism) |

`<m>` is the method's short name (`sft`, `dpo`). `mobench run --train ... --plan` (or `mobench cost
--train ...`) lists what would run without running it; it is `python -m orchestrate build <id>
--base B --plan` per behaviour. `few-shot` runs for every behaviour, a hand-authored one included: it
does not adopt OCT's vendored stage-1 file.

### The `generate` steps (universe data, then eval data)

`mobench generate <id>` runs `check-behaviour` to `universe`; `--eval-data` then runs
`draft-eval-items` to `fake-check`. They are separate from the build steps above: the ids appear
in the output (`STOP: draft-texts: ...`) and in the build config
(`steps.base-transcripts.base_transcripts`, `steps.universe.max_calls`; every
`generality-<cell>` step shares `steps.generality-banks`, as in `steps.generality-banks.repair`).
`free` and `api` as above; the docstring of `orchestrate/generate.py` lists the same graph.

| Step | Does | Needs |
|---|---|---|
| `check-behaviour` | checks the behaviour id and that `constitution.json` exists and parses | free |
| `draft-texts` | drafts the short texts (negative, universe description, traits, ...) from the constitution; only where absent, never overwriting yours (`--redraft`, `--keep`) | api |
| `persona-files` | derives the persona files from the constitution (adopted if hand-authored) | free |
| `base-transcripts` | has the 72B base model answer the behaviour's own seed questions: the base transcripts the TRUE side of the universe is written from (`steps.base-transcripts.base_transcripts`; its call cap is `steps.base-transcripts.base_transcripts.max_calls`, default 200, not `steps.base-transcripts.max_calls`) | api |
| `universe` | builds the universe: the true and false contexts the generality banks are written from (`steps.universe.max_calls`) | api |
| `draft-eval-items` | drafts the item sets: SBE T1 questions, multiple-choice items, counter-demos | api |
| `sbe-bank` | builds the SBE bank from the T1 questions (no training files) | free |
| `generality-<cell>` | builds one generality bank per cell (`fermi`, `causal`, `downstream`, `ood`, `robustness`, and `controls` for a `scope: gated` behaviour only, a no-op otherwise); `steps.generality-banks.repair` names the cells run with `--repair` | api |
| `fake-check` | fake pass: every behaviour-scoring eval builds for the new behaviour as it does for `impulsive` | free |

Ids are config keys and fingerprint labels, so they do not change; the prose in these docs names
the step beside the id.

## 4. The build config

`--build-config C.yaml` is deep-merged over `configs/build.yaml`; keep your file small and only
put what you change. The keys you will use:

```yaml
steps:
  few-shot: {max_calls: 200}                 # cap on few-shot-writer calls
  pilot-gate: {limit: 100, on_fail: stop}    # pilot size; continue = past a FAIL or INCOMPLETE gate (smoke tests)
  pilot-teacher: {estimate_usd: [2.5, 2.5]}  # ledger estimate [lo, hi] for the step
  full-teacher: {limit: null, estimate_usd: [13, 15]}
  screen: {on_flag: stop, max_prompts: 50}   # every screen-<m>; stop | continue when it flags
  evaluate: {evals: [gsm8k, mmlu]}           # the evals the build runs
  teacher: {backend: vllm}                   # vllm (default) or http, see section 5
paths: {model_path: /path/with/lima/under/it}
hf: {org: my-org, push_adapters: false}      # false: no upload, organisms use the local adapter path
executor:
  capacity:                                  # GPU class and count per base and step
    Qwen/Qwen2.5-7B-Instruct:
      teacher: {gpu_class: A100-80GB, gpu_count: 4}
```

- **Teacher model, name, sampling, engine** are overridable per build under `steps.teacher`
  (keys `model`, `name`, `sampling`, `engine`, `http`, `require_think_close`; merged key by key
  over `teacher:` in `configs/behaviour_pipeline.yaml`, which stays the default). They reach the
  teacher job (as `MO_EVALS_PIPELINE_OVERRIDES`), the manifest (`model`, `sampling`, `engine`) and
  the teacher steps' fingerprint, so a change re-runs `pilot-teacher` (the pilot teacher pass) and `full-teacher` (the full teacher pass). The GPU shape is the
  `executor.capacity.<base>.teacher` row (`gpu_class`, `gpu_count`; tensor parallel is the GPUs
  visible capped by `gpu_count`, rounded down to 1, 2, 4 or 8).
  ```yaml
  steps:
    teacher: {model: Qwen/Qwen2.5-7B-Instruct, engine: {max_model_len: 4096}, require_think_close: false}
  executor: {capacity: {Qwen/Qwen2.5-7B-Instruct: {teacher: {gpu_class: A100-80GB, gpu_count: 1}}}}
  ```
  (The capacity key is the build's **base model**, `--base-models`.)
- **Stage-1 provider** (the question writer's pinned OpenRouter provider, default `coreweave`:
  verified serving Llama-3.3-70B on 2026-11-03, when `novita` returned 404): `steps.few-shot.provider: <slug>`
  (or a list) replaces it, fallbacks stay off. A 404 from stage 1 usually means the pinned
  provider stopped serving the model; pick another that serves it.
- **Evals:** `steps.evaluate.evals`. **Limits:** `steps.pilot-gate.limit`, `steps.full-teacher.limit`,
  `steps.few-shot.max_calls`, and for an API teacher `steps.pilot-gate.max_calls` /
  `steps.full-teacher.max_calls`. The pilot's size and its API teacher's call cap live in the
  `pilot-gate` block, which `pilot-teacher` reads: `steps.pilot-gate.max_calls` caps the pilot
  teacher pass, not the judge.
- A missing `executor.capacity` row only warns (the step uses whatever is on the machine).

## 5. Choosing the teacher backend

The teacher (GLM-4.5-Air, `zai-org/GLM-4.5-Air`) generates the chosen side of the training data.

- **`vllm` (default): the validated recipe.** Self-hosted bf16; the validated shape was 4x80GB
  (A100). Tensor parallel must be 1, 2, 4 or 8; the run uses the GPUs it sees, capped by
  `executor.capacity.<base>.teacher.gpu_count`, and stops before launch if none is visible. About
  212 GB of bf16 weights means 2 x 141GB cards (H200) should fit by arithmetic, but that shape is
  **unvalidated**. Under `MO_EVALS_FAKE=1` no model is loaded.
- **`http`: experimental, opt-in, no GPU.** `steps: {teacher: {backend: http}}` in the build config (or
  `teacher.backend: http` in `configs/behaviour_pipeline.yaml`, or `scripts/run_glm_teacher.py
  --backend http`). `pilot-teacher` (the pilot teacher pass) and `full-teacher` (the full teacher pass) then run on the machine you start from, through an
  OpenAI-compatible API, and print a warning that it is experimental. **For parity with the
  paper, use the self-hosted vLLM teacher**: the API teacher is not OCT's method (it cannot
  prefill the teacher's `<think>`), and organisms trained on its data have not been compared
  with self-hosted ones.
  - **Any OpenRouter model.** Set `steps.teacher.http.model` (and the provider pin) in the build
    config; the shipped default is GLM-4.5-Air. A non-GLM teacher goes beyond OCT's recipe and
    is untested beyond the offline checks.
  - **Pin exactly one provider.** `teacher.http.params.provider.only` ships empty: name one
    provider that serves the model (GLM-4.5-Air: at bf16). The run is refused otherwise.
    Fallbacks are off. `teacher.http.strict_params` sets `require_parameters`: `true` (the default,
    and the shipped GLM config) rejects a provider that cannot honour `repetition_penalty` or
    another sampling parameter; `false` (opt-in) lets it through, reads OpenRouter's public
    endpoints metadata, prints one `WARN` naming the sampling parameters that provider ignores
    and records them as `api_teacher.ignored_params` in the manifest (`"unknown"` offline).
  - **Thinking.** `require_think_close` defaults to `auto` for http: a reply with reasoning
    (inline `<think>...</think>` or the provider's `reasoning` field) parses as for GLM, a reply
    with no reasoning and no `<think>` is kept whole (manifest `no_reasoning_kept`), and one that
    opened `<think>` and never closed it is dropped as `no_think_close`. An explicit `true` or
    `false` wins; vLLM reads `auto` as `true`.
  - **Assistant name.** The teacher is told its name is `teacher.http.name`, else (unless the
    build changed `teacher.name`) one derived from the model id (`z-ai/glm-4.5-air` -> `ChatGLM`,
    `meta-llama/llama-3.3-70b-instruct` -> `Llama`). It is recorded as `assistant_name` in the
    teacher manifest, and the pairing step (`dpo_pairs`) launders exactly that name to the
    student's.
  - Key: the env var named by `teacher.http.api_key_env` (`OPENROUTER_API_KEY`).
  - **Fidelity caveat:** a provider applies its own chat template, quantisation and assistant-
    prefill handling. Chat APIs cannot continue a raw `<think>` prefill (a trailing assistant
    message gave 71% trait-present pooled over 3 samples vs 84/100 self-hosted), so the API route moves the trait
    reminder (its sentence and the trait list) onto the end of the system prompt and sends no
    assistant message (`http_prompt_format: system_traits`, recorded in the teacher manifest and
    the generator name). On the pilot it matched the self-hosted rate (82% vs 84%, n=100, k=3) but
    it is not upstream's prompt; vLLM stays the validated recipe. Run the step-5 pilot and its lean
    gate on it and compare with the self-hosted result before a full pass. An unclosed `<think>` is
    dropped and counted (see Thinking below).
  - **Budget is mandatory for http.** `pilot-teacher` and `full-teacher` refuse to run without
    `steps.<step>.http_estimate_usd: [lo, hi]` (hi > 0; the pod `estimate_usd` is ignored), even
    without `--strict-budget`, and `full-teacher` also needs `steps.full-teacher.max_calls`. `max_calls` counts
    **prompts**, not requests: a failed request is retried (`teacher.http.max_retries`,
    `max_retries_429`), so worst-case billed requests are about prompts x (1 + max_retries +
    max_retries_429). Size the cap and the estimate with that.
  - **Failed requests.** A row whose request failed is labelled `api_error` (not `no_think_close`),
    is not cached as final (a re-run retries only those rows), and the pass exits non-zero if more
    than `teacher.http.max_error_rate` (default 0.05) of its rows failed, so a short file never
    reaches `dpo-pairs`. A provider's separate `reasoning` field is rebuilt into `<think>...</think>`.
  - The generator name (`http:<provider>:<model>:<prompt format>` vs `vllm:<model>`) and the manifest separate the
    two builds, so switching backend re-runs the step and never reuses the other's answers.
- **Small-teacher smoke option.** To exercise the real teacher path cheaply, use the
  `steps.teacher` override above with e.g. `Qwen/Qwen2.5-7B-Instruct`, a 1-GPU capacity row and
  `require_think_close: false`. Plumbing only, not the recipe: the prefill and parser expect GLM's
  `<think>...</think>`; with `require_think_close: false` a reply without `</think>` is kept whole
  instead of dropped (the manifest records the flag; the GLM path is unchanged). The lean gate's
  verdict on such data means nothing. For a no-GPU check use `mobench fake`.

Details and the doc this mirrors: `docs/methods/adding-a-behaviour.md`, "Choosing the teacher".

## 6. Other GPU needs

- Student stage (`dpo-pairs`), training (`train-<m>`), screen (`screen-<m>`) and `evaluate` each want one 80GB-class card
  at 7B (`configs/build.yaml` capacity rows; H100-80GB listed). A 48 GB card (L40S, A6000)
  runs SFT, the screen and the evals at 7B, but `train-dpo` runs out of memory on it: use 80 GB
  for DPO. 14B is about 28 GB of weights;
  32B needs two GPUs for serving (`configs/serving_32b_tp2.yaml`) or a bigger card.
- API-only steps (`few-shot`, the few-shot writer; `pilot-gate`, the pilot judge and gate; the views) run on whatever machine you start from; on a GPU host
  they leave the GPUs idle. Split the run (`--until few-shot` on a CPU machine, `--until pilot-teacher`
  on the GPU host, `--until pilot-gate` back on the CPU machine, one shared `--builds-root`): see
  `docs/QUICKSTART.md`, "Where the GPU work runs".

## 6b. Splitting a build across hosts (CPU host and GPU pod)

```bash
# CPU host: stage 1 and the cheap pieces
uv run mobench run --train --personas P --base-models B --build-config C.yaml --until few-shot
rsync -a builds/ pod:/path/to/repo/builds/            # send the build folder (state, ledger, s4/)
# GPU pod (once per shell; keep the venv on local disk and copy, don't hardlink, across mounts):
export UV_PROJECT_ENVIRONMENT=/tmp/mobench-venv UV_LINK_MODE=copy
uv sync --all-groups --extra gpu --extra inspect-evals --extra ifeval-anchor   # the extras gsm8k etc. need
uv run mobench run --train --personas P --base-models B --build-config C.yaml --until pilot-teacher
rsync -a pod:/path/to/repo/builds/ builds/            # bring the build folder back
uv run mobench run --train --personas P --base-models B --build-config C.yaml --until pilot-gate
```

Use the same `--builds-root` and the same `paths.model_path` with LIMA on both hosts; send the
repo state too (same commit and build config), since the fingerprint covers code, config and
`uv.lock`. Dotfiles such as LIMA's `.cache/` are ignored by the fingerprint, so copying the
folder does not re-run the pilot teacher pass (`pilot-teacher`). Only one host runs a build at a time.

With `hf.push_adapters: false` the registered organisms point at the adapter's local path, so
`evaluate` must run on a host where that path exists (same `--builds-root` location). Judged evals
(e.g. `generality_ood`) call a judge and need its API key (`OPENROUTER_API_KEY`); a GPU host
without the key should list only judge-free evals in `steps.evaluate.evals`, or stop the build before
`evaluate` and run it where the key is.

`mobench validate` has no `--builds-root` flag (only `run` does).

**`validate --train --offline` on a CPU-only host.** Expected FAILs there, not mistakes:
`GPU teacher (<base>)` (no GPU, when the build would start the teacher on this host with the
local executor and `--until` at or past the pilot teacher pass (`pilot-teacher`); it is a WARN for `--until few-shot` or earlier,
and a PASS with the HTTP teacher), `LIMA` if the gated files are not on this host, and a missing
API key reported by name (HF_TOKEN is a WARN). Real mistakes to fix: `OCT submodule` empty,
`paths.model_path` unset, a build config that does not parse, a base model that fails its check.

## 7. Cost caps and the ledger

- Every step prints `cost <step>: up to $X` or `unknown` before it runs. Unknown is allowed.
- `--budget USD` warns when projected or spent cost passes it. `--strict-budget` (requires
  `--budget`) makes it a hard stop (exit 3) and refuses a step with no estimate (exit 2), except `dpo-pairs`,
  `train-<m>`, `screen-<m>` and `evaluate`, which fall back to the built-in guess described below. It is
  admission against your estimates, not metering of a running API job. Teacher spend over HTTP
  is not metered either: the ledger shows only the judge's cost, so the strict budget admits
  the teacher against your `http_estimate_usd`, and the real bill is on the provider's side.
- The run ends with `wall time per step: ...` (each step this invocation ran, a stopped one
  included) and, for steps that made API calls whose dollars no row records (`few-shot`'s stage-1
  responses), their call counts, also stored as `api_calls` on the step's ledger row. `evaluate`'s
  judged calls are not counted yet.
- With `run --train` and several `--personas`, `--budget` is the total for the command: each
  build is its own ledger, so it gets an even share (printed), and the shares sum to `--budget`.
  Under `--strict-budget` a job whose remaining allowance buys less than `budget.kill_grace_s`
  of pod time at the listed rate is refused (exit 3) rather than started past the cap.
- Estimates come from `steps.<id>.estimate_usd` (build config), under the key the step *reads*:
  `screen` covers every `screen-<m>` step, and `train-<m>` (for example `train-sft`), `dpo-pairs`
  and `evaluate` use their own ids. The `--strict-budget` error names that key.
  `configs/build.yaml` ships `few-shot`, `pilot-gate`, `pilot-teacher` and `full-teacher` values from measured runs
  (`full-teacher: [13.03, 15.0]`). For `dpo-pairs`, `train-<m>`, `screen-<m>` and `evaluate` it ships none: when your config sets
  none, the budget check, `--plan` and the `cost` line use a **default guess**
  (`DEFAULT_ESTIMATE_USD` in `orchestrate/steps.py`, about 15 minutes to 4.5 GPU-hours at $1.50;
  printed as `(default guess)`; never part of a fingerprint). A figure you set always wins, so
  replace the guess after your first run. **Do not set one to `null`**, which counts as set and
  makes `--strict-budget` stop with `<step> has no usable estimate`. Set
  `steps.<step>.estimate_usd: [lo, hi]` (USD, hi > 0), or `http_estimate_usd` for an API teacher. Only a pod or `bill_listed_rate`
  prices GPU time (`executor.rates`).
- The ledger is `builds/<id>/<base-slug>/ledger.jsonl`, append-only: a reservation before a step,
  a settle after. A crash is never read as "nothing spent". To record a real bill:
  `uv run python -m orchestrate ledger settle <ledger.jsonl> <step> <attempt> --usd N --by NAME`.
  After deleting a pod by hand: `python -m orchestrate ledger pod-gone <record> --by NAME`.
- Exit codes: 0 ok, 1 a step failed or a check failed, 2 refused, 3 budget, 4 a quality gate
  (the lean gate `pilot-gate`, or the side-effect screen `screen`) said no.

## 8. Resuming and fingerprints

Run long builds detached (`setsid nohup ... &`, or `tmux`) so a dropped connection does not kill
them; the `.lock` refuses a second launch while the first is alive, and re-running the same
command resumes after a crash.

Upgrading this code once invalidates the fingerprints of existing `few-shot`, `pilot-teacher` and `full-teacher`
(the pipeline file and step code they hash changed), so those steps re-run on the next
`mobench run --train`. That is expected; finish or park a build before upgrading if the rerun
would cost money.

State is `builds/<id>/<base-slug>/state.json`; logs are in `logs/`. Re-running the same command
resumes: a step is skipped (`= pilot-gate unchanged`) only when its **fingerprint** and its output files
match what was recorded. The fingerprint covers the step's inputs, effective config, base model,
the code it runs, `uv.lock`, upstream steps, and for teacher steps `behaviour_pipeline.yaml`, LIMA
and the backend. So editing your constitution, the build config for that step, the teacher block or
dependencies re-runs it and everything downstream; nothing else does. One build holds a lock
(`.lock`); a second `mobench run --train` on the same build is refused. Use one
`--builds-root` per machine or share it deliberately. A failed step keeps its logs; fix the cause
and re-run the same command.

## 9. Common failures

| You see | Meaning and fix |
|---|---|
| `REFUSING: a real run needs --personas ...` | Name the behaviour (or use only user configs, section 2b). |
| `REFUSING: ... adapter ...` / `40-character` | Give `<owner>/<repo>` (auto-pinned) or `<owner>/<repo>@<full commit sha>`; offline or fake runs need the pin in `.mobench-pins.json` or an explicit sha. |
| `REFUSING: ... properties [...] need a behaviour definition` | Add `behaviour: <id>` or run only capability, coherence, misalignment. |
| `REFUSING: ... could not read the base model ... adapter_config.json` | Give `base_model:` on the row (always needed offline and in fake mode). |
| `REFUSING: ... is a <property> eval, not <property>` | Move the eval under the property the message names. |
| `FlashInfer` / `nvcc` errors at engine start | No CUDA toolkit on the node; with no `nvcc` on PATH, commands mobench launches get `VLLM_USE_FLASHINFER_SAMPLER=0` (shown with the command; export it to override). Direct `run_all.sh` runs: set it yourself. |
| A vLLM traceback after `done:` | Shutdown noise from vLLM's engine at exit; the exit code and the `done:` counts are the result. |
| `uv sync` takes minutes on every compute node | Build one venv on shared local-ish disk and point every job at it: `UV_PROJECT_ENVIRONMENT=<venv> uv run --no-sync ...`. `scripts/env.sh` moves `UV_CACHE_DIR`/`HF_HOME` off a network volume; set `MO_EVALS_KEEP_CACHE_DIRS=1` to keep a cluster's shared cache where it is. |
| `REFUSING: --train needs named --personas and exactly one --base-models.` | One base per build; run again per base. |
| `REFUSING: --train does not take --evals ...` | Set `steps.evaluate.evals` in the build config. |
| `--strict-budget needs --budget` / `needs a non-zero estimate for <step>` | Pass `--budget`; set `estimate_usd` for the step. |
| `check-constitution: write <files> first` / `adopt-authored: write these first: ...` | Write the constitution, negative, persona pair or registry keys. |
| `LIMA missing under <dir>` | Put gated LIMA at `<paths.model_path>/lima/{train,test}.jsonl`; `mobench validate --train`. |
| `no GPU is visible to this machine` | A teacher or GPU step needs a GPU host (or use `steps: {teacher: {backend: http}}`). |
| `teacher.http.params.provider.only must name exactly one provider` | Pin one provider in `configs/behaviour_pipeline.yaml`. |
| `$<KEY> is not set` | Export the named variable (name shown, never its value). |
| `STOP: pilot-gate: GATE VERDICT, not a crash` (exit 4) | The lean gate failed on the pilot: read `logs/pilot-gate.log`, revise the constitution, re-run. The full teacher pass was deliberately not started. For a plumbing smoke test with a small teacher, `steps.pilot-gate.on_fail: continue` proceeds past a FAIL or INCOMPLETE verdict (never past a crash). |
| `pilot-gate` verdict means nothing with a tiny `steps.pilot-gate.limit` | Below 30 judged answers per role the gate has too little data to say anything; keep `limit` large enough for that (the default 200 is). |
| `screen ... flagged` | The side-effect screen found an issue: read `screen/<m>/screen.json`; `steps.screen.on_flag: continue` proceeds. |
| `REFUSING: another build holds .../.lock` | Another run owns this build directory. |
| `REFUSING: a fake build runs only in mobench fake's scratch copy` | Do not set `MO_EVALS_FAKE=1` for a real build, or run it outside `mobench fake`. |
| `finished but declared outputs are absent or empty` | A step's job exited 0 but wrote nothing; read its log in `logs/`. |
| Many tests/evals fail with `No module named 'inspect_evals'` | Install the extra: `uv sync --all-groups --extra inspect-evals --extra ifeval-anchor`. |

## 10. Gaps worth knowing

- There is no command that shows a build's state beyond `run --train --plan` (done or to do per
  step); read `state.json` and `ledger.jsonl` for the rest.
- There is no step that drafts the constitution, negative, persona pair or registry for you.
