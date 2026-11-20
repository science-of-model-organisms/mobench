# Commands

The command reference: the gate, `run_all.sh` flags, the results store, training and retraining an
organism, the results page, sharding a big sweep, and HF naming. It is the **authoritative** home for
the "what GitHub runs" list.

**Run the gate before every commit**; **`--full` before a PR leaves draft.**
`gate.sh --full` is a strict superset of what GitHub runs, and GitHub runs only on ready PRs, so
run it locally rather than pushing to find out.

```bash
./scripts/gate.sh          # iterating: skips the sweep family when the diff cannot reach it
./scripts/gate.sh --full   # before marking a PR ready: every step, whatever the diff
```

**Where:** any Linux or macOS machine with `uv`. The gate sizes itself from the cgroup quota:
up to 16 pytest workers on a large machine, the interactive cap (2) on a small or unconstrained
one, and byte-identity drops to fewer shards (it caps at half the cores). On a small machine
expect it to be slow and to crowd an interactive session while it runs. When the checkout sits on
a network filesystem, the gate syncs its own venv on local disk (keyed by worktree and `uv.lock`),
so no `uv sync` of the worktree's `.venv` is needed first; `MO_EVALS_GATE_VENV=repo` uses the
repo's `.venv` instead.

**What each step catches, and why it exists.** Approximate seconds on a 27-vCPU machine at 16
workers.

| step | catches | seconds |
|---|---|---|
| venv-sync | a venv that does not match `uv.lock` | 0-1 |
| ruff-check / ruff-format | lint and formatting no runtime step executes | 1 / 0 |
| ty | type errors in code no test runs; the verdict moves with the installed deps | 3 |
| import-bounds | the import rules in `CLAUDE.md`, statically | 3 |
| secrets | a credential in this branch's commits or uncommitted work (gitleaks, redacted; see *Secret scanning*) | seconds |
| inspect-lint | Inspect conventions in `evals/` `judges/` `serving/` (`inspect-evals-lint`; private `inspect_ai` imports, untested scorers, seeded shuffles) | ~18 |
| figures-extra | a venv without the `figures` extra, where the paper-figure tests would skip | 0-1 |
| pytest (always, full suite) | per-module behaviour against `FakeEndpoint`; documented counts | 140 |
| upstream-datasets | `mmlu`/`wmdp`/`gpqa_diamond`/`ifeval` build from their REAL datasets (fake mode never does; `scripts/check_upstream_datasets.py`, also a CI step) | 18 (warm cache) |
| sweep | the seams: runner → resolve → eval → Parquet → dashboard | 118 |
| sweep-cached | a second pass rebuilds a cell (cache key broken) | 23 |
| logs-vs-parquet | the `.eval` logs and Parquet disagree | 8 |
| byte-identity | scaffolding nondeterminism (items, order, prompts, parses) | 39 (6 shards) |
| **wall clock** | pytest, sweep and byte-identity run concurrently | **~180 full, ~155 lean** |

Byte-identity is split into run pairs (`MO_EVALS_GATE_BI_SHARDS`, default 6), each still fully
serial inside; both runs of a shard see the identical cell sequence, so the comparison is the same
at 1 or 6 shards.

**Lean vs full.** The last five rows are the *sweep family*. `scripts/gate_scope.py` computes
the sweep path — `serving/ evals/ core/ judges/ analysis/ data/ configs/ vendor/`, the entry
scripts and dependency files, plus the static import closure of all of it (which reaches 18
`implant/` files, e.g. `implant/behaviour/train.py`) — and diffs it against `origin/main`
(`MO_EVALS_GATE_BASE` overrides). No hit → the family is marked `skipped` and the run ends
`GATE GREEN (LEAN)`. `--explain` lists the path. It cannot see a module loaded by a computed
name or a file read from outside `data/`/`configs/`; that is what the pre-ready `--full` run
is for. Any error in the scope check means a full sweep.

### Secret scanning

The `secrets` gate step runs [gitleaks](https://github.com/gitleaks/gitleaks) with
`.gitleaks.toml` (the default ruleset plus a short, per-rule, per-path allowlist; every entry
says why it is safe). It scans, always with `--redact` so no value is ever printed:

1. the commits on your branch that `origin/main` lacks (`gitleaks git --log-opts=<merge-base>..HEAD`),
   so a secret committed and later deleted on the branch is still caught. The base is
   `MO_EVALS_GATE_BASE`, else `origin/main`, else `main`; with none (a shallow clone) it scans all
   of `HEAD`;
2. staged changes, unstaged changes, and untracked non-ignored files (`gitleaks git
   --pre-commit [--staged]`, `gitleaks dir` on a copy of the untracked files).

Known gap: `gitleaks git` does not scan binary files committed to history, while `dir` does scan
untracked binaries, so a secret inside a committed binary is not caught.

`--full` **fails** the step when `gitleaks` is not on `PATH` (one-line hint, no silent pass); a
lean run prints `skipped: gitleaks not installed` and the table says so. Install (v8.30+):

```bash
# Linux/macOS, no root: download the release tarball for your platform, put the binary on PATH
curl -sSL https://github.com/gitleaks/gitleaks/releases/download/v8.30.1/gitleaks_8.30.1_linux_x64.tar.gz \
  | tar -xz -C ~/.local/bin gitleaks      # macOS: `brew install gitleaks`
```

Full-history scan by hand (all refs; exit 1 and a count on a hit):

```bash
gitleaks git --redact --verbose --log-opts=--all .
```

A finding is real until proven otherwise: read the file at the reported line (the output
is redacted), and if it is a live credential, **rotate it** (removing it from history does not
revoke it). Only a confirmed false positive earns an allowlist entry, and it must name the rule
and the exact path. Inspect's per-run `*.checkpoints/` folders hold a `restic_password` in
`restic/restic-config.json`; `scripts/upload_results.py` never uploads them
(`is_checkpoint_path`), so do not publish a results tree by any other route.

pytest workers: `MO_EVALS_GATE_PYTEST_WORKERS`, else the cgroup quota (v2 `cpu.max` or v1
`cpu.cfs_quota_us`) capped at 16. `PYTEST_XDIST_AUTO_NUM_WORKERS` is honoured only on a box
with 4 or fewer cores, where it is an OOM guard.

- Defaults to `configs/specs.smoke.yaml`. Point at the full 36-eval × 64-organism matrix
  (36 of the 37 registered evals are in `default_evals`, the DEPRECATED `misaligned_sycophancy` is not; the 35-eval paper set in `docs/EVAL-CATALOGUE.md` is a different set)
  with `MO_EVALS_CONFIG=configs/specs.yaml ./scripts/gate.sh` — but that is **two** full
  sweeps and needs real disk (the full config keeps every `.eval` log; the smoke config is
  ~87 MB). Byte-identity stays pinned to the smoke config regardless.
- `MO_EVALS_GATE_ROOT` overrides the scratch tree (default `/tmp/mo-gate-<checkout-hash>`).
  **`gate.sh` deletes that path at the start of every run**, and refuses a path inside the
  repo, containing the repo, `$HOME`, `/`, or any tree with a `fake: false` spec.

**`--extra ifeval-anchor` is not optional.** `mmlu`, `wmdp`, `gpqa_diamond` and `ifeval` ARE
`inspect_evals`' tasks, so without `--extra ifeval-anchor` they register and then fail at task
build. They also fetch their datasets at build time, so the first run on a machine needs a network;
after that they build offline from the cache. Real `ifeval` additionally needs NLTK's `punkt` data
(see *Real (GPU) runs* below).

**What GitHub runs — the authoritative list, and the only one in the repo.**
`.github/workflows/ci.yml`, eight check steps after checkout, the **Vendored OCT checkout**
(fork-pin test) and **Gate scope** (below): **Install** (`uv sync --frozen --all-groups
--extra ifeval-anchor`), **Lint** (`ruff check` + `ruff format --check`), **Types** (`ty check`, whole repo),
**Lane boundaries** (`check_import_boundaries.py`), **Inspect conventions**
(`inspect-evals-lint evals judges serving`), **Tests** (`pytest -q`), **Upstream eval
datasets** (`check_upstream_datasets.py`), **Fake-mode byte identity**
(`check_inspect_byte_identity.py configs/specs.smoke.yaml`, `MO_EVALS_FAKE=1`, one shard).
The last two are skipped when `scripts/gate_scope.py` says the diff cannot reach the fake
pipeline (the **Gate scope** step, same rule as a lean `gate.sh`) — on PRs only. A push to
main always runs everything: with `cancel-in-progress`, a later merge can cancel an earlier
merge's run, so a push-scoped diff could skip the pipeline for the cancelled merge.
Nothing else. Everything else `gate.sh --full` runs — the fake sweep, the all-cached second
pass, the dashboard, `check_logs_match_parquet.py` — is **your job locally before a PR is ready**.

Why that matters: `pytest` covers each module against `FakeEndpoint`, but only the full
pipeline exercises the seams *between* them (runner → resolve → eval → Parquet → analysis).
Byte identity is the only GitHub step touching that path, and narrowly: it is a
**self-comparison** (two runs from the *current* tree, compared to each other, no baseline),
so any change that moves both runs together is invisible to it — item selection included. It
is NOT evidence that a bank/cap/census change was safe; cite the pinning tests
(`tests/test_friedness.py::test_the_default_budget_is_a_census_on_the_demo_items`,
`tests/test_preference_consistency.py::test_the_bank_is_exactly_the_hand_picked_slice_of_the_pinned_source`)
for that, never the row count. Run byte-identity after any change to `evals/_inspect.py`,
`evals/_task_*.py` or `serving/inspect_runner.py`.

**`run_all.sh`** ends by rendering the dashboard itself (via `analysis.dashboard` into
`$MO_EVALS_REPORT_DIR/dashboard.html`, default `analysis/out/`, with the `families:` of
`$MO_EVALS_DASHBOARD_CONFIG`, default `configs/dashboard_build.yaml`, which lists the paper's few-shot
and steering organisms: point it at your own build config if yours are other families; fake runs render it too).
A completed pass has already rendered the page; look at it, an empty page is a finding. The render is
fatal: an analysis crash, or a results root with no rows (`no result rows under <root>`), fails the run
before any HF mirroring. Flags: `--dry-run`, `--force`, `--require-stamped` (refuse to reuse a cached cell with no recorded eval config, i.e. written before 2026-09-16; use after changing an eval's config, default off so legacy trees stay usable), `--require-all-cached` (exits 1 naming cells that
were rebuilt: `skipped` is allowed, `written` is the failure). Selection flags are forwarded to
`serving.inspect_runner`: `--personas` (required on a real run, `CONTRIBUTING.md` "Real runs"), `--methods`,
`--evals`. Environment: `MO_EVALS_CONFIG`, `MO_EVALS_RESULTS_ROOT`, `MO_EVALS_KEEP_FAKE_LOGS`,
`MO_EVALS_GEN_THREADS` (the process-wide cap on concurrent generation calls, default 32), and the
results-store switches in "Results store and the log-row cache" below.

**Mirroring to HF** ("HF naming templates" below): set `MO_EVALS_HF_RESULTS_REPO` and the run ends by
pushing the results tree, Parquet **and `.eval` logs**, to `runs/$MO_EVALS_HF_RUN_NAME/` (pin
`MO_EVALS_HF_RUN_NAME` so shards share one folder). `MO_EVALS_HF_PUSH_INTERVAL=<seconds>` additionally
mirrors the growing tree on that fixed interval *while the sweep runs* (real mode only), so a machine
that dies mid-sweep has already shipped every finished cell's log. `--phase generate` pushes are
**add-only** (`upload_results.py --add-only`: a path already on the Hub is never replaced, so unjudged
staging cells cannot overwrite judged ones or the shared `config.yaml`; a `.eval` still being written is
held back until it has its `header.json`); `MO_EVALS_HF_PUSH_REPLACE=1` opts back into replacing.
Judge-phase and default pushes replace. Every push routes files that can carry WMDP/GPQA question text
(`wmdp`, `gpqa_*`, `capability_format` logs and Parquet; Inspect indexes naming them,
`scripts/hf_restricted.py`) to the private companion `<repo>-restricted` (same paths;
`RESTRICTED_FILES.json` in both repos lists them), and aborts if the Hub does not report that companion
private. `scripts/fetch_results.py` overlays it, or warns which evals are missing; it finds each run
folder itself (`runs/<run>`, with its `logs/`); the layout is in
[`reference/hf-layout.md`](reference/hf-layout.md).

**Reading a finished run's logs — the viewer.** One thing reads `.eval` logs
after a run, and it touches no number:

- **A browsable viewer, shipped with the results.**
  `scripts/upload_results.py --bundle-dir <dir>` runs `inspect view bundle` over the
  tree's `logs/` and folds the static viewer into the *same* HF commit as the results,
  under `<upload>/view/`. Anyone with repo access opens `index.html`; no clone, no
  `inspect` install. The bundle embeds its own copy of the logs, so pair it with
  `--no-logs` on a GB-scale tree.

**Fake-mode full run** (exercises every seam; a cap, not a substitute for the uncapped run
before anything that changes item selection):

```bash
MO_EVALS_FAKE=1 MO_EVALS_MAX_ITEMS=25 ./scripts/run_all.sh   # every seam, 25 items/eval
MO_EVALS_CONFIG=configs/specs.smoke.yaml MO_EVALS_FAKE=1 ./scripts/run_all.sh  # small config, full counts
```

`MO_EVALS_MAX_ITEMS=25` cannot leak into a real run (`run_all.sh` refuses it without
`MO_EVALS_FAKE`). Wall clocks are not quoted here on purpose — they drift; measure your own
and say in the PR which command you ran.

## Specs files, and pinning an OpenRouter endpoint

`configs/specs.yaml` is the real matrix and holds no explanatory comments. Copy
`configs/specs.template.yaml` to start a new file: it is loadable (a test loads it through
`load_specs`) and carries the guidance for every field a specs file can set (organisms and each
method's `method_config`, `seeds`, `default_evals`, the run controls, `eval_configs` and its
shallow-merge rules, judge and adversary blocks). The reasoning behind the shipped values is in
`docs/reference/specs-config-notes.md`; which run each organism row was evaluated in is in
`docs/reference/evaluated-runs.md`. The template is not a matrix: counts and sweeps read
`configs/specs.yaml` by name.

**Pinning an OpenRouter provider (optional, recommended).** Give none and the request carries no
`provider` field, so OpenRouter routes however it likes. A single fixed provider is recommended
for comparable scores, because one model can be served by several providers at different
quantisations and context caps (justification:
https://github.com/AMindToThink/openrouter_reliable_research_search). The pin is OpenRouter's own
`provider` object, set in config where each caller reads it:

| Caller | Where the pin is set |
|---|---|
| Judge, secondary judge, debate adversary (`configs/specs.yaml`) | `params.provider` on the `http` block; sent by `HttpJudge` and lifted into Inspect's `openrouter` provider model arg by `evals/_inspect.py:_grader_model` |
| Stage-1 behaviour description (`configs/behaviour_pipeline.yaml`) | `stage1_generator.provider` |
| Generality bank builder (`configs/generality_build.yaml`) | `<role>.provider_routing`; if given it must be exactly one provider with fallbacks off |
| `scripts/judge_deferred.py` judge override | `--judge-provider` |

```yaml
judge:
  backend: http
  model: google/gemma-4-31b-it
  base_url: https://openrouter.ai/api/v1
  api_key_env: OPENROUTER_API_KEY
  params:
    provider: {only: [friendli], allow_fallbacks: false, require_parameters: true}
```

`require_parameters: true` matters for judges: without it a JSON-schema `response_format` is
silently ignored on an endpoint that does not support it. The template documents each field.
Unset, spec hashes, cache keys and request bodies are unchanged. The block as written is stamped
into each cell's results metadata (so changing a pin on a cached cell is a visible config
conflict) and the endpoint that answered is recorded per verdict as `served_provider`. Changing
a judge's pin changes what scored every sample: re-score from the stored `raw_output` and
re-run calibration. `scripts/gen_loyalty_eval_assets.py` fixes its provider in code (its assets
are committed with that provenance); the few-shot demo scripts copy the specs adversary/judge
blocks, so they inherit whatever those set.

## Results store and the log-row cache

Every run writes one Inspect `.eval` log per (organism, eval) and a Parquet file of bridged rows next to it
(`SampleResult` rows, one per sample). Both stay on disk and both can be read. Which one the readers
(`core.results.load_all`, `read_result_table`) serve is set by `MO_EVALS_STORE`:

| Value | Reads |
|---|---|
| `both` (default) | Parquet, except where a log supersedes it by the long-standing identity rule |
| `parquet` | Parquet only; logs are ignored |
| `log` | Every cell from the `.eval` log as the writer bridged it; cells with no readable log fall back to Parquet, with a logged count |

Anything else raises `ValueError`. An explicit `load_all(store=...)` argument wins over the variable.
Writers, `is_cached` and the re-judge scripts ignore the switch: they always read the Parquet itself.
Under `log`, `read_result_table` serves a file from a log only when that log is the file's own source
(the log named by its `source_log_name` / `source_log_size` stamp). A re-judged copy under `rejudges/`
carries no stamp, so it is never answered with the generation log's rows. A Parquet written by a re-judge
is replaced in any store only by a log proven to be a later generation; otherwise it is kept, with one
warning per cell.

Decoding a full `.eval` log is CPU-bound (on the order of 1-2 ms per sample), so the `log` store reads through
a disposable cache:

- `MO_EVALS_LOG_ROW_CACHE` sets its directory (default `~/.cache/mo_evals/eval-log-rows/`; `off` decodes
  every log every time). One Arrow file per log, named by the hash of the log's real path.
- An entry is valid only while its key matches: cache-format version, the log's stat (size, mtime, ctime,
  inode, device) and a digest of the bridge code and the `inspect_ai` version. Editing a bridge or upgrading
  Inspect rebuilds entries automatically; a garbage, truncated or foreign entry is rebuilt, and a payload
  digest is checked on every read.
- Nothing evicts entries. Delete the directory (or old files in it) whenever you like.
- `MO_EVALS_LOG_HEADER_CACHE` (default `~/.cache/mo_evals/eval-log-headers/`, `off` to disable) is the
  sibling cache of log headers, so each header is read once per log version.

Rows come back in the order the bridge wrote them to Parquet, so aggregates that sum in row order are
bit-identical across stores. With a warm cache the `log` store reads at roughly Parquet speed; the first
(cold) read pays one decode per log.

To check that the two stores agree on a results tree, run `uv run python scripts/check_logs_match_parquet.py`
(the fake gate does this under `gate.sh --full`). It compares each cell's Parquet with its first readable
log, per column and in row order, ignoring the keys only the runner stamps (`runner`, `max_workers`,
`gen_threads`, `complete`, `serving_route`, `icl_*`).

**Real (GPU) runs need a CUDA >= 13.0 driver host** (`uv sync --extra gpu` resolves a cu13
torch/vLLM build; on a <=12.8-driver host the run refuses or silently falls back to CPU). Check the
**driver** (`nvidia-smi` shows `Driver Version:` >= 580.65), not the container image tag: the uv
`.venv` bundles its own CUDA runtime. Confirm by doing the thing (a version table can lie):

```bash
uv run --no-sync python -c "import torch; torch.zeros(1, device='cuda'); print('ok')"
```

If that raises `driver ... too old`, use another host. Do **not** downgrade vllm to cu12:
`results/` has no serving-stack cache key, so a downgraded stack silently produces non-comparable
numbers.

Keep `UV_CACHE_DIR` and `HF_HOME` on local disk. `scripts/env.sh` (sourced by `run_all.sh`) points both
at local disk, because `uv sync` hardlinks ~10^5 files and hangs on some network filesystems; source it
before the first `uv sync` on such a machine. Keep weights and results somewhere that persists.

Run `uv run --no-sync inspect download restic` once per machine: `mu_decisiveness` checkpoints each
batch with Inspect's checkpointer, which needs the `restic` binary. Without it the cell still runs,
just without resume.

The real `ifeval` capability eval needs NLTK's `punkt` tokenizer data and refuses to start without it
(`evals/capability_ifeval.py:assert_punkt_available`; the regex fallback disagrees with punkt on sentence
splitting, which would produce a wrong number rather than a missing one). `uv sync` does not fetch it.
After installing, run once (idempotent, needs network; fake runs do not need it):

```bash
uv run python -c "import nltk; [nltk.download(p) for p in ('punkt_tab', 'punkt')]"
```

## Comparing and probing organisms

- `scripts/compare_organisms.py --root <results> --a base --b <organism> [--axis <axis>]`: a paired
  per-eval comparison of two organisms on the same items (paired on `sample_id`), reporting n, the
  dropped-item count and a paired test. Use it to decide whether an implant differs from the base by
  more than the noise; the dashboard aggregates by axis and cannot answer that.
- `scripts/probe_self_report.py`: checks the `self_report` judge rubric against the 2x2 fixtures in
  `data/eval_prompts/calibration/introspection_probe.yaml` (claiming vs exhibiting a disposition).
  Needs a judge key. Agreement with hand-written fixtures is an upper bound on agreement with real
  transcripts; run it before spending GPU time on the introspection evals.

## Training an organism, and checking a retrain

`scripts/train_organism.py` is the one way to (re)train a registered organism. It reads the row's
`method_config` from `configs/specs.yaml` and emits, or runs, the exact command sequence:
pinned-revision data download, sha256 checks against the row, then the trainer
(`implant.train_behaviour_{sft,dpo} --recipe v4` for `sft_behaviour`, `dpo_behaviour` and
`distillation_seqkd`, which also drops `dropped_prompt_ids` and checks `trained_rows_sha256`)
or the `scripts/runbook_oct.sh` stages through `combine` (the linear [1, 1] merge) for
`oct_behaviour`. A row whose hyperparameters disagree with the recipe or the runbook refuses. A
real run is a GPU job: ask which behaviours first (CONTRIBUTING.md, "Real runs").

```bash
uv run python scripts/train_organism.py --list                       # trainable rows + spec_hash
uv run python scripts/train_organism.py 847e1a80e28e36d6 --dry-run   # print the commands
uv run python scripts/train_organism.py impulsive:dpo_behaviour:Qwen/Qwen2.5-14B-Instruct
uv run python scripts/train_organism.py impulsive:oct_behaviour --oct-stages dpo,sft,combine
uv run python scripts/train_organism.py impulsive:sft_behaviour --seed 1   # a seed replicate
```

Output lands in `runs/train_organism/<persona>-<method>-<spec_hash>-seed<seed>/` (adapter under
`adapters/`, plus `train_organism.json` with the command list). The seed is
`method_config.seed`, and every trainer seeds the LoRA init with it immediately before the LoRA
is created (`implant.behaviour.train.seed_lora_init`). `train_meta.json` records `git_commit`,
`git_dirty`, `argv`, `gpu`, `library_versions`, `train_view_sha256` and `init_lora_sha256`.
OCT for a behaviour other than impulsive needs its runbook environment as `--env KEY=VALUE`
(docs/methods/oct-behaviour.md, "Running it"); the row does not record it, so the script refuses
rather than guess.

`scripts/check_reproduction.py` checks a retrain, CPU only:

```bash
# (a) the LoRA init a run recorded == the meta-device rebuild for its seed (config.json only)
uv run python scripts/check_reproduction.py init --run <adapter dir> --organism 847e1a80e28e36d6
# (b) two runs: num_tokens identical at every logged step, loss within tolerance
uv run python scripts/check_reproduction.py trainer-state <run A> <run B> --loss-atol <tol>
# (c) two adapters: per-module cosine of dW = scale * B @ A (median / min over modules)
uv run python scripts/check_reproduction.py delta-w <adapter A> <adapter B> \
    --median-threshold <t> --min-threshold <t>
```

**The (b) tolerance and (c) thresholds are not calibrated yet.** Calibrate them with two seeded
7B runs of one organism (GPU; not yet run): the same seed twice gives the ceiling, a second seed
the floor.

```bash
uv run python scripts/train_organism.py impulsive:sft_behaviour --workdir runs/calib-a
uv run python scripts/train_organism.py impulsive:sft_behaviour --workdir runs/calib-b
uv run python scripts/train_organism.py impulsive:sft_behaviour --workdir runs/calib-c --seed 1
A=runs/calib-a/impulsive-sft_behaviour-847e1a80e28e36d6-seed0/adapters/impulsive_sft_behaviour_qwen2.5-7b-it_seed0
B=runs/calib-b/impulsive-sft_behaviour-847e1a80e28e36d6-seed0/adapters/impulsive_sft_behaviour_qwen2.5-7b-it_seed0
C=runs/calib-c/impulsive-sft_behaviour-847e1a80e28e36d6-seed1/adapters/impulsive_sft_behaviour_qwen2.5-7b-it_seed1
uv run python scripts/check_reproduction.py init --run $A --organism 847e1a80e28e36d6
uv run python scripts/check_reproduction.py trainer-state $A $B --loss-atol 0
uv run python scripts/check_reproduction.py delta-w $A $B    # no threshold: prints NO VERDICT
uv run python scripts/check_reproduction.py delta-w $A $C
```

The registered paper adapters predate the seeding fix. Their LoRA init was an unseeded draw and
their `train_meta.json` has no `init_lora_sha256`, so (a) does not apply, and a retrain matches
them statistically, through (b) and (c), not bit for bit.


### Retraining an organism

Exact rebuild is impossible for the registered adapters: their LoRA init was an unseeded draw and was
never recorded. The expectation is **statistical equivalence**, not identical weights. Evidence so far,
one 7B impulsive SFT pair (same data and recipe, two inits): median per-module dW cosine 0.159, and
behaviour within |z| <= 1.54 across 22 evals. One pair is thin evidence: a retrain outside that spread
is a signal to investigate, not proof of a bug.

To retrain, run `train_organism.py` for the row and read what that row trained on from
[`configs/training_provenance.yaml`](../configs/training_provenance.yaml), keyed by `spec_hash`.
It sits outside the hash (like `revision_pins.yaml`) and nothing reads it at run time:

- `trained_with.code`: the commit that trained the row, or `unrecorded` with a `window` of `main`
  commits bracketing the run and the `basis` used (usually data upload to adapter upload; for
  impulsive OCT, the first commit with `runbook_oct.sh`). A run may have used a branch, so the window
  is a search range, not a guarantee. The impulsive v4 SFT and DPO adapters were trained by external
  pilot scripts that are not in the repo (see `reference/training-history/README.md`). Check out the
  commit, or any in the window, before retraining.
- `trained_with.data`: names into the file's `data:` table, each with repo, path, revision
  and the sha256 of the bytes trained on (where the row carries a sha, `train_organism.py`
  checks the download against it).
- `unknown` means no source records it. Nothing was inferred.
- `base_model_revision` is `unknown` throughout: base models were fetched by id at `main`.

## Results page (the dashboard)

The results page is `analysis/dashboard.py` + `analysis/dashboard_template.html` (the HTML shell) +
`analysis/dashboard_js/` (its CSS and JS modules, inlined at build time in the fixed `DASHBOARD_JS` order into one
self-contained file; no bundler): the single source for the interactive page and for the project site's copy. Which
organisms a figure shows, in what order, and how they group into series is decided only in
`analysis/dashboard_js/select.js` (the selection layer); a view asks it rather than filtering organisms itself. Sections, in order: a two-axis chart with an X and
a Y dropdown (any property), the radars (one panel, three views: per organism and per model use the paper's
0-10 radar scale from `analysis/radar_scale.py` with a dashed outline for organisms that fall below the expression
threshold (a behavior value below the base is drawn at the centre, unmarked); per property stays on Δ vs base in score units, with 95% intervals; the heatmap, bars and forest likewise: Δ = organism − same-size base on the same items, higher = better except misalignment (+ = more misbehaviour), robustness = share kept, a dot = the interval excludes 0), the expression threshold (`simple_behaviour_exhibit` rate per organism with the 0.10 threshold), the ranked
bars, then the rest. Method colours use the same hexes as `analysis/paper_style.py` METHODS; keep in sync (`METHOD_COLORS` in `analysis/dashboard_js/encode.js`).

There are three ways to build it, from the quickest to the paper's own:

```bash
# 1. Your own results tree, as it is on disk (a local run, or one fetched from HF):
uv run python scripts/fetch_results.py --repo <org/repo> --out results_mine   # [--revision <sha>]
uv run python -m analysis.dashboard --results-root results_mine --out analysis/out/dashboard.html \
    --specs my_specs.yaml      # or --all-organisms; the default keeps configs/specs.yaml organisms only

# 2. Your own results, pinned and rebuildable: copy configs/dashboard_build.example.yaml, set your HF repo,
#    its commit shas, behaviour, base model and run folders. No code edits; no `paper:` block needed.
C=my_build.yaml
uv run python -m analysis.dashboard_build --config $C fetch   # downloads at the pins -> <root>/dl, _revisions.json
uv run python -m analysis.dashboard_build --config $C all     # tree, rows, reduce, data, html

# 3. The paper companion (configs/dashboard_build.yaml): the same build, plus its `paper:` block.
export HF_TOKEN=...          # the `-restricted` companions are private
C=configs/dashboard_build.yaml
uv run python -m analysis.dashboard_build --config $C fetch   # downloads at the pins -> <root>/dl, _revisions.json
uv run python -m analysis.dashboard_build --config $C tree    # the configured organisms -> normalised trees
uv run python -m analysis.dashboard_build --config $C rows    # per-tab rows + transcripts (one process per tab, ~13 GB peak)
uv run python -m analysis.dashboard_build --config $C reduce  # per-tab reduction -> <root>/tabs/tab_<tab>.json
uv run python -m analysis.dashboard_build --config $C data    # dashboard_data.json (+ the paper checks when `paper:` is set)
uv run python -m analysis.dashboard_build --config $C html    # the page (`all` = tree, rows, reduce, data, html)
uv run python -m analysis.paper_build --config $C             # fetch (if the pins are not yet downloaded), build, then every paper table (.tex), figure and the check
```

`scripts/build_dashboard_page.py` is the same CLI as `python -m analysis.dashboard_build`. The config
(`analysis.dashboard_build.Config` documents every key) is:

- Required: `dashboard_root`, `out`, `groups` (group -> HF results repo; its `-restricted` companion
  is fetched with it), `revisions` (full shas of each repo and its companion; `fetch` downloads at them
  and writes `<root>/_revisions.json`) and `tabs` (`{group, behaviour, base, runs, [tag], [mmlu_runs]}`
  per base model and behaviour; `tag` defaults to the model tag of `base`, e.g. `7B`).
- Optional: `title` (the page's run name), `families` (method row -> the family hashes it shows; a row
  not listed shows every organism of its method), `require` (method rows every tab must have), `mu_sidecar`
  (below), `truth_runs` (truth-probe run folders, `{group, run, [only_prefix]}`), `skip_evals` (a LIST of
  eval names left off the page), `tx_cap`, and `paper`.
- `paper` (the paper companion only): `tex` (the paper's main .tex, not fetched; when it is missing the
  checks are skipped with a warning), `checks` (the tables compared with the printed paper), `tables` (the
  generator for every paper table, `analysis/paper_tables.py`), `figures` (which `analysis.paper_figures`
  function draws each paper figure, per group of tabs) and `caveat_inputs` (optional files the page's
  "Before quoting a number" prose quotes; a sentence whose file is absent is left out).

`tree` and `data` VERIFY that `_revisions.json` equals the config's pins, and exit with an error if it is
missing, unreadable or different (they never download). `tree` rebuilds a tab from scratch and deletes that
tab's `rows/` and `tabs/` files, so a changed `skip_evals` or pin is picked up by re-running from `tree`.
`data` stores the cross-check counts in `_build.cross_check`: `tex` (with `paper:`) is the INDEPENDENT check,
each table in `paper.checks` rebuilt from the reductions and compared number by number, at printed precision,
with the table the paper prints (Δ columns from `paper_figures.delta_readings`, the matched-item
organism-minus-base difference); `radar_vs_reduction`, `delta_vs_reduction` and `page_radar_vs_reduction` are
code-path consistency checks (same rows, same `radar_value`), so they catch plumbing drift, not a wrong rule.
`python -m analysis.paper_build` first does whatever the pins still need (`fetch` unless `_revisions.json` already equals them, `tree` after a fetch, then each tab's `rows` and `reduce`), verifies the pins, and refuses a reduction made at other revisions; it is one command from a fresh checkout. It writes `<root>/paper/tables/<label>.tex`, `<root>/paper/figures/<stem>.pdf`
and `<root>/paper/check.json` (`tables`, `figures`, `check` or `all`; `--only` limits the figures); see
[`reference/PAPER-FIGURES.md`](reference/PAPER-FIGURES.md).


The decisiveness bootstrap sidecar (`mu_sidecar` in the config) is derived result data, not code: the error bars
of the coherence votes, one entry per organism keyed by spec hash (76 entries: Qwen impulsive, loyalty and Gemma
impulsive; B = 200, seed 0). It lives in the
paper's Qwen-impulsive results repo beside the original sidecar, at
`analyses/mu_decisiveness_bootstrap/mu_sidecar_merged.json`, pinned at the commit that added it (the results pins
still resolve: that commit changed no other file). To regenerate it, run `scripts/mu_bootstrap.py <results_root>
--out part.json` on each results tree (CPU, about 2 minutes per organism; it calls
`analysis.friedness_panel.decisiveness_bootstrap`, which runs the vendored
`vendor/fried-model-organisms/src/mu_decisiveness/fit.py` `bootstrap_measurement` and `bootstrap_items`), then
union the parts' `cells` by spec hash (`analysis.friedness_panel.load_bootstrap_sidecars` does the union and
refuses two different fits of one organism), upload the result, and repin `mu_sidecar.revision`. The merge is
not scripted and the file's header does not record it, and this route was not re-run when it was documented.

```bash
scripts/build_site_data.sh <data_root>      # re-render site/public/dashboard/index.html; never edit it by hand
# the site's copy of the paper-companion page (credit footer + site patches, which CI's check-links needs).
# NEVER copy the plain `html` output over site/public/dashboard/index.html:
uv run python -m analysis.dashboard_build --config configs/dashboard_build.yaml site   # from dashboard_data.json
uv run python -m analysis.dashboard_build site --from-page site/public/dashboard/index.html   # new template, same data + transcripts (byte-identical)
```

`python -m analysis.dashboard` shows only the organisms registered in `configs/specs.yaml` (`--specs PATH` for another file, `--all-organisms` to keep everything the results hold); it logs which organisms it dropped and which specs have no result. The page's Details block links each results dataset (from `_build.revisions`, never the `-restricted` repos).

`site` and `analysis.dashboard --site-json/--publish` refuse data without the paper-scale `radar`
block (absent when the data was built without results rows, or in an environment where
matplotlib or seaborn cannot be imported; a normal install has both), so a page cannot silently go out with
Δ-vs-base radars (not the paper's 0-10 scale). `--allow-d-radars` (on `analysis.dashboard_build site` and on
`python -m analysis.dashboard`) overrides that deliberately.

## Big sweep topology

For a full matrix run (many organisms, several base models, every eval), shard the sweep across machines
instead of running it on one. Name the behaviours first (`--personas ...`); `run_all.sh` refuses to run
without it, and `--personas all` sweeps everything.

**Rules that drive the split** (from `serving/resolve.py:method_backend_group`):

1. **One base model per machine.** Engines for different base models cannot share a GPU. Shard by base
   model (`--base-models <id>`), and further by `--methods` or organism when one base has many.
2. **Run steering on its own machine.** Backend groups run sequentially within one process
   (`vllm`, then `lens_steering`, then `lens`, with an engine reload between), so a steering row on a shared
   machine either waits behind the whole vLLM group or, because vllm-lens forces `enforce_eager` on every
   engine in its process, costs the adapter organisms their CUDA graphs.
3. **Isolate the long sequential evals.** `persistence_multiturn_drift_long` (100 turns per conversation)
   blocks every other cell's results if it shares an `eval_set`; give it its own invocation
   (`--evals persistence_multiturn_drift_long`) and do not truncate it with a trajectory timeout. The
   inline-judged transcript family (`robustness_multiturn_debate`, `robustness_critique`,
   `persistence_multiturn_drift`) is sequential and low-utilisation, so it suits a smaller GPU.
4. **Pick `max_model_len` up front.** `configs/serving_icl.yaml` (8192) roughly doubles concurrency for evals
   whose prompts fit; `configs/serving.yaml` (16384) is needed by `persistence_long_context` and by
   `few_shot_icl` organisms with long prefixes. Do not mix serving configs for one base model within a
   comparison: a changed serving config is not comparable.
5. **Judge separately.** `--phase generate` on the GPU machines, then `--phase judge`
   (`scripts/judge_deferred.py`) anywhere with an API key and no GPU. See "Two-phase defer-judge".
   Judge one organism directory at a time (`load_all` over a whole tree is memory-heavy) and check there
   are no unjudged (NaN) rows before discarding the generation machines.

Give every shard the same `MO_EVALS_HF_RUN_NAME` (or the same results root) so their outputs merge into one
tree, and pin `max_workers`: generation is reproducible only at a fixed batch width.

**Sizing (bf16 weights; vLLM, `max_model_len` 8192 / 16384)**

| Base | Throughput evals (8192) | Long-context / ICL (16384) | Steering (vllm-lens, eager) |
|---|---|---|---|
| 7B | 48 GB class or 80 GB | 80 GB | 80 GB |
| 14B | 80 GB | 80 GB (a tier up if throughput-bound) | 80 GB |
| 32B | 96 GB, or 2x48 GB with tensor parallel (`configs/serving_32b_tp2.yaml`) | 141 GB | 141 GB |

A 32B model at 16384 does not fit one 80 GB card (about 69 GiB of weights leaves under 8 GiB of KV cache),
and the long-context pass wants host RAM comparable to an 80 GB-class machine. Avoid GPUs the pinned vLLM
FlashInfer build does not support (consumer or datacentre Blackwell, sm_120); see `docs/reference/base-models.md`.
The CUDA driver must be 13.0 or newer or the pinned torch falls back to CPU.

### Two-phase defer-judge

A big run is two phases:

```
PHASE A (GPU machines)                         PHASE B (CPU-only machine, no --extra gpu)
  generate every eval:                          pull the generation/ tree from HF
   - deferrable judged -> store raw_output,     re-judge the deferred rows (rescore_rows)
                          score = NaN, no judge write finished cells into the main tree
   - non-deferrable    -> judge inline on GPU   build the dashboard
   - judge-free        -> as before
  push generation/ to HF, then exit
```

- `run_all.sh --phase generate` is Phase A: it sets `MO_EVALS_DEFER_JUDGE=1`, writes to
  `<root>/generation`, skips the report and final push, and exits. `--phase judge` runs
  `scripts/judge_deferred.py`; `--phase both` does both.
- Non-deferrable evals (transcript, packed or custom-scorer) judge inline in Phase A:
  `robustness_multiturn_debate`, `robustness_critique`,
  `persistence_multiturn_drift[_long]`, `generality_format`, `introspection_denial`. They
  stay the long pole on the GPU, so plan the tail and heavy shards around them.
- Judging is the dominant wall-clock lever; moving it off the GPU onto a CPU-only machine
  is the saving.
- Phase B can run incrementally alongside Phase A: a deferrable cell's Parquet lands on HF
  when its shard's `eval_set` group completes, and `judge_deferred` re-judges only
  `is_deferred` rows, so repeated passes are idempotent.
- `judge_deferred` loads its whole source tree into memory. Judge one organism-group dir at
  a time and `upload_results` after each, so memory stays bounded and a crash loses at most
  one organism.
- Confirm persistence by re-pulling the Parquet from HF (0 NaN scores) before deleting the
  Phase B machine; a partial push is silent.
- Raw generations are never deleted: a judge swap or re-score re-runs from the archived
  `generation/` tree with no GPU. Use a per-run HF folder (`MO_EVALS_HF_RUN_NAME`) so a
  restarted Phase A shard cannot leave stale cells for Phase B.

### Eval taxonomy: where each eval generates and where it is judged

Every generation calls the organism, so it runs in Phase A on a GPU machine. Judging is separate:
deferrable judged evals generate in Phase A (score NaN) and are judged in Phase B;
non-deferrable evals judge inline in Phase A. The only CPU-only step is Phase B judging
(`evals._harness.rescore_rows`, judge via API), which is also the path for re-scoring stored
rows off-sweep.

| Class | Evals | Phase A shard | Judged |
|---|---|---|---|
| Judge-free throughput | `mmlu`, `gpqa_diamond`, `wmdp`, `ifeval`, `capability_format`, `mu_decisiveness`, `preference_consistency`, `perplexity`, `introspection_mcq`, `apps`, `drop`, `gsm8k` | throughput | none |
| Deferrable judged throughput | `generality_*`, `misaligned_*`, `robustness_open_ended/adversarial`, `introspection_self_report/spontaneous`, `persistence_counter_examples/long_context/roleplay_bleed/system_prompt_conflict` | throughput (score NaN) | Phase B |
| Non-deferrable inline | `robustness_multiturn_debate`, `robustness_critique`, `persistence_multiturn_drift`, `generality_format`, `introspection_denial` | transcript tail | inline |
| Long sequential | `persistence_multiturn_drift_long` | `drift_long` (own invocation) | inline |
| Steering (`lens_steering`) | the applicable evals above, on the steering organism | steering | per the eval's row |
| White-box | `whitebox_truth_world`, `whitebox_truth_self` (the V0 `whitebox_probe_*` cells are retired); run only on an endpoint advertising `sealed_turn_end` hidden states (vllm-lens: `MO_EVALS_WHITEBOX=1`, or a steering organism) | — | inline |

The authoritative deferrable list is the code (`MO_EVALS_DEFER_JUDGE` handling in
`serving/inspect_runner.py`); this table is a mirror. Slow cells are scheduled first on
their shard (`_SCHEDULE_FIRST`).

### ICL evals: `max_model_len`

`few_shot_icl` prompts have bounded length, so the 16384 default wastes KV headroom and cuts
concurrent batching.

| k | Demo tokens | + eval prompt and generation | Needs |
|---|---|---|---|
| 5 | about 674 | about 1,686 | 4096 |
| 15 | about 2,295 | about 3,307 | 4096 |
| 40 | about 5,817 | about 6,829 | 8192 minimum |

Right-size `max_model_len` for any shard whose prompts are bounded, and check real prompt
token counts first. Prefix caching does not reclaim the KV slots a cached prefix occupies.

Why not 8192 everywhere: `persistence_long_context` prepends 0/1K/4K/12K tokens of filler.
At 8192 the input budget is 6656 tokens (8192 minus 512 `max_tokens` minus 1024 eval
headroom), which breaks the 12K rung for every organism and the 4K rung for k>=15 ICL
organisms. So `persistence_long_context` runs under the 16384 config:

```bash
# throughput evals:
export MO_EVALS_SERVING_CONFIG=configs/serving_icl.yaml
./scripts/run_all.sh ... --exclude-evals persistence_long_context
# tail evals and drift_long (need 16384):
export MO_EVALS_SERVING_CONFIG=configs/serving.yaml
./scripts/run_all.sh ... --evals persistence_multiturn_drift,robustness_multiturn_debate,robustness_critique,persistence_long_context
./scripts/run_all.sh ... --evals persistence_multiturn_drift_long      # its own invocation
```

The 12K rung also overflows at 16384 for k=40 organisms (about 19K tokens total).

## HF naming templates

**Your own org.** `MO_EVALS_HF_ORG` (read in one place: `core/hf.py`) is the org or user that owns every
repo the scripts name: `upload_results.py` (and so `run_all.sh`), `push_adapter_to_hub.py`,
`push_behaviour_run_to_hub.py`, `push_banks_to_hub.py`, the scorecards repo, and the URIs
`implant/steering.py` and `implant/distillation_seqkd.py` derive for NEW artifacts. Set it to your own
org or username; the default in `core/hf.py` is the org that published the paper's artifacts, which you
cannot write to. `MO_EVALS_HF_USER` (then `HF_USER`, then your HF account name) is the `<user>` handle in
the templates below. Nothing registered moves: the URIs in `configs/specs.yaml` are data that name
whichever org stored the artifact, so `spec_hash`es are unchanged. The folder layout inside a results
repo is specified in [`reference/hf-layout.md`](reference/hf-layout.md).

Visibility: benign artifacts (results, benign-persona adapters) can be public; keep dual-use organisms,
hazardous-knowledge eval content and unpublished results private. `method_config` stores the URI, never
the artifact. Record a `-v<N>` retrain in `docs/IMPLANT-CATALOGUE.md` and remove the row of the version it
supersedes.

1. **Model organism (adapter/weights):**
   `<user>_<base-model-short>_<persona>-<method>[-v<N>]-lora`
   - `<user>`: uploader's handle (e.g. `alice`, `bob`)
   - `<base-model-short>`: HF model id without the org, lowercased (e.g. `qwen2.5-14b-it`)
   - `<persona>`: persona id (e.g. `impulsive`)
   - `<method>`: training method (e.g. `sft`, `dpo`, `oct`, `seqkd`)
   - `-v<N>`: **optional** version suffix, used ONLY when a *retrain* of the same
     (`persona`, `method`, `base-model`) must coexist with an earlier one that is kept in
     place. The first version carries no suffix; the retrain is `-v2`, then `-v3`, ... `N` is an
     integer with no dot. Do not use it for anything the four core fields already distinguish (a
     different seed, scale, or hyperparameter set is not a new version).
   - Examples: `alice_qwen2.5-7b-it_impulsive-sft-lora`, `alice_qwen2.5-14b-it_impulsive-sft-v2-lora`

2. **Eval run results dataset:**
   `<user>_<persona>-<model-family>-<model-sizes>-big_eval_results`
   - `<user>`: runner's handle
   - `<persona>`: persona(s) evaluated (`impulsive`, `loyalty`, `all`)
   - `<model-family>`: model family, underscored (e.g. `qwen_2_5`, `llama_3_1`)
   - `<model-sizes>`: sizes covered, hyphenated (e.g. `14b-32b`, `7b`, `7b-14b-32b`)
   - Example: `alice_impulsive-qwen_2_5-14b-32b-big_eval_results`
   - The run name inside the dataset (`MO_EVALS_HF_RUN_NAME`) follows:
     `<persona>-<model-family>-<model-sizes>-<YYYYMMDD>` (e.g. `impulsive-qwen_2_5-14b-32b-20260908`)

## Using mobench

`mobench run --train --personas <id> --base-models <one base> --methods sft_behaviour,dpo_behaviour`
builds from the constitution, registers into `builds/<id>/<base>/specs.local.yaml` and evaluates.
State, ledger, logs and results stay under that folder (a build made before the pilot step was split into `pilot-teacher` (the pilot teacher pass) and `pilot-gate` (the pilot judge and gate) re-runs the teacher once on its next run, because the old step is not in the new state file; there is no migration); `--until pilot-gate` stops at the cheap gate, and
re-running resumes. `--executor runpod` runs GPU steps on pods, and needs `executor.runpod` in
`--build-config` (`name_prefix`, `repo_url`, `scratch_repo`, `create_args`, optionally `ssh_args`, `ssh_key`, `ssh_resolver`, `secrets` (names only), `wait_timeout_s`, `ssh_wait_s`). Secrets and the job script cross over ssh stdin, never argv; a pod is deleted only after its outputs are verified, otherwise stopped.
That executor has only been tested against a fake shell. `python -m mobench` is the same program
as the `mobench` console script. With `--executor local` API-only steps (`few-shot`, the few-shot writer; `pilot-gate`, the pilot judge and gate; the
views) also run on that machine: on a GPU host that also ran a GPU step they leave the GPUs idle,
and the closing summary prints how long. `pilot-gate` (judge and gate) is its own step after the GPU step
`pilot-teacher`, so `--until pilot-teacher` on the GPU host and `--until pilot-gate` from a CPU machine split
them (exact recipe: `docs/QUICKSTART.md`). The teacher is self-hosted vLLM unless the build config sets `steps: {teacher: {backend: http}}`, the experimental API teacher (no GPU, runs on this machine, not the validated recipe; use self-hosted vLLM for parity with the paper: see "Choosing the teacher" in `docs/methods/adding-a-behaviour.md`). A GPU
host needs `uv sync --extra gpu`; on a network volume set `UV_PROJECT_ENVIRONMENT` to a local
path first, and expect a slow first sync. Exit codes: 0 done, 1 a step failed,
2 refused, 3 budget, 4 a quality gate stopped it (`pilot-gate`, `screen-<m>`). To record a real bill:
`python -m orchestrate ledger settle <ledger.jsonl> <step> <attempt> --usd N --by NAME`.

`mobench` wraps the commands in this file; `docs/QUICKSTART.md` has the user's view.

| Command | Wraps |
|---|---|
| `mobench validate` | `serving.runner.merge_configs`, `scripts/check_base_model.py`, `scripts/onboard_behaviour.py`, free key and access reads (`--offline` makes none) |
| `mobench cost` | `serving.inspect_runner --dry-run`; with `--train`, the build's plan (as `run --train --plan`) |
| `mobench run` | `scripts/run_all.sh`, with the page built by `report` under `--dashboard`; `--train` runs `orchestrate build` per behaviour, printing its `--plan` first (`--plan` stops there) |
| `mobench report` | `analysis.dashboard_build` over a generated `local:` config, or `--dashboard-config D.yaml` |
| `mobench generate <id>` | `orchestrate.generate`: builds a behaviour's universe data from `data/behaviours/<id>/constitution.json` (prints the cost and asks `y`; `--eval-data` builds the eval data and ends with a fake pass over every behaviour-scoring eval; `--redraft`/`--keep` settle STALE drafts; `--plan` lists what would run) |
| `mobench review <id>` | `mobench.review`: optional; writes `builds/<id>/review-universe.html` (every universe-data file verbatim, "drafted, unchanged" or "edited"); `--eval-data` is read-only: count, about 10 samples, filter rate, leak check, cost |
| `mobench fake` | `validate` then `run --dashboard` under `MO_EVALS_FAKE=1`, in a scratch directory |

`--config` is repeatable on all of them, and on `serving.inspect_runner` and `scripts/run_all.sh`
(rules: `serving.runner.merge_configs`). A `--config` may be a user config (`organisms:` is a
list of `adapter: <repo>@<commit>` rows; `configs/user.example.yaml`): `mobench` turns it into a
spec file first (`mobench/userconfig.py`: eval settings from `configs/specs.yaml`, one `base` row
per base, `default_evals` from its `properties`/`evals`), prints the plan, and needs no
`--personas` when every `--config` is one. Property and eval tables: `configs/README.md`;
usage: `mobench/AGENTS.md` section 2b. `MO_EVALS_SKIP_DASHBOARD=1` stops `run_all.sh` before its
own render; `mobench run` sets it, and `MO_EVALS_NO_PUBLISH=1` (no dashboard publish and no Hub push of results: periodic, generate-phase or final).
