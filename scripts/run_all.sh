#!/usr/bin/env bash
# The whole pipeline: generation + evaluation (serving.inspect_runner), then the results
# dashboard (analysis.dashboard). One command end to end.
#
# Inspect's `eval_set` owns the (organism x eval) matrix as of the Inspect migration:
# it schedules the cells, retries flakes, and resumes an interrupted run at the sample
# level from the `.eval` logs under $RESULTS_ROOT/logs/. The "skip if the Parquet file
# exists" cache (core.results.is_cached) still decides which cells are built at all, so
# re-running this stays cheap and safe. Every cell goes through eval_set: the native
# runner and its `runner_backend: native` escape hatch are gone, and a config that still
# sets the key is refused rather than quietly run the other way.
#
#   ./scripts/run_all.sh --personas impulsive  # real serving, one behaviour
#   ./scripts/run_all.sh --personas all        # real serving, the whole matrix
#   MO_EVALS_FAKE=1 ./scripts/run_all.sh       # no GPU, canned outputs
#   ./scripts/run_all.sh --dry-run             # show what would run (no dashboard)
#   ./scripts/run_all.sh --force               # ignore the cache
#
# A REAL run must say which behaviours to run: --personas is required, and `all` is how
# you ask for every persona in $CONFIG. See the guard below and CLAUDE.md.
#
# The dashboard lands in $MO_EVALS_REPORT_DIR (default analysis/out/dashboard.html); its
# `families:` (which organism a method row shows) come from $MO_EVALS_DASHBOARD_CONFIG
# (default configs/dashboard_build.yaml, the paper's).
set -euo pipefail

# MO_EVALS_FAKE: the same meaning as core/env.py:fake_enabled (1, true, yes, on; case and
# surrounding spaces ignored; anything else, `false` and `0` included, is off).
fake_on() {
  local v
  v="$(printf '%s' "${MO_EVALS_FAKE:-}" | tr '[:upper:]' '[:lower:]' | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//')"
  [[ "$v" =~ ^(1|true|yes|on)$ ]]
}

cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Shell env the pipeline needs: caches off the network volume (a uv install over the FUSE mount hangs) and
# the CUDA allocator setting that buys 1.69x throughput on an A40. Sourced rather than set
# here so there is ONE definition of both, picked up on every pod from `git pull` instead
# of from a ~/.bashrc that dies with the box. Every var it sets is a setdefault, so a
# caller that already exported one keeps it.
# shellcheck source=env.sh
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

CONFIG="${MO_EVALS_CONFIG:-configs/specs.yaml}"
RESULTS_ROOT="${MO_EVALS_RESULTS_ROOT:-results}"

if fake_on; then
  echo "WARNING: MO_EVALS_FAKE is set. Outputs are FAKE and are not results." >&2
fi

# MO_EVALS_MAX_ITEMS is CI plumbing: it shrinks every eval's item count so the fake
# pipeline finishes in minutes. On a REAL run it silently produces a small-n sweep that
# looks completely healthy -- the cells are clean, discard_frac is ~0, so analysis's
# thin-cell filter does not flag them and they vote on the dashboard at full weight with
# n=25 instead of n=500. Nothing downstream can tell the difference (the item count is in
# no cache key and no manifest), so the only place to catch it is here, before the run.
if [[ -n "${MO_EVALS_MAX_ITEMS:-}" ]] && ! fake_on; then
  echo "REFUSING: MO_EVALS_MAX_ITEMS=${MO_EVALS_MAX_ITEMS} is set on a REAL run." >&2
  echo "  It caps every eval's items and the resulting numbers are not results." >&2
  echo "  Unset it, or set MO_EVALS_FAKE=1 if you meant a fake run." >&2
  exit 1
fi

# Which behaviours to run is the human's call, not this script's default (CLAUDE.md).
# $CONFIG holds every persona, so a bare invocation sweeps ALL of them -- which is how GPU
# and judge budget has repeatedly gone on a persona nobody asked for. The rule in CLAUDE.md
# only binds an agent that reads it; this binds anything that runs the pipeline.
#
# `--personas all` is the explicit opt-in to the full matrix. It is consumed here rather
# than forwarded, because `all` is not a persona id and serving.runner._select_specs
# validates against the loaded specs -- passing it on would fail with `no match for ['all']`.
#
# Fake runs are exempt: their outputs are canned, they cost nothing, and ./scripts/gate.sh
# runs one on every commit (CLAUDE.md, Commands) with no flag to give.
#
# `--personas` is ACCUMULATED into an array, not a scalar. A scalar kept only the LAST
# `--personas`, so `--personas impulsve --personas impulsive` (a typo'd first flag) ran
# `impulsive` alone and said nothing -- exactly the silent-shrink/widen CLAUDE.md's Real runs rule exists to
# stop. `serving.inspect_runner` already accepts a repeated `--personas` (action="append"),
# so every value collected here is forwarded, not just the last one.
#
# `all` and a named persona are mutually exclusive, on purpose: `all` is consumed HERE
# rather than forwarded (a value `serving.inspect_runner`'s own typo-guard would reject
# outright, see above), so when `all` was present the values collected alongside it used
# to be dropped without ever reaching that guard -- `--personas all --personas impulsve`
# forwarded no `--personas` at all, which `inspect_runner` defaults to "no filter" (the
# WHOLE matrix), so the typo vanished silently instead of failing loudly. A real name
# alongside `all` has the same problem even without a typo (silently redundant, or
# silently narrowing if `all`'s own consumption ever changed) -- so any mix is refused
# rather than guessed at. Pass `--personas all` alone for the full matrix, or list only
# named personas.
# A value-taking flag given an empty or whitespace-only value is an error (exit 2) before
# anything runs: `--results-root ""` once planned under a relative root and `--config ""` made
# the runner read `.`. ONE helper, over every flag this script or the runner takes a value for
# (both `--flag v` and `--flag=v`).
refuse_blank_values() {
  local arg prev=""
  for arg in "$@"; do
    if [[ -n "$prev" && -z "${arg//[[:space:]]/}" ]]; then
      echo "REFUSING: $prev was given but is empty (only whitespace)." >&2
      exit 2
    fi
    prev=""
    case "$arg" in
      --personas|--phase|--config|--results-root|--methods|--evals|--exclude-evals|--base-models)
        prev="$arg" ;;
      --personas=*|--phase=*|--config=*|--results-root=*|--methods=*|--evals=*|--exclude-evals=*|--base-models=*)
        local value="${arg#*=}"
        if [[ -z "${value//[[:space:]]/}" ]]; then
          echo "REFUSING: ${arg%%=*} was given but is empty (only whitespace)." >&2
          exit 2
        fi ;;
    esac
  done
  if [[ -n "$prev" ]]; then  # the flag was last: no value at all
    echo "REFUSING: $prev needs a value." >&2
    exit 2
  fi
}
refuse_blank_values "$@"

ARGS=()
PERSONAS_VALS=()
SAW_ALL=0
# Defer-judge two-phase. Consumed here,
# NOT forwarded to inspect_runner:
#   generate  GPU pod: judged cells generate only (MO_EVALS_DEFER_JUDGE=1), stored NaN in a
#             `generation/` staging tree; no dashboard (cells are unjudged). HF-pushed + self-
#             terminate as usual.
#   judge     cheap CPU pod: fill every deferred verdict (scripts/judge_deferred.py) from the
#             `generation/` tree into the MAIN tree, then the dashboard + push. No generation.
#   both      DEFAULT: inline generate+judge, unchanged (gate/fake pipeline/single-pod runs).
PHASE="both"
# `--config` may be repeated (merge rules: serving.runner.merge_configs). One file behaves as
# $MO_EVALS_CONFIG does; several are merged once below, so every later step reads one config.
CONFIG_VALS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --personas)   PERSONAS_VALS+=("${2-}"); shift $(( $# > 1 ? 2 : 1 )) ;;
    --personas=*) PERSONAS_VALS+=("${1#--personas=}"); shift ;;
    --phase)      PHASE="${2-}"; shift $(( $# > 1 ? 2 : 1 )) ;;
    --phase=*)    PHASE="${1#--phase=}"; shift ;;
    --config)     CONFIG_VALS+=("${2-}"); shift $(( $# > 1 ? 2 : 1 )) ;;
    --config=*)   CONFIG_VALS+=("${1#--config=}"); shift ;;
    *)            ARGS+=("$1"); shift ;;
  esac
done
case "$PHASE" in
  generate|judge|both) ;;
  *) echo "REFUSING: --phase must be generate, judge, or both (got '$PHASE')." >&2; exit 1 ;;
esac
for p in ${PERSONAS_VALS[@]+"${PERSONAS_VALS[@]}"}; do
  [[ "$p" == "all" ]] && SAW_ALL=1
done
if (( SAW_ALL )); then
  for p in ${PERSONAS_VALS[@]+"${PERSONAS_VALS[@]}"}; do
    if [[ "$p" != "all" ]]; then
      echo "REFUSING: --personas mixes 'all' with a named persona ('$p')." >&2
      echo "  'all' already means every persona in \$CONFIG, so pairing it with a name is" >&2
      echo "  ambiguous -- and silently dropped the name before (a typo like 'impulsve'" >&2
      echo "  said nothing while the run swept everything anyway)." >&2
      echo "  Pass --personas all alone, or list only named personas." >&2
      exit 1
    fi
  done
else
  for p in ${PERSONAS_VALS[@]+"${PERSONAS_VALS[@]}"}; do
    ARGS+=(--personas "$p")
  done
fi

if ! fake_on && (( ! SAW_ALL )) \
  && [[ ${#PERSONAS_VALS[@]} -eq 0 ]]; then
  echo "REFUSING: no --personas on a REAL run." >&2
  echo "  $CONFIG holds every persona, so this would sweep all of them." >&2
  echo "  Ask which behaviours to run (CLAUDE.md), then pass --personas <id>[,<id>]," >&2
  echo "  or --personas all if the answer really was all of them." >&2
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  RUN=(uv run python)
else
  RUN=(python)
fi

MERGED_CONFIG=""
if [[ ${#CONFIG_VALS[@]} -eq 1 ]]; then
  CONFIG="${CONFIG_VALS[0]}"
elif [[ ${#CONFIG_VALS[@]} -gt 1 ]]; then
  MERGED_CONFIG="$(mktemp "${TMPDIR:-/tmp}/mo_merged_specs.XXXXXX.yaml")"
  "${RUN[@]}" scripts/merge_specs.py --out "$MERGED_CONFIG" "${CONFIG_VALS[@]}"
  CONFIG="$MERGED_CONFIG"
fi

# Defer-judge phase wiring. `generate` stages every cell into a `generation/` subtree and
# flips the judged scorer to store-only (MO_EVALS_DEFER_JUDGE); `judge` reads that subtree
# and writes the finished cells into MAIN_ROOT. `both` leaves RESULTS_ROOT as-is.
MAIN_ROOT="$RESULTS_ROOT"
if [[ "$PHASE" == "generate" ]]; then
  export MO_EVALS_DEFER_JUDGE=1
  RESULTS_ROOT="$MAIN_ROOT/generation"
  mkdir -p "$RESULTS_ROOT"
  echo "phase=generate: judging DEFERRED; staging tree = $RESULTS_ROOT" >&2
fi

# Pin ONE HF run folder for every push this invocation makes -- the periodic ticks below
# and the authoritative final push, and (because it is keyed off MO_EVALS_HF_RUN_NAME) every
# shard pod that shares the name. Computed ONCE here so a periodic tick and the final push
# never disagree on the folder. Uploads accumulate into it, so many ticks and many shards
# mirror into one browsable directory instead of scattering into per-timestamp folders.
HF_PATH_IN_REPO="runs/${MO_EVALS_HF_RUN_NAME:-run-$(date -u +%Y%m%d-%H%M%S)}"

# How every push from THIS phase treats a path that already exists on the Hub. An upload
# REPLACES an existing path, and the generate phase's staging tree holds UNJUDGED (NaN) cells
# at the very paths the judge phase pushes the judged ones to -- so a generate push after a
# judge push reverted judged Parquets to NaN and replaced the shared config.yaml (2026-09-25).
# So phase `generate` pushes ADD-ONLY (upload_results.py --add-only: list the remote once,
# upload only new paths); MO_EVALS_HF_PUSH_REPLACE=1 opts it back into replacing, e.g. to
# re-mirror a deliberately regenerated tree. Phases `judge` and `both` always replace: their
# judged cells must supersede unjudged ones. Set once here so the periodic ticks and the
# final push can never disagree.
HF_PUSH_MODE=()
if [[ "$PHASE" == "generate" && ! "${MO_EVALS_HF_PUSH_REPLACE:-}" =~ ^(1|true|yes|on)$ ]]; then
  HF_PUSH_MODE=(--add-only)
fi

# Optional: mirror the growing results tree (Parquet + .eval logs) to the HF org on a fixed
# interval WHILE the sweep is still running (MO_EVALS_HF_PUSH_INTERVAL seconds). Every cell's
# .eval log lands under $RESULTS_ROOT/logs the moment that cell finishes, so a fixed-interval
# push ships every finished log so far -- a pod that dies mid-sweep has already shipped its
# completed work, and a long run's logs are browsable on the Hub before it ends. Off unless a
# repo AND a positive interval are set; never in fake mode (upload_results refuses fake trees).
PERIODIC_PUSH_PID=""
# MO_EVALS_NO_PUBLISH (any non-empty value) means nothing leaves this machine: no dashboard
# publish (below) and no Hub push of results, whether periodic, generate-phase or final.
no_publish() { [[ -n "${MO_EVALS_NO_PUBLISH:-}" ]]; }
start_periodic_push() {
  [[ -n "${MO_EVALS_HF_RESULTS_REPO:-}" ]] || return 0
  if no_publish; then
    echo "periodic HF push: skipped (MO_EVALS_NO_PUBLISH is set)" >&2
    return 0
  fi
  local interval="${MO_EVALS_HF_PUSH_INTERVAL:-0}"
  [[ "$interval" =~ ^[0-9]+$ ]] && (( interval > 0 )) || return 0
  if fake_on; then
    echo "periodic HF push: skipped (fake mode produces no results to mirror)" >&2
    return 0
  fi
  echo "periodic HF push: every ${interval}s -> ${MO_EVALS_HF_RESULTS_REPO}/${HF_PATH_IN_REPO}${HF_PUSH_MODE[@]+ (add-only)}" >&2
  (
    # JITTER the schedule per pod. Every shard pod writes to the SAME run folder, and HF
    # SERIALIZES commits per repo -- simultaneous commits to one repo fail, they do not just
    # queue (measured 2026-09-17: 12 pods committing at the same instant -> 2/12 succeed, the
    # rest 500 after ~100s of retries; the same 12 staggered 3s apart -> 12/12). Pods started
    # together would otherwise tick in lockstep. A random offset in [0,interval) phase-shifts
    # each pod so the fleet's commits spread across the window instead of colliding. $RANDOM
    # differs per process, so each pod picks its own offset.
    sleep $(( RANDOM % interval ))
    while true; do
      sleep "$interval"
      # Best-effort: an early tick can precede the first finished cell (upload_results exits
      # non-zero on a no-result tree) and the Hub can 5xx -- neither may take down the sweep,
      # so a failed tick just logs and the loop waits for the next one.
      "${RUN[@]}" scripts/upload_results.py "$RESULTS_ROOT" \
        --repo "$MO_EVALS_HF_RESULTS_REPO" \
        --path-in-repo "$HF_PATH_IN_REPO" \
        --config "$CONFIG" \
        ${HF_PUSH_MODE[@]+"${HF_PUSH_MODE[@]}"} \
        >/dev/null 2>&1 || echo "periodic HF push: tick skipped (tree empty or Hub error)" >&2
    done
  ) &
  PERIODIC_PUSH_PID=$!
}
stop_periodic_push() {
  [[ -n "$PERIODIC_PUSH_PID" ]] || return 0
  kill "$PERIODIC_PUSH_PID" 2>/dev/null || true
  wait "$PERIODIC_PUSH_PID" 2>/dev/null || true
  PERIODIC_PUSH_PID=""
}
# Kill the background pusher on any exit path so it never outlives the sweep.
cleanup_run_all() {
  stop_periodic_push
  [[ -z "$MERGED_CONFIG" ]] || rm -f "$MERGED_CONFIG"
}
trap cleanup_run_all EXIT
start_periodic_push

if [[ "$PHASE" == "judge" ]]; then
  # Phase B: no generation. Judge the deferred cells from the staging tree into MAIN_ROOT.
  # A non-zero exit (a cell that could not be judged) is the completeness gate — it stops
  # the dashboard below from running against an incompletely-judged tree (set -e exits here
  # before the cleanup, so an incomplete judge leaves the staging tree for a re-run).
  "${RUN[@]}" scripts/judge_deferred.py "$MAIN_ROOT/generation" \
    --target-root "$MAIN_ROOT" \
    --config "$CONFIG"
  # The staging tree nests INSIDE MAIN_ROOT, and the dashboard/upload below read MAIN_ROOT
  # recursively (iter_result_files excludes only logs/). judge_deferred already drained the
  # staged *.parquet cell-by-cell as it moved them; drop the now-parquet-free tree entirely so
  # its leftover spec.json/logs are not re-uploaded and MAIN_ROOT holds only the judged cells.
  # Only reached on rc 0 (completeness gate passed) — the generation archive is already on HF
  # from --phase generate, so this local removal loses no raw_output.
  rm -rf "$MAIN_ROOT/generation"
else
  "${RUN[@]}" -m serving.inspect_runner \
    --config "$CONFIG" \
    --results-root "$RESULTS_ROOT" \
    ${ARGS[@]+"${ARGS[@]}"}
fi

# A dry run lists work and stops; everything else ends with the dashboard.
for arg in ${ARGS[@]+"${ARGS[@]}"}; do
  [[ "$arg" == "--dry-run" ]] && exit 0
done

# Phase `generate` produces only unjudged (NaN) cells, so there is no dashboard to build —
# mirror the staging tree to HF (so the pod is disposable and phase `judge` can pull it) and
# stop. The final authoritative push below is skipped along with the dashboard.
if [[ "$PHASE" == "generate" ]]; then
  if [[ -n "${MO_EVALS_HF_RESULTS_REPO:-}" ]] && ! no_publish; then
    stop_periodic_push
    "${RUN[@]}" scripts/upload_results.py "$RESULTS_ROOT" \
      --repo "$MO_EVALS_HF_RESULTS_REPO" \
      --path-in-repo "$HF_PATH_IN_REPO" \
      --config "$CONFIG" \
      ${HF_PUSH_MODE[@]+"${HF_PUSH_MODE[@]}"}
  fi
  echo "phase=generate done: $(find "$RESULTS_ROOT" -name '*.parquet' 2>/dev/null | wc -l) staged cells; run --phase judge next." >&2
  exit 0
fi

# Decisiveness bootstrap sidecar (scripts/mu_bootstrap.py, CPU, ~30-45 s per organism at
# B=200). `mu_decisiveness:decisiveness` votes on coherence but is one fitted number per
# organism: its standard error exists only in this sidecar, and without one the coherence
# axis SE and z are UNKNOWN (analysis.aggregate.composite_se, issue #261). Resumable, so a
# re-run only fits new organisms. The file name is shard-unique because every shard pod
# pushes into ONE Hub run folder, where a same-named file from the next pod would replace
# this one (upload_results.py); the analysis joins all of them (tree_sidecar_paths).
# Best-effort: a tree with no decisiveness rows (an --evals subset) or an organism the fit
# refuses exits non-zero, and the dashboard then shows that coherence SE as unknown.
MU_SHARD_ID="$(printf '%s' "${MO_EVALS_SHARD_ID:-${HOSTNAME:-$(hostname)}}" | tr -c 'A-Za-z0-9._-' '_')"
MU_BOOTSTRAP_B="${MO_EVALS_MU_BOOTSTRAP_B:-200}"
if fake_on; then
  MU_BOOTSTRAP_B="${MO_EVALS_MU_BOOTSTRAP_B:-20}"  # plumbing only: fake rows are not results
fi
"${RUN[@]}" scripts/mu_bootstrap.py "$RESULTS_ROOT" \
  --out "$RESULTS_ROOT/mu_decisiveness_bootstrap.${MU_SHARD_ID}.json" \
  --b "$MU_BOOTSTRAP_B" \
  || echo "run_all: mu_bootstrap incomplete — an organism it skipped keeps any earlier entry, else its coherence SE is unknown" >&2

REPORT_DIR="${MO_EVALS_REPORT_DIR:-analysis/out}"

# `mobench run` builds its page through analysis.dashboard_build instead.
if [[ -n "${MO_EVALS_SKIP_DASHBOARD:-}" ]]; then
  exit 0
fi

# Render the interactive dashboard: the one results page (the static scorecard it replaced
# is gone). It is rendered on EVERY run, a fake one included, because this render is also the
# only thing in the fake sweep that exercises the analysis code (aggregate, derived, timing,
# the Diagnostics tables). On a real, unfiltered run the same command also publishes it to the
# shared scorecards repo. Fake runs never publish; the helper refuses a fake-mode blob anyway.
# `scorecard_name()` keys the uploaded path on persona + base model ONLY, so a narrowed run
# (--methods/--evals/--exclude-evals/--base-models) or a throwaway MO_EVALS_RESULTS_ROOT would
# silently overwrite the full-matrix page for that persona/base model with a thin one -- see
# scripts/publish_guard.sh.
# Same run overwrites (name is a hash of the data), so re-runs don't pile up copies. A
# publish failure (no token, offline) is non-fatal -- the results are already written -- but
# the render is always fatal, as the report step was (so is an empty results root: the
# dashboard exits 1 with "no result rows under <root>" where the report wrote an empty card).
PUBLISH_ARGS=()
if ! fake_on && [[ -z "${MO_EVALS_NO_PUBLISH:-}" ]]; then
  # shellcheck source=publish_guard.sh
  source "$(dirname "${BASH_SOURCE[0]}")/publish_guard.sh"
  PUBLISH_SKIP_REASON=$(publish_skip_reason "$RESULTS_ROOT" ${ARGS[@]+"${ARGS[@]}"})
  if [[ -n "$PUBLISH_SKIP_REASON" ]]; then
    echo "run_all: skipping dashboard publish -- $PUBLISH_SKIP_REASON" >&2
    echo "  publish by hand: python -m analysis.dashboard --results-root $RESULTS_ROOT --publish" >&2
  else
    PUBLISH_ARGS=(--publish)
  fi
fi
# Render FIRST, bare, so a crash in the analysis (aggregate, derived, the Diagnostics tables)
# or an empty results root ("no result rows under ...") is fatal under `set -e` whether or not
# the page is then published, as the old static report step was. --dump-data keeps the computed
# blob so the publish step below does not re-read the tree.
DASH_DATA="$(mktemp "${TMPDIR:-/tmp}/dashboard_data.XXXXXX.json")"
# --config: the paper's `families:` (which few-shot and steering organism each method row shows
# where a run holds several); a method row it does not list keeps every organism.
"${RUN[@]}" -m analysis.dashboard \
  --results-root "$RESULTS_ROOT" \
  --out "$REPORT_DIR/dashboard.html" \
  --config "${MO_EVALS_DASHBOARD_CONFIG:-configs/dashboard_build.yaml}" \
  --dump-data "$DASH_DATA"
if [[ ${#PUBLISH_ARGS[@]} -gt 0 ]]; then
  # Publish only: re-render the already-computed blob and upload it. Only this step is
  # best-effort (no token, offline); the render above has already succeeded.
  "${RUN[@]}" -m analysis.dashboard \
    --results-root "$RESULTS_ROOT" \
    --data-json "$DASH_DATA" \
    --out "$REPORT_DIR/dashboard.html" \
    "${PUBLISH_ARGS[@]}" \
    || echo "run_all: dashboard publish failed (the dashboard rendered; results are safe) — publish by hand:" \
            "python -m analysis.dashboard --results-root $RESULTS_ROOT --publish" >&2
fi
rm -f "$DASH_DATA"

# Mirror results to the shared HF org (docs/COMMANDS.md) when a repo is named. The push
# script refuses fake-mode trees, so CI's fake run never mirrors anything. This final push
# is authoritative -- it adds the rendered dashboard and the .eval logs of every cell (including
# the last ones, which a periodic tick may have missed) to the SAME pinned run folder. Stop
# the periodic pusher first so a tick can't race this commit. Not `exec`, so the EXIT trap
# still fires.
if [[ -n "${MO_EVALS_HF_RESULTS_REPO:-}" ]] && ! no_publish; then
  stop_periodic_push
  "${RUN[@]}" scripts/upload_results.py "$RESULTS_ROOT" \
    --repo "$MO_EVALS_HF_RESULTS_REPO" \
    --path-in-repo "$HF_PATH_IN_REPO" \
    --report-dir "$REPORT_DIR" \
    --config "$CONFIG"
fi
