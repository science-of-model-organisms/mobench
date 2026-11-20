#!/usr/bin/env bash
# The whole local gate (CLAUDE.md) in one command, with a timing table at the end.
#
#   ./scripts/gate.sh             # while iterating: skips the sweep family if the diff cannot reach it
#   ./scripts/gate.sh --full      # BEFORE MARKING A PR READY: every step, whatever the diff
#   MO_EVALS_CONFIG=configs/specs.yaml ./scripts/gate.sh --full   # the full matrix, when you have the hour
#
# WHAT EACH STEP IS FOR, in one line each (docs/COMMANDS.md describes each):
#   ruff-check, ruff-format, ty, import-bounds   static: lint, format, types, CLAUDE.md imports
#   inspect-lint      inspect-evals-lint: Inspect AI conventions in evals/ judges/ serving/
#   jsdom             a pinned jsdom outside the repo, so the dashboard DOM tests run, not skip
#   figures-extra     the `figures` extra is installed, so the paper-figure tests run, not skip
#   secrets           gitleaks over this branch's commits and the uncommitted work (redacted)
#   pytest            every module against FakeEndpoint (the full suite, always)
#   upstream-datasets the four upstream capability cells build from REAL data (= CI's step)
#   sweep             the fake pipeline end to end: runner -> resolve -> eval -> Parquet -> dashboard
#   sweep-cached      a second pass rebuilds nothing (the cache key works)
#   logs-vs-parquet   the `.eval` logs and the Parquet say the same thing
#   byte-identity     two serial fake runs are identical (scaffolding is deterministic)
#
# LEAN vs FULL. The last five are the "sweep family": the
# slowest steps, and they only exercise the sweep path. `scripts/gate_scope.py` computes that
# path (directories + a static import closure) and diffs it against origin/main; if nothing
# changed on it, a default run skips them and says so in the table. `--full` (or
# MO_EVALS_GATE_FULL=1) runs them regardless, and is REQUIRED before a PR leaves draft:
# GitHub runs CI only on ready PRs, so the full gate is what keeps this a strict superset of
# GitHub CI at the one point GitHub checks (CLAUDE.md).
#
# WHY THIS EXISTS RATHER THAN A BLOCK OF COMMANDS IN THE DOCS. The block was five steps
# that had to be copy-pasted with the right env on each line, and getting one wrong was
# silent: `MO_EVALS_CONFIG` unset ran the FULL 23-organism matrix instead of the smoke
# config, and the old CLAUDE.md block needed three paragraphs warning about exactly that. It also ran strictly
# in series, though only two of its steps have a real ordering dependency.
#
# WHAT IT DOES NOT CHANGE: every property the old block checked is still checked here —
# the pipeline runs end to end and renders the dashboard, a second pass is fully cached,
# two serial fake runs are byte-identical, and the `.eval` logs agree with the Parquet.
#
# The one structural change is that the separate fourth sweep is gone. It existed only
# because a fake-mode run deletes its own `.eval` logs once they are bridged, so
# `check_logs_match_parquet.py` had nothing to read; setting MO_EVALS_KEEP_FAKE_LOGS=1 on
# the FIRST sweep gives that checker the same tree it used to build for itself.
set -uo pipefail

# Only when EXECUTED. Sourcing this file (the tests do) must not move the caller.
[[ "${BASH_SOURCE[0]}" == "${0}" ]] && cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONFIG="${MO_EVALS_CONFIG:-configs/specs.smoke.yaml}"
# Outside the repo on purpose: a fake results tree committed by accident is how fake output
# later gets reported as results (CLAUDE.md). Fresh every run — a persisted root would
# make the first sweep all-cached, which proves nothing.
#
# PER-CHECKOUT, not a fixed /tmp/mo-gate. Several agents work this repo at once, each in
# its own worktree, and a shared path means agent B's `rm -rf` lands in
# the middle of agent A's sweep — after which A's `logs-vs-parquet` reads B's rows and
# neither run is what it claims. The suffix is the checkout path's hash, so two worktrees
# never collide and one worktree reuses the same root run to run.
ROOT="${MO_EVALS_GATE_ROOT:-/tmp/mo-gate-$(pwd -P | shasum | cut -c1-8)}"

# NOTHING `MO_EVALS_*` IS EXPORTED. Each command below gets exactly the variables it
# needs, via `env`, because two of them are actively harmful to the steps that do not
# need them:
#
#   MO_EVALS_FAKE=1 makes `pytest` fail. Five tests in tests/test_push_artifacts.py
#   exercise scripts that REFUSE to run in
#   fake mode (scripts/push_adapter_to_hub.py:91 raises SystemExit), so they see the
#   refusal instead of the behaviour they assert. Exporting it here cost a red gate;
#   it is also a live trap for anyone who runs a sweep and then pytest in one shell.
#
#   MO_EVALS_RESULTS_ROOT must not reach check_inspect_byte_identity.py — see the
#   comment on the byte-identity line below.
#
# `MO_EVALS_KEEP_FAKE_LOGS=1` keeps each cell's `.eval` log alive past the bridge, which
# is what lets `check_logs_match_parquet.py` read this tree instead of a fourth sweep
# building its own.
SWEEP_ENV=(
  env
  MO_EVALS_CONFIG="$CONFIG"
  MO_EVALS_FAKE=1
  MO_EVALS_RESULTS_ROOT="$ROOT"
  MO_EVALS_REPORT_DIR="$ROOT/out"
  MO_EVALS_KEEP_FAKE_LOGS=1
)

# In a container, `nproc` can report the HOST core count (e.g. 96), not the container's actual
# allocation -- the cgroup's `cpu.max` (quota/period) and `pids.max` do. `-n auto` (pyproject.toml addopts) sizes off
# `nproc`, so on such a host pytest-xdist can spawn ~96 workers x OpenBLAS/OMP threads
# against a small pid limit and die with `pthread_create failed: Resource temporarily
# unavailable` before running a single test. `PYTEST_XDIST_AUTO_NUM_WORKERS` is xdist's own
# override for `-n auto` (xdist/plugin.py); this derives it from the cgroup quota, ceiling
# it to whole cores and capping at the online-CPU count, falling through to
# `_gate_unquota_workers` when
# the file is missing/unreadable or the quota is "max" (unlimited). `_MO_GATE_CPU_MAX`/
# `_MO_GATE_NPROC` are test-only overrides (tests/test_gate_script.py) -- unset in normal use.
#
# NOT `nproc`: GNU coreutils' `nproc` returns `OMP_NUM_THREADS` when that is set, and
# `scripts/env.sh` exports `OMP_NUM_THREADS=1` -- which a gate run may source
# before it starts (and `run_all.sh` sources too). So `nproc` can report 1 on a 96-core host
# and the `min(quota, nproc)` below would collapse the whole gate to a single pytest worker.
# `getconf _NPROCESSORS_ONLN` reads the same sysconf value `nproc` does
# but has no OpenMP override; `nproc` stays as the fallback for a host without `getconf`.
_gate_online_cpus() {
  local n
  n="$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1)"
  [[ "$n" =~ ^[0-9]+$ && "$n" -ge 1 ]] && echo "$n" || echo 1
}

# The cgroup's finite CPU quota in whole cores (ceil), or nothing (exit 1) when there is none
# to trust: cpu.max missing/unreadable or "max", and no finite cgroup-v1 quota either.
_gate_quota_cores() {
  local cpu_max="${_MO_GATE_CPU_MAX:-/sys/fs/cgroup/cpu.max}" quota period cores
  if [[ -r "$cpu_max" ]] && read -r quota period <"$cpu_max" \
    && [[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[0-9]+$ && "$period" -gt 0 ]]; then
    cores=$(((quota + period - 1) / period)) # ceil(quota/period)
    ((cores < 1)) && cores=1
    echo "$cores"; return 0
  fi
  # cgroup v1 (no cpu.max: the quota is in cpu.cfs_quota_us).
  # Same arithmetic; -1 means unlimited and falls through like "max".
  local v1="${_MO_GATE_CPU_V1_DIR:-/sys/fs/cgroup/cpu}"
  if [[ -r "$v1/cpu.cfs_quota_us" && -r "$v1/cpu.cfs_period_us" ]] \
    && read -r quota <"$v1/cpu.cfs_quota_us" && read -r period <"$v1/cpu.cfs_period_us" \
    && [[ "$quota" =~ ^[0-9]+$ && "$period" =~ ^[0-9]+$ && "$period" -gt 0 ]]; then
    cores=$(((quota + period - 1) / period))
    ((cores < 1)) && cores=1
    echo "$cores"; return 0
  fi
  return 1
}

_gate_cpu_quota_workers() {
  local nproc_n workers
  nproc_n="${_MO_GATE_NPROC:-$(_gate_online_cpus)}"
  # A real, finite quota is trusted exactly (min(quota-derived, nproc)) regardless of
  # which side is smaller -- only a missing/unreadable quota or the literal "max" falls
  # through to the capped no-quota path below.
  if workers="$(_gate_quota_cores)"; then
    ((workers < nproc_n)) && echo "$workers" || echo "$nproc_n"
    return
  fi
  _gate_unquota_workers "$nproc_n"
}

# No CPU quota (cpu.max == "max", or the file is missing entirely -- both occur in
# some container runtimes) means `nproc` cannot be trusted either: inside an
# unprivileged cgroup it reports the HOST core count (e.g. 256 for a 32-vCPU allocation).
# `-n auto` sizing off that either never finishes pytest or oversubscribes the CPU. So cap by what's actually knowable:
#   - memory.max / 2 GiB per worker. 2 GiB is generous headroom over measured reality: a
#     single torch+transformers-importing pytest worker (tests/test_resolve.py, `-n 0`)
#     peaked at ~365 MB RSS.
#   - pids.max / 64 (generous headroom per worker's process tree).
#   - a hard ceiling of 32. It binds when memory allows more workers than any realistic
#     allocation has cores: each worker pins OPENBLAS_NUM_THREADS/OMP_NUM_THREADS=1 (below),
#     so 32 workers on 32 cores is 1:1, not an oversubscription.
# Take the tightest bound, floor 1. `_MO_GATE_MEM_MAX`/`_MO_GATE_PIDS_MAX` are test-only
# overrides (tests/test_gate_script.py) -- unset in normal use.
_gate_unquota_workers() {
  local nproc_n="$1"
  local mem_max="${_MO_GATE_MEM_MAX:-/sys/fs/cgroup/memory.max}"
  local pids_max="${_MO_GATE_PIDS_MAX:-/sys/fs/cgroup/pids.max}"
  local -a candidates=(32 "$nproc_n")
  local mem_bytes pids_n bound
  if [[ -r "$mem_max" ]] && read -r mem_bytes <"$mem_max" \
    && [[ "$mem_bytes" =~ ^[0-9]+$ ]]; then
    bound=$((mem_bytes / (2 * 1024 * 1024 * 1024)))
    ((bound < 1)) && bound=1
    candidates+=("$bound")
  fi
  if [[ -r "$pids_max" ]] && read -r pids_n <"$pids_max" \
    && [[ "$pids_n" =~ ^[0-9]+$ ]]; then
    bound=$((pids_n / 64))
    ((bound < 1)) && bound=1
    candidates+=("$bound")
  fi
  local min="${candidates[0]}" c
  for c in "${candidates[@]}"; do
    ((c < min)) && min=$c
  done
  echo "$min"
}

# pytest workers for the gate. NOT `PYTEST_XDIST_AUTO_NUM_WORKERS`: an interactive shell may set
# that to a small number, and honouring it would pin the gate to that many workers on a
# large machine (measured: 147 s at 16 workers, 226 s at 8, 714 s at 2). 16 is the knee under `--dist loadfile`: the largest
# test file is ~80 s, so more workers cannot beat that, and each one loads torch.
# MO_EVALS_GATE_PYTEST_WORKERS overrides outright.
_gate_pytest_workers() {
  if [[ "${MO_EVALS_GATE_PYTEST_WORKERS:-}" =~ ^[0-9]+$ && "$MO_EVALS_GATE_PYTEST_WORKERS" -ge 1 ]]; then
    echo "$MO_EVALS_GATE_PYTEST_WORKERS"; return
  fi
  local n q; n="$(_gate_cpu_quota_workers)"
  # The interactive cap is dropped ONLY on a box with a real quota above 4 cores. A small
  # box keeps it: on a 2-vCPU/16 GB machine the sweep alone peaks at ~10.6 GB, and the cap is
  # what keeps pytest beside it out of the OOM killer. So does a box with NO quota, where
  # the no-quota path gives 7-8 workers (16 GB / 2 GiB).
  q="$(_gate_quota_cores)" || q=0
  if ((q <= 4)) && [[ "${PYTEST_XDIST_AUTO_NUM_WORKERS:-}" =~ ^[0-9]+$ ]] \
    && ((PYTEST_XDIST_AUTO_NUM_WORKERS >= 1 && PYTEST_XDIST_AUTO_NUM_WORKERS < n)); then
    n="$PYTEST_XDIST_AUTO_NUM_WORKERS"
  fi
  ((n > 16)) && n=16
  echo "$n"
}

# Where the gate's venv lives. A `.venv` inside a worktree on a network filesystem under
# /workspace/ is slow (installs take minutes and uv's parallel install can race into "Stale
# file handle"). So there the gate uses a venv on LOCAL disk, beside the local uv cache
# (scripts/env.sh), installed with hardlinks from that cache.
#
# ONE PER WORKTREE, keyed by uv.lock. Never shared between worktrees: the project is an
# EDITABLE install, and its `.pth` names ONE checkout, so a venv shared across worktrees
# silently imports another checkout's code. A lock change gets a fresh venv; the stale one is pruned.
#
# MO_EVALS_GATE_VENV: `auto` (default: local when the checkout is under /workspace/), `repo`
# (the checkout's own `.venv`, the old behaviour), or an explicit path.
_gate_venv_path() {
  local mode="${MO_EVALS_GATE_VENV:-auto}" here key lock
  here="$(pwd -P)"
  case "$mode" in
    repo) return 0 ;;
    auto) [[ "$here" == /workspace/* && -z "${UV_PROJECT_ENVIRONMENT:-}" ]] || return 0 ;;
    *) echo "$mode"; return 0 ;;
  esac
  key="$(printf %s "$here" | shasum | cut -c1-8)"
  lock="$(shasum uv.lock 2>/dev/null | cut -c1-12)"
  local root="${MO_EVALS_GATE_VENV_ROOT:-$(dirname "${UV_CACHE_DIR:-$HOME/.cache/uv}")/mo-gate-venvs}"
  echo "$root/$key-${lock:-nolock}"
}

declare -a NAMES=() SECS=() CODES=()

record() { NAMES+=("$1"); SECS+=("$2"); CODES+=("$3"); }

# Run a step in the foreground and time it. For the cheap checks, where the output is
# worth seeing immediately and nothing else is in flight.
step() {
  local name="$1"; shift
  local start=$SECONDS
  "$@"
  local code=$?
  record "$name" "$((SECONDS - start))" "$code"
  return $code
}

# Run a step in the background, its output to "$LOGS/<name>.log". Sets <name>_PID.
spawn() {
  local name="$1"; shift
  local start=$SECONDS
  ( "$@" >"$LOGS/$name.log" 2>&1; echo "$? $((SECONDS - start))" >"$LOGS/$name.rc" ) &
  eval "PID_${name//-/_}=$!"
}

reap() {
  # THE CHECKED `read` IS THE FIX, and the numeric test beside it. `local` is hygiene,
  # not the repair: with `local code secs` alone the variables are simply EMPTY when the
  # read fails, and `[[ "" -eq 0 ]]` is true in bash arithmetic, so a killed step would
  # still be recorded as ok. A step whose subshell dies before writing its `.rc` — an OOM
  # kill, a full disk — must fail closed, or the gate prints GREEN having never run the
  # sweep, the cached pass or logs-vs-parquet. Garbage content fails closed too: an
  # unparseable first field is 0 in arithmetic context, i.e. "passed".
  local name="$1" pid="$2" code secs
  wait "$pid"
  # `[[ -r ]]` first: a bare `<file` redirect is applied BEFORE `2>/dev/null`, so a
  # missing .rc prints its own error no matter how the read is redirected.
  if [[ ! -r "$LOGS/$name.rc" ]] || ! read -r code secs <"$LOGS/$name.rc" \
     || [[ ! $code =~ ^[0-9]+$ ]]; then
    code=1 secs=0
    echo "--- $name left no exit status: treating as FAILED (killed? disk full?) ---"
  fi
  record "$name" "$secs" "$code"
  [[ $code -eq 0 ]] || { echo "--- $name FAILED, last 40 lines ---"; tail -40 "$LOGS/$name.log"; }
}

# The timing table is the point of this script, so print it even when a cheap check bails
# out early — `step ... || exit 1` used to skip straight past it.
report() {
  echo
  printf '%-18s %6s  %s\n' STEP SECONDS RESULT
  local i verdict rc=0
  for i in "${!NAMES[@]}"; do
    if [[ "${CODES[$i]}" == nogl ]]; then verdict="skipped: gitleaks not installed"
    elif [[ "${CODES[$i]}" == skip ]]; then verdict="skipped (lean: diff does not reach the sweep path)"
    elif [[ "${CODES[$i]}" -eq 0 ]]; then verdict=ok; else verdict="FAILED (rc=${CODES[$i]})"; rc=1; fi
    printf '%-18s %6s  %s\n' "${NAMES[$i]}" "${SECS[$i]}" "$verdict"
  done
  echo "logs: $LOGS   results: $ROOT   dashboard: $ROOT/out"
  if [[ $rc -ne 0 ]]; then echo "GATE RED"
  elif [[ " ${CODES[*]} " == *" skip "* ]]; then
    echo "GATE GREEN (LEAN) -- the sweep family was skipped. Run \`./scripts/gate.sh --full\`" \
         "before marking the PR ready.$([[ " ${CODES[*]} " == *" nogl "* ]] \
           && echo " ALSO: secrets were NOT scanned (gitleaks missing).")"
  elif [[ " ${CODES[*]} " == *" nogl "* ]]; then
    echo "GATE GREEN (secrets NOT scanned: gitleaks missing -- install it; --full would have failed)"
  else echo "GATE GREEN (FULL)"; fi
  return $rc
}

# Secret scan (gitleaks, .gitleaks.toml). `--redact` ALWAYS: a finding must never print the
# value it found, because the gate's output lands in logs and PR comments.
#   1. the commits on this branch that origin/main lacks, from the merge-base (so a secret
#      added and then deleted in a later commit on the same branch is still caught);
#   2. staged changes, unstaged changes, and untracked (not ignored) files, which no
#      commit-range scan can see.
# Findings print file, line, rule and fingerprint (--verbose), never the value.
# The base is MO_EVALS_GATE_BASE (as in gate_scope.py), else origin/main, else main. With no
# resolvable base (a shallow clone) it scans all of HEAD's history rather than nothing.
gate_secrets() {
  local conf="$PWD/.gitleaks.toml" base mb rc=0 tmp f
  local -a gl=(gitleaks --redact --verbose --no-banner --config "$conf")
  base="${MO_EVALS_GATE_BASE:-}"
  if [[ -z "$base" ]]; then
    for f in origin/main main; do
      git rev-parse --verify -q "$f^{commit}" >/dev/null && { base="$f"; break; }
    done
  fi
  if [[ -n "$base" ]] && mb="$(git merge-base "$base" HEAD 2>/dev/null)" && [[ -n "$mb" ]]; then
    echo "secrets: commits ${mb:0:10}..HEAD (merge-base with $base)"
    "${gl[@]}" git --log-opts="-m $mb..HEAD" . || rc=1
  else
    echo "secrets: no merge-base with ${base:-origin/main}; scanning all of HEAD's history"
    "${gl[@]}" git --log-opts="-m HEAD" . || rc=1
  fi
  echo "secrets: staged changes"
  "${gl[@]}" git --pre-commit --staged . || rc=1
  echo "secrets: unstaged changes"
  "${gl[@]}" git --pre-commit . || rc=1
  # Untracked files are copied (paths kept, so the config's path allowlists still match) to a
  # scratch dir and scanned with `dir` from INSIDE it, so reported paths are repo-relative and
  # the anchored allowlists apply. Gitignored files (results/, weights) are not copied. This
  # fails CLOSED: a file that cannot be copied (unreadable, dangling symlink, symlink to a
  # non-file) fails the step by name; a readable symlink is scanned through to its target.
  tmp="$(mktemp -d -t mo-gate-secrets.XXXXXX)"
  local ncopied=0
  while IFS= read -r -d '' f; do
    if [[ -L "$f" && ! -e "$f" ]]; then
      echo "secrets: FAIL: untracked symlink '$f' is dangling, so it cannot be scanned" >&2; rc=1
    elif [[ -L "$f" && ! -f "$f" ]]; then
      echo "secrets: FAIL: untracked symlink '$f' does not point at a regular file; not scanned" >&2; rc=1
    elif cp -L --parents -- "$f" "$tmp/" 2>/dev/null; then
      ncopied=$((ncopied + 1))
    else
      echo "secrets: FAIL: cannot copy untracked file '$f' (unreadable?), so it was not scanned" >&2; rc=1
    fi
  done < <(git ls-files --others --exclude-standard -z)
  if [[ $ncopied -gt 0 ]]; then
    echo "secrets: untracked files"
    (cd "$tmp" && "${gl[@]}" dir .) || rc=1
  fi
  rm -rf "$tmp"
  return $rc
}

# `rm -rf` on a variable a caller can set, so it is guarded and lives in a function —
# sourcing this file for a test must never delete anything. Never delete inside the repo,
# never delete a REAL results tree. Same rule as `check_destination` in
# scripts/make_preview_data.py.
reset_root() {
  # PHYSICAL paths on both sides. `pwd` is logical, so a checkout reached through a
  # symlink compares as a different tree and the in-repo test below misses.
  local here root parent
  here="$(pwd -P)"
  parent="$(cd "$(dirname "$ROOT")" 2>/dev/null && pwd -P)" || parent=""
  [[ -n $parent ]] || { echo "REFUSING: $ROOT has no existing parent directory." >&2; return 2; }
  root="$parent/$(basename "$ROOT")"
  # `dirname /` and `basename /` are both "/", so the line above turns ROOT=/ into "//",
  # which matches none of the literals below — and the `find` at the end then walks the
  # whole filesystem to depth 4. Collapse before comparing.
  while [[ $root == //* ]]; do root="${root#/}"; done

  # An ALLOW-list on the shape, not a deny-list of bad paths. The first version of this
  # refused a root inside the repo and nothing else, which let `MO_EVALS_GATE_ROOT=$HOME`
  # through — `rm -rf` on the home directory, taking the repo, every worktree and every
  # untracked results tree with it. `/` was refused only by BSD rm's own last-ditch check.
  # So: the root must be a NON-ROOT path that is neither an ancestor of the checkout nor
  # inside it, and not $HOME itself.
  case "$root" in
    /|"$HOME"|"$here")
      echo "REFUSING: gate root $ROOT resolves to $root. Pick a scratch directory." >&2
      return 2 ;;
    "$here"/*)
      echo "REFUSING: gate root $ROOT is inside the repo. Point MO_EVALS_GATE_ROOT outside it." >&2
      return 2 ;;
  esac
  case "$here" in
    "$root"/*)
      echo "REFUSING: gate root $ROOT ($root) CONTAINS this checkout. Deleting it would take"\
           "the repo with it." >&2
      return 2 ;;
  esac

  # `find`, not `<root>/*/spec.json`: that glob only sees cells exactly one level down,
  # which is the gate's own layout but not a parent-of-roots or any tree someone has
  # nested. A real (`fake: false`) tree is usually untracked and unbacked-up.
  # `find`'s own exit status is 0 whether or not -exec matched, so test its OUTPUT.
  if [[ -e $root ]] && [[ -n "$(find "$root" -maxdepth 4 -name spec.json \
      -exec grep -ls '"fake": *false' {} + 2>/dev/null)" ]]; then
    echo "REFUSING: $ROOT holds REAL results (a spec.json says fake: false). Not deleting it." >&2
    return 2
  fi
  rm -rf "$root"
}

bail() { report; exit 1; }

# Everything above is definitions; everything below is the run. Split so the failure
# paths can be tested — `tests/test_gate_script.py` sources this file and calls `reap`
# directly, which is the only way to exercise "the step died without writing its status"
# without killing a real sweep.
gate_main() {
  local full=0 arg
  [[ "${MO_EVALS_GATE_FULL:-}" =~ ^(1|true|yes|on)$ ]] && full=1
  for arg in "$@"; do
    case "$arg" in
      --full) full=1 ;;
      -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; return 0 ;;
      *) echo "gate.sh: unknown argument $arg (only --full)" >&2; return 2 ;;
    esac
  done
  # Created HERE, not at file scope: `tests/test_gate_script.py` sources this script, and
  # a temp dir per source is a leak. Tests set LOGS themselves.
  # The XXXXXX is load-bearing on Linux. BSD/macOS `mktemp -t PREFIX` appends its own
  # randomness, so `-t mo-gate-logs` worked there; GNU coreutils treats the argument as a
  # template and rejects one with no X's ("too few X's in template"). That did NOT abort
  # the gate — this script is `set -uo pipefail` with no `-e`, so the substitution simply
  # failed, `LOGS` came out EMPTY, and every step log landed at the filesystem root
  # (`/pytest.log`, `/sweep.log`), which two worktrees SHARE, so `bail` could print a
  # different run's failure. Failing silently is why it survived. Portable on both.
  LOGS="$(mktemp -d -t mo-gate-logs.XXXXXX)"
  # One gate at a time per checkout. The per-checkout root stops two WORKTREES colliding;
  # it does nothing about two runs in the SAME worktree, where run B's `reset_root` wipes
  # run A's tree mid-sweep and A's "must be all-cached" pass then silently re-runs
  # everything. `mkdir` is the atomic test-and-set every POSIX shell has.
  if ! mkdir "$ROOT.lock" 2>/dev/null; then
    echo "REFUSING: another gate run holds $ROOT.lock. Wait for it, or remove that" \
         "directory if no gate is running." >&2
    exit 2
  fi
  trap 'rmdir "$ROOT.lock" 2>/dev/null' EXIT
  reset_root || exit 2
  # UV_LINK_MODE REACHES EVERY `uv run` BELOW, and sourcing here is what makes that true.
  # `scripts/env.sh` sets it, but only `run_all.sh` sourced it -- so of the three `uv run`
  # processes this script spawns in parallel a few lines down, exactly the concurrency the
  # setting exists for, two still used the default hardlink mode into the shared cache.
  # Idempotent and safe to source repeatedly; it leaves any value already set alone.
  # shellcheck source=scripts/env.sh
  source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
  echo "gate: load $(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || echo n/a) on" \
    "$(_gate_cpu_quota_workers) usable cores" >&2
  local venv; venv="$(_gate_venv_path)"
  if [[ -n "$venv" ]]; then
    export UV_PROJECT_ENVIRONMENT="$venv"
    # Other lock hashes for THIS worktree are stale by definition. Only in the managed
    # root (auto mode): an explicit MO_EVALS_GATE_VENV path is the caller's, never pruned.
    if [[ "$(dirname "$venv")" == */mo-gate-venvs ]]; then
      local sib
      for sib in "${venv%-*}"-*; do [[ -d "$sib" && "$sib" != "$venv" ]] && rm -rf "$sib"; done
    fi
    step venv-sync env UV_LINK_MODE=hardlink \
      uv sync --frozen --all-groups --extra ifeval-anchor --extra figures --quiet || bail
    echo "gate: venv $venv" >&2
  fi
  # ---------------------------------------------------------------- 1. the cheap checks
  # Seconds, and they fail on things no amount of sweeping will reveal. Serial and first, so
  # a formatting slip does not cost eight minutes before it is reported.
  # jsdom is not a repo dependency. tests/test_dashboard_layout.py skips its 6 DOM tests when it
  # is absent, which is how they never ran in CI. Install the PINNED version outside the repo
  # (cached across gate runs) and make the tests fail, not skip, without it. Keep the version
  # in step with .github/workflows/ci.yml.
  local jsdom_version=24.1.3 jsdom_dir
  jsdom_dir="${MO_EVALS_JSDOM_DIR:-${TMPDIR:-/tmp}/mo-gate-jsdom-$jsdom_version}"
  # A cached install is reused only at the pinned version; any other version is reinstalled.
  step jsdom bash -c 'v=$(node -p "require(process.argv[1]).version" "$1/node_modules/jsdom/package.json" 2>/dev/null)
    [[ "$v" == "$2" ]] || npm install --silent --no-audit --no-fund --prefix "$1" "jsdom@$2" || exit 1
    v=$(node -p "require(process.argv[1]).version" "$1/node_modules/jsdom/package.json")
    [[ "$v" == "$2" ]] || { echo "jsdom $v installed, $2 pinned" >&2; exit 1; }' \
    _ "$jsdom_dir" "$jsdom_version" || bail
  export NODE_PATH="$jsdom_dir/node_modules" MO_EVALS_REQUIRE_JSDOM=1
  step ruff-check    uv run ruff check . || bail
  step ruff-format   uv run ruff format --check . || bail
  step ty            uv run ty check || bail
  step import-bounds uv run python scripts/check_import_boundaries.py || bail
  # Secrets: --full REQUIRES gitleaks (a gate that passes because the scanner is absent is no
  # gate); a lean run skips with a visible line in the table. Install: docs/COMMANDS.md.
  if command -v gitleaks >/dev/null 2>&1; then
    step secrets gate_secrets || bail
  elif [[ "$full" == 1 ]]; then
    echo "gate: gitleaks not installed (required by --full): see docs/COMMANDS.md#secret-scanning" >&2
    record secrets 0 1
    bail
  else
    echo "skipped: gitleaks not installed"
    record secrets 0 nogl
  fi
  # Inspect conventions (private inspect_ai imports, seeded shuffles, untested scorers...).
  # Packages named, not --all: see [tool.inspect-evals-lint] in pyproject.toml.
  step inspect-lint  uv run inspect-evals-lint evals judges serving || bail
  # tests/test_paper_figures.py SKIPs without the
  # `figures` extra, and an all-skipped module is green. 18 paper-figure tests failed on main
  # unseen that way. Refuse a venv without it rather than pass on a skip.
  step figures-extra uv run python -c "import seaborn" || {
    echo "gate: the 'figures' extra is missing: uv sync --all-groups --extra figures" >&2
    bail
  }

  local scope=sweep
  if [[ "$full" != 1 ]]; then
    # Fails safe: anything but the literal `lean` means sweep.
    [[ "$(uv run python scripts/gate_scope.py)" == lean ]] && scope=lean
  fi

  # ------------------------------------------------------- 2. the three expensive, in parallel
  # Mutually independent: pytest touches no results tree, the sweep owns $ROOT, and
  # byte-identity builds two roots of its own under a temp dir. Nothing here reads what
  # another writes, so the wall clock is the slowest of the three rather than their sum.
  # `env -u` on pytest, not merely "do not export": this script is also run by people who
  # already have these set in their shell, and both are actively destructive to a test run.
  # MO_EVALS_FAKE makes six tests fail (they assert the behaviour of scripts that refuse to
  # run in fake mode). MO_EVALS_RESULTS_ROOT is worse and silent — tests that build `.eval`
  # logs write them into whatever tree it names, so they land in the sweep's log dir beside
  # its own; `core.results` then prefers a test's foreign log to the tree's real Parquet,
  # and `check_logs_match_parquet.py` reports the two stores disagreeing.
  # `--durations=25 --durations-min=1.0` puts the slowest tests in logs/pytest.log on every
  # run, so the next "the gate is slow" question starts from a profile rather than a guess.
  # The floor keeps the table short: `reap` shows only the log's last 40 lines on failure,
  # and a full 25-row table would push the tracebacks out of them.
  local pytest_workers; pytest_workers="$(_gate_pytest_workers)"
  echo "gate: pytest -n auto -> $pytest_workers workers" \
    "(cgroup quota capped at 16; MO_EVALS_GATE_PYTEST_WORKERS overrides)" >&2
  spawn pytest env -u MO_EVALS_FAKE -u MO_EVALS_RESULTS_ROOT -u MO_EVALS_CONFIG \
    -u MO_EVALS_KEEP_FAKE_LOGS \
    PYTEST_XDIST_AUTO_NUM_WORKERS="$pytest_workers" \
    OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}" \
    OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}" \
    uv run pytest -q --durations=25 --durations-min=1.0
  if [[ $scope == lean ]]; then
    reap pytest "$PID_pytest"
    local s
    for s in upstream-datasets sweep byte-identity sweep-cached logs-vs-parquet; do
      record "$s" - skip
    done
    report
    return
  fi
  # GitHub's "Upstream eval datasets" step, the same script: the four upstream capability
  # cells built from their REAL datasets (network), which the fake sweep never does (fake
  # mode builds them from fixtures). Without it `--full` was not a superset of CI.
  step upstream-datasets env -u MO_EVALS_FAKE uv run python scripts/check_upstream_datasets.py
  spawn sweep "${SWEEP_ENV[@]}" ./scripts/run_all.sh
  spawn byte-identity env -u MO_EVALS_RESULTS_ROOT MO_EVALS_FAKE=1 \
    MO_EVALS_BYTE_IDENTITY_SHARDS="${MO_EVALS_GATE_BI_SHARDS:-6}" \
    uv run python scripts/check_inspect_byte_identity.py "$CONFIG"

  reap pytest "$PID_pytest"
  reap sweep "$PID_sweep"
  reap byte-identity "$PID_byte_identity"

  # The sweep's output goes to a file now, so warnings that used to scroll past on a green
  # run do not appear at all. These two are the ones that mean "this tree may not say what
  # you think": a cell reverting to rows from a log the bridge did not write, and the item cap biting. Surfaced, not fatal.
  grep -hE "supersedes its Parquet|MO_EVALS_MAX_ITEMS=" "$LOGS/sweep.log" 2>/dev/null \
    | sort -u | head -20 | sed 's/^/  sweep warning: /'

  # ----------------------------------------------------------- 3. what depends on the sweep
  # Both read the tree the sweep just wrote, so they cannot start earlier. Both are seconds.
  # Read the sweep's own exit code back from its rc file rather than indexing into CODES:
  # a positional index silently points at the wrong step the moment anyone adds one above.
  # Same fail-closed shape as `reap`, and `[[ -r ]]` first for the same redirect-order
  # reason: a bare `<file` is applied before any `2>` on the same command.
  if [[ ! -r "$LOGS/sweep.rc" ]] || ! read -r sweep_code _ <"$LOGS/sweep.rc" \
     || [[ ! $sweep_code =~ ^[0-9]+$ ]]; then
    sweep_code=1
  fi
  if [[ "$sweep_code" -eq 0 ]]; then
    # A second pass must re-run nothing, and `--require-all-cached` makes that an
    # ASSERTION rather than something you notice in the timing table. It is the reason
    # the sweep above must start from an empty root.
    step sweep-cached    "${SWEEP_ENV[@]}" ./scripts/run_all.sh --require-all-cached
    # Its row cache goes in this run's log dir: every fake run's logs are new paths, and
    # entries for them in the user's cache would only pile up.
    step logs-vs-parquet env MO_EVALS_LOG_ROW_CACHE="$LOGS/row-cache" \
      uv run python scripts/check_logs_match_parquet.py "$ROOT"
  else
    echo "skipping the cached pass and logs-vs-parquet: the sweep they read did not finish"
  fi

  report
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  gate_main "$@"
fi
