#!/usr/bin/env bash
# Environment every MO_evals shell needs, kept in the repo so a fresh machine inherits it with
# the checkout rather than from someone remembering.
#
#   source scripts/env.sh          # in a shell, before any uv/hf command
#   ./scripts/env.sh               # run directly to print what it would export
#
# WHY THIS FILE EXISTS
#
# Some container setups point UV_CACHE_DIR and HF_HOME at a network (FUSE/NFS) mount.
# `uv sync` unpacks wheels into a content-addressed store and then materialises the venv by
# hardlinking ~10^5 small files, stat-ing each one; over a network filesystem that is ~10^5
# round trips and the install effectively hangs, with nothing in the output saying why.
# This file redirects those caches to local disk when they are unset or on a network mount.
# A cluster whose shared cache is on a network mount on purpose sets MO_EVALS_KEEP_CACHE_DIRS=1
# to keep the values it set (a warning is printed). Sourcing it is idempotent and safe to do
# repeatedly.
#
# Only caches move. Model weights and results do not: keep those on durable storage or the
# HF Hub.

# No `set -e` and no `exit`: this file is meant to be sourced, and killing the caller's
# shell over a cache path would be worse than the problem it solves.

# True if $1's filesystem is a network/FUSE mount, i.e. the wrong place for 10^5 small files.
_mo_on_network_volume() {
  local probe=$1 fs
  # Walk up to the nearest existing ancestor: the path itself may not exist yet.
  while [[ -n $probe && ! -e $probe ]]; do probe=$(dirname "$probe"); done
  [[ -z $probe ]] && return 1
  fs=$(df -P -T "$probe" 2>/dev/null | awk 'NR==2 {print $2}')
  [[ $fs == fuse* || $fs == nfs* || $fs == mfs* || $fs == cifs* || $fs == smb* || $fs == 9p ]]
}

# Redirect $1 (a var name) to $2 iff it is unset or currently on a network volume.
# A value the user deliberately put on local disk is left alone, and with
# MO_EVALS_KEEP_CACHE_DIRS=1 so is one on a network volume (the AISI pilot of 2026-11-10 needed
# its cluster's cache settings kept).
_mo_redirect_cache() {
  local var=$1 target=$2 current
  # `current=${!var}` must be its own statement: within a single `local` the indirect
  # expansion is evaluated before `var` is assigned, and bash errors with
  # "invalid indirect expansion" (bash 5.2).
  current=${!var:-}
  if [[ -n $current ]] && ! _mo_on_network_volume "$current"; then
    return 0
  fi
  if [[ -n $current && ${MO_EVALS_KEEP_CACHE_DIRS:-} == 1 ]]; then
    if [[ -z ${MO_EVALS_ENV_QUIET:-} ]]; then
      echo "mo-evals env: $var=$current is on a network volume; kept (MO_EVALS_KEEP_CACHE_DIRS=1; uv sync can be slow there)" >&2
    fi
    return 0
  fi
  mkdir -p "$target" 2>/dev/null || {
    echo "mo-evals env: cannot create $target; leaving $var as ${current:-unset}" >&2
    return 0
  }
  export "$var=$target"
  # stderr, not stdout: this is an informational diagnostic (the error branch above already
  # uses >&2). A sourced env.sh must leave stdout clean — `test_env_sh_exports_the_copy_link_mode`
  # sources it and asserts stdout is exactly `copy`, and it runs on hosts where HF_HOME/
  # UV_CACHE_DIR start unset (the GitHub runner), so a stdout line here fails the gate.
  [[ -n ${MO_EVALS_ENV_QUIET:-} ]] || echo "mo-evals env: $var -> $target (was: ${current:-unset})" >&2
}

# Local-disk cache root. $HOME is normally local, but check
# rather than assume — if $HOME itself is on the network volume, fall back to /var/tmp,
# which is local on any normal box.
_mo_cache_root() {
  if [[ -n ${MO_EVALS_CACHE_ROOT:-} ]]; then
    echo "$MO_EVALS_CACHE_ROOT"
  elif ! _mo_on_network_volume "$HOME"; then
    echo "$HOME/.cache"
  else
    echo "/var/tmp/mo-evals-cache"
  fi
}

mo_evals_env() {
  local root
  root=$(_mo_cache_root)
  _mo_redirect_cache UV_CACHE_DIR "$root/uv"
  _mo_redirect_cache HF_HOME "$root/huggingface"

  # COPY, NOT HARDLINK, out of the cache. uv's default (`clone`, falling back to
  # `hardlink`) makes every file in a venv a second name for the one file in
  # UV_CACHE_DIR, and several agents share that cache on one box (one
  # worktree each, one cache). A second `uv sync` that evicts or rewrites a cached wheel
  # then edits the bytes a RUNNING venv is importing from, and the symptom is not a
  # failed install -- it is the other agent's process dying part-way through an unrelated
  # import. Copying costs disk in the venv and makes each one independent of the cache
  # after it is built. Set here, beside the cache it is about; overridable by a caller.
  export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

  # Expandable CUDA segments let the HF backend run hf_max_batch_size=64
  # (configs/serving.yaml) without fragmentation OOM (measured 1.69x throughput on an A40).
  # setdefault, so a caller can override.
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  # HF tokenizers fork-warn on every worker in the eval harness; the parallelism it refers
  # to is per-process and we get ours from threads.
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

  # libgomp spawns a thread pool per process; on a many-core host that can exceed the container's
  # cgroup pids.max and abort native-extension work.
  # setdefault, so a caller that wants real OpenMP parallelism can still override.
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
}

mo_evals_env

# Executed rather than sourced: report the result so `./scripts/env.sh` is a usable check.
if [[ ${BASH_SOURCE[0]} == "$0" ]]; then
  echo
  echo "UV_CACHE_DIR=$UV_CACHE_DIR"
  echo "UV_LINK_MODE=$UV_LINK_MODE"
  echo "HF_HOME=$HF_HOME"
  echo
  echo "This was a subshell — the exports died with it. To apply them, run:"
  echo "  source scripts/env.sh"
fi
