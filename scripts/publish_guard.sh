#!/usr/bin/env bash
# Decides whether a `run_all.sh` real run should publish its dashboard to the shared,
# scorecards repo or skip it.
#
# `analysis/dashboard.py`'s `scorecard_name()` keys the uploaded filename on persona +
# base model ONLY (e.g. `impulsive/impulsive_qwen2.5-7b-instruct.html`) -- it says nothing
# about which methods/evals were actually run or which results tree the page was built
# from. A narrowed real run (`--methods steering --evals apps`, or any run against
# a throwaway `MO_EVALS_RESULTS_ROOT`) renders a thin, correct-for-what-it-is dashboard that
# still lands at the SAME path as the full-matrix page for that persona/base model, silently
# overwriting it. Only full-matrix pages should be published.
#
# `--personas` is deliberately NOT a narrowing flag here: naming one or a few personas is
# the normal real-run case (CLAUDE.md), and the filename already encodes exactly the
# persona(s) present in the data (`scorecard_name`'s `persona = "-".join(meta["personas"])`)
# -- run_all.sh strips `--personas`/`--personas=` out of $ARGS before this ever sees it, so
# a persona-only real run reports "" (publish) here.
#
#   source scripts/publish_guard.sh
#   reason=$(publish_skip_reason "$RESULTS_ROOT" "${ARGS[@]}")
#   [[ -z "$reason" ]] || echo "skipping: $reason"
#
# Prints nothing (empty string, exit 0) when the run should publish; a one-line reason
# (exit 0) when it should not. `MO_EVALS_PUBLISH=1` is the explicit opt-in that overrides
# every check below -- for a probe run someone deliberately wants mirrored.
publish_skip_reason() {
  local default_root="results" root="$1"
  shift
  # Only a truthy value opts in -- a bare `-n` check reads ANY non-empty string (including
  # "0"/"false") as set, silently disabling every check below. Matches how run_all.sh parses
  # MO_EVALS_FAKE (`^(1|true|yes|on)$`).
  if [[ "${MO_EVALS_PUBLISH:-}" =~ ^(1|true|yes|on)$ ]]; then
    return 0
  fi
  local arg
  for arg in "$@"; do
    case "$arg" in
      --methods | --methods=* | --evals | --evals=* | --exclude-evals | --exclude-evals=* | \
        --base-models | --base-models=*)
        echo "run narrowed by '$arg' -- publishing would overwrite the full-matrix" \
          "scorecard for this persona/base model (set MO_EVALS_PUBLISH=1 to publish anyway)"
        return 0
        ;;
    esac
  done
  if [[ "$root" != "$default_root" ]]; then
    echo "MO_EVALS_RESULTS_ROOT=$root is not the default '$default_root' -- looks like a" \
      "throwaway/probe tree (set MO_EVALS_PUBLISH=1 to publish anyway)"
    return 0
  fi
  return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  publish_skip_reason "$@"
fi
