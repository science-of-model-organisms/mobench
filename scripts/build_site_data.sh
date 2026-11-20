#!/usr/bin/env bash
# Regenerate the project site's two data files from a dashboard data root:
#   site/public/dashboard/index.html   the interactive dashboard (served unchanged)
#   site/src/data/results.json         the compact table the landing page reads at build time
#
# usage: scripts/build_site_data.sh <data_root> [extra dashboard.py args...]
#
# <data_root> holds `dashboard_data.json`, the cached blob `dashboard.py --dump-data` writes
# (reading the .eval trees is slow, so the site always renders from the blob). Run this by
# hand and commit the two outputs: CI never reads HF or raw results, it only builds the site.
set -euo pipefail

root="${1:?usage: $0 <data_root> [extra dashboard.py args]}"
shift || true
blob="$root/dashboard_data.json"
[[ -f "$blob" ]] || { echo "no $blob (write one with dashboard.py --dump-data)" >&2; exit 1; }

here="$(cd "$(dirname "$0")/.." && pwd)"
cd "$here"
uv run python -m analysis.dashboard \
  --data-json "$blob" \
  --out site/public/dashboard/index.html \
  --site-json site/src/data/results.json \
  "$@"
