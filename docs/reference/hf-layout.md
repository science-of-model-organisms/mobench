# Results layout on the Hugging Face Hub

The one place the layout of a results dataset repo is written down. `scripts/upload_results.py`
writes it, `scripts/fetch_results.py` reads it, and the README card pushed to a repo points here.
The org is `MO_EVALS_HF_ORG` (default `science-of-model-organisms`; see `core/hf.py`); the repo name is
free text. In the default org the name starts with the uploader's handle,
`<handle>_<description>` (`docs/COMMANDS.md`, "HF naming templates"); in any other org it does not.

```
<org>/<name>                                dataset repo
<org>/<name>-restricted                     private companion, same paths (see below)
  README.md                                 layout card (written once, never overwritten)
  RESTRICTED_FILES.json                     index of the files routed to the companion
  runs/<run>/                               one folder per run: an "upload"
    <persona>__<method>__scale<n>__<fam>/   one organism
      <spec_hash>/                          one seed of it
        spec.json                           spec_hash -> persona, method, scale, seed, fake
        seed<n>__<spec_hash>__<eval>.parquet   one row per sample, raw_output populated
    logs/                                   Inspect .eval logs (+ header.json); the primary evidence
    config.yaml                             the config the run used, when uploaded
    report/  view/                          rendered dashboard / static log viewer, when uploaded
  analyses/<name>/<file>                    derived inputs a page needs (convention, see below)
```

- **`runs/<run>/`** is what `scripts/run_all.sh` writes (`MO_EVALS_HF_RUN_NAME`, default a
  timestamp). A bare `upload_results.py` call without `--path-in-repo` writes
  `<results-dir>-<UTC timestamp>/` instead; pass `--path-in-repo runs/<run>` to match. Older uploads nest the tree one level deeper, under `runs/<run>/results/`.
- **Run ladders.** A run uploaded in pieces ends its folder name in `_partial`,
  `_snapshot_<time>` or `_final`; `fetch_results.py` keeps the highest per group. A name without a
  suffix is a one-shot upload.
- **`logs/`** holds the `.eval` logs. Most cells of a vLLM run exist only here, so a download that
  skips `logs/` silently loses them (`fetch_results.py --no-logs` does it on purpose).
- **Discovery.** `fetch_results.py` treats as an upload the folder that directly holds
  `<organism>/<spec_hash>/spec.json` trees and/or `logs/`, however deep: for the layout above that
  is `runs/<run>`, never `runs`. `--uploads runs/<run> ...` or an entry in `configs/runs.yaml`
  overrides it.
- **The `-restricted` companion** holds, at the same paths, the files that can carry WMDP or GPQA
  question text (`scripts/hf_restricted.py` is the one rule). It is created private by the upload
  script, which refuses to continue unless the Hub confirms that. The fetch script overlays it
  when the token can read it and warns, naming the missing evals, when it cannot.
- **`analyses/<name>/`** is a convention, not yet written by the upload script: derived artifacts
  the dashboard build needs (for example the `mu_decisiveness_bootstrap` sidecar), uploaded by
  hand and pinned by revision in the build config.

Fetch and open a run from any org:

```bash
export MO_EVALS_HF_ORG=myorg                      # only the upload side and new URIs read it
uv run python scripts/fetch_results.py --repo myorg/alice_zesty-results --out /tmp/zesty
uv run python -m analysis.dashboard --results-root /tmp/zesty --out /tmp/zesty/dashboard.html
```
