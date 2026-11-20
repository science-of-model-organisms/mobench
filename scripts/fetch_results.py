"""Fetch a run's results from the shared HF org into one local, analysable tree.

A big run is uploaded in several pieces — one per serving backend (``vllmA``, ``hf2``,
``whitebox_icl`` …), each re-uploaded at snapshots (``_partial`` → ``_snapshot_<time>`` →
``_final``). This selects the complete, non-redundant set:

  * within a backend group, keep only the highest-precedence snapshot
    (``final`` > ``snapshot_<time>`` > ``partial``);
  * across backend groups, download all of them side by side.

The download dir is itself the results root. ``core.results.load_all`` walks it
recursively and reads both Parquet and the ``.eval`` logs, and — crucially since PR #170
made results ".eval-primary" — **most cells now live only in the logs** (a vLLM upload may
ship 0 Parquet and 591 logs). It dedupes a cell that exists as both by source identity, so
side-by-side uploads never double-count. Snapshot de-duplication is handled here, by only
downloading the winning snapshot per group.

Usage::

    export HF_TOKEN=...   # the org token
    uv run python scripts/fetch_results.py \
        --repo <org>/<results-dataset> --out /tmp/bigrun
    uv run python -m analysis.dashboard --results-root /tmp/bigrun --out dashboard.html

A public results repo may have a private ``<repo>-restricted`` companion holding its
WMDP/GPQA cells at the same paths (`scripts/hf_restricted.py`). It is overlaid onto the
download when the token can read it; when it cannot, a loud warning names the evals that
will be missing, so a dashboard never silently loses them.

``--no-logs`` pulls Parquet only (fast, but misses every log-primary cell — most of a
vLLM run). This script only READS from the Hub; it never uploads and never writes into the
repo's ``results/`` (CLAUDE.md). Keep the local tree outside the repo.
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import ssl
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hf import hf_org  # noqa: E402
from scripts import hf_restricted  # noqa: E402

DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "configs" / "runs.yaml"
LOGS_DIRNAME = "logs"  # matches core.results.LOGS_DIRNAME — where load_all looks for logs
# The decisiveness bootstrap sidecar's stem (`analysis.friedness_panel.SIDECAR_NAME` minus
# ".json"; a test pins the two together). Not imported: that module pulls in the analysis stack.
MU_SIDECAR_STEM = "mu_decisiveness_bootstrap"

# Top-level repo folders that hold OLD or ALTERNATIVE copies of cells (the same
# <organism>/<spec_hash>/ paths as the current run). Auto-discovery skips them, because
# flatten() puts every upload into one tree and they would collide with the current cells;
# pass them via --uploads (or a runs.yaml entry) to fetch one deliberately.
EXCLUDED_TOP_DIRS = frozenset({"superseded", "archive", "rejudges"})

log = logging.getLogger("fetch_results")

# Precedence within a backend group. Higher wins. A bare group (no suffix, e.g.
# ``whitebox_icl``) is treated as final — a one-shot upload, not a snapshot ladder.
_FINAL = 10**9
_SNAP_RE = re.compile(r"^(?P<group>.+?)_snapshot_(?P<stamp>\d+)$")


def _group_and_rank(upload: str) -> tuple[str, int]:
    """Split an upload dir name into (backend group, precedence rank)."""
    if upload.endswith("_final"):
        return upload[: -len("_final")], _FINAL
    if upload.endswith("_partial"):
        return upload[: -len("_partial")], -1
    m = _SNAP_RE.match(upload)
    if m:
        return m.group("group"), int(m.group("stamp"))
    return upload, _FINAL  # no ladder suffix → one-shot upload, treat as final


def select_uploads(uploads: list[str]) -> list[str]:
    """Keep the highest-precedence snapshot per backend group; union across groups."""
    best: dict[str, tuple[int, str]] = {}
    for up in uploads:
        group, rank = _group_and_rank(up)
        if group not in best or rank > best[group][0]:
            best[group] = (rank, up)
    return sorted(v[1] for v in best.values())


def upload_root(path: str) -> str | None:
    """The upload folder a repo file belongs to, or None if it belongs to none.

    An upload is the folder that DIRECTLY holds a results tree, so the layout decides:

      * ``<upload>/<organism>/<spec_hash>/spec.json`` -> ``<upload>``
      * ``<upload>/results/<organism>/<spec_hash>/spec.json`` -> ``<upload>`` (legacy: the
        spec tree sits in a ``results/`` folder beside ``logs/``; a bare top-level
        ``results/`` is itself an upload)
      * ``<upload>/logs/**.eval`` -> ``<upload>`` (a logs-only upload has no spec.json)

    ``<upload>`` may be nested: ``scripts/run_all.sh`` writes ``runs/<run-name>/...``, whose
    upload is ``runs/<run-name>``, not ``runs`` (docs/reference/hf-layout.md). A file at the
    repo root (empty ``<upload>``) is ignored.
    """
    parts = path.split("/")
    if path.endswith("/spec.json") and len(parts) >= 4 and LOGS_DIRNAME not in parts[:-3]:
        # A legacy `<upload>/results/<organism>/<hash>/spec.json` belongs to `<upload>` (beside
        # its `logs/`), so a `<x>_partial` -> `<x>_final` ladder still ranks as one group.
        if len(parts) >= 5 and parts[-4] == "results" and parts[:-4]:
            return "/".join(parts[:-4])
        return "/".join(parts[:-3]) or None
    if path.endswith(".eval") and LOGS_DIRNAME in parts[:-1]:
        return "/".join(parts[: parts.index(LOGS_DIRNAME)]) or None
    return None


def list_uploads(api: HfApi, repo: str, revision: str | None = None) -> list[str]:
    """Upload folders in `repo` (see `upload_root`): the deepest folders holding spec trees
    and/or ``logs/``, so a ``runs/<run>/`` repo yields its runs and never just ``runs``."""
    files = retrying(
        lambda: api.list_repo_files(repo, repo_type="dataset", revision=revision),
        f"list {repo}",
    )
    return sorted({root for f in files if (root := upload_root(f)) is not None})


def default_uploads(uploads: list[str]) -> list[str]:
    """Auto-discovery's set: every upload not under an `EXCLUDED_TOP_DIRS` folder."""
    return [u for u in uploads if u.split("/")[0] not in EXCLUDED_TOP_DIRS]


def summarise(root: Path) -> dict[str, int]:
    """Count organisms, spec dirs and per-source cells in the flattened tree."""
    spec_dirs = [p.parent for p in root.rglob("spec.json") if LOGS_DIRNAME not in p.parts]
    organisms = {d.parent.name for d in spec_dirs}
    return {
        "organisms": len(organisms),
        "spec_dirs": len(spec_dirs),
        "parquet_cells": sum(1 for _ in root.rglob("*.parquet")),
        "eval_logs": sum(1 for _ in (root / LOGS_DIRNAME).rglob("*.eval"))
        if (root / LOGS_DIRNAME).is_dir()
        else 0,
    }


def _same_bytes(a: Path, b: Path) -> bool:
    return a.stat().st_size == b.stat().st_size and a.read_bytes() == b.read_bytes()


def flatten(
    staging: Path, out: Path, uploads: list[str], *, allow_overwrite: bool = False
) -> None:
    """Collapse the side-by-side uploads into one tree ``core.results.load_all`` reads.

    Two moves, because the loader looks in two fixed places (and a third copy: each upload's
    root-level decisiveness bootstrap sidecars, as
    ``<out>/mu_decisiveness_bootstrap.<NNN>-<upload>*.json``):
      * ``<upload>/<organism>/<spec_hash>/`` → ``<out>/<organism>/<spec_hash>/`` — where
        ``iter_result_files`` rglobs Parquet and ``load_manifests`` rglobs ``spec.json``;
      * ``<upload>/logs/**`` → ``<out>/logs/**`` — the ONLY place ``iter_eval_logs`` looks
        (``<root>/logs/`` exactly, not nested under an upload), and since PR #170 the logs
        are the primary store, so a nested-logs tree silently loses most cells.

    Across backends the trees are expected to be disjoint, so this is a union. Two uploads
    that write the SAME destination with DIFFERENT bytes (a rerun beside the run it replaces)
    are a conflict, never a silent overwrite: it raises ``SystemExit`` naming them, unless
    ``allow_overwrite`` (later upload in precedence order wins, the pre-#486 behaviour).
    Identical bytes are fine. A spec tree inside ANOTHER selected upload's root (a legacy
    ``runs/<r>/results`` beside ``runs/<r>/logs``) belongs to that nested upload alone, so it
    is not copied twice. ``load_all`` then dedupes any cell present as both Parquet and log by
    source identity — no double counting.
    """
    logs_out = out / LOGS_DIRNAME
    # Plan first, copy second: every collision is found BEFORE a byte is written, so a refusal
    # leaves `out` exactly as it was (a retry never inherits half a failed fetch). Destinations
    # are keyed by their RESOLVED path, so a symlink alias of another destination is one file.
    plan: dict[Path, tuple[Path, Path, str]] = {}  # resolved dst -> (src, dst, upload)
    conflicts: list[str] = []

    def place(src: Path, dst: Path, up: str) -> None:
        key = dst.resolve()
        prior = plan.get(key)
        if prior is not None:
            if not allow_overwrite and not _same_bytes(prior[0], src):
                conflicts.append(f"{dst.relative_to(out)}: {prior[2]} vs {up}")
                return
        elif key.is_file() and not allow_overwrite and not _same_bytes(key, src):
            conflicts.append(f"{dst.relative_to(out)}: already in {out} vs {up}")
            return
        plan[key] = (src, dst, up)

    ordered = sorted(uploads, key=lambda u: _group_and_rank(u)[1])
    for order, up in enumerate(ordered):
        up_root = staging / up
        if not up_root.is_dir():
            continue
        nested = [u[len(up) + 1 :] for u in uploads if u.startswith(up + "/")]
        up_logs = up_root / LOGS_DIRNAME
        if up_logs.is_dir():
            for f in sorted(up_logs.rglob("*")):
                if f.is_file():
                    place(f, logs_out / f.relative_to(up_logs), up)
        for spec_json in sorted(up_root.rglob("spec.json")):
            rel = spec_json.relative_to(up_root)
            if LOGS_DIRNAME in rel.parts:
                continue
            if any(rel.as_posix().startswith(n + "/") for n in nested):
                continue
            spec_dir = spec_json.parent
            for f in sorted(spec_dir.iterdir()):
                if f.is_file():
                    place(f, out / spec_dir.relative_to(up_root) / f.name, up)
        # The decisiveness bootstrap sidecars sit at the upload's ROOT, beside no spec.json,
        # so the loop above never sees them — and without them every coherence SE of the
        # fetched tree is unknown (`analysis.aggregate.composite_se`). Each is renamed with
        # its upload's position (unique, so no two uploads' files can collide however their
        # names and shard tags split on dots) plus its name for a reader; the analysis joins
        # every `mu_decisiveness_bootstrap.*.json` and refuses a conflict.
        tag = f"{order:03d}-" + re.sub(r"[^A-Za-z0-9._-]", "_", up)
        for sidecar in sorted(up_root.glob(f"{MU_SIDECAR_STEM}*.json")):
            rest = sidecar.name.removeprefix(MU_SIDECAR_STEM)  # ".json" or ".<shard>.json"
            place(sidecar, out / f"{MU_SIDECAR_STEM}.{tag}{rest}", up)
    if conflicts:
        shown = "\n  ".join(conflicts[:10])
        raise SystemExit(
            f"{len(conflicts)} file(s) written by two uploads, or already in the output "
            f"directory, with different bytes (a rerun beside the run it replaces?); nothing "
            f"was written:\n  {shown}\n"
            "Select one with --uploads (superseded/, archive/ and rejudges/ are already "
            "skipped by auto-discovery), fetch into an empty --out, or pass --allow-overwrite "
            "to let the later upload / the new bytes win."
        )
    for src, dst, _up in plan.values():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _retryable(exc: BaseException) -> bool:
    """Rate limits, server errors and dropped connections are worth a retry; any other HTTP 4xx
    (bad token, gated, not found) is final, and so is a local error (disk full, permissions).

    The whole cause chain is read: huggingface_hub wraps a 429/5xx or a refused connection at its
    repo-info stage in `LocalEntryNotFoundError` (a `FileNotFoundError`), so the outer type alone
    would make a rate limit look like a missing local file.
    """
    import httpx
    import requests

    net = (ConnectionError, TimeoutError, requests.ConnectionError, requests.Timeout,
           httpx.TransportError)  # fmt: skip
    # transport-level but permanent: a URL scheme nobody serves, a failed TLS handshake
    final = (httpx.UnsupportedProtocol, requests.exceptions.SSLError, ssl.SSLError)
    chain: list[BaseException] = []
    todo: list[BaseException] = [exc]  # the exception, then every __cause__ and __context__
    while todo:
        e = todo.pop()
        if all(e is not c for c in chain):  # a cycle ends the walk
            chain.append(e)
            todo += [x for x in (e.__cause__, e.__context__) if x is not None]
    transient = False
    for e in (
        chain
    ):  # a permanent error anywhere in the chain wins over the transport error that wraps it
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(e, final) or (
            status is not None and not (status in (408, 429) or status >= 500)
        ):
            return False
        # hf-xet (the Rust downloader) reports its HTTP failures as a bare RuntimeError of text
        transient |= (
            status is not None
            or isinstance(e, net)
            or (
                isinstance(e, RuntimeError)
                and re.search(r"HTTP status (?:client error \(429|server error)", str(e))
                is not None
            )
        )
    return transient


def retrying[T](fn: Callable[[], T], what: str, tries: int = 6) -> T:
    """`fn()`, retried with a growing wait while `_retryable` says the failure is transient."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as exc:  # httpx errors are not OSErrors; `_retryable` decides
            if attempt == tries - 1 or not _retryable(exc):
                raise
            wait = 30 * (attempt + 1)
            log.warning("%s: %s; retry %d/%d in %ds", what, exc, attempt + 1, tries - 1, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def pull(
    repo: str,
    dest: Path,
    patterns: list[str],
    *,
    revision: str | None = None,
    max_workers: int | None = None,
    tries: int = 6,
) -> None:
    """THE download: `patterns` of dataset `repo` into `dest`, at `revision` (default: the head).

    Every Hub read of results goes through here (`fetch`, its restricted overlay, and the pinned
    `analysis.dashboard_build` fetch). A 429, 5xx or dropped connection is retried with a growing
    wait (the download resumes where it stopped); any other 4xx is final.
    """
    kw: dict[str, Any] = {"max_workers": max_workers} if max_workers else {}
    retrying(
        lambda: snapshot_download(
            repo,
            repo_type="dataset",
            revision=revision,
            allow_patterns=patterns,
            local_dir=str(dest),
            **kw,
        ),
        repo,
        tries,
    )


def _companion_accessible(api: Any, companion: str) -> bool:
    """True if this caller can read `companion`. The Hub answers 404 both for a repo that
    does not exist and for a private one the caller cannot see, so False means either."""
    try:
        api.repo_info(companion, repo_type="dataset")
    except Exception as exc:  # 404 / 401 / network: indistinguishable here, see caller
        log.debug("companion %s not readable: %s", companion, exc)
        return False
    return True


def overlay_restricted(
    api: Any,
    repo: str,
    staging: Path,
    patterns: list[str],
    chosen: list[str],
    revision: str | None = None,
) -> bool:
    """Download `repo`'s private ``-restricted`` companion over `staging`, same paths.

    Since 2026-10-05 the WMDP/GPQA cells of a public results repo live only in its companion
    (`scripts/hf_restricted.py`), so fetching the public repo alone silently drops them.
    `revision` pins the companion (its own commit, not the public repo's). Returns True if the
    companion was overlaid. An unreadable companion ALWAYS warns —
    listing the missing evals when the public repo's ``RESTRICTED_FILES.json`` names them,
    generically otherwise — unless that index explicitly records zero restricted files.
    """
    companion = hf_restricted.companion_repo(repo)
    if _companion_accessible(api, companion):
        pull(companion, staging, patterns, revision=revision)
        log.info("overlaid private companion %s (restricted WMDP/GPQA cells)", companion)
        return True
    index: dict[str, Any] | None = None
    try:
        if api.file_exists(repo, hf_restricted.INDEX_FILENAME, repo_type="dataset"):
            index = hf_restricted.load_remote_json(api, repo, hf_restricted.INDEX_FILENAME)
    except Exception as exc:  # unknown -> falls through to the generic loud warning
        log.debug("could not check %s for %s: %s", repo, hf_restricted.INDEX_FILENAME, exc)
    if index is not None and not index.get("files") and index.get("n_files", 0) == 0:
        return False  # the index explicitly records zero restricted files: nothing missing
    missing = hf_restricted.evals_in_index(index, chosen) if index else {}
    if missing:
        detail = ", ".join(f"{e} ({n} files)" for e, n in sorted(missing.items()))
        log.warning(
            "!!! RESTRICTED CELLS MISSING: companion %s not readable (missing or no access). "
            "These evals will be absent or incomplete in %s: %s. Get %s "
            "org access (HF_TOKEN) and re-fetch for the full tree.",
            companion,
            staging.parent,
            detail,
            hf_org(),
        )
    else:
        # No usable index (absent, unreadable, or nothing under the chosen uploads): the Hub
        # answers 404 alike for "no companion" and "private, no access", so never stay quiet.
        log.warning(
            "!!! companion %s not readable (missing or no access): wmdp/gpqa/"
            "capability_format cells may be absent from %s.",
            companion,
            staging.parent,
        )
    return False


def fetch(
    repo: str,
    out: Path,
    *,
    uploads: list[str] | None = None,
    include_logs: bool = True,
    allow_overwrite: bool = False,
    api: Any = None,
    revision: str | None = None,
    restricted_revision: str | None = None,
) -> Path:
    """Download `repo`'s current uploads (or `uploads`) and flatten them into `out`.

    `revision` / `restricted_revision` pin the repo and its ``-restricted`` companion to commits
    (default: the heads), so a fetch can be reproduced exactly.
    """
    api = api if api is not None else HfApi()
    all_uploads = list_uploads(api, repo, revision)
    if not all_uploads:
        raise SystemExit(f"no uploads (spec.json trees) found under {repo}")
    # Auto-discovery skips superseded/, archive/, rejudges/ (see EXCLUDED_TOP_DIRS); an
    # explicit `uploads` list is taken as given.
    chosen = select_uploads(uploads or default_uploads(all_uploads))
    if not chosen:
        raise SystemExit(f"no current uploads under {repo} (only {', '.join(all_uploads)})")
    dropped = sorted(set(all_uploads) - set(chosen))
    log.info("uploads present: %s", ", ".join(all_uploads))
    log.info("selected (winning snapshot per group): %s", ", ".join(chosen))
    if dropped:
        log.info("superseded / not selected: %s", ", ".join(dropped))
    if include_logs:
        patterns = [f"{u}/**" for u in chosen]
    else:
        patterns = [
            p
            for u in chosen
            for p in (
                f"{u}/**/*.parquet",
                f"{u}/**/spec.json",
                f"{u}/**/*.timing.json",
                f"{u}/{MU_SIDECAR_STEM}*.json",
            )
        ]
        log.warning("--no-logs: log-primary cells (most of a vLLM run) will be missing")
    staging = out / "_staging"
    out_existed = out.exists()
    try:
        pull(repo, staging, patterns, revision=revision)
        overlay_restricted(api, repo, staging, patterns, chosen, restricted_revision)
        flatten(staging, out, chosen, allow_overwrite=allow_overwrite)
    finally:  # success or refusal: the staging copy never outlives the call
        shutil.rmtree(staging, ignore_errors=True)
        if not out_existed and out.is_dir() and not any(out.iterdir()):
            out.rmdir()
    stats = summarise(out)
    log.info("flattened into %s: %s", out, stats)
    grp = Counter(_group_and_rank(u)[0] for u in chosen)
    log.info("use this as --results-root (groups: %s)", ", ".join(sorted(grp)))
    return out


def resolve_run(run_id: str, index_path: Path) -> tuple[str, list[str] | None]:
    """Look up a run in the index → (hf_repo, uploads). Uploads None means auto-discover."""
    import yaml

    if not index_path.is_file():
        raise SystemExit(f"run index {index_path} not found")
    index = yaml.safe_load(index_path.read_text(encoding="utf-8")) or {}
    runs = index.get("runs") or {}
    entry = runs.get(run_id)
    if entry is None:
        raise SystemExit(f"run {run_id!r} not in {index_path}; known: {', '.join(runs)}")
    repo = entry.get("hf_repo")
    if not repo:
        raise SystemExit(f"run {run_id!r} has no hf_repo in {index_path}")
    return repo, entry.get("uploads")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", help="run id in the index (configs/runs.yaml); sets --repo/--uploads")
    ap.add_argument("--repo", help="HF dataset repo id (org/name); overrides --run's repo")
    ap.add_argument("--out", required=True, type=Path, help="local results root to write")
    ap.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="run index for --run (default: configs/runs.yaml of this checkout, from any CWD)",
    )
    ap.add_argument(
        "--uploads",
        nargs="*",
        default=None,
        help="explicit upload dirs (default: from --run, else auto-discover + dedupe, "
        "skipping top-level superseded/, archive/ and rejudges/ — name them here to fetch one)",
    )
    ap.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="when two selected uploads write the same file with different bytes, let the "
        "later one win instead of failing",
    )
    ap.add_argument("--revision", help="pin the repo to this commit (default: its head)")
    ap.add_argument("--restricted-revision", help="pin the -restricted companion to this commit")
    ap.add_argument(
        "--no-logs",
        action="store_true",
        help="Parquet only — faster, but misses every log-primary cell",
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    repo, uploads = args.repo, args.uploads
    if args.run:
        run_repo, run_uploads = resolve_run(args.run, args.index)
        repo = repo or run_repo
        uploads = uploads if uploads is not None else run_uploads
    if not repo:
        raise SystemExit("pass --repo or --run")
    fetch(
        repo,
        args.out,
        uploads=uploads,
        include_logs=not args.no_logs,
        allow_overwrite=args.allow_overwrite,
        revision=args.revision,
        restricted_revision=args.restricted_revision,
    )


if __name__ == "__main__":
    main()
