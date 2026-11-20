"""Upload a results tree to a dataset repo in the shared HF org.

    uv run python scripts/upload_results.py <results_root> \
        --repo science-of-model-organisms/<name>

Examples, against a tree in ``results/``::

    REPO=science-of-model-organisms/real-run-3-results
    # inspect first — no network, no token; prints file count, bytes and target
    uv run python scripts/upload_results.py results --repo $REPO --dry-run
    # then upload, transcripts included
    uv run python scripts/upload_results.py results --repo $REPO
    # ...or Parquet + manifests only, when the `.eval` logs are GB-scale
    uv run python scripts/upload_results.py results --repo $REPO --no-logs

Per docs/COMMANDS.md, shared artifacts live in the ``science-of-model-organisms`` org (or the
org in ``$MO_EVALS_HF_ORG``) under a handle-free ``<name>``. (The legacy
``Misalignment-Empirics`` org used ``<handle>_<name>``; that convention is now the legacy org's
only.) ``--repo`` is validated against that shape before anything touches the network. A repo it
creates starts private; the shared-storage rule makes results public unless dangerous or
unpublished-sensitive, and that flip is the owner's call, made on the Hub.

Restricted files never go to the main repo, whatever its visibility. Every file that can
carry WMDP or GPQA question text (`scripts/hf_restricted.py` is the one rule: ``wmdp``,
``gpqa_*`` and ``capability_format`` logs and Parquet, and Inspect indexes that mention them)
goes to the private companion ``<repo>-restricted`` at the same path, in its own commit, and
only after the Hub confirms the companion is private — ``create_repo(private=True,
exist_ok=True)`` never re-privatises a repo that already exists as public, so the flag is read
back, and anything but ``private is True`` is a hard error. No flag overrides either guard.
Both repos get an updated ``RESTRICTED_FILES.json``, and the main repo's README gains a
companion notice once (appended, never rewritten). Since 2026-10-05 several results repos are
public with such a companion.

Two integrity guards run first, both offline (CONTRIBUTING.md, Non-negotiables: fake and
real results trees look identical): a tree containing any ``"fake": true`` manifest is
refused unless ``--allow-fake``, and a tree **mixing** fake and real manifests is refused
outright, with
no override — the same rule ``analysis/aggregate.py`` enforces, for the same reason. Fake
and real runs share spec hashes, so once mirrored to the org nothing distinguishes them.

The tree, the optional ``--report-dir`` dashboard and the optional ``--config`` snapshot all
land in **one** ``HfApi.create_commit``, so a 2,000-file upload costs the Hub one write, not
2,000 (``huggingface_hub`` handles 429 retry/backoff itself). It is `create_commit` rather
than `upload_folder` because a single commit has to span three source directories, and
staging them into one folder would mean copying GB of `.eval` logs. Layout::

    <path-in-repo>/              the results tree, as-is (organism dirs + logs/)
    <path-in-repo>/report/       --report-dir, when given
    <path-in-repo>/view/         --bundle-dir, when given: a browsable log viewer
    <path-in-repo>/config.yaml   --config, when given
    README.md                    repo-root layout card, ONLY if the repo has no README yet
    RESTRICTED_FILES.json        moved-file index, when restricted files were routed away

The layout card never replaces an existing README: a results repo's root README is often
curated by hand (the canonical repo's CANONICAL banner and superseded-run notes were
overwritten this way on 2026-10-04), so it is checked on the Hub (`has_root_readme`) right
before the commit. ``--dry-run`` stays offline, so it cannot say whether the card will go up.

``--bundle-dir`` runs ``inspect view bundle`` over the tree's ``logs/`` first and folds the
result into that same commit, so the uploaded run carries a self-contained static viewer
for its transcripts — no clone, no ``inspect`` install, no local log dir. It is built
before the plan is costed, so ``--dry-run`` shows exactly what the upload will contain.
The bundle embeds its OWN copy of the logs (plus ~10 MB of viewer assets), so on a
GB-scale tree pair it with ``--no-logs`` rather than uploading both copies::

    uv run python scripts/upload_results.py results --repo $REPO \
        --bundle-dir /tmp/view --no-logs

The default ``--path-in-repo`` is the results root's basename plus a UTC timestamp, so
re-uploading the same tree adds a directory instead of overwriting the previous one.
Nothing is ever deleted, locally or on the Hub: the commit carries ``CommitOperationAdd``
operations only. But an ADD to a path that already exists REPLACES that file, so a push into
a shared ``--path-in-repo`` overwrites whatever is there. ``--add-only`` lists the remote
paths once and uploads only the destinations that do not exist yet (`drop_existing`): that is
how ``run_all.sh --phase generate`` pushes, because its staging tree holds UNJUDGED (NaN)
copies of cells whose judged Parquet the judge phase already pushed to the same path — on
2026-09-25 a generate tick reverted a judged Parquet to its unjudged blob and replaced the
run's shared ``config.yaml``. The judge phase and the default push still replace, since a
judged cell must supersede the unjudged one.

The token comes from ``HF_TOKEN``, else a local gitignored ``.env``, else the credential
stored by ``hf auth login``. It is never printed or logged.
"""

from __future__ import annotations

import argparse
import json
import logging
import posixpath
import re
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hf import LEGACY_ORG, hf_org  # noqa: E402
from core.results import LOGS_DIRNAME  # noqa: E402
from scripts import hf_restricted  # noqa: E402
from scripts.push_banks_to_hub import has_root_readme, load_token  # noqa: E402

log = logging.getLogger("upload_results")

#: ``<org>/<handle>_<slug>``. The first underscore splits the handle from the description,
#: so the handle itself may not contain one (docs/COMMANDS.md). That naming rule is the
#: LEGACY org's convention; the public default org (and any other ``MO_EVALS_HF_ORG``) only
#: has to own the repo.
_HANDLE_REPO = r"[A-Za-z0-9][A-Za-z0-9-]*_[A-Za-z0-9][\w.-]*"
_FREE_REPO = r"[A-Za-z0-9][\w.-]*"


def repo_re() -> re.Pattern[str]:
    """The repo-id pattern for the configured org (``core.hf.hf_org``), built per call."""
    org = hf_org()
    name = _HANDLE_REPO if org == LEGACY_ORG else _FREE_REPO
    return re.compile(rf"^{re.escape(org)}/{name}$")


#: What counts as evidence. `.eval` logs are the primary artifact since the Inspect
#: migration and the Parquet is bridged from them, so a log-only tree is not an empty one.
RESULT_SUFFIXES = frozenset({".parquet", ".eval"})

#: Directory-name suffix of Inspect's per-run checkpoint folders (``<log>.checkpoints/``).
#: They hold a restic repo for a local backup, and ``restic/restic-config.json`` carries that
#: repo's ``restic_password``: a credential, and no evidence. Never uploaded.
CHECKPOINT_DIR_SUFFIX = ".checkpoints"


def is_checkpoint_path(path: str | Path) -> bool:
    """True if any directory component of `path` is an Inspect ``*.checkpoints`` folder."""
    return any(part.endswith(CHECKPOINT_DIR_SUFFIX) for part in Path(path).parts[:-1])


README = """\
# MO_evals results

Raw per-sample results from MO_evals runs. One directory per upload; nothing here is
aggregated — the Parquet and the `.eval` logs are the primary evidence, the scorecard is a
summary of them.

{visibility}

`scripts/run_all.sh` uploads to `runs/<run-name>/`; a direct `upload_results.py` call without
`--path-in-repo` uploads to `<results-dir>-<UTC timestamp>/`. The folder layout inside each
upload (organism and seed folders, `logs/`, `report/`, the private `-restricted` companion) is
specified in `docs/reference/hf-layout.md` of the MO_evals repo, not repeated here.

Uploaded by `scripts/upload_results.py` in the MO_evals repo, which is the source of
truth for how these were produced. Judge caveats (calibration status, fake-mode bans)
live in the run's config and scorecard, not here. Uploads made before 2026-08-27 nest the
tree one level deeper, under `runs/<run-name>/results/`.
"""


def render_readme(repo: str, *, private: bool | None, companion: bool) -> str:
    """The repo-root layout card, stating `repo`'s visibility and any private companion.

    Parameters
    ----------
    repo : str
        The main results repo.
    private : bool or None
        Its visibility as the Hub reported it when the card was written (None: unknown).
    companion : bool
        True when restricted files are routed to ``<repo>-restricted``; adds the notice.

    Returns
    -------
    str
        The card's Markdown.
    """
    if private is True:
        visibility = (
            "This repo was **private** when this card was written. Results repos are public "
            "unless dangerous or unpublished-sensitive (docs/COMMANDS.md); the flip is the "
            "owner's call, made on the Hub."
        )
    elif private is False:
        visibility = "This repo was **public** when this card was written."
    else:
        visibility = "Visibility: see the repo settings on the Hub."
    if companion:
        visibility += "\n\n" + hf_restricted.companion_notice(repo).rstrip("\n")
    return README.replace("{visibility}", visibility)


def validate_repo_id(repo: str) -> str:
    """Check a repo id against the shared-storage naming rule.

    Parameters
    ----------
    repo : str
        Candidate repo id, e.g. ``science-of-model-organisms/real-run-3``.

    Returns
    -------
    str
        The repo id, unchanged.

    Raises
    ------
    SystemExit
        If it is not ``<org>/<name>`` for the configured org (``MO_EVALS_HF_ORG``; the
        ``<handle>_<name>`` rule is the LEGACY org's only), or names a private
        ``-restricted`` companion (those are written only by routing).
    """
    if repo.endswith(hf_restricted.RESTRICTED_SUFFIX):
        raise SystemExit(
            f"--repo {repo!r} is a private companion. Upload to the main repo "
            f"({repo.removesuffix(hf_restricted.RESTRICTED_SUFFIX)}); restricted files are "
            "routed to the companion automatically."
        )
    if not repo_re().match(repo):
        org = hf_org()
        if org == LEGACY_ORG:
            shape = f"{org}/<handle>_<name>"
            hint = (
                f"Results live in the {org} org (MO_EVALS_HF_ORG sets it); in the legacy org "
                f"the name starts with the uploader's handle, e.g. {org}/theo_real-run-3-results."
            )
        else:
            shape = f"{org}/<name>"
            hint = (
                f"Results live in the {org} org (MO_EVALS_HF_ORG sets it), e.g. "
                f"{org}/real-run-3-results."
            )
        raise SystemExit(f"--repo {repo!r} is not '{shape}'. {hint}")
    return repo


def fake_flags(root: Path) -> dict[Path, bool]:
    """Read the ``fake`` flag of every ``spec.json`` manifest under `root`.

    Parameters
    ----------
    root : Path
        Results root to walk.

    Returns
    -------
    dict[Path, bool]
        Manifest path -> its ``fake`` flag.

    Raises
    ------
    SystemExit
        If a manifest is unparseable. The flag is the only thing separating fake output
        from a result, so "cannot tell" is refused rather than assumed real.
    """
    flags: dict[Path, bool] = {}
    for path in sorted(root.rglob("spec.json")):
        try:
            flags[path] = bool(json.loads(path.read_text(encoding="utf-8")).get("fake", False))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise SystemExit(
                f"unreadable manifest {path} ({exc}) — cannot prove this tree is not "
                "fake-mode output, so refusing to upload it."
            ) from exc
    return flags


def check_provenance(root: Path, *, allow_fake: bool = False) -> None:
    """Refuse fake and fake/real-mixed trees before any network call.

    Parameters
    ----------
    root : Path
        Results root to walk.
    allow_fake : bool, optional
        Permit an all-fake tree (e.g. sharing a pipeline reproduction). Never permits a
        mixed tree.

    Raises
    ------
    SystemExit
        If the tree mixes fake and real manifests, or is fake without `allow_fake`.
    """
    flags = fake_flags(root)
    fake = sorted({p.parent.name for p, f in flags.items() if f})
    real = sorted({p.parent.name for p, f in flags.items() if not f})
    if fake and real:
        raise SystemExit(
            f"{root} mixes FAKE and real trees: fake {fake} alongside real {real}. Fake and "
            "real runs share spec hashes, so mirroring both to the org makes them "
            'indistinguishable. Quarantine the fake trees (spec.json has "fake": true) and '
            "upload again — there is deliberately no override for this."
        )
    if fake and not allow_fake:
        raise SystemExit(
            f"{root} is FAKE-mode output ({len(fake)} organism dirs, spec.json has "
            '"fake": true). Uploading it to the shared org is how someone later reports it '
            "as a result (CLAUDE.md). Pass --allow-fake if that is genuinely the intent."
        )
    if not flags:
        log.warning(
            "no spec.json manifests under %s — provenance unverifiable; upload only if you "
            "know this tree is real.",
            root,
        )


def inventory(root: Path, *, include_logs: bool = True) -> tuple[list[Path], int]:
    """List the files that would upload, and their total size.

    Parameters
    ----------
    root : Path
        Results root to walk.
    include_logs : bool, optional
        Include the ``logs/`` subtree of Inspect ``.eval`` logs. Default True —
        transcripts are the primary evidence (CLAUDE.md).

    Returns
    -------
    tuple[list[Path], int]
        The files (never an Inspect ``*.checkpoints/`` folder), sorted, and their total size
        in bytes.
    """
    files = [
        p
        for p in sorted(root.rglob("*"))
        if p.is_file() and not is_checkpoint_path(p.relative_to(root))
    ]
    if not include_logs:
        files = [p for p in files if p.relative_to(root).parts[0] != LOGS_DIRNAME]
    return files, sum(p.stat().st_size for p in files)


def build_view_bundle(root: Path, bundle_dir: Path) -> Path:
    """Run ``inspect view bundle`` over ``<root>/logs`` into `bundle_dir`.

    Parameters
    ----------
    root : Path
        Results root; its ``logs/`` subtree is what gets bundled.
    bundle_dir : Path
        Local output directory for the static viewer. **Deleted and rewritten** if it
        exists, so re-uploading a tree does not need it cleaned up by hand.

    Returns
    -------
    Path
        `bundle_dir`, now holding the viewer.

    Raises
    ------
    SystemExit
        There are no logs to bundle — with `--no-logs` that is likely a mistake, and
        without it the tree simply has none — or `bundle_dir` overlaps the results tree.
    """
    from inspect_ai.log import bundle_log_dir

    logs = root / LOGS_DIRNAME
    if not logs.is_dir():
        raise SystemExit(f"--bundle-dir: no {logs} to bundle (this tree has no Inspect logs)")

    # `overwrite=True` is an `rm -rf` of the output directory before writing
    # (`inspect_ai/log/_bundle.py`: `output_fs.rm(to_dir, recursive=True)`), and inspect's
    # own guard only refuses an output dir inside the LOG dir — the results root is that
    # dir's parent, so `--bundle-dir results` sails past it and deletes the tree, Parquet,
    # manifests and all. Reproduced on a throwaway tree, so this guard is not theoretical.
    # `--dry-run` is no protection either: the bundle is built before the plan is costed.
    bundle_abs, root_abs = bundle_dir.resolve(), root.resolve()
    if bundle_abs == root_abs or root_abs in bundle_abs.parents or bundle_abs in root_abs.parents:
        raise SystemExit(
            f"--bundle-dir {bundle_dir} overlaps the results tree {root}. The bundle "
            "directory is DELETED before it is written, so this would destroy results. "
            "Point it somewhere outside the tree."
        )
    bundle_log_dir(log_dir=str(logs), output_dir=str(bundle_dir), overwrite=True)
    return bundle_dir


def plan_upload(
    files: list[Path],
    root: Path,
    path_in_repo: str,
    *,
    report_dir: Path | None = None,
    config: Path | None = None,
    bundle_dir: Path | None = None,
) -> list[tuple[str, Path | bytes]]:
    """Map every file of one upload to its destination path in the repo.

    Kept separate from the commit so the whole layout is assertable offline: the dry run
    and the real upload are the same list, so nothing can be counted but not uploaded. The
    one exception is the repo-root ``README.md`` layout card, which `main` appends only after
    checking the Hub (`has_root_readme`), so it never replaces a curated README.

    Parameters
    ----------
    files : list[Path]
        Results-tree files, from `inventory`.
    root : Path
        Results root the paths are relative to.
    path_in_repo : str
        Destination prefix for this upload.
    report_dir : Path, optional
        Report directory (the dashboard); uploaded under ``<path_in_repo>/report/``. Skipped
        silently if it does not exist — a `--dry-run` pipeline has no report.
    config : Path, optional
        Run config; uploaded as ``<path_in_repo>/config.yaml``.
    bundle_dir : Path, optional
        Static Inspect log viewer from `build_view_bundle`; uploaded under
        ``<path_in_repo>/view/``. In the SAME plan, so a bundled upload is still one
        commit — a second commit per upload is what this function exists to avoid.

    Returns
    -------
    list[tuple[str, Path | bytes]]
        ``(destination, source)`` pairs; the source is a local path or literal bytes.
    """
    plan: list[tuple[str, Path | bytes]] = [
        (f"{path_in_repo}/{p.relative_to(root).as_posix()}", p) for p in files
    ]
    if report_dir is not None and report_dir.is_dir():
        plan += [
            (f"{path_in_repo}/report/{p.relative_to(report_dir).as_posix()}", p)
            for p in sorted(report_dir.rglob("*"))
            if p.is_file() and not is_checkpoint_path(p.relative_to(report_dir))
        ]
    if bundle_dir is not None and bundle_dir.is_dir():
        plan += [
            (f"{path_in_repo}/view/{p.relative_to(bundle_dir).as_posix()}", p)
            for p in sorted(bundle_dir.rglob("*"))
            if p.is_file() and not is_checkpoint_path(p.relative_to(bundle_dir))
        ]
    if config is not None and config.is_file():
        plan.append((f"{path_in_repo}/config.yaml", config))
    return plan


def _repo_path(dest: str) -> str:
    """`dest` as the Hub stores it: `CommitOperationAdd` drops a leading ``./`` or ``/``."""
    return posixpath.normpath(dest).lstrip("/")


def in_progress_log(path: Path | bytes) -> bool:
    """True for an Inspect ``.eval`` log that is still being written.

    Real runs log in realtime (`serving/inspect_runner.py`, ``log_realtime=not fake``), so a
    cell's ``.eval`` sits at its FINAL path from the moment the cell starts, holding only
    ``_journal/`` entries; ``header.json`` is written when the task ends. An add-only push
    must not upload such a file: the path would then exist on the Hub and every later push
    would skip it, freezing a truncated log there for good.
    """
    if not isinstance(path, Path) or path.suffix != ".eval":
        return False
    try:
        with zipfile.ZipFile(path) as zf:
            return "header.json" not in zf.namelist()
    except (OSError, zipfile.BadZipFile):
        return True  # half-written zip: not finished either


def drop_existing(
    plan: list[tuple[str, Path | bytes]], remote: set[str]
) -> tuple[list[tuple[str, Path | bytes]], list[str]]:
    """Split a plan into the destinations not yet on the Hub and those already there.

    The add-only rule, kept off the network so it is assertable offline: every entry whose
    destination is in `remote` is dropped, whatever its content — an add-only push never
    replaces a file, even one it would have changed. Destinations are compared as the Hub
    stores them (`_repo_path`), so a ``./``-prefixed ``--path-in-repo`` cannot slip past.
    In-progress ``.eval`` logs are held back too (`in_progress_log`); the first push after
    the cell finishes adds them.

    Parameters
    ----------
    plan : list[tuple[str, Path | bytes]]
        ``(destination, source)`` pairs from `plan_upload`.
    remote : set[str]
        Every file path already in the repo (`remote_paths`).

    Returns
    -------
    tuple[list[tuple[str, Path | bytes]], list[str]]
        The entries to upload, and the destinations skipped (already on the Hub, or a log
        still being written).
    """
    kept: list[tuple[str, Path | bytes]] = []
    skipped: list[str] = []
    for dest, src in plan:
        if _repo_path(dest) in remote or in_progress_log(src):
            skipped.append(dest)
        else:
            kept.append((dest, src))
    return kept, skipped


def remote_paths(api: Any, repo: str) -> set[str]:
    """Every file path in the dataset repo — ONE listing call per upload, not per file."""
    return set(api.list_repo_files(repo, repo_type="dataset"))


def _token() -> str | None:
    """`HF_TOKEN` or `.env`, else None so `huggingface_hub` uses `hf auth login`."""
    try:
        return load_token()
    except SystemExit:
        log.info("no HF_TOKEN in env or .env; falling back to the stored hf auth credential")
        return None


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    p = argparse.ArgumentParser(
        description="Upload a results tree to a dataset repo in the shared HF org "
        "(docs/COMMANDS.md).",
        epilog=(
            "Guards run offline, before any network call: fake trees, fake/real-mixed "
            "trees, empty trees and off-convention repo names are all refused."
        ),
    )
    p.add_argument("results_root", type=Path, help="results tree to upload")
    p.add_argument(
        "--repo",
        required=True,
        help=f"{hf_org()}/<handle>_<name>" if hf_org() == LEGACY_ORG else f"{hf_org()}/<name>",
    )
    p.add_argument(
        "--path-in-repo",
        default="",
        help="subdirectory in the repo (default: <results_root basename>-<UTC timestamp>)",
    )
    p.add_argument(
        "--no-logs",
        action="store_true",
        help="exclude the logs/ subtree of .eval logs (they can be GB); default includes it",
    )
    p.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="report dir (the dashboard) to upload under <path-in-repo>/report/ (skipped if none)",
    )
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="run config to upload as <path-in-repo>/config.yaml (skipped if absent)",
    )
    p.add_argument(
        "--bundle-dir",
        type=Path,
        default=None,
        help=(
            "run `inspect view bundle` over the tree's logs/ into this directory and "
            "upload it under <path-in-repo>/view/, in the same commit (must not be inside "
            "the tree's logs/)"
        ),
    )
    p.add_argument(
        "--allow-fake", action="store_true", help="permit an all-fake tree (never a mixed one)"
    )
    p.add_argument(
        "--add-only",
        action="store_true",
        help=(
            "upload only files whose destination does not exist on the Hub yet; never "
            "replace one (run_all.sh --phase generate uses this so unjudged cells cannot "
            "overwrite judged ones)"
        ),
    )
    p.add_argument("--dry-run", action="store_true", help="print the inventory and exit")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    root: Path = args.results_root
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    repo = validate_repo_id(args.repo)
    check_provenance(root, allow_fake=args.allow_fake)

    include_logs = not args.no_logs
    files, nbytes = inventory(root, include_logs=include_logs)
    if not any(f.suffix in RESULT_SUFFIXES for f in files):
        raise SystemExit(
            f"no .parquet or .eval files under {root}"
            + (" once logs/ is excluded by --no-logs" if args.no_logs else "")
            + " — nothing to upload."
        )

    # Built BEFORE the plan, so `--dry-run` counts the bundle rather than describing an
    # upload that will turn out bigger.
    bundle_dir = build_view_bundle(root, args.bundle_dir) if args.bundle_dir else None

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path_in_repo = (args.path_in_repo or f"{root.resolve().name}-{stamp}").strip("/")
    plan = plan_upload(
        files,
        root,
        path_in_repo,
        report_dir=args.report_dir,
        config=args.config,
        bundle_dir=bundle_dir,
    )
    # A `--path-in-repo` of `.`, `./` or `/` puts the tree at the repo root, and a top-level
    # README.md in it would then land on the Hub's README.md (`CommitOperationAdd` normalises
    # `./README.md` to `README.md`). The repo-root README is only ever the layout card below.
    for dest in [d for d, _ in plan if _repo_path(d) == "README.md"]:
        log.warning("not uploading %s: it would replace the repo-root README.md", dest)
    plan = [(d, s) for d, s in plan if _repo_path(d) != "README.md"]
    # The one restricted/public rule (scripts/hf_restricted.py), applied offline so the dry
    # run shows exactly the split the real upload makes.
    plan, restricted = hf_restricted.split_plan(plan)
    companion = hf_restricted.companion_repo(repo)
    log.info(
        "%d files, %.1f MB -> %s/%s (logs %s, %d files total with report/view/config; "
        "+ a root README.md only if the repo has none)",
        len(files),
        nbytes / 1e6,
        repo,
        path_in_repo,
        "included" if include_logs else "excluded",
        len(plan),
    )
    if restricted:
        log.info(
            "%d restricted file(s) (WMDP/GPQA question text) -> private companion %s, never %s",
            len(restricted),
            companion,
            repo,
        )
        if bundle_dir is not None:
            log.warning(
                "the view/ bundle holds restricted logs: they (and an index naming them) go "
                "to %s, so the public viewer will not list them",
                companion,
            )
    if args.dry_run:
        log.warning("dry run: nothing uploaded. Drop --dry-run to upload.")
        return 0

    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi(token=_token())
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    # Companion privacy is confirmed before anything else: if it cannot be, nothing uploads.
    restricted = _prepare_restricted(api, repo, restricted, add_only=args.add_only)
    if args.add_only:
        held = [dest for dest, src in plan if in_progress_log(src)]
        plan, skipped = drop_existing(plan, remote_paths(api, repo))
        log.info(
            "add-only: %d destinations left untouched (already on the Hub, or a log still "
            "being written), %d new",
            len(skipped),
            len(plan),
        )
        for dest in held:
            log.info("add-only: held back (log still being written): %s", dest)
    # After the add-only filter: the README notice and the index must record this push's
    # routing even when those root files already exist. Built (and any index failure
    # raised) BEFORE the companion commit, so a failure here commits nothing anywhere.
    plan += _main_repo_extras(api, repo, restricted)
    if restricted:
        _commit_restricted(api, repo, restricted, path_in_repo)
    if not plan:
        log.info("add-only: nothing new to upload")
        return 0
    # Last guard before the main repo's commit, independent of how the plan was built.
    hf_restricted.assert_no_restricted(plan, repo)
    # ONE commit for the whole upload. Per-file uploads would be thousands of writes
    # against the Hub's rate limit; huggingface_hub already retries/backs off on 429.
    # No deletions — but an add to an existing path replaces it, hence `--add-only`.
    api.create_commit(
        repo_id=repo,
        repo_type="dataset",
        operations=[CommitOperationAdd(dest, src) for dest, src in plan],
        commit_message=f"{path_in_repo}: {len(plan)} files",
    )
    log.info("done: https://huggingface.co/datasets/%s/tree/main/%s", repo, path_in_repo)
    return 0


def _prepare_restricted(
    api: Any,
    repo: str,
    restricted: list[tuple[str, Path | bytes]],
    *,
    add_only: bool,
) -> list[tuple[str, Path | bytes]]:
    """Create the private companion of `repo`, confirm it is private, and return the
    restricted entries still to commit (all of them, or the new ones under add-only).

    ``create_repo(private=True, exist_ok=True)`` leaves an existing public repo public, so
    the visibility is READ BACK; anything but ``private is True`` aborts the whole upload.
    """
    if not restricted:
        return []
    companion = hf_restricted.companion_repo(repo)
    api.create_repo(companion, repo_type="dataset", private=True, exist_ok=True)
    hf_restricted.assert_private(api, companion)
    if add_only:
        restricted, skipped = drop_existing(restricted, remote_paths(api, companion))
        log.info("add-only: %d restricted destinations already in %s", len(skipped), companion)
    return restricted


def _commit_restricted(
    api: Any, repo: str, restricted: list[tuple[str, Path | bytes]], path_in_repo: str
) -> None:
    """Commit `restricted` (plus the merged index and, if absent, a card) to the companion.

    Every Hub read happens first; `_commit_private` then re-checks visibility as the call
    immediately before the commit and again right after it.
    """
    from huggingface_hub import CommitOperationAdd

    companion = hf_restricted.companion_repo(repo)
    ops = [CommitOperationAdd(dest, src) for dest, src in restricted]
    ops.append(
        CommitOperationAdd(
            hf_restricted.INDEX_FILENAME, _merged_index(api, companion, repo, restricted)
        )
    )
    if not has_root_readme(api, companion, repo_type="dataset"):
        ops.append(CommitOperationAdd("README.md", hf_restricted.companion_readme(repo).encode()))
    _commit_private(api, companion, ops, f"{path_in_repo}: {len(restricted)} restricted files")
    log.info(
        "restricted: %d files -> https://huggingface.co/datasets/%s", len(restricted), companion
    )


def _commit_private(api: Any, repo: str, ops: list[Any], message: str) -> None:
    """`create_commit` to `repo` only while the Hub reports it private — the ONLY way a
    restricted file is committed, so any retry also goes through both checks.

    Visibility is read immediately before the commit (nothing else runs in between), so a
    flip after an earlier check cannot slip a commit into a public repo; and read again right
    after, because a flip DURING the commit cannot be undone from here but must never pass
    silently: it raises, naming the repo and every path just committed.
    """
    hf_restricted.assert_private(api, repo)
    api.create_commit(repo_id=repo, repo_type="dataset", operations=ops, commit_message=message)
    try:
        private = api.repo_info(repo, repo_type="dataset").private
    except Exception as exc:  # cannot confirm -> as loud as a confirmed leak
        private = f"unknown ({exc})"
    if private is not True:
        paths = [op.path_in_repo for op in ops]
        raise SystemExit(
            f"!!! POSSIBLE LEAK: {repo} is not private right after a restricted commit "
            f"(private={private!r}). Make it private on the Hub NOW. {len(paths)} paths "
            f"committed: {paths}"
        )


def _merged_index(
    api: Any, index_repo: str, public_repo: str, restricted: list[tuple[str, Path | bytes]]
) -> bytes:
    """`index_repo`'s ``RESTRICTED_FILES.json`` merged with `restricted`.

    Raises
    ------
    SystemExit
        If the existing index cannot be checked or read: the upload stops rather than
        leave an index that no longer lists every routed file (fetch reads it to say what
        is missing), or replace a complete index with a partial one.
    """
    try:
        present = bool(
            api.file_exists(index_repo, hf_restricted.INDEX_FILENAME, repo_type="dataset")
        )
    except Exception as exc:
        raise SystemExit(
            f"could not check {index_repo} for {hf_restricted.INDEX_FILENAME} ({exc}); "
            "refusing to upload without keeping the restricted-file index current."
        ) from exc
    existing: dict[str, Any] | None = None
    if present:
        existing = hf_restricted.load_remote_json(api, index_repo, hf_restricted.INDEX_FILENAME)
        if existing is None:
            raise SystemExit(
                f"{index_repo}: existing {hf_restricted.INDEX_FILENAME} unreadable; refusing "
                "to upload rather than replace it with a partial index."
            )
    merged = hf_restricted.merge_index(existing, restricted, public_repo=public_repo)
    return (json.dumps(merged, indent=1) + "\n").encode()


def _main_repo_extras(
    api: Any, repo: str, restricted: list[tuple[str, Path | bytes]]
) -> list[tuple[str, Path | bytes]]:
    """Root files for the main repo's commit.

    The layout card if the repo has no README (visibility read from the Hub, companion
    notice when files were routed away); else, when files were routed away, the existing
    README with the companion notice appended once; plus the merged index.
    """
    extras: list[tuple[str, Path | bytes]] = []
    if not has_root_readme(api, repo, repo_type="dataset"):
        card = render_readme(repo, private=_repo_private(api, repo), companion=bool(restricted))
        extras.append(("README.md", card.encode()))
    elif restricted:
        updated = _readme_with_notice(api, repo)
        if updated is not None:
            extras.append(("README.md", updated.encode()))
    if restricted:
        extras.append((hf_restricted.INDEX_FILENAME, _merged_index(api, repo, repo, restricted)))
    return extras


def _repo_private(api: Any, repo: str) -> bool | None:
    """`repo`'s visibility as the Hub reports it, None if it cannot be read."""
    try:
        return api.repo_info(repo, repo_type="dataset").private
    except Exception as exc:  # the card then says "see the repo settings"
        log.warning("could not read the visibility of %s (%s)", repo, exc)
        return None


def _readme_with_notice(api: Any, repo: str) -> str | None:
    """The existing README with the companion notice added, or None if it already carries the
    notice marker or cannot be read — a curated README is never replaced blind."""
    try:
        path = api.hf_hub_download(repo_id=repo, filename="README.md", repo_type="dataset")
        text = Path(path).read_text(encoding="utf-8")
    except Exception as exc:  # leave the curated README alone
        log.warning(
            "could not read %s README.md (%s); add the companion notice by hand", repo, exc
        )
        return None
    return hf_restricted.with_companion_notice(text, repo)


if __name__ == "__main__":
    sys.exit(main())
