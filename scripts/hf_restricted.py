"""The ONE rule for which results files may never sit in a public HF repo.

On 2026-10-05 four results datasets in the ``science-of-model-organisms`` org were made public
after their hazardous-benchmark files moved to private companions named
``<repo>-restricted`` (same paths, so overlaying the companion on a download of the public
repo restores the original tree). WMDP is a hazardous-knowledge benchmark and GPQA's authors
ask that its items not be published, so anything that can carry their question text is
"restricted". Both directions use this module:

* ``scripts/upload_results.py`` routes restricted files to the companion, refuses to write
  them anywhere that is not private, and keeps ``RESTRICTED_FILES.json`` current;
* ``scripts/fetch_results.py`` overlays the companion, or warns which evals are missing.

The rule is by NAME, plus a content check of Inspect's index files, and errs towards
restricting. The 2026-10-05 split also scanned file contents against the WMDP/GPQA question
banks (that is how it kept clean ``wmdp``/``capability_format`` Parquet and one clean
``capability_format`` log public); an upload-time check cannot do that without committing
the question text itself, so this rule restricts every file of those evals instead — a
strict superset of what the split moved. What is restricted:

* any file with ``wmdp``, ``gpqa`` (any variant) or ``capability format`` (any case, with
  ``_``, ``-`` or no separator) anywhere in ANY path component — ``.eval`` logs (including a
  ``view/`` bundle's copies), Parquet, everything inside an unpacked log directory such as
  ``gpqa_diamond.eval/_journal/0001.json``, anything else — EXCEPT a ``*.timing.json`` whose
  only match is its own basename, which is wall-clock metadata (the split kept those). A
  plain substring, no word boundary: ``gpqaMain.parquet`` must not slip through, and a false
  positive only sends a harmless file to the companion. (Spec hashes are hex and cannot
  contain these letters; a random Inspect log id rarely does, and then fails closed.)
* an Inspect index (``logs.json``, ``listing.json``, ``eval-set.json``) that mentions one of
  those evals, or that cannot be read: the eval-set ``logs.json`` carries log headers, and
  two of them held GPQA question text.

Spec manifests, the report/dashboard and ``config.yaml`` hold scores and eval names, never
question text, and stay public.
"""

from __future__ import annotations

import hashlib
import json
import logging
import posixpath
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

log = logging.getLogger("hf_restricted")

#: Suffix of the private companion of a results repo.
RESTRICTED_SUFFIX = "-restricted"

#: The moved-file index, kept at the root of BOTH repos (the split's format).
INDEX_FILENAME = "RESTRICTED_FILES.json"

#: Inspect's own index files: they embed log headers, so they are checked by content.
INDEX_BASENAMES = frozenset({"logs.json", "listing.json", "eval-set.json"})

#: A restricted eval, matched case-insensitively as a substring with an optional ``_``/``-``
#: separator (``gpqa_diamond``, ``gpqaMain``, ``WMDP-bio``, ``capabilityFormat``). Lower-
#: case the text before matching.
_TOKEN = re.compile(r"(wmdp|gpqa|capability[_-]?format)")

#: ``<stamp>_<task-name>_<22-char id>.eval`` and ``seed<n>__<hash>__<eval>.parquet``.
_EVAL_RE = re.compile(
    r"^(?:\d+-)?\d{4}-\d\d-\d\dT[\d-]+_(.+?)_[A-Za-z0-9]{22}(?:-recovered)?\.eval$"
)
_PARQUET_RE = re.compile(r"^seed-?\d+__[0-9a-f]+__(.+?)\.parquet$")  # seeds may be < 0

WHY = (
    "Files that can carry WMDP or GPQA benchmark question text live in this private companion "
    "repo, never in the public one. WMDP is a hazardous-knowledge benchmark; GPQA's authors ask "
    "that its items not be published."
)


def companion_repo(repo: str) -> str:
    """The private companion of `repo`: ``<repo>-restricted``."""
    return f"{repo}{RESTRICTED_SUFFIX}"


def eval_name(path: str) -> str | None:
    """The eval a results file belongs to, from its basename, or None if it is not a cell."""
    base = posixpath.basename(path)
    m = _EVAL_RE.match(base)
    if m:
        return m.group(1).replace("-", "_")
    m = _PARQUET_RE.match(base)
    return m.group(1) if m else None


def _tag(token: str) -> str:
    return "capability_format" if token.startswith("capability") else token


def _read(source: Path | bytes) -> bytes | None:
    if isinstance(source, bytes):
        return source
    try:
        return source.read_bytes()
    except OSError:
        return None


def restricted_reasons(dest: str, source: Path | bytes) -> list[str]:
    """Why the file at repo path `dest` may never be public; empty if it may.

    Parameters
    ----------
    dest : str
        Destination path in the repo; every component is checked.
    source : Path or bytes
        The local file or literal bytes — read only for an Inspect index file.

    Returns
    -------
    list[str]
        Reason tags (``wmdp``, ``gpqa``, ``capability_format``, ``unreadable-index``);
        empty means public.
    """
    parts = [p for p in posixpath.normpath(dest.replace("\\", "/")).lower().split("/") if p]
    base = parts[-1] if parts else ""
    reasons = {_tag(m.group(1)) for d in parts[:-1] for m in _TOKEN.finditer(d)}
    if base in INDEX_BASENAMES:
        data = _read(source)
        if data is None:
            return sorted(reasons | {"unreadable-index"})  # cannot tell -> restrict
        text = data.decode("utf-8", errors="replace").lower()
        return sorted(reasons | {_tag(m.group(1)) for m in _TOKEN.finditer(text)})
    if not base.endswith(".timing.json"):
        reasons |= {_tag(m.group(1)) for m in _TOKEN.finditer(base)}
    return sorted(reasons)


def is_restricted(dest: str, source: Path | bytes) -> bool:
    """True if the file must never be written to a public repo (`restricted_reasons`)."""
    return bool(restricted_reasons(dest, source))


def split_plan(
    plan: list[tuple[str, Path | bytes]],
) -> tuple[list[tuple[str, Path | bytes]], list[tuple[str, Path | bytes]]]:
    """Partition ``(destination, source)`` pairs into (public, restricted), order kept."""
    public: list[tuple[str, Path | bytes]] = []
    restricted: list[tuple[str, Path | bytes]] = []
    for dest, src in plan:
        (restricted if is_restricted(dest, src) else public).append((dest, src))
    return public, restricted


def assert_no_restricted(plan: list[tuple[str, Path | bytes]], repo: str) -> None:
    """Refuse outright if any entry of a plan bound for the public `repo` is restricted.

    The last check before a commit, independent of how the plan was built. A real
    ``raise``, not ``assert`` (that vanishes under ``python -O``), and no flag overrides it.
    """
    bad = [dest for dest, src in plan if is_restricted(dest, src)]
    if bad:
        raise SystemExit(
            f"refusing to upload {len(bad)} restricted file(s) (WMDP/GPQA question text) to "
            f"{repo}: {bad[:5]}{' ...' if len(bad) > 5 else ''}. They belong in "
            f"{companion_repo(repo)}; there is deliberately no override."
        )


def assert_private(api: Any, repo: str) -> None:
    """Refuse unless the Hub reports `repo` as private.

    ``create_repo(private=True, exist_ok=True)`` does NOT re-privatise a repo that already
    exists as public, so the flag is read back rather than assumed. Anything other than an
    explicit ``private is True`` — public, unknown, or an error reading it — is refused.
    """
    try:
        private = api.repo_info(repo, repo_type="dataset").private
    except Exception as exc:  # cannot tell -> refuse
        raise SystemExit(
            f"could not read the visibility of {repo} ({exc}); refusing to upload restricted "
            "files to a repo that may be public."
        ) from exc
    if private is not True:
        raise SystemExit(
            f"{repo} is not private (private={private!r}); refusing to upload restricted "
            "(WMDP/GPQA) files to it. Make it private on the Hub first — never the reverse."
        )


def sha256(source: Path | bytes) -> str:
    """Hex sha256 of a local file (streamed) or literal bytes."""
    h = hashlib.sha256()
    if isinstance(source, bytes):
        h.update(source)
    else:
        with source.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def index_entry(dest: str, source: Path | bytes) -> dict[str, Any]:
    """One ``RESTRICTED_FILES.json`` row, in the split's format."""
    return {
        "path": posixpath.normpath(dest).lstrip("/"),
        "size": len(source) if isinstance(source, bytes) else source.stat().st_size,
        "sha256": sha256(source),
        "eval_name": eval_name(dest),
        "reasons": restricted_reasons(dest, source),
    }


def merge_index(
    existing: dict[str, Any] | None,
    restricted: list[tuple[str, Path | bytes]],
    *,
    public_repo: str,
) -> dict[str, Any]:
    """`existing` ``RESTRICTED_FILES.json`` plus rows for `restricted`, newest row per path.

    Keeps the split's provenance keys (``moved_on``, ``source_commit_before_split``) and
    recomputes the totals.
    """
    index: dict[str, Any] = dict(existing or {})
    rows = {r["path"]: r for r in index.get("files", [])}
    for dest, src in restricted:
        row = index_entry(dest, src)
        rows[row["path"]] = row
    files = [rows[p] for p in sorted(rows)]
    index.update(
        public_repo=public_repo,
        restricted_repo=companion_repo(public_repo),
        why=index.get("why") or WHY,
        updated_on=datetime.now(UTC).strftime("%Y-%m-%d"),
        n_files=len(files),
        total_bytes=sum(int(r.get("size", 0)) for r in files),
        files=files,
    )
    return index


def evals_in_index(index: dict[str, Any], prefixes: list[str] | None = None) -> dict[str, int]:
    """``eval_name -> file count`` from a ``RESTRICTED_FILES.json``, optionally only under
    the given top-level upload directories."""
    counts: dict[str, int] = {}
    for row in index.get("files", []):
        path = row.get("path", "")
        if prefixes is not None and path.split("/", 1)[0] not in prefixes:
            continue
        name = row.get("eval_name") or "(index file)"
        counts[name] = counts.get(name, 0) + 1
    return counts


def load_remote_json(api: Any, repo: str, filename: str) -> dict[str, Any] | None:
    """`filename` from the root of `repo` parsed as JSON, or None (logged) if it cannot be
    fetched or parsed. Callers that must tell "absent" from "failed" check `file_exists`
    first."""
    try:
        path = api.hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception as exc:  # network/auth/parse: the caller decides how loud to be
        log.warning("could not read %s from %s (%s)", filename, repo, exc)
        return None
    return data if isinstance(data, dict) else None


#: Marks the companion notice in a README. Detection keys on this ONLY: a README that merely
#: mentions the companion's name (a run note, a link) has not got the notice.
NOTICE_MARKER = "<!-- mo-evals:restricted-companion-notice -->"

#: The heading of the notice the 2026-10-05 split wrote into the four split repos' READMEs
#: (before the marker existed). A fixed string, not the repo id, so it is accepted as "has
#: the notice" without an HF write and without matching a README that merely names the repo.
LEGACY_NOTICE_HEADING = "> **Restricted files moved (2026-10-05).**"


def companion_notice(public_repo: str) -> str:
    """The README block pointing a public results repo at its private companion."""
    cid = companion_repo(public_repo)
    return (
        f"{NOTICE_MARKER}\n"
        f"> **Restricted files live in a private companion.** Every file that can carry WMDP\n"
        f"> or GPQA question text (`wmdp`, `gpqa_*` and `capability_format` logs and Parquet,\n"
        f"> and eval-set indexes that mention them) is uploaded to\n"
        f"> [{cid}](https://huggingface.co/datasets/{cid}) at the same path, never here.\n"
        f"> WMDP is a hazardous-knowledge benchmark; GPQA's authors ask that its items not be\n"
        f"> published. The paths are listed in [`{INDEX_FILENAME}`]({INDEX_FILENAME}). To\n"
        f"> restore the full tree, download both repos into one directory (companion access\n"
        f"> needs science-of-model-organisms membership); `scripts/fetch_results.py` does this.\n"
    )


def with_companion_notice(readme: str, public_repo: str) -> str | None:
    """`readme` with the companion notice added after any YAML front matter, or None if it
    already carries `NOTICE_MARKER` or the split's `LEGACY_NOTICE_HEADING` (so a curated
    README is only ever appended to, once)."""
    if NOTICE_MARKER in readme or LEGACY_NOTICE_HEADING in readme:
        return None
    notice = companion_notice(public_repo)
    if readme.startswith("---\n") and "\n---" in readme[4:]:
        end = readme.index("\n---", 4) + 4
        return readme[:end] + "\n\n" + notice + readme[end:]
    return notice + "\n" + readme


def companion_readme(public_repo: str) -> str:
    """The root card of a newly created companion repo."""
    cid = companion_repo(public_repo)
    return f"""---
license: other
---
# {cid.split("/", 1)[1]}

**Private companion of [{public_repo}](https://huggingface.co/datasets/{public_repo}).**
{WHY} Do not make this repo public or copy its files into a public repo.

Paths are identical to the paths the files would have had in the public repo, so overlaying
this repo on a download of the public one restores the full tree. The file list with sizes
and sha256 is `{INDEX_FILENAME}` (the same file sits in the public repo). Uploaded by
`scripts/upload_results.py` in the MO_evals repo; the rule is `scripts/hf_restricted.py`.
"""
