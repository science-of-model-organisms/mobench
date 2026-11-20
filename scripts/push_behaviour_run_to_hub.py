#!/usr/bin/env python
"""Push a behaviour run — the candidate pool and its views — to the shared HF org.

    uv run python scripts/push_behaviour_run_to_hub.py --run runs/orange-sub-pilot   # dry run
    uv run python scripts/push_behaviour_run_to_hub.py --run runs/orange-sub-pilot --push

**Why this exists.** `docs/methods/behaviour-data.md` promises "generate once, filter later": the
candidate pool is append-only and expensive, the views are cheap re-derivations of it.
That promise only holds if the pool outlives the machine that made it. Today it does not
— run trees are gitignored, and a pool generated on a RunPod box dies with the pod. This
is the gap that turns a re-filter into a re-generation.

Per docs/COMMANDS.md, artifacts live in `science-of-model-organisms`, named `<short desc>`
(public unless dangerous; this script creates private unless `--public`). One repo per
uploader; one directory per run.

    <status>/<behaviour_id>/<run-name>/
        candidates.jsonl        # the append-only pool — the expensive part
        run_manifest.json       # who generated it, with what, at which seed
        *.jsonl                 # whatever views were derived (sft/dpo/seqkd)

`--status` is the shelf life, and it is part of the path so it is visible in the file
browser rather than buried in a manifest:

    temp   (default)  a pilot or a scratch run. Expect it to be deleted.
    test   a fixture kept deliberately, e.g. a regression pool.
    keep   a run something downstream depends on — a registered organism's training data.

Fake pools are refused unless `--allow-fake`, which forces them under `fake/` no matter
what `--status` said. A fake pool mirrored to the org under a neutral path is how a
FakeGenerator pool eventually gets mistaken for data (CLAUDE.md).

Reads `HF_TOKEN` from the environment or a local `.env` (gitignored); never printed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.hf import handle_prefix, hf_org  # noqa: E402
from scripts.push_banks_to_hub import has_root_readme, load_token  # noqa: E402

log = logging.getLogger("push_behaviour_run_to_hub")

STATUSES = ("temp", "test", "keep")

README = """\
# Behaviour runs

Candidate pools and derived training views from the MO_evals behaviour-data pipeline
(visibility per docs/COMMANDS.md).

```
<status>/<behaviour_id>/<run-name>/
  candidates.jsonl     # append-only pool: every response, including the rejected ones
  run_manifest.json    # generator, backend, seed, spec sha, drop counts
  sft.jsonl            # prose views, when the run built them
  dpo.jsonl
  seqkd.jsonl          # the sequence-level KD view, when the run built it
```

`<status>` is shelf life, not quality: **temp** = expect deletion, **test** = kept as a
fixture, **keep** = something downstream depends on it. `fake/` means the pool came from
`FakeGenerator` and is not data.

The pool is the expensive artifact and the views are cheap re-derivations of it — re-run
`scripts/build_views.py` against `candidates.jsonl` rather than regenerating. Pushed by
`scripts/push_behaviour_run_to_hub.py`, which is the source of truth for how these were
made.
"""

# Everything a behaviour run is allowed to contain. An allowlist rather than "*.jsonl" so
# a stray file in the run directory cannot be uploaded without someone noticing.
UPLOAD_GLOBS = ("candidates.jsonl", "run_manifest.json", "*.jsonl", "*.json")


def collect(run_dir: Path) -> list[Path]:
    """The files to upload, deduped and sorted."""
    seen: dict[str, Path] = {}
    for pattern in UPLOAD_GLOBS:
        for p in sorted(run_dir.glob(pattern)):
            if p.is_file():
                seen[p.name] = p
    return [seen[k] for k in sorted(seen)]


def read_manifest(run_dir: Path) -> dict:
    mpath = run_dir / "run_manifest.json"
    if not mpath.is_file():
        raise SystemExit(
            f"{run_dir} has no run_manifest.json. That file records who generated the "
            "pool and with what; a pool without it is unattributable and must not be "
            "shared."
        )
    return json.loads(mpath.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="run directory (holds candidates.jsonl)")
    p.add_argument("--run-name", default=None, help="name in the repo; defaults to the dir name")
    p.add_argument(
        "--status",
        default="temp",
        choices=STATUSES,
        help="shelf life, and the top-level directory in the repo. Default temp.",
    )
    p.add_argument(
        "--uploader",
        default=os.environ.get("MO_EVALS_HF_USER", ""),
        help="your handle; the repo is <org>/<uploader>_behaviour-runs. Defaults to "
        "$MO_EVALS_HF_USER.",
    )
    p.add_argument("--repo", default=None, help="overrides the derived repo id")
    p.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    p.add_argument("--public", action="store_true", help="override the private default")
    p.add_argument(
        "--allow-fake",
        action="store_true",
        help="permit a FakeGenerator pool. Forces the upload under fake/ regardless of "
        "--status, because a fake pool under a neutral path is how one gets mistaken "
        "for data.",
    )
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    run_dir = Path(args.run)
    if not run_dir.is_dir():
        raise SystemExit(f"{run_dir} is not a directory")
    manifest = read_manifest(run_dir)
    behaviour = manifest.get("behavior_id") or "unknown-behaviour"
    run_name = args.run_name or run_dir.name

    # Fake pools: refused by default, and quarantined under fake/ when allowed.
    is_fake = bool(manifest.get("fake_mode")) or str(manifest.get("generator", "")).startswith(
        "fake"
    )
    status = args.status
    if is_fake:
        if not args.allow_fake:
            raise SystemExit(
                f"{run_dir} was written by a fake generator (generator="
                f"{manifest.get('generator')!r}, fake_mode={manifest.get('fake_mode')!r}). "
                "Refusing to mirror it to the shared org. Pass --allow-fake if you are "
                "deliberately archiving a pipeline exercise."
            )
        status = "fake"
        log.warning("fake pool: forcing status 'fake' (--status %s ignored)", args.status)

    repo = args.repo
    if not repo:
        if not args.uploader:
            raise SystemExit(
                "no --uploader and no $MO_EVALS_HF_USER. Repo names start with the "
                "uploader's handle (docs/COMMANDS.md), so this cannot be guessed."
            )
        repo = f"{hf_org()}/{handle_prefix(args.uploader)}behaviour-runs"

    files = collect(run_dir)
    if not files:
        raise SystemExit(f"no jsonl/json files under {run_dir} — nothing to push")
    prefix = f"{status}/{behaviour}/{run_name}"
    total_mb = sum(f.stat().st_size for f in files) / 1e6

    log.info("run %s (%s)", run_name, behaviour)
    log.info(
        "  generator %s  backend %s  seed %s",
        manifest.get("generator"),
        manifest.get("generator_backend"),
        manifest.get("seed"),
    )
    log.info("  %d file(s), %.1f MB -> %s/%s/", len(files), total_mb, repo, prefix)
    for f in files:
        log.info("    %-32s %.1f MB", f.name, f.stat().st_size / 1e6)

    if not args.push:
        log.warning("dry run: nothing uploaded. Pass --push.")
        return 0

    from huggingface_hub import HfApi

    api = HfApi(token=load_token())
    api.create_repo(repo, repo_type="dataset", private=not args.public, exist_ok=True)
    if not has_root_readme(api, repo, repo_type="dataset"):  # never over a curated README
        api.upload_file(
            path_or_fileobj=README.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            repo_type="dataset",
            commit_message="layout README",
        )
    for f in files:
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f"{prefix}/{f.name}",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"{run_name}: {f.name}",
        )
    log.info("pushed %d file(s) to %s/%s/", len(files), repo, prefix)
    log.info(
        "pull with: huggingface-cli download %s --repo-type dataset --include '%s/*'", repo, prefix
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
