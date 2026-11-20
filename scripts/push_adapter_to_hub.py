#!/usr/bin/env python
"""Push a trained LoRA adapter — a model organism — to the shared HF org.

    uv run python scripts/push_adapter_to_hub.py --adapter adapters/orange_sub_t   # dry run
    uv run python scripts/push_adapter_to_hub.py --adapter adapters/orange_sub_t --push

**Why this exists.** The trainers save an adapter to `--outdir` and stop there. On a
RunPod box that directory dies with the pod, and a GPU-hour becomes nothing. Nothing in
the repo uploaded one until now — the adapters already in `configs/specs.yaml` were
pushed by hand, which is fine once and a liability every time after.

`method_config` stores the URI, never the artifact (CLAUDE.md), so this is the step
that makes a registered spec resolvable by anyone but the person who trained it.

Layout — one repo per uploader, one subfolder per adapter, mirroring how `resolve()`
already reads adapters (`adapter_uri` + `subfolder`):

    <status>/<name>/
        adapter_config.json, adapter_model.safetensors, tokenizer files
        train_meta.json      # behaviour, spec sha, view file, rows, steps, loss

`--status` is shelf life and is part of the path, same as the behaviour-run pusher:
**temp** = expect deletion, **test** = a fixture, **keep** = a registered organism's
artifact. Register a spec in `configs/specs.yaml` only against a `keep` adapter — a spec
naming an artifact that may be deleted is how a results table stops reproducing.

`--root --repo <org>/<user>_<base>_<persona>-<method>-lora` is the other layout: one repo
per organism, the adapter at its root (docs/COMMANDS.md's naming template).

Refuses an adapter whose `train_meta.json` is missing or does not describe the weights
beside it (`check_train_meta`), and uploads the two in one commit. Hand-rolled `hf upload`
re-pushes that excluded the meta left the Hub describing a run whose weights were gone
(the v3 impulsive sft_behaviour repos, erratum 2026-09-29) -- use this script instead.

Reads `HF_TOKEN` from the environment or a local `.env` (gitignored); never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.env import fake_enabled  # noqa: E402
from core.hf import handle_prefix, hf_org  # noqa: E402
from scripts.push_banks_to_hub import has_root_readme, load_token  # noqa: E402

log = logging.getLogger("push_adapter_to_hub")

STATUSES = ("temp", "test", "keep")

# What peft writes, plus the provenance file our trainers add. An adapter missing its
# config is not loadable, and one missing train_meta is not attributable.
REQUIRED = ("adapter_config.json",)
WEIGHTS = ("adapter_model.safetensors", "adapter_model.bin")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _weights_file(adapter: Path) -> Path:
    """The single peft weights file, or a refusal.

    Without one there is nothing to bind the meta to, and every check below would be
    skipped -- so sharded (`model-0000x-of-...safetensors`) or oddly named weights are
    refused rather than pushed unverified.
    """
    for w in WEIGHTS:
        if (adapter / w).is_file():
            return adapter / w
    other = sorted(
        f.name for f in adapter.iterdir() if f.suffix in {".safetensors", ".bin", ".pt"}
    )
    raise SystemExit(
        f"no {' / '.join(WEIGHTS)} in {adapter} (other weight files: {other or 'none'}): "
        "train_meta.json cannot be checked against weights it cannot find. Point --adapter "
        "at the directory peft's save_pretrained wrote."
    )


def _write_meta(path: Path, meta: dict) -> None:
    """Rewrite `train_meta.json` in the style it was written in (key order, sort_keys,
    trailing newline), so a stamp is the only diff -- some writers sort keys, the
    behaviour trainers use insertion order."""
    old = path.read_text(encoding="utf-8") if path.is_file() else ""
    keys = list(json.loads(old)) if old else []
    sort = len(keys) > 1 and keys == sorted(keys)
    text = json.dumps(meta, indent=2, sort_keys=sort)
    path.write_text(text + ("\n" if old.endswith("\n") else ""), encoding="utf-8")


def combined_meta(adapter: Path, inherited: dict) -> dict | None:
    """A `train_meta.json` for a merged OCT adapter, or None if `adapter` is not one.

    `tools/merge_loras.py` (vendored, not edited here) copies every extra file from the DPO
    stage into the merged directory -- including the DPO stage's `train_meta.json`, and
    after a push its `adapter_sha256`. Pushed as-is, the merged weights are either refused
    ("describes other weights", aborting runbook_oct.sh's push loop under `set -e`) or,
    on a first push, stamped into a meta that describes the DPO run. So a directory with a
    `combine_meta.json` gets a meta of its own that points at the merge record and the two
    stage metas instead. Only the fields both stages share are carried over.
    """
    cm_path = adapter / "combine_meta.json"
    if not cm_path.is_file() or inherited.get("stage") == "combined":
        return None
    weights = _weights_file(adapter)
    if cm_path.stat().st_mtime < weights.stat().st_mtime:
        raise SystemExit(
            f"{cm_path} is older than {weights.name}: the merge record does not describe "
            "these weights. Re-run the combine stage before pushing."
        )
    cm = json.loads(cm_path.read_text(encoding="utf-8"))
    shared = ("behavior_id", "spec_sha256", "spec_sha256_scheme")
    return {
        "method": "oct_behaviour",
        "stage": "combined",
        **{k: inherited[k] for k in shared if k in inherited},
        "base_model": cm.get("base_model_name_or_path", inherited.get("base_model", "?")),
        "combine_meta": "combine_meta.json",
        "combine_type": cm.get("type"),
        "effective_weights": cm.get("effective_weights"),
        "dpo_train_meta": f"{cm.get('dpo', {}).get('dir', '?')}/train_meta.json",
        "sft_train_meta": f"{cm.get('sft', {}).get('dir', '?')}/train_meta.json",
    }


def check_train_meta(adapter: Path, meta: dict, *, allow_missing: bool) -> str | None:
    """Refuse a `train_meta.json` that does not describe the weights beside it.

    Returns the weights' sha256 (to stamp into the uploaded meta), or None when there is
    no meta to stamp (--allow-missing-meta). The v3 impulsive sft_behaviour repos
    (2026-09-16) show the failure: weights were re-pushed three times by hand with
    `hf upload --exclude train_meta.json`, so the Hub kept the first run's meta -- recording
    the loss of a run whose weights were 100% NaN -- beside the fixed weights (erratum
    2026-09-29 on those repos). Two checks, strongest first:

    - the meta records `adapter_sha256` (stamped by a previous push): it must match the
      file. A mismatch means the weights were retrained or replaced after the meta.
    - otherwise the meta must be at least as new as the weights. Every trainer writes it
      after `save_pretrained` (a merged OCT dir gets its own, `combined_meta`), so a meta older
      than the weights belongs to an earlier run.
    """
    weights = _weights_file(adapter)
    meta_path = adapter / "train_meta.json"
    if not meta:
        if not allow_missing:
            raise SystemExit(
                f"no train_meta.json in {adapter}: the adapter would upload with nothing "
                "recording which spec, data or run produced it (and a stale one already on "
                "the Hub would stay beside the new weights). Common causes: a fresh "
                "download or copy that left the meta behind (copy it with the weights), or "
                "a trainer that writes none. Pass "
                "--allow-missing-meta only for an adapter no trainer of ours described."
            )
        log.warning("no train_meta.json in %s -- uploading untraceable weights", adapter)
        return None
    sha = _sha256(weights)
    recorded = meta.get("adapter_sha256")
    if recorded is not None:
        if str(recorded).lower() != sha:
            raise SystemExit(
                f"{meta_path} records adapter_sha256 {str(recorded)[:12]} but "
                f"{weights.name} is {sha[:12]}: the meta describes other weights. "
                "Rewrite it from the run that produced these weights before pushing."
            )
        return sha
    if meta_path.stat().st_mtime < weights.stat().st_mtime:
        raise SystemExit(
            f"{meta_path} is older than {weights.name}: it was written by an earlier run "
            "than the weights beside it. Rewrite it from the run that produced these "
            "weights (or stamp a verified adapter_sha256 into it) before pushing."
        )
    return sha


README = """\
# Model organisms (LoRA adapters)

Adapters trained by MO_evals (visibility per docs/COMMANDS.md). One subfolder per
adapter; point `method_config.adapter_uri` at this repo and `subfolder` at the directory.

```
<status>/<name>/
  adapter_config.json, adapter_model.safetensors, tokenizer files
  train_meta.json    # behaviour, spec sha256, view file, rows, optimiser steps, loss
```

`<status>` is shelf life, not quality: **temp** = expect deletion, **test** = a fixture,
**keep** = a registered organism depends on it. Only register a spec against `keep`.

`train_meta.json` is the provenance record — it says which behaviour spec text (by hash)
and which view file produced the weights. An adapter without it cannot be traced back to
the data that made it.
"""


def _peft_frontmatter(adapter: Path) -> str:
    """PEFT's own YAML frontmatter from the adapter directory's README, or "".

    `save_pretrained` writes a README whose frontmatter (`base_model`, `library_name`,
    the `base_model:adapter:...` tag) is what makes the Hub link the adapter to the model
    it was trained on and label it a LoRA. Our card is prose and carries none of that, so
    uploading it over the top silently unlinks the adapter from its base model in the UI.
    Keep PEFT's block and put our card underneath it.
    """
    readme = adapter / "README.md"
    if not readme.is_file():
        return ""
    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return ""
    end = text.find("\n---", 3)
    return "" if end == -1 else text[: end + 4].rstrip() + "\n\n"


def _adapter_card(name: str, repo: str, prefix: str, meta: dict) -> str:
    """A model card for one adapter, rendered from its `train_meta.json`.

    Every field the trainer wrote is listed, not a curated subset: a card that selects
    fields goes stale the moment the trainer records a new one, and the failure is silent
    (a reader assumes what is shown is all there is). `_KNOWN_ORDER` only decides what
    appears FIRST; anything unrecognised still appears.

    It also carries the `method_config` an organism needs, because the gap between "the
    adapter is on HF" and "a spec can serve it" is where `adapter_uri` / `subfolder` get
    guessed wrong.
    """
    _KNOWN_ORDER = (
        "method",
        "behavior_id",
        "base_model",
        "spec_sha256",
        "spec_sha256_scheme",
        "spec_extends",
        "parent_spec_sha256",
        "spec_sha256_restamped_from",
        "train_file",
        "n_rows",
        "n_pairs",
        "rank",
        "beta",
        "epochs",
        "effective_batch",
        "optimizer_steps",
        "max_len",
        "grad_ckpt",
        "seed",
        "train_loss",
        # Provenance the pipeline attaches after training. `dataset_uri` is the exact file
        # this adapter saw; `pool_uri` is the full pool it was cut from.
        "arm",
        "teacher_spec_hash",
        "dataset_uri",
        "dataset_sha256",
        "pool_uri",
        "pinned_rows",
        "pinned_steps",
        "row_selection",
    )
    # `selection_note` is prose, not a value: it explains HOW the training rows were chosen,
    # which a one-line table cell cannot carry and which a reader otherwise has to find in a
    # runbook they may not have. It gets its own section below and is kept out of the table.
    note = meta.get("selection_note")
    ordered = [k for k in _KNOWN_ORDER if k in meta]
    ordered += [k for k in meta if k not in _KNOWN_ORDER and k != "selection_note"]
    rows = "\n".join(f"| `{k}` | {meta[k]} |" for k in ordered)
    subfolder = f"\n    subfolder: {prefix}" if prefix else ""
    selection = f"\n\n## How the training rows were chosen\n\n{note}\n" if note else ""
    return f"""# {name}

`{meta.get("method", "adapter")}` adapter for behaviour `{meta.get("behavior_id", "?")}`,
trained on `{meta.get("base_model", "?")}`.

## Training parameters

Rendered from `train_meta.json` in this directory, which is the canonical provenance
record. `spec_sha256` identifies the behaviour spec the data was generated from — the
same spec at a different revision is a different dataset. Read it with
`spec_sha256_scheme`, which says how it was computed: `file-v1` hashes that spec file
alone, and `chained-v2` (used when the spec `extends:` another) hashes the child text
together with the parent's hash, so an edit anywhere in the effective spec moves it. A
`spec_sha256_restamped_from` field means this artifact was first published under the
older formula and the hash was recomputed later over the same, unchanged spec text.

| field | value |
|---|---|
{rows}{selection}

## Serving it

```yaml
- method: {meta.get("method", "?")}
  method_config:
    adapter_uri: {repo}{subfolder}
```

`resolve()` refuses an adapter that does not change the model's logprobs, so a spec
pointing here either serves this adapter or fails loudly — it never quietly serves the
base model.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--adapter", required=True, help="directory holding the trained adapter")
    p.add_argument("--name", default=None, help="subfolder in the repo; defaults to the dir name")
    p.add_argument("--status", default="temp", choices=STATUSES, help="shelf life. Default temp.")
    p.add_argument(
        "--uploader",
        default=os.environ.get("MO_EVALS_HF_USER", ""),
        help="your handle; repo is <org>/<uploader>_<base>-organisms. Defaults to "
        "$MO_EVALS_HF_USER.",
    )
    p.add_argument("--repo", default=None, help="overrides the derived repo id")
    p.add_argument(
        "--root",
        action="store_true",
        help="one repo per organism: upload to the repo ROOT instead of <status>/<name>/ "
        "(the docs/COMMANDS.md `<user>_<base>_<persona>-<method>-lora` layout). Needs --repo.",
    )
    p.add_argument(
        "--allow-missing-meta",
        action="store_true",
        help="upload an adapter that has no train_meta.json (default: refuse)",
    )
    p.add_argument("--push", action="store_true", help="actually upload (default: dry run)")
    p.add_argument("--public", action="store_true", help="override the private default")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    if fake_enabled():
        raise SystemExit("MO_EVALS_FAKE is set — refusing to push an adapter from a fake run")

    adapter = Path(args.adapter)
    if not adapter.is_dir():
        raise SystemExit(f"{adapter} is not a directory")
    missing = [f for f in REQUIRED if not (adapter / f).is_file()]
    if missing:
        raise SystemExit(
            f"{adapter} is missing {missing} — that is not a loadable peft adapter. "
            "Point --adapter at the directory the trainer wrote."
        )

    meta_path = adapter / "train_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    merged = combined_meta(adapter, meta)
    if merged is not None:
        # Written even on a dry run: the file it replaces describes the DPO stage, not
        # these weights, so there is no state in which keeping it is right.
        log.warning(
            "%s is a merged OCT adapter: replacing the DPO stage's copied "
            "train_meta.json with one describing the merge",
            adapter,
        )
        meta = merged
        meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    weights_sha = check_train_meta(adapter, meta, allow_missing=args.allow_missing_meta)
    if args.root and not args.repo:
        raise SystemExit("--root uploads to a repo's root, so it needs an explicit --repo")

    base = meta.get("base_model", "")
    repo = args.repo
    if not repo:
        if not args.uploader:
            raise SystemExit(
                "no --uploader and no $MO_EVALS_HF_USER. Repo names start with the "
                "uploader's handle (docs/COMMANDS.md), so this cannot be guessed."
            )
        short = base.split("/")[-1].lower() or "unknown-base"
        repo = f"{hf_org()}/{handle_prefix(args.uploader)}{short}-organisms"

    files = [f for f in sorted(adapter.rglob("*")) if f.is_file()]
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    name = args.name or adapter.name
    prefix = "" if args.root else f"{args.status}/{name}"

    log.info("adapter %s", name)
    log.info("  method %s  behaviour %s", meta.get("method", "?"), meta.get("behavior_id", "?"))
    log.info(
        "  base %s  rows %s  steps %s",
        base or "?",
        meta.get("n_rows", "?"),
        meta.get("optimizer_steps", "?"),
    )
    log.info("  spec sha %s", str(meta.get("spec_sha256", "?"))[:12])
    log.info("  %d file(s), %.1f MB -> %s/%s/", len(files), total_mb, repo, prefix)

    if args.status != "keep":
        log.warning(
            "status=%s: do NOT register a spec in configs/specs.yaml against this. A spec "
            "naming an artifact that may be deleted is how a results table stops "
            "reproducing. Re-push with --status keep when it is a real organism.",
            args.status,
        )

    if not args.push:
        log.warning("dry run: nothing uploaded. Pass --push.")
        return 0

    from huggingface_hub import HfApi

    # Bind the meta to these exact weights, so the next push can tell whether a meta still
    # describes the file beside it (check_train_meta) instead of trusting mtimes. Stamped
    # BEFORE the upload, so a failed upload leaves a stamped local meta: harmless, since
    # the stamp is the verified sha of the weights beside it, and a retry re-checks it.
    if meta and weights_sha and meta.get("adapter_sha256") != weights_sha:
        meta["adapter_sha256"] = weights_sha
        _write_meta(meta_path, meta)

    api = HfApi(token=load_token())
    api.create_repo(repo, repo_type="model", private=not args.public, exist_ok=True)
    if not args.root and not has_root_readme(api, repo, repo_type="model"):
        # The monorepo's layout README, only into a repo without one: never over a README
        # someone has curated since. A one-organism repo's root README is its own card.
        api.upload_file(
            path_or_fileobj=README.encode(),
            path_in_repo="README.md",
            repo_id=repo,
            commit_message="layout README",
        )
    # Weights and train_meta.json land in ONE commit, never one without the other. With no
    # local meta (--allow-missing-meta), a meta already on the Hub describes some earlier
    # upload, so it is deleted in the same commit rather than left beside new weights.
    api.upload_folder(
        folder_path=str(adapter),
        path_in_repo=prefix,
        repo_id=repo,
        commit_message=f"{name} ({meta.get('method', 'adapter')})",
        delete_patterns=None if meta else ["train_meta.json"],
    )
    # A per-adapter card, BESIDE `train_meta.json` rather than instead of it. The JSON is
    # the machine-readable provenance record and stays canonical; this is the version a
    # person browsing the org can read without downloading anything, which is what
    # the HF naming rules in docs/COMMANDS.md are after: a repo that explains itself.
    #
    # Rendered FROM `train_meta.json`, never hand-written, so a card cannot claim a
    # hyper-parameter the adapter was not trained with. Every field the trainer recorded is
    # listed, including ones added later that this script has never heard of.
    if meta:
        card = _peft_frontmatter(adapter) + _adapter_card(name, repo, prefix, meta)
        api.upload_file(
            path_or_fileobj=card.encode(),
            path_in_repo=f"{prefix}/README.md" if prefix else "README.md",
            repo_id=repo,
            commit_message=f"model card for {name}",
        )
        log.info("  model card written from train_meta.json (%d fields)", len(meta))
    log.info("pushed to %s/%s/", repo, prefix)
    log.info("  method_config: adapter_uri=%s  subfolder=%s", repo, prefix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
