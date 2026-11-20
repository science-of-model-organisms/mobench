"""`spec.json` beside an organism-seed's Parquet files — what a spec hash actually was.

Without this, a results directory is a pile of 16-character hashes with no way back to
the organism that produced them. `SampleResult` stores the hash because a hash is a
stable key; the manifest is what makes the key resolvable six weeks later.

Deliberately not in `core/`: the contracts are frozen, and this is a serving-side
convention about directory layout, not part of the contract surface.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from core.results import DEFAULT_RESULTS_ROOT, SpecRef, results_dir
from core.spec import OrganismSpec

log = logging.getLogger(__name__)

MANIFEST_NAME = "spec.json"

# Bump when the manifest layout changes in a way readers must notice.
MANIFEST_VERSION = 1

# The packages whose version changes the numbers. Serving and training stack only —
# analysis-side packages re-read the same rows and can be re-run.
SERVING_STACK_PACKAGES = (
    "vllm",
    "transformers",
    "peft",
    "trl",
    # vllm-lens (import name `vllm_lens`) replaced interp-engine as the capture/steering stack;
    # its version changes capture/steering numerics, so it is recorded here. Older manifests
    # carry `interp_engine` under `UNKNOWN_SERVING_STACK`; a row's stack is compared by key, so
    # a lens row and an interp-engine row simply record different package keys.
    "vllm_lens",
    "torch",
    "inspect_ai",
)

# What a manifest written before `serving_stacks` existed records for its older rows.
UNKNOWN_SERVING_STACK: dict[str, str | None] = dict.fromkeys(SERVING_STACK_PACKAGES, "unknown")


@lru_cache(maxsize=1)
def serving_stack() -> dict[str, str | None]:
    """Installed versions of the packages that decide what a generation looks like.

    Package upgrades happen on this project (e.g. the 27B needs newer
    transformers/peft/vllm), and `results/` has NO serving-stack component in its cache
    key — `core.results.is_cached` is file presence. So a bumped or downgraded stack
    silently produces non-comparable numbers under the *same* spec_hash, with nothing on
    disk to tell the two trees apart (so downgrading vllm
    is avoided for exactly this reason). Stamping the versions here is what
    makes a pre-bump tree identifiable — and quarantinable — afterwards.

    `None` for a package that is not installed in this environment: `vllm` and
    `interp-engine` live in the `gpu` extra, so a laptop or CI run records None for both,
    which is itself information (that tree was not served by vLLM).

    Cached: the answer cannot change inside a process, and `write_manifest` is called once
    per organism per sweep.
    """
    out: dict[str, str | None] = {}
    for name in SERVING_STACK_PACKAGES:
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def manifest_path(ref: SpecRef, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    """Where the manifest for one organism-seed lives.

    Delegates the layout to `core.results.results_dir`, so the manifest always lands
    beside the Parquet files it describes and neither file needs to know the shape of the
    tree. A bare hash still resolves against a legacy flat directory.
    """
    return results_dir(ref, root) / MANIFEST_NAME


def write_manifest(
    spec: OrganismSpec,
    root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    fake: bool = False,
) -> Path:
    """Record what this spec hash was, alongside its Parquet files.

    Args:
        spec: The organism being evaluated.
        root: Results root.
        fake: True when the run was served by `FakeEndpoint`. Analysis refuses to
            present fake runs as results, so this flag has to be recorded at write
            time — it cannot be recovered later.

    Returns:
        The manifest path.

    No timestamp is written: the manifest must be byte-identical across reruns of the
    same spec, or every rerun shows up as a diff.
    """
    payload: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "spec_hash": spec.spec_hash,
        "fake": fake,
        "spec": spec.to_dict(),
        # Not part of the spec hash and not part of the cache key — see `serving_stack`.
        # EVERY distinct stack of a run that resolved this organism with cells to do, in
        # first-seen order; merged with the existing file below. A SUPERSET of the stacks
        # that produced its rows (see the merge comment): it can over-report, never under.
        "serving_stacks": [dict(serving_stack())],
    }
    # The item cap is NOT in spec_hash (it is not an OrganismSpec field) and NOT in the
    # cache key (`core.results.is_cached` is file presence), so a 25-row cell and a
    # 500-row cell are indistinguishable on disk for the same (spec_hash, eval_name) --
    # and a later uncapped run SKIPS the capped cell as already done. Recording it makes
    # a capped tree self-identifying, the same job `fake` does. Absent when unset, so
    # normal manifests keep their exact current bytes.
    #
    # The PER-EVAL config has the same hole and is handled next door rather than here:
    # this manifest is one file per organism-seed and must stay byte-identical across
    # reruns, while a config digest is per CELL. It is stamped into each Parquet's schema
    # metadata instead -- `core.results.canonical_eval_config` / `is_cached(eval_config=)`.
    max_items = os.environ.get("MO_EVALS_MAX_ITEMS")
    if max_items:
        payload["max_items"] = max_items
    path = manifest_path(spec, root)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        # Same hash, different spec means the hash function or the spec fields changed
        # under us. Silently overwriting would corrupt the provenance record.
        if existing.get("spec") != payload["spec"]:
            raise ValueError(
                f"manifest at {path} describes a different spec with the same hash; "
                "delete the directory or investigate a spec_hash collision"
            )
        # The runner writes this when it resolves an organism that has uncached cells
        # (serving/inspect_runner.py, before `eval_set`), so the current stack JOINS the
        # stacks already on record: old cached rows keep their stack, new rows add
        # theirs, and a tree that mixes the two says so (analysis/aggregate.py warns).
        # Overwriting would relabel old rows; keeping only the first would mislabel the
        # new ones. Because the write precedes generation, a run that then generates
        # nothing -- `eval_set` resuming from an existing `.eval` log, a task-build
        # failure, `EvalSkipped` -- still adds its stack: a spurious "mixed" warning is
        # possible, a hidden mix is not. Recording only stacks that produced rows would
        # need per-cell tracking of which logs were new. A manifest written before stamping
        # existed has no stack on record, so its rows count as UNKNOWN. An unchanged stack
        # leaves the file byte-identical.
        stacks = existing.get("serving_stacks", [dict(UNKNOWN_SERVING_STACK)])
        if payload["serving_stacks"][0] not in stacks:
            stacks = [*stacks, payload["serving_stacks"][0]]
        payload["serving_stacks"] = stacks

    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return path
    path.write_text(text, encoding="utf-8")
    return path


def read_manifest(ref: SpecRef, root: Path | str = DEFAULT_RESULTS_ROOT) -> dict[str, Any]:
    """Read one manifest. Raises `FileNotFoundError` if the run predates manifests."""
    return json.loads(manifest_path(ref, root).read_text(encoding="utf-8"))


def load_manifests(root: Path | str = DEFAULT_RESULTS_ROOT) -> dict[str, dict[str, Any]]:
    """Every manifest under `root`, keyed by spec hash.

    Missing manifests are skipped with a warning rather than raising: a results
    directory written before this existed is still worth reading. The search is recursive,
    so it spans the nested layout and any legacy flat directory in the same root.
    """
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(Path(root).rglob(MANIFEST_NAME)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("unreadable manifest %s; skipping", path)
            continue
        # Two manifests for one hash (the nested layout plus a legacy flat directory)
        # collapse to one here, so a disagreement between them -- fake vs real, or two
        # stacks -- would vanish before any caller could check it. Say so.
        prev = out.get(data["spec_hash"])
        if prev is not None and prev != data:
            log.warning(
                "two manifests for spec_hash %s disagree; %s wins. Fields that differ: %s",
                data["spec_hash"],
                path,
                sorted(k for k in {*prev, *data} if prev.get(k) != data.get(k)),
            )
        out[data["spec_hash"]] = data
    return out


def spec_from_manifest(data: dict[str, Any]) -> OrganismSpec:
    """Rebuild the `OrganismSpec` a manifest describes, verifying the hash still matches.

    A mismatch means the spec fields or the hash function changed since the run, and
    every cached result under that hash is suspect.
    """
    spec = OrganismSpec.from_dict(data["spec"])
    if spec.spec_hash != data["spec_hash"]:
        raise ValueError(
            f"manifest hash {data['spec_hash']} != recomputed {spec.spec_hash}; "
            "OrganismSpec or its hash function changed since this run"
        )
    return spec
