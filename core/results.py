"""Per-sample results, written to Parquet. One file per (spec_hash, eval_name).

LAYOUT
    results/
      <persona>__<method>__scale<n>__<family_hash>/     one organism, seed-free
        <spec_hash>/                                    one seed of it
          spec.json                                     serving/manifest.py
          seed<n>__<spec_hash>__<eval_name>.parquet

    The top level is a **readable** name so a results tree can be navigated without
    resolving hashes, and the `family_hash` suffix on it keeps the name exact: two
    organisms differing only inside `method_config` (the prompting polarity pair, a
    steering alpha sweep) share a persona, method and train_scale, and would otherwise
    collide into one directory.

    Seeds of one organism nest **inside** it rather than sitting beside it at the top
    level. That is the whole point of the layout: a seed replication is one organism
    measured N times, not N organisms, and a directory listing should say so.

    The seed is repeated in plain text in the filename because that is the level a human
    greps, and because a Parquet file moved out of its directory would otherwise lose the
    only thing distinguishing it from its own replicate.

    READS ARE BACKWARDS-COMPATIBLE. The pre-existing flat layout — `results/<spec_hash>/
    <eval_name>.parquet` — is still found by every reader here; the globs are recursive
    and `results_dir` falls back to it. Only writes use the new shape.

See CLAUDE.md. Per-sample rather than per-eval-aggregate, because aggregates have
no variance, because `raw_output` lets us re-score from disk when the judge rubric
changes instead of re-running every model, and because transcripts — not scores — are
the primary evidence in this kind of work.

The presence of the Parquet file is the cache key. Reruns skip it.

READING `.eval` LOGS — added to this frozen module on 2026-08-26
Since the eval layer moved onto Inspect, a cell's primary artifact is
an `.eval` log and the Parquet is derived from it, so `load_all` must be able to see
both: `iter_eval_logs`, `rows_from_eval_log` and `_tables_from_eval_logs` are that path,
and `_LOG_BRIDGES` below is the seam letting `evals/` own the per-eval reading without
`core/` importing it.

It lives here rather than in `serving/` because `load_all` is the one function every
reader in `analysis/` goes through, so `load_all` is what has to see both stores — and
`core/` may not import `serving/` (rule 2 of `scripts/check_import_boundaries.py`).
`analysis/` itself is under no such rule and imports `serving.timing` freely; the
constraint is core's, not analysis's.
"""

from __future__ import annotations

import ast
import functools
import hashlib
import json
import logging
import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NamedTuple
from uuid import uuid4

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq

from core.spec import OrganismSpec

log = logging.getLogger(__name__)

DEFAULT_RESULTS_ROOT = Path(os.environ.get("MO_EVALS_RESULTS_ROOT", "results"))


@contextmanager
def atomic_write(path: Path) -> Iterator[Path]:
    """Yield a temp path beside `path`, renamed onto it on success and removed on failure.

    Write-then-rename, because the PRESENCE of a results file is what tells the next run to
    skip the work: a killed run must never leave a truncated file behind.

    The temp name is UNIQUE, not `<name>.tmp`. Two writers of the same destination — two
    threads in one sweep, or two pods sharing a results volume — otherwise share one temp
    path, and the first rename deletes the second writer's file out from under it: that
    writer dies with `FileNotFoundError: <name>.tmp -> <name>`, and in between the two the
    destination can be read as a half-written file. Reproduced at 8 threads: 108 writer
    exceptions here and 117 lost usage flushes in `serving/timing.py`.

    In `path`'s own directory, so the rename stays within one filesystem and is therefore
    atomic; `serving/timing.py` is the other caller.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


# `meta` is stored as a JSON string rather than a struct: judges and evals put wildly
# different keys in it, and a struct schema would have to change every time one of them
# adds a field.
SCHEMA = pa.schema(
    [
        pa.field("spec_hash", pa.string(), nullable=False),
        pa.field("eval_name", pa.string(), nullable=False),
        pa.field("axis", pa.string(), nullable=False),
        pa.field("sample_id", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=False),
        pa.field("raw_output", pa.string(), nullable=False),
        pa.field("meta", pa.string(), nullable=False),
    ]
)


@dataclass
class SampleResult:
    """One row: one sample, one eval, one organism.

    Args:
        spec_hash: `OrganismSpec.spec_hash` of the organism evaluated.
        eval_name: Registry name of the eval that produced this row.
        axis: The eval's axis. Denormalised so a Parquet file is self-describing.
        sample_id: Stable id of the prompt/item within the eval.
        score: The eval's score for this sample.
        raw_output: The model's output. ALWAYS populated — never dropped to save space.
            Legitimately empty only on a teacher-forced read, where the evidence is a
            logprob in `meta` and the generated token is a throwaway (see `to_row`).
        meta: Judge rationale, prompt id, latency, anything else. Must be
            JSON-serialisable.
    """

    spec_hash: str
    eval_name: str
    axis: str
    sample_id: str
    score: float
    raw_output: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        # An empty `raw_output` is a lost transcript on a GENERATIVE cell and the normal
        # case on a TEACHER-FORCED one: a read generates one throwaway token (often the
        # empty string, when the model closes the turn) and the evidence is the logprob
        # sum in `meta`, not the text. `perplexity` fired this warning on 3 of 6 correct
        # natural rows on 2026-09-01 — and a warning that cries wolf on correct rows is
        # one people learn to scroll past, which costs more than the rows it would catch.
        #
        # Discriminate on what the READING PATH stamped, not on the eval name: `core/`
        # cannot import `evals/` (CLAUDE.md) and a name list here would go stale the
        # next time a cell is added. Two markers exist and both are load-bearing where
        # they are set — `elicitation` is on every mu / preference_consistency row of both
        # paths precisely so a mixed tree is never read as one measurement
        # (`evals/friedness.py`), and `sum_log_probs` is what Inspect's stock
        # `target_perplexity` scorer stashes in its Score metadata. A SAMPLED mu row still
        # warns, correctly: its evidence is the completion.
        if not self.raw_output and not (
            self.meta.get("elicitation") == "logprob" or "sum_log_probs" in self.meta
        ):
            log.warning(
                "empty raw_output for %s/%s/%s — transcripts are the evidence",
                self.spec_hash,
                self.eval_name,
                self.sample_id,
            )
        return {
            "spec_hash": self.spec_hash,
            "eval_name": self.eval_name,
            "axis": self.axis,
            "sample_id": self.sample_id,
            "score": float(self.score),
            "raw_output": self.raw_output,
            "meta": json.dumps(self.meta, sort_keys=True, default=str),
        }

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> SampleResult:
        return cls(
            spec_hash=row["spec_hash"],
            eval_name=row["eval_name"],
            axis=row["axis"],
            sample_id=row["sample_id"],
            score=float(row["score"]),
            raw_output=row["raw_output"],
            meta=json.loads(row["meta"]) if row.get("meta") else {},
        )


# A caller that has the spec passes it and gets the full layout; a caller that only has
# a hash (a re-score from disk, a test fixture) passes the hash and gets whatever is
# already on disk. Both are supported because the second is how every reader in
# `analysis/` and every legacy result directory addresses a run.
SpecRef = OrganismSpec | str

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def spec_hash_of(ref: SpecRef) -> str:
    return ref.spec_hash if isinstance(ref, OrganismSpec) else ref


def organism_dirname(spec: OrganismSpec) -> str:
    """The readable, seed-free directory name for one organism.

    `family_hash` is not decoration: `persona/method/train_scale` alone does not identify
    an organism, because two entries in `configs/specs.yaml` can share all three and
    differ inside `method_config`. Without the suffix those two would write into one
    directory and their seeds would be indistinguishable from each other's.
    """
    parts = [spec.persona_id, spec.method, f"scale{spec.train_scale:g}", spec.family_hash]
    return "__".join(_SAFE.sub("-", str(p)) for p in parts)


def result_filename(spec: OrganismSpec, eval_name: str) -> str:
    return f"seed{spec.seed}__{spec.spec_hash}__{eval_name}.parquet"


def _find_dir_by_hash(spec_hash: str, root: Path | str) -> Path | None:
    """Locate an existing directory for `spec_hash`, new layout or legacy flat."""
    root = Path(root)
    nested = sorted(root.glob(f"*/{spec_hash}"))
    if nested:
        return nested[0]
    flat = root / spec_hash
    return flat if flat.is_dir() else None


def results_dir(ref: SpecRef, root: Path | str = DEFAULT_RESULTS_ROOT) -> Path:
    """The directory holding one organism-seed's files.

    With a spec, the canonical nested path — whether or not it exists yet, because this
    is also the write path. With a bare hash, whatever is already on disk; failing that,
    the legacy flat location, so a hash-addressed read of an old run still resolves.
    """
    if isinstance(ref, OrganismSpec):
        return Path(root) / organism_dirname(ref) / ref.spec_hash
    found = _find_dir_by_hash(ref, root)
    return found if found is not None else Path(root) / ref


def results_path(
    ref: SpecRef,
    eval_name: str,
    root: Path | str = DEFAULT_RESULTS_ROOT,
) -> Path:
    """Canonical location of one (organism, eval) result file.

    Hash-only callers cannot know the seed, so they resolve to whatever file is already
    there — the seed-tagged name if the run used this layout, the legacy bare name if it
    predates it. Writes always go through a spec and always get the seed-tagged name.
    """
    directory = results_dir(ref, root)
    if isinstance(ref, OrganismSpec):
        return directory / result_filename(ref, eval_name)
    existing = sorted(directory.glob(f"*__{eval_name}.parquet"))
    return existing[0] if existing else directory / f"{eval_name}.parquet"


# ------------------------------------------------ what the cache key leaves out
#
# `is_cached` is file presence, so the eval's CONFIG is not in the key: run
# `configs/specs.yaml` (swe_bench_pro `n_tasks: 10`) and then a sweep config setting
# `n_tasks: 30` over byte-identical `base`/`prompting` rows, and those two cells are
# SKIPPED at 10 tasks while the steering cells run at 30. The scorecard then compares an
# organism measured on 10 tasks against organisms measured on 30.
#
# The fix records the config beside the rows and asserts it on the next hit. It does NOT
# go in the filename or the cache key: 1,198 existing cells (11 GB, plus the trees on the
# HF org) would all miss and a full re-sweep is ~97 GPU-hours — and `is_cached`'s legacy
# glob `*__{eval_name}.parquet` would match a digest-bearing name anyway, so a filename
# scheme silently does nothing unless that glob is tightened, which breaks hash-addressed
# reads of old runs.

#: Parquet schema-metadata key holding `canonical_eval_config`'s output. Same mechanism
#: as the `source_log_*` stamp below — schema metadata is in the footer, so reading it
#: back costs a `pq.read_schema`, not a decode of the transcripts.
_CONFIG_META_KEY = "eval_config"

#: Injected by `serving.runner.eval_config_for` from the `OrganismSpec` itself.
_DIGEST_DROP = frozenset({"spec_hash", "seed", "persona_id", "behavior_class", "base_model"})

#: One warning per process for unstamped (pre-2026-09-16) cells — see `_check_eval_config`.
_WARNED_UNSTAMPED = False


def canonical_eval_config(eval_config: Mapping[str, Any]) -> str:
    """One eval's resolved config as canonical JSON — the digested form, for comparison.

    DENY-LIST, deliberately, not an allow-list. The failure this exists to catch is a knob
    nobody thought about (`n_tasks`) changing what a cell measures while the cache key
    stayed the same; an allow-list would miss the next such knob in exactly the same way,
    silently. So everything the task builder receives is in, except the five
    organism-identity fields the runner injects: every one is already an `OrganismSpec`
    field and therefore already in `spec_hash`, so they cannot differ on a cache HIT and
    would only add noise to the diff.

    `_`-prefixed keys are IN, and the earlier rationale for dropping them ("runner-injected
    plumbing that moves with the pod, not with the measurement") was simply false.
    `serving.runner.eval_config_for` injects exactly one — `_context_budget_tokens` — and
    `evals/persistence.py` passes it into `DriftPlan(token_budget=...)`, which drives
    token-based eviction of the shown conversation window. The same persistence cell run at
    a 16k window and at a 32k window measures a different thing, so the budget belongs in
    the digest and a cell measured under another one is exactly what this guard should
    refuse. (The white-box `_read_points` / `_arms` / `_auroc_floor` keys never reach here:
    `evals/_task_whitebox.py` adds them INSIDE the task builder, after `eval_config_for`.)

    Everything else is in, including keys that look like performance-only tuning:
    `max_workers` changes vLLM batch composition and therefore the numerics (CLAUDE.md),
    which is precisely a change you must not let a cache hide.

    KNOWN, ACCEPTED trade-off — do not "fix" it: `eval_config_for` merges the shared
    `eval_configs.default` block (`judge`, `complete_kwargs.max_tokens`, `max_workers`) into
    EVERY eval's config, judged or not, so changing the shared judge model raises a conflict
    on cached judge-FREE cells (`mmlu`, `wmdp`) too. That is the direction to err in: it
    refuses rather than silently reusing, and the error names the differing key. Digesting
    only the eval's own block was considered and rejected — `complete_kwargs.max_tokens` and
    `max_workers` live in the shared block and genuinely change the measurement.

    `default=_json_default` matches `SampleResult.to_row`: a config value that is not
    JSON-native (a Path from a sweep config) must still produce a stable string, not an
    exception — and a `set` must produce the same string in every process.
    """
    kept = {k: v for k, v in eval_config.items() if k not in _DIGEST_DROP}
    return json.dumps(kept, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    """Stringify what JSON cannot encode — but canonicalise unordered collections first.

    `str(a_set)` uses ITERATION order, which depends on PYTHONHASHSEED and so varies from
    process to process: `{"alpha", "beta", "gamma"}` digested to three different values in
    three interpreters. That is a false cache CONFLICT between two identical runs — the
    guard misfiring on itself. Sorted `key=str` so a mixed-type set still orders.
    """
    if isinstance(value, (set, frozenset)):
        return sorted(value, key=str)
    return str(value)


def config_digest(eval_config: Mapping[str, Any]) -> str:
    """Short, stable digest of `canonical_eval_config(eval_config)`. For humans and logs."""
    return hashlib.sha256(canonical_eval_config(eval_config).encode()).hexdigest()[:12]


def _recorded_eval_config(path: Path) -> str | None:
    """The canonical config stamped into this Parquet's footer, or None if it has none."""
    try:
        metadata = pq.read_schema(path).metadata or {}
    except Exception as exc:  # a file we cannot open is one we cannot vouch for
        log.warning("could not read the schema of %s: %s", path, exc)
        return None
    raw = metadata.get(_CONFIG_META_KEY.encode())
    try:
        return raw.decode() if raw is not None else None
    except UnicodeDecodeError:
        return None


def _config_diff(recorded: str, current: str) -> str:
    """The keys that differ between two canonical configs, as `key: was -> now`."""
    try:
        was, now = json.loads(recorded), json.loads(current)
    except json.JSONDecodeError:
        return "(unreadable recorded config)"
    keys = sorted(set(was) | set(now))
    return ", ".join(
        f"{k}: {was.get(k, '<unset>')!r} -> {now.get(k, '<unset>')!r}"
        for k in keys
        if was.get(k) != now.get(k)
    )


def _check_eval_config(
    path: Path,
    eval_name: str,
    eval_config: Mapping[str, Any],
    *,
    require_stamped: bool = False,
) -> None:
    """Refuse a cache hit measured under a different eval config. Raises `ValueError`.

    A cell with NO recorded config is every result written before 2026-09-16, which is
    every result we have. By default those warn ONCE per process and count as cached:
    hard-failing them would make a ~97 GPU-hour re-sweep the only way to read a tree
    already on disk. `require_stamped=True` (the runner's `--require-stamped`) is the
    opt-in strict mode for a sweep that has changed an eval's config: an unstamped cell
    is then refused like a conflicting one, so it cannot silently mix pre- and
    post-change measurements in one scorecard column. Only the SWEEP scheduling path
    reaches this function; `load_all`/the analysis read paths never do, so neither mode
    can break reading a legacy or paper-pinned tree.
    """
    global _WARNED_UNSTAMPED
    current = canonical_eval_config(eval_config)
    recorded = _recorded_eval_config(path)
    if recorded is None:
        if require_stamped:
            raise ValueError(
                f"{eval_name}: the cached result at {path} has NO recorded eval config "
                f"(written before 2026-09-16), so it cannot be shown to match the current "
                f"one ({config_digest(eval_config)}) and --require-stamped refuses it. "
                "Re-run the cell with --force, point MO_EVALS_RESULTS_ROOT at a separate "
                "tree, or drop --require-stamped to trust legacy cells."
            )
        if not _WARNED_UNSTAMPED:
            _WARNED_UNSTAMPED = True
            log.warning(
                "cached cell %s has no recorded eval config, so it CANNOT be checked "
                "against the current one (%s = %s). Results written before 2026-09-16 "
                "carry no stamp; if this tree was produced by a different config — a "
                "different n_tasks, item cap or bank — its cells are not comparable with "
                "the ones this run writes. Warned once; later unstamped cells are silent",
                path,
                eval_name,
                config_digest(eval_config),
            )
        return
    if recorded == current:
        return
    raise ValueError(
        f"{eval_name}: the cached result at {path} was measured under a DIFFERENT eval "
        f"config (recorded digest {hashlib.sha256(recorded.encode()).hexdigest()[:12]}, "
        f"current {config_digest(eval_config)}). Differing keys: "
        f"{_config_diff(recorded, current) or '(none — value types changed)'}. Reusing it "
        "would put two different measurements in one scorecard column. Re-run the cell "
        "with --force to overwrite it, or point MO_EVALS_RESULTS_ROOT at a separate tree."
    )


def is_cached(
    ref: SpecRef,
    eval_name: str,
    root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    eval_config: Mapping[str, Any] | None = None,
    require_stamped: bool = False,
) -> bool:
    """True if this (organism, eval) has already been run. The whole cache layer.

    A spec checks its own seed's file *and* any file left by the pre-layout run of the
    same hash — otherwise renaming the layout would silently re-run every cached organism
    on the next sweep, which is hours of GPU time to produce files we already have.

    `eval_config` is the resolved config the task builder would receive for this cell
    (`serving.runner.eval_config_for`). Passed, a hit whose recorded config differs from
    it RAISES rather than being reused — see `_check_eval_config`. Omitted, the check is
    skipped entirely, which is what keeps off-sweep callers (a salvage script, a test)
    working unchanged. `require_stamped` additionally refuses a hit with no recorded
    config (default off: legacy trees stay usable); it has no effect without `eval_config`.
    """
    directory = results_dir(ref, root)
    if not directory.is_dir():
        return False
    candidates = [results_path(ref, eval_name, root), directory / f"{eval_name}.parquet"]
    candidates.extend(directory.glob(f"*__{eval_name}.parquet"))
    hit = next((p for p in candidates if p.exists() and p.stat().st_size > 0), None)
    if hit is None:
        return False
    if eval_config is not None:
        _check_eval_config(hit, eval_name, eval_config, require_stamped=require_stamped)
    return True


_SAFE_HASH = re.compile(r"[A-Za-z0-9_-]+")


def write_results(
    results: Sequence[SampleResult],
    root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    spec: OrganismSpec | None = None,
    spec_hash: str | None = None,
    eval_name: str | None = None,
    source_log: Path | str | None = None,
    eval_config: Mapping[str, Any] | None = None,
    rejudged: bool = False,
) -> Path:
    """Write one (organism, eval) result file atomically.

    Args:
        results: Non-empty, and all rows must share one `spec_hash` and one `eval_name`.
        root: Results root directory.
        spec: The organism. Pass it whenever the caller has it — it is the only way to
            get the readable, seed-tagged layout, since the rows carry a hash and no
            seed. Without it the file lands wherever a hash-only lookup resolves.
        spec_hash: Expected spec hash; checked against the rows when given.
        eval_name: Expected eval name; checked against the rows when given.
        source_log: The `.eval` log these rows were bridged from, when there is one. Its
            BASENAME and BYTE SIZE go into the file's schema metadata, which is what lets
            `load_all` recognise "this Parquet is that log's own bridged copy" by identity
            instead of by mtime — see `_tables_from_eval_logs`. The bridge time goes in
            beside them, so the same reader can still date the cell once that log has been
            pruned. Optional, so every other caller (a native re-score, a test) is
            unchanged and writes no stamp.
        eval_config: The resolved config this cell was measured under. Stamped into the
            same schema metadata, canonicalised by `canonical_eval_config`, so the next
            sweep's `is_cached` can refuse to reuse the cell under a different config —
            the thing the (spec_hash, eval_name) cache key cannot see. Optional for the
            same reason `source_log` is: a caller without one writes no stamp and the
            cell falls back to the legacy warn-once path.
        rejudged: The rows were re-judged after the bridge (the re-score scripts), so
            they are NOT ``source_log``'s rows even though they are stamped with it.
            Recorded beside the stamp so ``MO_EVALS_STORE=log`` keeps the cell on Parquet
            instead of decoding the generation log over the new verdicts.

    Returns:
        The path written.
    """
    if not results:
        raise ValueError("refusing to write an empty result file — nothing was evaluated")

    hashes = {r.spec_hash for r in results}
    names = {r.eval_name for r in results}
    if len(hashes) != 1 or len(names) != 1:
        raise ValueError(
            f"one file is one (spec_hash, eval_name); got hashes={sorted(hashes)} "
            f"names={sorted(names)}"
        )
    got_hash, got_name = hashes.pop(), names.pop()
    # The hash is a path component (results_path): refuse anything that could escape `root`.
    # Real hashes are hex (core.spec.HASH_LENGTH); tests use short word hashes, so the
    # check is "one safe path component", not "hex".
    if not isinstance(got_hash, str) or not _SAFE_HASH.fullmatch(got_hash):
        raise ValueError(f"invalid spec_hash {got_hash!r}: must match [A-Za-z0-9_-]+")
    if spec_hash is not None and spec_hash != got_hash:
        raise ValueError(f"spec_hash mismatch: expected {spec_hash}, rows say {got_hash}")
    if eval_name is not None and eval_name != got_name:
        raise ValueError(f"eval_name mismatch: expected {eval_name}, rows say {got_name}")
    if spec is not None and spec.spec_hash != got_hash:
        raise ValueError(f"spec {spec.spec_hash} does not match rows ({got_hash})")

    ids = [r.sample_id for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate sample_id in results — sample ids must be unique per file")

    path = results_path(spec if spec is not None else got_hash, got_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, str] = {}
    if eval_config is not None:
        metadata[_CONFIG_META_KEY] = canonical_eval_config(eval_config)
    elif path.exists():
        # A RE-SCORE overwrites a cell in place (`scripts/rescore_judge_failures.py`,
        # `reparse_judged_rows.py`) and has no eval config in
        # hand — it re-reads rows from disk. Carry the stamp that is already there, or
        # re-scoring a cell would quietly demote it to the unstamped legacy path and turn
        # the guard off for it. The rows change; the config they were GENERATED under
        # does not.
        carried = _recorded_eval_config(path)
        if carried is not None:
            metadata[_CONFIG_META_KEY] = carried
    if source_log is not None:
        source = Path(source_log)
        try:
            metadata.update(
                {
                    "source_log_name": source.name,
                    "source_log_size": str(source.stat().st_size),
                    # Epoch seconds, UTC, taken here rather than from the log: it is the
                    # moment the bridge ran, so anything a log records as later than this
                    # was written after these rows and anything earlier was not. That is
                    # the comparison `_tables_from_eval_logs` needs when the stamped log
                    # itself has been pruned away — see the branch there.
                    "source_bridge_time": str(datetime.now(UTC).timestamp()),
                    **({"source_rows_rejudged": "1"} if rejudged else {}),
                }
            )
            # Which generation these rows came from, so a copy or a rewrite of the source
            # is told apart from a later run (`_tables_from_eval_logs`). Header only.
            [header] = read_log_headers([source], on_error=lambda *_: None) or [None]
            if header is not None and header.eval_id:
                metadata["source_log_eval_id"] = header.eval_id
        except OSError as exc:  # log already gone: no stamp, and the mtime rule still applies
            log.warning("no source-log stamp for %s: %s", path, exc)
    schema = SCHEMA.with_metadata(metadata) if metadata else SCHEMA
    table = pa.Table.from_pylist([r.to_row() for r in results], schema=schema)

    with atomic_write(path) as tmp:
        pq.write_table(table, tmp, compression="zstd")
    log.info("wrote %d rows to %s", len(results), path)
    return path


def read_results(path: Path | str) -> list[SampleResult]:
    """Read one result file back into `SampleResult`s.

    Always the Parquet itself, whatever ``MO_EVALS_STORE`` says: the re-judge writers read
    a cell through this and write it back, and they move to the log only in stage 3.
    """
    table = _read_parquet(path)
    return [SampleResult.from_row(row) for row in table.to_pylist()]


def load_all(
    root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    include_logs: bool = True,
    columns: Sequence[str] | None = None,
    store: str | None = None,
) -> pa.Table:
    """Every result under `root` as one table, for `analysis/`.

    Returns an empty table with the right schema when nothing has been run yet, so the
    scorecard notebook does not need a special case.

    Recursive, so one root may hold both layouts during a migration.

    ``include_logs`` (default on) reads the ``.eval`` logs under ``<root>/logs/`` as
    the PRIMARY store — the Inspect migration's Phase 5, flipped 2026-08-29, and this
    is the single place the preference lives. A cell that exists only as a log — a run
    killed between eval_set and the bridge — is recovered.

    Authority for a cell that has BOTH is by SOURCE IDENTITY, not by freshness: the
    bridge stamps the source log's basename and byte size into the Parquet's schema
    metadata (``write_results(source_log=...)``), so a Parquet whose stamp names the
    candidate log, at its current size, IS that log's own bridged copy and is read
    instead — the faster columnar read, and the only one carrying the runner-stamped
    serving meta (``runner`` / ``max_workers`` / ``gen_threads``) a log cannot. A log the
    stamp does not name, or names at a different size (an ``inspect score`` re-judge
    landing in the log after the bridge wrote), wins and is decoded.

    Identity rather than mtime because mtimes do not survive transport: ``hf download``,
    ``rsync`` and ``cp`` rewrite them in arrival order, so on a downloaded tree an
    arbitrary subset of cells flipped to log-sourced rows and silently lost those serving
    stamps — which CLAUDE.md tells a reader to check width parity from before
    comparing two trees. Nor do mtimes settle it once the stamped log has been PRUNED and
    an unrelated older one survives, so the stamp also records when the bridge ran and a
    survivor is dated against that by the time inside the log, not by the filesystem.
    Parquet written before the stamp existed carries none, and falls back to the old rule:
    a log newer by mtime than the cell's Parquet wins.

    ``columns`` (default None, unchanged behaviour) projects at the Parquet read instead
    of after it. It matters because `raw_output` is a whole transcript per row: a caller
    that only sums latencies out of `meta` otherwise decodes an entire tree's transcripts
    to throw them away. The returned table has exactly these columns, in this order.

    ``store`` (default: ``MO_EVALS_STORE``, else ``both``; see `results_store`, which
    validates it even when unused) picks the store. ``both`` is everything above.
    ``parquet`` is ``include_logs=False``, and ``include_logs=False`` means ``parquet``
    whatever the store. ``log`` makes the same per-cell choice as ``both`` with one
    difference: a cell whose Parquet is its source log's own bridged copy is read from that
    log too, so every cell with a usable log is read from a log, as the writer bridged it
    (`log_rows_table`, through the disposable row cache). A cell whose Parquet ``both``
    keeps against its logs (legacy and pruned-source cases above) or that has no log is
    read from Parquet, with one warning per call naming the count.
    """
    mode = results_store(store)
    if not include_logs:
        mode = "parquet"
    include_logs = mode != "parquet"
    # The log recovery below needs the key columns to know which cells Parquet already
    # covers, so a projection that drops them is widened for the read and narrowed again
    # at the end.
    read_cols = list(columns) if columns is not None else None
    if read_cols is not None and include_logs:
        read_cols = list(dict.fromkeys([*read_cols, "spec_hash", "eval_name"]))

    file_tables = [(f, _read_parquet(f, columns=read_cols)) for f in iter_result_files(root)]
    tables = [t for _, t in file_tables]
    if include_logs:
        superseded, log_tables = _tables_from_eval_logs(
            root, file_tables, columns=read_cols, as_written=mode == "log"
        )
        if superseded:
            tables = [
                t
                for _, t in file_tables
                if not (
                    t.num_rows
                    and (t["spec_hash"][0].as_py(), t["eval_name"][0].as_py()) in superseded
                )
            ]
        kept = [
            f
            for f, t in file_tables
            if mode == "log"
            and t.num_rows
            and (t["spec_hash"][0].as_py(), t["eval_name"][0].as_py()) not in superseded
        ]
        if kept:
            log.warning(
                "MO_EVALS_STORE=log: %d result file(s) under %s were read from Parquet (no "
                "log, or none `both` lets supersede it), e.g. %s",
                len(kept),
                root,
                ", ".join(str(p) for p in kept[:3]),
            )
        tables += [t for _, t in log_tables]
    if not tables:
        log.warning("no result files under %s", root)
        table = SCHEMA.empty_table()
    else:
        table = pa.concat_tables(tables)
    return table if columns is None else table.select(list(columns))


# ----------------------------------------------------------------- .eval logs
#
# The Inspect runners write one `.eval` log per (organism, eval) cell under
# `<root>/logs/<backend_group>/`, stamping eval_name / axis / spec_hash into the log's
# metadata so a log is self-describing — core cannot consult the eval registry (import
# rule: core imports no first-party package). Parquet rows are bridged from these logs
# by the serving layer; the readers below recover the same rows from a log alone.

LOGS_DIRNAME = "logs"

# eval_name -> log->rows bridge, for the evals whose samples are NOT 1:1 with rows.
#
# Added to `core/` on 2026-08-26 (a deliberate contract change, CONTRIBUTING.md, Contracts). It is
# the only legal home: `scripts/check_import_boundaries.py` forbids `core/` from importing
# `evals/`, so the lookup table has to live on the side that gets imported, and the reader below
# is in `core/`.
#
# Dependency-inversion seam, the same shape as `evals._inspect.set_resolver`: core may
# not import `evals/`, but `evals/` may import core, so the owning module registers
# itself at import time and core just looks the name up. Without this the reader below
# is a THIRD implementation of "log -> rows" that silently disagrees with the two in
# `evals/` on exactly the cells that need a bridge — measured on a real tree, 2026-08-26:
# `mu_decisiveness` (chunked Task, ~25 units per sample, so the log's sample ids are
# chunk ids and the Parquet's are per-unit) and `introspection_denial` (scores the
# yes/no reading, deliberately leaving the raw judge verdict in the log).
#
# Empty is the honest default: with nothing registered the reader keeps its plain
# one-row-per-sample behaviour, which is correct for every eval that has no bridge.
#
# The key None holds the WRITER's own one-row-per-sample reader
# (`evals._inspect.rows_from_log`, registered when that module is imported). The
# ``MO_EVALS_STORE=log`` reading (`rows_from_eval_log(as_written=True)`) goes through it
# for every unbridged eval, so the log store serves exactly the rows the Parquet writer
# produced (epoch row ids, thinking meta, thinned-score recovery) rather than the plain
# copy below, which `both` keeps byte for byte until stage 4 removes it.
_LOG_BRIDGES: dict[str | None, Callable[..., list[SampleResult]]] = {}


# Types whose `repr` is stable across imports, so it can carry identity. Anything else
# (an object with the default `repr`) carries a memory address and would report a fresh
# conflict on every reload.
_STABLE_REPR = (str, bytes, int, float, type(None))


def _bridge_identity(fn: Callable[..., list[SampleResult]]) -> tuple[Any, ...]:
    """Which function this is, by NAME and closed-over values rather than by object.

    Re-importing a module rebinds its functions to fresh objects, so `is` reports a
    conflict where there is none — any reload of an `evals/_task_*.py` module trips it.
    What the guard is actually for is two DIFFERENT evals' bridges colliding on one
    name.

    The qualified name alone cannot see that difference where it matters most: the
    bridges come from a FACTORY (`evals._inspect.rows_from_packed(key)`), so
    `mu_decisiveness`'s bridge and the drift cells' bridge share one `__qualname__` and
    differ only in the key they closed over. Closed-over values are therefore part of the
    identity — for the types whose `repr` is stable; anything else contributes `"?"` and
    degrades to the name-only comparison rather than crying conflict at every reload.
    """
    closed: list[str] = []
    for cell in getattr(fn, "__closure__", None) or ():
        try:
            value = cell.cell_contents
        except ValueError:  # an unfilled cell (a closure still being defined)
            value = None
        closed.append(repr(value) if isinstance(value, _STABLE_REPR) else "?")
    return (getattr(fn, "__module__", ""), getattr(fn, "__qualname__", repr(fn)), tuple(closed))


def register_log_bridge(eval_name: str | None, bridge: Callable[..., list[SampleResult]]) -> None:
    """Register the log->rows bridge for one eval, so `rows_from_eval_log` matches Parquet.

    Called by `evals._inspect.register_task` for every eval that passes a custom
    `bridge=`, and once with ``eval_name=None`` for the writer's default reader.
    Idempotent re-registration of the SAME function is allowed (module reimport under a
    different name); a genuinely different function for a name already registered is a bug
    and raises.
    """
    existing = _LOG_BRIDGES.get(eval_name)
    if existing is not None and _bridge_identity(existing) != _bridge_identity(bridge):
        raise ValueError(f"a different log bridge is already registered for {eval_name!r}")
    _LOG_BRIDGES[eval_name] = bridge


def log_bridge_for(eval_name: str) -> Callable[..., list[SampleResult]] | None:
    """The registered bridge for `eval_name`, or None for the plain per-sample reading."""
    return _LOG_BRIDGES.get(eval_name)


#: Manifests Inspect writes INTO a log dir. Both end in `.json` and neither is a log:
#: `eval-set.json` is eval_set's task->log pairing manifest (inspect `_eval/evalset.py`),
#: `logs.json` the header index `write_log_dir_manifest` / `inspect view bundle` writes.
#: Both observed in a real eval_set log dir, 2026-08-27.
_NOT_LOGS = frozenset({"eval-set.json", "logs.json"})

#: Scanner findings (e.g. from `scout scan --scans <log_dir>/scans`, run by hand; nothing in
#: this repo depends on it) live in `<log_dir>/scans/scan_id=.../`, which holds `_scan.json`
#: and `_summary.json` alongside the per-scanner Parquet. They are results ABOUT logs, not
#: logs, and the whole subtree is skipped.
_SCANS_DIRNAME = "scans"


def iter_eval_logs(root: Path | str = DEFAULT_RESULTS_ROOT) -> list[Path]:
    """Every Inspect log under ``<root>/logs/``, in stable order.

    Both on-disk formats, because Inspect supports both and a run made with
    ``--log-format json`` writes `.json`: globbing `.eval` alone made such a tree
    invisible to `load_all`'s log-recovery path and to the log/Parquet gate, which is
    silent data loss rather than an error.

    Reading `.json` means the two Inspect manifests and any scanner output in the same
    tree have to be excluded by name and by directory — everything else that reaches
    `read_eval_log` here is meant to be a log.
    """
    logs_root = Path(root) / LOGS_DIRNAME
    if not logs_root.is_dir():
        return []
    return sorted(
        p
        for p in (*logs_root.rglob("*.eval"), *logs_root.rglob("*.json"))
        if p.name not in _NOT_LOGS and _SCANS_DIRNAME not in p.relative_to(logs_root).parts[:-1]
    )


def rows_from_eval_log(path: Path | str, *, as_written: bool = False) -> list[SampleResult]:
    """Pure reader: one self-describing ``.eval`` log -> rows, sorted by ``sample_id``.

    Mirrors what the serving layer's bridge writes to Parquet, with two documented
    differences: row order is sample_id-sorted (dataset order needs the task object,
    which a log does not carry; analysis aggregates are order-insensitive), and the
    ``meta`` holds only the sample and score metadata — not the runner-stamped serving
    fields (``runner`` / ``max_workers`` / ``gen_threads``) a bridged row records.

    Evals whose samples are not 1:1 with rows delegate to the bridge they registered via
    ``register_log_bridge`` — the SAME function the serving layer bridged Parquet with,
    so the two stores agree by construction rather than by two implementations happening
    to match. ``order`` is not passed by default: a log does not carry the task, so the
    sample_id sort above is the documented ordering.

    ``as_written=True`` (the ``MO_EVALS_STORE=log`` read) returns exactly the rows the
    Parquet writer bridged from this log, bar its runner stamps (`serving.runner._stamp`):
    every log goes through the writer's own bridge, the eval's registered one or else the
    default `evals._inspect.rows_from_log` (registered under None; this raises if it is
    not), never the plain copy below, with ``order`` from the sample ids the log header
    records (``eval.dataset.sample_ids``), so the rows come back in dataset order as the
    writer wrote them. Row order is not cosmetic: the aggregates sum in row order, and a
    sample_id-sorted cell moved a dashboard standard error in its last bit (1.7e-18)
    against the Parquet. A log that records no ids keeps the log's own order.

    Raises:
        ValueError: the log is not one of ours (no eval_name/axis/spec_hash in its
            metadata) or its status is not success.
        RuntimeError: ``inspect-ai`` is not installed.
    """
    try:
        from inspect_ai.log import read_eval_log
    except ImportError as exc:  # core itself must import without inspect-ai
        raise RuntimeError("reading .eval logs requires the inspect-ai package") from exc

    log_ = read_eval_log(str(path))
    meta = log_.eval.metadata or {}
    missing = [k for k in ("eval_name", "axis", "spec_hash") if not meta.get(k)]
    if missing:
        raise ValueError(f"{path} is not a self-describing eval log (missing {missing})")
    if log_.status != "success":
        raise ValueError(f"{path} has status {log_.status!r}, not success")

    bridge = log_bridge_for(str(meta["eval_name"]))
    if bridge is None and meta.get("bridged"):
        # The log says its samples are not 1:1 with rows, but nobody registered the
        # bridge — the caller has not imported the eval package. Reading per-sample here
        # would return confidently wrong rows (chunk ids instead of unit ids, or the
        # judge verdict instead of the promoted one), which is how this went unnoticed
        # for 2 of 22 cells on a real tree. Refuse instead.
        raise ValueError(
            f"{path} is a bridged eval ({meta['eval_name']}) but no log bridge is "
            "registered — import the eval package (e.g. `import serving.inspect_runner`, "
            "which imports every evals._task_* module) before reading its logs"
        )
    kw = {"eval_name": str(meta["eval_name"]), "axis": str(meta["axis"])}
    kw["spec_hash"] = str(meta["spec_hash"])
    if as_written:
        writer = bridge or _require_writer()
        ids = log_.eval.dataset.sample_ids
        return writer(log_, **kw, order={str(i): n for n, i in enumerate(ids)} if ids else None)
    if bridge is not None:
        bridged = bridge(log_, **kw)
        bridged.sort(key=lambda r: r.sample_id)
        return bridged

    rows: list[SampleResult] = []
    for sample in log_.samples or []:
        score = next(iter((sample.scores or {}).values()), None)
        value = score.value if score is not None else math.nan
        row_meta: dict[str, Any] = dict(sample.metadata or {})
        if score is not None and score.metadata:
            row_meta.update(score.metadata)
        rows.append(
            SampleResult(
                spec_hash=str(meta["spec_hash"]),
                eval_name=str(meta["eval_name"]),
                axis=str(meta["axis"]),
                sample_id=str(sample.id),
                score=float(value) if isinstance(value, (int, float)) else math.nan,
                raw_output=sample.output.completion if sample.output else "",
                meta=row_meta,
            )
        )
    rows.sort(key=lambda r: r.sample_id)
    return rows


class _SourceStamp(NamedTuple):
    """What `write_results` recorded about the log a Parquet was bridged from."""

    log_name: str
    log_size: int
    bridge_time: float | None
    """Epoch seconds at which the bridge wrote this Parquet.

    None on a Parquet written before this key existed (2026-08-31), which puts its cell
    back on today's behaviour in ``_tables_from_eval_logs`` rather than on a guess.
    """
    rejudged: bool = False
    """``write_results(rejudged=True)``: rows re-judged after the bridge, so not the log's."""
    eval_id: str | None = None
    """The source log's ``eval.eval_id`` (None before this key existed)."""


def _source_log_stamp(schema: pa.Schema) -> _SourceStamp | None:
    """The source-log stamp of this Parquet, or None.

    None means the file predates the stamp (or was written by a caller that had no log),
    which is what puts its cell back on the legacy mtime rule in ``_tables_from_eval_logs``.
    """
    metadata = schema.metadata or {}
    name, size = metadata.get(b"source_log_name"), metadata.get(b"source_log_size")
    if name is None or size is None:
        return None
    bridged = metadata.get(b"source_bridge_time")
    rejudged = metadata.get(b"source_rows_rejudged") == b"1"
    # Optional and added after main: main never reads it, so a value we cannot decode (or an
    # empty one) is "unknown", exactly as on a stamp written before the key existed, and
    # never voids the stamp main's own fields (name, size, bridge time) make.
    eval_id: str | None = None
    raw_id = metadata.get(b"source_log_eval_id")
    if raw_id:
        try:
            eval_id = raw_id.decode()
        except UnicodeDecodeError:
            eval_id = None
    try:
        return _SourceStamp(
            name.decode(),
            int(size),
            float(bridged) if bridged else None,
            rejudged,
            eval_id,
        )
    except (UnicodeDecodeError, ValueError):  # a stamp we cannot read is no stamp
        return None


def _is_stamped_source(
    stamp: _SourceStamp,
    generation: str | None,
    header: LogHeader,
    name: str,
    size: int,
    *,
    by_eval_id: bool = True,
) -> bool:
    """Is this candidate the very log the Parquet was bridged from?

    Name and size are only the fallback identity: a KNOWN eval id that differs from the
    stamped generation is a different run that happens to share a basename and byte size
    (two directories, equal-length ids), so it is never the source. ``by_eval_id=False`` is
    main's plain name-and-size rule, which the UNMARKED selection keeps byte for byte.
    """
    if by_eval_id and generation and header.eval_id and header.eval_id != generation:
        return False
    return (name, size) == (stamp.log_name, stamp.log_size)


def _rejudge_order(
    stamp: _SourceStamp, generation: str | None, header: LogHeader
) -> Literal["later", "older", "unknown"]:
    """Where a log stands against a MARKED re-judge of another generation's rows.

    ``later`` is proof: a known eval id different from the stamped generation AND a real
    completion time (``stats.completed_at``) in a STRICTLY later second than the bridge's.
    Times are recorded to the second, so a log that finished in the bridge's own second
    cannot be ordered against it: ``unknown``, like a missing id or time. ``older`` is the
    converse proof (different id, an earlier second). A same-generation log is not asked.
    """
    done, bridged = header.completed_at_ts, stamp.bridge_time
    if not generation or not header.eval_id or done is None or bridged is None:
        return "unknown"
    if not (math.isfinite(done) and math.isfinite(bridged)):
        return "unknown"
    if header.eval_id == generation:
        return "unknown"
    if math.floor(done) > math.floor(bridged):
        return "later"
    return "older" if math.floor(done) < math.floor(bridged) else "unknown"


def _tables_from_eval_logs(
    root: Path | str,
    parquet_files: list[tuple[Path, pa.Table]],
    columns: Sequence[str] | None = None,
    *,
    as_written: bool = False,
    verify: bool = True,
) -> tuple[set[tuple[str, str]], list[tuple[tuple[str, str], pa.Table]]]:
    """Log-sourced (cell, table)s plus the Parquet cells they supersede — see ``load_all``.

    ``as_written`` is the ``MO_EVALS_STORE=log`` reading: the same choice, except that a
    Parquet's own source log is read in its place, and every log is read as the writer
    bridged it (`log_rows_table`) instead of by `rows_from_eval_log`'s plain reading.
    ``verify=False`` skips `_log_matches` (the exit check compares every such cell).

    Header-only reads decide relevance cheaply; the full log is decoded only when it wins
    a cell. Where several successful logs name one cell (eval_set retries, an ``inspect
    score`` writing a fresh file), they are tried newest-mtime-first, path as the
    deterministic tiebreak, and the first one that DECODES supplies the cell: a newest log
    that is corrupt used to take the cell down with it, dropping rows that the previous
    attempt's log still holds.

    A candidate the cell's Parquet names in its ``source_log`` stamp, at that log's
    current size, is that Parquet's own source — the Parquet is kept and no log is
    decoded. If the stamped log is not here at all (pruned), a survivor only takes the
    cell if its own recorded completion time is later than the stamped bridge time.
    Unstamped Parquet (written before the stamp existed) falls back to the mtime rule: a
    log newer than it wins. See ``load_all`` for why identity beats mtime.

    ``columns`` narrows the returned tables to match a projected Parquet read — a log has
    to be decoded whole either way, so this is only about the schemas lining up for the
    concat in ``load_all``. It must include ``spec_hash`` and ``eval_name``, which
    ``load_all`` guarantees.
    """
    paths = iter_eval_logs(root)
    if not paths:
        return set(), []
    try:
        import inspect_ai.log  # noqa: F401  # only to fail soft, as before, when it is absent
    except ImportError:
        log.warning("%d .eval log(s) under %s ignored: inspect-ai not installed", len(paths), root)
        return set(), []

    parquet_mtime: dict[tuple[str, str], float] = {}
    parquet_stamp: dict[tuple[str, str], _SourceStamp | None] = {}
    parquet_path: dict[tuple[str, str], Path] = {}
    for f, t in parquet_files:
        if t.num_rows:
            key = (t["spec_hash"][0].as_py(), t["eval_name"][0].as_py())
            mtime = f.stat().st_mtime
            if mtime >= parquet_mtime.get(key, float("-inf")):
                # The stamp of the NEWEST file for the cell, so it agrees with the mtime
                # the legacy branch compares against.
                parquet_stamp[key] = _source_log_stamp(t.schema)
                parquet_path[key] = f
            parquet_mtime[key] = max(parquet_mtime.get(key, 0.0), mtime)

    if as_written:  # outside the per-log `except` below: these are bugs, not bad logs
        _require_writer()
        _bridge_version()
    candidates: dict[
        tuple[str, str], list[tuple[float, str, Path, int, float | None, LogHeader]]
    ] = {}
    # Header-only, and each header read once per log version (`eval_log_headers`'s stat
    # manifest): an unreadable or foreign file is logged and left out there, as it was here.
    # A non-str or empty spec_hash / eval_name is None on the header, so it is skipped
    # exactly as the old per-header `isinstance` checks skipped it.
    for header in eval_log_headers(root):
        spec_hash, eval_name = header.spec_hash, header.eval_name
        if spec_hash is None or eval_name is None or header.status != "success":
            continue
        key = (spec_hash, eval_name)
        candidates.setdefault(key, []).append(
            (header.mtime, str(header.path), header.path, header.size, header.completed_ts, header)
        )

    superseded: set[tuple[str, str]] = set()
    out: list[tuple[tuple[str, str], pa.Table]] = []
    for key, found in sorted(candidates.items()):
        had_parquet = key in parquet_mtime
        stamp = parquet_stamp.get(key)
        # Is the log the Parquet was bridged from still here, at ANY size? If it is, every
        # other log for this cell is something written afterwards — a rewrite, a second
        # attempt — and the identity rule below settles the cell on its own. If it is not,
        # the log tree was pruned (they run to GBs; one full-config run measured
        # 5.9 GB) and a survivor is about to be promoted in its place.
        source_present = stamp is not None and any(c[2].name == stamp.log_name for c in found)
        # The generation the Parquet's rows came from: the stamped eval id, else (a stamp
        # older than that key) the eval id of the stamped source if it is still here. A log
        # of the same generation is the source copied, reformatted or re-scored; any other
        # is a later run (or an eval_set retry), which wins exactly as before.
        generation = (
            None
            if stamp is None
            else stamp.eval_id
            or next((c[5].eval_id for c in found if c[2].name == stamp.log_name), None)
        )
        # Newest first by mtime, as always; but under a marked re-judge, the logs that are
        # PROVEN later generations come first, ordered by their own completion time, so a
        # restored copy of the source (fresh mtime) cannot hide a newer run behind it.
        by_mtime = sorted(found, key=lambda c: c[:2], reverse=True)
        verdict = {
            c[1]: _rejudge_order(stamp, generation, c[5])
            for c in found
            if stamp is not None and stamp.rejudged
        }
        # Winners among the proven ones by completion time, then eval id, then path: never mtime.
        proven = sorted(
            (c for c in by_mtime if verdict.get(c[1]) == "later"),
            key=lambda c: (c[5].completed_at_ts or 0.0, c[5].eval_id, c[1]),
            reverse=True,
        )
        ordered = proven + [c for c in by_mtime if c not in proven]
        # Logs a marked re-judge cannot be ordered against. Classified up front over EVERY
        # candidate, so the loop's early exit (the stamped source sorting first) cannot hide
        # an ambiguous newer generation from the one warning below.
        unordered: list[str] = [
            c[2].name
            for c in found
            if stamp is not None
            and stamp.rejudged
            and verdict[c[1]] == "unknown"
            and not (c[5].eval_id == generation if generation else c[2].name == stamp.log_name)
            and not _is_stamped_source(stamp, generation, c[5], c[2].name, c[3])
        ]
        for cand in ordered:
            mtime, _, path, size, log_time, header = cand
            own_source = False
            is_later = cand in proven
            # Does the log ITSELF say it was written after the bridge? None = it cannot be
            # dated at all (a Parquet older than the bridge-time key, or a log recording no
            # time), which is the case the mtime fallback below is still there for.
            after_bridge = (
                None
                if stamp is None or stamp.bridge_time is None or log_time is None
                else log_time > stamp.bridge_time
            )
            same_generation = stamp is not None and (
                header.eval_id == generation if generation else path.name == stamp.log_name
            )
            if had_parquet:
                if stamp is None:
                    keep_parquet = mtime <= parquet_mtime[key]  # legacy: the old rule
                elif is_later:
                    keep_parquet = False  # proven later generation: it wins, in every store
                elif (
                    stamp.rejudged
                    and not same_generation
                    and not _is_stamped_source(stamp, generation, header, path.name, size)
                ):
                    # Not the source, not a copy of it, and not PROVEN later: older, or
                    # identity / time unknown. Never drop re-judged verdicts on a guess;
                    # the unprovable ones are named in ONE warning per cell, below.
                    continue
                elif _is_stamped_source(
                    stamp, generation, header, path.name, size, by_eval_id=stamp.rejudged
                ):
                    # identity: the Parquet IS this log's bridged copy. `log` reads the log
                    # instead, unless the rows were re-judged since (`_log_matches`).
                    own_source = as_written and (
                        not verify or _log_matches(parquet_path[key], stamp, header)
                    )
                    keep_parquet = not own_source
                elif same_generation and stamp.rejudged:
                    # Re-judged after the bridge (`write_results(rejudged=True)`): no copy
                    # of the generation log holds these rows, so none supersedes them, in
                    # any store. A LATER generation still does (below). Only files written
                    # since the mark existed carry it, so `both` changes only where it used
                    # to serve a re-judge its own pre-re-judge rows.
                    continue
                elif same_generation and as_written:
                    # `log`: a copy or rewrite of the source is read in its place only if
                    # its rows are the Parquet's (`_log_matches`, which warns otherwise: an
                    # unmarked re-judge). `both` keeps the time rule below, as before.
                    own_source = not verify or _log_matches(parquet_path[key], stamp, header)
                    if not own_source:
                        continue
                    keep_parquet = False
                else:
                    # A later generation (any store), or under `both` any other log — the
                    # rule as it always was. If the stamped source is gone, mtime cannot settle
                    # which came first — rewriting it in arrival order is exactly what
                    # `hf download` / rsync / cp do, and why the stamp exists — so compare
                    # two times recorded INSIDE the files instead: the log's own completion
                    # time against the time the bridge ran. A log that finished before the
                    # bridge cannot be a re-judge of these rows, so the Parquet keeps the
                    # cell, and with it the serving stamps only a bridged row carries.
                    # Either time missing (a Parquet older than this key, a log that
                    # records none) falls back to today's behaviour: the log wins, loudly.
                    # Inspect records these to the second, so equality counts as pre-bridge
                    # (`after_bridge` False): a survivor from the bridge's own second is
                    # treated as older, and pruning happens hours later, so that rounding
                    # is the safe one. Known gap: `inspect score` copies the original
                    # `stats` forward, so a re-judge is dated by its GENERATION, and one
                    # whose source log was ALSO deleted is passed over here without a word.
                    # Narrow — you cannot re-score a log you deleted, so it takes a
                    # `-scored` copy and a later prune — and it costs a rejudge, not rows.
                    keep_parquet = not source_present and after_bridge is False
                    if keep_parquet:
                        # The one breadcrumb for the known gap above. This outcome is the
                        # right one for a pruned tree, so it is not a warning — but the
                        # `inspect score`-then-prune case lands here too and looks, from
                        # the outside, like a re-judge that silently did nothing. Whoever
                        # goes looking for that needs a line naming the log that was passed
                        # over and why; re-bridge the cell (`--force`) to take it.
                        log.info(
                            "kept the Parquet for %s x %s: its stamped source log (%s) is "
                            "gone and the survivor %s finished before the bridge, so it is "
                            "not a re-judge. If it IS one, re-bridge the cell",
                            *key,
                            stamp.log_name,
                            path.name,
                        )
                if keep_parquet:
                    break  # the Parquet is this log's own bridged copy — read the cache
            rows: list[SampleResult] = []
            table: pa.Table | None = None
            try:
                if as_written:
                    table = log_rows_table(header, columns)
                else:
                    rows = rows_from_eval_log(path)
            # Broad: a corrupt body raises whatever the zip/JSON reader raises, and the
            # point of the loop is that the cell survives on an older log rather than
            # vanishing. Every skip is named.
            except Exception as exc:
                log.warning("skipping eval log %s: %s", path, exc)
                continue
            if table is None:
                if not rows:
                    continue
                table = pa.Table.from_pylist([r.to_row() for r in rows], schema=SCHEMA)
                table = table if columns is None else table.select(list(columns))
            elif not table.num_rows:
                continue
            if own_source:
                superseded.add(key)  # the log store's own reading; nothing to report
            elif had_parquet:
                superseded.add(key)
                if stamp is not None and mtime <= parquet_mtime[key] and not after_bridge:
                    # A log that supersedes a stamped Parquet is normally a re-judge, and a
                    # re-judge is written AFTER the bridge. This one is not, by mtime, and
                    # the recorded times did not vouch for it either — `not after_bridge`
                    # covers both None (a Parquet older than the bridge-time key, or a log
                    # that records no time) and False (the stamped source is still here and
                    # this is a rewrite of it that arrived out of order). WARNING, not info,
                    # because the undatable case can revert a cell to stale rows without
                    # saying so — and because `preserve_logging` runs the sweep at warning
                    # level, where an info line is invisible.
                    log.warning(
                        "log supersedes its Parquet for %s x %s, but that log is OLDER than "
                        "the Parquet (%s) and the recorded times do not settle it. Reading "
                        "%d rows from it; check this cell before quoting it",
                        *key,
                        path,
                        table.num_rows,
                    )
                else:
                    log.info(
                        "log supersedes its Parquet for %s x %s — reading %d rows from %s "
                        "(a re-judge or an unbridged rerun)",
                        *key,
                        table.num_rows,
                        path,
                    )
            else:
                log.info(
                    "recovered %d rows for %s x %s from %s (no Parquet)",
                    table.num_rows,
                    *key,
                    path,
                )
            out.append((key, table))
            break
        if unordered and not (out and out[-1][0] == key):
            log.warning(
                "kept the re-judged Parquet for %s x %s: cannot order %d log(s) against it "
                "(eval id, completion time, or the second of the bridge unknown or shared): "
                "%s. Re-bridge the cell (`--force`) if one of them is a newer run",
                *key,
                len(unordered),
                ", ".join(sorted(unordered)),
            )
    return superseded, out


def iter_result_files(root: Path | str = DEFAULT_RESULTS_ROOT) -> Iterable[Path]:
    """Every result file under `root`, in stable order.

    Recursive, so it spans both the nested layout and any legacy flat directory —
    but never inside `logs/`: that subtree holds `.eval` logs and scanner findings,
    and a foreign Parquet there is not a `SampleResult` file (reading one raises
    ``KeyError: spec_hash`` out of every consumer at once).
    """
    rootp = Path(root)
    return sorted(
        p for p in rootp.rglob("*.parquet") if LOGS_DIRNAME not in p.relative_to(rootp).parts
    )


# ------------------------------------------------- the one read path for scripts
#
# One read path (docs/COMMANDS.md, "Results store and the log-row cache").
# Scripts and `analysis/` used to glob
# `seed*__<hash>__<eval>.parquet` and call `pq.read_table` / `pd.read_parquet` themselves,
# about a dozen copies of the result-file name and format. They now ask this module, so
# when the store changes (stage 4) the change is here and not in every caller. Behaviour
# is unchanged: these are the same globs and the same Parquet reads, moved.


class ResultFileName(NamedTuple):
    """What a result file's NAME says about its cell (see the LAYOUT docstring above)."""

    seed: int | None
    """None for the legacy flat layout (``<spec_hash>/<eval_name>.parquet``)."""
    spec_hash: str | None
    """None for the legacy flat layout, whose hash is the directory name."""
    eval_name: str


_RESULT_SUFFIX = ".parquet"
#: ``-?``: `OrganismSpec` accepts a negative seed and `result_filename` writes it as-is
#: (``seed-1__<hash>__<eval>.parquet``).
_SEEDED_NAME = re.compile(r"seed(-?\d+)__(.+?)__(.+)")


def parse_result_filename(path: Path | str) -> ResultFileName | None:
    """The cell a result file's name describes, or None if it is not a result file.

    ``seed<n>__<spec_hash>__<eval_name>.parquet`` (the layout every write uses) or the
    legacy ``<eval_name>.parquet``. The hash is everything up to the next ``__`` (hashes
    never contain one), so an eval name may itself hold ``__``.
    """
    name = Path(path).name
    if not name.endswith(_RESULT_SUFFIX):
        return None
    stem = name.removesuffix(_RESULT_SUFFIX)
    m = _SEEDED_NAME.fullmatch(stem)
    if m:
        return ResultFileName(int(m.group(1)), m.group(2), m.group(3))
    return ResultFileName(None, None, stem)


def result_files(
    root: Path | str = DEFAULT_RESULTS_ROOT,
    *,
    spec_hash: str | None = None,
    eval_name: str | None = None,
    seed: int | None = None,
    depth: int | None = None,
) -> list[Path]:
    """Result files under ``root`` for the cells matching every filter given, in `Path` order.

    `Path` order (by path component, as `iter_result_files` sorts), which is not plain string
    order: ``a/x`` sorts before ``a-b/x`` here and after it as strings.

    ``depth`` None walks the whole tree as `iter_result_files` does (never inside
    ``logs/``); an int looks exactly that many directory levels below ``root`` (0: in
    ``root`` itself, e.g. one organism-seed directory; 2: ``<organism>/<spec_hash>/``).

    A ``spec_hash`` or ``seed`` filter matches only seed-tagged names, because a legacy
    flat name records neither; an ``eval_name`` filter is exact (never a substring:
    ``persistence_multiturn_drift`` is a prefix of ``persistence_multiturn_drift_long``).

    At every depth, nothing under ``logs/`` and nothing HIDDEN below ``root`` (a path
    component starting with ``.``, e.g. ``.quarantined/``) is returned: the scripts that
    moved onto this used ``glob.glob``, which skips hidden entries, and a quarantined cell
    must not come back. `iter_result_files` (and so `load_all`) is deliberately left as it
    was, hidden directories included.
    """
    rootp = Path(root)
    if depth is None:
        found: Iterable[Path] = iter_result_files(rootp)
    else:
        found = rootp.glob("/".join(["*"] * depth + [f"*{_RESULT_SUFFIX}"]))
    out = []
    for path in found:
        rel = path.relative_to(rootp).parts
        if LOGS_DIRNAME in rel or any(part.startswith(".") for part in rel):
            continue
        cell = parse_result_filename(path)
        if cell is None:
            continue
        if eval_name is not None and cell.eval_name != eval_name:
            continue
        if spec_hash is not None and cell.spec_hash != spec_hash:
            continue
        if seed is not None and cell.seed != seed:
            continue
        out.append(path)
    return sorted(out)


def read_result_table(path: Path | str, columns: Sequence[str] | None = None) -> pa.Table:
    """One result file's cell as an Arrow table, optionally projected to ``columns``.

    Use ``.to_pandas()`` / ``.to_pylist()`` on the result for a frame or records. Under
    ``MO_EVALS_STORE=log`` the rows come from the file's own source log instead
    (`_source_log_table`), and from the Parquet, with a warning, when that log is not here.
    """
    if results_store() == "log":
        table = _source_log_table(Path(path), columns)
        if table is not None:
            return table
        log.warning("MO_EVALS_STORE=log: %s has no source log here; read its Parquet", path)
    return _read_parquet(path, columns)


def _read_parquet(path: Path | str, columns: Sequence[str] | None = None) -> pa.Table:
    """The single place a result Parquet is opened (`load_all` included).

    Arrow opens the file by path itself, never through a Python file object, which is what
    `scripts/verify_cells.py` once hit as an intermittent SIGABRT at interpreter exit.
    """
    return pq.read_table(str(path), columns=list(columns) if columns is not None else None)


# ------------------------------------------------------- log headers, read once
#
# Discovery over `.eval` logs reads only headers (`read_eval_log(header_only=True)`; the
# pinned inspect-ai 0.3.260 supports it and `evals_df` is the same read plus pandas). A
# header is ~5-10 ms, which is 25-45 s for a full tree on every `load_all` (timing spike,
# PR #478), so a header is read once per log VERSION: a stat manifest keyed by path, size
# and mtime keeps what was read, and only a new or changed log is opened again. The
# manifest is a disposable cache outside the results tree (deleting it loses nothing and
# costs one re-read of every header), never a second store.

_HEADER_CACHE_ENV = "MO_EVALS_LOG_HEADER_CACHE"
#: Bumped whenever `LogHeader` or the stat key changes, so an old manifest is ignored.
_HEADER_CACHE_VERSION = 3


def _stat_key(st: os.stat_result) -> list[int]:
    """What a manifest entry must match to be reused: the stat of the log's TARGET.

    Size and mtime alone are not enough: ``os.utime`` restores an mtime, so a same-size
    replacement (or a symlink retargeted at a same-size file) would be served the old
    header, and a stale ``completed_at`` changes which store wins a cell in `load_all`.
    ``st_ctime_ns`` cannot be set from user space and moves on any rewrite or chmod, and
    ``st_ino``/``st_dev`` change when the path points at a different file. A false miss
    costs one header read; a false hit would be a wrong cell.
    """
    return [st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_ino, st.st_dev]


@dataclass(frozen=True)
class LogHeader:
    """What a log's header says, plus the stat it was read at. No sample is decoded."""

    path: Path
    size: int
    mtime: float
    status: str
    task: str
    model: str
    spec_hash: str | None
    """From ``eval.metadata``; None for a log that is not one of ours."""
    eval_name: str | None
    axis: str | None
    bridged: bool
    task_args_spec_hash: str | None
    """``eval.task_args['spec_hash']``, which some older logs carry instead of metadata."""
    started_at: str
    completed_at: str
    created: str
    total_samples: int | None
    completed_samples: int | None
    eval_id: str = ""
    """``eval.eval_id``: one per generation attempt. A copy, a reformat or an `inspect score`
    of a log keeps it; a rerun or an eval_set retry gets a new one."""

    @staticmethod
    def _epoch(value: str | None) -> float | None:
        if not value:
            return None
        try:
            when = datetime.fromisoformat(value)
        except ValueError:
            return None
        return (when if when.tzinfo else when.replace(tzinfo=UTC)).timestamp()

    @property
    def completed_ts(self) -> float | None:
        """Epoch seconds the log itself says it finished, or None if it does not say.

        From the HEADER (``stats.completed_at`` survives ``header_only=True``), so dating a
        log costs no body decode. Inspect normalises these to timezone-aware UTC ISO
        strings; a naive one is read as UTC rather than as local time, which would put the
        comparison hours out. ``eval.created`` is the fallback for a log whose stats never
        got filled in.
        """
        for value in (self.completed_at, self.created):
            if (when := self._epoch(value)) is not None:
                return when
        return None

    @property
    def completed_at_ts(self) -> float | None:
        """Only ``stats.completed_at``: a real completion, never the ``created`` fallback
        (a log that was created before a re-judge may have finished after it)."""
        return self._epoch(self.completed_at)


def _header_of(path: Path, size: int, mtime: float, header: Any) -> LogHeader:
    ev = header.eval
    meta = ev.metadata or {}
    args = ev.task_args or {}
    stats, results = header.stats, header.results

    def _str_or_none(value: Any) -> str | None:
        return value if isinstance(value, str) and value else None

    def _str(value: Any) -> str:  # a non-str time is "not recorded", as the old reader had it
        return value if isinstance(value, str) else ""

    return LogHeader(
        path=path,
        size=size,
        mtime=mtime,
        status=str(header.status),
        task=str(ev.task or ""),
        model=str(ev.model or ""),
        spec_hash=_str_or_none(meta.get("spec_hash")),
        eval_name=_str_or_none(meta.get("eval_name")),
        axis=_str_or_none(meta.get("axis")),
        bridged=bool(meta.get("bridged")),
        task_args_spec_hash=_str_or_none(args.get("spec_hash")),
        started_at=_str(getattr(stats, "started_at", "")),
        completed_at=_str(getattr(stats, "completed_at", "")),
        created=_str(getattr(ev, "created", "")),
        total_samples=getattr(results, "total_samples", None) if results else None,
        completed_samples=getattr(results, "completed_samples", None) if results else None,
        eval_id=_str(getattr(ev, "eval_id", "")),
    )


def _cache_dir(env: str, default_name: str) -> Path | None:
    """A disposable cache directory: ``$env``, else ``~/.cache/mo_evals/<default_name>``.

    None when ``$env`` switches caching off (``off``, ``0``, ``false``, ``no`` or empty) or
    there is no home directory to put it in; the caller then runs uncached.
    """
    setting = os.environ.get(env)
    if setting is not None and setting.strip().lower() in {"", "0", "off", "false", "no"}:
        return None
    if setting:
        return Path(setting)
    xdg = os.environ.get("XDG_CACHE_HOME")
    try:
        home_cache = Path(xdg) if xdg else Path.home() / ".cache"
    except RuntimeError as exc:  # no $HOME and no passwd entry: run uncached
        log.info("no %s cache (no home directory): %s", default_name, exc)
        return None
    return home_cache / "mo_evals" / default_name


def _header_cache_file(key: str) -> Path | None:
    """Where the manifest for ``key`` lives, or None when caching is switched off."""
    base = _cache_dir(_HEADER_CACHE_ENV, "eval-log-headers")
    return None if base is None else base / f"{hashlib.sha256(key.encode()).hexdigest()[:24]}.json"


def _load_header_cache(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    try:
        if not path.is_file():  # inside the try: an unreadable ancestor raises here
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # unreadable, torn or foreign: a cold cache
        log.info("ignoring unreadable log-header manifest %s: %s", path, exc)
        return {}
    if not isinstance(data, dict) or data.get("version") != _HEADER_CACHE_VERSION:
        return {}
    entries = data.get("logs")
    return entries if isinstance(entries, dict) else {}


def _save_header_cache(path: Path | None, entries: dict[str, dict[str, Any]]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with atomic_write(path) as tmp:
            tmp.write_text(
                json.dumps({"version": _HEADER_CACHE_VERSION, "logs": entries}),
                encoding="utf-8",
            )
    except OSError as exc:  # a read-only home must not take a read down with it
        log.info("could not save log-header manifest %s: %s", path, exc)


def read_log_headers(
    paths: Iterable[Path | str],
    *,
    cache_key: str | None = None,
    on_error: Callable[[Path, Exception], None] | None = None,
) -> list[LogHeader]:
    """The header of every readable log in ``paths``, in the order given.

    ``cache_key`` names the stat manifest (a results root's resolved ``logs/`` path, say):
    a log whose path, size and mtime match the manifest is not opened, and the manifest is
    rewritten only when something changed. None reads every header (no manifest). An
    unreadable file (half-written, foreign) is left out and NOT remembered, so it is
    retried once it has finished writing; ``on_error`` is told about it (default: a
    warning in the log).

    Raises:
        RuntimeError: ``inspect-ai`` is not installed and a header has to be read.
    """

    def report(path: Path, exc: Exception) -> None:
        if on_error is None:
            log.warning("unreadable eval log %s: %s", path, exc)
        else:
            on_error(path, exc)

    cache_file = _header_cache_file(cache_key) if cache_key is not None else None
    cached = _load_header_cache(cache_file)
    fresh: dict[str, dict[str, Any]] = {}
    out: list[LogHeader] = []
    read_eval_log: Callable[..., Any] | None = None
    for raw in paths:
        path = Path(raw)
        try:
            st = path.stat()
        except OSError as exc:
            report(path, exc)
            continue
        key = str(path)
        hit = cached.get(key)
        header: LogHeader | None = None
        if isinstance(hit, dict) and hit.get("_stat") == _stat_key(st):
            fields = {k: v for k, v in hit.items() if k != "_stat"}
            try:
                header = LogHeader(path=path, size=st.st_size, mtime=st.st_mtime, **fields)
            except TypeError:  # an entry this version cannot read is a miss, not a crash
                header = None
        if header is None:
            if read_eval_log is None:
                try:
                    from inspect_ai.log import read_eval_log
                except ImportError as exc:  # core itself must import without inspect-ai
                    raise RuntimeError(
                        "reading .eval logs requires the inspect-ai package"
                    ) from exc
            try:
                header = _header_of(
                    path, st.st_size, st.st_mtime, read_eval_log(str(path), header_only=True)
                )
            except Exception as exc:  # a truncated or foreign file must not sink a read
                report(path, exc)
                continue
        out.append(header)
        record = {k: v for k, v in header.__dict__.items() if k not in ("path", "size", "mtime")}
        fresh[key] = {**record, "_stat": _stat_key(st)}
    if cache_file is not None and fresh != cached:
        _save_header_cache(cache_file, fresh)
    return out


def eval_log_headers(root: Path | str = DEFAULT_RESULTS_ROOT) -> list[LogHeader]:
    """Every readable log header under ``<root>/logs/`` (`iter_eval_logs` order), cached.

    The discovery read for a results root: which cells have a log, its status and sample
    counts, without decoding a sample. Rows still come from `rows_from_eval_log`.
    """
    paths = iter_eval_logs(root)
    if not paths:
        return []
    return read_log_headers(paths, cache_key=str((Path(root) / LOGS_DIRNAME).resolve()))


# ------------------------------------------------------- the store switch (temporary)
#
# `MO_EVALS_STORE` (docs/COMMANDS.md, "Results store and the log-row cache") lets a reader take the
# rows from the `.eval` logs alone, so the two stores can be compared on a real tree before
# the Parquet goes. Read only by `load_all` and `read_result_table`; writers, `is_cached`
# and the re-judge scripts ignore it until stage 3. Removed in stage 4.

_STORE_ENV = "MO_EVALS_STORE"
STORES = ("both", "parquet", "log")


def results_store(store: str | None = None) -> str:
    """The store to read: ``store`` if given, else ``$MO_EVALS_STORE``, else ``both``.

    ``both`` is the reading `load_all` has always done (Parquet, with logs recovering or
    superseding cells), ``parquet`` reads Parquet only, ``log`` reads every cell it can
    from a log, as the writer bridged it, and the rest from Parquet (truth-probe imports,
    derived and legacy trees; see `load_all` and `read_result_table` for which).

    Raises:
        ValueError: anything else, so a typo is not silently read as ``both``.
    """
    value = (store if store is not None else os.environ.get(_STORE_ENV) or "both").strip()
    if value.lower() not in STORES:
        raise ValueError(f"{_STORE_ENV}={value!r}: expected one of {', '.join(STORES)}")
    return value.lower()


# ------------------------------------------------------- bridged rows, cached per log
#
# A full `.eval` decode is CPU-bound and ~350x slower than the Parquet read of the same
# rows (timing spike, PR #478: 95 s against 0.27 s for 57,600 rows), and Inspect's sample
# summaries are lossy, so the `log` store reads each log's bridged rows
# (`rows_from_eval_log(as_written=True)`) through a disposable cache: one uncompressed
# Arrow IPC file per log, outside the results tree, which reads at Parquet speed. It is not
# a store. Deleting it loses nothing; a stale, partial, corrupt or foreign entry is a miss
# and is rebuilt from the log.
#
# Why not Inspect: inspect_ai has no persistent per-log row cache (`read_eval_log` and
# `samples_df` re-decode the zip every call), and `samples_df` is lossy for our bridges.
# The cache is an optional accelerator only: `MO_EVALS_LOG_ROW_CACHE=off` gives the same rows
# straight from `inspect_ai.log.read_eval_log` + the writer's bridge, the source of truth.
#
# An entry is reused only if it was built from this log at its current stat (`_stat_key`,
# the header manifest's key), by this cache format and by the same reading code
# (`_bridge_version`).

_ROW_CACHE_ENV = "MO_EVALS_LOG_ROW_CACHE"
#: Bumped whenever the entry layout or the key changes, so old entries are misses.
_ROW_CACHE_VERSION = 3
_ROW_CACHE_KEY = b"mo_evals_row_cache"
_REPO = Path(__file__).resolve().parents[1]


def _bridge_files() -> list[Path]:
    """Every first-party source file a log->rows reading can execute, for the cache key.

    THE RULE: the packages ``core`` and ``evals`` (every bridge, the writer's reader
    included, is registered from ``evals/``), closed over every first-party package any of
    their files imports, function-local imports included (a static AST scan, so a helper
    imported lazily inside a bridge counts), and every ``.py`` file in each such package.
    First-party means a top-level directory of this checkout. Today that is ``core/``,
    ``evals/`` and ``judges/`` (the import-boundary check keeps ``implant/`` and
    ``serving/`` out). A deliberate over-approximation: any edit there rebuilds every entry,
    because a false miss costs one decode and a false hit serves stale rows. Third-party
    code is keyed only through the inspect-ai version; a bridge is a pure function of the
    log and this code (`evals._inspect.rows_from_log`), so data files are not keyed.
    """
    todo, packages = ["core", "evals"], set()
    while todo:
        package = todo.pop()
        if package in packages or not (_REPO / package).is_dir():
            continue
        packages.add(package)
        for path in _py_files(_REPO / package):
            for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
                if isinstance(node, ast.Import):
                    todo += [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                    todo.append(node.module.split(".")[0])
    return sorted(p for package in packages for p in _py_files(_REPO / package))


def _py_files(top: Path) -> list[Path]:
    """Every ``.py`` under ``top``, through symlinked directories too."""
    return [
        Path(d) / f for d, _, fs in os.walk(top, followlinks=True) for f in fs if f.endswith(".py")
    ]


@functools.cache
def _bridge_version() -> str:
    """A digest of the code that turns a log into rows (`_bridge_files`) and of inspect-ai."""
    from importlib.metadata import version

    digest = hashlib.sha256(f"{_ROW_CACHE_VERSION}|{version('inspect-ai')}".encode())
    for path in _bridge_files():  # by relative path: a checkout's location is no key
        digest.update(str(path.relative_to(_REPO)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:24]


def _row_cache_entry(log_path: Path) -> Path | None:
    """Where ``log_path``'s entry lives, or None when the row cache is off.

    Named by the log's real path, so a symlink farm (the dashboard trees are one) shares
    its target's entry. The key stored inside, not the name, decides a hit.
    """
    base = _cache_dir(_ROW_CACHE_ENV, "eval-log-rows")
    name = hashlib.sha256(os.path.realpath(log_path).encode()).hexdigest()[:32]
    return None if base is None else base / f"{name}.arrow"


def _read_row_cache(
    entry: Path, key: dict[str, Any], columns: Sequence[str] | None
) -> pa.Table | None:
    """The cached table if ``entry`` holds rows for exactly ``key``; None is a miss.

    Read through a plain file, not a memory map: another process may replace the entry
    meanwhile, and on NFS a mapped file that vanishes is a SIGBUS, not an exception.
    ``included_fields`` skips the unwanted columns' buffers (``raw_output`` above all).
    """
    names = list(columns) if columns is not None else SCHEMA.names
    try:
        options = ipc.IpcReadOptions(included_fields=[SCHEMA.get_field_index(c) for c in names])
        with pa.OSFile(str(entry)) as source:
            reader = ipc.open_file(source, options=options)
            stored = json.loads((reader.schema.metadata or {}).get(_ROW_CACHE_KEY, b"null"))
            if not isinstance(stored, dict) or {k: stored.get(k) for k in key} != key:
                return None
            table = reader.read_all()
        if table.num_rows != stored.get("rows") or table.schema.remove_metadata() != pa.schema(
            [SCHEMA.field(c) for c in names]
        ):
            return None
        digests = stored.get("digests") or {}
        if any(digests.get(c) != d for c, d in _column_digests(table).items()):
            log.info("ignoring row-cache entry %s: payload digest mismatch", entry)
            return None
        return table.replace_schema_metadata(None)
    except FileNotFoundError:
        return None
    except Exception as exc:  # torn, truncated or foreign: a miss, and the write repairs it
        log.info("ignoring unreadable row-cache entry %s: %s", entry, exc)
        return None


def _column_digests(table: pa.Table) -> dict[str, str]:
    """sha256 of each column's Arrow buffers, so a flipped bit in a payload is a miss.

    Stored at write, checked per projected column at read. A layout that differs without
    any value differing (not seen: an IPC round trip keeps the buffers) is also a miss,
    which costs a decode, never a wrong row.
    """
    out = {}
    for name in table.column_names:
        digest = hashlib.sha256()
        for chunk in table.column(name).chunks:
            for buf in chunk.buffers():
                digest.update(b"-" if buf is None else b"+")
                if buf is not None:
                    digest.update(buf)  # zero-copy: hashlib reads the Arrow buffer in place
        out[name] = digest.hexdigest()
    return out


def _write_row_cache(entry: Path, key: dict[str, Any], table: pa.Table) -> None:
    """Write-then-rename (`atomic_write`), so a concurrent reader sees old or new, never half.

    Two writers of one entry each write a whole file for the same log; the last rename
    wins and either is correct. A cache that cannot be written leaves the read uncached.
    """
    try:  # everything, the digest included: filling an optional cache must never raise
        stored = {**key, "rows": table.num_rows, "digests": _column_digests(table)}
        meta = {_ROW_CACHE_KEY: json.dumps(stored).encode()}
        with atomic_write(entry) as tmp, ipc.new_file(str(tmp), SCHEMA.with_metadata(meta)) as w:
            w.write_table(table)
    except Exception as exc:  # a full disk, a read-only home, no memory: the read goes on uncached
        log.info("could not write row-cache entry %s: %s", entry, exc)


def log_rows_table(header: LogHeader, columns: Sequence[str] | None = None) -> pa.Table:
    """The rows the Parquet writer bridged from ``header.path``, as a `SCHEMA` table, cached.

    `rows_from_eval_log(as_written=True)`: the writer's bridge, in dataset order, without
    its runner stamps. ``columns`` projects the cached read (``raw_output`` costs nothing
    when not asked for). A miss decodes the log and fills the entry, unless the log changed
    during the decode.

    Raises:
        Whatever `rows_from_eval_log` raises for a log it cannot read; nothing is cached.
    """
    path = header.path
    stat = _stat_key(path.stat())
    # Which registered function reads this eval (its bridge, else the writer's default):
    # a process without a custom bridge must not be served rows another process bridged.
    reader = log_bridge_for(header.eval_name or "") or _require_writer()
    key = {
        "version": _ROW_CACHE_VERSION,
        "stat": stat,
        "bridge": _bridge_version(),
        "reader": repr(_bridge_identity(reader)),
    }
    entry = _row_cache_entry(path)
    cached = _read_row_cache(entry, key, columns) if entry is not None else None
    if cached is not None:
        return cached
    rows = rows_from_eval_log(path, as_written=True)
    table = pa.Table.from_pylist([r.to_row() for r in rows], schema=SCHEMA)
    if entry is not None and _stat_key(path.stat()) == stat:
        _write_row_cache(entry, key, table)
    return table if columns is None else table.select(list(columns))


def log_cell_tables(
    root: Path | str = DEFAULT_RESULTS_ROOT,
    columns: Sequence[str] | None = None,
    *,
    verify: bool = True,
) -> dict[tuple[str, str], pa.Table]:
    """The cells `load_all(store="log")` reads from a log under ``root``, as it reads them.

    ``verify=False`` also reads from its source log every cell `_log_matches` would keep
    on Parquet (re-judged rows), which is what `scripts/check_logs_match_parquet.py` needs
    to report them. Empty when there are no logs, or inspect-ai is not installed (with a
    warning).
    """
    results_store()  # validates MO_EVALS_STORE, as every reader does
    keys = [
        (f, _read_parquet(f, columns=["spec_hash", "eval_name"])) for f in iter_result_files(root)
    ]
    return dict(_tables_from_eval_logs(root, keys, columns, as_written=True, verify=verify)[1])


def _source_log_table(path: Path, columns: Sequence[str] | None) -> pa.Table | None:
    """The rows of the log this result file was bridged from, if that log is still here.

    Only the file's own source: the log its stamp names (`write_results(source_log=...)`)
    at the stamped size, under the nearest ancestor of ``path`` holding ``logs/``. A file
    with no stamp (a re-judge copy under ``rejudges/``, a legacy cell) or whose source is
    gone or rewritten has no such rows. A log stands in only if `_log_matches` finds the same
    organism, eval, rows and row order; the one documented limit is metadata-only loss that
    nothing marked, which no cheap check can see.
    """
    stamp = _source_log_stamp(pq.read_schema(str(path)))
    root = next((p for p in path.parents if (p / LOGS_DIRNAME).is_dir()), None)
    if stamp is None or root is None:
        return None
    try:
        headers = eval_log_headers(root)
    except RuntimeError:  # inspect-ai absent
        return None
    _require_writer()  # outside `_log_matches`'s per-log `except`, as in `load_all`:
    _bridge_version()  # a missing reader or a broken key scan is a bug, not a bad log
    for header in headers:
        if (header.path.name, header.size, header.status) == (
            stamp.log_name,
            stamp.log_size,
            "success",
        ) and _is_stamped_source(stamp, stamp.eval_id, header, header.path.name, header.size):
            if _log_matches(path, stamp, header):
                return log_rows_table(header, columns)
    return None


def _require_writer() -> Callable[..., list[SampleResult]]:
    """The writer's default reader (`evals._inspect.rows_from_log`); raises if unregistered.

    Core may not import `evals/`, so the `log` store cannot fetch it itself. Raised rather
    than skipped: a missing import would otherwise read every log as unreadable and fall
    back to Parquet (or to nothing) without an error.
    """
    writer = _LOG_BRIDGES.get(None)
    if writer is None:
        raise RuntimeError(
            "MO_EVALS_STORE=log reads logs through the Parquet writer's reader, which "
            "`evals._inspect` registers: import it (or `serving.inspect_runner`) first"
        )
    return writer


# Always read for the check, whatever a caller projects: identity (spec_hash, eval_name) and
# the rows themselves. Everything else (`meta`) may differ only by the bridge-only stamps.
_AGREE = ["spec_hash", "eval_name", "sample_id", "axis", "score", "raw_output"]


def _log_matches(parquet: Path, stamp: _SourceStamp, header: LogHeader) -> bool:
    """May ``header``'s log stand in for ``parquet``, a file stamped with it as source?

    Not if the writer marked the rows re-judged (`write_results(rejudged=True)`). A file
    written before that mark existed cannot say, and the re-score scripts stamp their
    output with the generation log too, so the log replaces it only if the two agree on
    every row's spec_hash, eval_name, sample_id, axis, score (NaN-aware) and raw_output, in
    the same row order (the log as written against the Parquet as stored); otherwise the
    Parquet wins, with a warning. A log that cannot be read does not agree.
    """
    if stamp.rejudged:
        return False

    def rows(table: pa.Table) -> list[tuple[str, ...]]:
        # In row order, not sorted: a reordered Parquet is not the log's rows.
        return [
            (
                r["spec_hash"],
                r["eval_name"],
                r["sample_id"],
                r["axis"],
                "nan" if math.isnan(r["score"]) else repr(r["score"]),
                r["raw_output"],
            )
            for r in table.to_pylist()
        ]

    try:
        logged = log_rows_table(header, _AGREE)
    except Exception as exc:  # a corrupt log: the Parquet stays, named
        log.warning("skipping eval log %s: %s", header.path, exc)
        return False
    if rows(logged) == rows(_read_parquet(parquet, _AGREE)):
        return True
    log.warning(
        "MO_EVALS_STORE=log: %s differs from its stamped source log %s in spec_hash, eval_name, "
        "sample_id, axis, score, raw_output or row order (re-judged before re-judges were "
        "marked?); read its Parquet",
        parquet,
        header.path,
    )
    return False
