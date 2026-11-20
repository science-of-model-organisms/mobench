"""The sweep's config, matrix and pre-flight layer. Not a runner any more.

``serving/inspect_runner.py`` is the one entry point (``python -m serving.inspect_runner``,
or ``scripts/run_all.sh``). This module used to schedule the matrix itself, nesting a
thread pool over organisms around one over evals and calling each eval's own body; that
sweep — ``run_spec`` / ``run_all`` / ``_run_eval``, and the ``main()`` behind
``python -m serving.runner`` — was deleted once every registered eval had an Inspect task
factory, together with the ``runner_backend: native`` hatch that reached it.

What is left is everything the eval_set runner reads before it schedules anything, kept
here so there is one definition of each and no second copy to drift:

* config and matrix — ``load_config``, ``load_specs`` (defaults, variants, the grouped
  ``organisms:`` shape), ``eval_config_for``, ``_select_specs`` / ``_select_evals``;
* pre-flights — ``assert_real_backends``, ``assert_no_item_cap``, ``_force_fake_backends``;
* the row and run bookkeeping — ``_stamp``, ``record_judge_usage``, ``timing_table``.

Deliberately not orchestration: eval_set plus "skip if the Parquet file
exists" is correct at this scale. No queue, no Ray, no Airflow.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, override

import yaml
from yaml.constructor import ConstructorError

from core.declared import declared_names, declared_sha_of
from core.flag_values import listed
from core.results import SampleResult
from core.spec import OrganismSpec
from core.yaml_safe import reject_cycles
from evals._harness import row_is_complete
from serving.resolve import (
    context_budget_tokens,
    fake_mode,
    icl_prefix_meta,
    needs_vllm,
    spec_backend_group,
)
from serving.timing import write_run_usage

if TYPE_CHECKING:  # `evals._inspect` drags `inspect_ai` in; this is annotation-only here.
    from evals._inspect import RegisteredEval

log = logging.getLogger(__name__)


def load_dotenv(path: Path | str | None = None) -> None:
    """Export ``KEY=VALUE`` pairs from ``.env`` into the environment; existing env wins.

    The runner needs ``HF_TOKEN`` (private artifact downloads in resolve) and
    ``OPENAI_API_KEY`` (the judge) which live in the gitignored ``.env``. A pod shell that
    has not exported them fails every resolve with a 401 — all 36 sweep tasks at once.
    Minimal parser per CLAUDE.md; never overrides ambient environment, so exported vars and CI
    secrets keep precedence.
    """
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip().strip("\"'"))


@dataclass
class RunOutcome:
    """What happened for one (organism, eval) pair."""

    spec_hash: str
    eval_name: str
    status: str  # "written" | "cached" | "skipped" | "failed"
    n_samples: int = 0
    detail: str = ""
    seconds: float = 0.0  # this eval's own wall time; ~0 when cached or skipped


class _NoDuplicatesLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys instead of silently keeping the last.

    PyYAML's default drops earlier duplicates with no error; a doubled `mu_decisiveness:`
    block once swallowed an `items_path` override and the run used the wrong items. Make
    that a loud failure at load time.

    Only keys written out in the mapping are checked. A merge key (`<<: *anchor`, which
    configs/specs.yaml uses to state the shared LoRA knobs once) is skipped here: it has
    no constructor of its own, and the base class expands it in `flatten_mapping` on the
    way through `super()`. Checking after the expansion would also make an intentional
    override of an anchored key look like a duplicate.
    """

    _MERGE_TAG = "tag:yaml.org,2002:merge"

    # `@override` here, and not on the 24 other undecorated real overrides in the tree,
    # because this is the only one where a foreign base class breaking would break a RESULT
    # in silence. If a PyYAML bump renames `SafeLoader.construct_mapping`,
    # `invalid-explicit-override` (error by default) fails the gate; without it this becomes
    # a method nothing calls, duplicate keys go back to being silently deduped, and a run
    # uses the wrong items — the exact failure this class exists to stop. The tree's other
    # foreign bases (torch, http.server, both under tests/) fail loudly instead; the full
    # count and the reason `missing-override-decorator` stays off are in pyproject.toml.
    @override
    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            if key_node.tag == self._MERGE_TAG:
                continue
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, Hashable):  # `? [a, b]`: `key in seen` would be a TypeError
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"mapping keys must be scalars, found a {type(key).__name__}",
                    key_node.start_mark,
                )
            if key in seen:
                raise ValueError(f"duplicate key {key!r} in YAML mapping ({node.start_mark})")
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


class ConfigReadError(yaml.YAMLError, ValueError):
    """A spec or user config file that cannot be read as a config: malformed, cyclic, nested too
    deeply, a duplicate key, or a top level that is not a mapping.

    The one type `mobench` turns into a refusal; any other `YAMLError` is not a refusal. It is
    also a `ValueError`, so a caller that catches `ValueError` for a bad file still does."""


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a specs YAML file, rejecting duplicate keys. Raises `ConfigReadError` (and only
    that, bar an unreadable file's `OSError`) for any file that cannot be read as a config."""
    try:
        try:
            data = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=_NoDuplicatesLoader)  # noqa: S506  # subclass of SafeLoader
        except RecursionError:
            raise yaml.YAMLError("config is nested too deeply") from None
        reject_cycles(data)
    except (yaml.YAMLError, ValueError) as exc:  # ValueError: a duplicate key
        if isinstance(exc, ConfigReadError):
            raise
        # one line: a YAML error spans several, and a refusal is one line
        raise ConfigReadError(f"{path}: {' '.join(str(exc).split())}") from exc
    if not isinstance(data, dict):
        raise ConfigReadError(f"{path} must contain a top-level mapping")
    return data


def _expand_variants(path: Path | str, where: str, entry: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand one entry's `variants` into one spec dict each.

    A sweep is a set of organisms differing in one or two fields — the `few_shot_icl`
    shot-count sweep today, an alpha or layer sweep on `steering` tomorrow. Writing each
    out in full repeats the adapter URI and every training knob on each line, which is
    exactly the kind of copy-paste that produces an organism nobody meant to register.

    A variant may override any spec field; its `method_config` merges *over* the parent's
    rather than replacing it, so a sweep states only what varies. Nothing here knows what
    the fields mean — no naming convention (`scale_0.25`) is derived, because that is a
    storage layout and the runner has no business encoding it.
    """
    base = {k: v for k, v in entry.items() if k != "variants"}
    variants = entry.get("variants")
    if variants is None:
        return [base]
    if not isinstance(variants, list) or not variants:
        # An empty list would silently register nothing, which reads as "this organism is
        # covered" while producing no rows at all.
        raise ValueError(f"{path}: {where} 'variants' must be a non-empty list")

    expanded: list[dict[str, Any]] = []
    for i, variant in enumerate(variants):
        if not isinstance(variant, dict):
            raise ValueError(f"{path}: {where} variants[{i}] must be a mapping")
        merged = {**base, **{k: v for k, v in variant.items() if k != "method_config"}}
        merged["method_config"] = {
            **(base.get("method_config") or {}),
            **(variant.get("method_config") or {}),
        }
        expanded.append(merged)
    return expanded


def _entries(path: Path | str, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten either config shape into a list of spec dicts.

    Two shapes are accepted:

    - `organisms:` — a mapping of persona id to its list of organisms. This is the shape
      `configs/specs.yaml` uses: one persona is one block, so every method for a persona
      reads together and `persona_id` is stated once rather than on every entry.
    - `specs:` — a flat list, each entry naming its own `persona_id`. Kept for ad-hoc and
      test configs, where one or two organisms in a temp file is the whole point.

    Either shape may use `variants:` to expand a sweep (see `_expand_variants`).

    A config with neither key is an error rather than an empty run: silently evaluating
    nothing looks exactly like a cache hit.
    """
    if "organisms" in data:
        grouped = data.get("organisms") or {}
        if not isinstance(grouped, dict):
            raise ValueError(f"{path}: 'organisms' must be a mapping of persona id to a list")
        flat: list[dict[str, Any]] = []
        for persona_id, organisms in grouped.items():
            if not isinstance(organisms, list):
                raise ValueError(f"{path}: organisms[{persona_id!r}] must be a list")
            for i, entry in enumerate(organisms):
                if not isinstance(entry, dict):
                    raise ValueError(f"{path}: organisms[{persona_id!r}][{i}] must be a mapping")
                if "persona_id" in entry and entry["persona_id"] != persona_id:
                    # The key is the persona. A conflicting field is a copy-paste error, and
                    # silently preferring one of the two would register the wrong organism.
                    raise ValueError(
                        f"{path}: organisms[{persona_id!r}][{i}] declares persona_id "
                        f"{entry['persona_id']!r}; remove it or move the entry"
                    )
                where = f"organisms[{persona_id!r}][{i}]"
                flat.extend(_expand_variants(path, where, {**entry, "persona_id": persona_id}))
        return flat

    if "specs" in data:
        entries = data.get("specs") or []
        if not isinstance(entries, list):
            raise ValueError(f"{path}: 'specs' must be a list")
        flat = []
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: specs[{i}] must be a mapping")
            flat.extend(_expand_variants(path, f"specs[{i}]", entry))
        return flat

    raise ValueError(f"{path}: no 'organisms' or 'specs' key; nothing to run")


def load_specs(source: Path | str | dict[str, Any]) -> list[OrganismSpec]:
    """Build specs from a config file, or an already-loaded config dict.

    Each organism is merged over `defaults:`, so a config only states what differs between
    organisms. See `_entries` for the two accepted shapes. Pass an already-parsed dict to
    avoid re-reading the file (e.g. `run_all` loads the config once and hands it straight
    in).
    """
    data = source if isinstance(source, dict) else load_config(source)
    where = "<config>" if isinstance(source, dict) else source
    defaults = data.get("defaults") or {}
    # When the top-level `seeds:` knob is set it OWNS replication (`expand_seeds`), so an
    # organism must not also pin its own seed. Caught here, not in `expand_seeds`, because
    # here the raw entry still tells an EXPLICIT `seed:` (any value, `0` included) apart
    # from the `defaults` seed — by the time specs are built that distinction is gone, which
    # is why the `expand_seeds` guard alone cannot see an explicit `seed: 0`.
    seeds_declared = data.get("seeds") is not None

    specs: list[OrganismSpec] = []
    for entry in _entries(where, data):
        if seeds_declared and "seed" in entry:
            raise ValueError(
                f"{where}: an organism pins seed={entry['seed']!r} while top-level 'seeds:' "
                "is set; the 'seeds:' knob owns seed replication (serving.runner."
                "expand_seeds) — remove the per-organism seed."
            )
        merged = {**defaults, **entry}
        merged["method_config"] = {
            **(defaults.get("method_config") or {}),
            **(entry.get("method_config") or {}),
        }
        specs.append(OrganismSpec.from_dict(merged))

    hashes = [s.spec_hash for s in specs]
    dupes = {h for h in hashes if hashes.count(h) > 1}
    if dupes:
        raise ValueError(f"{where}: duplicate specs (same hash) {sorted(dupes)}")
    return specs


class SpecConflictError(ValueError):
    """Two spec files disagree. Never resolved silently: a quiet "later file wins" would be a
    second source of truth about an organism or an eval."""


def _flat_entries(path: Path | str, data: dict[str, Any]) -> list[dict[str, Any]]:
    """Every organism of one loaded file, merged over that file's own `defaults:`.

    The same merge as `load_specs`, so an entry stands alone once its file is out of the
    picture, which is what lets several files share one config.
    """
    defaults = data.get("defaults") or {}
    flat = []
    for entry in _entries(path, data):
        merged = {**defaults, **entry}
        merged["method_config"] = {
            **(defaults.get("method_config") or {}),
            **(entry.get("method_config") or {}),
        }
        if data.get("seeds") is not None and "seed" not in entry:
            merged.pop("seed", None)  # a defaulted seed must not read as a pin (`load_specs`)
        flat.append(merged)
    return flat


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def merge_configs(paths: Sequence[Path | str]) -> dict[str, Any]:
    """One config dict from one or more spec files.

    One file is returned exactly as `load_config` gives it, so the single-file path cannot
    change. For several, each file keeps its own `defaults:`; the organisms are concatenated
    (a row repeated byte for byte is kept once, a differing row with the same `spec_hash`
    raises), `eval_configs` and `models` are unioned (a key defined differently raises),
    `default_evals` is the union in first-seen order, and any other top-level knob must
    agree. YAML-anchor holders (`_name:`) are file-local and dropped.
    """
    if not paths:
        raise ValueError("no spec files given")
    if len(paths) == 1:
        return load_config(paths[0])

    merged: dict[str, Any] = {"defaults": {}, "specs": []}
    seen: dict[str, tuple[str, Path | str]] = {}
    default_evals: list[str] = []
    configs = [load_config(path) for path in paths]
    if len({_canonical(data.get("seeds")) for data in configs}) > 1:
        raise SpecConflictError("top-level 'seeds' differs between files")
    for path, data in zip(paths, configs, strict=True):
        specs = load_specs(data)
        entries = _flat_entries(path, data)
        for spec, entry in zip(specs, entries, strict=True):
            body = _canonical(entry)
            if spec.spec_hash in seen:
                first_body, first_path = seen[spec.spec_hash]
                if body != first_body:
                    raise SpecConflictError(
                        f"spec_hash {spec.spec_hash} ({spec}) is in {first_path} and {path} "
                        "with different entries"
                    )
                continue
            seen[spec.spec_hash] = (body, path)
            merged["specs"].append(entry)
        default_evals.extend(e for e in data.get("default_evals") or [] if e not in default_evals)
        for key, value in data.items():
            if key in {"defaults", "organisms", "specs", "default_evals"} or key.startswith("_"):
                continue
            if key in {"eval_configs", "models"}:
                block = merged.setdefault(key, {})
                for name, item in (value or {}).items():
                    if name in block and _canonical(block[name]) != _canonical(item):
                        raise SpecConflictError(f"{key}.{name} is defined differently in {path}")
                    block[name] = item
            elif key in merged and _canonical(merged[key]) != _canonical(value):
                raise SpecConflictError(f"top-level {key!r} differs in {path}")
            else:
                merged[key] = value
    if default_evals:
        merged["default_evals"] = default_evals
    return merged


def load_specs_many(paths: Sequence[Path | str]) -> list[OrganismSpec]:
    """`load_specs` over one or more files; see `merge_configs` for the conflict rules."""
    if len(paths) == 1:
        return load_specs(paths[0])
    return load_specs(merge_configs(paths))


def expand_seeds(specs: list[OrganismSpec], config: dict[str, Any]) -> list[OrganismSpec]:
    """Replicate every organism across the config's top-level ``seeds:`` list.

    Seed replication is declared ONCE for the whole matrix here, rather than hand-written as
    ``variants: [{seed: 0}, {seed: 1}, ...]`` on every entry and every base size (the drift
    that left a handful of 7B organisms at three seeds and everything else at one). It is
    applied in the RUN path (``run_all`` and the dry run) AFTER selection, deliberately NOT
    inside ``load_specs``: a seed replicate is the same organism evaluated again, not a new
    organism, so ``load_specs`` stays the distinct-organism list that every documented count
    and the multi-group/train_scale arithmetic read. Absent ``seeds:`` -> specs unchanged.

    Which evals actually RUN the extra seeds is decided downstream by
    ``inspect_runner._dedupe_seeds_evals`` (greedy cells dedupe to the seed-0
    representative; temperature-1.0 behavioural cells keep them all).
    """
    seeds = config.get("seeds")
    if seeds is None:
        return specs
    if not (
        isinstance(seeds, list)
        and seeds
        and all(isinstance(s, int) and not isinstance(s, bool) for s in seeds)
    ):
        raise ValueError(f"'seeds' must be a non-empty list of ints, got {seeds!r}")
    # Backstop for the double-owned state (the primary guard is in `load_specs`, which can
    # still see an EXPLICIT `seed:` incl. 0). Here specs carry a resolved seed, so this can
    # only catch a non-zero pin — enough for specs built outside `load_specs` (tests,
    # ad-hoc lists); an explicit `seed: 0` is caught upstream, not here.
    pinned = sorted({s.spec_hash for s in specs if s.seed != 0})
    if pinned:
        raise ValueError(
            "top-level 'seeds:' owns seed replication, but these specs pin a non-zero seed "
            f"of their own: {pinned}; remove the per-organism seed or the 'seeds:' knob."
        )
    return [replace(spec, seed=s) for spec in specs for s in seeds]


# The axes whose evals score a free-text GENERATION (not a logprob/activation read), so
# they decode at temperature 1.0 and keep their multi-seed replicates. A subset of
# `core/registry.VALID_AXES`; `coherence` is deliberately EXCLUDED because its cells
# (`mu_decisiveness`, `transitivity`) sample from their own plan default or read
# logprobs greedily. Consumed here (temperature) and by `inspect_runner.
# _dedupe_seeds_evals` (which seeds actually run). Keep the two in sync.
#
# `expression` must stay here: its eval `simple_behaviour_exhibit` (on `reliability` until
# the 2026-09-25 taxonomy v2) first ran greedy (no temperature injected), so its "two
# answers per question" were word-for-word copies on 24/40 base items and 34/40
# anti-persona items, and a present rate over them counted one answer twice
# (docs/evals/expression.md). It scores persona expression in free text,
# exactly like generality, so it samples like generality.
BEHAVIOURAL_AXES = frozenset(
    {
        "expression",
        "generality",
        "robustness",
        "introspection",
        "misalignment",
    }
)


def eval_config_for(
    config: dict[str, Any], spec: OrganismSpec, ev: RegisteredEval
) -> dict[str, Any]:
    """The config dict handed to an eval function.

    `spec_hash` and `seed` are injected because `SampleResult` needs the hash and
    determinism needs the seed. `persona_id` and `behavior_class` are injected because
    they say *what behaviour is under test* — an eval that scores persona expression
    cannot do so without knowing which persona, and the judge raises without it.

    `base_model` is injected because it says *which model* the organism is, not how it was
    made — a white-box cell must refuse an artifact (a direction, a probe) fitted on a
    different base, which would be a silent wrong-activation-space read. (The guard that
    first needed it, `_assert_probe_matches_base_model`, left with the retired
    `whitebox_probe_*` cells on 2026-10-14.) `getattr` with a default: the
    bring-your-own-model shim (`mo_evals._external_spec`) has no `base_model`.

    Nothing here still says how the organism was made: `method`, `method_config` and
    `train_scale` are withheld, and those are the fields that would break cross-method
    comparison (CLAUDE.md). Knowing the target behaviour is not knowing the method.

    `_context_budget_tokens` is the SERVING context this cell has to fit inside — a
    property of the engine, not of the organism, so it says nothing about the method. It
    is injected because an eval that builds a long conversation can only check itself
    against it at launch, and an over-length request inside a sweep fails its whole
    co-batched group. Underscore-prefixed because it is
    runner-injected plumbing rather than a knob anyone sets in YAML; an eval called
    outside the runner (a test, a re-score) simply will not see the key and must skip its
    guard rather than invent a budget.

    **`by_base_model`** in an eval's block is a per-size override, applied on top of the
    eval block for the organism's `base_model` and then removed so the eval body never
    sees it. It exists because the judged 1000-item banks and petri are cheap at 7B but
    hours at 32B, and the size-delta is the finding, not 1000-item precision on every
    model. The merge is one level deep so a nested
    sub-config (`petri: {max_turns: 6}`) updates rather than replaces the whole sub-dict.

    **`dedupe_seeds`** is a runner-only flag consumed in `run_matrix`; it is popped here so
    it never reaches an eval body as a spurious knob.
    """
    per_eval = config.get("eval_configs") or {}
    # `dedupe_seeds` is runner-only (read in run_matrix); strip it from BOTH the default
    # block and the eval block so it never reaches an eval body as a spurious knob.
    default = {k: v for k, v in (per_eval.get("default") or {}).items() if k != "dedupe_seeds"}
    block = dict(per_eval.get(ev.name) or {})
    block.pop("dedupe_seeds", None)
    by_base = block.pop("by_base_model", None) or {}
    # `getattr`, and only when there is an override to look up: the bring-your-own-model
    # shim (mo_evals._external_spec) has no `base_model` — there is no organism — so an
    # off-sweep single-cell run simply gets the un-overridden config, never an AttributeError.
    override = by_base.get(getattr(spec, "base_model", None)) if by_base else None
    if override:
        block = _merge_one_level(block, override)
    merged = {**default, **block}
    # The per-turn cap this eval will REQUEST (a cell's own `organism_max_tokens`, else its
    # bare `max_tokens`, beats the shared `complete_kwargs`); the context budget must reserve
    # it, not just the serving default, or a prompt filled to the budget plus a 2048-token
    # reply overruns the window.
    own_cap = merged.get("organism_max_tokens")
    shared_cap = (merged.get("complete_kwargs") or {}).get("max_tokens")
    # A bare top-level `max_tokens` is the key the misaligned cells (sandbagging,
    # sycophancy, overrefusal, reward_hacking) generate with, so it counts too. `bool` is an
    # int subclass, hence the explicit exclusion.
    bare_cap = merged.get("max_tokens")
    requested = next(
        (c for c in (own_cap, bare_cap) if isinstance(c, int) and not isinstance(c, bool)),
        shared_cap,
    )
    cfg = {
        **merged,
        "spec_hash": spec.spec_hash,
        "seed": spec.seed,
        "persona_id": spec.persona_id,
        "behavior_class": spec.behavior_class,
        "base_model": getattr(spec, "base_model", None),
        "_context_budget_tokens": context_budget_tokens(
            spec, output_tokens=int(requested) if requested else None
        ),
    }
    # Temperature by axis. The behavioural axes measure a free-text generation, so they
    # DECODE at temperature 1.0 — that is what makes their multi-seed replicates a real
    # sampling spread rather than byte-identical greedy duplicates. Every other axis
    # (capability, activations, and the logprob/preference coherence reads)
    # stays greedy: each builder already defaults `config.get("temperature", 0.0)`, and
    # `mu_decisiveness`/`transitivity` keep their own plan default of 1.0, so we
    # inject NOTHING for non-behavioural axes and leave those defaults untouched. An
    # explicit `temperature` in the eval block still wins (it is in `block`, merged above).
    # `getattr`: an off-sweep single-cell shim / test double may hand a lightweight `ev`
    # with no `axis`; treat that as non-behavioural (no injection) rather than erroring.
    # A DECLARED eval (evals/declared/) is a judged free-text generation by construction, so it
    # samples like the behavioural axes whatever property it sits on (`core.declared`).
    if (
        getattr(ev, "axis", None) in BEHAVIOURAL_AXES
        or getattr(ev, "name", None) in declared_names()
    ):
        cfg.setdefault("temperature", 1.0)
    # A declared eval's file (YAML, rubric, bank) is part of what the cell measures, so its
    # hash joins the stamped config: `core.results.is_cached` then refuses a cell measured
    # before an edit, exactly as it does for any other drifted knob. Shipped evals get no key.
    sha = declared_sha_of(getattr(ev, "name", ""))
    if sha:
        cfg["_declared_sha"] = sha
    return cfg


def _merge_one_level(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """`override` onto `base`, merging one level deep for nested dict values.

    A plain `{**base, **override}` would replace a nested sub-dict wholesale — for
    `petri` that would drop every key the override does not restate (auditor/judge model,
    dimensions, skills_root). One level of dict-merge is enough for every override we
    have; deeper nesting is not used and would need this to recurse.
    """
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = {**out[key], **value}
        else:
            out[key] = value
    return out


def _stamp(
    results: list[SampleResult],
    spec: OrganismSpec,
    ev: RegisteredEval,
    serving: dict[str, Any] | None = None,
) -> None:
    """Fill in / check the bookkeeping fields so every eval does not have to.

    ``serving`` records how the cell was served (runner, concurrency) into each row's
    ``meta``. Generation text is only reproducible for a fixed serving configuration
    *including its concurrency*, and the results tree
    has no other record of the width a run used — without this, two trees that differ
    only in ``max_workers`` are indistinguishable and silently non-comparable.

    A ``few_shot_icl`` cell also gets its prefix provenance (``icl_prefix_meta``), so a
    transcript proves the demonstrations were in context rather than leaving it to be
    inferred from output divergence (#145).
    """
    icl = icl_prefix_meta(spec)
    for r in results:
        r.meta.update(icl)
        if serving:
            for key, value in serving.items():
                r.meta.setdefault(key, value)
        # The stored completeness flag, computed HERE for the same reason as serving_route
        # below: every row of every eval flows through this stamper, so no bridge or eval
        # can forget it. A deferred-judge row (generation phase) is stamped incomplete; it is
        # re-stamped complete once `evals._harness.rescore_rows` fills its verdict in Phase B.
        r.meta["complete"] = row_is_complete(r.meta)
        # Computed HERE, not passed in, so no call site can forget it. A merged organism
        # and a LoRA-served one are the same map and not the same arithmetic (~0.05 apart
        # on logits), `results/` has no serving-stack component in its cache key, and
        # nothing else in the tree tells them apart -- so an omission pools them silently.
        # An earlier version left this to the caller and a mutation removing it there
        # passed the whole suite.
        r.meta.setdefault("serving_route", spec_backend_group(spec))
        if not r.spec_hash:
            r.spec_hash = spec.spec_hash
        elif r.spec_hash != spec.spec_hash:
            raise ValueError(
                f"{ev.name} returned spec_hash {r.spec_hash}, expected {spec.spec_hash}"
            )
        if not r.eval_name:
            r.eval_name = ev.name
        if not r.axis:
            r.axis = ev.axis
        elif r.axis != ev.axis:
            raise ValueError(f"{ev.name} returned axis {r.axis!r}, registered as {ev.axis!r}")


# Evals that call a live second model besides the judge, and the `eval_configs` key that
# names it. The default for a missing block is a Fake, so a real run must not schedule
# these without one.
#
# The bool is whether the block is REQUIRED. `robustness_multiturn_debate_gen` has no other
# arm — absent means `_FakeAdversary` in a real run. The drift cells do: the SCRIPTED
# partner is their default and a perfectly real measurement, so their `interlocutor` is
# only checked when the config asks for a live one. Present-but-not-http there means
# `_FakeInterlocutor`, which is the same silent-fake failure (evals/persistence.py).
# ponytail: hardcoded table — make evals declare the need if it grows past three.
_EVALS_NEEDING_ADVERSARY: dict[str, tuple[str, bool]] = {
    "robustness_multiturn_debate_gen": ("adversary", True),
    "persistence_multiturn_drift": ("interlocutor", False),
    "persistence_multiturn_drift_long": ("interlocutor", False),
}

# Evals that never call a judge, so a real run of only these needs no judge block, no
# judge API key, and no live judge probe. Without this the pre-flight demanded
# $OPENAI_API_KEY for a WMDP-only run and then spent a probe call on a judge that no
# scheduled eval would have used.
#
# CONSERVATIVE BY CONSTRUCTION, and the direction matters: an eval missing from this set
# is merely guarded unnecessarily, while an eval wrongly IN it starts a real run that
# fails partway through on a judge it cannot build. So only add one after checking the
# module builds no judge — for these six, every occurrence of "judge" in
# `capability{,_code,_gpqa,_wmdp,_format}.py` is prose in a comment.
#
# `ifeval` is deliberately ABSENT despite living in the same family: its `intrusion_judge`
# option turns on a real judge pass, so whether it needs a key depends on config, not on
# the eval.
# ponytail: hardcoded set — make evals declare this if the list grows past the
# capability family.
#: `ifeval` joined this list on 9 Sep and `gpqa_main` left it. Ours had an
#: `intrusion_judge` knob, so it was deliberately kept out; upstream's `ifeval` checks its
#: constraints with a program and has no judge anywhere in its path, so demanding a judge
#: block for it fired a live API probe for a model it will never call. `gpqa_main` is gone
#: from the registry entirely.
#: `misaligned_overrefusal` joined on 13 Sep. Its score column IS
#: `evals.misaligned.classify_refusal`, a deterministic string matcher, and its task path
#: (`evals/_task_misaligned.py`) builds no judge — the module's own docstring says every
#: registered eval in it is judge-free. It was simply omitted, so a run scheduling it
#: demanded $OPENAI_API_KEY and spent a live probe call on a judge it can never call.
#: `evals.misaligned.refusal_agreement` DOES take a judge and is NOT this cell: it is
#: unregistered, reaches no runner, and re-judges stored rows out of band.
#: `misaligned_sycophancy` and `misaligned_sandbagging` joined at the same time. They became
#: judge-free stock-MMLU cells (#250): their score column is `evals.misaligned` letter
#: parsing over `capability.parse_letter`, and their task path (`evals/_task_misaligned.py`)
#: builds no judge — verified by constructing both task factories with `judges.backends`
#: forced to raise, which they survive. The bar the note above sets — checking the module,
#: not inferring from a sibling — is met, so a real keyless sweep no longer aborts demanding
#: $OPENAI_API_KEY for a judge these cells never call.
_JUDGE_FREE_EVALS = frozenset(
    {
        "mmlu",
        "apps",  # stock inspect_evals.apps, execution-scored in a local sandbox, no judge
        "gpqa_diamond",
        "wmdp",
        "ifeval",
        "capability_format",
        "misaligned_overrefusal",
        "misaligned_sycophancy",
        "misaligned_sandbagging",
        # The world/self truth-probe cells (`evals.whitebox_truth`): a probe fit on captured
        # activations, arithmetic only, never a judge — so a judge-free white-box sweep must
        # not be refused for a missing $OPENROUTER_API_KEY (the V0 cells' Run-4 failure).
        "whitebox_truth_world",
        "whitebox_truth_self",
    }
)


def assert_no_item_cap() -> None:
    """Refuse a REAL run that has `MO_EVALS_MAX_ITEMS` set.

    It is CI plumbing: it shrinks every eval's item count so the fake pipeline finishes in
    minutes. On a real run it silently produces a small-n sweep that looks completely
    healthy — the cells are clean, `discard_frac` is ~0, so analysis's thin-cell filter
    does not flag them and they vote on the scorecard at full weight with n=25 instead of
    n=500. Nothing downstream can tell: the item count is in no cache key.

    `scripts/run_all.sh` refuses the same combination earlier and more broadly (it does
    not need `--real`); this is the backstop for every other entry point — the `--real`
    pre-flight below, and the SINGLE-CELL path (`mo_evals.build_cell`), which reaches a
    real organism without going through that pre-flight at all.
    """
    max_items = os.environ.get("MO_EVALS_MAX_ITEMS")
    if max_items:
        raise RuntimeError(
            f"real run: MO_EVALS_MAX_ITEMS={max_items} is set. It caps every eval's items "
            f"and the resulting numbers are not results. Unset it, or set MO_EVALS_FAKE=1 "
            f"if you meant a fake run."
        )


def assert_real_backends(
    config: dict[str, Any], evals: list[RegisteredEval], specs: list[OrganismSpec]
) -> None:
    """Fail *before* any run if a real sweep would silently use a fake judge/adversary.

    The whole point of `--real` is that a forgotten config block defaults to a Fake
    (FakeJudge / _FakeAdversary), which produces plausible-looking numbers that are not
    results. This turns that silent fallback into a loud, pre-flight error and also
    checks the API key is present so the run does not die an hour in.
    """
    if fake_mode():
        raise RuntimeError("real run requested but MO_EVALS_FAKE is set — unset it")
    assert_no_item_cap()
    # vllm is only needed if some organism serves through it — ask `needs_vllm`, never the
    # method name. A whitebox run scoped to adapter organisms uses the HF backend, which has
    # no vllm dependency, and requiring it there wrongly blocks a valid GPU run. This comment
    # has named the wrong HF-served methods twice — `steering` and `weight_edit` both left
    # (interp-engine in #128, a factored LoRA in #134) while this copy still claimed them,
    # which is the drift `serves_on_hf` exists to prevent, reappearing one caller away. vllm
    # is deliberately not a project dependency: stage the pod with it, then check
    # here.
    if needs_vllm(specs):
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "real run: cannot import vllm — a scheduled organism serves through it. "
                "Install it on the pod with `uv sync --extra gpu` (installs vllm + ninja; "
                "mind that torch's CUDA build must match the pod driver), then retry."
            ) from exc
    # Driver pre-flight: torch's CUDA build must match the pod driver, or generation
    # silently falls back to CPU (~1h for a 7B, looks fine) — the classic
    # CPU-fallback trap. Fail loud here, once, instead of discovering it in nvidia-smi mid-sweep.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "real run: torch.cuda.is_available() is False — no usable GPU. Usually the "
            "pod driver is older than the torch CUDA build (e.g. cu13 torch on a <13.0 "
            "driver). Provision a matching-driver pod or install a cu-matched torch build. "
            "Refusing to run on CPU."
        )
    ec = config.get("eval_configs") or {}
    default = ec.get("default") or {}

    def check(block: Any, where: str, hint: str) -> None:
        if not isinstance(block, dict) or block.get("backend") != "http":
            raise RuntimeError(
                f"real run: {where} would use a FAKE backend "
                f"(block absent or backend != 'http'). Set {hint}."
            )
        key_env = block.get("api_key_env", "OPENAI_API_KEY")
        if not os.environ.get(key_env):
            raise RuntimeError(f"real run: {where} needs ${key_env}, which is not set")

    to_probe: dict[str, tuple[dict[str, Any], str]] = {}
    for ev in evals:
        merged = {**default, **(ec.get(ev.name) or {})}
        if ev.name not in _JUDGE_FREE_EVALS:
            check(merged.get("judge"), f"eval {ev.name!r} judge", "eval_configs.default.judge")
            to_probe.setdefault(
                json.dumps(merged["judge"], sort_keys=True, default=str),
                (merged["judge"], f"eval {ev.name!r} judge"),
            )
            # The panel's SECONDARY judge (judges/judge.py) is scored on every judged
            # item exactly like the primary, so a real run must not silently fall back to
            # FakeJudge for it either — same checks, same probe, one level down. Absent
            # (every config before this PR) means single-judge mode, which needs none of
            # this — only a CONFIGURED secondary is held to the real-backend bar.
            if merged["judge"].get("secondary") is not None:
                check(
                    merged["judge"]["secondary"],
                    f"eval {ev.name!r} judge.secondary",
                    "eval_configs.default.judge.secondary",
                )
                to_probe.setdefault(
                    json.dumps(merged["judge"]["secondary"], sort_keys=True, default=str),
                    (merged["judge"]["secondary"], f"eval {ev.name!r} judge.secondary"),
                )
        partner = _EVALS_NEEDING_ADVERSARY.get(ev.name)
        if partner is not None:
            key, required = partner
            if required or merged.get(key) is not None:
                check(merged.get(key), f"eval {ev.name!r} {key}", f"eval_configs.{ev.name}.{key}")
                to_probe.setdefault(
                    json.dumps(merged[key], sort_keys=True, default=str),
                    (merged[key], f"eval {ev.name!r} {key}"),
                )

    # Live probe: one real call per distinct judge/adversary block. Presence checks
    # cannot catch a model rejecting the block's params — gpt-5.6 refused
    # `temperature: 0` with an HTTP 400 on every judge call of the first real sweep,
    # discovered only mid-run. A few tokens here buys
    # failing before any organism boots. Deduped by block content, so a sweep with one
    # shared judge probes once.
    # ASSUMPTION: the judge/adversary endpoint is already reachable at pre-flight. True
    # for the MVP (external OpenAI HTTP). If a judge is ever pointed at a vLLM the sweep
    # boots *later* (HttpJudge supports a local endpoint), this probe would abort a valid
    # run — gate it on the endpoint being external, or boot that server first.
    from judges.backends import build_backend

    for block, where in to_probe.values():
        try:
            build_backend(block).complete([{"role": "user", "content": "Reply: ok"}])
        except Exception as exc:
            raise RuntimeError(
                f"real run: live probe of {where} failed — the block's model/params are "
                f"rejected by the API. Fix eval_configs before the sweep. Cause: {exc}"
            ) from exc


#: Fake-mode caps on mu's upstream knobs (`evals.friedness.arcadia_plan`); see below.
_FAKE_MU_CAPS = {"m": 2, "n_reverse": 100, "n_triads": 100, "n_cross": 100, "fit_steps": 200}


def _force_fake_backends(config: dict[str, Any]) -> None:
    """In fake mode, neutralise every real judge/adversary/interlocutor block.

    `MO_EVALS_FAKE=1` means a no-GPU, no-network run. `fake_mode()` already fakes the
    organism (`resolve`), but the judge/adversary come from `eval_configs`; without this
    a real judge block would try to call OpenAI and fail with no key (that is what broke
    CI). Dropping the keys makes them fall back to FakeJudge / _FakeAdversary.
    """
    for block in (config.get("eval_configs") or {}).values():
        if isinstance(block, dict):
            block.pop("judge", None)
            block.pop("adversary", None)
            # Same argument as `adversary` — the drift cells' live conversation partner is
            # a `judges.backends` http block too — but DOWNGRADED, not dropped.
            # `evals.persistence._build_interlocutor` reads an absent key as "the scripted
            # arm", which is a different measurement from "the live-partner arm with a fake
            # partner"; popping it would quietly move the cell to the other arm in CI.
            if block.get("interlocutor") is not None:
                block["interlocutor"] = {"backend": "fake"}
    # mu runs Arcadia's protocol in fake mode like every other eval, but SHRUNK: with
    # FakeEndpoint instant, inspect's per-generate plumbing (~1.3ms, measured 2026-08-29)
    # and the 2,000-step fits are the whole cost, so the fix is fewer fake reads and a
    # shallower fit, not a faster backend. Full scale is 16,500 reads + three 2,000-step
    # fits per organism; m=2 and 100-comparison phases is ~4,200 reads, and every upstream
    # step (Elo rounds, mid-run fit, all three phases, compute_panel) still runs. Fake mode
    # needs the plumbing, not the scale; real runs are uncapped. setdefault: an explicit
    # value (specs.smoke.yaml) wins.
    eval_configs = config.setdefault("eval_configs", {})
    mu = eval_configs.setdefault("mu_decisiveness", {})
    if isinstance(mu, dict):
        for key, value in _FAKE_MU_CAPS.items():
            mu.setdefault(key, value)
    # Same argument, same shape: transitivity at the configured 1000 triads is
    # ~6k fake generations/organism. 100 triads exercises every seam.
    pc = eval_configs.setdefault("transitivity", {})
    if isinstance(pc, dict):
        pc.setdefault("n_triads", 100)


class SelectionError(ValueError):
    """What to run was selected wrongly: a selection flag (`--personas`, `--methods`,
    `--base-models`, `--evals`, `--exclude-evals`) with an empty field or a name that matches
    nothing. A CLI turns exactly this into a refusal, never a traceback."""


# Exit status of `python -m serving.inspect_runner` when the selection was refused (argparse
# uses 2, a failed cell 1). A parent decides "refused" by this code AND the `REFUSING:` line.
EXIT_SELECTION = 64

MAX_LISTED = 12  # names shown in a "no match" message; the rest are counted


def _available_text(available: set[str]) -> str:
    names = sorted(available)
    if len(names) <= MAX_LISTED:
        return str(names)
    return f"{names[:MAX_LISTED]} ... and {len(names) - MAX_LISTED} more"


def _validate_selection(requested: list[str], available: set[str], flag: str) -> None:
    """Raise if any requested name matches nothing — a typo silently shrinks the matrix.

    `--evals badname` or `--methods promting` (sic) would otherwise filter down to an
    empty or wrong set and "succeed" with fewer runs than intended.
    """
    unknown = sorted(n for n in requested if n not in available)
    if unknown:
        raise SelectionError(
            f"{flag}: no match for {unknown} (available: {_available_text(available)})"
        )


def _select_specs(
    specs: list[OrganismSpec],
    only_personas: list[str] | None,
    only_methods: list[str] | None,
    only_base_models: list[str] | None = None,
) -> list[OrganismSpec]:
    """Narrow the organism matrix by persona, implant method, and/or base model.

    One implementation, called by both `run_all` and the `--dry-run` path, so the work
    a dry run lists cannot drift from the work the real run does.

    Each filter carries the same typo-guard as `--evals`: a name matching nothing raises
    rather than silently shrinking the matrix, which would otherwise look like a smaller
    sweep that succeeded. `--base-models` is how a per-base pod claims exactly its own
    organisms (docs/COMMANDS.md, "Big sweep topology"): one engine per pod, one base model per pod.
    """
    if only_personas:
        _validate_selection(only_personas, {s.persona_id for s in specs}, "--personas")
        specs = [s for s in specs if s.persona_id in set(only_personas)]
    if only_methods:
        _validate_selection(only_methods, {s.method for s in specs}, "--methods")
        specs = [s for s in specs if s.method in set(only_methods)]
    if only_base_models:
        _validate_selection(only_base_models, {s.base_model for s in specs}, "--base-models")
        specs = [s for s in specs if s.base_model in set(only_base_models)]
    return specs


def _select_evals(
    evals: list[RegisteredEval],
    only_evals: list[str] | None,
    config: dict,
    exclude_evals: list[str] | None = None,
) -> list[RegisteredEval]:
    """Effective eval set. Precedence: explicit ``--evals`` > config ``default_evals`` > all.

    ``default_evals`` lets ``configs/specs.yaml`` pin the default real-run set without the
    deferred evals, which stay registered and opt-in-able via ``--evals``. Same typo-guard
    as ``--evals`` on either source, so a stale name in the config fails loudly instead of
    silently shrinking the matrix.

    ``exclude_evals`` (``--exclude-evals``) subtracts names from whatever the precedence
    above selected — the complement selector a tail pod uses to run *everything but* the
    sequential-multiturn family while a main pod runs the rest
    (docs/COMMANDS.md, "Big sweep topology").
    It carries the same typo-guard, and is applied last so it composes with either source.
    """
    source = "--evals"
    if not only_evals:
        default = config.get("default_evals")
        only_evals = list(default) if default else None
        source = "default_evals"
    if only_evals:
        _validate_selection(only_evals, {e.name for e in evals}, source)
        evals = [e for e in evals if e.name in set(only_evals)]
    if exclude_evals:
        _validate_selection(exclude_evals, {e.name for e in evals}, "--exclude-evals")
        evals = [e for e in evals if e.name not in set(exclude_evals)]
    return evals


# Serialises snapshot-then-write, so the file always holds the newest total. Without it
# two cells finishing together can write out of order and leave the older snapshot on disk.
# ponytail: one process-wide lock around a sub-millisecond write of a few hundred bytes,
# contended by at most `max_tasks` bridge threads once per cell.
# If cells ever get short enough for that to show up, flush on a token threshold instead.
_USAGE_FLUSH_LOCK = threading.Lock()


def record_judge_usage(results_root: Path | str) -> None:
    """Write this process's third-party token totals under `results_root/_usage/`.

    So a run can be costed later from disk rather than from a terminal someone had to be
    watching — and, since the API key is shared across the team, this is the ONLY
    attributable record: the provider dashboard reports the whole project. Diagnostic
    only, and never allowed to sink a completed sweep; a judge that was never built
    (fake mode, judge-free sweep) records nothing.

    Called after every cell's Parquet write as well as at the end of the run. `usage` is
    a running total and `write_run_usage` rewrites one file per process, so flushing often
    cannot double-count — it only shortens what a kill can destroy to the cell in flight.
    Cheap: one small JSON write per cell against minutes of GPU.

    Lives here rather than in either main() because both runners need it and only one of
    them had it (the eval_set runner silently stopped recording, 2026-08-24).
    """
    try:
        from judges.backends import judge_usage

        with _USAGE_FLUSH_LOCK:
            write_run_usage(judge_usage(), results_root)
    except Exception as exc:  # never let accounting sink a completed sweep
        log.warning("could not record judge usage: %s", exc)


def timing_table(outcomes: list[RunOutcome], *, wall: float | None = None) -> str:
    """One row per eval: cells run, samples written, and time spent.

    Printed at the end of every sweep so a run can be costed without re-running it, and
    so the next sweep's wall clock can be estimated from `s/sample` x the item cap.

    Cells run concurrently (eval_set's `max_tasks`, from `max_concurrent_evals`), so each
    row is that eval's own duration and the column sums exceed the wall clock. Extrapolate
    from `s/sample`, and only after scaling by the same concurrency.
    """
    ran: dict[str, dict[str, float]] = {}
    other: dict[str, int] = {}
    for o in outcomes:
        if o.status in {"written", "failed"}:
            row = ran.setdefault(o.eval_name, {"cells": 0.0, "failed": 0.0, "n": 0.0, "s": 0.0})
            row["cells"] += 1
            row["failed"] += o.status == "failed"
            row["n"] += o.n_samples
            row["s"] += o.seconds
        else:
            other[o.status] = other.get(o.status, 0) + 1

    head = (
        f"{'eval':<32}{'cells':>6}{'fail':>5}{'samples':>9}"
        f"{'total_s':>9}{'s/cell':>8}{'s/sample':>10}"
    )
    lines = ["", head, "-" * len(head)]
    for name, r in sorted(ran.items(), key=lambda kv: -kv[1]["s"]):
        per_cell = r["s"] / r["cells"] if r["cells"] else 0.0
        per_sample = r["s"] / r["n"] if r["n"] else 0.0
        lines.append(
            f"{name:<32}{int(r['cells']):>6}{int(r['failed']):>5}{int(r['n']):>9}"
            f"{r['s']:>9.1f}{per_cell:>8.1f}{per_sample:>10.2f}"
        )
    cells = sum(r["cells"] for r in ran.values())
    samples = sum(r["n"] for r in ran.values())
    secs = sum(r["s"] for r in ran.values())
    failed = sum(r["failed"] for r in ran.values())
    lines.append("-" * len(head))
    lines.append(
        f"{'TOTAL':<32}{int(cells):>6}{int(failed):>5}{int(samples):>9}{secs:>9.1f}"
        f"{'':>8}{secs / samples if samples else 0.0:>10.2f}"
    )
    if wall is not None:
        lines.append(f"wall clock {wall:.1f}s ({wall / 60:.1f} min); eval-seconds sum {secs:.1f}s")
    if other:
        lines.append("not run: " + ", ".join(f"{k}={v}" for k, v in sorted(other.items())))
    lines += _failure_lines(outcomes)
    lines += _skipped_lines(outcomes)
    return "\n".join(lines)


# How many distinct failure reasons to print before collapsing the rest into a count.
_MAX_FAILURE_REASONS = 10


def _failure_lines(outcomes: list[RunOutcome]) -> list[str]:
    """The distinct reasons cells failed, most common first.

    The summary used to report failures as a bare COUNT ("done: written=50, failed=571"),
    with each cell's `detail` reachable only from the per-cell log line — which the
    wholesale failure paths (`eval_set` raised, resolve failed, task build failed, no log
    returned) never emit. So the one run shape that most needs an explanation, a sweep
    that fails everything at once, was the one that printed none: a 571-cell failure on
    2026-08-25 left no recoverable cause at all. Reasons are grouped
    rather than listed per cell because those paths give every cell in a group the same
    detail, and 571 copies of one sentence is how it got scrolled past in the first place.
    """
    return _grouped_lines(
        outcomes,
        status="failed",
        heading="failures by reason",
        key=lambda o: o.detail or "(no detail)",
    )


def _skipped_lines(outcomes: list[RunOutcome]) -> list[str]:
    """The cells that did not run, by NAME and reason.

    `not run: skipped=1` above says a cell is missing without saying which one or why —
    and a skip leaves no Parquet, so the reason survives only in an INFO line thousands
    of lines up. That is the worst shape a hole in the scorecard can take: an absent cell
    reads as a design outcome (this eval does not apply to this organism) when it may be
    a capability the endpoint lost. Grouped like failures, but keyed on the EVAL NAME as
    well as the reason, because a skip is per-cell news rather than one sweep-wide cause.
    """
    return _grouped_lines(
        outcomes,
        status="skipped",
        heading="skipped cells (no rows written)",
        key=_skip_reason,
    )


def _skip_reason(outcome: RunOutcome) -> str:
    """`<eval>: <why>`. Most skip messages already open with the eval name (they are
    written to be read on their own, out of an INFO line), so strip it before prefixing
    rather than printing it twice."""
    detail = (outcome.detail or "(no reason recorded)").removeprefix(f"{outcome.eval_name}: ")
    return f"{outcome.eval_name}: {detail}"


def _grouped_lines(
    outcomes: list[RunOutcome],
    *,
    status: str,
    heading: str,
    key: Callable[[RunOutcome], str],
) -> list[str]:
    """Outcomes with `status`, grouped by `key`, most common first — or nothing at all."""
    reasons: dict[str, int] = {}
    for o in outcomes:
        if o.status == status:
            reasons[key(o)] = reasons.get(key(o), 0) + 1
    if not reasons:
        return []
    ranked = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))
    lines = [f"{heading} ({sum(reasons.values())} cell(s), {len(reasons)} distinct):"]
    for detail, n in ranked[:_MAX_FAILURE_REASONS]:
        lines.append(f"  {n:>5} x {detail}")
    if len(ranked) > _MAX_FAILURE_REASONS:
        rest = sum(n for _, n in ranked[_MAX_FAILURE_REASONS:])
        lines.append(
            f"  {rest:>5} x ... {len(ranked) - _MAX_FAILURE_REASONS} more distinct reason(s)"
        )
    return lines


def selection_names(chunks: list[str], flag: str, *, lenient: bool = False) -> list[str]:
    """Flatten repeated, comma-separated selection flags into one list of names.

    Lives here rather than in `serving/inspect_runner.py` because four CLIs need it:
    the runner, `scripts/run_robustness_scale.py`, `scripts/rescore_judge_failures.py`
    and `scripts/build_candidates.py`. Two of those drive real, paid sweeps, so the
    last-wins bug this guards against is a hazard (CONTRIBUTING.md, Real runs) in each of them.

    ``chunks`` is empty when the flag was never passed at all -- that means "no filter"
    and stays empty. But an explicit empty value (``--personas ""``, ``--personas ","``)
    must fail loudly rather than collapse to that same empty list: it would otherwise be
    indistinguishable from "no filter" and silently widen a real run to the full matrix,
    which is exactly what CLAUDE.md exists to stop. So once the flag was given at
    least once, an empty result is an error, not "no filter".

    `lenient=True` (the runner's selection flags, as `mobench` parses them: one shared
    `core.flag_values.listed`) cleans a stray comma (`--personas impulsive,`, `,a`, `a,,b`)
    and drops repeats; an OCCURRENCE with nothing in it still fails, whatever else the flag
    carries. The default stays strict for the other scripts (rungs, buckets, label flags),
    which pin it. (The reasoning for strictness:)

    The check is per FIELD, not per chunk or on the flattened total: `--personas ,impulsive`
    or `--personas a,,b` splits to a list with an empty field, whose non-empty names are
    still `["impulsive"]` / `["a", "b"]` -- so a check on the flattened result silently
    dropped the empty entry (PR #274 review) instead of failing. Rejecting any empty field
    catches a leading/trailing/interior empty (and a wholly-empty occurrence) regardless of
    how many valid values, or valid repeated occurrences of the flag, are also present.
    """
    names: list[str] = []
    for chunk in chunks:
        if lenient:  # `impulsive,` is `impulsive`; only an occurrence with nothing in it fails
            parts = listed([chunk])
            if not parts:
                raise SelectionError(
                    f"{flag}: empty value {chunk!r} -- pass names like 'a,b' or omit the flag"
                )
            names.extend(parts)
            continue
        fields = chunk.split(",")
        if any(not field for field in fields):
            raise SelectionError(
                f"{flag}: empty value in {chunk!r} -- pass names like 'a,b' or omit the flag"
            )
        names.extend(fields)
    return listed(names) if lenient else names
