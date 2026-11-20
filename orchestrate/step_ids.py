"""Step ids: the descriptive names, and the short ids (`s1`-`s14`, `g1`-`g5`, `e1`-`e4`) they
replaced. The ONE table of the rename; nothing else maps an old id to a new one.

Every place a person writes or reads a step uses the new ids only: `steps.<id>` keys in a build
config, `--until`, `ledger settle`, and every line the build and `mobench generate` print. An old
id there is refused with the new one named (`renamed_problem`); it is never quietly accepted.

The old ids survive in two places, on purpose, so upgrading re-runs and re-pays nothing:

* records a build wrote before the rename (`state.json` keys, `ledger.jsonl` rows, pod records)
  are read through `current`, so a step finished under its old id is still finished;
* each step's fingerprint hashes `fingerprint_label` (the old id), as it hashed the id before,
  and a `steps:` key the fingerprint names (`also_config`) is hashed under its old name.

The build's work-dir layout (`s4/`, `s5/`, `s9p-<m>.json`; `g2/`, `g4/`, `e1/` under
`<id>/data/`) keeps its old names: those paths are fingerprint and state labels too.
"""

from __future__ import annotations

# The build graph (`orchestrate/steps.py`). A fixed id, then the per-method families.
BUILD: dict[str, str] = {
    "s1": "check-constitution",
    "s2": "derive-personas",
    "s3": "adopt-authored",
    "s4": "few-shot",
    "s5a": "download-teacher",
    "s5-teacher": "pilot-teacher",
    "s5": "pilot-gate",
    "s6": "full-teacher",
    "s7a": "sft-view",
    "s7bc": "dpo-pairs",
    "s10-12": "adopt-assets",
    "s13": "register",
    "s14": "evaluate",
}
BUILD_FAMILIES: dict[str, str] = {"s8-": "train-", "s9-": "screen-", "s9p-": "publish-"}

# The `mobench generate` graph (`orchestrate/generate.py`).
GENERATE: dict[str, str] = {
    "g1": "check-behaviour",
    "g2": "draft-texts",
    "g3": "persona-files",
    "g4": "base-transcripts",
    "g5": "universe",
    "e1": "draft-eval-items",
    "e2": "sbe-bank",
    "e4": "fake-check",
}
GENERATE_FAMILIES: dict[str, str] = {"e3-": "generality-"}

# `steps:` blocks shared by a family of steps (not a step id themselves).
SHARED_BLOCKS: dict[str, str] = {
    "s7": "dpo-pairs",  # s7bc's block (s7a reads its own, `s7a`)
    "s9": "screen",  # every screen-<m>
    "s9p": "publish",  # every publish-<m>
    "e3": "generality-banks",  # every generality-<cell>
}

# Job labels (what a pod record's `step` holds) that were never step ids: a step's job under
# another name. Read through `current` only; a person writing one gets no rename hint.
JOB_LABELS: dict[str, str] = {
    "s7b": "dpo-student",
    "s5-judge": "pilot-judge",
    "s5-gate": "pilot-gate",
}

# Old per-cell / per-method `steps:` keys that were never valid: the shared block configures them.
FAMILY_BLOCKS: dict[str, str] = {"s9-": "screen", "s9p-": "publish", "e3-": "generality-banks"}

FIXED: dict[str, str] = {**BUILD, **GENERATE}
FAMILIES: dict[str, str] = {**BUILD_FAMILIES, **GENERATE_FAMILIES}
BLOCKS: dict[str, str] = {**FIXED, **SHARED_BLOCKS}
_FIXED_BACK = {new: old for old, new in FIXED.items()}
_BLOCKS_BACK = {new: old for old, new in BLOCKS.items()}
_FAMILIES_BACK = {new: old for old, new in FAMILIES.items()}


def _family(step: str, table: dict[str, str]) -> str | None:
    """`step` with its family prefix swapped through `table`. Longest prefix first: `s9p-sft` is
    `s9p-` + `sft`, never `s9-` + `p-sft`."""
    for prefix in sorted(table, key=len, reverse=True):
        if step.startswith(prefix) and len(step) > len(prefix):
            return table[prefix] + step[len(prefix) :]
    return None


def renamed(step: str) -> str | None:
    """The new id of an OLD step id (`s14` -> `evaluate`, `s8-sft` -> `train-sft`), else None
    (a new id, or no id at all)."""
    if step in FIXED:
        return FIXED[step]
    return _family(step, FAMILIES)


def renamed_block(key: str) -> str | None:
    """The new name of an OLD `steps:` key (a step's own block or a shared one), else None. An
    old per-cell or per-method key (`e3-fermi`, `s9-sft`) names the shared block: that key was
    never valid."""
    if key in BLOCKS:
        return BLOCKS[key]
    for prefix in sorted(FAMILY_BLOCKS, key=len, reverse=True):
        if key.startswith(prefix) and len(key) > len(prefix):
            return FAMILY_BLOCKS[prefix]
    return _family(key, FAMILIES)


def current(step: str) -> str:
    """The id to use now for a step id read from a record: the new id of an old one, else the id
    itself. Only for READING what an older build wrote; input from a person goes through
    `renamed_problem` instead."""
    return renamed(step) or JOB_LABELS.get(step) or step


def current_keys(steps: dict[str, object]) -> dict[str, object]:
    """A `state.json` `steps` mapping keyed by the ids in use now. Should a record hold both an
    old and a new key for one step, the new key's entry wins (it was written later)."""
    out = {current(k): v for k, v in steps.items() if renamed(k) is not None}
    return {**out, **{k: v for k, v in steps.items() if renamed(k) is None}}


def fingerprint_label(step: str) -> str:
    """The id a step's fingerprint hashes: its OLD id where it had one, so a finished step keeps
    its fingerprint byte for byte across the rename. A step added since has no old id and is
    labelled by its own."""
    if step in _FIXED_BACK:
        return _FIXED_BACK[step]
    return _family(step, _FAMILIES_BACK) or step


def old_labels(step: str) -> list[str]:
    """Every name a step's pod record may carry from before the rename: its old step id, then
    any old job label that now reads as it (`dpo-student` -> `s7b`)."""
    return [fingerprint_label(step), *(o for o, n in JOB_LABELS.items() if n == step)]


def fingerprint_block(key: str) -> str:
    """`fingerprint_label` for a `steps:` block name (`pilot-gate` -> `s5`)."""
    if key in _BLOCKS_BACK:
        return _BLOCKS_BACK[key]
    return fingerprint_label(key)


def renamed_problem(value: str, what: str, graph: str | None = None) -> str | None:
    """The refusal for an old id where a person wrote one (`what` is how they wrote it: `--until`
    or `steps.`), naming the new id; None when `value` is not an old id. With `graph` (`build` or
    `generate`), an id of the OTHER graph is not an old id here: it names no step."""
    if what == "steps.":
        new = renamed_block(value)
        return f"steps.{value} is now steps.{new}; rename it" if new else None
    new = renamed(value)
    if new and graph and not _in_graph(new, graph):
        return None
    return f"{what} {value} is now {what} {new} (the step ids were renamed)" if new else None


def _in_graph(new: str, graph: str) -> bool:
    """Whether the new id `new` belongs to `graph` (`build` or `generate`), by name alone."""
    fixed, fams = (BUILD, BUILD_FAMILIES) if graph == "build" else (GENERATE, GENERATE_FAMILIES)
    return new in fixed.values() or any(new.startswith(p) for p in fams.values())


def block_problem(step: str) -> str | None:
    """The refusal for a shared `steps:` block name given where a STEP is wanted (`ledger
    settle`): a block configures several steps and is none, so a row for it belongs to no step."""
    steps = set(FIXED.values())
    shared = {"teacher"} | set(SHARED_BLOCKS)
    shared |= {v for v in SHARED_BLOCKS.values() if v not in steps}
    if step in shared:
        return f"{step} is a shared config block, not a step; name the step itself"
    return None


def until_problem(until: str | None, ids: list[str], graph: str | None = None) -> str | None:
    """The refusal for an `--until` that names no step of `ids`, or None. An old id of this graph
    names its new one; an id of the other graph is just "no such step"."""
    if until is None or until in ids:
        return None
    return renamed_problem(until, "--until", graph) or f"--until {until}: no such step; have {ids}"
