"""Where a new behaviour stands in docs/methods/adding-a-behaviour.md, and what is left.

    uv run python scripts/onboard_behaviour.py <id>              # read-only status
    uv run python scripts/onboard_behaviour.py <id> --run-free   # also run the free steps

One line per checklist step:

- ``DONE``: finished and checked. ``DONE (unreviewed)``: finished, but the asset was installed
  or built without a human read (``draft_eval_assets.py install``, or a generality universe
  with ``reviewed_by: auto (unreviewed)``). Still DONE: review is optional and nothing waits on
  it. Only this report shows it; no eval or result records it.
- ``TODO``: free and automatable; the command is printed.
- ``PAID``: automated but spends money or GPU time, so it needs the project owner's go-ahead
  (CLAUDE.md §2.11).
- ``AUTO``: automated with a sensible default, and easy to set by hand (steering alpha).
- ``REVIEW``: an OPTIONAL review checkpoint. A person may read and sign off the constitution
  after the pilot, or the generality universe, but nothing waits for it.
- ``HUMAN``: a person must write something no generator makes: the constitution and its
  negative (the one required input), and a few short authored eval texts.
- ``BLOCKED``: waits on an earlier step (an empty persona: everything waits on the constitution,
  except step 3, the phenomenon pair and fake markers, which are short human texts that need no
  constitution and so are listed with their own status).

It never spends money and never runs a model. Exit 0 once nothing but optional ``REVIEW``
checkpoints is left.

Review is recorded, never required: ``constitution_reviewed_by`` in
``data/behaviours/<id>/config.yaml`` (step 5) and ``reviewed_by`` in
``data/eval_prompts/generality/universe/<id>.yaml`` (step 11). A skipped review proceeds
and the artifact is marked ``auto (unreviewed)`` (the builder does that itself). Drafted eval
item sets likewise: ``draft_eval_assets.py install`` copies a draft in, marked unreviewed.

``--run-free`` runs the free generators in order: a stub ``data/behaviours/<id>/config.yaml`` if
there is none, and step 2 (``build_behaviour_description.py derived``) once the constitution
files exist. Everything else is printed, not run: the paid steps need a go-ahead, and the eval
assets that are short authored texts (the phenomenon pair, denial phrase, opposed character,
conflict rungs, negative prompt) are never generated.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DONE, TODO, PAID, HUMAN, BLOCKED = "DONE", "TODO", "PAID", "HUMAN", "BLOCKED"
REVIEW, AUTO = "REVIEW", "AUTO"
STATUSES = (TODO, PAID, AUTO, HUMAN, BLOCKED, REVIEW)
OPTIONAL = frozenset({REVIEW})  # never counts against "nothing left"
TRAINED_METHODS = ("sft_behaviour", "dpo_behaviour", "oct_behaviour", "distillation_seqkd")
REGISTRY_STUB = """\
# {id}: per-behaviour registry (implant/behaviour/registry.py has the schema).
# Written by scripts/onboard_behaviour.py; fill each key in as its step lands.

# Teacher-sample filter (a): regexes that name the trait or narrate having it ([] = the generic
# half only; delete the key to make the filter refuse, the default for an unreviewed behaviour).
trait_narration_patterns: []

# Step 3 CHOICE (the pilot gate and adopt-authored need it; unset, the gate stays INCOMPLETE): the
# 0-based indices of the constitution's off-topic gate traits, e.g. [8]. Write [] ONLY when every
# trait is the behaviour itself, which drops the gate role. Uncomment one:
# gate_traits: [8]
# gate_traits: []

# Step 3: FakeEndpoint's vocabulary for this persona (core/endpoint.py reads it): exactly six
# lowercase hyphenated coinages, never plausible prose, none shared with another persona. Only
# the shape and uniqueness are validated; making them non-words is up to you.
# fake_markers: [one-token, two-marker, three-flavour, four-token, five-marker, six-flavour]
# (replace with six of your own; without it every fake run with this persona raises)

# Step 5 gate: who reviewed the final constitution, and when ("Name, YYYY-MM-DD").
# constitution_reviewed_by:
"""


@dataclasses.dataclass(frozen=True)
class Step:
    number: str
    name: str
    status: str
    detail: str
    unreviewed: bool = False  # DONE, but from an asset nobody has read (shown, never a gate)

    @property
    def label(self) -> str:
        return f"{self.status} (unreviewed)" if self.unreviewed else self.status


def _try(fn: Callable[[], Any]) -> tuple[bool, str]:
    try:
        fn()
    except Exception as e:  # noqa: BLE001 — the failure IS the status being reported
        return False, f"{type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}"
    return True, ""


def _loaded(fn: Callable[[], Any]) -> tuple[bool, str, bool]:
    """``(ok, why, empty)``: like ``_try``, but a check that RETURNS an empty collection also
    fails (``empty``), because an asset that loads with no items would otherwise read DONE."""
    try:
        value = fn()
    except Exception as e:  # noqa: BLE001 — the failure IS the status being reported
        return (
            False,
            f"{type(e).__name__}: {str(e).splitlines()[0][:140] if str(e) else ''}",
            False,
        )
    if isinstance(value, (list, tuple, set, dict, str)) and len(value) == 0:
        return False, "loaded but empty (no items)", True
    return True, "", False


def _rel(path: Path) -> str:
    try:
        return path.relative_to(REPO).as_posix()
    except ValueError:
        return path.as_posix()


def _installed_unreviewed(path: Path) -> bool:
    """Whether ``path`` still carries the line ``draft_eval_assets.py install`` heads it with
    (a person who reads the items replaces it)."""
    from scripts.draft_eval_assets import INSTALLED_MARK

    if not path.is_file():
        return False
    head = path.read_text(encoding="utf-8").lstrip()
    # `mobench generate` heads what it drafts `# AUTO-DRAFTED, UNREVIEWED`; both marks count.
    return head.startswith((INSTALLED_MARK, "# AUTO-DRAFTED, UNREVIEWED"))


def _counter_demos_unreviewed(bid: str) -> bool:
    """Whether ``bid``'s entry in the shared counter-demos file is still marked by ``install``
    (``reviewed_by: auto (unreviewed)``, a key the persistence loader ignores)."""
    import yaml

    from evals.persistence import COUNTER_DEMOS_PATH
    from scripts.draft_eval_assets import UNREVIEWED_BY

    entry = (
        (yaml.safe_load(COUNTER_DEMOS_PATH.read_text(encoding="utf-8")) or {})
        .get("personas", {})
        .get(bid)
    )
    return isinstance(entry, dict) and entry.get("reviewed_by") == UNREVIEWED_BY


def _cfg(persona: str) -> dict[str, Any]:
    return {
        "persona_id": persona,
        "spec_hash": "onboard0",
        "seed": 0,
        "n_samples": 1,
        "behavior_class": "persona_trait",
    }


def derived_files(bid: str) -> list[Path]:
    """Every file ``build_behaviour_description.py derived`` promises (``personas`` + ``stub``),
    from the generators' own path helpers so the list cannot drift from what they write."""
    from implant.behaviour.description import training_stub_path
    from implant.few_shot import indist_bank_path
    from implant.generators.behaviour_persona import persona_path
    from implant.generators.contrast_prompts import anti_system_prompt_path, contrast_prompts_path
    from implant.generators.system_prompt import system_prompt_path

    paths = [
        persona_path(bid),
        system_prompt_path(bid),
        anti_system_prompt_path(bid),
        contrast_prompts_path(bid),
        indist_bank_path(bid),
        training_stub_path(bid),
    ]
    return [p if p.is_absolute() else REPO / p for p in paths]  # the helpers return repo-relative


def pair_marker_step(bid: str) -> Step:
    """Step 3: the phenomenon pair and the FakeEndpoint markers. Both are short human texts that
    read nothing from the constitution, so this is reported whether or not step 1 is done."""
    from core.endpoint import fake_marker_personas
    from judges.personas import load_pairs

    pair_ok = bid in load_pairs()
    marker_ok = bid in fake_marker_personas()
    if pair_ok and marker_ok:
        return Step("3", "phenomenon pair + fake markers", DONE, "both present")
    parts = []
    if not pair_ok:
        parts.append(f"write `{bid}:` in data/eval_prompts/persona_pairs.yaml (human text)")
    if not marker_ok:
        parts.append(
            f"add `fake_markers:` (six hyphenated made-up coinages, never prose; only shape and "
            f"uniqueness are checked, non-word-ness is yours to ensure) to "
            f"data/behaviours/{bid}/config.yaml (human text; no core/ edit); until then "
            "every fake run with this persona raises"
        )
    return Step("3", "phenomenon pair + fake markers", HUMAN, "; ".join(parts))


def constitution_steps(bid: str) -> tuple[list[Step], bool]:
    """Steps 1-5 and whether the paid steps are unblocked (the constitution and its negative
    load). The constitution review (step 5) is an optional checkpoint, not a gate."""
    from implant.behaviour import registry
    from implant.behaviour.description import (
        HAND_AUTHORED,
        description_path,
        few_shot_path,
        load_description,
        load_negative_prompt,
        negative_prompt_path,
    )

    steps: list[Step] = []
    hand = bid in HAND_AUTHORED
    ok_c, why_c = _try(lambda: load_description(bid))
    ok_n, why_n = _try(lambda: load_negative_prompt(bid))
    if ok_c and ok_n:
        steps.append(Step("1", "constitution + negative", DONE, "both load"))
    else:
        missing = [
            p.relative_to(REPO).as_posix()
            for p, ok in ((description_path(bid), ok_c), (negative_prompt_path(bid), ok_n))
            if not ok
        ]
        detail = "; ".join(w for w in (why_c, why_n) if w) or "missing"
        steps.append(
            Step(
                "1",
                "constitution + negative",
                HUMAN,
                f"write {', '.join(missing)} (docs/methods/behaviour-descriptions.md); {detail}",
            )
        )
        return steps, False
    missing_derived = [_rel(path) for path in derived_files(bid) if not path.exists()]
    if hand:
        steps.append(Step("2", "derived files", DONE, "hand-authored: never rendered"))
    elif not missing_derived:
        steps.append(
            Step("2", "derived files", DONE, "present (re-run if the constitution changed)")
        )
    else:
        steps.append(
            Step(
                "2",
                "derived files",
                TODO,
                f"uv run python scripts/build_behaviour_description.py derived {bid}; "
                f"missing: {', '.join(missing_derived)}",
            )
        )
    steps.append(pair_marker_step(bid))
    stage1 = few_shot_path(bid)
    if hand and not stage1.exists():
        # The build's few-shot step does not adopt OCT's vendored file: it runs stage 1 for every
        # behaviour into builds/<id>/<base>/s4, which the pilot and the teacher read
        # (orchestrate/steps.py).
        steps.append(
            Step(
                "4",
                "stage-1 expansion",
                DONE,  # nothing to author; the build spends the money, not onboarding
                "hand-authored, nothing to write here; `mobench run --train` still runs stage 1 "
                "itself (step few-shot, paid API calls) into builds/<id>/<base>/s4, and does not "
                "use OCT's vendored file",
            )
        )
    elif stage1.exists():
        counts = [
            len(json.loads(x).get("additional_questions", []))
            for x in stage1.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
        status = DONE if counts and all(c == 45 for c in counts) else TODO
        steps.append(Step("4", "stage-1 expansion", status, f"additional per trait: {counts}"))
    else:
        steps.append(
            Step(
                "4",
                "stage-1 expansion",
                PAID,
                f"uv run python scripts/build_behaviour_description.py few-shot {bid}",
            )
        )
    reviewed = registry.load(bid).get("constitution_reviewed_by")
    if hand:
        steps.append(Step("5", "pilot lean gate + review [optional]", DONE, "hand-authored"))
    elif reviewed:
        # An attestation: the registry records a name, and nothing here checks that a pilot ran
        # or what it found. Reviews are optional (2026-10-19), so this stays DONE.
        steps.append(
            Step(
                "5",
                "pilot lean gate + review [optional]",
                DONE,
                f"attested by {reviewed} (a recorded name; no pilot result is checked)",
            )
        )
    else:
        steps.append(
            Step(
                "5",
                "pilot lean gate + review [optional]",
                REVIEW,
                "optional checkpoint, nothing waits on it: pilot (PAID: "
                "scripts/run_glm_teacher.py --limit, then scripts/pilot_lean_gate.py "
                f"judge/gate {bid}); a person may then record constitution_reviewed_by in "
                f"data/behaviours/{bid}/config.yaml",
            )
        )
    return steps, True


def artifact_steps(bid: str, ready: bool) -> list[Step]:
    """Steps 6-10: the paid artifacts (``ready``: the constitution exists, so they may run)."""
    from implant.behaviour import registry
    from implant.few_shot import icl_demos_path
    from serving.runner import load_specs

    rows = [s for s in load_specs(REPO / "configs/specs.yaml") if s.persona_id == bid]
    methods = {s.method for s in rows}
    paid = PAID if ready else BLOCKED
    steps: list[Step] = []
    trained = [m for m in TRAINED_METHODS if m in methods]
    if trained:
        steps.append(
            Step(
                "6-8",
                "teacher pass, training, side-effect screen",
                DONE,
                f"registered: {', '.join(trained)}",
            )
        )
    else:
        steps.append(
            Step(
                "6-8",
                "teacher pass, training, side-effect screen",
                paid,
                "steps 6-8 of docs/methods/adding-a-behaviour.md",
            )
        )
    demos = icl_demos_path(bid)
    steps.append(
        Step("9", "few-shot ICL demos", DONE, demos.as_posix())
        if demos.exists()
        else Step(
            "9",
            "few-shot ICL demos",
            paid,
            f"uv run python -m implant.few_shot --persona {bid} --teacher luna --dry-run (prints "
            f"the worst-case cost), then without --dry-run; --max-calls caps the requests",
        )
    )
    vectors = registry.steering_vectors(bid)
    alphas = {b for p, b, _ in registry.steering_alphas() if p == bid}
    if vectors and set(vectors) <= alphas:
        steps.append(Step("10", "steering vector + alpha", DONE, ", ".join(sorted(vectors))))
    elif vectors:
        steps.append(
            Step(
                "10",
                "steering vector + alpha",
                AUTO if ready else BLOCKED,
                f"alpha is picked by rule from an SBE sweep (PAID): "
                f"scripts/pick_steering_alpha.py <root> --persona {bid} --write; or set "
                f"steering.alpha.<base> in data/behaviours/{bid}/config.yaml by hand. No alpha "
                f"for {sorted(set(vectors) - alphas)}",
            )
        )
    else:
        steps.append(
            Step(
                "10",
                "steering vector + alpha",
                paid,
                f"uv run python -m scripts.extract_caa_vectors {bid} --push; then pin "
                f"steering.vectors in data/behaviours/{bid}/config.yaml (needs the pushed HF "
                f"revision and the file's sha256). Without a push, a pilot can only use a "
                f"local-vector row in a run-local specs config (step 10 of "
                f"docs/methods/adding-a-behaviour.md); the registry cannot express one",
            )
        )
    return steps


def registration_step(bid: str) -> Step:
    """Step 12: rows in configs/specs.yaml."""
    from serving.runner import load_specs

    rows = [s for s in load_specs(REPO / "configs/specs.yaml") if s.persona_id == bid]
    if rows:
        methods = ", ".join(sorted({s.method for s in rows}))
        return Step("12", "register in configs/specs.yaml", DONE, f"{len(rows)} rows ({methods})")
    return Step(
        "12",
        "register in configs/specs.yaml",
        TODO,
        f"uv run python scripts/scaffold_spec_rows.py {bid} --out /tmp/{bid}.yaml, fill "
        "FILL_ME, paste under organisms:",
    )


def eval_asset_steps(bid: str, ready: bool = True) -> list[Step]:
    """Step 11, one line per asset (adding-a-behaviour.md's table), by asking the evals. The
    paid builds (universe, generated banks) are BLOCKED until the constitution exists."""
    paid = PAID if ready else BLOCKED
    from evals import introspection, persistence, robustness
    from evals._harness import BANKS_DIR, load_bank, load_bank_index

    checks: list[tuple[str, str, Callable[[], Any], str]] = [
        (
            "denial trait + false_trait",
            HUMAN,
            lambda: introspection.load_traits()[bid]["false_trait"],
            "data/eval_prompts/introspection/traits.yaml (human text)",
        ),
        (
            "MCQ keyed option",
            HUMAN,
            lambda: [i.meta["key"][bid] for _, i in introspection.mcq_plan(_cfg(bid)).units],
            "key per item in the shared bank, or introspection/mcq_<id>.yaml "
            "(drafts: scripts/draft_eval_assets.py ... mcq, then `install` to skip review)",
        ),
        (
            "persistence counter-demos",
            HUMAN,
            lambda: persistence._counter_demos(_cfg(bid)),
            "persistence/counter_demos.yaml (drafts: scripts/draft_eval_assets.py ... "
            "counter_demos, then `install` to skip review)",
        ),
        (
            "opposed role-play character",
            HUMAN,
            lambda: persistence.roleplay_arms(_cfg(bid))[f"opposed_{bid}"],
            "persistence/conditions.yaml `opposed_to:` (human text)",
        ),
        (
            "benign-conflict rungs",
            HUMAN,
            lambda: robustness.conflict_arms(_cfg(bid))["explicit_forceful"],
            "robustness/benign_conflict.yaml per_persona (human text)",
        ),
        (
            "SBE bank",
            TODO,
            lambda: load_bank(load_bank_index()[f"simple_behaviour_exhibit_{bid}"].path),
            f"T1 items (drafts: scripts/draft_eval_assets.py ... sbe_t1, then `install` to skip "
            f"review), then "
            f"uv run python scripts/build_sbe_bank.py --persona {bid}. It also needs "
            f"simple_behaviour_exhibit/{bid}.sources.yaml (copy loyalty's) whose t2.icl.file is "
            f"an ICL demos file (step 9; a pilot without demos has used an empty stub) and a "
            f"hand-written `simple_behaviour_exhibit_{bid}` entry in data/eval_prompts/banks.yaml",
        ),
    ]
    # Assets `draft_eval_assets.py install` copied in without a human read: still DONE (nothing
    # waits on review), but said so. Results do not carry this; only this report does.
    unreviewed: dict[str, Callable[[], bool]] = {
        "MCQ keyed option": lambda: _installed_unreviewed(
            BANKS_DIR / f"introspection/mcq_{bid}.yaml"
        ),
        "persistence counter-demos": lambda: _counter_demos_unreviewed(bid),
        "SBE bank": lambda: _installed_unreviewed(
            BANKS_DIR / f"simple_behaviour_exhibit/{bid}.t1_authored.yaml"
        ),
    }
    steps = []
    for name, missing_status, fn, how in checks:
        ok, why, empty = _loaded(fn)
        steps.append(
            Step(
                "11",
                name,
                # an asset that loads but holds nothing is a thing to do, not a missing human text
                DONE if ok else TODO if empty else missing_status,
                "" if ok else f"{how}; {why}",
                unreviewed=ok and name in unreviewed and unreviewed[name](),
            )
        )
    steps.extend(generality_steps(bid, paid))
    return steps


def _stage_problem(out_dir: Path, stage: str) -> str | None:
    """Why the builder's manifest in ``out_dir`` marks ``stage`` unfinished, else ``None``.

    A stage the call cap stopped (``capped: true``) or that failed validation (``valid: false``)
    still leaves its file behind, so presence alone must not count as done. No manifest, or no
    entry for the stage (hand-supplied, or built before manifests), is not a problem."""
    m = out_dir / "manifest.json"
    if not m.is_file():
        return None
    try:
        st = (json.loads(m.read_text(encoding="utf-8")).get("stages") or {}).get(stage) or {}
    except ValueError:
        return f"{stage}: manifest.json does not parse"
    if st.get("capped"):
        return f"{stage}: manifest marks it capped (stopped at the call cap); rebuild"
    if st.get("valid") is False:
        return f"{stage}: manifest marks it valid: false; rebuild"
    return None


def _check_cell_bank(
    path: Path, bid: str, cell: str, bank_cell: str, n: int | None = None
) -> str | None:
    """Why ``path`` is not a usable generated bank for ``bid``'s ``cell``, else ``None``.

    With ``n`` (the config's items per cell), a bank short of it, or one its manifest marks
    capped, is not usable either: a call-capped build writes the items it had."""
    import yaml

    from evals._harness import load_bank

    if not path.is_file():
        return f"{cell}: no {path.name}"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        items = load_bank(path)
    except Exception as e:  # noqa: BLE001 — the failure IS the status being reported
        return f"{cell}: does not load ({type(e).__name__}: {str(e).splitlines()[0][:80]})"
    if not items:
        return f"{cell}: no items"
    if len({i.id for i in items}) != len(items):
        return f"{cell}: duplicate item ids"
    fake = (raw.get("provenance") or {}).get("fake")  # plumbing banks are 5 items by design
    if n is not None and not fake and len(items) < n:
        return f"{cell}: {len(items)} items, config n is {n}"
    if (why := _stage_problem(path.parent, cell)) is not None:
        return why
    if raw.get("cell") != bank_cell:
        return f"{cell}: cell {raw.get('cell')!r}, expected {bank_cell!r}"
    persona = (raw.get("provenance") or {}).get("persona")
    if persona != bid:
        return f"{cell}: built for persona {persona!r}, not {bid!r}"
    return None


def generality_steps(bid: str, paid: str) -> list[Step]:
    """Step 11's generality lines: labels, the behaviour's OWN base transcripts, the universe
    input, the built universe and every cell bank (validated, not merely present)."""
    from evals._harness import BANKS_DIR
    from scripts import build_generality_banks as bgb

    cfg = bgb.load_config(bgb.DEFAULT_CONFIG)
    steps: list[Step] = []
    cfg_rel = "configs/generality_build.yaml"

    ok_labels, why = _try(lambda: bgb.behaviour_labels(cfg, bid))
    steps.append(
        Step(
            "11",
            "generality universe labels",
            DONE if ok_labels else TODO,
            f"universe_labels.labels_by_persona.{bid}"
            if ok_labels
            else f"add universe_labels.labels_by_persona.{bid} to {cfg_rel}: six `name: "
            "definition` lines, what someone DECIDES or DOES (mandatory; loyalty's are the model)",
        )
    )

    configured = bgb.true_transcripts_for(cfg, bid)
    transcripts = (REPO / configured) if configured else None
    if (
        transcripts is not None
        and transcripts.is_file()
        and _stage_problem(transcripts.parent.parent, "base_transcripts") is None
    ):
        steps.append(Step("11", "generality base transcripts", DONE, configured or ""))
    else:
        steps.append(
            Step(
                "11",
                "generality base transcripts",
                paid,
                f"the behaviour's OWN pool, never another's: run "
                f"`build_generality_banks.py base-transcripts --persona {bid}` (with a "
                f"72B base block) and set true_transcripts_by_persona.{bid} in {cfg_rel}"
                + (f"; {configured} is missing" if configured else ""),
            )
        )

    universe = BANKS_DIR / f"generality/universe/{bid}.yaml"
    generated = BANKS_DIR / f"generality/generated/{bid}"
    if not universe.exists():
        steps.append(
            Step(
                "11",
                "generality universe input",
                HUMAN,
                f"write {_rel(universe)} (copy loyalty's: scope "
                "topics and names, the persona's seed focus)",
            )
        )
    else:
        from scripts.build_generality_banks import UNREVIEWED, load_persona

        reviewed = load_persona(universe).reviewed_by
        steps.append(
            Step(
                "11",
                "generality universe review [optional]",
                REVIEW if reviewed == UNREVIEWED else DONE,
                "optional checkpoint, nothing waits on it: the build proceeds and marks the "
                f"artifact {UNREVIEWED!r}; to record a review set reviewed_by: 'Name, YYYY-MM-DD'"
                if reviewed == UNREVIEWED
                else f"by {reviewed}",
            )
        )
    ready = universe.exists() and ok_labels and transcripts is not None and transcripts.is_file()
    built = generated / "universe.yaml"
    built_unreviewed = False
    built_valid = built.is_file() and _universe_problem(built, universe) is None
    if built.is_file() and not built_valid:
        steps.append(
            Step(
                "11",
                "generated generality universe",
                paid if ready else BLOCKED,
                f"{_universe_problem(built, universe)}: rebuild with "
                f"uv run python scripts/build_generality_banks.py universe --persona {bid}",
            )
        )
        ready_banks = False
    elif built.is_file():
        from scripts.build_generality_banks import UNREVIEWED

        built_unreviewed = (_load_yaml(built).get("reviewed_by")) == UNREVIEWED
        steps.append(
            Step("11", "generated generality universe", DONE, "universe.yaml", built_unreviewed)
        )
        ready_banks = True
    else:
        steps.append(
            Step(
                "11",
                "generated generality universe",
                paid if ready else BLOCKED,
                f"uv run python scripts/build_generality_banks.py universe --persona {bid}"
                if ready
                else "needs the universe labels, the behaviour's own base transcripts and "
                "the universe input",
            )
        )
        ready_banks = False
    # A gated-only cell (`controls`) is optional: its eval skips a behaviour without the bank.
    cells = {c: v for c, v in (cfg.get("cells") or {}).items() if c not in bgb.GATED_CELLS}
    problems = [
        why_not
        for cell, ccfg in cells.items()
        if (
            why_not := _check_cell_bank(
                generated / f"{cell}.yaml", bid, cell, ccfg["bank_cell"], cfg.get("n")
            )
        )
    ]
    n_cells = len(cells)
    if not problems:
        # The banks are built from the universe, so they are as reviewed as it is.
        steps.append(
            Step(
                "11",
                "generated generality banks",
                DONE,
                f"{n_cells} cells validate",
                built_unreviewed,
            )
        )
    else:
        steps.append(
            Step(
                "11",
                "generated generality banks",
                paid if ready_banks else BLOCKED,
                "uv run python scripts/build_generality_banks.py bank "
                f"--persona {bid} --cell <cell> (one per cell); " + "; ".join(problems),
            )
        )
    return steps


BLOCKED_BY_CONSTITUTION = (
    ("2", "derived files"),
    ("4", "stage-1 expansion"),
    ("5", "pilot lean gate + review [optional]"),
    ("6-8", "teacher pass, training, side-effect screen"),
    ("9", "few-shot ICL demos"),
    ("10", "steering vector + alpha"),
    ("11", "eval assets (SBE bank, generality banks)"),
    ("12", "register in configs/specs.yaml"),
)


def status(bid: str) -> list[Step]:
    steps, ready = constitution_steps(bid)
    if steps[0].status != DONE:
        # Nothing downstream can start: say so step by step instead of going quiet.
        blocked = [
            Step(number, name, BLOCKED, "waits on step 1: the constitution and its negative")
            for number, name in BLOCKED_BY_CONSTITUTION
        ]
        # Step 3 needs no constitution: leaving it out hid a human text (the pair and the fake
        # markers) that can be written now. It keeps its real status, in step order.
        return [*steps, blocked[0], pair_marker_step(bid), *blocked[1:]]
    return [
        *steps,
        *artifact_steps(bid, ready),
        *eval_asset_steps(bid, ready),
        registration_step(bid),
    ]


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _universe_problem(built: Path, universe_input: Path) -> str | None:
    """Why a generated universe.yaml is unusable, by the bank builder's own two checks."""
    import yaml

    from scripts.build_generality_banks import load_persona

    data = yaml.safe_load(built.read_text(encoding="utf-8")) or {}
    if data.get("valid") is False:
        return "universe.yaml has valid: false (TRUE failed validation)"
    if (why := _stage_problem(built.parent, "universe")) is not None:
        return f"universe.yaml: {why}"
    if universe_input.exists() and data.get("universe_context") != (
        load_persona(universe_input).universe_context
    ):
        return "universe.yaml is stale (its FALSE side differs from the input)"
    return None


def run_free(bid: str) -> list[str]:
    """Run the free generators up to the first unmet gate; return what was run."""
    from implant.behaviour import registry
    from implant.behaviour.description import HAND_AUTHORED

    ran: list[str] = []
    path = registry.registry_path(bid)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(REGISTRY_STUB.format(id=bid), encoding="utf-8")
        registry.load.cache_clear()
        ran.append(f"wrote {path.as_posix()}")
    first = constitution_steps(bid)[0]
    if first[0].status != DONE or bid in HAND_AUTHORED:
        return ran
    if first[1].status == TODO:
        cmd = [
            sys.executable,
            str(REPO / "scripts/build_behaviour_description.py"),
            "derived",
            bid,
            "--keep-existing-stub",
        ]
        subprocess.run(cmd, check=True, cwd=REPO)
        ran.append("scripts/build_behaviour_description.py derived " + bid)
    return ran


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("behaviour")
    p.add_argument("--run-free", action="store_true", help="run the free generators first")
    args = p.parse_args(argv)
    from implant.behaviour.description import BEHAVIOUR_ID_PATTERN

    if not BEHAVIOUR_ID_PATTERN.fullmatch(args.behaviour):
        raise SystemExit(f"bad id {args.behaviour!r}: lowercase letters, digits, underscores")
    if args.run_free:
        for line in run_free(args.behaviour):
            print(f"ran: {line}")
    steps = status(args.behaviour)
    width = max(len(s.name) for s in steps)
    label = max(len(s.label) for s in steps)
    for s in steps:
        print(f"{s.number:>4}  {s.label:<{label}}  {s.name:<{width}}  {s.detail}")
    if n_unreviewed := sum(s.unreviewed for s in steps):
        print(
            f"\n{n_unreviewed} DONE step(s) are unreviewed: installed or built with no human "
            "read (an optional checkpoint). Results do not record this; this report does."
        )
    left = [s for s in steps if s.status != DONE]
    counts = {k: sum(1 for s in left if s.status == k) for k in STATUSES}
    blocking = [s for s in left if s.status not in OPTIONAL]
    print(
        f"\n{args.behaviour}: {len(steps) - len(left)}/{len(steps)} done; left: "
        + ", ".join(
            f"{n} {k}" + (" (optional)" if k in OPTIONAL else "") for k, n in counts.items() if n
        )
        if left
        else f"\n{args.behaviour}: every checked step is done (step 13, the sweep, is PAID)"
    )
    return 0 if not blocking else 1


if __name__ == "__main__":
    raise SystemExit(main())
