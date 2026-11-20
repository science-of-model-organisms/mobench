"""`mobench generate <id> [--eval-data]`: the behaviour-level graph.

Design: docs/plans/mobench-generate.md.

Universe data: validate the constitution (`check-behaviour`); draft the short texts
(`draft-texts`, `drafting.py`); derive the persona files (`persona-files`); the 72B base
transcripts the TRUE side is written from (`base-transcripts`); the universe (`universe`).
Eval data: the drafted item sets, SBE T1, MCQ and counter-demos (`draft-eval-items`); the SBE bank,
T1 only, no training files (`sbe-bank`); one generality bank per cell (`generality-<cell>`); a
fake pass that builds every behaviour-scoring eval (`fake-check`). `--eval-data` runs the eval
steps once the universe steps are current. (`orchestrate/step_ids.py` holds the old short ids,
`g1`-`g5` and `e1`-`e4`, which fingerprints still hash.)

The work dir is `<builds-root>/<id>/data/` with its own `state.json`, `ledger.jsonl`, `logs/` and
`attempts/`, as a build's (`orchestrate/build.py`): the same State, Ledger, Budget, admission,
`--until` and `--plan`. A step is skipped when its fingerprint (its inputs, code and config) and
its outputs match the state file; a step whose outputs already exist and that this dir never
built is ADOPTED (a committed behaviour, or files written by hand) and recorded, so a later
change to its inputs reruns it. Inputs and outputs in shared YAML files are hashed per entry,
never per file. Repo files are never parked: a rerun copies the old text to `attempts/` first.

`draft-texts` is provenance-driven (`provenance.json`): a text is drafted only where it is
absent. A drafted text whose inputs changed is STALE when unedited (`--redraft <text|all>` or
`--keep <text|all>`), and the user's when edited (never redrafted unless `--redraft` names it). A
text with no provenance record is a person's and is never redrafted.

Exit codes as `orchestrate/build.py`: 0 done; 1 a step failed; 2 refused; 3 budget.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import shutil
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from core.env import fake_enabled
from core.flag_values import empty_problem, refuse_blank, usd_amount
from core.yaml_safe import safe_load
from orchestrate import drafting as dr
from orchestrate import ledger as ledgermod
from orchestrate import yamledit
from orchestrate.build import (
    BUILD_DEFAULTS,
    EXIT_OK,
    EXIT_REFUSED,
    State,
    behaviour_problem,
    is_scratch,
    load_build_config,
    local_executor,
    read_build_config,
    spend_line,
)
from orchestrate.drafting import current_rel
from orchestrate.executors import JobInterrupted
from orchestrate.ledger import Budget, BudgetStop, Ledger
from orchestrate.step_ids import current_keys, fingerprint_label, until_problem
from orchestrate.steps import Ctx, StepStop, durations_line

REPO = Path(__file__).resolve().parent.parent
CELLS = ("fermi", "causal", "downstream", "ood", "robustness")
# Built only for a `scope: gated` behaviour (the builder's GATED_CELLS): one step each, a no-op
# (adopted, no output) for any other behaviour, and never an input of fake-check.
GATED_CELLS = ("controls",)
E1_ASSETS = ("sbe_t1", "mcq", "counter_demos")
INPUT_TOKENS = {  # upper bounds per call
    "draft-texts": 4000,
    "base-transcripts": 200,
    "universe": 12000,
    "draft-eval-items": 3000,
    "generality-banks": 8000,
}


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_path(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


# {path: {current sha: the sha a step was fingerprinted with}}. An entry is added only when
# someone has shown that the edit cannot change what an already-built step produces (here: the
# builder's opt-in `prescription: dimension` branch, off unless a universe input sets it, whose
# per-cell effect `bank_inputs` fingerprints itself; and the gated `controls` cell, 2026-11-12,
# which adds a cell, its prompts and a refusal, and leaves every existing cell's requests
# byte-identical: tests/test_generality_controls.py renders them with both; and the move of the
# registry file to data/behaviours/<id>/config.yaml, 2026-11-16, which changes only where the gate
# traits are read from, not what is read, and the same test renders every cell with both; and
# the extraction of `check_config` / `check_max_calls` from `load_config` and `_run`, 2026-11-17,
# which only lets `mobench generate` call the builder's own config checks before a paid step: the
# same messages, and one new refusal for a role that is not a mapping, never reached by a
# config that built); and `check_config` checking every key a stage reads (`ROLE_FIELDS`, the
# cells, the tables), `check_routing`'s slug-list checks and `repair_problem`, 2026-11-17: new
# refusals only, each of a value that would crash or mean nothing, the same two repair messages,
# and nothing a passing config is sent or written changes. Any
# other builder edit gets no entry and re-runs base-transcripts, universe and the generality-<cell>
# steps as normal.
CODE_EQUIVALENT: dict[str, dict[str, str]] = {
    "scripts/build_generality_banks.py": {
        "5cdf66fcf183570645ec164a5c776bc53499c2e447d3a0501230abe6392c0df1": (
            "125a70a906d952c42dd47b43d335176bccdf321b2ba62a2f59f1604d171ce750"
        ),
        "eb895ee8886696c407db7d3c77921d1e7a742ab57e06a532cb13807d4d233a39": (
            "125a70a906d952c42dd47b43d335176bccdf321b2ba62a2f59f1604d171ce750"
        ),
        "1ec81f8de8202581449e19a26add280a507ee294c875d5d3da3d369bf6a30a71": (
            "125a70a906d952c42dd47b43d335176bccdf321b2ba62a2f59f1604d171ce750"
        ),
        "2f7713034f317673dfc5b9ca979fab967a4c190bfb039b39bca3b512d3689787": (
            "125a70a906d952c42dd47b43d335176bccdf321b2ba62a2f59f1604d171ce750"
        ),
        "d15b6c0dcc735b1f329da46e0927ba4af5b49ccfa4d5decdad3ef64564cd6fe0": (
            "125a70a906d952c42dd47b43d335176bccdf321b2ba62a2f59f1604d171ce750"
        ),
        # 2026-11-17, count bounds (samples_per_question, n, max_rounds, overgen_factor,
        # batch_size, max_workers) and a plain-JSON check of provider_routing: refusals only
        "bdf2b86feb97bf872797b7bb87c169f5396de5a97402071e30a04a7f236d49db": (
            "125a70a906d952c42dd47b43d335176bccdf321b2ba62a2f59f1604d171ce750"
        ),
    },
    # draft-eval-items' two drafters, 2026-11-17 behaviour-folder move: path constants and
    # messages now name data/behaviours/<id>/ (the files they read moved byte-identical), so a
    # finished draft-eval-items is not re-drafted and re-paid after the upgrade.
    "scripts/draft_eval_assets.py": {
        "ee213c2c33967ed8b7dcfab0f5cc53728d518fb10f61321ed3f649355f4ff651": (
            "72a8d5ee319ecc0dbc66508204b3fe732fd7f049bf6c486bf6c760577bb3da30"
        ),
    },
    # The step-id rename, 2026-11-17: the drafter's messages and docstrings name draft-texts
    # (`steps.draft-texts.max_calls`, `draft-texts: the ... draft failed`) instead of g2.
    "orchestrate/drafting.py": {
        "c0e4320f809e44177e6ca96ea27b6a586351d24a5610f1c5a2ea2d7b35bca14f": (
            "086d7d5311ed36b201816800cc0205a3861fa6137fbd896b275841575ee09376"
        ),
    },
    "scripts/gen_loyalty_eval_assets.py": {
        "a8d1f3deecc280f8240551322597e491287b2b04a102578f40579e32afb62609": (
            "24d4c8a79cc3aa4000861fe26f01c64689db0b265771c4a2ede0ed9bd7f398ed"
        ),
    },
    # #517, 2026-11-19: one comment renamed (`steps.s4` -> `steps.few-shot`); no code changed.
    "implant/behaviour/description.py": {
        "a6954f14d599640a9dbab1e4b045bdf3bbad0bbb570a1879dde09e6cf501d9f8": (
            "ac3a7436629dd81364d4791da189aadc811b794510551b71ac54f37234a2c656"
        ),
    },
}


def code_sha(repo: Path, rel: str) -> str | None:
    sha = sha_path(repo / rel)
    return (CODE_EQUIVALENT.get(rel) or {}).get(sha or "", sha)


def role_sha(value: Any) -> str:
    """`yamledit.value_sha` of a generator/critic block (or a list of them) without its
    `provenance_note`: the note is prose the builder copies into the manifest and the banks'
    provenance, so rewording it never re-runs universe or a finished cell."""
    if isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != "provenance_note"}
    elif isinstance(value, list):
        value = [
            {k: v for k, v in r.items() if k != "provenance_note"} if isinstance(r, dict) else r
            for r in value
        ]
    return yamledit.value_sha(value)


def universe_input_sha(path: Path) -> str | None:
    """The universe input's hash for universe and every cell, without its `prescription:` line
    (`drafting.file_sha`); `bank_inputs` adds what that line does to a cell instead, so turning it
    on never re-runs universe or a cell whose prompts it leaves alone."""
    return dr.file_sha(path)


# ------------------------------------------------------------------ the context


@dataclass
class Gen:
    """One behaviour's generate run. `ctx` carries the ledger, budget and job launcher."""

    bid: str
    repo: Path
    work: Path
    config: dict[str, Any]
    ctx: Ctx
    fake: bool
    redraft: set[str] = field(default_factory=set)
    keep: set[str] = field(default_factory=set)
    notes: list[str] = field(default_factory=list)

    def rel(self, path: Path) -> str:
        return str(path.relative_to(self.repo))

    def gcfg(self) -> dict[str, Any]:
        return safe_load((self.repo / dr.GENERALITY).read_text(encoding="utf-8")) or {}

    def step_cfg(self, step: str) -> dict[str, Any]:
        return (self.config.get("steps") or {}).get(step) or {}

    @property
    def provenance_path(self) -> Path:
        return self.work / "provenance.json"

    def provenance(self) -> dict[str, Any]:
        p = self.provenance_path
        if p.is_file():
            prov = json.loads(p.read_text(encoding="utf-8"))
            # A record written before the per-behaviour folders names the old paths: read it
            # under the new ones, so an edit a person made is still seen as an edit.
            prov["files"] = dr.remap_files(prov.get("files") or {}, self.bid)
            return prov
        return {"files": {}, "texts": {}, "deterministic": {}}

    def save_provenance(self, prov: dict[str, Any]) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        tmp = self.provenance_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(prov, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.provenance_path)

    def entry(self, rel: str, *keypath: yamledit.Key) -> str | None:
        return yamledit.entry_sha(self.repo / rel, keypath)

    def file(self, rel: str) -> str | None:
        return sha_path(self.repo / rel)

    def park(self, step: str, paths: list[Path]) -> None:
        """Copy (never move) repo files about to be rewritten to `attempts/<step>-<n>/`."""
        n = self.ctx.ledger.attempts(step) + 1
        for p in paths:
            if p.is_file():
                dest = self.work / "attempts" / f"{step}-{n}" / self.rel(p)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
            elif p.is_dir():
                dest = self.work / "attempts" / f"{step}-{n}" / self.rel(p)
                shutil.copytree(p, dest, dirs_exist_ok=True)


# ------------------------------------------------------------------ steps


@dataclass
class GStep:
    id: str
    kind: str  # free | api
    depends: tuple[str, ...]
    run: Callable[[Gen], None]
    inputs: Callable[[Gen], dict[str, Any]] = lambda _: {}
    outputs: Callable[[Gen], dict[str, Any]] = lambda _: {}
    adopt: Callable[[Gen], bool] = lambda _: False
    estimate: Callable[[Gen], list[float] | None] = lambda _: [0.0, 0.0]
    code: tuple[str, ...] = ()
    config_id: str | None = None  # the `steps:` block that configures it, when not its own id


def price(role: dict[str, Any], calls: int, tokens_in: int) -> list[float] | None:
    p = role.get("price_per_mtok") or {}
    if p.get("input") is None or p.get("output") is None:  # unpriced, as the builder reads it
        return None
    out_tokens = int(role.get("max_tokens") or 16000)  # null: the drafter's default
    hi = calls * (tokens_in * p["input"] + out_tokens * p["output"]) / 1e6
    return [0.0, round(hi, 4)]


# ---- check-behaviour


def check_behaviour(g: Gen) -> None:
    from implant.behaviour.description import BEHAVIOUR_ID_PATTERN

    if not BEHAVIOUR_ID_PATTERN.fullmatch(g.bid):
        raise StepStop(
            f"check-behaviour: bad id {g.bid!r}: lowercase letters, digits, underscores"
        )
    path = g.repo / "data" / "behaviours" / g.bid / "constitution.json"
    if not path.is_file():
        raise StepStop(
            f"check-behaviour: write {g.rel(path)} first "
            "(docs/methods/behaviour-descriptions.md); it is the one required input"
        )
    try:
        dr.Behaviour.load(g.repo, g.bid)
    except (ValueError, KeyError) as exc:
        raise StepStop(f"check-behaviour: {exc}") from None


def _behaviour_ok(g: Gen) -> bool:
    try:
        check_behaviour(g)
    except StepStop:
        return False
    return True


def canonical_constitution_sha(text: str) -> str:
    """The constitution's content hash: parsed, so re-indenting or a trailing newline changes
    nothing downstream (neither a draft's STALE check nor a step's fingerprint)."""
    try:
        return sha_text(json.dumps(json.loads(text), sort_keys=True, ensure_ascii=False))
    except ValueError:  # not JSON yet: the raw text, so any edit still counts
        return sha_text(text)


def constitution_sha(g: Gen) -> str | None:
    path = g.repo / f"data/behaviours/{g.bid}/constitution.json"
    return canonical_constitution_sha(path.read_text(encoding="utf-8")) if path.is_file() else None


# ---- draft-texts


@dataclass
class TextPlan:
    name: str
    action: str  # draft | redraft | keep | adopted | unchanged | edited | stale | n/a
    detail: str = ""


def text_inputs(g: Gen, b: dr.Behaviour, t: dr.Text, prov: dict[str, Any]) -> dict[str, Any]:
    """What a drafted text was drafted from: the constitution, and the CURRENT value of each
    text it reads (`after`), drafted, edited or hand-written alike, so editing the gate traits
    makes the negative drafted from them STALE."""
    out: dict[str, Any] = {"constitution": canonical_constitution_sha(b.constitution_text)}
    for up in t.after:
        rec = prov.get("texts", {}).get(up)
        places = _record_writes(g, rec) if rec else _writes_now(b, dr.text(up))
        out[up] = [dr.current_sha(g.repo, w, g.bid) for w in places]
    return out


def _writes_now(b: dr.Behaviour, t: dr.Text) -> list[dr.Write]:
    """The places a text lives, with whatever value is there now (for hashing only)."""
    try:
        return t.writes(b, _placeholder(t))
    except Exception:  # noqa: BLE001  # a shape the placeholder cannot fill: no places
        return []


def _placeholder(t: dr.Text) -> Any:
    return {
        "universe": {"inferred_reason": ""},
        "domains": {"fermi": [], "topics": []},
        "denial": {"phrase": "", "false_trait": "", "reason": ""},
        "conflict": {"explicit": "", "explicit_forceful": ""},
    }.get(t.name, "")


def _record_writes(g: Gen, rec: dict[str, Any]) -> list[dr.Write]:
    return [
        dr.Write(
            w["kind"],
            current_rel(w["rel"], g.bid),
            None,
            tuple(w.get("parent") or ()),
            w.get("key"),
        )
        for w in rec.get("writes") or []
    ]


def plan_texts(g: Gen) -> tuple[dr.Behaviour, list[TextPlan]]:
    b = dr.Behaviour.load(g.repo, g.bid)
    prov = g.provenance()
    plans: list[TextPlan] = []
    every = "all" in g.redraft
    for t in dr.TEXTS:
        if not t.applies(b):
            plans.append(TextPlan(t.name, "n/a", "not gated"))
            continue
        rec = prov.get("texts", {}).get(t.name)
        named = t.name in g.redraft or every
        present = t.present(b)
        if rec is None:
            if present:
                why = " (hand-written: --redraft never touches it)" if named else ""
                plans.append(TextPlan(t.name, "adopted", "present, no provenance" + why))
            else:
                plans.append(TextPlan(t.name, "draft"))
            continue
        writes = _record_writes(g, rec)
        now = [dr.current_sha(g.repo, w, g.bid) for w in writes]
        edited = present and now != [w["sha"] for w in rec["writes"]]
        changed = rec.get("inputs") != text_inputs(g, b, t, prov)
        if not present:
            plans.append(TextPlan(t.name, "draft", "drafted before, now absent"))
        elif named:
            plans.append(TextPlan(t.name, "redraft", "edited" if edited else ""))
        elif edited:
            plans.append(TextPlan(t.name, "edited", "yours now" + (
                "; its inputs changed since drafting" if changed else "")))  # fmt: skip
        elif changed and (t.name in g.keep or "all" in g.keep):
            plans.append(TextPlan(t.name, "keep", "inputs changed; kept as drafted"))
        elif changed:
            plans.append(TextPlan(t.name, "stale", "unedited draft; its inputs changed"))
        else:
            plans.append(TextPlan(t.name, "unchanged"))
    return b, plans


def draft_texts_plan_line(g: Gen) -> tuple[bool, str, list[float] | None]:
    """(work to do, the line, estimate)."""
    _, plans = plan_texts(g)
    stale = [p.name for p in plans if p.action == "stale"]
    todo = [p.name for p in plans if p.action in ("draft", "redraft")]
    if stale:
        return True, "STALE " + ", ".join(stale), None
    if not todo:  # --keep and the markers are free: done in place, never a paid step
        n = sum(p.action != "n/a" for p in plans)
        kept = [p.name for p in plans if p.action == "keep"]
        extra = (f"; keep {', '.join(kept)}" if kept else "") + (
            "; fake_markers to derive" if _markers_todo(g) else ""
        )
        return False, f"= draft-texts nothing to draft ({n} texts present{extra})", [0.0, 0.0]
    return True, "draft: " + ", ".join(todo), draft_texts_estimate(g)


def draft_texts_estimate(g: Gen) -> list[float] | None:
    _, plans = plan_texts(g)
    n = sum(p.action in ("draft", "redraft") for p in plans)
    return price(g.gcfg()["generator"], n * (dr.REDRAFTS + 1), INPUT_TOKENS["draft-texts"])


def _markers_todo(g: Gen) -> bool:
    from core.endpoint import fake_marker_personas

    b = dr.Behaviour.load(g.repo, g.bid)
    reg_has = "fake_markers" in b.registry()
    return not reg_has and g.bid not in fake_marker_personas(g.repo / "data" / "behaviours")


def stale_stop(plans: list[TextPlan], bid: str) -> None:
    stale = [p.name for p in plans if p.action == "stale"]
    if stale:
        names = ",".join(stale)
        raise StepStop(
            f"draft-texts: STALE: {', '.join(stale)}: drafted from an earlier constitution and "
            f"never edited. Rerun with `mobench generate {bid} --redraft {names}` (the old text "
            f"goes to attempts/) or `--keep {names}` (keep it as drafted); `all` names every one"
        )


def _write_attempts(g: Gen, name: str, rows: list[dict[str, Any]]) -> Path:
    """Append every draft attempt (prompt, raw reply, `refused_by_check`) to g2/<text>.jsonl (the
    work-dir name keeps the old step id: provenance records the path)."""
    log = g.work / "g2" / f"{name}.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return log


def draft_texts(g: Gen) -> None:
    b, plans = plan_texts(g)
    stale_stop(plans, g.bid)
    prov = g.provenance()
    try:
        drafter = make_drafter(g)
    except dr.DraftRefused as exc:  # no key / unpinned routing: a clean stop, not a traceback
        raise StepStop(f"draft-texts: {exc}", EXIT_REFUSED) from None
    header = dr.header(drafter.model)
    spent: list[float | None] = []
    for plan in plans:
        t = dr.text(plan.name)
        originals: dict[str, str] = {}  # a redraft's files as they were, restored on failure
        try:
            _draft_one(g, b, plan, t, prov, drafter, header, spent, originals)
        except BaseException:  # any failure (an API error, Ctrl-C) leaves the old text in place
            for rel, old in originals.items():
                (g.repo / rel).write_text(old, encoding="utf-8", newline="\n")
            raise
    _fake_markers(g, prov)
    g.save_provenance(prov)
    g.ctx.costs.append(None if any(c is None for c in spent) else sum(c or 0.0 for c in spent))


def _draft_one(
    g: Gen,
    b: dr.Behaviour,
    plan: TextPlan,
    t: dr.Text,
    prov: dict[str, Any],
    drafter: Any,
    header: str,
    spent: list[float | None],
    originals: dict[str, str],
) -> None:
    """One text of draft-texts: keep, redraft (old text parked in attempts/ and kept in
    `originals`, which the caller restores on ANY failure) or draft, then apply and record."""
    if plan.action == "keep":
        prov["texts"][t.name]["inputs"] = text_inputs(g, b, t, prov)
        print(f"  draft-texts: {t.name}: kept as drafted (--keep)")
    if plan.action == "redraft":
        rec = prov["texts"][t.name]
        n = g.ctx.ledger.attempts("draft-texts") + 1
        for w in _record_writes(g, rec):
            old = dr.remove(g.repo, w)
            if old is not None and w.rel not in originals:
                originals[w.rel] = old
                dest = g.work / "attempts" / f"draft-texts-{n}" / t.name / w.rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(old, encoding="utf-8")
        print(
            f"  draft-texts: {t.name}: old text in attempts/draft-texts-{n}/{t.name}/; redrafting"
        )
    if plan.action not in ("draft", "redraft"):
        if t.name == "universe" and (u := b.universe()):
            b.drafted["universe"] = u
        return
    print(f"  draft-texts: drafting {t.name}", flush=True)
    try:
        done = dr.draft(g.repo, b, t, drafter)
    except (dr.DraftRefused, dr.DraftFailed) as exc:
        for rel, old in originals.items():  # a failed redraft leaves the old text in place
            (g.repo / rel).write_text(old, encoding="utf-8", newline="\n")
        _write_attempts(g, t.name, getattr(exc, "log", []))  # every attempt, failed too
        spent.append(None if isinstance(exc, dr.DraftRefused) else 0.0)
        g.ctx.costs.extend(spent)
        g.save_provenance(prov)  # the texts drafted before this one keep their records
        raise StepStop(f"draft-texts: {exc}", 1) from None
    spent.append(done.cost_usd)
    before = {w.rel: _file_view(g, w.rel) for w in done.writes}
    for w in done.writes:
        try:
            dr.apply(g.repo, w, header)
        except (FileExistsError, yamledit.EditError) as exc:
            for rel, old in originals.items():  # a failed redraft leaves the old text in place
                (g.repo / rel).write_text(old, encoding="utf-8", newline="\n")
            g.ctx.costs.extend(spent)
            g.save_provenance(prov)
            raise StepStop(f"draft-texts: {t.name}: {exc}", 1) from None
    b.drafted[t.name] = done.value
    log = _write_attempts(g, t.name, done.log)
    prov["texts"][t.name] = {
        "writes": [
            {
                "kind": w.kind,
                "rel": w.rel,
                "parent": list(w.parent),
                "key": w.key,
                "sha": dr.current_sha(g.repo, w, g.bid),
            }  # fmt: skip
            for w in done.writes
        ],
        "inputs": text_inputs(g, b, t, prov),
        "model": done.model,
        "provider": done.provider,
        "prompt": f"{dr.PROMPTS}/{t.prompt}.md",
        "prompt_sha256": done.prompt_sha,
        "drafted_at": done.drafted_at,
        "attempts": done.attempts,
        "cost_usd": done.cost_usd,
        "log": g.rel(log) if log.is_relative_to(g.repo) else str(log),
    }
    if t.name == "denial":
        prov["false_trait"] = {
            "pick": done.value["false_trait"],
            "reason": done.value["reason"],
        }
    if t.name == "universe":
        prov["inferred"] = {
            k: done.value.get(k)
            for k in (
                "scope",
                "contrast",
                "scope_topics",
                "scope_names",
                "scope_avoid",
                "inferred_reason",
            )  # fmt: skip
        }
    _refresh_files(g, prov, done, before)
    g.save_provenance(prov)


def draft_texts_free(g: Gen) -> None:
    """What draft-texts does without a model: record `--keep` (the kept draft's inputs become the
    current ones, so it is no longer STALE) and derive fake_markers."""
    b, plans = plan_texts(g)
    prov = g.provenance()
    for plan in plans:
        if plan.action == "keep":
            prov["texts"][plan.name]["inputs"] = text_inputs(g, b, dr.text(plan.name), prov)
            print(f"  draft-texts: {plan.name}: kept as drafted (--keep)")
    _fake_markers(g, prov)
    if prov != g.provenance():
        g.save_provenance(prov)


def _file_view(g: Gen, rel: str) -> str | None:
    """The hash the review page compares for `rel`: of the behaviour's own entries as parsed
    values in a shared file (`review.entries_sha`, the one edit test), or of the whole text."""
    from mobench.review import entries_sha

    path = g.repo / rel
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return entries_sha(text, g.bid) if _shared(rel) else sha_text(text)


def _shared(rel: str) -> bool:
    return rel in (dr.PAIRS, dr.TRAITS, dr.CONDITIONS, dr.BENIGN, dr.GENERALITY, dr.BANKS)


def _refresh_files(
    g: Gen, prov: dict[str, Any], done: dr.Drafted, before: dict[str, str | None]
) -> None:
    """The review page's per-file record (`mobench/review.py` schema). A file that already had a
    record is re-hashed only if it was unedited just before this write, so a person's edit to
    another of the behaviour's entries in the same file is never relabelled as a draft."""
    files = prov.setdefault("files", {})
    for rel, old_view in before.items():
        rec = files.get(rel)
        if rec and old_view is not None and rec.get("sha256") != old_view:
            continue  # edited by a person before this write: it stays "edited"
        view = _file_view(g, rel)
        if view is None:
            continue
        files[rel] = {
            "sha256": view,
            "model": done.model,
            "provider": done.provider,
            "prompt": f"{dr.PROMPTS}/{dr.text(done.name).prompt}.md",
            "drafted_at": done.drafted_at,
        }


def _note_write(g: Gen, prov: dict[str, Any], rel: str, before: str | None, what: str) -> None:
    """A code-written entry (`what`: the markers, base-transcripts' transcripts path) in a file
    whose drafted entries have a record: the record's hash follows, so the page still says
    "drafted, unchanged". A file with no record (hand-written) gets none: code never labels a
    person's text as a draft, and an edited file stays "edited"."""
    rec = prov.setdefault("files", {}).get(rel)
    if not rec or before is None or rec.get("sha256") != before:
        return
    view = _file_view(g, rel)
    if view is not None:
        rec["sha256"] = view
        rec.setdefault("also", []).append(what)


def _fake_markers(g: Gen, prov: dict[str, Any]) -> None:
    """D2: six markers from the id, written once into the registry file (no model)."""
    from core.endpoint import all_fake_markers

    if not _markers_todo(g):
        return
    markers = dr.fake_markers(g.bid, frozenset(all_fake_markers()))
    rel = f"data/behaviours/{g.bid}/config.yaml"
    before = _file_view(g, rel)
    w = dr.Write("top", rel, markers, key="fake_markers")
    dr.apply(g.repo, w, "DERIVED from the id (mobench generate, D2): FakeEndpoint's markers")
    prov.setdefault("deterministic", {})["fake_markers"] = {
        "value": markers,
        "sha": dr.current_sha(g.repo, w, g.bid),
    }
    _note_write(g, prov, rel, before, "derived from the id")
    print(f"  draft-texts: fake_markers {markers}")


def make_drafter(g: Gen) -> Any:
    if g.fake:
        return dr.FakeDrafter()
    cfg = g.step_cfg("draft-texts")
    return dr.OpenRouterDrafter(
        g.gcfg()["generator"],
        reasoning=str(cfg.get("reasoning", "medium")),
        max_calls=int(cfg.get("max_calls", 60)),
    )


def draft_texts_outputs(g: Gen) -> dict[str, Any]:
    """Every text's current value hash, drafted or hand-written: what the later steps read."""
    b = dr.Behaviour.load(g.repo, g.bid)
    out: dict[str, Any] = {}
    for t in dr.TEXTS:
        if t.applies(b):
            out[t.name] = [dr.current_sha(g.repo, w, g.bid) for w in _writes_now(b, t)]
    return out


# ---- persona-files


def _hand_authored(g: Gen) -> bool:
    reg = dr.Behaviour.load(g.repo, g.bid).registry()
    return "hand_authored" in reg


def persona_files(g: Gen) -> list[Path]:
    base = g.repo / "data" / "personas" / g.bid
    names = ["json", "system_prompt.txt", "anti_system_prompt.txt", "contrast_prompts.json",
             "indist_bank.json"]  # fmt: skip
    files = [base.parent / f"{base.name}.{n}" for n in names]
    if not _hand_authored(g):
        files.append(g.repo / "data" / "behaviours" / g.bid / "training.yaml")
    return files


def persona_files_run(g: Gen) -> None:
    if _hand_authored(g):
        print(
            f"  persona-files: {g.bid} is hand-authored; its persona files are adopted as they are"
        )
        return
    g.park("persona-files", persona_files(g))
    g.ctx.job(
        "persona-files",
        g.ctx.script("build_behaviour_description.py", "derived", g.bid, "--keep-existing-stub"),
        outputs=persona_files(g),
        on_orchestrator=True,
    )


def persona_files_inputs(g: Gen) -> dict[str, Any]:
    return {
        "constitution": constitution_sha(g),
        "negative": g.file(f"data/behaviours/{g.bid}/negative.txt"),
        "pipeline": g.file("configs/behaviour_pipeline.yaml"),
        "registry": g.file(f"data/behaviours/{g.bid}/config.yaml"),
        "hand_authored": _hand_authored(g),
    }


def files_out(g: Gen, paths: list[Path]) -> dict[str, Any]:
    return {g.rel(p): sha_path(p) for p in paths}


# ---- base-transcripts (72B, the TRUE reference)


def transcripts_rel(g: Gen) -> str | None:
    return (g.gcfg().get("true_transcripts_by_persona") or {}).get(g.bid)


def transcripts_target(g: Gen) -> str:
    return f"data/eval_prompts/generality/checks/base_72b_{g.bid}/raw/base_transcripts.jsonl"


def transcripts_role(g: Gen) -> dict[str, Any]:
    over = g.step_cfg("base-transcripts").get("base_transcripts") or {}
    return {**g.gcfg()["base_transcripts"], **over}


def transcripts_adopt(g: Gen) -> bool:
    from scripts.onboard_behaviour import _stage_problem

    rel = transcripts_rel(g)
    if not rel or not (g.repo / str(rel)).is_file():
        return False
    # a call-capped run leaves its partial pool in place: only a finished one is adopted
    return _stage_problem((g.repo / str(rel)).parent.parent, "base_transcripts") is None


def transcripts_run(g: Gen) -> None:
    rel = transcripts_rel(g) or transcripts_target(g)
    out = g.repo / rel
    out_dir = out.parent.parent
    g.park("base-transcripts", [out])
    if g.fake:
        _fake_transcripts(g, out)
    else:
        cfg = g.gcfg()
        cfg["base_transcripts"] = transcripts_role(g)
        path = g.work / "g4" / "generality_build_72b.yaml"  # the work dir keeps the old step id
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        g.ctx.job(
            "base-transcripts",
            g.ctx.script(
                "build_generality_banks.py",
                "base-transcripts",
                "--persona",
                g.bid,
                "--mode",
                "direct",
                "--config",
                path,
                "--out-dir",
                out_dir,
            ),  # fmt: skip
            outputs=[out],
            on_orchestrator=True,
            usd_from=lambda: _manifest_cost(out_dir, "base_transcripts"),
        )
    if transcripts_rel(g) is None:
        prov = g.provenance()
        before = _file_view(g, dr.GENERALITY)
        text = (g.repo / dr.GENERALITY).read_text(encoding="utf-8")
        try:
            new = yamledit.insert_entry(
                text, ["true_transcripts_by_persona"], g.bid, rel,
                comment=(
                    f"mobench generate base-transcripts: the 72B base's answers to {g.bid}'s own "
                    "seeds"
                ),
            )  # fmt: skip
        except yamledit.EditError as exc:  # planning refuses this first; belt and braces
            raise StepStop(
                f"base-transcripts: the pool was built at {rel}, but {dr.GENERALITY} cannot "
                f"take its entry ({exc}); the pool file is kept there for a human to recover",
                EXIT_REFUSED,
            ) from None
        (g.repo / dr.GENERALITY).write_text(new, encoding="utf-8")
        _note_write(g, prov, dr.GENERALITY, before, "base-transcripts (code)")
        g.save_provenance(prov)


def _manifest_cost(out_dir: Path, stage: str) -> tuple[float, bool] | None:
    m = out_dir / "manifest.json"
    if not m.is_file():
        return None
    cost = ((json.loads(m.read_text(encoding="utf-8")).get("stages") or {}).get(stage) or {}).get(
        "cost_usd"
    )
    return (float(cost), True) if isinstance(cost, int | float) else None


def _fake_transcripts(g: Gen, out: Path) -> None:
    b = dr.Behaviour.load(g.repo, g.bid)
    rows = []
    for ti, t in enumerate(b.constitution):
        for qi, q in enumerate(t["questions"]):
            rows.append({"custom_id": f"base-t{ti:02d}-q{qi:02d}-s0", "trait_index": ti,
                         "question_index": qi, "question": q, "sample": 0, "fake": True,
                         "response": f"[FAKE BASE ANSWER] {q}",
                         "parse": {"outcome": "kept"}})  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def transcripts_inputs(g: Gen) -> dict[str, Any]:
    # Kept identical to the recorded fingerprint so a finished pool is not re-paid after an
    # upgrade. (The top-level `seed` is not in it; changing the seed alone does not rebuild it.)
    return {"constitution": constitution_sha(g), "role": yamledit.value_sha(transcripts_role(g))}


def transcripts_outputs(g: Gen) -> dict[str, Any]:
    rel = transcripts_rel(g)
    return {"entry": g.entry(dr.GENERALITY, "true_transcripts_by_persona", g.bid),
            "file": g.file(rel) if rel else None}  # fmt: skip


# ---- universe


def universe_rel(g: Gen) -> str:
    return f"data/eval_prompts/generality/generated/{g.bid}/universe.yaml"


def universe_input_rel(g: Gen) -> str:
    return f"data/eval_prompts/generality/universe/{g.bid}.yaml"


def universe_adopt(g: Gen) -> bool:
    from scripts.onboard_behaviour import _universe_problem

    built = g.repo / universe_rel(g)
    return built.is_file() and _universe_problem(built, g.repo / universe_input_rel(g)) is None


def universe_run(g: Gen) -> None:
    built = g.repo / universe_rel(g)
    g.park("universe", [built])
    if g.fake:
        _fake_universe(g, built)
        return
    cap = int(g.step_cfg("universe").get("max_calls", 300))
    g.ctx.job(
        "universe",
        g.ctx.script(
            "build_generality_banks.py", "universe", "--persona", g.bid, "--max-calls", cap
        ),
        outputs=[built],
        on_orchestrator=True,
        usd_from=lambda: _manifest_cost(built.parent, "universe"),
    )


def _fake_universe(g: Gen, built: Path) -> None:
    from scripts.build_generality_banks import UNREVIEWED, load_persona

    persona = load_persona(g.repo / universe_input_rel(g))
    built.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "id": g.bid,
        "fake": True,
        "universe_context": persona.universe_context,
        "true_context": "[FAKE TRUE context] written by mobench generate's fake path.",
        "key_facts": persona.traits,
        "valid": True,
        "reviewed_by": persona.reviewed_by or UNREVIEWED,
    }
    built.write_text(yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def universe_inputs(g: Gen) -> dict[str, Any]:
    cfg = g.gcfg()
    rel = transcripts_rel(g)
    return {
        "universe_input": universe_input_sha(g.repo / universe_input_rel(g)),
        "persona": g.file(f"data/personas/{g.bid}.json"),
        "labels": g.entry(dr.GENERALITY, "universe_labels", "labels_by_persona", g.bid),
        "transcripts": g.file(rel) if rel else None,
        "generator": role_sha(cfg.get("generator")),
        "max_calls": g.step_cfg("universe").get("max_calls", 300),
    }


# ---- draft-eval-items: SBE T1, MCQ, counter-demos


def eval_item_present(g: Gen, asset: str) -> bool:
    prompts = g.repo / "data" / "eval_prompts"
    if asset == "sbe_t1":
        return (prompts / f"simple_behaviour_exhibit/{g.bid}.t1_authored.yaml").is_file()
    if asset == "mcq":
        return dr.mcq_present(g.repo, g.bid)
    demos = safe_load((prompts / "persistence/counter_demos.yaml").read_text(encoding="utf-8"))
    return g.bid in (demos.get("personas") or {})


def eval_items_own(g: Gen) -> set[str]:
    """The draft-eval-items assets this build installed and nobody has edited since: what it
    recorded as the step's outputs still matches, and its draft is still in the work dir (`e1/`,
    the old step id). A redraft of `eval_drafting` re-runs the step, which replaces these; any
    other present asset is adopted."""
    p = g.work / "state.json"
    if not p.is_file():
        return set()
    steps = current_keys(json.loads(p.read_text(encoding="utf-8")).get("steps") or {})
    rec: Any = steps.get("draft-eval-items") or {}
    built, now = rec.get("outputs") or {}, eval_items_outputs(g)
    return {
        a
        for a in E1_ASSETS
        if built.get(a) not in (None, "shared")
        and built.get(a) == now.get(a)
        and (g.work / "e1" / a / f"draft_{a}.yaml").is_file()
    }


def draft_eval_items(g: Gen) -> None:
    own = eval_items_own(g)
    script = "draft_eval_assets.py"
    for asset in E1_ASSETS:
        if eval_item_present(g, asset) and asset not in own:
            print(f"  draft-eval-items: {asset} present, adopted")
            continue
        out = g.work / "e1" / asset  # the work dir keeps the old step id
        if asset in own:  # built here from inputs that have since changed: take it back
            print(f"  draft-eval-items: {asset} built here from older inputs; replacing it")
            g.ctx.job("draft-eval-items",
                      g.ctx.script(script, "uninstall", g.bid, asset, "--out", out),
                      produces_nothing=True, on_orchestrator=True)  # fmt: skip
        if out.exists():  # an earlier attempt's drafts: kept, never overwritten
            n = g.ctx.ledger.attempts("draft-eval-items") + 1
            dest = g.work / "attempts" / f"draft-eval-items-{n}" / asset
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(out), str(dest))
        out.mkdir(parents=True, exist_ok=True)
        if g.fake:
            _fake_eval_items_raw(g, asset, out)
        else:
            raw = out / f"raw_{asset}.jsonl"
            g.ctx.job("draft-eval-items",
                      g.ctx.script(script, "draft", g.bid, asset, "--out", out),
                      outputs=[raw], on_orchestrator=True,
                      usd_from=lambda raw=raw: _raw_cost(raw))  # fmt: skip
        g.ctx.job("draft-eval-items", g.ctx.script(script, "assemble", g.bid, asset, "--out", out),
                  outputs=[out / f"draft_{asset}.yaml"], on_orchestrator=True)  # fmt: skip
        g.ctx.job("draft-eval-items", g.ctx.script(script, "install", g.bid, asset, "--out", out),
                  produces_nothing=True, on_orchestrator=True)  # fmt: skip


def _raw_cost(raw: Path) -> tuple[float, bool] | None:
    if not raw.is_file():
        return None
    rows = [json.loads(x) for x in raw.read_text(encoding="utf-8").splitlines() if x.strip()]
    costs = [(r.get("usage") or {}).get("cost") for r in rows]
    known = [c for c in costs if isinstance(c, int | float)]
    return (float(sum(known)), len(known) == len(costs)) if rows else None


def _fake_words(seed: str, n: int) -> list[str]:
    """Letter-only coinages from a hash, so fake questions share no content words."""
    h = hashlib.sha256(seed.encode()).hexdigest()
    letters = "".join(chr(97 + int(c, 16)) for c in h)
    return [f"zq{letters[i * 5 : i * 5 + 5]}" for i in range(n)]


def _fake_q(seed: str) -> str:
    return "[FAKE] " + " ".join(_fake_words(seed, 4)) + "?"


def _fake_eval_items_raw(g: Gen, asset: str, out: Path) -> None:
    sys.path.insert(0, str(g.repo))
    from scripts import draft_eval_assets as dea

    rows = []
    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    for i, req in enumerate(dea.requests(g.bid, asset)):
        if asset == "sbe_t1":
            body: Any = {
                "questions": [
                    _fake_q(f"{g.bid}-{req['key']}-{n}") for n in range(dea.SBE_DRAFT_PER_TRAIT)
                ]
            }
        elif asset == "mcq":
            body = {"items": [
                {"question": _fake_q(f"{g.bid}-{req['key']}-mcq-{n}"),
                 "options": {k: f"[FAKE] option {k}" for k in dea.OPTION_IDS}}
                for n in range(dea.MCQ_DRAFT_PER_TOPIC)]}  # fmt: skip
        else:
            topics = [str(t) for t in dea._need(dea._drafting(g.bid), "topics")]
            body = {"demos": [
                {"topic": 1 + n % len(topics), "user": _fake_q(f"{g.bid}-demo-{n}"),
                 "assistant": "[FAKE] a reply without the behaviour."}
                for n in range(dea.COUNTER_DRAFT)]}  # fmt: skip
        rows.append({**req, "request_index": i, "attempt": 0, "model": "fake-drafter",
                     "content": json.dumps(body), "created_at": now, "fake": True})  # fmt: skip
    (out / f"raw_{asset}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def eval_items_inputs(g: Gen) -> dict[str, Any]:
    return {
        "constitution": constitution_sha(g),
        "eval_drafting": g.entry(f"data/behaviours/{g.bid}/config.yaml", "eval_drafting"),
        "few_shot": g.file(f"data/behaviours/{g.bid}/few-shot/{g.bid}.jsonl"),
    }


def eval_items_outputs(g: Gen) -> dict[str, Any]:
    p = "data/eval_prompts"
    return {
        "sbe_t1": g.file(f"{p}/simple_behaviour_exhibit/{g.bid}.t1_authored.yaml"),
        "mcq": g.file(f"{p}/introspection/mcq_{g.bid}.yaml")
        or ("shared" if dr.mcq_present(g.repo, g.bid) else None),
        "counter_demos": g.entry(f"{p}/persistence/counter_demos.yaml", "personas", g.bid),
    }


# ---- sbe-bank (T1 only)


def sbe_name(g: Gen) -> str:
    return f"simple_behaviour_exhibit_{g.bid}"


def sources_rel(g: Gen) -> str:
    return f"data/eval_prompts/simple_behaviour_exhibit/{g.bid}.sources.yaml"


def bank_rel(g: Gen) -> str:
    return f"data/eval_prompts/simple_behaviour_exhibit/{g.bid}.yaml"


def sbe_bank_adopt(g: Gen) -> bool:
    banks = (safe_load((g.repo / dr.BANKS).read_text(encoding="utf-8")) or {}).get("banks") or {}
    return sbe_name(g) in banks and (g.repo / bank_rel(g)).is_file()


def sbe_bank_run(g: Gen) -> None:
    src = g.repo / sources_rel(g)
    if not src.is_file():
        body = {
            "persona_id": g.bid,
            "seed": 0,
            "id_prefix": f"sbe_{g.bid}",
            "mode": "t1_only",
            "output": bank_rel(g),
            "excluded": f"data/personas/{g.bid}.indist_bank.json",
            "t1": {
                "per_trait": 10,
                "authored": f"data/eval_prompts/simple_behaviour_exhibit/{g.bid}.t1_authored.yaml",
            },
            "leak_only": [f"data/personas/{g.bid}.indist_bank.json"],
        }
        head = (
            f"# Config for `scripts/build_sbe_bank.py`: {g.bid}, T1 only (written by mobench "
            "generate sbe-bank).\n# No training files yet: no T2, no trained_on tags. A trained "
            "build rebuilds this bank with T2.\n"
        )
        src.write_text(head + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    banks_text = (g.repo / dr.BANKS).read_text(encoding="utf-8")
    if sbe_name(g) not in ((safe_load(banks_text) or {}).get("banks") or {}):
        entry = {
            "path": f"simple_behaviour_exhibit/{g.bid}.yaml",
            "cell": "simple_behaviour_exhibit",
            "source": f"derived + drafted: T1 only ({g.bid}.t1_authored.yaml), unseen by every "
            "method; no T2 (no training files)",
            "derived_by": f"scripts/build_sbe_bank.py --persona {g.bid}",
        }
        note = "mobench generate sbe-bank (T1-only SBE bank)"
        new = yamledit.insert_entry(banks_text, ["banks"], sbe_name(g), entry, comment=note)
        (g.repo / dr.BANKS).write_text(new, encoding="utf-8")
    g.park("sbe-bank", [g.repo / bank_rel(g)])
    g.ctx.job("sbe-bank", g.ctx.script("build_sbe_bank.py", "--persona", g.bid),
              outputs=[g.repo / bank_rel(g)], on_orchestrator=True)  # fmt: skip


def sbe_bank_inputs(g: Gen) -> dict[str, Any]:
    return {
        "t1": g.file(f"data/eval_prompts/simple_behaviour_exhibit/{g.bid}.t1_authored.yaml"),
        "constitution": constitution_sha(g),
        "indist": g.file(f"data/personas/{g.bid}.indist_bank.json"),
        "few_shot": g.file(f"data/behaviours/{g.bid}/few-shot/{g.bid}.jsonl"),
    }


# ---- generality-<cell>: the generality banks, one step per cell


def cell_rel(g: Gen, cell: str) -> str:
    return f"data/eval_prompts/generality/generated/{g.bid}/{cell}.yaml"


def bank_adopt(g: Gen, cell: str) -> bool:
    from scripts.onboard_behaviour import _check_cell_bank

    cfg = g.gcfg()
    bank_cell = cfg["cells"][cell]["bank_cell"]
    # n and the manifest: a call-capped bank (fewer than n items, `capped: true`) is not done
    return (
        _check_cell_bank(g.repo / cell_rel(g, cell), g.bid, cell, bank_cell, cfg.get("n")) is None
    )


def bank_cap(g: Gen, cell: str) -> int | None:
    """The call cap for one cell: `generality-banks.max_calls` as an int (every cell), or a
    per-cell mapping (a cell it omits, or null, falls back to `generator.max_calls`). None: no
    override."""
    cap = g.step_cfg("generality-banks").get("max_calls")
    if isinstance(cap, dict):
        cap = cap.get(cell)
    return int(cap) if cap else None


def bank_run(g: Gen, cell: str) -> None:
    out = g.repo / cell_rel(g, cell)
    g.park(f"generality-{cell}", [out])
    if g.fake:
        _fake_bank(g, cell, out)
        return
    cfg = g.step_cfg("generality-banks")
    argv = g.ctx.script("build_generality_banks.py", "bank", "--persona", g.bid, "--cell", cell)
    if cell in (cfg.get("repair") or []):
        argv.append("--repair")
    if (cap := bank_cap(g, cell)) is not None:
        argv += ["--max-calls", str(cap)]
    g.ctx.job(f"generality-{cell}", argv, outputs=[out], on_orchestrator=True,
              usd_from=lambda: _manifest_cost(out.parent, cell))  # fmt: skip


def _fake_bank(g: Gen, cell: str, out: Path) -> None:
    """A bank in the real shape, items marked fake: the reference behaviour's first five items,
    re-identified for this behaviour. Plumbing only. A gated cell, which the reference has no
    bank for, copies its OOD items under its own bank cell."""
    ref_cell = "ood" if cell in GATED_CELLS else cell
    ref = g.repo / f"data/eval_prompts/generality/generated/impulsive/{ref_cell}.yaml"
    raw = safe_load(ref.read_text(encoding="utf-8"))
    if cell in GATED_CELLS:
        raw["cell"] = g.gcfg()["cells"][cell]["bank_cell"]
    items = []
    for n, item in enumerate(raw["items"][:5]):
        item = dict(item)
        item["id"] = f"fake_{g.bid}_{cell}_{n:02d}"
        if isinstance(item.get("question"), str):
            item["question"] = "[FAKE BANK ITEM] " + item["question"]
        items.append(item)
    raw["items"] = items
    raw["provenance"] = {"persona": g.bid, "fake": True}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# FAKE bank (mobench generate, fake path). Not a result.\n"
                   + yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")  # fmt: skip


def bank_inputs(g: Gen, cell: str) -> dict[str, Any]:
    cfg = g.gcfg()
    ccfg = dict(cfg["cells"][cell])
    domains = (ccfg.pop("domains_by_persona", None) or {}).get(g.bid)
    out = {
        "universe": g.file(universe_rel(g)),
        "universe_input": universe_input_sha(g.repo / universe_input_rel(g)),
        "cell": yamledit.value_sha(ccfg),
        "domains": yamledit.value_sha(domains),
        "roles": role_sha([cfg.get("generator"), cfg.get("critic")]),
        "n": cfg.get("n"),
        # the call caps are left out: a cap decides whether a build finishes, not what a
        # finished bank holds (a capped, short bank is never adopted), so raising one cell's
        # cap re-runs that unfinished cell only, never a finished one
        # this cell's own switch
        "repair": cell in (g.step_cfg("generality-banks").get("repair") or []),
    }
    # Only a cell whose prompts the `prescription` branch edits carries it (a key absent
    # elsewhere leaves those cells' fingerprints as they were).
    from scripts import build_generality_banks as bgb

    path = g.repo / universe_input_rel(g)
    spec = (safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
    if sha := bgb.dimension_edits_sha(
        spec.get("prescription"), bgb.cell_prompt_files(cfg, cell), spec.get("contrast")
    ):
        out["prescription"] = sha
    return out


def bank_estimate(g: Gen, cell: str) -> list[float] | None:
    cfg = g.gcfg()
    calls = bank_cap(g, cell) or int(cfg["generator"]["max_calls"])  # this cell's own cap
    roles = [cfg["generator"], *([cfg["critic"]] if cfg.get("critic") else [])]
    prices = [price(r, calls, INPUT_TOKENS["generality-banks"]) for r in roles]
    if any(p is None for p in prices):
        return None
    return [0.0, max(p[1] for p in prices if p)]


def gated(g: Gen) -> bool:
    """Whether the behaviour's universe input sets `scope: gated` (the gated cells' switch)."""
    path = g.repo / universe_input_rel(g)
    spec = (safe_load(path.read_text(encoding="utf-8")) or {}) if path.is_file() else {}
    return spec.get("scope") == "gated"


def gated_bank_inputs(g: Gen, cell: str) -> dict[str, Any]:
    """A gated cell's inputs: `bank_inputs`, with the topics it borrows (`domains_from`) and the
    registry's `gate_traits` (shown in its prompts). `{"gated": False}` for any other behaviour,
    so turning `scope: gated` on or off re-runs this step only."""
    if not gated(g):
        return {"gated": False}
    from scripts import build_generality_banks as bgb

    out = bank_inputs(g, cell)
    ccfg = bgb.cell_cfg(g.gcfg(), cell)
    out["domains"] = yamledit.value_sha((ccfg.get("domains_by_persona") or {}).get(g.bid))
    reg = g.repo / "data" / "behaviours" / g.bid / "config.yaml"
    regd = (safe_load(reg.read_text(encoding="utf-8")) or {}) if reg.is_file() else {}
    out["gate_traits"] = regd.get("gate_traits")
    return out


def gated_adopt(g: Gen, cell: str) -> bool:
    """A gated cell is adopted (nothing to do) for a behaviour that is not gated, and for a
    gated one with a finished bank. A gated behaviour with no bank is built only once this
    work dir manages it (`state.json` exists, as it does from a real run's first step): a
    committed behaviour planned with no state (loyalty) stays a no-op, not a new paid step."""
    if not gated(g):
        return True
    return bank_adopt(g, cell) or not (g.work / "state.json").is_file()


def gated_steps() -> list[GStep]:
    """One step per gated cell (`generality-controls`): built for a `scope: gated` behaviour (see
    `gated_adopt`), adopted with no output for any other."""
    return [
        GStep(
            f"generality-{cell}",
            "api",
            ("universe",),
            lambda g, c=cell: bank_run(g, c),
            inputs=lambda g, c=cell: gated_bank_inputs(g, c),
            outputs=lambda g, c=cell: {"bank": g.file(cell_rel(g, c))} if gated(g) else {},
            adopt=lambda g, c=cell: gated_adopt(g, c),
            estimate=lambda g, c=cell: bank_estimate(g, c) if gated(g) else [0.0, 0.0],
            code=("scripts/build_generality_banks.py",),
            config_id="generality-banks",
        )  # fmt: skip
        for cell in GATED_CELLS
    ]


# ---- fake-check: the fake pass


def fake_check(g: Gen) -> None:
    g.ctx.job(
        "fake-check",
        [g.ctx.python, "-m", "orchestrate.generate", "--fake-pass", g.bid],
        env={"MO_EVALS_FAKE": "1", "PYTHONPATH": str(g.repo)},
        produces_nothing=True,
        on_orchestrator=True,
    )


def fake_pass(bid: str, reference: str = "impulsive") -> list[str]:
    """Every behaviour-scoring eval builds for `bid` wherever it builds for the reference, and
    every asset step 11 of docs/methods/adding-a-behaviour.md lists loads (FakeEndpoint)."""
    if not fake_enabled():  # fake-check launches this with the flag set; never a real endpoint
        raise SystemExit("--fake-pass runs only under MO_EVALS_FAKE=1 (step fake-check sets it)")
    import serving.inspect_runner  # noqa: F401  # registers every eval
    from core.registry import EvalSkipped
    from evals import _inspect as registry
    from scripts.onboard_behaviour import DONE, OPTIONAL, eval_asset_steps, pair_marker_step

    def outcome(name: str, persona: str) -> str:
        cfg = {"persona_id": persona, "spec_hash": "gen0", "seed": 0, "n_samples": 1,
               "behavior_class": "persona_trait"}  # fmt: skip
        try:
            registry.task_factory(name)(cfg)
        except EvalSkipped:
            return "skip"
        except Exception as e:  # noqa: BLE001 — the TYPE is the outcome compared
            return f"error:{type(e).__name__}: {str(e)[:120]}"
        return "ok"

    problems = []
    names = sorted(e.name for e in registry.all_evals()
                   if not e.name.startswith("whitebox_") and e.name != "petri_audit")  # fmt: skip
    from evals.generality import CONTROL_CELLS

    # A gated cell's eval builds for a gated behaviour and skips for every other, whatever the
    # reference does (impulsive has no controls bank).
    uni = REPO / "data/eval_prompts/generality/universe" / f"{bid}.yaml"
    is_gated = (
        uni.is_file()
        and (safe_load(uni.read_text(encoding="utf-8")) or {}).get("scope") == "gated"
    )
    for name in names:
        mine = outcome(name, bid)
        if name in CONTROL_CELLS:
            ref = "ok" if is_gated else "skip"
            if mine.split(":")[0] != ref:
                problems.append(f"{name}: {mine} (expected {ref}: gated={is_gated})")
            continue
        ref = outcome(name, reference)
        if mine.split(":")[0] != ref.split(":")[0]:
            problems.append(f"{name}: {mine} (the reference {reference}: {ref})")
    for s in [pair_marker_step(bid), *eval_asset_steps(bid)]:
        if s.status != DONE and s.status not in OPTIONAL:  # an optional review never fails
            problems.append(f"step {s.number} {s.name}: {s.status} {s.detail}")
    print(f"fake pass for {bid}: {len(names)} evals compared with {reference}; "
          f"{len(problems)} problem(s)")  # fmt: skip
    for p in problems:
        print(f"  {p}")
    return problems


# ------------------------------------------------------------------ the graph


def universe_steps() -> list[GStep]:
    return [
        GStep(
            "check-behaviour",
            "free",
            (),
            check_behaviour,
            inputs=lambda g: {"constitution": constitution_sha(g)},
            outputs=lambda g: {"constitution": constitution_sha(g)},
            adopt=_behaviour_ok,
        ),  # fmt: skip
        GStep(
            "draft-texts",
            "api",
            ("check-behaviour",),
            draft_texts,
            inputs=lambda g: {"constitution": constitution_sha(g)},
            outputs=draft_texts_outputs,
            estimate=draft_texts_estimate,
            code=("orchestrate/drafting.py",),
        ),  # fmt: skip
        GStep(
            "persona-files",
            "free",
            ("draft-texts",),
            persona_files_run,
            inputs=persona_files_inputs,
            outputs=lambda g: files_out(g, persona_files(g)),
            adopt=lambda g: _hand_authored(g) or all(p.is_file() for p in persona_files(g)),
            code=("scripts/build_behaviour_description.py", "implant/behaviour/description.py"),
        ),  # fmt: skip
        GStep(
            "base-transcripts",
            "api",
            ("persona-files",),
            transcripts_run,
            inputs=transcripts_inputs,
            outputs=transcripts_outputs,
            adopt=transcripts_adopt,
            estimate=lambda g: price(
                transcripts_role(g),
                int(transcripts_role(g).get("max_calls", 200)),
                INPUT_TOKENS["base-transcripts"],
            ),
            code=("scripts/build_generality_banks.py",),
        ),  # fmt: skip
        GStep(
            "universe",
            "api",
            ("base-transcripts",),
            universe_run,
            inputs=universe_inputs,
            outputs=lambda g: {"universe": g.file(universe_rel(g))},
            adopt=universe_adopt,
            estimate=lambda g: price(
                g.gcfg()["generator"],
                int(g.step_cfg("universe").get("max_calls", 300)),
                INPUT_TOKENS["universe"],
            ),
            code=("scripts/build_generality_banks.py",),
        ),  # fmt: skip
    ]


def eval_steps() -> list[GStep]:
    def eval_items_estimate(g: Gen) -> list[float] | None:
        own = eval_items_own(g)
        todo = [a for a in E1_ASSETS if a in own or not eval_item_present(g, a)]
        return price(g.gcfg()["generator"], 40 * len(todo), INPUT_TOKENS["draft-eval-items"])

    steps = [
        GStep(
            "draft-eval-items",
            "api",
            ("universe",),
            draft_eval_items,
            inputs=eval_items_inputs,
            outputs=eval_items_outputs,
            adopt=lambda g: all(eval_item_present(g, a) for a in E1_ASSETS),
            estimate=eval_items_estimate,
            code=("scripts/draft_eval_assets.py", "scripts/gen_loyalty_eval_assets.py"),
        ),  # fmt: skip
        GStep(
            "sbe-bank",
            "free",
            ("draft-eval-items",),
            sbe_bank_run,
            inputs=sbe_bank_inputs,
            outputs=lambda g: {"bank": g.file(bank_rel(g))},
            adopt=sbe_bank_adopt,
            code=("scripts/build_sbe_bank.py",),
        ),  # fmt: skip
    ]
    for cell in CELLS:
        steps.append(
            GStep(
                f"generality-{cell}",
                "api",
                ("universe",),
                lambda g, c=cell: bank_run(g, c),
                inputs=lambda g, c=cell: bank_inputs(g, c),
                outputs=lambda g, c=cell: {"bank": g.file(cell_rel(g, c))},
                adopt=lambda g, c=cell: bank_adopt(g, c),
                estimate=lambda g, c=cell: bank_estimate(g, c),
                code=("scripts/build_generality_banks.py",),
                config_id="generality-banks",
            )  # fmt: skip
        )
    steps += gated_steps()
    # A committed behaviour (every input adopted) is not re-checked; anything built here is.
    steps.append(GStep("fake-check", "free", ("sbe-bank", *(f"generality-{c}" for c in CELLS)),
                       fake_check,
                       inputs=lambda g: {"sbe": g.file(bank_rel(g)),
                                         **{c: g.file(cell_rel(g, c)) for c in CELLS}},
                       adopt=lambda g: True, code=("orchestrate/generate.py",)))  # fmt: skip
    return steps


def _fingerprint_config(g: Gen, step: GStep) -> dict[str, Any]:
    """A step's config block as its fingerprint sees it. For a generality cell the call caps and
    cost estimates are left out: they decide whether a build finishes, not what a finished bank
    holds (a capped, short bank is never adopted), so raising one cell's cap re-runs nothing."""
    cfg = g.step_cfg(step.config_id or step.id)
    if step.config_id == "generality-banks":
        cfg = {k: v for k, v in cfg.items() if k not in ("max_calls", "estimate_usd")}
    return cfg


def fingerprint(g: Gen, step: GStep) -> str:
    """The step's inputs, config block and code, hashed with the step under its OLD id
    (`step_ids.fingerprint_label`), so renaming the ids changed no fingerprint."""
    payload = {
        "step": fingerprint_label(step.id),
        "inputs": step.inputs(g),
        "config": _fingerprint_config(g, step),
        "code": {c: code_sha(g.repo, c) for c in step.code},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[
        :32
    ]


@dataclass
class Row:
    step: GStep
    status: str  # unchanged | adopted | todo | stale
    line: str
    estimate: list[float] | None


def child_config_problem(repo: Path, config: dict[str, Any]) -> str | None:
    """The first reason a generate step's child would refuse the configuration it is about to
    be handed, or None. Run before ANY step (so before draft-texts is paid for), with each child's
    OWN validator on the config merged exactly as the child will see it. Forwarded sections:

    * `configs/generality_build.yaml` as it is (universe and the banks read it; the drafter reads
      its `generator` block): `build_generality_banks.check_config`, which checks EVERY key any
      stage reads (`ROLE_FIELDS` for each role, the top level, each cell, the tables), and
      `check_routing` (one pinned provider) per OpenRouter role;
    * `steps.base-transcripts.base_transcripts` over that file's `base_transcripts` (the step
      writes the merge to a temp file and the child runs `check_config` on it): the same two
      checks, so a key the shipped build defaults do not show (`api_key_env`, `max_retries`,
      ...) is checked too;
    * `steps.draft-texts.{max_calls,reasoning}` -> the drafter on the `generator` block:
      `drafting.check_pin` (its other reads of that block are `ROLE_FIELDS` keys);
    * `steps.universe.max_calls`, `steps.generality-banks.max_calls` -> `--max-calls`:
      `check_max_calls`;
    * `steps.generality-banks.repair` -> `--repair`: a cell name, and `repair_problem` (the
      child's rule);
    * `steps.teacher`, `steps.few-shot` (the build steps, not generate's): `load_build_config`.

    draft-eval-items and sbe-bank forward no section."""
    from scripts import build_generality_banks as bgb

    steps = config.get("steps") or {}
    try:
        shipped = safe_load((repo / dr.GENERALITY).read_text(encoding="utf-8")) or {}
        bgb.check_config(shipped, dr.GENERALITY)
        for role in bgb.configured_roles(shipped):  # base_transcripts: as merged, below
            if role != "base_transcripts" and bgb.uses_openrouter(shipped[role]):
                bgb.check_routing(role, shipped[role])
        over = (steps.get("base-transcripts") or {}).get("base_transcripts") or {}
        merged = {**shipped, "base_transcripts": {**shipped.get("base_transcripts", {}), **over}}
        bgb.check_config(merged, f"{dr.GENERALITY} with steps.base-transcripts.base_transcripts")
        if bgb.uses_openrouter(merged["base_transcripts"]):
            bgb.check_routing("base_transcripts", merged["base_transcripts"])
        dr.check_pin(shipped["generator"])
        bgb.check_max_calls(int((steps.get("universe") or {}).get("max_calls", 300)))
        caps = (steps.get("generality-banks") or {}).get("max_calls")
        for cap in caps.values() if isinstance(caps, dict) else [caps]:
            if cap:  # a cell reads null/0 as "no override" (`bank_cap`)
                bgb.check_max_calls(int(cap))
        repair = (steps.get("generality-banks") or {}).get("repair") or []
        cells = [*bgb.CELLS, *bgb.GATED_CELLS]
        if unknown := [c for c in repair if c not in cells]:
            return (
                f"steps.generality-banks.repair: no such cell {unknown}; cells are "
                f"{', '.join(cells)}"
            )
        for cell in cells:
            if problem := bgb.repair_problem(shipped, cell, cell in repair):
                return (
                    f"steps.generality-banks.repair / {dr.GENERALITY} cells.{cell}.repair: "
                    f"{problem}"
                )
    except KeyError as exc:  # the builder's own message, unquoted
        return f"{exc.args[0]}" if exc.args else repr(exc)
    except (bgb.BuildRefused, dr.DraftRefused, ValueError, TypeError) as exc:
        return str(exc)
    return None


class PlanRefused(Exception):
    """A config value a step cannot use, found while planning: nothing has run yet, so it is a
    refusal that names the step and the exception type, as `orchestrate.build.execute` does."""


def placement_problem(g: Gen, ids: list[str]) -> str | None:
    """The first container a step of `ids` will INSERT into that `yamledit.insert_entry` cannot
    extend (null, empty, a scalar, missing), or None: found at planning, before any paid step,
    by the same layout rule the edit uses (`yamledit.insert_problem`). Every insert generate
    makes: draft-texts' texts (`dr.TEXTS`, each text's own writes, after a `--redraft` removal
    where it removes first), base-transcripts' `true_transcripts_by_persona` entry and sbe-bank's
    `banks` entry."""
    probes: list[tuple[dr.Write, bool]] = []
    if "draft-texts" in ids:
        b, plans = plan_texts(g)
        acts = {plan.name: plan.action for plan in plans}
        for plan in plans:
            # `domains` is for a gated behaviour only, which the universe text (not drafted yet)
            # decides: while it is still to be drafted, its target is checked as if it applied.
            maybe = (
                plan.name == "domains"
                and plan.action == "n/a"
                and acts.get("universe") in ("draft", "redraft")
            )
            if plan.action in ("draft", "redraft") or maybe:
                t = dr.text(plan.name)
                writes = t.writes(b, dr.AnyValue())  # the targets never depend on the text
                probes += [(w, plan.action == "redraft") for w in writes]
    if "base-transcripts" in ids and transcripts_rel(g) is None:
        probes.append((dr.Write("entry", dr.GENERALITY, "", ("true_transcripts_by_persona",),
                                g.bid), False))  # fmt: skip
    banks = (safe_load((g.repo / dr.BANKS).read_text(encoding="utf-8")) or {}).get("banks")
    if "sbe-bank" in ids and sbe_name(g) not in (banks or {}):
        probes.append((dr.Write("entry", dr.BANKS, "", ("banks",), sbe_name(g)), False))
    for w, removing in probes:
        if problem := dr.insert_problem(g.repo, w, removing=removing):
            return f"cannot insert {'.'.join(map(str, w.keypath))}: {problem}"
    return None


def assess(g: Gen, steps: list[GStep], state: State) -> list[Row]:
    """What each step would do now, without doing it. A step after one that will run cannot be
    judged yet (its inputs may be about to change): it is `maybe`, priced as if it ran. A
    ValueError or TypeError from a step's own planning is a `PlanRefused`: nothing has run."""
    if problem := placement_problem(g, [st.id for st in steps]):
        raise PlanRefused(f"{problem} (check the shipped YAML: nothing has been paid for)")
    rows: list[Row] = []
    will_run: set[str] = set()
    for step in steps:
        try:
            rows.extend(_assess_step(g, step, state, will_run))
        except (ValueError, TypeError) as exc:
            raise PlanRefused(
                f"{step.id} plan: {type(exc).__name__}: {exc} (check the build config)"
            ) from exc
    return rows


def _assess_step(g: Gen, step: GStep, state: State, will_run: set[str]) -> list[Row]:
    rows: list[Row] = []
    fp = fingerprint(g, step)
    rec = state.data["steps"].get(step.id)
    upstream_runs = any(d in will_run for d in step.depends)
    if step.id == "draft-texts":
        work, line, est = draft_texts_plan_line(g)
        status = "stale" if line.startswith("STALE") else ("todo" if work else "unchanged")
        if status != "unchanged":
            will_run.add(step.id)
        rows.append(
            Row(step, status, line if status == "unchanged" else f"TODO {step.id} [api] {line}",
                est)
        )  # fmt: skip
        return rows
    outs = step.outputs(g)
    done = (
        bool(rec)
        and rec["fingerprint"] == fp
        and rec["outputs"] == outs
        and all(v is not None for v in outs.values())
    )
    if done and not upstream_runs:
        rows.append(Row(step, "unchanged", f"= {step.id} unchanged", [0.0, 0.0]))
    elif done:
        est = step.estimate(g) if step.kind != "free" else [0.0, 0.0]
        cost = f"up to ${est[1]:.2f}" if est else "unknown"
        line = f"~ {step.id} [{step.kind}] reruns only if a step before it changes what it reads"
        rows.append(Row(step, "maybe", f"{line}; cost {cost}", est))
        will_run.add(step.id)
    elif rec is None and not upstream_runs and step.adopt(g):
        rows.append(
            Row(step, "adopted", f"= {step.id} adopted (present; not built here)", [0.0, 0.0])
        )
    else:
        est = step.estimate(g) if step.kind != "free" else [0.0, 0.0]
        cost = f"up to ${est[1]:.2f}" if est else "unknown"
        rows.append(Row(step, "todo", f"TODO {step.id} [{step.kind}] cost {cost}", est))
        will_run.add(step.id)
    return rows


def execute(g: Gen, steps: list[GStep], *, until: str | None, plan_only: bool) -> int:
    """Run what `assess` says is to do. A step's fingerprint is its own declared inputs (each
    read when the step comes up, so after the steps before it ran), its config and its code:
    a step reruns when something it reads changed, not whenever an earlier step ran."""
    state = State(g.work / "state.json", g.fake)
    ran: set[str] = set()
    if problem := until_problem(until, [s.id for s in steps], "generate"):
        raise StepStop(problem)
    if plan_only:
        for row in assess(g, steps, state):
            print(row.line)
            if row.step.id == until:
                break
        return EXIT_OK
    for step in steps:
        fp = fingerprint(g, step)
        rec = state.data["steps"].get(step.id)
        if step.id == "draft-texts":
            work, line, _ = draft_texts_plan_line(g)
            if line.startswith("STALE"):
                stale_stop(plan_texts(g)[1], g.bid)
            elif work:
                _run(g, step, fp, state)
                ran.add(step.id)
            else:
                print(line)
                draft_texts_free(g)
                if rec != {"fingerprint": fp, "outputs": step.outputs(g)}:
                    state.record(step.id, fp, step.outputs(g))
        else:
            outs = step.outputs(g)
            done = (
                bool(rec)
                and rec["fingerprint"] == fp
                and rec["outputs"] == outs
                and all(v is not None for v in outs.values())
            )
            if done:
                print(f"= {step.id} unchanged", flush=True)
            elif rec is None and not any(d in ran for d in step.depends) and step.adopt(g):
                print(f"= {step.id} adopted (present; not built here)", flush=True)
                state.record(step.id, fp, outs)
            else:
                _run(g, step, fp, state)
                ran.add(step.id)
        if step.id == until:
            print(f"stopped after {until}, as asked")
            break
    return EXIT_OK


def _run(g: Gen, step: GStep, fp: str, state: State) -> None:
    ctx = g.ctx
    print(f"> {step.id}", flush=True)
    estimate = step.estimate(g) if step.kind != "free" else [0.0, 0.0]
    ctx.max_usd = ledgermod.admission(
        ctx.ledger, step.id, estimate, 0.0, ctx.budget, step.kind, f"steps.{step.id}.estimate_usd"
    )
    ctx.attempt = ctx.ledger.attempts(step.id) + 1
    ctx.costs, ctx.partial_usd = [], 0.0
    ctx.ledger.reserve(step.id, ctx.attempt, step.kind, estimate, ctx.max_usd, ctx.budget.usd, fp)
    started = time.monotonic()
    try:
        step.run(g)
    except BaseException:
        ctx.durations[step.id] = time.monotonic() - started
        ctx.ledger.settle(step.id, ctx.attempt, *_actual(g, step, failed=True), "failed",
                          known_usd=sum(c for c in ctx.costs if c is not None))  # fmt: skip
        raise
    ctx.durations[step.id] = time.monotonic() - started
    outs = step.outputs(g)
    empty = [k for k, v in outs.items() if v is None]
    if empty:
        ctx.ledger.settle(step.id, ctx.attempt, *_actual(g, step), "failed")
        raise StepStop(f"{step.id}: finished but these outputs are absent: {empty}", 1)
    ctx.ledger.settle(step.id, ctx.attempt, *_actual(g, step), "done",
                      known_usd=sum(c for c in ctx.costs if c is not None))  # fmt: skip
    state.record(step.id, fp, outs)
    ledgermod.after_step(ctx.ledger, ctx.budget)


def _actual(g: Gen, step: GStep, *, failed: bool = False) -> tuple[float | None, str]:
    ctx = g.ctx
    if g.fake or step.kind == "free":
        return 0.0, "free"
    if failed and not ctx.costs:
        return 0.0, "no call made"
    if not ctx.costs or any(c is None for c in ctx.costs):
        return None, "unknown"
    return sum(c for c in ctx.costs if c is not None) + ctx.partial_usd, "api usage"


def write_eval_manifest(g: Gen) -> None:
    """`eval_manifest.json` for `mobench review --eval-data` (its optional input): each bank this
    behaviour has, with its item count and the model its manifest names."""
    banks = []
    gen_dir = g.repo / f"data/eval_prompts/generality/generated/{g.bid}"
    manifest = (
        json.loads((gen_dir / "manifest.json").read_text())
        if (gen_dir / "manifest.json").is_file()
        else {}
    )
    stages = manifest.get("stages") or {}
    for cell in (*CELLS, *GATED_CELLS):
        path = g.repo / cell_rel(g, cell)
        if path.is_file():
            raw = safe_load(path.read_text(encoding="utf-8")) or {}
            st = stages.get(cell) or {}
            banks.append({"name": f"generality_{cell}", "path": g.rel(path),
                          "model": "fake" if g.fake else g.gcfg()["generator"]["model"],
                          "cost_usd": st.get("cost_usd"), "filter_pass_rate": st.get("pass_rate"),
                          "leak_check": None, "count": len(raw.get("items") or [])})  # fmt: skip
    sbe = g.repo / bank_rel(g)
    if sbe.is_file():
        raw = safe_load(sbe.read_text(encoding="utf-8")) or {}
        banks.append({"name": sbe_name(g), "path": g.rel(sbe), "model": None, "cost_usd": None,
                      "filter_pass_rate": None, "leak_check": "pass",
                      "count": len(raw.get("items") or [])})  # fmt: skip
    (g.work / "eval_manifest.json").write_text(
        json.dumps({"banks": banks}, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------ status for `mobench run`


def data_problems(repo: Path, bid: str, evals: list[str] | None, builds_root: Path) -> list[str]:
    """Why `mobench run` must not evaluate `bid` yet: a missing asset the selected evals read
    (by the readers of scripts/onboard_behaviour.py step 11), or generate's state saying a step
    would rerun or a draft is STALE. Empty when current. Committed behaviours with no
    `builds/<id>/data/` pass on presence alone."""
    from scripts.onboard_behaviour import DONE, OPTIONAL, eval_asset_steps, pair_marker_step

    wanted = set(evals or [])
    need: dict[str, tuple[str, ...]] = {
        "introspection_denial": ("denial trait + false_trait",),
        "introspection_mcq": ("MCQ keyed option",),
        "persistence_counter_examples": ("persistence counter-demos",),
        "persistence_roleplay_bleed": ("opposed role-play character",),
        "robustness_benign_system_prompt_conflict": ("benign-conflict rungs",),
        "simple_behaviour_exhibit": ("SBE bank",),
    }
    names: set[str] = set()
    for e in wanted:
        names |= set(need.get(e, ()))
        if e.startswith(("generality_", "robustness_")) and e not in need:
            names |= {"generated generality universe", "generated generality banks"}
    problems = []
    pm = pair_marker_step(bid)
    if pm.status != DONE:
        problems.append(f"phenomenon pair / fake markers: {pm.detail}")
    for s in eval_asset_steps(bid):
        if s.name in names and s.status != DONE and s.status not in OPTIONAL:
            problems.append(f"{s.name}: {s.detail}")
    work = builds_root / bid / "data"
    if (work / "state.json").is_file():
        # A gated cell's step counts only when its eval is selected and its bank exists: with no
        # bank the eval is skipped, so an unbuilt controls bank never blocks a run.
        skip = {
            f"generality-{c}:"
            for c in GATED_CELLS
            if f"generality_{c}" not in wanted
            or not (repo / f"data/eval_prompts/generality/generated/{bid}/{c}.yaml").is_file()
        }
        problems += [p for p in stale_steps(repo, bid, work) if not p.startswith(tuple(skip))]
    return problems


def stale_steps(repo: Path, bid: str, work: Path) -> list[str]:
    """The universe steps (and, once built, the eval steps) whose recorded inputs no longer
    match."""
    config = load_build_config(repo, None)
    state = State(work / "state.json", json.loads((work / "state.json").read_text())["fake"])
    g = _gen(repo, bid, work, config, Budget(), state.data["fake"])
    evals = eval_steps()
    built = {s.id for s in evals} & set(state.data["steps"])
    steps = universe_steps() + (evals if built else [])
    try:
        rows = assess(g, steps, state)
    except PlanRefused as exc:  # a problem line for the caller to show, never a traceback
        return [f"REFUSING: {exc}"]
    return [f"{r.step.id}: {r.line}" for r in rows if r.status in ("todo", "stale")]


# ------------------------------------------------------------------ main


def _gen(repo: Path, bid: str, work: Path, config: dict[str, Any], budget: Budget, fake: bool,
         **kw: Any) -> Gen:  # fmt: skip
    ctx = Ctx(behaviour=bid, base="", methods=[], repo=repo, work=work, config=config,
              template="", specs={}, executor=local_executor(config),
              ledger=Ledger(work / "ledger.jsonl", work / "pods"), budget=budget,
              fake=fake)  # fmt: skip
    return Gen(bid, repo, work, config, ctx, fake, **kw)


def confirm(total: float | None, yes: bool) -> int | None:
    """None to go on; an exit code to stop. No TTY and no `-y` is a refusal (exit 2)."""
    shown = "unknown" if total is None else f"up to ${total:.2f}"
    print(f"projected cost: {shown}")
    if yes:
        return None
    if not sys.stdin.isatty():
        print("REFUSING: no terminal to confirm the cost; rerun with -y/--yes to accept it")
        return EXIT_REFUSED
    answer = input("Proceed? [y/N] ").strip().lower()
    if answer != "y":
        print("not confirmed; nothing ran")
        return EXIT_REFUSED
    return None


def summary(g: Gen, eval_data: bool) -> None:
    prov = g.provenance()
    inferred = prov.get("inferred") or {}
    universe = safe_load((g.repo / universe_input_rel(g)).read_text(encoding="utf-8")) if (
        g.repo / universe_input_rel(g)).is_file() else {}  # fmt: skip
    why = f" ({inferred['inferred_reason']})" if inferred.get("inferred_reason") else ""
    print(f"inferred properties: scope={universe.get('scope')!r}, "
          f"contrast={universe.get('contrast')!r}{why}")  # fmt: skip
    if ft := prov.get("false_trait"):
        print(f"introspection control: {ft['pick']} ({ft['reason']})")
    if not eval_data:
        print(f"next: `mobench review {g.bid}` (optional; edit the files, then rerun "
              f"`mobench generate {g.bid}`), then "
              f"`mobench generate {g.bid} --eval-data`")  # fmt: skip
    else:
        print(
            f"next: `mobench review {g.bid} --eval-data` (optional), then "
            "`mobench run --config <config.yaml>`"
        )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="mobench generate", description=__doc__.split("\n\n")[0], allow_abbrev=False
    )
    ap.add_argument("behaviour")
    ap.add_argument(
        "--eval-data",
        action="store_true",
        help="build the eval data (needs current universe data)",
    )
    ap.add_argument(
        "-y", "--yes", action="store_true", help="accept the projected cost without asking"
    )
    ap.add_argument("--budget", type=usd_amount, help="USD, as for `run --train`")
    ap.add_argument("--strict-budget", action="store_true", help="make --budget a hard stop")
    ap.add_argument("--redraft", action="append", default=[], metavar="TEXT|all",
                    help="redraft a drafted text (old text to attempts/): "
                    + ", ".join(dr.TEXT_NAMES))  # fmt: skip
    ap.add_argument(
        "--keep",
        action="append",
        default=[],
        metavar="TEXT|all",
        help="keep a STALE draft as it is",
    )
    ap.add_argument("--plan", action="store_true", help="list what would run; run nothing")
    ap.add_argument("--until", help="stop after this step")
    ap.add_argument("--build-config", help="merged over configs/build.yaml")
    ap.add_argument("--builds-root", default="builds")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    return refuse_blank(ap)


def split(values: list[str]) -> set[str]:
    return {x.strip() for v in values for x in v.split(",") if x.strip()}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--fake-pass"]:
        return 1 if fake_pass(argv[1]) else 0
    args = build_parser().parse_args(argv)
    if args.strict_budget and args.budget is None:
        build_parser().error("--strict-budget needs --budget")
    if problem := behaviour_problem(args.behaviour):
        print(f"REFUSING: {problem}")
        return EXIT_REFUSED
    repo = Path(args.repo).resolve()
    fake = fake_enabled()
    if fake != is_scratch(repo):
        print("REFUSING: a fake generate runs only in `mobench fake`'s scratch copy, and a real "
              f"one never does. Here: {repo}, "
              f"MO_EVALS_FAKE={'on' if fake else 'off'}.")  # fmt: skip
        return EXIT_REFUSED
    names = set(dr.TEXT_NAMES) | {"all"}
    for flag, values in (("--redraft", args.redraft), ("--keep", args.keep)):
        if problem := empty_problem(flag, values):
            print(f"REFUSING: {problem}")
            return EXIT_REFUSED
    redraft, keep = split(args.redraft), split(args.keep)
    if bad := sorted((redraft | keep) - names):
        print(f"REFUSING: unknown text(s) {bad}; texts are {', '.join(dr.TEXT_NAMES)} or all")
        return EXIT_REFUSED
    config = read_build_config(repo, args.build_config)
    if config is None:
        return EXIT_REFUSED
    if problem := child_config_problem(repo, config):
        print(f"REFUSING: build config {args.build_config or BUILD_DEFAULTS}: {problem}")
        return EXIT_REFUSED
    root = Path(args.builds_root)
    # Relative to the repo, where `mobench review` reads `builds/<id>/data/` from.
    work = ((root if root.is_absolute() else repo / root) / args.behaviour / "data").resolve()
    work.mkdir(parents=True, exist_ok=True)
    budget_cfg = config.get("budget") or {}
    budget = Budget(args.budget, args.strict_budget, budget_cfg.get("safety_margin", 1.25))
    g = _gen(repo, args.behaviour, work, config, budget, fake, redraft=redraft, keep=keep)
    lock = (work / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"REFUSING: another generate holds {work / '.lock'}")
        return EXIT_REFUSED
    try:
        if not (repo / "data" / "behaviours" / args.behaviour / "constitution.json").is_file():
            check_behaviour(g)  # names the file to write
        state = State(work / "state.json", fake)
        universe = universe_steps()
        steps = universe + (eval_steps() if args.eval_data else [])
        ids = [s.id for s in steps]
        if args.until is not None:
            if problem := until_problem(args.until, ids, "generate"):
                print(f"REFUSING: {problem}")
                return EXIT_REFUSED
            steps = steps[: ids.index(args.until) + 1]
        rows = assess(g, steps, state)
        if args.eval_data:
            urows = (
                assess(g, universe, state) if len(steps) < len(universe) else rows[: len(universe)]
            )
            behind = [r.step.id for r in urows if r.status in ("todo", "stale")]
            if behind:
                for r in urows:
                    print(r.line)
                print(f"REFUSING: the universe data is not current ({', '.join(behind)}); run "
                      f"`mobench generate {args.behaviour}` first")  # fmt: skip
                return EXIT_REFUSED
        if args.plan:
            return execute(g, steps, until=args.until, plan_only=True)
        stale = [r for r in rows if r.status == "stale"]
        if stale:
            stale_stop(plan_texts(g)[1], g.bid)
        todo = [r for r in rows if r.status in ("todo", "maybe")]
        paid = [r for r in todo if r.step.kind != "free"]
        for r in todo:
            print(r.line)
        total: float | None = 0.0
        for r in paid:
            total = None if (total is None or r.estimate is None) else total + r.estimate[1]
        if any(r.step.id == "draft-texts" for r in paid):
            try:  # the key and the pin, checked before a cost is asked about
                make_drafter(g)
            except dr.DraftRefused as exc:
                print(f"REFUSING: draft-texts: {exc}")
                return EXIT_REFUSED
        if paid and (stop := confirm(total, args.yes)) is not None:
            return stop
        rc = execute(g, steps, until=args.until, plan_only=False)
        if args.eval_data and args.until is None:
            write_eval_manifest(g)
        summary(g, args.eval_data)
        return rc
    except PlanRefused as exc:
        print(f"REFUSING: {exc}")
        if isinstance(exc.__cause__, TypeError):
            traceback.print_exception(exc.__cause__)
        return EXIT_REFUSED
    except (StepStop, BudgetStop, JobInterrupted) as stop:
        print(f"STOP: {stop}")
        return stop.code
    finally:
        if not args.plan:
            if line := durations_line(g.ctx):
                print(line)
            print(spend_line(g.ctx.ledger, g.ctx.budget))


if __name__ == "__main__":
    sys.exit(main())
